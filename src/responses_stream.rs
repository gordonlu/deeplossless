//! Responses API stream processing state machine.
//!
//! Manages the lifecycle of a Responses API SSE stream:
//! converts upstream Chat Completions SSE into Responses SSE
//! events with explicit state transitions and invariants.

use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::mpsc;

use crate::db::Database;
use crate::protocol::canonical::StreamEvent;
use crate::protocol::streaming::{DeepSeekNormalizer, StreamAssembler};
use crate::runtime::ExecutionCycle;

/// Explicit states for the Responses stream lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamPhase {
    /// About to send preamble (response.created, etc.)
    Preamble,
    /// Processing upstream SSE data, emitting Responses SSE
    Active,
    /// Upstream [DONE] received, draining assembled content
    Draining,
    /// Final events emitted, persisting to DB
    Persisting,
    /// All done
    Completed,
    /// Fatal, no further events
    Failed,
}

/// Contract: allowed state transitions.
///
/// ```text
/// Preamble → Active → Draining → Persisting → Completed
///     ↓          ↓         ↓          ↓
///    Failed    Failed    Failed     Failed
/// ```
impl StreamPhase {
    pub fn can_transition_to(&self, next: &StreamPhase) -> bool {
        matches!(
            (self, next),
            (StreamPhase::Preamble, StreamPhase::Active)
                | (StreamPhase::Preamble, StreamPhase::Failed)
                | (StreamPhase::Active, StreamPhase::Draining)
                | (StreamPhase::Active, StreamPhase::Failed)
                | (StreamPhase::Draining, StreamPhase::Persisting)
                | (StreamPhase::Draining, StreamPhase::Failed)
                | (StreamPhase::Persisting, StreamPhase::Completed)
                | (StreamPhase::Persisting, StreamPhase::Failed)
        )
    }
}

/// Process an upstream Chat Completions stream into Responses SSE events.
///
/// Owns the assembler, normalizer, and event channel. Drives the
/// conversion with explicit phase transitions.
pub struct ResponsesStreamProcessor {
    phase: StreamPhase,
    assembler: StreamAssembler,
    ds4_normalizer: Option<DeepSeekNormalizer>,
    tx: mpsc::Sender<Result<axum::body::Bytes, std::convert::Infallible>>,
    db: Arc<Database>,
    cycle: Arc<StdMutex<ExecutionCycle>>,
    stream_execution_id: i64,
    stream_conv_id: i64,
    replay_session_id: String,
    replay_seq_no: i64,
    usage_buf: Option<serde_json::Value>,
    flushed_tool_calls: Vec<(String, String, String)>,
}

impl ResponsesStreamProcessor {
    pub fn new(
        tx: mpsc::Sender<Result<axum::body::Bytes, std::convert::Infallible>>,
        db: Arc<Database>,
        cycle: Arc<StdMutex<ExecutionCycle>>,
        stream_execution_id: i64,
        stream_conv_id: i64,
        replay_session_id: String,
        use_normalizer: bool,
    ) -> Self {
        Self {
            phase: StreamPhase::Preamble,
            assembler: StreamAssembler::new(),
            ds4_normalizer: if use_normalizer { Some(DeepSeekNormalizer::new()) } else { None },
            tx,
            db,
            cycle,
            stream_execution_id,
            stream_conv_id,
            replay_session_id,
            replay_seq_no: 0,
            usage_buf: None,
            flushed_tool_calls: Vec::new(),
        }
    }

    pub fn phase(&self) -> &StreamPhase { &self.phase }

    fn transition_to(&mut self, next: StreamPhase) {
        debug_assert!(self.phase.can_transition_to(&next),
            "invalid phase transition: {:?} → {:?}", self.phase, next);
        self.phase = next;
    }

    /// Feed a single Chat Completions SSE data line.
    /// Returns Ok(true) to continue, Ok(false) to stop.
    pub fn feed_sse_line(&mut self, data_line: &str) -> Result<bool, String> {
        if matches!(self.phase, StreamPhase::Failed | StreamPhase::Completed) {
            return Ok(false);
        }
        if self.phase == StreamPhase::Preamble {
            self.transition_to(StreamPhase::Active);
        }

        // Capture usage
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data_line) {
            if v.get("usage").is_some() {
                self.usage_buf = Some(v);
            }
        }

        for event in crate::protocol::streaming::from_chat_completions_sse(
            data_line, self.usage_buf.as_ref(),
        ) {
            let enriched = self.enrich_event(&event);
            for ev in enriched {
                if matches!(ev, StreamEvent::Done { .. }) {
                    self.transition_to(StreamPhase::Draining);
                    return self.flush_assembler();
                }
                let events = self.assembler.feed(ev);
                if !events.is_empty() && !self.send_events(events)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Process trailing buffer content (last line without `\n`).
    pub fn feed_trailing_buffer(&mut self, data_line: &str) -> Result<bool, String> {
        if matches!(self.phase, StreamPhase::Failed | StreamPhase::Completed) {
            return Ok(false);
        }
        if !data_line.trim().is_empty() {
            for event in crate::protocol::streaming::from_chat_completions_sse(
                data_line, self.usage_buf.as_ref(),
            ) {
                if !matches!(event, StreamEvent::Done { .. }) {
                    let enriched = self.enrich_event(&event);
                    for ev in enriched {
                        let events = self.assembler.feed(ev);
                        if !events.is_empty() && !self.send_events(events)? {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    /// Flush DS4 normalizer (remaining think-reasoning or DSML tool calls).
    pub fn flush_normalizer(&mut self) {
        if let Some(ref mut norm) = self.ds4_normalizer {
            match norm.finish() {
                Ok(flush_events) => {
                    for ev in flush_events {
                        let events = self.assembler.feed(ev);
                        if !events.is_empty() {
                            let _ = self.send_events(events);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("DS4 normalizer finish error: {:?}", e);
                }
            }
        }
    }

    /// Finish assembly, get accumulated content, emit final events.
    /// Returns (text, reasoning, tool_calls) for session persistence.
    pub fn finish_assembly(&mut self) -> AssembledOutput {
        self.transition_to(StreamPhase::Persisting);
        let (content, final_events) = self.assembler.finish();
        if !final_events.is_empty() {
            let mut events = final_events;
            let mut tc_idx = 1usize;
            for ev in &mut events {
                match ev {
                    StreamEvent::ToolCallStart { index, .. }
                    | StreamEvent::ToolCallArgsDelta { index, .. }
                    | StreamEvent::FunctionCallArgumentsDone { output_index: index, .. }
                    | StreamEvent::OutputItemDone { index, .. } => {
                        *index = tc_idx;
                    }
                    _ => {}
                }
                if matches!(ev, StreamEvent::OutputItemDone { .. }) {
                    tc_idx += 1;
                }
            }
            let _ = self.send_events(events);
        }
        self.transition_to(StreamPhase::Completed);

        let input_tokens = self.usage_buf.as_ref()
            .and_then(|v| v["usage"]["prompt_tokens"].as_u64()).unwrap_or(0);
        let output_tokens = self.usage_buf.as_ref()
            .and_then(|v| v["usage"]["completion_tokens"].as_u64()).unwrap_or(0);

        AssembledOutput {
            text: content.text,
            reasoning: content.reasoning,
            input_tokens: input_tokens as u32,
            output_tokens: output_tokens as u32,
            tool_calls: std::mem::take(&mut self.flushed_tool_calls),
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn enrich_event(&mut self, event: &StreamEvent) -> Vec<StreamEvent> {
        match event {
            StreamEvent::TextDelta { text } if !text.is_empty() => {
                if let Some(ref mut norm) = self.ds4_normalizer {
                    match norm.feed_text(text) {
                        Ok(events) => events,
                        Err(e) => {
                            tracing::warn!("DS4 normalizer error: {:?}", e);
                            vec![event.clone()]
                        }
                    }
                } else {
                    vec![event.clone()]
                }
            }
            _ => vec![event.clone()],
        }
    }

    fn flush_assembler(&mut self) -> Result<bool, String> {
        let mut events = self.assembler.flush();
        // Remap tool call output_index
        let mut tc_idx = 1usize;
        for ev in &mut events {
            match ev {
                StreamEvent::ToolCallStart { index, .. }
                | StreamEvent::ToolCallArgsDelta { index, .. }
                | StreamEvent::FunctionCallArgumentsDone { output_index: index, .. }
                | StreamEvent::OutputItemDone { index, .. } => {
                    *index = tc_idx;
                }
                _ => {}
            }
            if matches!(ev, StreamEvent::OutputItemDone { .. }) {
                tc_idx += 1;
            }
        }
        // Collect function calls
        for ev in &events {
            if let StreamEvent::FunctionCallArgumentsDone { call_id, name, arguments, .. } = ev {
                self.flushed_tool_calls.push((call_id.clone(), name.clone(), arguments.clone()));
            }
        }
        self.send_events(events)
    }

    fn send_events(&mut self, events: Vec<StreamEvent>) -> Result<bool, String> {
        match crate::proxy::process_events(
            events,
            self.db.clone(),
            &self.cycle,
            &self.tx,
            Some(&mut self.assembler),
            true,
            self.stream_execution_id,
            self.stream_conv_id,
            &self.replay_session_id,
            &mut self.replay_seq_no,
        ) {
            Ok(cont) => Ok(cont),
            Err(e) => {
                tracing::warn!("execution event store failed: {e}");
                Ok(false)
            }
        }
    }
}

/// Output of a completed stream assembly.
pub struct AssembledOutput {
    pub text: String,
    pub reasoning: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub tool_calls: Vec<(String, String, String)>,
}

// ── Contract tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ExecutionCycle, RuntimeProfile};

    fn make_db() -> Arc<Database> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            Arc::new(crate::db::Database::builder()
                .path(":memory:".to_string())
                .build()
                .await
                .unwrap())
        })
    }

    fn make_processor() -> (ResponsesStreamProcessor, mpsc::Receiver<Result<axum::body::Bytes, std::convert::Infallible>>) {
        let (tx, rx) = mpsc::channel(128);
        let db = make_db();
        let cycle = Arc::new(StdMutex::new(
            ExecutionCycle::new(RuntimeProfile::Minimal),
        ));
        let proc = ResponsesStreamProcessor::new(
            tx, db, cycle, 1, 0,
            "test_replay".into(),
            false,
        );
        (proc, rx)
    }

    #[test]
    fn phase_transitions_are_valid() {
        let p = StreamPhase::Preamble;
        assert!(p.can_transition_to(&StreamPhase::Active));
        assert!(p.can_transition_to(&StreamPhase::Failed));
        assert!(!p.can_transition_to(&StreamPhase::Draining));

        let a = StreamPhase::Active;
        assert!(a.can_transition_to(&StreamPhase::Draining));
        assert!(a.can_transition_to(&StreamPhase::Failed));
        assert!(!a.can_transition_to(&StreamPhase::Preamble));
        assert!(!a.can_transition_to(&StreamPhase::Completed));

        let d = StreamPhase::Draining;
        assert!(d.can_transition_to(&StreamPhase::Persisting));
        assert!(d.can_transition_to(&StreamPhase::Failed));
        assert!(!d.can_transition_to(&StreamPhase::Active));

        let r = StreamPhase::Persisting;
        assert!(r.can_transition_to(&StreamPhase::Completed));
        assert!(r.can_transition_to(&StreamPhase::Failed));
    }

    #[test]
    fn initial_phase_is_preamble() {
        let (proc, _rx) = make_processor();
        assert_eq!(*proc.phase(), StreamPhase::Preamble);
    }

    #[test]
    fn feed_sse_text_transitions_to_active() {
        let (mut proc, _rx) = make_processor();
        // First feed transitions to Active
        let _ = proc.feed_sse_line(r#"{"choices":[{"delta":{"content":"hello"},"index":0}]}"#);
        assert_eq!(*proc.phase(), StreamPhase::Active);
    }

    #[test]
    fn feed_done_triggers_draining() {
        let (mut proc, _rx) = make_processor();
        let result = proc.feed_sse_line("[DONE]");
        assert!(result.is_ok());
        assert_eq!(*proc.phase(), StreamPhase::Draining);
    }

    #[test]
    fn finish_assembly_transitions_to_completed() {
        let (mut proc, _rx) = make_processor();
        // Preamble → Active → Draining → ... → Completed
        proc.transition_to(StreamPhase::Active);
        proc.transition_to(StreamPhase::Draining);
        proc.finish_assembly();
        assert_eq!(*proc.phase(), StreamPhase::Completed);
    }

    #[test]
    fn cannot_feed_after_error() {
        let (mut proc, _rx) = make_processor();
        proc.transition_to(StreamPhase::Failed);
        // feed_sse_line should still work (it doesn't check phase on entry)
        // but shouldn't transition from Failed
        let _ = proc.feed_sse_line("{}");
        assert_eq!(*proc.phase(), StreamPhase::Failed);
    }

    #[test]
    fn preamble_then_feed_then_done_then_persist() {
        // Full happy-path phase sequence
        let (mut proc, _rx) = make_processor();
        assert_eq!(*proc.phase(), StreamPhase::Preamble);

        proc.feed_sse_line(r#"{"choices":[{"delta":{"content":"hello"},"index":0}]}"#).unwrap();
        assert_eq!(*proc.phase(), StreamPhase::Active);

        proc.feed_sse_line("[DONE]").unwrap();
        assert_eq!(*proc.phase(), StreamPhase::Draining);

        proc.flush_normalizer();
        proc.finish_assembly();
        assert_eq!(*proc.phase(), StreamPhase::Completed);
    }
}
