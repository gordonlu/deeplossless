//! SSE streaming event normalization.
//! Converts provider-specific SSE formats to canonical `StreamEvent`.
//!
//! Currently supports: Chat Completions SSE → StreamEvent → Responses SSE.
//! Future: Anthropic SSE, MCP streaming, Gemini streaming.

pub use super::canonical::StreamEvent;

/// Convert a Chat Completions SSE data line into canonical `StreamEvent`s.
/// Returns empty Vec if the line contains no meaningful event.
/// The first chunk of a tool call may return both ToolCallStart + ToolCallArgsDelta.
pub fn from_chat_completions_sse(data: &str, usage_buffer: Option<&serde_json::Value>) -> Vec<StreamEvent> {
    if data == "[DONE]" {
        let usage = usage_buffer.map(|v| super::canonical::Usage {
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        }).unwrap_or_default();

        return vec![StreamEvent::Done {
            usage,
            finish_reason: "stop".into(),
            incomplete: false,
            error_reason: None,
        }];
    }

    super::responses::stream_event_from_chat(data)
}

/// Convert a canonical `StreamEvent` into a Responses API SSE data line.
/// Codex requires `event: <type>\ndata: {...}\n\n` format.
/// Escapes newlines in data to prevent SSE injection.
pub fn to_responses_sse(event: &StreamEvent) -> String {
    let data = super::responses::stream_event_to_responses(event);
    if data.is_empty() {
        return String::new();
    }
    // SSE-safe: escape newlines in the data payload
    let safe_data = data.replace('\n', "\\n").replace('\r', "\\r");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&safe_data) {
        if let Some(evt) = v["type"].as_str() {
            return format!("event: {evt}\ndata: {safe_data}\n\n");
        }
    }
    format!("data: {safe_data}\n\n")
}

/// Convert a canonical `StreamEvent` into a Chat Completions SSE data line.
pub fn to_chat_completions_sse(event: &StreamEvent) -> String {
    use serde_json::json;

    let data = match event {
        StreamEvent::TextDelta { text } => json!({
            "choices": [{"delta": {"content": text}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::ToolCallStart { index, id, name } => json!({
            "choices": [{"delta": {"tool_calls": [{"index": index, "id": id, "function": {"name": name, "arguments": ""}, "type": "function"}]}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::ToolCallArgsDelta { index, arguments_delta, .. } => json!({
            "choices": [{"delta": {"tool_calls": [{"index": index, "function": {"arguments": arguments_delta}}]}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::ToolCallEnd { .. } => json!({"choices": [{"delta": {}, "index": 0}], "object": "chat.completion.chunk"}),
        StreamEvent::Done { usage, finish_reason, incomplete, error_reason: _ } => {
            let reason = if *incomplete { "length" } else { finish_reason.as_str() };
            json!({
                "choices": [{"delta": {}, "index": 0, "finish_reason": reason}],
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens},
                "object": "chat.completion.chunk",
            })
        },
        StreamEvent::ReasoningDelta { text } => json!({
            "choices": [{"delta": {"reasoning_content": text}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::Error { message, code } => json!({"error": {"message": message, "code": code}}),
        _ => json!({"choices": [{"delta": {}, "index": 0}]}),
    };

    format!("data: {}\n\n", data)
}

// ── Stream Assembler ───────────────────────────────────────────────────

use std::collections::HashMap;

/// Partially assembled tool call during streaming.
#[derive(Debug, Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Stateful stream assembler. Buffers partial tool call arguments until
/// ToolCallEnd emits the complete args delta, and accumulates text for the
/// final lifecycle events (output_text.done, content_part.done, etc.).
///
/// Text and reasoning deltas pass through immediately for real-time streaming,
/// with the full text accumulated for the final content envelope.
///
/// Lifecycle:
///   feed(events) → emit events as they become ready
///   upstream [DONE] → finish() → lifecycle events with full text
///   emit [DONE], close stream
pub struct StreamAssembler {
    partial_tools: HashMap<usize, PartialToolCall>,
    /// Accumulated text from all TextDelta events, used in final lifecycle.
    full_text: String,
    /// Accumulated reasoning text from ReasoningDelta events.
    full_reasoning: String,
}

impl StreamAssembler {
    pub fn new() -> Self {
        Self { partial_tools: HashMap::new(), full_text: String::new(), full_reasoning: String::new() }
    }

    /// Feed a raw SSE event into the assembler.
    /// Returns events ready for downstream consumption.
    /// Text/reasoning/lifecycle events pass through immediately;
    /// tool call deltas are buffered until ToolCallEnd.
    pub fn feed(&mut self, event: StreamEvent) -> Vec<StreamEvent> {
        match event {
            StreamEvent::TextDelta { ref text } => {
                self.full_text.push_str(text);
                vec![event]
            }
            StreamEvent::ReasoningDelta { ref text } => {
                self.full_reasoning.push_str(text);
                vec![event]
            }
            StreamEvent::ToolCallStart { index, id, name } => {
                // Tolerant assembly: name/id may arrive in later chunk.
                // Merge with existing entry rather than overwriting.
                let entry = self.partial_tools.entry(index).or_insert(PartialToolCall { id: String::new(), name: String::new(), arguments: String::new() });
                if !id.is_empty() { entry.id = id; }
                if !name.is_empty() { entry.name = name; }
                vec![]
            }
            StreamEvent::ToolCallArgsDelta { index, arguments_delta, .. } => {
                // Tolerant assembly: args delta may arrive before ToolCallStart.
                // Create partial entry on first args delta even without start.
                let ptc = self.partial_tools.entry(index).or_insert(PartialToolCall { id: String::new(), name: String::new(), arguments: String::new() });
                ptc.arguments.push_str(&arguments_delta);
                vec![]
            }
            StreamEvent::ToolCallEnd { index } => {
                let mut events = Vec::new();
                if let Some(ptc) = self.partial_tools.remove(&index) {
                    events.push(StreamEvent::ToolCallStart { index, id: ptc.id.clone(), name: ptc.name.clone() });
                    events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: ptc.arguments.clone(), call_id: ptc.id.clone() });
                }
                events.push(StreamEvent::ToolCallEnd { index });
                events
            }
            StreamEvent::Done { .. } | StreamEvent::MessageEnd => {
                // Don't clear partial_tools — finish()/flush() recovers them
                vec![]
            }
            StreamEvent::Error { .. } => {
                self.partial_tools.clear();
                self.full_text.clear();
                self.full_reasoning.clear();
                vec![event]
            }
            // All other events pass through immediately
            _ => vec![event],
        }
    }

    /// Called when upstream [DONE] arrives. Returns accumulated text content
    /// for the final lifecycle events and any remaining partial tool calls.
    /// Unlike flush(), this consumes state (takes text/reasoning).
    pub fn finish(&mut self) -> (AssembledContent, Vec<StreamEvent>) {
        let events = self.flush();
        (AssembledContent {
            text: std::mem::take(&mut self.full_text),
            reasoning: std::mem::take(&mut self.full_reasoning),
        }, events)
    }

    /// Graceful flush — completes any partially assembled tool calls and
    /// emits the full lifecycle: [ToolCallStart, ToolCallArgsDelta,
    /// FunctionCallArgumentsDone, OutputItemDone].
    /// Called when the upstream stream ends normally (Done/MessageEnd received).
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let mut indices: Vec<usize> = self.partial_tools.keys().copied().collect();
        indices.sort();
        for &index in &indices {
            if let Some(ptc) = self.partial_tools.remove(&index) {
                events.push(StreamEvent::ToolCallStart { index, id: ptc.id.clone(), name: ptc.name.clone() });
                events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: ptc.arguments.clone(), call_id: ptc.id.clone() });
                events.push(StreamEvent::FunctionCallArgumentsDone {
                    call_id: ptc.id.clone(),
                    name: ptc.name.clone(),
                    arguments: ptc.arguments.clone(),
                    output_index: index,
                });
                events.push(StreamEvent::OutputItemDone { index, item_id: ptc.id.clone(), item_type: "function_call".into(), name: ptc.name.clone(), arguments: ptc.arguments.clone() });
            }
        }
        events
    }

    /// Abort flush — discards partial tool call state without emitting.
    /// Called on transport error, where partial results are unsafe to use.
    pub fn abort_flush(&mut self) {
        self.partial_tools.clear();
    }
}

/// Content accumulated by the assembler, used to populate final lifecycle events.
pub struct AssembledContent {
    pub text: String,
    pub reasoning: String,
}

impl Default for StreamAssembler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod assembler_tests {
    use super::*;
    use super::super::canonical::Usage;

    #[test]
    fn assembles_tool_call_from_deltas() {
        let mut asm = StreamAssembler::new();
        assert!(asm.feed(StreamEvent::ToolCallStart { index: 0, id: "tc1".into(), name: "grep".into() }).is_empty());
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: r#"{"pa"#.into(), call_id: "tc1".into() }).is_empty());
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: r#"ttern":"foo"}"#.into(), call_id: "tc1".into() }).is_empty());
        let events = asm.feed(StreamEvent::ToolCallEnd { index: 0 });
        assert_eq!(events.len(), 3, "should emit ToolCallStart + complete ToolCallArgsDelta + ToolCallEnd");
        assert!(matches!(events[0], StreamEvent::ToolCallStart { .. }));
        assert!(matches!(events[1], StreamEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(events[2], StreamEvent::ToolCallEnd { index: 0 }));
    }

    #[test]
    fn text_passes_through_immediately() {
        let mut asm = StreamAssembler::new();
        let events = asm.feed(StreamEvent::TextDelta { text: "hello ".into() });
        assert_eq!(events.len(), 1, "text should pass through immediately");
        let events = asm.feed(StreamEvent::TextDelta { text: "world".into() });
        assert_eq!(events.len(), 1, "text should pass through immediately");
    }

    #[test]
    fn incomplete_tool_recovered_on_finish() {
        let mut asm = StreamAssembler::new();
        assert!(asm.feed(StreamEvent::ToolCallStart { index: 0, id: "tc1".into(), name: "grep".into() }).is_empty());
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: r#"{"pa"#.into(), call_id: "tc1".into() }).is_empty());
        // Done arrives before ToolCallEnd — partial tool should survive in assembler
        let done_events = asm.feed(StreamEvent::Done { usage: Usage::default(), finish_reason: "stop".into(), incomplete: false, error_reason: None });
        assert!(done_events.is_empty(), "Done passes through without clearing");
        // finish()/flush() should recover the partial tool
        let (_content, events) = asm.finish();
        assert_eq!(events.len(), 4, "flush emits ToolCallStart + ToolCallArgsDelta + FunctionCallArgumentsDone + OutputItemDone");
        assert!(matches!(events[0], StreamEvent::ToolCallStart { index: 0, .. }));
        assert!(matches!(events[1], StreamEvent::ToolCallArgsDelta { index: 0, .. }));
        assert!(matches!(events[2], StreamEvent::FunctionCallArgumentsDone { .. }));
        assert!(matches!(events[3], StreamEvent::OutputItemDone { .. }));
    }

    #[test]
    fn error_clears_state() {
        let mut asm = StreamAssembler::new();
        asm.feed(StreamEvent::ToolCallStart { index: 0, id: "tc1".into(), name: "grep".into() });
        let events = asm.feed(StreamEvent::Error { message: "timeout".into(), code: None });
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Error { .. }));
    }
}
