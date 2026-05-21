//! SSE streaming event normalization.
//! Converts provider-specific SSE formats to canonical `StreamEvent`.
//!
//! Currently supports: Chat Completions SSE → StreamEvent → Responses SSE.
//! Future: Anthropic SSE, MCP streaming, Gemini streaming.

pub use super::canonical::StreamEvent;

/// Convert a Chat Completions SSE data line into a canonical `StreamEvent`.
/// Returns `None` if the line doesn't contain a meaningful event (e.g., `[DONE]`).
pub fn from_chat_completions_sse(data: &str, usage_buffer: Option<&serde_json::Value>) -> Option<StreamEvent> {
    if data == "[DONE]" {
        // Use accumulated usage from previous events
        let usage = usage_buffer.map(|v| super::canonical::Usage {
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        }).unwrap_or_default();

        return Some(StreamEvent::Done {
            usage,
            finish_reason: "stop".into(),
        });
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
        StreamEvent::ToolCallArgsDelta { index, arguments_delta } => json!({
            "choices": [{"delta": {"tool_calls": [{"index": index, "function": {"arguments": arguments_delta}}]}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::ToolCallEnd { .. } => json!({"choices": [{"delta": {}, "index": 0}], "object": "chat.completion.chunk"}),
        StreamEvent::Done { usage, finish_reason } => json!({
            "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens},
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
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
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
                self.partial_tools.insert(index, PartialToolCall { id, name, arguments: String::new() });
                vec![]
            }
            StreamEvent::ToolCallArgsDelta { index, arguments_delta } => {
                if let Some(ptc) = self.partial_tools.get_mut(&index) {
                    ptc.arguments.push_str(&arguments_delta);
                }
                vec![]
            }
            StreamEvent::ToolCallEnd { index } => {
                let mut events = Vec::new();
                if let Some(ptc) = self.partial_tools.remove(&index) {
                    events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: ptc.arguments });
                }
                events
            }
            StreamEvent::Done { .. } | StreamEvent::MessageEnd => {
                self.partial_tools.clear();
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
    /// for the final lifecycle events. Also clears any partial tool state.
    pub fn finish(&mut self) -> AssembledContent {
        self.partial_tools.clear();
        AssembledContent {
            text: std::mem::take(&mut self.full_text),
            reasoning: std::mem::take(&mut self.full_reasoning),
        }
    }

    /// Force-flush any buffered partial tool calls (called on stream close).
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        self.partial_tools.clear();
        vec![]
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
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: r#"{"pa"#.into() }).is_empty());
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: r#"ttern":"foo"}"#.into() }).is_empty());
        let events = asm.feed(StreamEvent::ToolCallEnd { index: 0 });
        assert_eq!(events.len(), 1, "should emit 1 event with complete args");
        assert!(matches!(events[0], StreamEvent::ToolCallArgsDelta { .. }));
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
    fn incomplete_tool_dropped_on_done() {
        let mut asm = StreamAssembler::new();
        assert!(asm.feed(StreamEvent::ToolCallStart { index: 0, id: "tc1".into(), name: "grep".into() }).is_empty());
        assert!(asm.feed(StreamEvent::ToolCallArgsDelta { index: 0, arguments_delta: "{".into() }).is_empty());
        let events = asm.feed(StreamEvent::Done { usage: Usage::default(), finish_reason: "stop".into() });
        assert!(events.is_empty(), "incomplete tool dropped on Done");
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
