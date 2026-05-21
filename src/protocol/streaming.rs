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
pub fn to_responses_sse(event: &StreamEvent) -> String {
    format!(
        "data: {}\n\n",
        super::responses::stream_event_to_responses(event)
    )
}

/// Convert a canonical `StreamEvent` into a Chat Completions SSE data line.
pub fn to_chat_completions_sse(event: &StreamEvent) -> String {
    use serde_json::json;

    let data = match event {
        StreamEvent::TextDelta { text } => json!({
            "choices": [{"delta": {"content": text}, "index": 0}],
            "object": "chat.completion.chunk",
        }),
        StreamEvent::ToolCallDelta { index, id, name, arguments_delta } => {
            let mut delta = json!({
                "index": index,
                "function": {"arguments": arguments_delta},
            });
            if !id.is_empty() {
                delta["id"] = json!(id);
            }
            if let Some(n) = name {
                delta["function"]["name"] = json!(n);
            }
            json!({
                "choices": [{"delta": {"tool_calls": [delta]}, "index": 0}],
                "object": "chat.completion.chunk",
            })
        }
        StreamEvent::Done { usage, finish_reason } => {
            json!({
                "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "object": "chat.completion.chunk",
            })
        }
        StreamEvent::Error { message, code } => {
            json!({"error": {"message": message, "code": code}})
        }
        _ => json!({"choices": [{"delta": {}, "index": 0}]}),
    };

    format!("data: {}\n\n", data)
}
