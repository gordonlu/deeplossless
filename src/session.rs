/// Normalized message: a universal representation that works across
/// OpenAI / DeepSeek / Claude tool schemas.
///
/// Tool calls are extracted from their schema-specific locations into
/// a unified `Vec<NormalizedToolCall>` so downstream logic (e.g.
/// `safe_split_points`) can reason about tool chains by role alone,
/// without fragile content heuristics.
#[derive(Debug, Clone)]
pub struct NormalizedMessage {
    pub role: String,
    pub content: String,
    /// Unified tool calls extracted from any schema variant.
    pub tool_calls: Vec<NormalizedToolCall>,
    /// tool_call_id for tool result messages.
    pub tool_call_id: Option<String>,
}

/// Unified tool call — extracted from assistant messages.
#[derive(Debug, Clone)]
pub struct NormalizedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Normalize a raw JSON message value from any API schema.
pub fn normalize_message(msg: &serde_json::Value) -> NormalizedMessage {
    let role = msg["role"].as_str().unwrap_or("unknown").to_string();
    let content = msg["content"].to_string();
    let tool_call_id = msg["tool_call_id"].as_str().map(|s| s.to_string());

    // Extract tool calls from multiple schema variants
    let mut tool_calls = Vec::new();

    // OpenAI/DeepSeek: assistant.tool_calls[]
    if let Some(calls) = msg["tool_calls"].as_array() {
        for tc in calls {
            if let (Some(id), Some(func)) = (tc["id"].as_str(), tc["function"].as_object()) {
                tool_calls.push(NormalizedToolCall {
                    id: id.to_string(),
                    name: func.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    arguments: func.get("arguments").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                });
            }
        }
    }

    // Claude: content blocks with type "tool_use" / "tool_result"
    if let Some(blocks) = msg["content"].as_array() {
        for block in blocks {
            match block["type"].as_str() {
                Some("tool_use") => {
                    if let Some(id) = block["id"].as_str() {
                        tool_calls.push(NormalizedToolCall {
                            id: id.to_string(),
                            name: block["name"].as_str().unwrap_or("").to_string(),
                            arguments: block["input"].to_string(),
                        });
                    }
                }
                Some("tool_result") => {}
                _ => {}
            }
        }
    }

    // Gemini: parts[].functionCall / parts[].functionResponse
    if let Some(parts) = msg["parts"].as_array() {
        for part in parts {
            if let Some(fc) = part["functionCall"].as_object() {
                tool_calls.push(NormalizedToolCall {
                    id: fc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    name: fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    arguments: fc.get("args").map(|v| v.to_string()).unwrap_or_default(),
                });
            }
        }
    }

    NormalizedMessage { role, content, tool_calls, tool_call_id }
}

/// Check if a normalized message represents a tool call.
pub fn is_tool_call(msg: &NormalizedMessage) -> bool {
    !msg.tool_calls.is_empty()
        || msg.role == "assistant" && msg.content.contains("tool_use")
        || msg.role == "model" && msg.content.contains("functionCall")
}

/// Check if a normalized message represents a tool result.
pub fn is_tool_result(msg: &NormalizedMessage) -> bool {
    msg.role == "tool" || msg.tool_call_id.is_some()
        || msg.role == "user" && msg.content.contains("tool_result")
        || msg.role == "function"
        || msg.role == "tool" && msg.content.contains("functionResponse")
}

/// Generate a stable session fingerprint from a messages array.
///
/// Uses the first `prefix_count` messages (typically system prompt + first
/// user turn) to create a hash that identifies the conversation across
/// requests.  This enables the proxy to associate multi-turn requests with
/// the same DAG conversation.
///
/// The fingerprint is the first 16 hex chars of SHA-256, computed over
/// the concatenation of `(role, content)` pairs for the prefix messages.
pub fn fingerprint(messages: &[serde_json::Value], prefix_count: usize) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for msg in messages.iter().take(prefix_count) {
        if let Some(role) = msg["role"].as_str() {
            hasher.update(role.as_bytes());
        }
        if let Some(content) = msg["content"].as_str() {
            hasher.update(content.as_bytes());
        } else if let Some(arr) = msg["content"].as_array() {
            for block in arr {
                if let Some(text) = block["text"].as_str() {
                    hasher.update(text.as_bytes());
                }
            }
        }
    }
    let result = hasher.finalize();
    hex::encode(&result[..8]) // first 8 bytes → 16 hex chars
}

/// Extract the model name from a request body.
pub fn model_name(body: &serde_json::Value) -> &str {
    body["model"].as_str().unwrap_or("unknown")
}

/// Check whether a request is streaming.
pub fn is_streaming(body: &serde_json::Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fingerprint_is_stable() {
        let msgs = json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"}
        ]);
        let a = fingerprint(msgs.as_array().unwrap(), 2);
        let b = fingerprint(msgs.as_array().unwrap(), 2);
        assert_eq!(a, b, "same input must produce same fingerprint");
    }

    #[test]
    fn fingerprint_differs_for_different_content() {
        let a = fingerprint(json!([{"role": "user", "content": "hello"}]).as_array().unwrap(), 1);
        let b = fingerprint(json!([{"role": "user", "content": "world"}]).as_array().unwrap(), 1);
        assert_ne!(a, b, "different content must differ");
    }

    #[test]
    fn fingerprint_handles_content_array() {
        let msgs = json!([{
            "role": "user",
            "content": [{"type": "text", "text": "hello"}]
        }]);
        let result = fingerprint(msgs.as_array().unwrap(), 1);
        assert_eq!(result.len(), 16, "hex digest should be 16 chars");
    }

    #[test]
    fn fingerprint_empty_messages() {
        let result = fingerprint(&[], 5);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn is_streaming_true() {
        let body = json!({"stream": true});
        assert!(is_streaming(&body));
    }

    #[test]
    fn is_streaming_false_by_default() {
        let body = json!({"model": "deepseek"});
        assert!(!is_streaming(&body));
    }

    // ── NormalizedMessage tests ─────────────────────────────────────────

    #[test]
    fn normalize_openai_tool_call() {
        let raw = json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"main.rs\"}"
                }
            }]
        });
        let msg = normalize_message(&raw);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "read_file");
        assert!(is_tool_call(&msg));
    }

    #[test]
    fn normalize_tool_result_by_role() {
        let raw = json!({
            "role": "tool",
            "content": "file content",
            "tool_call_id": "call_123"
        });
        let msg = normalize_message(&raw);
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
        assert!(is_tool_result(&msg));
    }

    #[test]
    fn normalize_plain_user_message() {
        let raw = json!({"role": "user", "content": "hello"});
        let msg = normalize_message(&raw);
        assert_eq!(msg.role, "user");
        assert!(msg.tool_calls.is_empty());
        assert!(msg.tool_call_id.is_none());
        assert!(!is_tool_call(&msg));
        assert!(!is_tool_result(&msg));
    }

    #[test]
    fn normalize_claude_tool_use() {
        let raw = json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": "ls"}
            ]
        });
        let msg = normalize_message(&raw);
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "bash");
        assert!(is_tool_call(&msg));
    }

    #[test]
    fn normalize_gemini_function_call() {
        let raw = json!({
            "role": "model",
            "parts": [
                {"text": "I'll look that up"},
                {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}}
            ]
        });
        let msg = normalize_message(&raw);
        assert_eq!(msg.role, "model");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert!(is_tool_call(&msg));
    }

    #[test]
    fn normalize_gemini_function_response() {
        let raw = json!({
            "role": "function",
            "parts": [
                {"functionResponse": {"name": "get_weather", "response": {"temp": 22}}}
            ]
        });
        let msg = normalize_message(&raw);
        assert_eq!(msg.role, "function");
        assert!(is_tool_result(&msg));
    }
}
