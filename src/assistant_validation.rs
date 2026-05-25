//! Assistant message validation — ensures reconstructed/forwarded messages
//! carry all runtime-critical fields before reaching the upstream API.
//!
//! DeepSeek thinking mode requires: content, tool_calls, reasoning_content.
//! Missing any of these in tool-call scenarios causes 400 errors or
//! silent reasoning corruption.

use serde_json::Value;

/// Fields that must be present on an assistant message for protocol correctness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CriticalField {
    Content,
    ToolCalls,
    ReasoningContent,
}

impl CriticalField {
    pub fn all() -> &'static [CriticalField] {
        &[CriticalField::Content, CriticalField::ToolCalls, CriticalField::ReasoningContent]
    }

    /// Fields required when the message has tool calls (thinking mode).
    pub fn for_tool_call() -> &'static [CriticalField] {
        &[CriticalField::Content, CriticalField::ToolCalls, CriticalField::ReasoningContent]
    }
}

/// Result of validating an assistant message.
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub missing_fields: Vec<CriticalField>,
    pub has_tool_calls: bool,
    pub has_reasoning: bool,
}

/// Check that an assistant message has all critical fields.
/// For tool-call messages, reasoning_content is mandatory.
pub fn validate_assistant_message(msg: &Value) -> ValidationResult {
    let has_tool_calls = msg.get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let has_reasoning = msg.get("reasoning_content")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let has_content = msg.get("content").is_some();

    let mut missing = Vec::new();
    if !has_content {
        missing.push(CriticalField::Content);
    }
    if has_tool_calls && !has_reasoning {
        missing.push(CriticalField::ReasoningContent);
    }
    if has_tool_calls {
        // tool_calls itself is present by definition if has_tool_calls is true
    }

    ValidationResult {
        valid: missing.is_empty(),
        missing_fields: missing,
        has_tool_calls,
        has_reasoning,
    }
}

/// Validate all assistant messages in a request body.
/// Returns the number of invalid messages found (0 = all valid).
pub fn validate_request_messages(body: &Value) -> usize {
    let Some(messages) = body["messages"].as_array() else { return 0 };
    let mut invalid = 0;
    for msg in messages {
        if msg["role"].as_str() != Some("assistant") { continue; }
        let result = validate_assistant_message(msg);
        if !result.valid {
            invalid += 1;
            tracing::warn!(target: "deeplossless::validation",
                missing=?result.missing_fields,
                has_tool_calls=result.has_tool_calls,
                has_reasoning=result.has_reasoning,
                "assistant message missing critical fields");
        }
    }
    invalid
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_tool_call_message() {
        let msg = json!({
            "role": "assistant",
            "content": null,
            "reasoning_content": "I need to search.",
            "tool_calls": [{"id":"c1","type":"function","function":{"name":"grep","arguments":"{}"}}]
        });
        let r = validate_assistant_message(&msg);
        assert!(r.valid, "missing: {:?}", r.missing_fields);
    }

    #[test]
    fn tool_call_missing_reasoning() {
        let msg = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id":"c1","type":"function","function":{"name":"grep","arguments":"{}"}}]
        });
        let r = validate_assistant_message(&msg);
        assert!(!r.valid);
        assert!(r.missing_fields.contains(&CriticalField::ReasoningContent));
    }

    #[test]
    fn text_only_message_is_valid() {
        let msg = json!({
            "role": "assistant",
            "content": "Hello"
        });
        let r = validate_assistant_message(&msg);
        assert!(r.valid);
    }

    #[test]
    fn validate_request_finds_invalid_messages() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [{"id":"c1","type":"function","function":{"name":"g","arguments":"{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ]
        });
        assert_eq!(validate_request_messages(&body), 1);
    }
}
