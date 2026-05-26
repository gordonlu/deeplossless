//! Assistant message validation — checks protocol-critical invariants.
//!
//! `reasoning_content` is an **optional** protocol extension — DeepSeek does
//! not guarantee it on every tool-call assistant message.  It must NOT be
//! treated as a required invariant, or validation noise will mask real issues.
//!
//! Validation policy:
//!   missing content             → ERROR (protocol violation)
//!   missing tool_calls on tool  → ERROR
//!   missing reasoning           → DEBUG  (optional extension)
//!   invalid role sequence       → ERROR  (future)
//!   replay mismatch             → WARN   (future)

use serde_json::Value;

/// Fields that are protocol-critical on an assistant message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CriticalField {
    Content,
    ToolCalls,
    /// Present but advisory-only — DeepSeek does not guarantee reasoning_content
    /// on every tool-call assistant message.  Missing it is NOT an error.
    ReasoningContent,
}

impl CriticalField {
    /// Only Content is truly required. ToolCalls is self-proving, Reasoning is optional.
    pub fn required() -> &'static [CriticalField] {
        &[CriticalField::Content]
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

/// Check that an assistant message has all required protocol fields.
/// `reasoning_content` is checked but treated as advisory — missing it
/// does NOT make the message invalid.
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
    // reasoning_content is optional — advisory only, never invalidates
    if has_tool_calls && !has_reasoning {
        missing.push(CriticalField::ReasoningContent);
    }

    ValidationResult {
        valid: missing.is_empty() || missing == vec![CriticalField::ReasoningContent],
        missing_fields: missing,
        has_tool_calls,
        has_reasoning,
    }
}

/// Validate all assistant messages in a request body.
/// Per-message at debug (avoid log flood), aggregate at warn.
/// Returns the number of messages with truly critical issues (not advisory).
pub fn validate_request_messages(body: &Value) -> usize {
    let Some(messages) = body["messages"].as_array() else { return 0 };
    let mut invalid = 0;
    let mut advisory_only = 0u32;
    for msg in messages {
        if msg["role"].as_str() != Some("assistant") { continue; }
        let result = validate_assistant_message(msg);
        if !result.valid {
            invalid += 1;
            tracing::debug!(target: "deeplossless::validation",
                missing=?result.missing_fields,
                "assistant message missing critical field");
        } else if result.has_tool_calls && !result.has_reasoning {
            advisory_only += 1;
        }
    }
    if advisory_only > 0 {
        tracing::debug!(target: "deeplossless::validation",
            count = advisory_only,
            "assistant messages without reasoning_content (advisory)");
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
    fn tool_call_missing_reasoning_is_advisory() {
        let msg = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id":"c1","type":"function","function":{"name":"grep","arguments":"{}"}}]
        });
        let r = validate_assistant_message(&msg);
        // reasoning is optional — missing it doesn't invalidate
        assert!(r.valid);
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
        // Missing content is critical; missing reasoning is advisory only
        let body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id":"c1","type":"function","function":{"name":"g","arguments":"{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ]
        });
        assert_eq!(validate_request_messages(&body), 1);
    }

    #[test]
    fn missing_reasoning_not_counted_as_invalid() {
        // tool_calls + content but no reasoning → advisory, not invalid
        let body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [{"id":"c1","type":"function","function":{"name":"g","arguments":"{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ]
        });
        assert_eq!(validate_request_messages(&body), 0);
    }
}
