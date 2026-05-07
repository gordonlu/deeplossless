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
        let a = fingerprint(&json!([{"role": "user", "content": "hello"}]).as_array().unwrap(), 1);
        let b = fingerprint(&json!([{"role": "user", "content": "world"}]).as_array().unwrap(), 1);
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
}
