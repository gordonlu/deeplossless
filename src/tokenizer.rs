use std::sync::LazyLock;
use tiktoken::CoreBPE;

/// Lazily-initialized cl100k_base BPE tokenizer (used by DeepSeek V3/V4).
static BPE: LazyLock<CoreBPE> = LazyLock::new(|| {
    tiktoken::p50k_base().expect("failed to load p50k_base tokenizer")
});

/// Count tokens in a text string using cl100k_base encoding.
/// Used for:
///   - Accurate token accounting when storing messages
///   - DAG node budget calculations
///   - Context assembly threshold decisions
pub fn count(text: &str) -> usize {
    BPE.encode_with_special_tokens(text).len()
}

/// Estimate token count for a JSON `Value` (message content).
/// Handles both plain strings and structured content arrays.
pub fn count_content(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(s) => count(s),
        serde_json::Value::Array(arr) => arr.iter().map(count_content).sum(),
        serde_json::Value::Object(obj) => {
            let mut total = 0;
            for (_k, v) in obj {
                total += count_content(v);
            }
            total
        }
        _ => 0,
    }
}

/// Estimate token overhead of a message's JSON structure (role + framing).
pub const MESSAGE_OVERHEAD_TOKENS: usize = 12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_short_text() {
        let n = count("hello world");
        assert!(n > 0 && n < 10, "expected ~2 tokens, got {n}");
    }

    #[test]
    fn token_count_chinese() {
        let n = count("你好世界");
        assert!(n > 0, "expected >0 tokens for Chinese text, got {n}");
    }

    #[test]
    fn token_count_empty() {
        assert_eq!(count(""), 0);
    }
}
