use std::sync::OnceLock;
use tiktoken::CoreBpe;

static BPE: OnceLock<&CoreBpe> = OnceLock::new();

fn encoding() -> &'static CoreBpe {
    *BPE.get_or_init(|| {
        tiktoken::get_encoding("deepseek_v3")
            .expect("failed to load deepseek_v3 tokenizer")
    })
}

/// Count tokens in a text string using DeepSeek V3/V4 tokenizer.
pub fn count(text: &str) -> usize {
    encoding().count(text)
}

/// Count tokens in a JSON `Value` (message content).
/// Handles plain strings, arrays (multi-block content), and nested structures.
pub fn count_content(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(s) => count(s),
        serde_json::Value::Array(arr) => arr.iter().map(count_content).sum(),
        serde_json::Value::Object(obj) => obj.values().map(count_content).sum(),
        _ => 0,
    }
}

/// Estimated token overhead per message for JSON framing (role + structure).
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

    #[test]
    fn count_content_handles_string() {
        let v = serde_json::json!("hello");
        assert!(count_content(&v) > 0);
    }

    #[test]
    fn count_content_handles_array() {
        let v = serde_json::json!(["hello", "world"]);
        assert!(count_content(&v) > 0);
    }
}
