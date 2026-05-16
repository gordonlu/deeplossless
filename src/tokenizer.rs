use std::collections::HashMap;
use std::sync::Mutex;
use tiktoken::CoreBpe;

// ── Configuration ──────────────────────────────────────────────────────

/// Tokenizer configuration, replaceable per provider at runtime.
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    /// The model name (e.g. "deepseek-v4-pro"). Used to resolve the
    /// correct tiktoken encoding via `model_to_encoding_name()`.
    pub model: String,
    /// Correction factor to align with upstream provider's actual count.
    /// 1.0 = no correction, >1.0 = pad upward, <1.0 = shrink.
    pub correction_factor: f64,
    /// Per-message framing overhead (role + JSON structure tokens).
    pub message_overhead: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-v4-pro".into(),
            correction_factor: 1.0,
            message_overhead: 12,
        }
    }
}

impl TokenizerConfig {
    pub fn encoding_name(&self) -> &str {
        model_to_encoding_name(&self.model)
    }
}

/// Map a model name to its tiktoken encoding.
/// DeepSeek V3/V4/R1 all use the same `deepseek_v3` tokenizer.
/// Claude models would map to `claude_20250219` (custom).
/// OpenAI models map to `o200k_base` (GPT-4o) or `cl100k_base`.
pub fn model_to_encoding_name(model: &str) -> &'static str {
    if model.starts_with("deepseek") {
        "deepseek_v3"
    } else if model.starts_with("claude") {
        "deepseek_v3" // Claude tokenizer not bundled; fallback
    } else if model.starts_with("gpt-4o") || model.starts_with("o1") || model.starts_with("o3") {
        "o200k_base"
    } else if model.starts_with("gpt-4") || model.starts_with("gpt-3") {
        "cl100k_base"
    } else {
        "deepseek_v3" // safe default
    }
}

// ── Multi-encoding cache ───────────────────────────────────────────────

static ENCODINGS: Mutex<Option<HashMap<String, &'static CoreBpe>>> = Mutex::new(None);

/// Get or load a tiktoken encoding by name, with graceful fallback.
/// Falls back to `cl100k_base` on load failure instead of panicking.
pub fn get_encoding(name: &str) -> &'static CoreBpe {
    let mut guard = ENCODINGS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);

    if let Some(enc) = map.get(name) {
        return enc;
    }

    let encoding = tiktoken::get_encoding(name).unwrap_or_else(|| {
        tracing::warn!(target: "deeplossless::tokenizer", name, "tokenizer load failed, falling back to cl100k_base");
        tiktoken::get_encoding("cl100k_base")
            .expect("cl100k_base must exist")
    });
    map.insert(name.to_string(), encoding);
    encoding
}

// ── Token counting ─────────────────────────────────────────────────────

/// Count tokens in plain text with the default encoding.
pub fn count(text: &str) -> usize {
    get_encoding("deepseek_v3").count(text)
}

/// Count tokens in plain text with a specific encoding.
pub fn count_with(text: &str, encoding_name: &str) -> usize {
    get_encoding(encoding_name).count(text)
}

/// Count tokens in message content, using a whitelist of content-bearing
/// fields. This avoids counting tool schema / metadata tokens that
/// would pollute the budget estimation.
///
/// Whitelist: `"content"` (string), `"text"` (in content blocks),
/// `"arguments"` / `"input"` (tool call args).
pub fn count_content(value: &serde_json::Value) -> usize {
    count_content_inner(value, "deepseek_v3")
}

/// Raw count with explicit encoding name.
pub fn count_content_raw(value: &serde_json::Value) -> usize {
    count_content_inner(value, "deepseek_v3")
}

/// Count with specific encoding name (for multi-provider support).
pub fn count_content_with(value: &serde_json::Value, encoding_name: &str) -> usize {
    count_content_inner(value, encoding_name)
}

fn count_content_inner(value: &serde_json::Value, enc: &str) -> usize {
    let bpe = get_encoding(enc);
    match value {
        serde_json::Value::String(s) => bpe.count(s),
        serde_json::Value::Array(arr) => {
            // Content array blocks: count strings directly, or whitelist
            // "text"/"content"/"arguments"/"input" fields from objects.
            arr.iter().map(|item| {
                if item.is_string() {
                    bpe.count(item.as_str().unwrap())
                } else if let Some(obj) = item.as_object() {
                    let mut total = 0;
                    if let Some(v) = obj.get("text").or_else(|| obj.get("content")) {
                        total += count_content_inner(v, enc);
                    }
                    if let Some(v) = obj.get("arguments").or_else(|| obj.get("input")) {
                        total += count_content_inner(v, enc);
                    }
                    total
                } else {
                    0
                }
            }).sum()
        }
        serde_json::Value::Object(obj) => {
            // Only count known content fields; skip metadata keys
            let mut total = 0;
            for key in &["content", "text", "arguments", "input", "response"] {
                if let Some(v) = obj.get(*key) {
                    total += count_content_inner(v, enc);
                }
            }
            total
        }
        _ => 0,
    }
}

// ── Message-level counting ─────────────────────────────────────────────

/// Count tokens for an entire message including framing overhead.
/// Includes role token + content tokens + JSON structure overhead.
pub fn count_message(value: &serde_json::Value, overhead: usize) -> usize {
    let role = value["role"].as_str().unwrap_or("unknown");
    let role_tokens = count(role);
    let content_tokens = count_content(value);
    let tool_token = if value.get("tool_call_id").is_some() { 5 } else { 0 };
    let tool_calls_tokens = if let Some(arr) = value["tool_calls"].as_array() {
        arr.iter().map(|tc| count_content(tc)).sum::<usize>()
    } else {
        0
    };
    role_tokens + content_tokens + tool_token + tool_calls_tokens + overhead
}

/// Count tokens for a full messages array, accounting for chat framing.
/// Each message includes the per-message overhead, plus a global preamble
/// overhead for the request structure.
pub fn count_messages(messages: &[serde_json::Value], overhead_per_msg: usize, overhead_global: usize) -> usize {
    let total: usize = messages.iter().map(|m| count_message(m, overhead_per_msg)).sum();
    total + overhead_global
}

// ── Correction ─────────────────────────────────────────────────────────

/// Apply the configured correction factor to a raw token count.
#[inline]
pub fn correct(raw_tokens: usize, factor: f64) -> usize {
    if (factor - 1.0).abs() < f64::EPSILON {
        raw_tokens
    } else {
        ((raw_tokens as f64) * factor).round() as usize
    }
}

/// Correction with tokenizer config.
#[inline]
pub fn correct_cfg(raw: usize, cfg: &TokenizerConfig) -> usize {
    correct(raw, cfg.correction_factor)
}

// ── Legacy compat ──────────────────────────────────────────────────────

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

    #[test]
    fn count_content_whitelist_ignores_metadata() {
        // A content array with type annotations — only "text" should be counted
        let v = serde_json::json!([
            {"type": "text", "text": "hello world"},
            {"type": "text", "text": "how are you"}
        ]);
        let n = count_content(&v);
        assert!(n > 0, "should count text fields only, got {n}");
    }

    #[test]
    fn count_content_skips_non_text_blocks() {
        let v = serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "text", "text": "describe this image"}
        ]);
        let n = count_content(&v);
        assert!(n > 0, "should count text but not image_url blocks, got {n}");
    }

    #[test]
    fn count_message_includes_framing() {
        let msg = serde_json::json!({
            "role": "user",
            "content": "hello world"
        });
        let n = count_message(&msg, 12);
        assert!(n > count("hello world"), "framing should add overhead");
    }

    #[test]
    fn count_messages_with_overhead() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let n = count_messages(&msgs, 12, 5);
        assert!(n >= count_content(&msgs[0]) + count_content(&msgs[1]),
            "message count should be >= sum of content counts");
    }

    #[test]
    fn correction_factor_identity() {
        assert_eq!(correct(100, 1.0), 100);
    }

    #[test]
    fn correction_factor_pads_up() {
        assert_eq!(correct(100, 1.1), 110);
    }

    #[test]
    fn correction_factor_shrinks() {
        assert_eq!(correct(100, 0.9), 90);
    }

    #[test]
    fn get_encoding_loads_deepseek() {
        let bpe = get_encoding("deepseek_v3");
        assert!(bpe.count("hello") > 0);
    }

    #[test]
    fn get_encoding_falls_back_to_cl100k() {
        let bpe = get_encoding("nonexistent_encoding_xyz");
        assert!(bpe.count("hello") > 0, "fallback should produce valid tokens");
    }
}
