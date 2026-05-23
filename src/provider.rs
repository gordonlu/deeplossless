//! Provider abstraction layer — decouples execution core from specific
//! LLM API implementations (D-4).
//!
//! The [`LlmProvider`] trait defines the contract for LLM communication.
//! Concrete implementations handle provider-specific authentication,
//! URL construction, and response normalization.

/// Abstract interface for LLM API communication.
pub trait LlmProvider: Send + Sync {
    /// Build the chat completion URL for this provider.
    fn chat_url(&self) -> String;

    /// Build the Authorization header value.
    fn auth_header(&self) -> String;

    /// Normalize a raw provider response into a standard content string.
    /// Returns None if the response has no usable content.
    fn extract_content(&self, raw: &serde_json::Value) -> Option<String>;

    /// Check if an HTTP status code is a rate-limit (429) response.
    fn is_rate_limited(&self, status: u16) -> bool {
        status == 429
    }
}

/// OpenAI-compatible provider (DeepSeek, vLLM, etc.).
pub struct OpenAiCompatibleProvider {
    pub api_url: String,
    pub api_key: String,
    pub model: String,
}

impl LlmProvider for OpenAiCompatibleProvider {
    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.api_url.trim_end_matches('/'))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    fn extract_content(&self, raw: &serde_json::Value) -> Option<String> {
        raw["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
    }
}
