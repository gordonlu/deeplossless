//! # Protocol Adapters
//!
//! Provider-neutral adapters used by the legacy Chat Completions / Anthropic
//! compatibility path and by the execution/event normalization layer.
//!
//! ## Transport architecture
//!
//! DeepSeek's native Responses API no longer round-trips through this module:
//!
//! ```text
//! OpenAI Responses client
//!        ↓ native Responses transport
//! DeepSeek /responses
//!        ↓ native Responses SSE / JSON
//! OpenAI Responses client
//! ```
//!
//! The canonical adapters remain for the compatibility surface:
//!
//! ```text
//! DeepSeek Chat / Anthropic Messages / legacy Responses fallback
//!        ↓ parse → normalize → canonical IR
//! execution/runtime pipeline
//!        ↓ canonical IR → provider serialization
//! provider compatibility transport
//! ```
//!
//! `responses` is therefore a **legacy compatibility adapter**, used only when
//! an upstream has no native `/responses` endpoint (404) and in compatibility
//! tests/diagnostics. New native Responses features must not be implemented by
//! translating them through `CanonicalRequest` or synthetic Responses SSE.

pub mod anthropic;
pub mod canonical;
pub mod chat_completions;
pub mod dsml;
/// Legacy Responses ↔ canonical adapter. Native DeepSeek Responses bypasses it.
pub mod responses;
pub mod streaming;

pub use canonical::{
    CanonicalRequest, CanonicalResponse, ContentPart, DeepSeekNativeCapabilities,
    FinishReason, Message, MessageMeta, ProviderCapabilities, ReasoningEffort,
    ReasoningEffortMode, ReasoningMode, ResponseFormat, ResponseStatus, Role,
    StreamEvent, StructuredOutputMode, ToolDef, ToolInvocation, ToolStreamingMode, Usage,
};

pub use dsml::parse_dsml_tool_calls;

/// Provider capability registry: maps client model names to DeepSeek names.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    overrides: Vec<(String, String)>,
    prefixes: Vec<(String, String, String)>,
    default: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            overrides: vec![
                ("gpt-5.5".into(), "deepseek-v4-pro".into()),
            ],
            prefixes: vec![
                ("gpt-".into(), "deepseek-flash".into(), "deepseek-v4-pro".into()),
                ("o1".into(), "deepseek-flash".into(), "deepseek-v4-pro".into()),
                ("o3".into(), "deepseek-flash".into(), "deepseek-v4-pro".into()),
            ],
            // V4.1 Flash is now the recommended general model and also owns vision.
            default: "deepseek-flash".into(),
        }
    }
}

impl ModelRegistry {
    pub fn new(
        overrides: Vec<(String, String)>,
        prefixes: Vec<(String, String, String)>,
        default: String,
    ) -> Self {
        Self { overrides, prefixes, default }
    }

    /// Return provider capabilities for a model.
    pub fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let m = model.to_lowercase();
        if m == "deepseek-flash"
            || m == "deepseek-v4-flash"
            || m == "deepseek-v4-flash-vision-exp"
        {
            ProviderCapabilities {
                tool_streaming: ToolStreamingMode::Parallel,
                reasoning: ReasoningMode::Full,
                structured_output: StructuredOutputMode::JsonSchema,
                multimodal: true,
            }
        } else if m.contains("deepseek") {
            ProviderCapabilities {
                tool_streaming: ToolStreamingMode::Parallel,
                reasoning: ReasoningMode::Full,
                structured_output: StructuredOutputMode::JsonSchema,
                multimodal: false,
            }
        } else {
            ProviderCapabilities {
                tool_streaming: ToolStreamingMode::Parallel,
                reasoning: ReasoningMode::Hidden,
                structured_output: StructuredOutputMode::JsonSchema,
                multimodal: false,
            }
        }
    }

    pub fn resolve(&self, model: &str) -> (String, bool) {
        let m = model.trim().to_lowercase();
        if m.is_empty() || m == "auto" {
            return (self.default.clone(), true);
        }
        for (exact, replacement) in &self.overrides {
            if &m == exact {
                return (replacement.clone(), true);
            }
        }
        for (prefix, mini_target, pro_target) in &self.prefixes {
            if m.starts_with(prefix) {
                if m.contains("mini") {
                    return (mini_target.clone(), true);
                }
                return (pro_target.clone(), true);
            }
        }
        // Explicit DeepSeek model names, including deprecated-but-supported
        // aliases such as deepseek-v4-flash, are preserved verbatim. DeepSeek
        // owns their server-side routing to the current underlying model.
        (model.to_string(), false)
    }

    pub fn map_model(&self, model: &str) -> String {
        let (result, matched) = self.resolve(model);
        if !matched && model != result {
            tracing::warn!(target: "deeplossless::protocol",
                "unknown model '{}', using '{}' (no mapping rule matched)", model, result);
        }
        result
    }
}

pub fn map_model(model: &str) -> String {
    ModelRegistry::default().map_model(model)
}
