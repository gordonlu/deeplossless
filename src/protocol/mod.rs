//! # Protocol Adapters
//!
//! Provider-neutral, model-neutral, transport-neutral translation layer.
//!
//! ## Architecture
//!
//! ```text
//! OpenAI Responses / DeepSeek Chat / Anthropic Messages
//!         ↓ adapter (dumb: parse → normalize → canonical IR)
//!   ExecutionState { steps, artifacts, plan, failures }
//!         ↓ runtime pipeline (DAG context, cache, failure detection)
//!   ExecutionState (updated)
//!         ↓ adapter (dumb: canonical IR → serialize to provider format)
//! OpenAI Responses / DeepSeek Chat / Anthropic Messages
//! ```
//!
//! ## Design rules
//!
//! 1. **IR is execution-centric, not message-centric.** `Message` is a
//!    serialization artifact of provider adapters. The core runtime operates
//!    on `ExecutionStep`, `ExecutionArtifact`, `PlanState`, `FailurePattern`.
//! 2. **Provider adapters are dumb.** Only parse, normalize, serialize.
//!    No business logic, no caching, no policy decisions.
//! 3. **Provider payload never enters core runtime.** All provider-specific
//!    fields are stripped during normalization.
//! 4. **Tool results are first-class runtime objects.** Not just "assistant
//!    message with tool content." This keeps cache/replay/provenance clean.
//! 5. **Reasoning is a distinct artifact type, not text.** Enables future
//!    distill/compact/reuse/invalidate operations on reasoning traces.
//! 6. **Streaming is fully abstracted.** `StreamEvent` enum covers text
//!    deltas, tool call deltas, reasoning deltas, and completion.

pub mod anthropic;
pub mod canonical;
pub mod chat_completions;
pub mod dsml;
pub mod responses;
pub mod streaming;

// Re-export the core types for convenience.
pub use canonical::{
    CanonicalRequest,
    CanonicalResponse,
    ContentPart,
    Message,
    MessageMeta,
    ResponseFormat,
    ResponseStatus,
    Role,
    StreamEvent,
    ToolDef,
    ToolInvocation,
    Usage,
    ProviderCapabilities,
    ToolStreamingMode,
    ReasoningMode,
    StructuredOutputMode,
    FinishReason,
    ReasoningEffort,
    ReasoningEffortMode,
    DeepSeekNativeCapabilities,
};

pub use dsml::parse_dsml_tool_calls;

/// Provider capability registry: maps upstream model names to local equivalents.
/// Explicit overrides take priority over prefix rules. Falls back to identity.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    /// Exact model name → local model (e.g. "gpt-5.5" → "deepseek-v4-pro")
    overrides: Vec<(String, String)>,
    /// Ordered prefix rules: (prefix, mini_suffix → local_flash, local_pro)
    prefixes: Vec<(String, String, String)>,
    /// Default model when no rule matches (e.g. "deepseek-v4-pro")
    default: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            overrides: vec![
                ("gpt-5.5".into(), "deepseek-v4-pro".into()),
            ],
            prefixes: vec![
                ("gpt-".into(), "deepseek-v4-flash".into(), "deepseek-v4-pro".into()),
                ("o1".into(), "deepseek-v4-flash".into(), "deepseek-v4-pro".into()),
                ("o3".into(), "deepseek-v4-flash".into(), "deepseek-v4-pro".into()),
            ],
            default: "deepseek-v4-pro".into(),
        }
    }
}

impl ModelRegistry {
    pub fn new(overrides: Vec<(String, String)>, prefixes: Vec<(String, String, String)>, default: String) -> Self {
        Self { overrides, prefixes, default }
    }

    /// Resolve an upstream model name to the local equivalent.
    /// Returns `(local_model, matched)` where `matched` is true if a rule applied.
    /// Return provider capabilities for a model. All DeepSeek V4 models support
    /// reasoning/thinking — required for correct replay and protocol assertions.
    pub fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let m = model.to_lowercase();
        if m.contains("deepseek") {
            ProviderCapabilities {
                tool_streaming: ToolStreamingMode::Parallel,
                reasoning: ReasoningMode::Full,
                structured_output: StructuredOutputMode::JsonSchema,
                multimodal: false,
            }
        } else {
            // Generic OpenAI-compatible — conservative defaults
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
        // 1. Exact overrides
        for (exact, replacement) in &self.overrides {
            if &m == exact {
                return (replacement.clone(), true);
            }
        }
        // 2. Prefix rules (first match wins)
        for (prefix, mini_target, pro_target) in &self.prefixes {
            if m.starts_with(prefix) {
                if m.contains("mini") {
                    return (mini_target.clone(), true);
                }
                return (pro_target.clone(), true);
            }
        }
        // 3. Pass through unknown models unchanged
        (model.to_string(), false)
    }

    /// Simple API: resolve or fall back to default with heuristic warning.
    pub fn map_model(&self, model: &str) -> String {
        let (result, matched) = self.resolve(model);
        if !matched && model != result {
            tracing::warn!(target: "deeplossless::protocol",
                "unknown model '{}', using '{}' (no mapping rule matched)",
                model, result);
        }
        result
    }
}

/// Convenience wrapper using the default registry.
pub fn map_model(model: &str) -> String {
    ModelRegistry::default().map_model(model)
}
