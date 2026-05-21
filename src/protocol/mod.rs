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

pub mod canonical;
pub mod chat_completions;
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
};
