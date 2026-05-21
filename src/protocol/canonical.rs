//! # Protocol IR
//!
//! Provider-neutral, model-neutral, transport-neutral intermediate representation.
//! This is the **protocol layer** — ingress/egress format translation.
//!
//! The **execution layer** lives in `crate::execution` (ExecutionUnit,
//! FailurePattern, PlanState, CodeChange). Protocol IR maps provider formats
//! to structured data; execution IR operates on runtime artifacts.
//!
//! ```text
//! Provider format
//!     ↓ adapter (dumb parse/normalize)
//! Protocol IR (this module)
//!     ↓ pipeline
//! Execution IR (crate::execution)
//!     ↓ runtime policy
//! Execution IR (updated)
//!     ↓ projection
//! Protocol IR
//!     ↓ adapter (dumb serialize)
//! Provider format
//! ```

use serde::{Deserialize, Serialize};

// ── Provider identity ──────────────────────────────────────────────────

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "unknown")]
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub reasoning: bool,
    pub tool_streaming: bool,
    pub multimodal: bool,
    pub json_schema: bool,
    pub parallel_tools: bool,
}

// ── Content parts ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "image")]
    Image {
        source_type: String,
        data: String,
        #[serde(default)]
        detail: String,
    },

    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        /// Structured arguments, not raw string — enables hashing/cache/diff.
        arguments: serde_json::Value,
    },

    /// Tool execution result — first-class runtime object, not "assistant message".
    #[serde(rename = "tool_result")]
    ToolResult {
        call_id: String,
        content: String,
    },
}

/// Reasoning/thinking trace — NOT a ContentPart.
/// Reasoning is a distinct execution artifact, not text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    pub text: String,
    #[serde(default)]
    pub summarized: bool,
    pub tokens: Option<u32>,
}

// ── Message ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<ContentPart>,
    #[serde(default)]
    pub meta: Option<MessageMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMeta {
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolInvocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    /// OpenAI Responses "developer" role — distinct from Tool.
    Developer,
}

// ── Instructions ───────────────────────────────────────────────────────

/// Multiple system-level instruction blocks (preserves provenance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionBlock {
    pub text: String,
    #[serde(default)]
    pub meta: Option<serde_json::Value>,
}

// ── Tools ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    /// Structured arguments — not raw String. Enables hash/cache/diff/replay.
    pub arguments: serde_json::Value,
}

// ── Request ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalRequest {
    /// System instructions (supports multiple blocks).
    #[serde(default)]
    pub instructions: Vec<InstructionBlock>,

    #[serde(default)]
    pub messages: Vec<Message>,

    #[serde(default)]
    pub tools: Vec<ToolDef>,

    pub model: String,

    #[serde(default)]
    pub stream: bool,

    pub max_tokens: Option<u32>,

    pub temperature: Option<f64>,

    pub response_format: Option<ResponseFormat>,

    /// Which provider generated this request.
    #[serde(default)]
    pub provider: ProviderKind,

    /// What the provider supports.
    #[serde(default)]
    pub capabilities: ProviderCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: Option<serde_json::Value>,
}

// ── Response ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalResponse {
    pub id: String,
    pub model: String,
    pub status: ResponseStatus,
    pub output: Vec<ContentPart>,

    /// Reasoning trace if the model produced one.
    #[serde(default)]
    pub reasoning_trace: Option<ReasoningTrace>,

    pub usage: Usage,

    #[serde(default)]
    pub provider_meta: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Incomplete,
    Error,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming events (with lifecycle state) ─────────────────────────────

/// Each tool call: Start → ArgsDelta* → End.
/// Each message: MessageStart → (TextDelta | ToolCall*) → MessageEnd.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { role: String },

    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    #[serde(rename = "tool_call_start")]
    ToolCallStart { index: usize, id: String, name: String },

    #[serde(rename = "tool_call_args_delta")]
    ToolCallArgsDelta { index: usize, arguments_delta: String },

    #[serde(rename = "tool_call_end")]
    ToolCallEnd { index: usize },

    #[serde(rename = "reasoning_delta")]
    ReasoningDelta { text: String },

    #[serde(rename = "message_end")]
    MessageEnd,

    #[serde(rename = "done")]
    Done { usage: Usage, finish_reason: String },

    #[serde(rename = "error")]
    Error { message: String, code: Option<String> },
}
