//! Canonical runtime representation — the intermediate representation (IR)
//! that all provider adapters translate to/from.
//!
//! Architecture:
//! ```text
//! OpenAI Responses / DeepSeek Chat / Anthropic Messages
//!         ↓ adapter::from_*
//!   CanonicalRequest
//!         ↓ runtime (pipeline: DAG, cache, failure detection)
//!   CanonicalResponse
//!         ↓ adapter::to_*
//! OpenAI Responses / DeepSeek Chat / Anthropic Messages
//! ```

use serde::{Deserialize, Serialize};

// ── Content parts (typed content items) ───────────────────────────────

/// A typed piece of content within a message.
/// This is the key abstraction — replaces free-form "content" strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Plain text.
    #[serde(rename = "text")]
    Text { text: String },

    /// An image reference (URL or base64).
    #[serde(rename = "image")]
    Image {
        /// "url" or "base64"
        source_type: String,
        /// URL or data URI
        data: String,
        /// Optional detail level
        #[serde(default)]
        detail: String,
    },

    /// A tool invocation from the assistant.
    #[serde(rename = "tool_call")]
    ToolCall {
        /// Unique ID for this invocation.
        id: String,
        /// Tool/function name.
        name: String,
        /// JSON-encoded arguments.
        arguments: String,
    },

    /// A tool execution result.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// Which tool call this responds to.
        call_id: String,
        /// The result content.
        content: String,
    },

    /// Reasoning/thinking content.
    #[serde(rename = "reasoning")]
    Reasoning { text: String },
}

// ── Message ────────────────────────────────────────────────────────────

/// A single message in the conversation. Each message has a role and
/// one or more typed content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<ContentPart>,
    /// Provider-specific metadata (e.g., tool_call_id for tool messages).
    #[serde(default)]
    pub meta: Option<MessageMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMeta {
    /// For tool result messages: which tool call this responds to.
    pub tool_call_id: Option<String>,
    /// For assistant messages with tool calls: the tool calls can be in meta.
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
}

// ── Tool definition ────────────────────────────────────────────────────

/// A tool that the model can invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Whether to enforce strict JSON schema validation.
    #[serde(default)]
    pub strict: bool,
}

/// A tool invocation (used in responses and in meta).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

// ── Request ────────────────────────────────────────────────────────────

/// Canonical request — the IR that all adapters produce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalRequest {
    /// System-level instructions. Separate from the message list.
    pub instructions: Option<String>,

    /// The conversation messages.
    pub messages: Vec<Message>,

    /// Available tools.
    #[serde(default)]
    pub tools: Vec<ToolDef>,

    /// Model name.
    pub model: String,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Max output tokens.
    pub max_tokens: Option<u32>,

    /// Sampling temperature.
    pub temperature: Option<f64>,

    /// Structured output format (JSON schema).
    pub response_format: Option<ResponseFormat>,

    /// Provider hint — which provider generated this request.
    /// Used by the pipeline for provider-specific behavior.
    #[serde(default)]
    pub provider_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: Option<serde_json::Value>,
}

// ── Response ───────────────────────────────────────────────────────────

/// Canonical response — the IR that adapters consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalResponse {
    /// Unique response ID.
    pub id: String,

    /// Model that produced the response.
    pub model: String,

    /// Response status.
    pub status: ResponseStatus,

    /// The output messages/content.
    pub output: Vec<ContentPart>,

    /// Token usage.
    pub usage: Usage,

    /// Provider metadata.
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

// ── Streaming events ───────────────────────────────────────────────────

/// Normalized streaming events — the IR for SSE streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// A chunk of text content.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// Content being appended.
        text: String,
    },

    /// A tool call is being built up.
    #[serde(rename = "tool_call_delta")]
    ToolCallDelta {
        /// Which tool call index.
        index: usize,
        /// Tool call ID.
        id: String,
        /// Tool name (usually on first chunk).
        name: Option<String>,
        /// Partial arguments JSON.
        arguments_delta: String,
    },

    /// Reasoning/thinking delta.
    #[serde(rename = "reasoning_delta")]
    ReasoningDelta { text: String },

    /// Response completed.
    #[serde(rename = "done")]
    Done {
        /// Final usage stats.
        usage: Usage,
        /// Finish reason from the model.
        finish_reason: String,
    },

    /// An error occurred.
    #[serde(rename = "error")]
    Error { message: String, code: Option<String> },
}
