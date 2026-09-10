use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelError {
    Upstream(UpstreamError),
    Protocol(ProtocolError),
    Stream(StreamError),
    Reasoning(ReasoningError),
    ToolCall(ToolCallError),
    Context(ContextError),
    Routing(RoutingError),
    Runtime(RuntimeError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity { Critical, Error, Warning, Info }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Recoverability { Retryable, Degradable, Terminal }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorPhase { Receive, Parse, Route, Emit, Persist }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetryHint {
    Backoff { base_ms: u64 },
    DowngradeModel,
    ReduceReasoningEffort,
    CompressContext,
    RequestModelFix,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorMeta {
    pub severity: Severity,
    pub recoverability: Recoverability,
    pub retry_hint: Option<RetryHint>,
    pub phase: ErrorPhase,
}

impl ModelError {
    pub fn meta(&self) -> ErrorMeta {
        match self {
            ModelError::Upstream(e) => e.meta(),
            ModelError::Protocol(e) => e.meta(),
            ModelError::Stream(e) => e.meta(),
            ModelError::Reasoning(e) => e.meta(),
            ModelError::ToolCall(e) => e.meta(),
            ModelError::Context(e) => e.meta(),
            ModelError::Routing(e) => e.meta(),
            ModelError::Runtime(e) => e.meta(),
        }
    }

    pub fn status_code(&self) -> u16 {
        match self {
            ModelError::Upstream(UpstreamError::AuthenticationFailed | UpstreamError::PermissionDenied) => 401,
            ModelError::Upstream(UpstreamError::RateLimited) => 429,
            ModelError::Upstream(UpstreamError::ModelNotFound) => 404,
            ModelError::Upstream(UpstreamError::QuotaExceeded) => 402,
            ModelError::Upstream(UpstreamError::ServerError) => 502,
            ModelError::Upstream(UpstreamError::Timeout) => 504,
            ModelError::Upstream(UpstreamError::ConnectionFailed) => 503,
            ModelError::Context(ContextError::ContextTooLong) => 413,
            ModelError::Routing(RoutingError::ReasoningEffortNotSupported | RoutingError::InvalidReasoningEffort) => 400,
            ModelError::Routing(RoutingError::ModelRouteUnavailable) => 503,
            _ => 502,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            ModelError::Upstream(_) => "UPSTREAM_ERROR",
            ModelError::Protocol(_) => "PROTOCOL_ERROR",
            ModelError::Stream(_) => "STREAM_ERROR",
            ModelError::Reasoning(_) => "REASONING_ERROR",
            ModelError::ToolCall(_) => "TOOL_CALL_ERROR",
            ModelError::Context(_) => "CONTEXT_ERROR",
            ModelError::Routing(_) => "ROUTING_ERROR",
            ModelError::Runtime(_) => "RUNTIME_ERROR",
        }
    }
}

// ── Upstream errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UpstreamError {
    RateLimited, AuthenticationFailed, PermissionDenied,
    ModelNotFound, QuotaExceeded, ServerError, Timeout, ConnectionFailed,
}

impl UpstreamError {
    fn meta(&self) -> ErrorMeta {
        match self {
            UpstreamError::RateLimited => ErrorMeta {
                severity: Severity::Warning, recoverability: Recoverability::Retryable,
                retry_hint: Some(RetryHint::Backoff { base_ms: 1000 }), phase: ErrorPhase::Receive,
            },
            UpstreamError::AuthenticationFailed | UpstreamError::PermissionDenied => ErrorMeta {
                severity: Severity::Critical, recoverability: Recoverability::Terminal,
                retry_hint: None, phase: ErrorPhase::Receive,
            },
            UpstreamError::ModelNotFound | UpstreamError::QuotaExceeded => ErrorMeta {
                severity: Severity::Error, recoverability: Recoverability::Terminal,
                retry_hint: None, phase: ErrorPhase::Receive,
            },
            UpstreamError::ServerError | UpstreamError::Timeout | UpstreamError::ConnectionFailed => ErrorMeta {
                severity: Severity::Error, recoverability: Recoverability::Retryable,
                retry_hint: Some(RetryHint::Backoff { base_ms: 500 }), phase: ErrorPhase::Receive,
            },
        }
    }
}

// ── Protocol errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProtocolError {
    InvalidJson, MissingChoices, MissingDelta,
    UnsupportedResponseShape, MixedProtocolShape,
    InvalidContent { detail: String },
}

impl ProtocolError {
    fn meta(&self) -> ErrorMeta {
        ErrorMeta { severity: Severity::Error, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Parse }
    }
}

// ── Stream errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamError {
    InvalidSseLine, InvalidJsonEvent, StreamInterrupted,
    DoneWithoutFinishReason, EventAfterDone,
}

impl StreamError {
    fn meta(&self) -> ErrorMeta {
        match self {
            StreamError::StreamInterrupted => ErrorMeta {
                severity: Severity::Error, recoverability: Recoverability::Retryable,
                retry_hint: Some(RetryHint::Backoff { base_ms: 200 }), phase: ErrorPhase::Parse,
            },
            _ => ErrorMeta { severity: Severity::Warning, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Parse },
        }
    }
}

// ── Reasoning errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningError {
    StructuredReasoningMalformed, ReasoningMixedIntoContent,
    ThinkTagUnclosed, ThinkTagInFinalAnswer,
    ReasoningStateMissingForToolChain,
}

impl ReasoningError {
    fn meta(&self) -> ErrorMeta {
        ErrorMeta { severity: Severity::Warning, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Parse }
    }
}

// ── Tool call errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolCallError {
    ToolCallsMalformed, DsmlMalformed, DsmlIncomplete, UnknownTool,
    MissingRequiredArgument, UnexpectedArgument, ArgumentTypeMismatch,
    ArgumentsJsonInvalid, ToolCallIdMissing, DuplicateToolCallId,
}

impl ToolCallError {
    fn meta(&self) -> ErrorMeta {
        match self {
            ToolCallError::DsmlIncomplete => ErrorMeta {
                severity: Severity::Warning, recoverability: Recoverability::Degradable,
                retry_hint: Some(RetryHint::RequestModelFix), phase: ErrorPhase::Parse,
            },
            _ => ErrorMeta { severity: Severity::Error, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Parse },
        }
    }
}

// ── Context errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContextError {
    ContextTooLong, StablePrefixTooLarge, DynamicSuffixTooLarge,
    EvidencePackTooLarge, ToolSchemaTooLarge,
}

impl ContextError {
    fn meta(&self) -> ErrorMeta {
        match self {
            ContextError::ContextTooLong => ErrorMeta {
                severity: Severity::Error, recoverability: Recoverability::Degradable,
                retry_hint: Some(RetryHint::CompressContext), phase: ErrorPhase::Route,
            },
            _ => ErrorMeta { severity: Severity::Warning, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Route },
        }
    }
}

// ── Routing errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RoutingError {
    ModelRouteUnavailable, InvalidReasoningEffort,
    ReasoningEffortNotSupported, FallbackExhausted,
}

impl RoutingError {
    fn meta(&self) -> ErrorMeta {
        match self {
            RoutingError::InvalidReasoningEffort => ErrorMeta {
                severity: Severity::Error, recoverability: Recoverability::Degradable,
                retry_hint: Some(RetryHint::ReduceReasoningEffort), phase: ErrorPhase::Route,
            },
            _ => ErrorMeta { severity: Severity::Error, recoverability: Recoverability::Terminal, retry_hint: None, phase: ErrorPhase::Route },
        }
    }
}

// ── Runtime errors ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimeError {
    CacheReadFailed, CacheWriteFailed, PersistenceFailed, Cancellation,
}

impl RuntimeError {
    fn meta(&self) -> ErrorMeta {
        ErrorMeta { severity: Severity::Warning, recoverability: Recoverability::Degradable, retry_hint: None, phase: ErrorPhase::Persist }
    }
}

// ── Classify upstream errors ──

pub fn classify_upstream(status: u16, body: &str) -> ModelError {
    match status {
        429 => ModelError::Upstream(UpstreamError::RateLimited),
        401 => ModelError::Upstream(UpstreamError::AuthenticationFailed),
        403 => ModelError::Upstream(UpstreamError::PermissionDenied),
        404 => ModelError::Upstream(UpstreamError::ModelNotFound),
        402 => ModelError::Upstream(UpstreamError::QuotaExceeded),
        500..=511 => ModelError::Upstream(UpstreamError::ServerError),
        _ => {
            if body.contains("rate") || body.contains("limit") {
                ModelError::Upstream(UpstreamError::RateLimited)
            } else if body.contains("timeout") || body.contains("timed out") {
                ModelError::Upstream(UpstreamError::Timeout)
            } else {
                ModelError::Upstream(UpstreamError::ServerError)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_meta() {
        let err = ModelError::Upstream(UpstreamError::RateLimited);
        let meta = err.meta();
        assert_eq!(meta.severity, Severity::Warning);
        assert_eq!(meta.recoverability, Recoverability::Retryable);
    }

    #[test]
    fn test_auth_not_retryable() {
        let err = ModelError::Upstream(UpstreamError::AuthenticationFailed);
        let meta = err.meta();
        assert_eq!(meta.recoverability, Recoverability::Terminal);
    }

    #[test]
    fn test_classify_upstream_429() {
        let err = classify_upstream(429, "rate limited");
        assert_eq!(err, ModelError::Upstream(UpstreamError::RateLimited));
    }

    #[test]
    fn test_classify_upstream_401() {
        let err = classify_upstream(401, "invalid api key");
        assert_eq!(err, ModelError::Upstream(UpstreamError::AuthenticationFailed));
    }

    #[test]
    fn test_stream_interrupted_retryable() {
        let err = ModelError::Stream(StreamError::StreamInterrupted);
        assert_eq!(err.meta().recoverability, Recoverability::Retryable);
    }
}
