//! Runtime event types — append-only runtime lifecycle events.
//!
//! Phase 2 (Event Runtime): these events replace direct mutation of
//! execution state. The existing mutable `ExecutionCycle` fields remain
//! as projections, derived from the RuntimeEvent lifecycle substream.
//!
//! `execution_events` also stores protocol replay/audit rows. Those rows
//! are not RuntimeEvent values unless they explicitly use this schema.
//!
//! # Scope
//! Execution lifecycle, retry lifecycle, cancellation.
//!
//! OUT of scope: DAG topology, compaction, cache entries, file observations.
//! MUST NOT add events for those subsystems.
//!
//! # Event payload policy
//! Events carry lifecycle metadata, NOT data dumps.
//! ToolCallCompleted has `tokens_spent` and `cache_hit`, not full JSON result.
//! The event stream is lifecycle truth, not storage.
//!
//! # Cooperative cancellation contract (frozen)
//! CancellationRequested signals the INTENT to cancel.
//! It does NOT imply immediate termination.
//! In-flight executions MAY complete normally after a cancellation request.
//! The runtime MUST emit one of:
//!   - CancellationAcknowledged  (execution stopped cleanly)
//!   - ExecutionCompleted         (normal completion after request)
//!   - CancellationCompleted      (all work has stopped)
//!     after a CancellationRequested.
//!
//! # Logical sequence
//! logical_seq is append-order monotonic (AtomicI64, not wall-clock).
//! Event ordering within a conversation is determined by logical_seq.

use serde::{Deserialize, Serialize};

/// Frozen schema version for RuntimeEvent enum.
/// Increment ONLY when adding new variants. NEVER change existing variant
/// shapes — this breaks replay compatibility.
pub const RUNTIME_EVENT_SCHEMA_VERSION: u32 = 1;

/// Frozen schema version for CancellationSource enum.
pub const CANCELLATION_SOURCE_SCHEMA_VERSION: u32 = 1;

/// A runtime event — the single source of truth for runtime lifecycle
/// transitions. Append-only. Never mutated after emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeEvent {
    // ── Execution lifecycle ─────────────────────────────────────────

    /// An execution cycle has started for a conversation.
    ExecutionStarted {
        conv_id: i64,
        logical_seq: i64,
        profile: String,
    },

    /// A tool call has been scheduled for execution.
    ToolCallScheduled {
        conv_id: i64,
        logical_seq: i64,
        tool_name: String,
        tool_call_id: String,
        span_id: String,
        /// Unique ID for this attempt (increments on retry).
        attempt: u32,
    },

    /// A tool call completed successfully.
    ToolCallCompleted {
        conv_id: i64,
        logical_seq: i64,
        tool_name: String,
        tool_call_id: String,
        span_id: String,
        attempt: u32,
        tokens_spent: u64,
        /// Whether result was served from cache.
        cache_hit: bool,
        /// Execution unit ID in the database (0 if not persisted).
        execution_unit_id: i64,
    },

    /// A tool call failed.
    ToolCallFailed {
        conv_id: i64,
        logical_seq: i64,
        tool_name: String,
        tool_call_id: String,
        span_id: String,
        attempt: u32,
        error_signature: String,
        /// Whether this failure can be retried.
        retryable: bool,
        execution_unit_id: i64,
    },

    // ── Retry lifecycle ─────────────────────────────────────────────

    /// A retry has been scheduled for a previously failed tool call.
    RetryScheduled {
        conv_id: i64,
        logical_seq: i64,
        tool_call_id: String,
        attempt: u32,
        suggested_fix: String,
    },

    /// A retry was aborted (max retries exceeded, or fatal error).
    RetryAborted {
        conv_id: i64,
        logical_seq: i64,
        tool_call_id: String,
        total_attempts: u32,
        reason: String,
    },

    // ── Cancellation lifecycle ──────────────────────────────────────

    /// A cancellation has been requested (shutdown, timeout, client disconnect).
    CancellationRequested {
        conv_id: i64,
        logical_seq: i64,
        source: CancellationSource,
    },

    /// A specific execution acknowledged the cancellation request and
    /// stopped cleanly before completing.
    CancellationAcknowledged {
        conv_id: i64,
        logical_seq: i64,
        tool_call_id: String,
        span_id: String,
    },

    /// Cancellation has completed and all in-flight work has stopped.
    CancellationCompleted {
        conv_id: i64,
        logical_seq: i64,
        /// Whether cancellation was clean (all tasks acknowledged or
        /// completed) vs. forced (some tasks timed out).
        clean: bool,
    },
}

/// What triggered the cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancellationSource {
    /// Process shutdown initiated.
    Shutdown,
    /// Client disconnected (SSE stream dropped).
    ClientDisconnect,
    /// Execution deadline exceeded.
    DeadlineExceeded,
    /// Explicit cancellation by the agent/runtime.
    ExplicitCancel,
}

impl RuntimeEvent {
    /// Logical sequence number for deterministic ordering.
    pub fn logical_seq(&self) -> i64 {
        match self {
            Self::ExecutionStarted { logical_seq, .. }
            | Self::ToolCallScheduled { logical_seq, .. }
            | Self::ToolCallCompleted { logical_seq, .. }
            | Self::ToolCallFailed { logical_seq, .. }
            | Self::RetryScheduled { logical_seq, .. }
            | Self::RetryAborted { logical_seq, .. }
            | Self::CancellationRequested { logical_seq, .. }
            | Self::CancellationAcknowledged { logical_seq, .. }
            | Self::CancellationCompleted { logical_seq, .. } => *logical_seq,
        }
    }

    /// Conversation this event belongs to.
    pub fn conv_id(&self) -> i64 {
        match self {
            Self::ExecutionStarted { conv_id, .. }
            | Self::ToolCallScheduled { conv_id, .. }
            | Self::ToolCallCompleted { conv_id, .. }
            | Self::ToolCallFailed { conv_id, .. }
            | Self::RetryScheduled { conv_id, .. }
            | Self::RetryAborted { conv_id, .. }
            | Self::CancellationRequested { conv_id, .. }
            | Self::CancellationAcknowledged { conv_id, .. }
            | Self::CancellationCompleted { conv_id, .. } => *conv_id,
        }
    }

    /// Human-readable event kind for logging and the `execution_events` table.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ExecutionStarted { .. } => "execution_started",
            Self::ToolCallScheduled { .. } => "tool_call_scheduled",
            Self::ToolCallCompleted { .. } => "tool_call_completed",
            Self::ToolCallFailed { .. } => "tool_call_failed",
            Self::RetryScheduled { .. } => "retry_scheduled",
            Self::RetryAborted { .. } => "retry_aborted",
            Self::CancellationRequested { .. } => "cancellation_requested",
            Self::CancellationAcknowledged { .. } => "cancellation_acknowledged",
            Self::CancellationCompleted { .. } => "cancellation_completed",
        }
    }
}
