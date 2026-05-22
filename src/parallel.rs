//! Parallel execution runtime: fork/join lifecycle, HappensBefore tracking,
//! parallel group governance (timeouts, error propagation, partial completion).
//!
//! # Design
//!
//! The proxy does NOT execute tool calls (they run at the agent side). Instead,
//! this module provides:
//!
//! - **Fork detection**: when the upstream LLM emits N tool calls in one turn,
//!   they form a parallel group. Each branch gets a unique span_id + shared
//!   parallel_group UUID.
//!
//! - **Join verification**: when tool results arrive in the next request, the
//!   tracker verifies all branches completed, inserts a join DAG node, and
//!   records HappensBefore edges between sequential groups.
//!
//! - **Governance**: timeout detection, fail-fast semantics (one failure
//!   invalidates sibling branches), partial completion tolerance.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Span identity ─────────────────────────────────────────────────────

/// Unique execution span identifier for distributed tracing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub String);

impl SpanId {
    pub fn new() -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let rand = fast_random_u16();
        Self(format!("sp_{ts:x}_{rand:x}"))
    }

    pub fn root() -> Self {
        Self("root".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a simple parallel group ID, unique within a conversation.
pub fn group_id(conv_id: i64, turn_index: usize) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let rand = fast_random_u16();
    format!("pg_{conv_id}_{turn_index}_{ts:x}_{rand:x}")
}

fn fast_random_u16() -> u16 {
    // Simple non-cryptographic random from wall-clock jitter
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    (n ^ (n >> 16)) as u16
}

// ── Span mode ─────────────────────────────────────────────────────────

/// Execution mode for an ExecutionSpan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanMode {
    Sequential,
    Parallel,
    Join,
}

impl SpanMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Join => "join",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "sequential" => Some(Self::Sequential),
            "parallel" => Some(Self::Parallel),
            "join" => Some(Self::Join),
            _ => None,
        }
    }
}

// ── ExecutionSpan ─────────────────────────────────────────────────────

/// A span in the execution tree. Tracks the lifecycle of a sequential or
/// parallel execution block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSpan {
    pub span_id: SpanId,
    pub parent_span_id: SpanId,
    pub mode: SpanMode,
    pub parallel_group: Option<String>,
    pub created_at: String,
}

impl ExecutionSpan {
    /// Create a root span (mode=sequential, parent=root).
    pub fn root_span() -> Self {
        Self {
            span_id: SpanId::new(),
            parent_span_id: SpanId::root(),
            mode: SpanMode::Sequential,
            parallel_group: None,
            created_at: iso_now(),
        }
    }

    /// Create a child span within this one.
    pub fn child(&self, mode: SpanMode) -> Self {
        Self {
            span_id: SpanId::new(),
            parent_span_id: self.span_id.clone(),
            mode,
            parallel_group: None,
            created_at: iso_now(),
        }
    }

    /// Tag this span with a parallel group ID.
    pub fn with_group(mut self, group: String) -> Self {
        self.parallel_group = Some(group);
        self
    }
}

// ── Branch status ─────────────────────────────────────────────────────

/// Status of a single branch within a parallel group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchStatus {
    Pending,
    Running,
    Completed,
    Failed,
    TimedOut,
}

impl BranchStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }
}

// ── Parallel governance ───────────────────────────────────────────────

/// Governance policy for a parallel execution group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGovernance {
    /// Max wall-clock seconds for all branches to complete.
    pub timeout_secs: f64,
    /// If true, one failure cancels all other branches (fail-fast).
    pub fail_fast: bool,
    /// If true, partial results are accepted (missing branches tolerated).
    pub allow_partial: bool,
    /// Max parallel branches recommended.
    pub max_concurrency: usize,
}

impl Default for ParallelGovernance {
    fn default() -> Self {
        Self {
            timeout_secs: 30.0,
            fail_fast: true,
            allow_partial: false,
            max_concurrency: 8,
        }
    }
}

// ── Parallel branch ───────────────────────────────────────────────────

/// A single branch within a parallel execution group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelBranch {
    pub span_id: SpanId,
    pub tool_name: String,
    pub tool_call_id: String,
    pub status: BranchStatus,
    pub error: Option<String>,
    pub execution_unit_id: Option<i64>,
}

// ── ForkJoinTracker ───────────────────────────────────────────────────

/// Tracks one parallel execution group from fork → all branches → join.
///
/// # Lifecycle
///
/// 1. **Fork**: created from an assistant message with N≥2 tool calls.
///    Each tool call becomes a `ParallelBranch`. Fork events are recorded
///    to `execution_events` and `lineage_edges`.
///
/// 2. **Branch updates**: as tool results arrive (in subsequent requests),
///    branches transition Pending → Running → Completed/Failed.
///
/// 3. **Join**: when all branches have resolved (or governance says to
///    proceed), a join DAG node is inserted, HappensBefore edges record
///    the ordering, and a join event is emitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkJoinTracker {
    pub group_id: String,
    pub conv_id: i64,
    pub turn_index: usize,
    pub parent_span: ExecutionSpan,
    pub branches: Vec<ParallelBranch>,
    pub governance: ParallelGovernance,
    pub created_at: String,
    pub completed_at: Option<String>,
    /// DAG node ID of the join node (set on completion).
    pub join_dag_node_id: Option<i64>,
}

impl ForkJoinTracker {
    /// Create a new tracker for a parallel group. This is the **fork** phase.
    pub fn fork(
        conv_id: i64,
        turn_index: usize,
        parent_span: &ExecutionSpan,
        tool_calls: &[ToolCallInfo],
        governance: ParallelGovernance,
    ) -> Self {
        let gid = group_id(conv_id, turn_index);
        let branches: Vec<ParallelBranch> = tool_calls
            .iter()
            .map(|tc| {
                let span = parent_span.child(SpanMode::Parallel).with_group(gid.clone());
                ParallelBranch {
                    span_id: span.span_id,
                    tool_name: tc.name.clone(),
                    tool_call_id: tc.call_id.clone(),
                    status: BranchStatus::Pending,
                    error: None,
                    execution_unit_id: None,
                }
            })
            .collect();

        Self {
            group_id: gid,
            conv_id,
            turn_index,
            parent_span: parent_span.clone(),
            branches,
            governance,
            created_at: iso_now(),
            completed_at: None,
            join_dag_node_id: None,
        }
    }

    /// Number of branches in this group.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Check if all branches are in a terminal state.
    pub fn is_completed(&self) -> bool {
        self.completed_at.is_some()
            || self.branches.iter().all(|b| matches!(b.status, BranchStatus::Completed | BranchStatus::Failed | BranchStatus::TimedOut))
    }

    /// Check if governance says we should proceed despite incomplete state.
    pub fn should_force_join(&self) -> bool {
        let terminal = self
            .branches
            .iter()
            .filter(|b| matches!(b.status, BranchStatus::Completed | BranchStatus::Failed | BranchStatus::TimedOut))
            .count();
        if terminal == 0 {
            return false;
        }
        if self.governance.allow_partial && terminal > 0 {
            return true;
        }
        if self.governance.fail_fast
            && self.branches.iter().any(|b| matches!(b.status, BranchStatus::Failed | BranchStatus::TimedOut))
        {
            return true;
        }
        terminal == self.branches.len()
    }

    /// Update a branch's status from an execution unit.
    pub fn record_branch_result(
        &mut self,
        tool_call_id: &str,
        exec_unit_id: i64,
        outcome: &crate::execution::ExecutionOutcome,
    ) {
        if let Some(branch) = self.branches.iter_mut().find(|b| b.tool_call_id == tool_call_id) {
            branch.execution_unit_id = Some(exec_unit_id);
            branch.status = match outcome {
                crate::execution::ExecutionOutcome::Success
                | crate::execution::ExecutionOutcome::CacheHit
                | crate::execution::ExecutionOutcome::Replayed => BranchStatus::Completed,
                crate::execution::ExecutionOutcome::RecoveredFailure
                | crate::execution::ExecutionOutcome::Blocked => {
                    branch.error = Some("tool execution failed".into());
                    BranchStatus::Failed
                }
                crate::execution::ExecutionOutcome::Stale => BranchStatus::TimedOut,
            };
        }
    }

    /// Mark a branch as failed due to timeout.
    pub fn timeout_branch(&mut self, tool_call_id: &str) {
        if let Some(branch) = self.branches.iter_mut().find(|b| b.tool_call_id == tool_call_id) {
            branch.status = BranchStatus::TimedOut;
            branch.error = Some("timeout".into());
        }
    }

    /// Complete the join phase. Returns the computed HappensBefore edges
    /// that should be inserted into the lineage_edges table.
    pub fn complete(&mut self, join_node_id: i64) -> Vec<HappensBeforeEdge> {
        self.completed_at = Some(iso_now());
        self.join_dag_node_id = Some(join_node_id);

        // Emit HappensBefore edges:
        //   - All branches in this group are concurrent (no ordering between them)
        //   - Each branch happens-before the join node
        //   - The parent span happens-before the group
        let mut edges = Vec::new();
        for branch in &self.branches {
            if let Some(eid) = branch.execution_unit_id {
                // branch execution happens-before join node
                edges.push(HappensBeforeEdge {
                    from_id: eid,
                    to_id: join_node_id,
                });
            }
        }
        edges
    }
}

/// A HappensBefore edge between two execution nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HappensBeforeEdge {
    pub from_id: i64,
    pub to_id: i64,
}

// ── Tool call info (input for fork) ───────────────────────────────────

/// Minimal tool call data needed to create a parallel branch.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub name: String,
    pub call_id: String,
}

// ── ParallelDetector ──────────────────────────────────────────────────

/// Detects parallel execution opportunities in normalized messages.
/// Scans for assistant messages with multiple tool calls → parallel group.
pub struct ParallelDetector;

impl ParallelDetector {
    /// Detect if an assistant message has multiple tool calls (parallelizable).
    /// Returns None for 0-1 tool calls, Some(tool_calls) for 2+.
    pub fn detect(msg: &crate::session::NormalizedMessage) -> Option<Vec<ToolCallInfo>> {
        if msg.role != "assistant" || msg.tool_calls.len() < 2 {
            return None;
        }
        Some(
            msg.tool_calls
                .iter()
                .map(|tc| ToolCallInfo {
                    name: tc.name.clone(),
                    call_id: tc.id.clone(),
                })
                .collect(),
        )
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn iso_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_id_generates_unique_values() {
        let a = SpanId::new();
        let b = SpanId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn group_id_includes_conv_id() {
        let gid = group_id(42, 3);
        assert!(gid.contains("42"), "group_id should contain conv_id");
        assert!(gid.contains("3"), "group_id should contain turn_index");
    }

    #[test]
    fn fork_creates_one_branch_per_tool() {
        let parent = ExecutionSpan::root_span();
        let tools = vec![
            ToolCallInfo { name: "grep".into(), call_id: "call_1".into() },
            ToolCallInfo { name: "read_file".into(), call_id: "call_2".into() },
            ToolCallInfo { name: "list_files".into(), call_id: "call_3".into() },
        ];
        let tracker = ForkJoinTracker::fork(1, 0, &parent, &tools, ParallelGovernance::default());
        assert_eq!(tracker.branch_count(), 3);
        assert!(tracker.group_id.contains("1"));
    }

    #[test]
    fn detect_multi_tool_turn() {
        let msg = crate::session::NormalizedMessage {
            role: "assistant".into(),
            content: "Let me check those files.".into(),
            tool_calls: vec![
                crate::session::NormalizedToolCall {
                    id: "call_1".into(),
                    name: "grep".into(),
                    arguments: r#"{"pattern":"foo"}"#.into(),
                },
                crate::session::NormalizedToolCall {
                    id: "call_2".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"bar"}"#.into(),
                },
            ],
            tool_call_id: None,
        };
        assert!(ParallelDetector::detect(&msg).is_some());
    }

    #[test]
    fn single_tool_not_parallel() {
        let msg = crate::session::NormalizedMessage {
            role: "assistant".into(),
            content: "Let me check.".into(),
            tool_calls: vec![crate::session::NormalizedToolCall {
                id: "call_1".into(),
                name: "grep".into(),
                arguments: "{}".into(),
            }],
            tool_call_id: None,
        };
        assert!(ParallelDetector::detect(&msg).is_none());
    }

    #[test]
    fn tracker_completion_all_success() {
        let parent = ExecutionSpan::root_span();
        let tools = vec![
            ToolCallInfo { name: "grep".into(), call_id: "c1".into() },
            ToolCallInfo { name: "read".into(), call_id: "c2".into() },
        ];
        let mut tracker = ForkJoinTracker::fork(1, 0, &parent, &tools, ParallelGovernance::default());
        assert!(!tracker.is_completed());

        tracker.record_branch_result("c1", 10, &crate::execution::ExecutionOutcome::Success);
        assert!(!tracker.is_completed());

        tracker.record_branch_result("c2", 11, &crate::execution::ExecutionOutcome::Success);
        assert!(tracker.is_completed());
    }

    #[test]
    fn tracker_fail_fast_completes_on_failure() {
        let parent = ExecutionSpan::root_span();
        let tools = vec![
            ToolCallInfo { name: "grep".into(), call_id: "c1".into() },
            ToolCallInfo { name: "read".into(), call_id: "c2".into() },
        ];
        let mut tracker = ForkJoinTracker::fork(1, 0, &parent, &tools, ParallelGovernance::default());
        tracker.record_branch_result("c1", 10, &crate::execution::ExecutionOutcome::RecoveredFailure);

        // With fail_fast=true, should_force_join returns true
        assert!(tracker.should_force_join());
    }

    #[test]
    fn tracker_happens_before_edges_correct() {
        let parent = ExecutionSpan::root_span();
        let tools = vec![
            ToolCallInfo { name: "grep".into(), call_id: "c1".into() },
            ToolCallInfo { name: "read".into(), call_id: "c2".into() },
        ];
        let mut tracker = ForkJoinTracker::fork(1, 0, &parent, &tools, ParallelGovernance::default());
        tracker.record_branch_result("c1", 10, &crate::execution::ExecutionOutcome::Success);
        tracker.record_branch_result("c2", 11, &crate::execution::ExecutionOutcome::Success);

        let edges = tracker.complete(42);
        // Both branches happen-before the join node
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|e| e.from_id == 10 && e.to_id == 42));
        assert!(edges.iter().any(|e| e.from_id == 11 && e.to_id == 42));
    }

    #[test]
    fn detect_zero_or_one_tool() {
        let msg = crate::session::NormalizedMessage {
            role: "user".into(),
            content: "hello".into(),
            tool_calls: vec![],
            tool_call_id: None,
        };
        assert!(ParallelDetector::detect(&msg).is_none());
    }

    #[test]
    fn span_mode_roundtrip() {
        assert_eq!(SpanMode::from_str("sequential"), Some(SpanMode::Sequential));
        assert_eq!(SpanMode::from_str("parallel"), Some(SpanMode::Parallel));
        assert_eq!(SpanMode::from_str("join"), Some(SpanMode::Join));
        assert_eq!(SpanMode::from_str("unknown"), None);
    }

    #[test]
    fn execution_span_creates_child() {
        let root = ExecutionSpan::root_span();
        let child = root.child(SpanMode::Parallel);
        assert_eq!(child.parent_span_id, root.span_id);
        assert_eq!(child.mode, SpanMode::Parallel);
    }
}
