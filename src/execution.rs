use serde::{Deserialize, Serialize};

/// Outcome of a tool execution within an agent reasoning loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionOutcome {
    /// Tool call succeeded, agent proceeded with reasoning.
    Success,
    /// Tool call failed but agent recovered (retry, fallback path).
    RecoveredFailure,
    /// Tool call failed and agent could not recover.
    Blocked,
}

impl ExecutionOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::RecoveredFailure => "recovered",
            Self::Blocked => "blocked",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "success" => Some(Self::Success),
            "recovered" => Some(Self::RecoveredFailure),
            "blocked" => Some(Self::Blocked),
            _ => None,
        }
    }
}

/// Atomic unit of agent memory: one complete "think → act → observe → reflect" cycle.
///
/// This is the core building block for agent memory — not a chat message,
/// but a recorded execution trace that survives compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionUnit {
    pub id: i64,
    pub conversation_id: i64,

    /// The assistant's reasoning before taking action (if present in response).
    pub reasoning_before: String,

    /// Tool call details.
    pub tool_name: String,
    pub tool_args: String,

    /// Tool execution result (truncated to preserve budget).
    pub tool_result: String,

    /// The assistant's reflection / next reasoning after seeing the result.
    pub reasoning_after: String,

    /// Inferred outcome of this execution step.
    pub outcome: ExecutionOutcome,

    /// DAG node IDs related to this execution (source leaves, summaries).
    pub related_nodes: Vec<i64>,

    /// ISO-8601 timestamp.
    pub created_at: String,
}

/// Lightweight reference for retrieval display.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionUnitRef {
    pub id: i64,
    pub tool_name: String,
    pub outcome: String,
    pub snippet: String,
}

impl ExecutionUnit {
    /// Create a new execution unit (id=0 until persisted).
    pub fn new(
        conversation_id: i64,
        reasoning_before: &str,
        tool_name: &str,
        tool_args: &str,
        tool_result: &str,
        reasoning_after: &str,
        outcome: ExecutionOutcome,
        related_nodes: &[i64],
    ) -> Self {
        Self {
            id: 0,
            conversation_id,
            reasoning_before: reasoning_before.to_string(),
            tool_name: tool_name.to_string(),
            tool_args: tool_args.to_string(),
            tool_result: tool_result.to_string(),
            reasoning_after: reasoning_after.to_string(),
            outcome,
            related_nodes: related_nodes.to_vec(),
            created_at: String::new(),
        }
    }

    /// Short display snippet for retrieval results.
    pub fn snippet(&self) -> String {
        let before: String = self.reasoning_before.chars().take(60).collect();
        let after: String = self.reasoning_after.chars().take(60).collect();
        format!(
            "🔧 {} → {} | before: {}… | after: {}…",
            self.tool_name,
            self.outcome.as_str(),
            if before.is_empty() { "(none)" } else { &before },
            if after.is_empty() { "(none)" } else { &after },
        )
    }
}

/// Group normalized messages into execution units.
/// Detects chains: assistant(tool_call) → tool(result) → assistant(response).
pub fn group_execution_chain(
    conv_id: i64,
    messages: &[crate::session::NormalizedMessage],
) -> Vec<ExecutionUnit> {
    let mut units = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];

        // Look for assistant message with tool calls
        if msg.role == "assistant" && !msg.tool_calls.is_empty() {
            let reasoning_before = msg.content.clone();
            let related = Vec::new(); // will be filled with DAG node IDs after insertion

            // Collect all tool call→result pairs for this assistant turn
            for tc in &msg.tool_calls {
                let mut tool_result = String::new();

                // Find matching tool result(s)
                let mut j = i + 1;
                while j < messages.len() {
                    let next = &messages[j];
                    if crate::session::is_tool_result(next)
                        && next.tool_call_id.as_deref() == Some(&tc.id)
                    {
                        if !tool_result.is_empty() {
                            tool_result.push('\n');
                        }
                        tool_result.push_str(&next.content);
                    }
                    // Stop at next assistant message (end of tool chain)
                    if next.role == "assistant" {
                        break;
                    }
                    j += 1;
                }
                i = j; // advance past tool results

                // Infer outcome from result using structured patterns
                let outcome = {
                    let lower = tool_result.to_lowercase();
                    if tool_result.contains("Error:") || tool_result.contains("error:")
                        || tool_result.contains("error[") || tool_result.contains("Error[")
                        || tool_result.contains("\nerror") || tool_result.starts_with("error")
                        || lower.contains("failed:") || lower.contains("exit code")
                        || lower.contains("timed out") || lower.contains("permission denied")
                        || lower.contains("not found") || lower.starts_with("err:")
                    {
                        ExecutionOutcome::RecoveredFailure
                    } else if tool_result.is_empty() {
                        ExecutionOutcome::Blocked
                    } else {
                        ExecutionOutcome::Success
                    }
                };

                // Get reasoning_after from the next assistant message
                let reasoning_after = if j < messages.len() && messages[j].role == "assistant" {
                    let after = messages[j].content.clone();
                    // Truncate to reasonable size
                    after.chars().take(300).collect()
                } else {
                    String::new()
                };

                units.push(ExecutionUnit::new(
                    conv_id,
                    &reasoning_before,
                    &tc.name,
                    &tc.arguments,
                    &tool_result,
                    &reasoning_after,
                    outcome,
                    &related,
                ));
            }
        }
        i += 1;
    }
    units
}

/// SQL migration for execution_units table.
pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS execution_units (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id   INTEGER NOT NULL REFERENCES conversations(id),
        reasoning_before  TEXT NOT NULL DEFAULT '',
        tool_name         TEXT NOT NULL DEFAULT '',
        tool_args         TEXT NOT NULL DEFAULT '',
        tool_result       TEXT NOT NULL DEFAULT '',
        reasoning_after   TEXT NOT NULL DEFAULT '',
        outcome           TEXT NOT NULL DEFAULT 'success',
        related_nodes     TEXT NOT NULL DEFAULT '[]',
        created_at        TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_execution_conv
        ON execution_units(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_execution_tool
        ON execution_units(tool_name);";

// ── Code Diff Memory (Phase 1.5) ──────────────────────────────────────

/// A recorded code change — the real unit of value in AI coding sessions.
/// 95% of tokens in coding conversations are repeated code; what matters is
/// what *changed* and whether it fixed the error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    pub id: i64,
    pub conversation_id: i64,

    /// File path affected (e.g. "src/dag.rs").
    pub file_path: String,

    /// The diff (minimal change representation, not full file).
    pub diff: String,

    /// Symbols (functions, types, variables) that were modified.
    pub symbols_changed: Vec<String>,

    /// Error messages before the change was applied.
    pub error_before: Vec<String>,

    /// Error messages after the change (empty = fix worked).
    pub error_after: Vec<String>,

    /// Link to the ExecutionUnit that produced this change.
    pub execution_unit_id: Option<i64>,

    pub created_at: String,
}

/// SQL migration for code_changes table.
pub const CODE_CHANGE_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS code_changes (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id     INTEGER NOT NULL REFERENCES conversations(id),
        file_path           TEXT NOT NULL DEFAULT '',
        diff                TEXT NOT NULL DEFAULT '',
        symbols_changed     TEXT NOT NULL DEFAULT '[]',
        error_before        TEXT NOT NULL DEFAULT '[]',
        error_after         TEXT NOT NULL DEFAULT '[]',
        execution_unit_id   INTEGER REFERENCES execution_units(id),
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_code_change_conv
        ON code_changes(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_code_change_file
        ON code_changes(file_path);
    CREATE INDEX IF NOT EXISTS idx_code_change_symbols
        ON code_changes(symbols_changed);";

// ── Failure Memory (v0.8) ────────────────────────────────────────────

/// A recorded failure pattern — not just the error, but the reasoning path
/// that led to it. The real cost is the failed reasoning, not the error itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub id: i64,
    pub conversation_id: i64,

    /// Error signature: normalized error message / symptom.
    pub signature: String,

    /// What fix was attempted.
    pub attempted_fix: String,

    /// Why the fix didn't work — the critical field.
    /// E.g., "Adding async mutex failed because SQLite connection still shared globally."
    pub why_failed: String,

    /// Assumptions that turned out to be wrong.
    pub invalidated_assumptions: Vec<String>,

    /// Files involved in the failure.
    pub related_files: Vec<String>,

    /// Link to the execution unit that recorded this failure.
    pub execution_unit_id: Option<i64>,

    pub created_at: String,
}

/// SQL migration for failure_patterns table.
pub const FAILURE_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS failure_patterns (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id         INTEGER NOT NULL REFERENCES conversations(id),
        signature               TEXT NOT NULL,
        attempted_fix           TEXT NOT NULL DEFAULT '',
        why_failed              TEXT NOT NULL DEFAULT '',
        invalidated_assumptions TEXT NOT NULL DEFAULT '[]',
        related_files           TEXT NOT NULL DEFAULT '[]',
        execution_unit_id       INTEGER REFERENCES execution_units(id),
        created_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_failure_sig
        ON failure_patterns(signature);
    CREATE INDEX IF NOT EXISTS idx_failure_conv
        ON failure_patterns(conversation_id);";

// ── Plan Persistence (v0.8) ──────────────────────────────────────────

/// Execution state for a plan — not the plan text, but the machine-readable
/// state that tracks what's done, what's blocked, and what assumptions hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanState {
    pub id: i64,
    pub conversation_id: i64,

    /// The goal this plan is working toward.
    pub goal: String,

    /// Steps awaiting execution (ordered).
    pub pending_steps: Vec<String>,

    /// Steps that completed successfully.
    pub completed_steps: Vec<String>,

    /// Steps blocked by external conditions (compile error, missing dep, etc.).
    pub blocked_steps: Vec<String>,

    /// Steps invalidated by intervening changes.
    pub invalidated_steps: Vec<String>,

    /// Assumptions at plan creation time. Critical — these are what get
    /// re-validated on plan reactivation.
    pub assumptions: Vec<String>,

    /// Whether this plan is still the active plan for the conversation.
    pub is_active: bool,

    pub created_at: String,
    pub updated_at: String,
}

/// SQL migration for plan_states table.
pub const PLAN_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS plan_states (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id     INTEGER NOT NULL REFERENCES conversations(id),
        goal                TEXT NOT NULL DEFAULT '',
        pending_steps       TEXT NOT NULL DEFAULT '[]',
        completed_steps     TEXT NOT NULL DEFAULT '[]',
        blocked_steps       TEXT NOT NULL DEFAULT '[]',
        invalidated_steps   TEXT NOT NULL DEFAULT '[]',
        assumptions         TEXT NOT NULL DEFAULT '[]',
        is_active           INTEGER NOT NULL DEFAULT 1,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_plan_conv
        ON plan_states(conversation_id);";
