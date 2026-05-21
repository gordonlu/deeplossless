use serde::{Deserialize, Serialize};

// ── Provenance Lineage (typed edges between execution nodes) ──────────

/// Typed lineage edge — what's the relationship between two execution nodes?
/// More specific than "parent_ids", enables replay/audit/explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageEdge {
    /// Node B was derived from node A (summary, compaction).
    DerivedFrom,
    /// Node B depends on node A's output (tool chain).
    DependsOn,
    /// Node B was invalidated because node A changed.
    InvalidatedBy,
    /// Node B's fix was suggested by failure pattern from node A.
    SuggestedBy,
    /// Node B corrected the error from node A.
    CorrectedBy,
}

impl LineageEdge {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::DependsOn => "depends_on",
            Self::InvalidatedBy => "invalidated_by",
            Self::SuggestedBy => "suggested_by",
            Self::CorrectedBy => "corrected_by",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "derived_from" => Some(Self::DerivedFrom),
            "depends_on" => Some(Self::DependsOn),
            "invalidated_by" => Some(Self::InvalidatedBy),
            "suggested_by" => Some(Self::SuggestedBy),
            "corrected_by" => Some(Self::CorrectedBy),
            _ => None,
        }
    }
}

/// SQL migration for lineage edges.
pub const LINEAGE_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS lineage_edges (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id     INTEGER NOT NULL,
        to_id       INTEGER NOT NULL,
        kind        TEXT NOT NULL DEFAULT 'depends_on',
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_lineage_from ON lineage_edges(from_id);
    CREATE INDEX IF NOT EXISTS idx_lineage_to ON lineage_edges(to_id);";

// ── Structured Reasoning ──────────────────────────────────────────────

/// Runtime-relevant reasoning kind — not free-form text, but typed execution intelligence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningKind {
    /// An assumption the agent is making.
    Assumption,
    /// A hypothesis being tested.
    Hypothesis,
    /// Validation of a previous hypothesis or assumption.
    Validation,
    /// Recognition that a previous approach failed.
    Failure,
    /// Resolution or fix applied.
    Resolution,
}

/// A structured reasoning step — execution-oriented, not chain-of-thought essay.
/// Links to execution units for provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// What kind of reasoning this step represents.
    pub kind: ReasoningKind,
    /// The reasoning content (concise, not verbose).
    pub content: String,
    /// Execution unit this step is linked to, if any.
    pub execution_unit_id: Option<i64>,
    /// Lineage: which previous reasoning step(s) this derives from.
    #[serde(default)]
    pub derived_from: Vec<i64>,
}

/// Normalize reasoning text to a canonical form for dedup.
/// Collapses whitespace, lowercases, trims — similar reasoning → same hash.
pub fn normalize_reasoning(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
        .chars()
        .take(200)
        .collect()
}

/// Maximum lineage edges per conversation before TTL compaction kicks in.
const LINEAGE_TTL: usize = 1000;

/// Apply TTL-based lineage compaction: keep only the most recent edges,
/// squash intermediate DerivedFrom chains, drop edges beyond TTL.
pub fn compact_lineage_with_ttl(
    edges: &[(i64, i64, LineageEdge)],
) -> Vec<(i64, i64, LineageEdge)> {
    if edges.len() <= LINEAGE_TTL {
        return compact_lineage(edges);
    }
    // Keep last TTL edges, compact the rest
    let recent = &edges[edges.len().saturating_sub(LINEAGE_TTL)..];
    compact_lineage(recent)
}

/// Compress lineage by transitive collapse of DerivedFrom edges.
/// A → B → C becomes A → C (preserves the full causation chain).
pub fn compact_lineage(
    edges: &[(i64, i64, LineageEdge)],
) -> Vec<(i64, i64, LineageEdge)> {
    use std::collections::{HashMap, HashSet};

    // Build adjacency: from → [(to, kind)]
    let mut adj: HashMap<i64, Vec<(i64, LineageEdge)>> = HashMap::new();
    for &(from, to, kind) in edges {
        adj.entry(from).or_default().push((to, kind));
    }

    // Transitive collapse: for each DerivedFrom chain, emit source → ultimate target
    let mut result = Vec::new();
    let mut collapsed: HashSet<(i64, i64)> = HashSet::new();

    for &(from, to, kind) in edges {
        match kind {
            LineageEdge::DerivedFrom => {
                // Follow the chain to find the ultimate target
                let mut current = to;
                let mut depth = 0;
                while depth < 10 {
                    if let Some(nexts) = adj.get(&current)
                        && let Some((next, LineageEdge::DerivedFrom)) = nexts.first() {
                        current = *next;
                        depth += 1;
                        continue;
                    }
                    break;
                }
                // Emit source → ultimate if not already covered
                if from != current && collapsed.insert((from, current)) {
                    result.push((from, current, LineageEdge::DerivedFrom));
                }
            }
            // Non-DerivedFrom edges pass through unchanged
            _ => {
                if collapsed.insert((from, to)) {
                    result.push((from, to, kind));
                }
            }
        }
    }
    result
}

/// Structured reasoning summary — preserves semantics for replay and retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSummary {
    pub assumptions: Vec<String>,
    pub failures: Vec<String>,
    pub resolutions: Vec<String>,
}

impl ReasoningSummary {
    /// Extract a structured summary from reasoning steps.
    pub fn from_steps(steps: &[ReasoningStep]) -> Self {
        let assumptions: Vec<String> = steps.iter()
            .filter(|s| s.kind == ReasoningKind::Assumption)
            .map(|s| s.content.clone())
            .collect();
        let failures: Vec<String> = steps.iter()
            .filter(|s| s.kind == ReasoningKind::Failure)
            .map(|s| s.content.clone())
            .collect();
        let resolutions: Vec<String> = steps.iter()
            .filter(|s| s.kind == ReasoningKind::Resolution)
            .map(|s| s.content.clone())
            .collect();
        Self { assumptions, failures, resolutions }
    }
}

/// Compress a sequence of reasoning steps into a structured summary.
/// Preserves Assumption + Failure + Resolution in typed fields.
pub fn distill_reasoning(steps: &[ReasoningStep]) -> String {
    let summary = ReasoningSummary::from_steps(steps);
    let mut out = String::new();
    for a in &summary.assumptions { out.push_str(&format!("Assumed: {a}\n")); }
    for f in &summary.failures { out.push_str(&format!("Failed: {f}\n")); }
    for r in &summary.resolutions { out.push_str(&format!("Resolved: {r}\n")); }
    out.trim().to_string()
}

// ── Execution Outcome ─────────────────────────────────────────────────

/// Outcome of a tool execution within an agent reasoning loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionOutcome {
    /// Tool call succeeded, agent proceeded with reasoning.
    Success,
    /// Tool call failed but agent recovered (retry, fallback path).
    RecoveredFailure,
    /// Tool call failed and agent could not recover.
    Blocked,
    /// Result served from cache — zero token execution.
    CacheHit,
    /// Cache entry existed but was stale (content changed since).
    Stale,
    /// Result replayed from prior execution (deterministic replay).
    Replayed,
}

impl ExecutionOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::RecoveredFailure => "recovered",
            Self::Blocked => "blocked",
            Self::CacheHit => "cache_hit",
            Self::Stale => "stale",
            Self::Replayed => "replayed",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "success" => Some(Self::Success),
            "recovered" => Some(Self::RecoveredFailure),
            "blocked" => Some(Self::Blocked),
            "cache_hit" => Some(Self::CacheHit),
            "stale" => Some(Self::Stale),
            "replayed" => Some(Self::Replayed),
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
    /// Legacy: raw arguments string. Prefer `tool_args_json`.
    pub tool_args: String,
    /// Structured arguments (v0.3).
    #[serde(default)]
    pub tool_args_json: Option<serde_json::Value>,

    /// Tool execution result (truncated to preserve budget).
    pub tool_result: String,

    /// The assistant's reflection / next reasoning after seeing the result.
    pub reasoning_after: String,

    /// Structured reasoning steps extracted from before/after (v0.3).
    #[serde(default)]
    pub reasoning_steps: Vec<ReasoningStep>,

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
        let tool_args_json = serde_json::from_str(tool_args).ok();
        // Outcome-guided reasoning extraction — infers kind from what happened, not keywords
        let reasoning_steps = {
            let mut steps = Vec::new();
            if !reasoning_before.is_empty() {
                let kind = match &outcome {
                    ExecutionOutcome::RecoveredFailure | ExecutionOutcome::Blocked =>
                        ReasoningKind::Failure,
                    ExecutionOutcome::CacheHit | ExecutionOutcome::Replayed =>
                        ReasoningKind::Validation,
                    _ => ReasoningKind::Assumption,
                };
                steps.push(ReasoningStep {
                    kind,
                    content: reasoning_before.chars().take(300).collect(),
                    execution_unit_id: None,
                    derived_from: vec![],
                });
            }
            if !reasoning_after.is_empty() {
                let kind = match &outcome {
                    ExecutionOutcome::Success => ReasoningKind::Resolution,
                    ExecutionOutcome::RecoveredFailure | ExecutionOutcome::Blocked =>
                        ReasoningKind::Hypothesis,
                    _ => ReasoningKind::Validation,
                };
                steps.push(ReasoningStep {
                    kind,
                    content: reasoning_after.chars().take(300).collect(),
                    execution_unit_id: None,
                    derived_from: vec![],
                });
            }
            steps
        };

        Self {
            id: 0,
            conversation_id,
            reasoning_before: reasoning_before.to_string(),
            tool_name: tool_name.to_string(),
            tool_args: tool_args.to_string(),
            tool_args_json,
            tool_result: tool_result.to_string(),
            reasoning_after: reasoning_after.to_string(),
            reasoning_steps,
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
            let related = Vec::new();

            // Collect all tool call→result pairs for this assistant turn
            let mut last_tool_end = i; // track how far we advanced
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
                    if next.role == "assistant" { break; }
                    j += 1;
                }
                last_tool_end = j; // remember furthest advance

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
            i = last_tool_end; // skip past all processed tool results
            continue; // skip the i += 1 at loop bottom
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

/// SQL migration for reasoning_steps table.
pub const REASONING_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS reasoning_steps (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        kind                TEXT NOT NULL DEFAULT 'assumption',
        content             TEXT NOT NULL DEFAULT '',
        execution_unit_id   INTEGER REFERENCES execution_units(id),
        derived_from        TEXT NOT NULL DEFAULT '[]',
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_reasoning_kind ON reasoning_steps(kind);";

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

    /// Canonical execution key for dedup (tool + normalized args hash).
    #[serde(default)]
    pub execution_key: String,

    /// Runtime environment fingerprint (model, provider, tool versions).
    /// Ensures failure patterns aren't misapplied across different environments.
    #[serde(default)]
    pub environment_fingerprint: String,

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
        execution_key           TEXT NOT NULL DEFAULT '',
        environment_fingerprint TEXT NOT NULL DEFAULT '',
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
