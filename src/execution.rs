use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicI64, Ordering};

/// Logical sequence counter for deterministic execution ordering (P0-6).
/// Incremented on each execution unit creation. Independent of wall clock.
static NEXT_LOGICAL_SEQ: AtomicI64 = AtomicI64::new(1);

/// Return the next logical sequence number for deterministic ordering.
pub fn next_logical_seq() -> i64 {
    NEXT_LOGICAL_SEQ.fetch_add(1, Ordering::Relaxed)
}

// ── Provenance Lineage (typed edges between execution nodes) ──────────

/// Typed lineage edge — what's the relationship between two execution nodes?
/// More specific than "parent_ids", enables replay/audit/explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Node A happens-before node B in wall-clock time (parallel-safe ordering).
    HappensBefore,
}

impl LineageEdge {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::DependsOn => "depends_on",
            Self::InvalidatedBy => "invalidated_by",
            Self::SuggestedBy => "suggested_by",
            Self::CorrectedBy => "corrected_by",
            Self::HappensBefore => "happens_before",
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
            "happens_before" => Some(Self::HappensBefore),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionOutcome {
    /// Tool call succeeded, agent proceeded with reasoning.
    #[default]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

    /// Tool call ID from the upstream LLM (for matching results in parallel groups).
    #[serde(default)]
    pub tool_call_id: String,

    /// Span ID for distributed tracing (ExecutionSpan).
    #[serde(default)]
    pub span_id: String,
    /// Parent span ID for nested/parallel execution.
    #[serde(default)]
    pub parent_span_id: String,
    /// Execution mode: "sequential" or "parallel".
    #[serde(default)]
    pub span_mode: String,
    /// UUID grouping parallel tool calls.
    #[serde(default)]
    pub parallel_group: String,

    /// Epoch timestamp (ms) for stable ordering (P0 audit).
    #[serde(default)]
    pub epoch_ms: i64,
    /// Replay session ID for replay lineage (P0 audit).
    #[serde(default)]
    pub replay_session_id: String,
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
        tool_call_id: &str,
    ) -> Self {
        Self::new_with_span(
            conversation_id, reasoning_before, tool_name, tool_args,
            tool_result, reasoning_after, outcome, related_nodes,
            "", "", "", "", tool_call_id,
        )
    }

    /// Create with full span/parallel metadata.
    pub fn new_with_span(
        conversation_id: i64,
        reasoning_before: &str,
        tool_name: &str,
        tool_args: &str,
        tool_result: &str,
        reasoning_after: &str,
        outcome: ExecutionOutcome,
        related_nodes: &[i64],
        span_id: &str,
        parent_span_id: &str,
        span_mode: &str,
        parallel_group: &str,
        tool_call_id: &str,
    ) -> Self {
        let tool_args_json = serde_json::from_str(tool_args).ok();
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
            span_id: span_id.to_string(),
            parent_span_id: parent_span_id.to_string(),
            span_mode: span_mode.to_string(),
            parallel_group: parallel_group.to_string(),
            tool_call_id: tool_call_id.to_string(),
            epoch_ms: 0,
            replay_session_id: String::new(),
        }
    }

    /// Short display snippet for retrieval results.
    pub fn snippet(&self) -> String {
        let before: String = self.reasoning_before.chars().take(60).collect();
        let after: String = self.reasoning_after.chars().take(60).collect();
        format!(
            "[exec] {} -> {} | before: {}... | after: {}...",
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
                    &tc.id,
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
        created_at        TEXT NOT NULL DEFAULT (datetime('now')),
        span_id           TEXT NOT NULL DEFAULT '',
        parent_span_id    TEXT NOT NULL DEFAULT '',
        span_mode         TEXT NOT NULL DEFAULT '',
        parallel_group    TEXT NOT NULL DEFAULT '',
        tool_call_id      TEXT NOT NULL DEFAULT '',
        epoch_ms          INTEGER NOT NULL DEFAULT 0,
        replay_session_id TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_execution_conv
        ON execution_units(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_execution_tool
        ON execution_units(tool_name);
    CREATE INDEX IF NOT EXISTS idx_execution_span
        ON execution_units(span_id);
    CREATE INDEX IF NOT EXISTS idx_execution_parallel
        ON execution_units(parallel_group);";

/// ALTER TABLE migration for existing databases (adds span/parallel columns).
pub const MIGRATION_ALTER_V5: &str = "
    ALTER TABLE execution_units ADD COLUMN span_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_units ADD COLUMN parent_span_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_units ADD COLUMN span_mode TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_units ADD COLUMN parallel_group TEXT NOT NULL DEFAULT '';";

/// ALTER TABLE migration for existing databases (adds tool_call_id column).
pub const MIGRATION_ALTER_V6: &str = "
    ALTER TABLE execution_units ADD COLUMN tool_call_id TEXT NOT NULL DEFAULT '';";

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

pub const EVENT_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS execution_events (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        -- NO FK constraint on execution_id — events are independent records.
        -- execution_id may reference an execution_units row when applicable
        -- (e.g. event_kind='execution_completed'), but is NULL for stream events.
        -- Events are NOT children of execution_units; they are the authoritative source.
        execution_id        INTEGER,
        event_kind          TEXT NOT NULL,
        event_payload       TEXT NOT NULL,
        seq_no              INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        span_id             TEXT NOT NULL DEFAULT '',
        parent_span_id      TEXT NOT NULL DEFAULT '',
        span_mode           TEXT NOT NULL DEFAULT '',
        parallel_group      TEXT NOT NULL DEFAULT '',
        tool_call_id        TEXT NOT NULL DEFAULT '',
        conv_id             INTEGER NOT NULL DEFAULT 0,
        epoch_ms            INTEGER NOT NULL DEFAULT 0,
        replay_session_id   TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_events_execution
        ON execution_events(execution_id, seq_no);
    CREATE INDEX IF NOT EXISTS idx_events_kind
        ON execution_events(event_kind);
    CREATE INDEX IF NOT EXISTS idx_events_span
        ON execution_events(span_id);
    CREATE INDEX IF NOT EXISTS idx_events_conv
        ON execution_events(conv_id);
    CREATE INDEX IF NOT EXISTS idx_events_replay
        ON execution_events(replay_session_id);";

/// ALTER TABLE migration for existing execution_events table (v0.5).
pub const EVENT_MIGRATION_ALTER_V5: &str = "
    ALTER TABLE execution_events ADD COLUMN span_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_events ADD COLUMN parent_span_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_events ADD COLUMN span_mode TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_events ADD COLUMN parallel_group TEXT NOT NULL DEFAULT '';";

/// ALTER TABLE migration for execution_events (v0.6 — audit P0 columns).
pub const EVENT_MIGRATION_ALTER_V6: &str = "
    ALTER TABLE execution_events ADD COLUMN tool_call_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_events ADD COLUMN conv_id INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE execution_events ADD COLUMN epoch_ms INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE execution_events ADD COLUMN replay_session_id TEXT NOT NULL DEFAULT '';";

/// ALTER TABLE migration for execution_units (v0.6 — audit P0 columns).
pub const EXECUTION_UNITS_ALTER_V6: &str = "
    ALTER TABLE execution_units ADD COLUMN epoch_ms INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE execution_units ADD COLUMN replay_session_id TEXT NOT NULL DEFAULT '';";

// ── Execution Scoring (v0.5.0) ─────────────────────────────────────────

/// Aggregate event counts for scoring — tallied from ExecutionUnit outcomes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventSummary {
    pub total_count: usize,
    pub success_count: usize,
    pub recovered_count: usize,
    pub blocked_count: usize,
    pub cache_hit_count: usize,
    pub stale_count: usize,
    pub replayed_count: usize,
}

impl EventSummary {
    /// Summarize outcomes from a slice of execution units.
    pub fn from_units(units: &[ExecutionUnit]) -> Self {
        let mut s = EventSummary {
            total_count: units.len(),
            ..Default::default()
        };
        for u in units {
            match u.outcome {
                ExecutionOutcome::Success => s.success_count += 1,
                ExecutionOutcome::RecoveredFailure => s.recovered_count += 1,
                ExecutionOutcome::Blocked => s.blocked_count += 1,
                ExecutionOutcome::CacheHit => s.cache_hit_count += 1,
                ExecutionOutcome::Stale => s.stale_count += 1,
                ExecutionOutcome::Replayed => s.replayed_count += 1,
            }
        }
        s
    }
}

/// Aggregate DAG metrics for scoring context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagMetrics {
    /// Number of leaf (raw) DAG nodes.
    pub leaf_count: i64,
    /// Number of summary nodes.
    pub summary_count: i64,
    /// Total tokens stored across all DAG nodes in conversation.
    pub total_tokens: i64,
    /// Maximum token budget (context window size).
    pub max_budget_tokens: i64,
    /// Number of reuse edges in the DAG.
    pub reuse_edge_count: i64,
    /// Total edges.
    pub total_edge_count: i64,
}

impl Default for DagMetrics {
    fn default() -> Self {
        Self {
            leaf_count: 0,
            summary_count: 0,
            total_tokens: 0,
            max_budget_tokens: 128_000,
            reuse_edge_count: 0,
            total_edge_count: 0,
        }
    }
}

/// Execution score — normalized metrics across five dimensions.
/// Each dimension is 0.0–1.0; higher is better for the composite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionScore {
    /// Ratio of successful tool calls to total (Success + CacheHit + Replayed / total).
    pub success_rate: f64,
    /// Ratio of failed/recovered tool calls (RecoveredFailure + Blocked / total).
    /// Higher = worse.
    pub retry_burden: f64,
    /// Fraction of results served from cache or replay.
    pub reuse_fraction: f64,
    /// Token cost relative to budget (clamped 0.0–1.0).
    pub latency_cost: f64,
    /// Estimated hallucination risk from stale caches, failures, and blocked calls.
    pub hallucination_risk: f64,
    /// Weighted composite score (0.0–1.0, higher = better execution).
    pub composite: f64,
}

impl Default for ExecutionScore {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            retry_burden: 0.0,
            reuse_fraction: 0.0,
            latency_cost: 0.0,
            hallucination_risk: 0.0,
            composite: 1.0,
        }
    }
}

impl ExecutionScore {
    /// Compute an execution score from event summaries and DAG metrics.
    ///
    /// Weights (from roadmap scoring dimensions):
    /// - success_rate: 0.35
    /// - retry_burden (inverted): 0.25
    /// - reuse_fraction: 0.15
    /// - latency_cost (inverted): 0.15
    /// - hallucination_risk (inverted): 0.10
    pub fn compute(summary: &EventSummary, dag: &DagMetrics) -> Self {
        let total = summary.total_count.max(1) as f64;

        let success_rate =
            (summary.success_count + summary.cache_hit_count + summary.replayed_count) as f64 / total;

        let retry_burden =
            (summary.recovered_count + summary.blocked_count) as f64 / total;

        let reuse_fraction =
            (summary.cache_hit_count + summary.replayed_count) as f64 / total;

        let budget = dag.max_budget_tokens.max(1) as f64;
        let token_ratio = (dag.total_tokens as f64 / budget).clamp(0.0, 1.0);
        let latency_cost = token_ratio;

        let stale_risk = summary.stale_count as f64 * 0.5;
        let failure_risk = summary.recovered_count as f64 * 0.3;
        let blocked_risk = summary.blocked_count as f64 * 0.2;
        let raw_risk = (stale_risk + failure_risk + blocked_risk) / total;
        let hallucination_risk = raw_risk.clamp(0.0, 1.0);

        let composite = success_rate * 0.35
            + (1.0 - retry_burden) * 0.25
            + reuse_fraction * 0.15
            + (1.0 - latency_cost) * 0.15
            + (1.0 - hallucination_risk) * 0.10;

        Self {
            success_rate,
            retry_burden,
            reuse_fraction,
            latency_cost,
            hallucination_risk,
            composite,
        }
    }

    /// Convenience: compute score directly from execution units + dag metrics.
    pub fn from_units(units: &[ExecutionUnit], dag: &DagMetrics) -> Self {
        let summary = EventSummary::from_units(units);
        Self::compute(&summary, dag)
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn default_score_is_perfect() {
        let score = ExecutionScore::default();
        assert!((score.composite - 1.0).abs() < 1e-9);
        assert!((score.success_rate - 1.0).abs() < 1e-9);
        assert!((score.retry_burden - 0.0).abs() < 1e-9);
    }

    #[test]
    fn event_summary_all_success() {
        let units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "ok".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Success,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
            ExecutionUnit {
                id: 2, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "read_file".into(),
                tool_args: String::new(), tool_result: "data".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::CacheHit,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let summary = EventSummary::from_units(&units);
        assert_eq!(summary.total_count, 2);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.cache_hit_count, 1);
    }

    #[test]
    fn score_pure_success_is_high() {
        let units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "ok".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Success,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let dag = DagMetrics::default();
        let score = ExecutionScore::from_units(&units, &dag);
        // 0.85 = max with zero reuse fraction (weighted: 0.35+0.25+0+0.15+0.10)
        assert!((score.composite - 0.85).abs() < 1e-9);
        assert!((score.success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn score_all_failures_is_low() {
        let units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "Error: not found".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Blocked,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
            ExecutionUnit {
                id: 2, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "build".into(),
                tool_args: String::new(), tool_result: "fail".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::RecoveredFailure,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let dag = DagMetrics::default();
        let score = ExecutionScore::from_units(&units, &dag);
        assert!(score.composite < 0.5, "composite should be low for all failures: {}", score.composite);
        assert!(score.success_rate < 0.1);
        assert!(score.retry_burden > 0.9);
    }

    #[test]
    fn score_cache_hits_higher_reuse() {
        let units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "ok".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Success,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
            ExecutionUnit {
                id: 2, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "cached".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::CacheHit,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let dag = DagMetrics::default();
        let score = ExecutionScore::from_units(&units, &dag);
        assert!(score.reuse_fraction > 0.4, "should have significant reuse: {}", score.reuse_fraction);
    }

    #[test]
    fn score_stale_cache_increases_risk() {
        let clean = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "ok".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Success,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let stale_units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "stale".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Stale,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        let dag = DagMetrics::default();
        let clean_score = ExecutionScore::from_units(&clean, &dag);
        let stale_score = ExecutionScore::from_units(&stale_units, &dag);
        assert!(stale_score.hallucination_risk > clean_score.hallucination_risk,
            "stale cache should increase hallucination risk");
    }

    #[test]
    fn score_dag_budget_scales_latency() {
        let units = vec![
            ExecutionUnit {
                id: 1, conversation_id: 1,
                reasoning_before: String::new(), tool_name: "grep".into(),
                tool_args: String::new(), tool_result: "ok".into(),
                reasoning_after: String::new(),
                outcome: ExecutionOutcome::Success,
                related_nodes: vec![], created_at: String::new(),
                tool_args_json: None, reasoning_steps: vec![],
            ..Default::default()
            },
        ];
        // Tight budget → high latency cost
        let tight_dag = DagMetrics { total_tokens: 100_000, max_budget_tokens: 128_000, ..Default::default() };
        // Generous budget → low latency cost
        let loose_dag = DagMetrics { total_tokens: 1_000, max_budget_tokens: 128_000, ..Default::default() };
        let tight_score = ExecutionScore::from_units(&units, &tight_dag);
        let loose_score = ExecutionScore::from_units(&units, &loose_dag);
        assert!(tight_score.latency_cost > loose_score.latency_cost,
            "tight budget should have higher latency cost");
        assert!(tight_score.composite < loose_score.composite,
            "tight budget should lower composite");
    }
}
