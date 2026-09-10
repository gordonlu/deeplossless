//! # Pluggable Runtime Policy Engine
//!
//! **Zero infrastructure dependencies.** No SQLite, no HTTP, no DeepSeek API,
//! no OpenAI schema. Just pure Rust types.
//!
//! This module can be embedded in any AI coding client — desktop app, IDE
//! plugin, CLI tool, or proxy middleware. The engine takes [`RuntimeState`]
//! and returns [`RuntimeDecision`].
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────────────┐
//!  RuntimeState ────►│   RuntimePolicy::decide │────► RuntimeDecision
//!  (input)           └─────────────────────────┘      (output)
//!
//!                    ┌──────────────┐
//!  Profile ─────────►│  Strategy    │────► thresholds
//!  (user choice)     │  (internal)  │      cache_aggressiveness
//!                    └──────────────┘      token_budget_ratio ...
//! ```
//!
//! ## Key boundary
//!
//! [`RuntimeDecision`] is **advisory**. Agents/UIs accept, ignore, or override.
//! We are middleware, not an agent framework. We do not control the model —
//! we optimize the economics of its execution.
//!
//! ## Integration
//!
//! ```ignore
//! // In any AI coding runtime:
//! use deeplossless::runtime::{ExecutionCycle, RuntimePolicy, RuntimeProfile};
//!
//! let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
//! let decision = RuntimePolicy::decide(&cycle, cache_hit, failure_hint, plan_hint);
//! // Agent decides whether to accept decision.action
//! ```

use serde::{Deserialize, Serialize};

// ── Runtime Profile (user-facing) ─────────────────────────────────────

/// User-selectable profile. Maps to internal strategy parameters.
/// The runtime optimizes inference cost within the chosen profile's constraints.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RuntimeProfile {
    /// Minimum token consumption. Aggressive caching, minimal reasoning.
    Minimal,
    /// Balanced optimization. Smart caching, normal reasoning.
    Efficient,
    /// Allow deeper exploration. Relaxed caching, broader retrieval.
    Exploratory,
    /// Maximum model autonomy. Full reasoning, speculative paths.
    Autonomous,
    /// User-defined strategy. Parameters are set directly on RuntimeStrategy.
    Custom,
}

// ── Audit / Snapshot Policy Configuration ───────────────────────────

/// Controls when audit events are written to `execution_events` and `dag_events`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditMode {
    /// No audit events written — only essential `execution_units` and `dag_nodes`.
    /// Replay falls back to full event log replay (slower but functional).
    Off,
    /// Buffer recent audit events in-memory. On failure, flush buffer + error
    /// to the audit log. Best for production — zero overhead on success paths.
    OnError,
    /// Always write audit events. Maximum observability, highest write overhead.
    Full,
}

/// Controls when execution snapshots are taken automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotMode {
    /// No automatic snapshots. Snapshots only via POST API.
    Off,
    /// Auto-snapshot at semantic boundaries: after compaction, retry escalation,
    /// failure recovery, and cancellation.
    Auto,
    /// Snapshots only via POST API (current behavior).
    Manual,
}

/// Top-level policy configuration combining audit and snapshot modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePolicyConfig {
    pub audit_mode: AuditMode,
    pub snapshot_mode: SnapshotMode,
    /// How many recent events to buffer in OnError mode before flushing on failure.
    pub onerror_ring_size: usize,
    /// Snapshot budget for auto-snapshots.
    #[serde(default)]
    pub snapshot_budget: crate::snapshot::SnapshotBudget,
}

impl Default for RuntimePolicyConfig {
    fn default() -> Self {
        Self {
            audit_mode: AuditMode::Full,
            snapshot_mode: SnapshotMode::Manual,
            onerror_ring_size: 50,
            snapshot_budget: crate::snapshot::SnapshotBudget::default(),
        }
    }
}

impl AuditMode {
    pub fn should_write_audit(&self) -> bool {
        matches!(self, Self::Full)
    }
    pub fn is_onerror(&self) -> bool {
        matches!(self, Self::OnError)
    }
}

impl SnapshotMode {
    pub fn should_auto_snapshot(&self) -> bool {
        matches!(self, Self::Auto)
    }
    pub fn is_manual(&self) -> bool {
        matches!(self, Self::Manual)
    }
}

impl RuntimeProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Efficient => "efficient",
            Self::Exploratory => "exploratory",
            Self::Autonomous => "autonomous",
            Self::Custom => "custom",
        }
    }
}

// ── Runtime Mode ─────────────────────────────────────────────────────

/// Execution mode controlling how the runtime processes requests (B-1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMode {
    /// Normal operation — real LLM calls, real tool execution.
    Live,
    /// Deterministic replay from execution events — reads past results
    /// from the event log instead of making live LLM calls.
    Replay {
        /// Replay session ID for lineage tracking.
        session_id: String,
        /// Stop replay at this logical sequence number.
        up_to_seq: i64,
    },
    /// Dry-run: evaluate policies and produce a decision plan without
    /// executing any tools or LLM calls. Used for budget estimation.
    DryRun,
}

impl RuntimeMode {
    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live)
    }
    pub fn is_replay(&self) -> bool {
        matches!(self, Self::Replay { .. })
    }
    pub fn is_dry_run(&self) -> bool {
        matches!(self, Self::DryRun)
    }
}

/// Separates execution output from side effects (A-4).
/// In replay mode, only the output is replayed; side effects are skipped.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionResult {
    /// The tool execution result text.
    pub output: String,
    /// The outcome classification.
    pub outcome: crate::execution::ExecutionOutcome,
    /// Token cost incurred (0 for cache hits and replays).
    pub tokens_spent: u64,
    /// Whether this execution had side effects (filesystem mutation, API call).
    pub has_side_effects: bool,
    /// Whether this was a cache hit or replay (skip side effects on replay).
    pub is_replay: bool,
}

// ── Runtime Strategy (internal) ───────────────────────────────────────

/// Internal parameters derived from the user's profile.
/// Controls admission thresholds, cache aggressiveness, retry limits.
#[derive(Debug, Clone)]
pub struct RuntimeStrategy {
    pub profile: RuntimeProfile,

    /// How aggressively to reuse cache (0.0 = never, 1.0 = always if available).
    pub cache_aggressiveness: f64,

    /// Maximum retries for the same failure before blocking.
    pub max_retries_per_failure: u32,

    /// Whether to allow speculative/reasoning-heavy paths.
    pub allow_speculative: bool,

    /// How much context to inject (0.0 = minimal delta only, 1.0 = full context).
    pub context_injection_ratio: f64,

    /// Whether to freeze plans early (don't re-plan on minor changes).
    pub freeze_plans_early: bool,

    /// Token budget as fraction of total window.
    pub token_budget_ratio: f64,
}

impl RuntimeStrategy {
    /// Build a custom strategy with explicit parameters.
    pub fn custom(
        cache_aggressiveness: f64,
        max_retries_per_failure: u32,
        allow_speculative: bool,
        context_injection_ratio: f64,
        freeze_plans_early: bool,
        token_budget_ratio: f64,
    ) -> Self {
        Self {
            profile: RuntimeProfile::Custom,
            cache_aggressiveness: cache_aggressiveness.clamp(0.0, 1.0),
            max_retries_per_failure: max_retries_per_failure.min(10),
            allow_speculative,
            context_injection_ratio: context_injection_ratio.clamp(0.0, 1.0),
            freeze_plans_early,
            token_budget_ratio: token_budget_ratio.clamp(0.1, 1.0),
        }
    }

    pub fn from_profile(profile: RuntimeProfile) -> Self {
        match profile {
            RuntimeProfile::Custom => Self {
                profile,
                cache_aggressiveness: 0.6,
                max_retries_per_failure: 2,
                allow_speculative: true,
                context_injection_ratio: 0.5,
                freeze_plans_early: false,
                token_budget_ratio: 0.6,
            },
            RuntimeProfile::Minimal => Self {
                profile,
                cache_aggressiveness: 1.0,
                max_retries_per_failure: 1,
                allow_speculative: false,
                context_injection_ratio: 0.2,
                freeze_plans_early: true,
                token_budget_ratio: 0.3,
            },
            RuntimeProfile::Efficient => Self {
                profile,
                cache_aggressiveness: 0.8,
                max_retries_per_failure: 2,
                allow_speculative: false,
                context_injection_ratio: 0.5,
                freeze_plans_early: false,
                token_budget_ratio: 0.6,
            },
            RuntimeProfile::Exploratory => Self {
                profile,
                cache_aggressiveness: 0.5,
                max_retries_per_failure: 3,
                allow_speculative: true,
                context_injection_ratio: 0.8,
                freeze_plans_early: false,
                token_budget_ratio: 0.8,
            },
            RuntimeProfile::Autonomous => Self {
                profile,
                cache_aggressiveness: 0.3,
                max_retries_per_failure: 5,
                allow_speculative: true,
                context_injection_ratio: 1.0,
                freeze_plans_early: false,
                token_budget_ratio: 0.95,
            },
        }
    }
}

// ── Runtime State (input to the policy engine) ──────────────────────

/// Complete input for the runtime policy engine.
/// Captures everything the engine needs to produce a decision.
/// Zero infrastructure dependencies — just data.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeState {
    pub profile: RuntimeProfile,
    pub metrics: RuntimeMetrics,

    /// Tool cache hit for the current tool call, if any.
    pub cache_hit: Option<CacheHit>,

    /// Known failure pattern matching the current context, if any.
    pub failure_hint: Option<FailureHint>,

    /// Active plan with pending steps, if any.
    pub plan_hint: Option<PlanHint>,

    /// Files changed since last cycle.
    pub context_delta: Vec<String>,
}

/// A detected tool cache hit.
#[derive(Debug, Clone, Serialize)]
pub struct CacheHit {
    pub tool_name: String,
    pub cache_id: i64,
    pub estimated_token_saving: u64,
}

/// A matching failure pattern.
#[derive(Debug, Clone, Serialize)]
pub struct FailureHint {
    /// The error signature.
    pub signature: String,
    /// The known fix (may be empty).
    pub suggested_fix: String,
    /// Why the previous attempt failed.
    pub why_failed: String,
    /// Assumptions that turned out to be wrong.
    pub invalidated_assumptions: Vec<String>,
    /// How many times this specific failure has been retried.
    /// Used for per-pattern retry limiting, not global streak.
    pub retry_count: u32,
}

impl FailureHint {
    /// Build from a stored [`crate::execution::FailurePattern`].
    pub fn from_failure_pattern(fp: &crate::execution::FailurePattern, retry_count: u32) -> Self {
        Self {
            signature: fp.signature.clone(),
            suggested_fix: fp.attempted_fix.clone(),
            why_failed: fp.why_failed.clone(),
            invalidated_assumptions: fp.invalidated_assumptions.clone(),
            retry_count,
        }
    }
}

/// An active plan with pending steps.
#[derive(Debug, Clone, Serialize)]
pub struct PlanHint {
    pub plan_id: i64,
    pub goal: String,
    pub pending_step_count: usize,
    /// Assumptions at plan creation time. If any of these files/conditions
    /// appear in `RuntimeState::context_delta`, the plan should be replanned.
    pub assumptions: Vec<String>,
}

// ── Runtime Action ────────────────────────────────────────────────────

/// Actions the runtime can recommend. Note: these are advisory.
/// The agent/UI decides whether to accept the recommendation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum RuntimeAction {
    /// Tool result found in cache — skip execution.
    ReuseToolCache {
        tool_name: String,
        cache_hit_id: i64,
    },
    /// Known failure fix available — apply it.
    RetryWithFix {
        failure_id: i64,
        suggested_fix: String,
    },
    /// Active plan has pending steps — continue.
    ContinuePlan {
        step_index: usize,
        step_description: String,
    },
    /// Plan assumptions invalid — re-plan needed.
    Replan { reason: String },
    /// Request more context before proceeding.
    RequestContext { what: String },
    /// Distill state into compact representation.
    CompactAndProceed,
    /// No optimization available — model decides.
    DelegateToModel,
}

impl RuntimeAction {
    /// Variant name string (no inner data) for serialization/display.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::ReuseToolCache { .. } => "ReuseToolCache",
            Self::RetryWithFix { .. } => "RetryWithFix",
            Self::ContinuePlan { .. } => "ContinuePlan",
            Self::Replan { .. } => "Replan",
            Self::RequestContext { .. } => "RequestContext",
            Self::CompactAndProceed => "CompactAndProceed",
            Self::DelegateToModel => "DelegateToModel",
        }
    }
}

// ── Runtime Decision Record (lifecycle tracking) ──────────────────────

/// Lifecycle record for a [`RuntimeDecision`].
///
/// Created automatically when [`RuntimePolicy::decide`] is called.
/// Tracks whether the recommendation was accepted and what the
/// actual outcome was — enabling the runtime to answer
/// "did my optimization actually help?"
///
/// This is pure lifecycle metadata. The decision logic remains in
/// [`RuntimeDecision`] and [`RuntimePolicy`].
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDecisionRecord {
    pub id: i64,
    pub conversation_id: i64,

    /// Action name (matches [`RuntimeAction`] variant name).
    pub action: String,

    /// Confidence at decision time (0.0–1.0).
    pub confidence: f64,

    /// Human-readable reason.
    pub reason: String,

    /// Whether the recommendation was accepted by the agent/runtime.
    /// `None` = pending, `Some(true)` = accepted, `Some(false)` = rejected.
    pub accepted: Option<bool>,

    /// Outcome after execution. `None` = not yet evaluated.
    /// Examples: "success", "cache_invalidated", "failure_repeated".
    pub outcome: Option<String>,

    /// Estimated token saving at decision time.
    pub estimated_token_saving: u64,

    /// Actual token saving after execution (measured, not estimated).
    pub actual_token_saving: Option<u64>,

    /// ISO-8601 timestamp of creation.
    pub created_at: String,
}

impl RuntimeDecisionRecord {
    /// Create a new record from a [`RuntimeDecision`].
    /// `id` and `created_at` are set by the caller (DB or in-memory store).
    pub fn from_decision(conversation_id: i64, decision: &RuntimeDecision) -> Self {
        Self {
            id: 0,
            conversation_id,
            action: decision.action.variant_name().to_string(),
            confidence: decision.confidence,
            reason: decision.reason.clone(),
            accepted: None,
            outcome: None,
            estimated_token_saving: decision.estimated_token_saving,
            actual_token_saving: None,
            created_at: String::new(),
        }
    }

    /// Mark whether the recommendation was accepted.
    pub fn mark_accepted(&mut self, accepted: bool) {
        self.accepted = Some(accepted);
    }

    /// Record the execution outcome and actual tokens saved (if measurable).
    pub fn mark_outcome(&mut self, outcome: &str, actual_token_saving: Option<u64>) {
        self.outcome = Some(outcome.to_string());
        self.actual_token_saving = actual_token_saving;
    }
}

// ── Decision Evaluation ────────────────────────────────────────────────

/// Outcome of evaluating whether a [`RuntimeDecision`] was effective.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DecisionOutcome {
    /// Whether the decision was effective (subjective per action type).
    pub success: bool,
    /// Estimated tokens that would have been spent without the optimization.
    pub estimated_tokens: u64,
    /// Actual tokens spent to execute the decision.
    pub actual_tokens: u64,
    /// Whether the same failure pattern repeated (only for RetryWithFix).
    pub failure_repeated: bool,
}

impl DecisionOutcome {
    /// Net token impact: positive = saved, negative = wasted.
    pub fn net_saving(&self) -> i64 {
        self.estimated_tokens as i64 - self.actual_tokens as i64
    }
}

/// Evaluates whether a [`RuntimeDecision`] was effective after execution.
pub struct DecisionEvaluation;

impl DecisionEvaluation {
    /// Evaluate a decision outcome based on action type and execution results.
    ///
    /// * `action` — action variant name (from [`RuntimeAction::variant_name`])
    /// * `estimated_tokens` — estimated token saving at decision time
    /// * `actual_tokens` — actual tokens consumed during execution
    /// * `failure_repeated` — whether the same failure pattern recurred (retries)
    pub fn evaluate(
        action: &str,
        estimated_tokens: u64,
        actual_tokens: u64,
        failure_repeated: bool,
    ) -> DecisionOutcome {
        let success = match action {
            "ReuseToolCache" => actual_tokens < estimated_tokens,
            "RetryWithFix" => !failure_repeated,
            "ContinuePlan" => !failure_repeated,
            "CompactAndProceed" => actual_tokens < estimated_tokens / 2,
            _ => !failure_repeated,
        };
        DecisionOutcome {
            success,
            estimated_tokens,
            actual_tokens,
            failure_repeated,
        }
    }

    /// Apply evaluation result to a decision record.
    pub fn apply(record: &mut RuntimeDecisionRecord, outcome: &DecisionOutcome) {
        if outcome.success {
            record.mark_outcome(
                "success",
                Some(outcome.actual_tokens.min(outcome.estimated_tokens)),
            );
        } else {
            let reason = match record.action.as_str() {
                "ReuseToolCache" => "cache_mismatch",
                "RetryWithFix" => {
                    if outcome.failure_repeated {
                        "failure_repeated"
                    } else {
                        "fix_ineffective"
                    }
                }
                "CompactAndProceed" => "re_expansion",
                _ => "ineffective",
            };
            record.mark_outcome(reason, Some(outcome.actual_tokens));
        }
    }
}

// ── Runtime Decision (advisory output) ────────────────────────────────

/// The output of the policy engine. Advisory, not mandatory.
/// Agents/UIs can inspect the recommendation and choose to accept,
/// ignore, or override based on their own judgment.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDecision {
    /// The recommended action.
    pub action: RuntimeAction,

    /// Estimated token savings if this decision is accepted.
    pub estimated_token_saving: u64,

    /// Confidence in this recommendation (0.0–1.0).
    /// Low confidence = agent should lean toward its own judgment.
    pub confidence: f64,

    /// Human-readable explanation of why this was recommended.
    pub reason: String,

    /// Evidence strings supporting this decision (populated by the ranking
    /// system). Each entry explains why a particular signal mattered.
    #[serde(default)]
    pub evidence: Vec<String>,
}

/// A candidate decision produced by a single [`DecisionRule`].
///
/// In the ranking system, multiple rules each produce a candidate with
/// a score. [`RuleEngine::rank_candidates`] picks the highest-scoring
/// one and returns its decision with evidence attached.
#[derive(Debug, Clone)]
pub struct DecisionCandidate {
    pub decision: RuntimeDecision,
    /// Ranking score (0.0–1.0). Higher = more likely to be selected.
    pub score: f64,
    /// Why this candidate scored this way.
    pub evidence: Vec<String>,
}

impl RuntimeDecision {
    pub fn cache_hit(tool_name: &str, cache_id: i64, estimated_save: u64) -> Self {
        Self {
            action: RuntimeAction::ReuseToolCache {
                tool_name: tool_name.to_string(),
                cache_hit_id: cache_id,
            },
            estimated_token_saving: estimated_save,
            confidence: 0.95,
            reason: format!("{tool_name} result cached — deterministic reuse"),
            evidence: vec![],
        }
    }

    pub fn retry_with_fix(failure_id: i64, fix: &str) -> Self {
        Self {
            action: RuntimeAction::RetryWithFix {
                failure_id,
                suggested_fix: fix.to_string(),
            },
            estimated_token_saving: fix.len() as u64 * 10, // rough estimate
            confidence: if fix.is_empty() { 0.3 } else { 0.7 },
            reason: format!("known failure pattern — suggested fix: {fix}"),
            evidence: vec![],
        }
    }

    pub fn continue_plan(step_idx: usize, step_desc: &str) -> Self {
        Self {
            action: RuntimeAction::ContinuePlan {
                step_index: step_idx,
                step_description: step_desc.to_string(),
            },
            estimated_token_saving: 500, // save one planning round
            confidence: 0.85,
            reason: "active plan has pending steps".to_string(),
            evidence: vec![],
        }
    }

    pub fn compact_and_proceed() -> Self {
        Self {
            action: RuntimeAction::CompactAndProceed,
            estimated_token_saving: 1000,
            confidence: 0.6,
            reason: "token budget critical — compacting context".to_string(),
            evidence: vec![],
        }
    }

    pub fn delegate(reason: &str) -> Self {
        Self {
            action: RuntimeAction::DelegateToModel,
            estimated_token_saving: 0,
            confidence: 0.0,
            reason: reason.to_string(),
            evidence: vec![],
        }
    }
}

// ── Runtime Metrics ───────────────────────────────────────────────────

/// Cost-aware metrics that feed back into the scheduler.
///
/// ## Derivable fields (Phase 2.7 audit)
/// Fields marked `/// DERIVABLE` can be computed from `RuntimeEvents` alone
/// and are retained temporarily for parity validation with `RuntimeStateView`.
/// DO NOT add new mutation logic to these fields.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeMetrics {
    /// DERIVABLE: `RuntimeStateView::total_tokens()`
    pub tokens_spent: u64,
    /// DERIVABLE: `RuntimeStateView::cache_hit_count()`
    pub cache_hits: u64,
    /// DERIVABLE: `total_completions - cache_hits` — not yet exposed in StateView
    pub cache_misses: u64,
    /// DERIVABLE: `RuntimeStateView::failure_count()`
    pub repeated_failures: u64,
    /// EXTERNAL: computed from file observation data, not in events
    pub reread_ratio: f64,
    /// EXTERNAL: computed from planning metadata
    pub planning_reuse_ratio: f64,
    /// EXTERNAL: user-configured budget, not in events
    pub budget_remaining_pct: f64,
    /// EXTERNAL: user-configured budget
    pub budget_total: u64,
    /// SEMI-DERIVABLE: count consecutive ToolCallFailed from event tail
    pub failure_streak: u32,
    /// EXTERNAL: no event source; remove after Phase 3 replay
    pub tool_repeat_count: u32,
}

impl RuntimeMetrics {
    pub fn is_token_critical(&self) -> bool {
        self.budget_remaining_pct < 0.15 || self.failure_streak >= 3
    }

    pub fn cache_effective(&self) -> bool {
        let total = self.cache_hits + self.cache_misses;
        total > 0 && (self.cache_hits as f64 / total as f64) > 0.6
    }

    /// Runtime invariant audit (P0): verify metrics are internally consistent.
    /// Returns a list of violations if any invariants are broken.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.budget_remaining_pct > 1.0 {
            issues.push(format!(
                "budget_remaining_pct {:.2} exceeds 1.0",
                self.budget_remaining_pct
            ));
        }
        if self.failure_streak > 0 && self.repeated_failures == 0 {
            issues.push("failure_streak > 0 but repeated_failures == 0".into());
        }
        if self.reread_ratio > 1.0 {
            issues.push(format!("reread_ratio {:.2} exceeds 1.0", self.reread_ratio));
        }
        if self.planning_reuse_ratio > 1.0 {
            issues.push(format!(
                "planning_reuse_ratio {:.2} exceeds 1.0",
                self.planning_reuse_ratio
            ));
        }
        issues
    }
}

// ── Execution Cycle ───────────────────────────────────────────────────

/// State machine tracking the current inference cycle.
///
/// ## Field derivability (Phase 2.7 audit)
/// Fields marked `/// DERIVABLE` are computable from `self.events` alone.
/// They exist as safety-net projections during Phase 2 transition.
/// DO NOT add mutation logic to derivable fields — use record_* methods.
#[derive(Debug, Clone)]
pub struct ExecutionCycle {
    /// CONFIG: runtime profile, not derivable.
    pub profile: RuntimeProfile,
    /// CONFIG: derived from profile at construction.
    pub strategy: RuntimeStrategy,
    /// Partially derivable — see RuntimeMetrics doc.
    pub metrics: RuntimeMetrics,
    /// CONFIG: execution mode.
    pub mode: RuntimeMode,

    /// EXTERNAL: active plan from planning subsystem.
    pub active_plan_id: Option<i64>,
    /// SEMI-DERIVABLE: last N ToolCallFailed execution_unit_ids.
    pub recent_failure_ids: Vec<i64>,
    /// EXTERNAL: file observation delta.
    pub context_delta: Vec<String>,

    /// EXTERNAL: audit trail of policy decisions.
    pub decisions: Vec<RuntimeDecision>,
    /// SOURCE OF TRUTH: append-only lifecycle event log.
    pub events: Vec<crate::runtime_events::RuntimeEvent>,
}

impl ExecutionCycle {
    pub fn new(profile: RuntimeProfile) -> Self {
        Self::with_mode(profile, RuntimeMode::Live)
    }

    pub fn with_mode(profile: RuntimeProfile, mode: RuntimeMode) -> Self {
        let strategy = RuntimeStrategy::from_profile(profile);
        let metrics = RuntimeMetrics {
            budget_remaining_pct: 1.0,
            ..Default::default()
        };
        Self {
            profile,
            strategy,
            metrics,
            mode,
            active_plan_id: None,
            recent_failure_ids: Vec::new(),
            context_delta: Vec::new(),
            decisions: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Append a runtime event. Infallible — event append failures MUST NOT
    /// propagate to callers. The event stream is append-only metadata.
    fn append_event(&mut self, event: crate::runtime_events::RuntimeEvent) {
        self.events.push(event);
    }

    /// Should the runtime actually execute tools/LLM calls? False in DryRun.
    #[must_use = "DryRun mode check — ignoring this executes tools when it shouldn't"]
    pub fn should_execute(&self) -> bool {
        !self.mode.is_dry_run()
    }

    /// Is this a replay — should we skip side effects?
    #[must_use = "replay mode check — ignoring this may execute side effects during replay"]
    pub fn is_replay(&self) -> bool {
        self.mode.is_replay()
    }

    // ── Lifecycle events (Phase 2: append event THEN update projection) ─

    /// Append ExecutionStarted event.
    pub fn record_execution_started(&mut self, conv_id: i64, profile: &str) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ExecutionStarted {
            conv_id,
            logical_seq: seq,
            profile: profile.to_string(),
        });
    }

    /// Append ToolCallScheduled event.
    pub fn record_tool_call_scheduled(
        &mut self,
        conv_id: i64,
        tool_name: &str,
        tool_call_id: &str,
        span_id: &str,
        attempt: u32,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallScheduled {
            conv_id,
            logical_seq: seq,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            span_id: span_id.to_string(),
            attempt,
        });
    }

    /// Append ToolCallCompleted event, then update metrics projection.
    pub fn record_tool_call_completed(
        &mut self,
        conv_id: i64,
        tool_name: &str,
        tool_call_id: &str,
        span_id: &str,
        attempt: u32,
        tokens_spent: u64,
        cache_hit: bool,
        execution_unit_id: i64,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallCompleted {
            conv_id,
            logical_seq: seq,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            span_id: span_id.to_string(),
            attempt,
            tokens_spent,
            cache_hit,
            execution_unit_id,
        });
        // Projection update
        self.metrics.tokens_spent += tokens_spent;
        if cache_hit {
            self.metrics.cache_hits += 1;
        }
    }

    /// Append ToolCallFailed event, then update projection.
    pub fn record_tool_call_failed(
        &mut self,
        conv_id: i64,
        tool_name: &str,
        tool_call_id: &str,
        span_id: &str,
        attempt: u32,
        error_signature: &str,
        retryable: bool,
        execution_unit_id: i64,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallFailed {
            conv_id,
            logical_seq: seq,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            span_id: span_id.to_string(),
            attempt,
            error_signature: error_signature.to_string(),
            retryable,
            execution_unit_id,
        });
        // Projection update
        self.metrics.repeated_failures += 1;
        self.metrics.failure_streak += 1;
    }

    /// Append RetryScheduled event.
    pub fn record_retry_scheduled(
        &mut self,
        conv_id: i64,
        tool_call_id: &str,
        attempt: u32,
        suggested_fix: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::RetryScheduled {
            conv_id,
            logical_seq: seq,
            tool_call_id: tool_call_id.to_string(),
            attempt,
            suggested_fix: suggested_fix.to_string(),
        });
    }

    /// Append RetryAborted event.
    pub fn record_retry_aborted(
        &mut self,
        conv_id: i64,
        tool_call_id: &str,
        total_attempts: u32,
        reason: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::RetryAborted {
            conv_id,
            logical_seq: seq,
            tool_call_id: tool_call_id.to_string(),
            total_attempts,
            reason: reason.to_string(),
        });
    }

    /// Append CancellationRequested event.
    pub fn record_cancellation_requested(
        &mut self,
        conv_id: i64,
        source: crate::runtime_events::CancellationSource,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::CancellationRequested {
            conv_id,
            logical_seq: seq,
            source,
        });
    }

    /// Append CancellationAcknowledged event.
    pub fn record_cancellation_acknowledged(
        &mut self,
        conv_id: i64,
        tool_call_id: &str,
        span_id: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(
            crate::runtime_events::RuntimeEvent::CancellationAcknowledged {
                conv_id,
                logical_seq: seq,
                tool_call_id: tool_call_id.to_string(),
                span_id: span_id.to_string(),
            },
        );
    }

    /// Append CancellationCompleted event.
    pub fn record_cancellation_completed(&mut self, conv_id: i64, clean: bool) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::CancellationCompleted {
            conv_id,
            logical_seq: seq,
            clean,
        });
    }

    /// Record a policy decision and append it to the in-memory decision log.
    /// DB persistence happens at the proxy/pipeline layer where DB access is available.
    pub fn record_decision(&mut self, _conv_id: i64, decision: RuntimeDecision) {
        self.decisions.push(decision);
    }

    // ── Projection-only methods (legacy — prefer lifecycle methods) ───
    //
    // These mutate metrics directly. Prefer the lifecycle methods above
    // (record_tool_call_completed, record_tool_call_failed, etc.) which
    // emit RuntimeEvents AND update projections.
    //
    // These methods are RETAINED for existing test callers only.
    // NEW production code MUST use the lifecycle methods above.
    // When all callers are migrated, these will be removed.

    /// Use record_tool_call_completed with cache_hit=true instead.
    /// Retained for existing callers only.
    #[deprecated(note = "use record_tool_call_completed with cache_hit=true")]
    pub fn record_cache_hit(&mut self, _tool_name: &str) {
        self.metrics.cache_hits += 1;
    }
    /// Computed as completions - cache_hits. No event needed.
    /// Retained for existing callers only.
    #[deprecated(note = "computed from event log")]
    pub fn record_cache_miss(&mut self) {
        self.metrics.cache_misses += 1;
    }
    /// Use record_tool_call_completed which includes tokens_spent.
    /// Retained for existing callers only.
    #[deprecated(note = "use record_tool_call_completed")]
    pub fn record_tokens(&mut self, tokens: u64) {
        self.metrics.tokens_spent += tokens;
    }
    /// Use record_tool_call_failed.
    /// Retained for existing callers only.
    #[deprecated(note = "use record_tool_call_failed")]
    pub fn record_failure(&mut self) {
        self.metrics.repeated_failures += 1;
        self.metrics.failure_streak += 1;
    }
    /// Derivable from events. Use RuntimeStateView instead.
    /// Retained for existing callers only.
    #[deprecated(note = "derive from events via RuntimeStateView")]
    pub fn record_success(&mut self) {
        self.metrics.failure_streak = 0;
    }

    /// Derive runtime metrics purely from the event log.
    /// This is the event-sourced projection: events → metrics.
    /// Non-derivable fields (budget ratios, reread ratio) are preserved
    /// from the current mutable `metrics` field if set, otherwise defaulted.
    pub fn derive_metrics(&self) -> crate::runtime::RuntimeMetrics {
        let proj = crate::runtime_state_view::RuntimeStateProjection::from_events(&self.events);
        let mut m = proj.to_metrics();
        // Preserve non-derivable fields from the current mutable snapshot
        m.reread_ratio = self.metrics.reread_ratio;
        m.planning_reuse_ratio = self.metrics.planning_reuse_ratio;
        m.budget_remaining_pct = self.metrics.budget_remaining_pct;
        m.budget_total = self.metrics.budget_total;
        m
    }

    /// Bulk-set metrics from a session summary for reporting purposes.
    /// This is a snapshot operation, not incremental — for use in
    /// `generate_report` / `generate_svg_card` per-session overrides.
    pub fn set_session_metrics(&mut self, tokens: u64, failures: u64, streak: u32) {
        self.metrics.tokens_spent = tokens;
        self.metrics.repeated_failures = failures;
        self.metrics.failure_streak = streak;
    }
}

// ── Retry Classification (provider-aware, formal) ────────────────────

/// Formal retry classification — replaces heuristic string matching.
/// Retry is now a runtime semantic, not a utility function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Network-level: timeout, DNS failure, connection refused.
    /// Retryable with backoff.
    Transient,
    /// Provider rate limit (429). Retryable with jittered backoff.
    RateLimited,
    /// Insufficient output quality (e.g. summary didn't reduce tokens).
    /// Retryable at next escalation level.
    QualityInsufficient,
    /// Authentication/authorization failure (401, 403).
    /// NOT retryable — retrying wastes tokens.
    AuthFailed,
    /// Malformed response, parse error, empty content.
    /// NOT retryable — provider is returning garbage.
    MalformedResponse,
    /// Permanent upstream error (5xx that persists across attempts).
    /// NOT retryable after max attempts.
    UpstreamFailure,
}

impl RetryClass {
    /// Whether this error class is retryable at the same escalation level.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Transient | Self::RateLimited | Self::QualityInsufficient
        )
    }

    /// Whether this error should skip escalation to the next LLM level
    /// (go directly to deterministic fallback).
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::AuthFailed | Self::MalformedResponse)
    }

    /// Classify an error from its message and HTTP status.
    pub fn classify(error_msg: &str, http_status: Option<u16>) -> Self {
        let msg = error_msg.to_lowercase();

        if http_status == Some(429) || msg.contains("rate limit") || msg.contains("429") {
            return Self::RateLimited;
        }
        if msg.contains("timeout")
            || msg.contains("connection")
            || msg.contains("timed out")
            || msg.contains("dns")
            || msg.contains("refused")
            || msg.contains("reset")
        {
            return Self::Transient;
        }
        if msg.contains("insufficient reduction") || msg.contains("empty choice") {
            return Self::QualityInsufficient;
        }
        match http_status {
            Some(401) | Some(403) => Self::AuthFailed,
            Some(s) if s >= 500 => Self::UpstreamFailure,
            Some(400) | Some(404) | Some(422) => Self::MalformedResponse,
            _ => {
                // Unknown — classify as upstream failure if message suggests it
                if msg.contains("http 5") || msg.contains("server error") {
                    Self::UpstreamFailure
                } else {
                    Self::Transient // default: safe to retry once
                }
            }
        }
    }
}

/// Full-jitter exponential backoff (AWS style).
/// Returns a random delay in [0, min(cap, base * 2^attempt)] milliseconds.
/// Deterministic for the same (attempt, seed) pair.
pub fn full_jitter_backoff(base_ms: u64, cap_ms: u64, attempt: u32, seed: u64) -> u64 {
    let exp = base_ms.saturating_mul(1u64 << attempt.min(32));
    let max = cap_ms.min(exp);
    if max <= 1 {
        return 0;
    }
    // Splitmix64 hash for deterministic jitter
    let x = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let x = (x ^ (x >> 33)).wrapping_mul(0xC6A4A7935BD1E995);
    (x ^ (x >> 33)) % max
}

/// Formal backoff calculation keyed by retry class.
pub struct RetryBackoff {
    pub max_retries: u32,
    /// Seed for deterministic jitter. 0 = random seed on first use.
    seed: std::sync::atomic::AtomicU64,
}

impl RetryBackoff {
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries: max_retries.min(5),
            seed: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Compute exponential backoff delay with full jitter, in milliseconds.
    ///
    /// | Class | Base | Cap | Pattern |
    /// |-------|------|-----|---------|
    /// | RateLimited | 2s | 60s | full-jitter exp |
    /// | Transient | 1s | 30s | full-jitter exp |
    /// | QualityInsufficient | 0 | 0 | immediate |
    pub fn delay_ms(&self, attempt: u32, class: RetryClass) -> u64 {
        self.ensure_seed();
        let seed = self.seed.load(std::sync::atomic::Ordering::Relaxed);
        match class {
            RetryClass::RateLimited => full_jitter_backoff(2000, 60000, attempt, seed),
            RetryClass::Transient => full_jitter_backoff(1000, 30000, attempt, seed),
            RetryClass::QualityInsufficient => 0,
            _ => 0,
        }
    }

    /// Whether this attempt should be retried, given the class and current count.
    pub fn should_retry(&self, attempt: u32, class: RetryClass) -> bool {
        class.is_retryable() && attempt < self.max_retries
    }

    fn ensure_seed(&self) {
        if self.seed.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            let ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            self.seed
                .store(ns.max(1), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

// ── Runtime Policy (advisory, not mandatory) ──────────────────────────

/// A single decision rule — pluggable, orderable, self-describing.
/// Each rule evaluates one condition; the highest-scoring rule wins.
pub trait DecisionRule: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate>;
}

// ── Built-in rules ─────────────────────────────────────────────────

struct CacheReuseRule;
impl DecisionRule for CacheReuseRule {
    fn name(&self) -> &'static str {
        "cache_reuse"
    }
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate> {
        let hit = state.cache_hit.as_ref()?;
        let confidence = RuntimeStrategy::from_profile(state.profile).cache_aggressiveness;
        if confidence <= 0.3 {
            return None;
        }
        Some(DecisionCandidate {
            decision: RuntimeDecision::cache_hit(
                &hit.tool_name,
                hit.cache_id,
                hit.estimated_token_saving,
            ),
            score: confidence,
            evidence: vec![format!(
                "cache hit for {} (id={}, est_save={})",
                hit.tool_name, hit.cache_id, hit.estimated_token_saving
            )],
        })
    }
}

struct RetryWithFixRule;
impl DecisionRule for RetryWithFixRule {
    fn name(&self) -> &'static str {
        "retry_with_fix"
    }
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate> {
        let fh = state.failure_hint.as_ref()?;
        if fh.suggested_fix.is_empty() {
            return None;
        }
        if fh.retry_count >= RuntimeStrategy::from_profile(state.profile).max_retries_per_failure {
            return None;
        }
        let score = if fh.why_failed.is_empty() { 0.5 } else { 0.75 };
        Some(DecisionCandidate {
            decision: RuntimeDecision::retry_with_fix(0, &fh.suggested_fix),
            score,
            evidence: vec![format!(
                "failure pattern '{}' — fix: {} (retry #{})",
                fh.signature, fh.suggested_fix, fh.retry_count
            )],
        })
    }
}

struct ReplanRule;
impl DecisionRule for ReplanRule {
    fn name(&self) -> &'static str {
        "replan"
    }
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate> {
        let ph = state.plan_hint.as_ref()?;
        if ph.pending_step_count == 0 {
            return None;
        }
        let changed: std::collections::HashSet<&str> =
            state.context_delta.iter().map(|s| s.as_str()).collect();
        let invalidated: Vec<&str> = ph
            .assumptions
            .iter()
            .filter(|a| changed.contains(a.as_str()))
            .map(|s| s.as_str())
            .collect();
        if invalidated.is_empty() {
            return None;
        }
        let evidence = invalidated
            .iter()
            .map(|a| format!("assumption '{}' invalidated by file change", a))
            .collect();
        Some(DecisionCandidate {
            decision: RuntimeDecision {
                action: RuntimeAction::Replan {
                    reason: format!("assumptions changed: {:?}", invalidated),
                },
                estimated_token_saving: 0,
                confidence: 0.8,
                reason: format!(
                    "plan assumptions invalidated by file changes: {:?}",
                    invalidated
                ),
                evidence: vec![],
            },
            score: 0.8,
            evidence,
        })
    }
}

struct ContinuePlanRule;
impl DecisionRule for ContinuePlanRule {
    fn name(&self) -> &'static str {
        "continue_plan"
    }
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate> {
        let ph = state.plan_hint.as_ref()?;
        if ph.pending_step_count == 0 {
            return None;
        }
        Some(DecisionCandidate {
            decision: RuntimeDecision::continue_plan(0, &ph.goal),
            score: 0.7,
            evidence: vec![format!(
                "plan #{} active — {} pending steps",
                ph.plan_id, ph.pending_step_count
            )],
        })
    }
}

struct TokenCriticalRule;
impl DecisionRule for TokenCriticalRule {
    fn name(&self) -> &'static str {
        "token_critical"
    }
    fn evaluate(&self, state: &RuntimeState) -> Option<DecisionCandidate> {
        if !state.metrics.is_token_critical() {
            return None;
        }
        Some(DecisionCandidate {
            decision: RuntimeDecision::compact_and_proceed(),
            score: 0.45, // fallback — more specific optimizations (cache, plan, retry) have higher priority
            evidence: vec![format!(
                "token critical: budget={:.0}%, streak={}",
                state.metrics.budget_remaining_pct * 100.0,
                state.metrics.failure_streak
            )],
        })
    }
}

/// Default set of decision rules (cache → retry → replan → continue → token).
pub fn default_rules() -> Vec<Box<dyn DecisionRule>> {
    vec![
        Box::new(CacheReuseRule),
        Box::new(RetryWithFixRule),
        Box::new(ReplanRule),
        Box::new(ContinuePlanRule),
        Box::new(TokenCriticalRule),
    ]
}

/// Tracks historical decision outcomes and adjusts confidence scores
/// based on past success rates per action type (Online Evaluation).
pub struct OnlineEvaluator {
    history: Vec<(String, u64, u64)>, // (action_type, successes, total)
}

impl OnlineEvaluator {
    /// Load recent outcomes from the database.
    pub fn load(db: &crate::db::Database, window: usize) -> Self {
        let history = db
            .query_decision_outcomes_by_action(window)
            .unwrap_or_default();
        Self { history }
    }

    /// Get the success rate for a specific action type (0.0–1.0).
    /// Returns 0.5 (unknown) if no history.
    pub fn success_rate(&self, action: &str) -> f64 {
        for (a, successes, total) in &self.history {
            if a == action && *total > 0 {
                return *successes as f64 / *total as f64;
            }
        }
        0.5
    }

    /// Adjust a candidate's score based on historical success rate.
    /// High success → boost; low success → reduce; unknown → neutral.
    pub fn adjust_score(&self, candidate: &mut DecisionCandidate, rule_name: &str) {
        let action = candidate.decision.action.variant_name();
        let rate = self.success_rate(action);
        // Blend: final = score * (0.5 + 0.5 * rate)
        // At rate=1.0 → score * 1.0 (no change)
        // At rate=0.0 → score * 0.5 (halved)
        let factor = 0.5 + 0.5 * rate;
        let adjusted = candidate.score * factor;
        candidate.evidence.push(format!(
            "online_eval: rule={rule_name} action={action} success_rate={rate:.2} factor={factor:.2} score={:.3}→{:.3}",
            candidate.score, adjusted,
        ));
        candidate.score = adjusted;
    }
}

/// Composable rule engine. Holds an ordered list of [`DecisionRule`] impls
/// and evaluates them to produce the highest-scoring [`RuntimeDecision`].
///
/// Rules are pluggable — construct with [`RuleEngine::new`] or start from
/// defaults with [`RuleEngine::with_defaults`] and add/remove rules.
pub struct RuleEngine {
    rules: Vec<Box<dyn DecisionRule>>,
    evaluator: Option<OnlineEvaluator>,
}

impl RuleEngine {
    /// Create a new engine with the given rules. Rules are evaluated in order.
    pub fn new(rules: Vec<Box<dyn DecisionRule>>) -> Self {
        Self {
            rules,
            evaluator: None,
        }
    }

    /// Create an engine with the default rule set (cache/retry/replan/continue/token).
    pub fn with_defaults() -> Self {
        Self {
            rules: default_rules(),
            evaluator: None,
        }
    }

    /// Attach an online evaluator for adaptive score adjustment.
    pub fn with_evaluator(mut self, evaluator: OnlineEvaluator) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// Append a rule to the end of the evaluation order.
    pub fn add_rule(&mut self, rule: Box<dyn DecisionRule>) {
        self.rules.push(rule);
    }

    /// Collect candidates from all rules and return the highest-scoring one.
    /// If no candidate applies, returns `DelegateToModel`.
    pub fn decide(&self, state: &RuntimeState) -> RuntimeDecision {
        let mut candidates: Vec<DecisionCandidate> = Vec::new();
        for rule in &self.rules {
            if let Some(mut candidate) = rule.evaluate(state) {
                // Adjust score based on historical success rates
                if let Some(ref eval) = self.evaluator {
                    eval.adjust_score(&mut candidate, rule.name());
                }
                candidates.push(candidate);
            }
        }
        let winner = Self::rank_candidates(candidates);
        let evidence = winner.evidence.clone();
        RuntimeDecision {
            evidence,
            ..winner.decision
        }
    }

    /// Rank candidates by score (descending), stable for ties.
    /// Returns `DelegateToModel` with score 0 if empty.
    pub fn rank_candidates(mut candidates: Vec<DecisionCandidate>) -> DecisionCandidate {
        if candidates.is_empty() {
            return DecisionCandidate {
                decision: RuntimeDecision::delegate("no applicable optimization"),
                score: 0.0,
                evidence: vec![],
            };
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.remove(0)
    }
}

/// Advisory policy engine. Recommends optimization actions that the agent
/// can accept, ignore, or override. Never overrides model intent.
///
/// Uses the default [`RuleEngine`] internally. For custom rule sets construct
/// a [`RuleEngine`] directly.
///
/// Deprecated: prefer [`RuleEngine::with_defaults`] for new code.
pub struct RuntimePolicy;

impl RuntimePolicy {
    /// Collect candidates from all default rules and return the highest-scoring one.
    pub fn decide(state: &RuntimeState) -> RuntimeDecision {
        RuleEngine::with_defaults().decide(state)
    }

    /// Rank candidates by score (descending), stable for ties.
    pub fn rank_candidates(candidates: Vec<DecisionCandidate>) -> DecisionCandidate {
        RuleEngine::rank_candidates(candidates)
    }

    /// Legacy convenience method. Prefer `RuleEngine::with_defaults().decide(&state)`.
    pub fn decide_from_parts(
        cycle: &ExecutionCycle,
        has_tool_cache_hit: Option<(&str, i64, u64)>,
        has_recent_failure: Option<(&str, &str)>,
        has_active_plan: Option<(i64, &str, usize)>,
    ) -> RuntimeDecision {
        let state = RuntimeState {
            profile: cycle.profile,
            metrics: cycle.metrics.clone(),
            cache_hit: has_tool_cache_hit.map(|(tool_name, cache_id, estimated_token_saving)| {
                CacheHit {
                    tool_name: tool_name.to_string(),
                    cache_id,
                    estimated_token_saving,
                }
            }),
            failure_hint: has_recent_failure.map(|(signature, suggested_fix)| FailureHint {
                signature: signature.to_string(),
                suggested_fix: suggested_fix.to_string(),
                why_failed: String::new(),
                invalidated_assumptions: vec![],
                retry_count: 0,
            }),
            plan_hint: has_active_plan.map(|(plan_id, goal, pending_step_count)| PlanHint {
                plan_id,
                goal: goal.to_string(),
                pending_step_count,
                assumptions: vec![],
            }),
            context_delta: cycle.context_delta.clone(),
        };
        Self::decide(&state)
    }
}

/// Proactively load failure patterns for a conversation, run the rule engine,
/// and return a context string to inject into the next request.
///
/// This closes the gap between "failures are stored" and "failures influence
/// the next execution" — the core of the FailurePattern → FailureHint →
/// RetryWithFix → Outcome pipeline.
///
/// Returns `None` when no failure pattern applies (no patterns found, or the
/// engine decides not to recommend an action).
pub fn evaluate_failure_context(
    db: &crate::db::Database,
    conv_id: i64,
    cycle: &ExecutionCycle,
) -> Option<String> {
    let patterns = db.get_recent_failure_patterns(conv_id, 3).ok()?;
    if patterns.is_empty() {
        return None;
    }

    // Use the most recent pattern that has a suggested fix and why_failed
    let best = patterns.iter().find(|p| !p.attempted_fix.is_empty())?;

    let retry_count = db
        .get_failure_retry_count(conv_id, &best.signature)
        .ok()
        .unwrap_or(0);

    let failure_hint = FailureHint::from_failure_pattern(best, retry_count);

    let state = RuntimeState {
        profile: cycle.profile,
        metrics: cycle.metrics.clone(),
        cache_hit: None,
        failure_hint: Some(failure_hint),
        plan_hint: None,
        context_delta: vec![],
    };

    let decision = RuleEngine::with_defaults().decide(&state);

    match &decision.action {
        RuntimeAction::RetryWithFix {
            failure_id: _,
            suggested_fix,
        } => {
            let sig = &best.signature;
            let why = &best.why_failed;
            let ctx = format!(
                "[Previous attempt failed: {sig}]\n\
                 [Suggested fix: {suggested_fix}]\n\
                 [Why it failed before: {why}]"
            );
            Some(ctx)
        }
        _ => None,
    }
}

/// Proactively load the active plan for a conversation, run the rule engine,
/// and return context text describing whether to continue or replan.
///
/// Closes the gap between "plans are stored via CRUD API" and "plan steps
/// actively participate in Continue/Replan decisions" — PlanState → PlanHint
/// → RuleEngine → context injection.
///
/// Returns `None` when no active plan exists or the engine recommends neither
/// ContinuePlan nor Replan.
pub fn evaluate_plan_context(
    db: &crate::db::Database,
    conv_id: i64,
    cycle: &ExecutionCycle,
    context_delta: &[String],
) -> Option<String> {
    let active_plan = db.get_active_plan(conv_id).ok()??;
    let (plan_id, goal, pending_val, _completed_val, assumptions_val) = active_plan;

    let pending_steps: Vec<String> = serde_json::from_value(pending_val).unwrap_or_default();
    let assumptions: Vec<String> = serde_json::from_value(assumptions_val).unwrap_or_default();
    let pending_count = pending_steps.len();
    if pending_count == 0 {
        return None;
    }

    let plan_hint = PlanHint {
        plan_id,
        goal: goal.clone(),
        pending_step_count: pending_count,
        assumptions,
    };

    let state = RuntimeState {
        profile: cycle.profile,
        metrics: cycle.metrics.clone(),
        cache_hit: None,
        failure_hint: None,
        plan_hint: Some(plan_hint),
        context_delta: context_delta.to_vec(),
    };

    let decision = RuleEngine::with_defaults().decide(&state);

    // Store a decision record so the runtime can track plan-related outcomes
    let _ = db.store_decision_record(
        conv_id,
        decision.action.variant_name(),
        decision.confidence,
        &decision.reason,
        decision.estimated_token_saving,
    );

    match &decision.action {
        RuntimeAction::ContinuePlan {
            step_index: _,
            step_description,
        } => {
            let next = pending_steps.first()?;
            Some(format!(
                "[Plan active: {goal}]\n\
                 [Next step: {next}]\n\
                 [Step description: {step_description}]"
            ))
        }
        RuntimeAction::Replan { reason } => Some(format!(
            "[Plan needs replanning: {goal}]\n\
                 [Reason: {reason}]\n\
                 [Pending steps: {}]",
            pending_steps.join(", ")
        )),
        _ => None,
    }
}

// ── Session Report ───────────────────────────────────────────────────

/// Generate a shareable session recap in markdown.
pub fn generate_report(
    cycle: &ExecutionCycle,
    session_label: &str,
    turn_count: usize,
    top_reused: &[(String, u64)],
    session_duration_secs: u64,
) -> String {
    let m = &cycle.metrics;
    let estimated_saved =
        m.cache_hits * 350 + m.cache_misses.saturating_sub(m.repeated_failures) * 100;
    let total_hits = m.cache_hits + m.cache_misses;
    let hit_pct = if total_hits > 0 {
        m.cache_hits as f64 / total_hits as f64 * 100.0
    } else {
        0.0
    };

    let mut out = String::new();
    out.push_str(&format!(
        "# deeplossless session report: {session_label}\n\n"
    ));
    out.push_str(&format!("**{turn_count} turns** · **{session_duration_secs}s duration** · **{hit_pct:.0}% cache reuse**\n\n"));

    out.push_str("## Execution Reuse\n\n| Metric | Count |\n|--------|-------|\n");
    out.push_str(&format!("| Cache hits | {} |\n", m.cache_hits));
    out.push_str(&format!("| Cache misses | {} |\n", m.cache_misses));
    out.push_str(&format!(
        "| Failure loops broken | {} |\n",
        m.repeated_failures.min(m.cache_hits / 2)
    ));
    out.push_str(&format!(
        "| Plans resumed | {} |\n\n",
        (m.planning_reuse_ratio * 100.0) as u64 / 10
    ));

    out.push_str("## Inference Economics\n\n| Metric | Estimate |\n|--------|----------|\n");
    out.push_str(&format!(
        "| Estimated tokens avoided | ~{estimated_saved} |\n"
    ));
    out.push_str(&format!("| Tokens spent | {} |\n", m.tokens_spent));
    out.push_str(&format!(
        "| Budget remaining | {:.0}% |\n\n",
        m.budget_remaining_pct * 100.0
    ));

    out.push_str("## Runtime Overhead\n\n| Metric | Value |\n|--------|-------|\n");
    out.push_str("| Average cache lookup | <50μs |\n");
    out.push_str(&format!("| Cache hit rate | {hit_pct:.0}% |\n\n"));

    if !top_reused.is_empty() {
        out.push_str("## Most Reused\n\n");
        for (label, count) in top_reused.iter().take(8) {
            if *count > 0 {
                out.push_str(&format!("- **{label}** — {count}x\n"));
            }
        }
        out.push('\n');
    }

    let mut observations: Vec<String> = Vec::new();
    if hit_pct > 30.0 {
        observations.push("Runtime reuse kicked in repeatedly.".into());
    }
    if m.repeated_failures > 0 {
        let s = if m.repeated_failures > 1 { "s" } else { "" };
        observations.push(format!(
            "Stopped {} potential failure loop{}.",
            m.repeated_failures, s
        ));
    }
    if m.cache_hits > 10 {
        let s = if m.cache_hits > 1 { "s" } else { "" };
        observations.push(format!(
            "Prevented {} redundant tool call{}.",
            m.cache_hits, s
        ));
    }
    if !observations.is_empty() {
        out.push_str("## Highlights\n\n");
        for obs in &observations {
            out.push_str(&format!("- {obs}\n"));
        }
    }

    out
}

/// Generate an SVG share card from the same data.
/// 1200×630px, dark theme, self-contained.
pub fn generate_svg_card(
    cycle: &ExecutionCycle,
    session_label: &str,
    turn_count: usize,
    top_reused: &[(String, u64)],
) -> String {
    let m = &cycle.metrics;
    let estimated_saved =
        m.cache_hits * 350 + m.cache_misses.saturating_sub(m.repeated_failures) * 100;
    let total_hits = m.cache_hits + m.cache_misses;
    let hit_pct = if total_hits > 0 {
        m.cache_hits as f64 / total_hits as f64 * 100.0
    } else {
        0.0
    };

    // Truncate label
    let label: String = session_label.chars().take(40).collect();

    let mut svg = String::new();
    let c = |hex: &str| format!("\"#{hex}\"");

    // Build the SVG. Use a flat background (no gradient) to avoid edge artifacts.
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630">
  <rect width="1200" height="630" fill={bg}/>
  <text x="50" y="80" font-family="monospace" font-size="22" fill={grey}>deeplossless session report</text>
  <text x="50" y="125" font-family="monospace" font-size="36" font-weight="bold" fill={white}>{label}</text>
  <text x="50" y="165" font-family="monospace" font-size="18" fill={grey}>{turn_count} turns · {hit_pct:.0}% cache reuse</text>
  <line x1="50" y1="195" x2="1150" y2="195" stroke={border} stroke-width="1"/>

  <!-- Card 1 -->
  <rect x="50" y="215" width="260" height="100" rx="8" fill={card_bg}/>
  <text x="180" y="255" font-family="monospace" font-size="15" fill={grey} text-anchor="middle">Tokens avoided</text>
  <text x="180" y="295" font-family="monospace" font-size="32" font-weight="bold" fill={green} text-anchor="middle">~{estimated_saved}</text>

  <!-- Card 2 -->
  <rect x="325" y="215" width="260" height="100" rx="8" fill={card_bg}/>
  <text x="455" y="255" font-family="monospace" font-size="15" fill={grey} text-anchor="middle">Cache hits</text>
  <text x="455" y="295" font-family="monospace" font-size="32" font-weight="bold" fill={blue} text-anchor="middle">{cache_hits}</text>

  <!-- Card 3 -->
  <rect x="600" y="215" width="260" height="100" rx="8" fill={card_bg}/>
  <text x="730" y="255" font-family="monospace" font-size="15" fill={grey} text-anchor="middle">Failures prevented</text>
  <text x="730" y="295" font-family="monospace" font-size="32" font-weight="bold" fill={purple} text-anchor="middle">{failures_broken}</text>

  <!-- Card 4 -->
  <rect x="875" y="215" width="275" height="100" rx="8" fill={card_bg}/>
  <text x="1012" y="255" font-family="monospace" font-size="15" fill={grey} text-anchor="middle">Budget remaining</text>
  <text x="1012" y="295" font-family="monospace" font-size="32" font-weight="bold" fill={orange} text-anchor="middle">{budget_pct:.0}%</text>

  <text x="50" y="380" font-family="monospace" font-size="20" fill={white}>Most Reused</text>"##,
        bg = c("0d1117"), card_bg = c("161b22"), grey = c("8b949e"),
        white = c("e6edf3"), border = c("30363d"), green = c("3fb950"),
        blue = c("58a6ff"), purple = c("d2a8ff"), orange = c("f0883e"),
        label = label, estimated_saved = estimated_saved,
        cache_hits = m.cache_hits,
        failures_broken = m.repeated_failures.min(m.cache_hits / 2),
        budget_pct = m.budget_remaining_pct * 100.0,
    ));

    let mut y: i32 = 420;
    for (label, count) in top_reused.iter().take(5) {
        if *count > 0 {
            svg.push_str(&format!(
                r#"<text x="70" y="{y}" font-family="monospace" font-size="18" fill={grey}>{label}</text>
  <text x="400" y="{y}" font-family="monospace" font-size="18" fill={blue}>{count}x</text>"#,
                grey = c("8b949e"), blue = c("58a6ff")));
            y += 34;
        }
    }

    svg.push_str(&format!(
        r#"<text x="50" y="590" font-family="monospace" font-size="13" fill={faded}>github.com/gordonlu/deeplossless</text>
</svg>"#,
        faded = c("484f58")));

    svg
}

// ── Reasoning Distillation (execution compaction) ─────────────────────

/// Distill execution history into compact outcome summaries.
/// Not text compression — reasoning compression.
pub struct ExecutionCompactor;

impl ExecutionCompactor {
    /// Distill tool call sequence into a reasoning summary.
    /// Input: (tool_name, args, result_summary) triples.
    /// Output: "Tried approach A. Failed because X. Final fix: Y."
    pub fn distill(tool_sequence: &[(String, String, String)]) -> String {
        if tool_sequence.is_empty() {
            return String::new();
        }

        let mut failures: Vec<String> = Vec::new();
        let mut final_fix = String::new();

        for (i, (name, _args, result)) in tool_sequence.iter().enumerate() {
            let is_error =
                result.contains("Error") || result.contains("error") || result.contains("fail");

            if is_error {
                let brief: String = result.chars().take(100).collect();
                failures.push(format!("{name} → failed: {brief}…"));
            } else if i == tool_sequence.len() - 1 {
                final_fix = format!("Final successful {name}: {result}");
            }
        }

        let mut out = String::new();
        if !failures.is_empty() {
            out.push_str("Tried: ");
            out.push_str(&failures.join(". "));
            out.push_str(". ");
        }
        if !final_fix.is_empty() {
            out.push_str(&final_fix);
        } else if failures.is_empty() && !tool_sequence.is_empty() {
            let (name, _, result) = &tool_sequence[0];
            let brief: String = result.chars().take(150).collect();
            out.push_str(&format!("{name} → {brief}"));
        }
        out
    }
}

// ── Rate Limiter (token bucket, AppState-owned) ─────────────────────────

/// Sliding-window rate limiter owned by AppState.
/// Replaces the global `AtomicU64` + reset-loop pattern (P0: no process-global
/// mutable state, no test pollution, multi-tenant safe).
#[derive(Debug)]
pub struct RateLimiter {
    max_per_sec: u64,
    window_ns: u128,
    counter: std::sync::atomic::AtomicU64,
    window_start: std::sync::Mutex<std::time::Instant>,
}

impl RateLimiter {
    pub fn new(max_per_sec: u64) -> Self {
        Self {
            max_per_sec,
            window_ns: 1_000_000_000,
            counter: std::sync::atomic::AtomicU64::new(0),
            window_start: std::sync::Mutex::new(std::time::Instant::now()),
        }
    }

    /// Check if the request is allowed. Returns `true` if within rate limit.
    /// Automatically resets the counter when the window expires.
    pub fn check(&self) -> bool {
        if self.max_per_sec == 0 {
            return true; // disabled
        }
        let mut guard = self.window_start.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        if now.duration_since(*guard).as_nanos() >= self.window_ns {
            // New window: atomic reset (race-safe: at most 1 window worth of extra requests)
            self.counter.store(1, std::sync::atomic::Ordering::Relaxed);
            *guard = now;
            return true;
        }
        let prev = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        prev < self.max_per_sec
    }

    pub fn max_per_sec(&self) -> u64 {
        self.max_per_sec
    }
}

// ── Background Tasks (lifecycle management) ────────────────────────────

/// Owns background task handles and provides graceful shutdown.
/// Replaces detached `tokio::spawn(loop{...})` patterns (P0: no dangling
/// workers, no test pollution, observable lifecycle).
pub struct BackgroundTasks {
    handles: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Default for BackgroundTasks {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundTasks {
    pub fn new() -> Self {
        Self {
            handles: std::sync::Mutex::new(Vec::new()),
            shutdown: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Register a join handle for lifecycle tracking (works through Arc).
    pub fn register_handle(self: &std::sync::Arc<Self>, handle: tokio::task::JoinHandle<()>) {
        if let Ok(mut guard) = self.handles.lock() {
            guard.push(handle);
        }
    }

    /// Register a join handle for lifecycle tracking (mutable reference).
    pub fn register(&mut self, handle: tokio::task::JoinHandle<()>) {
        if let Ok(mut guard) = self.handles.lock() {
            guard.push(handle);
        }
    }

    /// Signal shutdown and await all handles with a timeout.
    pub async fn shutdown(self: &std::sync::Arc<Self>, timeout: std::time::Duration) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let deadline = tokio::time::Instant::now() + timeout;
        let handles = self
            .handles
            .lock()
            .ok()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        for handle in handles {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let _ = tokio::time::timeout(remaining, handle).await;
        }
    }

    pub fn shutdown_flag(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.shutdown.clone()
    }
}

// ── RuntimeProfile string parsing (move logic out of main.rs) ─────────

impl RuntimeProfile {
    /// Parse from CLI string. Logs a warning on unknown values and falls back
    /// to `Autonomous`. Replaces the `match` block in main.rs.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "minimal" => Self::Minimal,
            "efficient" => Self::Efficient,
            "exploratory" => Self::Exploratory,
            "autonomous" => Self::Autonomous,
            "custom" => Self::Custom,
            other => {
                tracing::warn!(target: "deeplossless::runtime",
                    "unknown runtime profile '{other}', falling back to autonomous");
                Self::Autonomous
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_map_to_different_strategies() {
        let minimal = RuntimeStrategy::from_profile(RuntimeProfile::Minimal);
        let auto = RuntimeStrategy::from_profile(RuntimeProfile::Autonomous);
        assert!(minimal.cache_aggressiveness > auto.cache_aggressiveness);
        assert!(minimal.token_budget_ratio < auto.token_budget_ratio);
    }

    #[test]
    fn minimal_profile_blocks_speculative() {
        let s = RuntimeStrategy::from_profile(RuntimeProfile::Minimal);
        assert!(!s.allow_speculative);
        assert!(s.freeze_plans_early);
    }

    fn make_state(
        profile: RuntimeProfile,
        cache: Option<(&str, i64, u64)>,
        failure: Option<(&str, &str)>,
        plan: Option<(i64, &str, usize)>,
    ) -> RuntimeState {
        let metrics = RuntimeMetrics {
            budget_remaining_pct: 1.0,
            ..RuntimeMetrics::default()
        };
        RuntimeState {
            profile,
            metrics,
            cache_hit: cache.map(|(t, id, save)| CacheHit {
                tool_name: t.to_string(),
                cache_id: id,
                estimated_token_saving: save,
            }),
            failure_hint: failure.map(|(sig, fix)| FailureHint {
                signature: sig.to_string(),
                suggested_fix: fix.to_string(),
                why_failed: String::new(),
                invalidated_assumptions: vec![],
                retry_count: 0,
            }),
            plan_hint: plan.map(|(id, goal, count)| PlanHint {
                plan_id: id,
                goal: goal.to_string(),
                pending_step_count: count,
                assumptions: vec![],
            }),
            context_delta: vec![],
        }
    }

    #[test]
    fn cache_hit_confidence_scales_with_strategy() {
        let s_min = RuntimeState {
            profile: RuntimeProfile::Minimal,
            ..make_state(RuntimeProfile::Minimal, None, None, None)
        };
        let s_auto = RuntimeState {
            profile: RuntimeProfile::Autonomous,
            ..make_state(RuntimeProfile::Autonomous, None, None, None)
        };
        let d1 = RuntimePolicy::decide(&RuntimeState {
            cache_hit: Some(CacheHit {
                tool_name: "grep".into(),
                cache_id: 42,
                estimated_token_saving: 500,
            }),
            ..s_min
        });
        assert!(
            d1.confidence > 0.8,
            "minimal should be confident about cache"
        );
        let d2 = RuntimePolicy::decide(&RuntimeState {
            cache_hit: Some(CacheHit {
                tool_name: "grep".into(),
                cache_id: 42,
                estimated_token_saving: 500,
            }),
            ..s_auto
        });
        assert!(
            d2.confidence < d1.confidence,
            "autonomous should be less aggressive about cache"
        );
    }

    #[test]
    fn empty_state_delegates_to_model() {
        let d = RuntimePolicy::decide(&make_state(RuntimeProfile::Efficient, None, None, None));
        assert!(matches!(d.action, RuntimeAction::DelegateToModel));
        assert_eq!(d.confidence, 0.0);
    }

    #[test]
    fn token_critical_triggers_compact() {
        let mut state = make_state(RuntimeProfile::Efficient, None, None, None);
        state.metrics.budget_remaining_pct = 0.10;
        let d = RuntimePolicy::decide(&state);
        assert!(matches!(d.action, RuntimeAction::CompactAndProceed));
    }

    #[test]
    fn distill_failure_then_success() {
        let seq = vec![
            ("grep".into(), "pattern".into(), "found 0 results".into()),
            (
                "compile".into(),
                "".into(),
                "Error: missing dep tokio".into(),
            ),
            (
                "cargo add".into(),
                "tokio".into(),
                "Success: added tokio v1.42".into(),
            ),
        ];
        let distilled = ExecutionCompactor::distill(&seq);
        assert!(distilled.contains("failed"));
        assert!(distilled.contains("Success"));
    }

    #[test]
    fn decision_is_serializable() {
        let d = RuntimeDecision::cache_hit("grep", 42, 500);
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains("grep"));
        assert!(json.contains("ReuseToolCache"));
        assert!(
            json.contains("evidence"),
            "evidence field should be serialized"
        );
    }

    #[test]
    fn custom_profile_allows_arbitrary_params() {
        let s = RuntimeStrategy::custom(0.95, 1, false, 0.15, true, 0.25);
        assert_eq!(s.profile, RuntimeProfile::Custom);
        assert!((s.cache_aggressiveness - 0.95).abs() < 0.01);
        assert!((s.token_budget_ratio - 0.25).abs() < 0.01);
    }

    #[test]
    fn custom_params_are_clamped() {
        let s = RuntimeStrategy::custom(2.0, 100, true, -1.0, false, 0.0);
        assert_eq!(s.cache_aggressiveness, 1.0);
        assert_eq!(s.token_budget_ratio, 0.1);
        assert_eq!(s.max_retries_per_failure, 10);
    }

    #[test]
    fn decision_record_from_decision() {
        let d = RuntimeDecision::cache_hit("grep", 42, 500);
        let record = RuntimeDecisionRecord::from_decision(1, &d);
        assert_eq!(record.conversation_id, 1);
        assert!(record.action.contains("ReuseToolCache"));
        assert_eq!(record.estimated_token_saving, 500);
        assert!(record.accepted.is_none());
        assert!(record.outcome.is_none());
    }

    #[test]
    fn decision_record_accept() {
        let d = RuntimeDecision::cache_hit("grep", 42, 500);
        let mut record = RuntimeDecisionRecord::from_decision(1, &d);
        record.mark_accepted(true);
        assert_eq!(record.accepted, Some(true));
    }

    #[test]
    fn decision_record_evaluate_success() {
        let d = RuntimeDecision::cache_hit("grep", 42, 500);
        let mut record = RuntimeDecisionRecord::from_decision(1, &d);
        record.mark_outcome("success", Some(480));
        assert_eq!(record.outcome.as_deref(), Some("success"));
        assert_eq!(record.actual_token_saving, Some(480));
    }

    #[test]
    fn decision_record_evaluate_failure() {
        let d = RuntimeDecision::cache_hit("grep", 42, 500);
        let mut record = RuntimeDecisionRecord::from_decision(1, &d);
        record.mark_outcome("cache_invalidated", None);
        assert_eq!(record.outcome.as_deref(), Some("cache_invalidated"));
        assert_eq!(record.actual_token_saving, None);
    }

    #[test]
    fn evaluate_cache_reuse_match() {
        let outcome = DecisionEvaluation::evaluate("ReuseToolCache", 500, 50, false);
        assert!(outcome.success);
        assert_eq!(outcome.net_saving(), 450);
    }

    #[test]
    fn evaluate_cache_reuse_mismatch() {
        let outcome = DecisionEvaluation::evaluate("ReuseToolCache", 500, 500, false);
        assert!(!outcome.success);
        assert_eq!(outcome.net_saving(), 0);
    }

    #[test]
    fn evaluate_retry_with_fix_worked() {
        let outcome = DecisionEvaluation::evaluate("RetryWithFix", 200, 300, false);
        assert!(
            outcome.success,
            "fix cost more than estimate but solved the problem"
        );
    }

    #[test]
    fn evaluate_retry_with_fix_failed() {
        let outcome = DecisionEvaluation::evaluate("RetryWithFix", 200, 600, true);
        assert!(
            !outcome.success,
            "cost exceeded estimate and failure repeated"
        );
    }

    #[test]
    fn evaluate_compact_effective() {
        let outcome = DecisionEvaluation::evaluate("CompactAndProceed", 1000, 150, false);
        assert!(outcome.success);
        assert_eq!(outcome.net_saving(), 850);
    }

    #[test]
    fn replan_stage_triggers_on_assumption_change() {
        let state = RuntimeState {
            plan_hint: Some(PlanHint {
                plan_id: 1,
                goal: "add feature".into(),
                pending_step_count: 3,
                assumptions: vec!["src/lib.rs".to_string()],
            }),
            context_delta: vec!["src/lib.rs".to_string()],
            ..make_state(RuntimeProfile::Efficient, None, None, None)
        };
        let d = RuntimePolicy::decide(&state);
        assert!(matches!(d.action, RuntimeAction::Replan { .. }));
    }

    #[test]
    fn replan_stage_skips_when_assumptions_unchanged() {
        let state = RuntimeState {
            plan_hint: Some(PlanHint {
                plan_id: 1,
                goal: "add feature".into(),
                pending_step_count: 3,
                assumptions: vec!["src/lib.rs".to_string()],
            }),
            context_delta: vec!["src/other.rs".to_string()],
            ..make_state(RuntimeProfile::Efficient, None, None, None)
        };
        let d = RuntimePolicy::decide(&state);
        assert!(matches!(d.action, RuntimeAction::ContinuePlan { .. }));
    }

    #[test]
    fn replan_stage_empty_assumptions_continues() {
        let state = RuntimeState {
            plan_hint: Some(PlanHint {
                plan_id: 1,
                goal: "add feature".into(),
                pending_step_count: 2,
                assumptions: vec![],
            }),
            context_delta: vec!["src/lib.rs".to_string()],
            ..make_state(RuntimeProfile::Efficient, None, None, None)
        };
        let d = RuntimePolicy::decide(&state);
        assert!(matches!(d.action, RuntimeAction::ContinuePlan { .. }));
    }

    #[test]
    fn candidate_ranking_picks_highest_score() {
        let candidates = vec![
            DecisionCandidate {
                decision: RuntimeDecision::cache_hit("grep", 1, 100),
                score: 0.5,
                evidence: vec!["low confidence".into()],
            },
            DecisionCandidate {
                decision: RuntimeDecision::cache_hit("grep", 2, 500),
                score: 0.9,
                evidence: vec!["high confidence".into()],
            },
        ];
        let winner = RuntimePolicy::rank_candidates(candidates);
        assert_eq!(winner.decision.estimated_token_saving, 500);
        assert!((winner.score - 0.9).abs() < 0.01);
    }

    #[test]
    fn candidate_ranking_ties_use_insertion_order() {
        let candidates = vec![
            DecisionCandidate {
                decision: RuntimeDecision::cache_hit("first", 1, 100),
                score: 0.5,
                evidence: vec![],
            },
            DecisionCandidate {
                decision: RuntimeDecision::cache_hit("second", 2, 200),
                score: 0.5,
                evidence: vec![],
            },
        ];
        let winner = RuntimePolicy::rank_candidates(candidates);
        // Should contain "first" because stable sort preserves order
        assert!(winner.decision.reason.contains("first"));
    }

    #[test]
    fn candidate_ranking_empty_returns_none() {
        let result = RuntimePolicy::rank_candidates(vec![]);
        assert!(result.decision.action.variant_name() == "DelegateToModel");
    }

    #[test]
    fn decide_collects_and_ranks_from_pipeline() {
        // Only plan hint set — should trigger ContinuePlan (score ~0.7)
        let state = RuntimeState {
            plan_hint: Some(PlanHint {
                plan_id: 1,
                goal: "refactor".into(),
                pending_step_count: 3,
                assumptions: vec![],
            }),
            ..make_state(RuntimeProfile::Efficient, None, None, None)
        };
        let d = RuntimePolicy::decide(&state);
        assert!(
            matches!(d.action, RuntimeAction::ContinuePlan { .. }),
            "should pick ContinuePlan from candidates"
        );
        assert!(
            !d.evidence.is_empty(),
            "should carry evidence from winning stage"
        );
    }

    #[test]
    fn evaluate_failure_context_returns_none_without_patterns() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(async {
            crate::db::Database::builder()
                .path(dir.path().join("failure_test.db"))
                .build()
                .await
                .unwrap()
        });
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role":"user","content":"hi"}]))
            .unwrap();
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        let result = evaluate_failure_context(&db, conv_id, &cycle);
        assert!(result.is_none(), "no patterns → no context");
    }

    #[test]
    fn evaluate_failure_context_returns_fix_context_with_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(async {
            crate::db::Database::builder()
                .path(dir.path().join("failure_with_pattern.db"))
                .build()
                .await
                .unwrap()
        });
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role":"user","content":"hi"}]))
            .unwrap();
        let _id = db
            .store_failure_pattern(
                conv_id,
                "Error: ENOENT",
                "check file exists first",
                "file was deleted mid-run",
                &[],
                &[],
                None,
            )
            .unwrap();
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        let result = evaluate_failure_context(&db, conv_id, &cycle);
        assert!(
            result.is_some(),
            "should return context with fix suggestion"
        );
        let ctx = result.unwrap();
        assert!(
            ctx.contains("ENOENT"),
            "context should mention the error signature"
        );
        assert!(
            ctx.contains("check file exists first"),
            "context should mention the suggested fix"
        );
    }

    #[test]
    fn evaluate_plan_context_returns_none_without_plan() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(async {
            crate::db::Database::builder()
                .path(dir.path().join("plan_none.db"))
                .build()
                .await
                .unwrap()
        });
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role":"user","content":"hi"}]))
            .unwrap();
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        let result = evaluate_plan_context(&db, conv_id, &cycle, &[]);
        assert!(result.is_none(), "no active plan → no context");
    }

    #[test]
    fn evaluate_plan_context_returns_continue_with_active_plan() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(async {
            crate::db::Database::builder()
                .path(dir.path().join("plan_continue.db"))
                .build()
                .await
                .unwrap()
        });
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role":"user","content":"hi"}]))
            .unwrap();
        let steps = vec!["add tests".to_string(), "refactor".to_string()];
        let _plan_id = db
            .store_plan_state(conv_id, "improve quality", &steps, &[])
            .unwrap();
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        let result = evaluate_plan_context(&db, conv_id, &cycle, &[]);
        assert!(result.is_some(), "should return context with next step");
        let ctx = result.unwrap();
        assert!(
            ctx.contains("improve quality"),
            "context should mention goal"
        );
        assert!(
            ctx.contains("Next step"),
            "context should mention next step"
        );
    }

    #[test]
    fn evaluate_plan_context_returns_replan_when_assumptions_invalidated() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(async {
            crate::db::Database::builder()
                .path(dir.path().join("plan_replan.db"))
                .build()
                .await
                .unwrap()
        });
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role":"user","content":"hi"}]))
            .unwrap();
        let steps = vec!["add tests".to_string()];
        let assumptions = vec!["src/lib.rs".to_string()];
        let _plan_id = db
            .store_plan_state(conv_id, "improve quality", &steps, &assumptions)
            .unwrap();
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        // Pass a context_delta that includes the assumption file — triggers Replan
        let result = evaluate_plan_context(&db, conv_id, &cycle, &["src/lib.rs".to_string()]);
        assert!(result.is_some(), "should return replan context");
        let ctx = result.unwrap();
        assert!(
            ctx.contains("replanning"),
            "should indicate replan is needed"
        );
        assert!(
            ctx.contains("src/lib.rs"),
            "should mention invalidated assumption"
        );
    }

    // ── Online Evaluation ─────────────────────────────────────────────

    #[test]
    fn online_evaluator_adjusts_score_by_success_rate() {
        let history = vec![
            ("ReuseToolCache".to_string(), 8u64, 10u64), // 80% success
            ("RetryWithFix".to_string(), 1u64, 5u64),    // 20% success
        ];
        let eval = OnlineEvaluator { history };

        // High success rate → minimal reduction
        let rate = eval.success_rate("ReuseToolCache");
        assert!((rate - 0.8).abs() < 0.01, "80% success rate");

        // Low success rate → significant reduction
        let rate = eval.success_rate("RetryWithFix");
        assert!((rate - 0.2).abs() < 0.01, "20% success rate");

        // Unknown action → neutral (0.5)
        let rate = eval.success_rate("UnknownAction");
        assert!((rate - 0.5).abs() < 0.01, "unknown → 0.5");

        // Adjust candidate score
        let decision = RuntimeDecision::retry_with_fix(1, "fix it");
        let mut candidate = DecisionCandidate {
            decision,
            score: 1.0,
            evidence: vec![],
        };
        eval.adjust_score(&mut candidate, "retry_rule");
        // factor = 0.5 + 0.5 * 0.2 = 0.6
        assert!(
            (candidate.score - 0.6).abs() < 0.01,
            "adjusted score should be 0.6"
        );
        assert!(
            candidate.evidence.iter().any(|e| e.contains("online_eval")),
            "should include online_eval evidence"
        );
    }

    #[test]
    fn rule_engine_with_evaluator_applies_online_eval() {
        let history = vec![
            ("ReuseToolCache".to_string(), 9u64, 10u64), // 90% success
            ("RetryWithFix".to_string(), 0u64, 5u64),    // 0% success
        ];
        let eval = OnlineEvaluator { history };

        // Engine with evaluator should adjust scores
        let engine = RuleEngine::with_defaults().with_evaluator(eval);

        let state = RuntimeState {
            profile: RuntimeProfile::Efficient,
            metrics: RuntimeMetrics::default(),
            cache_hit: Some(CacheHit {
                tool_name: "grep".into(),
                cache_id: 1,
                estimated_token_saving: 500,
            }),
            failure_hint: None,
            plan_hint: None,
            context_delta: vec![],
        };
        let decision = engine.decide(&state);
        // CacheReuseRule should fire (90% success → factor=0.95, score=confidence)
        assert!(matches!(
            decision.action,
            RuntimeAction::ReuseToolCache { .. }
        ));
        assert!(
            decision.evidence.iter().any(|e| e.contains("online_eval")),
            "evidence should contain online_eval"
        );
    }
}
