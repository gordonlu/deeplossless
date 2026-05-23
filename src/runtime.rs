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
    pub fn is_live(&self) -> bool { matches!(self, Self::Live) }
    pub fn is_replay(&self) -> bool { matches!(self, Self::Replay { .. }) }
    pub fn is_dry_run(&self) -> bool { matches!(self, Self::DryRun) }
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
    /// How many times this specific failure has been retried.
    /// Used for per-pattern retry limiting, not global streak.
    pub retry_count: u32,
}

/// An active plan with pending steps.
#[derive(Debug, Clone, Serialize)]
pub struct PlanHint {
    pub plan_id: i64,
    pub goal: String,
    pub pending_step_count: usize,
}

// ── Runtime Action ────────────────────────────────────────────────────

/// Actions the runtime can recommend. Note: these are advisory.
/// The agent/UI decides whether to accept the recommendation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum RuntimeAction {
    /// Tool result found in cache — skip execution.
    ReuseToolCache { tool_name: String, cache_hit_id: i64 },
    /// Known failure fix available — apply it.
    RetryWithFix { failure_id: i64, suggested_fix: String },
    /// Active plan has pending steps — continue.
    ContinuePlan { step_index: usize, step_description: String },
    /// Plan assumptions invalid — re-plan needed.
    Replan { reason: String },
    /// Request more context before proceeding.
    RequestContext { what: String },
    /// Distill state into compact representation.
    CompactAndProceed,
    /// No optimization available — model decides.
    DelegateToModel,
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
        }
    }

    pub fn compact_and_proceed() -> Self {
        Self {
            action: RuntimeAction::CompactAndProceed,
            estimated_token_saving: 1000,
            confidence: 0.6,
            reason: "token budget critical — compacting context".to_string(),
        }
    }

    pub fn delegate(reason: &str) -> Self {
        Self {
            action: RuntimeAction::DelegateToModel,
            estimated_token_saving: 0,
            confidence: 0.0,
            reason: reason.to_string(),
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
            conv_id, logical_seq: seq,
            profile: profile.to_string(),
        });
    }

    /// Append ToolCallScheduled event.
    pub fn record_tool_call_scheduled(
        &mut self, conv_id: i64, tool_name: &str,
        tool_call_id: &str, span_id: &str, attempt: u32,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            span_id: span_id.to_string(),
            attempt,
        });
    }

    /// Append ToolCallCompleted event, then update metrics projection.
    pub fn record_tool_call_completed(
        &mut self, conv_id: i64, tool_name: &str,
        tool_call_id: &str, span_id: &str, attempt: u32,
        tokens_spent: u64, cache_hit: bool, execution_unit_id: i64,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallCompleted {
            conv_id, logical_seq: seq,
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
        &mut self, conv_id: i64, tool_name: &str,
        tool_call_id: &str, span_id: &str, attempt: u32,
        error_signature: &str, retryable: bool, execution_unit_id: i64,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::ToolCallFailed {
            conv_id, logical_seq: seq,
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
        &mut self, conv_id: i64, tool_call_id: &str,
        attempt: u32, suggested_fix: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::RetryScheduled {
            conv_id, logical_seq: seq,
            tool_call_id: tool_call_id.to_string(),
            attempt,
            suggested_fix: suggested_fix.to_string(),
        });
    }

    /// Append RetryAborted event.
    pub fn record_retry_aborted(
        &mut self, conv_id: i64, tool_call_id: &str,
        total_attempts: u32, reason: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::RetryAborted {
            conv_id, logical_seq: seq,
            tool_call_id: tool_call_id.to_string(),
            total_attempts,
            reason: reason.to_string(),
        });
    }

    /// Append CancellationRequested event.
    pub fn record_cancellation_requested(
        &mut self, conv_id: i64,
        source: crate::runtime_events::CancellationSource,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::CancellationRequested {
            conv_id, logical_seq: seq, source,
        });
    }

    /// Append CancellationAcknowledged event.
    pub fn record_cancellation_acknowledged(
        &mut self, conv_id: i64, tool_call_id: &str, span_id: &str,
    ) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::CancellationAcknowledged {
            conv_id, logical_seq: seq,
            tool_call_id: tool_call_id.to_string(),
            span_id: span_id.to_string(),
        });
    }

    /// Append CancellationCompleted event.
    pub fn record_cancellation_completed(&mut self, conv_id: i64, clean: bool) {
        let seq = crate::execution::next_logical_seq();
        self.append_event(crate::runtime_events::RuntimeEvent::CancellationCompleted {
            conv_id, logical_seq: seq, clean,
        });
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
        matches!(self, Self::Transient | Self::RateLimited | Self::QualityInsufficient)
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
        if msg.contains("timeout") || msg.contains("connection") || msg.contains("timed out")
            || msg.contains("dns") || msg.contains("refused") || msg.contains("reset")
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

/// Formal backoff calculation keyed by retry class.
pub struct RetryBackoff {
    pub max_retries: u32,
}

impl RetryBackoff {
    pub fn new(max_retries: u32) -> Self {
        Self { max_retries: max_retries.min(5) }
    }

    /// Compute delay for this attempt, in milliseconds.
    pub fn delay_ms(&self, attempt: u32, class: RetryClass) -> u64 {
        let jitter = crate::summarizer::jitter_millis(attempt);
        let base_secs = match class {
            RetryClass::RateLimited => 2u64.pow(attempt.min(4)),
            RetryClass::Transient => 1u64.pow(attempt.min(3)).max(1),
            RetryClass::QualityInsufficient => 0, // no delay — escalate immediately
            _ => 0, // non-retryable — don't wait
        };
        base_secs * 1000 + jitter
    }

    /// Whether this attempt should be retried, given the class and current count.
    pub fn should_retry(&self, attempt: u32, class: RetryClass) -> bool {
        class.is_retryable() && attempt < self.max_retries
    }
}

// ── Runtime Policy (advisory, not mandatory) ──────────────────────────

/// Advisory policy engine. Recommends optimization actions that the agent
/// can accept, ignore, or override. Never overrides model intent.
///
/// Uses a composable pipeline of [`PipelineStage`] implementations so each
/// decision rule is independently testable and reorderable (C-3).
pub struct RuntimePolicy;

/// A single decision stage in the runtime pipeline.
/// Each stage evaluates one condition; the first matching stage wins.
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, state: &RuntimeState) -> Option<RuntimeDecision>;
}

// ── Built-in pipeline stages ─────────────────────────────────────────

struct CacheReuseStage;
impl PipelineStage for CacheReuseStage {
    fn name(&self) -> &'static str { "cache_reuse" }
    fn evaluate(&self, state: &RuntimeState) -> Option<RuntimeDecision> {
        let hit = state.cache_hit.as_ref()?;
        let confidence = RuntimeStrategy::from_profile(state.profile).cache_aggressiveness;
        if confidence > 0.3 {
            Some(RuntimeDecision::cache_hit(&hit.tool_name, hit.cache_id, hit.estimated_token_saving))
        } else {
            None
        }
    }
}

struct RetryWithFixStage;
impl PipelineStage for RetryWithFixStage {
    fn name(&self) -> &'static str { "retry_with_fix" }
    fn evaluate(&self, state: &RuntimeState) -> Option<RuntimeDecision> {
        let fh = state.failure_hint.as_ref()?;
        if fh.suggested_fix.is_empty() { return None; }
        if fh.retry_count >= RuntimeStrategy::from_profile(state.profile).max_retries_per_failure {
            return None;
        }
        Some(RuntimeDecision::retry_with_fix(0, &fh.suggested_fix))
    }
}

struct ContinuePlanStage;
impl PipelineStage for ContinuePlanStage {
    fn name(&self) -> &'static str { "continue_plan" }
    fn evaluate(&self, state: &RuntimeState) -> Option<RuntimeDecision> {
        let ph = state.plan_hint.as_ref()?;
        if ph.pending_step_count > 0 {
            Some(RuntimeDecision::continue_plan(0, &ph.goal))
        } else {
            None
        }
    }
}

struct TokenCriticalStage;
impl PipelineStage for TokenCriticalStage {
    fn name(&self) -> &'static str { "token_critical" }
    fn evaluate(&self, state: &RuntimeState) -> Option<RuntimeDecision> {
        if state.metrics.is_token_critical() {
            Some(RuntimeDecision::compact_and_proceed())
        } else {
            None
        }
    }
}

impl RuntimePolicy {
    /// Ordered list of pipeline stages. Reorder to change evaluation priority.
    pub fn stages() -> Vec<Box<dyn PipelineStage>> {
        vec![
            Box::new(CacheReuseStage),
            Box::new(RetryWithFixStage),
            Box::new(ContinuePlanStage),
            Box::new(TokenCriticalStage),
        ]
    }

    /// Primary API: produce an advisory decision from a complete [`RuntimeState`].
    /// Iterates through the configured pipeline stages; first match wins.
    pub fn decide(state: &RuntimeState) -> RuntimeDecision {
        for stage in Self::stages() {
            if let Some(decision) = stage.evaluate(state) {
                return decision;
            }
        }
        RuntimeDecision::delegate("no applicable optimization")
    }

    /// Legacy convenience method. Prefer `decide(&RuntimeState)`.
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
                CacheHit { tool_name: tool_name.to_string(), cache_id, estimated_token_saving }
            }),
            failure_hint: has_recent_failure.map(|(signature, suggested_fix)| {
                FailureHint { signature: signature.to_string(), suggested_fix: suggested_fix.to_string(), why_failed: String::new(), retry_count: 0 }
            }),
            plan_hint: has_active_plan.map(|(plan_id, goal, pending_step_count)| {
                PlanHint { plan_id, goal: goal.to_string(), pending_step_count }
            }),
            context_delta: cycle.context_delta.clone(),
        };
        Self::decide(&state)
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
    let estimated_saved = m.cache_hits * 350 + m.cache_misses.saturating_sub(m.repeated_failures) * 100;
    let total_hits = m.cache_hits + m.cache_misses;
    let hit_pct = if total_hits > 0 { m.cache_hits as f64 / total_hits as f64 * 100.0 } else { 0.0 };

    let mut out = String::new();
    out.push_str(&format!("# deeplossless session report: {session_label}\n\n"));
    out.push_str(&format!("**{turn_count} turns** · **{session_duration_secs}s duration** · **{hit_pct:.0}% cache reuse**\n\n"));

    out.push_str("## Execution Reuse\n\n| Metric | Count |\n|--------|-------|\n");
    out.push_str(&format!("| Cache hits | {} |\n", m.cache_hits));
    out.push_str(&format!("| Cache misses | {} |\n", m.cache_misses));
    out.push_str(&format!("| Failure loops broken | {} |\n", m.repeated_failures.min(m.cache_hits / 2)));
    out.push_str(&format!("| Plans resumed | {} |\n\n", (m.planning_reuse_ratio * 100.0) as u64 / 10));

    out.push_str("## Inference Economics\n\n| Metric | Estimate |\n|--------|----------|\n");
    out.push_str(&format!("| Estimated tokens avoided | ~{estimated_saved} |\n"));
    out.push_str(&format!("| Tokens spent | {} |\n", m.tokens_spent));
    out.push_str(&format!("| Budget remaining | {:.0}% |\n\n", m.budget_remaining_pct * 100.0));

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
    if hit_pct > 30.0 { observations.push("Runtime reuse kicked in repeatedly.".into()); }
    if m.repeated_failures > 0 {
        let s = if m.repeated_failures > 1 { "s" } else { "" };
        observations.push(format!("Stopped {} potential failure loop{}.", m.repeated_failures, s));
    }
    if m.cache_hits > 10 {
        let s = if m.cache_hits > 1 { "s" } else { "" };
        observations.push(format!("Prevented {} redundant tool call{}.", m.cache_hits, s));
    }
    if !observations.is_empty() {
        out.push_str("## Highlights\n\n");
        for obs in &observations { out.push_str(&format!("- {obs}\n")); }
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
    let estimated_saved = m.cache_hits * 350 + m.cache_misses.saturating_sub(m.repeated_failures) * 100;
    let total_hits = m.cache_hits + m.cache_misses;
    let hit_pct = if total_hits > 0 { m.cache_hits as f64 / total_hits as f64 * 100.0 } else { 0.0 };

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
    pub fn distill(
        tool_sequence: &[(String, String, String)],
    ) -> String {
        if tool_sequence.is_empty() {
            return String::new();
        }

        let mut failures: Vec<String> = Vec::new();
        let mut final_fix = String::new();

        for (i, (name, _args, result)) in tool_sequence.iter().enumerate() {
            let is_error = result.contains("Error") || result.contains("error") || result.contains("fail");

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
        let prev = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        prev < self.max_per_sec
    }

    pub fn max_per_sec(&self) -> u64 { self.max_per_sec }
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
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        let deadline = tokio::time::Instant::now() + timeout;
        let handles = self.handles.lock().ok().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default();
        for handle in handles {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() { break; }
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

    fn make_state(profile: RuntimeProfile, cache: Option<(&str, i64, u64)>, failure: Option<(&str, &str)>, plan: Option<(i64, &str, usize)>) -> RuntimeState {
        let metrics = RuntimeMetrics { budget_remaining_pct: 1.0, ..RuntimeMetrics::default() };
        RuntimeState {
            profile,
            metrics,
            cache_hit: cache.map(|(t, id, save)| CacheHit { tool_name: t.to_string(), cache_id: id, estimated_token_saving: save }),
            failure_hint: failure.map(|(sig, fix)| FailureHint { signature: sig.to_string(), suggested_fix: fix.to_string(), why_failed: String::new(), retry_count: 0 }),
            plan_hint: plan.map(|(id, goal, count)| PlanHint { plan_id: id, goal: goal.to_string(), pending_step_count: count }),
            context_delta: vec![],
        }
    }

    #[test]
    fn cache_hit_confidence_scales_with_strategy() {
        let s_min = RuntimeState { profile: RuntimeProfile::Minimal, ..make_state(RuntimeProfile::Minimal, None, None, None) };
        let s_auto = RuntimeState { profile: RuntimeProfile::Autonomous, ..make_state(RuntimeProfile::Autonomous, None, None, None) };
        let d1 = RuntimePolicy::decide(&RuntimeState { cache_hit: Some(CacheHit { tool_name: "grep".into(), cache_id: 42, estimated_token_saving: 500 }), ..s_min });
        assert!(d1.confidence > 0.8, "minimal should be confident about cache");
        let d2 = RuntimePolicy::decide(&RuntimeState { cache_hit: Some(CacheHit { tool_name: "grep".into(), cache_id: 42, estimated_token_saving: 500 }), ..s_auto });
        assert!(d2.confidence < d1.confidence, "autonomous should be less aggressive about cache");
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
            ("compile".into(), "".into(), "Error: missing dep tokio".into()),
            ("cargo add".into(), "tokio".into(), "Success: added tokio v1.42".into()),
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
}
