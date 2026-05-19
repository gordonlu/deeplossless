//! Inference-aware Memory Runtime — optimizes inference economics, not agent behavior.
//!
//! Core philosophy: we are memory/runtime middleware, not an agent framework.
//! The runtime provides advisory optimization — cache reuse, delta injection,
//! failure avoidance, context compaction. It does NOT override model intent.
//!
//! Key boundary: `RuntimeDecision` is advisory. Agents/UIs accept, ignore, or override.

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
}

impl RuntimeProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Efficient => "efficient",
            Self::Exploratory => "exploratory",
            Self::Autonomous => "autonomous",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "minimal" => Some(Self::Minimal),
            "efficient" => Some(Self::Efficient),
            "exploratory" | "explore" => Some(Self::Exploratory),
            "autonomous" | "auto" => Some(Self::Autonomous),
            _ => None,
        }
    }
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
    pub fn from_profile(profile: RuntimeProfile) -> Self {
        match profile {
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
#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeMetrics {
    pub tokens_spent: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub repeated_failures: u64,
    pub reread_ratio: f64,
    pub planning_reuse_ratio: f64,
    pub budget_remaining_pct: f64,
    pub budget_total: u64,

    pub failure_streak: u32,
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
}

// ── Execution Cycle ───────────────────────────────────────────────────

/// State machine tracking the current inference cycle.
#[derive(Debug, Clone)]
pub struct ExecutionCycle {
    pub profile: RuntimeProfile,
    pub strategy: RuntimeStrategy,
    pub metrics: RuntimeMetrics,

    pub active_plan_id: Option<i64>,
    pub recent_failure_ids: Vec<i64>,
    pub cache_hits_this_cycle: Vec<String>,
    pub context_delta: Vec<String>,

    /// Decisions made this cycle (for audit).
    pub decisions: Vec<RuntimeDecision>,
}

impl ExecutionCycle {
    pub fn new(profile: RuntimeProfile) -> Self {
        let strategy = RuntimeStrategy::from_profile(profile);
        let mut metrics = RuntimeMetrics::default();
        metrics.budget_remaining_pct = 1.0;
        Self {
            profile,
            strategy,
            metrics,
            active_plan_id: None,
            recent_failure_ids: Vec::new(),
            cache_hits_this_cycle: Vec::new(),
            context_delta: Vec::new(),
            decisions: Vec::new(),
        }
    }

    pub fn record_cache_hit(&mut self, tool_name: &str) {
        self.metrics.cache_hits += 1;
        self.cache_hits_this_cycle.push(tool_name.to_string());
    }

    pub fn record_cache_miss(&mut self) {
        self.metrics.cache_misses += 1;
    }

    pub fn record_tokens(&mut self, tokens: u64) {
        self.metrics.tokens_spent += tokens;
    }

    pub fn record_failure(&mut self) {
        self.metrics.repeated_failures += 1;
        self.metrics.failure_streak += 1;
    }

    pub fn record_success(&mut self) {
        self.metrics.failure_streak = 0;
    }
}

// ── Runtime Policy (advisory, not mandatory) ──────────────────────────

/// Advisory policy engine. Recommends optimization actions that the agent
/// can accept, ignore, or override. Never overrides model intent.
///
/// Core responsibilities:
///   ✅ cache reuse       — avoid redundant tool execution
///   ✅ delta injection   — minimize repeated context
///   ✅ failure avoidance — prevent error loops
///   ✅ context compaction — distill when budget critical
///
/// Explicitly NOT:
///   ❌ autonomous planning    — model stays in control
///   ❌ multi-agent orchestration
///   ❌ self-reflection loops  — token black hole for DeepSeek
///   ❌ recursive reasoning    — model-killer for DeepSeek
pub struct RuntimePolicy;

impl RuntimePolicy {
    /// Produce an advisory decision based on current state.
    /// This is a recommendation, not a command.
    pub fn decide(
        cycle: &ExecutionCycle,
        has_tool_cache_hit: Option<(&str, i64, u64)>, // (tool_name, cache_id, estimated_save)
        has_recent_failure: Option<(&str, &str)>,     // (signature, fix)
        has_active_plan: Option<(i64, &str, usize)>,  // (plan_id, goal, pending_count)
    ) -> RuntimeDecision {
        // Rule 1: Tool cache hit with sufficient confidence
        if let Some((tool_name, cache_id, estimated_save)) = has_tool_cache_hit {
            let confidence = cycle.strategy.cache_aggressiveness;
            if confidence > 0.3 {
                return RuntimeDecision::cache_hit(tool_name, cache_id, estimated_save);
            }
        }

        // Rule 2: Failure with known fix
        if let Some((_sig, fix)) = has_recent_failure
            && !fix.is_empty()
            && cycle.metrics.failure_streak < cycle.strategy.max_retries_per_failure
        {
            return RuntimeDecision::retry_with_fix(0, fix);
        }

        // Rule 3: Active plan with pending steps
        if let Some((_plan_id, _goal, pending_count)) = has_active_plan
            && pending_count > 0
        {
            return RuntimeDecision::continue_plan(0, "");
        }

        // Rule 4: Token budget critical
        if cycle.metrics.is_token_critical() {
            return RuntimeDecision::compact_and_proceed();
        }

        // Rule 5: No optimization applies — model's turn
        RuntimeDecision::delegate("no applicable optimization")
    }
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

    #[test]
    fn cache_hit_confidence_scales_with_strategy() {
        let aggressive = ExecutionCycle::new(RuntimeProfile::Minimal);
        let relaxed = ExecutionCycle::new(RuntimeProfile::Autonomous);

        // Minimal profile: high confidence on cache hit
        let d1 = RuntimePolicy::decide(&aggressive, Some(("grep", 42, 500)), None, None);
        assert!(d1.confidence > 0.8, "minimal should be confident about cache");

        // Autonomous profile: lower confidence on same cache hit
        let d2 = RuntimePolicy::decide(&relaxed, Some(("grep", 42, 500)), None, None);
        assert!(d2.confidence < d1.confidence, "autonomous should be less aggressive about cache");
    }

    #[test]
    fn empty_state_delegates_to_model() {
        let cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        let d = RuntimePolicy::decide(&cycle, None, None, None);
        assert!(matches!(d.action, RuntimeAction::DelegateToModel));
        assert_eq!(d.confidence, 0.0);
    }

    #[test]
    fn token_critical_triggers_compact() {
        let mut cycle = ExecutionCycle::new(RuntimeProfile::Efficient);
        cycle.metrics.budget_remaining_pct = 0.10;
        let d = RuntimePolicy::decide(&cycle, None, None, None);
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
}
