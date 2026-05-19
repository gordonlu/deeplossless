//! Execution Runtime Layer — the spine that connects tool_cache, failures,
//! plans, and deltas into a closed-loop inference-aware runtime.
//!
//! This is not a memory module. It's the policy layer that determines
//! which tokens are worth producing and which are wasted recomputation.

use serde::{Deserialize, Serialize};

// ── Runtime Mode ──────────────────────────────────────────────────────

/// The runtime's operating mode determines token economics.
/// Different phases demand different policies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RuntimeMode {
    /// Minimum tokens. Aggressive cache reuse. Freeze plans early.
    FastIteration,
    /// Success rate first. Allow retries, deeper context.
    DeepDebug,
    /// Explore codebase structure. Cache-heavy, low reasoning.
    RepoExploration,
    /// Autonomous fix with full reasoning. Highest token budget.
    AutonomousFix,
}

impl RuntimeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FastIteration => "fast",
            Self::DeepDebug => "debug",
            Self::RepoExploration => "explore",
            Self::AutonomousFix => "autonomous",
        }
    }
}

// ── Runtime Action ────────────────────────────────────────────────────

/// The scheduler decides the next action — not the model.
/// This prevents agent entropy: wandering reasoning, tool spam, repeated planning.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum RuntimeAction {
    /// Tool result found in cache, skip execution entirely.
    ReuseToolCache { tool_name: String, cache_hit_id: i64 },
    /// Previous failure has a known fix, apply it.
    RetryWithFix { failure_id: i64, suggested_fix: String },
    /// Active plan has pending steps, execute the next one.
    ContinuePlan { step_index: usize, step_description: String },
    /// Plan assumptions are invalidated, need to re-plan.
    Replan { reason: String },
    /// Insufficient context to decide, request more.
    RequestContext { what: String },
    /// Distill current state into compact representation, then proceed.
    CompactAndProceed,
    /// No cached/planned action applies, fall through to LLM.
    DelegateToModel,
}

// ── Runtime Metrics ───────────────────────────────────────────────────

/// Cost-aware metrics that feed back into the scheduler.
/// The runtime uses these to dynamically adjust policy.
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

    /// Consecutive failures without progress.
    pub failure_streak: u32,
    /// How many times the same tool was called with same args.
    pub tool_repeat_count: u32,
}

impl RuntimeMetrics {
    /// Should we switch to a more conservative mode?
    pub fn is_token_critical(&self) -> bool {
        self.budget_remaining_pct < 0.15 || self.failure_streak >= 3
    }

    /// Is the cache effective enough to stay in fast mode?
    pub fn cache_effective(&self) -> bool {
        let total = self.cache_hits + self.cache_misses;
        total > 0 && (self.cache_hits as f64 / total as f64) > 0.6
    }
}

// ── Execution Cycle ───────────────────────────────────────────────────

/// The core state machine. Each turn is a state transition, not a new prompt.
#[derive(Debug, Clone)]
pub struct ExecutionCycle {
    pub mode: RuntimeMode,
    pub metrics: RuntimeMetrics,

    /// Active plan state ID (from plan_states table), if any.
    pub active_plan_id: Option<i64>,

    /// Recent failure pattern IDs that are still relevant.
    pub recent_failure_ids: Vec<i64>,

    /// Tool calls in the current cycle that hit cache.
    pub cache_hits_this_cycle: Vec<String>,

    /// Files changed since last cycle.
    pub context_delta: Vec<String>,
}

impl ExecutionCycle {
    pub fn new(mode: RuntimeMode) -> Self {
        let mut metrics = RuntimeMetrics::default();
        metrics.budget_remaining_pct = 1.0; // start with full budget
        Self {
            mode,
            metrics,
            active_plan_id: None,
            recent_failure_ids: Vec::new(),
            cache_hits_this_cycle: Vec::new(),
            context_delta: Vec::new(),
        }
    }

    /// Record a cache hit for metrics.
    pub fn record_cache_hit(&mut self, tool_name: &str) {
        self.metrics.cache_hits += 1;
        self.cache_hits_this_cycle.push(tool_name.to_string());
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&mut self) {
        self.metrics.cache_misses += 1;
    }

    /// Record tokens spent this cycle.
    pub fn record_tokens(&mut self, tokens: u64) {
        self.metrics.tokens_spent += tokens;
    }

    /// Record a failure occurrence.
    pub fn record_failure(&mut self) {
        self.metrics.repeated_failures += 1;
        self.metrics.failure_streak += 1;
    }

    /// Reset failure streak on success.
    pub fn record_success(&mut self) {
        self.metrics.failure_streak = 0;
    }
}

// ── Runtime Policy ────────────────────────────────────────────────────

/// The policy that maps state → action. This is the "intelligence" layer
/// that replaces free-form model reasoning with deterministic transitions.
pub struct RuntimePolicy;

impl RuntimePolicy {
    /// Decide the next action based on current state and available information.
    /// This is a deterministic function — no LLM call needed.
    pub fn decide(
        cycle: &ExecutionCycle,
        has_tool_cache_hit: Option<(&str, i64)>,
        has_recent_failure: Option<(&str, &str)>,
        has_active_plan: Option<(i64, &str, usize)>,
    ) -> RuntimeAction {
        // Rule 1: Tool cache hit → reuse immediately (fastest path)
        if let Some((tool_name, cache_id)) = has_tool_cache_hit {
            return RuntimeAction::ReuseToolCache {
                tool_name: tool_name.to_string(),
                cache_hit_id: cache_id,
            };
        }

        // Rule 2: Recent failure with known fix → retry with fix
        if let Some((_sig, fix)) = has_recent_failure
            && !fix.is_empty() {
                return RuntimeAction::RetryWithFix {
                    failure_id: 0,
                    suggested_fix: fix.to_string(),
                };
            }

        // Rule 3: Active plan with pending steps → continue
        if let Some((_plan_id, _goal, pending_count)) = has_active_plan
            && pending_count > 0 {
                return RuntimeAction::ContinuePlan {
                    step_index: 0,
                    step_description: String::new(),
                };
            }

        // Rule 4: Token critical → compact and reduce
        if cycle.metrics.is_token_critical() {
            return RuntimeAction::CompactAndProceed;
        }

        // Rule 5: No deterministic path → let model decide
        RuntimeAction::DelegateToModel
    }

    /// Select runtime mode based on heuristics.
    pub fn select_mode(
        tokens_spent: u64,
        budget_total: u64,
        failure_count: u32,
        is_exploration: bool,
    ) -> RuntimeMode {
        if is_exploration {
            return RuntimeMode::RepoExploration;
        }
        if failure_count >= 5 {
            return RuntimeMode::DeepDebug;
        }
        if tokens_spent > budget_total / 2 {
            return RuntimeMode::FastIteration;
        }
        RuntimeMode::AutonomousFix
    }
}

// ── Reasoning Distillation ────────────────────────────────────────────

/// Distill execution history into a compact outcome. This is "Runtime Compaction":
/// not compressing text, but distilling reasoning.
///
/// Input: a sequence of tool calls and results that form a reasoning path.
/// Output: a single sentence describing the approach and outcome.
pub struct ExecutionCompactor;

impl ExecutionCompactor {
    /// Distill a sequence of tool calls into a reasoning summary.
    /// E.g., "Tried approach A. Failed because X. Final fix: Y."
    pub fn distill(
        tool_sequence: &[(String, String, String)], // (tool_name, args, result_summary)
    ) -> String {
        if tool_sequence.is_empty() {
            return String::new();
        }

        let mut stages: Vec<String> = Vec::new();
        let mut final_fix = String::new();

        for (i, (name, _args, result)) in tool_sequence.iter().enumerate() {
            let is_error = result.contains("Error") || result.contains("error") || result.contains("fail");

            if is_error {
                let brief: String = result.chars().take(100).collect();
                stages.push(format!("{name} → failed: {brief}…"));
            } else if i == tool_sequence.len() - 1 {
                final_fix = format!("Final successful {name}: {}", result);
            }
        }

        let mut out = String::new();
        if !stages.is_empty() {
            out.push_str("Tried: ");
            out.push_str(&stages.join(". "));
            out.push_str(". ");
        }
        if !final_fix.is_empty() {
            out.push_str(&final_fix);
        } else if stages.is_empty() {
            // No failures, just a single success
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
    fn policy_cache_hit_takes_priority() {
        let cycle = ExecutionCycle::new(RuntimeMode::FastIteration);
        let action = RuntimePolicy::decide(
            &cycle,
            Some(("grep", 42)),
            Some(("deadlock", "use WAL mode")),
            Some((1, "fix bug", 3)),
        );
        assert!(matches!(action, RuntimeAction::ReuseToolCache { .. }),
            "cache hit should be highest priority, got {action:?}");
    }

    #[test]
    fn policy_falls_through_when_nothing_available() {
        let cycle = ExecutionCycle::new(RuntimeMode::FastIteration);
        let action = RuntimePolicy::decide(&cycle, None, None, None);
        assert!(matches!(action, RuntimeAction::DelegateToModel),
            "empty state should delegate to model");
    }

    #[test]
    fn token_critical_triggers_compact() {
        let mut cycle = ExecutionCycle::new(RuntimeMode::FastIteration);
        cycle.metrics.budget_remaining_pct = 0.10;
        let action = RuntimePolicy::decide(&cycle, None, None, None);
        assert!(matches!(action, RuntimeAction::CompactAndProceed),
            "low budget should trigger compaction");
    }

    #[test]
    fn distill_failure_then_success() {
        let seq = vec![
            ("grep".into(), "pattern".into(), "found 0 results".into()),
            ("compile".into(), "".into(), "Error: missing dependency tokio".into()),
            ("cargo add".into(), "tokio".into(), "Success: added tokio v1.42".into()),
        ];
        let distilled = ExecutionCompactor::distill(&seq);
        assert!(distilled.contains("failed"), "should mention failures");
        assert!(distilled.contains("Success"), "should mention success");
    }

    #[test]
    fn metrics_cache_effective() {
        let mut m = RuntimeMetrics::default();
        m.cache_hits = 8;
        m.cache_misses = 2;
        assert!(m.cache_effective());

        m.cache_hits = 2;
        m.cache_misses = 8;
        assert!(!m.cache_effective());
    }
}
