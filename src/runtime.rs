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

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "minimal" => Some(Self::Minimal),
            "efficient" => Some(Self::Efficient),
            "exploratory" | "explore" => Some(Self::Exploratory),
            "autonomous" | "auto" => Some(Self::Autonomous),
            "custom" => Some(Self::Custom),
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
        let metrics = RuntimeMetrics {
            budget_remaining_pct: 1.0,
            ..Default::default()
        };
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
    /// Primary API: produce an advisory decision from a complete [`RuntimeState`].
    /// This is the pure function — no side effects, no infrastructure.
    pub fn decide(state: &RuntimeState) -> RuntimeDecision {
        // Rule 1: Tool cache hit with sufficient confidence
        if let Some(ref hit) = state.cache_hit {
            let confidence = RuntimeStrategy::from_profile(state.profile).cache_aggressiveness;
            if confidence > 0.3 {
                return RuntimeDecision::cache_hit(&hit.tool_name, hit.cache_id, hit.estimated_token_saving);
            }
        }

        // Rule 2: Failure with known fix (per-pattern retry limit)
        if let Some(ref fh) = state.failure_hint
            && !fh.suggested_fix.is_empty()
            && fh.retry_count < RuntimeStrategy::from_profile(state.profile).max_retries_per_failure
        {
            return RuntimeDecision::retry_with_fix(0, &fh.suggested_fix);
        }

        // Rule 3: Active plan with pending steps
        if let Some(ref ph) = state.plan_hint
            && ph.pending_step_count > 0
        {
            return RuntimeDecision::continue_plan(0, &ph.goal);
        }

        // Rule 4: Token budget critical
        if state.metrics.is_token_critical() {
            return RuntimeDecision::compact_and_proceed();
        }

        // Rule 5: No optimization applies — model's turn
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
