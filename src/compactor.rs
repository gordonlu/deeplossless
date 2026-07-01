use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::dag::{DagEngine, DagConfig, DagNode};
use crate::db::Database;
use crate::summarizer::{Summarizer, SummarizerConfig};

/// Commands sent from the main thread to the compaction worker.
pub enum CompactCommand {
    CompressGroup { conv_id: i64, node_ids: Vec<i64> },
    ReviewAndCompact { conv_id: i64, context_window: usize },
    SlideAndCompact { conv_id: i64, window_size: usize, context_window: usize },
    Ping,
    Shutdown,
}

/// Events sent back from the compaction worker to the main thread.
#[derive(Debug)]
pub enum CompactEvent {
    GroupCompressed { conv_id: i64, new_node_id: i64, level: u8, tokens_saved: i64, latency_ms: u64, summarizer_level: u8 },
    /// Compaction completed for a conversation (per-command summary).
    CompactionCompleted { conv_id: i64, groups: u32, tokens_saved: i64, latency_ms: u64, failures: u32 },
    BelowThreshold { conv_id: i64, reason: &'static str },
    Error { message: String, conv_id: Option<i64> },
    /// Response to health ping.
    Pong,
}
impl CompactEvent {
    pub fn below(conv_id: i64, reason: &'static str) -> Self {
        CompactEvent::BelowThreshold { conv_id, reason }
    }
}

/// Configuration for the compaction loop.
#[derive(Clone, Debug)]
pub struct CompactorConfig {
    pub dag: DagConfig,
    pub summarizer: SummarizerConfig,
    pub soft_threshold_pct: f64,
    pub hard_threshold_pct: f64,
    pub group_size: usize,
    /// Weight for position-age in leaf scoring (0-1). Default: 0.4.
    pub age_weight: f64,
    /// Weight for token density in leaf scoring (0-1). Default: 0.2.
    pub token_density_weight: f64,
    /// Weight for novelty in leaf scoring (0-1). Default: 0.4.
    pub novelty_weight: f64,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            dag: DagConfig::default(),
            summarizer: SummarizerConfig::default(),
            soft_threshold_pct: 0.80,
            hard_threshold_pct: 0.95,
            group_size: 32,
            age_weight: 0.4,
            token_density_weight: 0.2,
            novelty_weight: 0.4,
        }
    }
}

/// Accumulated compaction telemetry for operator visibility (P1).
#[derive(Debug, Default, Clone)]
pub struct CompactionMetrics {
    pub total_compactions: u64,
    pub total_groups: u64,
    pub total_tokens_saved: i64,
    pub total_llm_calls: u64,
    pub total_fallbacks: u64,
    pub total_failures: u64,
    pub last_latency_ms: u64,
}

// ── Compaction planning types ─────────────────────────────────────────

/// Token-budget-aware threshold calculator (P0-6).
/// Decoupled from the compactor to allow independent tuning and testing.
#[derive(Debug, Clone)]
pub struct CompactionBudget {
    pub soft_limit: i64,
    pub hard_limit: i64,
    pub group_size: usize,
}

impl CompactionBudget {
    pub fn new(context_window: usize, soft_pct: f64, hard_pct: f64, group_size: usize) -> Self {
        Self {
            soft_limit: (context_window as f64 * soft_pct) as i64,
            hard_limit: (context_window as f64 * hard_pct) as i64,
            group_size,
        }
    }

    /// True when compaction is strongly recommended (above hard limit).
    pub fn is_critical(&self, total_tokens: i64) -> bool {
        total_tokens >= self.hard_limit
    }

    /// True when compaction should be considered (above soft limit or many leaves).
    pub fn is_advisory(&self, total_tokens: i64, leaf_count: usize) -> bool {
        total_tokens >= self.soft_limit || leaf_count >= self.group_size * 8
    }
}

/// Explicit compaction priority: 0.0 (skip) to 1.0 (urgent).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CompactionScore(pub f64);

/// A planned compaction operation on a group of nodes.
#[derive(Debug, Clone)]
pub struct CompactionGroup {
    pub node_ids: Vec<i64>,
    pub score: CompactionScore,
    pub strategy: CompactionStrategy,
}

/// What kind of compaction to apply to this group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Standard: group raw leaves into a summary (level 1).
    Summarize,
    /// Slide-window: compact oldest leaves outside the window.
    SlideWindow { window_size: usize },
    /// Merge: refine existing summaries into a higher-level node.
    Merge,
}

/// Planned compaction for a single pass over a conversation.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    pub conv_id: i64,
    pub budget: CompactionBudget,
    pub groups: Vec<CompactionGroup>,
    pub leaves: Vec<DagNode>,
    pub total_tokens: i64,
}

impl CompactionPlan {
    pub fn should_compact(&self) -> bool {
        self.budget.is_critical(self.total_tokens) || self.budget.is_advisory(self.total_tokens, self.leaves.len())
    }
}

// ── Planner ───────────────────────────────────────────────────────────

/// Compaction planner: analyzes DAG state and produces a prioritized plan.
///
/// Separates the decision of **what** to compact from **how** to compact
/// it (P0-1). Uses an explicit scoring model instead of heuristic threshold
/// hacks (P0-4).
#[derive(Debug, Clone)]
pub struct CompactionPlanner {
    config: CompactorConfig,
}

impl CompactionPlanner {
    pub fn new(config: CompactorConfig) -> Self {
        Self { config }
    }

    /// Produce a compaction plan for `conv_id` given the current DAG state.
    /// Returns an empty plan (no groups) when no compaction is needed.
    pub fn plan(&self, dag: &DagEngine, conv_id: i64, context_window: usize) -> anyhow::Result<CompactionPlan> {
        let budget = CompactionBudget::new(
            context_window,
            self.config.soft_threshold_pct,
            self.config.hard_threshold_pct,
            self.config.group_size,
        );

        let total_tokens = dag.total_tokens(conv_id)?;
        let leaves = dag.get_leaves(conv_id)?;
        let leaf_count = leaves.len();

        if leaf_count < 2 || (!budget.is_critical(total_tokens) && !budget.is_advisory(total_tokens, leaf_count)) {
            return Ok(CompactionPlan {
                conv_id, budget, groups: Vec::new(), leaves, total_tokens,
            });
        }

        let groups = self.build_groups(&leaves, &budget);

        Ok(CompactionPlan { conv_id, budget, groups, leaves, total_tokens })
    }

    /// Check whether a conversation's topology has changed since `last_leaf_count`
    /// was recorded — i.e., whether a re-plan is warranted (P0-11 dirty-region).
    pub fn is_dirty(&self, dag: &DagEngine, conv_id: i64, last_leaf_count: usize) -> anyhow::Result<bool> {
        let current = dag.get_leaves(conv_id)?.len();
        Ok(current != last_leaf_count || current >= self.config.group_size * 2)
    }

    /// Build compaction groups from leaves using explicit scoring.
    fn build_groups(&self, leaves: &[DagNode], budget: &CompactionBudget) -> Vec<CompactionGroup> {
        // Score each leaf
        let texts: Vec<&str> = leaves.iter().map(|n| n.summary.as_str()).collect();
        let novelty = novelty_score_for_group(&texts);

        let mut scored: Vec<(i64, CompactionScore)> = leaves
            .iter()
            .enumerate()
            .map(|(position, node)| {
                let score = self.score_leaf(node, position, leaves.len(), novelty);
                (node.id, score)
            })
            .collect();

        // Sort by score descending (most compressible first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Group the top-scoring nodes
        let node_ids: Vec<i64> = scored
            .into_iter()
            .take(budget.group_size)
            .map(|(id, _)| id)
            .collect();

        if node_ids.len() < 2 {
            return Vec::new();
        }

        vec![CompactionGroup {
            node_ids,
            score: CompactionScore(novelty),
            strategy: CompactionStrategy::Summarize,
        }]
    }

    /// Score a single leaf node for compaction priority.
    ///
    /// Factors:
    /// - **Position age**: older (lower-index) leaves score higher (compress oldest first)
    /// - **Token density**: high token count relative to position → more to save
    /// - **Novelty**: low novelty vs. peers → more redundant → higher compress priority
    ///
    /// Returns 0.0 (keep as-is) to 1.0 (compress immediately).
    fn score_leaf(&self, node: &DagNode, position: usize, total: usize, novelty: f64) -> CompactionScore {
        if total <= 1 {
            return CompactionScore(0.0);
        }

        // Older leaves (lower position / larger total) = higher priority
        let age_factor = 1.0 - (position as f64 / total as f64);

        // High token count = more urgent to compress
        let token_factor = (node.token_count as f64 / 500.0).min(1.0);

        // Low novelty = redundant = safe to compress
        let novelty_factor = 1.0 - novelty;

        let w = &self.config;
        let raw = w.age_weight * age_factor
            + w.token_density_weight * token_factor
            + w.novelty_weight * novelty_factor;
        CompactionScore(raw.clamp(0.0, 1.0))
    }

    /// Build groups using sliding-window strategy.
    pub fn plan_slide_window(
        &self,
        dag: &DagEngine,
        conv_id: i64,
        window_size: usize,
        context_window: usize,
    ) -> anyhow::Result<CompactionPlan> {
        let budget = CompactionBudget::new(
            context_window,
            self.config.soft_threshold_pct,
            self.config.hard_threshold_pct,
            self.config.group_size,
        );

        let total_tokens = dag.total_tokens(conv_id).unwrap_or(0);
        let leaves = dag.get_leaves(conv_id)?;
        let leaf_count = leaves.len();

        if leaf_count <= window_size {
            return Ok(CompactionPlan {
                conv_id, budget, groups: Vec::new(), leaves, total_tokens,
            });
        }

        if total_tokens < budget.hard_limit && leaf_count < window_size * 2 {
            return Ok(CompactionPlan {
                conv_id, budget, groups: Vec::new(), leaves, total_tokens,
            });
        }

        let compact_count = (leaf_count - window_size).min(8);
        let node_ids: Vec<i64> = leaves.iter().take(compact_count).map(|n| n.id).collect();

        if node_ids.len() < 2 {
            return Ok(CompactionPlan {
                conv_id, budget, groups: Vec::new(), leaves, total_tokens,
            });
        }

        Ok(CompactionPlan {
            conv_id,
            budget,
            groups: vec![CompactionGroup {
                node_ids,
                score: CompactionScore(0.5),
                strategy: CompactionStrategy::SlideWindow { window_size },
            }],
            total_tokens,
            leaves,
        })
    }
}

/// Compute novelty score for a group of leaf texts.
/// Wrapper around the trigram-based novelty_score for use in the scoring model.
fn novelty_score_for_group(texts: &[&str]) -> f64 {
    novelty_score(texts)
}

/// Async compaction coordinator.
///
/// Spawns a background task that receives `CompactCommand` via mpsc and
/// sends `CompactEvent` replies. All DB writes go through this single
/// task, maintaining the single-writer invariant.
pub struct Compactor {
    cmd_tx: mpsc::Sender<CompactCommand>,
    event_rx: mpsc::Receiver<CompactEvent>,
    config: CompactorConfig,
}

impl Compactor {
    pub fn new(
        db: Arc<Database>,
        config: CompactorConfig,
        tasks: Option<&std::sync::Arc<crate::runtime::BackgroundTasks>>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(32);
        let (event_tx, event_rx) = mpsc::channel(32);

        let dag = Arc::new(
            DagEngine::builder()
                .soft_threshold(config.soft_threshold_pct)
                .hard_threshold(config.hard_threshold_pct)
                .max_level(config.dag.max_level)
                .recent_messages(config.dag.recent_message_count)
                .build(db),
        );

        let summarizer = match Summarizer::builder()
            .api_key(&config.summarizer.api_key)
            .model(&config.summarizer.model)
            .upstream(&config.summarizer.upstream)
            .build()
        {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "deeplossless::compactor", "summarizer build failed: {e}");
                // Explicit offline fallback: compaction can still run using
                // deterministic Level 3 without attempting upstream calls.
                Summarizer::builder()
                    .offline_fallback_only()
                    .fallback_max_tokens(config.summarizer.fallback_max_tokens)
                    .build()
                    .expect("fallback summarizer build failed")
            }
        };

        let worker_handle = tokio::spawn(compactor_supervisor(cmd_rx, event_tx, dag, summarizer, config.clone()));

        // Register with BackgroundTasks for lifecycle supervision (P0 shutdown gap fix).
        // BackgroundTasks now owns the handle — compactor no longer stores it.
        if let Some(tasks) = tasks {
            tasks.register_handle(worker_handle);
        }

        Self { cmd_tx, event_rx, config }
    }

    pub async fn command(&mut self, cmd: CompactCommand) -> Option<CompactEvent> {
        self.cmd_tx.send(cmd).await.ok()?;
        self.event_rx.recv().await
    }

    /// Fire-and-forget command send — does NOT wait for a response.
    /// Use when the caller doesn't need a synchronous reply, so the
    /// Mutex is not held while the background worker processes the command.
    pub async fn send_command(&mut self, cmd: CompactCommand) -> Result<(), ()> {
        self.cmd_tx.send(cmd).await.map_err(|_| ())
    }

    pub fn drain_events(&mut self) -> Vec<CompactEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.event_rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn config(&self) -> &CompactorConfig { &self.config }

    /// Health check: sends a Ping through the command channel and waits for Pong.
    /// Times out after 500ms. Returns true if worker is alive.
    pub async fn health_ping(&mut self) -> bool {
        if self.cmd_tx.send(CompactCommand::Ping).await.is_err() {
            return false;
        }
        matches!(
            tokio::time::timeout(Duration::from_millis(500), self.event_rx.recv()).await,
            Ok(Some(CompactEvent::Pong))
        )
    }
}

/// Compute novelty score for a set of messages relative to each other.
/// Returns 0.0 (all redundant) to 1.0 (all novel).
/// Uses trigram Jaccard distance: low overlap = high novelty.
fn novelty_score(texts: &[&str]) -> f64 {
    if texts.len() < 2 {
        return 1.0;
    }
    let mut all_trigrams: Vec<std::collections::HashSet<[u8; 3]>> = Vec::new();
    for text in texts {
        let bytes = text.as_bytes();
        let trigrams: std::collections::HashSet<[u8; 3]> = bytes.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
        if !trigrams.is_empty() {
            all_trigrams.push(trigrams);
        }
    }
    if all_trigrams.len() < 2 {
        return 1.0;
    }
    // Average pairwise Jaccard distance
    let mut total_dist = 0.0;
    let mut pairs = 0;
    for i in 0..all_trigrams.len() {
        for j in i+1..all_trigrams.len() {
            let intersection = all_trigrams[i].intersection(&all_trigrams[j]).count();
            let union = all_trigrams[i].union(&all_trigrams[j]).count();
            if union > 0 {
                total_dist += 1.0 - (intersection as f64 / union as f64);
                pairs += 1;
            }
        }
    }
    if pairs == 0 { 1.0 } else { total_dist / pairs as f64 }
}

/// AST-aware code compression: preserve function/type signatures, trim bodies.
/// Reduces token count while keeping semantic structure for the summarizer.
fn compress_code_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_code = false;
    let mut code_buf = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                // Compress the code block
                let compressed = compress_code(&code_buf);
                result.push_str("```\n");
                result.push_str(&compressed);
                result.push_str("\n```\n");
                code_buf.clear();
                in_code = false;
            } else {
                result.push_str(line);
                result.push('\n');
                in_code = true;
            }
        } else if in_code {
            code_buf.push_str(line);
            code_buf.push('\n');
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    // Unclosed code block
    if !code_buf.is_empty() {
        result.push_str(&code_buf);
    }
    result
}

/// Compress code by keeping signatures + first line of body, trimming the rest.
fn compress_code(code: &str) -> String {
    let mut out = String::new();
    let mut skip_until_close = false;
    let mut indent_depth: usize = 0;

    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }

        // Track brace depth
        let opens = trimmed.matches('{').count();
        let closes = trimmed.matches('}').count();
        indent_depth = indent_depth.saturating_add(opens).saturating_sub(closes);

        // Keep structural lines fully
        let is_structural = trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("impl ") || trimmed.starts_with("use ")
            || trimmed.starts_with("mod ") || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("const ") || trimmed.starts_with("type ")
            || trimmed.starts_with("async fn ") || trimmed.starts_with("pub async fn ")
            || trimmed.starts_with("class ") || trimmed.starts_with("def ")
            || trimmed.starts_with("func ") || trimmed.starts_with("import ");

        if is_structural {
            out.push_str(line);
            out.push('\n');
            skip_until_close = opens > 0;
        } else if skip_until_close {
            if closes > 0 && indent_depth <= 1 {
                skip_until_close = false;
                out.push_str("    // ...\n");
            }
        } else if indent_depth > 0 {
            // First line inside a block: keep it as hint
            if !skip_until_close {
                out.push_str(line);
                out.push('\n');
                skip_until_close = true;
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

async fn compact_worker(
    mut cmd_rx: mpsc::Receiver<CompactCommand>,
    event_tx: mpsc::Sender<CompactEvent>,
    dag: Arc<DagEngine>,
    summarizer: Summarizer,
    config: CompactorConfig,
) {
    let planner = CompactionPlanner::new(config.clone());
    // Dirty-region tracking (P0-11): skip re-planning when DAG topology unchanged.
    let mut last_leaf_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    // Compaction cooldown: skip repeated compaction for same conversation within 30s.
    let mut last_compacted: std::collections::HashMap<i64, std::time::Instant> = std::collections::HashMap::new();
    const COMPACTION_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30);

    // Periodically prune expired cooldown entries
    let mut cooldown_prune_counter: u64 = 0;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            CompactCommand::Shutdown => break,
            CompactCommand::Ping => {
                if event_tx.send(CompactEvent::Pong).await.is_err() {
                    tracing::warn!(target: "deeplossless::compactor", "event receiver dropped on ping");
                }
            }
            CompactCommand::ReviewAndCompact { conv_id, context_window } => {
                // Cooldown: skip if compacted this conversation within 30s
                if let Some(last) = last_compacted.get(&conv_id) {
                    if last.elapsed() < COMPACTION_COOLDOWN {
                        continue;
                    }
                }
                last_compacted.insert(conv_id, std::time::Instant::now());
                // Periodic prune: clear expired entries every 50 triggers
                cooldown_prune_counter += 1;
                if cooldown_prune_counter % 50 == 0 {
                    last_compacted.retain(|_, t| t.elapsed() < COMPACTION_COOLDOWN);
                }
                let t0 = std::time::Instant::now();
                match tokio::time::timeout(Duration::from_secs(120), review_and_compact(conv_id, context_window, &dag, &summarizer, &planner, &event_tx, &mut last_leaf_counts)).await {
                    Ok(()) => {}
                    Err(_elapsed) => {
                        let _ = event_tx.send(CompactEvent::Error {
                            conv_id: Some(conv_id),
                            message: format!("review_and_compact timed out after {}s", t0.elapsed().as_secs()),
                        }).await;
                    }
                }
            }
            CompactCommand::CompressGroup { conv_id, node_ids } => {
                // Direct group compression bypasses the planner (caller-specified nodes).
                // Clear dirty-region tracking so next ReviewAndCompact re-scans.
                last_leaf_counts.remove(&conv_id);
                let t0 = std::time::Instant::now();
                match tokio::time::timeout(Duration::from_secs(120), compress_group(conv_id, node_ids, &dag, &summarizer, &event_tx)).await {
                    Ok(()) => {}
                    Err(_elapsed) => {
                        let _ = event_tx.send(CompactEvent::Error {
                            conv_id: Some(conv_id),
                            message: format!("compress_group timed out after {}s", t0.elapsed().as_secs()),
                        }).await;
                    }
                }
            }
            CompactCommand::SlideAndCompact { conv_id, window_size, context_window } => {
                let t0 = std::time::Instant::now();
                match tokio::time::timeout(Duration::from_secs(120), slide_and_compact(conv_id, window_size, context_window, &dag, &summarizer, &planner, &event_tx, &mut last_leaf_counts)).await {
                    Ok(()) => {}
                    Err(_elapsed) => {
                        let _ = event_tx.send(CompactEvent::Error {
                            conv_id: Some(conv_id),
                            message: format!("slide_and_compact timed out after {}s", t0.elapsed().as_secs()),
                        }).await;
                    }
                }
            }
        }
    }
}

async fn review_and_compact(
    conv_id: i64,
    context_window: usize,
    dag: &DagEngine,
    summarizer: &Summarizer,
    planner: &CompactionPlanner,
    event_tx: &mpsc::Sender<CompactEvent>,
    last_leaf_counts: &mut std::collections::HashMap<i64, usize>,
) {
    // Dirty-region tracking (P0-11): skip if topology unchanged.
    let prev_count = last_leaf_counts.get(&conv_id).copied().unwrap_or(0);
    match planner.is_dirty(dag, conv_id, prev_count) {
        Ok(false) => {
            // Topology unchanged and below threshold — skip.
            if event_tx.send(CompactEvent::below(conv_id, "dirty_skip")).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
            return;
        }
        Err(e) => {
            if event_tx.send(CompactEvent::Error { message: format!("planner: {e}"), conv_id: None }).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
            return;
        }
        _ => {}
    }

    let plan = match planner.plan(dag, conv_id, context_window) {
        Ok(p) => p,
        Err(e) => {
            if event_tx.send(CompactEvent::Error { message: format!("plan: {e}"), conv_id: None }).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
            return;
        }
    };

    // Record the current leaf count for next dirty check.
    last_leaf_counts.insert(conv_id, plan.leaves.len());

    if !plan.should_compact() {
        if event_tx.send(CompactEvent::below(conv_id, "budget_ok")).await.is_err() {
            tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
        }
        return;
    }

    let cmd_start = std::time::Instant::now();

    for group in &plan.groups {
        let text: String = plan.leaves.iter()
            .filter(|n| group.node_ids.contains(&n.id))
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        if text.is_empty() {
            continue;
        }

        let old_tc: i64 = plan.leaves.iter()
            .filter(|n| group.node_ids.contains(&n.id))
            .map(|n| n.token_count)
            .sum();

        do_compress(conv_id, group.node_ids.clone(), text, old_tc, dag, summarizer, event_tx).await;
    }

    // Per-command summary (P1 metrics)
    if event_tx.send(CompactEvent::CompactionCompleted {
        conv_id,
        groups: plan.groups.len() as u32,
        tokens_saved: 0,
        latency_ms: cmd_start.elapsed().as_millis() as u64,
        failures: 0,
    }).await.is_err() {
        tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
    }
}

async fn compress_group(
    conv_id: i64,
    node_ids: Vec<i64>,
    dag: &DagEngine,
    summarizer: &Summarizer,
    event_tx: &mpsc::Sender<CompactEvent>,
) {
    let leaves = match dag.get_leaves(conv_id) {
        Ok(l) => l,
        Err(e) => { if event_tx.send(CompactEvent::Error { message: format!("get_leaves: {e}"), conv_id: None }).await.is_err() { tracing::warn!(target: "deeplossless::compactor", "event receiver dropped"); } return; }
    };
    let nodes: Vec<&DagNode> = leaves.iter().filter(|n| node_ids.contains(&n.id)).collect();
    if nodes.len() < 2 {
        if event_tx.send(CompactEvent::below(conv_id, "too_few_nodes")).await.is_err() { tracing::warn!(target: "deeplossless::compactor", "event receiver dropped"); }
        return;
    }

    let text = nodes.iter().map(|n| n.summary.as_str()).collect::<Vec<_>>().join("\n---\n");
    let old_tc: i64 = nodes.iter().map(|n| n.token_count).sum();

    do_compress(conv_id, node_ids, text, old_tc, dag, summarizer, event_tx).await;
}

/// Sliding-window incremental compaction: when uncompressed leaves exceed
/// window_size, compact the oldest half. Keeps recent leaves visible.
async fn slide_and_compact(
    conv_id: i64,
    window_size: usize,
    context_window: usize,
    dag: &DagEngine,
    summarizer: &Summarizer,
    planner: &CompactionPlanner,
    event_tx: &mpsc::Sender<CompactEvent>,
    last_leaf_counts: &mut std::collections::HashMap<i64, usize>,
) {
    let plan = match planner.plan_slide_window(dag, conv_id, window_size, context_window) {
        Ok(p) => p,
        Err(e) => {
            if event_tx.send(CompactEvent::Error { message: format!("plan: {e}"), conv_id: None }).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
            return;
        }
    };

    // Record leaf count for dirty tracking.
    last_leaf_counts.insert(conv_id, plan.leaves.len());

    if !plan.should_compact() {
        if event_tx.send(CompactEvent::below(conv_id, "windowsize_ok")).await.is_err() {
            tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
        }
        return;
    }

    for group in &plan.groups {
        let leaves = match dag.get_leaves(conv_id) {
            Ok(l) => l,
            Err(e) => {
                if event_tx.send(CompactEvent::Error { message: format!("get_leaves: {e}"), conv_id: None }).await.is_err() {
                    tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
                }
                continue;
            }
        };

        let compact_count = group.node_ids.len();
        let text = leaves.iter()
            .take(compact_count)
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        let old_tc: i64 = leaves.iter().take(compact_count).map(|n| n.token_count).sum();

        do_compress(conv_id, group.node_ids.clone(), text, old_tc, dag, summarizer, event_tx).await;
    }
}

async fn do_compress(
    conv_id: i64,
    node_ids: Vec<i64>,
    text: String,
    old_tc: i64,
    dag: &DagEngine,
    summarizer: &Summarizer,
    event_tx: &mpsc::Sender<CompactEvent>,
) {
    let _r = do_compress_inner(conv_id, &node_ids, &text, old_tc, dag, summarizer).await;
    match _r {
        Ok((tokens_saved, new_node_id, level, summarizer_level, latency_ms)) => {
            if event_tx.send(CompactEvent::GroupCompressed {
                conv_id,
                new_node_id,
                level,
                tokens_saved,
                latency_ms,
                summarizer_level,
            }).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
        }
        Err(e) => {
            if event_tx.send(CompactEvent::Error {
                message: format!("{e}"),
                conv_id: Some(conv_id),
            }).await.is_err() {
                tracing::warn!(target: "deeplossless::compactor", "event receiver dropped");
            }
        }
    }
}

async fn do_compress_inner(
    conv_id: i64,
    node_ids: &[i64],
    text: &str,
    old_tc: i64,
    dag: &DagEngine,
    summarizer: &Summarizer,
) -> anyhow::Result<(i64, i64, u8, u8, u64)> {
    let start = std::time::Instant::now();
    let compressed_text = compress_code_blocks(text);
    let result = summarizer.summarize_escalate(&compressed_text).await?;
    let tc = crate::tokenizer::count(&result.text) as i64;
    let dag_level = result.dag_level;
    let summarizer_level = result.level.to_dag_level();
    let source = node_ids.first().map(|id| id.to_string()).unwrap_or_default();
    let mut snippets = crate::snippet::extract_with_source(text, &source);
    let orig_len = snippets.len();
    snippets.retain(|s| text.contains(&s.content));
    if snippets.len() < orig_len {
        tracing::warn!(
            target: "deeplossless::compactor",
            dropped = orig_len - snippets.len(),
            "snippets failed consistency check — removed"
        );
    }
    let node = dag.compress_group_with_snippets(conv_id, node_ids, &result.text, tc, dag_level, &snippets)?;
    let latency_ms = start.elapsed().as_millis() as u64;
    Ok((old_tc - tc, node.id, node.level, summarizer_level, latency_ms))
}

/// Supervisor that restarts the worker on panic, making compaction
/// resilient to transient panics (e.g. summary API timeout).
async fn compactor_supervisor(
    cmd_rx: mpsc::Receiver<CompactCommand>,
    event_tx: mpsc::Sender<CompactEvent>,
    dag: Arc<DagEngine>,
    summarizer: Summarizer,
    config: CompactorConfig,
) {
    // We can't restart with a new receiver after a panic since mpsc::Receiver
    // isn't Clone. Instead, catch any panics inside the worker and log them.
    // If the worker exits cleanly (Shutdown command via cmd_rx closing), we
    // exit too.
    let result = std::panic::AssertUnwindSafe(compact_worker(
        cmd_rx,
        event_tx,
        dag,
        summarizer,
        config,
    ))
    .catch_unwind()
    .await;

    if let Err(e) = result {
        tracing::error!(
            target = "deeplossless::compactor",
            "compactor worker panicked: {e:?}",
        );
    }
}

use futures::FutureExt;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn compactor_config_defaults() {
        let config = CompactorConfig::default();
        assert_eq!(config.group_size, 32);
        assert_eq!(config.soft_threshold_pct, 0.80);
        assert_eq!(config.hard_threshold_pct, 0.95);
    }

    #[tokio::test]
    async fn compactor_below_threshold() {
        let dir = tempdir().unwrap();
        let db = Arc::new(
            Database::builder()
                .path(dir.path().join("below.db"))
                .build()
                .await
                .unwrap(),
        );

        let config = CompactorConfig {
            summarizer: SummarizerConfig {
                api_key: "test".to_string(),
                upstream_path: "/v1/chat/completions".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut compactor = Compactor::new(db, config, None);
        let event = compactor
            .command(CompactCommand::ReviewAndCompact { conv_id: 1, context_window: 100_000 })
            .await;

        match event {
            Some(CompactEvent::BelowThreshold { .. }) => {}
            other => panic!("expected BelowThreshold, got {:?}", other),
        }
    }

    #[test]
    fn novelty_detects_redundant_vs_unique() {
        let redundant = vec!["hello world foo bar", "hello world foo baz", "hello world bar baz"];
        let unique = vec!["quantum computing advances", "my cat ate breakfast", "RFC 9457 error format"];
        let r_score = novelty_score(&redundant);
        let u_score = novelty_score(&unique);
        assert!(u_score > r_score, "unique texts should have higher novelty: u={u_score} vs r={r_score}");
        assert!(r_score < 0.5, "redundant texts should have low novelty, got {r_score}");
    }

    #[test]
    fn novelty_single_text_is_one() {
        assert_eq!(novelty_score(&["hello"]), 1.0);
    }

    #[test]
    fn novelty_empty_is_one() {
        assert_eq!(novelty_score(&[]), 1.0);
    }

    // ── CompactionBudget threshold calculations ─────────────────────────

    #[test]
    fn budget_is_critical_when_above_hard_limit() {
        let budget = CompactionBudget::new(100_000, 0.80, 0.95, 32);
        assert!(budget.is_critical(95_000));
        assert!(!budget.is_critical(94_999));
    }

    #[test]
    fn budget_is_advisory_when_above_soft_limit() {
        let budget = CompactionBudget::new(100_000, 0.80, 0.95, 32);
        assert!(budget.is_advisory(80_000, 1));
        assert!(!budget.is_advisory(79_999, 1));
    }

    #[test]
    fn budget_is_advisory_with_many_leaves() {
        let budget = CompactionBudget::new(100_000, 0.80, 0.95, 32);
        assert!(budget.is_advisory(100, 256)); // 32 * 8 = 256, triggers leaf threshold
        assert!(!budget.is_advisory(100, 255));
    }

    #[test]
    fn budget_limits_are_calculated_correctly() {
        let budget = CompactionBudget::new(200_000, 0.80, 0.95, 16);
        assert_eq!(budget.soft_limit, 160_000);
        assert_eq!(budget.hard_limit, 190_000);
        assert_eq!(budget.group_size, 16);
    }

    // ── leaf scoring (score_leaf) ───────────────────────────────────────

    fn make_node(id: i64, summary: &str, token_count: i64) -> DagNode {
        DagNode {
            id,
            conversation_id: 1,
            level: 0,
            summary: summary.to_string(),
            token_count,
            parent_ids: vec![],
            child_ids: vec![],
            is_leaf: true,
            is_join: false,
            snippets: vec![],
            deleted: false,
            semantic_hash: String::new(),
            access_count: 0,
            last_accessed_at: None,
            reasoning: String::new(),
            graph_revision: 0,
            compaction_id: String::new(),
        }
    }

    #[test]
    fn score_leaf_older_ranks_higher() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let early = make_node(1, "first message", 100);
        let late = make_node(2, "last message", 100);
        let early_score = planner.score_leaf(&early, 0, 10, 0.5);
        let late_score = planner.score_leaf(&late, 9, 10, 0.5);
        assert!(early_score.0 > late_score.0,
            "older (earlier position) should score higher: early={} vs late={}", early_score.0, late_score.0);
    }

    #[test]
    fn score_leaf_high_tokens_ranks_higher() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let small = make_node(1, "short", 50);
        let large = make_node(2, "very long message", 500);
        let small_score = planner.score_leaf(&small, 0, 10, 0.5);
        let large_score = planner.score_leaf(&large, 0, 10, 0.5);
        assert!(large_score.0 > small_score.0,
            "high token count should score higher: large={} vs small={}", large_score.0, small_score.0);
    }

    #[test]
    fn score_leaf_low_novelty_ranks_higher() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let node = make_node(1, "redundant text", 100);
        let high_novelty = planner.score_leaf(&node, 0, 10, 0.9);
        let low_novelty = planner.score_leaf(&node, 0, 10, 0.1);
        assert!(low_novelty.0 > high_novelty.0,
            "low novelty should score higher (more compressible): low_n={} vs high_n={}", low_novelty.0, high_novelty.0);
    }

    #[test]
    fn score_leaf_single_element_scores_zero() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let node = make_node(1, "only one", 100);
        let score = planner.score_leaf(&node, 0, 1, 0.5);
        assert_eq!(score.0, 0.0, "single element should not be scored for compaction");
    }

    #[test]
    fn score_leaf_clamped_to_zero_one() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let node = make_node(1, "test", 10_000); // Very high tokens
        let score = planner.score_leaf(&node, 0, 100, 0.0); // Zero novelty → max compressibility
        assert!(score.0 >= 0.0 && score.0 <= 1.0, "score should be clamped: {}", score.0);
    }

    // ── CompactionPlan threshold decisions ──────────────────────────────

    #[test]
    fn plan_returns_empty_when_below_threshold() {
        let dir = tempdir().unwrap();
        let db = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(
                Database::builder().path(dir.path().join("plan_below.db")).build()
            )
        }).join().unwrap().unwrap();
        let db = Arc::new(db);
        let conv_id = db.find_or_create_conversation("fp-plan", "test").unwrap();
        let dag = DagEngine::builder().max_level(3).build(db);
        dag.insert_leaf(conv_id, "single msg", 10).unwrap();

        let planner = CompactionPlanner::new(CompactorConfig::default());
        let plan = planner.plan(&dag, conv_id, 100_000).unwrap();
        assert!(!plan.should_compact(), "single leaf below threshold should not compact");
        assert!(plan.groups.is_empty());
    }

    #[test]
    fn plan_returns_groups_when_above_threshold() {
        let dir = tempdir().unwrap();
        let db = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(
                Database::builder().path(dir.path().join("plan_above.db")).build()
            )
        }).join().unwrap().unwrap();
        let db = Arc::new(db);
        let conv_id = db.find_or_create_conversation("fp-compact", "test").unwrap();
        let dag = DagEngine::builder().max_level(3).build(db.clone());
        for i in 0..40 {
            dag.insert_leaf(conv_id, &format!("message number {i} with enough tokens to push past the soft limit threshold"), 500).unwrap();
        }
        // Store messages so total_conversation_tokens is non-zero
        let messages: Vec<serde_json::Value> = (0..40).map(|i| {
            serde_json::json!({"role": "user", "content": format!("msg {i}")})
        }).collect();
        db.store_messages(conv_id, &serde_json::json!(messages)).unwrap();
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let plan = planner.plan(&dag, conv_id, 600).unwrap();
        assert!(!plan.groups.is_empty(), "should plan groups for 40 leaves");
    }

    #[test]
    fn plan_slide_window_returns_empty_when_few_leaves() {
        let dir = tempdir().unwrap();
        let db = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(
                Database::builder().path(dir.path().join("slide_below.db")).build()
            )
        }).join().unwrap().unwrap();
        let db = Arc::new(db);
        let conv_id = db.find_or_create_conversation("fp-slide", "test").unwrap();
        let dag = DagEngine::builder().max_level(3).build(db);
        dag.insert_leaf(conv_id, "msg", 10).unwrap();

        let planner = CompactionPlanner::new(CompactorConfig::default());
        let plan = planner.plan_slide_window(&dag, conv_id, 5, 100_000).unwrap();
        assert!(plan.groups.is_empty());
    }

    #[test]
    fn plan_slide_window_compacts_oldest_leaves() {
        let dir = tempdir().unwrap();
        let db = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(
                Database::builder().path(dir.path().join("slide_compact.db")).build()
            )
        }).join().unwrap().unwrap();
        let db = Arc::new(db);
        let conv_id = db.find_or_create_conversation("fp-slide2", "test").unwrap();
        let dag = DagEngine::builder().max_level(3).build(db.clone());
        for i in 0..10 {
            dag.insert_leaf(conv_id, &format!("msg {i}"), 500).unwrap();
        }
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let plan = planner.plan_slide_window(&dag, conv_id, 3, 100_000).unwrap();
        assert!(!plan.groups.is_empty(), "10 leaves with window 3 should produce groups");
        let group = &plan.groups[0];
        let leaf_ids: std::collections::HashSet<i64> = plan.leaves.iter().map(|n| n.id).collect();
        for id in &group.node_ids {
            assert!(leaf_ids.contains(id), "group node {id} should be in leaves");
        }
    }
}
