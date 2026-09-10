use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use tokio::sync::mpsc;

use crate::dag::{DagConfig, DagEngine, DagNode};
use crate::db::Database;
use crate::ground_truth::{SourceRef, TruthStream};
use crate::ground_truth_store::GroundTruthStore;
use crate::summarizer::{Summarizer, SummarizerConfig};

/// Commands sent from the main thread to the compaction worker.
pub enum CompactCommand {
    CompressGroup {
        conv_id: i64,
        node_ids: Vec<i64>,
    },
    ReviewAndCompact {
        conv_id: i64,
        context_window: usize,
    },
    SlideAndCompact {
        conv_id: i64,
        window_size: usize,
        context_window: usize,
    },
    Ping,
    Shutdown,
}

/// Events sent back from the compaction worker to the main thread.
#[derive(Debug)]
pub enum CompactEvent {
    GroupCompressed {
        conv_id: i64,
        new_node_id: i64,
        level: u8,
        tokens_saved: i64,
        latency_ms: u64,
        summarizer_level: u8,
    },
    CompactionCompleted {
        conv_id: i64,
        groups: u32,
        tokens_saved: i64,
        latency_ms: u64,
        failures: u32,
    },
    BelowThreshold {
        conv_id: i64,
        reason: &'static str,
    },
    Error {
        message: String,
        conv_id: Option<i64>,
    },
    Pong,
}

impl CompactEvent {
    pub fn below(conv_id: i64, reason: &'static str) -> Self {
        Self::BelowThreshold { conv_id, reason }
    }
}

#[derive(Clone, Debug)]
pub struct CompactorConfig {
    pub dag: DagConfig,
    pub summarizer: SummarizerConfig,
    pub soft_threshold_pct: f64,
    pub hard_threshold_pct: f64,
    pub group_size: usize,
    pub age_weight: f64,
    pub token_density_weight: f64,
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

    pub fn is_critical(&self, total_tokens: i64) -> bool {
        total_tokens >= self.hard_limit
    }

    pub fn is_advisory(&self, total_tokens: i64, leaf_count: usize) -> bool {
        total_tokens >= self.soft_limit || leaf_count >= self.group_size * 8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CompactionScore(pub f64);

#[derive(Debug, Clone)]
pub struct CompactionGroup {
    pub node_ids: Vec<i64>,
    pub score: CompactionScore,
    pub strategy: CompactionStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    Summarize,
    SlideWindow { window_size: usize },
    Merge,
}

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
        self.budget.is_critical(self.total_tokens)
            || self
                .budget
                .is_advisory(self.total_tokens, self.leaves.len())
    }
}

#[derive(Debug, Clone)]
pub struct CompactionPlanner {
    config: CompactorConfig,
}

impl CompactionPlanner {
    pub fn new(config: CompactorConfig) -> Self {
        Self { config }
    }

    pub fn plan(
        &self,
        dag: &DagEngine,
        conv_id: i64,
        context_window: usize,
    ) -> anyhow::Result<CompactionPlan> {
        let budget = CompactionBudget::new(
            context_window,
            self.config.soft_threshold_pct,
            self.config.hard_threshold_pct,
            self.config.group_size,
        );
        let total_tokens = dag.total_tokens(conv_id)?;
        let leaves = dag.get_leaves(conv_id)?;
        let leaf_count = leaves.len();
        if leaf_count < 2
            || (!budget.is_critical(total_tokens) && !budget.is_advisory(total_tokens, leaf_count))
        {
            return Ok(CompactionPlan {
                conv_id,
                budget,
                groups: Vec::new(),
                leaves,
                total_tokens,
            });
        }
        let groups = self.build_groups(&leaves, &budget);
        Ok(CompactionPlan {
            conv_id,
            budget,
            groups,
            leaves,
            total_tokens,
        })
    }

    pub fn is_dirty(
        &self,
        dag: &DagEngine,
        conv_id: i64,
        last_leaf_count: usize,
    ) -> anyhow::Result<bool> {
        let current = dag.get_leaves(conv_id)?.len();
        Ok(current != last_leaf_count || current >= self.config.group_size * 2)
    }

    fn build_groups(&self, leaves: &[DagNode], budget: &CompactionBudget) -> Vec<CompactionGroup> {
        let texts: Vec<&str> = leaves.iter().map(|n| n.summary.as_str()).collect();
        let novelty = novelty_score(&texts);
        let mut scored: Vec<(i64, CompactionScore)> = leaves
            .iter()
            .enumerate()
            .map(|(position, node)| {
                (
                    node.id,
                    self.score_leaf(node, position, leaves.len(), novelty),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

    fn score_leaf(
        &self,
        node: &DagNode,
        position: usize,
        total: usize,
        novelty: f64,
    ) -> CompactionScore {
        if total <= 1 {
            return CompactionScore(0.0);
        }
        let age_factor = 1.0 - position as f64 / total as f64;
        let token_factor = (node.token_count as f64 / 500.0).min(1.0);
        let novelty_factor = 1.0 - novelty;
        let raw = self.config.age_weight * age_factor
            + self.config.token_density_weight * token_factor
            + self.config.novelty_weight * novelty_factor;
        CompactionScore(raw.clamp(0.0, 1.0))
    }

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
        if leaf_count <= window_size
            || (total_tokens < budget.hard_limit && leaf_count < window_size * 2)
        {
            return Ok(CompactionPlan {
                conv_id,
                budget,
                groups: Vec::new(),
                leaves,
                total_tokens,
            });
        }
        let compact_count = (leaf_count - window_size).min(8);
        let node_ids: Vec<i64> = leaves.iter().take(compact_count).map(|n| n.id).collect();
        if node_ids.len() < 2 {
            return Ok(CompactionPlan {
                conv_id,
                budget,
                groups: Vec::new(),
                leaves,
                total_tokens,
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
            leaves,
            total_tokens,
        })
    }
}

pub struct Compactor {
    cmd_tx: mpsc::Sender<CompactCommand>,
    event_rx: mpsc::Receiver<CompactEvent>,
    config: CompactorConfig,
}

impl Compactor {
    pub fn new(
        db: Arc<Database>,
        config: CompactorConfig,
        tasks: Option<&Arc<crate::runtime::BackgroundTasks>>,
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
            Err(error) => {
                tracing::error!(target:"deeplossless::compactor", %error, "summarizer build failed");
                Summarizer::builder()
                    .offline_fallback_only()
                    .fallback_max_tokens(config.summarizer.fallback_max_tokens)
                    .build()
                    .expect("fallback summarizer build failed")
            }
        };
        let worker = tokio::spawn(compactor_supervisor(
            cmd_rx,
            event_tx,
            dag,
            summarizer,
            config.clone(),
        ));
        if let Some(tasks) = tasks {
            tasks.register_handle(worker);
        }
        Self {
            cmd_tx,
            event_rx,
            config,
        }
    }

    pub async fn command(&mut self, cmd: CompactCommand) -> Option<CompactEvent> {
        self.cmd_tx.send(cmd).await.ok()?;
        self.event_rx.recv().await
    }

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

    pub fn config(&self) -> &CompactorConfig {
        &self.config
    }

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

fn novelty_score(texts: &[&str]) -> f64 {
    if texts.len() < 2 {
        return 1.0;
    }
    let mut all_trigrams = Vec::<HashSet<[u8; 3]>>::new();
    for text in texts {
        let trigrams: HashSet<[u8; 3]> = text
            .as_bytes()
            .windows(3)
            .map(|w| [w[0], w[1], w[2]])
            .collect();
        if !trigrams.is_empty() {
            all_trigrams.push(trigrams);
        }
    }
    if all_trigrams.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut pairs = 0usize;
    for i in 0..all_trigrams.len() {
        for j in i + 1..all_trigrams.len() {
            let intersection = all_trigrams[i].intersection(&all_trigrams[j]).count();
            let union = all_trigrams[i].union(&all_trigrams[j]).count();
            if union > 0 {
                total += 1.0 - intersection as f64 / union as f64;
                pairs += 1;
            }
        }
    }
    if pairs == 0 {
        1.0
    } else {
        total / pairs as f64
    }
}

fn compress_code_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_code = false;
    let mut code_buf = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                result.push_str("```\n");
                result.push_str(&compress_code(&code_buf));
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
    if !code_buf.is_empty() {
        result.push_str(&code_buf);
    }
    result
}

fn compress_code(code: &str) -> String {
    let mut out = String::new();
    let mut skip_until_close = false;
    let mut indent_depth = 0usize;
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        let opens = trimmed.matches('{').count();
        let closes = trimmed.matches('}').count();
        indent_depth = indent_depth.saturating_add(opens).saturating_sub(closes);
        let structural = trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("mod ")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub async fn ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("func ")
            || trimmed.starts_with("import ");
        if structural {
            out.push_str(line);
            out.push('\n');
            skip_until_close = opens > 0;
        } else if skip_until_close {
            if closes > 0 && indent_depth <= 1 {
                skip_until_close = false;
                out.push_str("    // ...\n");
            }
        } else if indent_depth > 0 {
            out.push_str(line);
            out.push('\n');
            skip_until_close = true;
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
    let planner = CompactionPlanner::new(config);
    let mut last_leaf_counts = HashMap::<i64, usize>::new();
    let mut last_compacted = HashMap::<i64, std::time::Instant>::new();
    const COOLDOWN: Duration = Duration::from_secs(30);
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            CompactCommand::Shutdown => break,
            CompactCommand::Ping => {
                let _ = event_tx.send(CompactEvent::Pong).await;
            }
            CompactCommand::ReviewAndCompact {
                conv_id,
                context_window,
            } => {
                if last_compacted
                    .get(&conv_id)
                    .is_some_and(|t| t.elapsed() < COOLDOWN)
                {
                    continue;
                }
                last_compacted.insert(conv_id, std::time::Instant::now());
                let future = review_and_compact(
                    conv_id,
                    context_window,
                    &dag,
                    &summarizer,
                    &planner,
                    &event_tx,
                    &mut last_leaf_counts,
                );
                if tokio::time::timeout(Duration::from_secs(120), future)
                    .await
                    .is_err()
                {
                    let _ = event_tx
                        .send(CompactEvent::Error {
                            message: "review_and_compact timed out after 120s".into(),
                            conv_id: Some(conv_id),
                        })
                        .await;
                }
            }
            CompactCommand::CompressGroup { conv_id, node_ids } => {
                last_leaf_counts.remove(&conv_id);
                let future = compress_group(conv_id, node_ids, &dag, &summarizer, &event_tx);
                if tokio::time::timeout(Duration::from_secs(120), future)
                    .await
                    .is_err()
                {
                    let _ = event_tx
                        .send(CompactEvent::Error {
                            message: "compress_group timed out after 120s".into(),
                            conv_id: Some(conv_id),
                        })
                        .await;
                }
            }
            CompactCommand::SlideAndCompact {
                conv_id,
                window_size,
                context_window,
            } => {
                let future = slide_and_compact(
                    conv_id,
                    window_size,
                    context_window,
                    &dag,
                    &summarizer,
                    &planner,
                    &event_tx,
                    &mut last_leaf_counts,
                );
                if tokio::time::timeout(Duration::from_secs(120), future)
                    .await
                    .is_err()
                {
                    let _ = event_tx
                        .send(CompactEvent::Error {
                            message: "slide_and_compact timed out after 120s".into(),
                            conv_id: Some(conv_id),
                        })
                        .await;
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
    last_leaf_counts: &mut HashMap<i64, usize>,
) {
    let prev = last_leaf_counts.get(&conv_id).copied().unwrap_or(0);
    match planner.is_dirty(dag, conv_id, prev) {
        Ok(false) => {
            let _ = event_tx
                .send(CompactEvent::below(conv_id, "dirty_skip"))
                .await;
            return;
        }
        Err(error) => {
            let _ = event_tx
                .send(CompactEvent::Error {
                    message: format!("planner: {error}"),
                    conv_id: Some(conv_id),
                })
                .await;
            return;
        }
        Ok(true) => {}
    }
    let plan = match planner.plan(dag, conv_id, context_window) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = event_tx
                .send(CompactEvent::Error {
                    message: format!("plan: {error}"),
                    conv_id: Some(conv_id),
                })
                .await;
            return;
        }
    };
    last_leaf_counts.insert(conv_id, plan.leaves.len());
    if !plan.should_compact() {
        let _ = event_tx
            .send(CompactEvent::below(conv_id, "budget_ok"))
            .await;
        return;
    }
    let start = std::time::Instant::now();
    let mut groups = 0u32;
    let mut tokens_saved = 0i64;
    let mut failures = 0u32;
    for group in &plan.groups {
        let selected: Vec<&DagNode> = plan
            .leaves
            .iter()
            .filter(|leaf| group.node_ids.contains(&leaf.id))
            .collect();
        let text = selected
            .iter()
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        let old_tc = selected.iter().map(|n| n.token_count).sum();
        if text.is_empty() {
            continue;
        }
        match do_compress_inner(conv_id, &group.node_ids, &text, old_tc, dag, summarizer).await {
            Ok((saved, node_id, level, summarizer_level, latency_ms)) => {
                groups += 1;
                tokens_saved += saved;
                let _ = event_tx
                    .send(CompactEvent::GroupCompressed {
                        conv_id,
                        new_node_id: node_id,
                        level,
                        tokens_saved: saved,
                        latency_ms,
                        summarizer_level,
                    })
                    .await;
            }
            Err(error) => {
                failures += 1;
                let _ = event_tx
                    .send(CompactEvent::Error {
                        message: error.to_string(),
                        conv_id: Some(conv_id),
                    })
                    .await;
            }
        }
    }
    let _ = event_tx
        .send(CompactEvent::CompactionCompleted {
            conv_id,
            groups,
            tokens_saved,
            latency_ms: start.elapsed().as_millis() as u64,
            failures,
        })
        .await;
}

async fn compress_group(
    conv_id: i64,
    node_ids: Vec<i64>,
    dag: &DagEngine,
    summarizer: &Summarizer,
    event_tx: &mpsc::Sender<CompactEvent>,
) {
    let leaves = match dag.get_leaves(conv_id) {
        Ok(leaves) => leaves,
        Err(error) => {
            let _ = event_tx
                .send(CompactEvent::Error {
                    message: format!("get_leaves: {error}"),
                    conv_id: Some(conv_id),
                })
                .await;
            return;
        }
    };
    let selected: Vec<&DagNode> = leaves.iter().filter(|n| node_ids.contains(&n.id)).collect();
    if selected.len() < 2 {
        let _ = event_tx
            .send(CompactEvent::below(conv_id, "too_few_nodes"))
            .await;
        return;
    }
    let text = selected
        .iter()
        .map(|n| n.summary.as_str())
        .collect::<Vec<_>>()
        .join("\n---\n");
    let old_tc = selected.iter().map(|n| n.token_count).sum();
    emit_compression_result(
        conv_id,
        do_compress_inner(conv_id, &node_ids, &text, old_tc, dag, summarizer).await,
        event_tx,
    )
    .await;
}

async fn slide_and_compact(
    conv_id: i64,
    window_size: usize,
    context_window: usize,
    dag: &DagEngine,
    summarizer: &Summarizer,
    planner: &CompactionPlanner,
    event_tx: &mpsc::Sender<CompactEvent>,
    last_leaf_counts: &mut HashMap<i64, usize>,
) {
    let plan = match planner.plan_slide_window(dag, conv_id, window_size, context_window) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = event_tx
                .send(CompactEvent::Error {
                    message: format!("plan: {error}"),
                    conv_id: Some(conv_id),
                })
                .await;
            return;
        }
    };
    last_leaf_counts.insert(conv_id, plan.leaves.len());
    if !plan.should_compact() {
        let _ = event_tx
            .send(CompactEvent::below(conv_id, "windowsize_ok"))
            .await;
        return;
    }
    for group in &plan.groups {
        let leaves = match dag.get_leaves(conv_id) {
            Ok(leaves) => leaves,
            Err(error) => {
                let _ = event_tx
                    .send(CompactEvent::Error {
                        message: format!("get_leaves: {error}"),
                        conv_id: Some(conv_id),
                    })
                    .await;
                continue;
            }
        };
        let selected: Vec<&DagNode> = leaves
            .iter()
            .filter(|leaf| group.node_ids.contains(&leaf.id))
            .collect();
        let text = selected
            .iter()
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        let old_tc = selected.iter().map(|n| n.token_count).sum();
        emit_compression_result(
            conv_id,
            do_compress_inner(conv_id, &group.node_ids, &text, old_tc, dag, summarizer).await,
            event_tx,
        )
        .await;
    }
}

async fn emit_compression_result(
    conv_id: i64,
    result: anyhow::Result<(i64, i64, u8, u8, u64)>,
    event_tx: &mpsc::Sender<CompactEvent>,
) {
    match result {
        Ok((tokens_saved, new_node_id, level, summarizer_level, latency_ms)) => {
            let _ = event_tx
                .send(CompactEvent::GroupCompressed {
                    conv_id,
                    new_node_id,
                    level,
                    tokens_saved,
                    latency_ms,
                    summarizer_level,
                })
                .await;
        }
        Err(error) => {
            let _ = event_tx
                .send(CompactEvent::Error {
                    message: error.to_string(),
                    conv_id: Some(conv_id),
                })
                .await;
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
    if node_ids.len() < 2 {
        anyhow::bail!("compaction requires at least two source nodes");
    }

    // Closed-loop phase 0: pin exact source state before any lossy transform.
    let mut source_hashes = Vec::with_capacity(node_ids.len());
    for id in node_ids {
        let node = dag
            .get_node(*id)?
            .ok_or_else(|| anyhow::anyhow!("compaction source node {id} is missing"))?;
        source_hashes.push((node.id, node.semantic_hash));
    }
    let truth = GroundTruthStore::new(dag.db());
    let exact_source = truth.put_text(
        TruthStream::Named("compaction-source".into()),
        &format!("compaction:{conv_id}"),
        text,
        "compaction-input",
    )?;

    let compressed_text = compress_code_blocks(text);
    let result = summarizer.summarize_escalate(&compressed_text).await?;
    let tc = crate::tokenizer::count(&result.text) as i64;
    let dag_level = result.dag_level;
    let summarizer_level = result.level.to_dag_level();
    let source = node_ids.first().map(i64::to_string).unwrap_or_default();
    let mut snippets = crate::snippet::extract_with_source(text, &source);
    snippets.retain(|snippet| text.contains(&snippet.content));

    let node = dag.compress_group_with_snippets(
        conv_id,
        node_ids,
        &result.text,
        tc,
        dag_level,
        &snippets,
    )?;

    if let Err(error) =
        validate_committed_compaction(dag, node_ids, &source_hashes, &node, text, &exact_source)
    {
        let _ = dag.db().purge_dag_node(node.id);
        return Err(error);
    }

    // Attach the exact source landmark to the summary projection. This is not
    // the source of truth; it is a materialization pointer back to Ground Truth.
    let reasoning = serde_json::json!({
        "action": "compress",
        "source_count": node_ids.len(),
        "exact_source": exact_source,
        "closed_loop_validated": true,
    });
    dag.db()
        .update_node_reasoning(node.id, &reasoning.to_string())?;

    Ok((
        old_tc - tc,
        node.id,
        node.level,
        summarizer_level,
        start.elapsed().as_millis() as u64,
    ))
}

fn validate_committed_compaction(
    dag: &DagEngine,
    source_ids: &[i64],
    source_hashes: &[(i64, String)],
    summary: &DagNode,
    original_text: &str,
    exact_source: &SourceRef,
) -> anyhow::Result<()> {
    let expected: HashSet<i64> = source_ids.iter().copied().collect();
    let actual: HashSet<i64> = summary.child_ids.iter().copied().collect();
    if expected != actual {
        anyhow::bail!(
            "closed-loop compaction rejected: source coverage mismatch expected={expected:?} actual={actual:?}"
        );
    }

    for (source_id, expected_hash) in source_hashes {
        let source = dag.get_node(*source_id)?.ok_or_else(|| {
            anyhow::anyhow!("closed-loop compaction rejected: source node {source_id} disappeared")
        })?;
        if source.semantic_hash != *expected_hash {
            anyhow::bail!(
                "closed-loop compaction rejected: source node {source_id} changed during compaction"
            );
        }
    }

    for snippet in &summary.snippets {
        if !original_text.contains(&snippet.content) {
            anyhow::bail!(
                "closed-loop compaction rejected: precision snippet is not source-backed"
            );
        }
    }

    let truth = GroundTruthStore::new(dag.db());
    let materialized = truth.materialize_text(exact_source)?;
    if materialized.as_bytes() != original_text.as_bytes() {
        anyhow::bail!("closed-loop compaction rejected: exact source round-trip mismatch");
    }
    Ok(())
}

async fn compactor_supervisor(
    cmd_rx: mpsc::Receiver<CompactCommand>,
    event_tx: mpsc::Sender<CompactEvent>,
    dag: Arc<DagEngine>,
    summarizer: Summarizer,
    config: CompactorConfig,
) {
    let result =
        std::panic::AssertUnwindSafe(compact_worker(cmd_rx, event_tx, dag, summarizer, config))
            .catch_unwind()
            .await;
    if let Err(error) = result {
        tracing::error!(target:"deeplossless::compactor", ?error, "compactor worker panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
    fn compactor_config_defaults() {
        let config = CompactorConfig::default();
        assert_eq!(config.group_size, 32);
        assert_eq!(config.soft_threshold_pct, 0.80);
        assert_eq!(config.hard_threshold_pct, 0.95);
    }

    #[test]
    fn novelty_detects_redundant_vs_unique() {
        let redundant = [
            "hello world foo bar",
            "hello world foo baz",
            "hello world bar baz",
        ];
        let unique = [
            "quantum computing advances",
            "my cat ate breakfast",
            "RFC 9457 error format",
        ];
        assert!(novelty_score(&unique) > novelty_score(&redundant));
    }

    #[test]
    fn score_leaf_older_ranks_higher() {
        let planner = CompactionPlanner::new(CompactorConfig::default());
        let node = make_node(1, "message", 100);
        assert!(planner.score_leaf(&node, 0, 10, 0.5).0 > planner.score_leaf(&node, 9, 10, 0.5).0);
    }

    #[tokio::test]
    async fn exact_compaction_source_round_trips() {
        let dir = tempdir().unwrap();
        let db = Arc::new(
            Database::builder()
                .path(dir.path().join("compact-truth.db"))
                .build()
                .await
                .unwrap(),
        );
        let conv_id = db
            .find_or_create_conversation("compaction:test", "test")
            .unwrap();
        let dag = DagEngine::builder().max_level(3).build(db.clone());
        let a = dag.insert_leaf(conv_id, "alpha exact", 10).unwrap();
        let b = dag.insert_leaf(conv_id, "beta exact", 10).unwrap();
        let text = "alpha exact\n---\nbeta exact";
        let source = GroundTruthStore::new(&db)
            .put_text(
                TruthStream::Named("compaction-source".into()),
                &format!("compaction:{conv_id}"),
                text,
                "test",
            )
            .unwrap();
        let summary = dag
            .compress_group_with_snippets(conv_id, &[a.id, b.id], "alpha beta", 2, 1, &[])
            .unwrap();
        let hashes = vec![
            (a.id, a.semantic_hash.clone()),
            (b.id, b.semantic_hash.clone()),
        ];
        validate_committed_compaction(&dag, &[a.id, b.id], &hashes, &summary, text, &source)
            .unwrap();
    }
}
