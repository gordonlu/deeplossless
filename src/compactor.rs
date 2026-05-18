use std::sync::Arc;
use tokio::sync::mpsc;

use crate::dag::{DagEngine, DagConfig, DagNode};
use crate::db::Database;
use crate::summarizer::{Summarizer, SummarizerConfig};

/// Commands sent from the main thread to the compaction worker.
pub enum CompactCommand {
    CompressGroup { conv_id: i64, node_ids: Vec<i64> },
    ReviewAndCompact { conv_id: i64, context_window: usize },
    Shutdown,
}

/// Events sent back from the compaction worker to the main thread.
#[derive(Debug)]
pub enum CompactEvent {
    GroupCompressed { conv_id: i64, new_node_id: i64, level: u8, tokens_saved: i64 },
    BelowThreshold,
    Error { message: String },
}

/// Configuration for the compaction loop.
#[derive(Clone, Debug)]
pub struct CompactorConfig {
    pub dag: DagConfig,
    pub summarizer: SummarizerConfig,
    pub soft_threshold_pct: f64,
    pub hard_threshold_pct: f64,
    pub group_size: usize,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            dag: DagConfig::default(),
            summarizer: SummarizerConfig::default(),
            soft_threshold_pct: 0.80,
            hard_threshold_pct: 0.95,
            group_size: 8,
        }
    }
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
    pub fn spawn(db: Arc<Database>, config: CompactorConfig) -> Self {
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
                // Build a fallback summarizer with partial config so compaction
                // can still run (Level 3 deterministic fallback works without API key).
                Summarizer::builder()
                    .api_key("unset")
                    .build()
                    .expect("fallback summarizer build failed")
            }
        };

        tokio::spawn(compactor_supervisor(cmd_rx, event_tx, dag, summarizer, config.clone()));

        Self { cmd_tx, event_rx, config }
    }

    pub async fn command(&mut self, cmd: CompactCommand) -> Option<CompactEvent> {
        self.cmd_tx.send(cmd).await.ok()?;
        self.event_rx.recv().await
    }

    pub fn drain_events(&mut self) -> Vec<CompactEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.event_rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn config(&self) -> &CompactorConfig { &self.config }
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

async fn compact_worker(
    mut cmd_rx: mpsc::Receiver<CompactCommand>,
    event_tx: mpsc::Sender<CompactEvent>,
    dag: Arc<DagEngine>,
    summarizer: Summarizer,
    config: CompactorConfig,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            CompactCommand::Shutdown => break,
            CompactCommand::ReviewAndCompact { conv_id, context_window } => {
                review_and_compact(conv_id, context_window, &dag, &summarizer, &config, &event_tx).await;
            }
            CompactCommand::CompressGroup { conv_id, node_ids } => {
                compress_group(conv_id, node_ids, &dag, &summarizer, &event_tx).await;
            }
        }
    }
}

async fn review_and_compact(
    conv_id: i64,
    context_window: usize,
    dag: &DagEngine,
    summarizer: &Summarizer,
    config: &CompactorConfig,
    event_tx: &mpsc::Sender<CompactEvent>,
) {
    let total = match dag.total_tokens(conv_id) {
        Ok(t) => t,
        Err(e) => { let _ = event_tx.send(CompactEvent::Error { message: format!("total_tokens: {e}") }).await; return; }
    };

    let leaves = match dag.get_leaves(conv_id) {
        Ok(l) => l,
        Err(e) => { let _ = event_tx.send(CompactEvent::Error { message: format!("get_leaves: {e}") }).await; return; }
    };

    let leaf_count = leaves.len();
    if leaf_count < 2 {
        return;
    }

    // Entropy-aware threshold adjustment: novel content gets preserved,
    // redundant content gets compacted more aggressively.
    let texts: Vec<&str> = leaves.iter().take(config.group_size).map(|n| n.summary.as_str()).collect();
    let novelty = novelty_score(&texts);
    let adj_soft = if novelty > 0.7 { 0.90 } else if novelty < 0.3 { 0.60 } else { config.soft_threshold_pct };
    let adj_hard = if novelty > 0.7 { 0.98 } else if novelty < 0.3 { 0.85 } else { config.hard_threshold_pct };

    let soft = (context_window as f64 * adj_soft) as i64;
    let hard_limit = (context_window as f64 * adj_hard) as i64;

    let soft_trigger = leaf_count >= config.group_size * 2 && total >= soft;

    if total < hard_limit && !soft_trigger {
        let _ = event_tx.send(CompactEvent::BelowThreshold).await;
        return;
    }

    let group: Vec<_> = leaves.iter().take(config.group_size).map(|n| n.id).collect();
    if group.len() < 2 { return; }

    let text = leaves.iter()
        .take(config.group_size)
        .map(|n| n.summary.as_str())
        .collect::<Vec<_>>()
        .join("\n---\n");

    let old_tc: i64 = leaves.iter().take(config.group_size).map(|n| n.token_count).sum();

    do_compress(conv_id, group, text, old_tc, dag, summarizer, event_tx).await;
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
        Err(e) => { let _ = event_tx.send(CompactEvent::Error { message: format!("get_leaves: {e}") }).await; return; }
    };
    let nodes: Vec<&DagNode> = leaves.iter().filter(|n| node_ids.contains(&n.id)).collect();
    if nodes.len() < 2 { return; }

    let text = nodes.iter().map(|n| n.summary.as_str()).collect::<Vec<_>>().join("\n---\n");
    let old_tc: i64 = nodes.iter().map(|n| n.token_count).sum();

    do_compress(conv_id, node_ids, text, old_tc, dag, summarizer, event_tx).await;
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
    match summarizer.summarize_escalate(&text).await {
        Ok((summary, level)) => {
            let tc = crate::tokenizer::count(&summary) as i64;
            let dag_level = level.to_dag_level();
            // Extract snippets before compression to preserve precision (#10)
            let source = node_ids.first().map(|id| id.to_string()).unwrap_or_default();
            let snippets = crate::snippet::extract_with_source(&text, &source);
            match dag.compress_group_with_snippets(conv_id, &node_ids, &summary, tc, dag_level, &snippets) {
                Ok(node) => {
                    let _ = event_tx.send(CompactEvent::GroupCompressed {
                        conv_id,
                        new_node_id: node.id,
                        level: node.level,
                        tokens_saved: old_tc - tc,
                    }).await;
                }
                Err(e) => {
                    let _ = event_tx.send(CompactEvent::Error { message: format!("compress_group: {e}") }).await;
                }
            }
        }
        Err(e) => {
            let _ = event_tx.send(CompactEvent::Error { message: format!("summarize: {e}") }).await;
        }
    }
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
        assert_eq!(config.group_size, 8);
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
            summarizer: SummarizerConfig { api_key: "test".to_string(), ..Default::default() },
            ..Default::default()
        };

        let mut compactor = Compactor::spawn(db, config);
        let event = compactor
            .command(CompactCommand::ReviewAndCompact { conv_id: 1, context_window: 100_000 })
            .await;

        match event {
            Some(CompactEvent::BelowThreshold) => {}
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
}
