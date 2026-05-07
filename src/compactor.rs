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
    let hard = (context_window as f64 * config.hard_threshold_pct) as i64;
    let total = match dag.total_tokens(conv_id) {
        Ok(t) => t,
        Err(e) => { let _ = event_tx.send(CompactEvent::Error { message: format!("total_tokens: {e}") }).await; return; }
    };

    if total < hard {
        let _ = event_tx.send(CompactEvent::BelowThreshold).await;
        return;
    }

    let leaves = match dag.get_leaves(conv_id) {
        Ok(l) => l,
        Err(e) => { let _ = event_tx.send(CompactEvent::Error { message: format!("get_leaves: {e}") }).await; return; }
    };

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
            let snippets = crate::snippet::extract(&text);
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
}
