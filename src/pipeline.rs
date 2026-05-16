use std::sync::Arc;
use tokio::sync::Mutex;

use crate::compactor::{CompactCommand, CompactEvent, Compactor};
use crate::dag::{DagEngine, DagNode};
use crate::db::Database;
use crate::AppState;

/// Context window size for threshold calculations.
const CONTEXT_WINDOW: usize = 1_000_000;

/// Result of processing a chat completion request through the pipeline.
pub struct PipelineOutput {
    pub conv_id: i64,
    pub injected_body: serde_json::Value,
}

/// Render DAG context nodes into a structured, actionable context panel.
/// Each node shows its summary, snippets, and available operations.
pub fn render_dag_context(nodes: &[DagNode]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "<lcm_context>");

    let ids: Vec<i64> = nodes.iter().map(|n| n.id).collect();

    for node in nodes {
        let header = if node.is_leaf {
            format!("[msg {}] {} ({} tok)", node.id, node.summary, node.token_count)
        } else {
            format!(
                "[summary {}] L{} — {} ({} tok, {} parents)",
                node.id,
                node.level,
                node.summary,
                node.token_count,
                node.parent_ids.len()
            )
        };
        let _ = writeln!(out, "  {header}");

        for s in &node.snippets {
            let label = match s.snippet_type {
                crate::snippet::SnippetType::CodeBlock => "code",
                crate::snippet::SnippetType::FilePath => "path",
                crate::snippet::SnippetType::NumericConstant => "num",
                crate::snippet::SnippetType::ErrorMessage => "err",
                crate::snippet::SnippetType::ProperNoun => "ref",
            };
            let _ = writeln!(out, "    ├ {label}: {}", s.content);
        }

        if node.level > 0 {
            let _ = writeln!(out, "    └ /lcm/rollback {id}", id = node.id);
        }
    }

    if ids.len() >= 2 {
        let first = ids[0];
        let last = ids[ids.len() - 1];
        let _ = writeln!(out);
        let _ = writeln!(out, "  Operations:");
        let _ = writeln!(out, "    /lcm/compress conv_id={cid} from={first} to={last}", cid = nodes[0].conversation_id);
        let _ = writeln!(out, "    /lcm/delete conv_id={cid} id=<node_id>", cid = nodes[0].conversation_id);
        let _ = writeln!(out, "    /lcm/rollback conv_id={cid} id=<summary_id>", cid = nodes[0].conversation_id);
    }

    let _ = writeln!(out, "</lcm_context>");
    out
}

/// ChatPipeline: orchestrates the request processing steps
/// (fingerprint → store → compact → assemble → inject) without
/// coupling to HTTP transport.
pub struct ChatPipeline {
    db: Arc<Database>,
    dag: Arc<DagEngine>,
    compactor: Arc<Mutex<Compactor>>,
}

impl ChatPipeline {
    pub fn new(state: &AppState) -> Self {
        Self {
            db: state.db.clone(),
            dag: state.dag.clone(),
            compactor: state.compactor.clone(),
        }
    }

    /// Run the full pipeline: resolve conversation, persist messages,
    /// trigger compaction, assemble DAG context, inject into system messages.
    pub async fn process(
        &self,
        model: &str,
        req_body: &serde_json::Value,
    ) -> anyhow::Result<PipelineOutput> {
        let messages = &req_body["messages"];
        let msgs_arr = messages.as_array().map(|a| a.as_slice()).unwrap_or(&[]);

        // Resolve conversation via fingerprint
        let fp = crate::session::fingerprint(msgs_arr, 3);
        let conv_id = self.db.find_or_create_conversation(&fp, model)?;

        // Store messages and create DAG leaf nodes (async, non-blocking)
        let db = self.db.clone();
        let dag = self.dag.clone();
        let overhead = dag.config().token_overhead;
        let correction = dag.config().token_correction_factor;
        let msgs = messages.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = db.store_messages(conv_id, &msgs) {
                tracing::warn!(target: "deeplossless::pipeline", "failed to store messages: {e}");
                return;
            }
            if let Some(arr) = msgs.as_array() {
                for msg in arr {
                    let role = msg["role"].as_str().unwrap_or("");
                    if role == "user" || role == "assistant" {
                        let raw_tokens = crate::tokenizer::count_content_raw(msg) + overhead;
                        let tc = crate::tokenizer::correct(raw_tokens, correction) as i64;
                        let summary = msg["content"].to_string().chars().take(200).collect::<String>();
                        if let Err(e) = dag.insert_leaf(conv_id, &summary, tc) {
                            tracing::warn!(target: "deeplossless::pipeline", "failed to create DAG leaf: {e}");
                        }
                    }
                }
            }
        });

        // Trigger async compaction review (soft threshold)
        let mut compactor = self.compactor.lock().await;
        let _ = compactor
            .command(CompactCommand::ReviewAndCompact {
                conv_id,
                context_window: CONTEXT_WINDOW,
            })
            .await;
        for event in compactor.drain_events() {
            match event {
                CompactEvent::GroupCompressed { tokens_saved, .. } => {
                    tracing::debug!(target: "deeplossless::pipeline", conv_id, tokens_saved, "compaction completed");
                }
                CompactEvent::BelowThreshold => {}
                CompactEvent::Error { message } => {
                    tracing::warn!(target: "deeplossless::pipeline", conv_id, error = %message, "compaction error");
                }
            }
        }
        drop(compactor);

        // Assemble DAG context and inject into system messages
        let mut injected = req_body.clone();
        if let Ok(dag_ctx) = self.dag.assemble_context(conv_id, 2000) {
            if !dag_ctx.is_empty() {
                let ctx_text = render_dag_context(&dag_ctx);
                Self::inject_context(&mut injected, &ctx_text);
            }
        } else {
            tracing::debug!(target: "deeplossless::pipeline", conv_id, "DAG context assembly failed");
        }

        Ok(PipelineOutput { conv_id, injected_body: injected })
    }

    /// Inject `<lcm_context>` block only into the first system message
    /// to avoid prompt drift from repeated injection.
    /// `ctx_text` is the pre-rendered block from `render_dag_context`,
    /// which already includes the `<lcm_context>`...`</lcm_context>` tags.
    fn inject_context(body: &mut serde_json::Value, ctx_text: &str) {
        if let Some(arr) = body["messages"].as_array_mut() {
            for msg in arr.iter_mut() {
                if msg["role"] == "system" {
                    let existing = msg["content"].as_str().unwrap_or("");
                    msg["content"] = serde_json::json!(
                        format!("{}\n\n{}", existing, ctx_text)
                    );
                    break;
                }
            }
        }
    }
}
