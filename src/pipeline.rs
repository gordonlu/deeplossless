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
/// Three-tier layout: Summaries (by level DESC) → Snippets (by importance) → Recent Messages.
pub fn render_dag_context(nodes: &[DagNode]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "<lcm_context>");

    let summaries: Vec<&DagNode> = nodes.iter().filter(|n| n.level > 0).collect();
    let mut leaves: Vec<&DagNode> = nodes.iter().filter(|n| n.is_leaf).collect();
    leaves.sort_by_key(|n| n.id);
    let all_snippets: Vec<&crate::snippet::Snippet> = nodes.iter()
        .flat_map(|n| n.snippets.iter())
        .collect();

    // ── Tier 1: Summaries (highest level first) ──
    if !summaries.is_empty() {
        let _ = writeln!(out, "  ── Summaries ──");
        let mut sorted: Vec<&DagNode> = summaries.clone();
        sorted.sort_by_key(|n| (std::cmp::Reverse(n.level), n.id));
        for node in &sorted {
            let _ = writeln!(
                out,
                "  [summary {}] L{} {} ({} tok, {} sources)",
                node.id, node.level, node.summary, node.token_count, node.child_ids.len()
            );
            if !node.child_ids.is_empty() {
                let src_ids: Vec<String> = node.child_ids.iter().map(|id| format!("msg_{id}")).collect();
                let _ = writeln!(out, "    ← sources: {}", src_ids.join(", "));
            }
        }
        if !leaves.is_empty() || !all_snippets.is_empty() {
            let _ = writeln!(out);
        }
    }

    // ── Tier 2: Snippets (by importance) ──
    if !all_snippets.is_empty() {
        let _ = writeln!(out, "  ── Snippets ──");
        let mut sorted: Vec<&crate::snippet::Snippet> = all_snippets.clone();
        sorted.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.frequency.cmp(&b.frequency))
                .then(b.content.len().cmp(&a.content.len()))
        });
        for s in sorted.iter().take(8) {
            let label = match s.snippet_type {
                crate::snippet::SnippetType::CodeBlock => "code",
                crate::snippet::SnippetType::FilePath => "path",
                crate::snippet::SnippetType::NumericConstant => "num",
                crate::snippet::SnippetType::ErrorMessage => "err",
                crate::snippet::SnippetType::ProperNoun => "ref",
            };
            if s.source_node_id.is_empty() {
                let _ = writeln!(out, "  [{label}] {}", s.content);
            } else {
                let _ = writeln!(out, "  [{label}] {} (src: msg_{})", s.content, s.source_node_id);
            }
        }
        if !leaves.is_empty() {
            let _ = writeln!(out);
        }
    }

    // ── Tier 3: Recent messages ──
    if !leaves.is_empty() {
        let _ = writeln!(out, "  ── Recent Messages ──");
        for node in &leaves {
            let truncated: String = node.summary.chars().take(80).collect();
            let _ = writeln!(out, "  [msg {}] {} ({} tok)", node.id, truncated, node.token_count);
        }
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
                // Group tool chains into execution units (Phase 1.5)
                let normalized: Vec<crate::session::NormalizedMessage> = arr
                    .iter()
                    .map(crate::session::normalize_message)
                    .collect();
                let units = crate::execution::group_execution_chain(conv_id, &normalized);
                for unit in &units {
                    if let Err(e) = db.store_execution_unit(
                        conv_id,
                        &unit.reasoning_before,
                        &unit.tool_name,
                        &unit.tool_args,
                        &unit.tool_result,
                        &unit.reasoning_after,
                        unit.outcome.as_str(),
                        &unit.related_nodes,
                    ) {
                        tracing::warn!(target: "deeplossless::pipeline", "failed to store execution unit: {e}");
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
        let query = req_body["messages"].as_array()
            .and_then(|arr| arr.iter().rev().find(|m| m["role"] == "user"))
            .and_then(|m| m["content"].as_str());
        if let Ok(dag_ctx) = self.dag.assemble_context(conv_id, 2000, query) {
            if !dag_ctx.is_empty() {
                let mut ctx_text = render_dag_context(&dag_ctx);
                // Append active file conflicts so agents can avoid stepping on each other
                if let Ok(claims) = self.db.list_all_file_claims()
                    && !claims.is_empty() {
                        // Replace closing tag, add conflicts, re-add closing tag
                        ctx_text = ctx_text.trim_end_matches("</lcm_context>\n").to_string();
                        use std::fmt::Write;
                        let _ = writeln!(ctx_text, "  ── Active File Claims ──");
                        for (agent, path, op) in &claims {
                            let _ = writeln!(ctx_text, "  [{agent}] {op}: {path}");
                        }
                        let _ = writeln!(ctx_text, "</lcm_context>");
                    }
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
