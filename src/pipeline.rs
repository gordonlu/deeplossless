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
    lcm_context: bool,
}

impl ChatPipeline {
    pub fn new(state: &AppState) -> Self {
        Self {
            db: state.storage.db.clone(),
            dag: state.storage.dag.clone(),
            compactor: state.compactor.clone(),
            lcm_context: state.lcm_context,
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
            // Generate a unique replay session ID for this pipeline run
            let replay_session_id = {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                format!("rs_{}_{:x}", conv_id, now.as_nanos())
            };
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

                // ── Parallel execution detection ──────────────────────
                // Scan normalized messages for multi-tool turns and build
                // ForkJoinTrackers. Each turn with ≥2 tool calls becomes a
                // parallel group. Match units to groups by tool_call_id.
                let root_span = crate::parallel::ExecutionSpan::root_span();
                let mut active_trackers: Vec<crate::parallel::ForkJoinTracker> = Vec::new();
                // Stores the final HappensBefore edges for DB insertion
                let mut pending_hb_edges: Vec<crate::parallel::HappensBeforeEdge> = Vec::new();
                let mut next_turn_index: usize = 0;

                for msg in &normalized {
                    if let Some(tc_info) = crate::parallel::ParallelDetector::detect(msg) {
                        let tracker = crate::parallel::ForkJoinTracker::fork(
                            conv_id, next_turn_index, &root_span, &tc_info,
                            crate::parallel::ParallelGovernance::default(),
                        );
                        active_trackers.push(tracker);
                    }
                    if msg.role == "assistant" && !msg.tool_calls.is_empty() {
                        next_turn_index += 1;
                    }
                }

                // Process units with parallel awareness
                // Track the last unit ID to populate DependsOn lineage edges
                let mut last_exec_id: Option<i64> = None;
                for unit in &units {
                    // Find which active tracker this unit belongs to (if any)
                    let (span_id, parent_span_id, span_mode, parallel_group) =
                        if !unit.tool_call_id.is_empty() {
                            active_trackers
                                .iter()
                                .find(|t| t.branches.iter().any(|b| b.tool_call_id == unit.tool_call_id))
                                .map(|t| {
                                    let branch = t.branches.iter().find(|b| b.tool_call_id == unit.tool_call_id);
                                    match branch {
                                        Some(b) => (
                                            b.span_id.as_str().to_string(),
                                            t.parent_span.span_id.as_str().to_string(),
                                            crate::parallel::SpanMode::Parallel.as_str().to_string(),
                                            t.group_id.clone(),
                                        ),
                                        None => (String::new(), String::new(), String::new(), String::new()),
                                    }
                                })
                                .unwrap_or_default()
                        } else {
                            (String::new(), String::new(), String::new(), String::new())
                        };

                    let exec_id = match db.store_execution_unit_with_span(
                        conv_id,
                        &unit.reasoning_before,
                        &unit.tool_name,
                        &unit.tool_args,
                        &unit.tool_result,
                        &unit.reasoning_after,
                        unit.outcome.as_str(),
                        &unit.related_nodes,
                        &span_id,
                        &parent_span_id,
                        &span_mode,
                        &parallel_group,
                        &unit.tool_call_id,
                        &replay_session_id,
                    ) {
                        Ok(id) => id,
                        Err(e) => {
                            tracing::warn!(target: "deeplossless::pipeline", "failed to store execution unit: {e}");
                            continue;
                        }
                    };

                    // Record DependsOn lineage edge: consecutive units in a conversation
                    // form a dependency chain (unit N depends on unit N-1's output).
                    if let Some(prev_id) = last_exec_id {
                        if let Err(e) = db.insert_lineage_edge(prev_id, exec_id, "depends_on") {
                            tracing::warn!(target: "deeplossless::pipeline",
                                "failed to insert DependsOn edge: {e}");
                        }
                    }
                    last_exec_id = Some(exec_id);

                    // Update tracker with result
                    if !unit.tool_call_id.is_empty() {
                        for tracker in &mut active_trackers {
                            if let Err(e) = tracker.record_branch_result(&unit.tool_call_id, exec_id, &unit.outcome) {
                                tracing::debug!(target: "deeplossless::pipeline", "record_branch_result: {e}");
                            }
                        }
                    }

                    // Auto-populate tool cache from message history.
                    if crate::tool_cache::is_cacheable(&unit.tool_name)
                        && !unit.tool_result.is_empty() {
                        let (_name, args_hash) = crate::tool_cache::cache_key(
                            &unit.tool_name, &unit.tool_args);
                        let dependent_files = crate::tool_cache::extract_dependent_files(
                            &unit.tool_name, &unit.tool_args);
                        if let Err(e) = db.tool_cache_put(
                            &unit.tool_name, &args_hash,
                            &unit.tool_result, &dependent_files,
                        ) {
                            tracing::warn!(target: "deeplossless::pipeline", "failed to cache tool result: {e}");
                        }
                    }
                    // Auto-record failures from tool results
                    if unit.outcome != crate::execution::ExecutionOutcome::Success
                        && !unit.tool_result.is_empty() {
                        let sig: String = unit.tool_result.chars().take(120).collect();
                        let _ = db.store_failure_pattern(
                            conv_id, &sig,
                            "",
                            &unit.tool_result,
                            &[], &[], None,
                        );
                        tracing::debug!(target: "deeplossless::pipeline",
                            conv_id, tool = %unit.tool_name,
                            "auto-recorded failure pattern");
                    }
                }

                // Complete any finished parallel trackers and record HappensBefore edges.
                // For groups that should force-join, insert a join DAG node.
                for tracker in &active_trackers {
                    if !tracker.should_force_join() {
                        continue;
                    }
                    // Insert a join DAG node
                    let summary = format!(
                        "Parallel group {} ({} branches)",
                        tracker.group_id,
                        tracker.branch_count(),
                    );
                    match dag.db().insert_join_atomic(
                        conv_id,
                        &summary,
                        0,
                        &[],
                        &[],
                    ) {
                        Ok(join_node) => {
                            let edges = match tracker.clone().complete(join_node.id) {
                                Ok(e) => e,
                                Err(e) => {
                                    tracing::warn!(target: "deeplossless::pipeline", "complete failed: {e}");
                                    continue;
                                }
                            };
                            for hb in &edges {
                                if let Err(e) = db.insert_edge(
                                    hb.from_id, hb.to_id, "happens_before",
                                ) {
                                    tracing::warn!(target: "deeplossless::pipeline",
                                        "failed to insert HappensBefore edge: {e}");
                                }
                            }
                            pending_hb_edges.extend(edges);
                        }
                        Err(e) => {
                            tracing::warn!(target: "deeplossless::pipeline",
                                "failed to insert join DAG node: {e}");
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
                CompactEvent::BelowThreshold { .. } => {}
                CompactEvent::Error { message, .. } => {
                    tracing::warn!(target: "deeplossless::pipeline", conv_id, error = %message, "compaction error");
                }
                CompactEvent::CompactionCompleted { .. } => {}
                CompactEvent::Pong => {}
            }
        }
        drop(compactor);

        // Assemble DAG context and inject into system messages
        let mut injected = req_body.clone();
        let query = req_body["messages"].as_array()
            .and_then(|arr| arr.iter().rev().find(|m| m["role"] == "user"))
            .and_then(|m| m["content"].as_str());
        // LCM context appended as user message (not system prompt) — safe for
    // tool-call agents. Disabled by default; enable via --lcm-context.
    if self.lcm_context
            && let Ok(dag_ctx) = self.dag.assemble_context(conv_id, 2000, query)
            && !dag_ctx.is_empty() {
                let mut ctx_text = render_dag_context(&dag_ctx);
                if let Ok(claims) = self.db.list_all_file_claims()
                    && !claims.is_empty() {
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

        // Reasoning injection disabled — modifying request body changes model
    // reasoning trajectory. OpenCode and other agents handle reasoning_content
    // correctly in their own message construction. Our injection, even with
    // real content, can cause tool-call lifecycle disruption.
    let invalid = crate::assistant_validation::validate_request_messages(&injected);
    if invalid > 0 {
        tracing::warn!(target: "deeplossless::pipeline", invalid, "assistant messages missing critical fields");
    }

        Ok(PipelineOutput { conv_id, injected_body: injected })
    }

    /// For any assistant message that has `tool_calls` but is missing
    /// `reasoning_content`, inject captured reasoning_content from the
    /// previous response. Only injects when actual content is available.
    /// Empty strings cause DeepSeek to enter confused thinking mode.
    #[allow(dead_code)]
    fn inject_reasoning_content(&self, body: &mut serde_json::Value) {
        let model = body["model"].as_str().unwrap_or("").to_string();
        let Some(messages) = body["messages"].as_array_mut() else { return };
        let last_user = messages.iter().rev().find(|m| m["role"] == "user")
            .and_then(|m| m["content"].as_str()).unwrap_or("");
        let truncated: String = last_user.chars().take(80).collect();
        let key = format!("reasoning:{model}:{truncated}");
        let stored = self.db.get_reasoning(&key).ok().flatten();
        for msg in messages.iter_mut() {
            if msg["role"] != "assistant" { continue; }
            if msg.get("tool_calls").and_then(|v| v.as_array()).map(|a| a.is_empty()) == Some(true) { continue; }
            if msg.get("tool_calls").is_none() { continue; }
            if msg.get("reasoning_content").is_none() {
                // Only inject if we have actual captured reasoning. Empty string
                // triggers DeepSeek into thinking mode with no context → slow + hang.
                if let Some(rc) = &stored {
                    if !rc.is_empty() {
                        msg["reasoning_content"] = serde_json::json!(rc);
                        tracing::debug!(target: "deeplossless::pipeline", len=rc.len(), "injected reasoning_content");
                    }
                }
            }
        }
    }

    /// Inject LCM context as a tool-role message from `lcm_memory`.
    /// Tool results are "external capability" — optional memory, not
    /// planning authority. Uses retrieval hints (first 2 items, truncated)
    /// to avoid planner interference from full context dumps.
    /// Inject LCM context as a user message with `[lcm]` prefix.
    /// Tool role requires a paired assistant tool_call (DeepSeek rejects
    /// synthetic tool messages). User role is safe — the model treats it as
    /// additional context rather than a tool execution record.
    fn inject_context(body: &mut serde_json::Value, ctx_text: &str) {
        // Strip XML wrapper — context is now a user message, not system prompt
        let clean = ctx_text
            .trim_start_matches("<lcm_context>\n")
            .trim_end_matches("</lcm_context>\n")
            .trim();
        let hint: String = clean.lines()
            .filter(|l| !l.trim().is_empty())
            .take(2)
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(arr) = body["messages"].as_array_mut() {
            arr.push(serde_json::json!({
                "role": "user",
                "content": format!(
                    "[lcm] cached context (GET /v1/lcm/grep for full retrieval):\n{hint}"
                ),
            }));
        }
    }
}
