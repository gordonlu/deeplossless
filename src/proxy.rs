use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json, Router, routing::{get, post},
};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use crate::compactor::{CompactCommand, CompactEvent};
use crate::AppState;

/// Max context window for threshold calculations.
const CONTEXT_WINDOW: usize = 1_000_000;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/lcm/grep/{conv_id}", get(lcm_grep))
        .route("/v1/lcm/expand/{node_id}", get(lcm_expand))
        .route("/v1/lcm/status/{conv_id}", get(lcm_status))
        .route("/v1/lcm/snippets/{node_id}", get(lcm_snippets))
        .route("/v1/lcm/compress", post(lcm_compress))
        .route("/v1/lcm/delete", post(lcm_delete))
        .route("/v1/lcm/rollback", post(lcm_rollback))
        .route("/health", get(|| async { StatusCode::OK }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Extract API key from Authorization header on first request
    {
        let mut key = state.api_key.lock().unwrap();
        if key.is_none()
            && let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok())
            && let Some(bearer) = auth.strip_prefix("Bearer ")
                .or_else(|| auth.strip_prefix("bearer "))
        {
            *key = Some(bearer.to_string());
            tracing::debug!(target: "deeplossless", "API key extracted from request header");
        }
    }

    let req_body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")).into_response(),
    };

    let model = crate::session::model_name(&req_body);
    let streaming = crate::session::is_streaming(&req_body);
    let messages = req_body["messages"].clone();
    let msgs_arr = messages.as_array().map(|a| a.as_slice()).unwrap_or(&[]);

    // Resolve conversation via fingerprint
    let fp = crate::session::fingerprint(msgs_arr, 3);
    let conv_id = match state.db.find_or_create_conversation(&fp, model) {
        Ok(id) => id,
        Err(e) => {
            warn!("failed to resolve conversation: {e}");
            return (StatusCode::INTERNAL_SERVER_ERROR, "db error").into_response();
        }
    };

    // Store messages asynchronously
    let db = state.db.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = db.store_messages(conv_id, &messages) {
            warn!("failed to store messages: {e}");
        }
    });

    // Trigger async compaction review (soft threshold)
    {
        let mut compactor = state.compactor.lock().await;
        let _ = compactor
            .command(CompactCommand::ReviewAndCompact {
                conv_id,
                context_window: CONTEXT_WINDOW,
            })
            .await;
        // Drain and log any resulting events
        for event in compactor.drain_events() {
            match event {
                CompactEvent::GroupCompressed { tokens_saved, .. } => {
                    tracing::debug!(target: "deeplossless", conv_id, tokens_saved, "compaction completed");
                }
                CompactEvent::BelowThreshold => {}
                CompactEvent::Error { message } => {
                    tracing::warn!(target: "deeplossless", conv_id, error = %message, "compaction error");
                }
            }
        }
    }

    // Assemble DAG context and inject into ALL system messages.
    // DAG failures never block the request — the proxy falls back
    // to a plain passthrough if context assembly fails.
    let mut injected = req_body.clone();
    if let Ok(dag_ctx) = state.dag.assemble_context(conv_id, 2000) {
        if !dag_ctx.is_empty() {
            let ctx_text = render_dag_context(&dag_ctx);
            if let Some(arr) = injected["messages"].as_array_mut() {
                for msg in arr.iter_mut().filter(|m| m["role"] == "system") {
                    let existing = msg["content"].as_str().unwrap_or("");
                    msg["content"] = json!(format!("{}\n\n<lcm_context>\n{}\n</lcm_context>", existing, ctx_text));
                }
            }
        }
    } else {
        tracing::debug!(target: "deeplossless", conv_id, "DAG context assembly failed, running without injection");
    }

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let resp = match state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", get_cached_key(&state.api_key)))
        .header("Content-Type", "application/json")
        .json(&injected)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response(),
    };

    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().unwrap());

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);
        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        if tx.send(Ok::<_, std::convert::Infallible>(c)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("stream error: {e}");
                        break;
                    }
                }
            }
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert("content-type", content_type);
        response.headers_mut().insert("cache-control", "no-cache".parse().unwrap());
        response.headers_mut().insert("x-accel-buffering", "no".parse().unwrap());
        response
    } else {
        match resp.bytes().await {
            Ok(bytes) => {
                let mut response = Response::new(Body::from(bytes));
                *response.status_mut() = status;
                response.headers_mut().insert("content-type", content_type);
                response
            }
            Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response(),
        }
    }
}

/// Render DAG context nodes into a structured, actionable context panel.
/// Each node shows its summary, snippets, and available operations.
/// Get the cached API key, or "unset" if none has been provided yet.
/// The compactor will fall back to Level 3 (deterministic) if the key
/// is "unset".
fn get_cached_key(key: &std::sync::Mutex<Option<String>>) -> String {
    key.lock().unwrap().clone().unwrap_or_else(|| "unset".to_string())
}

fn render_dag_context(nodes: &[crate::dag::DagNode]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "<lcm_context>");

    // Collect for range-based operations
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

        // Snippets
        for s in &node.snippets {
            let label = match s.snippet_type {
                crate::snippet::SnippetType::CodeBlock => "code",
                crate::snippet::SnippetType::FilePath => "path",
                crate::snippet::SnippetType::NumericConstant => "num",
                crate::snippet::SnippetType::ErrorMessage => "err",
            };
            let _ = writeln!(out, "    ├ {label}: {}", s.content);
        }

        // Available actions for this node
        if node.level > 0 {
            let _ = writeln!(out, "    └ /lcm/rollback {id}", id = node.id);
        }
    }

    // Range operations at the bottom
    if ids.len() >= 2 {
        let first = ids.first().unwrap();
        let last = ids.last().unwrap();
        let _ = writeln!(out);
        let _ = writeln!(out, "  Operations:");
        let _ = writeln!(out, "    /lcm/compress conv_id={cid} from={first} to={last}", cid = nodes[0].conversation_id);
        let _ = writeln!(out, "    /lcm/delete conv_id={cid} id=<node_id>", cid = nodes[0].conversation_id);
        let _ = writeln!(out, "    /lcm/rollback conv_id={cid} id=<summary_id>", cid = nodes[0].conversation_id);
    }

    let _ = writeln!(out, "</lcm_context>");
    out
}

// ── LCM retrieval endpoints ────────────────────────────────────────────

async fn lcm_grep(
    State(state): State<AppState>,
    Path(conv_id): Path<i64>,
    query: String,
) -> Response {
    match state.db.search_messages(conv_id, &query) {
        Ok(results) => Json(json!({
            "conversation_id": conv_id,
            "query": query,
            "matches": results,
        }))
        .into_response(),
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("search error: {e}")).into_response()
        }
    }
}

async fn lcm_expand(
    State(state): State<AppState>,
    Path(node_id): Path<i64>,
) -> Response {
    // Expand a summary node to its children (original messages)
    match state.dag.get_children(node_id) {
        Ok(children) => {
            let node = state.dag.get_node(node_id).ok().flatten();
            Json(json!({
                "node_id": node_id,
                "summary": node.map(|n| n.summary),
                "children": children,
            }))
            .into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("expand error: {e}")).into_response()
        }
    }
}

async fn lcm_snippets(
    State(state): State<AppState>,
    Path(node_id): Path<i64>,
) -> Response {
    match state.dag.get_node(node_id) {
        Ok(Some(node)) => {
            Json(serde_json::json!({
                "node_id": node_id,
                "snippets": node.snippets,
                "summary": node.summary,
            }))
            .into_response()
        }
        Ok(None) => (StatusCode::NOT_FOUND, "node not found").into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("error: {e}")).into_response(),
    }
}

async fn lcm_status(
    State(state): State<AppState>,
    Path(conv_id): Path<i64>,
) -> Response {
    let leaves = state.dag.get_leaves(conv_id).unwrap_or_default();
    let tip = state.dag.get_active_tip(conv_id).ok().flatten();
    let total = state.dag.total_tokens(conv_id).unwrap_or(0);
    Json(json!({
        "conversation_id": conv_id,
        "total_tokens": total,
        "leaf_count": leaves.len(),
        "has_summary": tip.is_some(),
        "summary_level": tip.map(|n| n.level),
    }))
    .into_response()
}

// ── Context-ReAct operations ───────────────────────────────────────────

#[derive(serde::Deserialize)]
struct LcmRangeOp {
    #[allow(dead_code)]
    conv_id: i64,
    from: i64,
    to: i64,
}

#[derive(serde::Deserialize)]
struct LcmIdOp {
    #[allow(dead_code)]
    conv_id: i64,
    id: i64,
}

/// Compress a range of nodes into a summary.  Returns the new summary
/// text and node ID.  The caller (model) can then insert the summary
/// into the conversation via a follow-up message.
async fn lcm_compress(
    State(state): State<AppState>,
    Json(op): Json<LcmRangeOp>,
) -> Response {
    let leaves = match state.dag.get_leaves(op.conv_id) {
        Ok(l) => l,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("error: {e}")).into_response(),
    };
    let ids: Vec<i64> = leaves.iter().map(|n| n.id)
        .skip_while(|id| *id < op.from)
        .take_while(|id| *id <= op.to)
        .collect();
    if ids.len() < 2 {
        return (StatusCode::BAD_REQUEST, String::from("need at least 2 nodes to compress")).into_response();
    }
    let text = leaves.iter()
        .filter(|n| ids.contains(&n.id))
        .map(|n| n.summary.as_str())
        .collect::<Vec<_>>()
        .join("\n---\n");

    let summarizer = match crate::summarizer::Summarizer::builder()
        .api_key(&get_cached_key(&state.api_key))
        .model("deepseek-v4-flash")
        .upstream(&state.upstream)
        .build()
    {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("summarizer: {e}")).into_response(),
    };

    match summarizer.summarize_escalate(&text).await {
        Ok((summary, _level)) => {
            let tc = crate::tokenizer::count(&summary) as i64;
            let snippets = crate::snippet::extract(&text);
            match state.dag.compress_group_with_snippets(
                op.conv_id, &ids, &summary, tc, 1, &snippets,
            ) {
                Ok(node) => Json(json!({
                    "node_id": node.id,
                    "summary": node.summary,
                    "token_count": node.token_count,
                    "snippets": node.snippets,
                }))
                .into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("compress: {e}")).into_response(),
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("summarize: {e}")).into_response(),
    }
}

/// Soft-delete a node from the DAG (mark is_leaf = 0 so it won't appear
/// in active context assembly).  The raw data is still in the messages
/// table and can be recovered via lcm_expand.
async fn lcm_delete(
    State(state): State<AppState>,
    Json(op): Json<LcmIdOp>,
) -> Response {
    match state.dag.db().delete_dag_node(op.id) {
        Ok(_) => Json(json!({"deleted": op.id})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("delete: {e}")).into_response(),
    }
}

/// Rollback: return the state of a specific summary node so the model
/// can reconstruct its context to that point.  The caller receives the
/// summary text and its children, which it can use as a checkpoint.
async fn lcm_rollback(
    State(state): State<AppState>,
    Json(op): Json<LcmIdOp>,
) -> Response {
    let node = match state.dag.get_node(op.id) {
        Ok(Some(n)) => n,
        Ok(None) => return (StatusCode::NOT_FOUND, "node not found").into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("error: {e}")).into_response(),
    };
    let children = state.dag.get_children(op.id).unwrap_or_default();
    Json(json!({
        "rollback_to": op.id,
        "summary": node.summary,
        "level": node.level,
        "snippets": node.snippets,
        "children": children.iter().map(|c| json!({
            "id": c.id,
            "summary": c.summary,
            "token_count": c.token_count,
        })).collect::<Vec<_>>(),
    }))
    .into_response()
}
