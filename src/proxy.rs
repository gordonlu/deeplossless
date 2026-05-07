use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
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
        .route("/v1/lcm/grep/:conv_id", get(lcm_grep))
        .route("/v1/lcm/expand/:node_id", get(lcm_expand))
        .route("/v1/lcm/status/:conv_id", get(lcm_status))
        .route("/health", get(|| async { StatusCode::OK }))
}

async fn chat_completions(
    State(state): State<AppState>,
    body: String,
) -> Response {
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

    // Assemble DAG context and inject into system prompt
    let mut injected = req_body.clone();
    if let Some(system_msg) = injected["messages"]
        .as_array_mut()
        .and_then(|arr| arr.iter_mut().find(|m| m["role"] == "system"))
    {
        // Inject into existing system message
        let dag_ctx = state.dag.assemble_context(conv_id, 2000).unwrap_or_default();
        if !dag_ctx.is_empty() {
            let ctx_text = render_dag_context(&dag_ctx);
            let existing = system_msg["content"].as_str().unwrap_or("");
            let combined = format!("{}\n\n<lcm_context>\n{}\n</lcm_context>", existing, ctx_text);
            system_msg["content"] = json!(combined);
        }
    }

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let resp = match state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", state.api_key))
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

/// Render DAG context nodes into a readable text block for injection.
fn render_dag_context(nodes: &[crate::dag::DagNode]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for node in nodes {
        if node.is_leaf {
            let _ = writeln!(out, "[msg {}] {} ({} tok)", node.id, node.summary, node.token_count);
        } else {
            let _ = writeln!(
                out,
                "[summary {}] L{} — {} ({} tok, {} parents)",
                node.id,
                node.level,
                node.summary,
                node.token_count,
                node.parent_ids.len()
            );
        }
    }
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
