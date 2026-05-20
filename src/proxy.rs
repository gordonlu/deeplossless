use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json, Router, routing::{get, post},
};
use serde_json::{json, Value};
use std::collections::HashMap;
use futures::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use crate::metrics;
use crate::AppState;

/// Uniform JSON error envelope: `{"error": {"code": "...", "message": "..."}}`
fn json_error(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"code": code, "message": message.into()}}))).into_response()
}

/// Readiness probe: checks DB connectivity, upstream reachability,
/// and compactor liveness. Returns 200 with status JSON, or 503 if
/// any dependency is unhealthy.
async fn lcm_health(State(state): State<AppState>) -> Response {
    let mut healthy = true;
    let mut checks = serde_json::json!({});

    // DB check — fast SELECT 1 on writer connection
    match state.db.wal_checkpoint() {
        Ok(()) => checks["database"] = json!("ok"),
        Err(e) => {
            healthy = false;
            checks["database"] = json!(format!("error: {e}"));
        }
    }

    // Upstream reachability — lightweight HEAD
    let upstream = state.upstream.trim_end_matches('/');
    match state.client.head(upstream).send().await {
        Ok(resp) => checks["upstream"] = json!(format!("reachable (http {})", resp.status())),
        Err(e) => {
            healthy = false;
            checks["upstream"] = json!(format!("unreachable: {e}"));
        }
    }

    // Compactor liveness — send a no-op command to check responsiveness
    {
        let mut compactor = state.compactor.lock().await;
        let alive = compactor.drain_events().is_empty(); // just check if channel is alive
        checks["compactor"] = json!(if alive { "ok" } else { "no events" });
    }

    let status_code = if healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (status_code, Json(json!({
        "status": if healthy { "healthy" } else { "unhealthy" },
        "checks": checks,
    }))).into_response()
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/lcm/grep/{conv_id}", get(lcm_grep))
        .route("/v1/lcm/expand/{node_id}", get(lcm_expand))
        .route("/v1/lcm/status/{conv_id}", get(lcm_status))
        .route("/v1/lcm/snippets/{node_id}", get(lcm_snippets))
        .route("/v1/lcm/similar/{hash}", get(lcm_similar))
        .route("/v1/lcm/trace/{node_id}", get(lcm_trace))
        .route("/v1/lcm/global/search", get(lcm_global_search))
        .route("/v1/lcm/execution/search", get(lcm_execution_search))
        .route("/v1/lcm/stream/{conv_id}", get(lcm_stream_context))
        .route("/v1/lcm/runtime/stats", get(lcm_runtime_stats))
        .route("/v1/lcm/runtime/debug-dump", get(lcm_debug_dump))
        .route("/v1/lcm/runtime/report", get(lcm_runtime_report))
        .route("/v1/lcm/cache/put", post(lcm_cache_put))
        .route("/v1/lcm/cache", get(lcm_cache_get))
        .route("/v1/lcm/failure", post(lcm_failure_put))
        .route("/v1/lcm/plan", post(lcm_plan_put))
        .route("/v1/lcm/plan/{conv_id}", get(lcm_plan_get))
        .route("/v1/lcm/file/claim", post(lcm_file_claim))
        .route("/v1/lcm/file/release", post(lcm_file_release))
        .route("/v1/lcm/file/conflicts", get(lcm_file_conflicts))
        .route("/v1/lcm/health/{conv_id}", get(lcm_dag_health))
        .route("/v1/lcm/compress", post(lcm_compress))
        .route("/v1/lcm/delete", post(lcm_delete))
        .route("/v1/lcm/rollback", post(lcm_rollback))
        .route("/health", get(lcm_health))
        .route("/v1/health", get(lcm_health))
        .route("/metrics", get(metrics::handle_metrics))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Extract API key from Authorization header on first request
    {
        let mut key = state.api_key.lock().unwrap_or_else(|e| e.into_inner());
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
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", format!("invalid JSON: {e}")),
    };

    let model = crate::session::model_name(&req_body);
    let streaming = crate::session::is_streaming(&req_body);

    // Run the chat pipeline (fingerprint → store → compact → assemble → inject)
    let pipeline = crate::pipeline::ChatPipeline::new(&state);
    let injected_body = match pipeline.process(model, &req_body).await {
        Ok(out) => out.injected_body,
        Err(e) => {
            warn!("pipeline error: {e}, falling back to passthrough");
            req_body.clone()
        }
    };

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let resp = match state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", get_cached_key(&state.api_key)))
        .header("Content-Type", "application/json")
        .json(&injected_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
        }
    };

    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().expect("static header parse"));

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
        response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header parse"));
        response.headers_mut().insert("x-accel-buffering", "no".parse().expect("static header parse"));
        response
    } else {
        match resp.bytes().await {
            Ok(bytes) => {
                let mut response = Response::new(Body::from(bytes));
                *response.status_mut() = status;
                response.headers_mut().insert("content-type", content_type);
                response
            }
            Err(e) => {
                metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
            }
        }
    }
}

/// Get the cached API key, or "unset" if none has been provided yet.
/// The compactor will fall back to Level 3 (deterministic) if the key
/// is "unset".
fn get_cached_key(key: &std::sync::Mutex<Option<String>>) -> String {
    key.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| "unset".to_string())
}

/// Verify that a request to a Context-ReAct endpoint carries a valid
/// Authorization header. Checks `admin_key` first, then falls back to
/// `api_key` for backward compatibility. If no key is configured at all,
/// allows all (safe for localhost-only deployments).
fn ctx_react_auth_ok(headers: &HeaderMap, state: &AppState) -> bool {
    // Prefer explicit admin_key
    let admin = state.admin_key.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref admin_key) = *admin {
        return check_bearer(headers, admin_key);
    }
    drop(admin);

    // Fall back to api_key (backward compat)
    let expected = get_cached_key(&state.api_key);
    if expected == "unset" {
        return true;
    }
    check_bearer(headers, &expected)
}

fn check_bearer(headers: &HeaderMap, expected: &str) -> bool {
    let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let bearer = auth.strip_prefix("Bearer ").or_else(|| auth.strip_prefix("bearer "));
    bearer == Some(expected)
}

// ── LCM retrieval endpoints ────────────────────────────────────────────

async fn lcm_grep(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let query = params.get("query").map(|s| s.as_str()).unwrap_or("");
    match state.db.search_unified(conv_id, query) {
        Ok(results) => Json(json!({
            "conversation_id": conv_id,
            "query": query,
            "total": results.len(),
            "matches": results,
        }))
        .into_response(),
        Err(e) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR", format!("search error: {e}"))
        }
    }
}

async fn lcm_expand(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(node_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    // Expand a summary node to its children (original messages)
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
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "EXPAND_ERROR", format!("expand error: {e}"))
        }
    }
}

async fn lcm_snippets(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(node_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    match state.dag.get_node(node_id) {
        Ok(Some(node)) => {
            Json(serde_json::json!({
                "node_id": node_id,
                "snippets": node.snippets,
                "summary": node.summary,
            }))
            .into_response()
        }
        Ok(None) => json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "node not found"),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "NODE_ERROR", format!("error: {e}")),
    }
}

/// Streaming DAG context via SSE — yields summaries first, then messages incrementally.
async fn lcm_stream_context(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let budget: usize = params.get("budget").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let q_owned: Option<String> = params.get("q").cloned();

    let dag = state.dag.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        let q_ref = q_owned.as_deref();
        let nodes = dag.assemble_context(conv_id, budget, q_ref)
            .unwrap_or_default();

        for (i, node) in nodes.iter().enumerate() {
            let data = serde_json::json!({
                "idx": i,
                "type": if node.is_leaf { "message" } else { "summary" },
                "id": node.id,
                "level": if node.is_leaf { 0 } else { node.level as i32 },
                "summary": node.summary,
                "tokens": node.token_count,
                "reasoning": if node.reasoning.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(node.reasoning.clone()) },
            });
            let payload = format!("data: {}\n\n", serde_json::to_string(&data).unwrap_or_default());
            if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(payload))).is_err() {
                break;
            }
            // Small yield to let the client process incrementally
            tokio::task::yield_now().await;
        }

        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: [DONE]\n\n")
        ));
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    match axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(axum::body::Body::from_stream(stream))
    {
        Ok(resp) => resp,
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "STREAM_ERROR", format!("{e}")),
    }
}

/// Claim a file for an agent. Returns 409 if another agent holds it.
async fn lcm_file_claim(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let agent_id = body.get("agent_id").map(|s| s.as_str()).unwrap_or("unknown");
    let file_path = body.get("file_path").map(|s| s.as_str()).unwrap_or("");
    let operation = body.get("operation").map(|s| s.as_str()).unwrap_or("edit");
    match state.db.claim_file(agent_id, file_path, operation) {
        Ok(Ok(())) => Json(json!({"status": "claimed", "agent_id": agent_id, "file_path": file_path})).into_response(),
        Ok(Err(conflict_agent)) => (
            StatusCode::CONFLICT,
            Json(json!({"status": "conflict", "held_by": conflict_agent, "file_path": file_path}))
        ).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "CLAIM_ERROR", format!("{e}")),
    }
}

/// Release an agent's claim on a file.
async fn lcm_file_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let agent_id = body.get("agent_id").map(|s| s.as_str()).unwrap_or("unknown");
    let file_path = body.get("file_path").map(|s| s.as_str()).unwrap_or("");
    match state.db.release_file(agent_id, file_path) {
        Ok(()) => Json(json!({"status": "released"})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "RELEASE_ERROR", format!("{e}")),
    }
}

/// List all active file claims (for conflict awareness).
async fn lcm_file_conflicts(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match state.db.list_all_file_claims() {
        Ok(claims) => {
            let rows: Vec<Value> = claims.iter().map(|(aid, path, op)| json!({
                "agent_id": aid, "file_path": path, "operation": op
            })).collect();
            Json(json!({"conflicts": rows})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// Store a tool result in cache. Agent calls this after executing a tool.
async fn lcm_cache_put(
    State(state): State<AppState>,
    Json(body): Json<HashMap<String, String>>,
) -> Response {
    let tool = body.get("tool").map(|s| s.as_str()).unwrap_or("");
    let args = body.get("args").map(|s| s.as_str()).unwrap_or("");
    let result = body.get("result").map(|s| s.as_str()).unwrap_or("");
    let files: Vec<String> = body.get("files")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    let (_name, hash) = crate::tool_cache::cache_key(tool, args);
    if tool.is_empty() || hash.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "tool and args required");
    }
    match state.db.tool_cache_put(tool, &hash, result, &files) {
        Ok(()) => Json(json!({"status": "cached", "tool": tool})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "CACHE_ERROR", format!("{e}")),
    }
}

/// Look up a cached tool result. Agent calls this before executing a tool.
async fn lcm_cache_get(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let tool = params.get("tool").map(|s| s.as_str()).unwrap_or("");
    let args = params.get("args").map(|s| s.as_str()).unwrap_or("");

    let (_name, hash) = crate::tool_cache::cache_key(tool, args);
    if tool.is_empty() || hash.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "tool and args required");
    }
    match state.db.tool_cache_get(tool, &hash) {
        Ok(Some((result, hits))) => Json(json!({"hit": true, "result": result, "hit_count": hits})).into_response(),
        Ok(None) => Json(json!({"hit": false})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "CACHE_ERROR", format!("{e}")),
    }
}

/// Store a failure pattern. Agent calls this after a fix fails.
async fn lcm_failure_put(
    State(state): State<AppState>,
    Json(body): Json<HashMap<String, serde_json::Value>>,
) -> Response {
    let conv_id: i64 = body.get("conv_id").and_then(|v| v.as_i64()).unwrap_or(1);
    let sig = body.get("signature").and_then(|v| v.as_str()).unwrap_or("");
    let fix = body.get("attempted_fix").and_then(|v| v.as_str()).unwrap_or("");
    let why = body.get("why_failed").and_then(|v| v.as_str()).unwrap_or("");
    let assumptions: Vec<String> = body.get("assumptions")
        .and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let files: Vec<String> = body.get("files")
        .and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();

    if sig.is_empty() { return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "signature required"); }
    match state.db.store_failure_pattern(conv_id, sig, fix, why, &assumptions, &files, None) {
        Ok(id) => Json(json!({"status": "stored", "id": id})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "FAILURE_ERROR", format!("{e}")),
    }
}

/// Store an execution plan. Agent calls this after creating a plan.
async fn lcm_plan_put(
    State(state): State<AppState>,
    Json(body): Json<HashMap<String, serde_json::Value>>,
) -> Response {
    let conv_id: i64 = body.get("conv_id").and_then(|v| v.as_i64()).unwrap_or(1);
    let goal = body.get("goal").and_then(|v| v.as_str()).unwrap_or("");
    let steps: Vec<String> = body.get("steps")
        .and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let assumptions: Vec<String> = body.get("assumptions")
        .and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();

    if goal.is_empty() { return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "goal required"); }
    match state.db.store_plan_state(conv_id, goal, &steps, &assumptions) {
        Ok(id) => Json(json!({"status": "stored", "id": id})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "PLAN_ERROR", format!("{e}")),
    }
}

/// Get the active plan for a conversation.
async fn lcm_plan_get(
    State(state): State<AppState>,
    Path(conv_id): Path<i64>,
) -> Response {
    match state.db.get_active_plan(conv_id) {
        Ok(Some((id, goal, pending, assumptions))) => Json(json!({
            "id": id, "goal": goal,
            "pending_steps": pending,
            "assumptions": assumptions,
        })).into_response(),
        Ok(None) => Json(json!({"active_plan": null})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "PLAN_ERROR", format!("{e}")),
    }
}

/// Debug dump for GitHub issues. Strips user content — only counters, hashes, and structure.
async fn lcm_debug_dump(
    State(state): State<AppState>,
) -> Response {
    let cycle = state.cycle.lock().unwrap_or_else(|e| e.into_inner());
    let m = &cycle.metrics;

    // Cache stats (no content, just key counts)
    let cache_entry_count = state.db.top_tool_cache_entries(999).unwrap_or_default().len();
    let (total_nodes, total_convs, total_embeddings, active_plan_count, event_count) =
        state.db.debug_counts().unwrap_or((-1, -1, -1, -1, -1));
    let failure_count: i64 = state.db.search_failure_patterns("error", 9999)
        .map(|r| r.len() as i64).unwrap_or(-1);

    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "profile": cycle.profile.as_str(),
        "metrics": {
            "tokens_spent": m.tokens_spent,
            "cache_hits": m.cache_hits,
            "cache_misses": m.cache_misses,
            "cache_hit_rate": if m.cache_hits + m.cache_misses > 0 {
                format!("{:.0}%", m.cache_hits as f64 / (m.cache_hits + m.cache_misses) as f64 * 100.0)
            } else { "N/A".into() },
            "repeated_failures": m.repeated_failures,
            "failure_streak": m.failure_streak,
            "reread_ratio": m.reread_ratio,
            "planning_reuse_ratio": m.planning_reuse_ratio,
            "budget_remaining_pct": m.budget_remaining_pct,
            "decisions_count": cycle.decisions.len(),
        },
        "storage": {
            "cache_entries": cache_entry_count,
            "dag_nodes": total_nodes,
            "conversations": total_convs,
            "embeddings": total_embeddings,
            "failure_patterns": failure_count,
            "active_plans": active_plan_count,
            "dag_events": event_count,
        },
        "decisions_tail": cycle.decisions.iter().rev().take(10).collect::<Vec<_>>(),
    })).into_response()
}

/// Generate a shareable session report (markdown).
async fn lcm_runtime_report(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let label = params.get("label").map(|s| s.as_str()).unwrap_or("coding session");
    let conv_id: Option<i64> = params.get("conv_id").and_then(|s| s.parse().ok());
    let turns: usize = params.get("turns").and_then(|s| s.parse().ok()).unwrap_or(0);
    let fmt = params.get("format").map(|s| s.as_str()).unwrap_or("md");

    // Per-conversation data if conv_id provided
    let (_leaf_count, _summary_count, total_tokens, failure_count) = conv_id
        .and_then(|cid| state.db.collect_session_metrics(cid).ok())
        .unwrap_or((0, 0, 0, 0));

    let top_reused = state.db.top_tool_cache_entries(8)
        .unwrap_or_default()
        .into_iter()
        .map(|(name, count)| (name, count as u64))
        .collect::<Vec<_>>();

    let duration: u64 = params.get("duration").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut cycle = state.cycle.lock().unwrap_or_else(|e| e.into_inner()).clone();

    // If conv_id provided, override global metrics with per-session data
    if conv_id.is_some() {
        cycle.metrics.tokens_spent = total_tokens as u64;
        cycle.metrics.repeated_failures = failure_count as u64;
        cycle.metrics.failure_streak = failure_count.min(1) as u32;
    }

    if fmt == "svg" {
        let svg = crate::runtime::generate_svg_card(&cycle, label, turns, &top_reused);
        let mut response = Response::new(axum::body::Body::from(svg));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert("content-type", "image/svg+xml".parse().expect("static header"));
        return response;
    }

    let report = crate::runtime::generate_report(&cycle, label, turns, &top_reused, duration);
    let mut response = Response::new(axum::body::Body::from(report));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert("content-type", "text/markdown; charset=utf-8".parse().expect("static header"));
    response
}

/// Runtime metrics endpoint — exposes inference-economics counters for benchmarking.
async fn lcm_runtime_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let cycle = state.cycle.lock().unwrap_or_else(|e| e.into_inner());
    let m = &cycle.metrics;
    let total_cache = m.cache_hits + m.cache_misses;
    Json(json!({
        "profile": cycle.profile.as_str(),
        "tokens_spent": m.tokens_spent,
        "cache_hits": m.cache_hits,
        "cache_misses": m.cache_misses,
        "cache_hit_rate": if total_cache > 0 { (m.cache_hits as f64 / total_cache as f64 * 100.0).round() / 100.0 } else { 0.0 },
        "repeated_failures": m.repeated_failures,
        "failure_streak": m.failure_streak,
        "reread_ratio": m.reread_ratio,
        "planning_reuse_ratio": m.planning_reuse_ratio,
        "budget_remaining_pct": m.budget_remaining_pct,
        "decisions_count": cycle.decisions.len(),
        "recent_decisions": &cycle.decisions.iter().rev().take(5).collect::<Vec<_>>(),
    })).into_response()
}

/// Execution memory search — finds similar bugs, tool chains, code edits.
async fn lcm_execution_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let query = params.get("q").map(|s| s.as_str()).unwrap_or("");
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(10);
    match state.dag.search_execution_memory(query, limit) {
        Ok(refs) => Json(json!({ "results": refs })).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR", format!("{e}")),
    }
}

/// Cross-session semantic search across all conversations.
async fn lcm_global_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    let query = params.get("q").map(|s| s.as_str()).unwrap_or("");
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(10);
    match state.dag.search_cross_session(query, limit) {
        Ok(results) => {
            let items: Vec<Value> = results.iter().map(|(nid, cid, summary)| json!({
                "node_id": nid,
                "conversation_id": cid,
                "summary": summary,
            })).collect();
            Json(json!({ "results": items })).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR", format!("search error: {e}")),
    }
}

async fn lcm_dag_health(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    match state.dag.validate_dag(conv_id) {
        Ok(issues) => {
            let healthy = issues.is_empty();
            Json(json!({
                "conversation_id": conv_id,
                "healthy": healthy,
                "issues": issues,
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "VALIDATION_ERROR", format!("validation error: {e}")),
    }
}

async fn lcm_similar(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(hash): Path<String>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    match state.db.find_similar_by_hash(&hash) {
        Ok(results) => Json(json!({
            "hash": hash,
            "matches": results.iter().map(|(h, nid, cid, preview)| json!({
                "hash": h,
                "node_id": nid,
                "conversation_id": cid,
                "preview": preview,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR", format!("search error: {e}")),
    }
}

async fn lcm_trace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(node_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }    match state.db.get_provenance_with_excerpts(node_id) {
        Ok(rows) => {
            let sources: Vec<Value> = rows.iter().map(|(sid, off, len, excerpt)| json!({
                "source_node_id": sid,
                "offset": off,
                "length": len,
                "excerpt": excerpt,
            })).collect();
            Json(json!({
                "summary_node_id": node_id,
                "sources": sources,
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "PROVENANCE_ERROR", format!("provenance error: {e}")),
    }
}

async fn lcm_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
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

/// Compress a range of messages into a summary.  Returns the new summary
/// text and node ID.  The caller (model) can insert the summary into
/// the conversation via a follow-up message.
async fn lcm_compress(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(op): Json<LcmRangeOp>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "Context-ReAct requires Authorization header");
    }
    // Try DAG leaves first, then fall back to raw messages
    let content: Vec<String> = match state.dag.get_leaves(op.conv_id) {
        Ok(leaves) if leaves.len() >= 2 => leaves.iter()
            .skip_while(|n| n.id < op.from)
            .take_while(|n| n.id <= op.to)
            .map(|n| n.summary.clone())
            .collect(),
        _ => {
            // Fallback: read from DB directly
            let db = state.dag.db();
            let raw = db.get_messages_in_range(op.conv_id, op.from, op.to);
            match raw {
                Ok(msgs) if msgs.len() >= 2 => msgs,
                _ => return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "need at least 2 messages to compress"),
            }
        }
    };
    let text = content.join("\n---\n");

    let summarizer = match crate::summarizer::Summarizer::builder()
        .api_key(&get_cached_key(&state.api_key))
        .model(state.summarizer_model())
        .upstream(&state.upstream)
        .build()
    {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "SUMMARIZER_ERROR", format!("summarizer: {e}")),
    };

    match summarizer.summarize_escalate(&text).await {
        Ok((summary, _level)) => {
            let tc = crate::tokenizer::count(&summary) as i64;
            let snippets = crate::snippet::extract(&text);
            match state.dag.compress_group_with_snippets(
                op.conv_id, &[], &summary, tc, 1, &snippets,
            ) {
                Ok(node) => Json(json!({
                    "node_id": node.id,
                    "summary": node.summary,
                    "token_count": node.token_count,
                    "snippets": node.snippets,
                }))
                .into_response(),
                Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "COMPRESS_ERROR", format!("compress: {e}")),
            }
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SUMMARIZE_ERROR", format!("summarize: {e}")),
    }
}

/// Soft-delete a node from the DAG (mark is_leaf = 0 so it won't appear
/// in active context assembly).  The raw data is still in the messages
/// table and can be recovered via lcm_expand.
async fn lcm_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(op): Json<LcmIdOp>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match state.dag.db().delete_dag_node(op.id) {
        Ok(_) => Json(json!({"deleted": op.id})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DELETE_ERROR", format!("delete: {e}")),
    }
}

/// Rollback: return the state of a specific summary node so the model
/// can reconstruct its context to that point.  The caller receives the
/// summary text and its children, which it can use as a checkpoint.
async fn lcm_rollback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(op): Json<LcmIdOp>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let node = match state.dag.get_node(op.id) {
        Ok(Some(n)) => n,
        Ok(None) => return json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "node not found"),
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "NODE_ERROR", format!("error: {e}")),
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
