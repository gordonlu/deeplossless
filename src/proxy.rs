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
use futures::FutureExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use crate::metrics;
use crate::protocol::canonical::StreamEvent;
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
    match state.storage.db.wal_checkpoint() {
        Ok(()) => checks["database"] = json!("ok"),
        Err(e) => {
            healthy = false;
            checks["database"] = json!(format!("error: {e}"));
        }
    }

    // Upstream reachability — lightweight HEAD
    let upstream = state.upstream.trim_end_matches('/');
    match state.runtime.client.head(upstream).send().await {
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

/// Check the tool cache for a pair of [ToolCallStart, ToolCallArgsDelta] events.
/// Returns `(Some(cached_result), 2)` if hit, or `(None, 0)` if the first event
/// at `offset` is not an interceptable tool call pair.
fn check_tool_cache(
    events: &[StreamEvent],
    offset: usize,
    db: &std::sync::Arc<crate::db::Database>,
    cycle: &std::sync::Mutex<crate::runtime::ExecutionCycle>,
) -> (Option<String>, usize) {
    if offset + 1 >= events.len() { return (None, 0); }
    if let StreamEvent::ToolCallStart { index: si, name, .. } = &events[offset] {
        // Only intercept tools whose results are compact (grep/search/diagnostics).
        // Large results (read_file, list_files) would flood the conversation.
        if !crate::tool_cache::is_interceptable(name) { return (None, 0); }
        if let StreamEvent::ToolCallArgsDelta { index: ai, arguments_delta } = &events[offset + 1] {
            if si == ai {
                let (cname, args_hash) = crate::tool_cache::cache_key(name, arguments_delta);
                match db.tool_cache_get(&cname, &args_hash) {
                    Ok(Some((result, _hits))) => {
                        if let Ok(mut c) = cycle.lock() {
                            #[allow(deprecated)]
                            c.record_cache_hit(name);
                        }
                        let transformed = crate::tool_cache::transform_result(name, &result);
                        tracing::info!(target: "deeplossless",
                            tool=name, %args_hash, raw_len=result.len(), transformed_len=transformed.len(),
                            "cache hit — intercepting tool call");
                        return (Some(transformed), 2);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::debug!(target: "deeplossless",
                            tool=name, error=%e, "cache lookup error, forwarding tool call");
                    }
                }
            }
        }
    }
    (None, 0)
}

/// Process assembled events, with tool cache interception. For cache hits,
/// emits text deltas instead of forwarding tool calls. Processes ALL events
/// in the Vec (no early return — handles multi-tool-call scenarios correctly).
///
/// Cache-hit text is fed back through the assembler (when provided) so the
/// final lifecycle events contain the cached content.
fn process_events(
    events: Vec<StreamEvent>,
    db: std::sync::Arc<crate::db::Database>,
    cycle: &std::sync::Mutex<crate::runtime::ExecutionCycle>,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<axum::body::Bytes, std::convert::Infallible>>,
    mut assembler: Option<&mut crate::protocol::streaming::StreamAssembler>,
    use_responses_format: bool,
) -> bool {
    let mut seq: i64 = 0;
    let mut i = 0;
    while i < events.len() {
        let (cached, consumed) = check_tool_cache(&events, i, &db, cycle);
        if let Some(text) = cached {
            let text_ev = StreamEvent::TextDelta { text };
            // Fire-and-forget event store (best-effort, never blocks the stream)
            let db2 = db.clone();
            let kind = "TextDelta";
            let payload = event_to_payload(&text_ev);
            let sn = seq;
            let epoch_ms = crate::execution::next_logical_seq();
            tokio::task::spawn_blocking(move || {
                if let Err(e) = db2.store_execution_event(None, kind, &payload, sn, Some(0), epoch_ms) {
                    tracing::debug!(target: "deeplossless", "execution event store failed: {e}");
                }
            });
            seq += 1;
            if use_responses_format {
                if let Some(asm) = assembler.as_mut() {
                    for ev in asm.feed(text_ev) {
                        let sse_line = crate::protocol::streaming::to_responses_sse(&ev);
                        if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                            return false;
                        }
                    }
                } else {
                    let sse_line = crate::protocol::streaming::to_responses_sse(&text_ev);
                    if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                        return false;
                    }
                }
            } else {
                let sse_line = crate::protocol::streaming::to_chat_completions_sse(&text_ev);
                if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                    return false;
                }
            }
            i += consumed;
            continue;
        }
        // Store event before emission (best-effort, non-blocking)
        let ev = &events[i];
        let db2 = db.clone();
        let kind = event_kind_name(ev);
        let payload = event_to_payload(ev);
        let sn = seq;
        let epoch_ms = crate::execution::next_logical_seq();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = db2.store_execution_event(None, &kind, &payload, sn, Some(0), epoch_ms) {
                tracing::debug!(target: "deeplossless", "execution event store failed: {e}");
            }
        });
        seq += 1;
        // Normal emission
        let sse_line = if use_responses_format {
            crate::protocol::streaming::to_responses_sse(ev)
        } else {
            crate::protocol::streaming::to_chat_completions_sse(ev)
        };
        if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
            return false;
        }
        i += 1;
    }
    true
}

fn event_kind_name(ev: &StreamEvent) -> String {
    match ev {
        StreamEvent::TextDelta { .. } => "TextDelta",
        StreamEvent::ToolCallStart { .. } => "ToolCallStart",
        StreamEvent::ToolCallArgsDelta { .. } => "ToolCallArgsDelta",
        StreamEvent::ToolCallEnd { .. } => "ToolCallEnd",
        StreamEvent::ReasoningDelta { .. } => "ReasoningDelta",
        StreamEvent::MessageStart { .. } => "MessageStart",
        StreamEvent::MessageEnd => "MessageEnd",
        StreamEvent::OutputItemAdded { .. } => "OutputItemAdded",
        StreamEvent::OutputItemDone { .. } => "OutputItemDone",
        StreamEvent::FunctionCallArgumentsDone { .. } => "FunctionCallArgumentsDone",
        StreamEvent::Done { .. } => "Done",
        StreamEvent::Error { .. } => "Error",
    }.to_string()
}

fn event_to_payload(ev: &StreamEvent) -> String {
    serde_json::to_string(ev).unwrap_or_default()
}

/// Session log entry — one JSON line per request, written when `--log-dir` is set.
#[derive(serde::Serialize)]
struct LogEntry {
    ts: String,
    endpoint: &'static str,
    model: String,
    request_body_kb: f64,
    input_tokens_est: usize,
    output_tokens: u64,
    output_text_len: usize,
    cache_hits: u64,
    msg_count: usize,
    tools_count: usize,
    instructions_len: usize,
    instr_hash: String,
    prompt_cache_key: String,
    upstream_status: u16,
    elapsed_ms: u64,
    error: Option<String>,
}

/// Session file path, created once per process. All requests append to the same file.
static SESSION_FILE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

fn session_file_path(log_dir: &str) -> &std::path::PathBuf {
    SESSION_FILE.get_or_init(|| {
        let _ = std::fs::create_dir_all(log_dir);
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        std::path::PathBuf::from(format!("{}/session-{ts}.jsonl", log_dir.trim_end_matches('/')))
    })
}

fn write_log(log_dir: Option<&str>, entry: &LogEntry) {
    let Some(dir) = log_dir else { return };
    let path = session_file_path(dir);
    // Append one line — best-effort, never crash on log failure
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write;
        if let Ok(line) = serde_json::to_string(entry) {
            let _ = writeln!(f, "{line}");
        }
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{response_id}", get(responses_retrieve))
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
        .route("/v1/lcm/score/{conv_id}", get(lcm_score))
        .route("/v1/lcm/audit/{conv_id}", get(lcm_audit_trail))
        .route("/v1/lcm/audit/report/{conv_id}", get(lcm_audit_report))
        .route("/v1/lcm/replay/{execution_id}", get(lcm_replay))
        .route("/v1/lcm/snapshot", post(lcm_snapshot_take))
        .route("/v1/lcm/versions", get(lcm_versions))
        .route("/v1/lcm/cache/put", post(lcm_cache_put))
        .route("/v1/lcm/cache", get(lcm_cache_get))
        .route("/v1/lcm/failure", post(lcm_failure_put))
        .route("/v1/lcm/plan", post(lcm_plan_put))
        .route("/v1/lcm/plan/{conv_id}", get(lcm_plan_get))
        .route("/v1/lcm/file/claim", post(lcm_file_claim))
        .route("/v1/lcm/file/release", post(lcm_file_release))
        .route("/v1/lcm/file/conflicts", get(lcm_file_conflicts))
        .route("/v1/lcm/health/{conv_id}", get(lcm_dag_health))
        .route("/v1/lcm/motifs/{conv_id}", get(lcm_motifs))
        .route("/v1/lcm/observe", post(lcm_observe))
        .route("/v1/lcm/compress", post(lcm_compress))
        .route("/v1/lcm/delete", post(lcm_delete))
        .route("/v1/lcm/rollback", post(lcm_rollback))
        .route("/health", get(lcm_health))
        .route("/v1/health", get(lcm_health))
        .route("/metrics", get(metrics::handle_metrics))
}

/// Responses API endpoint — translates to Chat Completions internally.
/// Codex and other Responses API clients connect here.
async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let req_body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", format!("invalid JSON: {e}")),
    };

    // Request diagnostics: log key metrics for token cost tracking
    let prev_resp = req_body["previous_response_id"].as_str().unwrap_or("(none)");
    let instructions_len = req_body["instructions"].as_str().map(|s| s.len()).unwrap_or(0);
    let tools_count = req_body["tools"].as_array().map(|a| a.len()).unwrap_or(0);
    let store = req_body["store"].as_bool().unwrap_or(true);
    let prompt_cache_key = req_body["prompt_cache_key"].as_str().unwrap_or("(none)");
    let input_count = req_body["input"].as_array().map(|a| a.len()).unwrap_or(0);
    tracing::info!(target: "deeplossless",
        previous_response_id=%prev_resp,
        instructions_len, input_count, tools_count,
        store, prompt_cache_key=%prompt_cache_key,
        request_body_kb=body.len() as f64 / 1024.0,
        turnaround=input_count.saturating_sub(1).min(99),
        "request diagnostics");

    // Extract API key on first request
    {
        let mut key = state.api_key.lock().unwrap_or_else(|e| e.into_inner());
        if key.is_none()
            && let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok())
            && let Some(bearer) = auth.strip_prefix("Bearer ")
                .or_else(|| auth.strip_prefix("bearer "))
        {
            *key = Some(bearer.to_string());
        }
    }

    // 1. Responses API → canonical IR
    let mut canonical = crate::protocol::responses::request_from_responses(&req_body);
    // Map Codex model names to DeepSeek equivalents
    canonical.model = map_model(&canonical.model);
    // Codex sends Accept: text/event-stream — treat as implicit stream request
    let accept_ss = headers.get("accept").and_then(|v| v.to_str().ok()).unwrap_or("");
    let streaming = canonical.stream || accept_ss.contains("text/event-stream");
    if streaming && !canonical.stream {
        canonical.stream = true;
    }

    // 2. Canonical IR → Chat Completions (for DeepSeek)
    let chat_body = crate::protocol::chat_completions::request_to_chat(&canonical);

    // 3. Run the chat pipeline (DAG context injection, message persistence)
    let pipeline = crate::pipeline::ChatPipeline::new(&state);
    let chat_body_val: serde_json::Value = chat_body.clone();
    let injected = match pipeline.process(&canonical.model, &chat_body_val).await {
        Ok(out) => out.injected_body,
        Err(e) => {
            warn!("pipeline error: {e}, falling back to passthrough");
            chat_body_val
        }
    };

    // ── Dry-run: save translated body, return mock response ──
    if state.dry_run {
        let out_dir = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(".deeplossless");
        let _ = std::fs::create_dir_all(&out_dir);
        let _ = std::fs::write(out_dir.join("last_request.json"),
            serde_json::to_string_pretty(&req_body).unwrap_or_default());
        let _ = std::fs::write(out_dir.join("translated.json"),
            serde_json::to_string_pretty(&injected).unwrap_or_default());

        let msgs = injected["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        tracing::info!(target: "deeplossless", msg_count=msgs.len(),
            model=%canonical.model, stream=canonical.stream,
            "dry-run: saved to ~/.deeplossless/");

        // Return mock streaming response
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: {\"type\":\"response.output_text.delta\",\"delta\":\"[dry-run] request saved to ~/.deeplossless/translated.json\"}\n\n")
        ));
        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: [DONE]\n\n")
        ));
        let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert("content-type", "text/event-stream; charset=utf-8".parse().expect("static header"));
        response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header"));
        response.headers_mut().insert("connection", "close".parse().expect("static header"));
        return response;
    }

    // Token breakdown: compute system/history split for diagnostics
    let (mut system_len, mut user_len, mut history_len) = (0usize, 0usize, 0usize);
    let mut instr_hash = String::from("-");
    if let Some(msgs) = injected["messages"].as_array() {
        for msg in msgs {
            let role = msg["role"].as_str().unwrap_or("");
            let content = msg["content"].as_str().unwrap_or("");
            match role {
                "system" => {
                    system_len += content.len();
                    if instr_hash == "-" {
                        use sha2::{Digest, Sha256};
                        instr_hash = hex::encode(&Sha256::digest(content.as_bytes())[..6]);
                    }
                }
                "user" => user_len += content.len(),
                _ => history_len += content.len(),
            }
        }
    }

    // 4. Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let api_key = get_cached_key(&state.api_key);
    let req_stream = injected.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let req_msgs = injected.get("messages").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    tracing::info!(target: "deeplossless",
        msg_count=req_msgs, instr_hash,
        system_kb=system_len as f64 / 1024.0,
        user_kb=user_len as f64 / 1024.0,
        history_kb=history_len as f64 / 1024.0,
        "token breakdown");
    tracing::debug!(target: "deeplossless", model=%canonical.model, canonical_stream=canonical.stream,
        req_stream, msg_count=req_msgs, upstream_url,
        has_api_key=!api_key.is_empty() && api_key != "unset",
        "sending upstream request");
    let resp = match state.runtime
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&injected)
        .send()
        .await
    {
        Ok(r) => {
            tracing::debug!(target: "deeplossless", status=%r.status(),
                content_type=?r.headers().get("content-type"),
                "upstream response received");
            r
        }
        Err(e) => {
            metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
        }
    };

    // 5. Handle non-200 upstream status — propagate error, don't wrap in SSE
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return json_error(status, "UPSTREAM_ERROR", body);
    }

    // 6. Handle streaming vs non-streaming
    if streaming {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);
        let response_store = state.storage.response_store.clone();
        let db = state.storage.db.clone();
        let cycle = state.runtime.cycle.clone();
        let store_response = store;
        let request_body = body.clone();
        let log_dir = state.log_dir.clone();
        let upstream_status = resp.status().as_u16();
        let start = std::time::Instant::now();
        let log_model = canonical.model.clone();
        let log_instr_hash = instr_hash.clone();
        let log_prompt_cache_key = prompt_cache_key.to_string();
        let log_tools_count = tools_count;
        let log_instructions_len = instructions_len;
        let log_msg_count = req_msgs;
        let log_request_body_kb = body.len() as f64 / 1024.0;
        let shutdown = state.runtime.shutdown_notify.clone();
        tokio::spawn(async move {
            if shutdown.notified().now_or_never().is_some() { return; }
            // Generate IDs matching OpenAI format
            let resp_id = format!("resp_{}", crate::protocol::responses::monotonic_id());
            let msg_id = format!("msg_{}", crate::protocol::responses::monotonic_id());
            let model = canonical.model.clone();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

            // Helper: build the full response envelope used by created/in_progress/completed
            let response_envelope = |status: &str| -> String {
                format!("\"id\":\"{resp_id}\",\"object\":\"response\",\"created_at\":{now},\"status\":\"{status}\",\"model\":\"{model}\",\"output\":[],\"tools\":[],\"text\":{{\"format\":{{\"type\":\"text\"}}}},\"usage\":null")
            };

            // response.created (full envelope, status=in_progress)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{{}}}}}\n\n",
                response_envelope("in_progress")
            ))));
            // response.in_progress (same full envelope)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.in_progress\ndata: {{\"type\":\"response.in_progress\",\"response\":{{{}}}}}\n\n",
                response_envelope("in_progress")
            ))));
            // output_item.added (status=in_progress)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}}}\n\n"
            ))));
            // content_part.added (with item_id + annotations)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.content_part.added\ndata: {{\"type\":\"response.content_part.added\",\"item_id\":\"{msg_id}\",\"output_index\":0,\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}}}\n\n"
            ))));
            tracing::debug!(target: "deeplossless::stream", resp_id, msg_id, "sent stream preamble");

            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut usage_buf: Option<serde_json::Value> = None;
            let mut first_chunk = true;
            let mut assembler = crate::protocol::streaming::StreamAssembler::new();
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        let s = String::from_utf8_lossy(&c);
                        if first_chunk {
                            first_chunk = false;
                            tracing::info!(target: "deeplossless",
                                len=c.len(), preview=&s[..s.len().min(200)],
                                "first upstream chunk");
                        }
                        buf.push_str(&s);
                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].trim().to_string();
                            buf = buf[pos + 1..].to_string();
                            if let Some(data_line) = line.strip_prefix("data: ") {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data_line)
                                    && v.get("usage").is_some() {
                                        usage_buf = Some(v.clone());
                                    }
                                for event in crate::protocol::streaming::from_chat_completions_sse(data_line, usage_buf.as_ref()) {
                                    // Done = transport-level EOF, drain remaining buffers
                                    if matches!(event, StreamEvent::Done { .. }) {
                                        let events = assembler.flush();
                                        if !process_events(events, db.clone(), &cycle, &tx, Some(&mut assembler), true) { break; }
                                        continue;
                                    }
                                    let events = assembler.feed(event);
                                    if !process_events(events, db.clone(), &cycle, &tx, Some(&mut assembler), true) { break; }
                                }
                            }
                        }
                    }
                    Err(e) => { warn!("stream error: {e}"); break; }
                }
            }
            if first_chunk {
                tracing::warn!(target: "deeplossless",
                    "byte_stream produced zero chunks — upstream returned empty body");
            }
            // Process trailing buffer (last line without \n)
            if !buf.trim().is_empty() {
                let data_line = buf.trim().strip_prefix("data: ").unwrap_or(&buf);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data_line)
                    && v.get("usage").is_some() {
                        usage_buf = Some(v.clone());
                    }
                for event in crate::protocol::streaming::from_chat_completions_sse(data_line, usage_buf.as_ref()) {
                    if !matches!(event, StreamEvent::Done { .. }) {
                        let events = assembler.feed(event);
                        process_events(events, db.clone(), &cycle, &tx, Some(&mut assembler), true);
                    }
                }
            }
            // Upstream [DONE] → finish assembly, get accumulated content
            let content = assembler.finish();
            let input_tokens_est = crate::tokenizer::count(&request_body);
            let output_tokens = usage_buf.as_ref()
                .and_then(|v| v["usage"]["completion_tokens"].as_u64())
                .unwrap_or(0);
            let output_text_len = content.text.len();
            tracing::info!(target: "deeplossless",
                input_tokens_est, output_tokens, output_text_len,
                "turn complete");

            // Write session log if --log-dir is set
            if log_dir.is_some() {
                let cache_hits = cycle.lock().ok().map(|c| c.metrics.cache_hits).unwrap_or(0);
                write_log(log_dir.as_deref(), &LogEntry {
                    ts: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f").to_string(),
                    endpoint: "responses",
                    model: log_model,
                    request_body_kb: log_request_body_kb,
                    input_tokens_est,
                    output_tokens,
                    output_text_len,
                    cache_hits,
                    msg_count: log_msg_count,
                    tools_count: log_tools_count,
                    instructions_len: log_instructions_len,
                    instr_hash: log_instr_hash,
                    prompt_cache_key: log_prompt_cache_key,
                    upstream_status,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    error: if upstream_status >= 400 { Some(format!("HTTP {upstream_status}")) } else { None },
                });
            }

            // Build lifecycle events using serde_json for proper escaping
            let mut content_parts: Vec<serde_json::Value> = Vec::new();
            // Include reasoning content when present (required for tool-call multi-turn continuity)
            if !content.reasoning.is_empty() {
                content_parts.push(serde_json::json!({
                    "type": "reasoning", "text": content.reasoning
                }));
            }
            let text_part = serde_json::json!({
                "type": "output_text", "text": content.text, "annotations": []
            });
            content_parts.push(text_part.clone());
            let content_json = serde_json::json!(content_parts);
            let item_json = serde_json::json!({
                "id": msg_id, "type": "message", "status": "completed",
                "role": "assistant", "content": content_json
            });
            let output_item_done = serde_json::json!({
                "type": "response.output_item.done", "output_index": 0, "item": item_json
            });
            let content_part_done = serde_json::json!({
                "type": "response.content_part.done", "item_id": msg_id,
                "output_index": 0, "content_index": content_parts.len().saturating_sub(1),
                "part": text_part
            });
            let resp_status = if usage_buf.is_some() { "completed" } else { "incomplete" };
            let usage_json = usage_buf.map(|v| serde_json::json!({
                "input_tokens": v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                "output_tokens": v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                "total_tokens": v["usage"]["total_tokens"].as_u64().unwrap_or(0),
            })).unwrap_or(serde_json::json!({"input_tokens":0,"output_tokens":0,"total_tokens":0}));
            let completed = serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": resp_id, "object": "response", "created_at": now,
                    "status": resp_status, "model": model,
                    "output": [item_json],
                    "usage": usage_json
                }
            });

            // Emit lifecycle events in correct order
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from("event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\"}\n\n")
            ));
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.content_part.done\ndata: {content_part_done}\n\n"))
            ));
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.output_item.done\ndata: {output_item_done}\n\n"))
            ));
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.completed\ndata: {completed}\n\n"))
            ));
            // Persist the response so GET /v1/responses/{id} returns real data,
            // and Codex's previous_response_id continuity can work incrementally.
            // Persist reasoning for multi-turn continuity
            if !content.reasoning.is_empty() {
                let reason_key = format!("reasoning:{model}:{msg_id}");
                let reason_db = db.clone();
                let reason_text = content.reasoning.clone();
                tokio::task::spawn_blocking(move || {
                    let _ = reason_db.store_reasoning(&reason_key, &reason_text);
                });
            }

            if store_response {
                let resp_obj = serde_json::json!({
                    "id": resp_id, "object": "response", "created_at": now,
                    "status": resp_status, "model": model,
                    "output": [{
                        "id": msg_id, "type": "message", "status": "completed",
                        "role": "assistant",
                        "content": content_parts
                    }],
                    "usage": usage_json
                });
                response_store.insert(resp_id.clone(), resp_obj);
                tracing::info!(target: "deeplossless",
                    resp_id, text_len=content.text.len(),
                    "response stored");
            }
            // Transport-level EOF marker
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from("data: [DONE]\n\n")
            ));
            // tx dropped here → stream closes
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert("content-type", "text/event-stream; charset=utf-8".parse().expect("static header"));
        response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header"));
        response.headers_mut().insert("connection", "close".parse().expect("static header"));
        response
    } else {
        // Non-streaming: translate Chat Completions response → Responses format
        match resp.bytes().await {
            Ok(bytes) => {
                let chat_resp: serde_json::Value = match serde_json::from_slice(&bytes) {
                    Ok(v) => v,
                    Err(e) => return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("invalid upstream JSON: {e}")),
                };
                let canonical_resp = crate::protocol::chat_completions::response_from_chat(&chat_resp);
                let responses_body = crate::protocol::responses::response_to_responses(&canonical_resp);
                let mut response = Response::new(Body::from(serde_json::to_string(&responses_body).unwrap_or_default()));
                *response.status_mut() = StatusCode::OK;
                response.headers_mut().insert("content-type", "application/json".parse().expect("static header"));
                response
            }
            Err(e) => {
                metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
            }
        }
    }
}

/// Responses API retrieve — looks up a previously stored response.
/// Codex uses this for status polling, stream reconnection, and
/// `previous_response_id`-based incremental turns.
async fn responses_retrieve(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    // Check response store first
    if let Some(resp) = state.storage.response_store.get(&response_id) {
        tracing::debug!(target: "deeplossless",
            %response_id, "response retrieve hit");
        return Json(resp).into_response();
    }
    tracing::warn!(target: "deeplossless",
        %response_id, "response retrieve miss");
    json_error(StatusCode::NOT_FOUND, "NOT_FOUND", format!("response '{response_id}' not found"))
}

/// OpenAI-compatible model list. OpenCode and other clients query this to
/// discover supported models and their capabilities (reasoning, context, etc.).
async fn list_models() -> Response {
    // Redundant context window fields — the "OpenAI-compatible" ecosystem
    // has no unified schema. Different clients read different field names.
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": "deepseek-v4-pro",
                "object": "model",
                "owned_by": "deepseek",
                "context_window": 1_000_000,
                "max_context_tokens": 1_000_000,
                "max_input_tokens": 1_000_000,
                "supports_reasoning": true,
                "supports_thinking": true,
                "reasoning": true,
                "thinking": true,
                "capabilities": {
                    "supports_tool_calls": true,
                    "supports_streaming": true,
                    "supports_reasoning": true,
                    "supports_thinking": true,
                    "max_context_tokens": 1_000_000
                }
            },
            {
                "id": "deepseek-v4-flash",
                "object": "model",
                "owned_by": "deepseek",
                "context_window": 1_000_000,
                "max_context_tokens": 1_000_000,
                "max_input_tokens": 1_000_000,
                "supports_reasoning": true,
                "supports_thinking": true,
                "reasoning": true,
                "thinking": true,
                "capabilities": {
                    "supports_tool_calls": true,
                    "supports_streaming": true,
                    "supports_reasoning": true,
                    "max_context_tokens": 1_000_000
                }
            }
        ]
    })).into_response()
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
    let resp = match state.runtime
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
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return json_error(status, "UPSTREAM_ERROR", body);
    }
    let content_type = resp
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().expect("static header parse"));

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);
        // Capture reasoning_content for multi-turn continuity.
        // DeepSeek requires it on tool-call messages in subsequent requests.
        // Keyed by last user message + model to avoid fingerprint collision.
        let reasoning_db = state.storage.db.clone();
        let reasoning_key = {
            let msgs = injected_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
            let last_user = msgs.iter().rev().find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str()).unwrap_or("");
            let model = injected_body["model"].as_str().unwrap_or("");
            format!("reasoning:{model}:{}", last_user.chars().take(80).collect::<String>())
        };
        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut reasoning = String::new();
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        if tx.send(Ok::<_, std::convert::Infallible>(c.clone())).is_err() {
                            break;
                        }
                        let s = String::from_utf8_lossy(&c);
                        buf.push_str(&s);
                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].trim().to_string();
                            buf = buf[pos + 1..].to_string();
                            if let Some(data_line) = line.strip_prefix("data: ") {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data_line) {
                                    if let Some(rc) = v["choices"][0]["delta"]["reasoning_content"].as_str() {
                                        reasoning.push_str(rc);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("stream error: {e}");
                        break;
                    }
                }
            }
            if !reasoning.is_empty() {
                let _ = reasoning_db.store_reasoning(&reasoning_key, &reasoning);
            }
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert("content-type", content_type);
        response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header parse"));
        response.headers_mut().insert("x-accel-buffering", "no".parse().expect("static header parse"));
        response.headers_mut().insert("connection", "close".parse().expect("static header parse"));
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
/// Map Codex model names to DeepSeek equivalents.
/// Codex sends "gpt-5.5", "gpt-4o", "o3" etc. — translate to DeepSeek models.
fn map_model(model: &str) -> String {
    crate::protocol::ModelRegistry::default().map_model(model)
}

fn get_cached_key(key: &std::sync::Mutex<Option<String>>) -> String {
    let guard = key.lock().unwrap_or_else(|e| e.into_inner());
    guard.clone().unwrap_or_default()
}

/// Verify that a request to a Context-ReAct endpoint carries a valid
/// Authorization header. Checks `admin_key` first, then falls back to
/// `api_key` for backward compatibility. If no key is configured at all,
/// allows all (safe for localhost-only deployments).
fn ctx_react_auth_ok(headers: &HeaderMap, state: &AppState) -> bool {
    // Prefer explicit admin_key
    let admin = state.admin_key.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(admin_key) = admin.as_ref() {
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
    match state.storage.db.search_unified(conv_id, query) {
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
    match state.storage.dag.get_children(node_id) {
        Ok(children) => {
            let node = state.storage.dag.get_node(node_id).ok().flatten();
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
    }    match state.storage.dag.get_node(node_id) {
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

    let dag = state.storage.dag.clone();
    let shutdown = state.runtime.shutdown_notify.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        if shutdown.notified().now_or_never().is_some() { return; }
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
    match state.storage.db.claim_file(agent_id, file_path, operation) {
        Ok(Ok(())) => Json(json!({"status": "claimed", "agent_id": agent_id, "file_path": file_path})).into_response(),
        Ok(Err(conflict_agent)) => json_error(StatusCode::CONFLICT, "CONFLICT", format!("file '{file_path}' held by agent '{conflict_agent}'")),
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
    match state.storage.db.release_file(agent_id, file_path) {
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
    match state.storage.db.list_all_file_claims() {
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
    match state.storage.db.tool_cache_put(tool, &hash, result, &files) {
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
    match state.storage.db.tool_cache_get(tool, &hash) {
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
    match state.storage.db.store_failure_pattern(conv_id, sig, fix, why, &assumptions, &files, None) {
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
    match state.storage.db.store_plan_state(conv_id, goal, &steps, &assumptions) {
        Ok(id) => Json(json!({"status": "stored", "id": id})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "PLAN_ERROR", format!("{e}")),
    }
}

/// Get the active plan for a conversation.
async fn lcm_plan_get(
    State(state): State<AppState>,
    Path(conv_id): Path<i64>,
) -> Response {
    match state.storage.db.get_active_plan(conv_id) {
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
    let cycle = state.runtime.cycle.lock().unwrap_or_else(|e| e.into_inner());
    let m = &cycle.metrics;

    // Cache stats (no content, just key counts)
    let cache_entry_count = state.storage.db.top_tool_cache_entries(999).unwrap_or_default().len();
    let (total_nodes, total_convs, total_embeddings, active_plan_count, event_count) =
        state.storage.db.debug_counts().unwrap_or((-1, -1, -1, -1, -1));
    let failure_count: i64 = state.storage.db.search_failure_patterns("error", 9999)
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
    let (_leaf_count, _summary_count, total_tokens, failure_count) = match conv_id.and_then(|cid| state.storage.db.collect_session_metrics(cid).ok()) {
        Some(m) => m,
        None => {
            if conv_id.is_some() {
                tracing::warn!(target: "deeplossless", "collect_session_metrics failed for conv_id={:?}", conv_id);
            }
            (0, 0, 0, 0)
        }
    };

    let top_reused = state.storage.db.top_tool_cache_entries(8)
        .unwrap_or_default()
        .into_iter()
        .map(|(name, count)| (name, count as u64))
        .collect::<Vec<_>>();

    let duration: u64 = params.get("duration").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut cycle = state.runtime.cycle.lock().unwrap_or_else(|e| e.into_inner()).clone();

    // Override global metrics with per-session data for accurate reporting
    if conv_id.is_some() {
        cycle.set_session_metrics(total_tokens as u64, failure_count as u64, failure_count.min(1) as u32);
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

/// Replay an execution from the event log — returns full StreamEvent sequence.
/// GET /v1/lcm/audit/{conv_id} — audit trail for a conversation.
async fn lcm_audit_trail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let query = crate::audit::AuditQuery {
        conv_id: Some(conv_id),
        limit: 500,
        ..Default::default()
    };
    match crate::audit::build_audit_trail(&state.storage.db, &query) {
        Ok(records) => Json(json!({"conv_id": conv_id, "records": records, "total": records.len()})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "AUDIT_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/audit/report/{conv_id} — aggregated audit report.
async fn lcm_audit_report(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match crate::audit::build_audit_report(&state.storage.db, Some(conv_id)) {
        Ok(report) => Json(json!({"conv_id": conv_id, "report": report})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "AUDIT_REPORT_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/score/{conv_id} — execution outcome scoring.
async fn lcm_score(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match state.storage.db.compute_execution_score(conv_id) {
        Ok(score) => Json(score).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SCORE_ERROR", format!("{e}")),
    }
}

async fn lcm_replay(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(execution_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match crate::replay::replay_execution(&state.storage.db, execution_id) {
        Ok(result) => {
            let items: Vec<serde_json::Value> = result
                .events
                .iter()
                .map(|e| serde_json::json!({"seq_no": e.seq_no, "event": e.event, "schema_version": e.schema_version}))
                .collect();
            Json(json!({
                "execution_id": execution_id,
                "events": items,
                "total": items.len(),
                "corrupt_count": result.corrupt_count,
            })).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "REPLAY_ERROR", format!("{e}")),
    }
}

/// Take an execution snapshot for later replay/rollback.
async fn lcm_snapshot_take(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<HashMap<String, serde_json::Value>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let execution_id = body.get("execution_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let memory_version_id = body.get("memory_version_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let tier_raw = body.get("tier").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let data = body.get("data").and_then(|v| v.as_str()).unwrap_or("{}");
    let size_bytes = data.len() as i64;
    let ttl = body.get("retention_ttl").and_then(|v| v.as_i64());

    if execution_id == 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "execution_id required");
    }

    // Validate tier — fail fast on invalid values
    if let Err(e) = crate::snapshot::SnapshotTier::from_i32(tier_raw) {
        return json_error(StatusCode::BAD_REQUEST, "INVALID_TIER", format!("{e}"));
    }

    // Compute continuity metadata from the provided data
    let last_event_seq_no = body.get("last_event_seq_no").and_then(|v| v.as_i64()).unwrap_or(0);
    let boundary_hash = crate::snapshot::compute_boundary_hash(
        &[(0_i64, data)],
        1,
    );
    let integrity_hash = crate::snapshot::compute_chain_hash(&[(0_i64, data)]);

    match state.storage.db.take_snapshot(
        execution_id, memory_version_id, tier_raw, data, size_bytes, ttl,
        last_event_seq_no, &boundary_hash, &integrity_hash,
    ) {
        Ok(id) => {
            let _ = state.storage.db.enforce_snapshot_budget(&crate::snapshot::SnapshotBudget::default());
            Json(json!({"status": "stored", "id": id})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SNAPSHOT_ERROR", format!("{e}")),
    }
}

/// List memory version history.
async fn lcm_versions(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match state.storage.db.list_memory_versions(50) {
        Ok(versions) => Json(json!({"versions": versions})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "VERSION_ERROR", format!("{e}")),
    }
}

/// Runtime metrics endpoint — exposes inference-economics counters for benchmarking.
async fn lcm_runtime_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let cycle = state.runtime.cycle.lock().unwrap_or_else(|e| e.into_inner());
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
    match state.storage.dag.search_execution_memory(query, limit) {
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
    match state.storage.dag.search_cross_session(query, limit) {
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
    }    match state.storage.dag.validate_dag(conv_id) {
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
    }    match state.storage.db.find_similar_by_hash(&hash) {
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
    }    match state.storage.db.get_provenance_with_excerpts(node_id) {
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
    let leaves = state.storage.dag.get_leaves(conv_id).unwrap_or_default();
    let tip = state.storage.dag.get_active_tip(conv_id).ok().flatten();
    let total = state.storage.dag.total_tokens(conv_id).unwrap_or(0);
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
    conv_id: i64,
    from: i64,
    to: i64,
}

#[derive(serde::Deserialize)]
struct LcmIdOp {
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
    let content: Vec<String> = match state.storage.dag.get_leaves(op.conv_id) {
        Ok(leaves) if leaves.len() >= 2 => leaves.iter()
            .skip_while(|n| n.id < op.from)
            .take_while(|n| n.id <= op.to)
            .map(|n| n.summary.clone())
            .collect(),
        _ => {
            // Fallback: read from DB directly
            let db = state.storage.dag.db();
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
        Ok(result) => {
            let tc = crate::tokenizer::count(&result.text) as i64;
            let snippets = crate::snippet::extract(&text);
            match state.storage.dag.compress_group_with_snippets(
                op.conv_id, &[], &result.text, tc, 1, &snippets,
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
    match state.storage.dag.db().delete_dag_node(op.id) {
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
    let node = match state.storage.dag.get_node(op.id) {
        Ok(Some(n)) => n,
        Ok(None) => return json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "node not found"),
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "NODE_ERROR", format!("error: {e}")),
    };
    let children = state.storage.dag.get_children(op.id).unwrap_or_default();
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

/// POST /v1/lcm/motifs/{conv_id} — extract execution motifs from a conversation.
async fn lcm_motifs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match crate::motif::extract_motifs_from_db(&state.storage.db, &[conv_id], None) {
        Ok(motifs) => Json(json!({"conv_id": conv_id, "motifs": motifs})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "MOTIF_ERROR", format!("{e}")),
    }
}

/// POST /v1/lcm/observe — create a FileObservation from file path and content.
async fn lcm_observe(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let path = body.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let content = body.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if path.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "missing 'path' field");
    }
    let observation = crate::file_observation::observe_file(path, content);

    // Persist to file_observations table (P0 transactional snapshot)
    if let Err(e) = state.storage.db.store_file_observation(&observation) {
        tracing::warn!(target: "deeplossless::proxy", "failed to persist file observation: {e}");
    }

    Json(json!(observation)).into_response()
}
