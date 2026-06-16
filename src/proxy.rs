use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json, Router, routing::{get, post, delete},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use futures::StreamExt;
use futures::FutureExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::metrics;
use crate::protocol::canonical::StreamEvent;
use crate::AppState;

const STREAM_CHANNEL_CAPACITY: usize = 32;
const REASONING_CACHE_CAPACITY: usize = 1024;
type SseChunk = Result<axum::body::Bytes, std::convert::Infallible>;

/// Drop guard that ensures `data: [DONE]\n\n` is sent if the spawned task
/// exits without sending it (e.g., panic, early return).  Uses a cloned tx
/// so it's independent of the original sender's lifetime.
struct DoneGuard {
    tx: Option<tokio::sync::mpsc::Sender<SseChunk>>,
    armed: bool,
}

impl DoneGuard {
    fn new(tx: tokio::sync::mpsc::Sender<SseChunk>) -> Self {
        Self { tx: Some(tx), armed: true }
    }
    fn disarm(&mut self) { self.armed = false; }
}

impl Drop for DoneGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(tx) = self.tx.take() {
                let _ = tx.try_send(Ok::<_, std::convert::Infallible>(
                    axum::body::Bytes::from("data: [DONE]\n\n"),
                ));
            }
        }
    }
}

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

    // DB check — simple ping on a read connection (no writer lock)
    match state.storage.db.ping() {
        Ok(()) => checks["database"] = json!("ok"),
        Err(e) => {
            healthy = false;
            checks["database"] = json!(format!("error: {e}"));
        }
    }

    // Upstream reachability — lightweight HEAD with 3s timeout
    let upstream = state.upstream.trim_end_matches('/');
    match tokio::time::timeout(
        std::time::Duration::from_secs(3),
        state.runtime.client.head(upstream).send(),
    ).await {
        Ok(Ok(resp)) => checks["upstream"] = json!(format!("reachable (http {})", resp.status())),
        Ok(Err(e)) => {
            healthy = false;
            checks["upstream"] = json!(format!("unreachable: {e}"));
        }
        Err(_) => {
            healthy = false;
            checks["upstream"] = json!("timeout");
        }
    }

    // Compactor liveness — ping worker via channel (500ms timeout)
    match tokio::time::timeout(std::time::Duration::from_millis(500), state.compactor.lock()).await {
        Ok(mut compactor) => {
            checks["compactor"] = json!(if compactor.health_ping().await { "ok" } else { "worker unresponsive" });
        }
        Err(_) => {
            healthy = false;
            checks["compactor"] = json!("lock timeout");
        }
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
        if let StreamEvent::ToolCallArgsDelta { index: ai, arguments_delta, .. } = &events[offset + 1] {
            if si == ai {
                let (cname, args_hash) = crate::tool_cache::cache_key(name, arguments_delta);
                match db.tool_cache_get(&cname, &args_hash) {
                    Ok(Some((ref result, _hits))) => {
                        if let Ok(mut c) = cycle.lock() {
                            #[allow(deprecated)]
                            c.record_cache_hit(name);
                        }
                        let transformed = crate::tool_cache::transform_result(name, result);
                        tracing::info!(target: "deeplossless",
                            tool=name, %args_hash, raw_len=result.len(), transformed_len=transformed.len(),
                            "cache hit — intercepting tool call");
                        // Skip FunctionCallArgumentsDone + OutputItemDone if present (from flush())
                        let mut consumed = 2;
                        if offset + 3 < events.len()
                            && matches!(&events[offset + 2], StreamEvent::FunctionCallArgumentsDone { .. })
                            && matches!(&events[offset + 3], StreamEvent::OutputItemDone { .. })
                        {
                            consumed = 4;
                        }
                        return (Some(transformed), consumed);
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
    tx: &tokio::sync::mpsc::Sender<SseChunk>,
    mut assembler: Option<&mut crate::protocol::streaming::StreamAssembler>,
    use_responses_format: bool,
    execution_id: i64,
    conv_id: i64,
    replay_session_id: &str,
    next_seq_no: &mut i64,
) -> anyhow::Result<bool> {
    let mut i = 0;
    while i < events.len() {
        let (cached, consumed) = check_tool_cache(&events, i, &db, cycle);
        if let Some(text) = cached {
            let text_ev = StreamEvent::TextDelta { text };
            let kind = "TextDelta";
            let payload = event_to_payload(&text_ev);
            let sn = *next_seq_no;
            let epoch_ms = crate::execution::next_logical_seq();
            db.store_execution_event_with_replay(
                Some(execution_id),
                kind,
                &payload,
                sn,
                Some(conv_id),
                epoch_ms,
                replay_session_id,
            )?;
            *next_seq_no += 1;
            if use_responses_format {
                if let Some(asm) = assembler.as_mut() {
                    for ev in asm.feed(text_ev) {
                        let sse_line = crate::protocol::streaming::to_responses_sse(&ev);
                        if tx.try_send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                            return Ok(false);
                        }
                    }
                } else {
                    let sse_line = crate::protocol::streaming::to_responses_sse(&text_ev);
                    if tx.try_send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                        return Ok(false);
                    }
                }
            } else {
                let sse_line = crate::protocol::streaming::to_chat_completions_sse(&text_ev);
                if tx.try_send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
                    return Ok(false);
                }
            }
            i += consumed;
            continue;
        }
        // Store event before emission. Replay state must not silently diverge
        // from bytes sent downstream.
        let ev = &events[i];
        let kind = event_kind_name(ev);
        let payload = event_to_payload(ev);
        let sn = *next_seq_no;
        let epoch_ms = crate::execution::next_logical_seq();
        db.store_execution_event_with_replay(
            Some(execution_id),
            &kind,
            &payload,
            sn,
            Some(conv_id),
            epoch_ms,
            replay_session_id,
        )?;
        *next_seq_no += 1;
        // Normal emission
        let sse_line = if use_responses_format {
            crate::protocol::streaming::to_responses_sse(ev)
        } else {
            crate::protocol::streaming::to_chat_completions_sse(ev)
        };
        if tx.try_send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(sse_line))).is_err() {
            return Ok(false);
        }
        i += 1;
    }
    Ok(true)
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
        .route("/anthropic/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{response_id}", get(responses_retrieve))
        .route("/v1/lcm/grep/{conv_id}", get(lcm_grep_by_id))
        .route("/v1/lcm/grep", get(lcm_grep_by_fingerprint))
        .route("/v1/lcm/current", get(lcm_current_conv))
        .route("/v1/lcm/sessions", get(lcm_sessions_list))
        .route("/v1/lcm/sessions/{id}/events", get(lcm_session_events))
        .route("/v1/lcm/sessions/{id}/patches", get(lcm_session_patches))
        .route("/v1/lcm/sessions/{id}/system-prompt", get(lcm_session_system_prompt))
        .route("/v1/lcm/sessions/{id}/context-pressure", get(lcm_context_pressure))
        .route("/v1/lcm/latency", get(lcm_latency_records))
        .route("/v1/lcm/latency/summary", get(lcm_latency_summary))
        .route("/v1/lcm/cache/stability", get(lcm_cache_stability))
        .route("/v1/lcm/expand/{node_id}", get(lcm_expand))
        .route("/v1/lcm/status/{conv_id}", get(lcm_status))
        .route("/v1/lcm/snippets/{node_id}", get(lcm_snippets))
        .route("/v1/lcm/similar/{hash}", get(lcm_similar))
        .route("/v1/lcm/similar", get(lcm_similar_missing_hash))
        .route("/v1/lcm/trace/{node_id}", get(lcm_trace))
        .route("/v1/lcm/global/search", get(lcm_global_search))
        .route("/v1/lcm/execution/search", get(lcm_execution_search))
        .route("/v1/lcm/search", get(lcm_search_events))
        .route("/v1/lcm/diffs", get(lcm_diffs_list))
        .route("/v1/lcm/diffs/reconstruct", get(lcm_diff_reconstruct))
        .route("/v1/lcm/diffs/overlaps", get(lcm_diff_overlaps))
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
        .route("/v1/lcm/cache", delete(lcm_cache_delete))
        .route("/v1/lcm/failure", post(lcm_failure_put))
        .route("/v1/lcm/plan", post(lcm_plan_put))
        .route("/v1/lcm/plan/{conv_id}", get(lcm_plan_get))
        .route("/v1/lcm/plan", delete(lcm_plan_delete))
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
    let session_key = prompt_cache_key.to_string();
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
    // Before converting, load session history for conversation continuity.
    // Session store has the full conversation in Chat Completions format;
    // prepend it so DeepSeek sees the assistant's previous responses.
    if let Some(session_msgs) = state.storage.session_store.get(&session_key) {
        if !session_msgs.is_empty() {
            let wrapped = serde_json::json!({"messages": session_msgs});
            let session_canonical = crate::protocol::chat_completions::request_from_chat(&wrapped);
            // Replace with session history
            canonical.messages = session_canonical.messages;
            // Append tool results and new user message from Codex's raw input
            // (these are NOT duplicates — they're the execution results from the
            // previous turn's tool calls, followed by the user's next question).
            if let Some(input_arr) = req_body["input"].as_array() {
                let mut append_msgs: Vec<crate::protocol::canonical::Message> = Vec::new();
                for item in input_arr {
                    let item_type = item["type"].as_str().unwrap_or("");
                    if item_type == "function_call_output" {
                        // Deduplicate: skip if this call_id already exists in session
                        let call_id = item["call_id"].as_str()
                            .or_else(|| item["id"].as_str())
                            .unwrap_or("").to_string();
                        if call_id.is_empty() { continue; }
                        let already_in_session = canonical.messages.iter().any(|m|
                            m.role == crate::protocol::canonical::Role::Tool
                            && m.meta.as_ref().and_then(|meta| meta.tool_call_id.as_ref())
                                .map(|id| *id == call_id).unwrap_or(false)
                        );
                        if already_in_session {
                            tracing::debug!(target: "deeplossless", %call_id, "skipping duplicate tool message");
                            continue;
                        }
                        let output = item["output"].as_str().unwrap_or("").to_string();
                        append_msgs.push(crate::protocol::canonical::Message {
                            role: crate::protocol::canonical::Role::Tool,
                            parts: vec![crate::protocol::canonical::ContentPart::ToolResult {
                                call_id, content: output,
                            }],
                            meta: Some(crate::protocol::canonical::MessageMeta {
                                tool_call_id: item["call_id"].as_str().map(|s| s.to_string()),
                                tool_calls: vec![],
                            }),
                            reasoning: None,
                        });
                    }
                }
                // Also append the last user message
                if let Some(last_user) = input_arr.iter().rev()
                    .find(|item| item["role"].as_str() == Some("user"))
                    .and_then(|item| item["content"].as_array())
                    .and_then(|blocks| blocks.iter().find_map(|b| b["text"].as_str()))
                {
                    append_msgs.push(crate::protocol::canonical::Message {
                        role: crate::protocol::canonical::Role::User,
                        parts: vec![crate::protocol::canonical::ContentPart::Text {
                            text: last_user.to_string(),
                        }],
                        meta: None,
                        reasoning: None,
                    });
                }
                canonical.messages.extend(append_msgs);
            }

            // If the session ends with an incomplete assistant(tc) — no tool results
            // available yet — remove it to avoid orphaned tool_calls errors from DeepSeek.
            if let Some(msg) = canonical.messages.last() {
                if msg.role == crate::protocol::canonical::Role::Assistant {
                    let has_tc = msg.meta.as_ref()
                        .map(|m| !m.tool_calls.is_empty()).unwrap_or(false);
                    if has_tc {
                        let last_tc_idx = canonical.messages.len() - 1;
                        // Count tool messages after this assistant
                        let tool_count = canonical.messages[last_tc_idx + 1..].iter()
                            .filter(|m| m.role == crate::protocol::canonical::Role::Tool)
                            .count();
                        let tc_count = msg.meta.as_ref().map(|m| m.tool_calls.len()).unwrap_or(0);
                        if tool_count < tc_count {
                            canonical.messages.pop();
                            tracing::debug!(target: "deeplossless",
                                tc_count, tool_count,
                                "removed incomplete assistant(tc) from session");
                        }
                    }
                }
            }
            tracing::debug!(target: "deeplossless",
                session_len=session_msgs.len(),
                "loaded session history for continuity");
        }
    }

    let chat_body = crate::protocol::chat_completions::request_to_chat(&canonical);

    // 3. Run the chat pipeline (DAG context injection, message persistence).
    //    Gated by !state.no_pipeline so ACES torture mode — where the
    //    upstream IS the ACES mock and pipeline.process would call the
    //    summarizer (also pointed at the mock) and produce SSE parse
    //    errors — stays clean. The same gate exists on chat_completions
    //    and anthropic handlers; this site was missed when the responses
    //    handler was added, which is why Codex-mode runs logged
    //    "summarizer: response parse error" on every request.
    let mut chat_body_val: serde_json::Value = chat_body.clone();
    // Embed prompt_cache_key for pipeline reasoning injection (stable session key)
    chat_body_val["prompt_cache_key"] = serde_json::json!(prompt_cache_key);
    let (injected, stream_conv_id) = if state.no_pipeline {
        (chat_body_val, 0)
    } else {
        let pipeline = crate::pipeline::ChatPipeline::new(&state);
        match pipeline.process(&canonical.model, &chat_body_val).await {
            Ok(out) => (out.injected_body, out.conv_id),
            Err(e) => {
                warn!("pipeline error: {e}, falling back to passthrough");
                (chat_body_val, 0)
            }
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
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: {\"type\":\"response.output_text.delta\",\"delta\":\"[dry-run] request saved to ~/.deeplossless/translated.json\"}\n\n")
        )).await;
        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: [DONE]\n\n")
        )).await;
        let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
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
    if let Some(msgs) = injected["messages"].as_array() {
        tracing::debug!(target: "deeplossless", "upstream messages:");
        for (i, m) in msgs.iter().enumerate() {
            let role = m["role"].as_str().unwrap_or("?");
            let has_tc = m.get("tool_calls").is_some();
            let tci = m["tool_call_id"].as_str().unwrap_or("-");
            let rc = m["reasoning_content"].as_str().unwrap_or("-").chars().take(30).collect::<String>();
            let content = m["content"].as_str().unwrap_or("").chars().take(60).collect::<String>();
            tracing::debug!(target: "deeplossless", "  msg[{i}] role={role} rc={rc} tc={has_tc} tci={tci} content={content}");
        }
    }
    // Save input messages to session store for conversation continuity.
    // The assistant response will be appended after streaming completes.
    let session_input_msgs = injected["messages"].as_array().cloned().unwrap_or_default();
    if !session_input_msgs.is_empty() {
        state.storage.session_store.replace(&session_key, session_input_msgs.clone());
        tracing::debug!(target: "deeplossless", session_key, msg_count=session_input_msgs.len(), "saved session messages");
    }
    let upstream_start = std::time::Instant::now();
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
            let latency = upstream_start.elapsed().as_millis() as u64;
            let code: u16 = r.status().as_u16();
            metrics::record_latency("responses", code, Some(code), latency, None);
            tracing::debug!(target: "deeplossless", status=%r.status(),
                content_type=?r.headers().get("content-type"),
                latency_ms=latency,
                "upstream response received");
            r
        }
        Err(e) => {
            let latency = upstream_start.elapsed().as_millis() as u64;
            metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics::record_latency("responses", 502, None, latency, Some(format!("{e}")));
            let _ = state.storage.db.insert_event_simple(
                crate::event_store::EventType::Error,
                &session_key,
                &format!("{e}"),
                serde_json::json!({"source": "responses"}),
            );
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
        }
    };

    // 5. Handle non-200 upstream status — propagate error, don't wrap in SSE
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let _ = state.storage.db.insert_event_simple(
            crate::event_store::EventType::Error,
            &session_key,
            &body,
            serde_json::json!({"status_code": status.as_u16(), "source": "responses"}),
        );
        return json_error(status, "UPSTREAM_ERROR", body);
    }

    // 6. Handle streaming vs non-streaming
    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let stream = ReceiverStream::new(rx);
        let response_store = state.storage.response_store.clone();
        let session_store = state.storage.session_store.clone();
        let db = state.storage.db.clone();
        let cycle = state.runtime.cycle.clone();
        // Always store the response for previous_response_id continuity,
        // regardless of the store flag (which only controls GET retrieval).
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
        // Compute reasoning storage key: session (prompt_cache_key) + model.
        // Stable across all turns of the same session.
        let reasoning_key = format!("reasoning:{}:{}", canonical.model, session_key);
        let shutdown = state.runtime.shutdown_notify.clone();
        let record_dir = state.record.clone();
        let record_body = if record_dir.is_some() { Some(body.clone()) } else { None };
        if let Some(ref d) = record_dir {
            tracing::info!(target: "deeplossless::record", dir=%d, "response recording enabled");
        }
        let stream_execution_id = crate::execution::next_logical_seq();
        let replay_session_id = format!("rs_{stream_execution_id}");
        let response_replay_session_id = replay_session_id.clone();
        tokio::spawn(async move {
            let req_start = std::time::Instant::now();
            let mut replay_seq_no: i64 = 0;
            let mut _guard = DoneGuard::new(tx.clone());
            if shutdown.notified().now_or_never().is_some() {
                _guard.disarm();
                return;
            }
            // Protocol recorder: write raw request body
            let _rec = record_dir.as_ref().map(|dir| {
                let _ = std::fs::create_dir_all(dir);
                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                if let Some(body) = &record_body {
                    let req_path = format!("{dir}/req_{ts}.json");
                    match std::fs::write(&req_path, serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(body).unwrap_or_default()).unwrap_or_default()) {
                        Ok(()) => tracing::debug!(target: "deeplossless::record", path=%req_path, "recorded request"),
                        Err(e) => tracing::warn!(target: "deeplossless::record", path=%req_path, error=%e, "failed to record request"),
                    }
                }
                (dir.clone(), ts)
            });
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
            )))).await;
            // response.in_progress (same full envelope)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.in_progress\ndata: {{\"type\":\"response.in_progress\",\"response\":{{{}}}}}\n\n",
                response_envelope("in_progress")
            )))).await;
            // output_item.added (status=in_progress)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}}}\n\n"
            )))).await;
            // content_part.added (with item_id + annotations)
            let _ = tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                "event: response.content_part.added\ndata: {{\"type\":\"response.content_part.added\",\"item_id\":\"{msg_id}\",\"output_index\":0,\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}}}\n\n"
            )))).await;
            tracing::debug!(target: "deeplossless::stream", resp_id, msg_id, "sent stream preamble");

            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut usage_buf: Option<serde_json::Value> = None;
            let mut first_chunk = true;
            let mut all_bytes: Vec<u8> = Vec::new();
            let mut assembler = crate::protocol::streaming::StreamAssembler::new();
            let mut flushed_tool_calls: Vec<(String, String, String)> = Vec::new();
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        all_bytes.extend_from_slice(&c);
                        let s = String::from_utf8_lossy(&c);
                        if first_chunk {
                            first_chunk = false;
                            tracing::debug!(target: "deeplossless",
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
                                        let mut events = assembler.flush();
                                        // Remap tool call output_index from 0 (DeepSeek index)
                                        // to 1+ (beyond the text message at output_index: 0)
                                        {
                                            let mut tc_idx = 1usize;
                                            for ev in &mut events {
                                                match ev {
                                                    StreamEvent::ToolCallStart { index, .. }
                                                    | StreamEvent::ToolCallArgsDelta { index, .. }
                                                    | StreamEvent::FunctionCallArgumentsDone { output_index: index, .. }
                                                    | StreamEvent::OutputItemDone { index, .. } => {
                                                        *index = tc_idx;
                                                    }
                                                    _ => {}
                                                }
                                                if matches!(ev, StreamEvent::OutputItemDone { .. }) {
                                                    tc_idx += 1;
                                                }
                                            }
                                        }
                                        for ev in &events {
                                            if let StreamEvent::FunctionCallArgumentsDone { call_id, name, arguments, .. } = ev {
                                                flushed_tool_calls.push((call_id.clone(), name.clone(), arguments.clone()));
                                            }
                                        }
                                        match process_events(
                                            events,
                                            db.clone(),
                                            &cycle,
                                            &tx,
                                            Some(&mut assembler),
                                            true,
                                            stream_execution_id,
                                            stream_conv_id,
                                            &replay_session_id,
                                            &mut replay_seq_no,
                                        ) {
                                            Ok(true) => {}
                                            Ok(false) => break,
                                            Err(e) => {
                                                warn!("execution event store failed: {e}");
                                                break;
                                            }
                                        }
                                        continue;
                                    }
                                    let events = assembler.feed(event);
                                    match process_events(
                                        events,
                                        db.clone(),
                                        &cycle,
                                        &tx,
                                        Some(&mut assembler),
                                        true,
                                        stream_execution_id,
                                        stream_conv_id,
                                        &replay_session_id,
                                        &mut replay_seq_no,
                                    ) {
                                        Ok(true) => {}
                                        Ok(false) => break,
                                        Err(e) => {
                                            warn!("execution event store failed: {e}");
                                            break;
                                        }
                                    }
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
                        if let Err(e) = process_events(
                            events,
                            db.clone(),
                            &cycle,
                            &tx,
                            Some(&mut assembler),
                            true,
                            stream_execution_id,
                            stream_conv_id,
                            &replay_session_id,
                            &mut replay_seq_no,
                        ) {
                            warn!("execution event store failed: {e}");
                        }
                    }
                }
            }
            // Upstream [DONE] → finish assembly, get accumulated content
            let (content, mut final_events) = assembler.finish();
            // Send any remaining flush events (e.g., if the stream ended
            // without triggering the Done handler) with proper index remapping.
            if !final_events.is_empty() {
                let mut tc_idx = 1usize;
                for ev in &mut final_events {
                    match ev {
                        StreamEvent::ToolCallStart { index, .. }
                        | StreamEvent::ToolCallArgsDelta { index, .. }
                        | StreamEvent::FunctionCallArgumentsDone { output_index: index, .. }
                        | StreamEvent::OutputItemDone { index, .. } => {
                            *index = tc_idx;
                        }
                        _ => {}
                    }
                    if matches!(ev, StreamEvent::OutputItemDone { .. }) {
                        tc_idx += 1;
                    }
                }
                if let Err(e) = process_events(
                    final_events,
                    db.clone(),
                    &cycle,
                    &tx,
                    Some(&mut assembler),
                    true,
                    stream_execution_id,
                    stream_conv_id,
                    &replay_session_id,
                    &mut replay_seq_no,
                ) {
                    warn!("execution event store failed: {e}");
                }
            }
            let input_tokens = usage_buf.as_ref()
                .and_then(|v| v["usage"]["prompt_tokens"].as_u64())
                .unwrap_or(0);
            let output_tokens = usage_buf.as_ref()
                .and_then(|v| v["usage"]["completion_tokens"].as_u64())
                .unwrap_or(0);
            let output_text_len = content.text.len();
            tracing::debug!(target: "deeplossless",
                input_tokens, output_tokens, output_text_len,
                "turn complete");

            // Write session log if --log-dir is set
            if log_dir.is_some() {
                let cache_hits = cycle.lock().ok().map(|c| c.metrics.cache_hits).unwrap_or(0);
                write_log(log_dir.as_deref(), &LogEntry {
                    ts: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f").to_string(),
                    endpoint: "responses",
                    model: log_model,
                    request_body_kb: log_request_body_kb,
                    input_tokens_est: input_tokens as usize,
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

            // Store reasoning for multi-turn continuity (DeepSeek thinking mode
            // requires reasoning_content to be echoed back on tool-call assistants).
            if !content.reasoning.is_empty() {
                if let Err(e) = db.store_reasoning(&reasoning_key, &content.reasoning) {
                    tracing::debug!(target: "deeplossless", "store reasoning failed: {e}");
                }
                let _ = db.insert_event_simple(
                    crate::event_store::EventType::Reasoning,
                    &session_key,
                    &content.reasoning,
                    serde_json::json!({"model": canonical.model}),
                );
            }

            // Append assistant response to session for conversation continuity.
            // Build in Chat Completions format matching injected["messages"].
            let mut assistant_msg = serde_json::json!({"role": "assistant", "content": content.text});
            if !content.reasoning.is_empty() {
                assistant_msg["reasoning_content"] = serde_json::json!(content.reasoning);
            }
            if !flushed_tool_calls.is_empty() {
                let tcs: Vec<serde_json::Value> = flushed_tool_calls.iter().map(|(id, name, args)| {
                    serde_json::json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}})
                }).collect();
                assistant_msg["tool_calls"] = serde_json::json!(tcs);
            }
            // Load current session, append assistant response, save back
            let mut session = session_store.get(&session_key).unwrap_or_default();
            session.push(assistant_msg);
            session_store.replace(&session_key, session);
            tracing::debug!(target: "deeplossless", session_key, "appended assistant response to session");

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
            // content_index 0 matches content_part.added (always announces text at index 0)
            let content_part_done = serde_json::json!({
                "type": "response.content_part.done", "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": text_part
            });
            let resp_status = if usage_buf.is_some() { "completed" } else { "incomplete" };
            let usage_json = usage_buf.map(|v| serde_json::json!({
                "input_tokens": v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                "output_tokens": v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                "total_tokens": v["usage"]["total_tokens"].as_u64().unwrap_or(0),
            })).unwrap_or(serde_json::json!({"input_tokens":0,"output_tokens":0,"total_tokens":0}));
            // Build response.completed output. Only include the text message item;
            // function_call items were already emitted via SSE during flush().
            // Including them again in response.completed causes Codex to see
            // duplicates when it processes both SSE events and the final payload.
            let output_items: Vec<serde_json::Value> = vec![item_json.clone()];
            let completed = serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": resp_id, "object": "response", "created_at": now,
                    "status": resp_status, "model": model,
                    "output": output_items,
                    "parallel_tool_calls": true,
                    "usage": usage_json
                }
            });

            // Emit lifecycle events in correct order
            let output_text_done = serde_json::json!({
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": content.text
            });
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.output_text.done\ndata: {output_text_done}\n\n"))
            )).await;
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.content_part.done\ndata: {content_part_done}\n\n"))
            )).await;
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.output_item.done\ndata: {output_item_done}\n\n"))
            )).await;
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("event: response.completed\ndata: {completed}\n\n"))
            )).await;
            // Persist the response so GET /v1/responses/{id} returns real data,
            // and Codex's previous_response_id continuity can work incrementally.
            // Reasoning is stored by the reasoning_key block above.

            // Always store for previous_response_id continuity.
            // Compute tool calls for storage, filtering out cache-intercepted ones
            let effective_tool_calls: Vec<_> = flushed_tool_calls.iter().filter(|(_, name, arguments)| {
                let (cname, args_hash) = crate::tool_cache::cache_key(name, arguments);
                db.tool_cache_get(&cname, &args_hash).ok().flatten().is_none()
            }).cloned().collect();
            let mut stored_output: Vec<serde_json::Value> = Vec::new();
            for (call_id, name, arguments) in &effective_tool_calls {
                stored_output.push(serde_json::json!({
                    "type": "function_call", "call_id": call_id, "name": name, "arguments": arguments, "status": "completed"
                }));
            }
            stored_output.push(serde_json::json!({
                "id": msg_id, "type": "message", "status": "completed",
                "role": "assistant",
                "content": content_parts
            }));
            let resp_obj = serde_json::json!({
                "id": resp_id, "object": "response", "created_at": now,
                "status": resp_status, "model": model,
                "output": stored_output, "usage": usage_json,
                "prompt_cache_key": &session_key
            });
            response_store.insert(resp_id.clone(), resp_obj);
            tracing::debug!(target: "deeplossless",
                resp_id, text_len=content.text.len(),
                "response stored");
            // Write recorded upstream bytes
            if let Some((ref dir, ts)) = _rec {
                let rsp_path = format!("{dir}/rsp_{ts}.txt");
                match std::fs::write(&rsp_path, &all_bytes) {
                    Ok(()) => tracing::debug!(target: "deeplossless::record", path=%rsp_path, len=all_bytes.len(), "recorded response"),
                    Err(e) => tracing::warn!(target: "deeplossless::record", path=%rsp_path, error=%e, "failed to record response"),
                }
            }
            // Transport-level EOF marker
            _guard.disarm();
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from("data: [DONE]\n\n")
            )).await;
            let _ = db.insert_event_simple(
                crate::event_store::EventType::RequestEnd,
                &session_key,
                "",
                serde_json::json!({"duration_ms": req_start.elapsed().as_millis(), "source": "responses"}),
            );
            // tx dropped here → stream closes
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert("content-type", "text/event-stream; charset=utf-8".parse().expect("static header"));
        response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header"));
        response.headers_mut().insert("connection", "close".parse().expect("static header"));
        response.headers_mut().insert(
            "x-deeplossless-execution-id",
            stream_execution_id.to_string().parse().expect("numeric header"),
        );
        response.headers_mut().insert(
            "x-deeplossless-replay-session-id",
            response_replay_session_id.parse().expect("ascii header"),
        );
        response
    } else {
        // Non-streaming: translate Chat Completions response → Responses format
        match resp.bytes().await {
            Ok(bytes) => {
                // Record raw upstream response
                if let Some(ref dir) = state.record {
                    let _ = std::fs::create_dir_all(dir);
                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let req_path = format!("{dir}/req_{ts}.json");
                    match std::fs::write(&req_path, serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default()).unwrap_or_default()) {
                        Ok(()) => tracing::info!(target: "deeplossless::record", path=%req_path, "recorded non-streaming request"),
                        Err(e) => tracing::warn!(target: "deeplossless::record", path=%req_path, error=%e, "failed to record non-streaming request"),
                    }
                    let rsp_path = format!("{dir}/rsp_{ts}.txt");
                    match std::fs::write(&rsp_path, &bytes) {
                        Ok(()) => tracing::debug!(target: "deeplossless::record", path=%rsp_path, len=bytes.len(), "recorded non-streaming response"),
                        Err(e) => tracing::warn!(target: "deeplossless::record", path=%rsp_path, error=%e, "failed to record non-streaming response"),
                    }
                }
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
    json_error(
        StatusCode::NOT_FOUND,
        "NOT_FOUND",
        format!("response '{response_id}' not found; response continuity is ephemeral and may be lost after restart or FIFO eviction"),
    )
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

    // Pure passthrough: zero processing. Forward upstream URL, method, body, headers.
    // For isolating protocol bugs — when this works but the full pipeline doesn't.
    if state.passthrough && streaming {
        let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
        let resp = state.runtime.client.post(&upstream_url)
            .header("Authorization", format!("Bearer {}", get_cached_key(&state.api_key)))
            .header("Content-Type", "application/json")
            .body(body.clone())
            .send().await;
        match resp {
            Ok(r) => {
                let status = r.status();
                let headers = r.headers().clone();
                let byte_stream = r.bytes_stream();
                let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
                tokio::spawn(async move {
                    let mut stream = byte_stream;
                    while let Some(chunk) = stream.next().await {
                        if let Ok(c) = chunk {
                            if tx.send(Ok::<_, std::convert::Infallible>(c)).await.is_err() { break; }
                        }
                    }
                });
                let stream = ReceiverStream::new(rx);
                let mut response = Response::new(Body::from_stream(stream));
                *response.status_mut() = status;
                for (k, v) in headers.iter() {
                    if k != "transfer-encoding" {
                        response.headers_mut().insert(k.clone(), v.clone());
                    }
                }
                return response;
            }
            Err(e) => {
                metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("{e}"));
            }
        }
    }

    // System prompt normalization: strip timestamps/UUIDs to preserve
    // DeepSeek prefix cache hit rate.  Timestamps in system prompts break
    // cache because every request has a different prefix.
    let mut cache_normalized = false;
    let body_for_upstream = if state.cache_normalize {
        let normalized = normalize_system_prompt(req_body.clone());
        cache_normalized = normalized != req_body;
        normalized
    } else {
        req_body.clone()
    };

    // LCM context injection (--lcm-context-tokens, default 500)
    let lcm_budget = req_body.get("lcm_max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(state.lcm_context_tokens)
        .clamp(0, 8000) as usize;
    let body_for_upstream = if lcm_budget > 0 {
        let mut injected = body_for_upstream.clone();
        let msgs = req_body["messages"].as_array().cloned().unwrap_or_default();
        let fp = crate::session::fingerprint(msgs.as_slice(), 3);
        if let Ok(Some(cid)) = state.storage.db.find_conversation_by_fingerprint(&fp) {
            let query = msgs.iter().rev().find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str());
            if let Ok(nodes) = state.storage.dag.assemble_context(cid, lcm_budget, query) {
                if !nodes.is_empty() {
                    let ctx_text = crate::pipeline::render_dag_context(&nodes);
                    if let Some(arr) = injected["messages"].as_array_mut() {
                        if let Some(last_user) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                            let original = last_user["content"].as_str().unwrap_or("").to_string();
                            last_user["content"] = json!(if original.is_empty() {
                                ctx_text
                            } else {
                                format!("{ctx_text}\n\n{original}")
                            });
                        }
                    }
                }
            }
        }
        injected
    } else {
        body_for_upstream
    };

    // Pipeline: always run for storage. Only modify request body when
    // --lcm-context is enabled (context appended as user message).
    let injected_body = if state.lcm_context && !state.no_pipeline {
        let pipeline = crate::pipeline::ChatPipeline::new(&state);
        match pipeline.process(model, &body_for_upstream).await {
            Ok(out) => out.injected_body,
            Err(e) => {
                warn!("pipeline error: {e}, falling back to passthrough");
                body_for_upstream.clone()
            }
        }
    } else {
        if !state.no_pipeline {
            let pipeline = crate::pipeline::ChatPipeline::new(&state);
            let body = body_for_upstream.clone();
            let model_owned = model.to_string();
            tokio::task::spawn(async move {
                if let Err(e) = pipeline.process(&model_owned, &body).await {
                    tracing::warn!(target: "deeplossless::pipeline", "background pipeline failed: {e}");
                }
            });
        }
        body_for_upstream
    };

    // Resolve conversation ID for response header — lightweight fingerprint lookup
    let conv_id: Option<i64> = {
        let msgs_arr = req_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        let fp = crate::session::fingerprint(msgs_arr, 3);
        state.storage.db.find_conversation_by_fingerprint(&fp).ok().flatten()
    };

    if let Some(cid) = conv_id {
        track_cache_stability(&state, cid, &req_body);
    }

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let upstream_start = std::time::Instant::now();
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
            let latency = upstream_start.elapsed().as_millis() as u64;
            metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics::record_latency("chat_completions", 502, None, latency, Some(format!("{e}")));
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
        }
    };

    let status = resp.status();
    let latency = upstream_start.elapsed().as_millis() as u64;
    let code: u16 = status.into();
    metrics::record_latency("chat_completions", code, Some(code), latency, None);
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let fp = crate::session::fingerprint(req_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]), 3);
        let _ = state.storage.db.insert_event_simple(
            crate::event_store::EventType::Error,
            &fp,
            &body,
            serde_json::json!({"status_code": code, "source": "chat_completions_lcm"}),
        );
        return json_error(status, "UPSTREAM_ERROR", body);
    }
    let content_type = resp
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().expect("static header parse"));

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let stream = ReceiverStream::new(rx);
        // Protocol recorder — raw bytes, no parsing. Compare with direct DeepSeek.
        let record_dir = state.record.clone();
        let record_body = if record_dir.is_some() { Some(req_body.clone()) } else { None };
        // Capture reasoning_content for multi-turn continuity.
        let reasoning_db = state.storage.db.clone();
        let session_fingerprint = {
            let msgs = req_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
            crate::session::fingerprint(msgs, 3)
        };
        let reasoning_key = {
            let msgs = req_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
            let last_user = msgs.iter().rev().find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str()).unwrap_or("");
            let model = req_body["model"].as_str().unwrap_or("");
            format!("reasoning:{model}:{}", last_user.chars().take(80).collect::<String>())
        };
        tokio::spawn(async move {
            let req_start = std::time::Instant::now();
            // Protocol recorder: write raw request/response for diffing
            let _rec = record_dir.as_ref().map(|dir| {
                let _ = std::fs::create_dir_all(dir);
                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                if let Some(body) = &record_body {
                    let _ = std::fs::write(format!("{dir}/req_{ts}.json"), serde_json::to_string_pretty(body).unwrap_or_default());
                }
                (dir.clone(), ts)
            });
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut all_bytes: Vec<u8> = Vec::new();
            let mut prompt_tokens: u64 = 0;
            let mut completion_tokens: u64 = 0;
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        all_bytes.extend_from_slice(&c);
                        if tx.send(Ok::<_, std::convert::Infallible>(c.clone())).await.is_err() {
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
                                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                                        content.push_str(c);
                                    }
                                    if let Some(u) = v["usage"].as_object() {
                                        prompt_tokens = u.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
                                        completion_tokens = u.get("completion_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
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
                let _ = reasoning_db.insert_event_simple(
                    crate::event_store::EventType::Reasoning,
                    &reasoning_key,
                    &reasoning,
                    serde_json::json!({"source": "chat_completions"}),
                );
            }
            // Accumulate token usage from chat completions stream
            if let Some(cid) = conv_id {
                if prompt_tokens > 0 || completion_tokens > 0 {
                    let usage_db = state.storage.db.clone();
                    tokio::task::spawn_blocking(move || {
                        let _ = usage_db.accumulate_usage(cid, prompt_tokens, completion_tokens);
                    });
                }
            }
            // Save assistant response message from streaming output
            if let Some(cid) = conv_id {
                if !content.is_empty() {
                    let msg_db = state.storage.db.clone();
                    let msg_content = content.clone();
                    tokio::task::spawn_blocking(move || {
                        let assistant_msgs = serde_json::json!([{
                            "role": "assistant",
                            "content": msg_content
                        }]);
                        if let Err(e) = msg_db.store_messages(cid, &assistant_msgs) {
                            tracing::warn!(target: "deeplossless::stream", "failed to store assistant message: {e}");
                        }
                    });
                }
            }
            // Write recorded response bytes
            if let Some((ref dir, ts)) = _rec {
                let _ = std::fs::write(format!("{dir}/rsp_{ts}.txt"), &all_bytes);
            }
            let _ = reasoning_db.insert_event_simple(
                crate::event_store::EventType::RequestEnd,
                &session_fingerprint,
                "",
                serde_json::json!({"duration_ms": req_start.elapsed().as_millis(), "source": "chat_completions"}),
            );
            tracing::debug!(target: "deeplossless::stream",
                chunk_count=reasoning.len(), "STREAM CLOSED — task dropped, channel sender dropped, body EOF");
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert("content-type", content_type);
        if let Some(cid) = conv_id {
            response.headers_mut().insert("x-deeplossless-conv", cid.to_string().parse().expect("static header"));
        }
        if cache_normalized {
            let _ = response.headers_mut().insert("x-lcm-normalized", "1".parse().expect("static"));
        }
        if !state.no_header_mod {
            response.headers_mut().insert("cache-control", "no-cache".parse().expect("static header parse"));
            response.headers_mut().insert("x-accel-buffering", "no".parse().expect("static header parse"));
        }
        response
    } else {
        match resp.bytes().await {
            Ok(bytes) => {
                let mut response = Response::new(Body::from(bytes));
                *response.status_mut() = status;
                response.headers_mut().insert("content-type", content_type);
                if let Some(cid) = conv_id {
                    response.headers_mut().insert("x-deeplossless-conv", cid.to_string().parse().expect("static header"));
                }
                response.headers_mut().insert("x-deeplossless-lcm", "GET /v1/lcm/grep/{conv_id}?query= — search past context; GET /v1/lcm/cache?tool=&args= — check tool cache; GET /v1/lcm/status/{conv_id} — DAG health".parse().expect("static header"));
                response
            }
            Err(e) => {
                metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
            }
        }
    }
}

/// POST /v1/lcm/chat/completions — chat completions with automatic context injection.
/// Same interface as /v1/chat/completions, but assembles DAG context and merges it
#[allow(dead_code)]
/// into the last user message before forwarding to upstream. Reports context token
/// usage in the response via custom header `x-lcm-context-tokens`.
async fn lcm_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req_body): Json<serde_json::Value>,
) -> Response {
    // Auth
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }

    let model = req_body["model"].as_str().map(map_model).unwrap_or_else(|| "deepseek-chat".to_string());
    let streaming = req_body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let msgs = req_body["messages"].as_array().cloned().unwrap_or_default();
    if msgs.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "messages array is empty");
    }

    // Run pipeline for storage and context assembly
    let pipeline = crate::pipeline::ChatPipeline::new(&state);
    let (conv_id, context_text, _context_tokens) = match pipeline.process(&model, &req_body).await {
        Ok(out) => {
            // Assemble DAG context for injection
            let dag = state.storage.dag.clone();
            let token_budget = req_body.get("lcm_max_tokens").and_then(|v| v.as_u64()).unwrap_or(500).clamp(0, 8000) as usize;
            let (ctx_text, ctx_tokens) = if token_budget > 0 {
                let query = msgs.iter().rev().find(|m| m["role"] == "user")
                    .and_then(|m| m["content"].as_str());
                match dag.assemble_context(out.conv_id, token_budget, query) {
                    Ok(nodes) if !nodes.is_empty() => {
                        let text = crate::pipeline::render_dag_context(&nodes);
                        let tokens = crate::tokenizer::count(&text);
                        (text, tokens)
                    }
                    _ => (String::new(), 0),
                }
            } else {
                (String::new(), 0)
            };
            (Some(out.conv_id), ctx_text, ctx_tokens)
        }
        Err(e) => {
            tracing::warn!(target: "deeplossless::proxy", "pipeline error: {e}, injecting without storage");
            (None, String::new(), 0)
        }
    };

    // Inject context into the last user message
    let mut injected_body = req_body.clone();
    let mut context_token_count = 0usize;
    if !context_text.is_empty() {
        if let Some(arr) = injected_body["messages"].as_array_mut() {
            if let Some(last_user) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                let original = last_user["content"].as_str().unwrap_or("").to_string();
                let merged = if original.is_empty() {
                    context_text
                } else {
                    format!("{context_text}\n\n{original}")
                };
                context_token_count = crate::tokenizer::count(&merged) - crate::tokenizer::count(last_user["content"].as_str().unwrap_or(""));
                last_user["content"] = serde_json::json!(merged);
            }
        }
    }

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let upstream_start = std::time::Instant::now();
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
            let latency = upstream_start.elapsed().as_millis() as u64;
            metrics::UPSTREAM_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics::record_latency("lcm_chat_completions", 502, None, latency, Some(format!("{e}")));
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("upstream error: {e}"))
        }
    };

    let status = resp.status();
    let latency = upstream_start.elapsed().as_millis() as u64;
    let code: u16 = status.into();
    metrics::record_latency("lcm_chat_completions", code, Some(code), latency, None);
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let _ = state.storage.db.insert_event_simple(
            crate::event_store::EventType::Error,
            &crate::session::fingerprint(&msgs, 3),
            &body,
            serde_json::json!({"status_code": code, "source": "chat_completions"}),
        );
        return json_error(status, "UPSTREAM_ERROR", body);
    }

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let stream = ReceiverStream::new(rx);
        let reasoning_db = state.storage.db.clone();
        let session_fp = crate::session::fingerprint(&msgs, 3);
        let last_user = msgs.iter().rev().find(|m| m["role"] == "user")
            .and_then(|m| m["content"].as_str()).unwrap_or("");
        let reasoning_key = format!("reasoning:{}:{}", &model, last_user.chars().take(80).collect::<String>());
        tokio::spawn(async move {
            let req_start = std::time::Instant::now();
            let mut byte_stream = resp.bytes_stream();
            let mut reasoning = String::new();
            let mut prompt_tokens: u64 = 0;
            let mut completion_tokens: u64 = 0;
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        if tx.send(Ok::<_, std::convert::Infallible>(c.clone())).await.is_err() { break; }
                        let s = String::from_utf8_lossy(&c);
                        for line in s.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" { break; }
                                if let Ok(v) = serde_json::from_str::<Value>(data) {
                                    if let Some(rc) = v["choices"][0]["delta"]["reasoning_content"].as_str() {
                                        reasoning.push_str(rc);
                                    }
                                    if let Some(u) = v["usage"].as_object() {
                                        prompt_tokens = u.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
                                        completion_tokens = u.get("completion_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if !reasoning.is_empty() {
                let _ = reasoning_db.store_reasoning(&reasoning_key, &reasoning);
                let _ = reasoning_db.insert_event_simple(
                    crate::event_store::EventType::Reasoning,
                    &reasoning_key,
                    &reasoning,
                    serde_json::json!({"source": "chat_completions"}),
                );
            }
            if let Some(cid) = conv_id {
                if prompt_tokens > 0 || completion_tokens > 0 {
                    let usage_db = state.storage.db.clone();
                    tokio::task::spawn_blocking(move || {
                        let _ = usage_db.accumulate_usage(cid, prompt_tokens, completion_tokens);
                    });
                }
            }
            let _ = reasoning_db.insert_event_simple(
                crate::event_store::EventType::RequestEnd,
                &session_fp,
                "",
                serde_json::json!({"duration_ms": req_start.elapsed().as_millis(), "source": "chat_completions"}),
            );
        });
        let mut resp = axum::response::Response::new(
            axum::body::Body::from_stream(stream),
        );
        resp.headers_mut().insert("content-type", "text/event-stream".parse().unwrap());
        resp.headers_mut().insert("cache-control", "no-cache".parse().unwrap());
        resp.headers_mut().insert("x-lcm-context-tokens", context_token_count.to_string().parse().unwrap());
        resp
    } else {
        let mut resp = match resp.text().await {
            Ok(body) => {
                (StatusCode::OK, Json(json!(body))).into_response()
            }
            Err(e) => json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("{e}")),
        };
        resp.headers_mut().insert("x-lcm-context-tokens", context_token_count.to_string().parse().unwrap());
        resp
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
    // Allow localhost access — LCM endpoints are local-only tools.
    // Sandboxed agents (OpenClaw, etc.) can't access host env vars for auth.
    let is_local = headers.get("host")
        .and_then(|v| v.to_str().ok())
        .map(|h| h.starts_with("127.") || h.starts_with("localhost") || h.starts_with("[::1]") || h.starts_with("0.0.0.0"))
        .unwrap_or(true); // Missing Host header → assume local
    if is_local {
        return true;
    }

    let admin = state.admin_key.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(admin_key) = admin.as_ref() {
        return check_bearer(headers, admin_key);
    }
    drop(admin);

    let expected = get_cached_key(&state.api_key);
    if expected.is_empty() {
        return true; // No keys configured → allow all
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

/// Record a system prompt fingerprint for cache stability tracking.
/// Keeps the last 20 hashes per conversation.
fn track_cache_stability(state: &AppState, conv_id: i64, body: &serde_json::Value) {
    let system_content = body["messages"].as_array()
        .and_then(|arr| arr.first())
        .filter(|m| m["role"].as_str() == Some("system"))
        .and_then(|m| m["content"].as_str());

    let Some(content) = system_content else { return };
    if content.is_empty() { return }

    use sha2::{Digest, Sha256};
    let hash = format!("{:x}", Sha256::digest(content.as_bytes()));
    let short_hash = &hash[..12];

    let mut tracker = state.cache_stability.lock().unwrap_or_else(|e| e.into_inner());
    let hashes = tracker.entry(conv_id).or_default();
    hashes.push(short_hash.to_string());
    if hashes.len() > 20 {
        hashes.remove(0);
    }
}

/// Strip cache-breaking dynamic content from system prompts.
/// DeepSeek uses prefix-based caching — timestamps, UUIDs, and session IDs
/// in the system prompt make every request a cache miss.
///
/// Only modifies `messages[0]` if its role is "system".
fn normalize_system_prompt(mut body: serde_json::Value) -> serde_json::Value {
    let Some(messages) = body["messages"].as_array_mut() else { return body };
    if messages.is_empty() { return body }
    if messages[0].get("role").and_then(|v| v.as_str()) != Some("system") { return body }
    let Some(original) = messages[0].get("content").and_then(|v| v.as_str()) else { return body };
    if original.is_empty() { return body }

    let cleaned = original.to_string();
    let mut changes = 0u32;

    // Walk through the string and replace common cache-breakers with stable markers.
    // We use character-level scanning to avoid adding a regex dependency.
    let bytes = cleaned.as_bytes();
    let mut out = String::with_capacity(cleaned.len());
    let mut i = 0;
    while i < bytes.len() {
        // Try ISO 8601 timestamp: YYYY-MM-DDTHH:MM:SS...
        if i + 19 <= bytes.len()
            && bytes[i].is_ascii_digit()
            && bytes[i+4] == b'-' && bytes[i+7] == b'-'
            && (bytes[i+10] == b'T' || bytes[i+10] == b' ')
            && bytes[i+13] == b':' && bytes[i+16] == b':'
        {
            // Consume the full timestamp
            let start = i;
            i += 19;
            // Optional fractional seconds
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            }
            // Optional timezone
            if i < bytes.len() && (bytes[i] == b'Z' || bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
                if bytes[i-1] != b'Z' {
                    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b':') { i += 1; }
                }
            }
            out.push_str("[ts]");
            changes += 1;
            tracing::debug!(target: "deeplossless::cache", range=?start..i, "stripped timestamp");
            continue;
        }

        // Try UUID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX (36 chars, hex digits + dashes)
        if i + 36 <= bytes.len()
            && bytes[i].is_ascii_hexdigit()
            && bytes[i+8] == b'-' && bytes[i+13] == b'-'
            && bytes[i+18] == b'-' && bytes[i+23] == b'-'
        {
            let is_uuid = (i..i+36).all(|j| {
                matches!(bytes[j], b'-' | b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
            });
            if is_uuid {
                out.push_str("[uuid]");
                i += 36;
                changes += 2;
                continue;
            }
        }

        out.push(bytes[i] as char);
        i += 1;
    }

    if cleaned != out {
        messages[0]["content"] = serde_json::json!(out);
        tracing::debug!(
            target: "deeplossless::cache",
            before = cleaned.len(),
            after = out.len(),
            changes,
            "system prompt normalized for cache"
        );
    }

    body
}

/// POST /anthropic/v1/messages — Anthropic Messages API → DeepSeek Chat Completions.
/// Translates the request format, forwards to upstream, converts response back.
async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let t0 = std::time::Instant::now();
    // Extract API key from x-api-key header (Anthropic convention) or Authorization
    {
        let mut key = state.api_key.lock().unwrap_or_else(|e| e.into_inner());
        if key.is_none() {
            if let Some(ak) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
                *key = Some(ak.to_string());
            } else if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
                if let Some(bearer) = auth.strip_prefix("Bearer ").or_else(|| auth.strip_prefix("bearer ")) {
                    *key = Some(bearer.to_string());
                }
            }
        }
    }

    // Anthropic format keeps system prompt at top level.
    // Inject it into messages[0] so fingerprint, pipeline, and cache
    // normalization can all see it. This makes conversation IDs stable.
    let mut body = body;
    if let Some(system) = body.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr.iter()
                .filter_map(|b| b["text"].as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            if let Some(arr) = body["messages"].as_array_mut() {
                let existing_system = arr.first()
                    .and_then(|m| m.get("role"))
                    .and_then(|r| r.as_str()) == Some("system");
                if !existing_system {
                    arr.insert(0, json!({"role": "system", "content": system_text}));
                }
            }
        }
        // Remove top-level system to avoid duplication during translation
        body.as_object_mut().and_then(|o| o.remove("system"));
    }

    // Normalize system prompt for cache stability
    let body = if state.cache_normalize {
        normalize_system_prompt(body)
    } else {
        body
    };

    // Compute fingerprint from workspace path (CLI or git root).
    // Workspace identity belongs to the runtime layer, not the prompt.
    let fp = crate::session::fingerprint_anthropic(state.workspace.as_deref());
    // Synchronously create/find conversation so near-simultaneous requests
    // (e.g. Claude Code's stream+non-stream pair) share the same ID.
    let conv_id = state.storage.db.find_or_create_conversation(&fp, "claude").ok();
    if let Some(cid) = conv_id {
        track_cache_stability(&state, cid, &body);
    }

    // Pipeline: store messages and create DAG nodes via ChatPipeline.
    // Reuses the same pipeline as chat_completions — full audit trail,
    // execution events, parallel detection, and token accumulation.
    if !state.no_pipeline {
        let pipeline = crate::pipeline::ChatPipeline::new(&state);
        let body_clone = body.clone();
        let body_fp = fp.clone();
        tokio::task::spawn(async move {
            if let Err(e) = pipeline.process_with_fp("claude", &body_clone, 1, Some(&body_fp)).await {
                tracing::warn!(target: "deeplossless::pipeline", "anthropic pipeline failed: {e}");
            }
        });
    }

    // LCM context injection (honor --no-lcm-context flag)
    let lcm_budget = if state.lcm_context {
        body.get("lcm_max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(state.lcm_context_tokens)
            .clamp(0, 8000) as usize
    } else {
        0
    };
    let mut body = body;
    if lcm_budget > 0 {
        if let Some(cid) = conv_id {
            if let Ok(nodes) = state.storage.dag.assemble_context(cid, lcm_budget, None) {
                if !nodes.is_empty() {
                    let ctx_text = crate::pipeline::render_dag_context(&nodes);
                    if let Some(arr) = body["messages"].as_array_mut() {
                        if let Some(last_user) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                            let original = last_user["content"].as_str().unwrap_or("").to_string();
                            last_user["content"] = json!(if original.is_empty() {
                                ctx_text
                            } else {
                                format!("{ctx_text}\n\n{original}")
                            });
                        }
                    }
                }
            }
        }
    }

    // Translate Anthropic → DeepSeek Chat Completions
    let last_reasoning = {
        let cache = state.reasoning_cache.lock().unwrap_or_else(|e| e.into_inner());
        let rc = cache.get(&fp).cloned();
        tracing::debug!(target: "deeplossless::anthropic",
            fp = %fp,
            has_reasoning = rc.is_some(),
            reasoning_len = rc.as_ref().map(|r| r.len()).unwrap_or(0),
            "reasoning cache lookup");
        rc
    };
    let deepseek_body = crate::protocol::anthropic::request_to_deepseek(&body, last_reasoning.as_deref());
    let streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let overhead_ms = t0.elapsed().as_millis();

    // Forward to upstream
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let body_kb = serde_json::to_string(&deepseek_body).unwrap_or_default().len() as f64 / 1024.0;
    let orig_kb = serde_json::to_string(&body).unwrap_or_default().len() as f64 / 1024.0;
    tracing::debug!(target: "deeplossless::anthropic", orig_kb, translated_kb = body_kb, overhead_ms, "request sizes");
    // Record original Anthropic request body for protocol debugging
    if let Some(ref dir) = state.record {
        let _ = std::fs::create_dir_all(dir);
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        let _ = std::fs::write(format!("{dir}/req_{ts}.json"), serde_json::to_string_pretty(&body).unwrap_or_default());
    }
    let upstream_start = std::time::Instant::now();
    let resp = match state.runtime.client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", get_cached_key(&state.api_key)))
        .header("Content-Type", "application/json")
        .json(&deepseek_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let _ = state.storage.db.insert_event_simple(
                crate::event_store::EventType::Error,
                &fp,
                &format!("{e}"),
                serde_json::json!({"source": "anthropic_messages"}),
            );
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("{e}"))
        }
    };

    let status = resp.status();
    let latency = upstream_start.elapsed().as_millis() as u64;
    metrics::record_latency("anthropic", status.as_u16(), Some(status.as_u16()), latency, None);

    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let _ = state.storage.db.insert_event_simple(
            crate::event_store::EventType::Error,
            &fp,
            &body,
            serde_json::json!({"status_code": status.as_u16(), "source": "anthropic_messages"}),
        );
        return json_error(status, "UPSTREAM_ERROR", body);
    }

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let stream = ReceiverStream::new(rx);
        let reasoning_fp = fp.clone();
        let reasoning_cache = state.reasoning_cache.clone();
        let reasoning_db = state.storage.db.clone();
        tokio::spawn(async move {
            let req_start = std::time::Instant::now();
            let mut sse_state = crate::protocol::anthropic::AnthropicSseState::new();
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            // Send message_start event
            let _ = tx.send(Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"deepseek-v4-pro\",\"content\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n")
            )).await;
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => {
                        let s = String::from_utf8_lossy(&c);
                        buf.push_str(&s);
                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].trim().to_string();
                            buf = buf[pos + 1..].to_string();
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" { continue; }
                                for event in sse_state.convert(data) {
                                    if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(event))).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            // If upstream didn't send finish_reason (abnormal disconnect), close blocks
            if !sse_state.finish_seen() && sse_state.any_block_started() {
                for idx in sse_state.started_block_indices() {
                    let _ = tx.send(Ok::<_, std::convert::Infallible>(
                        axum::body::Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{idx}}}\n\n"))
                    )).await;
                }
                let _ = tx.send(Ok::<_, std::convert::Infallible>(
                    axum::body::Bytes::from("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n")
                )).await;
                let _ = tx.send(Ok::<_, std::convert::Infallible>(
                    axum::body::Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                )).await;
            }
            // Cache reasoning_content for the next turn
            if !sse_state.reasoning_content.is_empty() {
                let rc_len = sse_state.reasoning_content.len();
                let mut cache = reasoning_cache.lock().unwrap_or_else(|e| e.into_inner());
                if !cache.contains_key(&reasoning_fp)
                    && cache.len() >= REASONING_CACHE_CAPACITY
                    && let Some(evicted) = cache.keys().next().cloned()
                {
                    cache.remove(&evicted);
                    tracing::debug!(target: "deeplossless", key=%evicted, "reasoning cache capacity reached, evicted one entry");
                }
                cache.insert(reasoning_fp.clone(), sse_state.reasoning_content.clone());
                let _ = reasoning_db.insert_event_simple(
                    crate::event_store::EventType::Reasoning,
                    &reasoning_fp,
                    &sse_state.reasoning_content,
                    serde_json::json!({"source": "anthropic_messages"}),
                );
                tracing::debug!(target: "deeplossless::anthropic",
                    fp = %reasoning_fp, rc_len,
                    "reasoning cached from stream");
            }
            let _ = reasoning_db.insert_event_simple(
                crate::event_store::EventType::RequestEnd,
                &reasoning_fp,
                "",
                serde_json::json!({"duration_ms": req_start.elapsed().as_millis(), "source": "anthropic_messages"}),
            );
        });

        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert("content-type", "text/event-stream; charset=utf-8".parse().expect("static header"));
        response
    } else {
        match resp.bytes().await {
            Ok(bytes) => {
                // Record raw DeepSeek response for protocol debugging
                if let Some(ref dir) = state.record {
                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let _ = std::fs::write(format!("{dir}/rsp_{ts}.json"), serde_json::to_string_pretty(&serde_json::from_slice::<Value>(&bytes).unwrap_or_default()).unwrap_or_else(|_| String::from_utf8_lossy(&bytes).to_string()));
                }
                let deepseek_resp: Value = match serde_json::from_slice(&bytes) {
                    Ok(v) => v,
                    Err(e) => return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("invalid JSON: {e}")),
                };
                // Capture reasoning_content for next turn (thinking mode requirement)
                let rc = deepseek_resp["choices"][0]["message"]["reasoning_content"]
                    .as_str().unwrap_or("").to_string();
                if !rc.is_empty() {
                    let rc_len = rc.len();
                    let mut cache = state.reasoning_cache.lock().unwrap_or_else(|e| e.into_inner());
                    if !cache.contains_key(&fp)
                        && cache.len() >= REASONING_CACHE_CAPACITY
                        && let Some(evicted) = cache.keys().next().cloned()
                    {
                        cache.remove(&evicted);
                        tracing::debug!(target: "deeplossless::anthropic", key=%evicted, "reasoning cache capacity reached, evicted one entry");
                    }
                    cache.insert(fp.clone(), rc);
                    tracing::debug!(target: "deeplossless::anthropic",
                        fp = %fp, rc_len,
                        "reasoning cached from non-streaming response");
                }
                let anthropic_resp = crate::protocol::anthropic::response_to_anthropic(&deepseek_resp);
                // Accumulate token usage for non-streaming path
                if let Some(cid) = conv_id {
                    let it = deepseek_resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                    let ot = deepseek_resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                    if it > 0 || ot > 0 {
                        let usage_db = state.storage.db.clone();
                        tokio::task::spawn_blocking(move || {
                            let _ = usage_db.accumulate_usage(cid, it, ot);
                        });
                    }
                }
                Json(anthropic_resp).into_response()
            }
            Err(e) => json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", format!("{e}")),
        }
    }
}

// ── LCM retrieval endpoints ────────────────────────────────────────────

async fn lcm_grep_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conv_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let query = params.get("query").map(|s| s.as_str()).unwrap_or("");
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    match state.storage.db.search_unified(conv_id, query, limit) {
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

/// LCM grep by conversation fingerprint (UUID or session key).
/// Supports non-integer conversation identifiers that OpenCode generates.
async fn lcm_grep_by_fingerprint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let query = params.get("query").map(|s| s.as_str()).unwrap_or("");
    let fingerprint = params.get("fingerprint").map(|s| s.as_str()).unwrap_or("");

    // Resolve fingerprint to internal conversation ID
    let conv_id = if let Ok(Some(id)) = state.storage.db.find_conversation_by_fingerprint(fingerprint) {
        id
    } else if let Ok(id) = fingerprint.parse::<i64>() {
        id
    } else {
        return json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "conversation not found");
    };

    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    match state.storage.db.search_unified(conv_id, query, limit) {
        Ok(results) => Json(json!({
            "conversation_id": conv_id,
            "query": query,
            "total": results.len(),
            "matches": results,
        })).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR", format!("search error: {e}")),
    }
}

/// Return the most recent conversation ID for the current session.
/// AI agents can use this to discover which conversation to query via LCM endpoints.
async fn lcm_current_conv(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    match state.storage.db.last_conversation_id() {
        Ok(Some(id)) => Json(json!({"conversation_id": id})).into_response(),
        Ok(None) => Json(json!({"conversation_id": null, "hint": "No conversations yet. Make a request first."})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/sessions — list recent conversations with event counts.
async fn lcm_sessions_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    match state.storage.db.list_sessions(limit) {
        Ok(rows) => {
            let items: Vec<Value> = rows.iter().map(|(id, fp, model, count, tokens)| json!({
                "id": id,
                "fingerprint": fp,
                "model": model,
                "event_count": count,
                "total_tokens": tokens,
            })).collect();
            Json(json!({"sessions": items})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/search — structured event search.
/// Query params: event_type, tool, session, status, path, content (FTS), limit.
async fn lcm_search_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    use crate::event_store::{EventFilter, EventType};

    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }

    let filter = EventFilter {
        event_type: params.get("event_type").and_then(|s| EventType::from_event_str(s)),
        tool_name: params.get("tool").cloned(),
        session_id: params.get("session").cloned(),
        status: params.get("status").cloned(),
        path_pattern: params.get("path").map(|p| format!("%{p}%")),
        content_match: params.get("content").cloned(),
        limit: params.get("limit").and_then(|s| s.parse().ok()).or(Some(50)),
    };

    match state.storage.db.query_proxy_events(&filter) {
        Ok(events) => {
            let items: Vec<Value> = events.iter().map(|e| json!({
                "id": e.id,
                "event_type": e.event_type.as_str(),
                "session_id": e.session_id,
                "timestamp": e.timestamp,
                "tool_name": e.tool_name,
                "path": e.path,
                "status": e.status,
                "content": &e.content[..e.content.len().min(500)],
                "metadata": e.metadata,
            })).collect();
            Json(json!({"events": items, "count": items.len()})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/diffs — query file-level diffs.
/// Query params: session, file_path, tool_call_id, limit.
async fn lcm_diffs_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    use crate::diff_events::DiffQuery;
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let q = DiffQuery {
        session_id: params.get("session").cloned(),
        file_path: params.get("file_path").cloned(),
        tool_call_id: params.get("tool_call_id").cloned(),
        limit: params.get("limit").and_then(|s| s.parse().ok()).or(Some(50)),
    };
    match state.storage.db.query_diffs(&q) {
        Ok(diffs) => {
            let items: Vec<Value> = diffs.iter().map(|d| json!({
                "id": d.id, "session_id": d.session_id, "tool_call_id": d.tool_call_id,
                "file_path": d.file_path, "start_line": d.start_line, "end_line": d.end_line,
                "change_type": d.change_type.as_str(),
                "before_snippet": d.before_snippet, "after_snippet": d.after_snippet,
                "timestamp": d.timestamp,
            })).collect();
            Json(json!({"diffs": items, "count": items.len()})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/diffs/reconstruct — reconstruct file content from diffs.
/// Query: session, file_path, initial (optional base content).
async fn lcm_diff_reconstruct(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let session = params.get("session").cloned().unwrap_or_default();
    let file_path = params.get("file_path").cloned().unwrap_or_default();
    let initial = params.get("initial").cloned().unwrap_or_default();
    if session.is_empty() || file_path.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session and file_path required");
    }
    match state.storage.db.reconstruct_file(&session, &file_path, &initial) {
        Ok(content) => Json(json!({"content": content, "file_path": file_path})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/diffs/overlaps — find overlapping edits on same file region.
async fn lcm_diff_overlaps(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let session = params.get("session").cloned().unwrap_or_default();
    if session.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session required");
    }
    match state.storage.db.find_overlapping_edits(&session) {
        Ok(pairs) => {
            let items: Vec<Value> = pairs.iter().map(|(a, b)| json!({
                "first": {"tool_call_id": a.tool_call_id, "file_path": a.file_path, "lines": format!("{}-{}", a.start_line, a.end_line)},
                "second": {"tool_call_id": b.tool_call_id, "file_path": b.file_path, "lines": format!("{}-{}", b.start_line, b.end_line)},
            })).collect();
            Json(json!({"overlaps": items, "count": items.len()})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/sessions/{id}/events — execution events for a session.
async fn lcm_session_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session id must be positive");
    }
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let total = state.storage.db.count_session_events(id).unwrap_or(0);
    let tool_counts = state.storage.db.get_tool_category_counts(id).unwrap_or_default();
    match state.storage.db.get_session_events(id, limit) {
        Ok(rows) => {
            let items: Vec<Value> = rows.iter().map(|(ev_id, kind, payload, seq, ts)| json!({
                "id": ev_id,
                "type": kind,
                "payload": payload,
                "seq_no": seq,
                "timestamp": ts.replacen(' ', "T", 1) + "Z",
            })).collect();
            Json(json!({
                "session_id": id,
                "events": items,
                "total": total,
                "tool_counts": tool_counts.iter().map(|(k, v)| json!({"tool": k, "count": v})).collect::<Vec<_>>(),
            })).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
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
    let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);

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
            if tx.send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(payload))).await.is_err() {
                break;
            }
            // Small yield to let the client process incrementally
            tokio::task::yield_now().await;
        }

        let _ = tx.send(Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from("data: [DONE]\n\n")
        )).await;
    });

    let stream = ReceiverStream::new(rx);
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
    Json(body): Json<HashMap<String, serde_json::Value>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let agent_id = body.get("agent_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let file_path = body.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
    let operation = body.get("operation").and_then(|v| v.as_str()).unwrap_or("edit");
    let conv_id = body.get("conv_id").and_then(|v| v.as_i64()).unwrap_or(0);

    if file_path.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "file_path is required");
    }
    if conv_id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "conv_id must be a positive integer");
    }

    match state.storage.db.claim_file(agent_id, file_path, operation, conv_id) {
        Ok(Ok(())) => Json(json!({"status": "claimed", "agent_id": agent_id, "file_path": file_path, "conv_id": conv_id})).into_response(),
        Ok(Err(conflict_agent)) => json_error(StatusCode::CONFLICT, "CONFLICT", format!("file '{file_path}' held by agent '{conflict_agent}' in another conversation")),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "CLAIM_ERROR", format!("{e}")),
    }
}

/// Release an agent's claim on a file.
async fn lcm_file_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<HashMap<String, serde_json::Value>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let agent_id = body.get("agent_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let file_path = body.get("file_path").and_then(|v| v.as_str()).unwrap_or("");

    if file_path.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "file_path is required");
    }

    match state.storage.db.release_file(agent_id, file_path) {
        Ok(0) => json_error(StatusCode::NOT_FOUND, "NOT_FOUND", format!("no claim found for '{}' by agent '{}'", file_path, agent_id)),
        Ok(_) => Json(json!({"status": "released", "file_path": file_path})).into_response(),
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

/// GET /v1/lcm/latency — recent upstream request latency records.
async fn lcm_latency_records(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(50);
    let records = metrics::get_latency_records(limit);
    Json(json!({
        "count": records.len(),
        "records": records,
        "summary": metrics::get_latency_summary(),
    })).into_response()
}

/// GET /v1/lcm/latency/summary — aggregated latency statistics.
async fn lcm_latency_summary(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    Json(metrics::get_latency_summary()).into_response()
}

/// Delete a cached tool result.
async fn lcm_cache_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let tool = params.get("tool").map(|s| s.as_str()).unwrap_or("");
    let args = params.get("args").map(|s| s.as_str()).unwrap_or("");
    let (_name, hash) = crate::tool_cache::cache_key(tool, args);
    if tool.is_empty() || hash.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "tool and args required");
    }
    match state.storage.db.tool_cache_delete(tool, &hash) {
        Ok(()) => Json(json!({"status": "deleted"})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "CACHE_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/sessions/{id}/patches — tool results containing patch/diff content.
async fn lcm_session_patches(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session id must be positive");
    }
    let limit = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    match state.storage.db.get_session_patches(id, limit) {
        Ok(rows) => {
            let items: Vec<Value> = rows.iter().map(|(role, content)| json!({
                "role": role,
                "content": content,
            })).collect();
            Json(json!({"session_id": id, "patches": items})).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/sessions/{id}/system-prompt — system prompt history for a session.
/// Returns deduplicated consecutive entries (identical repeats collapsed).
async fn lcm_session_system_prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session id must be positive");
    }
    match state.storage.db.get_system_prompts_deduped(id) {
        Ok(rows) => {
            let items: Vec<Value> = rows.iter().map(|(msg_id, content, tokens, ts)| json!({
                "id": msg_id,
                "content": content,
                "token_count": tokens,
                "stored_at": ts.replacen(' ', "T", 1) + "Z",
            })).collect();
            Json(json!({
                "session_id": id,
                "count": items.len(),
                "prompts": items,
            })).into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

/// GET /v1/lcm/sessions/{id}/context-pressure
/// Returns context pressure analysis for the Context Pressure Dashboard.
/// Works across all 3 proxy endpoints (chat_completions, responses, anthropic).
async fn lcm_context_pressure(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "session id must be positive");
    }
    match state.storage.db.context_pressure_analysis(id) {
        Ok(data) => Json(data).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", format!("{e}")),
    }
}

#[allow(dead_code)]
/// POST /v1/lcm/inject — inject DAG context into the last user message.
/// Takes a messages array, assembles DAG context for the given conversation,
/// and appends it to the last `role: "user"` message. Returns the modified
/// messages with token accounting. Does NOT modify the proxy stream — the
/// client calls this explicitly before forwarding to any LLM provider.
#[derive(Deserialize)]
struct LcmInjectRequest {
    conv_id: i64,
    /// The messages array to inject context into. Must contain at least one user message.
    messages: Vec<serde_json::Value>,
    /// Optional search query to guide DAG context assembly.
    query: Option<String>,
    /// Token budget for context. Default 500, max 8000. Set 0 to skip assembly.
    max_tokens: Option<usize>,
}
#[allow(dead_code)]
async fn lcm_context_inject(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LcmInjectRequest>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if body.conv_id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "conv_id must be positive");
    }
    if body.messages.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "messages array is empty");
    }

    let mut messages = body.messages;
    let max_tokens = body.max_tokens.unwrap_or(500).clamp(0, 8000);
    let mut context_text = String::new();
    let mut context_tokens = 0usize;
    let mut node_count = 0usize;

    if max_tokens > 0 {
        let query = body.query.as_deref();
        let dag = state.storage.dag.clone();
        match dag.assemble_context(body.conv_id, max_tokens, query) {
            Ok(nodes) if !nodes.is_empty() => {
                context_text = crate::pipeline::render_dag_context(&nodes);
                context_tokens = crate::tokenizer::count(&context_text);
                node_count = nodes.len();
            }
            Ok(_) => {} // no context found — inject nothing
            Err(e) => {
                return json_error(StatusCode::INTERNAL_SERVER_ERROR, "DAG_ERROR", format!("{e}"));
            }
        }
    }

    let injected = !context_text.is_empty();

    // Inject context into the last user message
    if injected {
        if let Some(last_user) = messages.iter_mut().rev().find(|m| m["role"] == "user") {
            let original = last_user["content"].as_str().unwrap_or("").to_string();
            let merged = if original.is_empty() {
                context_text.clone()
            } else {
                format!("{context_text}\n\n{original}")
            };
            last_user["content"] = serde_json::json!(merged);
        }
    }

    let total_tokens: usize = messages.iter()
        .map(|m| crate::tokenizer::count(m["content"].as_str().unwrap_or("")))
        .sum();

    Json(json!({
        "conv_id": body.conv_id,
        "messages": messages,
        "context_token_count": context_tokens,
        "total_token_count": total_tokens,
        "max_tokens": max_tokens,
        "node_count": node_count,
        "injected": injected,
        "usage": {
            "context_tokens": context_tokens,
            "budget": max_tokens,
            "pct": if max_tokens > 0 { (context_tokens as f64 / max_tokens as f64 * 100.0).round() as u32 } else { 0 },
            "total": total_tokens,
        },
    })).into_response()
}

/// GET /v1/lcm/cache/stability — system prompt cache stability diagnostics.
/// Returns per-conversation hash history and stability metrics.
async fn lcm_cache_stability(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let tracker = state.cache_stability.lock().unwrap_or_else(|e| e.into_inner());
    let items: Vec<Value> = tracker.iter().map(|(conv_id, hashes)| {
        let total = hashes.len();
        let unique = hashes.iter().collect::<std::collections::HashSet<_>>().len();
        let cache_hit_rate = if total < 2 { 1.0 } else { 1.0 - (unique as f64 - 1.0) / (total as f64 - 1.0).max(1.0) };
        json!({
            "conversation_id": conv_id,
            "samples": total,
            "unique_hashes": unique,
            "stability_pct": (cache_hit_rate * 100.0).round() as u32,
            "recent": hashes.iter().rev().take(5).collect::<Vec<_>>(),
        })
    }).collect();
    Json(json!({"conversations": items})).into_response()
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
        Ok(id) => {
            // Link to runtime metrics so repeated_failures stays accurate
            if let Ok(mut cycle) = state.runtime.cycle.lock() {
                cycle.metrics.repeated_failures += 1;
            }
            Json(json!({"status": "stored", "id": id})).into_response()
        }
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
        Ok(Some((id, goal, pending, completed, assumptions))) => Json(json!({
            "id": id, "goal": goal,
            "pending_steps": pending,
            "completed_steps": completed,
            "assumptions": assumptions,
        })).into_response(),
        Ok(None) => Json(json!({"active_plan": null})).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "PLAN_ERROR", format!("{e}")),
    }
}

/// Delete a plan by ID.
async fn lcm_plan_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    let id: i64 = params.get("id").and_then(|s| s.parse().ok()).unwrap_or(0);
    if id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "plan id must be a positive integer");
    }
    match state.storage.db.deactivate_plan(id) {
        Ok(true) => Json(json!({"status": "deleted", "id": id})).into_response(),
        Ok(false) => json_error(StatusCode::NOT_FOUND, "NOT_FOUND", format!("plan {} not found", id)),
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
    let ttl = body.get("retention_ttl").and_then(|v| v.as_i64());

    if execution_id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "execution_id must be a positive integer");
    }

    // Validate tier — fail fast on invalid values
    let tier = match crate::snapshot::SnapshotTier::from_i32(tier_raw) {
        Ok(tier) => tier,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "INVALID_TIER", format!("{e}")),
    };

    let rows = match state.storage.db.get_execution_events(execution_id) {
        Ok(rows) => rows,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "SNAPSHOT_ERROR", format!("{e}")),
    };
    let last_event_seq_no = rows
        .iter()
        .map(|(_id, _kind, _payload, seq_no, _ts)| *seq_no)
        .max()
        .unwrap_or(0);
    let payload = match tier {
        crate::snapshot::SnapshotTier::Ephemeral => crate::snapshot::SnapshotPayload::Ephemeral {
            last_seq_no: last_event_seq_no,
            event_count: rows.len() as u32,
        },
        crate::snapshot::SnapshotTier::Structural
        | crate::snapshot::SnapshotTier::Full
        | crate::snapshot::SnapshotTier::Frozen => {
            let mut events = Vec::with_capacity(rows.len());
            for (_id, _kind, payload, seq_no, _ts) in &rows {
                let value = match serde_json::from_str::<serde_json::Value>(payload) {
                    Ok(value) => value,
                    Err(e) => return json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "SNAPSHOT_ERROR",
                        format!("execution event payload is not valid JSON at seq_no {seq_no}: {e}"),
                    ),
                };
                events.push((*seq_no, value));
            }
            match tier {
                crate::snapshot::SnapshotTier::Structural => {
                    crate::snapshot::SnapshotPayload::Structural { events }
                }
                crate::snapshot::SnapshotTier::Full => crate::snapshot::SnapshotPayload::Full { events },
                crate::snapshot::SnapshotTier::Frozen => {
                    crate::snapshot::SnapshotPayload::Frozen { events }
                }
                crate::snapshot::SnapshotTier::Ephemeral => unreachable!(),
            }
        }
    };
    let snapshot_event_payloads: Vec<(i64, String)> = match &payload {
        crate::snapshot::SnapshotPayload::Ephemeral { .. } => rows
            .iter()
            .map(|(_id, _kind, payload, seq_no, _ts)| (*seq_no, payload.clone()))
            .collect(),
        crate::snapshot::SnapshotPayload::Structural { events }
        | crate::snapshot::SnapshotPayload::Full { events }
        | crate::snapshot::SnapshotPayload::Frozen { events } => {
            let mut serialized = Vec::with_capacity(events.len());
            for (seq_no, value) in events {
                let payload = match serde_json::to_string(value) {
                    Ok(payload) => payload,
                    Err(e) => {
                        return json_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "SNAPSHOT_ERROR",
                            format!("snapshot event serialize failed at seq_no {seq_no}: {e}"),
                        );
                    }
                };
                serialized.push((*seq_no, payload));
            }
            serialized
        }
    };
    let event_refs: Vec<(i64, &str)> = snapshot_event_payloads
        .iter()
        .map(|(seq_no, payload)| (*seq_no, payload.as_str()))
        .collect();
    let data = match serde_json::to_string(&payload) {
        Ok(data) => data,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "SNAPSHOT_ERROR", format!("{e}")),
    };
    let size_bytes = data.len() as i64;
    let boundary_hash = crate::snapshot::compute_boundary_hash(
        &event_refs,
        crate::snapshot::BOUNDARY_EVENT_COUNT,
    );
    let integrity_hash = crate::snapshot::compute_chain_hash(&event_refs);

    match state.storage.db.take_snapshot(
        execution_id, memory_version_id, tier_raw, &data, size_bytes, ttl,
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

/// Runtime metrics endpoint — exposes inference-economics counters for diagnostics.
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
    if query.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "query parameter 'q' is required");
    }
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
            let items: Vec<Value> = results.iter().map(|(nid, cid, summary, excerpt)| json!({
                "node_id": nid,
                "conversation_id": cid,
                "summary": summary,
                "excerpt": excerpt,
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
    }
    if conv_id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "conversation_id must be positive");
    }
    // Verify the conversation exists before running expensive validation
    if !state.storage.db.conversation_exists(conv_id) {
        return json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "conversation not found");
    }
    match state.storage.dag.validate_dag(conv_id) {
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

/// Return 400 when hash is missing from the path.
async fn lcm_similar_missing_hash(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "hash is required in path: /v1/lcm/similar/{hash}")
}

async fn lcm_similar(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(hash): Path<String>,
) -> Response {
    if !ctx_react_auth_ok(&headers, &state) {
        return json_error(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "unauthorized");
    }
    if hash.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "hash is required");
    }
    match state.storage.db.find_similar_by_hash(&hash) {
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
    #[serde(default)]
    id: i64,
    #[serde(default, alias = "execution_id")]
    execution_id: i64,
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
    // Collect source nodes from DAG leaves in the requested range.
    // Fall back to raw messages only if there are no DAG leaves at all.
    let (text, source_ids, source_level): (String, Vec<i64>, u8) = match state.storage.dag.get_leaves(op.conv_id) {
        Ok(leaves) if leaves.len() >= 2 => {
            let in_range: Vec<&crate::dag::DagNode> = leaves.iter()
                .skip_while(|n| n.id < op.from)
                .take_while(|n| n.id <= op.to)
                .collect();
            if in_range.len() < 2 {
                return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "need at least 2 leaves in range");
            }
            let ids: Vec<i64> = in_range.iter().map(|n| n.id).collect();
            // New level = max(source levels) + 1, capped at effective max
            let max_lvl = in_range.iter().map(|n| n.level).max().unwrap_or(0);
            let level = (max_lvl + 1).min(3);
            let text = in_range.iter()
                .map(|n| n.summary.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n");
            (text, ids, level)
        }
        _ => {
            let db = state.storage.dag.db();
            let raw = db.get_messages_in_range(op.conv_id, op.from, op.to);
            match raw {
                Ok(msgs) if msgs.len() >= 2 => {
                    // Raw messages have no DAG node IDs — produce a level-1 summary
                    (msgs.join("\n---\n"), vec![], 1u8)
                }
                _ => return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "need at least 2 messages to compress"),
            }
        }
    };

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
                op.conv_id, &source_ids, &result.text, tc, source_level, &snippets,
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
    let node_id = if op.id > 0 { op.id } else { op.execution_id };
    if node_id <= 0 {
        return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", "id or execution_id must be positive");
    }
    let node = match state.storage.dag.get_node(node_id) {
        Ok(Some(n)) => n,
        Ok(None) => return json_error(StatusCode::NOT_FOUND, "NOT_FOUND", format!("node {} not found", op.id)),
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "NODE_ERROR", format!("error: {e}")),
    };
    // Actual DAG rollback: soft-delete all nodes created after target, and their
    // transitive children that depend exclusively on them.
    match state.storage.dag.rollback_to(node_id) {
        Ok(deleted) => {
            let children = state.storage.dag.get_children(node_id).unwrap_or_default();
            Json(json!({
                "rollback_to": node_id,
                "summary": node.summary,
                "level": node.level,
                "deleted_nodes": deleted,
                "children_remaining": children.iter().map(|c| json!({
                    "id": c.id,
                    "summary": c.summary,
                    "token_count": c.token_count,
                })).collect::<Vec<_>>(),
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "ROLLBACK_ERROR", format!("rollback failed: {e}")),
    }
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
