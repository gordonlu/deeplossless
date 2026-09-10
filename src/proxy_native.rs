//! Native DeepSeek Responses API transport.
//!
//! DeepSeek now accepts OpenAI Responses API requests directly. The native
//! path keeps Responses wire semantics end-to-end and only falls back to the
//! legacy Responses→Chat adapter when an upstream does not expose `/responses`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;

use crate::event_store::EventType;
use crate::ground_truth::TruthStream;
use crate::ground_truth_store::GroundTruthStore;
use crate::protocol::{ReasoningEffort, ReasoningEffortMode};
use crate::AppState;

const LOCAL_SESSION_FIELD: &str = "_deeplossless_session_id";
const STREAM_CHANNEL_CAPACITY: usize = 32;
static NEXT_EPHEMERAL_SESSION: AtomicU64 = AtomicU64::new(1);
type SseChunk = Result<axum::body::Bytes, std::convert::Infallible>;

pub use crate::proxy_legacy::upstream_chat_url;
pub(crate) use crate::proxy_legacy::process_events;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/responses", post(responses))
        .route("/responses", post(responses))
        .route("/v1/responses/{response_id}", get(responses_retrieve))
        .route("/responses/{response_id}", get(responses_retrieve))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .fallback(legacy_fallback)
}

async fn legacy_fallback(State(state): State<AppState>, req: Request) -> Response {
    run_legacy(state, req).await
}

async fn run_legacy(state: AppState, req: Request) -> Response {
    let service = crate::proxy_legacy::routes().with_state(state);
    match service.oneshot(req).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
}

async fn legacy_responses_fallback(state: &AppState, headers: &HeaderMap, body: &str) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json");
    if let Some(value) = headers.get("authorization") {
        builder = builder.header("authorization", value);
    }
    if let Some(value) = headers.get("accept") {
        builder = builder.header("accept", value);
    }
    let req = match builder.body(Body::from(body.to_owned())) {
        Ok(req) => req,
        Err(error) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                error.to_string(),
            );
        }
    };
    run_legacy(state.clone(), req).await
}

pub fn upstream_responses_url(upstream: &str) -> String {
    let base = upstream.trim_end_matches('/');
    if base.ends_with("/responses") {
        base.to_string()
    } else if base.ends_with("/chat/completions") {
        format!("{}/responses", base.trim_end_matches("/chat/completions"))
    } else {
        format!("{base}/responses")
    }
}

fn json_error(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error":{"code":code,"message":message.into()}}))).into_response()
}

fn cached_api_key(state: &AppState) -> String {
    state
        .api_key
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(|| "unset".to_string())
}

fn remember_api_key(state: &AppState, headers: &HeaderMap) {
    let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) else {
        return;
    };
    let Some(bearer) = auth
        .strip_prefix("Bearer ")
        .or_else(|| auth.strip_prefix("bearer "))
    else {
        return;
    };
    let mut key = state.api_key.lock().unwrap_or_else(|e| e.into_inner());
    if key.is_none() {
        *key = Some(bearer.to_string());
    }
}

fn stored_session_id(resp: &Value) -> Option<String> {
    resp.get(LOCAL_SESSION_FIELD)
        .and_then(Value::as_str)
        .or_else(|| resp.get("prompt_cache_key").and_then(Value::as_str))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
}

fn session_id_from_previous(state: &AppState, response_id: &str) -> Option<String> {
    if let Some(resp) = state.storage.response_store.get(response_id) {
        if let Some(session_id) = stored_session_id(&resp) {
            return Some(session_id);
        }
    }
    match state.storage.db.get_response_object(response_id) {
        Ok(Some(resp)) => {
            let session_id = stored_session_id(&resp);
            state
                .storage
                .response_store
                .insert(response_id.to_string(), resp);
            session_id
        }
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(target:"deeplossless::responses", %response_id, %error, "failed to load previous response");
            None
        }
    }
}

/// Create a unique local continuity key for a new stateless Responses chain.
/// Request-content hashes are intentionally not used: identical independent
/// first turns must never share DAG/fact/session state.
fn fallback_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_EPHEMERAL_SESSION.fetch_add(1, Ordering::Relaxed);
    format!("responses:{now:x}:{sequence:x}")
}

fn response_session_id(state: &AppState, body: &Value) -> String {
    if let Some(key) = body
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return key.to_string();
    }
    if let Some(previous) = body
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some(session_id) = session_id_from_previous(state, previous) {
            return session_id;
        }
    }
    fallback_session_id()
}

fn input_items(input: Option<&Value>) -> Vec<Value> {
    match input {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(text)) => {
            vec![json!({"type":"message","role":"user","content":text})]
        }
        _ => Vec::new(),
    }
}

fn merge_history(mut history: Vec<Value>, current: Vec<Value>) -> Vec<Value> {
    let max_overlap = history.len().min(current.len());
    let overlap = (1..=max_overlap)
        .rev()
        .find(|&n| history[history.len() - n..] == current[..n])
        .unwrap_or(0);
    history.extend(current.into_iter().skip(overlap));
    history
}

fn apply_reasoning_override(state: &AppState, body: &mut Value) {
    let effort = match state.reasoning_effort {
        ReasoningEffortMode::Passthrough => return,
        ReasoningEffortMode::Override(ReasoningEffort::None) => "none",
        ReasoningEffortMode::Override(ReasoningEffort::High) => "high",
        ReasoningEffortMode::Override(ReasoningEffort::Max) => "max",
    };
    if !body.get("reasoning").is_some_and(Value::is_object) {
        body["reasoning"] = json!({});
    }
    body["reasoning"]["effort"] = json!(effort);
}

fn build_upstream_body(
    state: &AppState,
    request: &Value,
    session_id: &str,
    force_stream: bool,
) -> (Value, Vec<Value>, Vec<Value>) {
    let mut body = request.clone();
    if let Some(model) = body.get("model").and_then(Value::as_str) {
        body["model"] = json!(crate::protocol::map_model(model));
    }
    if force_stream {
        body["stream"] = json!(true);
    }
    apply_reasoning_override(state, &mut body);

    let current_items = input_items(request.get("input"));
    let has_previous = request
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(|s| !s.is_empty());

    let history = if has_previous {
        state
            .storage
            .session_store
            .get(session_id)
            .or_else(|| match state.storage.db.get_response_session(session_id) {
                Ok(Some(items)) => {
                    state
                        .storage
                        .session_store
                        .replace(session_id, items.clone());
                    Some(items)
                }
                Ok(None) => None,
                Err(error) => {
                    tracing::warn!(target:"deeplossless::responses", %session_id, %error, "failed to load response session");
                    None
                }
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let merged = merge_history(history, current_items.clone());
    if has_previous {
        body["input"] = Value::Array(merged.clone());
    }

    if let Some(object) = body.as_object_mut() {
        object.remove("previous_response_id");
        object.remove("conversation");
        object.remove("store");
        object.remove("prompt_cache_key");
        object.remove("prompt_cache_retention");
    }
    (body, merged, current_items)
}

fn strip_local_fields(mut response: Value) -> Value {
    if let Some(object) = response.as_object_mut() {
        object.remove(LOCAL_SESSION_FIELD);
    }
    response
}

fn store_response_and_session(
    state: &AppState,
    session_id: &str,
    input_history: &[Value],
    response: &Value,
) {
    let Some(response_id) = response.get("id").and_then(Value::as_str) else {
        return;
    };
    let mut stored = response.clone();
    stored[LOCAL_SESSION_FIELD] = json!(session_id);
    state
        .storage
        .response_store
        .insert(response_id.to_string(), stored.clone());
    if let Err(error) = state
        .storage
        .db
        .store_response_object(response_id, session_id, &stored)
    {
        tracing::warn!(target:"deeplossless::responses", %response_id, %error, "failed to persist response object");
    }

    let mut session = input_history.to_vec();
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        session.extend(output.iter().cloned());
    }
    state
        .storage
        .session_store
        .replace(session_id, session.clone());
    if let Err(error) = state.storage.db.store_response_session(session_id, &session) {
        tracing::warn!(target:"deeplossless::responses", %session_id, %error, "failed to persist response session");
    }
}

fn text_from_content(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    content
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn observe_items(state: &AppState, session_id: &str, items: &[Value]) {
    let truth = GroundTruthStore::new(&state.storage.db);
    for item in items {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");

        if let Err(error) = truth.put_json(
            TruthStream::ProviderItems,
            session_id,
            item,
            if item_type.is_empty() {
                "responses-item"
            } else {
                item_type
            },
        ) {
            tracing::warn!(target:"deeplossless::ground_truth", %session_id, %error,
                "failed to persist exact Responses item");
        }

        match item_type {
            "message" | "" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("");
                let text = text_from_content(item.get("content").unwrap_or(&Value::Null));
                if text.is_empty() {
                    continue;
                }
                let event_type = match role {
                    "user" | "developer" => EventType::UserMessage,
                    "assistant" => EventType::AssistantMessage,
                    _ => continue,
                };
                let _ = state.storage.db.insert_event_simple(
                    event_type,
                    session_id,
                    &text,
                    json!({"source":"responses-native","role":role}),
                );
            }
            "reasoning" => {
                let text = item
                    .get("content")
                    .map(text_from_content)
                    .filter(|s| !s.is_empty())
                    .or_else(|| item.get("summary").map(text_from_content))
                    .unwrap_or_default();
                if !text.is_empty() {
                    let _ = state.storage.db.insert_event_simple(
                        EventType::Reasoning,
                        session_id,
                        &text,
                        json!({"source":"responses-native"}),
                    );
                }
            }
            "function_call" | "custom_tool_call" | "web_search_call" => {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(item_type);
                let arguments = item
                    .get("arguments")
                    .or_else(|| item.get("input"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let _ = state.storage.db.insert_event_simple(
                    EventType::ToolCall,
                    session_id,
                    &format!("{name}({arguments})"),
                    json!({"source":"responses-native","tool_name":name}),
                );
            }
            "function_call_output" | "custom_tool_call_output" => {
                let output = item
                    .get("output")
                    .map(text_from_content)
                    .unwrap_or_default();
                let _ = state.storage.db.insert_event_simple(
                    EventType::ToolResult,
                    session_id,
                    &output,
                    json!({"source":"responses-native","call_id":item.get("call_id").cloned().unwrap_or(Value::Null)}),
                );
            }
            _ => {}
        }
    }
}

fn terminal_response_from_frame(frame: &str) -> Option<Value> {
    let mut event_name = None;
    let mut data = None;
    for line in frame.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim());
        } else if let Some(value) = line.strip_prefix("data:") {
            data = Some(value.trim());
        }
    }
    let event_name = event_name?;
    if !matches!(
        event_name,
        "response.completed" | "response.incomplete" | "response.failed"
    ) {
        return None;
    }
    let value: Value = serde_json::from_str(data?).ok()?;
    value.get("response").cloned()
}

/// Remove and return one complete SSE frame from a byte buffer. Supporting
/// both LF and CRLF keeps the observer robust without ever changing the bytes
/// forwarded to the client.
fn take_sse_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n");
    let (position, delimiter_len) = match (lf, crlf) {
        (Some(a), Some(b)) if a <= b => (a, 2),
        (Some(_), Some(b)) => (b, 4),
        (Some(a), None) => (a, 2),
        (None, Some(b)) => (b, 4),
        (None, None) => return None,
    };
    let frame = buffer[..position].to_vec();
    buffer.drain(..position + delimiter_len);
    Some(frame)
}

async fn responses(State(state): State<AppState>, headers: HeaderMap, body: String) -> Response {
    let request: Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => {
            return json_error(StatusCode::BAD_REQUEST, "BAD_REQUEST", error.to_string())
        }
    };

    remember_api_key(&state, &headers);
    let session_id = response_session_id(&state, &request);
    let accept_sse = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));
    let streaming = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || accept_sse;
    let (mut upstream_body, input_history, current_items) =
        build_upstream_body(&state, &request, &session_id, streaming);

    observe_items(&state, &session_id, &current_items);

    let model = upstream_body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("deepseek-flash")
        .to_string();
    let projected = crate::responses_projection::project_and_assemble(
        &state,
        &session_id,
        &model,
        &current_items,
    )
    .await;
    if state.lcm_context {
        crate::responses_projection::inject_context(&mut upstream_body, &projected.context);
    }

    let _ = state.storage.db.insert_event_simple(
        EventType::RequestStart,
        &session_id,
        "",
        json!({
            "source":"responses-native",
            "model":upstream_body.get("model").cloned().unwrap_or(Value::Null),
            "stream":streaming,
            "input_items":current_items.len(),
            "projection_conv_id":projected.conv_id,
            "typed_plan_used":projected.typed_plan_used,
            "context_injected":state.lcm_context && !projected.context.is_empty(),
        }),
    );

    if state.dry_run {
        let out_dir = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(".deeplossless");
        let _ = std::fs::create_dir_all(&out_dir);
        let _ = std::fs::write(
            out_dir.join("translated.json"),
            serde_json::to_string_pretty(&upstream_body).unwrap_or_default(),
        );
        return Json(json!({
            "id":"resp_dry_run","object":"response","status":"completed",
            "model":upstream_body.get("model").cloned().unwrap_or(Value::Null),
            "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"[dry-run] native Responses request saved to ~/.deeplossless/translated.json"}]}],
            "store":false
        }))
        .into_response();
    }

    let upstream_url = upstream_responses_url(&state.upstream);
    let upstream = match state
        .runtime
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", cached_api_key(&state)))
        .header("Content-Type", "application/json")
        .json(&upstream_body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let _ = state.storage.db.insert_event_simple(
                EventType::Error,
                &session_id,
                &error.to_string(),
                json!({"source":"responses-native"}),
            );
            return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", error.to_string());
        }
    };

    let status = upstream.status();
    let upstream_headers = upstream.headers().clone();

    if status == StatusCode::NOT_FOUND {
        tracing::debug!(target:"deeplossless::responses", %upstream_url, "native Responses unavailable; using legacy adapter");
        return legacy_responses_fallback(&state, &headers, &body).await;
    }

    if !status.is_success() {
        let bytes = upstream.bytes().await.unwrap_or_default();
        let _ = state.storage.db.insert_event_simple(
            EventType::Error,
            &session_id,
            &String::from_utf8_lossy(&bytes),
            json!({"source":"responses-native","status":status.as_u16()}),
        );
        let mut response = Response::new(Body::from(bytes));
        *response.status_mut() = status;
        if let Some(content_type) = upstream_headers.get("content-type") {
            response
                .headers_mut()
                .insert("content-type", content_type.clone());
        }
        return response;
    }

    if streaming {
        let content_type = upstream_headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !content_type
            .to_ascii_lowercase()
            .contains("text/event-stream")
        {
            let raw = upstream.text().await.unwrap_or_default();
            return json_error(
                StatusCode::BAD_GATEWAY,
                "UPSTREAM_ERROR",
                format!("expected upstream text/event-stream, got {content_type}: {raw}"),
            );
        }
    }

    if !streaming {
        let bytes = match upstream.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => {
                return json_error(StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", error.to_string())
            }
        };
        let response_value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "UPSTREAM_ERROR",
                    format!("invalid upstream JSON: {error}"),
                )
            }
        };
        let output = response_value
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        observe_items(&state, &session_id, &output);
        store_response_and_session(&state, &session_id, &input_history, &response_value);
        let _ = state.storage.db.insert_event_simple(
            EventType::RequestEnd,
            &session_id,
            "",
            json!({"source":"responses-native","status":status.as_u16()}),
        );
        let mut response = Response::new(Body::from(bytes));
        *response.status_mut() = status;
        if let Some(content_type) = upstream_headers.get("content-type") {
            response
                .headers_mut()
                .insert("content-type", content_type.clone());
        }
        return response;
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<SseChunk>(STREAM_CHANNEL_CAPACITY);
    let stream_state = state.clone();
    let stream_session_id = session_id.clone();
    tokio::spawn(async move {
        let mut upstream_stream = upstream.bytes_stream();
        let mut parse_buffer = Vec::<u8>::new();
        let mut terminal_response = None;
        while let Some(chunk) = upstream_stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = stream_state.storage.db.insert_event_simple(
                        EventType::Error,
                        &stream_session_id,
                        &error.to_string(),
                        json!({"source":"responses-native","phase":"stream"}),
                    );
                    break;
                }
            };
            if tx.send(Ok(bytes.clone())).await.is_err() {
                break;
            }
            parse_buffer.extend_from_slice(&bytes);
            while let Some(frame) = take_sse_frame(&mut parse_buffer) {
                if let Ok(frame) = std::str::from_utf8(&frame) {
                    if let Some(response) = terminal_response_from_frame(frame) {
                        terminal_response = Some(response);
                    }
                }
            }
        }
        if let Some(response_value) = terminal_response {
            let output = response_value
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            observe_items(&stream_state, &stream_session_id, &output);
            store_response_and_session(
                &stream_state,
                &stream_session_id,
                &input_history,
                &response_value,
            );
        }
        let _ = stream_state.storage.db.insert_event_simple(
            EventType::RequestEnd,
            &stream_session_id,
            "",
            json!({"source":"responses-native","status":status.as_u16()}),
        );
    });

    let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    *response.status_mut() = status;
    response.headers_mut().insert(
        "content-type",
        upstream_headers
            .get("content-type")
            .cloned()
            .unwrap_or_else(|| "text/event-stream; charset=utf-8".parse().unwrap()),
    );
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().unwrap());
    response
}

async fn responses_retrieve(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Some(response) = state.storage.response_store.get(&response_id) {
        return Json(strip_local_fields(response)).into_response();
    }
    match state.storage.db.get_response_object(&response_id) {
        Ok(Some(response)) => {
            state
                .storage
                .response_store
                .insert(response_id, response.clone());
            Json(strip_local_fields(response)).into_response()
        }
        Ok(None) => json_error(StatusCode::NOT_FOUND, "NOT_FOUND", "response not found"),
        Err(error) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "STORAGE_ERROR",
            error.to_string(),
        ),
    }
}

fn model_entry(id: &str, version: &str, vision: bool) -> Value {
    json!({
        "id":id,
        "object":"model",
        "owned_by":"deepseek",
        "model_version":version,
        "context_window":1_000_000,
        "max_context_tokens":1_000_000,
        "max_input_tokens":1_000_000,
        "max_output_tokens":393_216,
        "supports_reasoning":true,
        "supports_thinking":true,
        "reasoning":true,
        "thinking":true,
        "supports_tool_calls":true,
        "supports_streaming":true,
        "supports_responses":true,
        "supports_vision":vision,
        "capabilities":{
            "supports_tool_calls":true,
            "supports_streaming":true,
            "supports_reasoning":true,
            "supports_thinking":true,
            "supports_responses":true,
            "supports_vision":vision,
            "max_context_tokens":1_000_000,
            "max_output_tokens":393_216
        }
    })
}

fn legacy_alias_entry(id: &str) -> Value {
    let mut entry = model_entry(id, "DeepSeek-V4.1-Flash", true);
    entry["deprecated"] = json!(true);
    entry["alias_for"] = json!("deepseek-flash");
    entry
}

async fn list_models() -> Response {
    Json(json!({
        "object":"list",
        "data":[
            model_entry("deepseek-flash", "DeepSeek-V4.1-Flash", true),
            model_entry("deepseek-v4-pro", "DeepSeek-V4-Pro-0813", false),
            legacy_alias_entry("deepseek-v4-flash"),
            legacy_alias_entry("deepseek-v4-flash-vision-exp")
        ]
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_url_uses_native_resource() {
        assert_eq!(
            upstream_responses_url("https://api.deepseek.com"),
            "https://api.deepseek.com/responses"
        );
        assert_eq!(
            upstream_responses_url("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1/responses"
        );
        assert_eq!(
            upstream_responses_url("https://api.deepseek.com/v1/chat/completions"),
            "https://api.deepseek.com/v1/responses"
        );
    }

    #[test]
    fn new_fallback_sessions_are_unique_even_for_identical_requests() {
        assert_ne!(fallback_session_id(), fallback_session_id());
    }

    #[test]
    fn merge_history_removes_only_exact_overlap() {
        let history = vec![json!({"id":1}), json!({"id":2})];
        let current = vec![json!({"id":2}), json!({"id":3})];
        assert_eq!(
            merge_history(history, current),
            vec![json!({"id":1}), json!({"id":2}), json!({"id":3})]
        );
    }

    #[test]
    fn terminal_frame_extracts_complete_response() {
        let frame = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"sequence_number\":9,\"response\":{\"id\":\"resp_1\",\"output\":[]}}\n"
        );
        let response = terminal_response_from_frame(frame).unwrap();
        assert_eq!(response["id"], "resp_1");
    }

    #[test]
    fn sse_frame_parser_accepts_crlf_and_preserves_utf8_across_chunks() {
        let event = "event: response.completed\r\ndata: {\"response\":{\"id\":\"响应_1\",\"output\":[]}}\r\n\r\n";
        let bytes = event.as_bytes();
        let split = event.find('响').unwrap() + 1;
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&bytes[..split]);
        assert!(take_sse_frame(&mut buffer).is_none());
        buffer.extend_from_slice(&bytes[split..]);
        let frame = take_sse_frame(&mut buffer).unwrap();
        let parsed = terminal_response_from_frame(std::str::from_utf8(&frame).unwrap()).unwrap();
        assert_eq!(parsed["id"], "响应_1");
    }

    #[test]
    fn local_session_metadata_is_not_exposed() {
        let response = strip_local_fields(json!({"id":"resp_1", LOCAL_SESSION_FIELD:"session-a"}));
        assert!(response.get(LOCAL_SESSION_FIELD).is_none());
    }

    #[test]
    fn v41_flash_is_multimodal_current_model() {
        let model = model_entry("deepseek-flash", "DeepSeek-V4.1-Flash", true);
        assert_eq!(model["capabilities"]["supports_tool_calls"], true);
        assert_eq!(model["supports_responses"], true);
        assert_eq!(model["supports_vision"], true);
    }

    #[test]
    fn legacy_flash_alias_points_to_current_flash() {
        let model = legacy_alias_entry("deepseek-v4-flash");
        assert_eq!(model["deprecated"], true);
        assert_eq!(model["alias_for"], "deepseek-flash");
    }
}
