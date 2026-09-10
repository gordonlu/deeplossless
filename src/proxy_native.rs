//! Native DeepSeek Responses API transport.
//!
//! DeepSeek now accepts OpenAI Responses API requests directly. This module
//! keeps the Responses wire protocol native end-to-end: requests go to the
//! upstream `/responses` endpoint and SSE bytes are forwarded unchanged.
//! DeepLossless still owns local continuity because DeepSeek's Responses API
//! is stateless and does not implement `previous_response_id`.

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
use sha2::{Digest, Sha256};
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;

use crate::event_store::EventType;
use crate::protocol::{ReasoningEffort, ReasoningEffortMode};
use crate::AppState;

const LOCAL_SESSION_FIELD: &str = "_deeplossless_session_id";
const STREAM_CHANNEL_CAPACITY: usize = 32;
type SseChunk = Result<axum::body::Bytes, std::convert::Infallible>;

// Preserve the public helpers used by existing callers/tests while the legacy
// Chat/Anthropic implementation remains in proxy.rs.
pub use crate::proxy_legacy::upstream_chat_url;
pub(crate) use crate::proxy_legacy::process_events;

/// Compose the new native Responses routes in front of the legacy router.
/// Unmatched requests are delegated verbatim to the old proxy implementation,
/// so Chat Completions, Anthropic, LCM APIs, metrics and health endpoints are
/// unaffected by this migration.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{response_id}", get(responses_retrieve))
        .route("/v1/models", get(list_models))
        .fallback(legacy_fallback)
}

async fn legacy_fallback(State(state): State<AppState>, req: Request) -> Response {
    let service = crate::proxy_legacy::routes().with_state(state);
    match service.oneshot(req).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
}

/// DeepSeek's documented OpenAI-compatible base URL is
/// `https://api.deepseek.com`; the Responses resource is `/responses`.
/// `/v1` bases are also accepted for OpenAI-style deployments.
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

fn cached_api_key(state: &AppState) -> String {
    state
        .api_key
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(|| "unset".to_string())
}

fn remember_api_key(state: &AppState, headers: &HeaderMap) {
    let Some(auth) = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
    else {
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
            tracing::warn!(target: "deeplossless::responses",
                %response_id, %error, "failed to load previous response");
            None
        }
    }
}

fn fallback_session_id(body: &Value) -> String {
    let seed = json!({
        "instructions": body.get("instructions").cloned().unwrap_or(Value::Null),
        "input": body.get("input").cloned().unwrap_or(Value::Null),
    });
    let encoded = serde_json::to_vec(&seed).unwrap_or_default();
    let digest = Sha256::digest(encoded);
    format!("responses:{}", hex::encode(&digest[..8]))
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

    fallback_session_id(body)
}

fn input_items(input: Option<&Value>) -> Vec<Value> {
    match input {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(text)) => vec![json!({
            "type": "message",
            "role": "user",
            "content": text,
        })],
        _ => Vec::new(),
    }
}

/// Merge an incremental Responses turn with locally retained history.
/// If a client already included a suffix of the history, remove the largest
/// exact overlap rather than duplicating items.
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
    let previous_id = request
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let history = if previous_id.is_some() {
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
                    tracing::warn!(target: "deeplossless::responses",
                        %session_id, %error, "failed to load response session");
                    None
                }
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let merged = merge_history(history, current_items.clone());
    if previous_id.is_some() {
        body["input"] = Value::Array(merged.clone());
    }

    // DeepSeek Responses is stateless. These OpenAI server-state fields are
    // handled locally (where applicable) and must not be relied on upstream.
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
        tracing::warn!(target: "deeplossless::responses",
            %response_id, %error, "failed to persist response object");
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
        tracing::warn!(target: "deeplossless::responses",
            %session_id, %error, "failed to persist response session");
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
    for item in items {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
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
                    json!({
                        "source":"responses-native",
                        "call_id":item.get("call_id").cloned().unwrap_or(Value::Null)
                    }),
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

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let request: Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":{"code":"BAD_REQUEST","message":error.to_string()}})),
            )
                .into_response();
        }
    };

    remember_api_key(&state, &headers);
    let session_id = response_session_id(&state, &request);
    let accept_sse = headers
        .get("accept")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    let streaming = request.get("stream").and_then(Value::as_bool).unwrap_or(false) || accept_sse;
    let (upstream_body, input_history, current_items) =
        build_upstream_body(&state, &request, &session_id, streaming);

    observe_items(&state, &session_id, &current_items);
    let _ = state.storage.db.insert_event_simple(
        EventType::RequestStart,
        &session_id,
        "",
        json!({
            "source":"responses-native",
            "model":upstream_body.get("model").cloned().unwrap_or(Value::Null),
            "stream":streaming,
            "input_items":current_items.len(),
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
            "id":"resp_dry_run",
            "object":"response",
            "status":"completed",
            "model":upstream_body.get("model").cloned().unwrap_or(Value::Null),
            "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"[dry-run] native Responses request saved to ~/.deeplossless/translated.json"}]}],
            "store":false,
        }))
        .into_response();
    }

    let upstream_url = upstream_responses_url(&state.upstream);
    tracing::debug!(target: "deeplossless::responses",
        %session_id, %upstream_url, streaming, "sending native Responses request");
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
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error":{"code":"UPSTREAM_ERROR","message":error.to_string()}})),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let upstream_headers = upstream.headers().clone();
    if !status.is_success() {
        let bytes = upstream.bytes().await.unwrap_or_default();
        let mut response = Response::new(Body::from(bytes.clone()));
        *response.status_mut() = status;
        if let Some(content_type) = upstream_headers.get("content-type") {
            response
                .headers_mut()
                .insert("content-type", content_type.clone());
        }
        let _ = state.storage.db.insert_event_simple(
            EventType::Error,
            &session_id,
            &String::from_utf8_lossy(&bytes),
            json!({"source":"responses-native","status":status.as_u16()}),
        );
        return response;
    }

    if !streaming {
        let bytes = match upstream.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error":{"code":"UPSTREAM_ERROR","message":error.to_string()}})),
                )
                    .into_response();
            }
        };
        if let Ok(response_value) = serde_json::from_slice::<Value>(&bytes) {
            let output = response_value
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            observe_items(&state, &session_id, &output);
            store_response_and_session(&state, &session_id, &input_history, &response_value);
        }
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
        let mut parse_buffer = String::new();
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

            // Forward the provider bytes unchanged. No synthetic lifecycle events
            // and, critically, no OpenAI/Chat `[DONE]` marker are added.
            if tx.send(Ok(bytes.clone())).await.is_err() {
                break;
            }

            parse_buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = parse_buffer.find("\n\n") {
                let frame = parse_buffer[..pos].to_string();
                parse_buffer.drain(..pos + 2);
                if let Some(response) = terminal_response_from_frame(&frame) {
                    terminal_response = Some(response);
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
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error":{"code":"NOT_FOUND","message":"response not found"}})),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"code":"STORAGE_ERROR","message":error.to_string()}})),
        )
            .into_response(),
    }
}

async fn list_models() -> Response {
    Json(json!({
        "object":"list",
        "data":[
            {
                "id":"deepseek-v4-flash",
                "object":"model",
                "owned_by":"deepseek",
                "model_version":"DeepSeek-V4-Flash-0731",
                "context_window":1_000_000,
                "max_context_tokens":1_000_000,
                "max_output_tokens":393_216,
                "supports_reasoning":true,
                "supports_tool_calls":true,
                "supports_responses":true
            },
            {
                "id":"deepseek-v4-pro",
                "object":"model",
                "owned_by":"deepseek",
                "model_version":"DeepSeek-V4-Pro-0813",
                "context_window":1_000_000,
                "max_context_tokens":1_000_000,
                "max_output_tokens":393_216,
                "supports_reasoning":true,
                "supports_tool_calls":true,
                "supports_responses":true
            },
            {
                "id":"deepseek-v4-flash-vision-exp",
                "object":"model",
                "owned_by":"deepseek",
                "model_version":"DeepSeek-V4-Flash-Vision-Exp",
                "context_window":1_000_000,
                "max_context_tokens":1_000_000,
                "max_output_tokens":393_216,
                "supports_reasoning":true,
                "supports_tool_calls":true,
                "supports_responses":true,
                "supports_vision":true
            }
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
            "data: {\"type\":\"response.completed\",\"sequence_number\":9,",
            "\"response\":{\"id\":\"resp_1\",\"output\":[]}}\n"
        );
        let response = terminal_response_from_frame(frame).unwrap();
        assert_eq!(response["id"], "resp_1");
    }

    #[test]
    fn local_session_metadata_is_not_exposed() {
        let response = strip_local_fields(json!({
            "id":"resp_1",
            LOCAL_SESSION_FIELD:"session-a"
        }));
        assert!(response.get(LOCAL_SESSION_FIELD).is_none());
    }
}
