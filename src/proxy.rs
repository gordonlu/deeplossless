use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use crate::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    body: String,
) -> Response {
    let req_body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")).into_response(),
    };

    // Store messages in SQLite (fire-and-forget)
    let model = req_body["model"].as_str().unwrap_or("unknown").to_string();
    let messages = req_body["messages"].clone();
    let db = state.db.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = db.store_messages(&model, &messages) {
            warn!("failed to store messages: {e}");
        }
    });

    // Forward the request
    let upstream_url = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let resp = match state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", state.api_key))
        .header("Content-Type", "application/json")
        .json(&req_body)
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

    if req_body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        // SSE streaming path — forward each chunk as it arrives
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("stream error: {e}");
                        break;
                    }
                };
                if tx.send(Ok::<_, std::convert::Infallible>(chunk)).is_err() {
                    break;
                }
            }
        });

        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert("content-type", content_type);
        response.headers_mut().insert(
            "cache-control",
            "no-cache".parse().unwrap(),
        );
        response.headers_mut().insert(
            "x-accel-buffering",
            "no".parse().unwrap(),
        );
        response
    } else {
        // Non-streaming path — buffer the full response
        match resp.bytes().await {
            Ok(bytes) => {
                let mut response = Response::new(Body::from(bytes));
                *response.status_mut() = status;
                response.headers_mut().insert("content-type", content_type);
                response
            }
            Err(e) => {
                (StatusCode::BAD_GATEWAY, format!("upstream body error: {e}")).into_response()
            }
        }
    }
}
