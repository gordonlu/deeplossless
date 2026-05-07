use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;
use std::sync::Arc;
use tracing::{info, warn};

use crate::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Parse the incoming request body
    let mut req_body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")).into_response();
        }
    };

    // Store messages in SQLite (fire-and-forget for now)
    let model = req_body["model"].as_str().unwrap_or("unknown").to_string();
    let messages = req_body["messages"].clone();
    let db = state.db.clone();
    tokio::spawn(async move {
        if let Err(e) = db.store_messages(&model, &messages).await {
            warn!("failed to store messages: {e}");
        }
    });

    // Build the upstream URL
    let upstream_path = format!("{}/v1/chat/completions", state.upstream.trim_end_matches('/'));
    let upstream_uri: Uri = upstream_path.parse().unwrap();

    // Forward the request
    let mut reqwest_headers = reqwest::header::HeaderMap::new();
    reqwest_headers.insert(
        "Authorization",
        format!("Bearer {}", state.api_key).parse().unwrap(),
    );
    reqwest_headers.insert("Content-Type", "application/json".parse().unwrap());

    let resp = match state
        .client
        .post(upstream_uri.to_string())
        .headers(reqwest_headers)
        .json(&req_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response();
        }
    };

    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let resp_body = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, format!("upstream body error: {e}")).into_response();
        }
    };

    // Build response
    let mut response = Response::new(Body::from(resp_body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        "content-type",
        resp_headers
            .get("content-type")
            .cloned()
            .unwrap_or_else(|| "application/json".parse().unwrap()),
    );

    response
}
