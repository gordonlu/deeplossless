//! Integration tests for the proxy layer with a mock upstream server.
//!
//! These tests start a real axum mock upstream + the deeplossless proxy,
//! then send requests through the proxy and verify correct forwarding,
//! storage, and DAG context injection.

use axum::{routing::post, Json, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Start a mock upstream server on a random port, returning the address
/// and a shutdown sender.
async fn start_mock_upstream() -> (SocketAddr, oneshot::Sender<()>) {
    let app = Router::new().route("/v1/chat/completions", post(mock_chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async { rx.await.ok(); })
            .await
            .unwrap();
    });

    (addr, tx)
}

async fn mock_chat(Json(body): Json<Value>) -> Json<Value> {
    let is_streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    if is_streaming {
        // Streaming responses aren't tested via JSON assertion
        Json(json!({"error": "streaming not supported in mock"}))
    } else {
        Json(json!({
            "id": "mock-cmpl-001",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Mock response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
    }
}

#[tokio::test]
async fn proxy_non_streaming_round_trip() {
    let (upstream_addr, _shutdown) = start_mock_upstream().await;

    // Build proxy pointed at the mock
    let db = Arc::new(
        deeplossless::db::Database::builder()
            .path(std::env::temp_dir().join(format!("proxy_test_{}", std::process::id())))
            .build()
            .await
            .unwrap(),
    );
    let dag = Arc::new(
        deeplossless::dag::DagEngine::builder()
            .build(db.clone()),
    );
    let compactor = Arc::new(tokio::sync::Mutex::new(
        deeplossless::compactor::Compactor::spawn(
            db.clone(),
            deeplossless::compactor::CompactorConfig {
                summarizer: deeplossless::summarizer::SummarizerConfig {
                    api_key: "test".to_string(),
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ));

    let state = deeplossless::AppState {
        upstream: format!("http://{}", upstream_addr),
        api_key: std::sync::Arc::new(std::sync::Mutex::new(Some("test-key".to_string()))),
        db,
        dag,
        compactor,
        client: reqwest::Client::new(),
    };

    let app = deeplossless::proxy::routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Send a non-streaming request through the proxy
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/chat/completions", proxy_addr))
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"}
            ],
            "max_tokens": 100,
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "proxy should return 200");

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Mock response");
}
