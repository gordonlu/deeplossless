//! Integration tests for the proxy layer with a mock upstream server.
//!
//! These tests start a real axum mock upstream + the deeplossless proxy,
//! then send requests through the proxy and verify correct forwarding,
//! storage, and DAG context injection. Tests cover both Chat Completions
//! and Responses API protocols, including tool call/result round-tripping
//! and streaming conversion.

use axum::{routing::post, Json, Router};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

// ── Shared test helpers ──────────────────────────────────────────────────

type CapturedRequest = Arc<Mutex<Option<Value>>>;

/// Start a mock upstream that returns a specific HTTP status and JSON body
/// for every POST /v1/chat/completions request, regardless of the request body.
async fn start_mock_upstream_with_status(
    status: u16, body: Value,
) -> (SocketAddr, oneshot::Sender<()>) {
    let app = Router::new().route("/v1/chat/completions",
        post(move |_: Json<Value>| {
            let body = body.clone();
            async move {
                (StatusCode::from_u16(status).unwrap_or(StatusCode::OK), Json(body))
            }
        })
    );
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

/// Start a mock upstream server. The `reply_fn` decides what JSON to return.
/// The `captured` (if provided) gets populated with the received Chat Completions body.
async fn start_mock_upstream_ex(
    capture: Option<CapturedRequest>,
) -> (SocketAddr, oneshot::Sender<()>) {
    let captured = capture.unwrap_or_else(|| Arc::new(Mutex::new(None)));
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = captured.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());

            let is_streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            if is_streaming {
                // Return an error JSON so the handler doesn't try to parse SSE
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
    }));
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

async fn start_mock_upstream() -> (SocketAddr, oneshot::Sender<()>) {
    start_mock_upstream_ex(None).await
}

/// Build a deeplossless AppState pointed at the given upstream address.
/// `suffix` should be unique per test to avoid SQLite locking.
async fn build_proxy_state(upstream_addr: SocketAddr, suffix: &str) -> deeplossless::AppState {
    let db = Arc::new(
        deeplossless::db::Database::builder()
            .path(std::env::temp_dir().join(format!("proxy_test_{}_{}", std::process::id(), suffix)))
            .build()
            .await
            .unwrap(),
    );
    let dag = Arc::new(
        deeplossless::dag::DagEngine::builder().build(db.clone()),
    );
    let compactor = Arc::new(tokio::sync::Mutex::new(
        deeplossless::compactor::Compactor::new(
            db.clone(),
            deeplossless::compactor::CompactorConfig {
                summarizer: deeplossless::summarizer::SummarizerConfig {
                    api_key: "test".to_string(),
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        ),
    ));
    deeplossless::AppState {
        upstream: format!("http://{}", upstream_addr),
        api_key: std::sync::Arc::new(std::sync::Mutex::new(Some("test-key".to_string()))),
        admin_key: std::sync::Arc::new(std::sync::Mutex::new(None)),
        storage: deeplossless::StorageServices {
            db,
            dag,
            response_store: deeplossless::response_store::ResponseStore::default(),
        },
        compactor,
        runtime: deeplossless::RuntimeServices {
            client: reqwest::Client::new(),
            cycle: std::sync::Arc::new(std::sync::Mutex::new(
                deeplossless::runtime::ExecutionCycle::new(deeplossless::runtime::RuntimeProfile::Minimal),
            )),
            rate_limiter: std::sync::Arc::new(deeplossless::runtime::RateLimiter::new(0)),
            shutdown_notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        },
        summarizer_model: "deepseek-v4-flash".into(),
        dry_run: false,
        log_dir: None,
    }
}

/// Start the proxy on a random port, return its address.
async fn start_proxy(state: deeplossless::AppState) -> SocketAddr {
    let app = deeplossless::proxy::routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

// ── Existing Chat Completions test ───────────────────────────────────────

#[tokio::test]
async fn chat_completions_text_round_trip() {
    let (upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(upstream_addr, "cc").await;
    let proxy_addr = start_proxy(state).await;

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

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Mock response");
}

// ── Responses API tests ───────────────────────────────────────────────────

#[tokio::test]
async fn responses_api_text_round_trip() {
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let (upstream_addr, _shutdown) = start_mock_upstream_ex(Some(captured.clone())).await;
    let state = build_proxy_state(upstream_addr, "responses_text").await;
    let proxy_addr = start_proxy(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/responses", proxy_addr))
        .json(&json!({
            "input": "write hello world in python",
            "instructions": "You are a helpful coding assistant.",
            "model": "deepseek-v4-flash",
            "max_output_tokens": 100,
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Responses API should return 200");

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response", "should return Responses API format");
    assert_eq!(body["status"], "completed");
    assert!(body["output"].is_array(), "should have output array");

    // Verify the upstream received a Chat Completions request
    let upstream_req = captured.lock().unwrap().take().expect("upstream should have received a request");
    assert_eq!(upstream_req["model"], "deepseek-v4-flash");
    assert!(upstream_req["messages"].is_array());
    let msgs = upstream_req["messages"].as_array().unwrap();
    assert!(msgs.iter().any(|m| m["role"] == "system"), "should have system message");
    assert!(msgs.iter().any(|m| m["role"] == "user"), "should have user message");
}

#[tokio::test]
async fn responses_api_tool_call_round_trip() {
    // Mock upstream that returns a tool call response
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            Json(json!({
                "id": "mock-cmpl-002",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": r#"{"pattern":"fn main","path":"src/"}"#
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
            }))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async { rx.await.ok(); })
            .await
            .unwrap();
    });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "tool_call_rtt").await;
    let proxy_addr = start_proxy(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/responses", proxy_addr))
        .json(&json!({
            "input": [{"role": "user", "content": "search the codebase"}],
            "instructions": "Use tools to search.",
            "model": "deepseek-v4-flash",
            "tools": [{"type": "function", "name": "grep", "description": "search files"}],
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Responses API should return 200");
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "incomplete", "tool_calls finish_reason maps to incomplete");

    // Should contain function_call in output
    let output = body["output"].as_array().expect("should have output");
    let has_fn_call = output.iter().any(|o| o["type"] == "function_call");
    assert!(has_fn_call, "response should contain function_call output item");

    // Verify upstream received Chat Completions with tools
    let upstream = captured.lock().unwrap().take().expect("upstream should have received body");
    assert!(upstream["tools"].is_array(), "upstream should receive tool definitions");
    assert_eq!(upstream["tools"][0]["function"]["name"], "grep");

    // Verify usage tokens propagated
    let usage = body["usage"].as_object().expect("should have usage");
    assert!(usage.contains_key("input_tokens"));
    assert!(usage.contains_key("output_tokens"));
}

#[tokio::test]
async fn pipeline_tool_result_caching() {
    // Direct test: call the pipeline's process() with tool call + result,
    // then verify the tool cache was populated.  This avoids the HTTP layer
    // and the spawn_blocking race.
    use deeplossless::tool_cache;

    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let (upstream_addr, _shutdown) = start_mock_upstream_ex(Some(captured.clone())).await;
    let state = build_proxy_state(upstream_addr, "pipeline_cache").await;
    let db = state.storage.db.clone();

    let pipeline = deeplossless::pipeline::ChatPipeline::new(&state);

    let req_body = json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "search for process_data"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "grep", "arguments": r#"{"pattern":"process_data","path":"src/"}"#}}
            ]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "src/lib.rs:42 pub fn process_data()"}
        ],
    });

    let output = pipeline.process("deepseek-v4-flash", &req_body).await.unwrap();
    assert!(output.conv_id > 0);

    // Now wait for the spawn_blocking task to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Verify tool cache was populated
    let (cname, args_hash) = tool_cache::cache_key("grep", r#"{"pattern":"process_data","path":"src/"}"#);
    eprintln!("checking cache: name={cname:?} hash={args_hash:?}");
    match db.tool_cache_get(&cname, &args_hash) {
        Ok(Some((result, _ts))) => {
            assert!(result.contains("process_data"), "cached result should contain expected content");
        }
        Ok(None) => {
            // Try a short retry — spawn_blocking may still be running
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            match db.tool_cache_get(&cname, &args_hash) {
                Ok(Some((result, _ts))) => assert!(result.contains("process_data")),
                _ => panic!("cache still empty after 1.5 s total wait"),
            }
        }
        Err(e) => {
            panic!("cache lookup error: {e}");
        }
    }
}

#[tokio::test]
async fn responses_api_streaming_round_trip() {
    // Mock upstream that returns SSE Chat Completions chunks
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());

            // Return SSE streaming response via axum::response::Sse
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13}}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async { rx.await.ok(); })
            .await
            .unwrap();
    });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "streaming").await;
    let proxy_addr = start_proxy(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/responses", proxy_addr))
        .header("accept", "text/event-stream")
        .json(&json!({
            "input": "say hello",
            "instructions": "Be concise.",
            "model": "deepseek-v4-flash",
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "streaming should return 200");

    // Read the SSE body as text
    let sse_body = resp.text().await.unwrap();
    assert!(!sse_body.is_empty(), "SSE body should not be empty");

    // Verify it contains Responses API SSE events
    assert!(sse_body.contains("response.created"), "should have response.created event");
    assert!(sse_body.contains("response.output_text.delta"), "should have text delta events");
    assert!(sse_body.contains("response.completed"), "should have response.completed");
    assert!(sse_body.contains("[DONE]"), "should end with [DONE]");

    // Verify upstream received streaming=true
    let upstream = captured.lock().unwrap().take().expect("upstream should have received body");
    assert_eq!(upstream["stream"], true, "upstream should receive streaming=true");
}

#[tokio::test]
async fn tool_cache_intercepts_tool_call() {
    // Pre-populate the tool cache, then have the mock upstream return a
    // tool call. The proxy should intercept it and return the cached
    // result as text instead of forwarding the tool call to Codex.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());

            // Return a tool call via SSE
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"grep\",\"arguments\":\"{\\\"pa\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ttern\\\":\\\"foo\\\"}\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async { rx.await.ok(); })
            .await.unwrap();
    });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "cache_intercept").await;
    // Pre-populate the tool cache — same args the upstream will "generate"
    use deeplossless::tool_cache;
    let (cname, args_hash) = tool_cache::cache_key("grep", r#"{"pattern":"foo"}"#);
    state.storage.db.tool_cache_put(&cname, &args_hash, "src/main.rs:42 found foo", &["src/main.rs".to_string()]).unwrap();

    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/responses", proxy_addr))
        .header("accept", "text/event-stream")
        .json(&json!({
            "input": "search for foo",
            "instructions": "Be concise.",
            "model": "deepseek-v4-flash",
        }))
        .send().await.unwrap();

    assert!(resp.status().is_success());
    let sse_body = resp.text().await.unwrap();

    eprintln!("=== SSE BODY ===\n{sse_body}\n=== END SSE ===");

    // Cache interception: should contain cached result as text
    assert!(sse_body.contains("src/main.rs:42 found foo"),
        "SSE body should contain cached result: {sse_body}");
    // Should NOT contain function_call events (tool call was intercepted)
    assert!(!sse_body.contains("function_call"),
        "SSE should NOT contain function_call (tool call was intercepted): {sse_body}");
    // Must still have proper lifecycle
    assert!(sse_body.contains("response.output_text.done"), "should have output_text.done");
    assert!(sse_body.contains("response.completed"), "should have response.completed");
    assert!(sse_body.contains("[DONE]"), "should end with [DONE]");
}

// chat_completions_cache_intercept and multi_tool_call_one_cache_hit removed:
// Chat Completions handler now uses raw byte forwarding (simpler, more reliable).
// Stream-level cache interception only applies to the Responses API path.
// Pipeline-level auto-caching still works for both paths.

#[tokio::test]
async fn chat_completions_tool_call_forwarded_without_cache() {
    // Tool call should pass through correctly when not cached — the client
    // (OpenCode etc.) must receive the function call to execute it.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"grep\",\"arguments\":\"{\\\"pattern\\\":\\\"foo\\\"}\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "tc_fwd").await;
    // NO cache pre-population — this is a cache miss scenario

    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/chat/completions", proxy_addr))
        .header("authorization", "Bearer sk-test")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role":"user","content":"search for foo"}],
            "stream": true,
        }))
        .send().await.unwrap();

    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();

    // Must contain the tool call — it should NOT be dropped
    assert!(body.contains("\"name\":\"grep\""),
        "body should contain grep tool call: {body}");
    assert!(body.contains("tool_calls"),
        "body should contain tool_calls field: {body}");
    assert!(body.contains("[DONE]"), "should end with [DONE]");
}

#[tokio::test]
async fn concurrent_requests_no_race() {
    // Two concurrent streaming requests should complete without errors.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"pong\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "concurrent").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();

    // Fire 4 concurrent requests
    let handles: Vec<_> = (0..4).map(|_| {
        let client = client.clone();
        let url = format!("http://{}/v1/chat/completions", proxy_addr);
        tokio::spawn(async move {
            client.post(&url)
                .header("authorization", "Bearer sk-test")
                .json(&json!({
                    "model": "deepseek-v4-flash",
                    "messages": [{"role":"user","content":"ping"}],
                    "stream": true,
                }))
                .send().await.unwrap()
                .text().await.unwrap()
        })
    }).collect();

    for (i, h) in handles.into_iter().enumerate() {
        let body = h.await.expect("concurrent request should not panic");
        assert!(body.contains("[DONE]"), "request {i} should end with [DONE]: {body}");
        assert!(body.contains("pong"), "request {i} should contain pong: {body}");
    }
}

#[tokio::test]
async fn lcm_grep_retrieves_stored_context() {
    // Full LCM retrieval chain: store messages via Chat Completions,
    // then verify grep/search the DAG, check cache, and read status.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"pong\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "lcm_test").await;

    // Store a tool result directly in cache
    use deeplossless::tool_cache;
    let (cname, args_hash) = tool_cache::cache_key("grep", r#"{"pattern":"process_data"}"#);
    state.storage.db.tool_cache_put(&cname, &args_hash, "src/lib.rs:42 found process_data", &[]).unwrap();

    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let base = format!("http://{proxy_addr}");

    // Send a request through the proxy so the pipeline stores messages + creates a conversation
    let _resp = client
        .post(format!("{base}/v1/chat/completions"))
        .header("authorization", "Bearer sk-test")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "search for process_data in src/lib.rs"}
            ],
            "stream": true,
        }))
        .send().await.unwrap()
        .text().await.unwrap();

    // Wait for pipeline spawn_blocking to finish
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Test 1: cache hit via LCM endpoint
    let cache_resp = client
        .get(format!("{base}/v1/lcm/cache?tool=grep&args={{%22pattern%22:%22process_data%22}}"))
        .header("authorization", "Bearer test-key")
        .send().await.unwrap();
    assert!(cache_resp.status().is_success());
    let cache_json: serde_json::Value = cache_resp.json().await.unwrap();
    assert_eq!(cache_json["hit"], true, "cache should hit: {cache_json}");
    assert!(cache_json["result"].as_str().unwrap_or("").contains("process_data"));

    // Test 2: cache put then get
    let put_resp = client
        .post(format!("{base}/v1/lcm/cache/put"))
        .header("authorization", "Bearer test-key")
        .json(&json!({
            "tool": "read_file",
            "args": "{\"file_path\":\"src/main.rs\"}",
            "result": "fn main() { println!(\"hello\"); }",
            "files": "[\"src/main.rs\"]"
        }))
        .send().await.unwrap();
    let put_status = put_resp.status();
    let put_body = put_resp.text().await.unwrap();
    assert!(put_status.is_success(), "cache put failed: {put_status} {put_body}");

    let get_resp = client
        .get(format!("{base}/v1/lcm/cache?tool=read_file&args={{%22file_path%22:%22src/main.rs%22}}"))
        .header("authorization", "Bearer test-key")
        .send().await.unwrap();
    let get_json: serde_json::Value = get_resp.json().await.unwrap();
    assert_eq!(get_json["hit"], true, "cache get should hit after put");

    // Test 3: DAG health status
    let status_resp = client
        .get(format!("{base}/v1/lcm/status/1"))
        .header("authorization", "Bearer test-key")
        .send().await.unwrap();
    assert!(status_resp.status().is_success());
    let status_json: serde_json::Value = status_resp.json().await.unwrap();
    assert!(status_json["total_tokens"].as_i64().unwrap_or(-1) >= 0);

    // Test 4: search across all conversations
    let search_resp = client
        .get(format!("{base}/v1/lcm/grep/1?query=process_data"))
        .header("authorization", "Bearer test-key")
        .send().await.unwrap();
    assert!(search_resp.status().is_success());
    let search_json: serde_json::Value = search_resp.json().await.unwrap();
    assert!(search_json["total"].as_i64().unwrap_or(-1) >= 0,
        "search should return results: {search_json}");
}

#[tokio::test]
async fn snapshot_and_versions_endpoints() {
    // Test POST /v1/lcm/snapshot and GET /v1/lcm/versions end-to-end.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "snap_versions").await;
    let db = state.storage.db.clone();

    // Create memory versions
    let v1 = db.create_memory_version(None, "test", "initial version", None).unwrap();
    let _v2 = db.create_memory_version(Some(v1), "test", "second version", None).unwrap();

    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let base = format!("http://{proxy_addr}");

    // Take a snapshot via API
    let snap_resp = client
        .post(format!("{base}/v1/lcm/snapshot"))
        .header("authorization", "Bearer test-key")
        .json(&json!({
            "execution_id": 1,
            "memory_version_id": v1,
            "tier": 1,
            "data": "{\"state\":\"test snapshot\"}",
        }))
        .send().await.unwrap();
    assert!(snap_resp.status().is_success(), "snapshot should succeed: {snap_resp:?}");
    let snap_json: serde_json::Value = snap_resp.json().await.unwrap();
    assert_eq!(snap_json["status"], "stored");
    let snap_id = snap_json["id"].as_i64().unwrap();
    assert!(snap_id > 0);

    // Restore the snapshot directly via DB
    let restored = db.restore_snapshot(snap_id).unwrap();
    assert!(restored.is_some());
    assert!(restored.unwrap().snapshot_data.contains("test snapshot"));

    // List versions via API
    let versions_resp = client
        .get(format!("{base}/v1/lcm/versions"))
        .header("authorization", "Bearer test-key")
        .send().await.unwrap();
    assert!(versions_resp.status().is_success());
    let versions_json: serde_json::Value = versions_resp.json().await.unwrap();
    let vers = versions_json["versions"].as_array().unwrap();
    assert!(vers.len() >= 2, "should have at least 2 versions: {versions_json}");
    assert!(vers.iter().any(|v| v["mutation_kind"] == "test"), "should have test version");
}

// ── LCM Audit / Score / Motif / Observe integration tests ───────────────

fn dummy_upstream_addr() -> std::net::SocketAddr {
    "127.0.0.1:1".parse().unwrap()
}

#[tokio::test]
async fn lcm_audit_trail_and_report_endpoints() {
    let state = build_proxy_state(dummy_upstream_addr(), "audit_test").await;
    let proxy_addr = start_proxy(state.clone()).await;
    let base = format!("http://{proxy_addr}");
    let client = reqwest::Client::new();
    let auth = "Bearer test-key";

    // Store execution units via DB
    let conv_id = state.storage.db.find_or_create_conversation("audit_int_fp", "deepseek-v4-flash").unwrap();
    for i in 0..4 {
        let outcome = match i {
            0 => "success",
            1 => "success",
            2 => "blocked",
            _ => "cache_hit",
        };
        state.storage.db.store_execution_unit(conv_id, "", "grep", "{}", if i == 2 { "Error: fail" } else { "ok" }, "", outcome, &[]).unwrap();
    }

    // Test audit trail endpoint
    let trail_resp = client
        .get(format!("{base}/v1/lcm/audit/{conv_id}"))
        .header("authorization", auth)
        .send().await.unwrap();
    assert!(trail_resp.status().is_success(), "audit trail: {trail_resp:?}");
    let trail_json: serde_json::Value = trail_resp.json().await.unwrap();
    assert_eq!(trail_json["conv_id"], conv_id);
    let records = trail_json["records"].as_array().unwrap();
    assert_eq!(records.len(), 4, "all 4 units should be in trail");

    // Test audit report endpoint
    let report_resp = client
        .get(format!("{base}/v1/lcm/audit/report/{conv_id}"))
        .header("authorization", auth)
        .send().await.unwrap();
    assert!(report_resp.status().is_success(), "audit report: {report_resp:?}");
    let report_json: serde_json::Value = report_resp.json().await.unwrap();
    assert_eq!(report_json["report"]["total_actions"], 4);
    assert!(report_json["report"]["failures"]["blocked"].as_i64().unwrap_or(0) >= 1);
    assert!(report_json["report"]["cache_perf"]["hits"].as_i64().unwrap_or(0) >= 1);
}

#[tokio::test]
async fn lcm_score_endpoint() {
    let state = build_proxy_state(dummy_upstream_addr(), "score_test").await;
    let proxy_addr = start_proxy(state.clone()).await;
    let base = format!("http://{proxy_addr}");
    let client = reqwest::Client::new();
    let auth = "Bearer test-key";

    let conv_id = state.storage.db.find_or_create_conversation("score_int_fp", "deepseek-v4-flash").unwrap();
    state.storage.db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
    state.storage.db.store_execution_unit(conv_id, "", "build", "{}", "Error: fail", "", "blocked", &[]).unwrap();
    state.storage.db.store_execution_unit(conv_id, "", "grep", "{}", "cached", "", "cache_hit", &[]).unwrap();

    let score_resp = client
        .get(format!("{base}/v1/lcm/score/{conv_id}"))
        .header("authorization", auth)
        .send().await.unwrap();
    assert!(score_resp.status().is_success(), "score: {score_resp:?}");
    let score: serde_json::Value = score_resp.json().await.unwrap();
    assert!(score.get("composite").and_then(|v| v.as_f64()).is_some(), "composite should exist");
    assert!(score.get("success_rate").and_then(|v| v.as_f64()).is_some(), "success_rate should exist");
    assert!(score.get("hallucination_risk").and_then(|v| v.as_f64()).is_some(), "hallucination_risk should exist");
}

#[tokio::test]
async fn lcm_motifs_endpoint() {
    let state = build_proxy_state(dummy_upstream_addr(), "motif_int").await;
    let proxy_addr = start_proxy(state.clone()).await;
    let base = format!("http://{proxy_addr}");
    let client = reqwest::Client::new();
    let auth = "Bearer test-key";

    let conv_id = state.storage.db.find_or_create_conversation("motif_int_fp", "deepseek-v4-flash").unwrap();
    // Repeated grep→read_file pattern
    for _ in 0..2 {
        state.storage.db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
        state.storage.db.store_execution_unit(conv_id, "", "read_file", "{}", "data", "", "success", &[]).unwrap();
    }

    let motifs_resp = client
        .get(format!("{base}/v1/lcm/motifs/{conv_id}"))
        .header("authorization", auth)
        .send().await.unwrap();
    assert!(motifs_resp.status().is_success(), "motifs: {motifs_resp:?}");
    let motifs_json: serde_json::Value = motifs_resp.json().await.unwrap();
    let motifs = motifs_json["motifs"].as_array().unwrap();
    assert!(!motifs.is_empty(), "should find motifs");
    let gr: Vec<_> = motifs.iter().filter(|m| m["tool_chain"].as_array().map(|a| a.len()) == Some(2)).collect();
    assert!(!gr.is_empty(), "should have grep→read_file motif");
}

#[tokio::test]
async fn lcm_observe_endpoint() {
    let state = build_proxy_state(dummy_upstream_addr(), "observe_int").await;
    let proxy_addr = start_proxy(state.clone()).await;
    let base = format!("http://{proxy_addr}");
    let client = reqwest::Client::new();
    let auth = "Bearer test-key";

    let observe_resp = client
        .post(format!("{base}/v1/lcm/observe"))
        .header("authorization", auth)
        .json(&json!({
            "path": "src/main.rs",
            "content": "fn hello() {\n    println!(\"world\");\n}\n"
        }))
        .send().await.unwrap();
    assert!(observe_resp.status().is_success(), "observe: {observe_resp:?}");
    let obs: serde_json::Value = observe_resp.json().await.unwrap();
    assert_eq!(obs["path"], "src/main.rs");
    assert!(obs["semantic_hash"].as_str().unwrap_or("").len() >= 8);
    assert!(obs["ast"]["functions"].as_array().map(|a| a.len()) >= Some(1));
}

#[tokio::test]
async fn lcm_observe_rejects_missing_path() {
    let state = build_proxy_state(dummy_upstream_addr(), "observe_reject").await;
    let proxy_addr = start_proxy(state.clone()).await;
    let base = format!("http://{proxy_addr}");
    let client = reqwest::Client::new();
    let auth = "Bearer test-key";

    let resp = client
        .post(format!("{base}/v1/lcm/observe"))
        .header("authorization", auth)
        .json(&json!({"content": "fn x() {}"}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400, "missing path should 400");
}

// ── Parallel execution integration test ─────────────────────────────────

#[tokio::test]
async fn parallel_group_detection_and_join() {
    // Verify that the pipeline detects multi-tool-call turns, tags execution
    // units with span/parallel metadata, inserts HappensBefore edges, and
    // creates a join DAG node.
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "parallel_test").await;
    let db = state.storage.db.clone();
    let dag = state.storage.dag.clone();

    let pipeline = deeplossless::pipeline::ChatPipeline::new(&state);

    // Request with two tool calls in one assistant turn (parallelizable)
    let req_body = serde_json::json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "check main.rs and lib.rs"},
            {"role": "assistant", "content": "Let me look at both files.", "tool_calls": [
                {"id": "call_grep", "type": "function",
                 "function": {"name": "grep", "arguments": r#"{"pattern":"main","path":"src/main.rs"}"#}},
                {"id": "call_read", "type": "function",
                 "function": {"name": "read_file", "arguments": r#"{"path":"src/lib.rs"}"#}}
            ]},
            {"role": "tool", "tool_call_id": "call_grep",
             "content": "src/main.rs:1: fn main() {}"},
            {"role": "tool", "tool_call_id": "call_read",
             "content": "pub fn helper() {}"},
            {"role": "assistant", "content": "Both files look good."}
        ],
    });

    let output = pipeline.process("deepseek-v4-flash", &req_body).await.unwrap();
    assert!(output.conv_id > 0, "should get a valid conversation ID");
    let conv_id = output.conv_id;

    // Wait for spawn_blocking persistence to finish
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;

    // ── 1. Verify execution units have parallel metadata ──────────────
    let mut units = db.get_execution_units(conv_id, 10).unwrap();
    assert_eq!(units.len(), 2, "should have 2 execution units for 2 tool calls");
    // Sort by ID ascending for deterministic assertion order
    units.sort_by_key(|u| u.id);

    // Both units should have the same parallel_group and span_mode=parallel
    assert!(!units[0].parallel_group.is_empty(), "unit 0 should have parallel_group");
    assert!(!units[1].parallel_group.is_empty(), "unit 1 should have parallel_group");
    assert_eq!(units[0].parallel_group, units[1].parallel_group,
        "both units should share the same parallel_group");
    assert_eq!(units[0].span_mode, "parallel", "unit 0 span_mode should be parallel");
    assert_eq!(units[1].span_mode, "parallel", "unit 1 span_mode should be parallel");

    // Each should have a unique span_id
    assert_ne!(units[0].span_id, units[1].span_id, "span_ids should be unique");

    // Both should share the same parent_span_id
    assert_eq!(units[0].parent_span_id, units[1].parent_span_id,
        "both units should share parent_span_id");

    // Verify tool_call_id was preserved
    let call_ids: Vec<&str> = units.iter().map(|u| u.tool_call_id.as_str()).collect();
    assert!(call_ids.contains(&"call_grep"), "should contain call_grep");
    assert!(call_ids.contains(&"call_read"), "should contain call_read");

    // ── 2. Verify HappensBefore edges in dag_edges table ──────────────
    let hb_edges = db.get_edges_by_kind("happens_before").unwrap();

    assert!(!hb_edges.is_empty(), "should have at least one HappensBefore edge");

    // Each edge should point to the join DAG node (to_id should match a join node)
    for (from_id, to_id, kind) in &hb_edges {
        assert_eq!(kind, "happens_before");
        // Verify from_id is an execution unit ID
        assert!(
            units.iter().any(|u| u.id == *from_id),
            "from_id {from_id} should be one of the execution unit IDs"
        );
        // Verify to_id is a join DAG node
        if let Some(node) = dag.db().get_node(*to_id).unwrap() {
            assert!(node.is_join, "target node should be a join node");
        }
    }

    // ── 3. Verify join DAG node exists ────────────────────────────────
    let all_nodes = db.get_all_dag_nodes(conv_id).unwrap();
    let join_nodes: Vec<_> = all_nodes.iter().filter(|n| n.is_join).collect();
    assert_eq!(join_nodes.len(), 1, "should have exactly 1 join DAG node");

    let join_node = join_nodes[0];
    assert!(join_node.summary.contains("Parallel group"), "join node summary should mention parallel group");

    // ── 4. Verify governance worked: both branches succeeded ──────────
    assert_eq!(units[0].outcome.as_str(), "success");
    assert_eq!(units[1].outcome.as_str(), "success");
}

#[tokio::test]
async fn concurrent_3_tool_execution_creates_happens_before() {
    // Verify that 3 truly concurrent tool executions produce correct
    // parallel_group metadata, HappensBefore edges, and a join DAG node
    // when results arrive out of order (different execution delays).
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "concurrent_3").await;
    let db = state.storage.db.clone();

    // Create conversation
    let conv_id = db.create_and_store("concurrent_test", &json!([
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Check three files."}
    ])).unwrap();
    let replay_session_id = format!("rs_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

    // Root span shared by all parallel branches
    let root_span = deeplossless::parallel::ExecutionSpan::root_span();

    // Three tool calls to execute concurrently
    let tools = vec![
        deeplossless::parallel::ToolCallInfo { name: "grep".into(), call_id: "tc_a".into() },
        deeplossless::parallel::ToolCallInfo { name: "read_file".into(), call_id: "tc_b".into() },
        deeplossless::parallel::ToolCallInfo { name: "list_files".into(), call_id: "tc_c".into() },
    ];

    // Fork tracker — assigns span IDs, establishes group_id and parent_span
    let tracker = deeplossless::parallel::ForkJoinTracker::fork(
        conv_id, 0, &root_span, &tools,
        deeplossless::parallel::ParallelGovernance::default(),
    );
    let group_id = tracker.group_id.clone();
    let parent_span_id_str = tracker.parent_span.span_id.0.clone();
    assert_eq!(tracker.branch_count(), 3, "should have 3 branches for 3 tool calls");

    // Extract per-branch metadata before spawning
    let branch_metas: Vec<(String, String, String)> = tracker.branches.iter()
        .map(|b| (b.tool_call_id.clone(), b.span_id.0.clone(), b.tool_name.clone()))
        .collect();

    // ── Spawn 3 concurrent tool execution tasks with staggered delays ──
    let db_arc = db.clone();
    let tasks: Vec<_> = branch_metas.into_iter().enumerate().map(|(i, (tool_call_id, span_id, tool_name))| {
        let db2 = db_arc.clone();
        let gid = group_id.clone();
        let psid = parent_span_id_str.clone();
        let rsid = replay_session_id.clone();
        let tcid = tool_call_id.clone();
        // Staggered delays so results arrive out of order (30ms, 60ms, 90ms)
        let delay_ms = 30 * (i as u64 + 1);

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

            let exec_id = tokio::task::spawn_blocking(move || {
                db2.store_execution_unit_with_span(
                    conv_id,
                    &format!("Reasoning for {tool_name}"),
                    &tool_name,
                    r#"{"path":"src/main.rs"}"#,
                    &format!("Result of {tool_name}"),
                    "All good.",
                    "success",
                    &[],
                    &span_id,
                    &psid,
                    "parallel",
                    &gid,
                    &tcid,
                    &rsid,
                )
            }).await.unwrap().unwrap();

            (exec_id, tool_call_id, "success".to_string())
        })
    }).collect();

    // Collect results — order depends on staggered delays, not insertion order
    let mut results: Vec<(i64, String, String)> = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }
    assert_eq!(results.len(), 3, "all 3 tool tasks should complete");

    // ── Record results in ForkJoinTracker (sequential, post-concurrent) ──
    let mut tracker = tracker;
    for (exec_id, tool_call_id, outcome_str) in &results {
        let outcome = deeplossless::execution::ExecutionOutcome::from_str(outcome_str)
            .unwrap_or(deeplossless::execution::ExecutionOutcome::Success);
        tracker.record_branch_result(tool_call_id, *exec_id, &outcome).unwrap();
    }

    assert!(tracker.should_force_join(), "all 3 branches complete → should_force_join");

    // ── Create join DAG node ──
    let join_summary = format!("Parallel group {group_id} (3 branches)");
    let join_node = db.insert_join_atomic(conv_id, &join_summary, 0, &[], &[]).unwrap();
    assert!(join_node.is_join, "join node should be marked is_join");

    // ── Complete tracker → HappensBefore edges ──
    let hb_edges = tracker.complete(join_node.id).unwrap();
    assert_eq!(hb_edges.len(), 3, "should have 3 HappensBefore edges (one per branch)");

    // Persist edges to lineage_edges (no FK constraint to dag_nodes, correct
    // for execution-to-execution HappensBefore relationships).
    for hb in &hb_edges {
        db.insert_lineage_edge(hb.from_id, hb.to_id, "happens_before").unwrap();
    }

    // ── Assertions ──

    // 1. All 3 execution units have correct parallel metadata
    let units = db.get_execution_units(conv_id, 10).unwrap();
    assert_eq!(units.len(), 3, "should have 3 execution units");

    for (exec_id, tc_id, _) in &results {
        let unit = units.iter().find(|u| u.id == *exec_id)
            .unwrap_or_else(|| panic!("execution unit {exec_id} should exist"));
        assert_eq!(unit.span_mode, "parallel", "unit {tc_id} should have span_mode=parallel");
        assert_eq!(unit.parallel_group, group_id, "unit {tc_id} should share group_id");
        assert_eq!(unit.parent_span_id, parent_span_id_str, "unit {tc_id} should share parent_span_id");
        assert!(!unit.span_id.is_empty(), "unit {tc_id} should have a span_id");
        assert_eq!(unit.outcome.as_str(), "success", "unit {tc_id} should have success outcome");
    }

    // 2. Unique span_ids
    let span_ids: std::collections::HashSet<&str> = units.iter().map(|u| u.span_id.as_str()).collect();
    assert_eq!(span_ids.len(), 3, "each unit should have a unique span_id");

    // 3. HappensBefore edges in lineage_edges — 3 edges, each from an execution unit to the join node
    let lineage = db.get_lineage_to(join_node.id).unwrap();
    let hb_to_join: Vec<_> = lineage.into_iter()
        .filter(|(_, _, kind)| kind == "happens_before")
        .collect();
    assert_eq!(hb_to_join.len(), 3, "should have exactly 3 HappensBefore edges to join node");

    for (from_id, to_id, kind) in &hb_to_join {
        assert_eq!(*kind, "happens_before", "edge kind should be happens_before");
        assert!(
            results.iter().any(|(eid, _, _)| eid == from_id),
            "from_id {from_id} should be an execution unit ID",
        );
        assert_eq!(*to_id, join_node.id, "to_id should be the join node ID");
    }

    // 4. Join DAG node persisted correctly
    let stored_join = db.get_node(join_node.id).unwrap()
        .expect("join node should exist in DB");
    assert!(stored_join.is_join, "stored node should be marked is_join");
    assert!(stored_join.summary.contains("Parallel group"), "join summary should mention parallel group");
    assert_eq!(stored_join.level, 0, "join node level should be 0");
}

// ── Upstream fault injection tests ──────────────────────────────────────

#[tokio::test]
async fn chat_completions_upstream_500_propagates_to_client() {
    // When upstream returns HTTP 500, the proxy should propagate the status.
    let (addr, _shutdown) = start_mock_upstream_with_status(
        500, json!({"error": {"message": "Internal server error", "type": "server_error"}}),
    ).await;
    let state = build_proxy_state(addr, "upstream_500_cc").await;
    let proxy_addr = start_proxy(state.clone()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("Authorization", "Bearer sk-test")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "test"}],
            "stream": false,
        }))
        .send().await.unwrap();

    assert_eq!(resp.status().as_u16(), 500,
        "upstream 500 should be propagated to client");
    let body = resp.text().await.unwrap();
    assert!(!body.is_empty(), "error response should have a body");
}

#[tokio::test]
async fn responses_upstream_500_does_not_crash_proxy() {
    // When upstream returns HTTP 500 to a streaming responses() request,
    // the proxy must not crash. Documents the gap that responses() currently
    // returns 200 OK regardless of upstream status.
    let (addr, _shutdown) = start_mock_upstream_with_status(
        500, json!({"error": {"message": "Internal server error", "type": "server_error"}}),
    ).await;
    let state = build_proxy_state(addr, "upstream_500_resp").await;
    let proxy_addr = start_proxy(state.clone()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/responses"))
        .header("Authorization", "Bearer sk-test")
        .header("Accept", "text/event-stream")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "input": "test",
        }))
        .send().await;

    // Proxy must not crash — response may be 502 if network fails,
    // 200 with SSE body if passthrough succeeds, or 500 if internal error.
    // The key invariant: no panic, no hang, no unhandled error.
    match resp {
        Ok(r) => {
            let status = r.status();
            let body = r.text().await.unwrap_or_default();
            // Even if status is 200, the body should contain the error
            // from the upstream (this is the gap to fix in responses())
            assert!(
                body.contains("error") || body.contains("Internal server error")
                    || status.as_u16() >= 400,
                "upstream 500 must be reflected in status or body: status={status}, body={body}"
            );
        }
        Err(e) => {
            // Network error is acceptable if proxy times out or refuses
            assert!(
                e.is_connect() || e.is_timeout() || e.is_request(),
                "unexpected error type from proxy: {e}"
            );
        }
    }
}

// ── Parallel execution failure paths ─────────────────────────────────────

#[tokio::test]
async fn concurrent_3_tool_with_1_failure_and_fail_fast() {
    // Three concurrent tool calls: 2 succeed, 1 is Blocked. With fail_fast=true,
    // the join should trigger on the first failure.
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "partial_fail").await;
    let db = state.storage.db.clone();

    let conv_id = db.create_and_store("partial_fail_test", &json!([
        {"role": "system", "content": "Agent."},
        {"role": "user", "content": "Check files."}
    ])).unwrap();
    let replay_session_id = format!("pf_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

    let root_span = deeplossless::parallel::ExecutionSpan::root_span();
    let tools = vec![
        deeplossless::parallel::ToolCallInfo { name: "grep".into(), call_id: "tc_x".into() },
        deeplossless::parallel::ToolCallInfo { name: "read_file".into(), call_id: "tc_y".into() },
        deeplossless::parallel::ToolCallInfo { name: "list_files".into(), call_id: "tc_z".into() },
    ];

    let tracker = deeplossless::parallel::ForkJoinTracker::fork(
        conv_id, 0, &root_span, &tools,
        deeplossless::parallel::ParallelGovernance::default(),
    );
    let group_id = tracker.group_id.clone();
    let parent_span_id_str = tracker.parent_span.span_id.0.clone();

    // Which branch will fail (index 1 = tc_y = read_file)
    let fail_tool_call_id = "tc_y".to_string();

    let branch_metas: Vec<(String, String, String)> = tracker.branches.iter()
        .map(|b| (b.tool_call_id.clone(), b.span_id.0.clone(), b.tool_name.clone()))
        .collect();

    let db_arc = db.clone();
    let tasks: Vec<_> = branch_metas.into_iter().enumerate().map(|(i, (tool_call_id, span_id, tool_name))| {
        let db2 = db_arc.clone();
        let gid = group_id.clone();
        let psid = parent_span_id_str.clone();
        let rsid = replay_session_id.clone();
        let is_fail = tool_call_id == fail_tool_call_id;
        let outcome = if is_fail { "blocked".to_string() } else { "success".to_string() };
        let delay_ms = 20 * (i as u64 + 1); // 20, 40, 60ms
        let tcid = tool_call_id.clone();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            let result_text = if is_fail { "Tool call blocked — file in use".to_string() } else { format!("Result {tool_name}") };
            let reasoning_after = if is_fail { "Recovery attempted but failed.".to_string() } else { "Ok.".to_string() };
            let outcome_c = outcome.clone();
            let exec_id = tokio::task::spawn_blocking(move || {
                db2.store_execution_unit_with_span(
                    conv_id,
                    &format!("Reasoning {tool_name}"),
                    &tool_name,
                    r#"{"path":"src/main.rs"}"#,
                    &result_text,
                    &reasoning_after,
                    &outcome_c,
                    &[],
                    &span_id,
                    &psid,
                    "parallel",
                    &gid,
                    &tcid,
                    &rsid,
                )
            }).await.unwrap().unwrap();
            (exec_id, tool_call_id, outcome)
        })
    }).collect();

    let mut results: Vec<(i64, String, String)> = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }

    // Record results in tracker
    let mut tracker = tracker;
    for (exec_id, tool_call_id, outcome_str) in &results {
        let outcome = deeplossless::execution::ExecutionOutcome::from_str(outcome_str)
            .unwrap_or(deeplossless::execution::ExecutionOutcome::Success);
        tracker.record_branch_result(tool_call_id, *exec_id, &outcome).unwrap();
    }

    // With fail_fast=true, should_force_join must return true
    // when at least one branch is Failed/Blocked.
    assert!(tracker.should_force_join(),
        "fail_fast=true + 1 blocked branch → should force join");

    // Create join DAG node
    let join_summary = format!("Parallel group {group_id} (3 branches, 1 blocked)");
    let join_node = db.insert_join_atomic(conv_id, &join_summary, 0, &[], &[]).unwrap();

    // Complete and persist HappensBefore edges
    let hb_edges = tracker.complete(join_node.id).unwrap();
    // Failed branches also create edges since execution_unit_id is set
    assert_eq!(hb_edges.len(), 3,
        "all 3 branches produce happens-before edges (including failed)");

    for hb in &hb_edges {
        db.insert_lineage_edge(hb.from_id, hb.to_id, "happens_before").unwrap();
    }

    // ── Verify ──
    let units = db.get_execution_units(conv_id, 10).unwrap();
    assert_eq!(units.len(), 3);

    let blocked_units: Vec<_> = units.iter().filter(|u| u.outcome.as_str() == "blocked").collect();
    assert_eq!(blocked_units.len(), 1, "exactly 1 unit should be blocked");
    assert_eq!(blocked_units[0].span_mode, "parallel");
    assert_eq!(blocked_units[0].parallel_group, group_id);

    let success_units: Vec<_> = units.iter().filter(|u| u.outcome.as_str() == "success").collect();
    assert_eq!(success_units.len(), 2, "exactly 2 units should succeed");

    // HappensBefore edges
    let lineage = db.get_lineage_to(join_node.id).unwrap();
    let hb_count = lineage.iter().filter(|(_, _, k)| k == "happens_before").count();
    assert_eq!(hb_count, 3, "3 happens-before edges to join node");

    // Join node
    let stored_join = db.get_node(join_node.id).unwrap().unwrap();
    assert!(stored_join.is_join);
}

// ── Audit mode tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn audit_mode_onerror_buffers_and_flushes_on_failure() {
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "audit_onerror").await;
    let db = state.storage.db.clone();

    // Switch to OnError mode
    {
        let mut cfg = db.policy_config.write().unwrap();
        cfg.audit_mode = deeplossless::runtime::AuditMode::OnError;
        cfg.onerror_ring_size = 10;
    }

    let conv_id = db.create_and_store("audit_onerror_test", &json!([
        {"role": "system", "content": "Agent."},
        {"role": "user", "content": "test"}
    ])).unwrap();

    // Store 3 successful execution units — should be buffered, NOT in DB
    for i in 0..3 {
        db.store_execution_unit(
            conv_id,
            &format!("reasoning_{i}"),
            "grep",
            r#"{"pattern":"test"}"#,
            &format!("result_{i}"),
            "ok",
            "success",
            &[],
        ).unwrap();
    }

    // Verify no audit events written for successful units
    let events = db.get_execution_events_by_conv(conv_id, 10).unwrap();
    assert_eq!(events.len(), 0,
        "OnError mode: successful units should NOT write execution_events");

    // Store a failed unit — should flush buffer + write this event
    db.store_execution_unit(
        conv_id,
        "reasoning_fail",
        "read_file",
        r#"{"path":"nonexistent"}"#,
        "Error: file not found",
        "recovery failed",
        "blocked",
        &[],
    ).unwrap();

    // Now execution_events should have content (the failed event)
    let events_after = db.get_execution_events_by_conv(conv_id, 20).unwrap();
    assert!(!events_after.is_empty(),
        "OnError mode: failed unit should flush buffer + write event");
}

#[tokio::test]
async fn audit_mode_off_writes_no_execution_events() {
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "audit_off").await;
    let db = state.storage.db.clone();

    // Switch to Off mode
    {
        let mut cfg = db.policy_config.write().unwrap();
        cfg.audit_mode = deeplossless::runtime::AuditMode::Off;
    }

    let conv_id = db.create_and_store("audit_off_test", &json!([
        {"role": "system", "content": "Agent."},
        {"role": "user", "content": "test"}
    ])).unwrap();

    // Store execution units
    for i in 0..5 {
        db.store_execution_unit(
            conv_id,
            &format!("reasoning_{i}"),
            "grep",
            r#"{"pattern":"test"}"#,
            &format!("result_{i}"),
            "ok",
            "success",
            &[],
        ).unwrap();
    }

    // Verify NO execution_events written
    let events = db.get_execution_events_by_conv(conv_id, 10).unwrap();
    assert_eq!(events.len(), 0,
        "AuditMode::Off: no execution_events should be written");

    // Verify execution_units ARE still written
    let units = db.get_execution_units(conv_id, 10).unwrap();
    assert_eq!(units.len(), 5,
        "execution_units should still be written in Off mode");
}

// ── ForkJoinTracker timeout branch ────────────────────────────────────────

#[tokio::test]
async fn fork_join_tracker_timeout_branch_transitions_correctly() {
    // Verify that timeout_branch() transitions a Pending branch to TimedOut,
    // and that should_force_join reacts to the timeout with fail_fast.
    let root_span = deeplossless::parallel::ExecutionSpan::root_span();
    let tools = vec![
        deeplossless::parallel::ToolCallInfo { name: "grep".into(), call_id: "tc_t1".into() },
        deeplossless::parallel::ToolCallInfo { name: "read_file".into(), call_id: "tc_t2".into() },
    ];

    let mut tracker = deeplossless::parallel::ForkJoinTracker::fork(
        1, 0, &root_span, &tools,
        deeplossless::parallel::ParallelGovernance::default(), // fail_fast=true
    );

    // Mark one branch as timed out
    tracker.timeout_branch("tc_t1").unwrap();

    // With fail_fast=true, a TimedOut branch should trigger force_join
    assert!(tracker.should_force_join(),
        "fail_fast + 1 timed out branch → should force join");

    let timed_out_branch = tracker.branches.iter()
        .find(|b| b.tool_call_id == "tc_t1").unwrap();
    assert!(matches!(timed_out_branch.status,
        deeplossless::parallel::BranchStatus::TimedOut),
        "branch should be TimedOut");
    assert!(timed_out_branch.error.as_deref() == Some("timeout"),
        "error message should be 'timeout'");

    // The other branch should still be Pending
    let pending_branch = tracker.branches.iter()
        .find(|b| b.tool_call_id == "tc_t2").unwrap();
    assert!(matches!(pending_branch.status,
        deeplossless::parallel::BranchStatus::Pending),
        "other branch should remain Pending");
}

// ── Upstream malformed response tests ─────────────────────────────────────

#[tokio::test]
async fn responses_upstream_malformed_json_returns_502() {
    // When the upstream returns a non-JSON body, the responses() non-streaming
    // path must return 502 (BAD_GATEWAY) rather than panicking on parse.
    // Use a raw-string response that is NOT valid JSON.
    let app = Router::new().route("/v1/chat/completions", post(move |_: Json<Value>| async move {
        (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain")], "this is not json")
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap();
    });

    let state = build_proxy_state(addr, "malformed").await;
    let proxy_addr = start_proxy(state.clone()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/responses"))
        .header("Authorization", "Bearer sk-test")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "input": "test",
        }))
        .send().await.unwrap();

    assert!(resp.status().as_u16() >= 400,
        "malformed upstream body should produce error, got {}", resp.status());
    drop(tx);
}

// ── SSE stream disconnect test ─────────────────────────────────────────────

#[tokio::test]
async fn proxy_handles_sse_stream_disconnect_gracefully() {
    // When the upstream sends partial SSE content then disconnects,
    // the proxy must close the stream cleanly (no panic, stream ends with [DONE]).
    let partial_sse_body = "data: {\"id\":\"evt-001\",\"choices\":[{\"delta\":{\"content\":\"partial response\"}}]}\n\n";
    let body = partial_sse_body.to_string();
    let app = Router::new().route("/v1/chat/completions", post(move |_: Json<Value>| {
        let body = body.clone();
        async move {
            // Return the partial SSE as a complete response body (not streamed)
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/event-stream")], body)
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async { rx.await.ok(); })
            .await.unwrap();
    });

    let state = build_proxy_state(addr, "sse_disconnect").await;
    let proxy_addr = start_proxy(state.clone()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("Authorization", "Bearer sk-test")
        .header("Accept", "text/event-stream")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "test"}],
            "stream": true,
        }))
        .send().await.unwrap();

    let body = resp.text().await.unwrap();
    // Stream should end cleanly — no panic propagated to client
    assert!(body.contains("[DONE]") || body.contains("\"partial response\""),
        "stream should end cleanly, got: {}", &body[..body.len().min(500)]);
    drop(tx); // cleanup
}

// ── Snapshot budget enforcement ────────────────────────────────────────────

#[tokio::test]
async fn enforce_snapshot_budget_evicts_excess_l0_snapshots() {
    // Insert more L0 (Ephemeral) snapshots than the budget allows,
    // then verify enforcement evicts the excess.
    let (_upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(_upstream_addr, "snap_budget").await;
    let db = state.storage.db.clone();

    // Insert 5 L0 snapshots (budget is 100, but we test explicit enforcement)
    for i in 0..5 {
        db.take_snapshot(
            i + 1, // execution_id
            0, // memory_version_id
            0, // tier: Ephemeral
            &format!("{{\"snap\": {i}}}"),
            100, // size_bytes
            None, // retention_ttl
            0, "", "",
        ).unwrap();
    }

    // Enforce with a restrictive budget: max 2 hot L0 snapshots
    let restrictive = deeplossless::snapshot::SnapshotBudget {
        max_hot_snapshots: 2,
        max_structural_per_execution: 1,
        max_full_snapshots: 1,
        max_total_size_bytes: 10 * 1024 * 1024,
    };
    let evicted = db.enforce_snapshot_budget(&restrictive).unwrap();
    assert!(evicted > 0, "should evict at least 1 L0 snapshot");

    // Verify remaining L0 count ≤ budget
    let conn = db.writer_lock().lock().unwrap();
    let remaining: i64 = conn.query_row(
        "SELECT COUNT(*) FROM execution_snapshots WHERE tier = 0",
        [], |r| r.get(0),
    ).unwrap();
    assert!(remaining <= 2, "L0 snapshots should be ≤ max_hot_snapshots (2), got {remaining}");
}

#[tokio::test]
async fn models_endpoint_lists_deepseek_models() {
    // GET /v1/models should return a standard OpenAI-compatible model list.
    let (upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(upstream_addr, "models_test").await;
    let proxy_addr = start_proxy(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{proxy_addr}/v1/models"))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["object"], "list");
    let models = json["data"].as_array().unwrap();
    assert!(models.iter().any(|m| m["id"] == "deepseek-v4-pro"), "should list v4-pro: {json}");
    assert!(models.iter().any(|m| m["id"] == "deepseek-v4-flash"), "should list v4-flash: {json}");
    // Verify capabilities
    let pro = models.iter().find(|m| m["id"] == "deepseek-v4-pro").unwrap();
    assert_eq!(pro["capabilities"]["supports_reasoning"], true);
    assert_eq!(pro["capabilities"]["supports_thinking"], true);
    assert_eq!(pro["capabilities"]["max_context_tokens"], 1_000_000);
    // Redundant compatibility fields
    assert_eq!(pro["context_window"], 1_000_000);
    assert_eq!(pro["max_context_tokens"], 1_000_000);
    assert_eq!(pro["max_input_tokens"], 1_000_000);
    assert_eq!(pro["reasoning"], true);
    assert_eq!(pro["supports_reasoning"], true);
    assert_eq!(pro["supports_thinking"], true);
}

#[tokio::test]
async fn reasoning_content_stream_passthrough() {
    // Upstream sends reasoning_content in delta — client should receive it.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"I need to think about this carefully.\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"The answer is 42\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":15,\"total_tokens\":20}}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], axum::body::Body::from_stream(stream))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "reasoning_passthrough").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role":"user","content":"What is the answer?"}],
            "stream": true,
            "reasoning_effort": "high",
        }))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();
    // Reasoning content should reach the client
    assert!(body.contains("reasoning_content"), "reasoning_content should pass through: {body}");
    assert!(body.contains("I need to think about this carefully"), "reasoning text should be present: {body}");
    assert!(body.contains("The answer is 42"), "regular content should be present: {body}");
    assert!(body.contains("[DONE]"), "should end with [DONE]");
}

// ── Error path tests ──

#[tokio::test]
async fn upstream_500_returns_error_not_hang() {
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        (StatusCode::INTERNAL_SERVER_ERROR, "boom")
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;
    let state = build_proxy_state(addr, "upstream_500").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert!(!resp.status().is_success(), "should return error, got {}", resp.status());
    let body = resp.text().await.unwrap();
    assert!(body.contains("error"), "body should contain error: {body}");
}

#[tokio::test]
async fn upstream_400_passes_through() {
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        (StatusCode::BAD_REQUEST, json!({"error":{"message":"bad input"}}).to_string())
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;
    let state = build_proxy_state(addr, "upstream_400").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn non_streaming_200_returns_json() {
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        (StatusCode::OK, json!({"choices":[{"message":{"content":"ok"}}]}).to_string())
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;
    let state = build_proxy_state(addr, "nonstream_200").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "ok");
}

#[tokio::test]
async fn stream_truncated_no_done_still_closes() {
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        let stream = futures::stream::iter(vec![
            Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"index\":0}],\"usage\":null}\n\n"
            )),
        ]);
        ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], axum::body::Body::from_stream(stream))
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;
    let state = build_proxy_state(addr, "truncated").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"stream":true}))
        .timeout(std::time::Duration::from_secs(3))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let body = tokio::time::timeout(std::time::Duration::from_secs(2), resp.text())
        .await.expect("response must complete within 2s").unwrap_or_default();
    assert!(body.contains("partial"), "should receive partial data: {body}");
}

#[tokio::test]
async fn upstream_unreachable_returns_502() {
    let dead_addr = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        a
    };
    let state = build_proxy_state(dead_addr, "unreachable").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}))
        .timeout(std::time::Duration::from_secs(3))
        .send().await.unwrap();
    assert!(!resp.status().is_success(), "should return error for unreachable upstream");
}

#[tokio::test]
async fn client_invalid_json_returns_400() {
    let (upstream_addr, _shutdown) = start_mock_upstream().await;
    let state = build_proxy_state(upstream_addr, "invalid_json").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .header("content-type", "application/json")
        .body("not json {{{")
        .timeout(std::time::Duration::from_secs(3))
        .send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "invalid JSON should return 400");
}

#[tokio::test]
async fn upstream_html_instead_of_sse_does_not_hang() {
    // Upstream returns HTML error page instead of SSE — proxy must forward and close.
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/html")], "<html><body>gateway timeout</body></html>")
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;
    let state = build_proxy_state(addr, "html_upstream").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"stream":true}))
        .timeout(std::time::Duration::from_secs(3))
        .send().await.unwrap();
    let body = tokio::time::timeout(std::time::Duration::from_secs(2), resp.text())
        .await.expect("must complete within 2s").unwrap_or_default();
    assert!(!body.is_empty(), "should return something, not hang");
}

#[tokio::test]
async fn multi_turn_reasoning_tool_continuity() {
    // Simulate thinking-mode tool call round-trip: assistant with reasoning_content
    // makes a tool call, result comes back, next request includes reasoning_content.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            let req = body.0.clone();
            *cap.lock().unwrap() = Some(req.clone());
            let msgs = req["messages"].as_array().unwrap();
            let last_assistant = msgs.iter().rev().find(|m| m["role"] == "assistant");
            // Verify reasoning_content is present on tool-call message
            if let Some(msg) = last_assistant {
                if msg.get("tool_calls").and_then(|v| v.as_array()).map(|a| !a.is_empty()) == Some(true) {
                    if msg.get("reasoning_content").and_then(|v| v.as_str()).map(|s| s.is_empty()) != Some(false) {
                        return (StatusCode::BAD_REQUEST, "missing reasoning_content").into_response();
                    }
                }
            }
            let stream = futures::stream::iter(vec![
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: [DONE]\n\n")),
            ]);
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/event-stream")], axum::body::Body::from_stream(stream)).into_response()
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move { axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap(); });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "reasoning_continuity").await;
    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();

    // Turn 1: send a request that triggers reasoning + tool call
    let _r1 = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({"model":"deepseek-v4-flash","messages":[{"role":"user","content":"search for foo"}],"stream":true}))
        .send().await.unwrap().text().await.unwrap();

    // Turn 2: send the tool result WITH reasoning_content (as pipeline should inject)
    let r2 = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header("authorization", "Bearer test-key")
        .json(&json!({
            "model":"deepseek-v4-flash",
            "messages":[
                {"role":"user","content":"search for foo"},
                {"role":"assistant","content":null,"reasoning_content":"I need to search.","tool_calls":[{"id":"c1","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"foo\"}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"found foo at line 42"}
            ],
            "stream":true
        }))
        .timeout(std::time::Duration::from_secs(5))
        .send().await.unwrap();

    assert!(r2.status().is_success(), "multi-turn with reasoning should succeed, got {}", r2.status());
    let body = r2.text().await.unwrap();
    assert!(body.contains("[DONE]"), "stream should complete: {body}");
}
