//! Integration tests for the proxy layer with a mock upstream server.
//!
//! These tests start a real axum mock upstream + the deeplossless proxy,
//! then send requests through the proxy and verify correct forwarding,
//! storage, and DAG context injection. Tests cover both Chat Completions
//! and Responses API protocols, including tool call/result round-tripping
//! and streaming conversion.

use axum::{routing::post, Json, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

// ── Shared test helpers ──────────────────────────────────────────────────

type CapturedRequest = Arc<Mutex<Option<Value>>>;

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
    deeplossless::AppState {
        upstream: format!("http://{}", upstream_addr),
        api_key: std::sync::Arc::new(std::sync::Mutex::new(Some("test-key".to_string()))),
        admin_key: std::sync::Arc::new(std::sync::Mutex::new(None)),
        db,
        dag,
        compactor,
        client: reqwest::Client::new(),
        summarizer_model: "deepseek-v4-flash".into(),
        cycle: std::sync::Arc::new(std::sync::Mutex::new(
            deeplossless::runtime::ExecutionCycle::new(deeplossless::runtime::RuntimeProfile::Minimal),
        )),
        dry_run: false,
        response_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
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
    let db = state.db.clone();

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
    state.db.tool_cache_put(&cname, &args_hash, "src/main.rs:42 found foo", &["src/main.rs".to_string()]).unwrap();

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

#[tokio::test]
async fn chat_completions_cache_intercept() {
    // Pre-populate the tool cache, then have the mock upstream return a
    // tool call via the /v1/chat/completions endpoint. The proxy should
    // intercept and return the cached result as a text delta.
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
        axum::serve(listener, app).with_graceful_shutdown(async { rx.await.ok(); }).await.unwrap();
    });
    let _shutdown = tx;

    let state = build_proxy_state(addr, "chat_cache").await;
    use deeplossless::tool_cache;
    let (cname, args_hash) = tool_cache::cache_key("grep", r#"{"pattern":"foo"}"#);
    state.db.tool_cache_put(&cname, &args_hash, "cached: found foo", &[]).unwrap();

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

    // Cache interception: should contain cached result
    assert!(body.contains("cached: found foo"), "body should contain cached result: {body}");
    // Should NOT forward the tool call
    assert!(!body.contains("function"), "body should NOT contain function_call: {body}");
    // Must have [DONE] termination
    assert!(body.contains("[DONE]"), "should end with [DONE]: {body}");
}

#[tokio::test]
async fn multi_tool_call_one_cache_hit() {
    // Two tool calls: grep (cached) + read_file (not cached).
    // Only the grep call should be intercepted; read_file should pass through.
    let captured: CapturedRequest = Arc::new(Mutex::new(None));
    let upstream_req = captured.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |body: Json<Value>| {
        let cap = upstream_req.clone();
        async move {
            *cap.lock().unwrap() = Some(body.0.clone());
            let stream = futures::stream::iter(vec![
                // Tool 0: grep (will cache-hit)
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"grep\",\"arguments\":\"{\\\"pattern\\\":\\\"foo\\\"}\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                // Tool 1: read_file (no cache)
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c2\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"file_path\\\":\\\"src/lib.rs\\\"}\"}}]},\"index\":0}],\"usage\":null}\n\n"
                )),
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                    "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":10,\"total_tokens\":20}}\n\n"
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

    let state = build_proxy_state(addr, "multi_tool").await;
    // Pre-populate grep cache only
    use deeplossless::tool_cache;
    let (cname, args_hash) = tool_cache::cache_key("grep", r#"{"pattern":"foo"}"#);
    state.db.tool_cache_put(&cname, &args_hash, "found foo at line 42", &[]).unwrap();

    let proxy_addr = start_proxy(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/chat/completions", proxy_addr))
        .header("authorization", "Bearer sk-test")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role":"user","content":"search foo and read lib.rs"}],
            "stream": true,
        }))
        .send().await.unwrap();

    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();

    // grep should be intercepted (cached)
    assert!(body.contains("found foo at line 42"),
        "body should contain cached grep result: {body}");
    // read_file should pass through (no cache for it)
    assert!(body.contains("read_file"),
        "body should contain read_file tool call (not cached): {body}");
    // Must have [DONE]
    assert!(body.contains("[DONE]"), "should end with [DONE]: {body}");
}

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
    state.db.tool_cache_put(&cname, &args_hash, "src/lib.rs:42 found process_data", &[]).unwrap();

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
    let db = state.db.clone();

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
    let conv_id = state.db.find_or_create_conversation("audit_int_fp", "deepseek-v4-flash").unwrap();
    for i in 0..4 {
        let outcome = match i {
            0 => "success",
            1 => "success",
            2 => "blocked",
            _ => "cache_hit",
        };
        state.db.store_execution_unit(conv_id, "", "grep", "{}", if i == 2 { "Error: fail" } else { "ok" }, "", outcome, &[]).unwrap();
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

    let conv_id = state.db.find_or_create_conversation("score_int_fp", "deepseek-v4-flash").unwrap();
    state.db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
    state.db.store_execution_unit(conv_id, "", "build", "{}", "Error: fail", "", "blocked", &[]).unwrap();
    state.db.store_execution_unit(conv_id, "", "grep", "{}", "cached", "", "cache_hit", &[]).unwrap();

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

    let conv_id = state.db.find_or_create_conversation("motif_int_fp", "deepseek-v4-flash").unwrap();
    // Repeated grep→read_file pattern
    for _ in 0..2 {
        state.db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
        state.db.store_execution_unit(conv_id, "", "read_file", "{}", "data", "", "success", &[]).unwrap();
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
    let db = state.db.clone();
    let dag = state.dag.clone();

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
