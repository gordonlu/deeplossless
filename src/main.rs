use axum::{
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    extract::Request,
};
use clap::Parser;
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;

use deeplossless::AppState;

/// Rate limiter state (written once at startup, read on every request).
static RATE_COUNT: AtomicU64 = AtomicU64::new(0);
static RATE_MAX: AtomicU64 = AtomicU64::new(100);

async fn rate_limit_mw(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let max = RATE_MAX.load(Ordering::Relaxed);
    if max == 0 {
        return Ok(next.run(req).await);
    }
    let prev = RATE_COUNT.fetch_add(1, Ordering::Relaxed);
    if prev >= max {
        deeplossless::metrics::RATE_LIMIT_HITS.fetch_add(1, Ordering::Relaxed);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    Ok(next.run(req).await)
}

#[derive(Parser)]
#[command(name = "deeplossless", version, about = "Inference-aware execution runtime for AI coding agents")]
struct Cli {
    /// Listen address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Upstream DeepSeek API base URL
    #[arg(long, default_value = "https://api.deepseek.com")]
    upstream: String,

    /// SQLite database path (supports ~/ and $HOME expansion)
    #[arg(long, default_value = "~/.deeplossless/lcm.db")]
    db_path: String,

    /// DeepSeek API key (optional — extracted from first request's
    /// Authorization header if not provided here).
    #[arg(long, env = "DEEPSEEK_API_KEY")]
    api_key: Option<String>,

    /// Separate admin key for LCM endpoint authentication. Takes priority
    /// over DEEPSEEK_API_KEY for LCM auth. If unset, LCM falls back to
    /// the DeepSeek key for backward compatibility.
    #[arg(long, env = "ADMIN_KEY")]
    admin_key: Option<String>,

    /// Model used for background summarization (Level 1 & 2 LLM calls).
    #[arg(long, default_value = "deepseek-v4-pro", env = "SUMMARIZER_MODEL")]
    summarizer_model: String,

    /// Max requests per second (0 to disable rate limiting).
    #[arg(long, default_value = "100", env = "RATE_LIMIT")]
    rate_limit: u64,

    /// Runtime profile (minimal, efficient, exploratory, autonomous, custom).
    #[arg(long, default_value = "autonomous", env = "RUNTIME_PROFILE")]
    runtime_profile: String,

    /// Dry-run mode: skip upstream, write translated request to
    /// ~/.deeplossless/translated.json and return a mock response.
    #[arg(long)]
    dry_run: bool,

    /// Enable per-request JSON logging to a directory. One JSON line per
    /// request, written to `<log-dir>/session-<timestamp>.jsonl`. Disabled
    /// by default (no log writes for normal users).
    #[arg(long)]
    log_dir: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser)]
enum Commands {
    /// Run a local demo (no API key needed)
    Demo,
    /// Translate a saved Responses API request to Chat Completions format.
    /// Reads the JSON file, runs the full protocol translation pipeline,
    /// and pretty-prints the result. No API call is made.
    Translate {
        /// Path to a JSON file containing a Responses API request body
        file: String,
    },
}

async fn run_demo() -> anyhow::Result<()> {
    use deeplossless::runtime::RuntimeProfile;
    let db = Arc::new(deeplossless::db::Database::builder()
        .path("/tmp/deeplossless_demo.db").build().await?);
    let dag = Arc::new(deeplossless::dag::DagEngine::builder()
        .max_level(3).recent_messages(20).build(db.clone()));
    let _cycle = Arc::new(StdMutex::new(
        deeplossless::runtime::ExecutionCycle::new(RuntimeProfile::Efficient)));

    // Quick smoke test — insert a conversation to verify DB/DAG work
    let conv_id = db.create_and_store("demo", &serde_json::json!([
        {"role":"user","content":"Hello, deeplossless!"}
    ]))?;
    db.tool_cache_put("grep", "hello", "src/main.rs:42: found", &["src/main.rs".to_string()])?;
    let _ = db.tool_cache_get("grep", "hello")?;
    dag.insert_leaf(conv_id, "greeting", 5)?;
    dag.assemble_context(conv_id, 1000, None)?;

    println!("\n  deeplossless v{}", env!("CARGO_PKG_VERSION"));
    println!("  Inference-aware execution runtime for AI coding agents\n");
    println!("  Smoke test: database OK, DAG OK, cache OK\n");
    println!("  Features:");
    println!("    Tool Result Cache        — deterministic reuse, partial invalidation");
    println!("    Failure Memory           — avoids repeated failed fixes");
    println!("    Plan Persistence         — resumable execution state");
    println!("    Semantic DAG             — embedding dedup, BM25 search");
    println!("    Protocol translation     — Responses API → Chat Completions");
    println!("    Runtime Policy           — advisory cache/retry/context decisions\n");
    println!("  Start the proxy:");
    println!("    deeplossless --api-key sk-...");
    println!("  Then check:");
    println!("    curl http://127.0.0.1:8080/v1/lcm/runtime/stats | jq .\n");
    println!("  Benchmarks (no API key needed):");
    println!("    cargo test --test long_session_benchmark -- --nocapture\n");
    let _ = std::fs::remove_file("/tmp/deeplossless_demo.db");
    Ok(())
}

/// Translate a saved Responses API request body to Chat Completions format.
/// Reads the file, runs the full protocol translation pipeline, and prints
/// the result. No API call is made — purely offline.
fn run_translate(file: &str) -> anyhow::Result<()> {
    use serde_json::Value;

    let raw = std::fs::read_to_string(file)?;
    let req_body: Value = serde_json::from_str(&raw)?;

    // Show input structure
    if let Some(input) = req_body["input"].as_array() {
        println!("Input items ({}) — role/type:", input.len());
        for (i, item) in input.iter().enumerate() {
            let typ = item["type"].as_str().unwrap_or("-");
            let role = item["role"].as_str().unwrap_or("-");
            let call_id = item["call_id"].as_str().or_else(|| item["tool_call_id"].as_str());
            let has_call_id = call_id.is_some();
            println!("  [{i}] role={role} type={typ} call_id={}", call_id.unwrap_or("-"));
            if role == "tool" && !has_call_id {
                println!("    ⚠ role=tool but no tool_call_id/call_id!");
            }
        }
        println!();
    }

    // Run translation: Responses → Canonical → Chat Completions
    let mut canonical = deeplossless::protocol::responses::request_from_responses(&req_body);
    canonical.model = map_model_protocol(&canonical.model);
    let chat_body = deeplossless::protocol::chat_completions::request_to_chat(&canonical);

    // Show translated messages
    let msgs = chat_body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    println!("Translated messages ({}) — role:", msgs.len());
    for (i, msg) in msgs.iter().enumerate() {
        let role = msg["role"].as_str().unwrap_or("?");
        let has_tcid = msg.get("tool_call_id").is_some();
        let has_tc = msg.get("tool_calls").is_some();
        let content_len = msg["content"].as_str().map(|s| s.len()).or_else(|| msg["content"].as_array().map(|a| a.len())).unwrap_or(0);
        let flags: Vec<&str> = if has_tcid { vec!["tool_call_id"] } else { vec![] };
        let flags2: Vec<&str> = if has_tc { vec!["tool_calls"] } else { vec![] };
        println!("  [{i}] role={role} content_len={content_len} flags={}{}", flags.join(","), if flags2.is_empty() { String::new() } else { format!(",{}", flags2.join(",")) });
        if role == "tool" && !has_tcid {
            println!("    ⚠ BUG: role=tool but missing tool_call_id!");
        }
    }

    println!();
    println!("Stream: {}", chat_body["stream"].as_bool().unwrap_or(false));
    println!("Model: {}", chat_body["model"].as_str().unwrap_or("?"));
    println!();

    // Full translated body (pretty)
    println!("── Translated Chat Completions body ──");
    println!("{}", serde_json::to_string_pretty(&chat_body)?);

    Ok(())
}

fn map_model_protocol(model: &str) -> String {
    let m = model.to_lowercase();
    if m.starts_with("gpt-") || m.starts_with("o1") || m.starts_with("o3") || m == "gpt-5.5" {
        if m.contains("mini") { return "deepseek-v4-flash".into(); }
        return "deepseek-v4-pro".into();
    }
    if model.is_empty() || model == "auto" { return "deepseek-v4-pro".into(); }
    model.to_string()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "deeplossless=info".into()),
        )
        .init();

    let cli = Cli::parse();

    if matches!(cli.command, Some(Commands::Demo)) {
        return run_demo().await;
    }
    if let Some(Commands::Translate { file }) = cli.command {
        return run_translate(&file);
    }

    let upstream = cli.upstream.clone();
    let db = Arc::new(
        deeplossless::db::Database::builder()
            .path(&cli.db_path)
            .build()
            .await?,
    );
    let dag = Arc::new(
        deeplossless::dag::DagEngine::builder()
            .build(db.clone()),
    );

    let initial_api_key = cli.api_key.clone()
        .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok());

    let compactor_config = deeplossless::compactor::CompactorConfig {
        summarizer: deeplossless::summarizer::SummarizerConfig {
            api_key: initial_api_key.clone().unwrap_or_default(),
            upstream: upstream.clone(),
            model: cli.summarizer_model.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let compactor = Arc::new(Mutex::new(
        deeplossless::compactor::Compactor::spawn(db.clone(), compactor_config),
    ));

    let state = AppState {
        upstream: cli.upstream,
        api_key: Arc::new(std::sync::Mutex::new(initial_api_key)),
        admin_key: Arc::new(std::sync::Mutex::new(cli.admin_key)),
        db,
        dag,
        compactor,
        client: reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?,
        summarizer_model: cli.summarizer_model,
        cycle: Arc::new(std::sync::Mutex::new(
            deeplossless::runtime::ExecutionCycle::new(match cli.runtime_profile.as_str() {
                "minimal" => deeplossless::runtime::RuntimeProfile::Minimal,
                "efficient" => deeplossless::runtime::RuntimeProfile::Efficient,
                "exploratory" => deeplossless::runtime::RuntimeProfile::Exploratory,
                "autonomous" => deeplossless::runtime::RuntimeProfile::Autonomous,
                "custom" => deeplossless::runtime::RuntimeProfile::Custom,
                other => {
                    tracing::warn!(target: "deeplossless", "unknown runtime profile '{other}', falling back to autonomous");
                    deeplossless::runtime::RuntimeProfile::Autonomous
                }
            })
        )),
        dry_run: cli.dry_run,
        response_store: Arc::new(std::sync::Mutex::new(HashMap::new())),
        log_dir: cli.log_dir,
    };

    // Reset rate counter every second
    RATE_MAX.store(cli.rate_limit, Ordering::Relaxed);
    if cli.rate_limit > 0 {
        tokio::spawn(async {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                RATE_COUNT.store(0, Ordering::Relaxed);
            }
        });
    }

    let app = deeplossless::proxy::routes()
        .with_state(state)
        .layer(axum::middleware::from_fn(deeplossless::metrics::middleware))
        .layer(CatchPanicLayer::new())
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn(rate_limit_mw))
        .layer(RequestBodyLimitLayer::new(20 * 1024 * 1024)); // 20 MB

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("deeplossless listening on {addr}");
    tracing::info!("upstream: {}", upstream);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Graceful shutdown: wait for SIGINT (Ctrl+C), then
    // drain pending requests with a 15-second timeout.
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutdown signal received, draining requests…");
        })
        .await?;

    tracing::info!("deeplossless stopped");

    Ok(())
}
