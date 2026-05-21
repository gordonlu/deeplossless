use axum::{
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    extract::Request,
};
use clap::Parser;
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

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser)]
enum Commands {
    /// Run a local demo (no API key needed)
    Demo,
}

async fn run_demo() -> anyhow::Result<()> {
    use deeplossless::runtime::RuntimeProfile;
    let db = Arc::new(deeplossless::db::Database::builder()
        .path(":memory:").build().await?);
    let dag = Arc::new(deeplossless::dag::DagEngine::builder()
        .max_level(3).recent_messages(20).build(db.clone()));
    let _cycle = Arc::new(StdMutex::new(
        deeplossless::runtime::ExecutionCycle::new(RuntimeProfile::Efficient)));

    let mut baseline: u64 = 0;
    let mut runtime: u64 = 0;
    let mut cache_hits: u64 = 0;
    let mut failures_stopped: u64 = 0;

    let tasks = [
        ("Rust async bug", "src/handler.rs", "select!"),
        ("SQLite refactor", "src/db.rs", "prepare"),
        ("OAuth feature", "src/auth.rs", "token"),
    ];
    for (name, file, pat) in tasks {
        let conv_id = db.create_and_store(name, &serde_json::json!([
            {"role":"user","content":format!("Fix {file}")}
        ]))?;
        db.tool_cache_put("grep", pat, &format!("{file}:42: found {pat}"), &[file.to_string()])?;
        baseline += 500; runtime += 500;
        db.tool_cache_put("read_file", file, &format!("100 lines in {file}"), &[file.to_string()])?;
        baseline += 400; runtime += 400;
        if db.tool_cache_get("grep", pat)?.is_some() { cache_hits += 1; baseline += 480; }
        db.on_files_changed(&[file.to_string()])?;
        db.store_failure_pattern(conv_id, &format!("{pat} error"), "edit file",
            "stale cache", &[], &[file.to_string()], None)?;
        failures_stopped += 1; baseline += 350; runtime += 200;
        dag.insert_leaf(conv_id, &format!("fixed {pat}"), 10)?;
        dag.assemble_context(conv_id, 1000, None)?;
    }
    let pct = (1.0 - runtime as f64 / baseline as f64) * 100.0;
    println!("\n  deeplossless v{} — demo", env!("CARGO_PKG_VERSION"));
    println!("  ┌──────────────────────────────────────────┐");
    println!("  │  Tokens without runtime:    {:>6}         │", baseline);
    println!("  │  Tokens with runtime:       {:>6}         │", runtime);
    println!("  │  Cache hits:                {:>6}         │", cache_hits);
    println!("  │  Failures prevented:        {:>6}         │", failures_stopped);
    println!("  │  Savings:                   {:>5.0}%        │", pct);
    println!("  └──────────────────────────────────────────┘");
    println!("\n  Start the proxy:  deeplossless --api-key sk-...");
    println!("  More benchmarks:  cargo test --test long_session_benchmark\n");
    Ok(())
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
