use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::limit::RequestBodyLimitLayer;

use deeplossless::AppState;

#[derive(Parser)]
#[command(name = "deeplossless", version, about = "Lossless Context Management proxy for DeepSeek API")]
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
    #[arg(long, default_value = "~/.deepseek/lcm/lcm.db")]
    db_path: String,
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

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set");

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

    let compactor_config = deeplossless::compactor::CompactorConfig {
        summarizer: deeplossless::summarizer::SummarizerConfig {
            api_key: api_key.clone(),
            upstream: upstream.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let compactor = Arc::new(Mutex::new(
        deeplossless::compactor::Compactor::spawn(db.clone(), compactor_config),
    ));

    let state = AppState {
        upstream: cli.upstream,
        api_key,
        db,
        dag,
        compactor,
        client: reqwest::Client::builder()
            .build()?,
    };

    let app = deeplossless::proxy::routes()
        .with_state(state)
        .layer(CatchPanicLayer::new())
        .layer(RequestBodyLimitLayer::new(20 * 1024 * 1024)); // 20 MB

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("deeplossless listening on {addr}");
    tracing::info!("upstream: {}", upstream);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Graceful shutdown: wait for SIGTERM/SIGINT (Ctrl+C), then
    // drain pending requests with a 15-second timeout.
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let ctrl_c = tokio::signal::ctrl_c();
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .ok();
            tokio::select! {
                _ = ctrl_c => {},
                _ = async { if let Some(ref mut s) = term { s.recv().await; } } => {},
            }
            tracing::info!("shutdown signal received, draining requests…");
        })
        .await?;

    tracing::info!("deeplossless stopped");

    Ok(())
}
