use axum::{Router, routing::post, http::StatusCode};
use clap::Parser;
use std::sync::Arc;
use tracing::info;

mod db;
mod proxy;
mod tokenizer;

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

    /// SQLite database path
    #[arg(long, default_value = "~/.deepseek/lcm/lcm.db")]
    db_path: String,
}

#[derive(Clone)]
struct AppState {
    upstream: String,
    api_key: String,
    db: Arc<db::Database>,
    client: reqwest::Client,
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

    let state = AppState {
        upstream: cli.upstream,
        api_key,
        db: Arc::new(db::Database::open(&cli.db_path).await?),
        client: reqwest::Client::builder()
            .build()?,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/health", post(|| async { StatusCode::OK }))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("deeplossless listening on {addr}");
    info!("upstream: {}", cli.upstream);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
