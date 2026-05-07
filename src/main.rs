use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;

mod compactor;
mod dag;
mod db;
mod proxy;
mod session;
mod summarizer;
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

    /// SQLite database path (supports ~/ and $HOME expansion)
    #[arg(long, default_value = "~/.deepseek/lcm/lcm.db")]
    db_path: String,
}

#[derive(Clone)]
struct AppState {
    upstream: String,
    api_key: String,
    db: Arc<db::Database>,
    dag: Arc<dag::DagEngine>,
    compactor: Arc<Mutex<compactor::Compactor>>,
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

    let upstream = cli.upstream.clone();
    let db = Arc::new(
        db::Database::builder()
            .path(&cli.db_path)
            .build()
            .await?,
    );
    let dag = Arc::new(
        dag::DagEngine::builder()
            .build(db.clone()),
    );

    let compactor_config = crate::compactor::CompactorConfig {
        summarizer: summarizer::SummarizerConfig {
            api_key: api_key.clone(),
            upstream: upstream.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let compactor = Arc::new(Mutex::new(
        crate::compactor::Compactor::spawn(db.clone(), compactor_config),
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

    let app = proxy::routes().with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("deeplossless listening on {addr}");
    tracing::info!("upstream: {}", upstream);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
