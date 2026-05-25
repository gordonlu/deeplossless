use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "deeplossless", version, about = "Inference-aware execution runtime for AI coding agents")]
pub(crate) struct Cli {
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

    /// Audit mode: full (always write), onerror (buffer, flush on failure), off.
    #[arg(long, default_value = "full")]
    audit_mode: String,

    /// Snapshot mode: auto (trigger at semantic boundaries), manual (POST API only), off.
    #[arg(long, default_value = "manual")]
    snapshot_mode: String,

    /// OnError ring buffer size — how many recent events to buffer before flushing on failure.
    #[arg(long, default_value = "50")]
    onerror_ring_size: usize,

    /// TLS certificate path (PEM). Uses auto-generated self-signed cert by default.
    #[arg(long)]
    tls_cert: Option<String>,

    /// TLS private key path (PEM).
    #[arg(long)]
    tls_key: Option<String>,

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
    /// Install the self-signed TLS certificate as system-trusted.
    /// After running this, HTTPS clients won't show certificate errors.
    Trust,
}

async fn run_demo() -> anyhow::Result<()> {
    use deeplossless::runtime::RuntimeProfile;
    let demo_db_path = std::env::temp_dir().join("deeplossless_demo.db");
    let db = Arc::new(deeplossless::db::Database::builder()
        .path(&demo_db_path).build().await?);
    let dag = Arc::new(deeplossless::dag::DagEngine::builder()
        .max_level(3).recent_messages(20).build(db.clone()));
    let _cycle = Arc::new(std::sync::Mutex::new(
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
    println!("    curl https://localhost:8080/v1/lcm/runtime/stats | jq .\n");
    println!("  Benchmarks (no API key needed):");
    println!("    cargo test --test long_session_benchmark -- --nocapture\n");
    let _ = std::fs::remove_file(&demo_db_path);
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
    canonical.model = deeplossless::protocol::map_model(&canonical.model);
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

/// Install the self-signed certificate as system-trusted.
/// Copies cert to the system CA directory and updates the trust store.
fn run_trust() -> anyhow::Result<()> {
    // When run via sudo, $HOME points to /root. Use SUDO_USER to find
    // the real user's home directory instead.
    let user_home = std::env::var("SUDO_USER").ok()
        .and_then(|u| {
            let path = format!("/home/{u}");
            if std::path::Path::new(&path).exists() { Some(path) } else { None }
        })
        .unwrap_or_else(|| shellexpand::tilde("~").to_string());
    let cert_path = format!("{user_home}/.deeplossless/cert.pem");
    if !std::path::Path::new(&cert_path).exists() {
        anyhow::bail!("No certificate found at {cert_path}. Start deeplossless first to generate one.");
    }
    #[cfg(target_os = "linux")]
    {
        let dest = "/usr/local/share/ca-certificates/deeplossless.crt";
        std::fs::copy(&cert_path, dest)?;
        std::process::Command::new("update-ca-certificates").status()?;
        println!("Certificate installed. Restart your terminal or run:");
        println!("  export NODE_EXTRA_CA_CERTS={cert_path}");
        println!("  opencode  # or your agent");
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("security")
            .args(["add-trusted-cert", "-d", "-k", "/Library/Keychains/System.keychain", &cert_path])
            .status()?;
        println!("Certificate installed to system keychain.");
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        println!("Automatic trust installation is not supported on this OS.");
        println!("Set NODE_EXTRA_CA_CERTS={cert_path} in your environment.");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
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
    if matches!(cli.command, Some(Commands::Trust)) {
        return run_trust();
    }

    let audit_mode = match cli.audit_mode.as_str() {
        "off" | "Off" => deeplossless::runtime::AuditMode::Off,
        "onerror" | "on_error" | "OnError" => deeplossless::runtime::AuditMode::OnError,
        _ => deeplossless::runtime::AuditMode::Full,
    };
    let snapshot_mode = match cli.snapshot_mode.as_str() {
        "off" | "Off" => deeplossless::runtime::SnapshotMode::Off,
        "auto" | "Auto" => deeplossless::runtime::SnapshotMode::Auto,
        _ => deeplossless::runtime::SnapshotMode::Manual,
    };
    let cfg = deeplossless::runtime_coordinator::CoordinatorConfig {
        upstream: cli.upstream,
        db_path: cli.db_path,
        api_key: cli.api_key,
        admin_key: cli.admin_key,
        summarizer_model: cli.summarizer_model,
        rate_limit: cli.rate_limit,
        runtime_profile: cli.runtime_profile,
        dry_run: cli.dry_run,
        log_dir: cli.log_dir,
        policy_config: deeplossless::runtime::RuntimePolicyConfig {
            audit_mode,
            snapshot_mode,
            onerror_ring_size: cli.onerror_ring_size,
            ..Default::default()
        },
    };
    let coordinator = deeplossless::runtime_coordinator::RuntimeCoordinator::build(cfg).await?;
    let upstream = coordinator.state.upstream.clone();

    let app = coordinator.router();

    let addr: std::net::SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    let addr_str = addr.to_string();
    tracing::info!("deeplossless v{} listening on {addr_str} (built {})",
        env!("CARGO_PKG_VERSION"),
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    tracing::info!("upstream: {upstream}");

    // SSE connections are long-lived — graceful shutdown would hang forever.
    // Handle signal ourselves: brief drain, then exit.
    let tls_dir = shellexpand::tilde("~/.deeplossless").to_string();
    let _ = std::fs::create_dir_all(&tls_dir);
    let default_cert = format!("{tls_dir}/cert.pem");
    let default_key = format!("{tls_dir}/key.pem");

    let (tls_cert_path, tls_key_path) =
        if let (Some(c), Some(k)) = (cli.tls_cert.as_ref(), cli.tls_key.as_ref()) {
            (c.clone(), k.clone())
        } else {
            // Auto-generate self-signed cert — generated once, reused on restart.
            if !std::path::Path::new(&default_cert).exists() {
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()])?;
                std::fs::write(&default_cert, cert.cert.pem())?;
                std::fs::write(&default_key, cert.key_pair.serialize_pem())?;
                tracing::info!("self-signed cert generated at {tls_dir}/");
            }
            (default_cert, default_key)
        };

    let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls_cert_path, &tls_key_path).await?;
    tracing::info!("TLS enabled — HTTPS on {addr_str}");
    if cli.tls_cert.is_none()
        && !std::path::Path::new("/usr/local/share/ca-certificates/deeplossless.crt").exists() {
        tracing::info!("Self-signed cert — run `sudo deeplossless trust` once, then restart your terminal");
    }
    let handle = axum_server::Handle::new();
    let shutdown_handle = handle.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_handle.shutdown();
    });
    axum_server::bind_rustls(addr, tls_config)
        .handle(handle)
        .serve(app.into_make_service())
        .await?;
    coordinator.shutdown(std::time::Duration::from_secs(2)).await;

    tracing::info!("deeplossless stopped");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn tls_flags_accepted() {
        let args = Cli::try_parse_from([
            "deeplossless", "--tls-cert", "/tmp/c.pem", "--tls-key", "/tmp/k.pem",
        ]).unwrap();
        assert_eq!(args.tls_cert.as_deref(), Some("/tmp/c.pem"));
        assert_eq!(args.tls_key.as_deref(), Some("/tmp/k.pem"));
    }

    #[test]
    fn tls_flags_optional() {
        let args = Cli::try_parse_from(["deeplossless"]).unwrap();
        assert!(args.tls_cert.is_none());
        assert!(args.tls_key.is_none());
    }
}
