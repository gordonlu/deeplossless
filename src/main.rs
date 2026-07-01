use clap::Parser;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

#[derive(Parser)]
#[command(name = "deeplossless", version, about = "Inference-aware execution runtime for AI coding agents")]
pub(crate) struct Cli {
    /// Listen address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port (HTTPS)
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Plain HTTP port for localhost agents (0 = disabled).
    /// Sandboxed agents (OpenClaw, etc.) may not trust self-signed certs.
    #[arg(long, default_value = "8081")]
    http_port: u16,

    /// Upstream DeepSeek API base URL
    #[arg(long, default_value = "https://api.deepseek.com")]
    upstream: String,

    /// Upstream API path suffix (e.g. /v1/chat/completions for DeepSeek,
    /// /chat/completions for Ark API).
    #[arg(long, default_value = "/v1/chat/completions")]
    upstream_path: String,

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

    /// Record raw request/response bytes for protocol debugging.
    /// Writes `<dir>/req_N.json` and `<dir>/rsp_N.txt` per request — no
    /// parsing, no reserialization. Diff these against direct DeepSeek traces.
    #[arg(long)]
    record: Option<String>,

    /// Pure byte-level passthrough — no pipeline, no context injection,
    /// no reasoning capture, no header modification. Identical to direct
    /// DeepSeek at the HTTP level. For isolating protocol bugs.
    #[arg(long)]
    passthrough: bool,

    /// Skip context injection only (still captures reasoning, modifies headers).
    /// For isolating whether DAG context injection causes the hang.
    #[arg(long)]
    no_pipeline: bool,

    /// Override DAG soft threshold (0.0–1.0). Default 0.80 = compact at 80% of
    /// context window. Set 0.01 for testing compaction with small conversations.
    #[arg(long)]
    dag_threshold: Option<f64>,

    /// Max LLM summarizer calls per session (0 = unlimited). Default 1000.
    /// Each call costs ~3K tokens. Caps background compaction cost.
    #[arg(long, default_value = "1000")]
    summarizer_budget: u64,

    /// Use upstream headers as-is (skip cache-control, x-accel-buffering, charset).
    /// For isolating whether header modification causes the hang.
    #[arg(long)]
    no_header_mod: bool,

    /// Enable DAG context injection into system messages. Off by default —
    /// LCM context changes model reasoning trajectory and breaks tool-call
    /// agents. Only enable for agents that explicitly understand LCM format.
    #[arg(long)]
    lcm_context: bool,

    /// Token budget for LCM context injection. Set >0 to enable (e.g. 512).
    /// Default 0 (off). Override per-request via `lcm_max_tokens` in body.
    #[arg(long, default_value = "1024")]
    lcm_context_tokens: u64,

    /// Disable LCM context injection entirely.
    #[arg(long)]
    no_lcm_context: bool,

    /// Disable system prompt cache normalization. By default, timestamps and
    /// UUIDs in system prompts are replaced with stable markers to preserve
    /// DeepSeek prefix cache hit rate. Use this flag to disable.
    #[arg(long)]
    no_cache_normalize: bool,


    /// Workspace root path for stable conversation identity.
    /// Auto-detected via `git rev-parse --show-toplevel` if not set.
    #[arg(long, env = "DEEPLOSSLESS_WORKSPACE")]
    workspace: Option<String>,

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

    /// Run built-in torture protocol compatibility test with deeplossless augmentation.
    #[arg(long)]
    torture: bool,

    /// Run protocol compatibility test scenarios.
    /// Starts a mock API that drives agent behavior via a state machine.
    /// Usage: --torture-aces hidden_bug
    /// Empty value (`--torture-aces=""`) or "all" runs the full suite:
    /// every base scenario in scenarios/ in sorted order, sequenced by
    /// idle-after-terminal detection.
    #[arg(long)]
    torture_aces: Option<String>,

    /// Default reasoning effort for DeepSeek-V4 (auto, high, max, none).
    /// Override per-request via reasoning_effort field.
    #[arg(long, default_value = "auto")]
    reasoning_effort: String,

    /// Parse DSML tool calls from DeepSeek-V4 response text.
    #[arg(long, default_value = "true")]
    dsml_parse: bool,

    /// Emit DSML tool calls to upstream (debug only, off by default).
    #[arg(long)]
    dsml_emit: bool,

    /// Quick instruction mode — optimizes system prompt for speed.
    #[arg(long)]
    quick_instruction: bool,

    /// Agent format used to select per-agent tool arg templates
    /// from `args_per_agent[format]` in the scenario YAML. Default is
    /// `openai` (camelCase field names: filePath, oldString, newString)
    /// which matches the Chat Completions protocol the mock speaks.
    /// Use `claude_code` (snake_case: file_path, old_string, new_string)
    /// when driving the mock from Claude Code, which translates the
    /// Anthropic protocol to Chat Completions.
    /// Use `codex` for the OpenAI Codex CLI / Responses API clients
    /// (note: Codex uses `apply_patch` for file edits, not `Edit`, so
    /// scenario YAMLs will need a separate `args_per_agent.codex`
    /// block with the apply_patch schema).
    #[arg(long, default_value = "openai")]
    agent_format: String,
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
    /// Drive protocol compatibility scenarios without an LLM — a diagnostic tool for
    /// verifying YAML state machines, pre_apply edits, and agent-format
    /// parameter mapping. No API key or LLM required.
    Drive {
        /// Scenario name or "all" for suite
        #[arg(default_value = "all")]
        scenario: String,
        /// Agent format: openai, claude_code, codex
        #[arg(long, default_value = "claude_code")]
        format: String,
    },
    /// Search proxy events with structured filters + FTS.
    /// Query the event index (proxy_events table) built by
    /// deeplossless during normal proxy operation.
    Search {
        /// Filter by event type: request_start, user_message, tool_call, etc.
        #[arg(long)]
        event_type: Option<String>,
        /// Filter by tool name: Read, Edit, Grep, Bash, etc.
        #[arg(long)]
        tool: Option<String>,
        /// Filter by session fingerprint or prompt_cache_key.
        #[arg(long)]
        session: Option<String>,
        /// Filter by status: success, error
        #[arg(long)]
        status: Option<String>,
        /// Filter by file path (SQL LIKE pattern: %config.rs)
        #[arg(long)]
        path: Option<String>,
        /// Full-text search on content column (FTS5)
        #[arg(long)]
        content: Option<String>,
        /// Max results, default 50
        #[arg(long, short = 'n', default_value = "50")]
        limit: usize,
    },
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
    println!("  Protocol compatibility tests (no API key needed):");
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
    let cert_dir = shellexpand::tilde("~/.deeplossless").to_string();
    let cert_path = format!("{cert_dir}/cert.pem");
    if !std::path::Path::new(&cert_path).exists() {
        anyhow::bail!("No cert at {cert_path}. Run deeplossless first to generate one.");
    }
    println!("Certificate: {cert_path}");
    println!();
    println!("Set the SSL_CERT_FILE environment variable so OpenSSL-based tools");
    println!("(including Codex/OpenCode) trust this certificate:");
    println!();
    #[cfg(target_os = "windows")]
    {
        println!("  setx SSL_CERT_FILE {}\\\\.deeplossless\\\\cert.pem", std::env::var("USERPROFILE").unwrap_or_default());
        println!();
        println!("Then restart your terminal or close+reopen VS Code.");
    }
    #[cfg(not(target_os = "windows"))]
    {
        println!("  export SSL_CERT_FILE={cert_path}");
        println!();
        println!("Add the line above to ~/.bashrc, ~/.zshrc, or equivalent.");
        println!("Then restart your terminal or run: source ~/.bashrc");
    }
    Ok(())
}

fn run_drive(scenario: &str, format: &str) -> anyhow::Result<()> {
    use std::path::PathBuf;

    let parent = PathBuf::from("/tmp/aces_drive");

    let verbose = true;

    if scenario == "all" || scenario.is_empty() {
        let outcomes = deeplossless::torture::driver::drive_suite(format, &parent, verbose);
        let mut ok = 0;
        let mut fail = 0;
        for outcome in &outcomes {
            match outcome {
                Ok(o) if o.success => ok += 1,
                Ok(o) => {
                    fail += 1;
                    eprintln!("  FAIL: {} → terminal={}", o.scenario, o.terminal_state);
                }
                Err(e) => {
                    fail += 1;
                    eprintln!("  FAIL: {e}");
                }
            }
        }
        eprintln!("\n── suite: {ok} OK, {fail} FAIL ──");
        if fail > 0 {
            anyhow::bail!("{fail} scenario(s) failed");
        }
    } else {
        let outcome = deeplossless::torture::driver::drive_scenario(scenario, format, &parent, verbose)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if !outcome.success {
            anyhow::bail!("scenario '{}' failed at state '{}'", scenario, outcome.terminal_state);
        }
    }

    Ok(())
}

async fn run_search(db_path: &str, filter: deeplossless::event_store::EventFilter) -> anyhow::Result<()> {
    let db = deeplossless::db::Database::builder()
        .path(db_path)
        .build()
        .await?;

    let events = db.query_proxy_events(&filter)?;

    if events.is_empty() {
        println!("no results");
        return Ok(());
    }

    for ev in &events {
        let type_str = ev.event_type.as_str();
        let tool_str = ev.tool_name.as_deref().unwrap_or("-");
        let path_str = ev.path.as_deref().unwrap_or("-");
        let status_str = ev.status.as_deref().unwrap_or("-");
        let session_part = if ev.session_id.len() > 16 {
            format!("{}…", &ev.session_id[..15])
        } else {
            ev.session_id.clone()
        };
        let content_preview = if ev.content.len() > 100 {
            format!("{}…", &ev.content[..99])
        } else {
            ev.content.clone()
        };
        println!(
            "{:<18} {:<12} {:<24} {:<8} {:<18} {}",
            type_str, tool_str, session_part, status_str, path_str, content_preview
        );
    }

    println!("\n── {} event(s) ──", events.len());
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
    if let Some(Commands::Drive { scenario, format }) = cli.command {
        return run_drive(&scenario, &format);
    }
    if let Some(Commands::Search { event_type, tool, session, status, path, content, limit }) = cli.command {
        use deeplossless::event_store::{EventFilter, EventType};
        let filter = EventFilter {
            event_type: event_type.as_deref().and_then(EventType::from_event_str),
            tool_name: tool,
            session_id: session,
            status,
            path_pattern: path.map(|p| format!("%{p}%")),
            content_match: content,
            limit: Some(limit),
        };
        return run_search(&cli.db_path, filter).await;
    }
    let mut cli = cli;

    if let Some(ref torture_aces) = cli.torture_aces {
        // Suite mode runs every base scenario; single mode runs the
        // one named scenario. The mock handles per-agent variant
        // selection internally via Scenario::load_with_format.
        let scenarios: Vec<String> = if torture_aces == "all" || torture_aces.is_empty() {
            match deeplossless::torture::scenario::Scenario::list_base() {
                Ok(names) if !names.is_empty() => names,
                Ok(_) => {
                    eprintln!("[aces] no base scenarios found in scenarios/");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("[aces] failed to list scenarios: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            vec![torture_aces.clone()]
        };

        eprintln!("[aces] starting ACES mock upstream...");
        eprintln!("[aces] suite: {} scenario(s)", scenarios.len());
        for (i, s) in scenarios.iter().enumerate() {
            eprintln!("[aces]   [{}/{}] {}", i + 1, scenarios.len(), s);
        }
        eprintln!("[aces] agent format: {}", cli.agent_format);

        let mock_state = deeplossless::torture::aces::start_mock(&scenarios, &cli.agent_format).await;
        cli.upstream = "http://127.0.0.1:9000".into();
        eprintln!("[aces] proxy upstream set to mock on port 9000");

        // For multi-scenario suites, wait for the entire suite to
        // complete (idleness-driven), then print a summary and exit.
        // For single-scenario runs, leave the mock running and let
        // the existing axum serve() loop handle requests.
        if scenarios.len() > 1 {
            // Print prompts for the operator: which prompt to feed
            // to the agent for each scenario. The agent stays
            // connected to the same proxy/mock; the mock cycles
            // through scenarios internally.
            eprintln!();
            eprintln!("[aces] ── Run order ────────────────────────────────────");
            for (i, s) in scenarios.iter().enumerate() {
                eprintln!("[aces]   [{}/{}] {} — run agent with this scenario's prompt",
                    i + 1, scenarios.len(), s);
            }
            eprintln!();
            eprintln!("[aces] waiting for suite to complete (5s idle per scenario)...");

            // Poll the suite_complete flag. The mock's background
            // idle-watcher sets it after the last scenario's idle
            // threshold fires. The user's main loop just waits.
            let poll_state = mock_state.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if poll_state.lock().unwrap().suite_complete {
                        let s = poll_state.lock().unwrap();
                        eprintln!();
                        eprintln!("═══════════════════════════════════════════");
                        eprintln!("  ACES Suite Complete");
                        eprintln!("═══════════════════════════════════════════");
                        for (i, (name, _)) in s.scenarios.iter().enumerate() {
                            eprintln!("  [{}/{}] {}", i + 1, s.scenarios.len(), name);
                        }
                        eprintln!("═══════════════════════════════════════════");
                        std::process::exit(0);
                    }
                }
            });
        }
    }

    let torture_workspace: Option<std::path::PathBuf> = if cli.torture {
        eprintln!("[torture] starting mock upstream...");
        if let Err(e) = torture_start_mock().await {
            eprintln!("[torture] mock server failed: {e}");
        }
        let ws = std::env::current_dir().ok();
        cli.upstream = "http://127.0.0.1:9000".into();
        eprintln!("[torture] proxy upstream set to mock on port 9000");
        eprintln!("[torture] connect your agent via HTTP to http://127.0.0.1:8081/v1/chat/completions");
        eprintln!("[torture] or via HTTPS to https://127.0.0.1:8080/v1/chat/completions");
        ws
    } else {
        None
    };

    let is_aces = cli.torture_aces.is_some();
    let is_torture = cli.torture;
    let mode_str = "full (runtime enabled)".to_string();

    if is_aces || is_torture {
        eprintln!();
        if is_aces {
        eprintln!("┌────────────────────────────────────────────────┐");
        eprintln!("│ Torture — Protocol Compatibility Test          │");
        eprintln!("├────────────────────────────────────────────────┤");
        let scenario_label = match cli.torture_aces.as_deref() {
            Some("") | Some("all") => "(suite)",
            Some(name) => name,
            None => "(suite)",
        };
        eprintln!("│ Scenario: {:<37}", scenario_label);
        eprintln!("│ Mode:     {mode_str}");
        eprintln!("│                                                │");
        eprintln!("│ Point your agent to:                           │");
        eprintln!("│ http://127.0.0.1:8081/v1/chat/completions      │");
        eprintln!("│ or HTTPS at 127.0.0.1:8080                     │");
        eprintln!("│                                                │");
        eprintln!("│ Scenario guides the interaction. Scores +      │");
        eprintln!("│ runtime metrics saved on exit.                 │");
        eprintln!("└────────────────────────────────────────────────┘");
    } else {
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("  Torture Protocol Test Started");
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("  mock upstream:  http://127.0.0.1:9000/v1/chat/completions");
        eprintln!("  proxy HTTP:     http://127.0.0.1:8081/v1/chat/completions");
        eprintln!("  proxy HTTPS:    https://127.0.0.1:8080/v1/chat/completions");
        eprintln!("  mode:           {mode_str}");
        eprintln!();
        if let Some(ref ws) = torture_workspace {
            eprintln!("  workspace:   {}", ws.display());
        }
        eprintln!("  HOW TO TEST");
        eprintln!("  1. Open an empty directory in your agent");
        eprintln!("  2. Point your agent to proxy:");
        eprintln!("     http://127.0.0.1:8081 (HTTP) or https://127.0.0.1:8080 (HTTPS)");
        eprintln!("  3. Work naturally — traces will generate files and exercise tools");
        eprintln!("  4. After all turns complete, test finishes and report is saved");
        eprintln!("═══════════════════════════════════════════════════════════════");
        }
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
    // Resolve workspace: explicit flag → git root → None
    let workspace = cli.workspace.clone().or_else(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    });
    if let Some(ref ws) = workspace {
        tracing::info!(workspace = %ws, "workspace identity");
    } else {
        tracing::warn!("no workspace set — use --workspace or run from a git repo for stable conversation identity");
    }

    let cfg = deeplossless::runtime_coordinator::CoordinatorConfig {
        dag_threshold: cli.dag_threshold,
        summarizer_budget: cli.summarizer_budget,
        upstream: cli.upstream,
        upstream_path: cli.upstream_path,
        db_path: cli.db_path,
        api_key: cli.api_key,
        admin_key: cli.admin_key,
        summarizer_model: cli.summarizer_model,
        rate_limit: cli.rate_limit,
        runtime_profile: cli.runtime_profile,
        dry_run: cli.dry_run,
        log_dir: cli.log_dir,
        record: cli.record,
        passthrough: cli.passthrough,
        // ACES drives its own scenario state machine — the deeplossless
        // pipeline (storage, compactor, summarizer) is pure noise here
        // and would call the mock's /v1/chat/completions from the
        // summarizer, producing the "expected value at line 1 column 1"
        // parse errors. Force the pipeline off in ACES mode.
        no_pipeline: cli.no_pipeline || cli.torture_aces.is_some(),
        no_header_mod: cli.no_header_mod,
        lcm_context: cli.lcm_context,
        cache_normalize: !cli.no_cache_normalize,
        lcm_context_tokens: if cli.no_lcm_context { 0 } else { cli.lcm_context_tokens },
        workspace,
        reasoning_effort: match cli.reasoning_effort.as_str() {
            "high" | "High" => deeplossless::protocol::ReasoningEffortMode::Override(deeplossless::protocol::ReasoningEffort::High),
            "max" | "Max" => deeplossless::protocol::ReasoningEffortMode::Override(deeplossless::protocol::ReasoningEffort::Max),
            "none" | "None" => deeplossless::protocol::ReasoningEffortMode::Override(deeplossless::protocol::ReasoningEffort::None),
            _ => deeplossless::protocol::ReasoningEffortMode::Passthrough,
        },
        dsml_parse: cli.dsml_parse,
        dsml_emit: cli.dsml_emit,
        quick_instruction: cli.quick_instruction,
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
    if coordinator.state.record.is_some() {
        tracing::info!(target: "deeplossless::record", "record flag enabled at startup");
    }

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
        && std::env::var("SSL_CERT_FILE").is_err() {
        tracing::info!("Run `deeplossless trust` once to configure HTTPS certificate trust (sets SSL_CERT_FILE).");
    }

    // Plain HTTP for sandboxed agents that can't trust self-signed certs (OpenClaw, etc.)
    if cli.http_port > 0 && cli.http_port != cli.port {
        let http_addr: std::net::SocketAddr = format!("127.0.0.1:{}", cli.http_port).parse()?;
        let http_app = app.clone();
        tracing::info!("HTTP on {http_addr} (for sandboxed local agents)");
        let tls_handle = axum_server::Handle::new();
        let http_handle = axum_server::Handle::new();
        let tls_h = tls_handle.clone();
        let http_h = http_handle.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            tls_h.shutdown();
            http_h.shutdown();
        });
        // HTTP: spawn in background, don't crash if port is busy
        tokio::spawn(async move {
            if let Err(e) = axum_server::bind(http_addr)
                .handle(http_handle)
                .serve(http_app.into_make_service())
                .await
            {
                tracing::warn!("HTTP server stopped: {e}");
            }
        });
        axum_server::bind_rustls(addr, tls_config)
            .handle(tls_handle)
            .serve(app.into_make_service())
            .await?;
    } else {
        let handle = axum_server::Handle::new();
        let h = handle.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            h.shutdown();
        });
        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await?;
    }
    coordinator.shutdown(std::time::Duration::from_secs(2)).await;

    tracing::info!("deeplossless stopped");

    Ok(())
}

async fn torture_start_mock() -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;

    let combined = deeplossless::torture::adversarial::combined_trace();
    let total_turns = combined.turns.len();
    let tool_calls_total: usize = combined.turns.iter().map(|t| t.tool_calls.len()).sum();
    let mode = "full";
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let report_path = format!("logs/torture-report-{}-{}.json", mode, ts);

    eprintln!("[torture] {} mode, {} turns", mode, total_turns);

    let state = std::sync::Arc::new(MockTortureState {
        turns: combined.turns,
        cursor: AtomicUsize::new(0),
        report_saved: AtomicUsize::new(0),
        started_at: std::time::Instant::now(),
    });
    let s = state.clone();
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(move |_body: axum::Json<serde_json::Value>| {
            let s = s.clone();
            async move {
                let idx = s.cursor.fetch_add(1, Ordering::Relaxed);

                if idx == total_turns {
                    s.cursor.store(total_turns, Ordering::Relaxed);
                    if s.report_saved.fetch_add(1, Ordering::Relaxed) == 0 {
                        eprintln!();
                        eprintln!("═══════════════════════════════════════════");
                        eprintln!("  Torture Protocol Test Complete");
                        eprintln!("═══════════════════════════════════════════");
                        eprintln!("  mode:        {}", mode);
                        eprintln!("  turns:       {}/{}", total_turns, total_turns);
                        eprintln!("  report:      {}", report_path);
                        eprintln!();

                        let elapsed = s.started_at.elapsed().as_secs_f64();
                        let runtime_metrics = match reqwest::Client::new()
                            .get("http://127.0.0.1:8081/v1/lcm/runtime/stats")
                            .header("Authorization", "Bearer test-key")
                            .send()
                            .await
                        {
                            Ok(r) => r.json::<serde_json::Value>().await.unwrap_or_default(),
                            Err(_) => serde_json::Value::Null,
                        };

                        let report = serde_json::json!({
                            "test": "deeplossless torture suite",
                            "version": "0.6.7",
                            "total_turns": total_turns,
                            "turns_served": total_turns,
                            "tool_calls_in_trace": tool_calls_total,
                            "elapsed_secs": elapsed,
                            "turns_per_sec": (total_turns as f64 / elapsed).round(),
                            "mode": mode,
                            "runtime": runtime_metrics,
                            "timestamp": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
                        });
                        let json_str = serde_json::to_string_pretty(&report).unwrap_or_default();
                        let _ = std::fs::create_dir_all("logs");
                        let _ = std::fs::write(&report_path, &json_str);
                    }
                }

                let turn_idx = idx.min(total_turns - 1);
                let turn = &s.turns[turn_idx];

                if idx > 0 && idx % 10 == 0 && idx < total_turns {
                    let pct = (idx as f64 / total_turns as f64 * 100.0).round();
                    eprintln!("[torture] {}/{} ({:.0}%)", idx, total_turns, pct);
                }

                let remaining = total_turns.saturating_sub(idx);

                let (delta_json, finish_reason) = if idx >= total_turns {
                    let msg = format!("[Protocol Compatibility Test Complete] All {} turns executed.", total_turns);
                    (format!("\"content\":{}", serde_json::to_string(&msg).unwrap_or_default()), "\"stop\"")
                } else if turn.tool_calls.is_empty() {
                    (format!("\"content\":{}", serde_json::to_string(&turn.completion).unwrap_or_default()), "\"stop\"")
                } else {
                    let tc_parts: Vec<String> = turn.tool_calls.iter().enumerate().map(|(ti, name)| {
                        let args_value: serde_json::Value = match name.as_str() {
                            "grep" | "rg" | "search" => serde_json::json!({"pattern":"fn foo","path":"src/lib.rs"}),
                            "read" | "cat" => serde_json::json!({"filePath":"src/lib.rs"}),
                            "edit" | "replace" => serde_json::json!({"filePath":"src/lib.rs","oldString":"fn foo() {}","newString":"fn foo() -> i32 { 42 }"}),
                            "bash" => {
                                let cmds = [
                                    "mkdir -p src && echo 'fn foo() {}' > src/lib.rs",
                                    "mkdir -p config && printf 'port=8080\\nhost=localhost\\n' > config/dev.conf",
                                    "echo 'Hello World' > README.md",
                                    "mkdir -p tests && echo 'mod tests;' > tests/mod.rs",
                                    "echo 'version = \"0.1.0\"' > Cargo.toml",
                                    "mkdir -p src && echo 'pub fn bar() -> u32 { 7 }' > src/bar.rs",
                                    "echo 'DEBUG=true' > .env",
                                    "printf 'fn main() {\\n    println!(\"hi\");\\n}\\n' > src/main.rs",
                                    "mkdir -p data && echo '{\"key\": 1}' > data/config.json",
                                    "echo 'use std::collections::HashMap;' > src/lib.rs",
                                ];
                                let cmd_idx = (idx + ti) % cmds.len();
                                serde_json::json!({"command": cmds[cmd_idx], "description": cmds[cmd_idx].split(' ').take(4).collect::<Vec<_>>().join(" ")})
                            },
                            "glob" | "find" | "ls" => serde_json::json!({"pattern":"src/**/*.rs"}),
                            _ => serde_json::json!({"key":"value"}),
                        };
                        let args_str = serde_json::to_string(&args_value).unwrap_or_default();
                        format!("{{\"index\":{ti},\"id\":\"call_{ti}\",\"function\":{{\"name\":{},\"arguments\":{}}},\"type\":\"function\"}}",
                            serde_json::to_string(name).unwrap_or_default(),
                            serde_json::to_string(&args_str).unwrap_or_default())
                    }).collect();
                    (format!("\"tool_calls\":[{}]", tc_parts.join(",")), "\"tool_calls\"")
                };

                let events = format!(
                    "data: {{\"choices\":[{{\"delta\":{{{}}},\"index\":0,\"finish_reason\":{}}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":{},\"total_tokens\":{},\"remaining_turns\":{}}}}}\n\ndata: [DONE]\n\n",
                    delta_json, finish_reason, turn.tokens, turn.tokens, remaining,
                );
                ([(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], events)
            }
        }));

    tokio::spawn(async move {
        match tokio::net::TcpListener::bind("127.0.0.1:9000").await {
            Ok(listener) => { axum::serve(listener, app).await.ok(); }
            Err(e) => eprintln!("[torture] failed to start mock server: {e}"),
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    Ok(())
}

struct MockTortureState {
    turns: Vec<deeplossless::torture::trace::Turn>,
    cursor: AtomicUsize,
    report_saved: AtomicUsize,
    started_at: std::time::Instant,
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

    #[test]
    fn torture_flag_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--torture"]).unwrap();
        assert!(args.torture);
    }

    #[test]
    fn torture_aces_all_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--torture-aces", "all"]).unwrap();
        assert_eq!(args.torture_aces.as_deref(), Some("all"));
    }

    #[test]
    fn torture_aces_empty_value_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--torture-aces="]).unwrap();
        assert_eq!(args.torture_aces.as_deref(), Some(""));
    }

    #[test]
    fn ds4_reasoning_effort_default_is_auto() {
        let args = Cli::try_parse_from(["deeplossless"]).unwrap();
        assert_eq!(args.reasoning_effort, "auto");
    }

    #[test]
    fn ds4_reasoning_effort_high_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--reasoning-effort", "high"]).unwrap();
        assert_eq!(args.reasoning_effort, "high");
    }

    #[test]
    fn ds4_reasoning_effort_max_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--reasoning-effort", "max"]).unwrap();
        assert_eq!(args.reasoning_effort, "max");
    }

    #[test]
    fn ds4_reasoning_effort_none_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--reasoning-effort", "none"]).unwrap();
        assert_eq!(args.reasoning_effort, "none");
    }

    #[test]
    fn ds4_dsml_parse_default_true() {
        let args = Cli::try_parse_from(["deeplossless"]).unwrap();
        assert!(args.dsml_parse);
    }

    #[test]
    fn ds4_dsml_emit_default_false() {
        let args = Cli::try_parse_from(["deeplossless"]).unwrap();
        assert!(!args.dsml_emit);
    }

    #[test]
    fn ds4_dsml_emit_true_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--dsml-emit"]).unwrap();
        assert!(args.dsml_emit);
    }

    #[test]
    fn ds4_quick_instruction_default_false() {
        let args = Cli::try_parse_from(["deeplossless"]).unwrap();
        assert!(!args.quick_instruction);
    }

    #[test]
    fn ds4_quick_instruction_true_accepted() {
        let args = Cli::try_parse_from(["deeplossless", "--quick-instruction"]).unwrap();
        assert!(args.quick_instruction);
    }
}
