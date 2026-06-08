use crate::torture::scenario::{Scenario, StateMachine, AgentEvent, extract_events_from_request};
use serde_json::Value;
use std::sync::Arc;
use std::sync::Mutex;

/// Start the ACES mock server (runs on port 9000 in background).
/// Returns a handle to the shared state so callers (e.g., the suite
/// runner) can poll `scenario_complete` and `suite_complete` to know
/// when the agent has finished the current scenario / the entire suite.
///
/// `scenario_names` may be a single name (single-scenario mode) or
/// multiple names (suite mode). Per-agent variant selection happens
/// inside [`Scenario::load_with_format`] so the same call works for
/// both.
pub async fn start_mock(scenario_names: &[String], agent_format: &str) -> std::sync::Arc<std::sync::Mutex<SharedState>> {
    if scenario_names.is_empty() {
        eprintln!("[aces] no scenarios provided");
        std::process::exit(1);
    }
    let mut scenarios: Vec<(String, Scenario)> = Vec::with_capacity(scenario_names.len());
    for name in scenario_names {
        match Scenario::load_with_format(name, Some(agent_format)) {
            Ok(s) => scenarios.push((name.clone(), s)),
            Err(e) => {
                eprintln!("[aces] failed to load scenario '{name}': {e}");
                std::process::exit(1);
            }
        }
    }
    let n = scenarios.len();
    let initial_scenario = scenarios[0].clone();
    let initial_machine = StateMachine::new(initial_scenario.1.clone());
    let agent_format = agent_format.to_string();

    // VFS is created lazily on the first request, in the agent's working
    // directory (extracted from the request's system prompt). This way the
    // mock's own cwd is irrelevant — files always land where the agent is.
    let state = Arc::new(Mutex::new(SharedState {
        scenarios,
        current_idx: 0,
        machine: initial_machine,
        reported: false,
        plugin_warned: false,
        terminal_sent: false,
        vfs_root: None,
        agent_format: agent_format.clone(),
        terminal_at: None,
        last_request_at: None,
        scenario_complete: false,
        suite_complete: n == 1, // single-scenario: "suite" of one is done once scenario_complete fires
    }));

    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post({
            let state = state.clone();
            move |body: axum::Json<Value>| {
                let state = state.clone();
                async move { handle_request(body.0, state).await }
            }
        }));

    tokio::spawn(async move {
        match tokio::net::TcpListener::bind("127.0.0.1:9000").await {
            Ok(listener) => {
                eprintln!("[aces] mock upstream listening on port 9000");
                eprintln!("[aces] agent format: {agent_format}");
                eprintln!("[aces] vfs will be created in the agent's working directory on first request");
                axum::serve(listener, app).await.ok();
            }
            Err(e) => eprintln!("[aces] failed to start mock: {e}"),
        }
    });

    // Background idle-watcher: once the terminal state has been
    // reached, wait for `IDLE_THRESHOLD` with no further agent
    // requests, then mark `scenario_complete = true` so the suite
    // runner can advance. This replaces wall-clock timeouts —
    // termination is driven by the agent's own behavior, so a slow
    // agent and a fast agent both finish cleanly.
    //
    // For multi-scenario suites, the watcher also advances the
    // current_idx, resets VFS, and prepares the next scenario.
    // When the last scenario completes, `suite_complete` is set
    // so the framework's main loop can exit.
    let watcher_state = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let advance_info = {
                let mut s = watcher_state.lock().unwrap();
                if s.suite_complete { return; }
                if let (Some(_terminal_at), Some(last_req)) = (s.terminal_at, s.last_request_at) {
                    // We only need terminal_at to be set + idle duration
                    // to have elapsed. last_request_at is updated at the
                    // very start of handle_request (before terminal_at is
                    // potentially set inside the same call), so
                    // `last_req < terminal_at` is normal — they're set
                    // in the same call but at different lines. Comparing
                    // them is meaningless; the idle threshold alone is
                    // what proves the agent has stopped reacting.
                    if !s.scenario_complete
                        && last_req.elapsed() >= std::time::Duration::from_secs(IDLE_THRESHOLD_SECS)
                    {
                        s.scenario_complete = true;
                        eprintln!("[aces] idle {}s after terminal — scenario complete", IDLE_THRESHOLD_SECS);

                        let status = if s.machine.is_success() { "ok" } else { "give_up" };
                        eprintln!("[aces]   → {}", status);

                        let next_idx = s.current_idx + 1;
                        if next_idx >= s.scenarios.len() {
                            s.suite_complete = true;
                            eprintln!("[aces] suite complete: {}/{} scenarios finished", next_idx, s.scenarios.len());
                            return;
                        }

                        // Advance to next scenario. Reset everything
                        // that was scenario-local: state machine,
                        // VFS root, idle tracking, terminal flag.
                        eprintln!("[aces] advancing to scenario [{}/{}]: {}",
                            next_idx + 1, s.scenarios.len(), s.scenarios[next_idx].0);
                        s.current_idx = next_idx;
                        let (name, scenario) = s.scenarios[next_idx].clone();
                        s.machine = StateMachine::new(scenario);
                        s.vfs_root = None;
                        s.terminal_sent = false;
                        s.reported = false;
                        s.plugin_warned = false;
                        s.terminal_at = None;
                        s.scenario_complete = false;
                        // last_request_at is left alone — the agent
                        // may have just sent a request before the
                        // threshold fired, and we want the next
                        // scenario's idle countdown to start clean
                        // when its terminal fires.
                        let _ = name; // currently logged above
                    }
                }
                if s.suite_complete { return; }
            };
            let _ = advance_info;
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    state
}

/// Seconds of no agent requests after the terminal state before the
/// mock declares the scenario "complete" and the suite runner may
/// advance. 5s is enough for the agent to receive the verify
/// message and either stop or start a post-terminal poll loop,
/// without dragging out the suite if the agent is already gone.
const IDLE_THRESHOLD_SECS: u64 = 5;

pub struct SharedState {
    /// Full suite: every scenario to run, in order. Length 1 for a
    /// single-scenario invocation; N for `--torture-aces all` mode.
    pub scenarios: Vec<(String, Scenario)>,
    /// Index of the currently-serving scenario in `scenarios`. Advances
    /// after each scenario's idle threshold fires.
    pub current_idx: usize,
    /// The active state machine for the current scenario. Replaced
    /// (in place) when advancing to the next scenario.
    pub machine: StateMachine,
    pub reported: bool,
    /// VFS root path, set lazily on the first request based on the agent's
    /// working directory (parsed from the request's system prompt).
    /// Recreated when advancing to a new scenario so each scenario's
    /// files start fresh.
    pub vfs_root: Option<std::path::PathBuf>,
    /// Agent format used to select per-agent tool arg templates. Default
    /// is "claude_code" (snake_case). Other values trigger lookup in
    /// `state.args_per_agent[format]`.
    pub agent_format: String,
    /// Whether we've already printed the plugin-interference warning for
    /// this scenario run (avoid spamming on every request).
    pub plugin_warned: bool,
    /// Whether we've already sent the "Task complete" terminal message.
    /// First terminal hit sends the full message; later hits send only
    /// `data: [DONE]\n\n` so the post-terminal agent polling loop
    /// doesn't spam the terminal text into the log.
    pub terminal_sent: bool,
    /// Time the terminal state was first reached for the current
    /// scenario. The mock is considered "done" once this is set AND
    /// no further requests arrive for the idle threshold. Reset
    /// when advancing to the next scenario.
    pub terminal_at: Option<std::time::Instant>,
    /// Time of the most recent request from the agent. Updated inside
    /// [`handle_request`]. Used together with `terminal_at` to detect
    /// "agent has finished reacting to the terminal message".
    pub last_request_at: Option<std::time::Instant>,
    /// Set to `true` by the background idle-watcher task once the
    /// terminal state has been reached and the idle threshold has
    /// elapsed with no further requests. Triggers a suite advance.
    pub scenario_complete: bool,
    /// Set to `true` once every scenario in the suite has completed
    /// (either reached terminal or hit the stuck-loop guard). The
    /// main loop polls this to know when the entire suite is done
    /// and the process can exit cleanly.
    pub suite_complete: bool,
}

pub(crate) fn create_vfs(scenario: &Scenario, base: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let _ = std::fs::remove_dir_all(base);
    std::fs::create_dir_all(base)?;
    for (rel, content) in &scenario.vfs.files {
        let full = base.join(rel);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, content)?;
    }
    Ok(base.to_path_buf())
}

/// Extract the agent's working directory from the request's system prompt.
/// Looks for "Working directory: /path" (Claude Code convention) and returns
/// the path. Returns None if no path is found.
pub(crate) fn extract_agent_cwd(body: &Value) -> Option<std::path::PathBuf> {
    let msgs = body["messages"].as_array()?;
    let sys = msgs.first()?;
    let content = match sys.get("content") {
        Some(c) => c,
        None => return None,
    };
    let text = match content.as_str() {
        Some(s) => s,
        None => {
            // content can be an array of blocks (Anthropic format)
            if let Some(arr) = content.as_array() {
                let mut buf = String::new();
                for b in arr {
                    if let Some(t) = b["text"].as_str() {
                        buf.push_str(t);
                        buf.push('\n');
                    }
                }
                Box::leak(buf.into_boxed_str()) as &str
            } else {
                return None;
            }
        }
    };
    for line in text.lines() {
        let lower = line.to_lowercase();
        let trimmed = line.trim();
        // Match "Working directory: /path" or "Current working directory: /path"
        if lower.starts_with("working directory:")
            || lower.starts_with("current working directory:")
            || lower.starts_with("primary working directory:")
        {
            if let Some((_, path)) = line.split_once(':') {
                let p = path.trim();
                if !p.is_empty() {
                    return Some(std::path::PathBuf::from(p));
                }
            }
        }
        // Fallback: bare absolute path on a labeled line
        if lower.contains("working directory") && trimmed.contains('/') {
            if let Some((_, path)) = line.split_once(':') {
                let p = path.trim();
                if !p.is_empty() && p.starts_with('/') {
                    return Some(std::path::PathBuf::from(p));
                }
            }
        }
    }
    None
}

pub(crate) fn resolve_vfs_path(template: &str, vfs_root: &std::path::Path) -> String {
    template.replace("${VFS}", &vfs_root.display().to_string())
}

/// Resolve `${VFS}` placeholders inside a tool args value. Used by
/// `build_sse` to render the final `arguments` string the LLM receives.
/// This is a pure value-level transform — agent-format selection happens
/// in `pick_args_for_format` (called by `build_sse`).
pub(crate) fn resolve_args(value: &Value, vfs_root: &std::path::Path) -> String {
    let resolved = resolve_recursive(value, vfs_root);
    serde_json::to_string(&resolved).unwrap_or_default()
}

/// Recursively walk `value` and replace `${VFS}` in any string leaf.
/// Top-level only handled non-string children as opaque clones, which
/// broke Codex's `apply_patch` whose `operation.path` is two levels
/// deep: { apply_patch: { operation: { path: "${VFS}/..." } } }.
fn resolve_recursive(value: &Value, vfs_root: &std::path::Path) -> Value {
    match value {
        Value::String(s) => Value::String(resolve_vfs_path(s, vfs_root)),
        Value::Object(map) => {
            let mut new_map = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                new_map.insert(k.clone(), resolve_recursive(v, vfs_root));
            }
            Value::Object(new_map)
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| resolve_recursive(v, vfs_root)).collect())
        }
        other => other.clone(),
    }
}

/// Handle a single Chat Completions request.
async fn handle_request(
    body: Value,
    state: Arc<Mutex<SharedState>>,
) -> ([(axum::http::HeaderName, &'static str); 1], String) {
    let mut s = state.lock().unwrap();

    // Stamp the request time so the idle-watcher can tell when the
    // agent has stopped reacting after the terminal state.
    s.last_request_at = Some(std::time::Instant::now());

    // Lazily resolve VFS root on first request. We try the agent's working
    // directory (from the system prompt) first, then fall back to the mock's
    // own cwd. Either way, files end up somewhere the agent can read them.
    if s.vfs_root.is_none() {
        let cwd = extract_agent_cwd(&body)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let base = cwd.join(format!("aces_vfs_{}", s.machine.scenario_name()));
        match create_vfs(s.machine.scenario(), &base) {
            Ok(p) => {
                eprintln!("[aces] vfs root: {}", p.display());
                s.vfs_root = Some(p);
            }
            Err(e) => {
                eprintln!("[aces] failed to create vfs at {}: {e}", base.display());
            }
        }
    }

    if !s.plugin_warned {
        if let Some(signal) = detect_plugin_interference(&body) {
            s.plugin_warned = true;
            eprintln!(
                "[aces] WARNING: {signal}\n\
                 [aces]   Plugins like claude-mem inject system reminders and spawn\n\
                 [aces]   sub-sessions with disabled tools. This corrupts tool results\n\
                 [aces]   and makes ACES scenarios unrunnable.\n\
                 [aces]   To fix: edit ~/.claude/settings.json and remove the\n\
                 [aces]   \"enabledPlugins\" entries (or set them to false), then\n\
                 [aces]   restart `claude` and this mock."
            );
        }
    }

    // Short-circuit: if the scenario already reported as terminal and we
    // sent the closing message, every subsequent request from the agent's
    // post-terminal polling loop returns just [DONE]. Skipping feed()
    // also suppresses the "no events, staying at X" log line that
    // would otherwise spam for each redundant request.
    if s.terminal_sent {
        return (content_type(), "data: [DONE]\n\n".to_string());
    }

    let vfs_root_for_feed = s.vfs_root.clone();
    let (is_terminal, advanced, had_events) = feed(&mut s.machine, &body, vfs_root_for_feed.as_deref());

    // Apply the current state's pre-apply edits to the VFS. Done after
    // feed() so a transition into a new state triggers that state's
    // pre-apply before the next build_sse. Idempotent — safe to call
    // every request.
    if let Some(root) = &s.vfs_root {
        apply_pre_apply(&s.machine, root);
    }

    if is_terminal {
        if !s.reported {
            s.reported = true;
            let n = s.machine.events.len();
            let t = s.machine.current_state_name();
            eprintln!("[aces] terminal: {t} ({n} events)");
        }
        // Mark the wall-clock time of first terminal hit. The idle
        // watcher will set `scenario_complete` once 5s passes with no
        // further agent requests, signalling the suite runner to
        // advance to the next scenario.
        if s.terminal_at.is_none() {
            s.terminal_at = Some(std::time::Instant::now());
        }
        // First terminal response: send the terminal state's `assistant`
        // text (scenario-controlled) so the agent has context to write its
        // own summary. Subsequent requests from the agent's post-terminal
        // polling loop get just [DONE] — the empty stream signals the
        // conversation is over without spamming the log. We deliberately
        // do NOT inject "do not use any more tools" or other mock-level
        // meta-instructions — that's for the scenario YAML to express
        // via the terminal state's own `assistant` field.
        if !s.terminal_sent {
            s.terminal_sent = true;
            let msg = s.machine.current_prompt()
                .unwrap_or("Task complete.");
            let content_event = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"index\":0}}]}}\n\n",
                serde_json::to_string(msg).unwrap_or_default()
            );
            let finish = "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":10,\"total_tokens\":10}}\n\n";
            return (content_type(), format!("{}{}data: [DONE]\n\n", content_event, finish));
        } else {
            return (content_type(), "data: [DONE]\n\n".to_string());
        }
    }

    // If state didn't advance AND there were events to process (agent sent tool
    // results but none matched), return detailed guidance instead of tool call
    // to prevent infinite loops. If no events (first request), just return the
    // current state's tool call/text.
    if !advanced && had_events {
        let expected_tools: Vec<String> = s.machine.current_tool_calls().iter().map(|t| t.to_string()).collect();
        let expected_args: Vec<String> = expected_tools.iter().filter_map(|t| {
            s.machine.current_tool_args_for(t, &s.agent_format).map(|a| {
                format!("{}({})", t, serde_json::to_string(&a).unwrap_or_default())
            })
        }).collect();
        let msg = format!(
            "I notice you're not using the expected tools. You must use: {}. Arguments: {}. Please call the correct tool now.",
            expected_tools.join(", "),
            expected_args.join(", ")
        );
        let sse = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"index\":0}}]}}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"index\":0,\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":10,\"total_tokens\":10}}}}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&msg).unwrap_or_default()
        );
        return (content_type(), sse);
    }

    let sse = match &s.vfs_root {
        Some(root) => build_sse(&s.machine, root, &s.agent_format),
        None => {
            // VFS not yet created (system prompt had no cwd and mock cwd
            // failed). Return a text message so the agent doesn't loop on
            // missing tool args.
            let msg = "Mock VFS could not be initialized (no working directory detected). Please ensure your system prompt includes 'Working directory: /path' or run the mock from a valid directory.";
            let content_event = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"index\":0}}]}}\n\n",
                serde_json::to_string(&msg).unwrap_or_default()
            );
            let finish = "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":10,\"total_tokens\":10}}\n\n";
            format!("{}{}data: [DONE]\n\n", content_event, finish)
        }
    };
    (content_type(), sse)
}

fn content_type() -> [(axum::http::HeaderName, &'static str); 1] {
    [(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")]
}

/// Scan a request body for telltale signs that a Claude Code plugin
/// (typically claude-mem) is intercepting tool calls. Returns a short
/// description of the first signal found, or `None` if the body looks
/// clean. Walks the `messages` array and inspects every string value
/// (content, tool_call arguments, etc.) — the plugin noise is appended
/// inside the tool result content string.
fn detect_plugin_interference(body: &Value) -> Option<&'static str> {
    let messages = body.get("messages")?.as_array()?;
    for msg in messages {
        // Tool result messages: content is a string (Chat Completions format)
        if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
            if content.contains("<observed_from_primary_session>") {
                return Some("tool result contains <observed_from_primary_session> (claude-mem plugin is active)");
            }
            if content.contains("not enabled in this context") {
                return Some("tool result contains 'not enabled in this context' (a plugin spawned a sub-session with --disallowedTools)");
            }
            if content.contains("<system-reminder>") && content.contains("Return either one or more <observation>") {
                return Some("tool result contains a claude-mem observation prompt");
            }
        }
        // Assistant messages may carry tool_calls; args can contain text too
        if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                if let Some(args) = tc.get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                {
                    if args.contains("<observed_from_primary_session>") {
                        return Some("tool_call arguments contain <observed_from_primary_session>");
                    }
                }
            }
        }
    }
    None
}

fn feed(machine: &mut StateMachine, body: &Value, vfs_root: Option<&std::path::Path>) -> (bool, bool, bool) {
    let state_before = machine.current_state_name().to_string();
    // File-content transitions (e.g., on_file_contains) are evaluated
    // BEFORE event matching — they check the actual VFS file state, not
    // the agent's tool call results. This makes Edit verification robust
    // against opaque tool result formats.
    if let Some(root) = vfs_root {
        if let Some(next) = check_file_transitions(machine, root) {
            machine.set_state(&next);
        }
    }
    let state_after_file_check = machine.current_state_name().to_string();
    if state_before != state_after_file_check {
        eprintln!("[aces] state: {} → {} (file check)", state_before, state_after_file_check);
        return (machine.is_terminal(), true, false);
    }
    let events = extract_events_from_request(body);
    let had_events = !events.is_empty();
    let mut advanced = false;
    for event in &events {
        eprintln!("[agent] {:?}", event);
        if machine.feed(event.clone()) {
            advanced = true;
        }
    }
    let state_after = machine.current_state_name().to_string();
    // Stuck-loop guard: if the same state has been entered more than
    // DEFAULT_MAX_STATE_VISITS times, force a terminal transition to
    // stop log spam and let the scenario settle.
    if !machine.is_terminal() && machine.is_stuck(crate::torture::scenario::DEFAULT_MAX_STATE_VISITS) {
        let visits = machine.visit_count(&state_after);
        eprintln!(
            "[aces] state {} visited {} times (>{}); forcing give_up to break loop",
            state_after, visits, crate::torture::scenario::DEFAULT_MAX_STATE_VISITS
        );
        if machine.scenario().states.contains_key("give_up") {
            machine.set_state("give_up");
        } else {
            eprintln!("[aces] (no 'give_up' state in scenario; staying put)");
        }
    } else if state_before != state_after {
        eprintln!("[aces] state: {} → {}", state_before, state_after);
    } else if events.is_empty() {
        eprintln!("[aces] no events, staying at {}", state_after);
    } else if !advanced {
        eprintln!("[aces] no transition matched, staying at {}", state_after);
    }
    (machine.is_terminal(), advanced, had_events)
}

/// Walk the current state's transitions; if any is `OnFileContains` and
/// the file's current text contains the expected substring, return the
/// next state name. The `file_path` may include `${VFS}` which is
/// resolved against the VFS root.
pub(crate) fn check_file_transitions(machine: &StateMachine, vfs_root: &std::path::Path) -> Option<String> {
    use crate::torture::scenario::Transition;
    let state = machine.scenario().states.get(machine.current_state_name())?;
    for t in &state.transitions {
        if let Transition::OnFileContains { file_path, contains, next } = t {
            let resolved = file_path.replace("${VFS}", &vfs_root.display().to_string());
            if let Ok(content) = std::fs::read_to_string(&resolved) {
                if content.contains(contains.as_str()) {
                    return Some(next.clone());
                }
            }
        }
    }
    None
}

/// Apply the current state's `pre_apply` edits to the VFS on disk. Each
/// edit is a `from` → `to` string replacement in `file_path` (with
/// `${VFS}` resolved). Idempotent: if `from` is not found (the file is
/// already in the post-edit state, or the edit is a no-op), nothing
/// happens. This is mock-only — used to make tests pass for agents
/// whose runtime can't actually execute the scenario's edit tool (e.g.
/// Codex's `apply_patch` doesn't survive the proxy's Chat-Completions →
/// Responses translation as a typed tool).
pub(crate) fn apply_pre_apply(machine: &StateMachine, vfs_root: &std::path::Path) {
    use crate::torture::scenario::FileEdit;
    let state = match machine.scenario().states.get(machine.current_state_name()) {
        Some(s) => s,
        None => return,
    };
    for edit in &state.pre_apply {
        let FileEdit { file_path, from, to } = edit;
        let resolved = file_path.replace("${VFS}", &vfs_root.display().to_string());
        let Ok(content) = std::fs::read_to_string(&resolved) else { continue };
        if !content.contains(from.as_str()) { continue; }
        let new_content = content.replacen(from.as_str(), to.as_str(), 1);
        match std::fs::write(&resolved, &new_content) {
            Ok(()) => eprintln!("[aces] pre_apply: {} ({} → {})", resolved, from, to),
            Err(e) => eprintln!("[aces] pre_apply failed for {resolved}: {e}"),
        }
    }
}

fn build_sse(machine: &StateMachine, vfs_root: &std::path::Path, agent_format: &str) -> String {
    let tool_calls = machine.current_tool_calls();

    if !tool_calls.is_empty() {
        let tc_json: Vec<String> = tool_calls.iter().enumerate().map(|(i, name)| {
            let args = machine.current_tool_args_for(name, agent_format)
                .unwrap_or_else(|| serde_json::json!({}));
            let args_str = resolve_args(&args, vfs_root);
            let args_field = serde_json::Value::String(args_str);
            let args_field_str = serde_json::to_string(&args_field).unwrap_or_default();
            format!(
                r#"{{"index":{i},"id":"toolu_{i}","function":{{"name":{},"arguments":{}}},"type":"function"}}"#,
                serde_json::to_string(name).unwrap_or_default(),
                args_field_str
            )
        }).collect();
        format!(
            r#"data: {{"choices":[{{"delta":{{"tool_calls":[{}]}},"index":0,"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":0,"completion_tokens":10,"total_tokens":10}}}}

data: [DONE]

"#,
            tc_json.join(",")
        )
    } else {
        let msg = machine.current_prompt().unwrap_or("Continue.");
        let content_event = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"index\":0}}]}}\n\n",
            serde_json::to_string(&msg).unwrap_or_default()
        );
        let finish = "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":10,\"total_tokens\":10}}\n\n";
        format!("{}{}data: [DONE]\n\n", content_event, finish)
    }
}
