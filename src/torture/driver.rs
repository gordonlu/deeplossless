//! 0-LLM scenario driver — a diagnostic tool, not a benchmark.
//!
//! Drives ACES scenarios to completion without an LLM by following the
//! scenario's state machine deterministically: for each state, the driver
//! applies `pre_apply` edits to the VFS, executes the state's tool calls
//! against the real filesystem, and feeds the resulting `AgentEvent`s back
//! into the state machine. It continues until a terminal state is reached
//! or `max_steps` is exceeded.
//!
//! This is useful for:
//! - Checking that the YAML state machine has no dead-ends
//! - Verifying `pre_apply` from→to strings match the actual VFS content
//! - Testing `args_per_agent` parameter mapping for a given agent format
//! - Debugging `on_file_contains` transitions without an LLM involved
//!
//! The driver's value is in the **diagnostic output** (`verbose = true`),
//! not in the score. Use it to validate your scenario YAML before hooking
//! up a real agent.

use crate::torture::aces::{apply_pre_apply, check_file_transitions, create_vfs};
use crate::torture::scenario::{AgentEvent, Scenario, StateMachine};
use std::path::{Path, PathBuf};

/// Result of driving a single scenario.
#[derive(Debug)]
pub struct DriveOutcome {
    pub scenario: String,
    pub terminal_state: String,
    pub events: Vec<AgentEvent>,
    pub steps: usize,
    pub success: bool,
}

/// Drive a single scenario to completion with 0 LLM intervention.
///
/// `format` selects the per-agent arg convention ("claude_code", "codex",
/// "openai"). `vfs_parent` is the parent directory under which a per-scenario
/// VFS subdirectory will be created.
///
/// When `verbose` is true, each step is printed to stderr with the state
/// name, the tool calls executed, and the next state.
pub fn drive_scenario(
    name: &str,
    format: &str,
    vfs_parent: &Path,
    verbose: bool,
) -> Result<DriveOutcome, String> {
    let scenario = Scenario::load_with_format(name, Some(format))
        .map_err(|e| format!("load scenario '{name}': {e}"))?;

    let vfs_root = vfs_parent.join(format!("aces_vfs_{name}"));
    create_vfs(&scenario, &vfs_root)
        .map_err(|e| format!("create vfs at {}: {e}", vfs_root.display()))?;

    let mut machine = StateMachine::new(scenario);

    let max_steps = 200;
    let mut step = 0;

    while !machine.is_terminal() {
        if step >= max_steps {
            return Err(format!(
                "scenario '{name}' did not terminate within {max_steps} steps (current state: {})",
                machine.current_state_name()
            ));
        }
        step += 1;

        let state_name = machine.current_state_name().to_string();

        apply_pre_apply(&machine, &vfs_root);

        let tool_calls = machine.current_tool_calls();
        let mut state_events: Vec<AgentEvent> = Vec::new();

        for tool in &tool_calls {
            let args = machine
                .current_tool_args_for(tool, format)
                .unwrap_or_else(|| serde_json::json!({}));
            let event = execute_tool_call(tool, &args, &vfs_root);
            state_events.push(event);
        }

        if state_events.is_empty() {
            return Err(format!(
                "scenario '{name}' stuck at state '{state_name}' — no tool calls and not terminal"
            ));
        }

        for event in &state_events {
            machine.feed(event.clone());
        }
        if let Some(next) = check_file_transitions(&machine, &vfs_root) {
            machine.set_state(&next);
        }

        if verbose {
            let tool_info: Vec<String> = state_events
                .iter()
                .map(|e| match e {
                    AgentEvent::Read(p, _) => format!("Read({})", short_path(p, &vfs_root)),
                    AgentEvent::Edit(p, r) => format!("Edit({}) [{}]", short_path(p, &vfs_root), r),
                    AgentEvent::Search(q, _) => format!("Search({})", q),
                    AgentEvent::Test(t, _) => format!("Test({})", t),
                    AgentEvent::Task(t, _) => format!("Task({})", t),
                    _ => format!("{}", e.args()),
                })
                .collect();
            eprintln!(
                "[{:>3}] {:<20} {:<50} → {}",
                step,
                truncate(&state_name, 20),
                truncate(&tool_info.join(", "), 50),
                machine.current_state_name()
            );
        }
    }

    let terminal = machine.current_state_name().to_string();
    let success = machine.is_success();
    let events = machine.events.clone();

    if verbose {
        match (success, &terminal) {
            (true, _) => eprintln!("  ✓ {} ({step} steps, {} events)", name, events.len()),
            (false, t) => eprintln!("  ✗ {} → stuck at {t} after {step} steps", name),
        }
    }

    Ok(DriveOutcome {
        scenario: name.to_string(),
        terminal_state: terminal,
        events,
        steps: step,
        success,
    })
}

/// Drive every base scenario in sequence, returning per-scenario outcomes.
pub fn drive_suite(format: &str, vfs_parent: &Path, verbose: bool) -> Vec<Result<DriveOutcome, String>> {
    let names = match Scenario::list_base() {
        Ok(n) if !n.is_empty() => n,
        Ok(_) => return vec![Err("no base scenarios found in scenarios/".to_string())],
        Err(e) => return vec![Err(format!("list_base: {e}"))],
    };

    if verbose {
        eprintln!("=== drive suite: {format} ===");
        eprintln!("   {} base scenarios found", names.len());
    }

    let mut out = Vec::with_capacity(names.len());
    for name in &names {
        if verbose {
            eprintln!("\n── {name} ──");
        }
        out.push(drive_scenario(name, format, vfs_parent, verbose));
    }

    if verbose {
        let ok = out.iter().filter(|o| o.as_ref().map(|o| o.success).unwrap_or(false)).count();
        let err = out.len() - ok;
        eprintln!("\n=== result: {ok} ok, {err} fail ===");
    }

    out
}

// ── Helpers ─────────────────────────────────────────────────────────

fn short_path(path: &str, vfs_root: &Path) -> String {
    let root_str = vfs_root.to_string_lossy();
    if path.starts_with(root_str.as_ref()) {
        path[root_str.len()..].trim_start_matches('/').to_string()
    } else {
        let segments: Vec<&str> = path.split('/').collect();
        if segments.len() <= 3 { path.to_string() }
        else { format!("{}/{}/{}", segments[0], segments[1], segments[segments.len()-1]) }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() }
    else { format!("{}…", &s[..max.saturating_sub(1)]) }
}

/// Execute a single tool call against the VFS and return the resulting event.
fn execute_tool_call(tool: &str, args: &serde_json::Value, vfs_root: &Path) -> AgentEvent {
    match tool {
        "Read" | "read" | "read_file" => {
            let path = pick_str(args, &["filePath", "file_path", "path"]);
            let resolved = resolve_path(&path, vfs_root);
            let result = std::fs::read_to_string(&resolved).unwrap_or_default();
            AgentEvent::Read(resolved.clone(), result)
        }

        "Edit" | "edit" | "edit_file" | "replace" | "Write" | "write" | "write_to_file" => {
            // For Write-style tools, take content from `content` field.
            // For Edit-style, take from old/new string fields.
            let content_field = pick_str(args, &["content"]);
            if !content_field.is_empty() {
                let path = pick_str(args, &["filePath", "file_path", "path"]);
                let resolved = resolve_path(&path, vfs_root);
                if let Some(parent) = Path::new(&resolved).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&resolved, &content_field);
                AgentEvent::Edit(resolved.clone(), "wrote".to_string())
            } else {
                let path = pick_str(args, &["filePath", "file_path", "path"]);
                let old = pick_str(args, &["oldString", "old_string"]);
                let new = pick_str(args, &["newString", "new_string"]);
                let resolved = resolve_path(&path, vfs_root);
                let result = apply_edit(&resolved, &old, &new);
                AgentEvent::Edit(resolved.clone(), result)
            }
        }

        "apply_patch" => {
            // Codex's apply_patch takes a nested operation.
            // { operation: { type, path, diff } }
            let op = &args["operation"];
            let path = pick_str(op, &["path"]);
            let diff = pick_str(op, &["diff"]);
            let resolved = resolve_path(&path, vfs_root);
            let result = apply_diff(&resolved, &diff);
            AgentEvent::Edit(resolved.clone(), result)
        }

        "Bash" | "bash" | "shell" | "execute_command" | "command" | "run" => {
            let cmd = pick_str(args, &["command"]);
            let (stdout, _stderr, _code) = run_bash(&cmd, vfs_root);
            // Match the mock's classification: bash with "test" or "grep" in
            // args is classified as Test or Search, otherwise Task.
            let lower = cmd.to_lowercase();
            if lower.contains("grep") || lower.contains("rg ") || lower.contains("search") {
                AgentEvent::Search(cmd, stdout)
            } else if lower.contains("test") || lower.contains("check") {
                AgentEvent::Test(cmd, stdout)
            } else {
                AgentEvent::Task(cmd, stdout)
            }
        }

        "Grep" | "grep" | "rg" | "search" | "search_file" | "find_in_files" => {
            let query = serde_json::to_string(args).unwrap_or_default();
            let (stdout, _, _) = run_bash(
                &format!("grep -rn '{}' . 2>/dev/null | head -50", query.escape_default()),
                vfs_root,
            );
            AgentEvent::Search(query, stdout)
        }

        "Glob" | "glob" => {
            let pattern = pick_str(args, &["pattern", "glob"]);
            let (stdout, _, _) = run_bash(
                &format!("find . -path '{}' 2>/dev/null | head -50", pattern),
                vfs_root,
            );
            AgentEvent::Search(pattern, stdout)
        }

        "test" | "cargo_test" | "run_test" | "pytest" | "npm_test" => {
            let args_str = serde_json::to_string(args).unwrap_or_default();
            let (stdout, _, _) = run_bash(&args_str, vfs_root);
            AgentEvent::Test(args_str, stdout)
        }

        other => {
            // Unknown tool — emit Other event so the state machine's
            // on_tool fallback can still advance.
            AgentEvent::Other(
                format!("{other}({})", serde_json::to_string(args).unwrap_or_default()),
                String::new(),
            )
        }
    }
}

fn pick_str(args: &serde_json::Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = args.get(*k).and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    String::new()
}

fn resolve_path(p: &str, vfs_root: &Path) -> String {
    if p.is_empty() {
        return vfs_root.display().to_string();
    }
    p.replace("${VFS}", &vfs_root.display().to_string())
}

/// Apply a simple find-and-replace edit. Returns a short status string.
fn apply_edit(path: &str, old: &str, new: &str) -> String {
    if path.is_empty() {
        return "missing path".to_string();
    }
    let Ok(content) = std::fs::read_to_string(path) else {
        return format!("read failed: {path}");
    };
    if !content.contains(old) {
        return "old not found".to_string();
    }
    // replacen(.., 1) matches the mock's pre_apply semantics.
    let updated = content.replacen(old, new, 1);
    match std::fs::write(path, &updated) {
        Ok(()) => "edited".to_string(),
        Err(e) => format!("write failed: {e}"),
    }
}

/// Apply a unified-diff body to a file. Minimal implementation: parse `@@`
/// hunks and apply each as find-and-replace on the chunk content.
fn apply_diff(path: &str, diff: &str) -> String {
    if path.is_empty() {
        return "missing path".to_string();
    }
    let Ok(content) = std::fs::read_to_string(path) else {
        return format!("read failed: {path}");
    };
    let mut updated = content.clone();
    for hunk in parse_diff_hunks(diff) {
        // hunk.old_text is the context+removed lines (what to find).
        // hunk.new_text is the context+added lines (what to replace with).
        if !updated.contains(&hunk.old_text) {
            return format!("hunk not found in {}", path);
        }
        updated = updated.replacen(&hunk.old_text, &hunk.new_text, 1);
    }
    match std::fs::write(path, &updated) {
        Ok(()) => "patched".to_string(),
        Err(e) => format!("write failed: {e}"),
    }
}

struct DiffHunk {
    old_text: String,
    new_text: String,
}

/// Minimal unified-diff parser. Only handles the file's single hunk or
/// multiple consecutive hunks — enough for the scenarios.
fn parse_diff_hunks(diff: &str) -> Vec<DiffHunk> {
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut current: Option<DiffHunk> = None;
    for line in diff.lines() {
        if line.starts_with("@@") {
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            current = Some(DiffHunk {
                old_text: String::new(),
                new_text: String::new(),
            });
            continue;
        }
        let Some(h) = current.as_mut() else { continue };
        if let Some(rest) = line.strip_prefix(' ') {
            // context line — appears in both old and new
            h.old_text.push_str(rest);
            h.old_text.push('\n');
            h.new_text.push_str(rest);
            h.new_text.push('\n');
        } else if let Some(rest) = line.strip_prefix('-') {
            // removed line — only in old
            h.old_text.push_str(rest);
            h.old_text.push('\n');
        } else if let Some(rest) = line.strip_prefix('+') {
            // added line — only in new
            h.new_text.push_str(rest);
            h.new_text.push('\n');
        }
    }
    if let Some(h) = current.take() {
        hunks.push(h);
    }
    hunks
}

/// Run a bash command in the VFS root. Returns (stdout, stderr, exit_code).
fn run_bash(cmd: &str, vfs_root: &Path) -> (String, String, i32) {
    if cmd.is_empty() {
        return (String::new(), "no command".to_string(), 1);
    }
    let output = std::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(vfs_root)
        .output();
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            let code = o.status.code().unwrap_or(-1);
            (stdout, stderr, code)
        }
        Err(e) => (String::new(), format!("spawn failed: {e}"), -1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_parent(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "aces_driver_test_{label}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn drive_hidden_bug_claude_code_reaches_terminal() {
        let parent = temp_parent("hidden_bug_cc");
        let outcome = drive_scenario("hidden_bug", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert!(!outcome.terminal_state.is_empty());
        // The hidden_bug state path includes verify.
        assert!(
            outcome.terminal_state == "verify",
            "expected terminal 'verify', got '{}'",
            outcome.terminal_state
        );
    }

    #[test]
    fn drive_01_fix_test_failure_claude_code() {
        let parent = temp_parent("01_cc");
        let outcome = drive_scenario("01_fix_test_failure", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_02_add_feature_claude_code() {
        let parent = temp_parent("02_cc");
        let outcome = drive_scenario("02_add_feature", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_03_refactor_rename_claude_code() {
        let parent = temp_parent("03_cc");
        let outcome = drive_scenario("03_refactor_rename", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_04_search_to_fix_claude_code() {
        let parent = temp_parent("04_cc");
        let outcome = drive_scenario("04_search_to_fix", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_05_multi_file_edit_claude_code() {
        let parent = temp_parent("05_cc");
        let outcome = drive_scenario("05_multi_file_edit", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_06_debug_from_logs_claude_code() {
        let parent = temp_parent("06_cc");
        let outcome = drive_scenario("06_debug_from_logs", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_07_security_fix_claude_code() {
        let parent = temp_parent("07_cc");
        let outcome = drive_scenario("07_security_fix", "claude_code", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_codex_format_reaches_terminal() {
        let parent = temp_parent("hidden_bug_codex");
        let outcome = drive_scenario("hidden_bug", "codex", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_openai_format_reaches_terminal() {
        let parent = temp_parent("hidden_bug_openai");
        let outcome = drive_scenario("hidden_bug", "openai", &parent, false).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_suite_runs_all_base_scenarios() {
        let parent = temp_parent("suite");
        let outcomes = drive_suite("claude_code", &parent, false);
        assert!(!outcomes.is_empty(), "suite should have at least one scenario");
        let failures: Vec<_> = outcomes
            .iter()
            .filter(|o| o.is_err() || !o.as_ref().unwrap().success)
            .collect();
        assert!(
            failures.is_empty(),
            "all base scenarios should drive to success, failures: {failures:?}"
        );
    }

    #[test]
    fn apply_diff_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "aces_diff_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("f.txt");
        std::fs::write(&f, "line1\nline2\nline3\n").unwrap();
        let diff = "@@ -1,3 +1,3 @@\n line1\n-line2\n+LINE2\n line3\n";
        let r = apply_diff(&f.display().to_string(), diff);
        assert_eq!(r, "patched");
        let after = std::fs::read_to_string(&f).unwrap();
        assert_eq!(after, "line1\nLINE2\nline3\n");
    }
}
