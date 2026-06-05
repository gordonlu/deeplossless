//! 0-LLM scenario driver.
//!
//! Drives ACES scenarios to completion without an LLM by:
//! 1. Loading the scenario and creating its VFS on disk
//! 2. For each state, applying the state's `pre_apply` edits to the VFS
//! 3. Executing the state's tool calls (Read/Edit/Bash/apply_patch/...) using
//!    real file-system and shell operations
//! 4. Feeding the resulting `AgentEvent` back into the state machine
//! 5. Continuing until a terminal state is reached
//!
//! This tests the agent's *tool layer* end-to-end: can the mock's tool calls
//! be parsed, routed to the right implementation, executed against the VFS,
//! and produce events that advance the state machine? The LLM (which normally
//! decides which tool to call) is replaced by a deterministic walk: the
//! driver follows whatever tool the state machine dictates.
//!
//! Scoring reflects the mock's design and the tool layer's correctness, not
//! the LLM's choices. The driver is useful for validating that the VFS is
//! set up correctly, the state machine is reachable from start to terminal,
//! and tool execution produces the expected events.

use crate::torture::aces::{apply_pre_apply, check_file_transitions, create_vfs};
use crate::torture::scenario::{score_run, AgentEvent, Scenario, ScenarioRun, StateMachine};
use std::path::{Path, PathBuf};

/// Result of driving a single scenario.
#[derive(Debug)]
pub struct DriveOutcome {
    pub scenario: String,
    pub terminal_state: String,
    pub events: Vec<AgentEvent>,
    pub score: f64,
    pub success: bool,
}

/// Drive a single scenario to completion with 0 LLM intervention.
///
/// `format` selects the per-agent arg convention ("claude_code", "codex",
/// "openai"). `vfs_parent` is the parent directory under which a per-scenario
/// VFS subdirectory will be created.
pub fn drive_scenario(
    name: &str,
    format: &str,
    vfs_parent: &Path,
) -> Result<DriveOutcome, String> {
    let scenario = Scenario::load_with_format(name, Some(format))
        .map_err(|e| format!("load scenario '{name}': {e}"))?;

    let vfs_root = vfs_parent.join(format!("aces_vfs_{name}"));
    create_vfs(&scenario, &vfs_root)
        .map_err(|e| format!("create vfs at {}: {e}", vfs_root.display()))?;

    let mut machine = StateMachine::new(scenario);
    let mut last_event: Option<AgentEvent> = None;

    // Bound the loop — scenarios should terminate in O(states) steps.
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

        // Apply pre_apply edits before responding to tool calls. This is
        // what the mock does in handle_request — the file on disk is fixed
        // before the agent's edit is "evaluated" via the transition check.
        apply_pre_apply(&machine, &vfs_root);

        // Get the current state's tool calls + args. The driver follows
        // them in order. If multiple tool_calls are listed, we execute all
        // of them; only the last event's effect on transitions matters
        // (matches what the mock's build_sse does — emits all in one chunk).
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
            // No tool calls in this state — shouldn't happen for non-terminal
            // states, but guard against infinite loops.
            return Err(format!(
                "scenario '{name}' stuck at state '{state_name}' — no tool calls and not terminal"
            ));
        }

        for event in &state_events {
            machine.feed(event.clone());
        }
        // After feeding events, check file-based transitions (OnFileContains
        // in the YAML). The mock does this in handle_request; we have to do
        // it explicitly because the state machine's feed() only matches on
        // event patterns, not file content.
        if let Some(next) = check_file_transitions(&machine, &vfs_root) {
            machine.set_state(&next);
        }
        last_event = state_events.last().cloned();
    }

    let terminal = machine.current_state_name().to_string();
    let success = machine.is_success();
    let events = machine.events.clone();

    let expected_search = machine.scenario().expected_search;
    let expected_read = machine.scenario().expected_read;
    let run = ScenarioRun {
        scenario: name.to_string(),
        events: events.clone(),
        terminal_state: Some(terminal.clone()),
        score: None,
        expected_search,
        expected_read,
    };
    let score = score_run(&run).total;

    // Suppress unused warning — last_event is useful for debugging.
    let _ = last_event;

    Ok(DriveOutcome {
        scenario: name.to_string(),
        terminal_state: terminal,
        events,
        score,
        success,
    })
}

/// Drive every base scenario in sequence, returning per-scenario outcomes.
pub fn drive_suite(format: &str, vfs_parent: &Path) -> Vec<Result<DriveOutcome, String>> {
    let names = match Scenario::list_base() {
        Ok(n) if !n.is_empty() => n,
        Ok(_) => return vec![Err("no base scenarios found in scenarios/".to_string())],
        Err(e) => return vec![Err(format!("list_base: {e}"))],
    };

    let mut out = Vec::with_capacity(names.len());
    for name in &names {
        out.push(drive_scenario(name, format, vfs_parent));
    }
    out
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
        let outcome = drive_scenario("hidden_bug", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert!(!outcome.terminal_state.is_empty());
        assert!(outcome.score > 0.0, "score should be > 0");
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
        let outcome = drive_scenario("01_fix_test_failure", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_02_add_feature_claude_code() {
        let parent = temp_parent("02_cc");
        let outcome = drive_scenario("02_add_feature", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_03_refactor_rename_claude_code() {
        let parent = temp_parent("03_cc");
        let outcome = drive_scenario("03_refactor_rename", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_04_search_to_fix_claude_code() {
        let parent = temp_parent("04_cc");
        let outcome = drive_scenario("04_search_to_fix", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_05_multi_file_edit_claude_code() {
        let parent = temp_parent("05_cc");
        let outcome = drive_scenario("05_multi_file_edit", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_06_debug_from_logs_claude_code() {
        let parent = temp_parent("06_cc");
        let outcome = drive_scenario("06_debug_from_logs", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_07_security_fix_claude_code() {
        let parent = temp_parent("07_cc");
        let outcome = drive_scenario("07_security_fix", "claude_code", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_codex_format_reaches_terminal() {
        let parent = temp_parent("hidden_bug_codex");
        let outcome = drive_scenario("hidden_bug", "codex", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_openai_format_reaches_terminal() {
        let parent = temp_parent("hidden_bug_openai");
        let outcome = drive_scenario("hidden_bug", "openai", &parent).expect("drive");
        assert!(outcome.success, "expected success, got {outcome:?}");
        assert_eq!(outcome.terminal_state, "verify");
    }

    #[test]
    fn drive_suite_runs_all_base_scenarios() {
        let parent = temp_parent("suite");
        let outcomes = drive_suite("claude_code", &parent);
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
