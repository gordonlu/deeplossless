use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ── Types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub vfs: VirtualFS,
    pub states: HashMap<String, State>,
    pub start: String,
    #[serde(default)]
    pub weights: Option<ScenarioWeights>,
    #[serde(default)]
    pub finish: Option<String>,
    #[serde(default)]
    pub fail: Option<String>,
    /// Whether the scenario expects the agent to do at least one
    /// `Search`/`Grep`/`Glob`. When `true` and the agent performs 0
    /// searches, the Search Efficiency score is 0/20 (penalty). When
    /// `false` (the default) and the agent performs 0 searches, the
    /// score is 20/20 (vacuous perfect — no need to search). Same
    /// semantics for `expected_read` on `Read` events.
    ///
    /// We deliberately do NOT cap search/read counts via budgets:
    /// an excellent agent might do 1 search where another needs 5,
    /// and the ratio-based metrics already reward "every search
    /// leading to action" without penalizing the higher-volume path.
    #[serde(default)]
    pub expected_search: bool,
    #[serde(default)]
    pub expected_read: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioWeights {
    #[serde(default = "default_exploration")]
    pub exploration: f64,
    #[serde(default = "default_reuse")]
    pub reuse: f64,
    #[serde(default = "default_verification")]
    pub verification: f64,
    #[serde(default = "default_efficiency")]
    pub efficiency: f64,
    #[serde(default = "default_correctness")]
    pub correctness: f64,
    #[serde(default = "default_tool_strategy")]
    pub tool_strategy: f64,
}

fn default_exploration() -> f64 { 15.0 }
fn default_reuse() -> f64 { 15.0 }
fn default_verification() -> f64 { 20.0 }
fn default_efficiency() -> f64 { 15.0 }
fn default_correctness() -> f64 { 15.0 }
fn default_tool_strategy() -> f64 { 20.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFS {
    #[serde(default)]
    pub files: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub assistant: String,
    #[serde(default)]
    pub tool_calls: Vec<String>,
    /// Default tool argument templates. Field names follow the Chat
    /// Completions / OpenAI camelCase convention (e.g. `filePath`,
    /// `oldString`, `newString`). Used for agent format `openai` (the
    /// CLI default), or for any other agent that has no entry in
    /// `args_per_agent`.
    #[serde(default)]
    pub args: std::collections::HashMap<String, serde_json::Value>,
    /// Per-agent-format overrides for `args`. Each entry REPLACES the
    /// default for that tool (no field-level merge) so the YAML can use
    /// the field names expected by the target agent (e.g. `file_path`/
    /// `old_string` for Claude Code). Example:
    /// ```yaml
    /// args_per_agent:
    ///   claude_code:
    ///     Edit:
    ///       file_path: "${VFS}/src/config.sh"
    ///       old_string: "TIMEOUT_DEFAULT=30"
    ///       new_string: "TIMEOUT_DEFAULT=60"
    /// ```
    #[serde(default)]
    pub args_per_agent:
        std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
    /// Edits the mock applies to the VFS file on disk *before* emitting
    /// the state's tool call. Used for agents whose runtime cannot
    /// execute the tool (e.g. Codex's typed `apply_patch` doesn't pass
    /// through deeplossless's Chat-Completions→Responses translation as
    /// a typed tool, so the agent's call ends up rejected). With
    /// `pre_apply`, the file is already in the post-edit state when the
    /// scenario's `on_file_contains` check fires, so the test still
    /// verifies the agent's navigation (read → decide → call edit) even
    /// though the actual disk write was done by the mock.
    #[serde(default)]
    pub pre_apply: Vec<FileEdit>,
    #[serde(default)]
    pub transitions: Vec<Transition>,
    #[serde(default)]
    pub finish: bool,
    #[serde(default)]
    pub fail: bool,
}

/// One pre-applied file edit: a simple `from` → `to` string replacement
/// in `file_path`. The mock resolves `${VFS}` in the path before
/// reading. If `from` is not found in the file, the edit is skipped
/// (idempotent — re-running is a no-op once the agent is done).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEdit {
    pub file_path: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Transition {
    OnTool { tool: String, next: String },
    OnRead { pattern: String, next: String },
    OnEdit { pattern: String, next: String },
    OnTest { result: String, next: String },
    OnTask { next: String },
    OnDone { next: String },
    #[serde(rename_all = "snake_case")]
    OnEditResult { pattern: String, result_contains: String, next: String },
    #[serde(rename_all = "snake_case")]
    OnTestResult { result_contains: String, next: String },
    /// File-content check: reads the VFS file and matches if the file's
    /// current text contains `contains`. Evaluated by the mock on every
    /// request, independent of agent tool calls. Useful when the agent's
    /// tool result format is opaque (e.g., claude code's Edit returns
    /// "updated successfully" without the new value). The `file_path` may
    /// include `${VFS}` — the mock resolves it against the scenario's
    /// VFS root before reading.
    #[serde(rename_all = "snake_case")]
    OnFileContains { file_path: String, contains: String, next: String },
}

// ── Agent event trace ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    Search(String, String),
    Read(String, String),
    Edit(String, String),
    Test(String, String),
    Task(String, String),
    Ask(String, String),
    Write(String, String),
    Other(String, String),
    Done,
}

impl AgentEvent {
    pub fn args(&self) -> &str {
        match self {
            AgentEvent::Search(a, _) | AgentEvent::Read(a, _) | AgentEvent::Edit(a, _)
            | AgentEvent::Test(a, _) | AgentEvent::Task(a, _) | AgentEvent::Ask(a, _)
            | AgentEvent::Write(a, _) | AgentEvent::Other(a, _) => a,
            AgentEvent::Done => "",
        }
    }
    pub fn result(&self) -> &str {
        match self {
            AgentEvent::Search(_, r) | AgentEvent::Read(_, r) | AgentEvent::Edit(_, r)
            | AgentEvent::Test(_, r) | AgentEvent::Task(_, r) | AgentEvent::Ask(_, r)
            | AgentEvent::Write(_, r) | AgentEvent::Other(_, r) => r,
            AgentEvent::Done => "",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioRun {
    pub scenario: String,
    pub events: Vec<AgentEvent>,
    pub terminal_state: Option<String>,
    pub score: Option<ScoreResult>,
    /// Copied from the `Scenario` so the scorer can decide whether 0
    /// searches/reads is "vacuous perfect" (default) or "missed the
    /// point" (when the YAML declared `expected_search: true`).
    #[serde(default)]
    pub expected_search: bool,
    #[serde(default)]
    pub expected_read: bool,
}

// ── Load ───────────────────────────────────────────────────────────

impl Scenario {
    pub fn from_yaml(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    pub fn load(name: &str) -> anyhow::Result<Self> {
        Self::load_with_format(name, None)
    }

    /// Load a scenario, preferring a per-agent variant when `agent_format`
    /// is set. Lookup order: `{name}.{format}.yaml` → `{name}.yaml`.
    /// Per-agent variants let the same logical scenario ship multiple
    /// tool-arg conventions (camelCase for Chat, snake_case for Claude
    /// Code, apply_patch for Codex) without bloating one YAML.
    pub fn load_with_format(name: &str, agent_format: Option<&str>) -> anyhow::Result<Self> {
        // Try several locations so the same scenario file works whether
        // launched from the project root or an empty benchmark directory.
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(fmt) = agent_format {
            for sub in ["scenarios", "../scenarios", "../../scenarios"] {
                candidates.push(std::path::PathBuf::from(format!("{sub}/{name}.{fmt}.yaml")));
            }
        }
        for sub in ["scenarios", "../scenarios", "../../scenarios"] {
            candidates.push(std::path::PathBuf::from(format!("{sub}/{name}.yaml")));
        }
        // Also try the binary's own directory (works for installed copies
        // where the binary lives next to scenarios/).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                if let Some(fmt) = agent_format {
                    for sub in ["scenarios", "../scenarios", "../../scenarios"] {
                        candidates.push(dir.join(sub).join(format!("{name}.{fmt}.yaml")));
                    }
                }
                for sub in ["scenarios", "../scenarios", "../../scenarios"] {
                    candidates.push(dir.join(sub).join(format!("{name}.yaml")));
                }
            }
        }
        for path in &candidates {
            if path.is_file() {
                return Self::from_yaml(path.to_str().unwrap_or(""));
            }
        }
        anyhow::bail!("scenario '{name}' not found in any of: {candidates:?}")
    }

    /// List all scenario file stems (base + per-agent variants). For
    /// suite discovery, prefer [`Scenario::list_base`] which strips the
    /// per-agent suffixes.
    pub fn list() -> anyhow::Result<Vec<String>> {
        let mut scenarios = Vec::new();
        let dir = std::path::Path::new("scenarios");
        if !dir.exists() { return Ok(scenarios); }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    scenarios.push(name.to_string());
                }
            }
        }
        scenarios.sort();
        Ok(scenarios)
    }

    /// List only base scenario names (no per-agent variants). A base
    /// scenario is one whose file stem has no `.` (e.g., `hidden_bug`
    /// is base; `hidden_bug.claude_code` and `hidden_bug.codex` are
    /// per-agent variants and excluded). The suite runner iterates
    /// over base scenarios and lets [`Scenario::load_with_format`]
    /// pick the right variant for the active `--agent-format`.
    pub fn list_base() -> anyhow::Result<Vec<String>> {
        Ok(Self::list()?
            .into_iter()
            .filter(|name| !name.contains('.'))
            .collect())
    }
}

// ── State machine engine ───────────────────────────────────────────

#[derive(Debug)]
pub struct StateMachine {
    scenario: Scenario,
    current_state: String,
    pub events: Vec<AgentEvent>,
    /// How many times each state has been entered. Used to detect stuck
    /// loops: if any state is visited more than `max_state_visits`, the
    /// mock forces a transition to `give_up` (or whatever terminal state
    /// the scenario defines) to break the cycle and stop log spam.
    state_visits: std::collections::HashMap<String, usize>,
}

/// Default cap on how many times any single state may be re-entered
/// before the mock forces a terminal transition. Keeps the loop bounded
/// when the agent can't make progress (e.g., keeps reading instead of
/// editing, or plugin noise confuses event extraction).
pub const DEFAULT_MAX_STATE_VISITS: usize = 5;

impl StateMachine {
    pub fn new(scenario: Scenario) -> Self {
        let start = scenario.start.clone();
        let mut visits = std::collections::HashMap::new();
        visits.insert(start.clone(), 1);
        Self {
            scenario,
            current_state: start,
            events: Vec::new(),
            state_visits: visits,
        }
    }

    pub fn weights(&self) -> ScenarioWeights {
        self.scenario.weights.clone().unwrap_or(ScenarioWeights {
            exploration: default_exploration(),
            reuse: default_reuse(),
            verification: default_verification(),
            efficiency: default_efficiency(),
            correctness: default_correctness(),
            tool_strategy: default_tool_strategy(),
        })
    }

    pub fn set_state(&mut self, name: &str) {
        *self.state_visits.entry(name.to_string()).or_insert(0) += 1;
        self.current_state = name.to_string();
    }

    /// Number of times `state` has been entered (including the initial
    /// entry for the start state).
    pub fn visit_count(&self, state: &str) -> usize {
        self.state_visits.get(state).copied().unwrap_or(0)
    }

    /// True when the current state has been entered more than `max`
    /// times. The mock uses this to force a terminal transition and
    /// stop a stuck loop from spamming logs.
    pub fn is_stuck(&self, max: usize) -> bool {
        self.visit_count(&self.current_state) > max
    }

    pub fn current_state_name(&self) -> &str {
        &self.current_state
    }

    pub fn scenario_name(&self) -> &str {
        &self.scenario.name
    }

    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    pub fn current_prompt(&self) -> Option<&str> {
        self.scenario.states.get(&self.current_state).map(|s| s.assistant.as_str())
    }

    pub fn current_tool_calls(&self) -> Vec<&str> {
        self.scenario.states.get(&self.current_state)
            .map(|s| s.tool_calls.iter().map(|t| t.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn current_tool_args(&self, tool: &str) -> Option<serde_json::Value> {
        self.scenario.states.get(&self.current_state)
            .and_then(|s| s.args.get(tool).cloned())
    }

    /// Like `current_tool_args` but selects args by agent format. If the
    /// state has an `args_per_agent[format][tool]` entry, that wins
    /// (full replacement, no field-level merge). Otherwise falls back to
    /// the default `args[tool]`. Returns `None` if neither is set.
    pub fn current_tool_args_for(&self, tool: &str, agent_format: &str) -> Option<serde_json::Value> {
        let state = self.scenario.states.get(&self.current_state)?;
        if let Some(per_agent) = state.args_per_agent.get(agent_format) {
            if let Some(args) = per_agent.get(tool) {
                return Some(args.clone());
            }
        }
        state.args.get(tool).cloned()
    }

    pub fn is_terminal(&self) -> bool {
        self.scenario.states.get(&self.current_state)
            .map(|s| s.finish || s.fail)
            .unwrap_or(false)
    }

    pub fn is_success(&self) -> bool {
        self.scenario.states.get(&self.current_state)
            .map(|s| s.finish)
            .unwrap_or(false)
    }

    /// Feed a parsed agent action and advance the state machine.
    /// Returns true if a transition was taken.
    pub fn feed(&mut self, event: AgentEvent) -> bool {
        self.events.push(event.clone());

        let state = match self.scenario.states.get(&self.current_state) {
            Some(s) => s,
            None => return false,
        };

        for transition in &state.transitions {
            if matches(&event, transition) {
                let next = transition.next_state().to_string();
                self.set_state(&next);
                return true;
            }
        }
        false
    }
}

fn matches(event: &AgentEvent, transition: &Transition) -> bool {
    match (event, transition) {
        (AgentEvent::Search(s, _), Transition::OnTool { tool, .. }) => tool.is_empty() || s.contains(tool),
        (AgentEvent::Read(p, _), Transition::OnTool { tool, .. }) => tool.is_empty() || p.contains(tool),
        (AgentEvent::Edit(p, _), Transition::OnTool { tool, .. }) => tool.is_empty() || p.contains(tool),
        (AgentEvent::Test(t, _), Transition::OnTool { tool, .. }) => tool.is_empty() || t.contains(tool),
        (AgentEvent::Task(t, _), Transition::OnTool { tool, .. }) => tool.is_empty() || t.contains(tool),
        (AgentEvent::Ask(a, _), Transition::OnTool { tool, .. }) => tool.is_empty() || a.contains(tool),
        (AgentEvent::Write(w, _), Transition::OnTool { tool, .. }) => tool.is_empty() || w.contains(tool),
        (AgentEvent::Other(o, _), Transition::OnTool { tool, .. }) => tool.is_empty() || o.contains(tool),
        (AgentEvent::Read(p, _), Transition::OnRead { pattern, .. }) => p.contains(pattern),
        (AgentEvent::Edit(p, _), Transition::OnEdit { pattern, .. }) => p.contains(pattern),
        (AgentEvent::Edit(p, r), Transition::OnEditResult { pattern, result_contains, .. }) => p.contains(pattern) && r.contains(result_contains),
        (AgentEvent::Test(t, _), Transition::OnTest { result, .. }) => t.contains(result),
        (AgentEvent::Test(_, r), Transition::OnTestResult { result_contains, .. }) => r.contains(result_contains),
        (AgentEvent::Task(_, _), Transition::OnTask { .. }) => true,
        (AgentEvent::Done, Transition::OnDone { .. }) => true,
        _ => false,
    }
}

impl Transition {
    pub fn next_state(&self) -> &str {
        match self {
            Transition::OnTool { next, .. } => next,
            Transition::OnRead { next, .. } => next,
            Transition::OnEdit { next, .. } => next,
            Transition::OnEditResult { next, .. } => next,
            Transition::OnTest { next, .. } => next,
            Transition::OnTestResult { next, .. } => next,
            Transition::OnFileContains { next, .. } => next,
            Transition::OnTask { next, .. } => next,
            Transition::OnDone { next, .. } => next,
        }
    }
}

// ── Scorer ─────────────────────────────────────────────────────────

/// All six score dimensions are scored on a 0-20 scale. Higher is better
/// for every dimension — the radar chart is symmetric. Composite is the
/// weighted sum of the six dimensions using [`COMPOSITE_WEIGHTS`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResult {
    pub reuse: f64,
    pub verification: f64,
    pub tool_strategy: f64,
    pub correctness: f64,
    pub search_efficiency: f64,
    pub context_efficiency: f64,
    pub total: f64,
}

/// Final composite weights. Hard metrics (correctness, verification) get
/// the most weight; diagnostic metrics (search/context efficiency) get
/// 10% each so they distinguish "excellent vs ordinary" agents without
/// overriding "did the work or not".
pub const COMPOSITE_WEIGHTS: &[(&str, f64)] = &[
    ("correctness", 0.30),
    ("verification", 0.20),
    ("tool_strategy", 0.20),
    ("reuse", 0.10),
    ("search_efficiency", 0.10),
    ("context_efficiency", 0.10),
];

// Causality windows for the new efficiency metrics. All are in events
// (not tool-class events), so they count any agent action toward the
// window. Small enough to be meaningful, large enough to tolerate
// plugin/system noise.
const SEARCH_READ_WINDOW: usize = 3;
const READ_ACTION_WINDOW: usize = 5;
const RE_READ_WINDOW: usize = 10;

// Targeting step function. `files_matched` → score. Branch boundaries
// at 3 / 10 / 30 (with the 30+ branch being a 0.2 "shotgun" score).
// `None` means parse failed → fall back to neutral 0.5 (NOT 0; don't
// punish missing observability — different agents emit grep results
// in different formats).
fn targeting_score(files_matched: Option<usize>) -> f64 {
    match files_matched {
        Some(n) if n <= 3 => 1.0,
        Some(n) if n <= 10 => 0.8,
        Some(n) if n <= 30 => 0.5,
        Some(_) => 0.2,
        None => 0.5,
    }
}

/// Parse a search-tool result string and return the number of distinct
/// file paths it mentions. Heuristic — works for ripgrep's
/// `path:line:content` format and similar. Returns `None` when no
/// path-like tokens can be extracted (different agent = different
/// output format), so callers can fall back to a neutral score
/// instead of penalizing the agent for the harness's parse miss.
fn parse_search_result_files(result: &str) -> Option<usize> {
    if result.is_empty() {
        return None;
    }
    let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut any_parsed = false;
    for line in result.lines() {
        // ripgrep default: "path/to/file.rs:42:fn foo()"
        // The first `:` separates path from the rest.
        if let Some((path, _rest)) = line.split_once(':') {
            // Path heuristic: contains `/` (typical) or has a file
            // extension. Filters out plain content lines.
            if path.contains('/') || path.contains('.') {
                files.insert(path.to_string());
                any_parsed = true;
            }
        }
    }
    if any_parsed { Some(files.len()) } else { None }
}

/// Compute Search Efficiency (0-20). Three sub-metrics:
/// - `search_to_read` (50%): fraction of Search events followed by a
///   Read within `SEARCH_READ_WINDOW` events. Measures "did the agent
///   act on its search results?"
/// - `search_targeting` (25%): step function on files matched per
///   search. Parse failure → 0.5 (neutral).
/// - `search_novelty` (25%): fraction of Search events whose result
///   file-set is unique. Compared by SET, not by query string — so
///   narrowing a search from "timeout" to "request_timeout" with a
///   different file set counts as novel exploration, not redundancy.
pub fn compute_search_efficiency(events: &[AgentEvent], expected: bool) -> f64 {
    let searches: Vec<(usize, &AgentEvent)> = events.iter().enumerate()
        .filter(|(_, e)| matches!(e, AgentEvent::Search(_, _)))
        .collect();

    if searches.is_empty() {
        return if expected { 0.0 } else { 20.0 };
    }

    // 1. search_to_read
    let to_read_count = searches.iter().filter(|(i, _)| {
        let start = i + 1;
        let end = (start + SEARCH_READ_WINDOW).min(events.len());
        events[start..end].iter().any(|e| matches!(e, AgentEvent::Read(_, _)))
    }).count();
    let search_to_read = to_read_count as f64 / searches.len() as f64;

    // 2. search_targeting
    let targeting_avg: f64 = searches.iter().map(|(_, e)| {
        if let AgentEvent::Search(_, result) = e {
            targeting_score(parse_search_result_files(result))
        } else { 0.5 }
    }).sum::<f64>() / searches.len() as f64;

    // 3. search_novelty
    let mut seen_sets: Vec<std::collections::HashSet<String>> = Vec::new();
    let mut unique_count = 0;
    for (_, e) in &searches {
        if let AgentEvent::Search(_, result) = e {
            let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
            for line in result.lines() {
                if let Some((path, _)) = line.split_once(':') {
                    if path.contains('/') || path.contains('.') {
                        set.insert(path.to_string());
                    }
                }
            }
            if !seen_sets.iter().any(|s| s == &set) {
                unique_count += 1;
                seen_sets.push(set);
            }
        }
    }
    let novelty = unique_count as f64 / searches.len() as f64;

    (search_to_read * 0.5 + targeting_avg * 0.25 + novelty * 0.25) * 20.0
}

/// Compute Context Efficiency (0-20). Three sub-metrics:
/// - `read_to_action` (50%): fraction of Read events followed by an
///   Edit or Test within `READ_ACTION_WINDOW` events. Measures "did
///   the agent consume the context it loaded?"
/// - `read_precision` (30%): `|files_read ∩ files_edited| / |files_read|`.
///   Measures "did the agent's exploration lead to a target?"
/// - `reread_penalty` (20%): fraction of reads that are *wasteful*
///   re-reads. A re-read is NOT penalized when:
///   1. It's within `RE_READ_WINDOW` events of the prior read of the
///      same file (short-window re-reads are common and fine), OR
///   2. There's an Edit or Test on the same file between the two
///      reads (justifies re-loading the new state).
pub fn compute_context_efficiency(events: &[AgentEvent], expected: bool) -> f64 {
    let reads: Vec<(usize, &AgentEvent)> = events.iter().enumerate()
        .filter(|(_, e)| matches!(e, AgentEvent::Read(_, _)))
        .collect();

    if reads.is_empty() {
        return if expected { 0.0 } else { 20.0 };
    }

    // 1. read_to_action
    let to_action_count = reads.iter().filter(|(i, _)| {
        let start = i + 1;
        let end = (start + READ_ACTION_WINDOW).min(events.len());
        events[start..end].iter().any(|e|
            matches!(e, AgentEvent::Edit(_, _) | AgentEvent::Test(_, _)))
    }).count();
    let read_to_action = to_action_count as f64 / reads.len() as f64;

    // 2. read_precision
    let files_read: std::collections::HashSet<&str> = reads.iter()
        .filter_map(|(_, e)| if let AgentEvent::Read(p, _) = e { Some(p.as_str()) } else { None })
        .collect();
    let files_edited: std::collections::HashSet<&str> = events.iter()
        .filter_map(|e| if let AgentEvent::Edit(p, _) = e { Some(p.as_str()) } else { None })
        .collect();
    let precision = if files_read.is_empty() {
        0.0
    } else {
        files_read.intersection(&files_edited).count() as f64 / files_read.len() as f64
    };

    // 3. reread_penalty
    let mut last_read_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut re_reads = 0usize;
    for (i, e) in events.iter().enumerate() {
        if let AgentEvent::Read(p, _) = e {
            if let Some(&last) = last_read_idx.get(p) {
                let within_window = (i - last) < RE_READ_WINDOW;
                let had_action = events[last + 1..i].iter().any(|ev| match ev {
                    AgentEvent::Edit(pp, _) | AgentEvent::Test(pp, _) => pp == p,
                    _ => false,
                });
                if !within_window && !had_action {
                    re_reads += 1;
                }
            }
            last_read_idx.insert(p.clone(), i);
        }
    }
    let reread_penalty = re_reads as f64 / reads.len() as f64;

    (read_to_action * 0.5 + precision * 0.3 + (1.0 - reread_penalty) * 0.2) * 20.0
}

pub fn score_run(run: &ScenarioRun) -> ScoreResult {
    let events = &run.events;
    let is_success = run.terminal_state.as_deref() == Some("verify");

    let edit_count = events.iter().filter(|e| matches!(e, AgentEvent::Edit(_, _))).count();
    let test_count = events.iter().filter(|e| matches!(e, AgentEvent::Test(_, _))).count();
    let search_count = events.iter().filter(|e| matches!(e, AgentEvent::Search(_, _))).count();
    let read_count = events.iter().filter(|e| matches!(e, AgentEvent::Read(_, _))).count();

    let first_edit = events.iter().position(|e| matches!(e, AgentEvent::Edit(_, _))).unwrap_or(usize::MAX);
    let last_edit = events.iter().rposition(|e| matches!(e, AgentEvent::Edit(_, _))).unwrap_or(0);

    let searches_before = events[..first_edit.min(events.len())].iter()
        .filter(|e| matches!(e, AgentEvent::Search(_, _))).count();
    let reads_before_set: std::collections::HashSet<&str> = events[..first_edit.min(events.len())].iter()
        .filter_map(|e| if let AgentEvent::Read(p, _) = e { Some(p.as_str()) } else { None })
        .collect();
    let unique_tests_after: std::collections::HashSet<&str> = events[last_edit..].iter()
        .filter_map(|e| if let AgentEvent::Test(t, _) = e { Some(t.as_str()) } else { None })
        .collect();

    // ── Reuse (20): Search→Read(10) + ReadResultUse(10) ──
    // Did the agent load files that informed its decisions, and did
    // it use the search results it got back?
    let discovery_reuse: f64 = {
        let read_after_search = events.iter()
            .scan(false, |searched, e| {
                if let AgentEvent::Search(_, _) = e { *searched = true; }
                Some(matches!(e, AgentEvent::Read(_, _)) && *searched)
            })
            .filter(|&b| b).count();
        (read_after_search as f64 * 2.5).min(10.0)
    };
    let result_reuse: f64 = {
        if search_count > 0 && read_count >= search_count { 10.0 }
        else if search_count > 0 && read_count > 0 { 5.0 }
        else if read_count > 0 { 3.0 }
        else { 0.0 }
    };
    let reuse: f64 = (discovery_reuse + result_reuse).clamp(0.0, 20.0);

    // ── Verification (20): Attempt(5) + Relevant(10) + Recovery(5) ──
    let val_attempt: f64 = {
        if test_count > 0 { 5.0 } else { 0.0 }
    };
    let relevant_val: f64 = {
        if unique_tests_after.is_empty() { 0.0 }
        else if read_count > 0 { 10.0 }
        else { 5.0 }
    };
    let failure_recovery: f64 = {
        if test_count >= 3 { 5.0 }
        else if test_count >= 2 { 3.0 }
        else if test_count >= 1 { 1.0 }
        else { 0.0 }
    };
    let verification: f64 = (val_attempt + relevant_val + failure_recovery).clamp(0.0, 20.0);

    // ── Correctness (20): Goal(15) + RootCause(2) + Quality(3) ──
    let goal_completion: f64 = {
        if is_success { 15.0 } else if run.terminal_state.is_some() { 5.0 } else { 0.0 }
    };
    let root_cause_fixed: f64 = {
        if searches_before >= 2 && reads_before_set.len() >= 2 { 2.0 }
        else if searches_before >= 1 { 1.0 }
        else { 0.0 }
    };
    let solution_quality: f64 = {
        if is_success && searches_before >= 1 && !reads_before_set.is_empty() { 3.0 }
        else if is_success { 1.0 }
        else { 0.0 }
    };
    let correctness: f64 = (goal_completion + root_cause_fixed + solution_quality).clamp(0.0, 20.0);

    // ── Tool Strategy (20): Selection(10) + Sequencing(10) ──
    let tool_selection: f64 = {
        let has_search = search_count > 0;
        let has_read = read_count > 0;
        let has_test = test_count > 0;
        if has_search && has_read && has_test { 10.0 }
        else if has_search && has_read { 7.0 }
        else if has_search { 5.0 }
        else if has_read { 3.0 }
        else { 1.0 }
    };
    let tool_sequencing: f64 = {
        if edit_count == 0 {
            10.0
        } else {
            let search_before = searches_before > 0;
            let read_before = !reads_before_set.is_empty();
            let test_after = events[last_edit..].iter().any(|e| matches!(e, AgentEvent::Test(_, _)));
            if search_before && read_before && test_after { 10.0 }
            else if search_before && test_after { 7.0 }
            else if search_before && read_before { 5.0 }
            else if search_before { 3.0 }
            else { 1.0 }
        }
    };
    let tool_strategy: f64 = (tool_selection + tool_sequencing).clamp(0.0, 20.0);

    // ── Search Efficiency + Context Efficiency (diagnostic) ──
    let search_efficiency = compute_search_efficiency(events, run.expected_search);
    let context_efficiency = compute_context_efficiency(events, run.expected_read);

    let total: f64 = (correctness * COMPOSITE_WEIGHTS[0].1
        + verification * COMPOSITE_WEIGHTS[1].1
        + tool_strategy * COMPOSITE_WEIGHTS[2].1
        + reuse * COMPOSITE_WEIGHTS[3].1
        + search_efficiency * COMPOSITE_WEIGHTS[4].1
        + context_efficiency * COMPOSITE_WEIGHTS[5].1)
        .clamp(0.0, 20.0);

    ScoreResult {
        reuse,
        verification,
        tool_strategy,
        correctness,
        search_efficiency,
        context_efficiency,
        total,
    }
}

// ── 2D Agent Profile ──────────────────────────────────────────────

/// Returns (prep_score, verify_score) on a 0-10 scale for 2D profiling.
pub fn agent_profile(events: &[AgentEvent]) -> (f64, f64) {
    let first_edit = events.iter().position(|e| matches!(e, AgentEvent::Edit(_, _))).unwrap_or(usize::MAX);
    let last_edit = events.iter().rposition(|e| matches!(e, AgentEvent::Edit(_, _))).unwrap_or(0);

    let searches_before = events[..first_edit.min(events.len())].iter()
        .filter(|e| matches!(e, AgentEvent::Search(_, _))).count();
    let reads_before: std::collections::HashSet<&str> = events[..first_edit.min(events.len())].iter()
        .filter_map(|e| if let AgentEvent::Read(p, _) = e { Some(p.as_str()) } else { None })
        .collect();

    let tests_after = events[last_edit..].iter()
        .filter(|e| matches!(e, AgentEvent::Test(_, _))).count();
    let unique_reads: std::collections::HashSet<&str> = events.iter()
        .filter_map(|e| if let AgentEvent::Read(p, _) = e { Some(p.as_str()) } else { None })
        .collect();

    let prep: f64 = ((searches_before as f64 * 2.5).min(5.0) + (reads_before.len() as f64 * 2.5).min(5.0)).min(10.0);
    let verify: f64 = ((tests_after as f64 * 3.0).min(6.0) + (unique_reads.len() as f64 * 1.5).min(4.0)).min(10.0);

    (prep, verify)
}

// ── Event extraction from Chat Completions request ─────────────────

/// Parse a Chat Completions request body into AgentEvents.
/// Also pairs tool_calls with their tool_results from the request.
pub fn extract_events_from_request(body: &serde_json::Value) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    let msgs = match body["messages"].as_array() {
        Some(a) => a,
        None => return events,
    };

    // Build a map of tool_call_id → tool result content
    let mut result_map: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for msg in msgs.iter() {
        if msg["role"].as_str() == Some("tool") {
            let id = msg["tool_call_id"].as_str().unwrap_or("");
            let content = msg["content"].as_str().unwrap_or("");
            if !id.is_empty() {
                result_map.insert(id, content);
            }
        }
    }

    // Find the last assistant message with tool_calls
    let last_asst = msgs.iter().rposition(|m| {
        m["role"].as_str() == Some("assistant")
            && m["tool_calls"].is_array()
            && !m["tool_calls"].as_array().map_or(true, |a| a.is_empty())
    });

    let start_idx = match last_asst {
        Some(i) => i,
        None => {
            let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap_or("?")).collect();
            return events;
        }
    };

    let last = &msgs[start_idx];
    if let Some(tcs) = last["tool_calls"].as_array() {
        let names: Vec<&str> = tcs.iter().map(|tc| tc["function"]["name"].as_str().unwrap_or("?")).collect();
        eprintln!("[aces] last assistant tool_calls: {:?}", names);
    }

    // Process the assistant message and following tool results
    for msg in msgs[start_idx..].iter() {
        let role = match msg["role"].as_str() {
            Some(r) => r,
            None => continue,
        };

        match role {
            "assistant" => {
                if let Some(tcs) = msg["tool_calls"].as_array() {
                    for tc in tcs {
                        let name = tc["function"]["name"].as_str().unwrap_or("");
                        let args = tc["function"]["arguments"].as_str().unwrap_or("");
                        let id = tc["id"].as_str().unwrap_or("");
                        let result = result_map.get(id).copied().unwrap_or("");
                        match name {
                            "grep" | "Grep" | "rg" | "search" | "search_file" | "find_in_files" | "glob" | "Glob" => {
                                events.push(AgentEvent::Search(args.to_string(), result.to_string()));
                            }
                            "read" | "Read" | "read_file" | "view" | "cat" => {
                                events.push(AgentEvent::Read(extract_path(args), result.to_string()));
                            }
                            "edit" | "Edit" | "edit_file" | "replace" | "write" | "Write" | "write_to_file" | "apply_patch" => {
                                events.push(AgentEvent::Edit(extract_path(args), result.to_string()));
                            }
                            "test" | "cargo_test" | "run_test" | "pytest" | "npm_test" => {
                                events.push(AgentEvent::Test(args.to_string(), result.to_string()));
                            }
                            "bash" | "Bash" | "shell" | "execute_command" | "command" | "run" => {
                                if let Some(cat_path) = extract_cat_path(args) {
                                    events.push(AgentEvent::Read(cat_path, result.to_string()));
                                } else if args.contains("test") || args.contains("check") {
                                    events.push(AgentEvent::Test(args.to_string(), result.to_string()));
                                } else if args.contains("grep") || args.contains("rg") || args.contains("search") {
                                    events.push(AgentEvent::Search(args.to_string(), result.to_string()));
                                } else {
                                    events.push(AgentEvent::Task(args.to_string(), result.to_string()));
                                }
                            }
                            "task" | "subagent" | "agent" | "Agent" | "delegate" => {
                                events.push(AgentEvent::Task(args.to_string(), result.to_string()));
                            }
                            "ask" | "question" | "ask_user" | "ask_followup_question" | "AskUserQuestion" => {
                                events.push(AgentEvent::Ask(args.to_string(), result.to_string()));
                            }
                            _ => {
                                events.push(AgentEvent::Other(format!("{name}({args})"), result.to_string()));
                            }
                        }
                    }
                }
            }
            "tool" => {}
            "user" => {
                let content = msg["content"].as_str().unwrap_or("");
                if content == "Done" || content == "done" || content.contains("task complete") {
                    events.push(AgentEvent::Done);
                }
            }
            _ => {}
        }
    }
    events
}

fn extract_path(args: &str) -> String {
    // Try to parse JSON args and extract path/filePath/file_path
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(args) {
        // Codex apply_patch nests the path inside operation
        if let Some(p) = v["operation"]["path"].as_str() { return p.to_string(); }
        if let Some(p) = v["path"].as_str() { return p.to_string(); }
        if let Some(p) = v["file_path"].as_str() { return p.to_string(); }
        if let Some(p) = v["filePath"].as_str() { return p.to_string(); }
        if let Some(p) = v["pattern"].as_str() { return p.to_string(); }
        if let Some(p) = v["command"].as_str() { return p.to_string(); }
    }
    args.to_string()
}

/// If `args` describe a `cat <path>` shell command, return the path.
/// Codex's native read pattern is `shell` with `cat`; we want the mock
/// to treat that as a Read event so the scenario's `on_read` transitions
/// fire (matching on the path, not the whole command string).
pub(crate) fn extract_cat_path(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    let cmd = v["command"].as_str()?.trim();
    let mut parts = cmd.split_whitespace();
    if parts.next()? != "cat" {
        return None;
    }
    let candidate = parts.next()?;
    if candidate.starts_with('-') {
        return None;
    }
    Some(candidate.to_string())
}
