//! Conservative automatic production of typed execution facts.
//!
//! Facts are derived only from exact tool outputs persisted in the Ground Truth
//! ledger. The producer deliberately avoids free-form LLM extraction: it emits
//! facts only when the tool contract or a machine-readable result gives us
//! something deterministic to say.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::db::Database;
use crate::file_observation::{FileObservation, observe_file};
use crate::ground_truth::{
    ExecutionFact, FactStatus, PlanFactDependency, ResourceRef, TruthStream,
};
use crate::ground_truth_store::GroundTruthStore;
use crate::tool_cache::{ToolKind, extract_dependent_files};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutoFactReport {
    pub facts_created: usize,
    pub dependencies_created: usize,
    pub file_observations_created: usize,
    pub facts_invalidated: usize,
    pub validation_steps_completed: usize,
    pub validation_steps_blocked: usize,
}

/// Project deterministic typed facts from newly-arrived Responses tool outputs.
///
/// `history` is the native Responses history after local continuity expansion,
/// so a tool output can resolve the matching earlier `function_call` /
/// `custom_tool_call` by call id. `current_items` contains only newly-arrived
/// client input, preventing old history from being projected repeatedly.
pub fn produce_from_responses_items(
    db: &Database,
    session_id: &str,
    conv_id: i64,
    history: &[Value],
    current_items: &[Value],
) -> anyhow::Result<AutoFactReport> {
    let store = GroundTruthStore::new(db);
    let mut report = AutoFactReport::default();

    let mut existing_facts: HashMap<String, ExecutionFact> = store
        .load_execution_facts(session_id)?
        .into_iter()
        .map(|fact| (fact.id.clone(), fact))
        .collect();

    let active_plan = db.get_active_plan(conv_id)?;
    let (plan_id, goal, current_step, current_step_index) = match active_plan {
        Some((id, goal, pending, completed, _assumptions)) => {
            let pending: Vec<String> = serde_json::from_value(pending).unwrap_or_default();
            let completed: Vec<String> = serde_json::from_value(completed).unwrap_or_default();
            (
                Some(id),
                goal,
                pending.first().cloned().unwrap_or_default(),
                completed.len(),
            )
        }
        None => (None, String::new(), String::new(), 0),
    };
    let plan_text = format!("{goal}\n{current_step}");
    let mut existing_deps: HashSet<(i64, Option<usize>, String, bool)> = match plan_id {
        Some(id) => store
            .load_plan_dependencies(session_id, id)?
            .into_iter()
            .map(|dep| (dep.plan_id, dep.step_index, dep.fact_id, dep.required))
            .collect(),
        None => HashSet::new(),
    };

    for item in current_items {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        if !matches!(
            item_type,
            "function_call_output" | "custom_tool_call_output"
        ) {
            continue;
        }
        let Some(call_id) = item
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let Some(call) = find_call(history, current_items, call_id) else {
            continue;
        };
        let name = call.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let args = call_arguments(call);
        let output = output_text(item.get("output"));
        if output.is_empty() {
            continue;
        }

        let source = store.put_json(
            TruthStream::ToolPayloads,
            session_id,
            item,
            if item_type.is_empty() {
                "tool-output"
            } else {
                item_type
            },
        )?;

        let kind = ToolKind::from_name(name);
        let paths = extract_dependent_files(name, &args);

        if kind == ToolKind::ReadFile && paths.len() == 1 {
            let path = paths[0].clone();
            let obs = observe_file(&path, &output);
            let obs_source = store.put_json(
                TruthStream::FileObservations,
                session_id,
                &obs,
                "auto-read-file-observation",
            )?;
            db.store_file_observation(&obs)?;
            report.file_observations_created += 1;

            let current_resource = ResourceRef::File {
                path: obs.path.clone(),
                content_hash: obs.content_hash.clone(),
            };
            for fact in existing_facts.values_mut() {
                if fact.invalidate_if_changed(&current_resource) {
                    store.put_execution_fact(session_id, fact)?;
                    report.facts_invalidated += 1;
                }
            }

            let mut fact = ExecutionFact::new(
                format!("file:{}", obs.path),
                format!(
                    "Observed {} at content version {} ({} lines, {} bytes).",
                    obs.path, obs.content_hash, obs.line_count, obs.size_bytes
                ),
            );
            fact.evidence.extend([source.clone(), obs_source.clone()]);
            fact.resources.push(current_resource);
            persist_fact_if_changed(
                &store,
                session_id,
                &mut existing_facts,
                fact.clone(),
                &mut report,
            )?;

            if let Some(id) = plan_id {
                let required = exact_path_mentioned(&current_step, &obs.path);
                let relevant = required || path_mentioned(&plan_text, &obs.path);
                if relevant {
                    persist_dependency_if_new(
                        &store,
                        session_id,
                        &mut existing_deps,
                        PlanFactDependency {
                            plan_id: id,
                            step_index: Some(current_step_index),
                            fact_id: fact.id.clone(),
                            required,
                        },
                        &mut report,
                    )?;
                }

                for symbol in plan_relevant_symbols(&obs, &plan_text).into_iter().take(32) {
                    let mut symbol_fact = ExecutionFact::new(
                        format!("symbol:{}::{symbol}", obs.path),
                        format!(
                            "Symbol {symbol} is present in {} at content version {}.",
                            obs.path, obs.content_hash
                        ),
                    );
                    symbol_fact.evidence.push(obs_source.clone());
                    symbol_fact.resources.push(ResourceRef::Symbol {
                        file_path: obs.path.clone(),
                        symbol: symbol.clone(),
                        content_hash: obs.content_hash.clone(),
                    });
                    persist_fact_if_changed(
                        &store,
                        session_id,
                        &mut existing_facts,
                        symbol_fact.clone(),
                        &mut report,
                    )?;
                    persist_dependency_if_new(
                        &store,
                        session_id,
                        &mut existing_deps,
                        PlanFactDependency {
                            plan_id: id,
                            step_index: Some(current_step_index),
                            fact_id: symbol_fact.id,
                            required: current_step.contains(symbol.as_str()),
                        },
                        &mut report,
                    )?;
                }
            }
            continue;
        }

        if kind == ToolKind::Bash {
            if let Some(exit_code) = parse_exit_code(&output) {
                let command = command_text(&args);
                let display = command_preview(&args);
                let fact_id = validation_fact_id(&command);
                let mut fact = ExecutionFact::new(
                    fact_id.clone(),
                    if exit_code == 0 {
                        format!("Validation command {display} succeeded with exit code 0.")
                    } else {
                        format!("Validation command {display} failed with exit code {exit_code}.")
                    },
                );
                if exit_code != 0 {
                    fact.status = FactStatus::Unknown;
                }
                fact.evidence.push(source.clone());
                fact.resources.push(ResourceRef::ToolResult {
                    call_id: call_id.to_string(),
                    content_hash: source.content_hash.clone(),
                });
                persist_fact_if_changed(
                    &store,
                    session_id,
                    &mut existing_facts,
                    fact,
                    &mut report,
                )?;

                if let Some(id) = plan_id
                    && validation_step_matches_command(&current_step, &command)
                {
                    persist_dependency_if_new(
                        &store,
                        session_id,
                        &mut existing_deps,
                        PlanFactDependency {
                            plan_id: id,
                            step_index: Some(current_step_index),
                            fact_id,
                            required: true,
                        },
                        &mut report,
                    )?;
                    if evolve_validation_step(db, id, &current_step, exit_code == 0)? {
                        if exit_code == 0 {
                            report.validation_steps_completed += 1;
                        } else {
                            report.validation_steps_blocked += 1;
                        }
                    }
                }
            }
            continue;
        }

        if matches!(
            kind,
            ToolKind::Grep | ToolKind::ListFiles | ToolKind::SymbolSearch | ToolKind::Diagnostics
        ) && !paths.is_empty()
        {
            let mut fact = ExecutionFact::new(
                format!("tool-result:{call_id}"),
                format!(
                    "Tool {name} returned exact evidence for {}.",
                    paths.join(", ")
                ),
            );
            fact.evidence.push(source.clone());
            fact.resources.push(ResourceRef::ToolResult {
                call_id: call_id.to_string(),
                content_hash: source.content_hash.clone(),
            });
            persist_fact_if_changed(
                &store,
                session_id,
                &mut existing_facts,
                fact.clone(),
                &mut report,
            )?;

            if let Some(id) = plan_id
                && paths.iter().any(|path| path_mentioned(&plan_text, path))
            {
                persist_dependency_if_new(
                    &store,
                    session_id,
                    &mut existing_deps,
                    PlanFactDependency {
                        plan_id: id,
                        step_index: Some(current_step_index),
                        fact_id: fact.id,
                        required: false,
                    },
                    &mut report,
                )?;
            }
        }
    }

    Ok(report)
}

fn persist_fact_if_changed(
    store: &GroundTruthStore<'_>,
    session_id: &str,
    existing: &mut HashMap<String, ExecutionFact>,
    fact: ExecutionFact,
    report: &mut AutoFactReport,
) -> anyhow::Result<()> {
    if existing.get(&fact.id) == Some(&fact) {
        return Ok(());
    }
    store.put_execution_fact(session_id, &fact)?;
    existing.insert(fact.id.clone(), fact);
    report.facts_created += 1;
    Ok(())
}

fn persist_dependency_if_new(
    store: &GroundTruthStore<'_>,
    session_id: &str,
    existing: &mut HashSet<(i64, Option<usize>, String, bool)>,
    dep: PlanFactDependency,
    report: &mut AutoFactReport,
) -> anyhow::Result<()> {
    let key = (
        dep.plan_id,
        dep.step_index,
        dep.fact_id.clone(),
        dep.required,
    );
    if !existing.insert(key) {
        return Ok(());
    }
    store.put_plan_dependency(session_id, &dep)?;
    report.dependencies_created += 1;
    Ok(())
}

fn find_call<'a>(history: &'a [Value], current: &'a [Value], call_id: &str) -> Option<&'a Value> {
    current
        .iter()
        .rev()
        .chain(history.iter().rev())
        .find(|item| {
            matches!(
                item.get("type").and_then(Value::as_str),
                Some("function_call") | Some("custom_tool_call")
            ) && item
                .get("call_id")
                .and_then(Value::as_str)
                .or_else(|| item.get("id").and_then(Value::as_str))
                == Some(call_id)
        })
}

fn call_arguments(call: &Value) -> String {
    let value = call.get("arguments").or_else(|| call.get("input"));
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_default(),
        None => String::new(),
    }
}

fn output_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
        _ => String::new(),
    }
}

fn parse_exit_code(text: &str) -> Option<i32> {
    const MARKERS: &[&str] = &[
        "Process exited with code ",
        "process exited with code ",
        "exit code: ",
        "exit_code=",
        "\"exit_code\":",
    ];
    for marker in MARKERS {
        let Some(pos) = text.find(marker) else {
            continue;
        };
        let tail = text[pos + marker.len()..].trim_start();
        let token: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if !token.is_empty() {
            if let Ok(code) = token.parse() {
                return Some(code);
            }
        }
    }
    None
}

fn command_text(args: &str) -> String {
    let parsed = serde_json::from_str::<Value>(args).ok();
    parsed
        .as_ref()
        .and_then(|value| value.get("command").or_else(|| value.get("cmd")))
        .and_then(Value::as_str)
        .unwrap_or(args)
        .trim()
        .to_string()
}

fn command_preview(args: &str) -> String {
    let command = command_text(args);
    let preview: String = command.chars().take(120).collect();
    if preview.is_empty() {
        "<unknown>".into()
    } else {
        format!("`{preview}`")
    }
}

fn validation_fact_id(command: &str) -> String {
    let normalized = command
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let digest = Sha256::digest(normalized.as_bytes());
    format!("validation:{}", hex::encode(&digest[..8]))
}

fn validation_step_matches_command(step: &str, command: &str) -> bool {
    if step.trim().is_empty() || command.trim().is_empty() {
        return false;
    }
    let step_lower = step.to_lowercase();
    let command_lower = command.to_lowercase();
    const MUTATION_TERMS: &[&str] = &[
        "fix ",
        "edit ",
        "implement ",
        "change ",
        "update ",
        "write ",
        "create ",
        "refactor ",
    ];
    if MUTATION_TERMS.iter().any(|term| step_lower.contains(term)) {
        return false;
    }
    const VALIDATION_TERMS: &[&str] = &[
        "test", "check", "build", "lint", "verify", "validate", "clippy", "pytest",
    ];
    if !VALIDATION_TERMS
        .iter()
        .any(|term| step_lower.contains(term))
    {
        return false;
    }
    if step_lower.contains(&command_lower) {
        return true;
    }
    let command_terms: Vec<&str> = command_lower
        .split(|c: char| c.is_whitespace() || matches!(c, ';' | '&' | '|'))
        .filter(|term| !term.is_empty() && !term.starts_with('-'))
        .take(2)
        .collect();
    !command_terms.is_empty() && command_terms.iter().all(|term| step_lower.contains(term))
}

fn evolve_validation_step(
    db: &Database,
    plan_id: i64,
    step: &str,
    success: bool,
) -> anyhow::Result<bool> {
    let conn = db.writer_conn();
    let tx = conn.unchecked_transaction()?;
    let row = tx.query_row(
        "SELECT pending_steps, completed_steps, blocked_steps FROM plan_states WHERE id = ?1 AND is_active = 1",
        rusqlite::params![plan_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        },
    );
    let (pending_json, completed_json, blocked_json) = match row {
        Ok(row) => row,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let mut pending: Vec<String> = serde_json::from_str(&pending_json).unwrap_or_default();
    let mut completed: Vec<String> = serde_json::from_str(&completed_json).unwrap_or_default();
    let mut blocked: Vec<String> = serde_json::from_str(&blocked_json).unwrap_or_default();
    if pending.first().map(String::as_str) != Some(step) {
        return Ok(false);
    }

    if success {
        pending.remove(0);
        if !completed.iter().any(|existing| existing == step) {
            completed.push(step.to_string());
        }
        blocked.retain(|existing| existing != step);
    } else if !blocked.iter().any(|existing| existing == step) {
        blocked.push(step.to_string());
    } else {
        return Ok(false);
    }

    tx.execute(
        "UPDATE plan_states SET pending_steps = ?1, completed_steps = ?2, blocked_steps = ?3, updated_at = datetime('now') WHERE id = ?4",
        rusqlite::params![
            serde_json::to_string(&pending)?,
            serde_json::to_string(&completed)?,
            serde_json::to_string(&blocked)?,
            plan_id,
        ],
    )?;
    tx.commit()?;
    Ok(true)
}

fn exact_path_mentioned(text: &str, path: &str) -> bool {
    if text.is_empty() || path.is_empty() {
        return false;
    }
    let text = text.replace('\\', "/");
    let path = path.replace('\\', "/");
    text.contains(&path)
}

fn path_mentioned(text: &str, path: &str) -> bool {
    if exact_path_mentioned(text, path) {
        return true;
    }
    let normalized = path.replace('\\', "/");
    let basename = normalized.rsplit('/').next().unwrap_or(&normalized);
    basename.len() >= 3 && text.contains(basename)
}

fn plan_relevant_symbols(obs: &FileObservation, plan_text: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    for function in &obs.ast.functions {
        if function.name.len() >= 2 && plan_text.contains(&function.name) {
            symbols.push(function.name.clone());
        }
    }
    for ty in &obs.ast.types {
        if ty.name.len() >= 2 && plan_text.contains(&ty.name) {
            symbols.push(ty.name.clone());
        }
    }
    symbols.sort();
    symbols.dedup();
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use crate::ground_truth_store::GroundTruthStore;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn parses_codex_exit_code_without_guessing_from_prose() {
        assert_eq!(
            parse_exit_code("Process exited with code 0\nFinal output:\nok"),
            Some(0)
        );
        assert_eq!(parse_exit_code("exit_code=101"), Some(101));
        assert_eq!(parse_exit_code("tests look successful"), None);
    }

    #[test]
    fn validation_step_matching_rejects_compound_work() {
        assert!(validation_step_matches_command(
            "run cargo test",
            "cargo test --lib"
        ));
        assert!(validation_step_matches_command(
            "check with cargo check",
            "cargo check"
        ));
        assert!(!validation_step_matches_command(
            "fix parser and run cargo test",
            "cargo test"
        ));
        assert!(!validation_step_matches_command(
            "implement parser",
            "cargo test"
        ));
    }

    #[tokio::test]
    async fn read_file_creates_versioned_fact_and_required_plan_dependency() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("auto-facts.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        let plan_id = db
            .store_plan_state(
                conv_id,
                "fix config loader",
                &["inspect src/config.rs ConfigLoader".into()],
                &[],
            )
            .unwrap();

        let call = json!({
            "type":"function_call",
            "call_id":"call_1",
            "name":"read_file",
            "arguments":"{\"path\":\"src/config.rs\"}"
        });
        let output = json!({
            "type":"function_call_output",
            "call_id":"call_1",
            "output":"pub struct ConfigLoader;\nfn load() {}"
        });
        let report = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call, output.clone()],
            std::slice::from_ref(&output),
        )
        .unwrap();

        assert!(report.facts_created >= 2);
        assert_eq!(report.file_observations_created, 1);
        let truth = GroundTruthStore::new(&db);
        let facts = truth.load_execution_facts("s1").unwrap();
        assert!(facts.iter().any(|fact| fact.id == "file:src/config.rs"));
        assert!(
            facts
                .iter()
                .any(|fact| fact.id == "symbol:src/config.rs::ConfigLoader")
        );
        let deps = truth.load_plan_dependencies("s1", plan_id).unwrap();
        assert!(deps.iter().any(|dep| dep.fact_id == "file:src/config.rs"
            && dep.required
            && dep.step_index == Some(0)));
    }

    #[tokio::test]
    async fn dependency_uses_absolute_current_step_index() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("step-index.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        let plan_id = db
            .store_plan_state(
                conv_id,
                "ship config",
                &["run cargo test".into(), "inspect src/config.rs".into()],
                &[],
            )
            .unwrap();

        let test_call = json!({"type":"function_call","call_id":"t1","name":"shell","arguments":"{\"command\":\"cargo test\"}"});
        let test_out = json!({"type":"function_call_output","call_id":"t1","output":"Process exited with code 0"});
        produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[test_call, test_out.clone()],
            std::slice::from_ref(&test_out),
        )
        .unwrap();

        let read_call = json!({"type":"function_call","call_id":"r1","name":"read_file","arguments":"{\"path\":\"src/config.rs\"}"});
        let read_out =
            json!({"type":"function_call_output","call_id":"r1","output":"pub struct Config;"});
        produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[read_call, read_out.clone()],
            std::slice::from_ref(&read_out),
        )
        .unwrap();

        let deps = GroundTruthStore::new(&db)
            .load_plan_dependencies("s1", plan_id)
            .unwrap();
        assert!(deps.iter().any(|dep| dep.fact_id == "file:src/config.rs"
            && dep.required
            && dep.step_index == Some(1)));
    }

    #[tokio::test]
    async fn changed_read_invalidates_old_symbol_version_before_replacing_it() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("auto-invalidate.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        db.store_plan_state(
            conv_id,
            "fix loader",
            &["inspect src/config.rs ConfigLoader".into()],
            &[],
        )
        .unwrap();

        let call = json!({"type":"function_call","call_id":"c1","name":"read_file","arguments":"{\"path\":\"src/config.rs\"}"});
        let out1 = json!({"type":"function_call_output","call_id":"c1","output":"pub struct ConfigLoader;"});
        produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call.clone(), out1.clone()],
            std::slice::from_ref(&out1),
        )
        .unwrap();

        let call2 = json!({"type":"function_call","call_id":"c2","name":"read_file","arguments":"{\"path\":\"src/config.rs\"}"});
        let out2 = json!({"type":"function_call_output","call_id":"c2","output":"pub struct ConfigLoader { pub enabled: bool }"});
        let report = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call, out1, call2, out2.clone()],
            std::slice::from_ref(&out2),
        )
        .unwrap();
        assert!(report.facts_invalidated >= 1);
        let truth = GroundTruthStore::new(&db);
        let facts = truth.load_execution_facts("s1").unwrap();
        let file_fact = facts
            .iter()
            .find(|fact| fact.id == "file:src/config.rs")
            .unwrap();
        assert_eq!(file_fact.status, FactStatus::Valid);
    }

    #[tokio::test]
    async fn shell_fact_requires_explicit_exit_code() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("shell-facts.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        let call = json!({"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"command\":\"cargo test\"}"});
        let vague =
            json!({"type":"function_call_output","call_id":"c1","output":"tests look successful"});
        let report = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call.clone(), vague.clone()],
            std::slice::from_ref(&vague),
        )
        .unwrap();
        assert_eq!(report.facts_created, 0);

        let exact = json!({"type":"function_call_output","call_id":"c1","output":"Process exited with code 0\nFinal output:\nok"});
        let report = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call, exact.clone()],
            std::slice::from_ref(&exact),
        )
        .unwrap();
        assert_eq!(report.facts_created, 1);
    }

    #[tokio::test]
    async fn failed_then_successful_validation_evolves_only_validation_step() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("plan-evolution.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        let plan_id = db
            .store_plan_state(
                conv_id,
                "ship parser",
                &["run cargo test".into(), "implement docs".into()],
                &[],
            )
            .unwrap();

        let call1 = json!({"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"command\":\"cargo test\"}"});
        let fail = json!({"type":"function_call_output","call_id":"c1","output":"Process exited with code 101"});
        let failed = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call1, fail.clone()],
            std::slice::from_ref(&fail),
        )
        .unwrap();
        assert_eq!(failed.validation_steps_blocked, 1);
        let facts = GroundTruthStore::new(&db)
            .load_execution_facts("s1")
            .unwrap();
        let validation = facts
            .iter()
            .find(|fact| fact.id.starts_with("validation:"))
            .unwrap();
        assert_eq!(validation.status, FactStatus::Unknown);
        let deps = GroundTruthStore::new(&db)
            .load_plan_dependencies("s1", plan_id)
            .unwrap();
        assert!(
            deps.iter()
                .any(|dep| dep.fact_id == validation.id && dep.required)
        );

        let call2 = json!({"type":"function_call","call_id":"c2","name":"shell","arguments":"{\"command\":\"cargo test\"}"});
        let pass = json!({"type":"function_call_output","call_id":"c2","output":"Process exited with code 0"});
        let passed = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call2, pass.clone()],
            std::slice::from_ref(&pass),
        )
        .unwrap();
        assert_eq!(passed.validation_steps_completed, 1);

        let active = db.get_active_plan(conv_id).unwrap().unwrap();
        let pending: Vec<String> = serde_json::from_value(active.2).unwrap();
        let completed: Vec<String> = serde_json::from_value(active.3).unwrap();
        assert_eq!(pending, vec!["implement docs"]);
        assert_eq!(completed, vec!["run cargo test"]);
        let facts = GroundTruthStore::new(&db)
            .load_execution_facts("s1")
            .unwrap();
        let validation = facts
            .iter()
            .find(|fact| fact.id.starts_with("validation:"))
            .unwrap();
        assert_eq!(validation.status, FactStatus::Valid);
    }

    #[tokio::test]
    async fn successful_test_does_not_complete_mutating_compound_step() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("plan-no-overreach.db"))
            .build()
            .await
            .unwrap();
        let conv_id = db
            .find_or_create_conversation("s1", "deepseek-flash")
            .unwrap();
        db.store_plan_state(
            conv_id,
            "fix parser",
            &["fix parser and run cargo test".into()],
            &[],
        )
        .unwrap();
        let call = json!({"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"command\":\"cargo test\"}"});
        let pass = json!({"type":"function_call_output","call_id":"c1","output":"Process exited with code 0"});
        let report = produce_from_responses_items(
            &db,
            "s1",
            conv_id,
            &[call, pass.clone()],
            std::slice::from_ref(&pass),
        )
        .unwrap();
        assert_eq!(report.validation_steps_completed, 0);
        let active = db.get_active_plan(conv_id).unwrap().unwrap();
        let pending: Vec<String> = serde_json::from_value(active.2).unwrap();
        assert_eq!(pending, vec!["fix parser and run cargo test"]);
    }
}
