//! Internal lossless projection pipeline for native Responses transport.
//!
//! Wire protocol stays Responses end-to-end. Native provider items are Ground
//! Truth; this module derives a working projection for DAG/context selection.
//! Unlike the historical Chat pipeline it does not truncate content before
//! creating level-0 leaves. Loss is allowed only after a recoverable source has
//! been established.

use serde_json::{json, Value};

use crate::compactor::CompactCommand;
use crate::dynamic_context::{assemble_dynamic_context, DynamicContextHints};
use crate::ground_truth::{FactStatus, PlanFactDependency, ResourceRef};
use crate::ground_truth_store::GroundTruthStore;
use crate::pipeline::render_dag_context;
use crate::typed_fact_producer::AutoFactReport;
use crate::AppState;

const CONTEXT_WINDOW: usize = 1_000_000;

#[derive(Debug, Clone, Default)]
pub struct NativeProjectionOutput {
    pub conv_id: Option<i64>,
    pub context: String,
    pub typed_plan_used: bool,
    pub auto_facts: AutoFactReport,
}

pub async fn project_and_assemble(
    state: &AppState,
    session_id: &str,
    model: &str,
    current_items: &[Value],
) -> NativeProjectionOutput {
    if state.no_pipeline {
        return NativeProjectionOutput::default();
    }

    let messages = project_messages(current_items);
    if messages.is_empty() {
        return NativeProjectionOutput::default();
    }

    let conv_id = match state
        .storage
        .db
        .find_or_create_conversation(session_id, model)
    {
        Ok(id) => id,
        Err(error) => {
            tracing::warn!(target:"deeplossless::responses_projection", %error,
                "failed to resolve native Responses conversation");
            return NativeProjectionOutput::default();
        }
    };

    let history = state
        .storage
        .session_store
        .get(session_id)
        .or_else(|| {
            state
                .storage
                .db
                .get_response_session(session_id)
                .ok()
                .flatten()
        })
        .unwrap_or_default();
    let auto_facts = match crate::typed_fact_producer::produce_from_responses_items(
        &state.storage.db,
        session_id,
        conv_id,
        &history,
        current_items,
    ) {
        Ok(report) => report,
        Err(error) => {
            tracing::warn!(target:"deeplossless::typed_facts", %session_id, %error,
                "automatic typed fact projection failed");
            AutoFactReport::default()
        }
    };

    let db = state.storage.db.clone();
    let dag = state.storage.dag.clone();
    let stored_messages = Value::Array(messages.clone());
    let ingest = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let message_ids = db.store_messages_with_ids(conv_id, &stored_messages)?;
        if let Some(items) = stored_messages.as_array() {
            for (message, message_id) in items.iter().zip(message_ids.iter().copied()) {
                let role = message.get("role").and_then(Value::as_str).unwrap_or("");
                if !matches!(role, "user" | "assistant" | "tool") {
                    continue;
                }
                let content = message
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if content.is_empty() {
                    continue;
                }
                let raw_tokens = crate::tokenizer::count(content) + dag.config().token_overhead;
                let token_count = crate::tokenizer::correct(
                    raw_tokens,
                    dag.config().token_correction_factor,
                ) as i64;
                dag.insert_message_leaf(conv_id, message_id, content, token_count)?;
            }
        }
        Ok(())
    })
    .await;

    match ingest {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(target:"deeplossless::responses_projection", %error,
                "native Responses projection ingestion failed");
        }
        Err(error) => {
            tracing::warn!(target:"deeplossless::responses_projection", %error,
                "native Responses projection worker failed");
        }
    }

    // Native Responses keeps exact level-0 projection data regardless of LCM mode,
    // but compaction is only useful when the LCM working view is actually consumed.
    if state.lcm_context
        && let Ok(mut compactor) = state.compactor.try_lock()
    {
        let _ = compactor
            .send_command(CompactCommand::ReviewAndCompact {
                conv_id,
                context_window: CONTEXT_WINDOW,
            })
            .await;
    }

    let query = messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str);

    let hints = build_dynamic_hints(state, session_id, conv_id);
    let budget = state.lcm_context_tokens.clamp(0, 8_000) as usize;
    let dag_context = if state.lcm_context && budget > 0 {
        match assemble_dynamic_context(&state.storage.dag, conv_id, budget, query, &hints) {
            Ok(nodes) if !nodes.is_empty() => render_dag_context(&nodes),
            Ok(_) => String::new(),
            Err(error) => {
                tracing::warn!(target:"deeplossless::responses_projection", %error,
                    "native Responses dynamic DAG assembly failed");
                String::new()
            }
        }
    } else {
        String::new()
    };

    let (plan_context, typed_plan_used) = if state.lcm_context {
        evaluate_typed_plan(state, session_id, conv_id)
    } else {
        (None, false)
    };

    let context = match (dag_context.is_empty(), plan_context.as_deref()) {
        (false, Some(plan)) => format!("{dag_context}\n\n{plan}"),
        (false, None) => dag_context,
        (true, Some(plan)) => plan.to_string(),
        (true, None) => String::new(),
    };

    NativeProjectionOutput {
        conv_id: Some(conv_id),
        context,
        typed_plan_used,
        auto_facts,
    }
}

fn build_dynamic_hints(
    state: &AppState,
    session_id: &str,
    conv_id: i64,
) -> DynamicContextHints {
    let active = state.storage.db.get_active_plan(conv_id).ok().flatten();
    let (plan_id, goal, current_step) = match active {
        Some((plan_id, goal, pending, _completed, _assumptions)) => {
            let pending: Vec<String> = serde_json::from_value(pending).unwrap_or_default();
            (Some(plan_id), Some(goal), pending.first().cloned())
        }
        None => (None, None, None),
    };

    let store = GroundTruthStore::new(&state.storage.db);
    let mut semantic = store
        .rebuild_execution_state(session_id, plan_id, goal, current_step)
        .unwrap_or_default();

    if let Ok(cycle) = state.runtime.cycle.lock() {
        for path in &cycle.context_delta {
            semantic
                .changed_resources
                .insert(format!("file:{path}"));
        }
    }

    if let Ok(patterns) = state.storage.db.get_recent_failure_patterns(conv_id, 3) {
        for pattern in patterns {
            semantic.blocked_reasons.push(format!(
                "{} {} {}",
                pattern.signature, pattern.why_failed, pattern.attempted_fix
            ));
        }
    }

    DynamicContextHints::from_execution_state(&semantic)
}

pub fn inject_context(body: &mut Value, context: &str) {
    if context.trim().is_empty() {
        return;
    }
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    let context_item = json!({
        "type": "message",
        "role": "developer",
        "content": [{"type":"input_text","text":context}],
    });
    let insert_at = items
        .iter()
        .rposition(|item| item.get("role").and_then(Value::as_str) == Some("user"))
        .unwrap_or(items.len());
    items.insert(insert_at, context_item);
}

fn project_messages(items: &[Value]) -> Vec<Value> {
    let mut messages = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        match item_type {
            "message" | "" => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user");
                let content = content_text(item.get("content"));
                if !content.is_empty() {
                    messages.push(json!({"role":role,"content":content}));
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                let content = content_text(item.get("output"));
                if !content.is_empty() {
                    messages.push(json!({
                        "role":"tool",
                        "content":content,
                        "tool_call_id":item.get("call_id").and_then(Value::as_str).unwrap_or(""),
                    }));
                }
            }
            _ => {}
        }
    }
    messages
}

fn content_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    value
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .or_else(|| block.get("output_text").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn dependency_applies_to_step(dependency: &PlanFactDependency, step_index: usize) -> bool {
    dependency.step_index.is_none() || dependency.step_index == Some(step_index)
}

fn evaluate_typed_plan(
    state: &AppState,
    session_id: &str,
    conv_id: i64,
) -> (Option<String>, bool) {
    let Some((plan_id, goal, pending_value, completed_value, _legacy_assumptions)) =
        state.storage.db.get_active_plan(conv_id).ok().flatten()
    else {
        return (None, false);
    };

    let pending: Vec<String> = serde_json::from_value(pending_value).unwrap_or_default();
    let completed: Vec<String> = serde_json::from_value(completed_value).unwrap_or_default();
    if pending.is_empty() {
        return (None, false);
    }
    let current_step_index = completed.len();

    let store = GroundTruthStore::new(&state.storage.db);
    let mut semantic = match store.rebuild_execution_state(
        session_id,
        Some(plan_id),
        Some(goal.clone()),
        pending.first().cloned(),
    ) {
        Ok(state) => state,
        Err(error) => {
            tracing::warn!(target:"deeplossless::typed_plan", %error,
                "failed to rebuild typed plan state");
            return (None, false);
        }
    };

    let active_dependencies: Vec<&PlanFactDependency> = semantic
        .plan_dependencies
        .iter()
        .filter(|dependency| {
            dependency.plan_id == plan_id
                && dependency_applies_to_step(dependency, current_step_index)
        })
        .collect();
    if active_dependencies.is_empty() {
        return (None, false);
    }

    let changed_paths = state
        .runtime
        .cycle
        .lock()
        .map(|cycle| cycle.context_delta.clone())
        .unwrap_or_default();
    for path in changed_paths {
        for fact in semantic.facts.values_mut() {
            if fact.resources.iter().any(|resource| match resource {
                ResourceRef::File { path: expected, .. } => expected == &path,
                ResourceRef::Symbol { file_path, .. } => file_path == &path,
                _ => false,
            }) {
                fact.status = FactStatus::Stale;
            }
        }
    }

    let required_fact_ids: Vec<String> = active_dependencies
        .iter()
        .filter(|dependency| dependency.required)
        .map(|dependency| dependency.fact_id.clone())
        .collect();
    let missing: Vec<String> = required_fact_ids
        .iter()
        .filter(|id| !semantic.facts.contains_key(*id))
        .cloned()
        .collect();
    let stale: Vec<String> = required_fact_ids
        .iter()
        .filter_map(|id| semantic.facts.get(id))
        .filter(|fact| fact.status != FactStatus::Valid)
        .map(|fact| fact.id.clone())
        .collect();
    let unbacked: Vec<String> = required_fact_ids
        .iter()
        .filter_map(|id| semantic.facts.get(id))
        .filter(|fact| fact.evidence.is_empty())
        .map(|fact| fact.id.clone())
        .collect();
    let unrecoverable: Vec<String> = required_fact_ids
        .iter()
        .filter_map(|id| semantic.facts.get(id))
        .filter(|fact| {
            fact.evidence
                .iter()
                .any(|source| !store.is_recoverable(source))
        })
        .map(|fact| fact.id.clone())
        .collect();

    if !missing.is_empty()
        || !stale.is_empty()
        || !unbacked.is_empty()
        || !unrecoverable.is_empty()
    {
        let reason = format!(
            "typed plan dependencies invalid: missing={missing:?}, stale={stale:?}, unbacked={unbacked:?}, unrecoverable={unrecoverable:?}"
        );
        let _ = state
            .storage
            .db
            .store_decision_record(conv_id, "Replan", 0.95, &reason, 0);
        return (
            Some(format!(
                "[Plan needs replanning: {goal}]\n[Reason: {reason}]\n[Pending steps: {}]",
                pending.join(", ")
            )),
            true,
        );
    }

    let next = pending.first().cloned().unwrap_or_default();
    let _ = state.storage.db.store_decision_record(
        conv_id,
        "ContinuePlan",
        0.95,
        "all current-step required typed facts are valid, source-backed, and recoverable",
        0,
    );
    (
        Some(format!(
            "[Plan active: {goal}]\n[Next step: {next}]\n[Typed dependencies verified: {} required facts]",
            required_fact_ids.len()
        )),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_native_messages_and_tool_outputs_without_truncation() {
        let long = "x".repeat(10_000);
        let items = vec![
            json!({"type":"message","role":"user","content":[{"type":"input_text","text":long}]}),
            json!({"type":"function_call_output","call_id":"c1","output":"tests failed"}),
        ];
        let messages = project_messages(&items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"].as_str().unwrap().len(), 10_000);
        assert_eq!(messages[1]["role"], "tool");
    }

    #[test]
    fn context_is_inserted_before_latest_user() {
        let mut body = json!({
            "input": [
                {"type":"message","role":"assistant","content":"old"},
                {"type":"message","role":"user","content":"now"}
            ]
        });
        inject_context(&mut body, "grounded context");
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["role"], "developer");
        assert_eq!(input[2]["role"], "user");
    }

    #[test]
    fn completed_step_dependencies_do_not_apply_to_current_step() {
        let old = PlanFactDependency {
            plan_id: 1,
            step_index: Some(0),
            fact_id: "old".into(),
            required: true,
        };
        let current = PlanFactDependency {
            plan_id: 1,
            step_index: Some(1),
            fact_id: "current".into(),
            required: true,
        };
        let global = PlanFactDependency {
            plan_id: 1,
            step_index: None,
            fact_id: "global".into(),
            required: true,
        };
        assert!(!dependency_applies_to_step(&old, 1));
        assert!(dependency_applies_to_step(&current, 1));
        assert!(dependency_applies_to_step(&global, 1));
    }
}
