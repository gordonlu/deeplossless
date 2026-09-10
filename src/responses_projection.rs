//! Internal projection pipeline for native Responses transport.
//!
//! Wire protocol stays Responses end-to-end. This module derives a minimal
//! Chat-shaped *projection* solely so the existing DAG/compaction pipeline can
//! ingest the turn, then assembles DeepLossless context back into a native
//! Responses input item. Projection loss never affects Ground Truth storage.

use serde_json::{json, Value};

use crate::ground_truth::{FactStatus, ResourceRef};
use crate::ground_truth_store::GroundTruthStore;
use crate::pipeline::{render_dag_context, ChatPipeline};
use crate::AppState;

#[derive(Debug, Clone, Default)]
pub struct NativeProjectionOutput {
    pub conv_id: Option<i64>,
    pub context: String,
    pub typed_plan_used: bool,
}

/// Ingest the current native Responses turn into the LCM projection pipeline
/// and assemble context for injection into the upstream native request.
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

    let shadow = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });
    let pipeline = ChatPipeline::new(state);
    let processed = match pipeline
        .process_with_fp(model, &shadow, 1, Some(session_id))
        .await
    {
        Ok(out) => out,
        Err(error) => {
            tracing::warn!(target:"deeplossless::responses_projection", %error,
                "native Responses projection pipeline failed");
            return NativeProjectionOutput::default();
        }
    };

    let conv_id = processed.conv_id;
    let query = messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str);

    let budget = state.lcm_context_tokens.clamp(0, 8_000) as usize;
    let dag_context = if state.lcm_context && budget > 0 {
        match state.storage.dag.assemble_context(conv_id, budget, query) {
            Ok(nodes) if !nodes.is_empty() => render_dag_context(&nodes),
            Ok(_) => String::new(),
            Err(error) => {
                tracing::warn!(target:"deeplossless::responses_projection", %error,
                    "native Responses DAG assembly failed");
                String::new()
            }
        }
    } else {
        String::new()
    };

    let (plan_context, typed_plan_used) = evaluate_typed_plan(
        state,
        session_id,
        conv_id,
    );

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
    }
}

/// Inject derived context as a developer message immediately before the latest
/// user message. The persisted session history remains the original provider
/// items; this mutation applies only to the one upstream request.
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
        "content": [{
            "type": "input_text",
            "text": context,
        }],
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
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = content_text(item.get("content"));
                if !content.is_empty() {
                    messages.push(json!({"role": role, "content": content}));
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                let content = content_text(item.get("output"));
                if !content.is_empty() {
                    messages.push(json!({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": item.get("call_id").and_then(Value::as_str).unwrap_or(""),
                    }));
                }
            }
            _ => {}
        }
    }
    messages
}

fn content_text(value: Option<&Value>) -> String {
    let Some(value) = value else { return String::new(); };
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    value
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    block.get("text").and_then(Value::as_str)
                        .or_else(|| block.get("output_text").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Prefer typed Plan→Fact dependencies. Legacy string assumptions are evaluated
/// by the old Runtime path when no typed dependencies have been registered.
fn evaluate_typed_plan(
    state: &AppState,
    session_id: &str,
    conv_id: i64,
) -> (Option<String>, bool) {
    let Some((plan_id, goal, pending_value, _completed, _legacy_assumptions)) =
        state.storage.db.get_active_plan(conv_id).ok().flatten()
    else {
        return (None, false);
    };

    let pending: Vec<String> = serde_json::from_value(pending_value).unwrap_or_default();
    if pending.is_empty() {
        return (None, false);
    }

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
    if semantic.plan_dependencies.is_empty() {
        return (None, false);
    }

    // context_delta represents resources known to have changed during the
    // execution cycle. Typed resource identity replaces assumption-string
    // matching. A matching File/Symbol dependency is stale immediately.
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

    let required_fact_ids: Vec<String> = semantic
        .plan_dependencies
        .iter()
        .filter(|dependency| dependency.plan_id == plan_id && dependency.required)
        .map(|dependency| dependency.fact_id.clone())
        .collect();

    let missing: Vec<String> = required_fact_ids
        .iter()
        .filter(|id| !semantic.facts.contains_key(*id))
        .cloned()
        .collect();
    let stale: Vec<String> = semantic
        .stale_required_plan_facts()
        .into_iter()
        .map(|fact| fact.id.clone())
        .collect();
    let unbacked: Vec<String> = semantic
        .unbacked_required_plan_facts()
        .into_iter()
        .map(|fact| fact.id.clone())
        .collect();
    let unrecoverable: Vec<String> = semantic
        .plan_dependencies
        .iter()
        .filter(|dependency| dependency.plan_id == plan_id && dependency.required)
        .filter_map(|dependency| semantic.facts.get(&dependency.fact_id))
        .filter(|fact| fact.evidence.iter().any(|source| !store.is_recoverable(source)))
        .map(|fact| fact.id.clone())
        .collect();

    if !missing.is_empty() || !stale.is_empty() || !unbacked.is_empty() || !unrecoverable.is_empty() {
        let reason = format!(
            "typed plan dependencies invalid: missing={missing:?}, stale={stale:?}, unbacked={unbacked:?}, unrecoverable={unrecoverable:?}"
        );
        let _ = state.storage.db.store_decision_record(
            conv_id,
            "Replan",
            0.95,
            &reason,
            0,
        );
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
        "all required typed facts are valid, source-backed, and recoverable",
        500,
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
    fn projects_native_messages_and_tool_outputs() {
        let items = vec![
            json!({"type":"message","role":"user","content":[{"type":"input_text","text":"fix it"}]}),
            json!({"type":"function_call_output","call_id":"c1","output":"tests failed"}),
        ];
        let messages = project_messages(&items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"], "fix it");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "c1");
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
}
