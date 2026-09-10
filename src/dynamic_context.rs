//! Execution-state-aware working-context recall.
//!
//! Implements the useful part of AttnCompress without proxy-attention/PPL:
//! historical DAG nodes are re-scored when the current plan, resource changes,
//! failure state or query makes them relevant again.

use std::collections::HashSet;

use crate::context_pack::DynamicSignals;
use crate::dag::{DagEngine, DagNode};
use crate::ground_truth::{ExecutionStateProjection, ResourceRef};

/// Dynamic hints derived from the current execution state.
#[derive(Debug, Clone, Default)]
pub struct DynamicContextHints {
    pub plan_terms: Vec<String>,
    pub changed_resources: Vec<String>,
    pub failure_terms: Vec<String>,
}

impl DynamicContextHints {
    pub fn from_execution_state(state: &ExecutionStateProjection) -> Self {
        let mut plan_terms = Vec::new();
        let mut changed_resources = state.changed_resources.iter().cloned().collect::<Vec<_>>();
        let mut failure_terms = Vec::new();

        if let Some(goal) = &state.goal {
            extend_terms(&mut plan_terms, goal);
        }
        if let Some(step) = &state.current_step {
            extend_terms(&mut plan_terms, step);
        }
        if let Some(plan_id) = state.active_plan_id {
            for dep in state
                .plan_dependencies
                .iter()
                .filter(|dep| dep.plan_id == plan_id && dep.required)
            {
                if let Some(fact) = state.facts.get(&dep.fact_id) {
                    extend_terms(&mut plan_terms, &fact.statement);
                    for resource in &fact.resources {
                        match resource {
                            ResourceRef::File { path, .. } => {
                                plan_terms.push(path.clone());
                            }
                            ResourceRef::Symbol { file_path, symbol, .. } => {
                                plan_terms.push(file_path.clone());
                                plan_terms.push(symbol.clone());
                            }
                            ResourceRef::ToolResult { call_id, .. } => {
                                plan_terms.push(call_id.clone());
                            }
                            ResourceRef::Artifact { artifact_id, .. } => {
                                plan_terms.push(format!("artifact:{artifact_id}"));
                            }
                            ResourceRef::Environment { key, .. } => {
                                plan_terms.push(key.clone());
                            }
                        }
                    }
                }
            }
        }

        for blocker in &state.blocked_reasons {
            extend_terms(&mut failure_terms, blocker);
        }

        dedup_terms(&mut plan_terms);
        dedup_terms(&mut changed_resources);
        dedup_terms(&mut failure_terms);
        Self {
            plan_terms,
            changed_resources,
            failure_terms,
        }
    }

    pub fn has_execution_signal(&self) -> bool {
        !self.plan_terms.is_empty()
            || !self.changed_resources.is_empty()
            || !self.failure_terms.is_empty()
    }
}

/// Assemble normal DAG context, then dynamically recall older nodes that are
/// strongly relevant to current execution state. The returned set remains
/// token-budget bounded, coverage-aware and duplicate-free.
///
/// When dynamic signals exist, 25% of the budget is initially reserved for
/// recall so recent context cannot consume the entire window before an old but
/// execution-critical node has a chance to compete. Any unused reserve is then
/// filled from the normal context order.
pub fn assemble_dynamic_context(
    dag: &DagEngine,
    conv_id: i64,
    token_budget: usize,
    query: Option<&str>,
    hints: &DynamicContextHints,
) -> anyhow::Result<Vec<DagNode>> {
    let normal = dag.assemble_context(conv_id, token_budget, query)?;
    if token_budget == 0 || !hints.has_execution_signal() {
        return Ok(normal);
    }

    let reserve = (token_budget / 4).max(1);
    let base_budget = token_budget.saturating_sub(reserve);
    let mut selected = Vec::new();
    let mut used_tokens = 0i64;
    let mut selected_ids = HashSet::new();
    let mut covered_ids = HashSet::new();

    // Start from the ordinary selection but leave deterministic room for recall.
    for node in &normal {
        let tc = node.token_count.max(0);
        if used_tokens + tc > base_budget as i64 {
            continue;
        }
        add_node(
            dag,
            node.clone(),
            &mut selected,
            &mut selected_ids,
            &mut covered_ids,
            &mut used_tokens,
        )?;
    }

    let candidates = dag.db().get_all_dag_nodes(conv_id)?;
    let max_id = candidates.iter().map(|node| node.id).max().unwrap_or(1) as f64;
    let query_terms = query.map(tokenize_terms).unwrap_or_default();

    let mut scored = Vec::<(f64, DagNode)>::new();
    for node in candidates {
        if selected_ids.contains(&node.id)
            || covered_ids.contains(&node.id)
            || node.token_count <= 0
            || node.summary.is_empty()
            || summary_would_duplicate_selected_leaf(dag, &node, &selected_ids)?
        {
            continue;
        }
        let signals = signals_for_node(&node, hints, &query_terms);
        let dynamic = signals.boost();
        // Do not recall on weak generic similarity alone. A node needs a
        // meaningful execution-state signal or a strong direct query match.
        if dynamic < 0.12 && signals.semantic_similarity < 0.7 {
            continue;
        }
        let recency = (node.id as f64 / max_id).clamp(0.0, 1.0);
        let memory = dag.compute_memory_score(&node);
        let score = 0.65 * dynamic + 0.20 * memory + 0.15 * recency;
        scored.push((score, node));
    }

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.id.cmp(&a.1.id))
    });
    for (_score, node) in scored.into_iter().take(8) {
        if selected_ids.contains(&node.id)
            || covered_ids.contains(&node.id)
            || summary_would_duplicate_selected_leaf(dag, &node, &selected_ids)?
        {
            continue;
        }
        if used_tokens + node.token_count > token_budget as i64 {
            continue;
        }
        add_node(
            dag,
            node,
            &mut selected,
            &mut selected_ids,
            &mut covered_ids,
            &mut used_tokens,
        )?;
    }

    // Give unused recall budget back to the ordinary assembler order.
    for node in normal {
        if selected_ids.contains(&node.id)
            || covered_ids.contains(&node.id)
            || summary_would_duplicate_selected_leaf(dag, &node, &selected_ids)?
        {
            continue;
        }
        let tc = node.token_count.max(0);
        if used_tokens + tc > token_budget as i64 {
            continue;
        }
        add_node(
            dag,
            node,
            &mut selected,
            &mut selected_ids,
            &mut covered_ids,
            &mut used_tokens,
        )?;
    }

    debug_assert!(used_tokens <= token_budget as i64);
    Ok(selected)
}

fn add_node(
    dag: &DagEngine,
    node: DagNode,
    selected: &mut Vec<DagNode>,
    selected_ids: &mut HashSet<i64>,
    covered_ids: &mut HashSet<i64>,
    used_tokens: &mut i64,
) -> anyhow::Result<()> {
    if node.level > 0 {
        covered_ids.extend(dag.covered_leaf_ids(node.id)?);
    }
    selected_ids.insert(node.id);
    *used_tokens += node.token_count.max(0);
    selected.push(node);
    Ok(())
}

fn summary_would_duplicate_selected_leaf(
    dag: &DagEngine,
    node: &DagNode,
    selected_ids: &HashSet<i64>,
) -> anyhow::Result<bool> {
    if node.level == 0 {
        return Ok(false);
    }
    let covered = dag.covered_leaf_ids(node.id)?;
    Ok(covered.iter().any(|id| selected_ids.contains(id)))
}

fn signals_for_node(
    node: &DagNode,
    hints: &DynamicContextHints,
    query_terms: &HashSet<String>,
) -> DynamicSignals {
    let text = node_text(node);
    let text_lower = text.to_lowercase();
    let plan_dependency = term_match_score(&text_lower, &hints.plan_terms);
    let changed_resource = resource_match_score(&text_lower, &hints.changed_resources);
    let failure_lineage = term_match_score(&text_lower, &hints.failure_terms);
    let precision = if node.snippets.is_empty() {
        0.0
    } else {
        node.snippets
            .iter()
            .map(|snippet| snippet.importance as f64)
            .fold(0.0, f64::max)
            .clamp(0.0, 1.0)
    };
    let semantic_similarity = lexical_similarity(&text_lower, query_terms);
    DynamicSignals {
        plan_dependency,
        changed_resource,
        failure_lineage,
        precision,
        semantic_similarity,
    }
}

fn node_text(node: &DagNode) -> String {
    let mut text = node.summary.clone();
    for snippet in &node.snippets {
        text.push('\n');
        text.push_str(&snippet.content);
    }
    text
}

fn term_match_score(text_lower: &str, terms: &[String]) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let hits = terms
        .iter()
        .filter(|term| term.len() >= 2 && text_lower.contains(&term.to_lowercase()))
        .count();
    (hits as f64 / terms.len().min(4) as f64).clamp(0.0, 1.0)
}

fn resource_match_score(text_lower: &str, resources: &[String]) -> f64 {
    resources
        .iter()
        .any(|resource| {
            let raw = resource
                .strip_prefix("file:")
                .or_else(|| resource.strip_prefix("symbol:"))
                .unwrap_or(resource);
            !raw.is_empty() && text_lower.contains(&raw.to_lowercase())
        })
        .then_some(1.0)
        .unwrap_or(0.0)
}

fn lexical_similarity(text_lower: &str, query_terms: &HashSet<String>) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let text_terms = tokenize_terms(text_lower);
    if text_terms.is_empty() {
        return 0.0;
    }
    let intersection = query_terms.intersection(&text_terms).count();
    let union = query_terms.union(&text_terms).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn tokenize_terms(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '/' && c != '.' && c != '-')
        .map(str::trim)
        .filter(|term| term.len() >= 2)
        .map(|term| term.to_lowercase())
        .collect()
}

fn extend_terms(out: &mut Vec<String>, text: &str) {
    out.extend(tokenize_terms(text));
}

fn dedup_terms(terms: &mut Vec<String>) {
    let mut seen = HashSet::new();
    terms.retain(|term| seen.insert(term.to_lowercase()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_resource_creates_strong_dynamic_signal() {
        let hints = DynamicContextHints {
            plan_terms: vec!["src/config.rs".into(), "ConfigLoader".into()],
            changed_resources: vec!["file:src/config.rs".into()],
            failure_terms: vec![],
        };
        let node = DagNode {
            id: 1,
            conversation_id: 1,
            level: 0,
            summary: "read src/config.rs and inspect ConfigLoader".into(),
            token_count: 10,
            parent_ids: vec![],
            child_ids: vec![],
            snippets: vec![],
            is_leaf: true,
            is_join: false,
            deleted: false,
            semantic_hash: String::new(),
            access_count: 0,
            last_accessed_at: None,
            reasoning: String::new(),
            graph_revision: 0,
            compaction_id: String::new(),
        };
        let signals = signals_for_node(&node, &hints, &HashSet::new());
        assert!(signals.plan_dependency >= 0.5);
        assert_eq!(signals.changed_resource, 1.0);
        assert!(signals.boost() > 0.3);
    }

    #[test]
    fn lexical_similarity_rewards_query_overlap() {
        let query = tokenize_terms("cargo test config failure");
        assert!(lexical_similarity("cargo test config failure", &query) > 0.9);
        assert_eq!(lexical_similarity("unrelated weather text", &query), 0.0);
    }

    #[test]
    fn hints_derive_plan_terms_without_falsely_marking_resource_changed() {
        use crate::ground_truth::{ExecutionFact, PlanFactDependency};
        let mut state = ExecutionStateProjection {
            goal: Some("fix config loader".into()),
            active_plan_id: Some(7),
            current_step: Some("edit src/config.rs".into()),
            ..Default::default()
        };
        let mut fact = ExecutionFact::new("f", "ConfigLoader ignores env");
        fact.resources.push(ResourceRef::File {
            path: "src/config.rs".into(),
            content_hash: "abc".into(),
        });
        state.insert_fact(fact);
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 7,
            step_index: Some(0),
            fact_id: "f".into(),
            required: true,
        });
        let hints = DynamicContextHints::from_execution_state(&state);
        assert!(hints.plan_terms.iter().any(|term| term.contains("config")));
        assert!(hints.changed_resources.is_empty());
    }

    #[test]
    fn execution_signal_detection_ignores_empty_hints() {
        assert!(!DynamicContextHints::default().has_execution_signal());
        assert!(DynamicContextHints {
            plan_terms: vec!["src/lib.rs".into()],
            ..Default::default()
        }
        .has_execution_signal());
    }
}
