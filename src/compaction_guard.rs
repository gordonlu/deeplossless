//! Closed-loop validation for lossy context projection.
//!
//! The guard validates *continuation-critical recoverability*, not summary
//! quality. It deliberately avoids an extra LLM call on the normal path.
//! Required plan facts must remain present, valid, source-backed, and every
//! referenced exact payload must successfully materialize from Ground Truth.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::ground_truth::{
    CompactionValidationReport, ExecutionStateProjection, SourceRef, validate_compaction,
};
use crate::ground_truth_store::GroundTruthStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionDecision {
    Accept,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedLoopValidation {
    pub decision: CompactionDecision,
    pub report: CompactionValidationReport,
    /// 0..=100. Intended for telemetry / deciding whether a future semantic
    /// continuation probe is worth paying for.
    pub risk_score: u8,
}

pub struct CompactionGuard<'a> {
    truth: GroundTruthStore<'a>,
}

impl<'a> CompactionGuard<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self {
            truth: GroundTruthStore::new(db),
        }
    }

    /// Validate a candidate projection by materializing the exact evidence
    /// required by the active plan from SQLite.
    pub fn validate(
        &self,
        before: &ExecutionStateProjection,
        after: &ExecutionStateProjection,
    ) -> ClosedLoopValidation {
        let required_sources = required_active_plan_sources(before);
        let recoverable = self.truth.recoverable_set(required_sources.iter());
        let report = validate_compaction(before, after, &recoverable);
        let risk_score = risk_score(&report, required_sources.len());
        ClosedLoopValidation {
            decision: if report.ok {
                CompactionDecision::Accept
            } else {
                CompactionDecision::Reject
            },
            report,
            risk_score,
        }
    }
}

/// Exact evidence needed to continue the active plan before compaction.
pub fn required_active_plan_sources(state: &ExecutionStateProjection) -> HashSet<SourceRef> {
    let Some(plan_id) = state.active_plan_id else {
        return HashSet::new();
    };

    let mut sources = HashSet::new();
    for dep in state
        .plan_dependencies
        .iter()
        .filter(|dep| dep.plan_id == plan_id && dep.required)
    {
        if let Some(fact) = state.facts.get(&dep.fact_id) {
            sources.extend(fact.evidence.iter().cloned());
        }
    }
    sources
}

fn risk_score(report: &CompactionValidationReport, source_count: usize) -> u8 {
    if report.ok {
        // A successful deterministic check can still be semantically risky
        // when many exact sources are involved. Keep this deliberately low;
        // a future semantic probe can use the score as a trigger.
        return (source_count.min(20) * 2) as u8;
    }

    let mut score = 50usize;
    score += report.missing_sources.len() * 15;
    score += report.missing_plan_facts.len() * 20;
    score += report.stale_plan_facts.len() * 15;
    score += report.unbacked_plan_facts.len() * 20;
    score.min(100) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground_truth::{ExecutionFact, PlanFactDependency, TruthStream};
    use tempfile::tempdir;

    #[tokio::test]
    async fn guard_accepts_when_exact_plan_evidence_materializes() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("guard-ok.db"))
            .build()
            .await
            .unwrap();
        let store = GroundTruthStore::new(&db);
        let src = store
            .put_text(
                TruthStream::ToolPayloads,
                "s1",
                "cargo test: ok",
                "test-output",
            )
            .unwrap();

        let mut state = ExecutionStateProjection {
            active_plan_id: Some(3),
            ..Default::default()
        };
        let mut fact = ExecutionFact::new("tests-pass", "tests pass");
        fact.evidence.push(src);
        state.insert_fact(fact);
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 3,
            step_index: Some(1),
            fact_id: "tests-pass".into(),
            required: true,
        });

        let result = CompactionGuard::new(&db).validate(&state, &state);
        assert_eq!(result.decision, CompactionDecision::Accept);
        assert!(result.report.ok);
        assert!(result.risk_score < 50);
    }

    #[tokio::test]
    async fn guard_rejects_when_source_locator_is_broken() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("guard-bad.db"))
            .build()
            .await
            .unwrap();
        let store = GroundTruthStore::new(&db);
        let mut src = store
            .put_text(
                TruthStream::ToolPayloads,
                "s1",
                "exact output",
                "tool-output",
            )
            .unwrap();
        src.payload_ref = Some("proxy-event:s1:999999".into());

        let mut state = ExecutionStateProjection {
            active_plan_id: Some(3),
            ..Default::default()
        };
        let mut fact = ExecutionFact::new("f", "critical result");
        fact.evidence.push(src);
        state.insert_fact(fact);
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 3,
            step_index: None,
            fact_id: "f".into(),
            required: true,
        });

        let result = CompactionGuard::new(&db).validate(&state, &state);
        assert_eq!(result.decision, CompactionDecision::Reject);
        assert!(!result.report.ok);
        assert_eq!(result.report.missing_sources.len(), 1);
        assert!(result.risk_score >= 50);
    }

    #[test]
    fn required_sources_ignore_non_required_dependencies() {
        let mut state = ExecutionStateProjection {
            active_plan_id: Some(1),
            ..Default::default()
        };
        let mut required = ExecutionFact::new("required", "r");
        required
            .evidence
            .push(SourceRef::new(TruthStream::ExecutionEvents, 1, b"r"));
        let mut optional = ExecutionFact::new("optional", "o");
        optional
            .evidence
            .push(SourceRef::new(TruthStream::ExecutionEvents, 2, b"o"));
        state.insert_fact(required);
        state.insert_fact(optional);
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 1,
            step_index: None,
            fact_id: "required".into(),
            required: true,
        });
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 1,
            step_index: None,
            fact_id: "optional".into(),
            required: false,
        });
        let sources = required_active_plan_sources(&state);
        assert_eq!(sources.len(), 1);
        assert!(sources.iter().all(|source| source.seq_no == 1));
    }
}
