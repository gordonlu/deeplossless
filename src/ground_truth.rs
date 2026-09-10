//! Ground-truth references and typed execution facts.
//!
//! This module hardens a core DeepLossless invariant:
//!
//! **Compression may be lossy; truth must not be.**
//!
//! Runtime projections, DAG summaries, plan assumptions and snapshots may all
//! be discarded and rebuilt.  Any decision-critical derived state should point
//! back to immutable evidence through [`SourceRef`].  `SourceRef` is deliberately
//! transport/storage neutral: today it can address `execution_events`; later it
//! may point at an external content-addressed payload store without changing
//! the projection model.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Immutable truth stream that owns the referenced evidence.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TruthStream {
    /// Ordered execution/audit event ledger.
    ExecutionEvents,
    /// Original provider request/response item stream.
    ProviderItems,
    /// File observations captured at a concrete content hash.
    FileObservations,
    /// Exact tool input/output payloads.
    ToolPayloads,
    /// Other append-only stream identified by a stable name.
    Named(String),
}

/// Exact address of immutable source evidence.
///
/// A projection MUST NOT claim exact recoverability merely because it has a
/// DAG node id. Exact recovery requires a source address plus a content hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    pub stream: TruthStream,
    /// Monotonic sequence / row identity inside the truth stream.
    pub seq_no: i64,
    /// SHA-256 of the complete uncompressed payload bytes.
    pub content_hash: String,
    /// Exact payload byte length. Useful for detecting truncated materialization.
    pub payload_len: u64,
    /// Optional stable locator for externalized payloads (blob id, object key).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<String>,
}

impl SourceRef {
    pub fn new(stream: TruthStream, seq_no: i64, payload: &[u8]) -> Self {
        Self {
            stream,
            seq_no,
            content_hash: sha256_hex(payload),
            payload_len: payload.len() as u64,
            payload_ref: None,
        }
    }

    pub fn with_payload_ref(mut self, payload_ref: impl Into<String>) -> Self {
        self.payload_ref = Some(payload_ref.into());
        self
    }

    /// Verify that materialized bytes are the exact payload referenced here.
    pub fn verify(&self, payload: &[u8]) -> bool {
        self.payload_len == payload.len() as u64 && self.content_hash == sha256_hex(payload)
    }
}

fn sha256_hex(payload: &[u8]) -> String {
    hex::encode(Sha256::digest(payload))
}

/// A current-world resource whose version can invalidate a derived fact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResourceRef {
    File {
        path: String,
        content_hash: String,
    },
    Symbol {
        file_path: String,
        symbol: String,
        content_hash: String,
    },
    ToolResult {
        call_id: String,
        content_hash: String,
    },
    Artifact {
        artifact_id: i64,
        version_hash: String,
    },
    Environment {
        key: String,
        value_hash: String,
    },
}

impl ResourceRef {
    /// Stable resource identity independent of its observed version.
    pub fn identity(&self) -> String {
        match self {
            Self::File { path, .. } => format!("file:{path}"),
            Self::Symbol {
                file_path, symbol, ..
            } => format!("symbol:{file_path}::{symbol}"),
            Self::ToolResult { call_id, .. } => format!("tool:{call_id}"),
            Self::Artifact { artifact_id, .. } => format!("artifact:{artifact_id}"),
            Self::Environment { key, .. } => format!("env:{key}"),
        }
    }

    pub fn version(&self) -> &str {
        match self {
            Self::File { content_hash, .. }
            | Self::Symbol { content_hash, .. }
            | Self::ToolResult { content_hash, .. } => content_hash,
            Self::Artifact { version_hash, .. } => version_hash,
            Self::Environment { value_hash, .. } => value_hash,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FactStatus {
    #[default]
    Valid,
    Stale,
    Unknown,
}

/// A compact execution fact projected from exact evidence.
///
/// Facts are disposable projections. `evidence` is the bridge back to truth;
/// `resources` describes what can make the fact stale in the live world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionFact {
    pub id: String,
    pub statement: String,
    #[serde(default)]
    pub evidence: Vec<SourceRef>,
    #[serde(default)]
    pub resources: Vec<ResourceRef>,
    #[serde(default)]
    pub status: FactStatus,
}

impl ExecutionFact {
    pub fn new(id: impl Into<String>, statement: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            statement: statement.into(),
            evidence: Vec::new(),
            resources: Vec::new(),
            status: FactStatus::Valid,
        }
    }

    pub fn is_source_backed(&self) -> bool {
        !self.evidence.is_empty()
    }

    /// Mark this fact stale when a resource with the same stable identity has
    /// been observed at a different version.
    pub fn invalidate_if_changed(&mut self, current: &ResourceRef) -> bool {
        if self.resources.iter().any(|expected| {
            expected.identity() == current.identity() && expected.version() != current.version()
        }) {
            self.status = FactStatus::Stale;
            true
        } else {
            false
        }
    }
}

/// Typed replacement for `PlanState.assumptions: Vec<String>`.
///
/// This is intentionally a sidecar first: existing PlanState serialization and
/// database schema remain compatible while callers migrate incrementally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanFactDependency {
    pub plan_id: i64,
    /// Index into PlanState.pending/completed logical step ordering, when known.
    pub step_index: Option<usize>,
    pub fact_id: String,
    /// A dependency can be informative without blocking the plan on staleness.
    pub required: bool,
}

/// Current execution state derived from truth, not a compressed transcript.
///
/// Lifecycle metrics remain in `RuntimeStateProjection`; this type owns the
/// semantic state needed to continue a task: facts, active plan dependencies,
/// invalidations and current resource versions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionStateProjection {
    pub goal: Option<String>,
    pub active_plan_id: Option<i64>,
    pub current_step: Option<String>,
    pub facts: BTreeMap<String, ExecutionFact>,
    pub plan_dependencies: Vec<PlanFactDependency>,
    pub changed_resources: BTreeSet<String>,
    pub blocked_reasons: Vec<String>,
}

impl ExecutionStateProjection {
    pub fn insert_fact(&mut self, fact: ExecutionFact) {
        self.facts.insert(fact.id.clone(), fact);
    }

    /// Apply a newly observed resource version and return the fact ids that
    /// became stale. This is deterministic; no LLM/string matching is involved.
    pub fn observe_resource(&mut self, current: &ResourceRef) -> Vec<String> {
        self.changed_resources.insert(current.identity());
        let mut stale = Vec::new();
        for fact in self.facts.values_mut() {
            if fact.invalidate_if_changed(current) {
                stale.push(fact.id.clone());
            }
        }
        stale
    }

    /// Required facts for the active plan that are no longer valid.
    pub fn stale_required_plan_facts(&self) -> Vec<&ExecutionFact> {
        let Some(plan_id) = self.active_plan_id else {
            return Vec::new();
        };
        self.plan_dependencies
            .iter()
            .filter(|dep| dep.plan_id == plan_id && dep.required)
            .filter_map(|dep| self.facts.get(&dep.fact_id))
            .filter(|fact| fact.status != FactStatus::Valid)
            .collect()
    }

    /// Every required active-plan fact must be backed by at least one exact
    /// source reference before the state can claim lossless recoverability.
    pub fn unbacked_required_plan_facts(&self) -> Vec<&ExecutionFact> {
        let Some(plan_id) = self.active_plan_id else {
            return Vec::new();
        };
        self.plan_dependencies
            .iter()
            .filter(|dep| dep.plan_id == plan_id && dep.required)
            .filter_map(|dep| self.facts.get(&dep.fact_id))
            .filter(|fact| !fact.is_source_backed())
            .collect()
    }
}

/// Deterministic closed-loop check around a compaction/projection update.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionValidationReport {
    pub ok: bool,
    /// Source refs needed before compaction but not recoverable afterwards.
    pub missing_sources: Vec<SourceRef>,
    /// Required active-plan facts omitted from the compacted projection.
    pub missing_plan_facts: Vec<String>,
    /// Required plan facts present but already stale/unknown.
    pub stale_plan_facts: Vec<String>,
    /// Required plan facts that have no exact source evidence at all.
    pub unbacked_plan_facts: Vec<String>,
}

/// Validate continuation-critical equivalence without another model call.
///
/// `recoverable_after` should include source refs directly present in the new
/// working view *and* refs reachable through its summaries/landmarks. This
/// allows context to be aggressively compressed while keeping exact recovery.
pub fn validate_compaction(
    before: &ExecutionStateProjection,
    after: &ExecutionStateProjection,
    recoverable_after: &HashSet<SourceRef>,
) -> CompactionValidationReport {
    let mut required_sources = HashSet::new();
    if let Some(plan_id) = before.active_plan_id {
        for dep in before
            .plan_dependencies
            .iter()
            .filter(|dep| dep.plan_id == plan_id && dep.required)
        {
            if let Some(fact) = before.facts.get(&dep.fact_id) {
                required_sources.extend(fact.evidence.iter().cloned());
            }
        }
    }

    let mut missing_sources: Vec<SourceRef> = required_sources
        .into_iter()
        .filter(|source| !recoverable_after.contains(source))
        .collect();
    missing_sources.sort_by_key(|source| source.seq_no);

    let mut missing_plan_facts = Vec::new();
    if let Some(plan_id) = before.active_plan_id {
        for dep in before
            .plan_dependencies
            .iter()
            .filter(|dep| dep.plan_id == plan_id && dep.required)
        {
            if !after.facts.contains_key(&dep.fact_id) {
                missing_plan_facts.push(dep.fact_id.clone());
            }
        }
    }

    let stale_plan_facts: Vec<String> = after
        .stale_required_plan_facts()
        .into_iter()
        .map(|fact| fact.id.clone())
        .collect();
    let unbacked_plan_facts: Vec<String> = after
        .unbacked_required_plan_facts()
        .into_iter()
        .map(|fact| fact.id.clone())
        .collect();

    let ok = missing_sources.is_empty()
        && missing_plan_facts.is_empty()
        && stale_plan_facts.is_empty()
        && unbacked_plan_facts.is_empty();

    CompactionValidationReport {
        ok,
        missing_sources,
        missing_plan_facts,
        stale_plan_facts,
        unbacked_plan_facts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(seq: i64, text: &str) -> SourceRef {
        SourceRef::new(TruthStream::ExecutionEvents, seq, text.as_bytes())
    }

    #[test]
    fn source_ref_detects_truncation_and_mutation() {
        let src = source(7, "complete tool output");
        assert!(src.verify(b"complete tool output"));
        assert!(!src.verify(b"complete tool"));
        assert!(!src.verify(b"complete tool Output"));
    }

    #[test]
    fn typed_resource_change_invalidates_fact() {
        let mut state = ExecutionStateProjection::default();
        let mut fact = ExecutionFact::new("fact-1", "config enables foo");
        fact.evidence.push(source(10, "foo=true"));
        fact.resources.push(ResourceRef::File {
            path: "src/config.rs".into(),
            content_hash: "old".into(),
        });
        state.insert_fact(fact);

        let stale = state.observe_resource(&ResourceRef::File {
            path: "src/config.rs".into(),
            content_hash: "new".into(),
        });
        assert_eq!(stale, vec!["fact-1"]);
        assert_eq!(state.facts["fact-1"].status, FactStatus::Stale);
    }

    #[test]
    fn same_resource_version_does_not_invalidate() {
        let mut fact = ExecutionFact::new("f", "stable");
        fact.resources.push(ResourceRef::Environment {
            key: "rustc".into(),
            value_hash: "1.90".into(),
        });
        assert!(!fact.invalidate_if_changed(&ResourceRef::Environment {
            key: "rustc".into(),
            value_hash: "1.90".into(),
        }));
        assert_eq!(fact.status, FactStatus::Valid);
    }

    #[test]
    fn compaction_fails_when_required_evidence_is_not_recoverable() {
        let src = source(42, "exact evidence");
        let mut before = ExecutionStateProjection {
            active_plan_id: Some(9),
            ..Default::default()
        };
        let mut fact = ExecutionFact::new("fact-a", "the test currently fails");
        fact.evidence.push(src.clone());
        before.insert_fact(fact.clone());
        before.plan_dependencies.push(PlanFactDependency {
            plan_id: 9,
            step_index: Some(0),
            fact_id: "fact-a".into(),
            required: true,
        });

        let after = before.clone();
        let report = validate_compaction(&before, &after, &HashSet::new());
        assert!(!report.ok);
        assert_eq!(report.missing_sources, vec![src]);
    }

    #[test]
    fn compaction_passes_when_fact_and_exact_source_remain_recoverable() {
        let src = source(42, "exact evidence");
        let mut state = ExecutionStateProjection {
            active_plan_id: Some(9),
            ..Default::default()
        };
        let mut fact = ExecutionFact::new("fact-a", "the test currently fails");
        fact.evidence.push(src.clone());
        state.insert_fact(fact);
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 9,
            step_index: Some(0),
            fact_id: "fact-a".into(),
            required: true,
        });

        let recoverable = HashSet::from([src]);
        let report = validate_compaction(&state, &state, &recoverable);
        assert!(report.ok, "{report:?}");
    }

    #[test]
    fn unbacked_required_fact_prevents_lossless_claim() {
        let mut state = ExecutionStateProjection {
            active_plan_id: Some(1),
            ..Default::default()
        };
        state.insert_fact(ExecutionFact::new("f", "unsupported assumption"));
        state.plan_dependencies.push(PlanFactDependency {
            plan_id: 1,
            step_index: None,
            fact_id: "f".into(),
            required: true,
        });
        let report = validate_compaction(&state, &state, &HashSet::new());
        assert!(!report.ok);
        assert_eq!(report.unbacked_plan_facts, vec!["f"]);
    }
}
