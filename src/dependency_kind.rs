//! Dependency taxonomy — the canonical vocabulary for dependency
//! relationships across all subsystems (Phase 3.1).
//!
//! This is a SEMANTIC vocabulary, not a storage schema. Each variant
//! describes WHAT kind of relationship exists, regardless of WHERE it
//! is stored (dag_edges, lineage_edges, tool_cache, etc.).
//!
//! Implementation note: subsystems MAY store these relationships in
//! different tables. The taxonomy unifies the MEANING, not the storage.

/// Canonical dependency kind — the single taxonomy for all dependency
/// relationships in the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    // ── Topology dependencies (stored in dag_edges) ──────────────────

    /// A summary node covers source nodes. Child → Parent.
    /// Created by: compress_group_with_snippets, insert_leaf.
    Coverage,

    /// A higher-level summary refines lower-level summaries.
    /// Created by: merge_nodes.
    Refinement,

    /// Cross-conversation semantic reuse of a summary.
    /// Created by: dedup_and_reuse.
    CrossSessionReuse,

    // ── Execution ordering dependencies ──────────────────────────────

    /// Unit B was executed after Unit A in sequence.
    /// Stored in: lineage_edges (kind = "depends_on").
    SequentialOrdering,

    /// Parallel execution group — branches happen-before the join.
    /// Stored in: dag_edges (kind = "happens_before").
    ParallelJoin,

    // ── Artifact dependencies ────────────────────────────────────────

    /// An execution read from a file. Cache invalidation trigger.
    /// Stored in: tool_cache (dependent_files).
    ReadsFile,

    /// An execution wrote to / produced a file.
    /// NOT YET ACTIVE — no producer exists.
    ProducesFile,

    /// An execution searched a file (grep-like).
    /// NOT YET ACTIVE — no producer exists.
    SearchesFile,

    // ── Lineage dependencies (Phase 2+ candidates) ───────────────────

    /// Unit B was derived from (summarized from) Unit A.
    /// Defined but no active producer.
    Derivation,

    /// Unit B was invalidated because Unit A changed.
    /// Defined but no active producer.
    Invalidation,

    /// Unit B's fix was suggested by Unit A's failure pattern.
    /// Defined but no active producer.
    FailureCorrection,
}

impl DependencyKind {
    /// Storage-neutral string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Coverage => "coverage",
            Self::Refinement => "refinement",
            Self::CrossSessionReuse => "cross_session_reuse",
            Self::SequentialOrdering => "sequential_ordering",
            Self::ParallelJoin => "parallel_join",
            Self::ReadsFile => "reads_file",
            Self::ProducesFile => "produces_file",
            Self::SearchesFile => "searches_file",
            Self::Derivation => "derivation",
            Self::Invalidation => "invalidation",
            Self::FailureCorrection => "failure_correction",
        }
    }

    /// Whether this kind represents a topology relationship (dag_edges).
    pub fn is_topology(&self) -> bool {
        matches!(self, Self::Coverage | Self::Refinement | Self::CrossSessionReuse)
    }

    /// Whether this kind represents an execution ordering relationship.
    pub fn is_ordering(&self) -> bool {
        matches!(self, Self::SequentialOrdering | Self::ParallelJoin)
    }

    /// Whether this kind represents a file artifact dependency.
    pub fn is_artifact(&self) -> bool {
        matches!(self, Self::ReadsFile | Self::ProducesFile | Self::SearchesFile)
    }

    /// Whether this kind has an active producer in production code.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Coverage
                | Self::Refinement
                | Self::CrossSessionReuse
                | Self::SequentialOrdering
                | Self::ParallelJoin
                | Self::ReadsFile
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_active_kinds_have_producers() {
        let active = [
            DependencyKind::Coverage,
            DependencyKind::Refinement,
            DependencyKind::CrossSessionReuse,
            DependencyKind::SequentialOrdering,
            DependencyKind::ParallelJoin,
            DependencyKind::ReadsFile,
        ];
        for k in &active {
            assert!(k.is_active(), "{k:?} should be active");
        }
    }

    #[test]
    fn inactive_kinds_are_marked_inactive() {
        assert!(!DependencyKind::ProducesFile.is_active());
        assert!(!DependencyKind::SearchesFile.is_active());
        assert!(!DependencyKind::Derivation.is_active());
        assert!(!DependencyKind::Invalidation.is_active());
        assert!(!DependencyKind::FailureCorrection.is_active());
    }

    #[test]
    fn topology_kinds_are_correctly_classified() {
        assert!(DependencyKind::Coverage.is_topology());
        assert!(DependencyKind::Refinement.is_topology());
        assert!(DependencyKind::CrossSessionReuse.is_topology());
        assert!(!DependencyKind::SequentialOrdering.is_topology());
        assert!(!DependencyKind::ReadsFile.is_topology());
    }

    #[test]
    fn artifact_kinds_are_correctly_classified() {
        assert!(DependencyKind::ReadsFile.is_artifact());
        assert!(DependencyKind::ProducesFile.is_artifact());
        assert!(DependencyKind::SearchesFile.is_artifact());
        assert!(!DependencyKind::Coverage.is_artifact());
    }
}
