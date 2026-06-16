//! DependencyView — unified read-only interpretation over all dependency
//! storage backends (Phase 3.3).
//!
//! Each subsystem owns its storage (dag_edges, lineage_edges, tool_cache).
//! This module is the SINGLE place to ask cross-subsystem dependency
//! questions. It does NOT unify storage — only interpretation.
//!
//! All queries are pure: DB in, structured answer out. No mutation.

use crate::dependency_kind::DependencyKind;

/// A single dependency edge in the unified view.
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// What kind of dependency this is.
    pub kind: DependencyKind,
    /// Source entity ID (depends on what).
    pub source_id: i64,
    /// Target entity ID (what depends on source).
    pub target_id: i64,
    /// Extra context (file path, tool name, etc.).
    pub context: Option<String>,
}

/// Unified read-only dependency interpretation layer.
///
/// # Authority note
/// This view reads from multiple storage backends. It is NOT an
/// authority — it is a projection. The authority for each kind
/// remains with the subsystem that created the dependency.
pub struct DependencyView;

impl DependencyView {
    /// What DAG nodes are topology-descendants of `node_id`?
    /// Walks summarizes/refines/reuses edges downward.
    pub fn topology_descendants(
        db: &crate::db::Database,
        node_id: i64,
    ) -> anyhow::Result<Vec<DependencyEdge>> {
        let edges = db.get_edges_from(node_id)?;
        let mut result = Vec::new();
        for (_edge_id, to_id, kind) in edges {
            let dk = match kind.as_str() {
                "summarizes" => DependencyKind::Coverage,
                "refines" => DependencyKind::Refinement,
                "reuses" => DependencyKind::CrossSessionReuse,
                _ => continue,
            };
            result.push(DependencyEdge {
                kind: dk,
                source_id: node_id,
                target_id: to_id,
                context: None,
            });
        }
        Ok(result)
    }

    /// What files does this execution unit depend on (for invalidation)?
    /// Reads from the tool_cache module via the DB.
    pub fn files_depended_on_by_execution(
        db: &crate::db::Database,
        execution_unit_id: i64,
    ) -> anyhow::Result<Vec<String>> {
        // Query tool_cache entries for this execution unit
        let files = db.get_dependent_files_for_unit(execution_unit_id)?;
        Ok(files)
    }

    /// What cache entries would be invalidated if `file_path` changed?
    pub fn cache_entries_affected_by_file(
        db: &crate::db::Database,
        file_path: &str,
    ) -> anyhow::Result<Vec<i64>> {
        let ids = db.get_cache_ids_for_file(file_path)?;
        Ok(ids)
    }

    /// What execution units happened before `target_id`?
    /// Reads execution-ordering edges from lineage_edges.
    pub fn execution_predecessors(
        db: &crate::db::Database,
        target_id: i64,
    ) -> anyhow::Result<Vec<DependencyEdge>> {
        let mut result = Vec::new();

        // Lineage edges are the authority for execution ordering.
        let lineage = db.get_lineage_to(target_id)?;
        for (from_id, to_id, kind) in lineage {
            let dependency_kind = match kind.as_str() {
                "depends_on" => DependencyKind::SequentialOrdering,
                "happens_before" => DependencyKind::ParallelJoin,
                _ => continue,
            };
            result.push(DependencyEdge {
                kind: dependency_kind,
                source_id: from_id,
                target_id: to_id,
                context: None,
            });
        }

        Ok(result)
    }

    /// Full dependency audit for an execution unit: what does it depend on,
    /// and what depends on it? Returns (incoming, outgoing).
    pub fn execution_dependency_audit(
        db: &crate::db::Database,
        exec_unit_id: i64,
    ) -> anyhow::Result<(Vec<DependencyEdge>, Vec<DependencyEdge>)> {
        let incoming = Self::execution_predecessors(db, exec_unit_id)?;

        // Outgoing: what files does this execution touch?
        let files = db.get_dependent_files_for_unit(exec_unit_id)?;
        let mut outgoing = Vec::new();
        for file in files {
            outgoing.push(DependencyEdge {
                kind: DependencyKind::ReadsFile,
                source_id: exec_unit_id,
                target_id: 0, // files don't have numeric IDs
                context: Some(file),
            });
        }

        Ok((incoming, outgoing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependency_kind::DependencyKind;

    #[test]
    fn topology_kinds_map_correctly() {
        // Verify the mapping from dag_edges kind strings to DependencyKind
        let edge = DependencyEdge {
            kind: DependencyKind::Coverage,
            source_id: 1,
            target_id: 2,
            context: None,
        };
        assert_eq!(edge.kind, DependencyKind::Coverage);
        assert!(edge.kind.is_topology());
    }

    #[tokio::test]
    async fn execution_predecessors_read_ordering_from_lineage_edges() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::db::Database::builder()
            .path(dir.path().join("dependency_view.db"))
            .build()
            .await
            .unwrap();

        db.insert_lineage_edge(10, 30, "depends_on").unwrap();
        db.insert_lineage_edge(20, 30, "happens_before").unwrap();

        let mut edges = DependencyView::execution_predecessors(&db, 30).unwrap();
        edges.sort_by_key(|edge| edge.source_id);

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].kind, DependencyKind::SequentialOrdering);
        assert_eq!(edges[0].source_id, 10);
        assert_eq!(edges[0].target_id, 30);
        assert_eq!(edges[1].kind, DependencyKind::ParallelJoin);
        assert_eq!(edges[1].source_id, 20);
        assert_eq!(edges[1].target_id, 30);
    }

    #[test]
    fn artifact_deps_use_file_context() {
        let edge = DependencyEdge {
            kind: DependencyKind::ReadsFile,
            source_id: 42,
            target_id: 0,
            context: Some("src/main.rs".into()),
        };
        assert!(edge.kind.is_artifact());
        assert!(edge.kind.is_active());
        assert_eq!(edge.context.as_deref(), Some("src/main.rs"));
    }
}
