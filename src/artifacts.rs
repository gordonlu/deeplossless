//! Artifact versioning and execution dependencies.
//! Foundation for cache invalidation correctness — not just file-level,
//! but execution-to-artifact dependency tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Artifact Version ──────────────────────────────────────────────────

/// A versioned artifact — not just a file path, but path + content identity.
/// mtime alone is unreliable for cache correctness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ArtifactVersion {
    /// File path (relative to repo root).
    pub path: String,

    /// SHA-256 of file contents. Primary identity signal.
    pub content_hash: String,

    /// ISO-8601 timestamp of last observed modification.
    pub observed_at: String,
}

impl ArtifactVersion {
    /// Create from path + content. Content hash identifies version.
    pub fn new(path: &str, content: &str) -> Self {
        use sha2::{Digest, Sha256};
        let hash = hex::encode(&Sha256::digest(content.as_bytes())[..8]);
        Self {
            path: path.to_string(),
            content_hash: hash,
            observed_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// True if two versions refer to the same content (can reuse cache).
    pub fn is_same_content(&self, other: &ArtifactVersion) -> bool {
        self.path == other.path && self.content_hash == other.content_hash
    }
}

// ── Execution Dependency ──────────────────────────────────────────────

/// How an execution node depends on an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyKind {
    /// The execution read the artifact content.
    Read,
    /// The execution searched the artifact (grep, search).
    Search,
    /// The execution parsed the artifact (tree-sitter, AST).
    Parse,
    /// The artifact was a build input (Cargo.toml, Makefile).
    BuildInput,
    /// The artifact was an output of the execution.
    Produced,
}

/// An edge from an execution to an artifact version.
/// When the artifact changes, all dependent executions are invalidated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// The canonical execution key that depends on this artifact.
    pub execution_key: String,
    /// The artifact version at time of execution.
    pub artifact: ArtifactVersion,
    /// How the execution used the artifact.
    pub kind: DependencyKind,
}

/// Reverse index: artifact path → set of execution keys that depend on it.
/// Enables O(affected) invalidation.
#[derive(Debug, Clone, Default)]
pub struct DependencyIndex {
    /// artifact_path → [{execution_key, artifact_version, kind}]
    pub edges: HashMap<String, Vec<DependencyEdge>>,
}

impl DependencyIndex {
    pub fn new() -> Self { Self { edges: HashMap::new() } }

    /// Record that an execution depends on an artifact at a specific version.
    pub fn record(&mut self, exec_key: &str, artifact: ArtifactVersion, kind: DependencyKind) {
        self.edges
            .entry(artifact.path.clone())
            .or_default()
            .push(DependencyEdge { execution_key: exec_key.to_string(), artifact, kind });
    }

    /// Invalidate all executions that depend on a changed artifact.
    /// Returns the set of execution keys that should be invalidated.
    /// `is_stale` checks if the current version differs from the recorded version.
    pub fn invalidate(&mut self, new_version: &ArtifactVersion) -> Vec<String> {
        let mut invalidated = Vec::new();
        if let Some(edges) = self.edges.get_mut(&new_version.path) {
            // Keep only non-matching versions; remove matching ones (still valid)
            edges.retain(|e| {
                if e.artifact.is_same_content(new_version) {
                    false // same content, still valid
                } else {
                    invalidated.push(e.execution_key.clone());
                    true // keep for tracking but mark as invalidated
                }
            });
        }
        invalidated
    }

    /// Get all execution keys that depend on a given artifact path.
    pub fn dependencies_of(&self, path: &str) -> Vec<&DependencyEdge> {
        self.edges.get(path).map(|v| v.iter().collect()).unwrap_or_default()
    }

    /// Remove all dependencies for an execution key (when invalidated).
    pub fn remove(&mut self, exec_key: &str) {
        for edges in self.edges.values_mut() {
            edges.retain(|e| e.execution_key != exec_key);
        }
    }
}

// ── SQL migration ─────────────────────────────────────────────────────

pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS artifact_versions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        path        TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        observed_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_artifact_path ON artifact_versions(path);

    CREATE TABLE IF NOT EXISTS dependency_edges (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_key   TEXT NOT NULL,
        artifact_path   TEXT NOT NULL,
        kind            TEXT NOT NULL DEFAULT 'read',
        recorded_at     TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_dep_exec ON dependency_edges(execution_key);
    CREATE INDEX IF NOT EXISTS idx_dep_artifact ON dependency_edges(artifact_path);";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_content_version_matches() {
        let v1 = ArtifactVersion::new("src/main.rs", "fn main() {}");
        let v2 = ArtifactVersion::new("src/main.rs", "fn main() {}");
        assert!(v1.is_same_content(&v2));
    }

    #[test]
    fn different_content_does_not_match() {
        let v1 = ArtifactVersion::new("src/main.rs", "fn main() {}");
        let v2 = ArtifactVersion::new("src/main.rs", "fn main() { todo!() }");
        assert!(!v1.is_same_content(&v2));
    }

    #[test]
    fn dependency_index_invalidates_on_content_change() {
        let mut idx = DependencyIndex::new();
        let v1 = ArtifactVersion::new("Cargo.toml", "[dependencies]\ntokio = \"1\"");
        idx.record("exec_grep_tokio", v1, DependencyKind::Search);

        let v2 = ArtifactVersion::new("Cargo.toml", "[dependencies]\ntokio = \"1.42\"");
        let invalidated = idx.invalidate(&v2);
        assert!(!invalidated.is_empty(), "should invalidate on content change");
    }

    #[test]
    fn dependency_index_keeps_same_content() {
        let mut idx = DependencyIndex::new();
        let v1 = ArtifactVersion::new("Cargo.toml", "[dependencies]\ntokio = \"1\"");
        idx.record("exec_grep_tokio", v1, DependencyKind::Search);

        let v2 = ArtifactVersion::new("Cargo.toml", "[dependencies]\ntokio = \"1\"");
        let invalidated = idx.invalidate(&v2);
        assert!(invalidated.is_empty(), "same content should not invalidate");
    }

    #[test]
    fn remove_cleans_edges() {
        let mut idx = DependencyIndex::new();
        let v1 = ArtifactVersion::new("src/main.rs", "fn main() {}");
        idx.record("exec_grep", v1, DependencyKind::Read);
        idx.remove("exec_grep");
        assert!(idx.dependencies_of("src/main.rs").is_empty());
    }
}
