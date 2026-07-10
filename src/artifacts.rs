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

/// Section-level hash for structured files (TOML sections, Rust modules, etc.).
/// Avoids invalidating ALL cache for a file when only one section changes.
pub fn hash_sections(path: &str, content: &str) -> Vec<(String, String)> {
    use sha2::{Digest, Sha256};

    if path == "Cargo.toml" || path.ends_with(".toml") {
        // TOML: hash each [section] independently
        let mut sections = Vec::new();
        let mut current_section = "[root]".to_string();
        let mut current_content = String::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                if !current_content.trim().is_empty() {
                    let h = hex::encode(&Sha256::digest(current_content.as_bytes())[..8]);
                    sections.push((current_section.clone(), h));
                }
                current_section = trimmed.to_string();
                current_content = String::new();
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }
        if !current_content.trim().is_empty() {
            let h = hex::encode(&Sha256::digest(current_content.as_bytes())[..8]);
            sections.push((current_section, h));
        }
        sections
    } else {
        // Default: whole-file hash
        let h = hex::encode(&Sha256::digest(content.as_bytes())[..8]);
        vec![(path.to_string(), h)]
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
/// When the artifact changes, the edge is marked dirty (lazy invalidation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// The canonical execution key that depends on this artifact.
    pub execution_key: String,
    /// The artifact version at time of execution.
    pub artifact: ArtifactVersion,
    /// How the execution used the artifact.
    pub kind: DependencyKind,
    /// Whether this edge is stale (artifact changed since execution).
    #[serde(default)]
    pub dirty: bool,
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
            .push(DependencyEdge { execution_key: exec_key.to_string(), artifact, kind, dirty: false });
    }

    /// Mark dependent executions as dirty when an artifact changes.
    /// Returns the set of execution keys that are now dirty.
    /// Does NOT cascade — dirty propagation is lazy (validate on next access).
    pub fn mark_dirty(&mut self, new_version: &ArtifactVersion) -> Vec<String> {
        let mut dirty = Vec::new();
        if let Some(edges) = self.edges.get_mut(&new_version.path) {
            for e in edges.iter_mut() {
                if !e.artifact.is_same_content(new_version) {
                    e.dirty = true;
                    dirty.push(e.execution_key.clone());
                }
            }
        }
        dirty
    }

    /// Check if an execution key is dirty (stale). Used for lazy validation.
    /// Returns true if any dependency of this execution has changed.
    pub fn is_dirty(&self, exec_key: &str) -> bool {
        self.edges.values().any(|edges| {
            edges.iter().any(|e| e.execution_key == exec_key && e.dirty)
        })
    }

    /// Validate (clean) an execution key — called when the execution is re-run
    /// and produces the same result. Removes dirty flag.
    pub fn validate(&mut self, exec_key: &str) {
        for edges in self.edges.values_mut() {
            for e in edges.iter_mut() {
                if e.execution_key == exec_key {
                    e.dirty = false;
                }
            }
        }
    }

    /// Invalidate all executions that depend on a changed artifact.
    /// Returns the set of execution keys that should be invalidated.
    /// Keeps edges for tracking but marks as stale.
    pub fn invalidate(&mut self, new_version: &ArtifactVersion) -> Vec<String> {
        self.mark_dirty(new_version)
    }

    /// Compact the dependency index: remove edges that are clean and
    /// from executions older than `max_age_versions` versions.
    pub fn compact(&mut self, max_age_versions: usize) {
        for edges in self.edges.values_mut() {
            let len = edges.len();
            edges.retain(|e| e.dirty || len <= max_age_versions);
        }
        self.edges.retain(|_, v| !v.is_empty());
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

// ── Execution Artifact ─────────────────────────────────────────────────

/// A complete execution artifact — the record of a tool execution.
///
/// Artifact = input + dependency snapshot + output + environment context.
/// This upgrades the tool cache from a simple key-value store to a full
/// execution artifact system, supporting replay, dependency validation,
/// and cache correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionArtifact {
    pub id: i64,
    /// Canonical execution key: tool_name + args_hash (for dedup/lookup).
    pub execution_key: String,
    /// Normalized tool name.
    pub tool_name: String,
    /// Full input arguments text (not just the hash).
    pub args: String,
    /// SHA-256 hash of arguments (for backward-compatible lookup).
    pub args_hash: String,
    /// Tool output / execution result.
    pub result: String,
    /// Files this execution depended on.
    pub dependent_files: Vec<String>,
    /// Content hashes of dependent files at execution time.
    pub file_hashes: Vec<String>,
    /// Runtime environment: model + provider + tool versions.
    pub environment_fingerprint: String,
    /// ISO-8601 timestamp of creation.
    pub created_at: String,
    /// Number of times this artifact has been reused (cache hits).
    pub hit_count: i64,
}

impl ExecutionArtifact {
    /// Create from execution components.
    pub fn new(
        execution_key: &str,
        tool_name: &str,
        args: &str,
        args_hash: &str,
        result: &str,
        dependent_files: Vec<String>,
        file_hashes: Vec<String>,
        environment_fingerprint: &str,
    ) -> Self {
        Self {
            id: 0,
            execution_key: execution_key.to_string(),
            tool_name: tool_name.to_string(),
            args: args.to_string(),
            args_hash: args_hash.to_string(),
            result: result.to_string(),
            dependent_files,
            file_hashes,
            environment_fingerprint: environment_fingerprint.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            hit_count: 0,
        }
    }

    /// Record a dependency edge for each dependent file.
    /// Returns the edges to persist.
    pub fn dependency_edges(&self) -> Vec<(String, ArtifactVersion, DependencyKind)> {
        self.dependent_files.iter().zip(self.file_hashes.iter())
            .map(|(path, hash)| {
                let av = ArtifactVersion {
                    path: path.clone(),
                    content_hash: hash.clone(),
                    observed_at: self.created_at.clone(),
                };
                (self.execution_key.clone(), av, DependencyKind::Read)
            })
            .collect()
    }
}

/// SQL migration for execution_artifacts table.
pub const ARTIFACT_MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS execution_artifacts (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_key           TEXT NOT NULL,
        tool_name               TEXT NOT NULL,
        args                    TEXT NOT NULL DEFAULT '',
        args_hash               TEXT NOT NULL,
        result                  TEXT NOT NULL,
        dependent_files         TEXT NOT NULL DEFAULT '[]',
        file_hashes             TEXT NOT NULL DEFAULT '[]',
        environment_fingerprint TEXT NOT NULL DEFAULT '',
        created_at              TEXT NOT NULL DEFAULT (datetime('now')),
        hit_count               INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_artifact_exec_key ON execution_artifacts(execution_key);
    CREATE INDEX IF NOT EXISTS idx_artifact_tool_hash ON execution_artifacts(tool_name, args_hash);";

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

    #[test]
    fn execution_artifact_creates_dependency_edges() {
        let artifact = ExecutionArtifact::new(
            "grep:abc123", "grep", "search foo",
            "abc123", "found 3 results",
            vec!["src/main.rs".to_string()],
            vec!["hash1".to_string()],
            "deepseek-chat",
        );
        let edges = artifact.dependency_edges();
        assert!(!edges.is_empty(), "should create an edge for each dependent file");
        assert_eq!(edges[0].0, "grep:abc123", "execution_key should match");
        assert_eq!(edges[0].1.path, "src/main.rs", "should reference the dependent file");
        assert_eq!(edges[0].2, DependencyKind::Read, "default kind should be Read");
    }

    #[test]
    fn execution_artifact_roundtrip_fields() {
        let artifact = ExecutionArtifact::new(
            "read_file:def456", "read_file", "/path/to/file.txt",
            "def456", "file contents here",
            vec!["/path/to/file.txt".to_string()],
            vec!["hash2".to_string()],
            "deepseek-chat",
        );
        assert_eq!(artifact.tool_name, "read_file");
        assert_eq!(artifact.args, "/path/to/file.txt");
        assert_eq!(artifact.result, "file contents here");
        assert_eq!(artifact.hit_count, 0);
        assert!(artifact.execution_key.contains("def456"));
    }
}
