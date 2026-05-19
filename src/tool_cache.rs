use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Tools that benefit most from caching — deterministic, expensive, frequently repeated.
const CACHEABLE_TOOLS: &[&str] = &[
    "grep", "search_content", "search_file", "search_code",
    "read_file", "read",
    "list_files", "ls", "tree",
    "symbol_search", "document_symbols", "workspace_symbol",
    "diagnostics", "diagnostic",
];

/// A tool call key: `hash(tool_name + normalized_args + workspace_hash)`.
/// Workspace hash captures relevant file state for invalidation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ToolCacheKey {
    pub tool_name: String,
    /// JSON-sorted normalized arguments.
    pub normalized_args: String,
    /// Hash of relevant file paths at time of execution (for invalidation).
    pub files_hash: String,
}

/// Cached tool execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCacheEntry {
    pub id: i64,
    pub tool_name: String,
    pub args_hash: String,
    /// JSON result from the tool execution.
    pub result: String,
    /// Files that this result depends on (for partial invalidation).
    pub dependent_files: Vec<String>,
    /// ISO-8601 timestamp.
    pub created_at: String,
    /// Number of cache hits.
    pub hit_count: i64,
}

/// Check if a tool is worth caching.
pub fn is_cacheable(tool_name: &str) -> bool {
    CACHEABLE_TOOLS.iter().any(|t| tool_name.contains(t) || t.contains(tool_name))
}

/// Normalize tool arguments for cache key generation.
/// Sorts JSON keys, strips whitespace-only differences.
pub fn normalize_args(args: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(args) {
        Ok(val) => serde_json::to_string(&val).unwrap_or_else(|_| args.to_string()),
        Err(_) => args.trim().to_lowercase(),
    }
}

/// Compute a cache key from tool name, normalized args, and workspace state.
pub fn cache_key(tool_name: &str, args: &str, workspace_files: &[String]) -> ToolCacheKey {
    use sha2::{Digest, Sha256};
    let normalized = normalize_args(args);

    // Hash the workspace file paths
    let mut fh = Sha256::new();
    let mut sorted_files: Vec<&String> = workspace_files.iter().collect();
    sorted_files.sort();
    for f in &sorted_files {
        fh.update(f.as_bytes());
    }
    let files_hash = hex::encode(&fh.finalize()[..8]);

    ToolCacheKey {
        tool_name: tool_name.to_string(),
        normalized_args: normalized.clone(),
        files_hash,
    }
}

/// Extract files mentioned in tool args (for partial invalidation).
/// E.g., grep "pattern" → args may contain file paths.
pub fn extract_dependent_files(tool_name: &str, args: &str) -> Vec<String> {
    let mut files = Vec::new();
    let arg_str = args.to_lowercase();

    // File-reading tools: extract paths from args
    if tool_name.contains("read") || tool_name.contains("grep") || tool_name.contains("search") {
        for word in arg_str.split_whitespace() {
            let clean = word.trim_matches(|c: char| c == '"' || c == '\'' || c == ',');
            if clean.contains('.')
                && !clean.starts_with("--")
                && !clean.starts_with('-')
                && clean.len() > 3
            {
                files.push(clean.to_string());
            }
        }
    }

    // Deduplicate and sort
    let seen: HashSet<String> = files.into_iter().collect();
    let mut result: Vec<String> = seen.into_iter().collect();
    result.sort();
    result
}

/// Invalidate cache entries whose dependent files overlap with changed_files.
pub fn should_invalidate(entry: &ToolCacheEntry, changed_files: &[String]) -> bool {
    if changed_files.is_empty() {
        return false;
    }
    let changed: HashSet<&str> = changed_files.iter().map(|s| s.as_str()).collect();
    entry.dependent_files.iter().any(|f| changed.contains(f.as_str()))
}

/// SQL migration for tool_cache table.
pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS tool_cache (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_name       TEXT NOT NULL,
        args_hash       TEXT NOT NULL,
        result          TEXT NOT NULL,
        dependent_files TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        hit_count       INTEGER NOT NULL DEFAULT 1
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_cache_key
        ON tool_cache(tool_name, args_hash);
    CREATE INDEX IF NOT EXISTS idx_tool_cache_hits
        ON tool_cache(hit_count DESC);";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sorts_json_keys() {
        let a = normalize_args(r#"{"pattern":"foo","path":"src"}"#);
        let b = normalize_args(r#"{"path":"src","pattern":"foo"}"#);
        assert_eq!(a, b, "JSON key order should not affect normalization");
    }

    #[test]
    fn is_cacheable_tools() {
        assert!(is_cacheable("grep"));
        assert!(is_cacheable("search_content"));
        assert!(is_cacheable("read_file"));
        assert!(is_cacheable("list_files"));
        assert!(!is_cacheable("execute_command"));
        assert!(!is_cacheable("unknown_tool"));
    }

    #[test]
    fn extract_dependent_files_from_args() {
        let files = extract_dependent_files("grep", r#"{"pattern":"foo","path":"src/main.rs"}"#);
        assert!(files.iter().any(|f| f.contains("src/main.rs")));
    }

    #[test]
    fn should_invalidate_when_dependency_changes() {
        let entry = ToolCacheEntry {
            id: 1,
            tool_name: "grep".into(),
            args_hash: "abc".into(),
            result: "found".into(),
            dependent_files: vec!["src/main.rs".into()],
            created_at: String::new(),
            hit_count: 1,
        };
        assert!(should_invalidate(&entry, &["src/main.rs".into()]));
        assert!(!should_invalidate(&entry, &["README.md".into()]));
        assert!(!should_invalidate(&entry, &[]));
    }
}
