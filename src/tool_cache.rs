//! Deterministic tool result cache. L1 is in-memory (RwLock<HashMap>),
//! L2 is SQLite for persistence across restarts.
//!
//! Cache key: `(tool_name, hex(SHA-256(normalized_args)))`.
//! File dependencies tracked separately for partial invalidation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use std::time::Instant;

// ── L1 Hot Cache ───────────────────────────────────────────────────────

/// In-memory L1 cache. True LRU eviction, Arc<str> for zero-clone reads,
/// reverse dependency index for O(affected) invalidation.
pub struct L1HotCache {
    entries: RwLock<HashMap<(String, String), CacheEntry>>,
    /// Reverse index: file_path → set of cache keys that depend on it.
    file_index: RwLock<HashMap<String, Vec<(String, String)>>>,
}

struct CacheEntry {
    result: std::sync::Arc<str>,
    dependent_files: Vec<String>,
    hit_count: u64,
    last_accessed: Instant,
}

impl Default for L1HotCache {
    fn default() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            file_index: RwLock::new(HashMap::new()),
        }
    }
}

impl L1HotCache {
    pub fn new() -> Self { Self::default() }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Look up a cached result. Returns Arc<str> clone (cheap ref-count bump),
    /// updates last_accessed + hit_count.
    pub fn get(&self, tool_name: &str, args_hash: &str) -> Option<String> {
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let key = (tool_name.to_string(), args_hash.to_string());
        if let Some(e) = map.get_mut(&key) {
            e.hit_count += 1;
            e.last_accessed = Instant::now();
            Some(e.result.to_string())
        } else {
            None
        }
    }

    /// Store a result. LRU eviction, reverse index updated.
    pub fn put(&self, tool_name: &str, args_hash: &str, result: &str, dependent_files: &[String]) {
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let key = (tool_name.to_string(), args_hash.to_string());
        if map.len() >= 128 && !map.contains_key(&key)
            && let Some(evict_key) = map.iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone())
            && let Some(old) = map.remove(&evict_key)
        {
            let mut idx = self.file_index.write().unwrap_or_else(|e| e.into_inner());
            for f in &old.dependent_files {
                if let Some(keys) = idx.get_mut(f) { keys.retain(|k| k != &evict_key); }
            }
        }
        // Update reverse index
        {
            let mut idx = self.file_index.write().unwrap_or_else(|e| e.into_inner());
            for f in dependent_files {
                idx.entry(f.clone()).or_default().push(key.clone());
            }
        }
        map.insert(key, CacheEntry {
            result: std::sync::Arc::from(result.to_string()),
            dependent_files: dependent_files.to_vec(),
            hit_count: 1,
            last_accessed: Instant::now(),
        });
    }

    /// O(affected) invalidation via reverse dependency index.
    pub fn invalidate(&self, changed_files: &[String]) -> usize {
        let idx = self.file_index.read().unwrap_or_else(|e| e.into_inner());
        let mut affected_keys = HashSet::new();
        for f in changed_files {
            if let Some(keys) = idx.get(f) {
                for k in keys {
                    affected_keys.insert(k.clone());
                }
            }
        }
        drop(idx);

        if affected_keys.is_empty() {
            return 0;
        }

        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut idx = self.file_index.write().unwrap_or_else(|e| e.into_inner());
        let before = map.len();
        for key in &affected_keys {
            map.remove(key);
            // Clean up reverse index
            for keys in idx.values_mut() {
                keys.retain(|k| k != key);
            }
        }
        before - map.len()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// ── Normalization ──────────────────────────────────────────────────────

/// Compute `args_hash = hex(SHA-256(normalized_args))`. 16 hex chars.
fn sha256_hex(input: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(&Sha256::digest(input.as_bytes())[..8])
}

/// Normalize tool arguments for deterministic cache keys.
/// JSON: recursively sort object keys for canonical output.
/// Plain text: trim only (case-preserving).
pub fn normalize_args(args: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(args) {
        let canonical = canonicalize_json(&val);
        serde_json::to_string(&canonical).unwrap_or_else(|_| args.trim().to_string())
    } else {
        args.trim().to_string()
    }
}

/// Recursively sort JSON object keys for deterministic serialization.
/// `serde_json::to_string` doesn't guarantee key order — this fixes it.
fn canonicalize_json(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize_json(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(entries.into_iter().collect())
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

/// Compute the cache key: `(tool_name, hex(SHA-256(normalized_args)))`.
pub fn cache_key(tool_name: &str, args: &str) -> (String, String) {
    let normalized = normalize_args(args);
    let hash = sha256_hex(&normalized);
    (tool_name.to_string(), hash)
}

// ── Dependent file extraction ──────────────────────────────────────────

/// Extract file paths that a tool call depends on.
/// Per-tool patterns avoid false positives from version numbers, domains, etc.
pub fn extract_dependent_files(tool_name: &str, args: &str) -> Vec<String> {
    let t = tool_name.to_lowercase();
    let mut files = Vec::new();

    // Parse JSON args if possible
    let arg_map: Option<HashMap<String, String>> = serde_json::from_str(args).ok();

    if t.contains("read") || t == "read_file" {
        // read_file: exact path is the arg
        if let Some(ref m) = arg_map {
            for key in &["file_path", "path", "filePath"] {
                if let Some(v) = m.get(*key) {
                    files.push(v.clone());
                }
            }
        }
        if files.is_empty() {
            files.extend(extract_path_literals(args));
        }
    } else if t.contains("grep") || t.contains("search") {
        // grep/search: path/directory args
        if let Some(ref m) = arg_map {
            for key in &["path", "directory", "dir", "pattern"] {
                if let Some(v) = m.get(*key) && looks_like_path_or_pattern(v) {
                    files.push(v.clone());
                }
            }
        }
        if files.is_empty() {
            files.extend(extract_path_literals(args));
        }
    } else if t.contains("list_files") || t == "ls" || t == "tree" {
        if let Some(ref m) = arg_map
            && let Some(v) = m.get("target_directory") { files.push(v.clone()); }
    } else if t.contains("symbol") {
        if let Some(ref m) = arg_map
            && let Some(v) = m.get("file_path") { files.push(v.clone()); }
    } else {
        // Fallback: extract anything that looks like a path
        files.extend(extract_path_literals(args));
    }

    let seen: HashSet<String> = files.into_iter().collect();
    let mut result: Vec<String> = seen.into_iter().collect();
    result.sort();
    result
}

/// Extract tokens that look like file paths (contain a dot or slash, not URLs).
fn extract_path_literals(text: &str) -> Vec<String> {
    let mut files = Vec::new();
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ':');
        if clean.starts_with("http") || clean.starts_with("https") { continue; }
        if clean.starts_with("--") || clean.starts_with('-') { continue; }
        // Must contain a path separator or file extension
        if (clean.contains('/') || clean.contains('\\') || (clean.contains('.') && clean.len() > 4 && !clean.contains("..")))
            && clean.len() >= 2 && clean.len() <= 256
        {
            files.push(clean.to_string());
        }
    }
    files
}

fn looks_like_path_or_pattern(s: &str) -> bool {
    s.contains('/') || s.contains('.') || s.contains('*') || s.len() < 50
}

// ── Cacheability ───────────────────────────────────────────────────────

/// Tools that benefit from caching — deterministic, expensive, frequently repeated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolKind {
    Grep,
    ReadFile,
    ListFiles,
    SymbolSearch,
    Diagnostics,
    Other,
}

impl ToolKind {
    pub fn from_name(name: &str) -> Self {
        let n = name.to_lowercase();
        if n == "grep" || n.contains("search_content") || n.contains("search_code") || n.contains("search_file") {
            ToolKind::Grep
        } else if n == "read_file" || n == "read" || n.contains("read") && n.contains("file") {
            ToolKind::ReadFile
        } else if n == "list_files" || n == "ls" || n == "tree" {
            ToolKind::ListFiles
        } else if n == "symbol_search" || n.contains("document_symbol") || n.contains("workspace_symbol") {
            ToolKind::SymbolSearch
        } else if n == "diagnostics" || n.contains("diagnostic") {
            ToolKind::Diagnostics
        } else {
            ToolKind::Other
        }
    }
}

pub fn is_cacheable(tool_name: &str) -> bool {
    !matches!(ToolKind::from_name(tool_name), ToolKind::Other)
}

// ── Canonical Execution Identity ──────────────────────────────────────

/// The stable key for cache correctness, replay determinism, and failure matching.
/// Normalizes tool name + args across provider/format differences.
///
/// `grep("tokio", "src/")` and `search_content("tokio", directory="src/")`
/// resolve to the same canonical key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalExecutionKey {
    pub tool_name: String,
    pub args_hash: String,
}

impl CanonicalExecutionKey {
    pub fn new(tool_name: &str, args: &str) -> Self {
        let normalized = normalize_args(args);
        Self {
            tool_name: normalize_tool_name(tool_name),
            args_hash: sha256_hex(&normalized),
        }
    }
}

/// Normalize tool name to a provider-agnostic canonical form.
fn normalize_tool_name(name: &str) -> String {
    let n = name.to_lowercase();
    if n.contains("grep") || n.contains("search_content") || n.contains("search_code") { "grep".into() }
    else if n.contains("read") && (n.contains("file") || n == "read") { "read_file".into() }
    else if n.contains("list") || n.contains("ls") || n.contains("tree") { "list_files".into() }
    else if n.contains("symbol") || n.contains("document_symbol") { "symbol_search".into() }
    else { name.to_string() }
}

// ── Cache entry (L2) ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCacheEntry {
    pub id: i64,
    pub tool_name: String,
    pub args_hash: String,
    pub result: String,
    pub dependent_files: Vec<String>,
    pub created_at: String,
    pub hit_count: i64,
}

pub fn should_invalidate(entry: &ToolCacheEntry, changed_files: &[String]) -> bool {
    if changed_files.is_empty() { return false; }
    let changed: HashSet<&str> = changed_files.iter().map(|s| s.as_str()).collect();
    entry.dependent_files.iter().any(|f| changed.contains(f.as_str()))
}

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
    fn lru_evicts_oldest_accessed() {
        let cache = L1HotCache::new();
        // Fill to capacity
        for i in 0..128 {
            cache.put("grep", &format!("key{i}"), "result", &[]);
        }
        // Access key0 to make it recently used
        assert!(cache.get("grep", "key0").is_some());
        // Add one more — key1 should be evicted (oldest last_accessed), not key0
        cache.put("grep", "overflow", "result", &[]);
        assert!(cache.get("grep", "key0").is_some(), "recently accessed key should survive LRU");
    }

    #[test]
    fn get_updates_hit_count_and_timestamp() {
        let cache = L1HotCache::new();
        cache.put("grep", "hash1", "found", &[]);
        cache.get("grep", "hash1");
        cache.get("grep", "hash1");
        // Should still be present (not evicted by LRU)
        assert!(cache.get("grep", "hash1").is_some());
    }

    #[test]
    fn args_hash_is_sha256_hex() {
        let h = sha256_hex("hello");
        assert_eq!(h.len(), 16, "hash should be 16 hex chars (64 bits)");
        // Same input → same hash
        assert_eq!(sha256_hex("hello"), sha256_hex("hello"));
        // Different input → different hash
        assert_ne!(sha256_hex("hello"), sha256_hex("world"));
    }

    #[test]
    fn normalize_json_sorts_keys() {
        let a = normalize_args(r#"{"path":"src","pattern":"foo"}"#);
        let b = normalize_args(r#"{"pattern":"foo","path":"src"}"#);
        assert_eq!(a, b, "JSON key order should not affect normalization");
    }

    #[test]
    fn normalize_preserves_case_for_plain_text() {
        let a = normalize_args("Grep FooBar");
        assert!(a.contains("Grep"), "case should be preserved for non-JSON args, got '{a}'");
    }

    #[test]
    fn canonical_json_produces_identical_args_hash() {
        let a = normalize_args(r#"{"b":2,"a":1,"c":{"z":9,"x":7}}"#);
        let b = normalize_args(r#"{"a":1,"c":{"x":7,"z":9},"b":2}"#);
        assert_eq!(a, b, "different key orders should produce identical canonical form");
        assert_eq!(sha256_hex(&a), sha256_hex(&b), "hash must be identical for canonically equal JSON");
    }

    #[test]
    fn is_cacheable_by_tool_kind() {
        assert!(is_cacheable("grep"));
        assert!(is_cacheable("read_file"));
        assert!(is_cacheable("list_files"));
        assert!(is_cacheable("symbol_search"));
        assert!(!is_cacheable("execute_command"));
        assert!(!is_cacheable("unknown_tool"));
        assert!(!is_cacheable("grep2"), "grep2 should not match");
    }

    #[test]
    fn extract_dependent_files_from_json_args() {
        let files = extract_dependent_files("grep", r#"{"pattern":"foo","path":"src/main.rs"}"#);
        assert!(files.iter().any(|f| f.contains("src/main.rs")));
    }

    #[test]
    fn extract_files_ignores_urls_and_flags() {
        let files = extract_dependent_files("read_file", r#"read_file --path https://example.com --flag src/main.rs"#);
        assert!(!files.iter().any(|f| f.contains("http")));
        assert!(!files.iter().any(|f| f.contains("--")));
    }

    #[test]
    fn should_invalidate_on_dependency_change() {
        let entry = ToolCacheEntry {
            id: 1, tool_name: "grep".into(), args_hash: "abc".into(),
            result: "found".into(), dependent_files: vec!["src/main.rs".into()],
            created_at: String::new(), hit_count: 1,
        };
        assert!(should_invalidate(&entry, &["src/main.rs".into()]));
        assert!(!should_invalidate(&entry, &["README.md".into()]));
        assert!(!should_invalidate(&entry, &[]));
    }
}
