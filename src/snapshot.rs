//! Execution snapshots — replay acceleration with budget-aware retention.
//!
//! Snapshots are replay accelerators, not the source of truth. The immutable
//! event log (`execution_events`, `tool_calls`, `tool_results`) is the true
//! source. Snapshots can be deleted — replay just becomes slower.
//!
//! # Retention Tiers
//!
//! | Tier | Name       | Purpose                     | Eviction     |
//! |------|------------|-----------------------------|--------------|
//! | 0    | Ephemeral  | Crash recovery, short replay| Ring buffer  |
//! | 1    | Structural | Planner state, memory ver,  | Per-exec cap |
//! |      |            | tool lifecycle, DAG pos     |              |
//! | 2    | Full       | Complete checkpoint (sparse)| Soft limit   |
//! | 3    | Frozen     | Enterprise immutable        | Never        |
//!
//! # Design invariants
//!
//! - Snapshots are append-only — never overwritten
//! - Snapshots reference `memory_version_id`, never embed the full graph
//! - Deletion is always safe — event log enables full replay
//! - Semantic boundaries trigger snapshots, not wall-clock time

use serde::Serialize;

/// Snapshot budget — prevents unbounded storage growth.
#[derive(Debug, Clone)]
pub struct SnapshotBudget {
    /// Hard limit on total snapshot storage (bytes). Default 100 MB.
    pub max_total_size_bytes: u64,
    /// Maximum L0 (ephemeral) snapshots before ring-buffer eviction.
    pub max_hot_snapshots: usize,
    /// Maximum L1 (structural) snapshots per execution.
    pub max_structural_per_execution: usize,
    /// Maximum L2 (full) snapshots total.
    pub max_full_snapshots: usize,
}

impl Default for SnapshotBudget {
    fn default() -> Self {
        Self {
            max_total_size_bytes: 100 * 1024 * 1024, // 100 MB
            max_hot_snapshots: 100,
            max_structural_per_execution: 3,
            max_full_snapshots: 10,
        }
    }
}

/// Snapshot tier determines retention policy and compaction behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotTier {
    /// L0 — ring-buffer eviction, short TTL
    Ephemeral = 0,
    /// L1 — per-execution cap, small size, high replay value
    Structural = 1,
    /// L2 — sparse, for long/high-value executions
    Full = 2,
    /// L3 — immutable, enterprise audit
    Frozen = 3,
}

impl SnapshotTier {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Structural,
            2 => Self::Full,
            3 => Self::Frozen,
            _ => Self::Ephemeral,
        }
    }
}

/// A stored snapshot record.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionSnapshot {
    pub id: i64,
    pub execution_id: i64,
    pub memory_version_id: i64,
    pub tier: i32,
    pub snapshot_data: String,
    pub size_bytes: i64,
    pub retention_ttl: Option<i64>, // seconds, NULL = permanent
    pub created_at: String,
}

/// A memory version record — one row per mutation point.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryVersion {
    pub id: i64,
    pub parent_version_id: Option<i64>,
    pub mutation_kind: String,
    pub mutation_desc: String,
    pub dag_root_id: Option<i64>,
    pub created_at: String,
}

pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS memory_versions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_version_id   INTEGER,
        mutation_kind       TEXT NOT NULL,
        mutation_desc       TEXT NOT NULL,
        dag_root_id         INTEGER,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS execution_snapshots (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_id        INTEGER NOT NULL,
        memory_version_id   INTEGER NOT NULL,
        tier                INTEGER NOT NULL DEFAULT 0,
        snapshot_data       TEXT NOT NULL,
        size_bytes          INTEGER NOT NULL DEFAULT 0,
        retention_ttl       INTEGER,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_snapshots_execution
        ON execution_snapshots(execution_id, tier);
    CREATE INDEX IF NOT EXISTS idx_snapshots_version
        ON execution_snapshots(memory_version_id);
    CREATE INDEX IF NOT EXISTS idx_memory_versions_parent
        ON memory_versions(parent_version_id);
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_are_reasonable() {
        let b = SnapshotBudget::default();
        assert!(b.max_total_size_bytes > 0);
        assert!(b.max_hot_snapshots > 0);
        assert_eq!(b.max_full_snapshots, 10);
    }

    #[test]
    fn tier_from_i32() {
        assert_eq!(SnapshotTier::from_i32(0), SnapshotTier::Ephemeral);
        assert_eq!(SnapshotTier::from_i32(1), SnapshotTier::Structural);
        assert_eq!(SnapshotTier::from_i32(2), SnapshotTier::Full);
        assert_eq!(SnapshotTier::from_i32(3), SnapshotTier::Frozen);
        assert_eq!(SnapshotTier::from_i32(99), SnapshotTier::Ephemeral);
    }
}
