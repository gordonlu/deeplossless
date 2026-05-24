//! Execution snapshots — replay acceleration with budget-aware retention.
//!
//! Snapshots are replay accelerators, not the source of truth. The immutable
//! event log (`execution_events`) is the true source. Snapshots can be deleted
//! — replay just becomes slower.
//!
//! # Design invariants
//!
//! - Snapshots are append-only — never overwritten
//! - Snapshots reference `memory_version_id`, never embed the full graph
//! - Deletion is always safe — event log enables full replay
//! - Semantic boundaries trigger snapshots, not wall-clock time
//! - `SnapshotTier::from_i32()` fails on unknown values — no silent downgrade

use serde::{Deserialize, Serialize};

/// Current snapshot schema version. Bump when snapshot payload format changes.
pub const SCHEMA_VERSION: i32 = 1;

/// Snapshot budget — prevents unbounded storage growth.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            max_total_size_bytes: 100 * 1024 * 1024,
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
    /// Convert an i32 to a SnapshotTier. Returns an error for unknown values
    /// instead of silently downgrading.
    pub fn from_i32(v: i32) -> anyhow::Result<Self> {
        match v {
            0 => Ok(Self::Ephemeral),
            1 => Ok(Self::Structural),
            2 => Ok(Self::Full),
            3 => Ok(Self::Frozen),
            other => anyhow::bail!("invalid snapshot tier: {other}"),
        }
    }
}

/// Typed snapshot payload. Each tier stores an appropriate representation
/// rather than duplicating the full event log as a giant JSON blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotPayload {
    /// L0: lightweight state for crash recovery and short replay.
    /// Stores only the last event seq_no and event count.
    Ephemeral {
        last_seq_no: i64,
        event_count: u32,
    },
    /// L1: structured memory topology + execution state.
    /// A compressed projection of the relevant event subsequence.
    Structural {
        events: Vec<(i64, serde_json::Value)>,
    },
    /// L2: full checkpoint with serialized state.
    /// Sparse — for long/high-value executions.
    Full {
        events: Vec<(i64, serde_json::Value)>,
    },
    /// L3: immutable audit record.
    Frozen {
        events: Vec<(i64, serde_json::Value)>,
    },
}

impl SnapshotPayload {
    /// Number of events stored in this payload.
    pub fn event_count(&self) -> usize {
        match self {
            Self::Ephemeral { .. } => 0,
            Self::Structural { events } | Self::Full { events } | Self::Frozen { events } => {
                events.len()
            }
        }
    }

    /// Last seq_no stored in this payload.
    pub fn last_seq_no(&self) -> i64 {
        match self {
            Self::Ephemeral { last_seq_no, .. } => *last_seq_no,
            Self::Structural { events } | Self::Full { events } | Self::Frozen { events } => {
                events.last().map(|(s, _)| *s).unwrap_or(0)
            }
        }
    }
}

/// A stored snapshot record with continuity verification fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    pub id: i64,
    pub execution_id: i64,
    pub memory_version_id: i64,
    pub schema_version: i32,
    pub tier: i32,
    pub snapshot_data: String,
    pub last_event_seq_no: i64,
    pub boundary_hash: String,
    pub integrity_hash: String,
    pub size_bytes: i64,
    pub retention_ttl: Option<i64>,
    pub created_at: String,
}

impl ExecutionSnapshot {
    /// Parse the typed payload from the opaque JSON string.
    pub fn payload(&self) -> anyhow::Result<Option<SnapshotPayload>> {
        if self.snapshot_data.is_empty() || self.snapshot_data == "{}" {
            return Ok(None);
        }
        Ok(serde_json::from_str(&self.snapshot_data).ok())
    }
}

/// A memory version record — one row per mutation point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVersion {
    pub id: i64,
    pub parent_version_id: Option<i64>,
    pub mutation_kind: String,
    pub mutation_desc: String,
    pub dag_root_id: Option<i64>,
    pub created_at: String,
}

/// Compute a chain integrity hash for an event sequence.
/// Uses SHA-256 over the concatenation of (seq_no || payload).
pub fn compute_chain_hash(events: &[(i64, &str)]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    for (seq, payload) in events {
        hasher.update(seq.to_le_bytes());
        hasher.update(payload.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Compute boundary hash — hash of the last N events for continuity checking.
pub fn compute_boundary_hash(events: &[(i64, &str)], n: usize) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    let start = events.len().saturating_sub(n);
    for (seq, payload) in &events[start..] {
        hasher.update(seq.to_le_bytes());
        hasher.update(payload.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// ALTER migration for v0.6.1: adds schema versioning and integrity fields.
/// Run only if `schema_version` column does not exist (checked in db.rs init).
pub const ALTER_MIGRATION: &str = "
    ALTER TABLE execution_snapshots ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE execution_snapshots ADD COLUMN last_event_seq_no INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE execution_snapshots ADD COLUMN boundary_hash TEXT NOT NULL DEFAULT '';
    ALTER TABLE execution_snapshots ADD COLUMN integrity_hash TEXT NOT NULL DEFAULT '';
";

pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS memory_versions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_version_id   INTEGER REFERENCES memory_versions(id),
        mutation_kind       TEXT NOT NULL,
        mutation_desc       TEXT NOT NULL,
        dag_root_id         INTEGER,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS execution_snapshots (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_id        INTEGER NOT NULL,
        memory_version_id   INTEGER NOT NULL,
        schema_version      INTEGER NOT NULL DEFAULT 1,
        tier                INTEGER NOT NULL DEFAULT 0,
        snapshot_data       TEXT NOT NULL,
        last_event_seq_no   INTEGER NOT NULL DEFAULT 0,
        boundary_hash       TEXT NOT NULL DEFAULT '',
        integrity_hash      TEXT NOT NULL DEFAULT '',
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
        assert_eq!(b.max_total_size_bytes, 100 * 1024 * 1024); // 100 MB
        assert_eq!(b.max_hot_snapshots, 100);
        assert_eq!(b.max_structural_per_execution, 3);
        assert_eq!(b.max_full_snapshots, 10);
    }

    #[test]
    fn tier_from_i32_valid() {
        assert_eq!(SnapshotTier::from_i32(0).unwrap(), SnapshotTier::Ephemeral);
        assert_eq!(SnapshotTier::from_i32(1).unwrap(), SnapshotTier::Structural);
        assert_eq!(SnapshotTier::from_i32(2).unwrap(), SnapshotTier::Full);
        assert_eq!(SnapshotTier::from_i32(3).unwrap(), SnapshotTier::Frozen);
    }

    #[test]
    fn tier_from_i32_invalid_errors() {
        assert!(SnapshotTier::from_i32(-1).is_err());
        assert!(SnapshotTier::from_i32(4).is_err());
        assert!(SnapshotTier::from_i32(99).is_err());
    }

    #[test]
    fn payload_ephemeral_round_trip() {
        let p = SnapshotPayload::Ephemeral { last_seq_no: 42, event_count: 10 };
        let json = serde_json::to_string(&p).unwrap();
        let back: SnapshotPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.last_seq_no(), 42);
        assert_eq!(back.event_count(), 0);
    }

    #[test]
    fn chain_hash_deterministic() {
        let events = vec![(1_i64, r#"{"type":"text_delta","text":"hello"}"#)];
        let h1 = compute_chain_hash(&events);
        let h2 = compute_chain_hash(&events);
        assert_eq!(h1, h2);
    }

    #[test]
    fn chain_hash_differs_for_different_events() {
        let a = vec![(1_i64, r#"{"type":"text_delta","text":"hello"}"#)];
        let b = vec![(1_i64, r#"{"type":"text_delta","text":"world"}"#)];
        assert_ne!(compute_chain_hash(&a), compute_chain_hash(&b));
    }

    #[test]
    fn boundary_hash_with_n_events() {
        let events = vec![
            (1_i64, r#"{"type":"text_delta","text":"a"}"#),
            (2_i64, r#"{"type":"text_delta","text":"b"}"#),
            (3_i64, r#"{"type":"text_delta","text":"c"}"#),
        ];
        let h2 = compute_boundary_hash(&events, 2);
        let h3 = compute_boundary_hash(&events, 3);
        assert_ne!(h2, h3);
    }

    #[test]
    fn payload_event_count() {
        let events = vec![(1_i64, serde_json::json!({}))];
        let p = SnapshotPayload::Full { events };
        assert_eq!(p.event_count(), 1);
        assert_eq!(p.last_seq_no(), 1);
    }

    #[test]
    fn empty_payload_last_seq_no_is_zero() {
        let p = SnapshotPayload::Ephemeral { last_seq_no: 0, event_count: 0 };
        assert_eq!(p.last_seq_no(), 0);
        assert_eq!(p.event_count(), 0);
    }
}
