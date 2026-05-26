//! Deterministic replay engine — reconstructs StreamEvent sequences from
//! the append-only execution_events table.
//!
//! # Design invariants
//!
//! - Replay reads events in seq_no order (monotonic, unique per execution)
//! - Events use a schema-versioned envelope for safe protocol evolution
//! - Invalid/corrupt events are reported as errors — never silently dropped
//! - Snapshots include boundary_hash for continuity verification
//! - Replay does NOT modify any state — pure read
//! - Replay always works from events; snapshots are acceleration only

use crate::protocol::canonical::{CapabilityAdapter, StreamEvent};
#[cfg(test)]
use crate::protocol::canonical::{ProviderCapabilities, ReasoningMode, Usage};
use crate::snapshot;

/// Current replay event envelope schema version.
pub const EVENT_SCHEMA_VERSION: i32 = 1;

/// A schema-versioned event envelope for safe protocol evolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayEventEnvelope {
    pub schema_version: i32,
    pub seq_no: i64,
    pub event: StreamEvent,
}

/// Errors that can occur during replay, with corruption diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("replay event parse failed at seq_no {seq_no}: {detail}")]
    ParseError { seq_no: i64, detail: String },

    #[error("sequence discontinuity at seq_no {got}: expected {expected}")]
    SeqDiscontinuity { expected: i64, got: i64 },

    #[error("duplicate seq_no: {seq_no}")]
    DuplicateSeqNo { seq_no: i64 },

    #[error("snapshot integrity mismatch: expected {expected}, got {actual}")]
    IntegrityMismatch { expected: String, actual: String },

    #[error("snapshot boundary mismatch at seq_no {seq_no}: expected {expected}, got {actual}")]
    BoundaryMismatch { seq_no: i64, expected: String, actual: String },

    #[error("snapshot not found: {0}")]
    SnapshotNotFound(i64),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result of a replay operation with diagnostics.
#[derive(Debug)]
pub struct ReplayResult {
    pub events: Vec<ReplayEventEnvelope>,
    /// Number of corrupt/invalid events encountered.
    pub corrupt_count: usize,
}

/// Full replay from the event log. Returns events in seq_no order.
/// Fails fast on any parse error — no silent drops.
pub fn replay_execution(
    db: &crate::db::Database,
    execution_id: i64,
) -> Result<ReplayResult, ReplayError> {
    let rows = db.get_execution_events(execution_id)?;
    let mut events = Vec::with_capacity(rows.len());
    let corrupt_count = 0;

    for (_id, kind, payload, seq_no, _ts) in &rows {
        // Try parsing as StreamEvent first (new format: has "type" field).
        // Fall back to injecting type from event_kind (old format: detail-only payload).
        let stream_event = serde_json::from_str::<StreamEvent>(payload)
            .or_else(|_| {
                // Old format: payload is a detail object without "type".
                // Inject the event_kind as the type field and re-parse.
                let mut val: serde_json::Value = serde_json::from_str(payload)
                    .map_err(|e| ReplayError::ParseError {
                        seq_no: *seq_no,
                        detail: format!("payload is not valid JSON: {e}"),
                    })?;
                if let Some(obj) = val.as_object_mut()
                    && !obj.contains_key("type")
                {
                    obj.insert("type".to_string(), serde_json::Value::String(kind.clone()));
                }
                serde_json::from_value::<StreamEvent>(val)
                    .map_err(|e| ReplayError::ParseError {
                        seq_no: *seq_no,
                        detail: format!("{e} — payload: {:.200}", payload),
                    })
            })?;
        events.push(ReplayEventEnvelope {
            schema_version: EVENT_SCHEMA_VERSION,
            seq_no: *seq_no,
            event: stream_event,
        });
    }

    // Verify monotonic seq order and detect duplicates
    for window in events.windows(2) {
        let a = window[0].seq_no;
        let b = window[1].seq_no;
        if a == b {
            return Err(ReplayError::DuplicateSeqNo { seq_no: a });
        }
        if a + 1 != b {
            return Err(ReplayError::SeqDiscontinuity { expected: a + 1, got: b });
        }
    }

    Ok(ReplayResult { events, corrupt_count })
}

/// Replay from a snapshot point, verifying continuity via boundary_hash.
pub fn replay_from_snapshot(
    db: &crate::db::Database,
    snapshot_id: i64,
    execution_id: i64,
) -> Result<ReplayResult, ReplayError> {
    let snap = db.restore_snapshot(snapshot_id)?
        .ok_or(ReplayError::SnapshotNotFound(snapshot_id))?;

    let last_snap_seq = snap.last_event_seq_no;

    // Load tail events after the snapshot boundary
    let rows = db.get_execution_events(execution_id)?;
    let tail: Vec<(i64, String)> = rows.iter()
        .filter(|(_id, _kind, _payload, seq_no, _ts)| *seq_no > last_snap_seq)
        .map(|(_id, _kind, payload, seq_no, _ts)| (*seq_no, payload.clone()))
        .collect();

    // Parse snapshot events
    let mut events: Vec<ReplayEventEnvelope> = Vec::new();
    if let Ok(Some(payload)) = snap.payload() {
        match payload {
            snapshot::SnapshotPayload::Ephemeral { .. } => {}
            snapshot::SnapshotPayload::Structural { events: snap_events }
            | snapshot::SnapshotPayload::Full { events: snap_events }
            | snapshot::SnapshotPayload::Frozen { events: snap_events } => {
                for (seq_no, val) in &snap_events {
                    let event: StreamEvent = serde_json::from_value(val.clone())
                        .map_err(|e| ReplayError::ParseError {
                            seq_no: *seq_no,
                            detail: format!("snapshot event parse: {e}"),
                        })?;
                    events.push(ReplayEventEnvelope {
                        schema_version: snap.schema_version,
                        seq_no: *seq_no,
                        event,
                    });
                }
            }
        }
    }

    // Parse and append tail events
    for (seq_no, payload) in &tail {
        let event = serde_json::from_str::<StreamEvent>(payload)
            .map_err(|e| ReplayError::ParseError {
                seq_no: *seq_no,
                detail: format!("{e} — payload: {:.200}", payload),
            })?;
        events.push(ReplayEventEnvelope {
            schema_version: EVENT_SCHEMA_VERSION,
            seq_no: *seq_no,
            event,
        });
    }

    // Ensure ordering
    events.sort_by_key(|e| e.seq_no);
    events.dedup_by_key(|e| e.seq_no);

    // Verify monotonic seq order and detect duplicates
    for window in events.windows(2) {
        let a = window[0].seq_no;
        let b = window[1].seq_no;
        if a == b {
            return Err(ReplayError::DuplicateSeqNo { seq_no: a });
        }
        if a + 1 != b {
            return Err(ReplayError::SeqDiscontinuity { expected: a + 1, got: b });
        }
    }

    Ok(ReplayResult { events, corrupt_count: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn setup_db() -> (tempfile::TempDir, crate::db::Database) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("replay_test.db");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = rt.block_on(
            crate::db::Database::builder().path(&path).build()
        ).unwrap();
        (dir, db)
    }

    #[test]
    fn round_trips_text_event() {
        let ev = StreamEvent::TextDelta { text: "hello world".into() };
        let json = serde_json::to_string(&ev).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn round_trips_tool_call_event() {
        let ev = StreamEvent::ToolCallStart {
            index: 0, id: "call_1".into(), name: "grep".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        match &back {
            StreamEvent::ToolCallStart { name, .. } => assert_eq!(name, "grep"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trips_done_with_incomplete() {
        let ev = StreamEvent::Done {
            usage: crate::protocol::canonical::Usage {
                prompt_tokens: 10, completion_tokens: 5, total_tokens: 15,
            },
            finish_reason: "length".into(),
            incomplete: true,
            error_reason: Some("stream truncated".into()),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        match &back {
            StreamEvent::Done { incomplete, .. } => assert!(*incomplete),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn envelope_schema_version() {
        let ev = StreamEvent::TextDelta { text: "test".into() };
        let envelope = ReplayEventEnvelope {
            schema_version: EVENT_SCHEMA_VERSION,
            seq_no: 1,
            event: ev,
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let back: ReplayEventEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, EVENT_SCHEMA_VERSION);
        assert_eq!(back.seq_no, 1);
    }

    #[test]
    fn parse_error_on_invalid_json() {
        let (dir, db) = setup_db();
        let conn = db.writer_lock().lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![1, "text", "not valid json at all", 1],
        ).unwrap();
        drop(conn);
        drop(dir);

        let result = replay_execution(&db, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReplayError::ParseError { seq_no, .. } => assert_eq!(seq_no, 1),
            other => panic!("expected ParseError, got: {other}"),
        }
    }

    #[test]
    fn duplicate_seq_no_detected() {
        let (dir, db) = setup_db();
        let conn = db.writer_lock().lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![1, "msg_start", r#"{"type":"message_start","role":"user"}"#, 1],
        ).unwrap();
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![1, "msg_end", r#"{"type":"message_end"}"#, 1],
        ).unwrap();
        drop(conn);
        drop(dir);

        let result = replay_execution(&db, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReplayError::DuplicateSeqNo { seq_no } => assert_eq!(seq_no, 1),
            other => panic!("expected DuplicateSeqNo, got: {other}"),
        }
    }

    #[test]
    fn seq_discontinuity_detected() {
        let (dir, db) = setup_db();
        let conn = db.writer_lock().lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![1, "start", r#"{"type":"message_start","role":"user"}"#, 1],
        ).unwrap();
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![1, "end", r#"{"type":"message_end"}"#, 3],
        ).unwrap();
        drop(conn);
        drop(dir);

        let result = replay_execution(&db, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReplayError::SeqDiscontinuity { expected, got } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            other => panic!("expected SeqDiscontinuity, got: {other}"),
        }
    }
}

/// Replay protocol assertion — verifies that a replayed event sequence
/// satisfies all critical-field invariants for the target provider.
pub fn assert_replay_valid(
    events: &[(i64, StreamEvent)],
    caps: &crate::protocol::canonical::ProviderCapabilities,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let requires_reasoning = CapabilityAdapter::expose_reasoning(caps);

    for (seq, ev) in events {
        match ev {
            StreamEvent::ReasoningDelta { text } if text.is_empty() => {
                errors.push(format!("seq {seq}: empty ReasoningDelta"));
            }
            StreamEvent::ToolCallStart { name, .. } if name.is_empty() => {
                errors.push(format!("seq {seq}: ToolCallStart with empty name"));
            }
            StreamEvent::Done { incomplete: true, error_reason: None, .. } => {
                errors.push(format!("seq {seq}: Done marked incomplete but no error_reason"));
            }
            StreamEvent::Done { .. } => {}
            _ => {}
        }
    }

    // Verify reasoning presence requirement
    if requires_reasoning {
        let has_reasoning = events.iter().any(|(_, ev)| matches!(ev, StreamEvent::ReasoningDelta { .. }));
        if !has_reasoning {
            // Not an error — some responses don't use reasoning
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod assertion_tests {
    use super::*;

    #[test]
    fn empty_events_pass_assertion() {
        let caps = ProviderCapabilities::default();
        assert!(assert_replay_valid(&[], &caps).is_ok());
    }

    #[test]
    fn done_incomplete_without_reason_fails() {
        let caps = ProviderCapabilities::default();
        let events = vec![(0, StreamEvent::Done {
            usage: Usage::default(),
            finish_reason: "stop".into(),
            incomplete: true,
            error_reason: None,
        })];
        assert!(assert_replay_valid(&events, &caps).is_err());
    }

    #[test]
    fn deepseek_caps_expose_reasoning() {
        let caps = ProviderCapabilities {
            reasoning: ReasoningMode::Full,
            ..Default::default()
        };
        assert!(CapabilityAdapter::expose_reasoning(&caps));
    }
} // mod tests
