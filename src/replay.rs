//! Deterministic replay engine — reconstructs StreamEvent sequences from
//! the append-only execution_events table.
//!
//! Replay is the verification substrate for memory mutation. Before accepting
//! any evolved memory topology, replay verifies it against historical
//! executions.
//!
//! # Design invariants
//!
//! - Replay reads events in seq_no order (deterministic ordering)
//! - Events are stored as full StreamEvent JSON (lossless)
//! - Replay does NOT modify any state — pure read
//! - Snapshots are acceleration, not prerequisite — replay always works from events

use crate::protocol::canonical::StreamEvent;

/// Replay an execution from the event log. Returns events in seq_no order.
pub fn replay_execution(
    db: &crate::db::Database,
    execution_id: i64,
) -> anyhow::Result<Vec<(i64, StreamEvent)>> {
    let rows = db.get_execution_events(execution_id)?;
    let mut events = Vec::with_capacity(rows.len());
    for (_id, _kind, payload, seq_no, _ts) in rows {
        if let Ok(ev) = serde_json::from_str::<StreamEvent>(&payload) {
            events.push((seq_no, ev));
        }
    }
    // Already ordered by seq_no from the query, but ensure determinism
    events.sort_by_key(|(seq, _)| *seq);
    Ok(events)
}

/// Replay events from a snapshot + tail events.
/// Snapshot provides the jump point; tail events complete the replay.
pub fn replay_from_snapshot(
    db: &crate::db::Database,
    snapshot_id: i64,
    execution_id: i64,
) -> anyhow::Result<Vec<(i64, StreamEvent)>> {
    let snap = db.restore_snapshot(snapshot_id)?;
    let mut events = Vec::new();

    // Reconstruct snapshot state from snapshot_data (stored as JSON)
    if let Some(ref s) = snap {
        if let Ok(snap_events) = serde_json::from_str::<Vec<(i64, StreamEvent)>>(&s.snapshot_data) {
            events.extend(snap_events);
        }
    }

    // Append tail events from the event log
    let last_seq = events.last().map(|(s, _)| *s).unwrap_or(0);
    let tail = db.get_execution_events(execution_id)?;
    for (_id, _kind, payload, seq_no, _ts) in tail {
        if seq_no > last_seq {
            if let Ok(ev) = serde_json::from_str::<StreamEvent>(&payload) {
                events.push((seq_no, ev));
            }
        }
    }

    events.sort_by_key(|(seq, _)| *seq);
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_round_trips_text_event() {
        let ev = StreamEvent::TextDelta { text: "hello world".into() };
        let json = serde_json::to_string(&ev).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn replay_round_trips_tool_call_event() {
        let ev = StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "grep".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        match &back {
            StreamEvent::ToolCallStart { name, .. } => assert_eq!(name, "grep"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn replay_round_trips_done_with_incomplete() {
        let ev = StreamEvent::Done {
            usage: crate::protocol::canonical::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
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
}
