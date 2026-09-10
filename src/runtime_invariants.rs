//! Runtime invariant enforcement (Phase 4.1).
//!
//! These are runtime assertions that validate the frozen architecture
//! contracts at execution time. Violations produce explain() output
//! and telemetry, not just test failures.
//!
//! All assertions are `debug_assert!` level — enabled in debug/test,
//! compiled out in release. For production invariant monitoring,
//! use the check_* variants which return `Vec<String>`.

use crate::runtime_events::RuntimeEvent;
use crate::runtime_state_view::RuntimeStateView;

/// Assert that logical_seq values in an event slice are strictly
/// ascending within each conversation. Violation means replay ordering
/// is broken.
pub fn assert_monotonic_logical_seq(events: &[RuntimeEvent], conv_id: i64) {
    let ordered = RuntimeStateView::ordered_by_seq(events, conv_id);
    for w in ordered.windows(2) {
        debug_assert!(
            w[0].logical_seq() < w[1].logical_seq(),
            "LOGICAL_SEQ VIOLATION: seq {} >= seq {} in conv={conv_id}\n{}",
            w[0].logical_seq(),
            w[1].logical_seq(),
            RuntimeStateView::explain(&[w[0].clone(), w[1].clone()])
        );
    }
}

/// Assert that projection parity holds for a conversation.
/// Violation means mutable metrics drifted from the event log.
pub fn assert_projection_parity(
    events: &[RuntimeEvent],
    conv_id: i64,
    expected_tokens: u64,
    expected_cache_hits: usize,
    expected_failures: usize,
    expected_completions: usize,
) {
    let diffs = RuntimeStateView::projection_parity_check(
        events, conv_id,
        expected_tokens, expected_cache_hits,
        expected_failures, expected_completions,
    );
    debug_assert!(
        diffs.is_empty(),
        "PROJECTION DRIFT conv={conv_id}:\n{}\n--- Event log ---\n{}",
        diffs.join("\n"),
        RuntimeStateView::dump_events(events)
    );
}

/// Assert that a cancellation lifecycle is well-formed within a
/// conversation. Violation means cancellation contract is broken.
pub fn assert_cancellation_well_formed(events: &[RuntimeEvent], conv_id: i64) {
    let mut requested_count = 0usize;
    let mut completed_count = 0usize;
    let mut last_requested_seq: Option<i64> = None;
    let mut last_completed_seq: Option<i64> = None;

    for e in events {
        if e.conv_id() != conv_id { continue; }
        match e {
            RuntimeEvent::CancellationRequested { logical_seq, .. } => {
                requested_count += 1;
                last_requested_seq = Some(*logical_seq);
            }
            RuntimeEvent::CancellationCompleted { logical_seq, .. } => {
                completed_count += 1;
                last_completed_seq = Some(*logical_seq);
            }
            _ => {}
        }
    }

    // Multiple cancellations without interleaving completion is suspicious
    debug_assert!(
        requested_count <= completed_count + 1,
        "CANCELLATION VIOLATION conv={conv_id}: {requested_count} requests, {completed_count} completions\n{}",
        RuntimeStateView::dump_events(events)
    );

    // Completed must follow requested
    if let (Some(rs), Some(cs)) = (last_requested_seq, last_completed_seq) {
        debug_assert!(
            cs > rs,
            "CANCELLATION VIOLATION conv={conv_id}: completed seq {cs} before requested seq {rs}"
        );
    }
}

/// Assert that no tool call has a retry with non-increasing attempt number.
pub fn assert_retry_attempt_monotonic(events: &[RuntimeEvent], tool_call_id: &str) {
    let mut last_attempt: u32 = 0;
    for e in events {
        let attempt = match e {
            RuntimeEvent::RetryScheduled { tool_call_id: tcid, attempt, .. } if tcid == tool_call_id => Some(*attempt),
            RuntimeEvent::ToolCallScheduled { tool_call_id: tcid, attempt, .. } if tcid == tool_call_id => Some(*attempt),
            _ => None,
        };
        if let Some(a) = attempt {
            debug_assert!(
                a >= last_attempt,
                "RETRY ORDERING VIOLATION tcid={tool_call_id}: attempt {a} after {last_attempt}\n{}",
                RuntimeStateView::dump_events(events)
            );
            last_attempt = a;
        }
    }
}

/// Non-debug variant: check well-formedness of cancellation lifecycle.
/// Returns violations. Empty vec means healthy.
pub fn check_cancellation_well_formed(events: &[RuntimeEvent], conv_id: i64) -> Vec<String> {
    let mut issues = Vec::new();
    let mut requested = false;
    let mut completed = false;

    for e in events {
        if e.conv_id() != conv_id { continue; }
        match e {
            RuntimeEvent::CancellationRequested { .. } => {
                if requested && !completed {
                    issues.push(format!(
                        "conv={conv_id}: CancellationRequested without prior CancellationCompleted"
                    ));
                }
                requested = true;
                completed = false;
            }
            RuntimeEvent::CancellationCompleted { .. } => {
                if !requested {
                    issues.push(format!(
                        "conv={conv_id}: CancellationCompleted without prior CancellationRequested"
                    ));
                }
                completed = true;
            }
            _ => {}
        }
    }

    if requested && !completed {
        issues.push(format!("conv={conv_id}: pending CancellationRequested never resolved"));
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::{CancellationSource, RuntimeEvent};

    #[test]
    fn monotonic_seq_passes_for_ordered_events() {
        let events = vec![
            RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 1, tool_name: "g".into(), tool_call_id: "c1".into(), span_id: "s1".into(), attempt: 1 },
            RuntimeEvent::ToolCallCompleted { conv_id: 1, logical_seq: 2, tool_name: "g".into(), tool_call_id: "c1".into(), span_id: "s1".into(), attempt: 1, tokens_spent: 10, cache_hit: false, execution_unit_id: 1 },
        ];
        assert_monotonic_logical_seq(&events, 1); // should not panic
    }

    #[test]
    #[should_panic(expected = "LOGICAL_SEQ")]
    fn monotonic_seq_panics_on_duplicate() {
        let events = vec![
            RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 1, tool_name: "g".into(), tool_call_id: "c1".into(), span_id: "s1".into(), attempt: 1 },
            RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 1, tool_name: "g".into(), tool_call_id: "c2".into(), span_id: "s2".into(), attempt: 1 },
        ];
        assert_monotonic_logical_seq(&events, 1);
    }

    #[test]
    fn cancellation_check_passes_for_normal_lifecycle() {
        let events = vec![
            RuntimeEvent::ExecutionStarted { conv_id: 1, logical_seq: 1, profile: "e".into() },
            RuntimeEvent::CancellationRequested { conv_id: 1, logical_seq: 2, source: CancellationSource::Shutdown },
            RuntimeEvent::CancellationCompleted { conv_id: 1, logical_seq: 3, clean: true },
        ];
        let issues = check_cancellation_well_formed(&events, 1);
        assert!(issues.is_empty(), "issues: {:?}", issues);
    }

    #[test]
    fn cancellation_check_detects_orphan_completion() {
        let events = vec![
            RuntimeEvent::CancellationCompleted { conv_id: 1, logical_seq: 1, clean: true },
        ];
        let issues = check_cancellation_well_formed(&events, 1);
        assert!(!issues.is_empty());
    }
}
