//! RuntimeStateView — derived read-only state from the append-only
//! event log. Phase 2.5: validates event schema completeness by
//! computing runtime truth from events alone.
//!
//! This is a PURE function: events in, state out. No mutation.
//! Existing mutable projections remain as safety net — they are
//! NOT replaced by this view until parity is validated.
//!
//! Scope: execution lifecycle, retry lifecycle, cancellation.
//! OUT of scope: DAG, compaction, cache, observation.

use crate::runtime_events::RuntimeEvent;

/// Read-only derived state from an event log slice.
/// Computed on demand; never cached or materialized.
pub struct RuntimeStateView;

impl RuntimeStateView {
    /// Count retries scheduled for a conversation.
    #[must_use]
    pub fn retry_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::RetryScheduled { conv_id: c, .. } if *c == conv_id))
            .count()
    }

    /// Count retries aborted for a conversation.
    pub fn retry_aborted_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::RetryAborted { conv_id: c, .. } if *c == conv_id))
            .count()
    }

    /// Total tokens spent in a conversation (sum of ToolCallCompleted tokens).
    #[must_use]
    pub fn total_tokens(events: &[RuntimeEvent], conv_id: i64) -> u64 {
        events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::ToolCallCompleted { conv_id: c, tokens_spent, .. } = e {
                    if *c == conv_id { Some(*tokens_spent) } else { None }
                } else {
                    None
                }
            })
            .sum()
    }

    /// Count of cache hits in a conversation.
    pub fn cache_hit_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| {
                matches!(e, RuntimeEvent::ToolCallCompleted { conv_id: c, cache_hit: true, .. } if *c == conv_id)
            })
            .count()
    }

    /// Total tool call completions in a conversation.
    pub fn completed_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::ToolCallCompleted { conv_id: c, .. } if *c == conv_id))
            .count()
    }

    /// Total tool call failures in a conversation.
    pub fn failure_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::ToolCallFailed { conv_id: c, .. } if *c == conv_id))
            .count()
    }

    /// Most recent ToolCallFailed event for a conversation, if any.
    pub fn last_failure(events: &[RuntimeEvent], conv_id: i64) -> Option<&RuntimeEvent> {
        events
            .iter()
            .rev()
            .find(|e| matches!(e, RuntimeEvent::ToolCallFailed { conv_id: c, .. } if *c == conv_id))
    }

    /// Current attempt number for a tool call (highest attempt seen).
    /// Returns 0 if never scheduled.
    pub fn current_attempt(events: &[RuntimeEvent], tool_call_id: &str) -> u32 {
        events
            .iter()
            .filter_map(|e| match e {
                RuntimeEvent::ToolCallScheduled { tool_call_id: tcid, attempt, .. } if tcid == tool_call_id => Some(*attempt),
                RuntimeEvent::ToolCallCompleted { tool_call_id: tcid, attempt, .. } if tcid == tool_call_id => Some(*attempt),
                RuntimeEvent::ToolCallFailed { tool_call_id: tcid, attempt, .. } if tcid == tool_call_id => Some(*attempt),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// Whether a conversation has an active (unresolved) cancellation.
    /// True if CancellationRequested exists without a subsequent CancellationCompleted.
    pub fn is_cancellation_pending(events: &[RuntimeEvent], conv_id: i64) -> bool {
        let mut requested = false;
        for e in events {
            match e {
                RuntimeEvent::CancellationRequested { conv_id: c, .. } if *c == conv_id => {
                    requested = true;
                }
                RuntimeEvent::CancellationCompleted { conv_id: c, .. } if *c == conv_id => {
                    requested = false;
                }
                _ => {}
            }
        }
        requested
    }

    /// Count of CancellationAcknowledged events in a conversation.
    pub fn cancellation_ack_count(events: &[RuntimeEvent], conv_id: i64) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::CancellationAcknowledged { conv_id: c, .. } if *c == conv_id))
            .count()
    }

    /// Whether a tool call was ultimately retried (has RetryScheduled).
    pub fn was_retried(events: &[RuntimeEvent], tool_call_id: &str) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RetryScheduled { tool_call_id: tcid, .. } if tcid == tool_call_id))
    }

    /// Extract execution ordering: events sorted by logical_seq for a conversation.
    /// Useful for replay reconstruction and ordering validation.
    pub fn ordered_by_seq(events: &[RuntimeEvent], conv_id: i64) -> Vec<&RuntimeEvent> {
        let mut filtered: Vec<&RuntimeEvent> = events
            .iter()
            .filter(|e| e.conv_id() == conv_id)
            .collect();
        filtered.sort_by_key(|e| e.logical_seq());
        filtered
    }

    /// Compute a projection diff: compare event-derived values against
    /// a provided metrics snapshot. Returns discrepancies as human-readable
    /// strings. Empty Vec means parity. Silently ignoring the result is a bug.
    #[must_use = "projection parity result must be checked — empty vec means healthy"]
    pub fn projection_parity_check(
        events: &[RuntimeEvent],
        conv_id: i64,
        expected_tokens: u64,
        expected_cache_hits: usize,
        expected_failures: usize,
        expected_completions: usize,
    ) -> Vec<String> {
        let mut diffs = Vec::new();

        let et = Self::total_tokens(events, conv_id);
        if et != expected_tokens {
            diffs.push(format!(
                "tokens: events={et}, projection={expected_tokens}"
            ));
        }

        let ec = Self::cache_hit_count(events, conv_id);
        if ec != expected_cache_hits {
            diffs.push(format!(
                "cache_hits: events={ec}, projection={expected_cache_hits}"
            ));
        }

        let ef = Self::failure_count(events, conv_id);
        if ef != expected_failures {
            diffs.push(format!(
                "failures: events={ef}, projection={expected_failures}"
            ));
        }

        let ed = Self::completed_count(events, conv_id);
        if ed != expected_completions {
            diffs.push(format!(
                "completions: events={ed}, projection={expected_completions}"
            ));
        }

        diffs
    }

    /// Fully derivable: ToolCallCompleted with cache_hit=true → tool_name.
    pub fn cached_tool_names(events: &[RuntimeEvent], conv_id: i64) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::ToolCallCompleted { conv_id: c, cache_hit: true, tool_name, .. } = e {
                    if *c == conv_id { Some(tool_name.clone()) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Comprehensive runtime inspection report (Phase 4.2/4.4).
    /// Combines event summary, timeline, parity stats, and invariant
    /// violations into a single human-readable output.
    #[must_use]
    pub fn inspect(events: &[RuntimeEvent], conv_id: i64) -> String {
        let mut out = String::new();

        // Header
        let total = events.len();
        let completions = Self::completed_count(events, conv_id);
        let failures = Self::failure_count(events, conv_id);
        let retries = Self::retry_count(events, conv_id);
        let tokens = Self::total_tokens(events, conv_id);
        let cached = Self::cache_hit_count(events, conv_id);

        out.push_str("══════ Runtime Inspection conv=");
        out.push_str(&conv_id.to_string());
        out.push_str(" ══════\n\n");

        // Summary
        out.push_str(&format!(
            "Events: {total} | Completions: {completions} | Failures: {failures} | Retries: {retries}\n\
             Tokens: {tokens} | Cache hits: {cached}\n"
        ));

        let pending = Self::is_cancellation_pending(events, conv_id);
        out.push_str(&format!("Cancellation: {}\n", if pending { "PENDING" } else { "resolved" }));

        // Invariant violations
        let cancel_issues = crate::runtime_invariants::check_cancellation_well_formed(events, conv_id);
        if !cancel_issues.is_empty() {
            out.push_str("\n── INVARIANT VIOLATIONS ──\n");
            for issue in &cancel_issues {
                out.push_str(&format!("  ✗ {issue}\n"));
            }
        }

        // Ordering check
        let ordered = Self::ordered_by_seq(events, conv_id);
        let mut seq_violations = 0usize;
        for w in ordered.windows(2) {
            if w[0].logical_seq() >= w[1].logical_seq() {
                seq_violations += 1;
            }
        }
        if seq_violations > 0 {
            out.push_str(&format!("  ✗ {seq_violations} logical_seq ordering violations\n"));
        }

        // Timeline
        out.push_str("\n── Timeline ──\n");
        out.push_str(&Self::explain(events));

        // Retry details
        out.push_str("\n── Retry Summary ──\n");
        let retry_targets: std::collections::HashSet<String> = events.iter()
            .filter_map(|e| match e {
                RuntimeEvent::RetryScheduled { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();
        if retry_targets.is_empty() {
            out.push_str("  (no retries)\n");
        } else {
            for tcid in &retry_targets {
                let max_attempt = Self::current_attempt(events, tcid);
                out.push_str(&format!("  {tcid}: max attempt {max_attempt}\n"));
            }
        }

        out
    }

    /// Dump events to a human-readable log format for debugging.
    /// Output format: `[seq] Kind { key: value, ... }`
    pub fn dump_events(events: &[RuntimeEvent]) -> String {
        let mut out = String::new();
        for e in events {
            let seq = e.logical_seq();
            out.push_str(&format!("[{seq:04}] {}\n", Self::format_event(e)));
        }
        out
    }

    /// Produce a grouped timeline view for debugging. Tool calls are
    /// grouped by tool_call_id, showing retry chains and cancellation.
    /// Output format:
    /// ```text
    /// Execution conv=1 profile=efficient
    ///   tc_1: grep [attempt=1] ✓ (100 tokens)
    ///   tc_2: read_file [attempt=1] ✗ ENOENT
    ///     → retry [attempt=2] fix="check path"
    ///     → ✓ (50 tokens, cache)
    ///   ⚡ CANCEL Shutdown
    ///     • tc_pending acknowledged
    ///     • clean shutdown
    /// ```
    pub fn explain(events: &[RuntimeEvent]) -> String {
        let mut out = String::new();
        // Collect tool calls by tcid for grouping
        let mut tool_calls: std::collections::BTreeMap<String, Vec<&RuntimeEvent>> = std::collections::BTreeMap::new();
        let mut meta_events: Vec<&RuntimeEvent> = Vec::new();

        for e in events {
            let tcid = match e {
                RuntimeEvent::ToolCallScheduled { tool_call_id, .. }
                | RuntimeEvent::ToolCallCompleted { tool_call_id, .. }
                | RuntimeEvent::ToolCallFailed { tool_call_id, .. }
                | RuntimeEvent::RetryScheduled { tool_call_id, .. }
                | RuntimeEvent::RetryAborted { tool_call_id, .. }
                | RuntimeEvent::CancellationAcknowledged { tool_call_id, .. } => {
                    Some(tool_call_id.clone())
                }
                _ => None,
            };
            if let Some(tcid) = tcid {
                tool_calls.entry(tcid).or_default().push(e);
            } else {
                meta_events.push(e);
            }
        }

        // Print execution headers
        for e in &meta_events {
            match e {
                RuntimeEvent::ExecutionStarted { conv_id, profile, .. } => {
                    out.push_str(&format!("Execution conv={conv_id} profile={profile}\n"));
                }
                RuntimeEvent::CancellationRequested { source, .. } => {
                    out.push_str(&format!("  ⚡ CANCEL {source:?}\n"));
                }
                RuntimeEvent::CancellationCompleted { clean, .. } => {
                    let label = if *clean { "clean shutdown" } else { "FORCED shutdown" };
                    out.push_str(&format!("    • {label}\n"));
                }
                _ => {}
            }
        }

        // Print tool call groups
        for (tcid, evs) in &tool_calls {
            let mut prev_attempt: u32 = 0;
            for e in evs {
                match e {
                    RuntimeEvent::ToolCallScheduled { tool_name, attempt, .. } => {
                        if *attempt > prev_attempt {
                            out.push_str(&format!("    → retry [attempt={attempt}]"));
                        } else {
                            out.push_str(&format!("  {tcid}: {tool_name} [attempt={attempt}]"));
                        }
                        prev_attempt = *attempt;
                    }
                    RuntimeEvent::ToolCallCompleted { tokens_spent, cache_hit, .. } => {
                        let ch = if *cache_hit { ", cache" } else { "" };
                        out.push_str(&format!(" ✓ ({tokens_spent} tokens{ch})\n"));
                    }
                    RuntimeEvent::ToolCallFailed { error_signature, retryable, .. } => {
                        let re = if *retryable { " (retryable)" } else { " (fatal)" };
                        out.push_str(&format!(" ✗ {error_signature}{re}\n"));
                    }
                    RuntimeEvent::RetryScheduled { suggested_fix, .. } => {
                        out.push_str(&format!(" fix=\"{suggested_fix}\"\n"));
                    }
                    RuntimeEvent::RetryAborted { total_attempts, reason, .. } => {
                        out.push_str(&format!("    → ABORTED after {total_attempts} attempts: {reason}\n"));
                    }
                    RuntimeEvent::CancellationAcknowledged { span_id, .. } => {
                        out.push_str(&format!("    • {tcid} acknowledged ({span_id})\n"));
                    }
                    _ => {}
                }
            }
        }

        out
    }

    fn format_event(e: &RuntimeEvent) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        match e {
            RuntimeEvent::ExecutionStarted { conv_id, logical_seq, profile } => {
                write!(s, "ExecutionStarted conv={conv_id} seq={logical_seq} profile={profile}").unwrap();
            }
            RuntimeEvent::ToolCallScheduled { conv_id, tool_name, tool_call_id, span_id, attempt, .. } => {
                write!(s, "ToolCallScheduled conv={conv_id} tool={tool_name} tcid={tool_call_id} span={span_id} attempt={attempt}").unwrap();
            }
            RuntimeEvent::ToolCallCompleted { conv_id, tool_name, tool_call_id, attempt, tokens_spent, cache_hit, execution_unit_id, .. } => {
                let ch = if *cache_hit { " (cache)" } else { "" };
                write!(s, "ToolCallCompleted conv={conv_id} tool={tool_name} tcid={tool_call_id} attempt={attempt} tokens={tokens_spent} unit={execution_unit_id}{ch}").unwrap();
            }
            RuntimeEvent::ToolCallFailed { conv_id, tool_name, tool_call_id, attempt, error_signature, retryable, execution_unit_id, .. } => {
                let re = if *retryable { " retryable" } else { "" };
                write!(s, "ToolCallFailed conv={conv_id} tool={tool_name} tcid={tool_call_id} attempt={attempt} error={error_signature} unit={execution_unit_id}{re}").unwrap();
            }
            RuntimeEvent::RetryScheduled { conv_id, tool_call_id, attempt, suggested_fix, .. } => {
                write!(s, "RetryScheduled conv={conv_id} tcid={tool_call_id} attempt={attempt} fix=\"{suggested_fix}\"").unwrap();
            }
            RuntimeEvent::RetryAborted { conv_id, tool_call_id, total_attempts, reason, .. } => {
                write!(s, "RetryAborted conv={conv_id} tcid={tool_call_id} attempts={total_attempts} reason=\"{reason}\"").unwrap();
            }
            RuntimeEvent::CancellationRequested { conv_id, source, .. } => {
                write!(s, "CancellationRequested conv={conv_id} source={source:?}").unwrap();
            }
            RuntimeEvent::CancellationAcknowledged { conv_id, tool_call_id, span_id, .. } => {
                write!(s, "CancellationAcknowledged conv={conv_id} tcid={tool_call_id} span={span_id}").unwrap();
            }
            RuntimeEvent::CancellationCompleted { conv_id, clean, .. } => {
                let cl = if *clean { " clean" } else { " FORCED" };
                write!(s, "CancellationCompleted conv={conv_id}{cl}").unwrap();
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_events() -> Vec<RuntimeEvent> {
        vec![
            RuntimeEvent::ExecutionStarted { conv_id: 1, logical_seq: 1, profile: "efficient".into() },
            RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 2, tool_name: "grep".into(), tool_call_id: "tc_1".into(), span_id: "sp_1".into(), attempt: 1 },
            RuntimeEvent::ToolCallCompleted { conv_id: 1, logical_seq: 3, tool_name: "grep".into(), tool_call_id: "tc_1".into(), span_id: "sp_1".into(), attempt: 1, tokens_spent: 100, cache_hit: false, execution_unit_id: 10 },
            RuntimeEvent::ExecutionStarted { conv_id: 2, logical_seq: 4, profile: "efficient".into() },
            RuntimeEvent::ToolCallScheduled { conv_id: 2, logical_seq: 5, tool_name: "read_file".into(), tool_call_id: "tc_2".into(), span_id: "sp_2".into(), attempt: 1 },
            RuntimeEvent::ToolCallFailed { conv_id: 2, logical_seq: 6, tool_name: "read_file".into(), tool_call_id: "tc_2".into(), span_id: "sp_2".into(), attempt: 1, error_signature: "ENOENT".into(), retryable: true, execution_unit_id: 0 },
            RuntimeEvent::RetryScheduled { conv_id: 2, logical_seq: 7, tool_call_id: "tc_2".into(), attempt: 2, suggested_fix: "check path".into() },
            RuntimeEvent::ToolCallScheduled { conv_id: 2, logical_seq: 8, tool_name: "read_file".into(), tool_call_id: "tc_2".into(), span_id: "sp_3".into(), attempt: 2 },
            RuntimeEvent::ToolCallCompleted { conv_id: 2, logical_seq: 9, tool_name: "read_file".into(), tool_call_id: "tc_2".into(), span_id: "sp_3".into(), attempt: 2, tokens_spent: 50, cache_hit: true, execution_unit_id: 11 },
            RuntimeEvent::CancellationRequested { conv_id: 2, logical_seq: 10, source: crate::runtime_events::CancellationSource::Shutdown },
            RuntimeEvent::CancellationAcknowledged { conv_id: 2, logical_seq: 11, tool_call_id: "tc_pending".into(), span_id: "sp_4".into() },
            RuntimeEvent::CancellationCompleted { conv_id: 2, logical_seq: 12, clean: true },
        ]
    }

    #[test]
    fn total_tokens_per_conv() {
        let events = make_events();
        assert_eq!(RuntimeStateView::total_tokens(&events, 1), 100);
        assert_eq!(RuntimeStateView::total_tokens(&events, 2), 50);
        assert_eq!(RuntimeStateView::total_tokens(&events, 99), 0);
    }

    #[test]
    fn cache_hit_count() {
        let events = make_events();
        assert_eq!(RuntimeStateView::cache_hit_count(&events, 1), 0);
        assert_eq!(RuntimeStateView::cache_hit_count(&events, 2), 1);
    }

    #[test]
    fn failure_and_completion_counts() {
        let events = make_events();
        assert_eq!(RuntimeStateView::failure_count(&events, 1), 0);
        assert_eq!(RuntimeStateView::failure_count(&events, 2), 1);
        assert_eq!(RuntimeStateView::completed_count(&events, 1), 1);
        assert_eq!(RuntimeStateView::completed_count(&events, 2), 1);
    }

    #[test]
    fn retry_count() {
        let events = make_events();
        assert_eq!(RuntimeStateView::retry_count(&events, 1), 0);
        assert_eq!(RuntimeStateView::retry_count(&events, 2), 1);
        assert_eq!(RuntimeStateView::retry_aborted_count(&events, 2), 0);
    }

    #[test]
    fn current_attempt_tracking() {
        let events = make_events();
        assert_eq!(RuntimeStateView::current_attempt(&events, "tc_1"), 1);
        assert_eq!(RuntimeStateView::current_attempt(&events, "tc_2"), 2);
        assert_eq!(RuntimeStateView::current_attempt(&events, "nonexistent"), 0);
    }

    #[test]
    fn was_retried() {
        let events = make_events();
        assert!(!RuntimeStateView::was_retried(&events, "tc_1"));
        assert!(RuntimeStateView::was_retried(&events, "tc_2"));
    }

    #[test]
    fn cancellation_lifecycle() {
        let events = make_events();
        // conv 2 has CancellationRequested → Acknowledged → Completed
        // So cancellation is resolved (not pending)
        assert!(!RuntimeStateView::is_cancellation_pending(&events, 2));
        // conv 1 has no cancellation events
        assert!(!RuntimeStateView::is_cancellation_pending(&events, 1));
        assert_eq!(RuntimeStateView::cancellation_ack_count(&events, 2), 1);
    }

    #[test]
    fn cancellation_pending_without_completion() {
        let events = vec![
            RuntimeEvent::CancellationRequested { conv_id: 1, logical_seq: 1, source: crate::runtime_events::CancellationSource::Shutdown },
        ];
        assert!(RuntimeStateView::is_cancellation_pending(&events, 1));
    }

    #[test]
    fn ordering_by_logical_seq() {
        let events = make_events();
        let ordered = RuntimeStateView::ordered_by_seq(&events, 2);
        let seqs: Vec<i64> = ordered.iter().map(|e| e.logical_seq()).collect();
        // Must be ascending
        for w in seqs.windows(2) {
            assert!(w[0] < w[1], "seq not ascending: {:?}", seqs);
        }
    }

    #[test]
    fn projection_parity_check_passes() {
        let events = make_events();
        let diffs = RuntimeStateView::projection_parity_check(
            &events, 1,
            100, // expected_tokens
            0,   // expected_cache_hits
            0,   // expected_failures
            1,   // expected_completions
        );
        assert!(diffs.is_empty(), "unexpected diffs: {:?}", diffs);
    }

    #[test]
    fn projection_parity_check_detects_mismatch() {
        let events = make_events();
        let diffs = RuntimeStateView::projection_parity_check(
            &events, 2,
            999, // wrong tokens
            0,   // wrong cache_hits
            0,   // wrong failures
            0,   // wrong completions
        );
        assert_eq!(diffs.len(), 4, "should detect 4 mismatches");
    }

    #[test]
    fn last_failure_returns_most_recent() {
        let events = make_events();
        let failure = RuntimeStateView::last_failure(&events, 2);
        assert!(failure.is_some());
        if let Some(RuntimeEvent::ToolCallFailed { error_signature, .. }) = failure {
            assert_eq!(error_signature, "ENOENT");
        } else {
            panic!("expected ToolCallFailed");
        }
    }
}
