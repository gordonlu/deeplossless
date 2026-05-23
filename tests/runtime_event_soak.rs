//! Runtime event soak tests — long-run parity validation (Phase 2.6).
//!
//! Verifies that projections never drift from the event log under
//! extended lifetime: many retries, interleaved cancellation,
//! partial failure, mixed completion, replay ordering.

use deeplossless::runtime_events::{CancellationSource, RuntimeEvent};
use deeplossless::runtime_state_view::RuntimeStateView;

#[test]
fn soak_100_retries_no_parity_drift() {
    let mut events = Vec::new();
    let mut expected_tokens: u64 = 0;
    let mut expected_cache_hits: usize = 0;
    let mut expected_failures: usize = 0;
    let mut expected_completions: usize = 0;
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;

    seq += 1;
    events.push(RuntimeEvent::ExecutionStarted {
        conv_id, logical_seq: seq, profile: "efficient".into(),
    });

    // 100 tool calls, each with 1-3 retries
    for call_idx in 0..100 {
        let tcid = format!("tc_{call_idx}");
        let max_attempts = (call_idx % 3 + 1) as u32; // 1, 2, or 3 attempts
        let will_fail_all = call_idx % 5 == 0; // every 5th call fails all attempts

        for attempt in 1..=max_attempts {
            seq += 1;
            events.push(RuntimeEvent::ToolCallScheduled {
                conv_id, logical_seq: seq,
                tool_name: format!("tool_{}", call_idx % 5),
                tool_call_id: tcid.clone(),
                span_id: format!("sp_{}_{}", tcid, attempt),
                attempt,
            });

            let is_last = attempt == max_attempts;
            if is_last && !will_fail_all {
                // Complete on last attempt
                seq += 1;
                let tokens = (call_idx * 10 + attempt as usize * 5) as u64;
                let cache_hit = call_idx % 3 == 0;
                events.push(RuntimeEvent::ToolCallCompleted {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{}", call_idx % 5),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{}_{}", tcid, attempt),
                    attempt,
                    tokens_spent: tokens,
                    cache_hit,
                    execution_unit_id: call_idx as i64 * 100 + attempt as i64,
                });
                expected_completions += 1;
                expected_tokens += tokens;
                if cache_hit { expected_cache_hits += 1; }
            } else {
                // Fail (retryable unless last)
                seq += 1;
                let retryable = !is_last;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{}", call_idx % 5),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{}_{}", tcid, attempt),
                    attempt,
                    error_signature: format!("ERR_{}", attempt),
                    retryable,
                    execution_unit_id: 0,
                });
                expected_failures += 1;

                if retryable {
                    seq += 1;
                    events.push(RuntimeEvent::RetryScheduled {
                        conv_id, logical_seq: seq,
                        tool_call_id: tcid.clone(),
                        attempt: attempt + 1,
                        suggested_fix: format!("fix_{}", attempt),
                    });
                } else {
                    seq += 1;
                    events.push(RuntimeEvent::RetryAborted {
                        conv_id, logical_seq: seq,
                        tool_call_id: tcid.clone(),
                        total_attempts: attempt,
                        reason: "max retries".into(),
                    });
                }
            }
        }
    }

    // Verify event-derived counts match expected
    assert_eq!(
        RuntimeStateView::total_tokens(&events, conv_id),
        expected_tokens,
        "tokens must match after 100 calls"
    );
    assert_eq!(
        RuntimeStateView::cache_hit_count(&events, conv_id),
        expected_cache_hits,
        "cache hits must match"
    );
    assert_eq!(
        RuntimeStateView::completed_count(&events, conv_id),
        expected_completions,
        "completions must match"
    );
    assert_eq!(
        RuntimeStateView::failure_count(&events, conv_id),
        expected_failures,
        "failures must match"
    );

    // projection_parity_check must pass
    let diffs = RuntimeStateView::projection_parity_check(
        &events, conv_id,
        expected_tokens, expected_cache_hits,
        expected_failures, expected_completions,
    );
    assert!(diffs.is_empty(), "parity diffs: {:?}", diffs);
}

#[test]
fn soak_interleaved_cancellation() {
    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;

    seq += 1;
    events.push(RuntimeEvent::ExecutionStarted {
        conv_id, logical_seq: seq, profile: "efficient".into(),
    });

    // Start 5 tool calls
    for i in 0..5 {
        let tcid = format!("tc_{i}");
        seq += 1;
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(),
            tool_call_id: tcid.clone(),
            span_id: format!("sp_{i}"),
            attempt: 1,
        });
    }

    // Complete 3, then cancel
    for i in 0..3 {
        seq += 1;
        events.push(RuntimeEvent::ToolCallCompleted {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(),
            tool_call_id: format!("tc_{i}"),
            span_id: format!("sp_{i}"),
            attempt: 1,
            tokens_spent: 10,
            cache_hit: false,
            execution_unit_id: i * 100,
        });
    }

    // Request cancellation
    seq += 1;
    events.push(RuntimeEvent::CancellationRequested {
        conv_id, logical_seq: seq,
        source: deeplossless::runtime_events::CancellationSource::Shutdown,
    });

    // Remaining 2 acknowledge
    for i in 3..5 {
        seq += 1;
        events.push(RuntimeEvent::CancellationAcknowledged {
            conv_id, logical_seq: seq,
            tool_call_id: format!("tc_{i}"),
            span_id: format!("sp_{i}"),
        });
    }

    // Complete cancellation
    seq += 1;
    events.push(RuntimeEvent::CancellationCompleted {
        conv_id, logical_seq: seq, clean: true,
    });

    // Verify cancellation resolved
    assert!(!RuntimeStateView::is_cancellation_pending(&events, conv_id));
    // All 5 should have acknowledged
    assert_eq!(RuntimeStateView::cancellation_ack_count(&events, conv_id), 2);
    // 3 completed before cancel
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), 3);
    // 0 failures
    assert_eq!(RuntimeStateView::failure_count(&events, conv_id), 0);
    // Tokens only from the 3 completed
    assert_eq!(RuntimeStateView::total_tokens(&events, conv_id), 30);
}

#[test]
fn soak_replay_ordering_is_ascending() {
    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;

    // Build a non-trivial event stream
    for i in 0..50 {
        seq += 1;
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(),
            tool_call_id: format!("tc_{i}"),
            span_id: format!("sp_{i}"),
            attempt: 1,
        });
        seq += 1;
        let fail = i % 3 == 0;
        if fail {
            events.push(RuntimeEvent::ToolCallFailed {
                conv_id, logical_seq: seq,
                tool_name: "grep".into(),
                tool_call_id: format!("tc_{i}"),
                span_id: format!("sp_{i}"),
                attempt: 1,
                error_signature: "ERR".into(),
                retryable: true,
                execution_unit_id: 0,
            });
            seq += 1;
            events.push(RuntimeEvent::RetryScheduled {
                conv_id, logical_seq: seq,
                tool_call_id: format!("tc_{i}"),
                attempt: 2,
                suggested_fix: "retry".into(),
            });
        } else {
            events.push(RuntimeEvent::ToolCallCompleted {
                conv_id, logical_seq: seq,
                tool_name: "grep".into(),
                tool_call_id: format!("tc_{i}"),
                span_id: format!("sp_{i}"),
                attempt: 1,
                tokens_spent: (i * 3) as u64,
                cache_hit: i % 4 == 0,
                execution_unit_id: (i * 10) as i64,
            });
        }
    }

    // Verify strict ascending order
    let ordered = RuntimeStateView::ordered_by_seq(&events, conv_id);
    let seqs: Vec<i64> = ordered.iter().map(|e| e.logical_seq()).collect();
    for w in seqs.windows(2) {
        assert!(w[0] < w[1], "replay ordering violated: {:?}", seqs);
    }

    // Verify event count
    assert_eq!(ordered.len(), events.len());
}

#[test]
fn soak_partial_failure_and_mixed_completion() {
    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;
    let mut expected_tokens: u64 = 0;
    let mut expected_cache_hits: usize = 0;
    let mut expected_failures: usize = 0;
    let mut expected_completions: usize = 0;

    seq += 1;
    events.push(RuntimeEvent::ExecutionStarted {
        conv_id, logical_seq: seq, profile: "efficient".into(),
    });

    // Interleaved: complete, fail, complete, fail+retry+complete, fail+abort
    let scenarios = [
        ("complete", 0),
        ("fail_once", 0),
        ("complete", 1),
        ("retry_then_complete", 2),
        ("fail_abort", 3),
    ];

    for (i, (scenario, tool_idx)) in scenarios.iter().enumerate() {
        let tcid = format!("tc_{}", i as i64);
        seq += 1;
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: format!("tool_{tool_idx}"),
            tool_call_id: tcid.clone(),
            span_id: format!("sp_{i}"),
            attempt: 1,
        });

        match *scenario {
            "complete" => {
                seq += 1;
                let tokens: u64 = 50;
                events.push(RuntimeEvent::ToolCallCompleted {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}"),
                    attempt: 1,
                    tokens_spent: tokens,
                    cache_hit: true,
                    execution_unit_id: i as i64 * 100,
                });
                expected_completions += 1;
                expected_tokens += tokens;
                expected_cache_hits += 1;
            }
            "fail_once" => {
                seq += 1;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}"),
                    attempt: 1,
                    error_signature: "ERR".into(),
                    retryable: false,
                    execution_unit_id: 0,
                });
                expected_failures += 1;
            }
            "retry_then_complete" => {
                seq += 1;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}"),
                    attempt: 1,
                    error_signature: "ERR".into(),
                    retryable: true,
                    execution_unit_id: 0,
                });
                expected_failures += 1;
                seq += 1;
                events.push(RuntimeEvent::RetryScheduled {
                    conv_id, logical_seq: seq,
                    tool_call_id: tcid.clone(),
                    attempt: 2,
                    suggested_fix: "fix".into(),
                });
                seq += 1;
                events.push(RuntimeEvent::ToolCallScheduled {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}_r"),
                    attempt: 2,
                });
                seq += 1;
                let tokens: u64 = 30;
                events.push(RuntimeEvent::ToolCallCompleted {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}_r"),
                    attempt: 2,
                    tokens_spent: tokens,
                    cache_hit: false,
                    execution_unit_id: i as i64 * 100 + 1,
                });
                expected_completions += 1;
                expected_tokens += tokens;
            }
            "fail_abort" => {
                seq += 1;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: format!("tool_{tool_idx}"),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}"),
                    attempt: 1,
                    error_signature: "FATAL".into(),
                    retryable: false,
                    execution_unit_id: 0,
                });
                expected_failures += 1;
                seq += 1;
                events.push(RuntimeEvent::RetryAborted {
                    conv_id, logical_seq: seq,
                    tool_call_id: tcid.clone(),
                    total_attempts: 1,
                    reason: "fatal".into(),
                });
            }
            _ => unreachable!(),
        }
    }

    // Verify parity
    assert_eq!(RuntimeStateView::total_tokens(&events, conv_id), expected_tokens);
    assert_eq!(RuntimeStateView::cache_hit_count(&events, conv_id), expected_cache_hits);
    assert_eq!(RuntimeStateView::failure_count(&events, conv_id), expected_failures);
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), expected_completions);

    let diffs = RuntimeStateView::projection_parity_check(
        &events, conv_id,
        expected_tokens, expected_cache_hits,
        expected_failures, expected_completions,
    );
    assert!(diffs.is_empty(), "soak parity diffs: {:?}", diffs);

    // Verify specific tool calls
    assert!(RuntimeStateView::was_retried(&events, "tc_3")); // retry_then_complete
    assert!(!RuntimeStateView::was_retried(&events, "tc_0")); // complete
    assert_eq!(RuntimeStateView::current_attempt(&events, "tc_3"), 2);
    assert_eq!(RuntimeStateView::current_attempt(&events, "tc_0"), 1);
}

#[test]
fn replay_dry_run_parity_and_ordering() {
    use deeplossless::runtime_events::{CancellationSource, RuntimeEvent};
    use deeplossless::runtime_state_view::RuntimeStateView;

    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;

    seq += 1;
    events.push(RuntimeEvent::ExecutionStarted {
        conv_id, logical_seq: seq, profile: "efficient".into(),
    });

    // Build a realistic session: 3 tools, some fail, some retry, cancel
    let scenarios: Vec<(&str, bool, u32)> = vec![
        ("grep", false, 1),       // simple success
        ("read_file", true, 3),   // fails 3 times, aborts
        ("edit_file", true, 2),   // fails once, retry succeeds
        ("cargo_check", false, 1), // succeeds
    ];

    let mut expected_completions = 0usize;
    let mut expected_failures = 0usize;
    let mut expected_tokens: u64 = 0;
    let mut expected_cache: usize = 0;

    for (i, (tool_name, will_retry, max_attempts)) in scenarios.iter().enumerate() {
        let tcid = format!("tc_{i}");
        let abort_after = if *will_retry { *max_attempts } else { 1 };

        for attempt in 1..=abort_after {
            seq += 1;
            events.push(RuntimeEvent::ToolCallScheduled {
                conv_id, logical_seq: seq,
                tool_name: tool_name.to_string(),
                tool_call_id: tcid.clone(),
                span_id: format!("sp_{i}_{attempt}"),
                attempt,
            });

            let is_last = attempt == abort_after;
            let last_retry_success = *will_retry && is_last && attempt > 1 && i == 2;

            if !is_last || last_retry_success {
                if is_last {
                    // Retry succeeded
                    seq += 1;
                    let t = (i * 30) as u64;
                    events.push(RuntimeEvent::ToolCallCompleted {
                        conv_id, logical_seq: seq,
                        tool_name: tool_name.to_string(),
                        tool_call_id: tcid.clone(),
                        span_id: format!("sp_{i}_{attempt}"),
                        attempt,
                        tokens_spent: t,
                        cache_hit: false,
                        execution_unit_id: (i * 10 + attempt as usize) as i64,
                    });
                    expected_completions += 1;
                    expected_tokens += t;
                } else {
                    // Will retry
                    seq += 1;
                    events.push(RuntimeEvent::ToolCallFailed {
                        conv_id, logical_seq: seq,
                        tool_name: tool_name.to_string(),
                        tool_call_id: tcid.clone(),
                        span_id: format!("sp_{i}_{attempt}"),
                        attempt,
                        error_signature: format!("ERR_{attempt}"),
                        retryable: true,
                        execution_unit_id: 0,
                    });
                    expected_failures += 1;
                    seq += 1;
                    events.push(RuntimeEvent::RetryScheduled {
                        conv_id, logical_seq: seq,
                        tool_call_id: tcid.clone(),
                        attempt: attempt + 1,
                        suggested_fix: format!("retry_fix_{i}_{attempt}"),
                    });
                }
            } else if *will_retry && is_last && !last_retry_success {
                // Abort
                seq += 1;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: tool_name.to_string(),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}_{attempt}"),
                    attempt,
                    error_signature: format!("FATAL_{attempt}"),
                    retryable: false,
                    execution_unit_id: 0,
                });
                expected_failures += 1;
                seq += 1;
                events.push(RuntimeEvent::RetryAborted {
                    conv_id, logical_seq: seq,
                    tool_call_id: tcid.clone(),
                    total_attempts: attempt,
                    reason: "max retries".into(),
                });
            } else {
                // Simple success
                seq += 1;
                let t = (i * 25) as u64;
                events.push(RuntimeEvent::ToolCallCompleted {
                    conv_id, logical_seq: seq,
                    tool_name: tool_name.to_string(),
                    tool_call_id: tcid.clone(),
                    span_id: format!("sp_{i}_{attempt}"),
                    attempt,
                    tokens_spent: t,
                    cache_hit: i == 3,
                    execution_unit_id: (i * 10 + 1) as i64,
                });
                expected_completions += 1;
                expected_tokens += t;
                if i == 3 { expected_cache += 1; }
            }
        }
    }

    // Cancel near end
    seq += 1;
    events.push(RuntimeEvent::CancellationRequested {
        conv_id, logical_seq: seq,
        source: CancellationSource::Shutdown,
    });
    seq += 1;
    events.push(RuntimeEvent::CancellationCompleted {
        conv_id, logical_seq: seq, clean: true,
    });

    // REPLAY: validate ordering
    let ordered = RuntimeStateView::ordered_by_seq(&events, conv_id);
    for w in ordered.windows(2) {
        assert!(w[0].logical_seq() < w[1].logical_seq(),
            "replay ordering broken at seq {} -> {}",
            w[0].logical_seq(), w[1].logical_seq());
    }

    // REPLAY: validate parity
    assert_eq!(RuntimeStateView::total_tokens(&events, conv_id), expected_tokens);
    assert_eq!(RuntimeStateView::cache_hit_count(&events, conv_id), expected_cache);
    assert_eq!(RuntimeStateView::failure_count(&events, conv_id), expected_failures);
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), expected_completions);

    let diffs = RuntimeStateView::projection_parity_check(
        &events, conv_id,
        expected_tokens, expected_cache, expected_failures, expected_completions,
    );
    assert!(diffs.is_empty(), "replay parity broken: {:?}", diffs);

    // Verify specific retry behavior
    assert_eq!(RuntimeStateView::retry_count(&events, conv_id), 3); // read_file(2) + edit_file(1)
    assert_eq!(RuntimeStateView::retry_aborted_count(&events, conv_id), 1); // read_file
    assert!(RuntimeStateView::was_retried(&events, "tc_1")); // read_file
    assert!(RuntimeStateView::was_retried(&events, "tc_2")); // edit_file
    assert!(!RuntimeStateView::was_retried(&events, "tc_0")); // grep
    assert!(!RuntimeStateView::is_cancellation_pending(&events, conv_id));
}

// ── Semantic Regression / Invariant Contract Tests (Phase 2.7) ────────

#[test]
fn invariant_cancellation_requested_must_precede_resolved() {
    use deeplossless::runtime_events::{CancellationSource, RuntimeEvent};
    use deeplossless::runtime_state_view::RuntimeStateView;

    let conv_id: i64 = 1;
    let correct = vec![
        RuntimeEvent::CancellationRequested { conv_id, logical_seq: 1, source: CancellationSource::Shutdown },
        RuntimeEvent::CancellationAcknowledged { conv_id, logical_seq: 2, tool_call_id: "tc".into(), span_id: "sp".into() },
        RuntimeEvent::CancellationCompleted { conv_id, logical_seq: 3, clean: true },
    ];
    assert!(!RuntimeStateView::is_cancellation_pending(&correct, conv_id));
}

#[test]
fn invariant_retry_must_increment_attempt() {
    use deeplossless::runtime_events::RuntimeEvent;
    use deeplossless::runtime_state_view::RuntimeStateView;

    let tcid = "tc_1";
    let events = vec![
        RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 1, tool_name: "grep".into(), tool_call_id: tcid.into(), span_id: "s1".into(), attempt: 1 },
        RuntimeEvent::ToolCallFailed { conv_id: 1, logical_seq: 2, tool_name: "grep".into(), tool_call_id: tcid.into(), span_id: "s1".into(), attempt: 1, error_signature: "ERR".into(), retryable: true, execution_unit_id: 0 },
        RuntimeEvent::RetryScheduled { conv_id: 1, logical_seq: 3, tool_call_id: tcid.into(), attempt: 2, suggested_fix: "fix".into() },
        RuntimeEvent::ToolCallScheduled { conv_id: 1, logical_seq: 4, tool_name: "grep".into(), tool_call_id: tcid.into(), span_id: "s2".into(), attempt: 2 },
        RuntimeEvent::ToolCallCompleted { conv_id: 1, logical_seq: 5, tool_name: "grep".into(), tool_call_id: tcid.into(), span_id: "s2".into(), attempt: 2, tokens_spent: 10, cache_hit: false, execution_unit_id: 42 },
    ];

    assert_eq!(RuntimeStateView::current_attempt(&events, tcid), 2);
    assert!(RuntimeStateView::was_retried(&events, tcid));
    assert_eq!(RuntimeStateView::retry_count(&events, 1), 1);
}

#[test]
fn invariant_logical_seq_must_be_strictly_monotonic() {
    use deeplossless::runtime_events::RuntimeEvent;
    use deeplossless::runtime_state_view::RuntimeStateView;

    let mut events = Vec::new();
    for seq in 1..=100 {
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id: 1, logical_seq: seq,
            tool_name: "grep".into(), tool_call_id: format!("tc_{}", seq % 5),
            span_id: format!("sp_{seq}"), attempt: 1,
        });
    }
    let ordered = RuntimeStateView::ordered_by_seq(&events, 1);
    for w in ordered.windows(2) {
        assert!(w[0].logical_seq() < w[1].logical_seq(),
            "logical_seq must be strictly ascending");
    }
}

#[test]
fn invariant_no_duplicate_completion_per_attempt() {
    use deeplossless::runtime_events::RuntimeEvent;
    use deeplossless::runtime_state_view::RuntimeStateView;

    let conv_id: i64 = 1;
    // Duplicate completion (shouldn't happen, but documents event-vs-projection gap)
    let events = vec![
        RuntimeEvent::ToolCallScheduled { conv_id, logical_seq: 1, tool_name: "grep".into(), tool_call_id: "tc".into(), span_id: "s1".into(), attempt: 1 },
        RuntimeEvent::ToolCallCompleted { conv_id, logical_seq: 2, tool_name: "grep".into(), tool_call_id: "tc".into(), span_id: "s1".into(), attempt: 1, tokens_spent: 10, cache_hit: false, execution_unit_id: 1 },
        RuntimeEvent::ToolCallCompleted { conv_id, logical_seq: 3, tool_name: "grep".into(), tool_call_id: "tc".into(), span_id: "s1".into(), attempt: 1, tokens_spent: 20, cache_hit: false, execution_unit_id: 2 },
    ];
    // Events count both; a projection would only count 1 (gap documentation)
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), 2);
    assert_eq!(RuntimeStateView::total_tokens(&events, conv_id), 30);
    assert_eq!(RuntimeStateView::current_attempt(&events, "tc"), 1);
}

// ── Operational Reality Tests (Phase 5) ─────────────────────────────

/// 1000 execution cycles with random outcomes, verify no parity drift.
#[test]
fn operational_soak_1000_cycles_no_parity_drift() {
    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;
    let mut expected_tokens: u64 = 0;
    let mut expected_completions: usize = 0;
    let mut expected_failures: usize = 0;
    let mut expected_cache: usize = 0;
    let mut expected_retries: usize = 0;

    seq += 1;
    events.push(RuntimeEvent::ExecutionStarted { conv_id, logical_seq: seq, profile: "efficient".into() });

    for cycle in 0..1000 {
        let tcid = format!("tc_{cycle}");
        let will_fail = cycle % 7 == 0;
        let will_retry = will_fail && cycle % 3 != 0;
        let attempts: u32 = if will_retry { 2 + (cycle as u32 % 3) } else { 1 };

        for attempt in 1..=attempts {
            seq += 1;
            events.push(RuntimeEvent::ToolCallScheduled {
                conv_id, logical_seq: seq,
                tool_name: "grep".into(),
                tool_call_id: tcid.clone(),
                span_id: format!("sp_{cycle}_{attempt}"),
                attempt,
            });

            let is_last = attempt == attempts;
            if !will_fail || (will_retry && is_last && attempt > 1) {
                seq += 1;
                let tokens = (cycle % 50 + 1) as u64 * 5;
                let cache = cycle % 4 == 0;
                events.push(RuntimeEvent::ToolCallCompleted {
                    conv_id, logical_seq: seq,
                    tool_name: "grep".into(), tool_call_id: tcid.clone(),
                    span_id: format!("sp_{cycle}_{attempt}"), attempt,
                    tokens_spent: tokens, cache_hit: cache,
                    execution_unit_id: (cycle * 10 + attempt as usize) as i64,
                });
                expected_tokens += tokens;
                expected_completions += 1;
                if cache { expected_cache += 1; }
            } else {
                seq += 1;
                let retryable = !is_last;
                events.push(RuntimeEvent::ToolCallFailed {
                    conv_id, logical_seq: seq,
                    tool_name: "grep".into(), tool_call_id: tcid.clone(),
                    span_id: format!("sp_{cycle}_{attempt}"), attempt,
                    error_signature: format!("ERR_{attempt}"), retryable,
                    execution_unit_id: 0,
                });
                expected_failures += 1;
                if retryable {
                    seq += 1;
                    events.push(RuntimeEvent::RetryScheduled {
                        conv_id, logical_seq: seq,
                        tool_call_id: tcid.clone(),
                        attempt: attempt + 1,
                        suggested_fix: "retry".into(),
                    });
                    expected_retries += 1;
                }
            }
        }

        // Random cancellation every ~50 cycles
        if cycle > 0 && cycle % 47 == 0 {
            seq += 1;
            events.push(RuntimeEvent::CancellationRequested {
                conv_id, logical_seq: seq,
                source: deeplossless::runtime_events::CancellationSource::Shutdown,
            });
            seq += 1;
            events.push(RuntimeEvent::CancellationCompleted {
                conv_id, logical_seq: seq, clean: true,
            });
        }
    }

    // Verify no parity drift after 1000 cycles
    assert_eq!(RuntimeStateView::total_tokens(&events, conv_id), expected_tokens);
    assert_eq!(RuntimeStateView::cache_hit_count(&events, conv_id), expected_cache);
    assert_eq!(RuntimeStateView::failure_count(&events, conv_id), expected_failures);
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), expected_completions);
    assert_eq!(RuntimeStateView::retry_count(&events, conv_id), expected_retries);

    // Verify ordering
    let ordered = RuntimeStateView::ordered_by_seq(&events, conv_id);
    for w in ordered.windows(2) {
        assert!(w[0].logical_seq() < w[1].logical_seq());
    }

    // Verify explain() doesn't panic on large event set
    let _ = RuntimeStateView::explain(&events);
    let _ = RuntimeStateView::inspect(&events, conv_id);
}

/// Retry storm: 500 failures with aggressive retry, verify attempt tracking.
#[test]
fn operational_retry_storm_attempt_tracking() {
    let mut events = Vec::new();
    let conv_id: i64 = 1;
    let mut seq: i64 = 0;
    let storm_tcid = "tc_storm";

    for fail_count in 1..=500 {
        let attempt = fail_count as u32;
        seq += 1;
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(), tool_call_id: storm_tcid.into(),
            span_id: format!("sp_s{attempt}"), attempt,
        });
        seq += 1;
        let retryable = fail_count < 500;
        events.push(RuntimeEvent::ToolCallFailed {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(), tool_call_id: storm_tcid.into(),
            span_id: format!("sp_s{attempt}"), attempt,
            error_signature: format!("ERR_{attempt}"), retryable,
            execution_unit_id: 0,
        });
        if retryable {
            seq += 1;
            events.push(RuntimeEvent::RetryScheduled {
                conv_id, logical_seq: seq,
                tool_call_id: storm_tcid.into(),
                attempt: attempt + 1,
                suggested_fix: format!("fix_{attempt}"),
            });
        }
    }
    // Final abort
    seq += 1;
    events.push(RuntimeEvent::RetryAborted {
        conv_id, logical_seq: seq,
        tool_call_id: storm_tcid.into(),
        total_attempts: 500,
        reason: "max retries".into(),
    });

    // Verify
    assert_eq!(RuntimeStateView::current_attempt(&events, storm_tcid), 500);
    assert_eq!(RuntimeStateView::retry_count(&events, conv_id), 499);
    assert_eq!(RuntimeStateView::failure_count(&events, conv_id), 500);
    assert_eq!(RuntimeStateView::completed_count(&events, conv_id), 0);
}

/// Dependency churn: many cross-conversation events, verify per-conv isolation.
#[test]
fn operational_cross_conv_isolation() {
    let mut events = Vec::new();
    let mut seq: i64 = 0;

    // 5 conversations, interleaved
    for cycle in 0..200 {
        let conv_id: i64 = (cycle % 5) + 1;
        seq += 1;
        if cycle % 20 == 0 {
            events.push(RuntimeEvent::ExecutionStarted {
                conv_id, logical_seq: seq, profile: "efficient".into(),
            });
        }
        seq += 1;
        let tcid = format!("tc_{cycle}");
        events.push(RuntimeEvent::ToolCallScheduled {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(), tool_call_id: tcid.clone(),
            span_id: format!("sp_{cycle}"), attempt: 1,
        });
        seq += 1;
        events.push(RuntimeEvent::ToolCallCompleted {
            conv_id, logical_seq: seq,
            tool_name: "grep".into(), tool_call_id: tcid,
            span_id: format!("sp_{cycle}"), attempt: 1,
            tokens_spent: 10, cache_hit: false,
            execution_unit_id: cycle as i64,
        });
    }

    // Verify per-conversation isolation
    for conv_id in 1..=5 {
        let comps = RuntimeStateView::completed_count(&events, conv_id);
        assert!(comps > 0, "conv {conv_id} should have completions");
        let other_conv = (conv_id % 5) + 1;
        let other_comps = RuntimeStateView::completed_count(&events, other_conv);
        assert!(other_comps > 0);
        // Each conv should have roughly equal counts
        let diff = (comps as i64 - other_comps as i64).abs();
        assert!(diff <= 2, "conv {conv_id} vs {other_conv}: unbalanced ({comps} vs {other_comps})");
    }
}

