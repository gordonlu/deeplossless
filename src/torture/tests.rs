use crate::torture::adversarial::{self, BaseTemplate, Mutation, NoiseType, TurnSpec};
use crate::torture::trace::{self, ScenarioTrace};

// ── Synthetic Trace Generators ────────────────────────────────────

#[test]
fn gen_loop_creates_correct_length() {
    let t = trace::gen_loop("test", "grep", 5);
    assert_eq!(t.turns.len(), 5);
    assert!(t.name.contains("5x"));
    for turn in &t.turns {
        assert_eq!(turn.tool_calls, vec!["grep"]);
        assert!(turn.completion.contains("grep"));
    }
}

#[test]
fn gen_cache_stress_has_hits() {
    let t = trace::gen_cache_stress("test", 10, 0.6);
    assert_eq!(t.turns.len(), 10);
    let zero_token_count = t.turns.iter().filter(|t| t.tokens == 0).count();
    assert!(zero_token_count >= 4, "should have some cache hits at 60% rate");
}

#[test]
fn gen_chaos_has_failures() {
    let t = trace::gen_chaos("test", &["grep", "read", "edit"], 1.0);
    assert_eq!(t.turns.len(), 3);
    assert!(t.turns.iter().all(|t| t.completion.contains("ERROR")));
}

// ── ScenarioTrace Serialization ───────────────────────────────────

#[test]
fn scenario_trace_roundtrip_json() {
    let t = trace::gen_loop("roundtrip", "test", 3);
    let json = serde_json::to_string(&t).unwrap();
    let back: ScenarioTrace = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, t.name);
    assert_eq!(back.turns.len(), 3);
    assert_eq!(back.turns[0].tool_calls, vec!["test"]);
}

// ── Built-in Traces ───────────────────────────────────────────────

#[test]
fn builtin_traces_are_valid() {
    let traces = trace::builtin_traces();
    assert!(!traces.is_empty(), "should have at least one built-in trace");
    for t in &traces {
        assert!(!t.name.is_empty());
        assert!(!t.turns.is_empty());
    }
}

// ── Adversarial generation ────────────────────────────────────────

#[test]
fn base_template_render_uses_counter() {
    let t = BaseTemplate {
        name: "test".into(),
        description: "".into(),
        turns: vec![
            TurnSpec { role: "user".into(), content_template: "q{n}".into(), tool_calls: vec![] },
            TurnSpec { role: "assistant".into(), content_template: "a{n}".into(), tool_calls: vec![] },
        ],
    };
    let trace = t.render();
    assert_eq!(trace.turns.len(), 2);
    assert_eq!(trace.turns[0].prompt, "q0");
    assert_eq!(trace.turns[1].prompt, "a1");
}

#[test]
fn builtin_templates_have_content() {
    for t in adversarial::builtin_templates() {
        assert!(!t.name.is_empty());
        assert!(!t.turns.is_empty());
        for spec in &t.turns {
            assert!(!spec.content_template.is_empty());
        }
    }
}

#[test]
fn mutation_duplicate_turn() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[Mutation::DuplicateTurn { index: 0, times: 2, vary_context: false }], &[], "test");
    assert_eq!(trace.turns.len(), base.turns.len() + 2);
    assert_eq!(trace.turns[0].prompt, trace.turns[1].prompt);
    assert_eq!(trace.turns[1].prompt, trace.turns[2].prompt);
}

#[test]
fn mutation_duplicate_with_vary() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[Mutation::DuplicateTurn { index: 0, times: 2, vary_context: true }], &[], "test");
    assert_eq!(trace.turns.len(), base.turns.len() + 2);
    assert_ne!(trace.turns[0].prompt, trace.turns[1].prompt);
    assert!(trace.turns[1].prompt.contains("[attempt 1]"));
}

#[test]
fn mutation_reorder_turns() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[Mutation::ReorderTurns { a: 0, b: 1 }], &[], "test");
    assert_eq!(trace.turns[0].prompt, base.turns[1].content_template.replace("{n}", "1"));
    assert_eq!(trace.turns[1].prompt, base.turns[0].content_template.replace("{n}", "0"));
}

#[test]
fn mutation_drop_turn() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[Mutation::DropTurn(0)], &[], "test");
    assert_eq!(trace.turns.len(), base.turns.len() - 1);
    assert_eq!(trace.turns[0].prompt, base.turns[1].content_template.replace("{n}", "1"));
}

#[test]
fn mutation_identical_prompt() {
    let base = &adversarial::builtin_templates()[2];
    let trace = adversarial::generate(base, &[Mutation::IdenticalPrompt { source: 0, target: 2 }], &[], "test");
    assert_eq!(trace.turns[0].prompt, trace.turns[2].prompt);
}

#[test]
fn noise_trailing_whitespace() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[], &[(0, NoiseType::TrailingWhitespace)], "test");
    assert!(trace.turns[0].prompt.ends_with("  \n"));
}

#[test]
fn noise_unicode_homoglyph() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[], &[(0, NoiseType::UnicodeHomoglyph)], "test");
    assert_ne!(trace.turns[0].prompt, base.turns[0].content_template.replace("{n}", "0"));
}

#[test]
fn noise_truncate() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[], &[(0, NoiseType::Truncate(3))], "test");
    assert_eq!(trace.turns[0].prompt.len(), 3);
}

#[test]
fn adversarial_all_generated_unique_names() {
    let traces = adversarial::all_adversarial();
    assert_eq!(traces.len(), 9);
    let mut names = std::collections::HashSet::new();
    for t in &traces {
        assert!(names.insert(&t.name), "duplicate name: {}", t.name);
        assert!(!t.turns.is_empty());
    }
}

#[test]
fn full_pipeline_smoke() {
    let base = &adversarial::builtin_templates()[0];
    let trace = adversarial::generate(base, &[
        Mutation::DuplicateTurn { index: 0, times: 3, vary_context: false },
    ], &[(1, NoiseType::TrailingWhitespace)], "smoke");

    let json = serde_json::to_string(&trace).unwrap();
    let loaded: ScenarioTrace = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.turns.len(), trace.turns.len());

    for t in adversarial::all_adversarial() {
        let json = serde_json::to_string(&t).unwrap();
        let loaded: ScenarioTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.turns.len(), t.turns.len());
    }
}

// ── ACES Scenario tests ───────────────────────────────────────────

#[test]
fn scenario_score_search_before_edit() {
    use crate::torture::scenario::*;
    let run = ScenarioRun {
        scenario: "test".into(),
        events: vec![
            AgentEvent::Search("foo".into(), "src/lib.rs:1:hit".into()),
            AgentEvent::Read("src/lib.rs".into(), String::new()),
            AgentEvent::Edit("src/lib.rs".into(), String::new()),
        ],
        terminal_state: Some("success".into()),
        score: None,
        expected_search: false,
        expected_read: false,
    };
    let score = score_run(&run);
    assert!(score.reuse > 0.0, "should have reuse for search before edit (got {})", score.reuse);
    assert!(score.tool_strategy > 0.0, "should have tool strategy (got {})", score.tool_strategy);
    assert!(score.search_efficiency > 0.0, "search followed by read should score (got {})", score.search_efficiency);
    assert!(score.context_efficiency > 0.0, "read followed by edit should score (got {})", score.context_efficiency);
}

#[test]
fn scenario_score_no_search_before_edit() {
    use crate::torture::scenario::*;
    let run = ScenarioRun {
        scenario: "test".into(),
        events: vec![
            AgentEvent::Edit("src/lib.rs".into(), String::new()),
            AgentEvent::Done,
        ],
        terminal_state: Some("success".into()),
        score: None,
        expected_search: false,
        expected_read: false,
    };
    let score = score_run(&run);
    // No searches and no reads → vacuous perfect (20/20)
    assert_eq!(score.search_efficiency, 20.0, "no searches, no expected_search → vacuous perfect");
    assert_eq!(score.context_efficiency, 20.0, "no reads, no expected_read → vacuous perfect");
    // Reached terminal but not verify → only 5/20 correctness goal
    assert_eq!(score.correctness, 5.0, "reached terminal but not verify");
}

#[test]
fn scenario_score_no_search_when_expected_is_penalty() {
    use crate::torture::scenario::*;
    let run = ScenarioRun {
        scenario: "test".into(),
        events: vec![
            AgentEvent::Read("src/lib.rs".into(), String::new()),
            AgentEvent::Edit("src/lib.rs".into(), String::new()),
        ],
        terminal_state: Some("success".into()),
        score: None,
        expected_search: true,   // agent was supposed to search
        expected_read: false,
    };
    let score = score_run(&run);
    assert_eq!(score.search_efficiency, 0.0, "expected_search=true with 0 searches → 0/20");
    assert_eq!(score.context_efficiency, 20.0, "read→edit within window → action counts");
}

#[test]
fn scenario_score_verify_after_edit() {
    use crate::torture::scenario::*;
    let run = ScenarioRun {
        scenario: "test".into(),
        events: vec![
            AgentEvent::Search("foo".into(), "src/lib.rs:1:hit".into()),
            AgentEvent::Read("src/lib.rs".into(), String::new()),
            AgentEvent::Edit("src/lib.rs".into(), String::new()),
            AgentEvent::Test("cargo test".into(), String::new()),
        ],
        terminal_state: Some("verify".into()),
        score: None,
        expected_search: false,
        expected_read: false,
    };
    let score = score_run(&run);
    assert!(score.verification > 0.0, "should have verification (got {})", score.verification);
    assert!(score.tool_strategy > 0.0, "should have tool strategy (got {})", score.tool_strategy);
    assert!(score.correctness >= 10.0, "correctness should be >=10 (got {})", score.correctness);
}

// ── Search Efficiency tests ───────────────────────────────────────

#[test]
fn search_efficiency_targeting_step_function() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_search_efficiency;

    // 3 files → targeting=1.0; no Read within window → to_read=0; novelty=1.0
    // (0 * 0.5 + 1.0 * 0.25 + 1.0 * 0.25) * 20 = 10.0
    let three = vec![AgentEvent::Search("q".into(), "a.rs:1:x\nb.rs:1:x\nc.rs:1:x".into())];
    let s = compute_search_efficiency(&three, false);
    assert!((s - 10.0).abs() < 0.01, "3 files targeting=1.0, to_read=0, novelty=1.0 → 10.0 got {s}");

    // 10 files → targeting=0.8; to_read=0; novelty=1.0
    // (0 * 0.5 + 0.8 * 0.25 + 1.0 * 0.25) * 20 = 0.45 * 20 = 9.0
    let ten = "a.rs:1:x\nb.rs:1:x\nc.rs:1:x\nd.rs:1:x\ne.rs:1:x\nf.rs:1:x\ng.rs:1:x\nh.rs:1:x\ni.rs:1:x\nj.rs:1:x";
    let events = vec![AgentEvent::Search("q".into(), ten.into())];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 9.0).abs() < 0.01, "10 files targeting=0.8 → 9.0 got {s}");

    // 100 files → targeting=0.2; to_read=0; novelty=1.0
    // (0 * 0.5 + 0.2 * 0.25 + 1.0 * 0.25) * 20 = 6.0
    let many = (0..100).map(|i| format!("f{i}.rs:1:x")).collect::<Vec<_>>().join("\n");
    let events = vec![AgentEvent::Search("q".into(), many)];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 6.0).abs() < 0.01, "100 files targeting=0.2 → 6.0 got {s}");
}

#[test]
fn search_efficiency_targeting_parse_failure_is_neutral() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_search_efficiency;

    // Empty result → parse fail → targeting=0.5 (neutral)
    // to_read=0; novelty=1.0
    // (0 * 0.5 + 0.5 * 0.25 + 1.0 * 0.25) * 20 = 0.375 * 20 = 7.5
    let events = vec![AgentEvent::Search("q".into(), String::new())];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 7.5).abs() < 0.01, "empty result → 0.5 neutral expected 7.5 got {s}");

    // Plain text with no `:` → parse fail → same
    let events = vec![AgentEvent::Search("q".into(), "no colons here just text".into())];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 7.5).abs() < 0.01, "no-colon text → 0.5 expected 7.5 got {s}");
}

#[test]
fn search_efficiency_to_read_window() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_search_efficiency;

    // Search → Read in same event window: to_read=1.0
    // 1 file → targeting=1.0; novelty=1.0
    // (1.0 * 0.5 + 1.0 * 0.25 + 1.0 * 0.25) * 20 = 20.0
    let events = vec![
        AgentEvent::Search("q".into(), "f.rs:1:x".into()),
        AgentEvent::Read("f.rs".into(), String::new()),
    ];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 20.0).abs() < 0.01, "search+read+1file → 20.0 got {s}");

    // Search → 3 non-Read events → Read: to_read=0
    // 1 file → targeting=1.0; novelty=1.0
    // (0 * 0.5 + 1.0 * 0.25 + 1.0 * 0.25) * 20 = 10.0
    let events = vec![
        AgentEvent::Search("q".into(), "f.rs:1:x".into()),
        AgentEvent::Task("a".into(), String::new()),
        AgentEvent::Task("b".into(), String::new()),
        AgentEvent::Task("c".into(), String::new()),
        AgentEvent::Read("f.rs".into(), String::new()),
    ];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 10.0).abs() < 0.01, "search with no read in 3 events → to_read=0 expected 10.0 got {s}");
}

#[test]
fn search_efficiency_novelty_by_result_set_not_query() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_search_efficiency;

    // Same query, different result sets → novel
    // to_read=1.0 (both searches followed by read); targeting=1.0; novelty=1.0
    // (1.0 * 0.5 + 1.0 * 0.25 + 1.0 * 0.25) * 20 = 20.0
    let events = vec![
        AgentEvent::Search("timeout".into(), "auth.rs:1:x\nlogin.rs:1:x".into()),
        AgentEvent::Read("auth.rs".into(), String::new()),
        AgentEvent::Search("timeout".into(), "db.rs:1:x".into()),
        AgentEvent::Read("db.rs".into(), String::new()),
    ];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 20.0).abs() < 0.01, "same query different result sets → novelty=1.0 got {s}");

    // Same query, same result set → NOT novel
    // to_read=0.5 (first has read, second has no read within 3)
    // targeting=1.0; novelty=0.5
    // (0.5 * 0.5 + 1.0 * 0.25 + 0.5 * 0.25) * 20 = (0.25 + 0.25 + 0.125) * 20 = 12.5
    let events = vec![
        AgentEvent::Search("timeout".into(), "auth.rs:1:x\nlogin.rs:1:x".into()),
        AgentEvent::Read("auth.rs".into(), String::new()),
        AgentEvent::Search("timeout".into(), "auth.rs:5:y\nlogin.rs:3:z".into()),
    ];
    let s = compute_search_efficiency(&events, false);
    assert!((s - 12.5).abs() < 0.01, "same result set → novelty=0.5 expected 12.5 got {s}");
}

// ── Context Efficiency tests ──────────────────────────────────────

#[test]
fn context_efficiency_read_to_action_window() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_context_efficiency;

    // Read → Edit within 5: counts
    let events = vec![
        AgentEvent::Read("f.rs".into(), String::new()),
        AgentEvent::Edit("f.rs".into(), String::new()),
    ];
    let s = compute_context_efficiency(&events, false);
    // read_to_action=1.0, precision=1.0 (read and edit same file), reread=0
    // (1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2) * 20 = 20.0
    assert!((s - 20.0).abs() < 0.01, "read→edit same file → 20.0 got {s}");

    // Read with no follow-up action
    let events = vec![
        AgentEvent::Read("f.rs".into(), String::new()),
        AgentEvent::Done,
    ];
    let s = compute_context_efficiency(&events, false);
    // read_to_action=0, precision=0 (no edits), reread=0
    // (0 * 0.5 + 0 * 0.3 + 1.0 * 0.2) * 20 = 4.0
    assert!((s - 4.0).abs() < 0.01, "read→done (no action) → 4.0 got {s}");
}

#[test]
fn context_efficiency_reread_within_window_no_penalty() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_context_efficiency;

    // Two reads of same file, within 10 events, no Edit between → no penalty
    let events = vec![
        AgentEvent::Read("f.rs".into(), String::new()),
        AgentEvent::Read("f.rs".into(), String::new()),
        AgentEvent::Edit("f.rs".into(), String::new()),
    ];
    let s = compute_context_efficiency(&events, false);
    // read_to_action: 1/2 = 0.5 (only first read followed by Edit in window)
    //   wait, second read is followed by Edit, so to_action=2/2=1.0
    // precision: f.rs read and edited → 1.0
    // reread: 1 re-read, within window, no Edit/Test between → no penalty → 0
    // (1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2) * 20 = 20.0
    assert!((s - 20.0).abs() < 0.01, "reread within window → no penalty expected 20.0 got {s}");
}

#[test]
fn context_efficiency_reread_with_edit_resets_window() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_context_efficiency;

    // Read → Edit → Read (re-read justified by Edit)
    let events = vec![
        AgentEvent::Read("f.rs".into(), String::new()),
        AgentEvent::Edit("f.rs".into(), String::new()),
        AgentEvent::Read("f.rs".into(), String::new()),
    ];
    let s = compute_context_efficiency(&events, false);
    // read_to_action: 1st read → Edit within window = yes
    //                2nd read → no Edit/Test after = no
    //   so 1/2 = 0.5
    // precision: 1.0
    // reread: 2nd read has Edit(Test) on f.rs between 1st and 2nd → had_action = true → no penalty
    // (0.5 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2) * 20 = (0.25 + 0.30 + 0.20) * 20 = 15.0
    assert!((s - 15.0).abs() < 0.01, "Read→Edit→Read should be 15.0 got {s}");
}

#[test]
fn context_efficiency_reread_outside_window_penalized() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_context_efficiency;

    // 11 events of no-op filler, then re-read
    let mut events: Vec<AgentEvent> = vec![AgentEvent::Read("f.rs".into(), String::new())];
    for _ in 0..11 {
        events.push(AgentEvent::Task("noop".into(), String::new()));
    }
    events.push(AgentEvent::Read("f.rs".into(), String::new()));
    events.push(AgentEvent::Edit("f.rs".into(), String::new()));

    let s = compute_context_efficiency(&events, false);
    // 2 reads: first at idx 0, second at idx 12
    // distance = 12, > 10 (RE_READ_WINDOW)
    // no Edit/Test on f.rs between idx 0 and idx 12
    // → had_action = false, within_window = false → re_reads += 1
    // read_to_action: 1st read → no action within 5 events (Task is not Edit/Test) → no
    //                 2nd read → Edit within 5 → yes
    //   so 1/2 = 0.5
    // precision: f.rs edited → 1.0
    // reread_penalty: 1/2 = 0.5
    // (0.5 * 0.5 + 1.0 * 0.3 + 0.5 * 0.2) * 20 = (0.25 + 0.30 + 0.10) * 20 = 13.0
    assert!((s - 13.0).abs() < 0.01, "reread outside window → penalty expected 13.0 got {s}");
}

#[test]
fn context_efficiency_precision_read_but_not_edited() {
    use crate::torture::scenario::*;
    use crate::torture::scenario::compute_context_efficiency;

    // Read 3 files, edit 1 of them
    let events = vec![
        AgentEvent::Read("a.rs".into(), String::new()),
        AgentEvent::Read("b.rs".into(), String::new()),
        AgentEvent::Read("c.rs".into(), String::new()),
        AgentEvent::Edit("a.rs".into(), String::new()),
    ];
    let s = compute_context_efficiency(&events, false);
    // read_to_action: only the last read (c.rs) is NOT followed by action... wait
    //   reads at 0, 1, 2. After idx 2, next is Edit(a.rs) → 3rd read → action yes
    //   but Edit is on a.rs not c.rs. Does that matter? `read_to_action` checks if
    //   any Edit/Test is in the window, not whether it touches the same file.
    //   So all 3 reads are followed by Edit within 5 → to_action=1.0
    // precision: 1/3 (a.rs read and edited, b.rs and c.rs read but not edited)
    // reread: 0
    // (1.0 * 0.5 + 0.333 * 0.3 + 1.0 * 0.2) * 20 = (0.5 + 0.1 + 0.2) * 20 = 16.0
    assert!((s - 16.0).abs() < 0.5, "1/3 precision expected ~16.0 got {s}");
}

#[test]
fn scenario_state_machine_transitions() {
    use crate::torture::scenario::*;
    let yaml = r#"
name: test
description: test
start: start
vfs:
  files:
    "src/lib.rs": "fn hello() {}"
states:
  start:
    assistant: "Find and fix the bug"
    transitions:
      - type: on_tool
        tool: "hello"
        next: found
  found:
    assistant: "Found it"
    finish: true
"#;
    let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
    let mut sm = StateMachine::new(scenario);
    assert_eq!(sm.current_state_name(), "start");
    assert_eq!(sm.current_prompt(), Some("Find and fix the bug"));

    assert!(sm.feed(AgentEvent::Search("hello".into(), String::new())));
    assert_eq!(sm.current_state_name(), "found");
    assert!(sm.is_terminal());
    assert!(sm.is_success());
}

#[test]
fn scenario_state_machine_edit_result_verification() {
    use crate::torture::scenario::*;
    // The verify transition should only fire when the Edit result actually
    // contains the expected string. Wrong-content edits must NOT advance.
    let yaml = r#"
name: test
description: test
start: edit
vfs:
  files: {}
states:
  edit:
    assistant: "Edit the file"
    transitions:
      - type: on_edit_result
        pattern: "config.sh"
        result_contains: "TIMEOUT_DEFAULT=60"
        next: verify
      - type: on_tool
        tool: ""
        next: edit
  verify:
    assistant: "Verified"
    finish: true
"#;
    let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
    let mut sm = StateMachine::new(scenario);

    // Wrong content: OnEditResult doesn't match, but the on_tool wildcard
    // transition fires (stays at edit by looping). This verifies that
    // the verify transition ONLY fires for correct content.
    assert!(sm.feed(AgentEvent::Edit(
        "config.sh".into(),
        "TIMEOUT_DEFAULT=999".into(),
    )));
    assert_eq!(sm.current_state_name(), "edit");

    // Right content: advances to verify
    assert!(sm.feed(AgentEvent::Edit(
        "config.sh".into(),
        "The file has been updated. Now contains:\nTIMEOUT_DEFAULT=60\n".into(),
    )));
    assert_eq!(sm.current_state_name(), "verify");
    assert!(sm.is_terminal());
}

#[test]
fn scenario_extract_events_from_request() {
    use crate::torture::scenario::*;
    let body = serde_json::json!({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "edit", "arguments": "{\"filePath\":\"x\"}"}}
            ]},
            {"role": "user", "content": "Done"}
        ]
    });
    let events = extract_events_from_request(&body);
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Edit(_, _))), "should see edit");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Done)), "should see Done after last assistant");
}

#[test]
fn scenario_extract_only_last_turn() {
    use crate::torture::scenario::*;
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "fix timeout"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{\"pattern\":\"foo\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "edit", "arguments": "{\"filePath\":\"x\"}"}}
            ]}
        ]
    });
    let events = extract_events_from_request(&body);
    assert_eq!(events.len(), 1, "only last assistant tool_calls");
    assert!(matches!(events[0], AgentEvent::Edit(_, _)));
}

#[test]
fn extract_path_handles_snake_case() {
    use crate::torture::scenario::*;
    // Claude Code uses snake_case (file_path); the older camelCase (filePath)
    // and other tools' field names (path, pattern, command) must all work.
    let cases = [
        (r#"{"file_path":"/tmp/x"}"#, "/tmp/x"),
        (r#"{"filePath":"/tmp/y"}"#, "/tmp/y"),
        (r#"{"path":"/tmp/z"}"#, "/tmp/z"),
        (r#"{"pattern":"foo"}"#, "foo"),
        (r#"{"command":"ls"}"#, "ls"),
    ];
    for (args, expected) in cases {
        // Use extract_events_from_request to indirectly exercise extract_path
        let body = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": args}}
                ]}
            ]
        });
        let events = extract_events_from_request(&body);
        match &events[0] {
            AgentEvent::Read(p, _) => assert_eq!(p, expected, "args: {args}"),
            other => panic!("expected Read, got {other:?}"),
        }
    }
}

#[test]
fn extract_path_handles_codex_apply_patch() {
    use crate::torture::scenario::*;
    // Codex uses apply_patch with path nested inside operation
    // (see Codex Responses API: operation.{type, path, diff}).
    let body = serde_json::json!({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "apply_patch", "arguments": r#"{"operation":{"type":"update_file","path":"src/lib.rs","diff":"@@ -1 +1 @@\n-old\n+new"}}"#}}
            ]}
        ]
    });
    let events = extract_events_from_request(&body);
    match &events[0] {
        AgentEvent::Edit(p, _) => assert_eq!(p, "src/lib.rs"),
        other => panic!("expected Edit, got {other:?}"),
    }
}

#[test]
fn extract_cat_path_handles_shell_cat() {
    use crate::torture::scenario::*;
    // Codex reads files via `shell` with `cat`. The mock should treat
    // that as a Read event so the scenario's on_read transitions fire
    // on the path (not the whole command).
    let cases = [
        (r#"{"command":"cat src/config.sh"}"#, Some("src/config.sh")),
        (r#"{"command":"  cat   src/lib.rs  "}"#, Some("src/lib.rs")),
        (r#"{"command":"ls -la"}"#, None),
        (r#"{"command":"grep foo bar"}"#, None),
        (r#"{"command":"cat -n src/x"}"#, None),  // cat with flag, not the canonical form
        (r#"{}"#, None),
    ];
    for (args, expected) in cases {
        let got = extract_cat_path(args);
        assert_eq!(got.as_deref(), expected, "args: {args}");
    }
}

#[test]
fn shell_cat_event_classified_as_read() {
    use crate::torture::scenario::*;
    let body = serde_json::json!({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "shell", "arguments": r#"{"command":"cat src/config.sh"}"#}}
            ]}
        ]
    });
    let events = extract_events_from_request(&body);
    match &events[0] {
        AgentEvent::Read(p, _) => assert_eq!(p, "src/config.sh"),
        other => panic!("expected Read from shell+cat, got {other:?}"),
    }
}

#[test]
fn scenario_list_empty_dir() {
    use crate::torture::scenario::*;
    let _list = Scenario::list().unwrap_or_default();
}

#[test]
fn scenario_hidden_bug_loads() {
    use crate::torture::scenario::*;
    let scenario = Scenario::load("hidden_bug").expect("hidden_bug.yaml should load");
    assert_eq!(scenario.name, "hidden_bug");
    assert!(scenario.states.len() >= 7, "should have many states, got {}", scenario.states.len());
    let has_args = scenario.states.get("reading")
        .map(|s| s.args.contains_key("Read"))
        .unwrap_or(false);
    assert!(has_args, "reading state should have Read args");
    // Verify the transitions form a proper DAG (no dead states)
    let all_next: std::collections::HashSet<&str> = scenario.states.values()
        .flat_map(|s| s.transitions.iter().map(|t| t.next_state()))
        .collect();
    // Every referenced next state must exist
    for next in &all_next {
        assert!(scenario.states.contains_key(*next), "scenario '{s}' references non-existent state '{next}'",
            s = scenario.name);
    }
    // Every non-terminal state must be reachable from start
    for state_name in scenario.states.keys() {
        if scenario.states[state_name].finish || scenario.states[state_name].fail { continue; }
        if state_name == &scenario.start { continue; }
        assert!(all_next.contains(state_name.as_str()),
            "scenario '{}': state '{state_name}' is unreachable from start", scenario.name);
    }
}

#[test]
fn scenario_all_load_and_validate() {
    use crate::torture::scenario::*;
    let names = Scenario::list().unwrap_or_default();
    assert!(!names.is_empty(), "no scenarios found in scenarios/");
    for name in &names {
        let scenario = Scenario::load(name).unwrap_or_else(|e| {
            panic!("scenario '{name}' failed to load: {e}");
        });
        // Check transitions reference valid states
        let all_next: std::collections::HashSet<&str> = scenario.states.values()
            .flat_map(|s| s.transitions.iter().map(|t| t.next_state()))
            .collect();
        for next in &all_next {
            assert!(scenario.states.contains_key(*next),
                "scenario '{name}' references non-existent state '{next}'");
        }
    }
}

#[test]
fn vfs_creates_files_on_disk() {
    use crate::torture::scenario::*;
    use crate::torture::aces::create_vfs;
    // Load scenario from absolute path (relative would fail after chdir).
    let scenario_path = std::path::Path::new("scenarios/hidden_bug.yaml").canonicalize()
        .expect("scenarios/hidden_bug.yaml should be canonicalizable");
    let scenario = Scenario::from_yaml(scenario_path.to_str().unwrap())
        .expect("hidden_bug.yaml should load");
    // Use a tmp base path to avoid polluting the project tree.
    let base = std::env::temp_dir().join(format!("aces_vfs_test_{}", std::process::id()));
    let root = create_vfs(&scenario, &base).expect("vfs should create");
    assert!(root.is_dir(), "vfs root should be a directory");
    for rel in scenario.vfs.files.keys() {
        let full = root.join(rel);
        assert!(full.is_file(), "vfs file should exist: {}", full.display());
        let content = std::fs::read_to_string(&full).expect("file should be readable");
        assert!(!content.is_empty(), "vfs file should have content: {}", rel);
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn extract_agent_cwd_finds_working_directory_line() {
    use crate::torture::aces::extract_agent_cwd;
    use serde_json::json;
    // Claude Code system prompt typically includes this label.
    let body = json!({
        "messages": [
            {"role": "system", "content": "You are Claude Code.\nWorking directory: /home/user/project\n"}
        ]
    });
    let cwd = extract_agent_cwd(&body).expect("should extract cwd");
    assert_eq!(cwd, std::path::PathBuf::from("/home/user/project"));
}

#[test]
fn extract_agent_cwd_handles_block_content() {
    use crate::torture::aces::extract_agent_cwd;
    use serde_json::json;
    // Anthropic format uses content blocks for system messages.
    let body = json!({
        "messages": [
            {"role": "system", "content": [
                {"type": "text", "text": "Working directory: /Users/dev/work"}
            ]}
        ]
    });
    let cwd = extract_agent_cwd(&body).expect("should extract from blocks");
    assert_eq!(cwd, std::path::PathBuf::from("/Users/dev/work"));
}

#[test]
fn extract_agent_cwd_returns_none_when_absent() {
    use crate::torture::aces::extract_agent_cwd;
    use serde_json::json;
    let body = json!({
        "messages": [
            {"role": "system", "content": "You are an agent with no path info."}
        ]
    });
    assert!(extract_agent_cwd(&body).is_none());
}

#[test]
fn resolve_vfs_substitutes_placeholder() {
    use crate::torture::aces::resolve_vfs_path;
    let root = std::path::Path::new("/tmp/aces_vfs_test");
    assert_eq!(
        resolve_vfs_path("${VFS}/src/config.sh", root),
        "/tmp/aces_vfs_test/src/config.sh"
    );
    assert_eq!(
        resolve_vfs_path("no placeholder here", root),
        "no placeholder here"
    );
    assert_eq!(resolve_vfs_path("${VFS}", root), "/tmp/aces_vfs_test");
}

#[test]
fn resolve_args_substitutes_nested_strings() {
    use crate::torture::aces::resolve_args;
    use serde_json::json;
    let root = std::path::Path::new("/tmp/aces_vfs_x");
    let args = json!({
        "filePath": "${VFS}/src/config.sh",
        "oldString": "TIMEOUT_DEFAULT=30",
        "newString": "TIMEOUT_DEFAULT=60",
        "count": 1
    });
    let resolved = resolve_args(&args, root);
    assert!(resolved.contains("/tmp/aces_vfs_x/src/config.sh"), "filePath should be substituted: {resolved}");
    assert!(resolved.contains("\"TIMEOUT_DEFAULT=30\""), "oldString should be preserved");
    assert!(resolved.contains("\"TIMEOUT_DEFAULT=60\""), "newString should be preserved");
    assert!(resolved.contains("\"count\":1"), "non-string fields should be preserved");
}

#[test]
fn resolve_args_substitutes_deeply_nested_strings() {
    use crate::torture::aces::resolve_args;
    use serde_json::json;
    // Codex's apply_patch nests the file path two levels deep:
    //   { apply_patch: { operation: { type, path: "${VFS}/...", diff } } }
    // The recursive resolver must walk into nested objects (and arrays)
    // to substitute ${VFS} in the leaf string, not just top-level keys.
    let root = std::path::Path::new("/tmp/aces_vfs_x");
    let args = json!({
        "apply_patch": {
            "operation": {
                "type": "update_file",
                "path": "${VFS}/src/config.sh",
                "diff": "@@ -3 +3 @@\n-TIMEOUT_DEFAULT=30\n+TIMEOUT_DEFAULT=60\n"
            }
        },
        "commands": ["${VFS}/a", "${VFS}/b"],
        "n": 1
    });
    let resolved = resolve_args(&args, root);
    let v: serde_json::Value = serde_json::from_str(&resolved).unwrap();
    assert_eq!(v["apply_patch"]["operation"]["path"], "/tmp/aces_vfs_x/src/config.sh",
        "nested path should be substituted, got: {resolved}");
    assert_eq!(v["apply_patch"]["operation"]["type"], "update_file");
    assert!(v["apply_patch"]["operation"]["diff"].as_str().unwrap().contains("TIMEOUT_DEFAULT=30"));
    assert_eq!(v["commands"][0], "/tmp/aces_vfs_x/a");
    assert_eq!(v["commands"][1], "/tmp/aces_vfs_x/b");
    assert_eq!(v["n"], 1);
}

#[test]
fn current_tool_args_for_selects_agent_format() {
    use crate::torture::scenario::StateMachine;
    // Default args use Chat API (camelCase); claude_code override uses
    // snake_case. Loading the real hidden_bug.yaml would also work but
    // this is faster and self-contained.
    let yaml = r#"
name: format_test
description: test
start: s
vfs:
  files: {}
states:
  s:
    assistant: ""
    tool_calls: [Edit]
    args:
      Edit:
        filePath: "${VFS}/config.sh"
        oldString: "x=1"
        newString: "x=2"
    args_per_agent:
      claude_code:
        Edit:
          file_path: "${VFS}/config.sh"
          old_string: "x=1"
          new_string: "x=2"
    transitions: []
"#;
    let scenario: crate::torture::scenario::Scenario =
        serde_yaml::from_str(yaml).expect("parse yaml");
    let sm = StateMachine::new(scenario);
    // Default format is openai → camelCase
    let openai_args = sm.current_tool_args_for("Edit", "openai").unwrap();
    assert!(openai_args.get("filePath").is_some(), "openai args should have filePath: {openai_args}");
    assert!(openai_args.get("oldString").is_some(), "openai args should have oldString");
    assert!(openai_args.get("file_path").is_none(), "openai args should NOT have snake_case file_path");
    // claude_code format → snake_case
    let cc_args = sm.current_tool_args_for("Edit", "claude_code").unwrap();
    assert!(cc_args.get("file_path").is_some(), "claude_code args should have file_path: {cc_args}");
    assert!(cc_args.get("old_string").is_some(), "claude_code args should have old_string");
    assert!(cc_args.get("filePath").is_none(), "claude_code args should NOT have camelCase filePath");
    // Unknown format → falls back to default
    let fallback = sm.current_tool_args_for("Edit", "unknown_format").unwrap();
    assert!(fallback.get("filePath").is_some(), "unknown format should fall back to default: {fallback}");
}

#[test]
fn visit_counter_tracks_state_reentries() {
    use crate::torture::scenario::StateMachine;
    let yaml = r#"
name: loop_test
description: test
start: a
vfs: { files: {} }
states:
  a:
    assistant: ""
    tool_calls: []
    transitions:
      - type: on_tool
        tool: ""
        next: b
  b:
    assistant: ""
    tool_calls: []
    transitions:
      - type: on_tool
        tool: ""
        next: a
  give_up:
    assistant: ""
    tool_calls: []
    finish: true
    transitions: []
"#;
    let scenario: crate::torture::scenario::Scenario =
        serde_yaml::from_str(yaml).expect("parse yaml");
    let mut sm = StateMachine::new(scenario);
    // Initial state 'a' is visited once
    assert_eq!(sm.visit_count("a"), 1);
    assert!(!sm.is_stuck(5));
    // Bounce a <-> b several times
    for _ in 0..10 {
        sm.feed(crate::torture::scenario::AgentEvent::Other("x".into(), "".into()));
    }
    // 10 events, alternating a/b → each visited ~5-6 times
    let a_count = sm.visit_count("a");
    let b_count = sm.visit_count("b");
    assert!(a_count >= 5 && b_count >= 5, "a={a_count} b={b_count}");
    // The current state should be 'stuck' at threshold 5
    let cur = sm.current_state_name().to_string();
    if sm.visit_count(&cur) > 5 {
        assert!(sm.is_stuck(5), "should report stuck when current state visited > 5 times");
    }
}

#[test]
fn file_contains_transition_parses_yaml() {
    // Just verify the YAML shape parses (file check itself is a mock-side
    // concern; the scenario file is read by the mock at runtime).
    use crate::torture::scenario::Scenario;
    let yaml = r#"
name: file_test
description: test
start: s
vfs: { files: {} }
states:
  s:
    assistant: ""
    tool_calls: []
    transitions:
      - type: on_file_contains
        file_path: "${VFS}/config.sh"
        contains: "TIMEOUT_DEFAULT=60"
        next: done
  done:
    assistant: ""
    finish: true
    transitions: []
"#;
    let scenario: Scenario = serde_yaml::from_str(yaml).expect("parse yaml");
    let s = scenario.states.get("s").unwrap();
    assert_eq!(s.transitions.len(), 1);
    match &s.transitions[0] {
        crate::torture::scenario::Transition::OnFileContains { file_path, contains, next } => {
            assert_eq!(file_path, "${VFS}/config.sh");
            assert_eq!(contains, "TIMEOUT_DEFAULT=60");
            assert_eq!(next, "done");
        }
        other => panic!("expected OnFileContains, got {other:?}"),
    }
}

#[test]
fn pre_apply_parses_yaml() {
    // Mock-only escape hatch: the scenario declares a from→to replacement
    // the mock applies to the VFS before emitting the state's tool call.
    // Necessary for agents whose runtime can't execute the scenario's
    // tool (Codex's typed apply_patch/shell don't survive deeplossless's
    // Chat-Completions → Responses translation).
    use crate::torture::scenario::{FileEdit, Scenario};
    let yaml = r#"
name: pre_apply_test
description: test
start: edit
vfs: { files: {} }
states:
  edit:
    assistant: ""
    tool_calls: [apply_patch]
    pre_apply:
      - file_path: "${VFS}/config.sh"
        from: "TIMEOUT_DEFAULT=30"
        to: "TIMEOUT_DEFAULT=60"
    transitions: []
  done:
    assistant: ""
    finish: true
    transitions: []
"#;
    let scenario: Scenario = serde_yaml::from_str(yaml).expect("parse yaml");
    let s = scenario.states.get("edit").unwrap();
    assert_eq!(s.pre_apply.len(), 1);
    assert_eq!(
        s.pre_apply[0],
        FileEdit {
            file_path: "${VFS}/config.sh".to_string(),
            from: "TIMEOUT_DEFAULT=30".to_string(),
            to: "TIMEOUT_DEFAULT=60".to_string(),
        }
    );
}

#[test]
fn list_base_excludes_per_agent_variants() {
    // list_base() must return only base scenario names — those whose
    // file stem has no dot. Per-agent variants (e.g. hidden_bug.claude_code)
    // are excluded because they're selected by load_with_format() based
    // on the active --agent-format. This is what makes `--torture-aces all`
    // run each logical scenario exactly once, not once per variant.
    use crate::torture::scenario::Scenario;
    let names = Scenario::list_base().expect("list_base should not fail");
    // No name in the result should contain a dot.
    for n in &names {
        assert!(!n.contains('.'), "list_base returned per-agent variant '{n}'");
    }
    // Sanity: hidden_bug is a real base scenario in scenarios/.
    // Other per-agent variants like hidden_bug.claude_code should not appear.
    assert!(!names.iter().any(|n| n.ends_with(".claude_code")));
    assert!(!names.iter().any(|n| n.ends_with(".codex")));
}

#[test]
fn load_with_format_prefers_per_agent_variant() {
    // load_with_format("hidden_bug", Some("claude_code")) should pick
    // hidden_bug.claude_code.yaml over hidden_bug.yaml. The variants
    // differ in tool arg conventions (snake_case vs camelCase), so
    // picking the wrong one breaks claude code runs.
    use crate::torture::scenario::Scenario;
    let base = Scenario::load("hidden_bug").expect("hidden_bug.yaml should load");
    let cc = Scenario::load_with_format("hidden_bug", Some("claude_code"))
        .expect("hidden_bug.claude_code.yaml should load");
    // Different file = different args conventions. Spot-check a
    // known divergence: the edit state uses snake_case in the
    // claude_code variant.
    let edit_args = cc.states.get("understand_constraints").unwrap()
        .args.get("Edit").expect("Edit args present");
    assert!(edit_args.get("file_path").is_some(),
        "claude_code variant should use snake_case file_path");
    let edit_args_base = base.states.get("understand_constraints").unwrap()
        .args.get("Edit").expect("Edit args present in base");
    assert!(edit_args_base.get("filePath").is_some(),
        "base variant should use camelCase filePath");
}
