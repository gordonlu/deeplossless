//! Simulated coding session — exercises the full inference-economics pipeline
//! and collects benchmark metrics without requiring a running proxy.
//!
//! Run: cargo test --test simulated_session -- --nocapture
//!
//! This simulates: cache hits, failure patterns, plan reuse, tool chains,
//! repeated reads, delta injection, and execution unit recording.

use std::sync::Arc;
use std::sync::Mutex;

/// Simulate a 5-turn coding session and collect all runtime metrics.
#[tokio::test]
async fn simulated_coding_session() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        deeplossless::db::Database::builder()
            .path(dir.path().join("sim_session.db"))
            .build()
            .await
            .unwrap(),
    );

    let dag = Arc::new(
        deeplossless::dag::DagEngine::builder()
            .max_level(3)
            .recent_messages(20)
            .build(db.clone()),
    );

    let cycle = Arc::new(Mutex::new(
        deeplossless::runtime::ExecutionCycle::new(
            deeplossless::runtime::RuntimeProfile::Efficient,
        ),
    ));
    let cycle2 = cycle.clone();

    // ── Turn 1: Start task, explore repo ─────────────────────────────
    let conv_id = db
        .create_and_store("sim", &serde_json::json!([
            {"role": "user", "content": "The build is failing with 'cannot find tokio'. Fix it."}
        ]))
        .unwrap();

    // Simulate: agent runs grep to find the error
    let leaf1 = dag.insert_leaf(conv_id, "grep tokio Cargo.toml", 8).unwrap();
    db.tool_cache_put(
        "grep",
        "tokio Cargo.toml",
        "line 12: tokio = \"0.1\"  ← outdated, need 1.42",
        &["Cargo.toml".into()],
    )
    .unwrap();

    // ── Turn 2: Read file, try fix, compile fails ─────────────────────
    let leaf2 = dag.insert_leaf(conv_id, "read Cargo.toml", 5).unwrap();

    // Check tool cache — grep again with same args
    {
        let mut c = cycle.lock().unwrap();
        let _ = deeplossless::runtime::RuntimePolicy::decide(
            &c,
            Some(("grep", 1, 500)),
            None,
            None,
        );
        c.record_cache_hit("grep");
        c.record_tokens(0); // cache hit, no tokens spent
    }

    // Simulate: compile fails
    let leaf3 = dag.insert_leaf(conv_id, "cargo build → Error: tokio 0.1 not found", 10).unwrap();
    db.store_failure_pattern(
        conv_id,
        "tokio version not found",
        "upgrade Cargo.toml from 0.1 to 1.42",
        "Cargo.toml had old tokio version 0.1; upgrade resolves the dependency",
        &["Cargo.toml must have tokio >= 1.0".into()],
        &["Cargo.toml".into()],
        None,
    )
    .unwrap();

    {
        let mut c = cycle.lock().unwrap();
        c.record_tokens(350);
    }

    // ── Turn 3: Apply fix from failure memory, retry ──────────────────
    // Search failure patterns for "tokio"
    let failures = db.search_failure_patterns("tokio", 5).unwrap();
    assert!(!failures.is_empty(), "failure pattern should be searchable");

    // Policy should recommend RetryWithFix
    {
        let c = cycle.lock().unwrap();
        let d = deeplossless::runtime::RuntimePolicy::decide(
            &c,
            None,
            Some(("tokio version not found", "upgrade Cargo.toml from 0.1 to 1.42")),
            None,
        );
        println!("  Turn 3 decision: {} (confidence: {}, save: {} tok)",
            d.reason, d.confidence, d.estimated_token_saving);
        assert!(d.estimated_token_saving > 0, "failure fix should save tokens");
    }

    // Simulate: apply fix, compile succeeds
    let leaf4 = dag.insert_leaf(conv_id, "cargo build → success", 5).unwrap();

    // Store code change (delta)
    db.store_code_change(
        conv_id,
        "Cargo.toml",
        "tokio = \"1.42\"",
        &["tokio".into()],
        &["cannot find tokio".into()],
        &[],
        None,
    )
    .unwrap();

    {
        let mut c = cycle.lock().unwrap();
        c.record_tokens(200);
        c.record_success();
    }

    // ── Turn 4: Repeated grep (cache hit) ─────────────────────────────
    // Same grep again — should be a cache hit
    if let Some((_result, _hits)) = db
        .tool_cache_get("grep", "tokio Cargo.toml")
        .unwrap()
    {
        let c = cycle.lock().unwrap();
        let d = deeplossless::runtime::RuntimePolicy::decide(
            &c,
            Some(("grep", 1, 500)),
            None,
            None,
        );
        println!("  Turn 4 decision: {} (confidence: {}, save: {} tok)",
            d.reason, d.confidence, d.estimated_token_saving);
        drop(c);
        let mut c = cycle.lock().unwrap();
        c.record_cache_hit("grep");
        c.record_tokens(0); // deterministic cache, zero tokens
    }

    // ── Turn 5: Continue with plan ────────────────────────────────────
    db.store_plan_state(
        conv_id,
        "fix all build errors",
        &["upgrade deps".into(), "fix imports".into(), "verify build".into()],
        &["all deps are on crates.io".into()],
    )
    .unwrap();

    if let Some((_id, _goal, _pending, _assumptions)) = db.get_active_plan(conv_id).unwrap() {
        let c = cycle.lock().unwrap();
        let d = deeplossless::runtime::RuntimePolicy::decide(
            &c,
            None,
            None,
            Some((1, "fix all build errors", 2)),
        );
        println!("  Turn 5 decision: {} (confidence: {}, save: {} tok)",
            d.reason, d.confidence, d.estimated_token_saving);
    }

    // ── Run compactor to exercise DAG ─────────────────────────────────
    let _ = dag.assemble_context(conv_id, 2000, Some("tokio")).unwrap();
    let ctx = dag.assemble_context(conv_id, 500, None).unwrap();
    assert!(!ctx.is_empty());

    // ── Execution unit recording ──────────────────────────────────────
    let unit_id = db
        .store_execution_unit(
            conv_id,
            "The build fails with tokio not found. Need to check Cargo.toml.",
            "grep",
            r#"{"pattern":"tokio"}"#,
            "line 12: tokio = \"0.1\" — outdated",
            "Found outdated tokio version. Upgrading to 1.42.",
            "success",
            &[],
        )
        .unwrap();
    let units = db.get_execution_units(conv_id, 10).unwrap();
    assert!(!units.is_empty());
    println!("  Execution units stored: {}", units.len());

    // ── Search execution memory ───────────────────────────────────────
    let exec_results = db.search_execution_memory("tokio", 5).unwrap();
    println!("  Execution memory results: {}", exec_results.len());

    // ── Code changes search ───────────────────────────────────────────
    let code_results = db.search_code_changes("Cargo.toml", 5).unwrap();
    assert!(!code_results.is_empty());
    println!("  Code changes found: {}", code_results.len());

    // ── GC by score ───────────────────────────────────────────────────
    let deleted = dag.gc_by_score(conv_id, 10).unwrap();
    println!("  GC deleted (by score): {} nodes", deleted);

    // ── Event sourcing ────────────────────────────────────────────────
    let events = db.get_events(conv_id, 10).unwrap();
    println!("  DAG events recorded: {}", events.len());

    // ── Validation ────────────────────────────────────────────────────
    let issues = dag.validate_dag(conv_id).unwrap();
    assert!(issues.is_empty(), "DAG should be healthy, got: {issues:?}");

    // ── FINAL METRICS ─────────────────────────────────────────────────
    let final_metrics = {
        let c = cycle2.lock().unwrap();
        c.metrics.clone()
    };

    println!();
    println!("  ╔══════════════════════════════════════╗");
    println!("  ║  Simulated Session Metrics          ║");
    println!("  ╠══════════════════════════════════════╣");
    println!("  ║  Tokens spent:        {:>8}       ║", final_metrics.tokens_spent);
    println!("  ║  Cache hits:          {:>8}       ║", final_metrics.cache_hits);
    println!("  ║  Cache misses:        {:>8}       ║", final_metrics.cache_misses);
    let total = final_metrics.cache_hits + final_metrics.cache_misses;
    let rate = if total > 0 {
        format!("{:.0}%", final_metrics.cache_hits as f64 / total as f64 * 100.0)
    } else {
        "N/A".into()
    };
    println!("  ║  Cache hit rate:      {:>8}       ║", rate);
    println!("  ║  Repeated failures:   {:>8}       ║", final_metrics.repeated_failures);
    println!("  ║  Failure streak:      {:>8}       ║", final_metrics.failure_streak);
    println!("  ║  Reread ratio:        {:>8.2}     ║", final_metrics.reread_ratio);
    println!("  ║  Planning reuse:      {:>8.2}     ║", final_metrics.planning_reuse_ratio);
    println!("  ║  Budget remaining:    {:>8.0}%     ║", final_metrics.budget_remaining_pct * 100.0);
    println!("  ╚══════════════════════════════════════╝");
    println!();

    // Token savings estimate:
    // - Cache hit saved 500 tokens (grep execution)
    // - Failure avoidance saved ~300 tokens (no blind retry)
    // - Plan continuation saves ~500 tokens (no replanning)
    let estimated_saved = 500 + 300 + 500;
    let total_without = final_metrics.tokens_spent + estimated_saved;
    println!("  Estimated token savings: {} / {} ({:.0}%)",
        estimated_saved,
        total_without,
        estimated_saved as f64 / total_without as f64 * 100.0,
    );
}
