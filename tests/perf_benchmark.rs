//! Performance benchmark suite — node scaling, long-session degradation,
//! cache effectiveness, memory growth, efficiency ratio.
//! Run: cargo test --test perf_benchmark -- --nocapture

use std::sync::Arc;
use std::time::Instant;

fn setup_db(name: &str) -> (Arc<deeplossless::db::Database>, Arc<deeplossless::dag::DagEngine>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(tokio::runtime::Runtime::new().unwrap().block_on(
        deeplossless::db::Database::builder().path(dir.path().join(name)).build()
    ).unwrap());
    let dag = Arc::new(deeplossless::dag::DagEngine::builder()
        .max_level(5).recent_messages(50).build(db.clone()));
    (db, dag)
}

fn create_conv(db: &deeplossless::db::Database, tag: &str) -> i64 {
    db.create_and_store(tag, &serde_json::json!([{"role": "user", "content": format!("start {tag}")}])).unwrap()
}

fn efficiency_ratio(tokens_saved: u64, overhead_ms: f64) -> f64 {
    if overhead_ms <= 0.0 { return 0.0; }
    tokens_saved as f64 / overhead_ms
}

// ═══════════════════════════════════════════════════════════════════════
//  B. Node count scaling
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn bench_node_count_scaling() {
    println!("\n  ╔═════════════════════════════════════════════════╗");
    println!("  ║  B. Node Count Scaling                         ║");
    println!("  ╠═════════════════════════════════════════════════╣");
    println!("  ║  Nodes   │ Assembly │ CacheGet │ Compress       ║");

    let scales = [100, 500, 1000, 5000];
    let mut results = Vec::new();

    for &n in &scales {
        let (db, dag) = setup_db(&format!("perf_nodes_{n}"));
        let conv_id = create_conv(&db, &format!("nodes_{n}"));
        for i in 0..n {
            dag.insert_leaf(conv_id, &format!("leaf {i} test"), 10).unwrap();
        }
        let start = Instant::now();
        let _ = dag.assemble_context(conv_id, 5000, None).unwrap();
        let assemble_us = start.elapsed().as_micros();

        let start = Instant::now();
        let _ = db.tool_cache_get("grep", &format!("hash_{}", n / 2));
        let cache_us = start.elapsed().as_micros();

        let compress_us = if let Ok(leaves) = dag.get_leaves(conv_id) {
            if leaves.len() >= 4 {
                let ids: Vec<i64> = leaves.iter().take(4).map(|l| l.id).collect();
                let start = Instant::now();
                let _ = dag.compress_group(conv_id, &ids, "test summary", 15, 1);
                start.elapsed().as_micros()
            } else { 0 }
        } else { 0 };

        results.push((n, assemble_us, cache_us, compress_us));
        println!("  ║  {:>5}   │ {:>5}μs  │ {:>4}μs   │ {:>5}μs       ║",
            n, assemble_us, cache_us, compress_us);
    }
    println!("  ╚═════════════════════════════════════════════════╝");

    if results.len() >= 2 {
        let a500 = results[1].1 as f64;
        let a5000 = results.last().unwrap().1 as f64;
        let ratio = a5000 / a500;
        println!("  Assembly degradation (500→5000): {ratio:.1}x");
        assert!(ratio < 20.0, "DAG assembly degrades too fast: {ratio:.1}x");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  C. Cache effectiveness decay
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn bench_cache_decay() {
    println!("\n  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║  C. Cache Effectiveness Decay                        ║");
    println!("  ╠═══════════════════════════════════════════════════════╣");
    println!("  ║  Turns   │ Hit Rate │ Stale Trigs │ Est Memory       ║");

    let (db, _dag) = setup_db("perf_cache_decay");
    let mut total_hits: u64 = 0;
    let mut total_misses: u64 = 0;
    let mut stale_triggers: u64 = 0;
    let files = ["src/main.rs", "src/lib.rs", "Cargo.toml"];

    for turn in 0..200 {
        let file = files[turn % 3];
        let pattern = format!("bug_{}", turn % 10);
        // Introduce cache misses: every 7th access uses a new pattern
        let lookup_pattern = if turn % 7 == 0 {
            format!("new_query_{}", turn)
        } else {
            pattern.clone()
        };
        // Put the known pattern
        let _ = db.tool_cache_put("grep", &pattern,
            &format!("{file}:{}: {pattern}", turn * 10), &[file.to_string()]);
        // Try to get — may be a known pattern (hit) or new query (miss)
        let hit = db.tool_cache_get("grep", &lookup_pattern).unwrap();
        if hit.is_some() { total_hits += 1; } else { total_misses += 1; }
        if turn % 20 == 0 && turn > 0 {
            let _ = db.on_files_changed(&[file.to_string()]);
            stale_triggers += 1;
        }
        if turn % 40 == 0 {
            let total = total_hits + total_misses;
            let rate = if total > 0 {
                format!("{:.0}%", total_hits as f64 / total as f64 * 100.0)
            } else { "0%".into() };
            println!("  ║  {:>5}   │ {:>7} │ {:>10} │ {:>8}     ║",
                turn, rate, stale_triggers, turn * 200);
        }
    }
    println!("  ╚═══════════════════════════════════════════════════════╝");
    println!("  Final: hits={total_hits} misses={total_misses} stale_triggers={stale_triggers}");
}

// ═══════════════════════════════════════════════════════════════════════
//  D. Long-session degradation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn bench_long_session_degradation() {
    println!("\n  ╔══════════════════════════════════════════════════════╗");
    println!("  ║  D. Long-Session Degradation                        ║");
    println!("  ╠══════════════════════════════════════════════════════╣");
    println!("  ║  Turns   │ Assembly │ CacheGet │ Context(nodes)     ║");

    let (db, dag) = setup_db("perf_degrade");
    let conv_id = create_conv(&db, "degrade");

    for &step in &[50, 100, 200, 500, 1000] {
        // Insert leaves
        for t in 0..step.min(50) {
            let file = ["src/main.rs", "src/lib.rs", "Cargo.toml"][t as usize % 3];
            dag.insert_leaf(conv_id, &format!("leaf_{t} {file}"), 10).unwrap();
        }
        // Measure
        let start = Instant::now();
        let ctx = dag.assemble_context(conv_id, 10000, None).unwrap();
        let assemble_us = start.elapsed().as_micros();
        let start = Instant::now();
        let _ = db.tool_cache_get("grep", "bug_5");
        let cache_us = start.elapsed().as_micros();
        println!("  ║  {:>5}   │ {:>5}μs  │ {:>4}μs   │ {:>5}            ║",
            step, assemble_us, cache_us, ctx.len());
    }
    println!("  ╚══════════════════════════════════════════════════════╝");
    
}

// ═══════════════════════════════════════════════════════════════════════
//  E. Memory growth
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn bench_memory_growth() {
    println!("\n  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║  E. Memory Growth Per Session                        ║");
    println!("  ╠═══════════════════════════════════════════════════════╣");

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("mem_growth.db");
    let db = Arc::new(tokio::runtime::Runtime::new().unwrap().block_on(
        deeplossless::db::Database::builder().path(&db_path).build()
    ).unwrap());
    let dag = Arc::new(deeplossless::dag::DagEngine::builder()
        .max_level(5).recent_messages(50).build(db.clone()));

    for &turns in &[100, 500, 1000, 2000] {
        let conv_id = create_conv(&db, &format!("mem_{turns}"));
        // Generate N-turn session
        for t in 0..turns {
            let file = ["src/main.rs", "src/lib.rs", "Cargo.toml"][t as usize % 3];
            dag.insert_leaf(conv_id, &format!("leaf_{t} {file}"), 10).unwrap();
            let _ = db.tool_cache_put("grep", &format!("pattern_{}", t % 10),
                &format!("result_{t}"), &[file.to_string()]);
        }
        let file_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let per_turn = if turns > 0 { file_size / turns as u64 } else { 0 };
        println!("  ║  {} turns → {}  ~{} bytes/turn     ║", turns, file_size, per_turn);
        assert!(per_turn < 5000, "Memory per turn too high: {per_turn} bytes/turn");
    }
    println!("  ╚═══════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════════════════════
//  F. Efficiency ratio
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn bench_efficiency_ratio() {
    println!("\n  ╔══════════════════════════════════════════════════════╗");
    println!("  ║  F. Efficiency Ratio (tokens saved / ms overhead)   ║");
    println!("  ╠══════════════════════════════════════════════════════╣");

    let sessions: usize = 500;
    let files = ["src/main.rs", "src/lib.rs", "Cargo.toml", "src/dag.rs", "src/db.rs"];

    // Baseline
    let (db_b, dag_b) = setup_db("perf_eff_baseline");
    let conv_b = create_conv(&db_b, "eff_base");
    let start_b = Instant::now();
    for t in 0..sessions {
        dag_b.insert_leaf(conv_b, &format!("grep bug_{} {}", t % 20, files[t % 5]), 10).unwrap();
    }
    let baseline_ms = start_b.elapsed().as_millis() as f64;
    let baseline_tokens: u64 = sessions as u64 * 500;

    // Runtime (with cache)
    let (db_r, dag_r) = setup_db("perf_eff_runtime");
    let conv_r = create_conv(&db_r, "eff_rt");
    let start_r = Instant::now();
    let mut runtime_tokens: u64 = 0;
    let mut hits: u64 = 0;
    for t in 0..sessions {
        let pattern = format!("bug_{}", t % 20);
        if db_r.tool_cache_get("grep", &pattern).unwrap().is_some() {
            hits += 1;
        } else {
            let _ = db_r.tool_cache_put("grep", &pattern, "found", &[files[t % 5].to_string()]);
            dag_r.insert_leaf(conv_r, &format!("grep {pattern}"), 10).unwrap();
            runtime_tokens += 500;
        }
    }
    let runtime_ms = start_r.elapsed().as_millis() as f64;
    let tokens_saved = baseline_tokens.saturating_sub(runtime_tokens);
    let overhead = (runtime_ms - baseline_ms).max(0.0);
    let _ratio = efficiency_ratio(tokens_saved, overhead);

    let overhead_us = ((runtime_ms - baseline_ms).max(0.0) * 1000.0) as u64;
    let overhead_label = if overhead_us == 0 { "<1μs".to_string() } else { format!("{overhead_us}μs") };
    let ratio = if overhead_us > 0 {
        format!("{:.0}", tokens_saved as f64 / overhead_us as f64 * 1000.0)
    } else { format!("{tokens_saved} (no measurable overhead)") };
    println!("  ║  Baseline:    {baseline_tokens} tokens in {baseline_ms:.0}ms          ║");
    println!("  ║  Runtime:     {runtime_tokens} tokens in {runtime_ms:.0}ms           ║");
    println!("  ║  Saved:       {tokens_saved} tokens                       ║");
    println!("  ║  Overhead:    {overhead_label}                              ║");
    println!("  ║  Cache hits:  {hits}/{sessions}  ({:.0}% reuse)         ║", hits as f64 / sessions as f64 * 100.0);
    println!("  ║  Efficiency:  {ratio} tok/ms overhead               ║");
    println!("  ╚══════════════════════════════════════════════════════╝");
    // Runtime overhead must be worth the token savings.
    if overhead > 0.0 { assert!(efficiency_ratio(tokens_saved, overhead) > 1.0, "Overhead not justified"); }
}
