use criterion::{Criterion, black_box, criterion_group, criterion_main};

// ── Helpers ────────────────────────────────────────────────────────────

fn make_db() -> deeplossless::db::Database {
    let dir = std::env::temp_dir().join(format!("deeplossless_bench_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    tokio::runtime::Runtime::new().unwrap().block_on(
        deeplossless::db::Database::builder()
            .path(dir.join("bench.db"))
            .build()
    ).unwrap()
}

fn make_conversation_text(size: usize) -> String {
    let lines = vec![
        "User: Can you fix the build error in src/main.rs?",
        "Assistant: Let me check the error message. The issue is on line 42.",
        "User: The error is E0425: cannot find value `MAX_RETRY` in this scope.",
        "Assistant: I'll add the missing constant. Port 8080 should be configurable.",
        "User: According to Gartner 2026, 67% of projects face this issue.",
        "Assistant: Let me check the Cargo.toml for the dependency version.",
        "User: We need to upgrade reqwest from 0.11 to 0.12.",
        "Assistant: Done. The build passes now at 500ms compile time.",
    ];
    let mut text = String::new();
    for _ in 0..size {
        for line in &lines {
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

// ── Benchmarks ────────────────────────────────────────────────────────

fn bench_token_count(c: &mut Criterion) {
    let text = make_conversation_text(1000); // ~8000 lines
    c.bench_function("tokenizer/count_8k_lines", |b| {
        b.iter(|| deeplossless::tokenizer::count(black_box(&text)))
    });
}

fn bench_snippet_extraction(c: &mut Criterion) {
    let text = make_conversation_text(500); // ~4000 lines
    c.bench_function("snippet/extract_4k_lines", |b| {
        b.iter(|| deeplossless::snippet::extract(black_box(&text)))
    });
}

fn bench_session_fingerprint(c: &mut Criterion) {
    let msgs: Vec<serde_json::Value> = (0..10)
        .map(|i| serde_json::json!({"role": "user", "content": format!("message {i}")}))
        .collect();
    c.bench_function("session/fingerprint_10_msgs", |b| {
        b.iter(|| deeplossless::session::fingerprint(black_box(&msgs), 3))
    });
}

fn bench_dag_context_assembly(c: &mut Criterion) {
    // Setup: create DB with DAG nodes
    let db = make_db();
    let conv_id = db.create_and_store("bench", &serde_json::json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test"}
    ])).unwrap();
    let dag = deeplossless::dag::DagEngine::builder()
        .recent_messages(20)
        .build(std::sync::Arc::new(db));

    // Insert 1000 leaf nodes and compress
    for i in 0..1000 {
        dag.insert_leaf(conv_id, &format!("message {i}"), 10).unwrap();
    }

    c.bench_function("dag/assemble_1k_leaves", |b| {
        b.iter(|| dag.assemble_context(black_box(conv_id), black_box(10000), None).unwrap())
    });
}

fn bench_snippet_preservation(c: &mut Criterion) {
    // Measure: what % of embedded key values survive extraction?
    let text = r#"
    The server listens on port 8080. According to Gartner 2026, 67% of
    projects fail. Error: E0425 in /src/main.rs at line 42.
    DeepSeek-V4 handles 128K context. Latency is 500ms.
    "#;

    c.bench_function("snippet/key_value_extraction", |b| {
        b.iter(|| deeplossless::snippet::extract(black_box(text)))
    });
}

fn bench_dag_gc(c: &mut Criterion) {
    let db = make_db();
    let conv_id = db.create_and_store("bench", &serde_json::json!([
        {"role": "user", "content": "test"}
    ])).unwrap();
    let dag = deeplossless::dag::DagEngine::builder()
        .build(std::sync::Arc::new(db));

    for i in 0..500 {
        dag.insert_leaf(conv_id, &format!("message {i}"), 10).unwrap();
    }

    c.bench_function("dag/gc_500_leaves", |b| {
        b.iter(|| dag.collect_garbage(black_box(conv_id), black_box(true)).unwrap())
    });
}

fn bench_fts5_insert(c: &mut Criterion) {
    let db = make_db();
    let msgs: Vec<serde_json::Value> = (0..100)
        .map(|i| serde_json::json!({"role": "user", "content": format!("Message with some content and numbers like {i} and paths like src/main.rs")}))
        .collect();
    let conv_id = db.create_and_store("bench", &serde_json::json!([
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"}
    ])).unwrap();

    c.bench_function("db/insert_100_messages_with_fts", |b| {
        b.iter(|| {
            db.store_messages(black_box(conv_id), &serde_json::json!(msgs)).unwrap();
        })
    });
}

// ── Runtime benchmarks (simulated coding session) ────────────────────

fn bench_tool_cache_hit_rate(c: &mut Criterion) {
    let cycle = deeplossless::runtime::ExecutionCycle::new(
        deeplossless::runtime::RuntimeProfile::Efficient
    );

    c.bench_function("runtime/cache_hit_decision", |b| {
        b.iter(|| {
            let d = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                Some(("grep", 42, 500)),
                None,
                None,
            );
            black_box(d)
        })
    });
}

fn bench_failure_avoidance(c: &mut Criterion) {
    let cycle = deeplossless::runtime::ExecutionCycle::new(
        deeplossless::runtime::RuntimeProfile::Efficient
    );

    c.bench_function("runtime/failure_retry_decision", |b| {
        b.iter(|| {
            let d = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                None,
                Some(("sqlite deadlock", "use WAL mode + reader pool")),
                None,
            );
            black_box(d)
        })
    });
}

fn bench_plan_continuation(c: &mut Criterion) {
    let cycle = deeplossless::runtime::ExecutionCycle::new(
        deeplossless::runtime::RuntimeProfile::Efficient
    );

    c.bench_function("runtime/plan_continue_decision", |b| {
        b.iter(|| {
            let d = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                None,
                None,
                Some((1, "fix build error", 3)),
            );
            black_box(d)
        })
    });
}

fn bench_full_decision_cycle(c: &mut Criterion) {
    let mut cycle = deeplossless::runtime::ExecutionCycle::new(
        deeplossless::runtime::RuntimeProfile::Efficient
    );
    // Simulate state: some failures, some cache
    cycle.metrics.cache_hits = 5;
    cycle.metrics.cache_misses = 2;
    cycle.metrics.failure_streak = 1;
    cycle.metrics.budget_remaining_pct = 0.75;

    c.bench_function("runtime/full_decision_cycle", |b| {
        b.iter(|| {
            // Simulate a full cycle: check cache, check failures, check plan, fall through
            let d1 = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                Some(("grep", 1, 300)),
                None,
                Some((1, "task", 2)),
            );
            let d2 = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                None,
                Some(("deadlock", "use reader pool")),
                None,
            );
            let d3 = deeplossless::runtime::RuntimePolicy::decide(
                &cycle,
                None,
                None,
                None,
            );
            black_box((d1, d2, d3))
        })
    });
}

fn bench_reasoning_distillation(c: &mut Criterion) {
    let seq: Vec<(String, String, String)> = (0..20).map(|i| {
        if i % 3 == 0 {
            ("grep".into(), "pattern".into(), format!("Error: no match {i}"))
        } else if i == 19 {
            ("compile".into(), "".into(), "Success: build passes".into())
        } else {
            ("read_file".into(), format!("src/mod{i}.rs"), "fn main() {}".into())
        }
    }).collect();

    c.bench_function("runtime/distill_20_tool_calls", |b| {
        b.iter(|| deeplossless::runtime::ExecutionCompactor::distill(black_box(&seq)))
    });
}

criterion_group!(benches,
    bench_token_count,
    bench_snippet_extraction,
    bench_session_fingerprint,
    bench_dag_context_assembly,
    bench_snippet_preservation,
    bench_dag_gc,
    bench_fts5_insert,
    bench_tool_cache_hit_rate,
    bench_failure_avoidance,
    bench_plan_continuation,
    bench_full_decision_cycle,
    bench_reasoning_distillation,
);
criterion_main!(benches);
