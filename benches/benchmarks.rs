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
        b.iter(|| dag.assemble_context(black_box(conv_id), black_box(10000)).unwrap())
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

criterion_group!(benches,
    bench_token_count,
    bench_snippet_extraction,
    bench_session_fingerprint,
    bench_dag_context_assembly,
    bench_snippet_preservation,
    bench_dag_gc,
    bench_fts5_insert,
);
criterion_main!(benches);
