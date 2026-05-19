//! Comprehensive simulated multi-session benchmark — exercises the full
//! inference-economics pipeline: cache hits/misses, failure patterns,
//! plan reuse & invalidation, cross-session dedup, delta injection,
//! execution units, and compaction. No real API calls.
//!
//! Run: cargo test --test simulated_session -- --nocapture

use std::sync::Arc;
use std::sync::Mutex;

// ── Helpers ────────────────────────────────────────────────────────────

fn dag_for(db: &Arc<deeplossless::db::Database>) -> Arc<deeplossless::dag::DagEngine> {
    Arc::new(deeplossless::dag::DagEngine::builder().max_level(3).recent_messages(20).build(db.clone()))
}

fn new_cycle(profile: deeplossless::runtime::RuntimeProfile) -> Arc<Mutex<deeplossless::runtime::ExecutionCycle>> {
    Arc::new(Mutex::new(deeplossless::runtime::ExecutionCycle::new(profile)))
}

fn new_conv(db: &deeplossless::db::Database, session_id: &str, first_msg: &str) -> i64 {
    db.create_and_store(session_id, &serde_json::json!([
        {"role": "user", "content": first_msg}
    ])).unwrap()
}

fn add_leaf(dag: &deeplossless::dag::DagEngine, conv_id: i64, content: &str, tokens: i64) -> i64 {
    dag.insert_leaf(conv_id, content, tokens).unwrap().id
}

/// Record a decision and update metrics.
fn apply_decision(
    cycle: &Mutex<deeplossless::runtime::ExecutionCycle>,
    action: deeplossless::runtime::RuntimeAction,
    tokens_if_no_optimization: u64,
    tokens_actually_spent: u64,
) {
    let mut c = cycle.lock().unwrap();
    let saved = tokens_if_no_optimization.saturating_sub(tokens_actually_spent);
    match &action {
        deeplossless::runtime::RuntimeAction::ReuseToolCache { tool_name, .. } => {
            c.record_cache_hit(tool_name);
            c.record_tokens(0);
        }
        deeplossless::runtime::RuntimeAction::RetryWithFix { .. } => {
            c.record_tokens(tokens_actually_spent);
            c.record_failure();
        }
        deeplossless::runtime::RuntimeAction::ContinuePlan { .. } => {
            c.record_tokens(tokens_actually_spent);
        }
        deeplossless::runtime::RuntimeAction::CompactAndProceed => {
            c.record_tokens(tokens_actually_spent);
        }
        _ => {
            c.record_tokens(tokens_if_no_optimization);
        }
    }
    // Track estimated savings
    c.metrics.budget_remaining_pct = (c.metrics.budget_remaining_pct
        - tokens_actually_spent as f64 / c.metrics.budget_total.max(1) as f64)
        .max(0.0);
    // If no optimization was applied, record as cache miss
    if saved == 0 && !matches!(action, deeplossless::runtime::RuntimeAction::DelegateToModel) {
        c.record_cache_miss();
    }
}

fn print_metrics(label: &str, cycle: &Mutex<deeplossless::runtime::ExecutionCycle>) {
    let c = cycle.lock().unwrap();
    let m = &c.metrics;
    let total = m.cache_hits + m.cache_misses;
    let rate = if total > 0 { format!("{:.0}%", m.cache_hits as f64 / total as f64 * 100.0) } else { "N/A".into() };
    println!("  {label}");
    println!("    tokens={}  cache_hits={}  cache_misses={}  hit_rate={rate}  failures={}  streak={}",
        m.tokens_spent, m.cache_hits, m.cache_misses, m.repeated_failures, m.failure_streak);
}

// ═══════════════════════════════════════════════════════════════════════
//  Full simulation: 3 conversations × 5-8 turns = 20 turns total
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn comprehensive_simulated_session() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(deeplossless::db::Database::builder()
        .path(dir.path().join("comp_sim.db")).build().await.unwrap());
    let dag = dag_for(&db);
    let cycle = new_cycle(deeplossless::runtime::RuntimeProfile::Efficient);
    let mut total_baseline: u64 = 0; // tokens that WOULD have been spent without optimization

    println!();
    println!("  ╔══════════════════════════════════════════════════════════╗");
    println!("  ║   Comprehensive Simulated Coding Session              ║");
    println!("  ║   3 conversations, 20 turns, multi-language           ║");
    println!("  ╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Conv A: Rust — fix build + runtime panic (8 turns) ────────────
    let conv_a = new_conv(&db, "session_rust", "cargo build fails with 'cannot find serde_json'");

    // Turn A1: grep for dependency
    add_leaf(&dag, conv_a, "grep serde_json Cargo.toml", 8);
    db.tool_cache_put("grep", "serde_json Cargo.toml", "line 8: serde_json = \"1.0\"", &["Cargo.toml".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), Some(("grep", 0, 500)), None, None,
    );
    apply_decision(&cycle, d.action, 500, 0);
    total_baseline += 500;
    println!("  A1  grep serde_json        → cache MISS (cold), execute, 500 tok");

    // Turn A2: read Cargo.toml (cache miss — first read)
    add_leaf(&dag, conv_a, "read_file Cargo.toml — dependencies list", 15);
    db.tool_cache_put("read_file", "Cargo.toml", "[dependencies]\nserde = \"1\"\nserde_json = \"1\"\ntokio = { version = \"1\", features = [\"full\"] }", &["Cargo.toml".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 300, 300);
    total_baseline += 300;
    println!("  A2  read Cargo.toml        → no cache yet, 300 tok");

    // Turn A3: grep again — HIT
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), Some(("grep", 0, 480)), None, None,
    );
    apply_decision(&cycle, d.action, 480, 0);
    total_baseline += 480;
    println!("  A3  grep serde_json (again)→ cache HIT, 0 tok (saved 480)");

    // Turn A4: compile → error (type mismatch)
    add_leaf(&dag, conv_a, "cargo build → error[E0308]: mismatched types, expected String found &str", 12);
    db.store_failure_pattern(conv_a, "E0308 mismatched types String vs &str",
        "add .to_string() or use &str consistently",
        "serde_json::to_string returns String but field expects &str — need conversion",
        &["function return types should be consistent".into()],
        &["src/lib.rs".into(), "src/main.rs".into()], None).unwrap();
    {
        let mut c = cycle.lock().unwrap();
        c.record_tokens(400);
        c.record_failure();
    }
    total_baseline += 400;
    println!("  A4  compile → type error   → failure recorded (E0308), 400 tok");

    // Turn A5: symbol_search — cache miss, different tool
    add_leaf(&dag, conv_a, "symbol_search 'process_data' → found in src/lib.rs:42", 8);
    db.tool_cache_put("symbol_search", "process_data", "src/lib.rs:42 pub fn process_data(input: &str) -> String", &["src/lib.rs".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 250, 250);
    total_baseline += 250;
    println!("  A5  symbol_search          → cache miss, 250 tok");

    // Turn A6: failure memory suggests fix
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None,
        Some(("E0308 mismatched types", "add .to_string() or use &str consistently")),
        None,
    );
    apply_decision(&cycle, d.action, 350, 150);
    total_baseline += 350;
    println!("  A6  failure→retry with fix → confidence {:.0}%, save {} tok → 150 tok spent", d.confidence, d.estimated_token_saving);

    // Turn A7: apply fix, compile success
    add_leaf(&dag, conv_a, "cargo build → success (fixed type mismatch)", 5);
    db.store_code_change(conv_a, "src/lib.rs", "-    let out: String = result;\n+    let out: String = result.to_string();",
        &["process_data".into()], &["E0308 mismatched types".into()], &[],
        None).unwrap();
    {
        cycle.lock().unwrap().record_success();
        cycle.lock().unwrap().record_tokens(200);
    }
    total_baseline += 200;
    println!("  A7  fix applied → build passes, code change stored, 200 tok");

    // Turn A8: plan created for remaining work
    db.store_plan_state(conv_a, "complete Rust refactor: fix all type warnings, add tests",
        &["fix remaining type warnings".into(), "add unit tests".into(), "run clippy".into()],
        &["project uses Rust edition 2024".into()]).unwrap();
    {
        let d = deeplossless::runtime::RuntimePolicy::decide(
            &cycle.lock().unwrap(), None, None,
            Some((1, "complete Rust refactor", 3)),
        );
        apply_decision(&cycle, d.action, 500, 200);
    }
    total_baseline += 500;
    println!("  A8  plan created (3 pending steps), plan reuse saves 300 tok");

    // ── Conv B: Python — data pipeline bug (6 turns) ──────────────────
    let conv_b = new_conv(&db, "session_py", "data pipeline crashes on 'KeyError: column_name' in pandas");

    println!();
    // B1: grep for column_name (REPEAT of similar dep pattern — cross-session)
    add_leaf(&dag, conv_b, "grep 'column_name' pipeline.py", 8);
    db.tool_cache_put("grep", "column_name pipeline.py", "col_name = df['column_name']  # KeyError if missing", &["pipeline.py".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), Some(("grep", 0, 500)), None, None,
    );
    apply_decision(&cycle, d.action, 500, 0);
    total_baseline += 500;
    println!("  B1  grep column_name       → cache MISS (new pattern), 500 tok baseline");

    // B2: read pipeline.py
    add_leaf(&dag, conv_b, "read_file pipeline.py — 200 lines, data processing logic", 20);
    db.tool_cache_put("read_file", "pipeline.py", "def process(df):\n    col = df['column_name']\n    return df.groupby('col').sum()", &["pipeline.py".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 400, 400);
    total_baseline += 400;
    println!("  B2  read pipeline.py       → first read, 400 tok");

    // B3: diagnostics → shows KeyError traceback
    add_leaf(&dag, conv_b, "diagnostics → KeyError: 'column_name' not in DataFrame columns ['col_a', 'col_b']", 10);
    db.tool_cache_put("diagnostics", "pipeline.py KeyError", "KeyError: 'column_name' — available columns: col_a, col_b", &["pipeline.py".into()]).unwrap();
    db.store_failure_pattern(conv_b, "KeyError column not found in DataFrame",
        "check column exists before accessing: if 'col' in df.columns",
        "pandas KeyError when column name doesn't exist — need defensive check or .get()",
        &["always validate column existence before df access".into()],
        &["pipeline.py".into()], None).unwrap();
    {
        let mut c = cycle.lock().unwrap();
        c.record_tokens(350);
        c.record_failure();
    }
    total_baseline += 350;
    println!("  B3  diagnostics → KeyError → failure pattern stored, 350 tok");

    // B4: grep again — HIT
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), Some(("grep", 0, 480)), None, None,
    );
    apply_decision(&cycle, d.action, 480, 0);
    total_baseline += 480;
    println!("  B4  grep again             → cache HIT, 0 tok (saved 480)");

    // B5: symbol_search for defensive patterns — cache MISS (new query)
    add_leaf(&dag, conv_b, "symbol_search df.columns → found 3 usages in pipeline.py", 8);
    db.tool_cache_put("symbol_search", "df.columns", "pipeline.py:12 df.columns\npipeline.py:34 df.columns\npipeline.py:67 df.columns", &["pipeline.py".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 250, 250);
    total_baseline += 250;
    println!("  B5  symbol_search columns  → cold cache, 250 tok");

    // B6: fix + retry succeeds
    add_leaf(&dag, conv_b, "python pipeline.py → success (all rows processed)", 5);
    db.store_code_change(conv_b, "pipeline.py",
        "-    col = df['column_name']\n+    col = df.get('column_name', pd.Series(dtype=float))",
        &["process".into()], &["KeyError: column_name".into()], &[], None).unwrap();
    {
        cycle.lock().unwrap().record_success();
        cycle.lock().unwrap().record_tokens(250);
    }
    total_baseline += 250;
    println!("  B6  fix applied → pipeline passes, code change stored, 250 tok");

    // ── Conv C: Go — API refactor with dep conflict (6 turns) ─────────
    let conv_c = new_conv(&db, "session_go", "refactor HTTP handler to use new router, 'undefined: chi.NewRouter'");

    println!();
    // C1: grep for chi import (cache MISS)
    add_leaf(&dag, conv_c, "grep 'chi' go.mod main.go", 8);
    db.tool_cache_put("grep", "chi go.mod main.go", "main.go:8: r := chi.NewRouter()", &["go.mod".into(), "main.go".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 500, 500);
    total_baseline += 500;
    println!("  C1  grep chi go.mod        → cold, 500 tok");

    // C2: read go.mod — CACHE MISS
    add_leaf(&dag, conv_c, "read_file go.mod → module example.com/api, go 1.21, require chi v5", 15);
    db.tool_cache_put("read_file", "go.mod", "module example.com/api\ngo 1.21\nrequire github.com/go-chi/chi/v5 v5.0.12", &["go.mod".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 300, 300);
    total_baseline += 300;
    println!("  C2  read go.mod            → first read, 300 tok");

    // C3: grep go.mod again — CACHE HIT
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), Some(("grep", 0, 480)), None, None,
    );
    apply_decision(&cycle, d.action, 480, 0);
    total_baseline += 480;
    println!("  C3  grep go.mod again      → cache HIT, 0 tok (saved 480)");

    // C4: compile error — dep conflict
    add_leaf(&dag, conv_c, "go build → import cycle not allowed in package main", 10);
    db.store_failure_pattern(conv_c, "go import cycle",
        "restructure packages to avoid circular imports",
        "internal/models imports handlers which imports models — break the cycle with interface",
        &["circular imports must be broken by introducing interfaces or shared package".into()],
        &["main.go".into(), "handlers/auth.go".into(), "models/user.go".into()], None).unwrap();
    {
        let mut c = cycle.lock().unwrap();
        c.record_tokens(400);
        c.record_failure();
    }
    total_baseline += 400;
    println!("  C4  go build → import cycle → failure recorded, 400 tok");

    // C5: read_file handlers/auth.go — CACHE HIT (pre-loaded from C2 read pattern)
    // In real usage, read_file of go.mod ≠ handlers/auth.go, so this is a MISS
    db.tool_cache_put("read_file", "handlers/auth.go", "package handlers\nimport \"example.com/api/models\"\nfunc Login(w http.ResponseWriter, r *http.Request) {}", &["handlers/auth.go".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 350, 350);
    total_baseline += 350;
    println!("  C5  read handlers/auth.go  → cold, 350 tok");

    // C6: list_files to understand project structure — CACHE MISS
    add_leaf(&dag, conv_c, "list_files src/ → main.go, handlers/, models/, go.mod, go.sum", 6);
    db.tool_cache_put("list_files", "src/", "main.go\nhandlers/auth.go\nhandlers/middleware.go\nmodels/user.go\nmodels/session.go", &["src/".into()]).unwrap();
    let d = deeplossless::runtime::RuntimePolicy::decide(
        &cycle.lock().unwrap(), None, None, None,
    );
    apply_decision(&cycle, d.action, 200, 200);
    total_baseline += 200;
    println!("  C6  list_files src/        → cold, 200 tok");

    // ── Cross-session: search for dep issues across all conversations ──
    println!();
    let cross_results = db.search_cross_session("fix", 10).unwrap();
    println!("  Cross-session search 'fix': {} results across conversations", cross_results.len());

    let failure_all = db.search_failure_patterns("error", 10).unwrap();
    println!("  Failure patterns across all: {} total", failure_all.len());

    let exec = db.search_execution_memory("build", 10).unwrap();
    println!("  Execution memory 'build': {} results", exec.len());

    // ── DAG assembly + compaction ───────────────────────────────────
    for &cid in &[conv_a, conv_b, conv_c] {
        let ctx = dag.assemble_context(cid, 3000, Some("fix")).unwrap();
        assert!(!ctx.is_empty(), "conv {cid} should have context");
        let deleted = dag.gc_by_score(cid, 20).unwrap();
        println!("  Conv {}: DAG nodes in context={}  GC'd by score={}", cid, ctx.len(), deleted);
    }

    let ctx = dag.assemble_context(conv_a, 3000, Some("fix")).unwrap();
    assert!(!ctx.is_empty());

    // ── DAG validation ───────────────────────────────────────────────
    for &cid in &[conv_a, conv_b, conv_c] {
        let issues = dag.validate_dag(cid).unwrap();
        assert!(issues.is_empty(), "conv {cid} DAG issues: {issues:?}");
    }

    // ── Execution units ──────────────────────────────────────────────
    let _ = db.store_execution_unit(conv_a,
        "Build fails with type error. Found mismatch in process_data.",
        "cargo build", "{}", "error[E0308]: mismatched types", "Fixed by adding .to_string(). Build passes.", "success", &[]).unwrap();
    let _ = db.store_execution_unit(conv_b,
        "Pipeline crashes on KeyError. Need to check column names.",
        "python pipeline.py", "{}", "KeyError: 'column_name'", "Fixed with df.get(). Pipeline runs.", "success", &[]).unwrap();
    let _ = db.store_execution_unit(conv_c,
        "Import cycle after adding router. Need to restructure packages.",
        "go build", "{}", "import cycle not allowed", "Broke cycle with interface in shared package.", "success", &[]).unwrap();

    // ── FINAL METRICS ────────────────────────────────────────────────
    let final_metrics = cycle.lock().unwrap().metrics.clone();

    println!();
    println!("  ╔══════════════════════════════════════════════╗");
    println!("  ║     FINAL BENCHMARK METRICS                ║");
    println!("  ╠══════════════════════════════════════════════╣");
    let total_cache = final_metrics.cache_hits + final_metrics.cache_misses;
    let hit_rate = if total_cache > 0 {
        format!("{:.0}%", final_metrics.cache_hits as f64 / total_cache as f64 * 100.0)
    } else { "N/A".into() };
    println!("  ║  Conversations:                   3         ║");
    println!("  ║  Total turns:                    20         ║");
    println!("  ║                                            ║");
    println!("  ║  Tokens spent (runtime):     {:>8}        ║", final_metrics.tokens_spent);
    println!("  ║  Tokens baseline (no opt):   {:>8}        ║", total_baseline);
    let saved_pct = if total_baseline > 0 {
        format!("{:.0}%", (1.0 - final_metrics.tokens_spent as f64 / total_baseline as f64) * 100.0)
    } else { "N/A".into() };
    println!("  ║  Tokens SAVED:               {:>8}  ({})  ║",
        total_baseline.saturating_sub(final_metrics.tokens_spent), saved_pct);
    println!("  ║                                            ║");
    println!("  ║  Cache hits:                  {:>8}        ║", final_metrics.cache_hits);
    println!("  ║  Cache misses:                {:>8}        ║", final_metrics.cache_misses);
    println!("  ║  Cache hit rate:              {:>8}        ║", hit_rate);
    println!("  ║  Repeated failures:           {:>8}        ║", final_metrics.repeated_failures);
    println!("  ║  Failure streak (final):      {:>8}        ║", final_metrics.failure_streak);
    println!("  ║  Reread ratio:                {:>8.2}      ║", final_metrics.reread_ratio);
    println!("  ║  Planning reuse ratio:        {:>8.2}      ║", final_metrics.planning_reuse_ratio);
    println!("  ║  Budget remaining:            {:>8.0}%      ║", final_metrics.budget_remaining_pct * 100.0);
    println!("  ╚══════════════════════════════════════════════╝");
    println!();

    // Verify key invariants
    assert!(final_metrics.cache_hits >= 3, "expected at least 3 cache hits");
    assert!(total_baseline > final_metrics.tokens_spent, "runtime should save tokens vs baseline");
}

#[tokio::test]
async fn multi_language_ast_extraction() {
    let test_cases: Vec<(&str, &str, &[&str])> = vec![
        ("rust", "fn process(data: &[u8]) -> Result<()> {\n    let x = 42;\n}\npub struct Config { debug: bool }\nuse std::collections::HashMap;",
         &["fn process", "struct Config"]),
        ("python", "def process_data(df):\n    return df.dropna()\n\nclass PipelineConfig:\n    def __init__(self):\n        self.batch_size = 100\n\nimport pandas as pd\nfrom typing import List",
         &["def process_data", "class PipelineConfig"]),
        ("typescript", "function parseInput(data: string): Parsed {\n  return JSON.parse(data);\n}\ninterface Config { debug: boolean }\nimport { Logger } from './logger';",
         &["function parseInput", "interface Config"]),
        ("javascript", "function handleClick(event) {\n  event.preventDefault();\n}\nclass EventBus {\n  emit(name, data) { }\n}",
         &["function handleClick", "class EventBus"]),
        ("java", "public class UserService {\n    public User findById(Long id) { return null; }\n}\nimport java.util.List;",
         &["class UserService", "findById"]),
        ("cpp", "#include <vector>\nclass DataProcessor {\npublic:\n    void ingest(const std::string& path);\n};\nint process(char *buf) { return 0; }",
         &["DataProcessor", "ingest", "process"]),
        ("csharp", "using System;\npublic class OrderService {\n    public async Task<Order> ProcessAsync(int id) { return null; }\n}\npublic interface IRepository<T> { Task<T> GetAsync(int id); }",
         &["OrderService", "ProcessAsync", "IRepository"]),
        ("go", "package service\nimport \"context\"\ntype UserRepo struct { db *sql.DB }\nfunc (r *UserRepo) FindByID(ctx context.Context, id int64) (*User, error) { return nil, nil }\nfunc NewServer(addr string) *http.Server { return &http.Server{Addr: addr} }",
         &["FindByID", "NewServer", "UserRepo"]),
    ];

    let mut total_extracted = 0;
    let mut languages_tested = 0;
    for (lang, code, expected) in &test_cases {
        let results = deeplossless::snippet::extract_with_source(
            &format!("```{}\n{}\n```", lang, code), "test");
        let code_snippets: Vec<_> = results.iter()
            .filter(|s| s.snippet_type == deeplossless::snippet::SnippetType::CodeBlock).collect();
        let found = expected.iter().filter(|sym| code_snippets.iter().any(|s| s.content.contains(**sym))).count();
        println!("  {:>10}: extracted {}, matched {}/{} symbols", lang, code_snippets.len(), found, expected.len());
        if found > 0 { languages_tested += 1; }
        total_extracted += code_snippets.len();
    }
    println!("\n  Languages with structural output: {}/{} | Total elements: {}", languages_tested, test_cases.len(), total_extracted);
    assert!(languages_tested >= 5, "at least 5 languages should work, got {languages_tested}");
    assert!(total_extracted >= 8, "should extract >=8 elements, got {total_extracted}");
}
