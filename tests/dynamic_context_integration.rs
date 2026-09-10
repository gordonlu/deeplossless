use std::sync::Arc;

use deeplossless::dag::DagEngine;
use deeplossless::db::Database;
use deeplossless::dynamic_context::{assemble_dynamic_context, DynamicContextHints};

#[tokio::test]
async fn old_plan_relevant_leaf_reenters_working_context_under_budget() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        Database::builder()
            .path(dir.path().join("dynamic-recall.db"))
            .build()
            .await
            .unwrap(),
    );
    // Pin the recent window so the fixture proves recall behavior rather than
    // depending on the production default (currently 20 messages).
    let dag = DagEngine::builder().recent_messages(4).build(db.clone());
    let conv_id = db
        .find_or_create_conversation("dynamic-recall", "deepseek-flash")
        .unwrap();

    let old = dag
        .insert_leaf(
            conv_id,
            "critical ConfigLoader evidence lives in src/config.rs",
            10,
        )
        .unwrap();

    // Push the critical evidence outside the explicitly configured recent window.
    for i in 0..8 {
        dag.insert_leaf(conv_id, &format!("recent unrelated event {i}"), 10)
            .unwrap();
    }

    let budget = 80;
    let ordinary = dag.assemble_context(conv_id, budget, None).unwrap();
    assert!(
        ordinary.iter().all(|node| node.id != old.id),
        "fixture must put the old node outside ordinary recent context"
    );

    let hints = DynamicContextHints {
        plan_terms: vec!["ConfigLoader".into(), "src/config.rs".into()],
        ..Default::default()
    };
    let recalled = assemble_dynamic_context(&dag, conv_id, budget, None, &hints).unwrap();

    assert!(
        recalled.iter().any(|node| node.id == old.id),
        "execution-state signal should recall old plan-relevant evidence"
    );
    let used: i64 = recalled.iter().map(|node| node.token_count.max(0)).sum();
    assert!(used <= budget as i64, "dynamic recall must respect token budget");
}
