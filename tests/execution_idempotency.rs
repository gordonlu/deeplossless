use deeplossless::db::Database;

#[tokio::test]
async fn execution_replay_returns_existing_durable_id() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder()
        .path(dir.path().join("execution-replay.db"))
        .build()
        .await
        .unwrap();
    let conv_id = db
        .find_or_create_conversation("execution-replay", "test")
        .unwrap();

    let first = db
        .store_execution_unit_with_span_once(
            conv_id,
            "before",
            "read_file",
            "{}",
            "ok",
            "after",
            "success",
            &[],
            "",
            "",
            "",
            "",
            "call-1",
            "replay-1",
        )
        .unwrap();
    let second = db
        .store_execution_unit_with_span_once(
            conv_id,
            "before",
            "read_file",
            "{}",
            "ok",
            "after",
            "success",
            &[],
            "",
            "",
            "",
            "",
            "call-1",
            "replay-2",
        )
        .unwrap();

    assert!(first.1);
    assert!(!second.1);
    assert_eq!(first.0, second.0);
    assert_ne!(second.0, 0);

    let units = db.get_execution_units(conv_id, 10).unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].id, first.0);
}

#[tokio::test]
async fn lineage_insert_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder()
        .path(dir.path().join("lineage-idempotent.db"))
        .build()
        .await
        .unwrap();

    let first = db.insert_lineage_edge(10, 20, "depends_on").unwrap();
    let second = db.insert_lineage_edge(10, 20, "depends_on").unwrap();

    assert_eq!(first, second);
    let edges = db.get_lineage_to(20).unwrap();
    assert_eq!(edges, vec![(10, 20, "depends_on".to_string())]);
}
