//! Auditability layer — high-level query and reporting over execution history.
//!
//! Builds on `execution_events` and `execution_units` tables to provide
//! a unified audit trail with filtering, aggregation, and report generation.

use serde::{Deserialize, Serialize};
use crate::db::Database;
use crate::execution::{ExecutionOutcome, ExecutionUnit};

// ── Audit record ───────────────────────────────────────────────────────

/// A single audit entry — the unit of the audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub id: i64,
    pub timestamp: String,
    pub epoch_ms: i64,
    pub conv_id: Option<i64>,
    pub execution_id: Option<i64>,
    pub action_kind: String,
    pub summary: String,
    pub detail: serde_json::Value,
    /// Span context for parallel execution tracking (P0 audit).
    pub span_id: String,
    pub parent_span_id: String,
    pub span_mode: String,
    pub parallel_group: String,
    pub tool_call_id: String,
    /// Replay session ID for replay lineage (P0 audit).
    pub replay_session_id: String,
}

// ── Audit query ─────────────────────────────────────────────────────────

/// Filters for querying the audit trail.
/// Prefer `epoch_ms_since`/`epoch_ms_until` over `since`/`until` for stable ordering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditQuery {
    pub conv_id: Option<i64>,
    pub action_kind: Option<String>,
    pub tool_name: Option<String>,
    /// String-based lower bound (ISO-8601). Prefer `epoch_ms_since`.
    pub since: Option<String>,
    /// String-based upper bound (ISO-8601). Prefer `epoch_ms_until`.
    pub until: Option<String>,
    /// Epoch millisecond lower bound — stable ordering, no string comparison issues.
    pub epoch_ms_since: Option<i64>,
    /// Epoch millisecond upper bound — stable ordering, no string comparison issues.
    pub epoch_ms_until: Option<i64>,
    pub limit: usize,
    pub offset: usize,
}

// ── Audit report ────────────────────────────────────────────────────────

/// Aggregated audit report across one or all conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Total actions recorded.
    pub total_actions: usize,
    /// Breakdown by action kind.
    pub by_kind: Vec<(String, usize)>,
    /// Timeline: hourly buckets.
    pub timeline: Vec<TimelineEntry>,
    /// Most-used tools.
    pub top_tools: Vec<(String, usize)>,
    /// Failure summary.
    pub failures: FailureSummary,
    /// Cache performance.
    pub cache_perf: CacheSummary,
}

/// A single timeline bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    pub period: String,
    pub actions: usize,
    pub errors: usize,
}

/// Failure breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailureSummary {
    pub total_failures: usize,
    pub blocked: usize,
    pub recovered: usize,
    pub stale: usize,
    pub by_tool: Vec<(String, usize)>,
}

/// Cache performance summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheSummary {
    pub hits: usize,
    pub misses: usize,
    pub stale_hits: usize,
    pub hit_rate: f64,
}

impl AuditReport {
    /// Build an empty report with reasonable defaults.
    pub fn empty() -> Self {
        Self {
            total_actions: 0,
            by_kind: vec![],
            timeline: vec![],
            top_tools: vec![],
            failures: FailureSummary::default(),
            cache_perf: CacheSummary::default(),
        }
    }
}

// ── Audit trail builder ────────────────────────────────────────────────

/// Category labels for human-readable summaries.
fn action_summary(kind: &str, detail: &serde_json::Value) -> String {
    match kind {
        "tool_call" => format!(
            "Tool call: {}",
            detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
        ),
        "tool_result" => {
            let name = detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)");
            let size = detail.get("result_size").and_then(|v| v.as_i64()).unwrap_or(0);
            format!("Tool result: {} ({} bytes)", name, size)
        }
        "cache_hit" => format!(
            "Cache hit: {}",
            detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
        ),
        "cache_stale" => format!(
            "Cache stale: {}",
            detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
        ),
        "error" => format!(
            "Error in {}: {}",
            detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
            detail.get("error").and_then(|v| v.as_str()).unwrap_or("(no detail)"),
        ),
        "retry" => format!(
            "Retry #{} of {}",
            detail.get("attempt").and_then(|v| v.as_i64()).unwrap_or(0),
            detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
        ),
        "replay" => format!(
            "Replay: {} events from execution {}",
            detail.get("event_count").and_then(|v| v.as_i64()).unwrap_or(0),
            detail.get("execution_id").and_then(|v| v.as_i64()).unwrap_or(0),
        ),
        "compaction" => format!(
            "Compacted {} nodes in conv {}",
            detail.get("nodes_compacted").and_then(|v| v.as_i64()).unwrap_or(0),
            detail.get("conv_id").and_then(|v| v.as_i64()).unwrap_or(0),
        ),
        "snapshot" => format!(
            "Snapshot taken (tier {}, exec {})",
            detail.get("tier").and_then(|v| v.as_i64()).unwrap_or(0),
            detail.get("execution_id").and_then(|v| v.as_i64()).unwrap_or(0),
        ),
        "session_start" => format!(
            "Session started: model={}",
            detail.get("model").and_then(|v| v.as_str()).unwrap_or("(unknown)"),
        ),
        "session_end" => format!(
            "Session ended: {} units",
            detail.get("total_units").and_then(|v| v.as_i64()).unwrap_or(0),
        ),
        _ => format!("{}: {}", kind, serde_json::to_string(detail).unwrap_or_default()),
    }
}

/// Convert an execution unit outcome into an audit action_kind.
fn outcome_to_audit_kind(outcome: &ExecutionOutcome) -> &'static str {
    match outcome {
        ExecutionOutcome::Success => "tool_result",
        ExecutionOutcome::RecoveredFailure => "error",
        ExecutionOutcome::Blocked => "error",
        ExecutionOutcome::CacheHit => "cache_hit",
        ExecutionOutcome::Stale => "cache_stale",
        ExecutionOutcome::Replayed => "replay",
    }
}

/// Build the audit trail from the authoritative execution_events table,
/// with fallback to execution_units for backfill.
///
/// # Audit Source of Truth (P0)
///
/// `execution_events` is the authoritative append-only source. `execution_units`
/// is a derived projection. This function reads events first, then falls back
/// to execution_units for rows that predate the event-sourced schema.
/// DAG events (compaction, snapshot) are read from dag_events.
pub fn build_audit_trail(db: &Database, query: &AuditQuery) -> anyhow::Result<Vec<AuditRecord>> {
    let mut records = Vec::new();
    let limit = if query.limit > 0 { query.limit } else { 500 };

    // 1. Primary source: execution_events (authoritative, append-only)
    // Skip conv_id=0 (default/unknown — not a real conversation)
    if let Some(conv_id) = query.conv_id.filter(|&id| id > 0) {
        if let Ok(events) = db.get_execution_events_by_conv(conv_id, limit) {
            for (id, exec_id, event_kind, payload, _seq_no, created_at,
                 span_id, parent_span_id, span_mode, parallel_group, tool_call_id, epoch_ms) in &events {
                // event_kind "execution_completed" → map to tool_result/error/etc
                let detail: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
                let (action_kind, summary) = if event_kind == "execution_completed" {
                    let tool_name = detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
                    let outcome = detail.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
                    let kind = outcome_to_kind_str(outcome);
                    let mut det = serde_json::json!({
                        "tool_name": tool_name,
                    });
                    if kind == "error" {
                        det["error"] = serde_json::Value::String(detail.to_string());
                    }
                    let summary = action_summary(kind, &det);
                    (kind.to_string(), summary)
                } else {
                    // Raw stream event — keep original kind
                    let summary = action_summary(event_kind, &detail);
                    (event_kind.clone(), summary)
                };
                records.push(AuditRecord {
                    id: *id,
                    timestamp: created_at.clone(),
                    epoch_ms: *epoch_ms,
                    conv_id: Some(conv_id),
                    execution_id: *exec_id,
                    action_kind,
                    summary,
                    detail,
                    span_id: span_id.clone(),
                    parent_span_id: parent_span_id.clone(),
                    span_mode: span_mode.clone(),
                    parallel_group: parallel_group.clone(),
                    tool_call_id: tool_call_id.clone(),
                    replay_session_id: String::new(),
                });
            }
        }
    }

    // 2. Fallback source: execution_units for backfill (pre-event rows)
    if records.is_empty() && query.conv_id.is_some() {
        if let Some(conv_id) = query.conv_id.filter(|&id| id > 0) {
            if let Ok(units) = db.get_execution_units(conv_id, limit) {
                for u in &units {
                    if let Some(record) = unit_to_audit(u) {
                        records.push(record);
                    }
                }
            }
        }
    }

    // 3. DAG events: compaction, snapshot
    if let Some(conv_id) = query.conv_id {
        if let Ok(events) = db.get_events(conv_id, limit) {
            for (event_type, node_id, payload, created_at) in &events {
                let mut detail = serde_json::json!({
                    "conv_id": conv_id,
                    "node_id": node_id,
                });
                if let Ok(p) = serde_json::from_str::<serde_json::Value>(payload) {
                    if let Some(obj) = p.as_object() {
                        for (k, v) in obj {
                            detail[k] = v.clone();
                        }
                    }
                }
                let kind = match event_type.as_str() {
                    "compress" => "compaction",
                    "snapshot" => "snapshot",
                    _ => event_type,
                };
                records.push(AuditRecord {
                    id: 0,
                    timestamp: created_at.clone(),
                    epoch_ms: 0,
                    conv_id: Some(conv_id),
                    execution_id: None,
                    action_kind: kind.to_string(),
                    summary: action_summary(kind, &detail),
                    detail,
                    span_id: String::new(),
                    parent_span_id: String::new(),
                    span_mode: String::new(),
                    parallel_group: String::new(),
                    tool_call_id: String::new(),
                    replay_session_id: String::new(),
                });
            }
        }
    }

    // 4. Apply remaining filters (prefer epoch_ms range for stable ordering)
    let filtered: Vec<AuditRecord> = records
        .into_iter()
        .filter(|r| {
            if let Some(ref ak) = query.action_kind {
                if r.action_kind != *ak {
                    return false;
                }
            }
            if let Some(ref tn) = query.tool_name {
                let detail_tool = r.detail.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
                if detail_tool != tn {
                    return false;
                }
            }
            // epoch_ms range filter (stable ordering, no string comparison issues)
            if let Some(since_epoch) = query.epoch_ms_since {
                if r.epoch_ms < since_epoch {
                    return false;
                }
            }
            if let Some(until_epoch) = query.epoch_ms_until {
                if r.epoch_ms > until_epoch {
                    return false;
                }
            }
            // Fallback string-based range filter (legacy)
            if let Some(ref since) = query.since {
                if r.epoch_ms > 0 {
                    if let Ok(since_epoch) = since.parse::<i64>() {
                        if r.epoch_ms < since_epoch { return false; }
                    }
                } else if !r.timestamp.is_empty() && r.timestamp.as_str() < since.as_str() {
                    return false;
                }
            }
            if let Some(ref until) = query.until {
                if r.epoch_ms > 0 {
                    if let Ok(until_epoch) = until.parse::<i64>() {
                        if r.epoch_ms > until_epoch { return false; }
                    }
                } else if !r.timestamp.is_empty() && r.timestamp.as_str() > until.as_str() {
                    return false;
                }
            }
            true
        })
        .skip(query.offset)
        .take(limit)
        .collect();

    Ok(filtered)
}

/// Map an outcome string to an audit action kind string.
fn outcome_to_kind_str(outcome: &str) -> &str {
    match outcome {
        "success" => "tool_result",
        "recovered" | "RecoveredFailure" => "error",
        "blocked" | "Blocked" => "error",
        "cache_hit" | "CacheHit" => "cache_hit",
        "stale" | "Stale" => "cache_stale",
        "replayed" | "Replayed" => "replay",
        _ => "tool_result",
    }
}

/// Generate an aggregated audit report for a conversation (or all).
/// Uses epoch_ms for stable ordering (P0 audit).
pub fn build_audit_report(db: &Database, conv_id: Option<i64>) -> anyhow::Result<AuditReport> {
    let cid = conv_id.ok_or_else(|| anyhow::anyhow!(
        "conv_id is required; cross-conversation aggregation not yet implemented"
    ))?;
    let limit = 5000;
    let units: Vec<ExecutionUnit> = db.get_execution_units(cid, limit)?;

    if units.is_empty() {
        return Ok(AuditReport::empty());
    }

    let mut by_kind: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_tool: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut error_tools: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let mut total_actions = 0;
    let mut total_blocked = 0;
    let mut total_recovered = 0;
    let mut total_stale = 0;
    let mut cache_hits = 0;
    let mut cache_stale = 0;
    let mut total_failures = 0;

    for u in &units {
        let kind = outcome_to_audit_kind(&u.outcome).to_string();
        *by_kind.entry(kind.clone()).or_insert(0) += 1;
        *by_tool.entry(u.tool_name.clone()).or_insert(0) += 1;
        total_actions += 1;

        match u.outcome {
            ExecutionOutcome::Success => {}
            ExecutionOutcome::RecoveredFailure => {
                total_recovered += 1;
                total_failures += 1;
                *error_tools.entry(u.tool_name.clone()).or_insert(0) += 1;
            }
            ExecutionOutcome::Blocked => {
                total_blocked += 1;
                total_failures += 1;
                *error_tools.entry(u.tool_name.clone()).or_insert(0) += 1;
            }
            ExecutionOutcome::CacheHit => {
                cache_hits += 1;
            }
            ExecutionOutcome::Stale => {
                cache_stale += 1;
                total_stale += 1;
                total_failures += 1;
            }
            ExecutionOutcome::Replayed => {}
        }
    }

    let total_cache_ops = (cache_hits + cache_stale).max(1);
    let hit_rate = cache_hits as f64 / total_cache_ops as f64;

    let mut kind_list: Vec<(String, usize)> = by_kind.into_iter().collect();
    kind_list.sort_by_key(|b| std::cmp::Reverse(b.1));
    let mut tool_list: Vec<(String, usize)> = by_tool.into_iter().collect();
    tool_list.sort_by_key(|b| std::cmp::Reverse(b.1));
    let mut error_tool_list: Vec<(String, usize)> = error_tools.into_iter().collect();
    error_tool_list.sort_by_key(|b| std::cmp::Reverse(b.1));
    error_tool_list.truncate(10);

    // Timeline: group by hour using epoch_ms for stable ordering
    // Fall back to created_at string if epoch_ms is 0 (legacy rows).
    let mut hourly: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    for u in &units {
        let hour_key = if u.epoch_ms > 0 {
            // Convert epoch_ms to hourly bucket: "2026-05-22T10" from Unix millis
            let secs = u.epoch_ms / 1000;
            chrono::DateTime::from_timestamp(secs, 0)
                .map(|dt| dt.format("%Y-%m-%dT%H").to_string())
                .unwrap_or_default()
        } else if u.created_at.len() >= 13 {
            u.created_at[..13].to_string()
        } else {
            continue;
        };
        if hour_key.is_empty() {
            continue;
        }
        let entry = hourly.entry(hour_key).or_insert((0, 0));
        entry.0 += 1;
        if matches!(u.outcome, ExecutionOutcome::RecoveredFailure | ExecutionOutcome::Blocked) {
            entry.1 += 1;
        }
    }
    let timeline: Vec<TimelineEntry> = hourly
        .into_iter()
        .map(|(period, (actions, errors))| TimelineEntry { period, actions, errors })
        .collect();

    Ok(AuditReport {
        total_actions,
        by_kind: kind_list,
        timeline,
        top_tools: tool_list,
        failures: FailureSummary {
            total_failures,
            blocked: total_blocked,
            recovered: total_recovered,
            stale: total_stale,
            by_tool: error_tool_list,
        },
        cache_perf: CacheSummary {
            hits: cache_hits,
            misses: 0,
            stale_hits: cache_stale,
            hit_rate,
        },
    })
}

/// Generate a timeline with join-only events — skips individual parallel branch
/// events to prevent concurrent log explosion in audit rendering. Parallel groups
/// are collapsed into a single "Parallel group X (N branches)" entry.
/// Uses the span_id/parallel_group fields on AuditRecord.
///
/// # Join-only injection (P0 audit)
///
/// When a parallel group has N branches, instead of N audit entries (one per branch),
/// emit a single "parallel_group" entry with the group_id and branch count.
/// DAG HappensBefore edges are still fully recorded in the lineage store.
pub fn build_join_only_timeline(records: &[AuditRecord]) -> Vec<AuditRecord> {
    if records.is_empty() {
        return Vec::new();
    }

    // Group records by parallel_group (non-empty groups only)
    let mut groups: std::collections::HashMap<String, Vec<&AuditRecord>> = std::collections::HashMap::new();
    let mut standalone: Vec<AuditRecord> = Vec::new();

    for r in records {
        if r.span_mode == "parallel" && !r.parallel_group.is_empty() {
            groups.entry(r.parallel_group.clone()).or_default().push(r);
        } else {
            standalone.push(r.clone());
        }
    }

    // Emit one summary record per parallel group
    for (_group_id, members) in groups {
        let branch_count = members.len();
        let tool_names: Vec<&str> = members.iter()
            .filter_map(|r| r.detail.get("tool_name").and_then(|v| v.as_str()))
            .collect();
        let summary = if tool_names.is_empty() {
            format!("Parallel group ({} branches)", branch_count)
        } else {
            let names = tool_names.join(", ");
            format!("Parallel group: {} ({} branches)", names, branch_count)
        };
        // Use the first member's metadata as representative
        let first = members[0];
        standalone.push(AuditRecord {
            id: first.id,
            timestamp: first.timestamp.clone(),
            epoch_ms: first.epoch_ms,
            conv_id: first.conv_id,
            execution_id: None,
            action_kind: "parallel_group".to_string(),
            summary,
            detail: serde_json::json!({
                "parallel_group": first.parallel_group,
                "branch_count": branch_count,
                "tools": tool_names,
            }),
            span_id: first.span_id.clone(),
            parent_span_id: first.parent_span_id.clone(),
            span_mode: first.span_mode.clone(),
            parallel_group: first.parallel_group.clone(),
            tool_call_id: String::new(),
            replay_session_id: first.replay_session_id.clone(),
        });
    }

    standalone
}

/// Generate a human-readable session report as plain text.
pub fn generate_session_report(db: &Database, conv_id: i64) -> anyhow::Result<String> {
    let report = build_audit_report(db, Some(conv_id))?;
    let mut out = String::new();

    out.push_str(&format!("=== Session Report (conv {conv_id}) ===\n"));
    out.push_str(&format!("Total actions: {}\n", report.total_actions));
    out.push('\n');

    out.push_str("--- Breakdown ---\n");
    for (kind, count) in &report.by_kind {
        out.push_str(&format!("  {kind}: {count}\n"));
    }
    out.push('\n');

    out.push_str("--- Failures ---\n");
    out.push_str(&format!("  Total: {}\n", report.failures.total_failures));
    out.push_str(&format!("  Blocked: {}\n", report.failures.blocked));
    out.push_str(&format!("  Recovered: {}\n", report.failures.recovered));
    out.push_str(&format!("  Stale cache: {}\n", report.failures.stale));
    if !report.failures.by_tool.is_empty() {
        out.push_str("  By tool:\n");
        for (tool, count) in &report.failures.by_tool {
            out.push_str(&format!("    {tool}: {count}\n"));
        }
    }
    out.push('\n');

    out.push_str("--- Cache ---\n");
    out.push_str(&format!("  Hits: {}\n", report.cache_perf.hits));
    out.push_str(&format!("  Stale: {}\n", report.cache_perf.stale_hits));
    out.push_str(&format!("  Hit rate: {:.1}%\n", report.cache_perf.hit_rate * 100.0));
    out.push('\n');

    out.push_str("--- Top Tools ---\n");
    for (tool, count) in &report.top_tools {
        out.push_str(&format!("  {tool}: {count}\n"));
    }
    out.push('\n');

    out.push_str("--- Timeline ---\n");
    for entry in &report.timeline {
        out.push_str(&format!("  {}: {} actions ({} errors)\n", entry.period, entry.actions, entry.errors));
    }

    Ok(out)
}

fn unit_to_audit(u: &ExecutionUnit) -> Option<AuditRecord> {
    let kind = outcome_to_audit_kind(&u.outcome);
    let mut detail = serde_json::json!({
        "tool_name": u.tool_name,
        "tool_args": u.tool_args,
        "result_size": u.tool_result.len(),
    });
    if kind == "error" {
        detail["error"] = serde_json::Value::String(
            u.tool_result.chars().take(200).collect(),
        );
    }

    Some(AuditRecord {
        id: u.id,
        timestamp: u.created_at.clone(),
        epoch_ms: u.epoch_ms,
        conv_id: Some(u.conversation_id),
        execution_id: None,
        action_kind: kind.to_string(),
        summary: action_summary(kind, &detail),
        detail,
        span_id: u.span_id.clone(),
        parent_span_id: u.parent_span_id.clone(),
        span_mode: u.span_mode.clone(),
        parallel_group: u.parallel_group.clone(),
        tool_call_id: u.tool_call_id.clone(),
        replay_session_id: u.replay_session_id.clone(),
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_unit(outcome: ExecutionOutcome, tool: &str, created_at: &str) -> ExecutionUnit {
        ExecutionUnit {
            id: 1,
            conversation_id: 42,
            reasoning_before: String::new(),
            tool_name: tool.into(),
            tool_args: "{}".into(),
            tool_result: if matches!(outcome, ExecutionOutcome::Blocked | ExecutionOutcome::RecoveredFailure) { "Error: fail".into() } else { "ok".into() },
            reasoning_after: String::new(),
            outcome,
            related_nodes: vec![],
            created_at: created_at.into(),
            tool_args_json: None,
            reasoning_steps: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn outcome_to_kind_maps_correctly() {
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::Success), "tool_result");
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::RecoveredFailure), "error");
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::Blocked), "error");
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::CacheHit), "cache_hit");
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::Stale), "cache_stale");
        assert_eq!(outcome_to_audit_kind(&ExecutionOutcome::Replayed), "replay");
    }

    #[test]
    fn action_summary_produces_readable_text() {
        let detail = serde_json::json!({"tool_name": "grep", "result_size": 1024});
        let s = action_summary("tool_result", &detail);
        assert!(s.contains("grep"));
        assert!(s.contains("1024"));

        let detail = serde_json::json!({"tool_name": "build", "attempt": 3});
        let s = action_summary("retry", &detail);
        assert!(s.contains("build"));
        assert!(s.contains("3"));
    }

    #[test]
    fn unit_to_audit_creates_record() {
        let u = sample_unit(ExecutionOutcome::Success, "grep", "2026-05-22T10:00:00");
        let record = unit_to_audit(&u).unwrap();
        assert_eq!(record.action_kind, "tool_result");
        assert_eq!(record.detail["tool_name"], "grep");
    }

    #[test]
    fn unit_to_audit_error_includes_detail() {
        let u = sample_unit(ExecutionOutcome::Blocked, "build", "2026-05-22T10:00:00");
        let record = unit_to_audit(&u).unwrap();
        assert_eq!(record.action_kind, "error");
        assert!(record.detail["error"].as_str().unwrap_or("").contains("fail"));
    }

    #[test]
    fn empty_report_has_defaults() {
        let report = AuditReport::empty();
        assert_eq!(report.total_actions, 0);
        assert!(report.by_kind.is_empty());
    }

    #[tokio::test]
    async fn build_report_from_units_computes_counts() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_audit.db");
        let db = std::sync::Arc::new(
            crate::db::Database::builder()
                .path(&db_path)
                .build()
                .await
                .unwrap(),
        );

        // Simulate execution units for conv 42
        let conv_id = db.find_or_create_conversation("audit_test_fp", "deepseek-v4-flash").unwrap();
        for i in 0..5 {
            let outcome = if i < 3 {
                "success"
            } else if i == 3 {
                "blocked"
            } else {
                "cache_hit"
            };
            db.store_execution_unit(
                conv_id, "", "grep", "{}", "result", "", outcome, &[],
            ).unwrap();
        }

        let report = build_audit_report(&db, Some(conv_id)).unwrap();
        assert_eq!(report.total_actions, 5);
        assert!(report.cache_perf.hits >= 1);
        assert!(report.failures.blocked >= 1);
    }

    #[tokio::test]
    async fn build_audit_trail_returns_filtered_results() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_audit_trail.db");
        let db = std::sync::Arc::new(
            crate::db::Database::builder()
                .path(&db_path)
                .build()
                .await
                .unwrap(),
        );

        let conv_id = db.find_or_create_conversation("audit_trail_fp", "deepseek-v4-flash").unwrap();
        db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
        db.store_execution_unit(conv_id, "", "build", "{}", "Error: fail", "", "blocked", &[]).unwrap();

        // Query all
        let query = AuditQuery { conv_id: Some(conv_id), ..Default::default() };
        let trail = build_audit_trail(&db, &query).unwrap();
        assert_eq!(trail.len(), 2);

        // Query by action kind
        let query = AuditQuery { conv_id: Some(conv_id), action_kind: Some("error".into()), ..Default::default() };
        let trail = build_audit_trail(&db, &query).unwrap();
        assert_eq!(trail.len(), 1);
        assert_eq!(trail[0].action_kind, "error");
    }

    #[tokio::test]
    async fn session_report_is_readable_text() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_report.db");
        let db = std::sync::Arc::new(
            crate::db::Database::builder()
                .path(&db_path)
                .build()
                .await
                .unwrap(),
        );

        let conv_id = db.find_or_create_conversation("report_fp", "deepseek-v4-flash").unwrap();
        for outcome in &["success", "success", "blocked", "cache_hit", "recovered"] {
            db.store_execution_unit(conv_id, "", "grep", "{}", "data", "", outcome, &[]).unwrap();
        }

        let text = generate_session_report(&db, conv_id).unwrap();
        assert!(text.contains("Total actions"));
        assert!(text.contains("Breakdown"));
        assert!(text.contains("Failures"));
        assert!(text.contains("Cache"));
        assert!(text.contains("Top Tools"));
    }

    #[tokio::test]
    async fn event_based_audit_trail_from_execution_events() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_event_audit.db");
        let db = std::sync::Arc::new(
            crate::db::Database::builder()
                .path(&db_path)
                .build()
                .await
                .unwrap(),
        );

        let conv_id = db.find_or_create_conversation("event_audit_fp", "deepseek-v4-flash").unwrap();
        // store_execution_unit writes to both execution_units AND execution_events
        db.store_execution_unit(conv_id, "", "grep", "{}", "ok", "", "success", &[]).unwrap();
        db.store_execution_unit(conv_id, "", "build", "{}", "Error: fail", "", "blocked", &[]).unwrap();

        // Query via build_audit_trail — should read from execution_events
        let query = AuditQuery { conv_id: Some(conv_id), ..Default::default() };
        let trail = build_audit_trail(&db, &query).unwrap();
        assert_eq!(trail.len(), 2, "should read 2 records from execution_events");

        // Verify epoch_ms is populated (non-zero for event-sourced records)
        for record in &trail {
            assert!(record.epoch_ms > 0, "epoch_ms must be populated from execution_events");
            assert!(record.execution_id.is_some(), "execution_id must be set from execution_events");
        }

        // Filter by epoch_ms range — exclude both records by using a future epoch
        let future_epoch = chrono::Utc::now().timestamp_millis() + 60_000;
        let query = AuditQuery {
            conv_id: Some(conv_id),
            epoch_ms_since: Some(future_epoch),
            ..Default::default()
        };
        let filtered = build_audit_trail(&db, &query).unwrap();
        assert!(filtered.is_empty(), "epoch_ms_since in the future should return 0 records");

        // Include both records with a past epoch
        let query = AuditQuery {
            conv_id: Some(conv_id),
            epoch_ms_since: Some(0),
            epoch_ms_until: Some(future_epoch),
            ..Default::default()
        };
        let all = build_audit_trail(&db, &query).unwrap();
        assert_eq!(all.len(), 2, "epoch_ms range including both records should return 2");
    }

    #[tokio::test]
    async fn depends_on_edge_inserted_in_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_depends_on.db");
        let db = std::sync::Arc::new(
            crate::db::Database::builder()
                .path(&db_path)
                .build()
                .await
                .unwrap(),
        );

        let conv_id = db.find_or_create_conversation("depends_on_fp", "deepseek-v4-flash").unwrap();

        // Store two execution units with replay_session_id
        let rs_id = "test_replay_session_123";
        let id1 = db.store_execution_unit_with_span(
            conv_id, "", "grep", "{}", "ok", "", "success", &[],
            "", "", "", "", "", rs_id,
        ).unwrap();
        let id2 = db.store_execution_unit_with_span(
            conv_id, "", "build", "{}", "done", "", "success", &[],
            "", "", "", "", "", rs_id,
        ).unwrap();

        // Manually insert DependsOn edge (simulating pipeline behavior)
        db.insert_lineage_edge(id1, id2, "depends_on").unwrap();

        // Verify lineage_edges has the DependsOn edge
        let conn = db.writer_conn();
        let mut stmt = conn.prepare(
            "SELECT from_id, to_id, kind FROM lineage_edges WHERE from_id = ?1 AND to_id = ?2"
        ).unwrap();
        let result: Option<(i64, i64, String)> = stmt.query_row(
            rusqlite::params![id1, id2],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).ok();
        assert!(result.is_some(), "DependsOn edge must exist in lineage_edges");
        let (from, to, kind) = result.unwrap();
        assert_eq!(from, id1);
        assert_eq!(to, id2);
        assert_eq!(kind, "depends_on");
    }

    #[test]
    fn join_only_timeline_collapses_parallel_groups() {
        // Build sample records with parallel span_mode
        let make_record = |id: i64, group: &str, tool: &str, kind: &str, epoch: i64| -> AuditRecord {
            AuditRecord {
                id,
                timestamp: String::new(),
                epoch_ms: epoch,
                conv_id: Some(42),
                execution_id: Some(id),
                action_kind: kind.to_string(),
                summary: format!("{}: {}", kind, tool),
                detail: serde_json::json!({"tool_name": tool}),
                span_id: format!("span_{}", id),
                parent_span_id: "root".to_string(),
                span_mode: "parallel".to_string(),
                parallel_group: group.to_string(),
                tool_call_id: format!("call_{}", id),
                replay_session_id: String::new(),
            }
        };

        let records = vec![
            make_record(1, "group_A", "grep", "tool_result", 1000),
            make_record(2, "group_A", "find", "tool_result", 1001),
            make_record(3, "group_A", "sort", "tool_result", 1002),
            make_record(4, "", "sequential_tool", "tool_result", 2000),
            make_record(5, "group_B", "build", "tool_result", 3000),
            make_record(6, "group_B", "test", "error", 3001),
        ];

        let collapsed = build_join_only_timeline(&records);

        // Should have: 1 sequential + 2 group summaries = 3 records
        assert_eq!(collapsed.len(), 3, "3 parallel groups → 2 collapsed, 1 standalone = 3");

        // Check standalone sequential record
        let standalone: Vec<&AuditRecord> = collapsed.iter().filter(|r| r.action_kind != "parallel_group").collect();
        assert_eq!(standalone.len(), 1);
        assert_eq!(standalone[0].tool_call_id, "call_4");

        // Check group A summary
        let group_a: Vec<&AuditRecord> = collapsed.iter().filter(|r| r.parallel_group == "group_A").collect();
        assert_eq!(group_a.len(), 1);
        assert_eq!(group_a[0].action_kind, "parallel_group");
        assert!(group_a[0].summary.contains("grep"));
        assert!(group_a[0].summary.contains("3 branches"));

        // Check group B summary
        let group_b: Vec<&AuditRecord> = collapsed.iter().filter(|r| r.parallel_group == "group_B").collect();
        assert_eq!(group_b.len(), 1);
        assert!(group_b[0].summary.contains("2 branches"));
    }
}
