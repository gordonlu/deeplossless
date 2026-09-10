//! SQLite-backed persistence for exact source evidence and typed plan dependencies.
//!
//! This is intentionally implemented on top of the existing append-only
//! `proxy_events` carrier first.  Callers only depend on [`SourceRef`] and this
//! facade, so the carrier can later move to a dedicated blob/table without
//! changing projection semantics.

use std::collections::HashSet;

use serde::Serialize;
use serde_json::{json, Value};

use crate::db::Database;
use crate::event_store::{EventFilter, EventType, ProxyEvent};
use crate::ground_truth::{PlanFactDependency, SourceRef, TruthStream};

const KIND_EXACT_SOURCE: &str = "ground_truth_exact_source";
const KIND_PLAN_DEPENDENCY: &str = "plan_fact_dependency";

/// SQLite-backed exact-evidence facade.
pub struct GroundTruthStore<'a> {
    db: &'a Database,
}

impl<'a> GroundTruthStore<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Persist UTF-8 evidence exactly as observed and return a verified SourceRef.
    ///
    /// `payload_ref` carries both the stable session bucket and row id. The
    /// caller never needs to understand this representation.
    pub fn put_text(
        &self,
        stream: TruthStream,
        session_id: &str,
        payload: &str,
        label: &str,
    ) -> anyhow::Result<SourceRef> {
        let provisional = SourceRef::new(stream.clone(), 0, payload.as_bytes());
        let event = ProxyEvent {
            id: None,
            // `proxy_events` is the persistence carrier, not the semantic
            // taxonomy. Metadata below identifies the exact-source record.
            event_type: EventType::Reasoning,
            session_id: session_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_name: None,
            path: None,
            status: Some("ground_truth".into()),
            content: payload.to_string(),
            metadata: json!({
                "deeplossless_kind": KIND_EXACT_SOURCE,
                "truth_stream": stream,
                "content_hash": provisional.content_hash,
                "payload_len": provisional.payload_len,
                "label": label,
            }),
        };
        let id = self.db.insert_proxy_event(&event)?;
        Ok(SourceRef {
            seq_no: id,
            payload_ref: Some(format!("proxy-event:{session_id}:{id}")),
            ..provisional
        })
    }

    /// Persist a JSON value using a deterministic serialization.
    pub fn put_json<T: Serialize>(
        &self,
        stream: TruthStream,
        session_id: &str,
        value: &T,
        label: &str,
    ) -> anyhow::Result<SourceRef> {
        let payload = serde_json::to_string(value)?;
        self.put_text(stream, session_id, &payload, label)
    }

    /// Materialize exact source text and verify length+SHA-256 before returning.
    pub fn materialize_text(&self, source: &SourceRef) -> anyhow::Result<String> {
        let payload_ref = source
            .payload_ref
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("source has no materializable payload_ref"))?;
        let (session_id, expected_id) = parse_payload_ref(payload_ref)?;

        let events = self.db.query_proxy_events(&EventFilter {
            session_id: Some(session_id.to_string()),
            ..Default::default()
        })?;
        let event = events
            .into_iter()
            .find(|event| event.id == Some(expected_id))
            .ok_or_else(|| anyhow::anyhow!("ground-truth source row {expected_id} not found"))?;

        if event.metadata.get("deeplossless_kind").and_then(Value::as_str)
            != Some(KIND_EXACT_SOURCE)
        {
            anyhow::bail!("row {expected_id} is not an exact-source record");
        }
        if !source.verify(event.content.as_bytes()) {
            anyhow::bail!("ground-truth source {expected_id} failed hash/length verification");
        }
        Ok(event.content)
    }

    pub fn is_recoverable(&self, source: &SourceRef) -> bool {
        self.materialize_text(source).is_ok()
    }

    pub fn recoverable_set<'b>(
        &self,
        sources: impl IntoIterator<Item = &'b SourceRef>,
    ) -> HashSet<SourceRef> {
        sources
            .into_iter()
            .filter(|source| self.is_recoverable(source))
            .cloned()
            .collect()
    }

    /// Persist a typed Plan→Fact dependency as append-only sidecar state.
    pub fn put_plan_dependency(
        &self,
        session_id: &str,
        dependency: &PlanFactDependency,
    ) -> anyhow::Result<i64> {
        let payload = serde_json::to_string(dependency)?;
        self.db.insert_proxy_event(&ProxyEvent {
            id: None,
            event_type: EventType::Reasoning,
            session_id: session_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_name: None,
            path: None,
            status: Some("projection_sidecar".into()),
            content: payload,
            metadata: json!({
                "deeplossless_kind": KIND_PLAN_DEPENDENCY,
                "plan_id": dependency.plan_id,
                "fact_id": dependency.fact_id,
                "required": dependency.required,
            }),
        })
    }

    /// Load typed dependencies for one plan. Duplicate append-only rows are
    /// collapsed by `(plan_id, step_index, fact_id, required)` identity.
    pub fn load_plan_dependencies(
        &self,
        session_id: &str,
        plan_id: i64,
    ) -> anyhow::Result<Vec<PlanFactDependency>> {
        let events = self.db.query_proxy_events(&EventFilter {
            session_id: Some(session_id.to_string()),
            ..Default::default()
        })?;
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for event in events {
            if event.metadata.get("deeplossless_kind").and_then(Value::as_str)
                != Some(KIND_PLAN_DEPENDENCY)
                || event.metadata.get("plan_id").and_then(Value::as_i64) != Some(plan_id)
            {
                continue;
            }
            let dep: PlanFactDependency = match serde_json::from_str(&event.content) {
                Ok(dep) => dep,
                Err(_) => continue,
            };
            let key = (dep.plan_id, dep.step_index, dep.fact_id.clone(), dep.required);
            if seen.insert(key) {
                out.push(dep);
            }
        }
        out.sort_by_key(|dep| (dep.step_index.unwrap_or(usize::MAX), dep.fact_id.clone()));
        Ok(out)
    }
}

fn parse_payload_ref(payload_ref: &str) -> anyhow::Result<(&str, i64)> {
    let rest = payload_ref
        .strip_prefix("proxy-event:")
        .ok_or_else(|| anyhow::anyhow!("unsupported payload_ref: {payload_ref}"))?;
    let (session_id, id) = rest
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid payload_ref: {payload_ref}"))?;
    let id = id.parse::<i64>()?;
    Ok((session_id, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn exact_payload_round_trips_and_verifies() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("truth.db"))
            .build()
            .await
            .unwrap();
        let store = GroundTruthStore::new(&db);
        let src = store
            .put_text(TruthStream::ToolPayloads, "s1", "full tool output", "tool-result")
            .unwrap();
        assert!(src.seq_no > 0);
        assert_eq!(store.materialize_text(&src).unwrap(), "full tool output");
        assert!(store.is_recoverable(&src));
    }

    #[tokio::test]
    async fn plan_dependencies_persist_as_typed_sidecar() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("deps.db"))
            .build()
            .await
            .unwrap();
        let store = GroundTruthStore::new(&db);
        let dep = PlanFactDependency {
            plan_id: 9,
            step_index: Some(2),
            fact_id: "fact-config".into(),
            required: true,
        };
        store.put_plan_dependency("s1", &dep).unwrap();
        assert_eq!(store.load_plan_dependencies("s1", 9).unwrap(), vec![dep]);
    }
}
