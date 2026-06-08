/// Minimal LCG for deterministic randomness (no external dep).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A single recorded interaction for replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub prompt: String,
    pub completion: String,
    pub tokens: u64,
    #[serde(default)]
    pub tool_calls: Vec<String>,
}

/// A full scenario trace — can be loaded from JSON or constructed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioTrace {
    pub name: String,
    pub description: String,
    pub turns: Vec<Turn>,
}

impl ScenarioTrace {
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        serde_json::from_str(&data).map_err(|e| format!("parse {path}: {e}"))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let data = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, &data).map_err(|e| format!("write {path}: {e}"))
    }
}

// ── Synthetic generators ─────────────────────────────────────────

/// Generate a loop trace: same tool call repeated N times.
pub fn gen_loop(name: &str, tool: &str, n: usize) -> ScenarioTrace {
    let mut turns = Vec::with_capacity(n);
    for i in 0..n {
        turns.push(Turn {
            prompt: format!("{tool} iteration {i}"),
            completion: format!("ran {tool}\n"),
            tokens: 50,
            tool_calls: vec![tool.to_string()],
        });
    }
    ScenarioTrace {
        name: format!("loop_{name}_{n}x"),
        description: format!("{tool} repeated {n} times"),
        turns,
    }
}

/// Generate a cache stress trace: alternating cache hit/miss.
pub fn gen_cache_stress(name: &str, total: usize, hit_rate: f64) -> ScenarioTrace {
    let mut turns = Vec::with_capacity(total);
    for i in 0..total {
        let is_hit = (i as f64) < (total as f64 * hit_rate);
        turns.push(Turn {
            prompt: format!("search query {i}"),
            completion: if is_hit { format!("(cached) result {i}") } else { format!("result {i}") },
            tokens: if is_hit { 0 } else { 100 },
            tool_calls: vec![format!("search_{i}")],
        });
    }
    ScenarioTrace {
        name: format!("cache_{name}_{total}t_{}", (hit_rate * 100.0) as u32),
        description: format!("{total} turns with {:.0}% cache hit rate", hit_rate * 100.0),
        turns,
    }
}

/// Generate a chaos trace: tool calls that may fail randomly.
pub fn gen_chaos(name: &str, tools: &[&str], failure_rate: f64) -> ScenarioTrace {
    let mut rng = SimpleRng::new(42);
    let mut turns = Vec::new();
    for tool in tools.iter() {
        let fails = rng.next_f64() < failure_rate;
        turns.push(Turn {
            prompt: format!("run {tool}"),
            completion: if fails {
                format!("ERROR: {tool} failed")
            } else {
                format!("{tool} success")
            },
            tokens: if fails { 0 } else { 80 },
            tool_calls: vec![tool.to_string()],
        });
    }
    ScenarioTrace {
        name: format!("chaos_{name}_{}t", tools.len()),
        description: format!("{:.0}% failure rate across {} tools", failure_rate * 100.0, tools.len()),
        turns,
    }
}

/// Pre-built synthetic traces for smoke testing.
pub fn builtin_traces() -> Vec<ScenarioTrace> {
    vec![
        gen_loop("grep", "grep foo", 8),
        gen_cache_stress("read", 10, 0.6),
        gen_chaos("tools", &["grep", "read", "edit", "glob", "bash"], 0.3),
    ]
}
