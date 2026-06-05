use crate::torture::trace::{ScenarioTrace, Turn};

// ── Base templates ───────────────────────────────────────────────

/// A conversational turn template with variable substitution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnSpec {
    pub role: String,
    pub content_template: String,
    pub tool_calls: Vec<String>,
}

/// An archetypal conversation pattern before mutation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaseTemplate {
    pub name: String,
    pub description: String,
    pub turns: Vec<TurnSpec>,
}

impl BaseTemplate {
    /// Render into a ScenarioTrace, substituting {n} with a counter.
    pub fn render(&self) -> ScenarioTrace {
        let turns: Vec<Turn> = self.turns.iter().enumerate().map(|(i, spec)| {
            let prompt = spec.content_template.replace("{n}", &i.to_string());
            Turn {
                prompt,
                completion: format!("response to: {}", spec.content_template.replace("{n}", &i.to_string())),
                tokens: 50,
                tool_calls: spec.tool_calls.clone(),
            }
        }).collect();
        ScenarioTrace {
            name: self.name.clone(),
            description: self.description.clone(),
            turns,
        }
    }
}

/// Built-in conversation archetypes.
pub fn builtin_templates() -> Vec<BaseTemplate> {
    vec![
        BaseTemplate {
            name: "simple_search".into(),
            description: "Create files, then search and edit them".into(),
            turns: vec![
                TurnSpec { role: "user".into(), content_template: "create a Rust project with a lib.rs that has a foo function".into(), tool_calls: vec!["bash".into()] },
                TurnSpec { role: "assistant".into(), content_template: "project created".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "find the foo function".into(), tool_calls: vec!["grep".into()] },
                TurnSpec { role: "assistant".into(), content_template: "found in src/lib.rs: fn foo()".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "change foo to return 42".into(), tool_calls: vec!["edit".into()] },
                TurnSpec { role: "assistant".into(), content_template: "edited".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "show me the file".into(), tool_calls: vec!["read".into()] },
                TurnSpec { role: "assistant".into(), content_template: "```rust\nfn foo() -> i32 { 42 }\n```".into(), tool_calls: vec![] },
            ],
        },
        BaseTemplate {
            name: "multi_tool".into(),
            description: "Multiple tool calls in parallel".into(),
            turns: vec![
                TurnSpec { role: "user".into(), content_template: "create config files for dev, staging, and prod".into(), tool_calls: vec!["bash".into()] },
                TurnSpec { role: "assistant".into(), content_template: "configs created".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "read all config files and find the port setting".into(), tool_calls: vec!["read".into(), "glob".into(), "grep".into()] },
                TurnSpec { role: "assistant".into(), content_template: "found port 8080 in all configs".into(), tool_calls: vec![] },
            ],
        },
        BaseTemplate {
            name: "follow_up".into(),
            description: "User asks question, assistant answers, user follows up".into(),
            turns: vec![
                TurnSpec { role: "user".into(), content_template: "what is the capital of France".into(), tool_calls: vec![] },
                TurnSpec { role: "assistant".into(), content_template: "Paris".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "what about Germany".into(), tool_calls: vec![] },
                TurnSpec { role: "assistant".into(), content_template: "Berlin".into(), tool_calls: vec![] },
                TurnSpec { role: "user".into(), content_template: "population of {n}".into(), tool_calls: vec!["search".into()] },
            ],
        },
    ]
}

// ── Mutation rules ────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Mutation {
    /// Repeat a turn N times. If `vary_context`, slightly change the prompt each time.
    DuplicateTurn { index: usize, times: usize, vary_context: bool },
    /// Swap two turns.
    ReorderTurns { a: usize, b: usize },
    /// Remove a turn.
    DropTurn(usize),
    /// Modify a turn's prompt by appending noise.
    NoisePrompt { index: usize, noise_type: NoiseType },
    /// Flip tool calls between two turns (tests state contamination).
    SwapToolCalls { a: usize, b: usize },
    /// Change a turn's prompt to match another turn's (tests cache correctness).
    IdenticalPrompt { source: usize, target: usize },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NoiseType {
    TrailingWhitespace,
    PrefixText(String),
    UnicodeHomoglyph, // e.g. replace 'o' with 'ο' (Greek omicron)
    Truncate(usize),
}

// ── Noise injection ───────────────────────────────────────────────

fn inject_noise(text: &str, noise: &NoiseType) -> String {
    match noise {
        NoiseType::TrailingWhitespace => format!("{text}  \n"),
        NoiseType::PrefixText(p) => format!("{p}: {text}"),
        NoiseType::UnicodeHomoglyph => text.replace('o', "ο").replace('a', "α").replace('e', "ε"),
        NoiseType::Truncate(n) => text.chars().take(*n).collect(),
    }
}

// ── Adversarial pipeline ──────────────────────────────────────────

/// Generate an adversarial trace from a base template + mutations.
pub fn generate(base: &BaseTemplate, mutations: &[Mutation], noises: &[(usize, NoiseType)], suffix: &str) -> ScenarioTrace {
    let mut trace = base.render();
    trace.name = format!("{}_{}", trace.name, suffix);

    // Apply mutations
    for mutation in mutations {
        apply_mutation(&mut trace, mutation);
    }

    // Apply noise
    for (idx, noise) in noises {
        if *idx < trace.turns.len() {
            trace.turns[*idx].prompt = inject_noise(&trace.turns[*idx].prompt, noise);
        }
    }

    trace
}

fn apply_mutation(trace: &mut ScenarioTrace, mutation: &Mutation) {
    match mutation {
        Mutation::DuplicateTurn { index, times, vary_context } => {
            let idx = *index;
            let times = *times;
            let vary = *vary_context;
            if idx < trace.turns.len() {
                let original = trace.turns[idx].clone();
                for i in 0..times {
                    let mut dup = original.clone();
                    if vary {
                        dup.prompt = format!("{} [attempt {}]", dup.prompt, i + 1);
                    }
                    trace.turns.insert(idx + 1 + i, dup);
                }
            }
        }
        Mutation::ReorderTurns { a, b } => {
            let (a, b) = (*a, *b);
            if a < trace.turns.len() && b < trace.turns.len() {
                trace.turns.swap(a, b);
            }
        }
        Mutation::DropTurn(idx) => {
            let idx = *idx;
            if idx < trace.turns.len() {
                trace.turns.remove(idx);
            }
        }
        Mutation::NoisePrompt { index, noise_type } => {
            let idx = *index;
            if idx < trace.turns.len() {
                let prompt = trace.turns[idx].prompt.clone();
                trace.turns[idx].prompt = inject_noise(&prompt, noise_type);
            }
        }
        Mutation::SwapToolCalls { a, b } => {
            let (a, b) = (*a, *b);
            if a < trace.turns.len() && b < trace.turns.len() {
                trace.turns.swap(a, b);
            }
        }
        Mutation::IdenticalPrompt { source, target } => {
            let (src_idx, tgt_idx) = (*source, *target);
            if src_idx < trace.turns.len() && tgt_idx < trace.turns.len() {
                let src = trace.turns[src_idx].prompt.clone();
                trace.turns[tgt_idx].prompt = src;
            }
        }
    }
}

// ── High-level generators ─────────────────────────────────────────

/// Generate traces that target cache correctness:
/// same prompt reappears in different contexts.
pub fn gen_cache_adversarial() -> Vec<ScenarioTrace> {
    let base = &builtin_templates()[0]; // simple_search
    vec![
        generate(base, &[Mutation::IdenticalPrompt { source: 0, target: 2 }], &[], "cache_identical"),
        generate(base, &[Mutation::DuplicateTurn { index: 0, times: 3, vary_context: false }], &[], "cache_duplicate"),
        generate(base, &[], &[(0, NoiseType::UnicodeHomoglyph)], "cache_homoglyph"),
    ]
}

/// Generate traces that target loop detection:
/// repeated patterns with variations.
pub fn gen_loop_adversarial() -> Vec<ScenarioTrace> {
    let base = &builtin_templates()[0];
    vec![
        // ABAB pattern
        generate(base, &[Mutation::DuplicateTurn { index: 0, times: 5, vary_context: true }], &[], "loop_abab"),
        // Same prompt repeated
        generate(base, &[Mutation::DuplicateTurn { index: 0, times: 3, vary_context: false }], &[], "loop_identical"),
        // Identical prompts at different positions
        generate(base, &[
            Mutation::IdenticalPrompt { source: 0, target: 2 },
            Mutation::IdenticalPrompt { source: 0, target: 3 },
        ], &[], "loop_identical_across_turns"),
    ]
}

/// Generate traces that target state contamination:
/// tool calls/results leaking between turns.
pub fn gen_state_adversarial() -> Vec<ScenarioTrace> {
    let base = &builtin_templates()[1]; // multi_tool
    vec![
        // Swap tool calls between turns
        generate(base, &[Mutation::SwapToolCalls { a: 0, b: 1 }], &[], "state_swap_tools"),
        // Reorder turns
        generate(base, &[Mutation::ReorderTurns { a: 0, b: 1 }], &[], "state_reorder"),
        // Drop a turn
        generate(base, &[Mutation::DropTurn(0)], &[], "state_drop_turn"),
    ]
}

pub fn all_adversarial() -> Vec<ScenarioTrace> {
    let mut all = Vec::new();
    all.extend(gen_cache_adversarial());
    all.extend(gen_loop_adversarial());
    all.extend(gen_state_adversarial());
    all
}

/// Single comprehensive trace combining all adversarial categories.
pub fn combined_trace() -> ScenarioTrace {
    let traces = all_adversarial();
    ScenarioTrace {
        name: "torture".into(),
        description: "Combined adversarial: cache, loop, state contamination".into(),
        turns: traces.into_iter().flat_map(|t| t.turns).collect(),
    }
}
