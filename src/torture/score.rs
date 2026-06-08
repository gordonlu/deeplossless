use crate::torture::scenario::{AgentEvent, ScenarioRun, score_run};

pub fn print_report(run: &ScenarioRun) {
    eprintln!();
    eprintln!("═══════════════════════════════════════════");
    eprintln!("  Scenario Complete: {}", run.scenario);
    eprintln!("═══════════════════════════════════════════");
    eprintln!("  Events: {}", run.events.len());
    eprintln!("  Terminal: {:?}", run.terminal_state);
    eprintln!();

    let score = run.score.as_ref().cloned().unwrap_or_else(|| score_run(run));
    eprintln!("  ── Scores (all 0-20, higher = better) ──");
    eprintln!("  Correctness         {:>5.1}", score.correctness);
    eprintln!("  Verification        {:>5.1}", score.verification);
    eprintln!("  Tool Strategy       {:>5.1}", score.tool_strategy);
    eprintln!("  Reuse               {:>5.1}", score.reuse);
    eprintln!("  Search Efficiency   {:>5.1}", score.search_efficiency);
    eprintln!("  Context Efficiency  {:>5.1}", score.context_efficiency);
    eprintln!("  ─────────────────────");
    eprintln!("  Composite (weighted) {:>5.2}/20", score.total);
    eprintln!();
    eprintln!("  ── Agent Profile ──");
    eprintln!("  {}", classify_agent(&run.events));
    eprintln!();

    // 2D profile plot
    let (prep, verify) = crate::torture::scenario::agent_profile(&run.events);
    let prep_i = prep.round() as usize;
    let verify_i = verify.round() as usize;
    eprintln!("  ── Agent Profile ──");
    eprintln!("  Prep (search+read before edit): {:>2}/10  Verify (tests+reads after edit): {:>2}/10",
        prep_i, verify_i);
    eprintln!("  Quadrant: {}", classify_agent(&run.events));
    eprintln!();
}

fn classify_agent(events: &[AgentEvent]) -> String {
    let (prep, verify) = crate::torture::scenario::agent_profile(events);

    match (prep >= 6.0, verify >= 6.0) {
        (true, true) => "Thorough (researches & verifies)".to_string(),
        (true, false) => "Explorer (researches, skips verification)".to_string(),
        (false, true) => "Minimalist (acts fast, verifies heavily)".to_string(),
        (false, false) => "YOLO (acts first, asks never)".to_string(),
    }
}
