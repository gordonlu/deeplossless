use deeplossless::torture::adversarial;
use deeplossless::torture::scenario::{Scenario, ScenarioRun, StateMachine, extract_events_from_request, score_run};
use deeplossless::torture::trace::ScenarioTrace;
use serde_json::Value;

struct MockServerState {
    trace: ScenarioTrace,
    cursor: std::sync::atomic::AtomicUsize,
}

fn usage() {
    eprintln!("Torture Suite — Protocol Compatibility Test");
    eprintln!();
    eprintln!("COMMANDS:");
    eprintln!("  gen                     Generate adversarial traces to traces/");
    eprintln!("  list                    List available scenarios/traces");
    eprintln!("  serve   <file> --port   Start local mock Chat Completions server");
    eprintln!("  run     <scenario>      Run a scenario (start mock, wait for agent)");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  target/release/torture gen");
    eprintln!("  target/release/torture list");
    eprintln!("  target/release/torture run hidden_bug");
    eprintln!("  target/release/torture serve traces/simple_search_cache_identical.json --port 9000 --host 127.0.0.1");
}

fn list_traces() {
    let t = adversarial::combined_trace();
    println!("Traces:");
    println!("  combined  — {} ({} turns)", t.description, t.turns.len());
    for adv in adversarial::all_adversarial() {
        println!("  {}  — {} ({} turns)", adv.name, adv.description, adv.turns.len());
    }
    if let Ok(scenarios) = Scenario::list() {
        if !scenarios.is_empty() {
            println!();
            println!("Scenarios:");
            for s in &scenarios {
                println!("  {}", s);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); std::process::exit(1); }

    match args[1].as_str() {
        "list" => list_traces(),

        "gen" => {
            let out = if args.len() > 2 && args[2] == "--out" {
                args[3].clone()
            } else if args.len() > 2 {
                args[2].clone()
            } else {
                "traces".to_string()
            };
            std::fs::create_dir_all(&out).ok();

            let t = adversarial::combined_trace();
            let path = format!("{}/{}.json", out, t.name);
            t.save(&path).unwrap();
            println!("Generated {}  ({} turns)", path, t.turns.len());

            let template = serde_json::json!({
                "name": "my_scenario",
                "description": "Describe your test scenario here",
                "turns": [
                    {
                        "prompt": "User's first message to the agent",
                        "completion": "Expected assistant response for this turn",
                        "tokens": 50,
                        "tool_calls": ["tool_name_1", "tool_name_2"]
                    },
                    {
                        "prompt": "User's follow-up message",
                        "completion": "response with no tool calls",
                        "tokens": 20,
                        "tool_calls": []
                    }
                ]
            });
            let tmpl_path = format!("{}/template.json", out);
            std::fs::write(&tmpl_path, serde_json::to_string_pretty(&template).unwrap()).ok();
            println!("Generated {}  (edit this to create custom traces)", tmpl_path);
        }

        "adversarial" => {
            let out = if args.len() > 2 && args[2] == "--out" {
                args[3].clone()
            } else if args.len() > 2 {
                args[2].clone()
            } else {
                "traces".to_string()
            };
            std::fs::create_dir_all(&out).ok();
            let traces = adversarial::all_adversarial();
            for t in &traces {
                let path = format!("{}/{}.json", out, t.name);
                t.save(&path).unwrap();
                println!("Generated {path}  ({} turns)", t.turns.len());
            }
            println!();
            println!("Adversarial trace summary: {} total", traces.len());
        }

        "run" => {
            if args.len() < 3 {
                eprintln!("Usage: run <scenario> [--port <n>]");
                std::process::exit(1);
            }
            let scenario_name = &args[2];
            let mut port = 9000u16;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--port" { i += 1; port = args[i].parse().unwrap_or(9000); }
                i += 1;
            }

            let scenario = Scenario::load(scenario_name).unwrap_or_else(|e| {
                eprintln!("Error loading scenario '{scenario_name}': {e}");
                std::process::exit(1);
            });

            eprintln!("╔═══════════════════════════════════════════════╗");
            eprintln!("║  Torture — Protocol Compatibility Test        ║");
            eprintln!("╠═══════════════════════════════════════════════╣");
            eprintln!("║  Scenario: {}", scenario.name);
            eprintln!("║  {} ", scenario.description);
            eprintln!("║                                           ");
            eprintln!("║  Mock API: http://127.0.0.1:{}/v1/chat/completions", port);
            eprintln!("║                                           ");
            eprintln!("║  Point your agent to the mock API above.   ");
            eprintln!("║  The scenario will guide the interaction.  ");
            eprintln!("║  When complete, results will be shown.     ");
            eprintln!("╚═══════════════════════════════════════════════╝");

            let sm = std::sync::Arc::new(std::sync::Mutex::new(StateMachine::new(scenario)));
            let app = axum::Router::new()
                .route("/v1/chat/completions", axum::routing::post(move |body: axum::Json<Value>| {
                    let sm = sm.clone();
                    async move {
                        let mut machine = sm.lock().unwrap();

                        let events = extract_events_from_request(&body.0);
                        for event in &events {
                            eprintln!("[agent] {:?}", event);
                            machine.feed(event.clone());
                        }

                        let msg = machine.current_prompt().unwrap_or("Continue.").to_string();
                        let is_terminal = machine.is_terminal();

                        if is_terminal {
                            let expected_search = machine.scenario().expected_search;
                            let expected_read = machine.scenario().expected_read;
                            let pre_apply_used = machine
                                .scenario()
                                .states
                                .values()
                                .any(|state| !state.pre_apply.is_empty());
                            let run = ScenarioRun {
                                scenario: machine.current_state_name().to_string(),
                                events: machine.events.clone(),
                                terminal_state: Some(machine.current_state_name().to_string()),
                                score: None,
                                expected_search,
                                expected_read,
                                pre_apply_used,
                            };
                            let scored = ScenarioRun {
                                scenario: run.scenario.clone(),
                                events: run.events.clone(),
                                terminal_state: run.terminal_state.clone(),
                                score: Some(score_run(&run)),
                                expected_search: run.expected_search,
                                expected_read: run.expected_read,
                                pre_apply_used: run.pre_apply_used,
                            };
                            deeplossless::torture::score::print_report(&scored);
                        }

                        drop(machine);

                        axum::Json(serde_json::json!({
                            "id": "mock_aces",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "mock",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": msg},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10}
                        }))
                    }
                }));

            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await.unwrap();
                axum::serve(listener, app).await.unwrap();
            });
        }

        "serve" => {
            let mut trace_path = String::new();
            let mut port = 9000u16;
            let mut host = "127.0.0.1".to_string();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--port" => { i += 1; port = args[i].parse().unwrap_or(9000); }
                    "--host" => { i += 1; host = args.get(i).cloned().unwrap_or_else(|| "127.0.0.1".into()); }
                    _ if trace_path.is_empty() => { trace_path = args[i].clone(); }
                    _ => {}
                }
                i += 1;
            }
            if trace_path.is_empty() {
                eprintln!("Usage: serve <file> [--port <n>] [--host 127.0.0.1]");
                std::process::exit(1);
            }

            let trace = ScenarioTrace::load(&trace_path).unwrap_or_else(|e| {
                eprintln!("Error loading trace: {e}"); std::process::exit(1);
            });
            let total = trace.turns.len();
            eprintln!("Serving {} turns from {} on http://{}:{}", total, trace_path, host, port);

            let state = std::sync::Arc::new(MockServerState {
                trace,
                cursor: std::sync::atomic::AtomicUsize::new(0),
            });

            let app = axum::Router::new()
                .route("/v1/chat/completions", axum::routing::post(move |_body: axum::Json<Value>| {
                    let s = state.clone();
                    async move {
                        let idx = s.cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let turn = &s.trace.turns[idx.min(total - 1)];
                        if idx + 1 >= total {
                            eprintln!("[serve] trace exhausted ({} turns)", total);
                        }
                        axum::Json(serde_json::json!({
                            "id": "mock_chatcmpl",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "mock",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": turn.completion},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": turn.tokens, "total_tokens": turn.tokens}
                        }))
                    }
                }));

            eprintln!("Listening on http://{}:{}", host, port);
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await.unwrap();
                axum::serve(listener, app).await.unwrap();
            });
        }

        _ => { usage(); std::process::exit(1); }
    }
}
