<img src="asset/banner.png" alt="DeepLossless" width="100%">

# deeplossless

An **inference-aware coding runtime** that reduces repeated work in long AI
coding sessions. It sits as an OpenAI-compatible proxy between your client
and the DeepSeek API.

```bash
cargo install deeplossless
deeplossless --api-key sk-...
# Point any OpenAI-compatible client at https://localhost:8080/v1
```

Long context windows are not memory. Repeated inference is waste.

---

## Quick Start

```bash
# Try without API key — runs a local demo
deeplossless demo

# Proxy mode: set once
export DEEPSEEK_API_KEY=sk-...
deeplossless

# Or let the proxy extract your key from the first request
```

OpenAI-compatible clients: point `base_url` to `https://localhost:8080/v1`.

## What It Does

```
Long coding session (3 tasks, 86 turns)

Vanilla Agent                          DeepLossless Runtime
────────────────────────────────────── ──────────────────────────────────────
21,070 tokens                          13,500 tokens
14 repeated replans                    5 replans
8 repeated failures                    3 failures
11 repo rereads                        9 rereads avoided

                                       ↓36% total tokens
                                       ↓64% replanning
                                       ↓62% repeated failures
```

**Try it yourself — no API key needed:**
```bash
git clone https://github.com/gordonlu/deeplossless.git && cd deeplossless
cargo test --test long_session_benchmark -- --nocapture
cargo test --test simulated_session -- --nocapture
```

## What Gets Reused

- **Repeated tool calls** — cached results returned inline, zero API tokens spent
- **File reads** — structured summaries instead of raw content dumps
- **Failed fixes** — remembers what didn't work and why
- **Plans** — persists execution state across turns, avoids replanning

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8080` | Listen port |
| `--api-key` | `DEEPSEEK_API_KEY` env | DeepSeek API key |
| `--admin-key` | `ADMIN_KEY` env | Admin key for LCM endpoints |
| `--upstream` | `https://api.deepseek.com` | Upstream API base URL |
| `--db-path` | `~/.deeplossless/lcm.db` | SQLite database path |
| `--rate-limit` | `100` | Max requests/second |
| `--summarizer-model` | `deepseek-v4-pro` | Model for background summarization |
| `--dry-run` | disabled | Save request bodies, skip upstream |
| `--log-dir` | disabled | Per-request JSON logging |
| `--record` | disabled | Record raw request/response for protocol debugging |
| `--tls-cert` | auto-generated | Custom TLS certificate (PEM) |
| `--tls-key` | auto-generated | Custom TLS private key (PEM) |
| `--lcm-context` | disabled | Enable DAG context injection (for LCM-aware agents) |

TLS is always on. A self-signed certificate is auto-generated at `~/.deeplossless/`.
Run `deeplossless trust` once to configure it.

Set via `RUNTIME_PROFILE=minimal|efficient|exploratory|autonomous|custom`.

## Codex + DeepSeek

```bash
# 1. Start the proxy
deeplossless

# 2. Codex config (~/.codex/config.toml)
[model_providers.localproxy]
name = "deeplossless"
base_url = "https://localhost:8080/v1"
wire_api = "responses"

# 3. Trust the certificate (once)
deeplossless trust

# 4. Run
codex
```

Protocol translation and tool cache interception work transparently.
DAG context injection is disabled by default — use `--lcm-context` to enable
it for agents that understand LCM format. Manual agent hooks require
Codex-side integration.

## Session Report

```bash
curl https://localhost:8080/v1/lcm/runtime/report?label=fix+build
```

```
# deeplossless session report: fix build
50 turns · 180s duration · 42% cache reuse
21 cache hits · 3 failure loops broken · ~8,400 tokens avoided
```

## Runtime Profiles

| Profile | Cache | Retries | Context | Budget | Best for |
|---------|-------|---------|---------|--------|----------|
| Minimal | 100% | 1 | 20% | 30% | Budget-conscious |
| Efficient | 80% | 2 | 50% | 60% | Daily coding |
| Exploratory | 50% | 3 | 80% | 80% | Debugging |
| Autonomous | 30% | 5 | 100% | 95% | Complex tasks |

## Tech Docs

- [Technical Reference](docs/tech-reference.md) — architecture, API reference, benchmarks, runtime strategies
- [Architecture Documents](docs/architecture/) — invariants, lifecycle, replay model, dependency authority
- [Runtime Event Schema](docs/architecture/runtime-events.md) — frozen event contract
- [Dependency Model](docs/architecture/dependency-model.md) — dependency taxonomy and authority boundaries

## Requirements

- Rust 1.80+
- DeepSeek API key (for proxy mode; benchmarks run without)

## License

MIT
