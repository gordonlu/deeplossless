# deeplossless — Inference-aware execution memory for AI coding agents

You are connected through deeplossless, a local proxy that adds execution
memory, tool caching, and context retrieval to your DeepSeek API connection.

## Discover your conversation ID

Every response includes an `x-deeplossless-conv` header with your conversation
ID. You can also query it:

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://localhost:8080/v1/lcm/current
# → {"conversation_id": 8}
```

## Search past execution context

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/grep/{conv_id}?query=<search terms>"
```

Returns relevant past messages, summaries, and code snippets.

## Check tool cache before executing

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/cache?tool=grep&args=<json>"
# → {"hit": true, "result": "src/main.rs:42 found foo"}
```

If `hit: true`, use the cached result instead of executing the tool.

## Store tool results for future reuse

```bash
curl -sk -X POST -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tool":"grep","args":"{\"pattern\":\"foo\"}","result":"src/main.rs:42","files":"[\"src/main.rs\"]"}' \
  https://localhost:8080/v1/lcm/cache/put
```

## DAG health and status

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/status/{conv_id}"
```

## Runtime metrics

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  https://localhost:8080/v1/lcm/runtime/stats
```

## Session report

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/runtime/report?label=my+session&turns=50"
```

## Connection info

- Base URL: `https://localhost:8080/v1` (must include `/v1`)
- API key: same as passed to deeplossless
- Health check: `curl -sk https://localhost:8080/health`
- Models: `deepseek-v4-pro` (1M context), `deepseek-v4-flash` (1M context)
- TLS: self-signed cert, use `-k` flag or run `deeplossless trust`
