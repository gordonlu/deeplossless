# deeplossless — execution memory

Connected through deeplossless — context search, tool caching, session memory.

## ⚠️ BEFORE every tool call — check cache first

**Always check the cache before executing** any deterministic tool
(grep, read_file, list_files, search, diagnostics, symbol_search).

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/cache?tool=<name>&args=<json>"
```
Response: `{"hit": true, "result": "..."}` → use `result`, skip execution.
If `hit: false` → execute normally, then store the result (see below).

## AFTER every tool call — store the result

```bash
curl -sk -X POST -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tool":"<name>","args":"<json>","result":"<output>","files":"[\"<path>\"]"}' \
  https://localhost:8080/v1/lcm/cache/put
```

This populates the cache for future turns. The proxy also auto-caches from
conversation history, but explicit `put` is more reliable.

## Discover your conversation ID

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://localhost:8080/v1/lcm/current
# → {"conversation_id": 8}
```

## Search past context

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/grep/{id}?query=<search>"
```

## Session stats

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://localhost:8080/v1/lcm/runtime/stats
```

## Notes

- All endpoints need `Authorization: Bearer $DEEPSEEK_API_KEY`
- Use `-k` flag — TLS cert is self-signed (or run `deeplossless trust` once)
- Base API URL is `https://localhost:8080/v1`
- Models: `deepseek-v4-pro` / `deepseek-v4-flash` (both 1M context)
- Cache store `files` param expects JSON string: `"[\"src/main.rs\"]"`
