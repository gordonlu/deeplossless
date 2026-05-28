//! Anthropic Messages API ↔ DeepSeek Chat Completions adapter.
//! Minimal implementation for basic text streaming and tool use.

use serde_json::{json, Value};

/// Convert Anthropic Messages request → DeepSeek Chat Completions format.
/// Pass `last_reasoning_content` (from the previous response) to satisfy DeepSeek's
/// thinking-mode requirement that `reasoning_content` be echoed back in the next turn.
pub fn request_to_deepseek(body: &Value, last_reasoning_content: Option<&str>) -> Value {
    let model = body["model"].as_str().unwrap_or("deepseek-v4-pro");
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(4096);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut messages = Vec::new();

    // Anthropic system prompt → Chat Completions system message
    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            messages.push(json!({"role": "system", "content": text}));
        } else if let Some(arr) = system.as_array() {
            let text = arr.iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
    }

    // Anthropic messages → Chat Completions messages
    if let Some(msgs) = body["messages"].as_array() {
        for msg in msgs {
            let role = msg["role"].as_str().unwrap_or("user");
            let content = &msg["content"];

            if let Some(text) = content.as_str() {
                let mut m = json!({"role": role, "content": text});
                if role == "assistant" {
                    if let Some(rc) = last_reasoning_content {
                        m["reasoning_content"] = json!(rc);
                    }
                }
                messages.push(m);
            } else if let Some(blocks) = content.as_array() {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                let mut tool_results = Vec::new();

                for block in blocks {
                    match block["type"].as_str().unwrap_or("") {
                        "text" => {
                            if let Some(t) = block["text"].as_str() {
                                text_parts.push(t.to_string());
                            }
                        }
                        "thinking" => {
                            if let Some(t) = block["thinking"].as_str() {
                                text_parts.push(format!("[thinking] {t}"));
                            }
                        }
                        "tool_use" => {
                            tool_calls.push(json!({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": serde_json::to_string(&block["input"]).unwrap_or_default(),
                                }
                            }));
                        }
                        "tool_result" => {
                            let tc_content = block["content"].as_str()
                                .map(|s| json!(s))
                                .unwrap_or_else(|| {
                                    // Anthropic content blocks → extract text
                                    if let Some(arr) = block["content"].as_array() {
                                        let text = arr.iter()
                                            .filter_map(|b| b["text"].as_str())
                                            .collect::<Vec<_>>()
                                            .join("\n");
                                        json!(if text.is_empty() { block["content"].to_string() } else { text })
                                    } else {
                                        block["content"].clone()
                                    }
                                });
                            tool_results.push(json!({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": tc_content,
                            }));
                        }
                        _ => {}
                    }
                }

                // Push tool results BEFORE text content so they follow
                // tool_calls consecutively (DeepSeek requirement).
                for tr in tool_results {
                    messages.push(tr);
                }
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    // Combine text and tool_calls into a single message.
                    // DeepSeek requires reasoning_content on ALL assistant messages
                    // in multi-turn conversations when thinking mode is active.
                    let mut assistant_msg = json!({"role": role});
                    if !text_parts.is_empty() {
                        assistant_msg["content"] = json!(text_parts.join("\n"));
                    }
                    if !tool_calls.is_empty() {
                        assistant_msg["tool_calls"] = json!(tool_calls);
                    }
                    if role == "assistant" {
                        if let Some(rc) = last_reasoning_content {
                            assistant_msg["reasoning_content"] = json!(rc);
                        }
                    }
                    messages.push(assistant_msg);
                }
            }
        }
    }

    // Repair: DeepSeek requires tool messages to immediately follow tool_calls.
    // In Anthropic format, user messages may combine tool_result blocks
    // with text, creating gaps between tool_calls and their tool responses.
    // Fix: move all tool messages after each tool_calls group to directly
    // follow the tool_calls message, preserving text order otherwise.
    {
        let mut repaired: Vec<Value> = Vec::with_capacity(messages.len());
        let mut i = 0;
        while i < messages.len() {
            let tc_count = messages[i].get("tool_calls")
                .and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            if tc_count == 0 {
                repaired.push(messages[i].clone());
                i += 1;
                continue;
            }
            repaired.push(messages[i].clone());
            let mut tools: Vec<Value> = Vec::new();
            let mut others: Vec<Value> = Vec::new();
            let mut j = i + 1;
            while j < messages.len() && messages[j].get("tool_calls").is_none() {
                if messages[j]["role"] == "tool" {
                    tools.push(messages[j].clone());
                } else {
                    others.push(messages[j].clone());
                }
                j += 1;
            }
            repaired.append(&mut tools);
            repaired.append(&mut others);
            i = j;
        }
        messages = repaired;
    }

    // Strip tool_calls from the final assistant if no tool results follow
    // (the model's pending tool use that hasn't been executed yet).
    for i in (0..messages.len()).rev() {
        if messages[i].get("tool_calls").is_none() { continue; }
        let next_is_tool = messages.get(i+1).map(|m| m["role"] == "tool").unwrap_or(false);
        if !next_is_tool {
            messages[i].as_object_mut().map(|o| o.remove("tool_calls"));
            if messages[i].get("content").is_none() {
                messages.remove(i);
            }
        }
        break;
    }

    // Translate Anthropic tools → OpenAI function format
    let tools: Value = body.get("tools").map(|arr| {
        Value::Array(arr.as_array().unwrap_or(&vec![]).iter().map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description").unwrap_or(&Value::Null),
                    "parameters": tool.get("input_schema").unwrap_or(&json!({"type": "object", "properties": {}}))
                }
            })
        }).collect())
    }).unwrap_or(Value::Null);

    let mut req = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });

    if !tools.is_null() {
        req["tools"] = tools;
    }

    // Translate Anthropic tool_choice → OpenAI format
    if let Some(tc) = body.get("tool_choice") {
        req["tool_choice"] = match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") | Some("any") => json!("auto"),
            Some("tool") => json!({
                "type": "function",
                "function": {"name": tc.get("name").unwrap_or(&Value::Null)}
            }),
            _ => json!("auto"),
        };
    }

    // Temperature
    if let Some(temp) = body.get("temperature") {
        req["temperature"] = temp.clone();
    }
    // Top-p
    if let Some(top_p) = body.get("top_p") {
        req["top_p"] = top_p.clone();
    }
    // Stop sequences
    if let Some(stop) = body.get("stop_sequences") {
        req["stop"] = stop.clone();
    }
    // Thinking/reasoning: DeepSeek V4 thinks by default. Only disable if the
    // client explicitly opts out. Reasoning content is echoed back via
    // last_reasoning_content for multi-turn continuity.
    if body.get("thinking").is_some() {
        req["reasoning_effort"] = json!("medium");
    }

    // Debug: audit tool_calls → tool message pairing
    for (i, m) in messages.iter().enumerate() {
        if let Some(tc) = m.get("tool_calls").and_then(|v| v.as_array()) {
            let tc_ids: Vec<&str> = tc.iter()
                .filter_map(|c| c["id"].as_str())
                .collect();
            let tool_ids: Vec<&str> = messages[i+1..].iter()
                .take_while(|n| n["role"] == "tool")
                .filter_map(|n| n["tool_call_id"].as_str())
                .collect();
            tracing::debug!(target: "deeplossless::anthropic",
                msg_index = i, tc_count = tc_ids.len(),
                ?tc_ids, ?tool_ids,
                "tool_calls audit");
        }
    }

    req
}

/// Convert DeepSeek Chat Completions non-streaming response → Anthropic format.
pub fn response_to_anthropic(deepseek: &Value) -> Value {
    let choice = match deepseek["choices"].as_array().and_then(|a| a.first()) {
        Some(c) => c,
        None => {
            return json!({
                "id": format!("msg_{}", chrono::Utc::now().timestamp_millis()),
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            });
        }
    };
    let message = &choice["message"];
    let content = message["content"].as_str().unwrap_or("");
    let model = deepseek["model"].as_str().unwrap_or("deepseek-v4-pro");
    let usage = &deepseek["usage"];

    let mut blocks: Vec<Value> = Vec::new();

    // Text content (if any)
    if !content.is_empty() {
        blocks.push(json!({"type": "text", "text": content}));
    }

    // Tool calls (may coexist with text in DeepSeek responses)
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            let func = &tc["function"];
            let input: Value = serde_json::from_str(func["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
            blocks.push(json!({
                "type": "tool_use",
                "id": tc["id"],
                "name": func["name"],
                "input": input,
            }));
        }
    }

    json!({
        "id": format!("msg_{}", chrono::Utc::now().timestamp_millis()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": choice["finish_reason"].as_str().unwrap_or("end_turn"),
        "usage": {
            "input_tokens": usage["prompt_tokens"].as_u64().unwrap_or(0),
            "output_tokens": usage["completion_tokens"].as_u64().unwrap_or(0),
        }
    })
}

/// Stateful SSE converter that tracks content block lifecycle so
/// `content_block_start` for text is emitted before the first text delta.
#[derive(Default)]
pub struct AnthropicSseState {
    text_block_index: Option<usize>,
    thinking_block_index: Option<usize>,
    tool_use_block_indices: Vec<usize>,
    /// Token usage captured from the finish chunk, for accumulation.
    pub last_input_tokens: u64,
    pub last_output_tokens: u64,
    /// Accumulated reasoning_content from DeepSeek deltas (thinking mode).
    pub reasoning_content: String,
    /// Next available content block index, monotonically increasing.
    next_block_index: usize,
}

impl AnthropicSseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a DeepSeek SSE data line into Anthropic SSE event strings.
    /// Maintains state to correctly emit `content_block_start` before the
    /// first delta of each content type (text, thinking, or tool_use).
    pub fn convert(&mut self, data: &str) -> Vec<String> {
        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let delta = &v["choices"][0]["delta"];
        let mut events = Vec::new();

        // Tool call chunks — may carry id+name (start), arguments (delta), or both
        if let Some(tc) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for call in tc.iter() {
                let name = call["function"]["name"].as_str().unwrap_or("");
                let id = call["id"].as_str().unwrap_or("");
                let args = call["function"]["arguments"].as_str().unwrap_or("");
                // Use DeepSeek's global tool index (from 'index' field), not
                // the local position in this chunk's tool_calls array.
                let idx = call.get("index").and_then(|v| v.as_i64()).unwrap_or(self.next_block_index as i64) as usize;

                // New tool call starting → content_block_start
                if !id.is_empty() {
                    if idx >= self.next_block_index {
                        self.next_block_index = idx + 1;
                    }
                    if !self.tool_use_block_indices.contains(&idx) {
                        self.tool_use_block_indices.push(idx);
                    }
                    events.push(format!("event: content_block_start\ndata: {}\n\n", json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                    })));
                }

                // Arguments (may appear in same chunk or subsequent chunks)
                if !args.is_empty() {
                    events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    })));
                }
            }
            return events;
        }

        // Reasoning content — accumulate for next turn, emit as thinking delta.
        // A content_block_start for thinking is emitted before the first delta only.
        if let Some(rc) = delta["reasoning_content"].as_str() {
            if !rc.is_empty() {
                self.reasoning_content.push_str(rc);
                let is_first = self.thinking_block_index.is_none();
                let thinking_idx = self.thinking_block_index.unwrap_or_else(|| {
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    self.thinking_block_index = Some(idx);
                    idx
                });
                if is_first {
                    events.push(format!("event: content_block_start\ndata: {}\n\n", json!({
                        "type": "content_block_start",
                        "index": thinking_idx,
                        "content_block": {"type": "thinking", "thinking": ""}
                    })));
                }
                events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                    "type": "content_block_delta",
                    "index": thinking_idx,
                    "delta": {"type": "thinking_delta", "thinking": rc}
                })));
            }
            return events;
        }

        // Text delta
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                let is_first = self.text_block_index.is_none();
                let text_idx = self.text_block_index.unwrap_or_else(|| {
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    self.text_block_index = Some(idx);
                    idx
                });
                if is_first {
                    events.push(format!("event: content_block_start\ndata: {}\n\n", json!({
                        "type": "content_block_start",
                        "index": text_idx,
                        "content_block": {"type": "text", "text": ""}
                    })));
                }
                events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                    "type": "content_block_delta",
                    "index": text_idx,
                    "delta": {"type": "text_delta", "text": text}
                })));
            }
            return events;
        }

        // Finish — emit content_block_stop for each started block, then message_delta + message_stop
        if let Some(reason) = v["choices"][0]["finish_reason"].as_str() {
            let stop_reason = match reason {
                "stop" => "end_turn",
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                _ => reason,
            };
            // Close each content block
            for idx in &self.started_block_indices() {
                events.push(format!("event: content_block_stop\ndata: {}\n\n", json!({
                    "type": "content_block_stop",
                    "index": idx
                })));
            }
            let usage = &v["usage"];
            let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
            let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
            self.last_input_tokens = prompt_tokens;
            self.last_output_tokens = completion_tokens;
            events.push(format!("event: message_delta\ndata: {}\n\n", json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {
                    "output_tokens": completion_tokens
                }
            })));
            events.push(format!("event: message_stop\ndata: {}\n\n", json!({
                "type": "message_stop"
            })));
            return events;
        }

        events
    }

    /// Whether any content block was started during conversion.
    /// If false after all deltas, the response body was empty.
    pub fn any_block_started(&self) -> bool {
        self.text_block_index.is_some()
            || self.thinking_block_index.is_some()
            || !self.tool_use_block_indices.is_empty()
    }

    /// All content block indices that were started (for emitting content_block_stop).
    pub fn started_block_indices(&self) -> Vec<usize> {
        let mut indices = self.tool_use_block_indices.clone();
        if let Some(idx) = self.thinking_block_index {
            indices.push(idx);
        }
        if let Some(idx) = self.text_block_index {
            indices.push(idx);
        }
        indices.sort();
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_translation_simple_text() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": false
        });
        let deepseek = request_to_deepseek(&body, None);
        assert_eq!(deepseek["model"], "claude-sonnet-4-5");
        assert_eq!(deepseek["max_tokens"], 1024);
        assert_eq!(deepseek["stream"], false);
        let msgs = deepseek["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2); // system + user
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hello");
    }

    #[test]
    fn request_translation_system_as_array() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 500,
            "system": [{"type": "text", "text": "Rule 1"}, {"type": "text", "text": "Rule 2"}],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let deepseek = request_to_deepseek(&body, None);
        let msgs = deepseek["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Rule 1\nRule 2");
    }

    #[test]
    fn request_translation_with_tool_use() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Read file"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tc1", "name": "read_file", "input": {"path": "/tmp/test"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tc1", "content": "file contents here"}
                ]}
            ]
        });
        let deepseek = request_to_deepseek(&body, None);
        let msgs = deepseek["messages"].as_array().unwrap();
        // Should have: user, assistant(tool_calls), tool
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].get("tool_calls").is_some());
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "tc1");
    }

    #[test]
    fn request_translation_preserves_temperature() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let deepseek = request_to_deepseek(&body, None);
        assert_eq!(deepseek["temperature"], 0.7);
    }

    #[test]
    fn response_translation_basic() {
        let deepseek_resp = json!({
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let anthropic = response_to_anthropic(&deepseek_resp);
        assert_eq!(anthropic["type"], "message");
        assert_eq!(anthropic["role"], "assistant");
        assert_eq!(anthropic["stop_reason"], "stop");
        assert_eq!(anthropic["content"][0]["text"], "Hello!");
        assert_eq!(anthropic["usage"]["input_tokens"], 10);
        assert_eq!(anthropic["usage"]["output_tokens"], 5);
    }

    #[test]
    fn sse_text_delta_conversion() {
        let data = r#"{"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let results = AnthropicSseState::new().convert(data);
        assert_eq!(results.len(), 2); // content_block_start + text_delta
        assert!(results[0].contains("content_block_start"));
        assert!(results[1].contains("content_block_delta"));
        assert!(results[1].contains("text_delta"));
        assert!(results[1].contains("Hi"));
    }

    #[test]
    fn sse_text_delta_only_emits_block_start_once() {
        let mut state = AnthropicSseState::new();
        let r1 = state.convert(r#"{"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}"#);
        let r2 = state.convert(r#"{"choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}"#);
        assert!(r1[0].contains("content_block_start"));
        assert!(!r2.iter().any(|e| e.contains("content_block_start")));
        assert!(state.any_block_started());
    }

    #[test]
    fn sse_finish_conversion() {
        let data = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5,"prompt_tokens":10,"total_tokens":15}}"#;
        let results = AnthropicSseState::new().convert(data);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("message_delta"));
        assert!(results[0].contains("stop_reason"));
        assert!(results[1].contains("message_stop"));
    }

    #[test]
    fn sse_empty_delta_returns_empty() {
        let data = r#"{"choices":[{"index":0,"delta":{},"finish_reason":null}]}"#;
        assert!(AnthropicSseState::new().convert(data).is_empty());
    }

    #[test]
    fn sse_tool_call_with_id_and_args() {
        let data = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/test\"}"}}]},"finish_reason":null}]}"#;
        let results = AnthropicSseState::new().convert(data);
        assert_eq!(results.len(), 2); // content_block_start + input_json_delta
        assert!(results[0].contains("tool_use"));
        assert!(results[0].contains("content_block_start"));
        assert!(results[1].contains("input_json_delta"));
        assert!(results[1].contains("partial_json"));
    }

    #[test]
    fn sse_tool_call_args_only_no_id() {
        // Subsequent chunk with only arguments (no id)
        let data = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\" more\"}"}}]},"finish_reason":null}]}"#;
        let results = AnthropicSseState::new().convert(data);
        assert_eq!(results.len(), 1); // input_json_delta only, no block_start
        assert!(results[0].contains("input_json_delta"));
    }

    // ── Round-trip: real Claude Code fixtures → DeepSeek format ──────

    /// Realistic fixture: user message with content blocks (Claude Code format).
    #[test]
    fn user_message_preserves_role() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let last_user = msgs.iter().rev().find(|m| m["role"] == "user").unwrap();
        assert_eq!(last_user["content"], "hello world",
            "user message must keep role=user, not become assistant");
    }

    /// Tool result in user message keeps role=tool in DeepSeek format.
    #[test]
    fn tool_result_becomes_tool_message() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "grep for main"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "main"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": [{"type": "text", "text": "src/main.rs:5: fn main()"}]}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();

        // Should have: user, assistant(tool_calls), tool_result
        let tool_msgs: Vec<_> = msgs.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_msgs.len(), 1, "should have exactly 1 tool message");
        assert_eq!(tool_msgs[0]["tool_call_id"], "toolu_01",
            "tool_call_id must match tool_use id");
        // Content should be extracted from array
        let content = tool_msgs[0]["content"].as_str().unwrap();
        assert_eq!(content, "src/main.rs:5: fn main()",
            "tool_result content must be extracted from content blocks");
    }

    /// Tool call with no results (model's pending tool use) gets stripped.
    #[test]
    fn pending_tool_calls_are_stripped() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "grep for main"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "main"}}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        // The assistant with pending tool_calls should have its tool_calls stripped
        // (and the message removed if it has no content).
        let assistants_with_tc: Vec<_> = msgs.iter()
            .filter(|m| m.get("tool_calls").is_some())
            .collect();
        assert!(assistants_with_tc.is_empty(),
            "no assistant should have pending tool_calls without results. msgs={msgs:?}");
    }

    /// Reasoning content from cache is injected into assistant messages.
    #[test]
    fn reasoning_content_injection() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "what's new?"}
            ]
        });
        let ds = request_to_deepseek(&body, Some("Let me think about this..."));
        let msgs = ds["messages"].as_array().unwrap();
        // The assistant message should have reasoning_content
        let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        assert_eq!(assistant["reasoning_content"], "Let me think about this...",
            "assistant messages must include reasoning_content from cache");
    }

    /// Combined text + tool_use without tool result → tool_calls stripped, text kept.
    #[test]
    fn combined_text_and_tool_use() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "find main"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me search for that"},
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "main"}}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        assert!(assistant["content"].as_str().unwrap_or("").contains("Let me search"),
            "text must be preserved");
        // Pending tool_use without result → stripped
        assert!(assistant.get("tool_calls").is_none(),
            "pending tool_calls without result must be stripped");
    }

    /// Multiple tool calls with interleaved text results → tool msgs kept consecutive.
    #[test]
    fn multi_tool_with_text_results() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "find and edit"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "old"}},
                    {"type": "tool_use", "id": "toolu_02", "name": "Edit", "input": {"path": "file.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "found: line 42"},
                    {"type": "text", "text": "here is the grep result"}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "edited successfully"},
                    {"type": "text", "text": "done editing"}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();

        // Find the assistant with tool_calls
        let tc_pos = msgs.iter().position(|m| m["tool_calls"].is_array()).unwrap();
        let tc_count = msgs[tc_pos]["tool_calls"].as_array().unwrap().len();
        // Count consecutive tool messages after tool_calls
        let tool_count = msgs[tc_pos+1..].iter()
            .take_while(|m| m["role"] == "tool")
            .count();
        assert!(tool_count >= tc_count,
            "must have at least {tc_count} consecutive tool messages after tool_calls, got {tool_count}. msgs={msgs:?}");
        // All expected tool_call_ids must be in the tool messages
        let tool_ids: Vec<&str> = msgs[tc_pos+1..tc_pos+1+tool_count].iter()
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        assert!(tool_ids.contains(&"toolu_01"), "toolu_01 must be in {tool_ids:?}");
        assert!(tool_ids.contains(&"toolu_02"), "toolu_02 must be in {tool_ids:?}");
    }

    /// System prompt as array of text blocks (Anthropic format) → single system message.
    #[test]
    fn system_as_content_blocks() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "system": [
                {"type": "text", "text": "You are Claude Code."},
                {"type": "text", "text": "You help with software."}
            ],
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let system = msgs.iter().find(|m| m["role"] == "system").unwrap();
        let content = system["content"].as_str().unwrap();
        assert!(content.contains("You are Claude Code"),
            "system content blocks must be joined");
        assert!(content.contains("You help with software"),
            "all system text blocks must be included");
    }

    // ── Edge cases for request_to_deepseek ────────────────────────────

    #[test]
    fn request_with_empty_messages() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "messages": []
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        assert!(msgs.is_empty(), "empty input = empty output");
    }

    #[test]
    fn request_with_thinking_blocks() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "think step by step"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Let me reason about this..."},
                    {"type": "text", "text": "Here is the answer."}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        let content = assistant["content"].as_str().unwrap_or("");
        assert!(content.contains("[thinking]"), "thinking blocks should be marked");
        assert!(content.contains("Here is the answer"), "text should follow thinking");
    }

    #[test]
    fn request_with_tool_choice_any() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "tool_choice": {"type": "any"},
            "tools": [{"name": "bash", "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "run ls"}]
        });
        let ds = request_to_deepseek(&body, None);
        assert_eq!(ds["tool_choice"], json!("auto"), "\"any\" should map to \"auto\"");
    }

    #[test]
    fn request_with_tool_choice_specific_tool() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "tool_choice": {"type": "tool", "name": "bash"},
            "tools": [{"name": "bash", "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "run ls"}]
        });
        let ds = request_to_deepseek(&body, None);
        assert_eq!(ds["tool_choice"]["type"], "function");
        assert_eq!(ds["tool_choice"]["function"]["name"], "bash");
    }

    #[test]
    fn request_preserves_top_p_and_stop() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "top_p": 0.9,
            "stop_sequences": ["\n\n", "Observation:"],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let ds = request_to_deepseek(&body, None);
        assert_eq!(ds["top_p"], 0.9);
        assert_eq!(ds["stop"], json!(["\n\n", "Observation:"]));
    }

    #[test]
    fn request_thinking_field_sets_reasoning_effort() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "think hard"}]
        });
        let ds = request_to_deepseek(&body, None);
        assert_eq!(ds["reasoning_effort"], "medium",
            "thinking field should set reasoning_effort");
    }

    // ── Edge cases for response_to_anthropic ──────────────────────────

    #[test]
    fn response_empty_choices_returns_fallback() {
        let ds = json!({"model": "deepseek", "choices": [], "usage": {}});
        let anth = response_to_anthropic(&ds);
        assert_eq!(anth["type"], "message");
        assert!(anth["content"].as_array().unwrap().is_empty(),
            "empty choices should produce empty content");
        assert_eq!(anth["usage"]["input_tokens"], 0);
    }

    #[test]
    fn response_with_text_and_tool_calls_merges_both() {
        let ds = json!({
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check that file.",
                    "tool_calls": [{
                        "id": "call_01",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let anth = response_to_anthropic(&ds);
        let blocks = anth["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "text + tool_use should both be present");
        assert_eq!(blocks[0]["type"], "text");
        assert!(blocks[0]["text"].as_str().unwrap().contains("Let me check"));
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "call_01");
    }

    #[test]
    fn response_with_tool_calls_only_no_text() {
        let ds = json!({
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "bash", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        });
        let anth = response_to_anthropic(&ds);
        let blocks = anth["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "only tool_use block");
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["name"], "bash");
    }

    // ── Edge cases for AnthropicSseState.convert ──────────────────────

    #[test]
    fn sse_text_delta_starts_at_sequential_index() {
        let mut state = AnthropicSseState::new();
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#);
        // Should be content_block_start + text_delta
        assert_eq!(r.len(), 2);
        assert!(r[0].contains(r#""index":0"#), "first content block starts at 0");
    }

    #[test]
    fn sse_tool_call_separate_chunks_use_global_index() {
        let mut state = AnthropicSseState::new();
        // Chunk 1: first tool call (index 0)
        let r1 = state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#);
        assert!(r1[0].contains(r#""index":0"#), "tool call 0 at index 0");
        // Chunk 2: second tool call (index 1) — separate chunk
        let r2 = state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"tc2","function":{"name":"edit","arguments":""}}]},"finish_reason":null}]}"#);
        assert!(r2[0].contains(r#""index":1"#), "tool call 1 at index 1");
        // Chunk 3: args for both tools
        let r3 = state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\""}},{"index":1,"function":{"arguments":"\"file.txt\""}}]},"finish_reason":null}]}"#);
        assert_eq!(r3.len(), 2, "two argument deltas");
        // Verify finish emits stops at correct indices
        let finish = state.convert(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"prompt_tokens":5,"total_tokens":15}}"#);
        assert_eq!(finish.len(), 4, "2 stops + message_delta + message_stop");
        assert!(finish[0].contains(r#""index":0"#), "stop for tool 0");
        assert!(finish[1].contains(r#""index":1"#), "stop for tool 1");
    }

    #[test]
    fn sse_thinking_block_emits_start_then_delta() {
        let mut state = AnthropicSseState::new();
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{"reasoning_content":"Let me think..."},"finish_reason":null}]}"#);
        assert_eq!(r.len(), 2, "content_block_start + thinking_delta");
        assert!(r[0].contains("content_block_start"), "first event is block start");
        assert!(r[0].contains("thinking"), "block type is thinking");
        assert!(r[1].contains("thinking_delta"), "delta is thinking_delta");
    }

    #[test]
    fn sse_thinking_then_text_use_different_indices() {
        let mut state = AnthropicSseState::new();
        // Thinking block
        state.convert(r#"{"choices":[{"index":0,"delta":{"reasoning_content":"Thinking..."},"finish_reason":null}]}"#);
        // Text block (should use index 1, not 0)
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{"content":"Answer."},"finish_reason":null}]}"#);
        assert!(r[0].contains(r#""index":1"#), "text block should use index 1 (not 0)");
        // Finish should stop both
        let finish = state.convert(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5,"prompt_tokens":3,"total_tokens":8}}"#);
        assert!(finish[0].contains(r#""index":0"#), "stop for thinking at 0");
        assert!(finish[1].contains(r#""index":1"#), "stop for text at 1");
    }

    #[test]
    fn sse_tool_use_after_thinking_uses_next_index() {
        let mut state = AnthropicSseState::new();
        // Thinking
        state.convert(r#"{"choices":[{"index":0,"delta":{"reasoning_content":"Let me plan..."},"finish_reason":null}]}"#);
        // Text
        state.convert(r#"{"choices":[{"index":0,"delta":{"content":"I will search."},"finish_reason":null}]}"#);
        // Tool call
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"tc1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#);
        assert!(r[0].contains(r#""index":2"#), "tool call uses global index 2");
    }

    #[test]
    fn sse_finish_without_any_blocks_started() {
        let mut state = AnthropicSseState::new();
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":0,"prompt_tokens":0,"total_tokens":0}}"#);
        // No content_block_stop since no blocks started
        assert_eq!(r.len(), 2, "message_delta + message_stop only");
        assert!(r[0].contains("message_delta"));
    }

    #[test]
    fn sse_finish_reason_length() {
        let mut state = AnthropicSseState::new();
        state.convert(r#"{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#);
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":100,"prompt_tokens":4096,"total_tokens":4196}}"#);
        assert!(r[0].contains("stop_reason"));
        assert!(r[0].contains("max_tokens"), "length should map to max_tokens");
    }

    #[test]
    fn sse_finish_reason_tool_calls() {
        let mut state = AnthropicSseState::new();
        state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#);
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":20,"prompt_tokens":10,"total_tokens":30}}"#);
        assert!(r[0].contains("stop_reason"));
        assert!(r[0].contains("tool_use"), "tool_calls should map to tool_use");
    }

    #[test]
    fn sse_malformed_json_returns_empty() {
        let mut state = AnthropicSseState::new();
        let r = state.convert("not json at all");
        assert!(r.is_empty(), "malformed JSON should produce empty result");
    }

    #[test]
    fn sse_empty_content_does_not_emit_block_start() {
        let mut state = AnthropicSseState::new();
        let r = state.convert(r#"{"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#);
        assert!(r.is_empty(), "empty content string should produce no events");
        // Subsequent real content should work
        let r2 = state.convert(r#"{"choices":[{"index":0,"delta":{"content":"real"},"finish_reason":null}]}"#);
        assert_eq!(r2.len(), 2, "real content still works after empty skip");
    }

    #[test]
    fn started_block_indices_ordering() {
        let mut state = AnthropicSseState::new();
        // Start text (gets index 0)
        state.convert(r#"{"choices":[{"index":0,"delta":{"content":"text"},"finish_reason":null}]}"#);
        // Start tool (gets index 1 via explicit global index)
        state.convert(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#);
        let indices = state.started_block_indices();
        // Check they are sorted
        for w in indices.windows(2) {
            assert!(w[0] < w[1], "indices must be sorted: {:?}", indices);
        }
    }

    #[test]
    fn any_block_started_false_when_no_blocks() {
        let state = AnthropicSseState::new();
        assert!(!state.any_block_started(), "fresh state has no blocks");
    }

    #[test]
    fn any_block_started_true_after_text() {
        let mut state = AnthropicSseState::new();
        state.convert(r#"{"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#);
        assert!(state.any_block_started(), "text block should register as started");
    }

    /// Orphan tool message (tool_result without preceding tool_calls) in request.
    #[test]
    fn request_orphan_tool_message_preserved() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "what happened?"},
                {"role": "tool", "tool_call_id": "orphan_01", "content": "result from previous turn"}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let tool_msgs: Vec<_> = msgs.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_msgs.len(), 1, "orphan tool message preserved");
        assert_eq!(tool_msgs[0]["tool_call_id"], "orphan_01");
    }

    /// Multiple consecutive assistant messages with tool_calls.
    #[test]
    fn request_consecutive_tool_calls_preserved() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "do two things"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tc1", "name": "Grep", "input": {"pattern": "foo"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "result1"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tc2", "name": "Edit", "input": {"path": "x"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc2", "content": "result2"}]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let tc_count = msgs.iter().filter(|m| m.get("tool_calls").is_some()).count();
        assert_eq!(tc_count, 2, "both assistant tool_calls should be preserved");
    }

    /// Pending tool_calls with content text in the same assistant message
    /// should strip tool_calls but keep the text.
    #[test]
    fn request_pending_tool_with_text_keeps_text() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "search and tell me"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll search for that"},
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "main"}}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        assert!(assistant.get("tool_calls").is_none(), "pending tool_calls stripped");
        assert_eq!(assistant["content"], "I'll search for that", "text preserved");
    }

    /// Response with non-streaming tool_use content.
    #[test]
    fn response_with_tool_use_parses_input_json() {
        let ds = json!({
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\": \"ls -la\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        });
        let anth = response_to_anthropic(&ds);
        let blocks = anth["content"].as_array().unwrap();
        assert_eq!(blocks[0]["input"]["command"], "ls -la");
    }

    /// Response with malformed tool arguments JSON should not crash.
    #[test]
    fn response_tool_arguments_malformed_json_does_not_panic() {
        let ds = json!({
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "bash", "arguments": "not valid json"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        });
        let anth = response_to_anthropic(&ds);
        let blocks = anth["content"].as_array().unwrap();
        assert_eq!(blocks[0]["input"], json!({}),
            "malformed JSON should fall back to empty object");
    }

    /// No system prompt at all.
    #[test]
    fn request_no_system_prompt_omits_system_message() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let system_count = msgs.iter().filter(|m| m["role"] == "system").count();
        assert_eq!(system_count, 0, "no system prompt = no system message");
    }

    /// Reasoning content not injected when cache is None.
    #[test]
    fn reasoning_content_not_injected_when_none() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "what's up?"}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        let assistants: Vec<_> = msgs.iter().filter(|m| m["role"] == "assistant").collect();
        assert_eq!(assistants.len(), 1, "assistant messages preserved");
        // No reasoning_content injected since we passed None
        assert!(assistants[0].get("reasoning_content").is_none(),
            "no reasoning_content when cache is None");
    }

    /// Content blocks with only text and tool_results (no tool_use) are flattened correctly.
    #[test]
    fn request_tool_result_only_no_tool_use() {
        let body = json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "read file"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tc1", "name": "read_file", "input": {"path": "/tmp/x"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tc1", "content": "file contents here"}
                ]}
            ]
        });
        let ds = request_to_deepseek(&body, None);
        let msgs = ds["messages"].as_array().unwrap();
        // After repair, tool should directly follow tool_calls
        let tc_pos = msgs.iter().position(|m| m["tool_calls"].is_array()).unwrap();
        assert_eq!(msgs[tc_pos + 1]["role"], "tool",
            "tool must immediately follow tool_calls");
        assert_eq!(msgs[tc_pos + 1]["tool_call_id"], "tc1");
    }
}
