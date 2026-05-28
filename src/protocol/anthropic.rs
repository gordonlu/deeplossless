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

    // DeepSeek requires tool messages to immediately follow tool_calls.
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
            // Found tool_calls — push it, then collect all following tool msgs
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
            // Place tool messages directly after tool_calls, then other messages
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
    let choice = &deepseek["choices"][0];
    let message = &choice["message"];
    let content = message["content"].as_str().unwrap_or("");
    let model = deepseek["model"].as_str().unwrap_or("deepseek-v4-pro");
    let usage = &deepseek["usage"];

    let mut response = json!({
        "id": format!("msg_{}", chrono::Utc::now().timestamp_millis()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": choice["finish_reason"].as_str().unwrap_or("end_turn"),
        "usage": {
            "input_tokens": usage["prompt_tokens"].as_u64().unwrap_or(0),
            "output_tokens": usage["completion_tokens"].as_u64().unwrap_or(0),
        }
    });

    // Tool calls
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        let blocks: Vec<Value> = tool_calls.iter().map(|tc| {
            let func = &tc["function"];
            let input: Value = serde_json::from_str(func["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
            json!({
                "type": "tool_use",
                "id": tc["id"],
                "name": func["name"],
                "input": input,
            })
        }).collect();
        response["content"] = json!(blocks);
    }

    response
}

/// Stateful SSE converter that tracks content block lifecycle so
/// `content_block_start` for text is emitted before the first text delta.
#[derive(Default)]
pub struct AnthropicSseState {
    text_block_started: bool,
    tool_use_block_indices: Vec<usize>,
    /// Token usage captured from the finish chunk, for accumulation.
    pub last_input_tokens: u64,
    pub last_output_tokens: u64,
    /// Accumulated reasoning_content from DeepSeek deltas (thinking mode).
    pub reasoning_content: String,
}

impl AnthropicSseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a DeepSeek SSE data line into Anthropic SSE event strings.
    /// Maintains state to correctly emit `content_block_start` before the
    /// first delta of each content type (text or tool_use).
    pub fn convert(&mut self, data: &str) -> Vec<String> {
        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let delta = &v["choices"][0]["delta"];
        let mut events = Vec::new();

        // Tool call chunks — may carry id+name (start), arguments (delta), or both
        if let Some(tc) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for (i, call) in tc.iter().enumerate() {
                let name = call["function"]["name"].as_str().unwrap_or("");
                let id = call["id"].as_str().unwrap_or("");
                let args = call["function"]["arguments"].as_str().unwrap_or("");

                // New tool call starting → content_block_start
                if !id.is_empty() {
                    self.tool_use_block_indices.push(i);
                    events.push(format!("event: content_block_start\ndata: {}\n\n", json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                    })));
                }

                // Arguments (may appear in same chunk or subsequent chunks)
                if !args.is_empty() {
                    events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    })));
                }
            }
            return events;
        }

        // Reasoning content — accumulate for next turn, emit as thinking delta
        if let Some(rc) = delta["reasoning_content"].as_str() {
            if !rc.is_empty() {
                self.reasoning_content.push_str(rc);
                events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": rc}
                })));
            }
            return events;
        }

        // Text delta
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                if !self.text_block_started {
                    self.text_block_started = true;
                    events.push(format!("event: content_block_start\ndata: {}\n\n", json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""}
                    })));
                }
                events.push(format!("event: content_block_delta\ndata: {}\n\n", json!({
                    "type": "content_block_delta",
                    "index": 0,
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
        self.text_block_started || !self.tool_use_block_indices.is_empty()
    }

    /// Iterator of block indices that were started (for emitting content_block_stop).
    pub fn started_block_indices(&self) -> Vec<usize> {
        let mut indices = self.tool_use_block_indices.clone();
        if self.text_block_started {
            indices.push(0);
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
}
