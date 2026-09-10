//! OpenAI Responses API ↔ canonical IR adapter.

use super::canonical::*;

pub fn request_from_responses(body: &serde_json::Value) -> CanonicalRequest {
    let model = body["model"].as_str().unwrap_or("").to_string();
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_output_tokens"].as_u64().map(|n| n as u32);
    let temperature = body["temperature"].as_f64();

    let mut instructions: Vec<InstructionBlock> = Vec::new();
    if let Some(s) = body["instructions"].as_str() {
        instructions.push(InstructionBlock { text: s.to_string(), meta: None });
    }

    let tools: Vec<ToolDef> = body["tools"].as_array()
        .map(|arr| arr.iter().filter(|t| t["type"] == "function").map(|t| ToolDef {
            name: t["name"].as_str().unwrap_or("").to_string(),
            description: t["description"].as_str().unwrap_or("").to_string(),
            parameters: t.get("parameters").cloned().unwrap_or(serde_json::json!({})),
            strict: t["strict"].as_bool().unwrap_or(true),
        }).collect())
        .unwrap_or_default();

    let mut messages: Vec<Message> = Vec::new();
    let mut pending_reasoning: Option<String> = None;
    if let Some(arr) = body["input"].as_array() {
        for item in arr {
            let item_type = item["type"].as_str().unwrap_or("");
            let role_str = item["role"].as_str().unwrap_or("user");
            // Developer messages are system-level instructions — append to
            // instructions and skip from the messages array.
            if role_str == "developer" || role_str == "system" {
                if let Some(content) = item["content"].as_array() {
                    for block in content {
                        if let Some(t) = block["text"].as_str() {
                            instructions.push(InstructionBlock { text: t.to_string(), meta: None });
                        }
                    }
                } else if let Some(s) = item["content"].as_str() {
                    instructions.push(InstructionBlock { text: s.to_string(), meta: None });
                }
                continue;
            }
            // Reasoning items attach to the previous assistant message.
            // If no assistant message exists yet (e.g. reasoning before function_call),
            // store pending so it's attached when the assistant message is created.
            // Format: {"type": "reasoning", "text": "..."} or
            //         {"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]}
            if item_type == "reasoning" {
                let reasoning_text = item["text"].as_str()
                    .or_else(|| item["summary"].as_array()
                        .and_then(|s| s.first())
                        .and_then(|b| b["text"].as_str()))
                    .unwrap_or("");
                if !reasoning_text.is_empty() {
                    if let Some(last) = messages.iter_mut().rev()
                        .find(|m| m.role == Role::Assistant)
                    {
                        last.reasoning = Some(ReasoningTrace {
                            text: reasoning_text.to_string(),
                            summarized: false,
                            tokens: None,
                        });
                    } else {
                        pending_reasoning = Some(reasoning_text.to_string());
                    }
                }
                continue;
            }
            let role = match role_str {
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => if item_type == "function_call_output" { Role::Tool } else { Role::User },
            };
            let mut parts = Vec::new();
            let mut meta = None;
            let mut reasoning: Option<String> = None;
            if let Some(content) = item["content"].as_array() {
                for block in content {
                    match block["type"].as_str().unwrap_or("") {
                        "input_text" | "output_text" => {
                            if let Some(t) = block["text"].as_str() {
                                parts.push(ContentPart::Text { text: t.to_string() });
                            }
                        }
                        "reasoning" => {
                            if let Some(t) = block["text"].as_str() {
                                reasoning = Some(t.to_string());
                            }
                        }
                        "input_image" => {
                            if let Some(url) = block["image_url"].as_str() {
                                parts.push(ContentPart::Image { source_type: "url".into(), data: url.to_string(), detail: "auto".into() });
                            }
                            }
                        _ => {}
                    }
                }
            } else if let Some(s) = item["content"].as_str() {
                parts.push(ContentPart::Text { text: s.to_string() });
            }
            // function_call items — merge into previous assistant message, or
            // create one if none exists (Codex may send function_call standalone).
            // In Responses API, function_call is a separate output item, but
            // Chat Completions requires tool_calls inside the assistant message.
            // IMPORTANT: only merge if no non-tool messages intervene (user messages
            // between turns break DeepSeek's adjacency requirement).
            if item_type == "function_call" {
                let id = item["call_id"].as_str()
                    .or_else(|| item["id"].as_str())
                    .unwrap_or("").to_string();
                let name = item["name"].as_str().unwrap_or("").to_string();
                let args_str = item["arguments"].as_str().unwrap_or(&item["arguments"].to_string()).to_string();
                let args: serde_json::Value = serde_json::from_str(&args_str).unwrap_or(serde_json::json!({}));
                let tc = ToolInvocation { id: id.clone(), name: name.clone(), arguments: args.clone() };
                // Check if there's a non-tool message between the last assistant and here
                let last_assistant_pos = messages.iter().rposition(|m| m.role == Role::Assistant);
                let intervening_non_tool = messages.iter().rev()
                    .take_while(|m| m.role != Role::Assistant)
                    .any(|m| m.role != Role::Tool);
                // Capture fallback reasoning before any mutable borrow of messages.
                // DeepSeek thinking mode requires reasoning_content on every assistant
                // message — inherit from previous assistants if no pending reasoning.
                let fallback_reasoning = pending_reasoning.take()
                    .or_else(|| messages.iter().rev().find_map(|m| m.reasoning.as_ref().map(|r| r.text.clone())));
                match last_assistant_pos {
                    Some(idx) if !intervening_non_tool => {
                        // Merge into existing assistant (no intervening non-tool messages)
                        let last = &mut messages[idx];
                        last.parts.push(ContentPart::ToolCall { id, name, arguments: args });
                        let meta = last.meta.get_or_insert_with(|| MessageMeta { tool_call_id: None, tool_calls: vec![] });
                        meta.tool_calls.push(tc);
                        if last.reasoning.is_none() {
                            if let Some(r) = fallback_reasoning.clone() {
                                last.reasoning = Some(ReasoningTrace { text: r, summarized: false, tokens: None });
                            }
                        }
                    }
                    _ => {
                        // Non-tool message between (user boundary) or no assistant at all
                        // Create a new assistant message with the tool call.
                        let reasoning = fallback_reasoning.clone()
                            .map(|text| ReasoningTrace { text, summarized: false, tokens: None });
                        messages.push(Message {
                            role: Role::Assistant,
                            parts: vec![ContentPart::ToolCall { id, name, arguments: args }],
                            meta: Some(MessageMeta { tool_call_id: None, tool_calls: vec![tc] }),
                            reasoning,
                        });
                    }
                }
                continue;
            }
            // function_call_output items
            if item_type == "function_call_output" {
                let call_id = item["call_id"].as_str()
                    .or_else(|| item["id"].as_str())
                    .unwrap_or("").to_string();
                let content = item["output"].as_str()
                    .map(|s| s.to_string())
                    .or_else(|| item["output"].as_array().map(|arr| {
                        arr.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("\n")
                    }))
                    .unwrap_or_default();
                parts.push(ContentPart::ToolResult { call_id: call_id.clone(), content });
                meta = Some(MessageMeta { tool_call_id: Some(call_id), tool_calls: vec![] });
            }
            // Chat Completions-style tool messages (role=tool + tool_call_id)
            if role == Role::Tool {
                let explicit_call_id = item["tool_call_id"].as_str().map(|s| s.to_string());
                if let Some(ref call_id) = explicit_call_id {
                    if parts.iter().any(|p| matches!(p, ContentPart::ToolResult { .. })) {
                        if meta.as_ref().and_then(|m| m.tool_call_id.as_ref()).is_none() {
                            meta = Some(MessageMeta { tool_call_id: Some(call_id.clone()), tool_calls: vec![] });
                        }
                    } else {
                        let content = parts.iter()
                            .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                            .collect::<Vec<_>>().join("\n");
                        parts.clear();
                        parts.push(ContentPart::ToolResult { call_id: call_id.clone(), content });
                        meta = Some(MessageMeta { tool_call_id: Some(call_id.clone()), tool_calls: vec![] });
                    }
                }
            }
            let reasoning = reasoning.or_else(|| item["reasoning_content"].as_str().map(|s| s.to_string()));
            let reasoning = reasoning.map(|text| ReasoningTrace { text, summarized: false, tokens: None });
            messages.push(Message { role, parts, meta, reasoning });
        }
    } else if let Some(s) = body["input"].as_str() {
        messages.push(Message { role: Role::User, parts: vec![ContentPart::Text { text: s.to_string() }], meta: None, reasoning: None });
    }

    let response_format = body["text"]["format"].as_object()
        .filter(|f| f["type"] == "json_schema")
        .map(|f| ResponseFormat { format_type: "json_schema".into(), json_schema: f.get("json_schema").cloned() });

    CanonicalRequest {
        instructions, messages, tools, model, stream, max_tokens, temperature, response_format,
        provider: ProviderKind::OpenAI,
        capabilities: ProviderCapabilities { tool_streaming: ToolStreamingMode::Parallel, reasoning: ReasoningMode::Partial, structured_output: StructuredOutputMode::JsonSchema, multimodal: true },
        deepseek_native: Default::default(),
    }
}

pub fn response_to_responses(cr: &CanonicalResponse) -> serde_json::Value {
    use serde_json::json;
    let mut output = Vec::new();
    for part in &cr.output {
        match part {
            ContentPart::Text { text } => {
                output.push(json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}));
            }
            ContentPart::ToolCall { id, name, arguments } => {
                output.push(json!({"type": "function_call", "call_id": id, "name": name, "arguments": arguments.to_string()}));
            }
            _ => {}
        }
    }
    if let Some(ref rt) = cr.reasoning_trace {
        output.push(json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": rt.text}]}));
    }
    let resp_id = if cr.id.starts_with("chatcmpl-") { cr.id.replacen("chatcmpl-", "resp_", 1) } else if cr.id.is_empty() { format!("resp_{}", monotonic_id()) } else { cr.id.clone() };
    json!({
        "id": resp_id, "object": "response", "model": cr.model,
        "status": match cr.status { ResponseStatus::Completed => "completed", ResponseStatus::Incomplete => "incomplete", ResponseStatus::Error => "failed" },
        "output": output,
        "usage": {"input_tokens": cr.usage.prompt_tokens, "output_tokens": cr.usage.completion_tokens, "total_tokens": cr.usage.total_tokens},
    })
}

// ── Streaming ──────────────────────────────────────────────────────────

pub fn stream_event_from_chat(data: &str) -> Vec<StreamEvent> {
    let Some(v) = serde_json::from_str::<serde_json::Value>(data).ok() else { return vec![] };
    let Some(choices) = v["choices"].as_array() else { return vec![] };
    let Some(choice) = choices.first() else { return vec![] };
    let delta = &choice["delta"];

    // Accumulate events — a single SSE chunk may contain multiple event types
    // (e.g. content + reasoning, content + finish_reason, tool_calls + finish_reason).
    let mut events: Vec<StreamEvent> = Vec::new();

    // 1. Text content (must be non-empty; empty content may accompany finish_reason)
    if let Some(content) = delta["content"].as_str() {
        if !content.is_empty() {
            events.push(StreamEvent::TextDelta { text: content.to_string() });
        }
    }

    // 2. Finish reason — always emit Done. The proxy handler catches Done
    //    and never forwards it to Codex — it's just a trigger for flush().
    //    This ensures the assembler's flush() is called for tool_calls too,
    //    emitting FunctionCallArgumentsDone/OutputItemDone downstream.
    if let Some(reason) = choice["finish_reason"].as_str() {
        events.push(StreamEvent::Done {
            usage: Usage {
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
            },
            finish_reason: reason.to_string(),
            incomplete: reason == "length",
            error_reason: None,
        });
    }

    // 3. Reasoning content
    if let Some(reasoning) = delta["reasoning_content"].as_str() {
        events.push(StreamEvent::ReasoningDelta { text: reasoning.to_string() });
    }

    // 4. Tool calls — accumulate multiple tool calls from the same chunk
    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        for tc in tool_calls {
            let index = tc["index"].as_u64().unwrap_or(0) as usize;
            if let Some(name) = tc["function"]["name"].as_str() {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let args = tc["function"]["arguments"].as_str().unwrap_or("").to_string();
                events.push(StreamEvent::ToolCallStart { index, id: id.clone(), name: name.to_string() });
                if !args.is_empty() {
                    events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: args, call_id: id });
                }
            } else {
                let args = tc["function"]["arguments"].as_str().unwrap_or("").to_string();
                if !args.is_empty() {
                    events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: args, call_id: String::new() });
                }
            }
        }
    }

    // 5. Error
    if let Some(err) = v["error"].as_object() {
        events.push(StreamEvent::Error { message: err["message"].as_str().unwrap_or("").to_string(), code: err["code"].as_str().map(|s| s.to_string()) });
    }

    events
}

pub fn stream_event_to_responses(event: &StreamEvent) -> String {
    use serde_json::json;
    match event {
        StreamEvent::TextDelta { text } => json!({"type": "response.output_text.delta", "delta": text}).to_string(),
        StreamEvent::ToolCallStart { index, id, name } => json!({"type": "response.output_item.added", "item": {"call_id": id, "type": "function_call", "name": name, "arguments": "", "status": "in_progress"}, "output_index": index}).to_string(),
        StreamEvent::ToolCallArgsDelta { index, arguments_delta, call_id } => json!({"type": "response.function_call_arguments.delta", "output_index": index, "delta": arguments_delta, "item_id": call_id}).to_string(),
        StreamEvent::ToolCallEnd { index } => json!({"type": "response.output_item.done", "output_index": index}).to_string(),
        StreamEvent::FunctionCallArgumentsDone { call_id, name, arguments, output_index } => json!({"type": "response.function_call_arguments.done", "item_id": call_id, "name": name, "arguments": arguments, "output_index": output_index}).to_string(),
        StreamEvent::OutputItemAdded { index, item_type } => json!({"type": "response.output_item.added", "output_index": index, "item": {"type": item_type}}).to_string(),
        StreamEvent::OutputItemDone { index, item_id, item_type, name, arguments } => {
            let mut item = serde_json::json!({"call_id": item_id, "type": item_type});
            if item_type == "function_call" {
                item["name"] = serde_json::json!(name);
                item["arguments"] = serde_json::json!(arguments);
            }
            serde_json::json!({"type": "response.output_item.done", "output_index": index, "item": item}).to_string()
        }
        StreamEvent::ReasoningDelta { text } => json!({"type": "response.reasoning_text.delta", "delta": text}).to_string(),
        StreamEvent::Done { usage, .. } => json!({"type": "response.completed", "response": {"usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}}}).to_string(),
        StreamEvent::Error { message, code } => json!({"type": "error", "error": {"type": "server_error", "code": code.as_deref().unwrap_or("unknown"), "message": message}}).to_string(),
        _ => String::new(),
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);
pub fn monotonic_id() -> String { format!("{:016x}", ID_COUNTER.fetch_add(1, Ordering::Relaxed)) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reasoning_content_from_sse() {
        let data = r#"{"choices":[{"delta":{"reasoning_content":"Let me think about this."},"index":0}]}"#;
        let events = stream_event_from_chat(data);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ReasoningDelta { text } => assert!(text.contains("Let me think")),
            other => panic!("expected ReasoningDelta, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_before_reasoning() {
        // content takes priority over reasoning_content — standard delta
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"index":0}]}"#;
        let events = stream_event_from_chat(data);
        assert!(!events.is_empty());
    }

    #[test]
    fn parse_empty_delta_returns_empty() {
        let data = r#"{"choices":[{"delta":{},"index":0}]}"#;
        let events = stream_event_from_chat(data);
        assert!(events.is_empty());
    }
}
