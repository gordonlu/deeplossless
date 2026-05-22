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
            let role = match role_str {
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => if item_type == "function_call_output" { Role::Tool } else { Role::User },
            };
            let mut parts = Vec::new();
            let mut meta = None;
            if let Some(content) = item["content"].as_array() {
                for block in content {
                    match block["type"].as_str().unwrap_or("") {
                        "input_text" | "output_text" => {
                            if let Some(t) = block["text"].as_str() {
                                parts.push(ContentPart::Text { text: t.to_string() });
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
            // function_call items
            if item_type == "function_call" {
                let id = item["call_id"].as_str().unwrap_or("").to_string();
                let name = item["name"].as_str().unwrap_or("").to_string();
                let args_str = item["arguments"].as_str().unwrap_or(&item["arguments"].to_string()).to_string();
                let args: serde_json::Value = serde_json::from_str(&args_str).unwrap_or(serde_json::json!({}));
                parts.push(ContentPart::ToolCall { id: id.clone(), name: name.clone(), arguments: args.clone() });
                meta = Some(MessageMeta { tool_call_id: None, tool_calls: vec![ToolInvocation { id, name, arguments: args }] });
            }
            // function_call_output items
            if item_type == "function_call_output" {
                let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                let content = item["output"].as_str().unwrap_or("").to_string();
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
            messages.push(Message { role, parts, meta });
        }
    } else if let Some(s) = body["input"].as_str() {
        messages.push(Message { role: Role::User, parts: vec![ContentPart::Text { text: s.to_string() }], meta: None });
    }

    let response_format = body["text"]["format"].as_object()
        .filter(|f| f["type"] == "json_schema")
        .map(|f| ResponseFormat { format_type: "json_schema".into(), json_schema: f.get("json_schema").cloned() });

    CanonicalRequest {
        instructions, messages, tools, model, stream, max_tokens, temperature, response_format,
        provider: ProviderKind::OpenAI,
        capabilities: ProviderCapabilities { tool_streaming: ToolStreamingMode::Parallel, reasoning: ReasoningMode::Partial, structured_output: StructuredOutputMode::JsonSchema, multimodal: true },
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
    if let Some(content) = delta["content"].as_str() {
        return vec![StreamEvent::TextDelta { text: content.to_string() }];
    }
    if let Some(tool_calls) = delta["tool_calls"].as_array()
        && let Some(tc) = tool_calls.first() {
        let index = tc["index"].as_u64().unwrap_or(0) as usize;
        if let Some(name) = tc["function"]["name"].as_str() {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let args = tc["function"]["arguments"].as_str().unwrap_or("").to_string();
            let mut events = vec![StreamEvent::ToolCallStart { index, id, name: name.to_string() }];
            if !args.is_empty() {
                events.push(StreamEvent::ToolCallArgsDelta { index, arguments_delta: args });
            }
            return events;
        }
        let args = tc["function"]["arguments"].as_str().unwrap_or("").to_string();
        if !args.is_empty() {
            return vec![StreamEvent::ToolCallArgsDelta { index, arguments_delta: args }];
        }
    }
    if let Some(reason) = choice["finish_reason"].as_str() {
        return vec![StreamEvent::Done {
            usage: Usage {
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
            },
            finish_reason: reason.to_string(),
            incomplete: false,
            error_reason: None,
        }];
    }
    if let Some(err) = v["error"].as_object() {
        return vec![StreamEvent::Error { message: err["message"].as_str().unwrap_or("").to_string(), code: err["code"].as_str().map(|s| s.to_string()) }];
    }
    vec![]
}

pub fn stream_event_to_responses(event: &StreamEvent) -> String {
    use serde_json::json;
    match event {
        StreamEvent::TextDelta { text } => json!({"type": "response.output_text.delta", "delta": text}).to_string(),
        StreamEvent::ToolCallStart { index, id, name } => json!({"type": "response.output_item.added", "item": {"id": id, "type": "function_call", "name": name}, "output_index": index}).to_string(),
        StreamEvent::ToolCallArgsDelta { index, arguments_delta, .. } => json!({"type": "response.function_call_arguments.delta", "output_index": index, "delta": arguments_delta}).to_string(),
        StreamEvent::ToolCallEnd { index } => json!({"type": "response.output_item.done", "output_index": index}).to_string(),
        StreamEvent::FunctionCallArgumentsDone { call_id, name, arguments } => json!({"type": "response.function_call_arguments.done", "item_id": call_id, "name": name, "arguments": arguments}).to_string(),
        StreamEvent::OutputItemAdded { index, item_type } => json!({"type": "response.output_item.added", "output_index": index, "item": {"type": item_type}}).to_string(),
        StreamEvent::OutputItemDone { index } => json!({"type": "response.output_item.done", "output_index": index}).to_string(),
        StreamEvent::Done { usage, .. } => json!({"type": "response.completed", "response": {"usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}}}).to_string(),
        StreamEvent::Error { message, code } => json!({"type": "error", "error": {"type": "server_error", "code": code.as_deref().unwrap_or("unknown"), "message": message}}).to_string(),
        _ => String::new(),
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);
pub fn monotonic_id() -> String { format!("{:016x}", ID_COUNTER.fetch_add(1, Ordering::Relaxed)) }
