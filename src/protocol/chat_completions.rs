//! OpenAI Chat Completions ↔ canonical IR adapter.
//! Used for DeepSeek and other Chat Completions-compatible providers.

use super::canonical::*;

/// Convert a Chat Completions request JSON into canonical IR.
pub fn request_from_chat(body: &serde_json::Value) -> CanonicalRequest {
    let model = body["model"].as_str().unwrap_or("").to_string();
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_tokens"].as_u64().map(|n| n as u32)
        .or_else(|| body["max_completion_tokens"].as_u64().map(|n| n as u32));
    let temperature = body["temperature"].as_f64();

    // Extract tools
    let tools: Vec<ToolDef> = body["tools"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let func = &t["function"];
                    Some(ToolDef {
                        name: func["name"].as_str()?.to_string(),
                        description: func["description"].as_str().unwrap_or("").to_string(),
                        parameters: func["parameters"].clone(),
                        strict: t["strict"].as_bool().unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Extract messages
    let mut instructions: Option<String> = None;
    let messages: Vec<Message> = body["messages"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|msg| {
                    let role_str = msg["role"].as_str().unwrap_or("user");
                    let role = match role_str {
                        "system" => Role::System,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::User,
                    };

                    let mut parts = Vec::new();

                    // Content: string or array
                    let content = &msg["content"];
                    if let Some(s) = content.as_str() {
                        if !s.is_empty() {
                            parts.push(ContentPart::Text {
                                text: s.to_string(),
                            });
                        }
                    } else if let Some(arr) = content.as_array() {
                        for block in arr {
                            if block["type"] == "text" || block["type"] == "input_text" {
                                if let Some(t) = block["text"].as_str() {
                                    parts.push(ContentPart::Text {
                                        text: t.to_string(),
                                    });
                                }
                            } else if block["type"] == "image_url"
                                && let Some(url) = block["image_url"]["url"].as_str() {
                                    parts.push(ContentPart::Image {
                                        source_type: "url".into(),
                                        data: url.to_string(),
                                        detail: block["image_url"]["detail"]
                                            .as_str()
                                            .unwrap_or("auto")
                                            .into(),
                                    });
                                }
                        }
                    }

                    // System message → instructions
                    if role == Role::System {
                        if let Some(ContentPart::Text { text }) = parts.first() {
                            instructions = Some(text.clone());
                        }
                        parts.clear();
                    }

                    // Tool calls from assistant
                    let mut meta = None;
                    if let Some(tool_calls) = msg["tool_calls"].as_array() {
                        let tcs: Vec<ToolInvocation> = tool_calls
                            .iter()
                            .map(|tc| ToolInvocation {
                                id: tc["id"].as_str().unwrap_or("").to_string(),
                                name: tc["function"]["name"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string(),
                                arguments: tc["function"]["arguments"]
                                    .as_str()
                                    .unwrap_or("{}")
                                    .to_string(),
                            })
                            .collect();
                        for tc in &tcs {
                            parts.push(ContentPart::ToolCall {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            });
                        }
                        meta = Some(MessageMeta {
                            tool_call_id: None,
                            tool_calls: tcs,
                        });
                    }

                    // Tool call ID for tool messages
                    if role == Role::Tool {
                        let tc_id = msg["tool_call_id"].as_str().map(|s| s.to_string());
                        if let Some(ContentPart::Text { text }) = parts.first() {
                            parts[0] = ContentPart::ToolResult {
                                call_id: tc_id.clone().unwrap_or_default(),
                                content: text.clone(),
                            };
                        }
                        meta = Some(MessageMeta {
                            tool_call_id: tc_id,
                            tool_calls: vec![],
                        });
                    }

                    Message {
                        role,
                        parts,
                        meta,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Structured output
    let response_format = body["response_format"].as_object().map(|rf| {
        ResponseFormat {
            format_type: rf["type"].as_str().unwrap_or("text").to_string(),
            json_schema: rf.get("json_schema").cloned(),
        }
    });

    CanonicalRequest {
        instructions,
        messages,
        tools,
        model,
        stream,
        max_tokens,
        temperature,
        response_format,
        provider_hint: "chat_completions".into(),
    }
}

/// Convert canonical IR back to Chat Completions request JSON.
pub fn request_to_chat(req: &CanonicalRequest) -> serde_json::Value {
    use serde_json::json;

    let mut msgs: Vec<serde_json::Value> = Vec::new();

    // System instructions → first message
    if let Some(ref inst) = req.instructions {
        msgs.push(json!({"role": "system", "content": inst}));
    }

    for msg in &req.messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };

        // Collect text parts into content string
        let text: Vec<&str> = msg
            .parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect();
        let content = if text.len() == 1 {
            text[0].to_string()
        } else if text.len() > 1 {
            text.join("\n")
        } else {
            String::new()
        };

        // Tool calls
        let tool_calls: Vec<serde_json::Value> = msg
            .parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } = p
                {
                    Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    }))
                } else {
                    None
                }
            })
            .collect();

        let mut m = json!({"role": role, "content": content});
        if !tool_calls.is_empty() {
            m["tool_calls"] = json!(tool_calls);
        }
        if role == "tool"
            && let Some(ref meta) = msg.meta
                && let Some(ref tc_id) = meta.tool_call_id {
                    m["tool_call_id"] = json!(tc_id);
                }
        msgs.push(m);
    }

    let mut body = json!({
        "model": req.model,
        "messages": msgs,
        "stream": req.stream,
    });

    if let Some(mt) = req.max_tokens {
        body["max_tokens"] = json!(mt);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if !req.tools.is_empty() {
        let tools: Vec<serde_json::Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                    "strict": t.strict,
                })
            })
            .collect();
        body["tools"] = json!(tools);
    }
    if let Some(ref rf) = req.response_format {
        body["response_format"] = json!({
            "type": rf.format_type,
            "json_schema": rf.json_schema,
        });
    }

    body
}

/// Convert a Chat Completions response JSON into canonical IR.
pub fn response_from_chat(body: &serde_json::Value) -> CanonicalResponse {
    use serde_json::json;

    let id = body["id"].as_str().unwrap_or("").to_string();
    let model = body["model"].as_str().unwrap_or("").to_string();

    let choice = &body["choices"][0];
    let msg = &choice["message"];

    // Extract content and tool calls
    let mut output = Vec::new();
    if let Some(content) = msg["content"].as_str()
        && !content.is_empty() {
            output.push(ContentPart::Text {
                text: content.to_string(),
            });
        }
    if let Some(tool_calls) = msg["tool_calls"].as_array() {
        for tc in tool_calls {
            output.push(ContentPart::ToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: tc["function"]["arguments"]
                    .as_str()
                    .unwrap_or("{}")
                    .to_string(),
            });
        }
    }

    // Status
    let finish = choice["finish_reason"].as_str().unwrap_or("stop");
    let status = match finish {
        "stop" => ResponseStatus::Completed,
        "tool_calls" | "function_call" => ResponseStatus::Incomplete,
        "length" => ResponseStatus::Incomplete,
        _ => ResponseStatus::Completed,
    };

    // Usage
    let usage = Usage {
        prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: body["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or(0) as u32,
        total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
    };

    CanonicalResponse {
        id,
        model,
        status,
        output,
        usage,
        provider_meta: Some(json!({
            "finish_reason": finish,
            "object": body["object"].as_str().unwrap_or(""),
        })),
    }
}
