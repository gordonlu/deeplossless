//! OpenAI Responses API ↔ canonical IR adapter.
//! Translates the new Responses API format to/from our canonical representation.

use super::canonical::*;

/// Convert a Responses API request JSON into canonical IR.
pub fn request_from_responses(body: &serde_json::Value) -> CanonicalRequest {
    let model = body["model"].as_str().unwrap_or("").to_string();
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_output_tokens"].as_u64().map(|n| n as u32);
    let temperature = body["temperature"].as_f64();
    let instructions = body["instructions"].as_str().map(|s| s.to_string());

    // Extract tools — Responses uses flat format (no "function" wrapper)
    let tools: Vec<ToolDef> = body["tools"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|t| t["type"] == "function")
                .map(|t| ToolDef {
                    name: t["name"].as_str().unwrap_or("").to_string(),
                    description: t["description"].as_str().unwrap_or("").to_string(),
                    parameters: t.get("parameters").cloned().unwrap_or(serde_json::json!({})),
                    strict: t["strict"].as_bool().unwrap_or(true), // Responses defaults to strict
                })
                .collect()
        })
        .unwrap_or_default();

    // Extract messages from input
    let messages: Vec<Message> = if let Some(arr) = body["input"].as_array() {
        arr.iter()
            .map(|item| {
                let role_str = item["role"].as_str().unwrap_or("user");
                let role = match role_str {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    "developer" | "tool" => Role::Tool,
                    _ => Role::User,
                };

                let mut parts = Vec::new();
                let mut meta = None;

                if let Some(content) = item["content"].as_array() {
                    for block in content {
                        match block["type"].as_str().unwrap_or("") {
                            "input_text" => {
                                if let Some(t) = block["text"].as_str() {
                                    parts.push(ContentPart::Text {
                                        text: t.to_string(),
                                    });
                                }
                            }
                            "input_image" => {
                                if let Some(url) = block["image_url"].as_str() {
                                    parts.push(ContentPart::Image {
                                        source_type: "url".into(),
                                        data: url.to_string(),
                                        detail: "auto".into(),
                                    });
                                }
                            }
                            "output_text" => {
                                if let Some(t) = block["text"].as_str() {
                                    parts.push(ContentPart::Text {
                                        text: t.to_string(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                } else if let Some(s) = item["content"].as_str() {
                    // content can be a plain string in Responses too
                    parts.push(ContentPart::Text {
                        text: s.to_string(),
                    });
                }

                // Handle function_call items in input (from prior turns)
                if item["type"] == "function_call" {
                    let id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let args = item["arguments"]
                        .as_str()
                        .unwrap_or(&item["arguments"].to_string())
                        .to_string();
                    parts.push(ContentPart::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: args.clone(),
                    });
                    meta = Some(MessageMeta {
                        tool_call_id: None,
                        tool_calls: vec![ToolInvocation { id: id.clone(), name: name.clone(), arguments: args }],
                    });
                }

                // Handle function_call_output items
                if item["type"] == "function_call_output" {
                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                    let content = item["output"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    parts.push(ContentPart::ToolResult {
                        call_id: call_id.clone(),
                        content,
                    });
                    meta = Some(MessageMeta {
                        tool_call_id: Some(call_id),
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
    } else if let Some(s) = body["input"].as_str() {
        // input as a string: treat as a single user message
        vec![Message {
            role: Role::User,
            parts: vec![ContentPart::Text {
                text: s.to_string(),
            }],
            meta: None,
        }]
    } else {
        vec![]
    };

    // Structured output: Responses uses text.format.json_schema
    let response_format = body["text"]["format"]
        .as_object()
        .filter(|f| f["type"] == "json_schema")
        .map(|f| ResponseFormat {
            format_type: "json_schema".into(),
            json_schema: f.get("json_schema").cloned(),
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
        provider_hint: "responses".into(),
    }
}

/// Convert canonical IR back to Responses API response JSON.
pub fn response_to_responses(cr: &CanonicalResponse, _req_id: Option<&str>) -> serde_json::Value {
    use serde_json::json;

    let mut output = Vec::new();

    // Convert content parts to Responses output items
    for part in &cr.output {
        match part {
            ContentPart::Text { text } => {
                output.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }));
            }
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => {
                output.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                }));
            }
            ContentPart::ToolResult { .. } => {
                // Tool results are input-side items, not in output
            }
            ContentPart::Reasoning { text } => {
                output.push(json!({
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": text}],
                }));
            }
            ContentPart::Image { .. } => {
                // Images in output are rare; skip for now
            }
        }
    }

    // Generate a Responses-style ID from the Chat Completions ID
    let resp_id = if cr.id.starts_with("chatcmpl-") {
        cr.id.replacen("chatcmpl-", "resp_", 1)
    } else if cr.id.is_empty() {
        format!("resp_{}", uuid_simple())
    } else {
        cr.id.clone()
    };

    json!({
        "id": resp_id,
        "object": "response",
        "model": cr.model,
        "status": match cr.status {
            ResponseStatus::Completed => "completed",
            ResponseStatus::Incomplete => "incomplete",
            ResponseStatus::Error => "failed",
        },
        "output": output,
        "usage": {
            "input_tokens": cr.usage.prompt_tokens,
            "output_tokens": cr.usage.completion_tokens,
            "total_tokens": cr.usage.total_tokens,
        },
    })
}

/// Generate a simple UUID-like string without pulling in the uuid crate.
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{ts:016x}")
}

// ── Streaming ──────────────────────────────────────────────────────────

use super::streaming::StreamEvent;

/// Convert Chat Completions SSE delta → canonical StreamEvent.
pub fn stream_event_from_chat(data: &str) -> Option<StreamEvent> {
    let v: serde_json::Value = serde_json::from_str(data).ok()?;
    let choices = v["choices"].as_array()?;
    let choice = choices.first()?;
    let delta = &choice["delta"];

    // Text delta
    if let Some(content) = delta["content"].as_str() {
        return Some(StreamEvent::TextDelta {
            text: content.to_string(),
        });
    }

    // Tool call delta
    if let Some(tool_calls) = delta["tool_calls"].as_array()
        && let Some(tc) = tool_calls.iter().next() {
            let index = tc["index"].as_u64().unwrap_or(0) as usize;
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().map(|s| s.to_string());
            let args = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return Some(StreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta: args,
            });
        }

    // Finish
    if let Some(reason) = choice["finish_reason"].as_str() {
        let usage = Usage {
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["completion_tokens"]
                .as_u64()
                .unwrap_or(0) as u32,
            total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        };
        return Some(StreamEvent::Done {
            usage,
            finish_reason: reason.to_string(),
        });
    }

    // Error
    if let Some(err) = v["error"].as_object() {
        return Some(StreamEvent::Error {
            message: err["message"].as_str().unwrap_or("").to_string(),
            code: err["code"].as_str().map(|s| s.to_string()),
        });
    }

    None
}

/// Convert canonical StreamEvent → Responses API SSE format.
pub fn stream_event_to_responses(event: &StreamEvent) -> String {
    use serde_json::json;

    match event {
        StreamEvent::TextDelta { text } => {
            json!({
                "type": "response.output_text.delta",
                "delta": text,
            })
            .to_string()
        }
        StreamEvent::ToolCallDelta {
            index: _,
            id,
            name: _,
            arguments_delta,
        } => {
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": id,
                "delta": arguments_delta,
            })
            .to_string()
        }
        StreamEvent::Done { usage, finish_reason: _ } => {
            json!({
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                },
            })
            .to_string()
        }
        StreamEvent::Error { message, code } => {
            json!({
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": code.as_deref().unwrap_or("unknown"),
                    "message": message,
                },
            })
            .to_string()
        }
        _ => String::new(),
    }
}
