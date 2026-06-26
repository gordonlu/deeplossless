//! OpenAI Chat Completions ↔ canonical IR adapter.

use super::canonical::*;

pub fn request_from_chat(body: &serde_json::Value) -> CanonicalRequest {
    let model = body["model"].as_str().unwrap_or("").to_string();
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_tokens"].as_u64().map(|n| n as u32)
        .or_else(|| body["max_completion_tokens"].as_u64().map(|n| n as u32));
    let temperature = body["temperature"].as_f64();

    let tools: Vec<ToolDef> = body["tools"].as_array()
        .map(|arr| arr.iter().filter_map(|t| {
            let func = &t["function"];
            Some(ToolDef {
                name: func["name"].as_str()?.to_string(),
                description: func["description"].as_str().unwrap_or("").to_string(),
                parameters: func["parameters"].clone(),
                strict: t["strict"].as_bool().unwrap_or(false),
            })
        }).collect())
        .unwrap_or_default();

    let mut instructions = Vec::new();
    let messages: Vec<Message> = body["messages"].as_array()
        .map(|arr| arr.iter().map(|msg| {
            let role_str = msg["role"].as_str().unwrap_or("user");
            let role = match role_str {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            };
            let mut parts = Vec::new();
            let content = &msg["content"];
            if let Some(s) = content.as_str() {
                if !s.is_empty() { parts.push(ContentPart::Text { text: s.to_string() }); }
            } else if let Some(arr) = content.as_array() {
                for block in arr {
                    let bt = block["type"].as_str().unwrap_or("");
                    if bt == "text" || bt == "input_text" {
                        if let Some(t) = block["text"].as_str() {
                            parts.push(ContentPart::Text { text: t.to_string() });
                        }
                    } else if bt == "image_url"
                        && let Some(url) = block["image_url"]["url"].as_str() {
                        parts.push(ContentPart::Image {
                            source_type: "url".into(), data: url.to_string(),
                            detail: block["image_url"]["detail"].as_str().unwrap_or("auto").into(),
                        });
                    }
                }
            }
            // System → instructions (preserves provenance, doesn't clear)
            if role == Role::System {
                if let Some(ContentPart::Text { text }) = parts.first() {
                    instructions.push(InstructionBlock { text: text.clone(), meta: None });
                }
                parts.clear();
            }
            let mut meta = None;
            if let Some(tool_calls) = msg["tool_calls"].as_array() {
                let tcs: Vec<ToolInvocation> = tool_calls.iter().map(|tc| ToolInvocation {
                    id: tc["id"].as_str().unwrap_or("").to_string(),
                    name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                    arguments: tc["function"]["arguments"].as_str()
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or(serde_json::json!({})),
                }).collect();
                for tc in &tcs {
                    parts.push(ContentPart::ToolCall { id: tc.id.clone(), name: tc.name.clone(), arguments: tc.arguments.clone() });
                }
                meta = Some(MessageMeta { tool_call_id: None, tool_calls: tcs });
            }
            if role == Role::Tool {
                let tc_id = msg["tool_call_id"].as_str().map(|s| s.to_string());
                if let Some(ContentPart::Text { text }) = parts.first() {
                    parts[0] = ContentPart::ToolResult { call_id: tc_id.clone().unwrap_or_default(), content: text.clone() };
                }
                meta = Some(MessageMeta { tool_call_id: tc_id, tool_calls: vec![] });
            }
            Message { role, parts, meta, reasoning: None }
        }).collect())
        .unwrap_or_default();

    let response_format = body["response_format"].as_object().map(|rf| ResponseFormat {
        format_type: rf["type"].as_str().unwrap_or("text").to_string(),
        json_schema: rf.get("json_schema").cloned(),
    });

    CanonicalRequest {
        instructions, messages, tools, model, stream, max_tokens, temperature,
        response_format,
        provider: ProviderKind::OpenAI,
        capabilities: ProviderCapabilities { structured_output: StructuredOutputMode::JsonSchema, ..Default::default() },
        reasoning_effort: None,
        deepseek_native: Default::default(),
    }
}

pub fn request_to_chat(req: &CanonicalRequest) -> serde_json::Value {
    use serde_json::json;
    let mut msgs: Vec<serde_json::Value> = Vec::new();
    // Merge all instruction blocks into a single system message to avoid
    // system-message proliferation (which causes attention dilution and
    // token waste with stateless providers like DeepSeek).
    let system_text: Vec<&str> = req.instructions.iter().map(|i| i.text.as_str()).collect();
    if !system_text.is_empty() {
        msgs.push(json!({"role": "system", "content": system_text.join("\n\n")}));
    }
    for msg in &req.messages {
        let role = match msg.role {
            Role::System | Role::Developer => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let text: Vec<String> = msg.parts.iter().filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.clone()),
            ContentPart::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        }).collect();
        let content = text.join("\n");
        let tool_calls: Vec<serde_json::Value> = msg.parts.iter().filter_map(|p| {
            if let ContentPart::ToolCall { id, name, arguments } = p {
                Some(json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments.to_string()}}))
            } else { None }
        }).collect();
        let mut m = json!({"role": role});
        // Omit content when empty for assistant with tool_calls (matches API spec)
        if !content.is_empty() || tool_calls.is_empty() {
            m["content"] = json!(content);
        }
        if !tool_calls.is_empty() { m["tool_calls"] = json!(tool_calls); }
        if role == "tool"
            && let Some(ref meta) = msg.meta
                && let Some(ref tc_id) = meta.tool_call_id { m["tool_call_id"] = json!(tc_id); }
        if role == "assistant"
            && let Some(ref r) = msg.reasoning {
                m["reasoning_content"] = json!(r.text);
            }
        msgs.push(m);
    }
    let mut body = json!({"model": req.model, "messages": msgs, "stream": req.stream});
    if let Some(mt) = req.max_tokens { body["max_tokens"] = json!(mt); }
    if let Some(t) = req.temperature { body["temperature"] = json!(t); }
    if !req.tools.is_empty() {
        body["tools"] = json!(req.tools.iter().map(|t| json!({
            "type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}, "strict": t.strict
        })).collect::<Vec<_>>());
    }
    if let Some(ref rf) = req.response_format {
        body["response_format"] = json!({"type": rf.format_type, "json_schema": rf.json_schema});
    }
    // DeepSeek-V4 native options
    match req.deepseek_native.reasoning_effort {
        ReasoningEffortMode::Override(ReasoningEffort::High) => { body["reasoning_effort"] = json!("high"); }
        ReasoningEffortMode::Override(ReasoningEffort::Max) => { body["reasoning_effort"] = json!("max"); }
        ReasoningEffortMode::Override(ReasoningEffort::None) => { body["reasoning_effort"] = json!("none"); }
        ReasoningEffortMode::Passthrough => {}
    }
    if req.deepseek_native.dsml_parse { body["dsml_parse"] = json!(true); }
    if req.deepseek_native.dsml_emit { body["dsml_emit"] = json!(true); }
    if req.deepseek_native.quick_instruction { body["quick_instruction"] = json!(true); }
    body
}

pub fn response_from_chat(body: &serde_json::Value) -> CanonicalResponse {
    use serde_json::json;
    let choice = &body["choices"][0];
    let msg = &choice["message"];
    let mut output = Vec::new();
    if let Some(content) = msg["content"].as_str()
        && !content.is_empty() { output.push(ContentPart::Text { text: content.to_string() }); }
    if let Some(tool_calls) = msg["tool_calls"].as_array() {
        for tc in tool_calls {
            let args = tc["function"]["arguments"].as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::json!({}));
            output.push(ContentPart::ToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: args,
            });
        }
    }
    let reasoning_trace = msg["reasoning_content"].as_str().map(|text| ReasoningTrace {
        text: text.to_string(),
        summarized: false,
        tokens: None,
    });
    let finish = choice["finish_reason"].as_str().unwrap_or("stop");
    CanonicalResponse {
        id: body["id"].as_str().unwrap_or("").to_string(),
        model: body["model"].as_str().unwrap_or("").to_string(),
        status: match finish { "stop" => ResponseStatus::Completed, _ => ResponseStatus::Incomplete },
        output,
        reasoning_trace,
        usage: Usage {
            prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        },
            provider_meta: Some(json!({"finish_reason": finish})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_to_chat_reasoning_effort_passthrough() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { reasoning_effort: ReasoningEffortMode::Passthrough, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert!(body.get("reasoning_effort").is_none(), "Passthrough should omit reasoning_effort");
    }

    #[test]
    fn request_to_chat_reasoning_effort_high() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities {
                reasoning_effort: ReasoningEffortMode::Override(ReasoningEffort::High),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn request_to_chat_reasoning_effort_max() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities {
                reasoning_effort: ReasoningEffortMode::Override(ReasoningEffort::Max),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["reasoning_effort"], "max");
    }

    #[test]
    fn request_to_chat_reasoning_effort_none() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities {
                reasoning_effort: ReasoningEffortMode::Override(ReasoningEffort::None),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["reasoning_effort"], "none");
    }

    #[test]
    fn request_to_chat_dsml_parse_true() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { dsml_parse: true, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["dsml_parse"], true);
    }

    #[test]
    fn request_to_chat_dsml_parse_false() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { dsml_parse: false, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert!(body.get("dsml_parse").is_none(), "false dsml_parse should be omitted");
    }

    #[test]
    fn request_to_chat_dsml_emit_true() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { dsml_emit: true, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["dsml_emit"], true);
    }

    #[test]
    fn request_to_chat_dsml_emit_false() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { dsml_emit: false, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert!(body.get("dsml_emit").is_none(), "false dsml_emit should be omitted");
    }

    #[test]
    fn request_to_chat_quick_instruction_true() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { quick_instruction: true, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["quick_instruction"], true);
    }

    #[test]
    fn request_to_chat_quick_instruction_false() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            deepseek_native: DeepSeekNativeCapabilities { quick_instruction: false, ..Default::default() },
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert!(body.get("quick_instruction").is_none(), "false quick_instruction should be omitted");
    }

    #[test]
    fn request_to_chat_keeps_existing_fields() {
        let req = CanonicalRequest {
            model: "deepseek-v4".into(),
            stream: true,
            max_tokens: Some(4096),
            temperature: Some(0.7),
            ..Default::default()
        };
        let body = request_to_chat(&req);
        assert_eq!(body["model"], "deepseek-v4");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["temperature"], 0.7);
    }
}
