use crate::protocol::canonical::ToolInvocation;

#[derive(Debug, Clone)]
pub struct DsmlDocument {
    pub invokes: Vec<DsmlInvoke>,
}

#[derive(Debug, Clone)]
pub struct DsmlInvoke {
    pub name: String,
    pub parameters: Vec<DsmlParameter>,
}

#[derive(Debug, Clone)]
pub struct DsmlParameter {
    pub name: String,
    pub value: DsmlValue,
    pub raw_string: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DsmlValue {
    String(String),
    Number(serde_json::Number),
    Bool(bool),
    Null,
    Array(Vec<DsmlValue>),
    Object(std::collections::HashMap<String, DsmlValue>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DsmlError {
    MalformedSyntax { position: usize, detail: String },
    UnclosedTag { tag: String },
    InvalidParameterType { name: String, raw: String },
    NestedTooDeep,
}

impl std::fmt::Display for DsmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsmlError::MalformedSyntax { position, detail } => write!(f, "malformed DSML at position {position}: {detail}"),
            DsmlError::UnclosedTag { tag } => write!(f, "unclosed DSML tag: {tag}"),
            DsmlError::InvalidParameterType { name, raw } => write!(f, "invalid parameter type for '{name}': {raw}"),
            DsmlError::NestedTooDeep => write!(f, "DSML nesting too deep"),
        }
    }
}

/// Parse a complete DSML tool_calls block from assistant content.
pub fn parse_dsml_tool_calls(raw: &str) -> Result<Vec<ToolInvocation>, DsmlError> {
    let start_tag = "<|DSML|tool_calls|>";
    let start = raw.find(start_tag).ok_or(DsmlError::MalformedSyntax {
        position: 0, detail: "missing <|DSML|tool_calls|>".into(),
    })?;
    let content_after_start = &raw[start + start_tag.len()..];

    let end_tag = "</|DSML|tool_calls|>";
    let end = content_after_start.find(end_tag).ok_or(DsmlError::UnclosedTag {
        tag: "</|DSML|tool_calls|>".into(),
    })?;
    let body = &content_after_start[..end];

    let doc = parse_dsml_body(body)?;
    Ok(doc.invocations_to_canonical())
}

fn parse_dsml_body(body: &str) -> Result<DsmlDocument, DsmlError> {
    let mut invokes = Vec::new();
    let mut remaining = body.trim();
    while !remaining.is_empty() {
        remaining = remaining.trim();
        if remaining.starts_with("<|DSML|invoke|") {
            let invoke_end = remaining.find("</|DSML|invoke|>")
                .ok_or(DsmlError::UnclosedTag { tag: "</|DSML|invoke|>".into() })?;
            let invoke_body = &remaining[..invoke_end + "</|DSML|invoke|>".len()];
            let invoke = parse_invoke(invoke_body)?;
            invokes.push(invoke);
            remaining = &remaining[invoke_end + "</|DSML|invoke|>".len()..];
        } else {
            if let Some(next_invoke) = remaining.find("<|DSML|invoke|") {
                remaining = &remaining[next_invoke..];
            } else {
                break;
            }
        }
    }
    Ok(DsmlDocument { invokes })
}

fn parse_invoke(raw: &str) -> Result<DsmlInvoke, DsmlError> {
    let name = extract_attr(raw, "name").ok_or(DsmlError::MalformedSyntax {
        position: 0, detail: "invoke missing name attribute".into(),
    })?;
    let mut parameters = Vec::new();
    let mut remaining = raw;
    loop {
        if let Some(param_start) = remaining.find("<|DSML|parameter|") {
            remaining = &remaining[param_start + "<|DSML|parameter|".len()..];
            let param_end = remaining.find("</|DSML|parameter|>")
                .ok_or(DsmlError::UnclosedTag { tag: "</|DSML|parameter|>".into() })?;
            let param_raw = &remaining[..param_end + "</|DSML|parameter|>".len()];
            let param = parse_parameter(param_raw)?;
            parameters.push(param);
            remaining = &remaining[param_end + "</|DSML|parameter|>".len()..];
        } else {
            break;
        }
    }
    Ok(DsmlInvoke { name, parameters })
}

fn parse_parameter(raw: &str) -> Result<DsmlParameter, DsmlError> {
    let name = extract_attr(raw, "name").ok_or(DsmlError::MalformedSyntax {
        position: 0, detail: "parameter missing name attribute".into(),
    })?;
    let is_string = match extract_attr(raw, "string") {
        Some(v) => v == "true",
        None => true,
    };
    let value_start = raw.find('>').ok_or(DsmlError::MalformedSyntax {
        position: 0, detail: "parameter missing value".into(),
    })? + 1;
    let value_end = raw.rfind("</|DSML|parameter|>").unwrap_or(raw.len());
    let raw_value = raw[value_start..value_end].trim().to_string();
    let value = if is_string {
        DsmlValue::String(raw_value.clone())
    } else {
        parse_json_value(&raw_value)?
    };
    Ok(DsmlParameter { name, value, raw_string: raw_value })
}

fn extract_attr(raw: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    if let Some(start) = raw.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = raw[value_start..].find('"') {
            return Some(raw[value_start..value_start + end].to_string());
        }
    }
    None
}

fn parse_json_value(raw: &str) -> Result<DsmlValue, DsmlError> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| DsmlError::MalformedSyntax {
        position: 0, detail: format!("JSON parse error: {e}"),
    })?;
    Ok(json_to_dsml(v))
}

fn json_to_dsml(v: serde_json::Value) -> DsmlValue {
    match v {
        serde_json::Value::String(s) => DsmlValue::String(s),
        serde_json::Value::Number(n) => DsmlValue::Number(n),
        serde_json::Value::Bool(b) => DsmlValue::Bool(b),
        serde_json::Value::Null => DsmlValue::Null,
        serde_json::Value::Array(arr) => DsmlValue::Array(arr.into_iter().map(json_to_dsml).collect()),
        serde_json::Value::Object(obj) => DsmlValue::Object(obj.into_iter().map(|(k, v)| (k, json_to_dsml(v))).collect()),
    }
}

impl DsmlDocument {
    fn invocations_to_canonical(&self) -> Vec<ToolInvocation> {
        self.invokes.iter().enumerate().map(|(i, invoke)| {
            let args = dsml_params_to_json(&invoke.parameters);
            ToolInvocation {
                id: format!("dsml_{}", i),
                name: invoke.name.clone(),
                arguments: args,
            }
        }).collect()
    }
}

fn dsml_params_to_json(params: &[DsmlParameter]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for p in params {
        map.insert(p.name.clone(), dsml_value_to_json(&p.value));
    }
    serde_json::Value::Object(map)
}

fn dsml_value_to_json(v: &DsmlValue) -> serde_json::Value {
    match v {
        DsmlValue::String(s) => serde_json::Value::String(s.clone()),
        DsmlValue::Number(n) => serde_json::Value::Number(n.clone()),
        DsmlValue::Bool(b) => serde_json::Value::Bool(*b),
        DsmlValue::Null => serde_json::Value::Null,
        DsmlValue::Array(arr) => serde_json::Value::Array(arr.iter().map(dsml_value_to_json).collect()),
        DsmlValue::Object(obj) => {
            let map: serde_json::Map<_, _> = obj.iter().map(|(k, v)| (k.clone(), dsml_value_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
    }
}

/// Optional: emit DSML from canonical tool calls (test/debug only, not in main path).
#[allow(dead_code)]
pub fn emit_dsml_tool_calls(tool_calls: &[ToolInvocation]) -> String {
    let mut out = String::from("<|DSML|tool_calls|\n");
    for tc in tool_calls {
        out.push_str(&format!("<|DSML|invoke| name=\"{}\">\n", tc.name));
        if let serde_json::Value::Object(map) = &tc.arguments {
            for (key, val) in map {
                match val {
                    serde_json::Value::String(s) => {
                        out.push_str(&format!("<|DSML|parameter| name=\"{key}\" string=\"true\">{s}</|DSML|parameter|>\n"));
                    }
                    _ => {
                        out.push_str(&format!("<|DSML|parameter| name=\"{key}\" string=\"false\">{val}</|DSML|parameter|>\n"));
                    }
                }
            }
        }
        out.push_str("</|DSML|invoke|>\n");
    }
    out.push_str("</|DSML|tool_calls|>");
    out
}

#[derive(Debug, Default)]
pub struct StreamingDsmlParser {
    buffer: String,
}

impl StreamingDsmlParser {
    pub fn new() -> Self { Self { buffer: String::new() } }

    /// Feed a text chunk. Returns complete ToolInvocation vectors when full
    /// <|DSML|tool_calls|>...</|DSML|tool_calls|> documents close.
    pub fn feed(&mut self, text: &str) -> Result<Vec<ToolInvocation>, DsmlError> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        self.buffer.push_str(text);
        let mut results = Vec::new();
        loop {
            if let Some(start) = self.buffer.find("<|DSML|tool_calls|>") {
                let after_start = &self.buffer[start..];
                if let Some(end) = after_start.find("</|DSML|tool_calls|>") {
                    let full_doc = &after_start[..end + "</|DSML|tool_calls|>".len()];
                    let tool_calls = parse_dsml_tool_calls(full_doc)?;
                    results.extend(tool_calls);
                    let consumed = start + end + "</|DSML|tool_calls|>".len();
                    self.buffer = self.buffer[consumed..].to_string();
                } else {
                    break;
                }
            } else {
                if self.buffer.ends_with("<|DSML") || self.buffer.ends_with("<|DSML|") || self.buffer.ends_with("<|DSML|tool_calls") {
                } else {
                    self.buffer.clear();
                }
                break;
            }
        }
        Ok(results)
    }

    /// Flush remaining buffer at stream end (may return DsmlIncomplete).
    pub fn finish(&mut self) -> Result<Vec<ToolInvocation>, DsmlError> {
        if self.buffer.contains("<|DSML|") {
            Err(DsmlError::UnclosedTag { tag: "</|DSML|tool_calls|>".into() })
        } else {
            Ok(vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dsml(name: &str, param_name: &str, param_value: &str) -> String {
        format!(
            "<|DSML|tool_calls|>\n\
             <|DSML|invoke| name=\"{name}\">\n\
             <|DSML|parameter| name=\"{param_name}\" string=\"true\">{param_value}</|DSML|parameter|>\n\
             </|DSML|invoke|>\n\
             </|DSML|tool_calls|>"
        )
    }

    #[test]
    fn test_parse_simple_invoke() {
        let raw = sample_dsml("read_file", "path", "/foo/bar.txt");
        let result = parse_dsml_tool_calls(&raw).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[0].arguments["path"], "/foo/bar.txt");
    }

    #[test]
    fn test_parse_string_false_number() {
        let raw = r##"<|DSML|tool_calls|>
<|DSML|invoke| name="search">
<|DSML|parameter| name="max_results" string="false">10</|DSML|parameter|>
</|DSML|invoke|>
</|DSML|tool_calls|>"##;
        let result = parse_dsml_tool_calls(raw).unwrap();
        assert_eq!(result[0].arguments["max_results"], 10);
    }

    #[test]
    fn test_parse_string_false_bool() {
        let raw = r##"<|DSML|tool_calls|>
<|DSML|invoke| name="enable">
<|DSML|parameter| name="flag" string="false">true</|DSML|parameter|>
</|DSML|invoke|>
</|DSML|tool_calls|>"##;
        let result = parse_dsml_tool_calls(raw).unwrap();
        assert_eq!(result[0].arguments["flag"], true);
    }

    #[test]
    fn test_parse_string_false_object() {
        let raw = r##"<|DSML|tool_calls|>
<|DSML|invoke| name="update">
<|DSML|parameter| name="config" string="false">{"key":"val","count":3}</|DSML|parameter|>
</|DSML|invoke|>
</|DSML|tool_calls|>"##;
        let result = parse_dsml_tool_calls(raw).unwrap();
        assert_eq!(result[0].arguments["config"]["key"], "val");
        assert_eq!(result[0].arguments["config"]["count"], 3);
    }

    #[test]
    fn test_multiple_invokes() {
        let raw = "<|DSML|tool_calls|>\n\
                   <|DSML|invoke| name=\"read\">\n\
                   <|DSML|parameter| name=\"path\" string=\"true\">a.txt</|DSML|parameter|>\n\
                   </|DSML|invoke|>\n\
                   <|DSML|invoke| name=\"search\">\n\
                   <|DSML|parameter| name=\"q\" string=\"true\">hello</|DSML|parameter|>\n\
                   </|DSML|invoke|>\n\
                   </|DSML|tool_calls|>";
        let result = parse_dsml_tool_calls(raw).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "read");
        assert_eq!(result[1].name, "search");
    }

    #[test]
    fn test_malformed_missing_name() {
        let raw = "<|DSML|tool_calls|>\n<|DSML|invoke|>\n</|DSML|invoke|>\n</|DSML|tool_calls|>";
        assert!(parse_dsml_tool_calls(raw).is_err());
    }

    #[test]
    fn test_streaming_partial() {
        let mut parser = StreamingDsmlParser::new();
        assert!(parser.feed("<|DSML|tool_calls|>\n<|DSML|invoke| name=\"test\">\n").unwrap().is_empty());
        let results = parser.feed("<|DSML|parameter| name=\"x\" string=\"true\">val</|DSML|parameter|>\n</|DSML|invoke|>\n</|DSML|tool_calls|>").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test");
    }

    #[test]
    fn test_streaming_incomplete_at_end() {
        let mut parser = StreamingDsmlParser::new();
        parser.feed("<|DSML|tool_calls|>\n<|DSML|invoke| name=\"test\">\n").unwrap();
        let result = parser.finish();
        assert!(result.is_err());
    }
}
