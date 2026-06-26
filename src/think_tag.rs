use crate::protocol::StreamEvent;

#[derive(Debug, Clone, PartialEq)]
pub enum ThinkTagState {
    Idle,
    InThinkTag,
    InAnswer,
}

#[derive(Debug, Clone)]
pub struct ThinkTagResult {
    pub reasoning_text: String,
    pub complete: bool,
}

/// Streaming-aware `<think>...</think>` tag parser.
///
/// Maintains a rolling text buffer to detect tags split across chunks.
/// When text might contain a partial tag at the end, it's held in `pending`
/// until the next chunk resolves it.
#[derive(Debug, Clone)]
pub struct ThinkTagParser {
    state: ThinkTagState,
    reasoning_buffer: String,
    answer_buffer: String,
    /// Text held back because it might be a partial `<think>` or `</think>`.
    pending: String,
    consumed: bool,
}

impl ThinkTagParser {
    pub fn new() -> Self {
        Self {
            state: ThinkTagState::Idle,
            reasoning_buffer: String::new(),
            answer_buffer: String::new(),
            pending: String::new(),
            consumed: false,
        }
    }

    /// Process a text chunk. Returns normalized events.
    /// Should only be called when no structured reasoning has been received.
    pub fn feed_text(&mut self, text: &str) -> Vec<StreamEvent> {
        if self.consumed {
            return vec![];
        }

        // Combine with any pending text from the previous chunk
        let combined = if self.pending.is_empty() {
            text.to_string()
        } else {
            let mut buf = std::mem::take(&mut self.pending);
            buf.push_str(text);
            buf
        };

        // Check if the combined text ends with a potential partial tag
        // that shouldn't be emitted until the next chunk resolves it.
        let (safe, held) = Self::split_partial_tag(&combined);

        // Push held text back to pending
        self.pending = held;

        self.process_text(&safe)
    }

    /// Split text at the last possible partial tag boundary.
    /// Returns (safe_to_emit, held_back).
    fn split_partial_tag(text: &str) -> (String, String) {
        let max_tag_len = 7; // "<think>" or "</think>" max prefix
        if text.len() < 2 {
            // Too short to contain a partial tag — hold if non-empty
            if text.is_empty() {
                (String::new(), String::new())
            } else {
                (String::new(), text.to_string())
            }
        } else {
            // Check the tail for potential partial tag opener/closer
            let tail = &text[text.len().saturating_sub(max_tag_len)..];
            for i in (1..=max_tag_len).rev() {
                if tail.len() < i { continue; }
                let suffix = &tail[tail.len() - i..];
                // Could this be the start of <think> or </think>?
                if "<think>".starts_with(suffix) || "</think>".starts_with(suffix) || "think>".starts_with(suffix) {
                    let split = text.len() - i;
                    return (text[..split].to_string(), text[split..].to_string());
                }
            }
            // Also check for bare `<` that could start a tag
            if text.ends_with('<') {
                return (text[..text.len() - 1].to_string(), "<".to_string());
            }
            (text.to_string(), String::new())
        }
    }

    fn process_text(&mut self, text: &str) -> Vec<StreamEvent> {
        if text.is_empty() {
            return vec![];
        }
        let mut events = Vec::new();
        let mut remaining = text;

        loop {
            match self.state {
                ThinkTagState::Idle => {
                    if let Some(pos) = remaining.find("<think>") {
                        let before = &remaining[..pos];
                        if !before.is_empty() {
                            events.push(StreamEvent::TextDelta { text: before.to_string() });
                            self.answer_buffer.push_str(before);
                        }
                        remaining = &remaining[pos + 7..];
                        self.state = ThinkTagState::InThinkTag;
                    } else {
                        events.push(StreamEvent::TextDelta { text: remaining.to_string() });
                        self.answer_buffer.push_str(remaining);
                        break;
                    }
                }
                ThinkTagState::InThinkTag => {
                    if let Some(pos) = remaining.find("</think>") {
                        let reasoning = &remaining[..pos];
                        if !reasoning.is_empty() {
                            events.push(StreamEvent::ReasoningDelta { text: reasoning.to_string() });
                            self.reasoning_buffer.push_str(reasoning);
                        }
                        remaining = &remaining[pos + 8..];
                        self.state = ThinkTagState::InAnswer;
                    } else {
                        self.reasoning_buffer.push_str(remaining);
                        events.push(StreamEvent::ReasoningDelta { text: remaining.to_string() });
                        break;
                    }
                }
                ThinkTagState::InAnswer => {
                    events.push(StreamEvent::TextDelta { text: remaining.to_string() });
                    self.answer_buffer.push_str(remaining);
                    break;
                }
            }
        }
        events
    }

    /// Called when stream ends. Flushes any pending text.
    pub fn finish(&mut self) -> ThinkTagResult {
        self.consumed = true;
        // Flush pending text
        let pending_text = std::mem::take(&mut self.pending);
        if !pending_text.is_empty() {
            self.process_text(&pending_text);
        }
        let complete = matches!(self.state, ThinkTagState::InAnswer | ThinkTagState::Idle);
        // Emit trailing text after unclosed think tag
        if !pending_text.is_empty() {
            self.answer_buffer.push_str(&pending_text);
        }
        ThinkTagResult {
            reasoning_text: self.reasoning_buffer.clone(),
            complete,
        }
    }

    pub fn reset(&mut self) {
        self.state = ThinkTagState::Idle;
        self.reasoning_buffer.clear();
        self.answer_buffer.clear();
        self.pending.clear();
        self.consumed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_think_tag() {
        let mut p = ThinkTagParser::new();
        let events = p.feed_text("hello world");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::TextDelta { ref text } if text == "hello world"));
        let result = p.finish();
        assert!(result.complete);
        assert!(result.reasoning_text.is_empty());
    }

    #[test]
    fn test_full_think_tag_single_chunk() {
        let mut p = ThinkTagParser::new();
        let events = p.feed_text("hello<think>deep thought</think> world");
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], StreamEvent::TextDelta { ref text } if text == "hello"));
        assert!(matches!(events[1], StreamEvent::ReasoningDelta { ref text } if text == "deep thought"));
        assert!(matches!(events[2], StreamEvent::TextDelta { ref text } if text == " world"));
        let result = p.finish();
        assert!(result.complete);
    }

    #[test]
    fn test_opening_tag_split_across_chunks() {
        let mut p = ThinkTagParser::new();
        // "hello<thi" — "hello" is safe, "<thi" might be partial <think>, held
        let e1 = p.feed_text("hello<thi");
        assert_eq!(e1.len(), 1);
        assert!(matches!(e1[0], StreamEvent::TextDelta { ref text } if text == "hello"));
        // "nk>deep thought</think> done" — completes <think>, process normally
        let e2 = p.feed_text("nk>deep thought</think> done");
        assert_eq!(e2.len(), 2);
        assert!(matches!(e2[0], StreamEvent::ReasoningDelta { ref text } if text == "deep thought"));
        assert!(matches!(e2[1], StreamEvent::TextDelta { ref text } if text == " done"));
        let result = p.finish();
        assert!(result.complete);
    }

    #[test]
    fn test_closing_tag_split_across_chunks() {
        let mut p = ThinkTagParser::new();
        // "<think>deep tho" — "<think>" triggers reasoning, "deep tho" is reasoning
        // but "tho" might be part of "thought" — hold it? No, pending only holds
        // potential partial tags, not arbitrary text.
        let e1 = p.feed_text("<think>deep tho");
        assert_eq!(e1.len(), 1);
        assert!(matches!(e1[0], StreamEvent::ReasoningDelta { ref text } if text == "deep tho"));
        let e2 = p.feed_text("ught</think> done");
        assert_eq!(e2.len(), 2);
        assert!(matches!(e2[1], StreamEvent::TextDelta { ref text } if text == " done"));
        let result = p.finish();
        assert!(result.complete);
    }

    #[test]
    fn test_incomplete_think_block() {
        let mut p = ThinkTagParser::new();
        p.feed_text("<think>incomplete reasoning");
        let result = p.finish();
        assert!(!result.complete);
        assert_eq!(result.reasoning_text, "incomplete reasoning");
    }

    #[test]
    fn test_mid_content_think_gets_stuck_in_incomplete() {
        let mut p = ThinkTagParser::new();
        // "<think>" triggers — middle or not, parser treats it as think tag
        let events = p.feed_text("some code: <think>");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::TextDelta { ref text } if text == "some code: "));
        let result = p.finish();
        assert!(!result.complete);
    }

    #[test]
    fn test_reset() {
        let mut p = ThinkTagParser::new();
        p.feed_text("<think>r1</think>a1");
        p.reset();
        let e = p.feed_text("no tag");
        assert!(matches!(e[0], StreamEvent::TextDelta { .. }));
    }

    #[test]
    fn test_pending_text_flushed_on_finish() {
        let mut p = ThinkTagParser::new();
        // Text ending with possible partial tag
        let e1 = p.feed_text("hello<");
        assert_eq!(e1.len(), 1);
        assert!(matches!(e1[0], StreamEvent::TextDelta { ref text } if text == "hello"));
        // finish flushes "<"
        let result = p.finish();
        assert!(result.complete);
    }
}
