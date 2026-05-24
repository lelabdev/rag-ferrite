/// Recursive character text splitter with overlap (UTF-8 safe)
///
/// chunk_size: ~1000 chars (~250 tokens, optimal per RAG Cookbook)
/// overlap: 10% for context preservation
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<Chunk> {
    let separators = ["\n\n", "\n", ". ", " "];
    let overlap = (chunk_size as f64 * 0.1) as usize;

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let char_count = chars.len();

    let mut char_pos: usize = 0; // position in chars, not bytes

    while char_pos < char_count {
        let target_end = (char_pos + chunk_size).min(char_count);

        // Find best split in char range
        let split = if target_end < char_count {
            let slice: String = chars[char_pos..target_end].iter().collect();
            find_best_split_char(&slice, &separators)
        } else {
            target_end - char_pos
        };

        let char_end = char_pos + split;
        let content: String = chars[char_pos..char_end].iter().collect();
        let trimmed = content.trim().to_string();

        if !trimmed.is_empty() {
            // Calculate byte positions for start/end
            let byte_start: usize = chars[..char_pos].iter().collect::<String>().len();
            let byte_end: usize = chars[..char_end].iter().collect::<String>().len();

            // Detect content type from the chunk text
            let is_first = chunks.is_empty();
            let chunk_type = detect_chunk_type(&trimmed, is_first);

            chunks.push(Chunk {
                content: trimmed,
                index: chunks.len() as i32,
                start_pos: byte_start as i32,
                end_pos: byte_end as i32,
                chunk_type,
            });
        }

        // Advance with overlap
        let next = if char_end > overlap { char_end - overlap } else { char_end };
        if next <= char_pos { break; }
        char_pos = next;
    }

    // Merge last chunk with previous if it's too short (< 200 chars)
    if chunks.len() > 1 {
        let last_idx = chunks.len() - 1;
        if chunks[last_idx].content.len() < 200 {
            let last_end = chunks[last_idx].end_pos;
            let last_content = chunks.pop().unwrap();
            let prev = chunks.last_mut().unwrap();
            prev.content = format!("{}\n\n{}", prev.content, last_content.content);
            prev.end_pos = last_end;
        }
    }

    chunks
}

fn find_best_split_char(text: &str, separators: &[&str]) -> usize {
    for sep in separators {
        if let Some(p) = text.rfind(sep) {
            if p > text.len() * 3 / 10 {
                return text[..p].chars().count();
            }
        }
    }
    text.chars().count().saturating_sub(1)
}

/// Detect the content type of a chunk based on its text.
///
/// Detection priority (first match wins):
///   1. Code  — fenced code blocks or heavily indented lines
///   2. Table — pipe-delimited rows with separators
///   3. Heading — starts with #
///   4. List  — starts with list markers
///   5. ImageRef — contains markdown image syntax
///   6. Title — first chunk of the document
///   7. Text  — default fallback
pub fn detect_chunk_type(text: &str, is_first_chunk: bool) -> ChunkType {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return ChunkType::Text;
    }

    // 1. Code: fenced code blocks (```...```) or majority of lines indented 4+ spaces
    if detect_code(&lines) {
        return ChunkType::Code;
    }

    // 2. Table: contains | with | separator lines (|---|---|)
    if detect_table(&lines) {
        return ChunkType::Table;
    }

    // 3. Heading: first non-empty line starts with #
    if let Some(first) = lines.iter().find(|l| !l.is_empty()) {
        if first.starts_with('#') {
            return ChunkType::Heading;
        }
    }

    // 4. List: majority of non-empty lines start with list markers
    if detect_list(&lines) {
        return ChunkType::List;
    }

    // 5. ImageRef: contains markdown image syntax ![...](...)
    if text.contains("![") && text.contains("](") {
        return ChunkType::ImageRef;
    }

    // 6. Title: first chunk of the document
    if is_first_chunk {
        return ChunkType::Title;
    }

    // 7. Text: default
    ChunkType::Text
}

fn detect_code(lines: &[&str]) -> bool {
    // Check for fenced code blocks
    let fence_count = lines.iter().filter(|l| l.trim().starts_with("```")).count();
    if fence_count >= 2 {
        return true;
    }
    // If there's an opening fence and it's the last line or not closed,
    // still count it (chunk may split a code block)
    if fence_count >= 1 {
        return true;
    }

    // Check for majority of lines indented 4+ spaces (code-like)
    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.is_empty()).collect();
    if non_empty.len() >= 3 {
        let indented = non_empty.iter().filter(|l| l.starts_with("    ") || l.starts_with("\t")).count();
        if indented * 2 >= non_empty.len() {
            return true;
        }
    }

    false
}

fn detect_table(lines: &[&str]) -> bool {
    // Need at least one line with | and a separator line like |---|---|
    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.is_empty()).collect();
    if non_empty.len() < 2 {
        return false;
    }

    let pipe_lines = non_empty.iter().filter(|l| l.contains('|')).count();
    let has_separator = non_empty.iter().any(|l| {
        let trimmed = l.trim();
        trimmed.contains('|') && trimmed.contains('-') && trimmed.chars().all(|c| c == '|' || c == '-' || c == ' ' || c == ':')
    });

    pipe_lines >= 2 && has_separator
}

fn detect_list(lines: &[&str]) -> bool {
    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.is_empty()).collect();
    if non_empty.is_empty() {
        return false;
    }

    let list_lines = non_empty.iter().filter(|l| {
        let trimmed = l.trim();
        // Unordered: - or * followed by space
        (trimmed.starts_with("- ") || trimmed.starts_with("* "))
        // Ordered: 1. 2. etc.
        || trimmed.starts_with(|c: char| c.is_ascii_digit()) && trimmed.contains('.') && trimmed.find('.').map_or(false, |pos| pos <= 3)
    }).count();

    // Majority of lines are list items
    list_lines * 2 > non_empty.len()
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub index: i32,
    pub start_pos: i32,
    pub end_pos: i32,
    pub chunk_type: ChunkType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Text,
    Code,
    Title,
    Heading,
    List,
    Table,
    ImageRef,
}

impl ChunkType {
    /// Convert to the string stored in the DB chunk_type column.
    /// Values are lowercase to match rag_engine convention.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkType::Text => "text",
            ChunkType::Code => "code",
            ChunkType::Title => "title",
            ChunkType::Heading => "heading",
            ChunkType::List => "list",
            ChunkType::Table => "table",
            ChunkType::ImageRef => "image_ref",
        }
    }
}
