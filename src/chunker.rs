/// Recursive character text splitter with overlap (UTF-8 safe)
///
/// chunk_size: ~1000 chars (~250 tokens, optimal per RAG Cookbook)
/// overlap: 10% for context preservation
/// Also tracks: markdown headers (section_path), page breaks (page), content type (chunk_type)
pub fn chunk_text(text: &str, chunk_size: usize, overlap_ratio: f64, merge_threshold: usize) -> Vec<Chunk> {
    let separators = ["\n\n", "\n", ". ", " "];
    let overlap = (chunk_size as f64 * overlap_ratio) as usize;

    // Pre-scan: build section map from markdown headers
    let sections = extract_sections(text);
    // Pre-scan: build page break positions
    let page_breaks = find_page_breaks(text);

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let char_count = chars.len();

    let mut char_pos: usize = 0; // position in chars, not bytes
    let mut current_page: u32 = page_for_position(&page_breaks, 0);

    while char_pos < char_count {
        // Update page before chunking this segment
        current_page = page_for_position(&page_breaks, char_pos);

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
            // Look up section path based on byte position of chunk start
            let section_path = find_section_for_position(&sections, byte_start);
            // Determine page
            let chunk_page = if page_breaks.is_empty() {
                None
            } else {
                Some(current_page)
            };

            chunks.push(Chunk {
                content: trimmed,
                index: chunks.len() as i32,
                start_pos: byte_start as i32,
                end_pos: byte_end as i32,
                chunk_type,
                section_path,
                page: chunk_page,
            });
        }

        // Advance with overlap
        let next = if char_end > overlap { char_end - overlap } else { char_end };
        if next <= char_pos { break; }
        char_pos = next;
    }

    // Merge last chunk with previous if it's too short
    if chunks.len() > 1 {
        let last_idx = chunks.len() - 1;
        if chunks[last_idx].content.len() < merge_threshold {
            let last_end = chunks[last_idx].end_pos;
            let last_section = chunks[last_idx].section_path.clone();
            let last_page = chunks[last_idx].page;
            let last_content = chunks.pop().unwrap();
            let prev = chunks.last_mut().unwrap();
            prev.content = format!("{}\n\n{}", prev.content, last_content.content);
            prev.end_pos = last_end;
            // Preserve section path from the last chunk if it has one
            if prev.section_path.is_none() && last_section.is_some() {
                prev.section_path = last_section;
            }
            // Keep the earlier page number
            if prev.page.is_none() && last_page.is_some() {
                prev.page = last_page;
            }
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

// === Content Type Detection ===

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
    let fence_count = lines.iter().filter(|l| l.trim().starts_with("```")).count();
    if fence_count >= 1 {
        return true;
    }

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
        (trimmed.starts_with("- ") || trimmed.starts_with("* "))
        || trimmed.starts_with(|c: char| c.is_ascii_digit()) && trimmed.contains('.') && trimmed.find('.').map_or(false, |pos| pos <= 3)
    }).count();

    list_lines * 2 > non_empty.len()
}

// === Section Path Tracking ===

pub fn extract_sections(text: &str) -> Vec<(usize, String)> {
    let mut sections: Vec<(usize, String)> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut offset: usize = 0;

    for line in text.split('\n') {
        let trimmed = line.trim_start();
        let header_level = count_hash_prefix(trimmed);

        if header_level > 0 {
            let after_hashes = &trimmed[header_level..];
            let title = after_hashes.trim_start();

            if !title.is_empty() {
                stack.retain(|(lvl, _)| *lvl < header_level);
                stack.push((header_level, title.to_string()));

                let path = stack
                    .iter()
                    .map(|(_, t)| t.as_str())
                    .collect::<Vec<_>>()
                    .join(" > ");
                sections.push((offset, path));
            }
        }

        offset += line.len();
        if offset < text.len() {
            offset += 1;
        }
    }

    sections
}

pub fn count_hash_prefix(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut count = 0;
    for &b in bytes {
        if b == b'#' {
            count += 1;
        } else if b == b' ' || b == b'\t' {
            break;
        } else {
            return 0;
        }
    }
    if count >= 1 && count <= 6 {
        count
    } else {
        0
    }
}

pub fn find_section_for_position(sections: &[(usize, String)], byte_pos: usize) -> Option<String> {
    let mut result: Option<String> = None;
    for (offset, path) in sections {
        if *offset <= byte_pos {
            result = Some(path.clone());
        } else {
            break;
        }
    }
    result
}

// === Page Tracking ===

/// Find all page break positions in the text.
/// Returns a sorted Vec of (char_position, new_page_number).
fn find_page_breaks(text: &str) -> Vec<(usize, u32)> {
    let mut breaks = Vec::new();

    // 1. Form feed characters (\f = 0x0C)
    for (i, ch) in text.char_indices() {
        if ch == '\x0C' {
            breaks.push((i, 0));
        }
    }

    // 2. Literal "--- PAGE BREAK ---"
    let marker = "--- PAGE BREAK ---";
    let mut search_start = 0;
    while let Some(pos) = text[search_start..].find(marker) {
        let abs_pos = search_start + pos;
        breaks.push((abs_pos, 0));
        search_start = abs_pos + marker.len();
    }

    // 3. Pattern "\n\n---\n\n"
    let hr = "\n\n---\n\n";
    search_start = 0;
    while let Some(pos) = text[search_start..].find(hr) {
        let abs_pos = search_start + pos;
        breaks.push((abs_pos, 0));
        search_start = abs_pos + hr.len();
    }

    // Sort and deduplicate
    breaks.sort_by_key(|(pos, _)| *pos);
    breaks.dedup_by(|a, b| (a.0 as i64 - b.0 as i64).unsigned_abs() < 10);

    // Assign page numbers
    for (i, (_, page)) in breaks.iter_mut().enumerate() {
        *page = (i as u32) + 2;
    }

    breaks
}

/// Return the page number for a given char position.
fn page_for_position(breaks: &[(usize, u32)], char_pos: usize) -> u32 {
    match breaks.binary_search_by_key(&char_pos, |(pos, _)| *pos) {
        Ok(idx) => breaks[idx].1,
        Err(idx) => {
            if idx == 0 {
                1
            } else {
                breaks[idx - 1].1
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub index: i32,
    pub start_pos: i32,
    pub end_pos: i32,
    pub chunk_type: ChunkType,
    pub section_path: Option<String>,
    pub page: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Text,
    Title,
    Code,
    Heading,
    List,
    Table,
    ImageRef,
}

impl ChunkType {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text() {
        let text = "Hello, this is a short text.";
        let chunks = chunk_text(text, 1000, 0.1, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_pos, 0);
        assert_eq!(chunks[0].end_pos, text.len() as i32);
    }

    #[test]
    fn test_long_text() {
        // Generate text long enough to produce multiple chunks
        let paragraph = "This is a paragraph with some meaningful content. ".repeat(40);
        let text = paragraph.repeat(3);
        let chunks = chunk_text(&text, 1000, 0.1, 200);
        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
        // All chunks should have non-empty content
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
        }
        // Indices should be sequential
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i as i32);
        }
    }

    #[test]
    fn test_empty_input() {
        let chunks = chunk_text("", 1000, 0.1, 200);
        assert!(chunks.is_empty(), "Empty input should produce no chunks");
    }

    #[test]
    fn test_utf8_boundary() {
        // Multi-byte characters: emojis (4 bytes each), accented chars (2 bytes)
        let chars = "éàüö ñ ç ß 🙂🎉🔥 💩";
        let text = (chars.to_string() + "\n\n").repeat(100);
        let chunks = chunk_text(&text, 500, 0.1, 200);

        assert!(chunks.len() >= 1);

        // Verify no chunk panics on re-encoding — all content strings are valid
        for chunk in &chunks {
            assert!(chunk.content.chars().all(|c| !c.is_control() || c == '\n'),
                "Unexpected control char in chunk");
        }

        // Verify start_pos and end_pos point to valid UTF-8 boundaries
        let full_bytes = text.as_bytes();
        for chunk in &chunks {
            let start = chunk.start_pos as usize;
            let end = chunk.end_pos as usize;
            assert!(start < full_bytes.len());
            assert!(end <= full_bytes.len());
            assert!(std::str::from_utf8(&full_bytes[start..end]).is_ok(),
                "Chunk byte range {}..{} is not valid UTF-8", start, end);
        }
    }

    #[test]
    fn test_overlap() {
        // Create text with paragraph breaks so splits are predictable
        let paragraph = "Word ".repeat(250); // ~1250 chars
        let text = format!("{}\n\n{}\n\n{}", paragraph, paragraph, paragraph);
        let chunks = chunk_text(&text, 1000, 0.1, 200);

        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());

        // Verify overlap: the end of one chunk should share content with the start of the next
        if chunks.len() >= 2 {
            let first_end = &chunks[0].content;
            let second_start = &chunks[1].content;

            // With 10% overlap (100 chars), there should be some shared text
            let overlap_found = first_end
                .lines()
                .last()
                .map(|last_line| second_start.contains(last_line.trim()))
                .unwrap_or(false);

            // At minimum, chunks should not have a gap — the second chunk
            // should start at or before the first chunk's end position
            assert!(chunks[1].start_pos <= chunks[0].end_pos,
                "Chunks should overlap: second starts at {} but first ends at {}",
                chunks[1].start_pos, chunks[0].end_pos);

            // Overlap or adjacency is acceptable
            assert!(overlap_found || chunks[1].start_pos < chunks[0].end_pos,
                "Expected overlap between consecutive chunks");
        }
    }
}
