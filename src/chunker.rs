/// Recursive character text splitter with overlap (UTF-8 safe)
///
/// chunk_size: ~1000 chars (~250 tokens, optimal per RAG Cookbook)
/// overlap: 10% for context preservation
///
/// Also tracks markdown headers (# H1, ## H2, etc.) and builds a section path
/// for each chunk based on which header section it falls into.
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<Chunk> {
    let separators = ["\n\n", "\n", ". ", " "];
    let overlap = (chunk_size as f64 * 0.1) as usize;

    // Pre-scan: build section map from markdown headers
    let sections = extract_sections(text);

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

            // Look up section path based on byte position of chunk start
            let section_path = find_section_for_position(&sections, byte_start);

            chunks.push(Chunk {
                content: trimmed,
                index: chunks.len() as i32,
                start_pos: byte_start as i32,
                end_pos: byte_end as i32,
                chunk_type: ChunkType::Text,
                section_path,
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
            let last_section = chunks[last_idx].section_path.clone();
            let last_content = chunks.pop().unwrap();
            let prev = chunks.last_mut().unwrap();
            prev.content = format!("{}\n\n{}", prev.content, last_content.content);
            prev.end_pos = last_end;
            // Preserve section path from the last chunk if it has one
            if prev.section_path.is_none() && last_section.is_some() {
                prev.section_path = last_section;
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

pub fn extract_sections(text: &str) -> Vec<(usize, String)> {
    let mut sections: Vec<(usize, String)> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new(); // (level, title)
    let mut offset: usize = 0;

    for line in text.split('\n') {
        let trimmed = line.trim_start();
        let header_level = count_hash_prefix(trimmed);

        if header_level > 0 {
            // Skip the #'s and leading whitespace to get the title
            let after_hashes = &trimmed[header_level..];
            let title = after_hashes.trim_start();

            if !title.is_empty() {
                // Pop headers at same or deeper level (higher number = deeper)
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
        // Account for the '\n' separator
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
            return 0; // e.g. "##no-space" is not a header
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
            break; // sections are sorted by offset
        }
    }
    result
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub index: i32,
    pub start_pos: i32,
    pub end_pos: i32,
    pub chunk_type: ChunkType,
    pub section_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Text,
    #[allow(dead_code)]
    Title,
    #[allow(dead_code)]
    Code,
}
