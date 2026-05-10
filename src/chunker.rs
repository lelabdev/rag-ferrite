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

            chunks.push(Chunk {
                content: trimmed,
                index: chunks.len() as i32,
                start_pos: byte_start as i32,
                end_pos: byte_end as i32,
                chunk_type: ChunkType::Text,
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
    Title,
    Code,
}
