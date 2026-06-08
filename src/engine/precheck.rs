use anyhow::Result;

use super::get_conn;

/// Pre-ingestion check: language detection, duplicate detection, size warnings.
pub fn pre_check_document(content: &str, filename: &str, chunk_size: usize) -> crate::types::PreCheckReport {
    let mut warnings = Vec::new();

    let char_count = content.len();

    let extraction_ok = !content.trim().is_empty();

    if char_count < 100 {
        warnings.push(format!("Very short content ({} chars), may not provide useful retrieval results", char_count));
    }

    if char_count > 500_000 {
        warnings.push(format!("Large document ({} chars), ingestion may take a while", char_count));
    }

    let estimated_chunks = if char_count == 0 {
        0
    } else if char_count < chunk_size {
        1
    } else {
        char_count.div_ceil(chunk_size)
    };

    let language = detect_language(content);

    let is_duplicate = check_duplicate_source(filename);
    if is_duplicate {
        warnings.push(format!("A document named '{}' already exists in the index", filename));
    }

    crate::types::PreCheckReport {
        extraction_ok,
        char_count,
        estimated_chunks,
        language,
        is_duplicate,
        warnings,
    }
}

/// Simple language detection heuristic based on character frequency.
fn detect_language(text: &str) -> String {
    let sample = &text[..text.len().min(5000)];
    let mut french_accents = 0usize;
    let mut cjk_chars = 0usize;
    let mut latin_chars = 0usize;
    let mut arabic_chars = 0usize;
    let mut cyrillic_chars = 0usize;

    for ch in sample.chars() {
        match ch {
            'à' | 'â' | 'é' | 'è' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'ù' | 'û' | 'ü' | 'ÿ' | 'ç'
            | 'À' | 'Â' | 'É' | 'È' | 'Ê' | 'Ë' | 'Î' | 'Ï' | 'Ô' | 'Ù' | 'Û' | 'Ü' | 'Ÿ' | 'Ç' => {
                french_accents += 1;
                latin_chars += 1;
            }
            c if ('\u{4E00}'..='\u{9FFF}').contains(&c) => cjk_chars += 1,
            c if ('\u{3040}'..='\u{309F}').contains(&c) || ('\u{30A0}'..='\u{30FF}').contains(&c) => cjk_chars += 1,
            c if ('\u{0600}'..='\u{06FF}').contains(&c) || ('\u{0750}'..='\u{077F}').contains(&c) => arabic_chars += 1,
            c if ('\u{0400}'..='\u{04FF}').contains(&c) => cyrillic_chars += 1,
            'a'..='z' | 'A'..='Z' => latin_chars += 1,
            _ => {}
        }
    }

    let total = latin_chars + cjk_chars + arabic_chars + cyrillic_chars;
    if total == 0 {
        return "unknown".to_string();
    }

    if cjk_chars as f64 / total as f64 > 0.3 {
        return "cjk".to_string();
    }

    if arabic_chars as f64 / total as f64 > 0.3 {
        return "arabic".to_string();
    }

    if cyrillic_chars as f64 / total as f64 > 0.3 {
        return "cyrillic".to_string();
    }

    if french_accents >= 3 {
        return "french".to_string();
    }

    "english".to_string()
}

/// Check if a source with the given name already exists in the DB.
pub fn check_duplicate_source(filename: &str) -> bool {
    let conn = match get_conn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sources WHERE name = ?1",
            rusqlite::params![filename],
            |row| row.get(0),
        )
        .unwrap_or(0);
    count > 0
}

/// Post-chunking verification: checks coverage, empty chunks, and logs warnings.
pub fn verify_chunks(chunks: &[String], source: &str) -> crate::types::ChunkVerification {
    let total_chunks = chunks.len();
    let source_chars = source.len();
    let chunk_chars: usize = chunks.iter().map(|c| c.len()).sum();
    let coverage_ratio = if source_chars == 0 {
        1.0
    } else {
        chunk_chars as f64 / source_chars as f64
    };

    let mut warnings = Vec::new();

    // Warn on empty chunks
    let empty_count = chunks.iter().filter(|c| c.trim().is_empty()).count();
    if empty_count > 0 {
        warnings.push(format!("{} empty chunks found for source '{}'", empty_count, source));
    }

    // Warn if coverage < 90%
    if coverage_ratio < 0.9 {
        warnings.push(format!(
            "Low chunk coverage {:.1}% for source '{}' ({} source chars, {} chunk chars)",
            coverage_ratio * 100.0, source, source_chars, chunk_chars
        ));
    }

    crate::types::ChunkVerification {
        total_chunks,
        source_chars,
        chunk_chars,
        coverage_ratio,
        warnings,
    }
}

/// Update heat tracking for chunks returned in query results.
pub fn update_chunk_heat(chunk_ids: &[i64]) -> Result<()> {
    if chunk_ids.is_empty() {
        return Ok(());
    }
    let conn = get_conn()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as f64;

    for &id in chunk_ids {
        conn.execute(
            "UPDATE chunks SET query_count = query_count + 1, last_queried_at = ?1 WHERE id = ?2",
            rusqlite::params![now, id],
        )?;
    }
    tracing::debug!("Updated heat for {} chunks", chunk_ids.len());
    Ok(())
}
