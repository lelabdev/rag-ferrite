//! Soft tag routing — route queries to the best-matching collection based on tags.
//!
//! Extracts keywords from the query, matches them against the `collection_tags`
//! table, and returns the best collection if a clear winner emerges.
//! Soft routing: if no match, falls back to searching all collections.

use anyhow::Result;

use super::get_conn;
use crate::tag_rules::get_tag_rules;

/// Result of routing a query.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// The suggested collection (if any).
    pub collection: Option<String>,
    /// All collections that matched, with their score (sum of chunk_counts for matched tags).
    pub matches: Vec<(String, i64)>,
    /// The keywords extracted from the query.
    pub keywords: Vec<String>,
}

/// Route a query to the best-matching collection.
///
/// Algorithm:
/// 1. Extract keywords from the query (lowercase, strip stop words, apply synonyms)
/// 2. Look up `collection_tags` for those keywords
/// 3. Sum chunk_count per collection
/// 4. If the top collection has >2x the score of the runner-up → suggest it (strong signal)
/// 5. Otherwise → None (ambiguous, search all)
pub fn route_query(query: &str) -> Result<RouteResult> {
    let keywords = extract_keywords(query);

    if keywords.is_empty() {
        return Ok(RouteResult {
            collection: None,
            matches: vec![],
            keywords: vec![],
        });
    }

    // Query collection_tags for matching keywords
    let conn = get_conn()?;
    let placeholders: Vec<String> = (1..=keywords.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT collection, SUM(chunk_count) as total
         FROM collection_tags
         WHERE tag IN ({})
         GROUP BY collection
         ORDER BY total DESC",
        placeholders.join(",")
    );

    let params: Vec<&dyn rusqlite::types::ToSql> = keywords
        .iter()
        .map(|k| k as &dyn rusqlite::types::ToSql)
        .collect();

    let mut stmt = conn.prepare(&sql)?;
    let matches: Vec<(String, i64)> = stmt
        .query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(Result::ok)
        .collect();

    // Decide: strong signal if top collection has >2x runner-up score
    let collection = if matches.len() == 1 {
        // Only one collection matched → route there (unless it's "general" and we have others)
        Some(matches[0].0.clone())
    } else if matches.len() >= 2 {
        let (top_coll, top_score) = &matches[0];
        let (_, runner_up) = &matches[1];
        if *top_score >= runner_up * 2 {
            Some(top_coll.clone())
        } else {
            None // ambiguous
        }
    } else {
        None
    };

    // Don't route if the only suggestion is "general" — it's the default anyway
    let collection = collection.filter(|c| c != "general");

    Ok(RouteResult {
        collection,
        matches,
        keywords,
    })
}

/// Extract keywords from a query string.
/// Lowercases, removes stop words, applies synonyms from tag_rules.
fn extract_keywords(query: &str) -> Vec<String> {
    let rules = get_tag_rules();

    let stop_words: std::collections::HashSet<&str> = rules.stop_words.all().into_iter().collect();

    let strip_chars: &str = rules.rules.strip_chars.as_str();

    query
        .to_lowercase()
        .split_whitespace()
        .map(|word| word.trim_matches(|c| strip_chars.contains(c)).to_string())
        .filter(|word| word.len() >= rules.rules.min_length)
        .filter(|word| !stop_words.contains(word.as_str()))
        .map(|word| {
            // Apply synonyms
            rules.synonyms.get(&word).cloned().unwrap_or(word)
        })
        .collect()
}

/// Get a summary of which collections have which tags.
/// Useful for the suggest_collection endpoint.
pub fn get_tag_collection_map() -> Result<Vec<(String, String, i64)>> {
    let conn = get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT tag, collection, chunk_count
         FROM collection_tags
         ORDER BY tag, chunk_count DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords_basic() {
        // Just test that it doesn't crash and returns something
        let keywords = extract_keywords("how to build a rust web server");
        assert!(!keywords.is_empty());
        // "rust" should be in there (not a stop word)
        assert!(keywords.contains(&"rust".to_string()));
    }

    #[test]
    fn test_extract_keywords_empty() {
        let keywords = extract_keywords("");
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_extract_keywords_short_filtered() {
        // Words shorter than min_length (3) are filtered
        let keywords = extract_keywords("a an be to do");
        assert!(keywords.is_empty(), "Short words should be filtered");
    }
}
