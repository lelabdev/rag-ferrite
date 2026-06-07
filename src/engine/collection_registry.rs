//! Collection registry — tracks which collections are loaded in RAM.
//!
//! With lazy loading, only hot collections (heat_score > threshold) are searched.
//! Cold collections are skipped unless explicitly routed to by tag routing.
//! Once a collection is loaded by rag_engine, it stays in RAM (no unload API).

use anyhow::Result;
use std::collections::HashSet;
use std::sync::Mutex;

use super::get_conn;
use super::heat;

/// Collections currently loaded in RAM (activated by rag_engine).
/// Tracked locally — rag_engine doesn't expose an unload API, so once loaded = stays.
static LOADED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Mark a collection as loaded in RAM.
pub fn mark_loaded(collection: &str) {
    let mut guard = LOADED.lock().unwrap();
    let set = guard.get_or_insert_with(HashSet::new);
    set.insert(collection.to_string());
}

/// Check if a collection is already loaded in RAM.
pub fn is_loaded(collection: &str) -> bool {
    let guard = LOADED.lock().unwrap();
    guard.as_ref().is_some_and(|s| s.contains(collection))
}

/// Get all collections that are loaded.
pub fn loaded_collections() -> Vec<String> {
    let guard = LOADED.lock().unwrap();
    guard.as_ref().map(|s| s.iter().cloned().collect()).unwrap_or_default()
}

/// Collection info for lazy loading decisions.
#[derive(Debug, Clone)]
pub struct CollectionStatus {
    pub collection: String,
    pub heat_score: f64,
    pub query_count: i64,
    pub chunk_count: i64,
    pub is_loaded: bool,
    pub is_hot: bool,
}

/// Get status of all collections for lazy loading decisions.
pub fn get_all_statuses(heat_threshold: f64) -> Result<Vec<CollectionStatus>> {
    // Get chunk counts per collection (release connection before next call)
    let chunk_counts: Vec<(String, i64)> = {
        let conn = get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT collection_id, COUNT(*) as chunks
             FROM chunks
             GROUP BY collection_id",
        )?;
        stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(Result::ok)
        .collect()
    };

    // Get heat scores (uses its own get_conn internally)
    let heat_map = heat::get_all_heat()?;
    let loaded = loaded_collections();

    let statuses = chunk_counts
        .into_iter()
        .map(|(coll, chunks)| {
            let heat_entry = heat_map.iter().find(|h| h.collection == coll);
            let heat_score = heat_entry.map(|h| h.heat_score).unwrap_or(0.0);
            let query_count = heat_entry.map(|h| h.query_count).unwrap_or(0);
            let is_loaded = loaded.contains(&coll);
            let is_hot = heat_score >= heat_threshold;

            CollectionStatus {
                collection: coll,
                heat_score,
                query_count,
                chunk_count: chunks,
                is_loaded,
                is_hot,
            }
        })
        .collect();

    Ok(statuses)
}
