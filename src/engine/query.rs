use anyhow::Result;
use rag_engine::api::source_rag::{self, ChunkSearchResult};

use super::{data_dir, get_conn};

/// Fetch section_path for a batch of chunk IDs.
pub fn get_section_paths_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;

    // Build IN clause: SELECT id, section_path FROM chunks WHERE id IN (?,?,...)
    let placeholders: Vec<&str> = chunk_ids.iter().map(|_| "?").collect();
    let sql = format!(
        "SELECT id, section_path FROM chunks WHERE id IN ({})",
        placeholders.join(",")
    );
    let params: Vec<&i64> = chunk_ids.iter().collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
    })?;

    let mut map = std::collections::HashMap::new();
    // Pre-fill with None for any IDs not found in the DB
    for &id in chunk_ids {
        map.insert(id, None);
    }
    for row in rows {
        let (id, sp) = row?;
        map.insert(id, sp);
    }
    Ok(map)
}

/// Fetch page for a batch of chunk IDs.
pub fn get_pages_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<u32>>> {
    let conn = get_conn()?;
    let mut map = std::collections::HashMap::new();
    for &id in chunk_ids {
        let page: Option<u32> = conn.query_row(
            "SELECT page FROM chunks WHERE id = ?1",
            rusqlite::params![id],
            |row| row.get(0),
        ).unwrap_or(None);
        map.insert(id, page);
    }
    Ok(map)
}

/// Get chunks adjacent to a given chunk, enriched with section_path and page.
pub fn get_neighbors(source_id: i64, chunk_index: i64, before: i64, after: i64) -> Result<Vec<(ChunkSearchResult, Option<String>, Option<u32>)>> {
    let min_index = (chunk_index - before).max(0);
    let max_index = chunk_index + after;
    let chunks = source_rag::get_adjacent_chunks(source_id, min_index as i32, max_index as i32)?;

    // Fetch section_paths and pages for all chunks
    let chunk_ids: Vec<i64> = chunks.iter().map(|c| c.chunk_id).collect();
    let section_map = get_section_paths_for_chunk_ids(&chunk_ids)?;
    let page_map = get_pages_for_chunk_ids(&chunk_ids)?;

    let enriched = chunks
        .into_iter()
        .map(|c| {
            let sp = section_map.get(&c.chunk_id).cloned().flatten();
            let pg = page_map.get(&c.chunk_id).cloned().flatten();
            (c, sp, pg)
        })
        .collect();

    Ok(enriched)
}

/// Delete a source by ID
pub fn delete_source(source_id: i64) -> Result<()> {
    // Look up the collection before deleting, so we can rebuild its indexes
    let conn = get_conn()?;
    let collection_id: Option<String> = conn
        .query_row(
            "SELECT collection_id FROM sources WHERE id = ?1",
            rusqlite::params![source_id],
            |row| row.get(0),
        )
        .ok()
        .flatten();
    drop(conn);

    source_rag::delete_source(source_id)?;

    // Also delete orphaned chunks and their tags (rag_engine::delete_source may not clean them)
    {
        let conn = get_conn()?;
        // Delete tags for chunks belonging to this source (before deleting the chunks)
        conn.execute(
            "DELETE FROM chunk_tags WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = ?1)",
            rusqlite::params![source_id],
        )?;
        conn.execute("DELETE FROM chunks WHERE source_id = ?1", rusqlite::params![source_id])?;
    }

    // Rebuild indexes for the specific collection if found
    if let Some(ref coll) = collection_id {
        if let Err(e) = source_rag::rebuild_chunk_hnsw_index_for_collection(coll.clone()) {
            tracing::warn!("Failed to rebuild HNSW index for {}: {}", coll, e);
        }
        if let Err(e) = source_rag::rebuild_chunk_bm25_index_for_collection(coll.clone()) {
            tracing::warn!("Failed to rebuild BM25 index for {}: {}", coll, e);
        }
        // Persist updated HNSW index
        let index_path = format!("{}/hnsw_{}.index", data_dir(), coll);
        if let Err(e) = source_rag::save_collection_hnsw_index(coll.clone(), index_path) {
            tracing::warn!("Failed to save HNSW index for {}: {}", coll, e);
        }
    } else {
        // Fallback: rebuild all if we couldn't find the collection
        tracing::warn!("Could not find collection for source {}, rebuilding all indexes", source_id);
        let _ = source_rag::rebuild_chunk_hnsw_index();
        let _ = source_rag::rebuild_chunk_bm25_index();
    }

    Ok(())
}

/// List all sources across all collections.
///
/// Queries the `sources` table directly instead of using
/// `source_rag::list_sources()` which hardcodes the `__default__` collection.
pub fn list_sources() -> Result<Vec<source_rag::SourceEntry>> {
    let conn = get_conn()?;

    let mut stmt = conn.prepare(
        "SELECT id, name, created_at, metadata, status, collection_id
         FROM sources
         ORDER BY id DESC",
    )?;

    let entries: Vec<source_rag::SourceEntry> = stmt.query_map([], |row| {
        Ok(source_rag::SourceEntry {
            id: row.get(0)?,
            name: row.get(1)?,
            created_at: row.get(2)?,
            metadata: row.get(3)?,
            status: row.get(4)?,
            collection_id: row.get(5)?,
        })
    })?.filter_map(|e| e.ok()).collect();

    Ok(entries)
}
