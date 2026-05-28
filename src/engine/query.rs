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

/// Fetch page for a batch of chunk IDs using a single IN query.
pub fn get_pages_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<u32>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;
    let mut map = std::collections::HashMap::new();
    let placeholders: Vec<String> = chunk_ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect();
    let sql = format!("SELECT id, page FROM chunks WHERE id IN ({})", placeholders.join(","));
    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<u32>>(1)?))
    })?;
    for row in rows {
        let (id, page) = row?;
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
        super::rebuild_and_save_indexes(coll);
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

/// Parent info resolved from a child chunk.
pub struct ParentInfo {
    pub content: String,
    pub section_path: Option<String>,
    pub page: Option<u32>,
}

/// For a list of chunk IDs, resolve parents for any that are children.
/// Returns a map from child chunk ID → ParentInfo.
/// Chunks that are not children (recursive mode, or parents themselves) are not in the map.
pub fn resolve_parents(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, ParentInfo>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;

    // Find children (chunks with chunk_role = 'child' and a parent_id)
    let placeholders: Vec<&str> = chunk_ids.iter().map(|_| "?").collect();
    let sql = format!(
        "SELECT c.id, c.parent_id FROM chunks c WHERE c.id IN ({}) AND c.chunk_role = 'child'",
        placeholders.join(",")
    );
    let params: Vec<&i64> = chunk_ids.iter().collect();
    let mut stmt = conn.prepare(&sql)?;
    let child_rows: Vec<(i64, i64)> = stmt
        .query_map(rusqlite::params_from_iter(params.iter().copied()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if child_rows.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    // Batch-fetch parent content
    let parent_ids: Vec<i64> = child_rows.iter().map(|(_, pid)| *pid).collect();
    let parent_placeholders: Vec<&str> = parent_ids.iter().map(|_| "?").collect();
    let parent_sql = format!(
        "SELECT id, content, section_path, page FROM chunks WHERE id IN ({})",
        parent_placeholders.join(",")
    );
    let parent_params: Vec<&i64> = parent_ids.iter().collect();
    let mut parent_stmt = conn.prepare(&parent_sql)?;
    let mut parent_data: std::collections::HashMap<i64, (String, Option<String>, Option<u32>)> =
        std::collections::HashMap::new();
    for row in parent_stmt.query_map(rusqlite::params_from_iter(parent_params.iter().copied()), |row| {
        let id: i64 = row.get(0)?;
        let content: String = row.get(1)?;
        let section_path: Option<String> = row.get(2)?;
        let page: Option<u32> = row.get::<_, Option<i64>>(3)?.map(|p| p as u32);
        Ok((id, content, section_path, page))
    })? {
        if let Ok((id, content, sp, page)) = row {
            parent_data.insert(id, (content, sp, page));
        }
    }

    // Build result map: child_id → ParentInfo
    let mut result = std::collections::HashMap::new();
    for (child_id, parent_id) in &child_rows {
        if let Some((content, section_path, page)) = parent_data.get(parent_id) {
            result.insert(*child_id, ParentInfo {
                content: content.clone(),
                section_path: section_path.clone(),
                page: *page,
            });
        }
    }

    Ok(result)
}
