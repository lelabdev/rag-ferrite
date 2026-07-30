use crate::storage::sqlite;
use crate::types::ChunkSearchResult;
use anyhow::Result;

use super::get_conn;

/// Generate a SQL IN clause with N placeholders: "?,?,?,?,?" for N items.
pub fn in_placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

/// Fetch section_path for a batch of chunk IDs.
pub fn get_section_paths_for_chunk_ids(
    chunk_ids: &[i64],
) -> Result<std::collections::HashMap<i64, Option<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;

    let sql = format!(
        "SELECT id, section_path FROM chunks WHERE id IN ({})",
        in_placeholders(chunk_ids.len())
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
pub fn get_pages_for_chunk_ids(
    chunk_ids: &[i64],
) -> Result<std::collections::HashMap<i64, Option<u32>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;
    let mut map = std::collections::HashMap::new();
    let sql = format!(
        "SELECT id, page FROM chunks WHERE id IN ({})",
        in_placeholders(chunk_ids.len())
    );
    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();
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
/// Resolve source_id → source name for a list of IDs.
pub fn get_source_names(source_ids: &[i64]) -> Result<std::collections::HashMap<i64, String>> {
    let conn = crate::engine::get_conn()?;
    let mut map = std::collections::HashMap::new();
    for &id in source_ids {
        let name: String = conn.query_row(
            "SELECT name FROM sources WHERE id = ?1",
            rusqlite::params![id],
            |row| row.get(0),
        )?;
        map.insert(id, name);
    }
    Ok(map)
}

pub fn get_neighbors(
    source_id: i64,
    chunk_index: i64,
    before: i64,
    after: i64,
) -> Result<Vec<(ChunkSearchResult, Option<String>, Option<u32>)>> {
    let min_index = (chunk_index - before).max(0);
    let max_index = chunk_index + after;
    let chunks = sqlite::get_adjacent_chunks(source_id, min_index as i32, max_index as i32)?;

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
    sqlite::delete_source(source_id)?;

    // Indexes are not rebuilt synchronously on delete; the next flush refreshes them.
    Ok(())
}

/// List all sources across all collections.
///
/// Queries the `sources` table directly instead of using
/// `source_rag::list_sources()` which hardcodes the `__default__` collection.
pub fn list_sources() -> Result<Vec<crate::types::SourceEntry>> {
    let conn = get_conn()?;

    let mut stmt = conn.prepare(
        "SELECT id, name, created_at, metadata, status, collection_id
         FROM sources
         ORDER BY id DESC",
    )?;

    let entries: Vec<crate::types::SourceEntry> = stmt
        .query_map([], |row| {
            Ok(crate::types::SourceEntry {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                metadata: row.get(3)?,
                status: row.get(4)?,
                collection_id: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(entries)
}

/// Count chunks per source — used by list to show chunk_count.
pub fn count_chunks_per_source() -> Result<std::collections::HashMap<i64, i64>> {
    let conn = get_conn()?;
    let mut stmt =
        conn.prepare("SELECT source_id, COUNT(*) as cnt FROM chunks GROUP BY source_id")?;
    let counts: std::collections::HashMap<i64, i64> = stmt
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .collect();
    Ok(counts)
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
    let sql = format!(
        "SELECT c.id, c.parent_id FROM chunks c WHERE c.id IN ({}) AND c.chunk_role = 'child'",
        in_placeholders(chunk_ids.len())
    );
    let params: Vec<&i64> = chunk_ids.iter().collect();
    let mut stmt = conn.prepare(&sql)?;
    let child_rows: Vec<(i64, i64)> = stmt
        .query_map(rusqlite::params_from_iter(params.iter().copied()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if child_rows.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    // Batch-fetch parent content
    let parent_ids: Vec<i64> = child_rows.iter().map(|(_, pid)| *pid).collect();
    let parent_sql = format!(
        "SELECT id, content, section_path, page FROM chunks WHERE id IN ({})",
        in_placeholders(parent_ids.len())
    );
    let parent_params: Vec<&i64> = parent_ids.iter().collect();
    let mut parent_stmt = conn.prepare(&parent_sql)?;
    let mut parent_data: std::collections::HashMap<i64, (String, Option<String>, Option<u32>)> =
        std::collections::HashMap::new();
    for (id, content, sp, page) in parent_stmt
        .query_map(
            rusqlite::params_from_iter(parent_params.iter().copied()),
            |row| {
                let id: i64 = row.get(0)?;
                let content: String = row.get(1)?;
                let section_path: Option<String> = row.get(2)?;
                let page: Option<u32> = row.get::<_, Option<i64>>(3)?.map(|p| p as u32);
                Ok((id, content, section_path, page))
            },
        )?
        .flatten()
    {
        parent_data.insert(id, (content, sp, page));
    }

    // Build result map: child_id → ParentInfo
    let mut result = std::collections::HashMap::new();
    for (child_id, parent_id) in &child_rows {
        if let Some((content, section_path, page)) = parent_data.get(parent_id) {
            result.insert(
                *child_id,
                ParentInfo {
                    content: content.clone(),
                    section_path: section_path.clone(),
                    page: *page,
                },
            );
        }
    }

    Ok(result)
}
