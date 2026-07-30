use anyhow::Result;
use rusqlite::Connection;

use super::get_conn;

/// Create the chunk_tags table if it doesn't exist.
pub fn create_chunk_tags_table(db_path: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunk_tags (
            chunk_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (chunk_id, tag)
         );
         CREATE INDEX IF NOT EXISTS idx_chunk_tags_tag ON chunk_tags(tag);
         CREATE INDEX IF NOT EXISTS idx_chunk_tags_chunk_id ON chunk_tags(chunk_id);",
    )?;
    tracing::info!("chunk_tags table ready");
    Ok(())
}

/// Create the collection_tags table for tag routing (v5 design).
/// Maps each tag to the collections where it appears, with chunk counts.
pub fn create_collection_tags_table(db_path: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS collection_tags (
            tag TEXT NOT NULL,
            collection TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag, collection)
         );
         CREATE INDEX IF NOT EXISTS idx_collection_tags_tag ON collection_tags(tag);",
    )?;
    tracing::info!("collection_tags table ready");
    Ok(())
}

/// Update collection_tags after inserting tags for a source.
/// For each (tag, collection) pair, increment chunk_count.
pub fn update_collection_tags(
    source_id: i64,
    tags_per_chunk: &[(i32, Vec<String>)],
    collection_id: &str,
    conn: &Connection,
) -> Result<()> {
    // Count unique tags across all chunks of this source
    let mut tag_counts: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    for (_, tags) in tags_per_chunk {
        for tag in tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    for (tag, count) in &tag_counts {
        conn.execute(
            "INSERT INTO collection_tags (tag, collection, chunk_count)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(tag, collection) DO UPDATE SET chunk_count = chunk_count + ?3",
            rusqlite::params![tag, collection_id, count],
        )?;
    }

    if !tag_counts.is_empty() {
        tracing::debug!(
            "Updated collection_tags: {} unique tags for source {} in collection '{}'",
            tag_counts.len(),
            source_id,
            collection_id
        );
    }
    Ok(())
}

/// Insert tags for all chunks of a source, matched by original chunk_index.
/// tags_per_chunk contains (chunk_index, tags) pairs.
/// If `conn` is provided, uses it (avoids deadlock when caller already holds get_conn lock).
pub fn insert_chunk_tags(
    source_id: i64,
    tags_per_chunk: &[(i32, Vec<String>)],
    conn: Option<&Connection>,
) -> Result<()> {
    // If caller provided a connection, use it. Otherwise get our own.
    let owned_conn;
    let conn: &Connection = match conn {
        Some(c) => c,
        None => {
            owned_conn = get_conn()?;
            &owned_conn
        }
    };

    for &(chunk_index, ref tags) in tags_per_chunk {
        if tags.is_empty() {
            continue;
        }
        // Look up chunk_id by source_id + chunk_index
        let chunk_id: Option<i64> = conn
            .query_row(
                "SELECT id FROM chunks WHERE source_id = ?1 AND chunk_index = ?2",
                rusqlite::params![source_id, chunk_index],
                |row| row.get(0),
            )
            .ok();

        if let Some(cid) = chunk_id {
            for tag in tags {
                conn.execute(
                    "INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?1, ?2)",
                    rusqlite::params![cid, tag],
                )?;
            }
        }
    }
    tracing::debug!(
        "Inserted tags for {} chunks of source {}",
        tags_per_chunk.len(),
        source_id
    );
    Ok(())
}

/// Get chunk_ids that have **all** the given tags (AND logic).
/// With 1 tag: returns all chunks with that tag (broad).
/// With 2+ tags: returns only chunks matching every tag (intersection / precise).
pub fn get_chunk_ids_for_tags(tags: &[String]) -> anyhow::Result<Vec<i64>> {
    if tags.is_empty() {
        return Ok(Vec::new());
    }
    let conn = get_conn()?;

    if tags.len() == 1 {
        // Single tag: simple lookup
        let sql = "SELECT chunk_id FROM chunk_tags WHERE tag = ?1";
        let mut stmt = conn.prepare(sql)?;
        let ids: Vec<i64> = stmt
            .query_map(rusqlite::params![tags[0]], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        tracing::debug!("Found {} chunk_ids for 1 tag", ids.len());
        return Ok(ids);
    }

    // Multiple tags: INTERSECT — chunk must have ALL tags
    let selects: Vec<String> = tags
        .iter()
        .enumerate()
        .map(|(i, _)| format!("SELECT chunk_id FROM chunk_tags WHERE tag = ?{}", i + 1))
        .collect();
    let sql = selects.join(" INTERSECT ");
    let params: Vec<&dyn rusqlite::types::ToSql> = tags
        .iter()
        .map(|t| t as &dyn rusqlite::types::ToSql)
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let ids: Vec<i64> = stmt
        .query_map(params.as_slice(), |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    tracing::debug!(
        "Found {} chunk_ids for {} tags (AND intersection)",
        ids.len(),
        tags.len()
    );
    Ok(ids)
}

/// Fetch tags for a batch of chunk IDs using a single IN query.
pub fn get_tags_for_chunk_ids(
    chunk_ids: &[i64],
) -> Result<std::collections::HashMap<i64, Vec<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;
    let mut map = std::collections::HashMap::new();

    let sql = format!(
        "SELECT chunk_id, tag FROM chunk_tags WHERE chunk_id IN ({}) ORDER BY tag",
        crate::engine::query::in_placeholders(chunk_ids.len())
    );
    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (chunk_id, tag) = row?;
        map.entry(chunk_id).or_insert_with(Vec::new).push(tag);
    }
    Ok(map)
}
