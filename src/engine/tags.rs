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
         CREATE INDEX IF NOT EXISTS idx_chunk_tags_chunk_id ON chunk_tags(chunk_id);"
    )?;
    tracing::info!("chunk_tags table ready");
    Ok(())
}

/// Insert tags for all chunks of a source, matched by original chunk_index.
/// tags_per_chunk contains (chunk_index, tags) pairs.
/// If `conn` is provided, uses it (avoids deadlock when caller already holds get_conn lock).
pub fn insert_chunk_tags(source_id: i64, tags_per_chunk: &[(i32, Vec<String>)], conn: Option<&Connection>) -> Result<()> {
    // If caller provided a connection, use it. Otherwise get our own.
    let owned_conn;
    let conn: &Connection = match conn {
        Some(c) => c,
        None => { owned_conn = get_conn()?; &owned_conn }
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
    tracing::debug!("Inserted tags for {} chunks of source {}", tags_per_chunk.len(), source_id);
    Ok(())
}

/// Fetch tags for a batch of chunk IDs using a single IN query.
pub fn get_tags_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Vec<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;
    let mut map = std::collections::HashMap::new();

    let placeholders: Vec<String> = chunk_ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect();
    let sql = format!("SELECT chunk_id, tag FROM chunk_tags WHERE chunk_id IN ({}) ORDER BY tag", placeholders.join(","));
    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
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
