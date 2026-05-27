use anyhow::Result;

use super::get_conn;

/// Create the chunk_tags table if it doesn't exist.
pub fn create_chunk_tags_table(db_path: &str) -> Result<()> {
    let conn = rusqlite::Connection::open(db_path)?;
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

/// Insert tags for all chunks of a source, matched by chunk_index position.
/// tags_per_chunk[i] contains the tags for the i-th kept chunk.
pub fn insert_chunk_tags(source_id: i64, tags_per_chunk: &[Vec<String>]) -> Result<()> {
    let conn = get_conn()?;

    for (idx, tags) in tags_per_chunk.iter().enumerate() {
        if tags.is_empty() {
            continue;
        }
        // Look up chunk_id by source_id + chunk_index
        let chunk_id: Option<i64> = conn
            .query_row(
                "SELECT id FROM chunks WHERE source_id = ?1 AND chunk_index = ?2",
                rusqlite::params![source_id, idx as i32],
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

/// Fetch tags for a batch of chunk IDs.
pub fn get_tags_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Vec<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let conn = get_conn()?;

    let mut map = std::collections::HashMap::new();
    for &id in chunk_ids {
        let mut stmt = conn.prepare(
            "SELECT tag FROM chunk_tags WHERE chunk_id = ?1 ORDER BY tag"
        )?;
        let tags: Vec<String> = stmt
            .query_map(rusqlite::params![id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        if !tags.is_empty() {
            map.insert(id, tags);
        }
    }
    Ok(map)
}
