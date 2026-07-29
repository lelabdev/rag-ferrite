use anyhow::Result;

use super::{get_conn, sanitize_collection};

/// No-op: brute-force search reads embeddings directly from SQLite at query time.
/// Kept for API compatibility with the ingestion pipeline.
pub fn add_embeddings_to_buffer(_source_id: i64) {
    // Brute-force cosine search doesn't need an incremental buffer.
    // Embeddings are read directly from the chunks table during search.
}

/// Rebuild FTS5 index for BM25 search. Vector search is brute-force (no index needed).
pub fn rebuild_and_save_indexes(_collection_id: &str) {
    if let Err(e) = crate::storage::sqlite::rebuild_fts_index() {
        tracing::warn!("Failed to rebuild FTS5 index: {}", e);
    }
}

/// Move a source (and all its chunks) to a different collection.
pub fn reassign_source_collection(source_id: i64, new_collection: &str) -> Result<String> {
    let new_collection = sanitize_collection(new_collection)?;
    let conn = get_conn()?;

    // Get current collection
    let old_collection: String = conn.query_row(
        "SELECT collection_id FROM sources WHERE id = ?1",
        rusqlite::params![source_id],
        |row| row.get(0),
    )?;

    if old_collection == new_collection {
        return Ok(format!("Source {} already in collection '{}'", source_id, new_collection));
    }

    // Get source name for logging
    let source_name: Option<String> = conn.query_row(
        "SELECT name FROM sources WHERE id = ?1",
        rusqlite::params![source_id],
        |row| row.get(0),
    ).ok();

    // Update sources
    let _updated = conn.execute(
        "UPDATE sources SET collection_id = ?1 WHERE id = ?2",
        rusqlite::params![new_collection, source_id],
    )?;

    // Update chunks
    let chunks_updated = conn.execute(
        "UPDATE chunks SET collection_id = ?1 WHERE source_id = ?2",
        rusqlite::params![new_collection, source_id],
    )?;

    tracing::info!(
        "Reassigned source {} ({:?}): {} → {} ({} chunks moved)",
        source_id, source_name, old_collection, new_collection, chunks_updated
    );
    crate::pipeline::invalidate_cache();

    Ok(format!(
        "Source {} ({:?}) moved: {} → {} ({} chunks)",
        source_id, source_name, old_collection, new_collection, chunks_updated
    ))
}

/// Run WAL checkpoint to free disk space and reduce memory pressure.
pub fn wal_checkpoint() {
    match get_conn() {
        Ok(conn) => {
            match conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)") {
                Ok(_) => tracing::info!("WAL checkpoint completed"),
                Err(e) => tracing::warn!("WAL checkpoint failed: {}", e),
            }
        }
        Err(e) => tracing::warn!("WAL checkpoint: cannot get DB connection: {}", e),
    }
}
