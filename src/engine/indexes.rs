use anyhow::Result;

use rag_engine::api::incremental_index;
use rag_engine::api::source_rag;

use super::{get_conn, sanitize_collection, data_dir};

/// Add all embeddings for a source to the incremental buffer (immediately searchable).
pub fn add_embeddings_to_buffer(source_id: i64) {
    let conn = match get_conn() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Cannot get DB connection for incremental buffer: {}", e);
            return;
        }
    };
    let mut stmt = match conn.prepare(
        "SELECT c.id, c.embedding FROM chunks c WHERE c.source_id = ?1 AND c.embedding IS NOT NULL"
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Cannot prepare embedding query: {}", e);
            return;
        }
    };
    let rows: Vec<(i64, Vec<u8>)> = stmt.query_map([source_id], |row| {
        let id: i64 = row.get(0)?;
        let emb_bytes: Vec<u8> = row.get(1)?;
        Ok((id, emb_bytes))
    }).ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();

    let mut batch = Vec::new();
    for (id, emb_bytes) in &rows {
        // Embeddings stored as f32 NE bytes
        let embedding: Vec<f32> = emb_bytes.chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        batch.push((*id, embedding));
    }
    incremental_index::incremental_add_batch(batch);
    tracing::info!("Added {} embeddings to incremental buffer for source {}", rows.len(), source_id);
}

/// Rebuild and persist HNSW + BM25 indexes for a collection.
pub fn rebuild_and_save_indexes(collection_id: &str) {
    // Merge incremental buffer into HNSW before rebuild
    if incremental_index::needs_merge() {
        tracing::info!("Incremental buffer threshold reached, merging into HNSW index");
    }

    if let Err(e) = source_rag::rebuild_chunk_hnsw_index_for_collection(collection_id.to_string()) {
        tracing::warn!("Failed to rebuild HNSW index for {}: {}", collection_id, e);
    }
    if let Err(e) = source_rag::rebuild_chunk_bm25_index_for_collection(collection_id.to_string()) {
        tracing::warn!("Failed to rebuild BM25 index for {}: {}", collection_id, e);
    }
    let index_path = format!("{}/hnsw_{}.index", data_dir(), collection_id);
    if let Err(e) = source_rag::save_collection_hnsw_index(collection_id.to_string(), index_path) {
        tracing::warn!("Failed to save HNSW index for {}: {}", collection_id, e);
    }
    // Clear buffer after successful rebuild — all vectors are now in the main index
    incremental_index::clear_buffer();
}

/// Move a source (and all its chunks) to a different collection.
/// Rebuilds HNSW + BM25 indexes for both old and new collections.
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

    // Rebuild indexes for both collections
    rebuild_and_save_indexes(&old_collection);
    rebuild_and_save_indexes(&new_collection);

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
