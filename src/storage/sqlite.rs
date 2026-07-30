use crate::engine::get_conn;
use crate::types::{ChunkData, SearchFilter, SearchResult, SourceEntry};
use anyhow::Result;

/// Default collection ID used by the local SQLite storage layer.
pub const DEFAULT_COLLECTION_ID: &str = "__default__";

/// Result of adding a source.
#[derive(Debug, Clone)]
pub struct AddSourceResult {
    pub source_id: i64,
    pub is_duplicate: bool,
    pub chunk_count: i32,
    pub message: String,
}

/// Hash content for deduplication using the stable SHA-256 storage format.
fn hash_content(input: &str) -> String {
    use sha2::{Digest, Sha256};

    format!("{:x}", Sha256::digest(input.as_bytes()))
}

/// Legacy v1 hashes remain readable so existing sources deduplicate on their next access.
fn legacy_hash_content(input: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Add a source to a collection.
pub fn add_source_in_collection(
    collection_id: String,
    content: String,
    metadata: Option<String>,
    name: Option<String>,
) -> Result<AddSourceResult> {
    let scoped_content = format!("{}:{}", collection_id, content);
    let scoped_hash = hash_content(&scoped_content);
    let legacy_hash = legacy_hash_content(&scoped_content);
    let conn = get_conn()?;

    // Accept legacy hashes during the SHA-256 migration, then upgrade them lazily.
    let existing: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, content_hash FROM sources
         WHERE collection_id = ?1 AND content_hash IN (?2, ?3)
         LIMIT 1",
            rusqlite::params![collection_id, scoped_hash, legacy_hash],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();

    if let Some((id, existing_hash)) = existing {
        if existing_hash != scoped_hash {
            conn.execute(
                "UPDATE sources SET content_hash = ?1 WHERE id = ?2",
                rusqlite::params![scoped_hash, id],
            )?;
        }
        return Ok(AddSourceResult {
            source_id: id,
            is_duplicate: true,
            chunk_count: 0,
            message: format!("Source already exists (id={})", id),
        });
    }

    conn.execute(
        "INSERT INTO sources (content, content_hash, metadata, name, status, collection_id)
         VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
        rusqlite::params![content, scoped_hash, metadata, name, collection_id],
    )?;

    let source_id = conn.last_insert_rowid();
    crate::pipeline::invalidate_cache();
    Ok(AddSourceResult {
        source_id,
        is_duplicate: false,
        chunk_count: 0,
        message: "Source created".to_string(),
    })
}

/// Synchronize one embedded chunk with the FTS5 and sqlite-vec indexes.
pub(crate) fn add_chunk_to_indexes<C>(
    conn: &C,
    chunk_id: i64,
    content: &str,
    embedding: &[f32],
) -> Result<()>
where
    C: std::ops::Deref<Target = rusqlite::Connection>,
{
    let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_ne_bytes()).collect();

    conn.execute(
        "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
        rusqlite::params![content, chunk_id],
    )?;

    if conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
        [],
        |row| row.get::<_, i64>(0),
    )? == 0
    {
        anyhow::ensure!(
            !embedding.is_empty(),
            "Cannot create chunks_vec without embedding dimensions"
        );
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{}])",
            embedding.len()
        ))?;
    }

    conn.execute(
        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )?;
    Ok(())
}

/// Add chunks for a source.
pub fn add_chunks(source_id: i64, chunks: Vec<ChunkData>) -> Result<i32> {
    let mut conn = get_conn()?;
    let tx = conn.transaction()?;
    let count = chunks.len() as i32;

    for chunk in &chunks {
        let embedding_bytes: Vec<u8> = chunk
            .embedding
            .iter()
            .flat_map(|f| f.to_ne_bytes())
            .collect();

        tx.execute(
            "INSERT INTO chunks (source_id, collection_id, chunk_index, content, start_pos, end_pos, chunk_type, embedding)
             SELECT ?1, collection_id, ?2, ?3, ?4, ?5, ?6, ?7
             FROM sources WHERE id = ?1",
            rusqlite::params![
                source_id,
                chunk.chunk_index,
                chunk.content,
                chunk.start_pos,
                chunk.end_pos,
                chunk.chunk_type,
                embedding_bytes,
            ],
        )?;

        // Keep both secondary indexes synchronized with the canonical chunk row.
        let chunk_id = tx.last_insert_rowid();
        add_chunk_to_indexes(&tx, chunk_id, &chunk.content, &chunk.embedding)?;
    }
    tx.commit()?;

    tracing::info!("Added {} chunks for source {}", count, source_id);
    crate::pipeline::invalidate_cache();
    Ok(count)
}

/// Update source status.
pub fn update_source_status(source_id: i64, status: String) -> Result<()> {
    let conn = get_conn()?;
    conn.execute(
        "UPDATE sources SET status = ?1 WHERE id = ?2",
        rusqlite::params![status, source_id],
    )?;
    Ok(())
}

/// Delete a source and its chunks.
pub fn delete_source(source_id: i64) -> Result<()> {
    let mut conn = get_conn()?;
    let tx = conn.transaction()?;
    let chunk_ids_sql = "SELECT id FROM chunks WHERE source_id = ?1";

    let has_vec: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec')",
        [],
        |row| row.get(0),
    )?;
    if has_vec {
        tx.execute(
            &format!("DELETE FROM chunks_vec WHERE rowid IN ({chunk_ids_sql})"),
            rusqlite::params![source_id],
        )?;
    }
    tx.execute(
        &format!("DELETE FROM chunks_fts WHERE chunk_id IN ({chunk_ids_sql})"),
        rusqlite::params![source_id],
    )?;
    tx.execute(
        &format!("DELETE FROM chunk_tags WHERE chunk_id IN ({chunk_ids_sql})"),
        rusqlite::params![source_id],
    )?;
    tx.execute(
        "DELETE FROM chunks WHERE source_id = ?1",
        rusqlite::params![source_id],
    )?;
    tx.execute(
        "DELETE FROM sources WHERE id = ?1",
        rusqlite::params![source_id],
    )?;
    tx.commit()?;
    crate::pipeline::invalidate_cache();
    Ok(())
}

/// List all sources across all collections.
pub fn list_sources() -> Result<Vec<SourceEntry>> {
    let conn = get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT id, name, created_at, metadata, status, collection_id
         FROM sources ORDER BY id DESC",
    )?;
    let sources = stmt
        .query_map([], |row| {
            Ok(SourceEntry {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                metadata: row.get(3)?,
                status: row.get(4)?,
                collection_id: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(sources)
}

/// Get adjacent chunks for neighbor expansion.
pub fn get_adjacent_chunks(
    source_id: i64,
    min_index: i32,
    max_index: i32,
) -> Result<Vec<crate::types::ChunkSearchResult>> {
    let conn = get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT id, source_id, chunk_index, content, chunk_type, metadata
         FROM chunks
         WHERE source_id = ?1 AND chunk_index >= ?2 AND chunk_index <= ?3
         ORDER BY chunk_index ASC",
    )?;
    let chunks = stmt
        .query_map(rusqlite::params![source_id, min_index, max_index], |row| {
            Ok(crate::types::ChunkSearchResult {
                chunk_id: row.get(0)?,
                source_id: row.get(1)?,
                chunk_index: row.get(2)?,
                content: row.get(3)?,
                chunk_type: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                similarity: 0.0,
                metadata: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(chunks)
}

/// RRF constant (standard value)
const RRF_K: u32 = 60;

/// Initialize FTS5 + vec0 virtual tables.
/// Called once during DB init. Safe to call multiple times (IF NOT EXISTS).
pub fn init_db() -> Result<()> {
    let conn = get_conn()?;
    // Create FTS5 virtual table mirroring chunks content
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            chunk_id UNINDEXED,
            tokenize='porter unicode61'
        );",
    )?;

    // --- sqlite-vec setup ---
    // Detect existing dimensions from chunks table
    let dims: Option<usize> = conn
        .query_row(
            "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob.len() / 4)
            },
        )
        .ok();

    if let Some(d) = dims {
        // Create vec0 virtual table with detected dimensions
        let create_vec_sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{}]);",
            d
        );
        conn.execute_batch(&create_vec_sql)?;

        // Migrate existing embeddings into chunks_vec if empty
        let vec_count: i64 = conn
            .query_row("SELECT count(rowid) FROM chunks_vec", [], |row| row.get(0))
            .unwrap_or(0);
        let chunk_count: i64 = conn
            .query_row(
                "SELECT count(id) FROM chunks WHERE embedding IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if vec_count == 0 && chunk_count > 0 {
            tracing::info!(
                "Migrating {} embeddings to chunks_vec (dims={})",
                chunk_count,
                d
            );
            conn.execute_batch("INSERT INTO chunks_vec(rowid, embedding) SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL;")?;
            tracing::info!("Embeddings migrated to chunks_vec successfully");
        }
    } else {
        tracing::info!("No embeddings found yet, chunks_vec will be created on first ingest");
    }

    // Check if FTS is populated (has rows). If empty but chunks table has rows, rebuild.
    let fts_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks_fts", [], |row| row.get(0))?;
    let chunk_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
        [],
        |row| row.get(0),
    )?;

    if fts_count == 0 && chunk_count > 0 {
        tracing::info!("Populating FTS5 index with {} chunks", chunk_count);
        rebuild_fts_index_with_conn(&conn)?;
    }

    Ok(())
}

/// Rebuild the FTS5 index from scratch.
pub fn rebuild_fts_index() -> Result<()> {
    let conn = get_conn()?;
    rebuild_fts_index_with_conn(&conn)
}

fn rebuild_fts_index_with_conn(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch("DELETE FROM chunks_fts;")?;
    conn.execute_batch(
        "INSERT INTO chunks_fts (content, chunk_id)
         SELECT content, id FROM chunks WHERE embedding IS NOT NULL;",
    )?;
    tracing::info!("FTS5 index rebuilt");
    Ok(())
}

/// Add a single chunk to the FTS5 index (for incremental ingestion).
pub fn add_to_fts_index(chunk_id: i64, content: &str) -> Result<()> {
    let conn = get_conn()?;
    conn.execute(
        "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
        rusqlite::params![content, chunk_id],
    )?;
    Ok(())
}

/// Remove chunks for a source from the FTS5 index.
pub fn remove_source_from_fts(source_id: i64) -> Result<()> {
    let conn = get_conn()?;
    conn.execute(
        "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = ?1)",
        rusqlite::params![source_id],
    )?;
    Ok(())
}

/// Vector search using sqlite-vec (fast) with brute-force fallback.
pub fn search_vector(
    query_embedding: &[f32],
    limit: usize,
    filter: Option<&SearchFilter>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let allowed_ids = filtered_ids_for_filter(&conn, filter)?;
    search_vector_with_allowed_ids(query_embedding, limit, filter, allowed_ids.as_ref())
}

fn search_vector_with_allowed_ids(
    query_embedding: &[f32],
    limit: usize,
    filter: Option<&SearchFilter>,
    allowed_ids: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<(i64, f64)>> {
    // Try sqlite-vec first (fast path)
    if let Ok(results) = search_vector_sqlite_vec(query_embedding, limit, allowed_ids) {
        return Ok(results);
    }
    // Fallback to brute-force if sqlite-vec fails
    tracing::debug!("[vector_search] sqlite-vec unavailable, using brute-force fallback");
    search_vector_brute_force(query_embedding, limit, filter)
}

/// Fast vector search using sqlite-vec extension.
fn search_vector_sqlite_vec(
    query_embedding: &[f32],
    limit: usize,
    allowed_ids: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let query_bytes: Vec<u8> = query_embedding
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();
    let mut candidate_limit = limit.max(1);

    // sqlite-vec cannot apply the relational filter inside the KNN scan. Fetch
    // progressively larger windows until the filtered result is complete or
    // the index is exhausted, so narrow filters do not lose low-ranked matches.
    let scored = loop {
        let mut stmt = conn.prepare(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance"
        )?;
        let rows = stmt.query_map(
            rusqlite::params![&query_bytes, candidate_limit as i64],
            |row| {
                let chunk_id: i64 = row.get(0)?;
                let distance: f64 = row.get(1)?;
                Ok((chunk_id, 1.0 - distance))
            },
        )?;
        let raw: Vec<(i64, f64)> = rows.collect::<rusqlite::Result<_>>()?;
        let filtered: Vec<(i64, f64)> = raw
            .iter()
            .filter(|(id, _)| allowed_ids.is_none_or(|ids| ids.contains(id)))
            .copied()
            .take(limit)
            .collect();

        if allowed_ids.is_none() || filtered.len() >= limit || raw.len() < candidate_limit {
            break filtered;
        }
        candidate_limit = candidate_limit.saturating_mul(2);
    };

    tracing::debug!(
        "[vector_search] sqlite-vec returned {} filtered results",
        scored.len()
    );
    Ok(scored)
}

/// Brute-force vector search (fallback when sqlite-vec is unavailable).
fn search_vector_brute_force(
    query_embedding: &[f32],
    limit: usize,
    filter: Option<&SearchFilter>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let (where_clause, params) = build_filter_clause(filter);

    // Fetch all embeddings matching the filter
    let sql = format!(
        "SELECT c.id, c.embedding
         FROM chunks c
         LEFT JOIN sources s ON c.source_id = s.id
         WHERE c.embedding IS NOT NULL {}
         ORDER BY c.id",
        where_clause
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
        let id: i64 = row.get(0)?;
        let emb_bytes: Vec<u8> = row.get(1)?;
        Ok((id, emb_bytes))
    })?;

    // Compute cosine similarity for each
    let mut scored: Vec<(i64, f64)> = Vec::new();
    for row in rows.collect::<rusqlite::Result<Vec<_>>>()? {
        let embedding: Vec<f32> = row
            .1
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sim = cosine_similarity(query_embedding, &embedding);
        scored.push((row.0, sim));
    }

    // Sort by score descending and take top-k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    tracing::debug!(
        "[vector_search] {} candidates, returning top {}",
        scored.len(),
        limit
    );
    Ok(scored)
}

/// BM25 search using FTS5.
pub fn search_bm25(
    query: &str,
    limit: usize,
    filter: Option<&SearchFilter>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let allowed_ids = filtered_ids_for_filter(&conn, filter)?;
    search_bm25_with_allowed_ids(query, limit, allowed_ids.as_ref())
}

fn search_bm25_with_allowed_ids(
    query: &str,
    limit: usize,
    allowed_ids: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let mut candidate_limit = limit.max(1);

    loop {
        // FTS5 BM25 search — returns negative BM25 score (more negative = better)
        let mut stmt = conn.prepare(
            "SELECT chunk_id, bm25(chunks_fts) as score
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![query, candidate_limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let score: f64 = row.get(1)?;
            Ok((id, -score))
        })?;
        let raw: Vec<(i64, f64)> = rows.collect::<rusqlite::Result<_>>()?;
        let result: Vec<(i64, f64)> = raw
            .iter()
            .filter(|(id, _)| allowed_ids.is_none_or(|ids| ids.contains(id)))
            .copied()
            .take(limit)
            .collect();

        if allowed_ids.is_none() || result.len() >= limit || raw.len() < candidate_limit {
            tracing::debug!("[bm25_search] returning top {}", result.len());
            return Ok(result);
        }
        candidate_limit = candidate_limit.saturating_mul(2);
    }
}

/// Hybrid search combining vector + BM25 via Reciprocal Rank Fusion (RRF).
pub fn search_hybrid(
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: usize,
    filter: Option<SearchFilter>,
) -> Result<Vec<SearchResult>> {
    tracing::info!("[hybrid] Starting hybrid search, top_k: {}", top_k);

    let candidate_k = if filter.is_some() {
        top_k * 4
    } else {
        top_k * 2
    };

    // 1. Resolve the relational filter once and share it between retrieval paths.
    // This avoids materializing the same potentially large ID set twice.
    let conn = get_conn()?;
    let allowed_ids = filtered_ids_for_filter(&conn, filter.as_ref())?;
    let vector_results = search_vector_with_allowed_ids(
        &query_embedding,
        candidate_k,
        filter.as_ref(),
        allowed_ids.as_ref(),
    )?;
    let bm25_results =
        search_bm25_with_allowed_ids(&query_text, candidate_k, allowed_ids.as_ref())?;

    tracing::info!(
        "[hybrid] Raw candidates - Vector: {}, BM25: {}",
        vector_results.len(),
        bm25_results.len()
    );

    // 2. RRF fusion
    let mut vector_ranks: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (rank, (id, _)) in vector_results.iter().enumerate() {
        vector_ranks.insert(*id, rank + 1);
    }

    let mut bm25_ranks: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (rank, (id, _)) in bm25_results.iter().enumerate() {
        bm25_ranks.insert(*id, rank + 1);
    }

    // Merge all doc IDs
    let mut all_doc_ids: Vec<i64> = vector_ranks
        .keys()
        .chain(bm25_ranks.keys())
        .copied()
        .collect();
    all_doc_ids.sort();
    all_doc_ids.dedup();

    if all_doc_ids.is_empty() {
        return Ok(vec![]);
    }

    // Compute RRF scores
    let mut rrf_scores: Vec<(i64, f64, u32, u32)> = Vec::with_capacity(all_doc_ids.len());
    for doc_id in &all_doc_ids {
        let vec_rank = vector_ranks.get(doc_id).copied();
        let bm25_rank = bm25_ranks.get(doc_id).copied();

        let mut combined_score = 0.0;
        if let Some(rank) = vec_rank {
            combined_score += rrf_score(rank, RRF_K);
        }
        if let Some(rank) = bm25_rank {
            combined_score += rrf_score(rank, RRF_K);
        }

        rrf_scores.push((
            *doc_id,
            combined_score,
            vec_rank.unwrap_or(0) as u32,
            bm25_rank.unwrap_or(0) as u32,
        ));
    }

    rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rrf_scores.truncate(top_k);

    // 3. Batch fetch content/metadata
    if rrf_scores.is_empty() {
        return Ok(vec![]);
    }

    let id_list: Vec<String> = rrf_scores
        .iter()
        .map(|(id, _, _, _)| id.to_string())
        .collect();
    let id_str = id_list.join(",");

    let conn = get_conn()?;
    let sql = format!(
        "SELECT c.id, c.content, c.source_id, s.metadata, c.chunk_index
         FROM chunks c
         LEFT JOIN sources s ON c.source_id = s.id
         WHERE c.id IN ({})",
        id_str
    );

    let mut stmt = conn.prepare(&sql)?;
    let mut content_map: std::collections::HashMap<i64, (String, i64, Option<String>, u32)> =
        std::collections::HashMap::new();
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, u32>(4)?,
        ))
    })?;
    for row in rows.collect::<rusqlite::Result<Vec<_>>>()? {
        content_map.insert(row.0, (row.1, row.2, row.3, row.4));
    }

    // 4. Assemble results
    let mut results: Vec<SearchResult> = Vec::with_capacity(rrf_scores.len());
    for (doc_id, score, vec_rank, bm25_rank) in rrf_scores {
        if let Some((content, source_id, metadata, chunk_index)) = content_map.remove(&doc_id) {
            results.push(SearchResult {
                doc_id,
                content,
                score,
                vector_rank: vec_rank,
                bm25_rank,
                source_id,
                metadata,
                chunk_index,
            });
        }
    }

    tracing::info!("[hybrid] Returning {} results", results.len());
    Ok(results)
}

// ── Helpers ──

fn rrf_score(rank: usize, k: u32) -> f64 {
    1.0 / (k as f64 + rank as f64)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)) as f64
}

fn build_filter_clause(filter: Option<&SearchFilter>) -> (String, Vec<rusqlite::types::Value>) {
    let mut conditions = Vec::new();
    let mut params: Vec<rusqlite::types::Value> = Vec::new();

    if let Some(f) = filter {
        if let Some(ref collection_id) = f.collection_id {
            if !collection_id.trim().is_empty() {
                conditions.push("c.collection_id = ?".to_string());
                params.push(rusqlite::types::Value::Text(collection_id.clone()));
            }
        }
        if let Some(ref sids) = f.source_ids {
            if !sids.is_empty() {
                let placeholders: Vec<&str> = sids.iter().map(|_| "?").collect();
                conditions.push(format!("c.source_id IN ({})", placeholders.join(",")));
                for sid in sids {
                    params.push(rusqlite::types::Value::Integer(*sid));
                }
            }
        }
        if let Some(ref cids) = f.chunk_ids {
            if !cids.is_empty() {
                let placeholders: Vec<&str> = cids.iter().map(|_| "?").collect();
                conditions.push(format!("c.id IN ({})", placeholders.join(",")));
                for cid in cids {
                    params.push(rusqlite::types::Value::Integer(*cid));
                }
            }
        }
        if let Some(ref pattern) = f.metadata_like {
            if !pattern.trim().is_empty() {
                conditions.push("s.metadata LIKE ?".to_string());
                params.push(rusqlite::types::Value::Text(pattern.clone()));
            }
        }
    }

    let clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" AND {}", conditions.join(" AND "))
    };

    (clause, params)
}

fn filtered_ids_for_filter(
    conn: &rusqlite::Connection,
    filter: Option<&SearchFilter>,
) -> Result<Option<std::collections::HashSet<i64>>> {
    filter
        .map(|f| get_filtered_chunk_ids_with_conn(conn, f))
        .transpose()
}

fn get_filtered_chunk_ids_with_conn(
    conn: &rusqlite::Connection,
    filter: &SearchFilter,
) -> Result<std::collections::HashSet<i64>> {
    let (where_clause, params) = build_filter_clause(Some(filter));
    let sql = format!(
        "SELECT c.id FROM chunks c LEFT JOIN sources s ON c.source_id = s.id WHERE 1=1 {}",
        where_clause
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
        row.get::<_, i64>(0)
    })?;
    let ids: std::collections::HashSet<i64> = rows
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .collect();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_stable_sha256() {
        assert_eq!(
            hash_content("rag-ferrite"),
            "ce28433c5f2359d0ce622a0149c0f0989eea6d5e8fef331c584b754e77dc448d"
        );
    }

    #[test]
    fn filtered_ids_use_the_existing_connection() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE sources (id INTEGER PRIMARY KEY, metadata TEXT);
             CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_id INTEGER, collection_id TEXT);",
        )
        .unwrap();
        conn.execute("INSERT INTO sources (id, metadata) VALUES (1, 'keep')", [])
            .unwrap();
        conn.execute(
            "INSERT INTO chunks (id, source_id, collection_id) VALUES (7, 1, 'docs')",
            [],
        )
        .unwrap();

        let filter = SearchFilter {
            collection_id: Some("docs".to_string()),
            metadata_like: Some("keep".to_string()),
            ..Default::default()
        };
        let ids = filtered_ids_for_filter(&conn, Some(&filter)).unwrap();
        assert_eq!(ids.unwrap().into_iter().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn no_filter_does_not_materialize_an_allowed_id_set() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        assert!(filtered_ids_for_filter(&conn, None).unwrap().is_none());
    }

    #[test]
    fn filtered_candidates_preserve_matches_outside_initial_window() {
        let allowed: std::collections::HashSet<i64> = [99_i64].into_iter().collect();
        let candidates = [(1_i64, 1.0), (2, 0.9), (99, 0.8)];
        let filtered: Vec<_> = candidates
            .iter()
            .filter(|(id, _)| allowed.contains(id))
            .copied()
            .collect();
        assert_eq!(filtered, vec![(99, 0.8)]);
    }
}
