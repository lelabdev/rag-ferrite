use anyhow::Result;
use crate::types::{SearchResult, SearchFilter, SourceEntry, ChunkData};
use crate::engine::get_conn;

/// Default collection ID (replaces rag_engine constant)
pub const DEFAULT_COLLECTION_ID: &str = "__default__";

/// Result of adding a source (replaces rag_engine::AddSourceResult)
#[derive(Debug, Clone)]
pub struct AddSourceResult {
    pub source_id: i64,
    pub is_duplicate: bool,
    pub chunk_count: i32,
    pub message: String,
}

/// Hash content for deduplication (SHA-256 hex).
fn hash_content(input: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Add a source to a collection (replaces rag_engine::add_source_in_collection)
pub fn add_source_in_collection(
    collection_id: String,
    content: String,
    metadata: Option<String>,
    name: Option<String>,
) -> Result<AddSourceResult> {
    let scoped_hash = hash_content(&format!("{}:{}", collection_id, content));
    let conn = get_conn()?;

    // Check for duplicate
    let existing: Option<i64> = conn.query_row(
        "SELECT id FROM sources WHERE collection_id = ?1 AND content_hash = ?2",
        rusqlite::params![collection_id, scoped_hash],
        |row| row.get(0),
    ).ok();

    if let Some(id) = existing {
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
    Ok(AddSourceResult {
        source_id,
        is_duplicate: false,
        chunk_count: 0,
        message: "Source created".to_string(),
    })
}

/// Add chunks for a source (replaces rag_engine::add_chunks)
pub fn add_chunks(source_id: i64, chunks: Vec<ChunkData>) -> Result<i32> {
    let conn = get_conn()?;
    let count = chunks.len() as i32;

    for chunk in &chunks {
        let embedding_bytes: Vec<u8> = chunk.embedding.iter()
            .flat_map(|f| f.to_ne_bytes())
            .collect();

        conn.execute(
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

        // Also add to FTS5 index
        let chunk_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
            rusqlite::params![chunk.content, chunk_id],
        )?;

        // Also add to chunks_vec (sqlite-vec) for fast vector search
        // Ignore errors if chunks_vec doesn't exist yet (dims not detected)
        let _ = conn.execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
            rusqlite::params![chunk_id, embedding_bytes],
        );
    }

    tracing::info!("Added {} chunks for source {}", count, source_id);
    Ok(count)
}

/// Update source status (replaces rag_engine::update_source_status)
pub fn update_source_status(source_id: i64, status: String) -> Result<()> {
    let conn = get_conn()?;
    conn.execute(
        "UPDATE sources SET status = ?1 WHERE id = ?2",
        rusqlite::params![status, source_id],
    )?;
    Ok(())
}

/// Delete a source and its chunks (replaces rag_engine::delete_source)
pub fn delete_source(source_id: i64) -> Result<()> {
    let conn = get_conn()?;
    // Remove from FTS5 index
    conn.execute(
        "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = ?1)",
        rusqlite::params![source_id],
    )?;
    // Delete chunks
    conn.execute("DELETE FROM chunks WHERE source_id = ?1", rusqlite::params![source_id])?;
    // Delete source
    conn.execute("DELETE FROM sources WHERE id = ?1", rusqlite::params![source_id])?;
    Ok(())
}

/// List all sources across all collections (replaces rag_engine::list_sources)
pub fn list_sources() -> Result<Vec<SourceEntry>> {
    let conn = get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT id, name, created_at, metadata, status, collection_id
         FROM sources ORDER BY id DESC"
    )?;
    let sources = stmt.query_map([], |row| {
        Ok(SourceEntry {
            id: row.get(0)?,
            name: row.get(1)?,
            created_at: row.get(2)?,
            metadata: row.get(3)?,
            status: row.get(4)?,
            collection_id: row.get(5)?,
        })
    })?.filter_map(|r| r.ok()).collect();
    Ok(sources)
}

/// Get adjacent chunks for neighbor expansion (replaces rag_engine::get_adjacent_chunks)
pub fn get_adjacent_chunks(source_id: i64, min_index: i32, max_index: i32) -> Result<Vec<crate::types::ChunkSearchResult>> {
    let conn = get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT id, source_id, chunk_index, content, chunk_type, metadata
         FROM chunks
         WHERE source_id = ?1 AND chunk_index >= ?2 AND chunk_index <= ?3
         ORDER BY chunk_index ASC"
    )?;
    let chunks = stmt.query_map(rusqlite::params![source_id, min_index, max_index], |row| {
        Ok(crate::types::ChunkSearchResult {
            chunk_id: row.get(0)?,
            source_id: row.get(1)?,
            chunk_index: row.get(2)?,
            content: row.get(3)?,
            chunk_type: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            similarity: 0.0,
            metadata: row.get(5)?,
        })
    })?.filter_map(|r| r.ok()).collect();
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
        );"
    )?;

    // --- sqlite-vec setup ---
    // Detect existing dimensions from chunks table
    let dims: Option<usize> = conn.query_row(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1",
        [],
        |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob.len() / 4)
        },
    ).ok();

    if let Some(d) = dims {
        // Create vec0 virtual table with detected dimensions
        let create_vec_sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{}]);",
            d
        );
        conn.execute_batch(&create_vec_sql)?;

        // Migrate existing embeddings into chunks_vec if empty
        let vec_count: i64 = conn.query_row("SELECT count(rowid) FROM chunks_vec", [], |row| row.get(0)).unwrap_or(0);
        let chunk_count: i64 = conn.query_row("SELECT count(id) FROM chunks WHERE embedding IS NOT NULL", [], |row| row.get(0)).unwrap_or(0);

        if vec_count == 0 && chunk_count > 0 {
            tracing::info!("Migrating {} embeddings to chunks_vec (dims={})", chunk_count, d);
            conn.execute_batch("INSERT INTO chunks_vec(rowid, embedding) SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL;")?;
            tracing::info!("Embeddings migrated to chunks_vec successfully");
        }
    } else {
        tracing::info!("No embeddings found yet, chunks_vec will be created on first ingest");
    }
    
    // Check if FTS is populated (has rows). If empty but chunks table has rows, rebuild.
    let fts_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks_fts", [], |row| row.get(0))?;
    let chunk_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL", [], |row| row.get(0))?;
    
    if fts_count == 0 && chunk_count > 0 {
        tracing::info!("Populating FTS5 index with {} chunks", chunk_count);
        rebuild_fts_index()?;
    }
    
    Ok(())
}

/// Rebuild the FTS5 index from scratch.
pub fn rebuild_fts_index() -> Result<()> {
    let conn = get_conn()?;
    conn.execute_batch("DELETE FROM chunks_fts;")?;
    conn.execute_batch(
        "INSERT INTO chunks_fts (content, chunk_id)
         SELECT content, id FROM chunks WHERE embedding IS NOT NULL;"
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
    // Try sqlite-vec first (fast path)
    if let Ok(results) = search_vector_sqlite_vec(query_embedding, limit, filter) {
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
    filter: Option<&SearchFilter>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    let query_bytes: Vec<u8> = query_embedding.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let mut stmt = conn.prepare(
        "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance"
    )?;
    let rows = stmt.query_map(rusqlite::params![query_bytes, limit as i64], |row| {
        let chunk_id: i64 = row.get(0)?;
        let distance: f64 = row.get(1)?;
        Ok((chunk_id, 1.0 - distance))
    })?;
    let mut scored: Vec<(i64, f64)> = rows.filter_map(|r| r.ok()).collect();
    if let Some(f) = filter {
        let valid_ids = get_filtered_chunk_ids(f)?;
        scored.retain(|(id, _)| valid_ids.contains(id));
    }
    scored.truncate(limit);
    tracing::debug!("[vector_search] sqlite-vec returned {} results", scored.len());
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
    for row in rows.filter_map(|r| r.ok()) {
        let embedding: Vec<f32> = row.1
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sim = cosine_similarity(query_embedding, &embedding);
        scored.push((row.0, sim));
    }
    
    // Sort by score descending and take top-k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    
    tracing::debug!("[vector_search] {} candidates, returning top {}", scored.len(), limit);
    Ok(scored)
}

/// BM25 search using FTS5.
pub fn search_bm25(
    query: &str,
    limit: usize,
    filter: Option<&SearchFilter>,
) -> Result<Vec<(i64, f64)>> {
    let conn = get_conn()?;
    
    // FTS5 BM25 search — returns negative BM25 score (more negative = better)
    let fts_results: Vec<(i64, f64)> = {
        let mut stmt = conn.prepare(
            "SELECT chunk_id, bm25(chunks_fts) as score
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![query, limit * 4], |row| {
            let id: i64 = row.get(0)?;
            let score: f64 = row.get(1)?;
            // FTS5 bm25() returns negative values (more negative = better match)
            // Normalize to positive: negate it
            Ok((id, -score))
        })?;
        rows.filter_map(|r| r.ok()).collect()
    };
    
    // Apply filter if present
    let filtered = if let Some(f) = filter {
        let valid_ids = get_filtered_chunk_ids(f)?;
        fts_results.into_iter()
            .filter(|(id, _)| valid_ids.contains(id))
            .collect::<Vec<_>>()
    } else {
        fts_results
    };
    
    let mut result = filtered;
    result.truncate(limit);
    
    tracing::debug!("[bm25_search] returning top {}", result.len());
    Ok(result)
}

/// Hybrid search combining vector + BM25 via Reciprocal Rank Fusion (RRF).
pub fn search_hybrid(
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: usize,
    filter: Option<SearchFilter>,
) -> Result<Vec<SearchResult>> {
    tracing::info!("[hybrid] Starting hybrid search, top_k: {}", top_k);
    
    let candidate_k = if filter.is_some() { top_k * 4 } else { top_k * 2 };
    
    // 1. Run vector and BM25 search
    let vector_results = search_vector(&query_embedding, candidate_k, filter.as_ref())?;
    let bm25_results = search_bm25(&query_text, candidate_k, filter.as_ref())?;
    
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
    let mut all_doc_ids: Vec<i64> = vector_ranks.keys().chain(bm25_ranks.keys()).copied().collect();
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
        
        rrf_scores.push((*doc_id, combined_score, vec_rank.unwrap_or(0) as u32, bm25_rank.unwrap_or(0) as u32));
    }
    
    rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rrf_scores.truncate(top_k);
    
    // 3. Batch fetch content/metadata
    if rrf_scores.is_empty() {
        return Ok(vec![]);
    }
    
    let id_list: Vec<String> = rrf_scores.iter().map(|(id, _, _, _)| id.to_string()).collect();
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
    let mut content_map: std::collections::HashMap<i64, (String, i64, Option<String>, u32)> = std::collections::HashMap::new();
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, u32>(4)?,
        ))
    })?;
    for row in rows.filter_map(|r| r.ok()) {
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

fn get_filtered_chunk_ids(filter: &SearchFilter) -> Result<std::collections::HashSet<i64>> {
    let (where_clause, params) = build_filter_clause(Some(filter));
    let conn = get_conn()?;
    let sql = format!(
        "SELECT c.id FROM chunks c LEFT JOIN sources s ON c.source_id = s.id WHERE 1=1 {}",
        where_clause
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
        row.get::<_, i64>(0)
    })?;
    let ids: std::collections::HashSet<i64> = rows.filter_map(|r| r.ok()).collect();
    Ok(ids)
}
