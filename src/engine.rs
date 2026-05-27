use anyhow::Result;
use std::time::Instant;
use rag_engine::api::{
    db_pool,
    hybrid_search,
    simple,
    source_rag::{self, ChunkData, ChunkSearchResult},
};
use rag_engine::api::source_rag::DEFAULT_COLLECTION_ID;
use crate::chunker;
use crate::embedding::EmbeddingProvider;
use crate::extractor;
use crate::llm::{ContextResult, LlmProvider};
use crate::types::{BenchmarkDetail, BenchmarkResult, ChunkVerification, GoldenEntry, IngestionReport};

/// Stored DB path so list_sources/stats can query across all collections.
static DB_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Get the data directory from DB_PATH.
fn data_dir() -> String {
    DB_PATH.get()
        .map(|p| std::path::Path::new(p).parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_else(|| ".".to_string()))
        .unwrap_or_else(|| ".".to_string())
}

/// Initialize rag_engine: logger + DB pool + schema + reranker
pub fn init(data_dir: &std::path::Path, config: &crate::config::Config) -> Result<()> {
    simple::init_core();
    let db_path = data_dir.join("rag.sqlite3");
    std::fs::create_dir_all(data_dir)?;
    let db_path_str = db_path.to_string_lossy().to_string();
    db_pool::init_db_pool(db_path_str.clone(), 4)?;
    source_rag::init_source_db()?;

    // Migration: add section_path column to chunks (backward-compatible)
    let conn = rusqlite::Connection::open(&db_path_str)?;
    let has_section_path: bool = conn.prepare("SELECT section_path FROM chunks LIMIT 1").is_ok();
    if !has_section_path {
        tracing::info!("Migrating: adding section_path column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN section_path TEXT DEFAULT NULL")?;
    }

    // Migration: add page column to chunks (backward-compatible)
    let has_page: bool = conn.prepare("SELECT page FROM chunks LIMIT 1").is_ok();
    if !has_page {
        tracing::info!("Migrating: adding page column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN page INTEGER DEFAULT NULL")?;
    }
    drop(conn);

    // Create chunk_tags table for auto-tagging
    create_chunk_tags_table(&db_path_str)?;

    let _ = DB_PATH.set(db_path_str);
    tracing::info!("rag_engine DB initialized at {}", db_path.display());

    Ok(())
}

/// Ingest a text document into the RAG
pub async fn ingest_text(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    content: &str,
    source_name: &str,
    metadata: Option<&str>,
    collection: Option<&str>,
    max_concurrent: usize,
    relevance_scoring: bool,
    min_relevance_score: f32,
) -> Result<(i64, IngestionReport)> {
    let total_start = Instant::now();
    let collection_id = collection.unwrap_or(DEFAULT_COLLECTION_ID).to_string();
    let meta = metadata.map(|m| m.to_string()).unwrap_or_default();
    let source = source_rag::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?;

    // Custom recursive character chunker (faster, no freeze on large docs)
    let chunk_size = 800;

    let single_section = if content.len() < chunk_size {
        // Even for single-chunk docs, extract section path from the beginning
        let sections = chunker::extract_sections(content);
        chunker::find_section_for_position(&sections, 0)
    } else {
        None
    };

    // Skip chunking for small sources — single chunk is better than 2 tiny ones
    let chunks = if content.len() < chunk_size {
        tracing::info!("Source below chunk size ({} chars), ingesting as single chunk", content.len());
        vec![chunker::Chunk {
            content: content.trim().to_string(),
            index: 0,
            start_pos: 0,
            end_pos: content.len() as i32,
chunk_type: chunker::detect_chunk_type(content.trim(), true),
            section_path: single_section,
            page: None,
        }]
    } else {
        chunker::chunk_text(content, chunk_size)
    };
    tracing::info!("Chunked into {} chunks (size={})", chunks.len(), chunk_size);

    // Post-chunking verification
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let verification = verify_chunks(&chunk_texts, source_name);
    if !verification.warnings.is_empty() {
        for warning in &verification.warnings {
            tracing::warn!("Chunk verification: {}", warning);
        }
    }

    // Contextual retrieval: generate context prefixes + relevance scores via LLM
    let mut context_failures = 0usize;
    let llm_start = Instant::now();
    let context_results: Vec<ContextResult> = if let Some(llm_provider) = llm {
        tracing::info!("Generating context prefixes for {} chunks via LLM...", chunks.len());

        // Process in batches of 20 for rate limiting
        let mut all_results: Vec<ContextResult> = Vec::with_capacity(chunks.len());
        for batch in chunk_texts.chunks(20) {
            let results = llm_provider.generate_context_batch(content, batch, max_concurrent).await;
            for result in results {
                match result {
                    Ok(ctx_result) => {
                        if ctx_result.context.is_none() {
                            context_failures += 1;
                        }
                        all_results.push(ctx_result)
                    }
                    Err(e) => {
                        tracing::warn!("Context generation failed for chunk: {}, using raw content", e);
                        context_failures += 1;
                        all_results.push(ContextResult { context: None, relevance_score: None, extracted_metadata: None, tags: Vec::new() });
                    }
                }
            }
        }
        let with_ctx = all_results.iter().filter(|c| c.context.is_some()).count();
        tracing::info!("Generated {}/{} context prefixes", with_ctx, chunks.len());
        all_results
    } else {
        vec![ContextResult { context: None, relevance_score: None, extracted_metadata: None, tags: Vec::new() }; chunks.len()]
    };
    let llm_duration_ms = llm_start.elapsed().as_millis() as u64;

    // Filter chunks by relevance score if enabled
    let mut filtered_count = 0usize;
    let kept: Vec<(usize, &chunker::Chunk, &ContextResult)> = chunks
        .iter()
        .enumerate()
        .zip(context_results.iter())
        .filter(|((_, _), ctx_result)| {
            if relevance_scoring {
                if let Some(score) = ctx_result.relevance_score {
                    if score < min_relevance_score {
                        filtered_count += 1;
                        tracing::info!("Filtered chunk (score={:.1} < threshold={:.1})", score, min_relevance_score);
                        return false;
                    }
                }
            }
            true
        })
        .map(|((idx, chunk), ctx_result)| (idx, chunk, ctx_result))
        .collect();

    // Compute relevance statistics from all context results
    let relevance_scores: Vec<f64> = context_results
        .iter()
        .filter_map(|c| c.relevance_score.map(|s| s as f64))
        .collect();
    let avg_relevance = if relevance_scores.is_empty() { 0.0 } else { relevance_scores.iter().sum::<f64>() / relevance_scores.len() as f64 };
    let min_relevance = relevance_scores.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let max_relevance = relevance_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0);

    if filtered_count > 0 {
        tracing::info!("Relevance scoring: filtered {}/{} chunks (threshold={:.1})", filtered_count, chunks.len(), min_relevance_score);
    }

    if kept.is_empty() {
        tracing::warn!("All chunks filtered by relevance scoring, ingesting anyway");
    }

    // Build final texts for embedding: context prefix + chunk content
    let final_texts: Vec<String> = kept
        .iter()
        .map(|(_, chunk, ctx_result)| {
            match &ctx_result.context {
                Some(context) => format!("{}\n\n{}", context, chunk.content),
                None => chunk.content.clone(),
            }
        })
        .collect();

    // Batch embed all chunks (with context prefixes)
    let embed_start = Instant::now();
    let embeddings = embedder.embed_batch(&final_texts).await?;
    let embedding_duration_ms = embed_start.elapsed().as_millis() as u64;

    // Store original chunk content (not the prefixed version)
    // Collect section_paths and pages for post-insert UPDATE (rag_engine ChunkData doesn't have these)
    let section_paths: Vec<Option<String>> = kept
        .iter()
        .map(|(_, chunk, _)| chunk.section_path.clone())
        .collect();
    let pages: Vec<Option<u32>> = kept
        .iter()
        .map(|(_, chunk, _)| chunk.page)
        .collect();

    // Collect auto-generated tags before kept is consumed (for chunk_tags table)
    let tags_per_chunk: Vec<Vec<String>> = kept
        .iter()
        .map(|(_, _, ctx_result)| ctx_result.tags.clone())
        .collect();

    let chunk_data: Vec<ChunkData> = kept
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|((_, chunk, _), emb)| ChunkData {
            content: chunk.content.clone(),
            chunk_index: chunk.index,
            start_pos: chunk.start_pos,
            end_pos: chunk.end_pos,
            chunk_type: chunk.chunk_type.as_str().to_string(),
            embedding: emb,
        })
        .collect();

    let count = source_rag::add_chunks(source.source_id, chunk_data)?;
    tracing::info!("Ingested {} chunks for source {} ({})", count, source.source_id, source_name);

    // Store section_path for each chunk (separate UPDATE since rag_engine doesn't know about it)
    update_chunk_section_paths(source.source_id, &section_paths)?;
    update_chunk_pages(source.source_id, &pages)?;

    // Store auto-generated tags for each chunk in chunk_tags table
    if tags_per_chunk.iter().any(|t| !t.is_empty()) {
        insert_chunk_tags(source.source_id, &tags_per_chunk)?;
    }

    // Mark source as completed
    if let Err(e) = source_rag::update_source_status(source.source_id, "completed".to_string()) {
        tracing::warn!("Failed to update source status: {}", e);
    }

    // Rebuild indexes for the target collection
    if let Err(e) = source_rag::rebuild_chunk_hnsw_index_for_collection(collection_id.clone()) {
        tracing::warn!("Failed to rebuild HNSW index for {}: {}", collection_id, e);
    }
    if let Err(e) = source_rag::rebuild_chunk_bm25_index_for_collection(collection_id.clone()) {
        tracing::warn!("Failed to rebuild BM25 index for {}: {}", collection_id, e);
    }

    // Persist HNSW index to disk for fast startup
    let index_path = format!("{}/hnsw_{}.index", data_dir(), collection_id);
    if let Err(e) = source_rag::save_collection_hnsw_index(collection_id.clone(), index_path) {
        tracing::warn!("Failed to save HNSW index: {}", e);
    }

    let total_duration_ms = total_start.elapsed().as_millis() as u64;

    let report = IngestionReport {
        total_chunks: chunks.len(),
        filtered_chunks: filtered_count,
        avg_relevance,
        min_relevance,
        max_relevance,
        context_failures,
        total_duration_ms,
        embedding_duration_ms,
        llm_duration_ms,
        source_name: source_name.to_string(),
    };

    Ok((source.source_id, report))
}

/// Ingest a file (PDF, TXT, MD)
/// Uses our custom extractor for reliable text extraction
pub async fn ingest_file(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    file_path: &str,
    collection: Option<&str>,
    max_concurrent: usize,
    relevance_scoring: bool,
    min_relevance_score: f32,
) -> Result<(i64, IngestionReport)> {
    // Use our custom extractor instead of rag_engine's document_parser
    let text = extractor::extract_text(file_path)?;

    let name = std::path::Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.to_string());
    ingest_text(
        embedder,
        llm,
        &text,
        &name,
        Some(&format!("{{\"path\":\"{}\"}}", file_path)),
        collection,
        max_concurrent,
        relevance_scoring,
        min_relevance_score,
    )
    .await
}

/// Update section_path for all chunks of a source, matched by chunk_index.
fn update_chunk_section_paths(source_id: i64, section_paths: &[Option<String>]) -> Result<()> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    for (idx, path) in section_paths.iter().enumerate() {
        if let Some(sp) = path {
            conn.execute(
                "UPDATE chunks SET section_path = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![sp, source_id, idx as i32],
            )?;
        }
    }

    Ok(())
}

/// Update page for all chunks of a source, matched by chunk_index.
fn update_chunk_pages(source_id: i64, pages: &[Option<u32>]) -> Result<()> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;
    for (idx, page) in pages.iter().enumerate() {
        if let Some(p) = page {
            conn.execute(
                "UPDATE chunks SET page = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![*p as i64, source_id, idx as i32],
            )?;
        }
    }
    Ok(())
}

/// Fetch section_path for a batch of chunk IDs.
pub fn get_section_paths_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    let mut map = std::collections::HashMap::new();
    for &id in chunk_ids {
        let sp: Option<String> = conn
            .query_row(
                "SELECT section_path FROM chunks WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        map.insert(id, sp);
    }
    Ok(map)
}

/// Delete a source by ID
pub fn delete_source(source_id: i64) -> Result<()> {
    // Look up the collection before deleting, so we can rebuild its indexes
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;
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
        let conn = rusqlite::Connection::open(db_path)?;
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

/// Search with hybrid fusion (BM25 + vector + RRF)
/// Optionally expands short queries via LLM for better retrieval.
pub async fn search_hybrid(
    embedder: &EmbeddingProvider,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    search_hybrid_with_expansion(embedder, None, query, limit, filter).await
}

/// Search with optional query expansion for short/ambiguous queries.
pub async fn search_hybrid_with_expansion(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    // Activate the correct collection's indexes before searching
    if let Some(ref f) = filter {
        if let Some(ref coll) = f.collection_id {
            let index_path = format!("{}/hnsw_{}.index", data_dir(), coll);
            if let Err(e) = source_rag::activate_collection_for_hybrid_search(coll.clone(), index_path) {
                tracing::warn!("Failed to activate collection '{}': {}", coll, e);
            }
        }
    }

    // Expand short queries (< 5 words) if LLM is available
    let queries = if let Some(llm_provider) = llm {
        let word_count = query.split_whitespace().count();
        if word_count <= 5 {
            match llm_provider.expand_query(query).await {
                Ok(expansions) => {
                    tracing::info!("Query expansion: {:?}", expansions);
                    expansions
                }
                Err(e) => {
                    tracing::warn!("Query expansion failed: {}, using original", e);
                    vec![query.to_string()]
                }
            }
        } else {
            vec![query.to_string()]
        }
    } else {
        vec![query.to_string()]
    };

    // Run hybrid search for each query variant
    let mut all_results: Vec<hybrid_search::HybridSearchResult> = Vec::new();
    let mut seen_doc_ids = std::collections::HashSet::new();

    for q in &queries {
        let query_embedding = embedder.embed(q).await?;
        let filter_clone = filter.clone();

        if let Ok(results) = hybrid_search::search_hybrid(
            q.to_string(),
            query_embedding,
            limit as u32,
            None,
            filter_clone,
        ) {
            for result in results {
                // Deduplicate by doc_id
                if seen_doc_ids.insert(result.doc_id) {
                    all_results.push(result);
                }
            }
        }
    }

    // Sort by score descending
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(limit);

    Ok(all_results)
}

/// Fetch page for a batch of chunk IDs.
pub fn get_pages_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<u32>>> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;
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

/// List all sources across all collections.
///
/// Queries the `sources` table directly instead of using
/// `source_rag::list_sources()` which hardcodes the `__default__` collection.
pub fn list_sources() -> Result<Vec<source_rag::SourceEntry>> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

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

/// Get stats across all collections.
pub fn stats() -> Result<Stats> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    let count: usize = conn.query_row(
        "SELECT COUNT(*) FROM sources",
        [],
        |row| row.get::<_, i64>(0),
    )? as usize;

    Ok(Stats {
        document_count: count,
    })
}

pub struct Stats {
    pub document_count: usize,
}

/// Pre-ingestion document quality check.
/// Analyzes content and returns a report before committing to chunking+embedding.
pub fn pre_check_document(content: &str, filename: &str) -> crate::types::PreCheckReport {
    let mut warnings = Vec::new();

    let char_count = content.len();

    // Extraction check: non-empty after trimming
    let extraction_ok = !content.trim().is_empty();

    // Empty content warning
    if char_count < 100 {
        warnings.push(format!("Very short content ({} chars), may not provide useful retrieval results", char_count));
    }

    // Size warning
    if char_count > 500_000 {
        warnings.push(format!("Large document ({} chars), ingestion may take a while", char_count));
    }

    // Estimated chunks (matching the 800-char chunk size used in ingest_text)
    let chunk_size = 800;
    let estimated_chunks = if char_count == 0 {
        0
    } else if char_count < chunk_size {
        1
    } else {
        (char_count + chunk_size - 1) / chunk_size
    };

    // Language detection via simple heuristic
    let language = detect_language(content);

    // Duplicate detection: check if a source with the same name already exists
    let is_duplicate = check_duplicate_source(filename);
    if is_duplicate {
        warnings.push(format!("A document named '{}' already exists in the index", filename));
    }

    crate::types::PreCheckReport {
        extraction_ok,
        char_count,
        estimated_chunks,
        language,
        is_duplicate,
        warnings,
    }
}

/// Simple language detection heuristic based on character frequency.
fn detect_language(text: &str) -> String {
    let sample = &text[..text.len().min(5000)];
    let mut french_accents = 0usize;
    let mut cjk_chars = 0usize;
    let mut latin_chars = 0usize;
    let mut arabic_chars = 0usize;
    let mut cyrillic_chars = 0usize;

    for ch in sample.chars() {
        match ch {
            'à' | 'â' | 'é' | 'è' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'ù' | 'û' | 'ü' | 'ÿ' | 'ç'
            | 'À' | 'Â' | 'É' | 'È' | 'Ê' | 'Ë' | 'Î' | 'Ï' | 'Ô' | 'Ù' | 'Û' | 'Ü' | 'Ÿ' | 'Ç' => {
                french_accents += 1;
                latin_chars += 1;
            }
            c if c >= '\u{4E00}' && c <= '\u{9FFF}' => cjk_chars += 1,
            c if c >= '\u{3040}' && c <= '\u{309F}' || c >= '\u{30A0}' && c <= '\u{30FF}' => cjk_chars += 1,
            c if c >= '\u{0600}' && c <= '\u{06FF}' || c >= '\u{0750}' && c <= '\u{077F}' => arabic_chars += 1,
            c if c >= '\u{0400}' && c <= '\u{04FF}' => cyrillic_chars += 1,
            'a'..='z' | 'A'..='Z' => latin_chars += 1,
            _ => {}
        }
    }

    let total = latin_chars + cjk_chars + arabic_chars + cyrillic_chars;
    if total == 0 {
        return "unknown".to_string();
    }

    // CJK detection
    if cjk_chars as f64 / total as f64 > 0.3 {
        return "cjk".to_string();
    }

    // Arabic detection
    if arabic_chars as f64 / total as f64 > 0.3 {
        return "arabic".to_string();
    }

    // Cyrillic detection
    if cyrillic_chars as f64 / total as f64 > 0.3 {
        return "cyrillic".to_string();
    }

    // French vs English: if notable French accent count, assume French
    if french_accents >= 3 {
        return "french".to_string();
    }

    "english".to_string()
}

/// Check if a source with the given name already exists in the DB.
fn check_duplicate_source(filename: &str) -> bool {
    let db_path = match DB_PATH.get() {
        Some(p) => p,
        None => return false,
    };
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sources WHERE name = ?1",
            rusqlite::params![filename],
            |row| row.get(0),
        )
        .unwrap_or(0);
    count > 0
}

/// Run a benchmark against a golden dataset.
/// For each entry, queries the engine and checks if expected source_ids appear in top results.
pub async fn run_benchmark(
    embedder: &EmbeddingProvider,
    entries: Vec<GoldenEntry>,
    collection: Option<String>,
    limit: usize,
) -> Result<BenchmarkResult> {
    let mut details = Vec::with_capacity(entries.len());
    let mut total_score = 0.0;
    let mut hits = 0usize;

    for entry in &entries {
        let filter = if collection.is_some() {
            Some(hybrid_search::SearchFilter {
                source_ids: None,
                metadata_like: None,
                collection_id: collection.clone(),
            })
        } else {
            None
        };

        let results = search_hybrid(embedder, &entry.question, limit, filter).await.unwrap_or_default();

        // Collect unique source_ids from results
        let found_ids: Vec<i64> = results
            .iter()
            .map(|r| r.source_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Score: fraction of expected source_ids that appear in found_ids
        let matched = entry.relevant_source_ids.iter().filter(|id| found_ids.contains(id)).count();
        let score = if entry.relevant_source_ids.is_empty() {
            0.0
        } else {
            matched as f64 / entry.relevant_source_ids.len() as f64
        };
        let is_hit = matched > 0;

        if is_hit {
            hits += 1;
        }
        total_score += score;

        details.push(BenchmarkDetail {
            query: entry.question.clone(),
            expected_source_ids: entry.relevant_source_ids.clone(),
            found_source_ids: found_ids,
            score,
            is_hit,
        });
    }

    let total_queries = entries.len();
    let misses = total_queries - hits;
    let avg_score = if total_queries > 0 {
        total_score / total_queries as f64
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        total_queries,
        hits,
        misses,
        avg_score,
        details,
    })
}

/// Graph data for document similarity visualization.
pub fn get_graph_data(
    collection: Option<&str>,
    threshold: f32,
    max_edges: usize,
) -> Result<crate::types::GraphData> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    // 1. Get sources, optionally filtered by collection
    let sources: Vec<(i64, Option<String>, String, i32)> = if let Some(coll) = collection {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s WHERE s.collection_id = ?1 ORDER BY s.id",
        )?;
        stmt.query_map(rusqlite::params![coll], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s ORDER BY s.id",
        )?;
        stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    };

    if sources.is_empty() {
        return Ok(crate::types::GraphData {
            nodes: vec![],
            edges: vec![],
        });
    }

    // Build nodes
    let nodes: Vec<crate::types::GraphNode> = sources
        .iter()
        .map(|(id, name, collection_id, chunk_count)| crate::types::GraphNode {
            id: *id,
            name: name.clone().unwrap_or_else(|| format!("doc_{}", id)),
            collection: collection_id.clone(),
            chunk_count: *chunk_count,
        })
        .collect();

    let source_ids: Vec<i64> = sources.iter().map(|(id, _, _, _)| *id).collect();

    // 2. Load chunk embeddings per source and compute centroids
    let mut centroids: std::collections::HashMap<i64, Vec<f32>> = std::collections::HashMap::new();

    for source_id in &source_ids {
        let mut stmt = conn.prepare(
            "SELECT embedding FROM chunks WHERE source_id = ?1",
        )?;
        let embeddings: Vec<Vec<f32>> = stmt
            .query_map(rusqlite::params![source_id], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(decode_f32_embedding(&blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|v| if v.is_empty() { None } else { Some(v) })
            .collect();

        if embeddings.is_empty() {
            continue;
        }

        // Compute centroid (average of all chunk embeddings)
        let dims = embeddings[0].len();
        let mut centroid = vec![0.0f32; dims];
        let count = embeddings.len() as f32;
        for emb in &embeddings {
            for (i, val) in emb.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in centroid.iter_mut() {
            *val /= count;
        }
        centroids.insert(*source_id, centroid);
    }

    // 3. Compute pairwise cosine similarity
    let mut edges: Vec<crate::types::GraphEdge> = Vec::new();
    let ids_with_centroids: Vec<i64> = source_ids
        .iter()
        .filter(|id| centroids.contains_key(id))
        .copied()
        .collect();

    for i in 0..ids_with_centroids.len() {
        for j in (i + 1)..ids_with_centroids.len() {
            let id_a = ids_with_centroids[i];
            let id_b = ids_with_centroids[j];
            let a = &centroids[&id_a];
            let b = &centroids[&id_b];
            let sim = cosine_similarity(a, b);
            if sim >= threshold {
                edges.push(crate::types::GraphEdge {
                    source: id_a,
                    target: id_b,
                    similarity: (sim * 10000.0).round() / 10000.0, // 4 decimal places
                });
            }
        }
    }

    // 4. Sort by similarity desc, keep max_edges
    edges.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges.truncate(max_edges);

    Ok(crate::types::GraphData { nodes, edges })
}

/// Post-chunking verification: checks coverage, empty chunks, and logs warnings.
fn verify_chunks(chunks: &[String], source: &str) -> ChunkVerification {
    let total_chunks = chunks.len();
    let source_chars = source.len();
    let chunk_chars: usize = chunks.iter().map(|c| c.len()).sum();
    let coverage_ratio = if source_chars == 0 {
        1.0
    } else {
        chunk_chars as f64 / source_chars as f64
    };

    let mut warnings = Vec::new();

    // Warn on empty chunks
    let empty_count = chunks.iter().filter(|c| c.trim().is_empty()).count();
    if empty_count > 0 {
        warnings.push(format!("{} empty chunks found for source '{}'", empty_count, source));
    }

    // Warn if coverage < 90%
    if coverage_ratio < 0.9 {
        warnings.push(format!(
            "Low chunk coverage {:.1}% for source '{}' ({} source chars, {} chunk chars)",
            coverage_ratio * 100.0, source, source_chars, chunk_chars
        ));
    }

    ChunkVerification {
        total_chunks,
        source_chars,
        chunk_chars,
        coverage_ratio,
        warnings,
    }
}

/// Decode a BLOB of native-endian f32 bytes into a Vec<f32>.
fn decode_f32_embedding(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 {
        return Vec::new();
    }
    blob.chunks(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Compute cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Create the chunk_tags table if it doesn't exist.
fn create_chunk_tags_table(db_path: &str) -> Result<()> {
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
fn insert_chunk_tags(source_id: i64, tags_per_chunk: &[Vec<String>]) -> Result<()> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

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
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

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

