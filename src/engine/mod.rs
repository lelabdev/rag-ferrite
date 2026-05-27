use anyhow::Result;
use std::time::Instant;
use rag_engine::api::{
    db_pool,
    simple,
    source_rag::{self, ChunkData},
};
use rag_engine::api::source_rag::DEFAULT_COLLECTION_ID;
use crate::chunker;
use crate::embedding::EmbeddingProvider;
use crate::extractor;
use crate::llm::{ContextResult, LlmProvider};
use crate::types::{ChunkVerification, IngestionReport};

pub mod search;
pub mod query;
pub mod benchmark;
pub mod tags;

// Re-export public items from sub-modules
pub use search::{search_hybrid, search_hybrid_with_expansion};
pub use query::{get_section_paths_for_chunk_ids, get_neighbors, delete_source, list_sources};
pub use benchmark::{run_benchmark, get_graph_data};
pub use tags::{create_chunk_tags_table, insert_chunk_tags, get_tags_for_chunk_ids};

/// Stored DB path so list_sources/stats can query across all collections.
static DB_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Centralized SQLite connection — avoids ad-hoc `Connection::open` calls.
static DB_CONN: std::sync::OnceLock<std::sync::Mutex<rusqlite::Connection>> = std::sync::OnceLock::new();

/// Obtain a locked handle to the shared SQLite connection.
pub fn get_conn() -> Result<std::sync::MutexGuard<'static, rusqlite::Connection>> {
    DB_CONN
        .get()
        .ok_or_else(|| anyhow::anyhow!("DB not initialized"))?
        .lock()
        .map_err(|e| anyhow::anyhow!("DB connection lock poisoned: {}", e))
}

/// Get the data directory from DB_PATH.
pub(crate) fn data_dir() -> String {
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

    let _ = DB_PATH.set(db_path_str.clone());

    // Store a shared connection for all subsequent get_conn() calls
    let shared_conn = rusqlite::Connection::open(&db_path_str)?;
    shared_conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
    let _ = DB_CONN.set(std::sync::Mutex::new(shared_conn));

    tracing::info!("rag_engine DB initialized at {}", db_path.display());

    Ok(())
}

/// Sanitize a collection ID: only allow alphanumeric, underscore, and hyphen.
/// Returns an error if the result is empty after sanitization.
pub fn sanitize_collection(collection: &str) -> Result<String> {
    let sanitized: String = collection
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized.is_empty() {
        anyhow::bail!("Invalid collection ID: '{}' contains no valid characters", collection);
    }
    Ok(sanitized)
}

/// Options for ingestion controlling concurrency and relevance filtering.
pub struct IngestOptions {
    pub max_concurrent: usize,
    pub relevance_scoring: bool,
    pub min_relevance_score: f64,
    pub chunk_size: usize,
}

/// Ingest a text document into the RAG
pub async fn ingest_text(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    content: &str,
    source_name: &str,
    metadata: Option<&str>,
    collection: Option<&str>,
    options: IngestOptions,
) -> Result<(i64, IngestionReport)> {
    let total_start = Instant::now();
    if content.trim().is_empty() {
        anyhow::bail!("Cannot ingest empty content");
    }
    let collection_id = sanitize_collection(collection.unwrap_or(DEFAULT_COLLECTION_ID))?;
    let meta = metadata.map(|m| m.to_string()).unwrap_or_default();
    let source = source_rag::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?;

    // Custom recursive character chunker (faster, no freeze on large docs)
    let chunk_size = options.chunk_size;

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
            let results = llm_provider.generate_context_batch(content, batch, options.max_concurrent).await;
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
            if options.relevance_scoring {
                if let Some(score) = ctx_result.relevance_score {
                    if (score as f64) < options.min_relevance_score {
                        filtered_count += 1;
                        tracing::info!("Filtered chunk (score={:.1} < threshold={:.1})", score, options.min_relevance_score);
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
    let min_relevance = if relevance_scores.is_empty() { 0.0 } else { relevance_scores.iter().cloned().fold(f64::INFINITY, f64::min) };
    let max_relevance = relevance_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0);

    if filtered_count > 0 {
        tracing::info!("Relevance scoring: filtered {}/{} chunks (threshold={:.1})", filtered_count, chunks.len(), options.min_relevance_score);
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
    // Collect section_paths, pages, and tags indexed by original chunk.index
    // (not enumerate position, which may differ after relevance filtering)
    let section_paths: Vec<(i32, Option<String>)> = kept
        .iter()
        .map(|(_, chunk, _)| (chunk.index, chunk.section_path.clone()))
        .collect();
    let pages: Vec<(i32, Option<u32>)> = kept
        .iter()
        .map(|(_, chunk, _)| (chunk.index, chunk.page))
        .collect();

    // Collect auto-generated tags before kept is consumed (for chunk_tags table)
    let tags_per_chunk: Vec<(i32, Vec<String>)> = kept
        .iter()
        .map(|(_, chunk, ctx_result)| (chunk.index, ctx_result.tags.clone()))
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
    if tags_per_chunk.iter().any(|(_, t)| !t.is_empty()) {
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
    options: IngestOptions,
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
        options,
    )
    .await
}

/// Update section_path for all chunks of a source, matched by chunk_index.
fn update_chunk_section_paths(source_id: i64, section_paths: &[(i32, Option<String>)]) -> Result<()> {
    let conn = get_conn()?;

    for &(chunk_index, ref path) in section_paths {
        if let Some(sp) = path {
            conn.execute(
                "UPDATE chunks SET section_path = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![sp, source_id, chunk_index],
            )?;
        }
    }

    Ok(())
}

/// Update page for all chunks of a source, matched by chunk_index.
fn update_chunk_pages(source_id: i64, pages: &[(i32, Option<u32>)]) -> Result<()> {
    let conn = get_conn()?;
    for &(chunk_index, page) in pages {
        if let Some(p) = page {
            conn.execute(
                "UPDATE chunks SET page = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![p as i64, source_id, chunk_index],
            )?;
        }
    }
    Ok(())
}

/// Get stats across all collections.
pub fn stats() -> Result<Stats> {
    let conn = get_conn()?;

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

    // Estimated chunks (using default chunk size for estimation)
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
    let conn = match get_conn() {
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
