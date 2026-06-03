use anyhow::Result;
use std::time::Instant;
use rag_engine::api::{
    db_pool,
    simple,
    source_rag::{self, ChunkData},
};
use rag_engine::api::incremental_index;
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
    db_pool::init_db_pool(db_path_str.clone(), config.advanced.db_pool_size as u32)?;
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

    // Migration: add parent_id and chunk_type columns for parent-child chunking
    let has_parent_id: bool = conn.prepare("SELECT parent_id FROM chunks LIMIT 1").is_ok();
    if !has_parent_id {
        tracing::info!("Migrating: adding parent_id and chunk_type columns to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN parent_id INTEGER DEFAULT NULL")?;
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN chunk_role TEXT DEFAULT NULL")?;
    }

    // Migration: make embedding nullable (parents don't have embeddings)
    let embedding_notnull: bool = {
        let mut stmt = conn.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'")?;
        let sql: String = stmt.query_row([], |row| row.get(0))?;
        sql.contains("embedding BLOB NOT NULL")
    };
    if embedding_notnull {
        tracing::info!("Migrating: making embedding column nullable for parent-child support");
        conn.execute_batch(
            "CREATE TABLE chunks_new (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                collection_id TEXT NOT NULL DEFAULT '__default__',
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                chunk_type TEXT DEFAULT 'general',
                embedding BLOB,
                embedding_i8 BLOB,
                embedding_scale REAL,
                section_path TEXT DEFAULT NULL,
                page INTEGER DEFAULT NULL,
                parent_id INTEGER DEFAULT NULL,
                chunk_role TEXT DEFAULT NULL
            );
            INSERT INTO chunks_new SELECT * FROM chunks;
            DROP TABLE chunks;
            ALTER TABLE chunks_new RENAME TO chunks;"
        )?;
    }
    drop(conn);

    // Create chunk_tags table for auto-tagging
    create_chunk_tags_table(&db_path_str)?;

    let _ = DB_PATH.set(db_path_str.clone());

    // Store a shared connection for all subsequent get_conn() calls
    let shared_conn = rusqlite::Connection::open(&db_path_str)?;
    shared_conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL; PRAGMA busy_timeout={}; PRAGMA cache_size=-{};",
        config.advanced.db_busy_timeout_ms, config.advanced.db_cache_size_mb
    ))?;
    let _ = DB_CONN.set(std::sync::Mutex::new(shared_conn));

    // Check embedding dimension mismatch: compare DB vectors with configured dimensions
    if let Some(config_dims) = config.embedding.dimensions {
        let conn = get_conn()?;
        let db_dims: Option<usize> = conn.query_row(
            "SELECT vector FROM chunks WHERE vector IS NOT NULL LIMIT 1",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob.len() / 4) // f32 = 4 bytes
            },
        ).ok();
        drop(conn);

        if let Some(stored_dims) = db_dims {
            if stored_dims != config_dims {
                anyhow::bail!(
                    "Embedding dimension mismatch: DB has {} but config says {}. Re-ingest all documents or update config.",
                    stored_dims, config_dims
                );
            }
            tracing::info!("Embedding dimensions verified: {} (DB matches config)", stored_dims);
        }
    }

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
    pub context_batch_size: usize,
    pub context_max_retries: usize,
    pub chunk_overlap_ratio: f64,
    pub merge_last_chunk_threshold: usize,
    /// Chunking strategy: "recursive", "parent_child", or "auto"
    pub chunking_strategy: String,
    /// Parent chunk max chars (for parent_child mode)
    pub parent_max_chars: usize,
    /// Child chunk max chars (for parent_child mode)
    pub child_max_chars: usize,
    /// Child chunk overlap (for parent_child mode)
    pub child_overlap: usize,
    /// Auto-switch threshold (for "auto" mode)
    pub auto_threshold: usize,
    /// Min child chars — consecutive children below this are merged
    pub child_min_chars: usize,
    /// Defer HNSW + BM25 index rebuild to explicit flush (saves RAM during batch ingestion)
    pub defer_index_rebuild: bool,
    /// WAL checkpoint every N parents committed (0 = disabled)
    pub wal_checkpoint_interval: usize,
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
    let source_id = source_rag::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?.source_id;

    // Custom recursive character chunker (faster, no freeze on large docs)
    let chunk_size = options.chunk_size;

    // Resolve chunking strategy
    let strategy = chunker::resolve_chunking_strategy(
        &options.chunking_strategy,
        content.len(),
        options.auto_threshold,
    );
    tracing::info!("Chunking strategy: {} (content_len={}, threshold={})", strategy, content.len(), options.auto_threshold);

    if strategy == "parent_child" {
        return ingest_text_parent_child(
            embedder, llm, content, source_name, metadata, Some(&collection_id), options,
        ).await;
    }

    // ── Recursive chunking (original path) ──

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
        chunker::chunk_text(content, chunk_size, options.chunk_overlap_ratio, options.merge_last_chunk_threshold)
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
    let mut context_skipped = 0usize;
    let llm_start = Instant::now();
    let context_results: Vec<ContextResult> = if let Some(llm_provider) = llm {
        tracing::info!("Generating context prefixes for {} chunks via LLM...", chunks.len());
        let (results, failures, skipped) = generate_contexts(
            llm_provider, content, &chunk_texts,
            options.context_batch_size, options.context_max_retries,
            options.child_min_chars,
        ).await;
        let with_ctx = results.iter().filter(|c| c.context.is_some()).count();
        tracing::info!(
            "Context: {}/{} contextualized, {} skipped, {} failed",
            with_ctx, chunks.len(), skipped, failures
        );
        context_failures = failures;
        context_skipped = skipped;
        results
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
            if options.relevance_scoring
                && let Some(score) = ctx_result.relevance_score
                    && (score as f64) < options.min_relevance_score {
                        filtered_count += 1;
                        tracing::info!("Filtered chunk (score={:.1} < threshold={:.1})", score, options.min_relevance_score);
                        return false;
                    }
            true
        })
        .map(|((idx, chunk), ctx_result)| (idx, chunk, ctx_result))
        .collect();

    // Compute relevance statistics
    let (avg_relevance, min_relevance, rel_max) = compute_relevance_stats(&context_results);

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
        .zip(embeddings)
        .map(|((_, chunk, _), emb)| ChunkData {
            content: chunk.content.clone(),
            chunk_index: chunk.index,
            start_pos: chunk.start_pos,
            end_pos: chunk.end_pos,
            chunk_type: chunk.chunk_type.as_str().to_string(),
            embedding: emb,
        })
        .collect();

    let count = source_rag::add_chunks(source_id, chunk_data)?;
    tracing::info!("Ingested {} chunks for source {} ({})", count, source_id, source_name);

    // Store section_path for each chunk (separate UPDATE since rag_engine doesn't know about it)
    update_chunk_section_paths(source_id, &section_paths)?;
    update_chunk_pages(source_id, &pages)?;

    // Store auto-generated tags for each chunk in chunk_tags table
    if tags_per_chunk.iter().any(|(_, t)| !t.is_empty()) {
        insert_chunk_tags(source_id, &tags_per_chunk, None)?;
    }

    // Mark source as completed
    if let Err(e) = source_rag::update_source_status(source_id, "completed".to_string()) {
        tracing::warn!("Failed to update source status: {}", e);
    }

    // Rebuild indexes only if not deferred (batch ingestion defers to explicit flush)
    if !options.defer_index_rebuild {
        rebuild_and_save_indexes(&collection_id);
    } else {
        // Add embeddings to incremental buffer — immediately searchable without full rebuild
        add_embeddings_to_buffer(source_id);
        tracing::info!("Embeddings added to incremental buffer (defer_index_rebuild=true)");
    }

    let total_duration_ms = total_start.elapsed().as_millis() as u64;

    let report = IngestionReport {
        total_chunks: chunks.len(),
        filtered_chunks: filtered_count,
        avg_relevance,
        min_relevance,
        max_relevance: rel_max,
        context_failures,
        context_skipped,
        total_duration_ms,
        embedding_duration_ms,
        llm_duration_ms,
        source_name: source_name.to_string(),
    };

    Ok((source_id, report))
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
        Some(&serde_json::json!({"path": file_path}).to_string()),
        collection,
        options,
    )
    .await
}

/// Ingest text using parent-child chunking strategy.
/// Parents are stored without embeddings, children are embedded for search.
/// Generate context for a batch of chunk texts using LLM (batch + individual retry).
/// Chunks below `min_chars` are skipped (no LLM call) and counted separately.
/// Shared by both recursive and parent-child ingestion paths.
async fn generate_contexts(
    llm_provider: &LlmProvider,
    whole_document: &str,
    chunk_texts: &[String],
    batch_size: usize,
    max_retries: usize,
    min_chars: usize,
) -> (Vec<ContextResult>, usize, usize) {
    let mut failures = 0usize;
    let mut skipped = 0usize;
    let mut results = Vec::with_capacity(chunk_texts.len());

    // Separate chunks into contextualize (long enough) and skip (too short)
    let mut long_indices = Vec::new();
    for (i, text) in chunk_texts.iter().enumerate() {
        if text.len() < min_chars {
            skipped += 1;
            results.push(ContextResult {
                context: None,
                relevance_score: None,
                extracted_metadata: None,
                tags: Vec::new(),
            });
        } else {
            long_indices.push(i);
        }
    }

    // Batch: send groups of long chunks in one LLM call each
    for batch_idx in long_indices.chunks(batch_size) {
        let batch_texts: Vec<String> = batch_idx.iter().map(|&i| chunk_texts[i].clone()).collect();
        tracing::info!("Calling generate_context_for_parent ({} chunks)...", batch_texts.len());
        let batch_results = llm_provider.generate_context_for_parent(whole_document, &batch_texts).await;
        for (j, result) in batch_results.into_iter().enumerate() {
            let global_i = batch_idx[j];
            match result {
                Ok(ctx) => {
                    if ctx.context.is_none() { failures += 1; }
                    // Insert at correct position
                    while results.len() <= global_i {
                        results.push(ContextResult {
                            context: None, relevance_score: None,
                            extracted_metadata: None, tags: Vec::new(),
                        });
                    }
                    results[global_i] = ctx;
                }
                Err(e) => {
                    tracing::warn!("Context generation failed: {}, using raw content", e);
                    failures += 1;
                    while results.len() <= global_i {
                        results.push(ContextResult {
                            context: None, relevance_score: None,
                            extracted_metadata: None, tags: Vec::new(),
                        });
                    }
                    results[global_i] = ContextResult {
                        context: None, relevance_score: None,
                        extracted_metadata: None, tags: Vec::new(),
                    };
                }
            }
        }
    }

    // Ensure results vector is full size
    while results.len() < chunk_texts.len() {
        results.push(ContextResult {
            context: None, relevance_score: None,
            extracted_metadata: None, tags: Vec::new(),
        });
    }

    // Retry failed chunks individually (up to max_retries)
    if max_retries > 0 {
        let mut retry_count = 0usize;
        for &i in &long_indices {
            if results[i].context.is_some() { continue; }
            for attempt in 1..=max_retries {
                match llm_provider.generate_context(whole_document, &chunk_texts[i]).await {
                    Ok(ctx) if ctx.context.is_some() => {
                        tracing::debug!("Retry {}/{} succeeded for chunk {}", attempt, max_retries, i);
                        failures = failures.saturating_sub(1);
                        retry_count += 1;
                        results[i] = ctx;
                        break;
                    }
                    Ok(_) => {} // context still None
                    Err(e) => {
                        tracing::debug!("Retry {}/{} failed for chunk {}: {}", attempt, max_retries, i, e);
                    }
                }
            }
        }
        if retry_count > 0 {
            tracing::info!("Retry recovered {}/{} failed chunks", retry_count, long_indices.len());
        }
    }

    (results, failures, skipped)
}

/// Compute relevance statistics from context results.
fn compute_relevance_stats(context_results: &[ContextResult]) -> (f64, f64, f64) {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = 0.0f64;
    for c in context_results {
        if let Some(s) = c.relevance_score {
            let s = s as f64;
            count += 1;
            sum += s;
            if s < min { min = s; }
            if s > max { max = s; }
        }
    }
    let avg = if count == 0 { 0.0 } else { sum / count as f64 };
    let min_val = if count == 0 { 0.0 } else { min };
    (avg, min_val, max)
}

/// Result of processing a single parent (LLM + embedding done, DB write pending).
struct ParentProcessResult {
    p_idx: usize,
    context_failures: usize,
    context_skipped: usize,
    filtered_count: usize,
    kept_data: Vec<KeptChild>,
    parent_chunk: chunker::Chunk,
    relevance_scores: Vec<f64>,
    llm_duration_ms: u64,
    embedding_duration_ms: u64,
}

struct KeptChild {
    content: String,
    chunk_index: i32,
    start_pos: i32,
    end_pos: i32,
    embedding: Vec<f32>,
    tags: Vec<String>,
}

/// Process a single parent group: generate context + embed children.
/// This is the CPU/IO-intensive part that can run in parallel.
async fn process_parent(
    p_idx: usize,
    total_parents: usize,
    children: Vec<chunker::Chunk>,
    parent_chunk: chunker::Chunk,
    llm: Option<LlmProvider>,
    embedder: EmbeddingProvider,
    whole_document: String,
    batch_size: usize,
    max_retries: usize,
    child_min_chars: usize,
    relevance_scoring: bool,
    min_relevance_score: f64,
) -> Result<ParentProcessResult> {
    if children.is_empty() {
        return Ok(ParentProcessResult {
            p_idx, context_failures: 0, context_skipped: 0, filtered_count: 0,
            kept_data: vec![], parent_chunk, relevance_scores: vec![],
            llm_duration_ms: 0, embedding_duration_ms: 0,
        });
    }

    let child_texts: Vec<String> = children.iter().map(|c| c.content.clone()).collect();

    // LLM context generation
    let (context_results, context_failures, context_skipped, llm_ms) = if let Some(ref llm_provider) = llm {
        tracing::info!("Processing parent {}/{} ({} children)...", p_idx + 1, total_parents, children.len());
        let t = Instant::now();
        let (results, failures, skipped) = generate_contexts(
            llm_provider, &whole_document, &child_texts,
            batch_size, max_retries, child_min_chars,
        ).await;
        let dur = t.elapsed().as_millis() as u64;
        (results, failures, skipped, dur)
    } else {
        (vec![ContextResult { context: None, relevance_score: None, extracted_metadata: None, tags: Vec::new()}; child_texts.len()], 0, 0, 0)
    };

    // Filter by relevance and build final texts
    let mut filtered_count = 0usize;
    let mut relevance_scores: Vec<f64> = Vec::new();
    let mut final_texts: Vec<String> = Vec::new();
    let mut kept_children: Vec<(usize, &chunker::Chunk, &ContextResult)> = Vec::new();

    for (c_idx, child) in children.iter().enumerate() {
        let ctx = &context_results[c_idx];
        if relevance_scoring
            && let Some(score) = ctx.relevance_score
                && (score as f64) < min_relevance_score {
                    filtered_count += 1;
                    continue;
                }
        if let Some(score) = ctx.relevance_score {
            relevance_scores.push(score as f64);
        }
        let text = match &ctx.context {
            Some(context) => format!("{}\n\n{}", context, child.content),
            None => child.content.clone(),
        };
        final_texts.push(text);
        kept_children.push((c_idx, child, ctx));
    }

    if kept_children.is_empty() {
        return Ok(ParentProcessResult {
            p_idx, context_failures, context_skipped, filtered_count,
            kept_data: vec![], parent_chunk, relevance_scores,
            llm_duration_ms: llm_ms, embedding_duration_ms: 0,
        });
    }

    // Embed
    tracing::info!("Embedding {} texts...", final_texts.len());
    let t = Instant::now();
    let embeddings = embedder.embed_batch(&final_texts).await?;
    let embed_ms = t.elapsed().as_millis() as u64;

    // Build kept data
    let kept_data: Vec<KeptChild> = kept_children.iter().zip(embeddings)
        .map(|((_, child, ctx_result), embedding)| KeptChild {
            content: child.content.clone(),
            chunk_index: child.index,
            start_pos: child.start_pos,
            end_pos: child.end_pos,
            embedding,
            tags: ctx_result.tags.clone(),
        })
        .collect();

    Ok(ParentProcessResult {
        p_idx, context_failures, context_skipped, filtered_count,
        kept_data, parent_chunk, relevance_scores,
        llm_duration_ms: llm_ms, embedding_duration_ms: embed_ms,
    })
}

/// Commit a processed parent and its children to the database.
fn commit_parent_to_db(
    source_id: i64,
    parent_chunk: &chunker::Chunk,
    kept_data: &[KeptChild],
) -> Result<usize> {
    let conn = get_conn()?;
    let mut parent_tags: Vec<(i32, Vec<String>)> = Vec::new();

    // Store parent chunk (direct INSERT — single connection, no pool contention)
    conn.execute(
        "INSERT INTO chunks (source_id, content, chunk_index, start_pos, end_pos, chunk_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![source_id, parent_chunk.content, parent_chunk.index, parent_chunk.start_pos, parent_chunk.end_pos, "parent"],
    )?;
    let parent_db_id: i64 = conn.query_row(
        "SELECT last_insert_rowid()",
        [],
        |row| row.get(0),
    )?;
    if let Some(sp) = &parent_chunk.section_path {
        conn.execute("UPDATE chunks SET section_path = ?1 WHERE id = ?2", rusqlite::params![sp, parent_db_id])?;
    }
    if let Some(p) = parent_chunk.page {
        conn.execute("UPDATE chunks SET page = ?1 WHERE id = ?2", rusqlite::params![p as i64, parent_db_id])?;
    }

    // Store children (direct INSERT — avoids pool contention)
    for child in kept_data {
        let embedding_blob = if child.embedding.is_empty() {
            None
        } else {
            // Serialize Vec<f32> to bytes (little-endian f32 array)
            let bytes: Vec<u8> = child.embedding.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Some(bytes)
        };
        conn.execute(
            "INSERT INTO chunks (source_id, content, chunk_index, start_pos, end_pos, chunk_type, parent_id, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![source_id, child.content, child.chunk_index, child.start_pos, child.end_pos, "child", parent_db_id, embedding_blob],
        )?;
        if !child.tags.is_empty() {
            parent_tags.push((child.chunk_index, child.tags.clone()));
        }
    }

    if parent_tags.iter().any(|(_, t)| !t.is_empty()) {
        insert_chunk_tags(source_id, &parent_tags, Some(&conn))?;
    }

    Ok(kept_data.len())
}

async fn ingest_text_parent_child(
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

    // Reuse existing source if one with the same name exists in this collection
    let source_id: i64 = {
        let conn = get_conn()?;
        let existing_id: Option<i64> = conn.query_row(
            "SELECT id FROM sources WHERE name = ?1 AND collection_id = ?2 LIMIT 1",
            rusqlite::params![source_name, collection_id],
            |row| row.get::<_, i64>(0),
        ).ok();
        match existing_id {
            Some(id) => {
                tracing::info!("Reusing existing source id={} for '{}' (resume)", id, source_name);
                id
            }
            None => {
                source_rag::add_source_in_collection(
                    collection_id.clone(),
                    content.to_string(),
                    if meta.is_empty() { None } else { Some(meta) },
                    Some(source_name.to_string()),
                )?.source_id
            }
        }
    };

    // Step 1: Parent-child chunking
    let groups = chunker::chunk_text_parent_child(
        content,
        options.parent_max_chars,
        options.child_max_chars,
        options.child_overlap,
        options.merge_last_chunk_threshold,
        options.child_min_chars,
    );

    let total_parents = groups.len();
    let total_children: usize = groups.iter().map(|g| g.children.len()).sum();
    tracing::info!(
        "Parent-child chunking: {} parents, {} children",
        total_parents, total_children
    );

    // Parallel processing: use JoinSet to process up to max_concurrent parents simultaneously.
    // LLM context generation and embedding happen in parallel.
    // DB commits happen sequentially as results arrive.
    let concurrency = options.max_concurrent.max(1);
    let mut context_failures = 0usize;
    let mut context_skipped = 0usize;
    let mut filtered_count = 0usize;
    let mut total_kept = 0usize;
    let mut llm_duration_ms: u64 = 0;
    let mut embedding_duration_ms: u64 = 0;
    let mut all_relevance_scores: Vec<f64> = Vec::new();

    // Check how many parents already have chunks (resume support)
    let existing_parent_count: usize = {
        let conn = get_conn()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT parent_id) FROM chunks WHERE source_id = ?1 AND parent_id IS NOT NULL",
            rusqlite::params![source_id],
            |row| row.get::<_, i64>(0),
        )?;
        count as usize
    };
    if existing_parent_count > 0 {
        tracing::info!("Resume: {} parents already committed, starting from parent {}", existing_parent_count, existing_parent_count + 1);
    }

    // Collect parents to process (skip already committed)
    let parents_to_process: Vec<(usize, &chunker::ParentChildGroup)> = groups.iter()
        .enumerate()
        .filter(|(p_idx, group)| {
            *p_idx >= existing_parent_count && !group.children.is_empty()
        })
        .collect();

    if !parents_to_process.is_empty() {
        let llm_clone = llm.cloned();
        let embedder_clone = embedder.clone();
        let content_owned = content.to_string();
        let batch_size = options.context_batch_size;
        let max_retries = options.context_max_retries;
        let child_min_chars_val = options.child_min_chars;
        let relevance_scoring = options.relevance_scoring;
        let min_relevance_score = options.min_relevance_score;

        let mut join_set: tokio::task::JoinSet<Result<ParentProcessResult>> = tokio::task::JoinSet::new();
        let mut parent_iter = parents_to_process.into_iter();

        // Seed the join set with up to `concurrency` tasks
        for _ in 0..concurrency {
            if let Some((p_idx, group)) = parent_iter.next() {
                let children: Vec<chunker::Chunk> = group.children.clone();
                let parent_chunk = group.parent.clone();
                let llm = llm_clone.clone();
                let embedder = embedder_clone.clone();
                let doc = content_owned.clone();
                join_set.spawn(async move {
                    process_parent(
                        p_idx, total_parents, children, parent_chunk,
                        llm, embedder, doc, batch_size, max_retries, child_min_chars_val,
                        relevance_scoring, min_relevance_score,
                    ).await
                });
            }
        }

        // Process results as they arrive, spawning new tasks to maintain concurrency
        while let Some(result) = join_set.join_next().await {
            // Spawn next parent before processing result (keeps pipeline full)
            if let Some((p_idx, group)) = parent_iter.next() {
                let children: Vec<chunker::Chunk> = group.children.clone();
                let parent_chunk = group.parent.clone();
                let llm = llm_clone.clone();
                let embedder = embedder_clone.clone();
                let doc = content_owned.clone();
                join_set.spawn(async move {
                    process_parent(
                        p_idx, total_parents, children, parent_chunk,
                        llm, embedder, doc, batch_size, max_retries, child_min_chars_val,
                        relevance_scoring, min_relevance_score,
                    ).await
                });
            }

            // Handle completed result
            let processed = match result {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
            };

            context_failures += processed.context_failures;
            context_skipped += processed.context_skipped;
            filtered_count += processed.filtered_count;
            llm_duration_ms += processed.llm_duration_ms;
            embedding_duration_ms += processed.embedding_duration_ms;
            all_relevance_scores.extend(&processed.relevance_scores);

            // Commit to DB (sequential — SQLite)
            if !processed.kept_data.is_empty() {
                let kept_count = commit_parent_to_db(source_id, &processed.parent_chunk, &processed.kept_data)?;
                total_kept += kept_count;
                tracing::info!("Parent {}/{} committed ({} children stored)", processed.p_idx + 1, total_parents, kept_count);

                // Periodic WAL checkpoint to prevent WAL bloat during ingestion
                if options.wal_checkpoint_interval > 0 && total_kept % options.wal_checkpoint_interval == 0 {
                    tracing::info!("Periodic WAL checkpoint ({} children committed)", total_kept);
                    wal_checkpoint();
                }
            }
        }
    }

    // Mark source as completed
    if let Err(e) = source_rag::update_source_status(source_id, "completed".to_string()) {
        tracing::warn!("Failed to update source status: {}", e);
    }

    // Rebuild indexes only if not deferred (batch ingestion defers to explicit flush)
    if !options.defer_index_rebuild {
        rebuild_and_save_indexes(&collection_id);
    } else {
        // Add embeddings to incremental buffer — immediately searchable without full rebuild
        add_embeddings_to_buffer(source_id);
        tracing::info!("Embeddings added to incremental buffer (defer_index_rebuild=true)");
    }

    let total_duration_ms = total_start.elapsed().as_millis() as u64;

    // Compute relevance stats from accumulated scores
    let avg_relevance = if all_relevance_scores.is_empty() { 0.0 } else { all_relevance_scores.iter().sum::<f64>() / all_relevance_scores.len() as f64 };
    let min_relevance = all_relevance_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let rel_max = all_relevance_scores.iter().cloned().fold(0.0f64, f64::max);
    let min_relevance = if min_relevance == f64::INFINITY { 0.0 } else { min_relevance };

    let report = IngestionReport {
        total_chunks: total_children,
        filtered_chunks: filtered_count,
        avg_relevance,
        min_relevance,
        max_relevance: rel_max,
        context_failures,
        context_skipped,
        total_duration_ms,
        embedding_duration_ms,
        llm_duration_ms,
        source_name: source_name.to_string(),
    };

    let contextualized = total_children - context_failures - context_skipped - filtered_count;
    tracing::info!(
        "Parent-child ingestion complete: {} parents, {} children in {}ms | {} contextualized, {} skipped, {} failed, {} filtered",
        total_parents, total_children, total_duration_ms, contextualized, context_skipped, context_failures, filtered_count
    );

    Ok((source_id, report))
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
pub fn pre_check_document(content: &str, filename: &str, chunk_size: usize) -> crate::types::PreCheckReport {
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

    // Estimated chunks
    let estimated_chunks = if char_count == 0 {
        0
    } else if char_count < chunk_size {
        1
    } else {
        char_count.div_ceil(chunk_size)
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
            c if ('\u{4E00}'..='\u{9FFF}').contains(&c) => cjk_chars += 1,
            c if ('\u{3040}'..='\u{309F}').contains(&c) || ('\u{30A0}'..='\u{30FF}').contains(&c) => cjk_chars += 1,
            c if ('\u{0600}'..='\u{06FF}').contains(&c) || ('\u{0750}'..='\u{077F}').contains(&c) => arabic_chars += 1,
            c if ('\u{0400}'..='\u{04FF}').contains(&c) => cyrillic_chars += 1,
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
        Err(e) => {
            tracing::warn!("Duplicate check failed for '{}': {}", filename, e);
            return false;
        }
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

/// Rebuild and persist HNSW + BM25 indexes for a collection.
pub fn list_collections() -> Vec<String> {
    let conn = match get_conn() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut stmt = match conn.prepare("SELECT DISTINCT collection_id FROM chunks") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut collections = Vec::new();
    let rows = stmt.query_map([], |row| row.get::<_, String>(0));
    if let Ok(mapped) = rows {
        for c in mapped.flatten() { collections.push(c); }
    }
    collections
}

/// Add all embeddings for a source to the incremental buffer (immediately searchable).
fn add_embeddings_to_buffer(source_id: i64) {
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
