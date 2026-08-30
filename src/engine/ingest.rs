use anyhow::Result;
use std::time::Instant;

use crate::storage::sqlite::{self, DEFAULT_COLLECTION_ID};
use crate::types::ChunkData;

use crate::chunker;
use crate::embedding::EmbeddingProvider;
use crate::extractor;
use crate::llm::{ContextResult, LlmProvider};
use crate::params::IngestConfig;
use crate::types::IngestionReport;

use super::activity_log;
use super::chunk_counter;
use super::{
    add_embeddings_to_buffer, get_conn, insert_chunk_tags, rebuild_and_save_indexes,
    sanitize_collection, update_collection_tags, verify_chunks, wal_checkpoint,
};

/// Ingest a text document into the RAG (recursive chunking path).
pub async fn ingest_text(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    content: &str,
    source_name: &str,
    metadata: Option<&str>,
    collection: Option<&str>,
    options: IngestConfig,
) -> Result<(i64, IngestionReport)> {
    let total_start = Instant::now();
    if content.trim().is_empty() {
        anyhow::bail!("Cannot ingest empty content");
    }
    let collection_id = sanitize_collection(collection.unwrap_or(DEFAULT_COLLECTION_ID))?;
    let meta = metadata.map(|m| m.to_string()).unwrap_or_default();
    // Custom recursive character chunker (faster, no freeze on large docs)
    let chunk_size = options.chunk_size;

    // Resolve chunking strategy
    let strategy = chunker::resolve_chunking_strategy(
        &options.chunking_strategy,
        content.len(),
        options.auto_threshold,
    );
    tracing::info!(
        "Chunking strategy: {} (content_len={}, threshold={})",
        strategy,
        content.len(),
        options.auto_threshold
    );

    if strategy == "parent_child" {
        return ingest_text_parent_child(
            embedder,
            llm,
            content,
            source_name,
            metadata,
            Some(&collection_id),
            options,
        )
        .await;
    }

    let source = sqlite::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?;
    if source.is_duplicate {
        anyhow::bail!("{}", source.message);
    }
    let source_id = source.source_id;

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
        tracing::info!(
            "Source below chunk size ({} chars), ingesting as single chunk",
            content.len()
        );
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
        chunker::chunk_text(
            content,
            chunk_size,
            options.chunk_overlap_ratio,
            options.merge_last_chunk_threshold,
        )
    };
    tracing::info!("Chunked into {} chunks (size={})", chunks.len(), chunk_size);
    activity_log::push(
        "chunking",
        format!(
            "Chunked '{}' into {} chunks (size={})",
            source_name,
            chunks.len(),
            chunk_size
        ),
    );

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
        tracing::info!(
            "Generating context prefixes for {} chunks via LLM...",
            chunks.len()
        );
        activity_log::push(
            "llm",
            format!("Starting contextual retrieval for {} chunks", chunks.len()),
        );
        let (results, failures, skipped) = generate_contexts(
            llm_provider,
            content,
            &chunk_texts,
            options.context_batch_size,
            options.context_max_retries,
            options.child_min_chars,
        )
        .await;
        let with_ctx = results.iter().filter(|c| c.context.is_some()).count();
        tracing::info!(
            "Context: {}/{} contextualized, {} skipped, {} failed",
            with_ctx,
            chunks.len(),
            skipped,
            failures
        );
        context_failures = failures;
        context_skipped = skipped;
        results
    } else {
        vec![ContextResult::default(); chunks.len()]
    };
    let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
    activity_log::push(
        "llm",
        format!("Contextual retrieval done in {}ms", llm_duration_ms),
    );

    // Filter chunks by relevance score if enabled
    let mut filtered_count = 0usize;
    let kept: Vec<(usize, &chunker::Chunk, &ContextResult)> = chunks
        .iter()
        .enumerate()
        .zip(context_results.iter())
        .filter(|((_, _), ctx_result)| {
            if options.relevance_scoring
                && let Some(score) = ctx_result.relevance_score
                && (score as f64) < options.min_relevance_score
            {
                filtered_count += 1;
                tracing::info!(
                    "Filtered chunk (score={:.1} < threshold={:.1})",
                    score,
                    options.min_relevance_score
                );
                return false;
            }
            true
        })
        .map(|((idx, chunk), ctx_result)| (idx, chunk, ctx_result))
        .collect();

    // Compute relevance statistics
    let (avg_relevance, min_relevance, rel_max) = compute_relevance_stats(&context_results);

    if filtered_count > 0 {
        tracing::info!(
            "Relevance scoring: filtered {}/{} chunks (threshold={:.1})",
            filtered_count,
            chunks.len(),
            options.min_relevance_score
        );
    }

    if kept.is_empty() {
        sqlite::delete_source(source_id)?;
        anyhow::bail!("All chunks were filtered by relevance scoring; source was not ingested");
    }

    // Build final texts for embedding: context prefix + chunk content
    let final_texts: Vec<String> = kept
        .iter()
        .map(|(_, chunk, ctx_result)| match &ctx_result.context {
            Some(context) => format!("{}\n\n{}", context, chunk.content),
            None => chunk.content.clone(),
        })
        .collect();

    // Batch embed all chunks (with context prefixes)
    let embed_start = Instant::now();
    activity_log::push(
        "embedding",
        format!("Embedding {} texts...", final_texts.len()),
    );
    let embeddings = embedder.embed_batch(&final_texts).await?;
    embedder.validate_batch(final_texts.len(), &embeddings)?;
    let embedding_duration_ms = embed_start.elapsed().as_millis() as u64;
    activity_log::push(
        "embedding",
        format!(
            "Embedding done in {}ms ({} texts)",
            embedding_duration_ms,
            final_texts.len()
        ),
    );

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

    let count = sqlite::add_chunks(source_id, chunk_data)?;
    tracing::info!(
        "Ingested {} chunks for source {} ({})",
        count,
        source_id,
        source_name
    );

    // Store section_path separately because it is application metadata.
    update_chunk_section_paths(source_id, &section_paths)?;
    update_chunk_pages(source_id, &pages)?;

    // Store auto-generated tags for each chunk in chunk_tags table
    if tags_per_chunk.iter().any(|(_, t)| !t.is_empty()) {
        insert_chunk_tags(source_id, &tags_per_chunk, None)?;
        // Update collection_tags for routing
        if let Ok(conn) = get_conn() {
            let _ = update_collection_tags(source_id, &tags_per_chunk, &collection_id, &conn);
        }
    }

    // Mark source as completed
    if let Err(e) = sqlite::update_source_status(source_id, "completed".to_string()) {
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

/// Ingest a file (PDF, TXT, MD).
/// Uses our custom extractor for reliable text extraction.
pub async fn ingest_file(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    file_path: &str,
    collection: Option<&str>,
    options: IngestConfig,
) -> Result<(i64, IngestionReport)> {
    // Use the local extractor for supported document formats.
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
pub(super) async fn generate_contexts(
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
                tags: Vec::new(),
            });
        } else {
            long_indices.push(i);
        }
    }

    // Batch: send groups of long chunks in one LLM call each
    for batch_idx in long_indices.chunks(batch_size) {
        let batch_texts: Vec<String> = batch_idx.iter().map(|&i| chunk_texts[i].clone()).collect();
        tracing::info!(
            "Calling generate_context_for_parent ({} chunks)...",
            batch_texts.len()
        );
        let batch_results = llm_provider
            .generate_context_for_parent(whole_document, &batch_texts)
            .await;
        for (j, result) in batch_results.into_iter().enumerate() {
            let global_i = batch_idx[j];
            match result {
                Ok(ctx) => {
                    if ctx.context.is_none() {
                        failures += 1;
                    }
                    // Insert at correct position
                    while results.len() <= global_i {
                        results.push(ContextResult::default());
                    }
                    results[global_i] = ctx;
                }
                Err(e) => {
                    tracing::warn!("Context generation failed: {}, using raw content", e);
                    activity_log::push("error", format!("Context generation failed: {}", e));
                    failures += 1;
                    while results.len() <= global_i {
                        results.push(ContextResult::default());
                    }
                    results[global_i] = ContextResult::default();
                }
            }
        }
    }

    // Ensure results vector is full size
    while results.len() < chunk_texts.len() {
        results.push(ContextResult::default());
    }

    // Retry failed chunks individually (up to max_retries)
    if max_retries > 0 {
        let mut retry_count = 0usize;
        for &i in &long_indices {
            if results[i].context.is_some() {
                continue;
            }
            for attempt in 1..=max_retries {
                match llm_provider
                    .generate_context(whole_document, &chunk_texts[i])
                    .await
                {
                    Ok(ctx) if ctx.context.is_some() => {
                        tracing::debug!(
                            "Retry {}/{} succeeded for chunk {}",
                            attempt,
                            max_retries,
                            i
                        );
                        failures = failures.saturating_sub(1);
                        retry_count += 1;
                        results[i] = ctx;
                        break;
                    }
                    Ok(_) => {} // context still None
                    Err(e) => {
                        tracing::debug!(
                            "Retry {}/{} failed for chunk {}: {}",
                            attempt,
                            max_retries,
                            i,
                            e
                        );
                    }
                }
            }
        }
        if retry_count > 0 {
            tracing::info!(
                "Retry recovered {}/{} failed chunks",
                retry_count,
                long_indices.len()
            );
        }
    }

    (results, failures, skipped)
}

/// Compute relevance statistics from context results.
pub(super) fn compute_relevance_stats(context_results: &[ContextResult]) -> (f64, f64, f64) {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = 0.0f64;
    for c in context_results {
        if let Some(s) = c.relevance_score {
            let s = s as f64;
            count += 1;
            sum += s;
            if s < min {
                min = s;
            }
            if s > max {
                max = s;
            }
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

pub(super) struct KeptChild {
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
    _collection_id: String,
) -> Result<ParentProcessResult> {
    if children.is_empty() {
        return Ok(ParentProcessResult {
            p_idx,
            context_failures: 0,
            context_skipped: 0,
            filtered_count: 0,
            kept_data: vec![],
            parent_chunk,
            relevance_scores: vec![],
            llm_duration_ms: 0,
            embedding_duration_ms: 0,
        });
    }

    let child_texts: Vec<String> = children.iter().map(|c| c.content.clone()).collect();

    // LLM context generation
    let (context_results, context_failures, context_skipped, llm_ms) =
        if let Some(ref llm_provider) = llm {
            tracing::info!(
                "Processing parent {}/{} ({} children)...",
                p_idx + 1,
                total_parents,
                children.len()
            );
            activity_log::push(
                "llm",
                format!(
                    "Parent {}/{}: contextual retrieval for {} children",
                    p_idx + 1,
                    total_parents,
                    children.len()
                ),
            );
            let t = Instant::now();
            let (results, failures, skipped) = generate_contexts(
                llm_provider,
                &whole_document,
                &child_texts,
                batch_size,
                max_retries,
                child_min_chars,
            )
            .await;
            let dur = t.elapsed().as_millis() as u64;
            if failures > 0 {
                activity_log::push(
                    "error",
                    format!(
                        "Parent {}/{}: {} context generation failures",
                        p_idx + 1,
                        total_parents,
                        failures
                    ),
                );
            }
            activity_log::push(
                "llm",
                format!(
                    "Parent {}/{}: context done in {}ms ({} ok, {} skip, {} fail)",
                    p_idx + 1,
                    total_parents,
                    dur,
                    results.iter().filter(|r| r.context.is_some()).count(),
                    skipped,
                    failures
                ),
            );
            (results, failures, skipped, dur)
        } else {
            (vec![ContextResult::default(); child_texts.len()], 0, 0, 0)
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
            && (score as f64) < min_relevance_score
        {
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
            p_idx,
            context_failures,
            context_skipped,
            filtered_count,
            kept_data: vec![],
            parent_chunk,
            relevance_scores,
            llm_duration_ms: llm_ms,
            embedding_duration_ms: 0,
        });
    }

    // Embed
    tracing::info!("Embedding {} texts...", final_texts.len());
    activity_log::push(
        "embedding",
        format!(
            "Parent {}/{}: embedding {} texts",
            p_idx + 1,
            total_parents,
            final_texts.len()
        ),
    );
    let t = Instant::now();
    let embeddings = embedder.embed_batch(&final_texts).await?;
    embedder.validate_batch(final_texts.len(), &embeddings)?;
    let embed_ms = t.elapsed().as_millis() as u64;
    activity_log::push(
        "embedding",
        format!(
            "Parent {}/{}: embedding done in {}ms",
            p_idx + 1,
            total_parents,
            embed_ms
        ),
    );

    // Build kept data
    let kept_data: Vec<KeptChild> = kept_children
        .iter()
        .zip(embeddings)
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
        p_idx,
        context_failures,
        context_skipped,
        filtered_count,
        kept_data,
        parent_chunk,
        relevance_scores,
        llm_duration_ms: llm_ms,
        embedding_duration_ms: embed_ms,
    })
}

/// Commit a processed parent and its children to the database.
pub(super) fn commit_parent_to_db(
    source_id: i64,
    logical_parent_index: usize,
    parent_chunk: &chunker::Chunk,
    kept_data: &[KeptChild],
    collection_id: &str,
) -> Result<usize> {
    let mut conn = get_conn()?;
    let tx = conn.transaction()?;
    let mut parent_tags: Vec<(i32, Vec<String>)> = Vec::new();

    tx.execute(
        "INSERT INTO chunks (source_id, collection_id, content, chunk_index, start_pos, end_pos, chunk_type, chunk_role, logical_parent_index) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![source_id, collection_id, parent_chunk.content, parent_chunk.index, parent_chunk.start_pos, parent_chunk.end_pos, "parent", "parent", logical_parent_index as i64],
    )?;
    let parent_db_id = tx.last_insert_rowid();
    if let Some(sp) = &parent_chunk.section_path {
        tx.execute(
            "UPDATE chunks SET section_path = ?1 WHERE id = ?2",
            rusqlite::params![sp, parent_db_id],
        )?;
    }
    if let Some(p) = parent_chunk.page {
        tx.execute(
            "UPDATE chunks SET page = ?1 WHERE id = ?2",
            rusqlite::params![p as i64, parent_db_id],
        )?;
    }

    // Store children (direct INSERT — avoids pool contention)
    for child in kept_data {
        let embedding_blob = if child.embedding.is_empty() {
            None
        } else {
            // Serialize Vec<f32> to bytes (little-endian f32 array)
            let bytes: Vec<u8> = child
                .embedding
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Some(bytes)
        };
        tx.execute(
            "INSERT INTO chunks (source_id, collection_id, content, chunk_index, start_pos, end_pos, chunk_type, chunk_role, parent_id, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![source_id, collection_id, child.content, child.chunk_index, child.start_pos, child.end_pos, "child", "child", parent_db_id, embedding_blob],
        )?;
        let child_db_id = tx.last_insert_rowid();
        crate::storage::sqlite::add_chunk_to_indexes(
            &tx,
            child_db_id,
            &child.content,
            &child.embedding,
        )?;
        if !child.tags.is_empty() {
            parent_tags.push((child.chunk_index, child.tags.clone()));
        }
    }

    for (chunk_index, tags) in &parent_tags {
        let chunk_id: i64 = tx.query_row(
            "SELECT id FROM chunks WHERE source_id = ?1 AND chunk_index = ?2",
            rusqlite::params![source_id, chunk_index],
            |row| row.get(0),
        )?;
        for tag in tags {
            tx.execute(
                "INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?1, ?2)",
                rusqlite::params![chunk_id, tag],
            )?;
        }
    }
    tx.commit()?;

    if parent_tags.iter().any(|(_, t)| !t.is_empty()) {
        let conn = get_conn()?;
        let _ = update_collection_tags(source_id, &parent_tags, collection_id, &conn);
    }

    crate::pipeline::invalidate_cache();
    Ok(kept_data.len())
}

/// Parent-child chunking ingestion path.
async fn ingest_text_parent_child(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    content: &str,
    source_name: &str,
    metadata: Option<&str>,
    collection: Option<&str>,
    options: IngestConfig,
) -> Result<(i64, IngestionReport)> {
    let total_start = Instant::now();
    if content.trim().is_empty() {
        anyhow::bail!("Cannot ingest empty content");
    }
    let collection_id = sanitize_collection(collection.unwrap_or(DEFAULT_COLLECTION_ID))?;
    let meta = metadata.map(|m| m.to_string()).unwrap_or_default();

    let source = sqlite::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?;
    let source_id = if source.is_duplicate {
        let status: String = {
            let conn = get_conn()?;
            conn.query_row(
                "SELECT status FROM sources WHERE id = ?1",
                rusqlite::params![source.source_id],
                |row| row.get(0),
            )?
        };
        if status == "pending" {
            tracing::info!(
                "Resuming pending source id={} for '{}'",
                source.source_id,
                source_name
            );
            source.source_id
        } else {
            anyhow::bail!("{}", source.message);
        }
    } else {
        source.source_id
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
        total_parents,
        total_children
    );
    activity_log::push(
        "chunking",
        format!(
            "Parent-child chunking: {} parents, {} children for '{}'",
            total_parents, total_children, source_name
        ),
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

    // Resume from exact committed logical parent indices, not completion count.
    let committed_parent_indices: std::collections::HashSet<i64> = {
        let conn = get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT logical_parent_index FROM chunks
             WHERE source_id = ?1 AND chunk_role = 'parent' AND logical_parent_index IS NOT NULL",
        )?;
        stmt.query_map(rusqlite::params![source_id], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    if !committed_parent_indices.is_empty() {
        tracing::info!(
            "Resume: {} exact parent groups already committed",
            committed_parent_indices.len()
        );
    }

    let parents_to_process: Vec<(usize, &chunker::ParentChildGroup)> = groups
        .iter()
        .enumerate()
        .filter(|(p_idx, group)| {
            !committed_parent_indices.contains(&(*p_idx as i64)) && !group.children.is_empty()
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

        let mut join_set: tokio::task::JoinSet<Result<ParentProcessResult>> =
            tokio::task::JoinSet::new();
        let mut parent_iter = parents_to_process.into_iter();

        // Seed the join set with up to `concurrency` tasks
        for _ in 0..concurrency {
            if let Some((p_idx, group)) = parent_iter.next() {
                let children: Vec<chunker::Chunk> = group.children.clone();
                let parent_chunk = group.parent.clone();
                let llm = llm_clone.clone();
                let embedder = embedder_clone.clone();
                let doc = content_owned.clone();
                let col_id = collection_id.clone();
                join_set.spawn(async move {
                    process_parent(
                        p_idx,
                        total_parents,
                        children,
                        parent_chunk,
                        llm,
                        embedder,
                        doc,
                        batch_size,
                        max_retries,
                        child_min_chars_val,
                        relevance_scoring,
                        min_relevance_score,
                        col_id,
                    )
                    .await
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
                let col_id = collection_id.clone();
                join_set.spawn(async move {
                    process_parent(
                        p_idx,
                        total_parents,
                        children,
                        parent_chunk,
                        llm,
                        embedder,
                        doc,
                        batch_size,
                        max_retries,
                        child_min_chars_val,
                        relevance_scoring,
                        min_relevance_score,
                        col_id,
                    )
                    .await
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
                let kept_count = commit_parent_to_db(
                    source_id,
                    processed.p_idx,
                    &processed.parent_chunk,
                    &processed.kept_data,
                    &collection_id,
                )?;
                total_kept += kept_count;
                tracing::info!(
                    "Parent {}/{} committed ({} children stored)",
                    processed.p_idx + 1,
                    total_parents,
                    kept_count
                );
                activity_log::push(
                    "info",
                    format!(
                        "Parent {}/{} committed ({} children stored)",
                        processed.p_idx + 1,
                        total_parents,
                        kept_count
                    ),
                );
                // Increment global chunk counter for real-time progress
                chunk_counter::add(kept_count);

                // Periodic WAL checkpoint to prevent WAL bloat during ingestion
                if options.wal_checkpoint_interval > 0
                    && total_kept % options.wal_checkpoint_interval == 0
                {
                    tracing::info!(
                        "Periodic WAL checkpoint ({} children committed)",
                        total_kept
                    );
                    wal_checkpoint();
                }
            }
        }
    }

    if total_kept == 0 {
        sqlite::delete_source(source_id)?;
        anyhow::bail!(
            "All parent-child chunks were filtered by relevance scoring; source was not ingested"
        );
    }

    // Mark source as completed
    if let Err(e) = sqlite::update_source_status(source_id, "completed".to_string()) {
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
    let avg_relevance = if all_relevance_scores.is_empty() {
        0.0
    } else {
        all_relevance_scores.iter().sum::<f64>() / all_relevance_scores.len() as f64
    };
    let min_relevance = all_relevance_scores
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let rel_max = all_relevance_scores.iter().cloned().fold(0.0f64, f64::max);
    let min_relevance = if min_relevance == f64::INFINITY {
        0.0
    } else {
        min_relevance
    };

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
        total_parents,
        total_children,
        total_duration_ms,
        contextualized,
        context_skipped,
        context_failures,
        filtered_count
    );

    Ok((source_id, report))
}

/// Update section_path for all chunks of a source, matched by chunk_index.
pub(super) fn update_chunk_section_paths(
    source_id: i64,
    section_paths: &[(i32, Option<String>)],
) -> Result<()> {
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
pub(super) fn update_chunk_pages(source_id: i64, pages: &[(i32, Option<u32>)]) -> Result<()> {
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
