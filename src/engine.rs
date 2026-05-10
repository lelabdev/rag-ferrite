use anyhow::Result;
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
use crate::llm::LlmProvider;

/// Initialize rag_engine: logger + DB pool + schema
pub fn init(data_dir: &std::path::Path) -> Result<()> {
    simple::init_core();
    let db_path = data_dir.join("rag.sqlite3");
    std::fs::create_dir_all(data_dir)?;
    db_pool::init_db_pool(db_path.to_string_lossy().to_string(), 4)?;
    source_rag::init_source_db()?;
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
) -> Result<i64> {
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
    let chunks = chunker::chunk_text(content, chunk_size);
    tracing::info!("Chunked into {} chunks (size={})", chunks.len(), chunk_size);

    // Contextual retrieval: generate context prefixes via LLM
    let contexts: Vec<Option<String>> = if let Some(llm_provider) = llm {
        tracing::info!("Generating context prefixes for {} chunks via LLM...", chunks.len());
        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        // Process in batches of 20 for rate limiting
        let mut all_contexts: Vec<Option<String>> = Vec::with_capacity(chunks.len());
        for batch in chunk_texts.chunks(20) {
            let results = llm_provider.generate_context_batch(content, batch).await;
            for result in results {
                match result {
                    Ok(ctx) => all_contexts.push(Some(ctx)),
                    Err(e) => {
                        tracing::warn!("Context generation failed for chunk: {}, using raw content", e);
                        all_contexts.push(None);
                    }
                }
            }
        }
        let with_ctx = all_contexts.iter().filter(|c| c.is_some()).count();
        tracing::info!("Generated {}/{} context prefixes", with_ctx, chunks.len());
        all_contexts
    } else {
        vec![None; chunks.len()]
    };

    // Build final texts for embedding: context prefix + chunk content
    let final_texts: Vec<String> = chunks
        .iter()
        .zip(contexts.iter())
        .map(|(chunk, ctx)| {
            match ctx {
                Some(context) => format!("{}\n\n{}", context, chunk.content),
                None => chunk.content.clone(),
            }
        })
        .collect();

    // Batch embed all chunks (with context prefixes)
    let embeddings = embedder.embed_batch(&final_texts).await?;

    // Store original chunk content (not the prefixed version)
    let chunk_data: Vec<ChunkData> = chunks
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(c, emb)| ChunkData {
            content: c.content.clone(),
            chunk_index: c.index,
            start_pos: c.start_pos,
            end_pos: c.end_pos,
            chunk_type: format!("{:?}", c.chunk_type),
            embedding: emb,
        })
        .collect();

    let count = source_rag::add_chunks(source.source_id, chunk_data)?;
    tracing::info!("Ingested {} chunks for source {} ({})", count, source.source_id, source_name);

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
    let index_path = format!("/home/loops/services/rag-ferrite/data/hnsw_{}.index", collection_id);
    if let Err(e) = source_rag::save_collection_hnsw_index(collection_id.clone(), index_path) {
        tracing::warn!("Failed to save HNSW index: {}", e);
    }

    Ok(source.source_id)
}

/// Ingest a file (PDF, TXT, MD)
/// Uses our custom extractor for reliable text extraction
pub async fn ingest_file(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    file_path: &str,
    collection: Option<&str>,
) -> Result<i64> {
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
    )
    .await
}

/// Delete a source by ID
pub fn delete_source(source_id: i64) -> Result<()> {
    source_rag::delete_source(source_id)?;
    // Rebuild all indexes (don't know which collection the source was in)
    let _ = source_rag::rebuild_chunk_hnsw_index();
    let _ = source_rag::rebuild_chunk_bm25_index();
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
            let index_path = format!("/home/loops/services/rag-ferrite/data/hnsw_{}.index", coll);
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

/// Get chunks adjacent to a given chunk
pub fn get_neighbors(source_id: i64, chunk_index: i64, before: i64, after: i64) -> Result<Vec<ChunkSearchResult>> {
    let min_index = (chunk_index - before).max(0);
    let max_index = chunk_index + after;
    let chunks = source_rag::get_adjacent_chunks(source_id, min_index as i32, max_index as i32)?;
    Ok(chunks)
}

/// List all sources
pub fn list_sources() -> Result<Vec<source_rag::SourceEntry>> {
    Ok(source_rag::list_sources()?)
}

/// Get stats
pub fn stats() -> Result<Stats> {
    let sources = source_rag::list_sources()?;
    Ok(Stats {
        document_count: sources.len(),
    })
}

pub struct Stats {
    pub document_count: usize,
}
