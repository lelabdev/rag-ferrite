use anyhow::Result;
use rag_engine::api::{
    db_pool,
    hybrid_search,
    semantic_chunker,
    simple,
    source_rag::{self, ChunkData, ChunkSearchResult},
};
use crate::embedding::EmbeddingProvider;
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
) -> Result<i64> {
    let source = source_rag::add_source(
        content.to_string(),
        metadata.map(|m| m.to_string()),
        Some(source_name.to_string()),
    )?;

    // Semantic chunking
    let chunks = semantic_chunker::semantic_chunk(content.to_string(), 600);

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

    // Rebuild indexes
    let _ = source_rag::rebuild_chunk_hnsw_index();
    let _ = source_rag::rebuild_chunk_bm25_index();

    Ok(source.source_id)
}

/// Ingest a file (PDF, DOCX, TXT, MD)
pub async fn ingest_file(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    file_path: &str,
) -> Result<i64> {
    let bytes = std::fs::read(file_path)?;
    let text = rag_engine::api::document_parser::extract_text_from_document(bytes)?;

    let name = std::path::Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.to_string());

    ingest_text(embedder, llm, &text, &name, Some(&format!("{{\"path\":\"{}\"}}", file_path))).await
}

/// Delete a source by ID
pub fn delete_source(source_id: i64) -> Result<()> {
    source_rag::delete_source(source_id)?;
    let _ = source_rag::rebuild_chunk_hnsw_index();
    let _ = source_rag::rebuild_chunk_bm25_index();
    Ok(())
}

/// Search with hybrid fusion (BM25 + vector + RRF)
pub async fn search_hybrid(
    embedder: &EmbeddingProvider,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    let query_embedding = embedder.embed(query).await?;
    let results = hybrid_search::search_hybrid(
        query.to_string(),
        query_embedding,
        limit as u32,
        None,
        filter,
    )?;
    Ok(results)
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
