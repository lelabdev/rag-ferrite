use anyhow::Result;
use rag_engine::api::{
    db_pool,
    hybrid_search,
    semantic_chunker,
    simple,
    source_rag::{self, ChunkData, ChunkSearchResult},
};
use crate::embedding::EmbeddingProvider;

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

    // Batch embed all chunks
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = embedder.embed_batch(&texts).await?;

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
pub async fn ingest_file(embedder: &EmbeddingProvider, file_path: &str) -> Result<i64> {
    let bytes = std::fs::read(file_path)?;
    let text = rag_engine::api::document_parser::extract_text_from_document(bytes)?;

    let name = std::path::Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.to_string());

    ingest_text(embedder, &text, &name, Some(&format!("{{\"path\":\"{}\"}}", file_path))).await
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
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    let query_embedding = embedder.embed(query).await?;
    let results = hybrid_search::search_hybrid(
        query.to_string(),
        query_embedding,
        limit as u32,
        None,
        None,
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
