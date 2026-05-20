use serde::Serialize;

/// Shared result type for hybrid search, used by both MCP and HTTP.
#[derive(Debug, Serialize)]
pub struct HybridResult {
    pub doc_id: i64,
    pub content: String,
    pub score: f64,
    pub vector_rank: u32,
    pub bm25_rank: u32,
    pub source_id: i64,
    pub chunk_index: u32,
    pub metadata: Option<String>,
}

impl From<rag_engine::api::hybrid_search::HybridSearchResult> for HybridResult {
    fn from(r: rag_engine::api::hybrid_search::HybridSearchResult) -> Self {
        HybridResult {
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
            vector_rank: r.vector_rank,
            bm25_rank: r.bm25_rank,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            metadata: r.metadata,
        }
    }
}

impl From<crate::reranker::RerankedResult> for HybridResult {
    fn from(r: crate::reranker::RerankedResult) -> Self {
        HybridResult {
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
            vector_rank: r.vector_rank,
            bm25_rank: r.bm25_rank,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            metadata: r.metadata,
        }
    }
}

/// Source info for listing documents.
#[derive(Debug, Serialize)]
pub struct SourceInfo {
    pub id: i64,
    pub name: Option<String>,
    pub created_at: i64,
    pub metadata: Option<String>,
    pub status: Option<String>,
    pub collection_id: String,
}

impl From<rag_engine::api::source_rag::SourceEntry> for SourceInfo {
    fn from(s: rag_engine::api::source_rag::SourceEntry) -> Self {
        SourceInfo {
            id: s.id,
            name: s.name,
            created_at: s.created_at,
            metadata: s.metadata,
            status: s.status,
            collection_id: s.collection_id,
        }
    }
}

/// Chunk result for neighbor expansion.
#[derive(Debug, Serialize)]
pub struct ChunkResult {
    pub chunk_id: i64,
    pub source_id: i64,
    pub chunk_index: i32,
    pub content: String,
    pub score: f64,
    pub metadata: Option<String>,
}

impl From<rag_engine::api::source_rag::ChunkSearchResult> for ChunkResult {
    fn from(r: rag_engine::api::source_rag::ChunkSearchResult) -> Self {
        ChunkResult {
            chunk_id: r.chunk_id,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            content: r.content,
            score: r.similarity,
            metadata: r.metadata,
        }
    }
}
