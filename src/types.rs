use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Post-chunking verification report.
#[derive(Debug, Serialize)]
pub struct ChunkVerification {
    pub total_chunks: usize,
    pub source_chars: usize,
    pub chunk_chars: usize,
    pub coverage_ratio: f64,
    pub warnings: Vec<String>,
}

/// Pre-ingestion quality check report.
#[derive(Debug, Serialize, Deserialize)]
pub struct PreCheckReport {
    pub extraction_ok: bool,
    pub char_count: usize,
    pub estimated_chunks: usize,
    pub language: String,
    pub is_duplicate: bool,
    pub warnings: Vec<String>,
}

/// Ingestion quality report returned after ingesting a document.
#[derive(Debug, Serialize)]
pub struct IngestionReport {
    pub total_chunks: usize,
    pub filtered_chunks: usize,
    pub avg_relevance: f64,
    pub min_relevance: f64,
    pub max_relevance: f64,
    pub context_failures: usize,
    pub total_duration_ms: u64,
    pub embedding_duration_ms: u64,
    pub llm_duration_ms: u64,
    pub source_name: String,
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    /// LLM reranker score (0.0-1.0). None = not reranked, use `score`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f64>,
    /// Auto-generated tags for this chunk.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
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
            section_path: None,
            page: None,
            rerank_score: None,
            tags: Vec::new(),
        }
    }
}

impl From<rag_engine::api::hybrid_search::HybridSearchResult> for crate::reranker::RerankedResult {
    fn from(r: rag_engine::api::hybrid_search::HybridSearchResult) -> Self {
        crate::reranker::RerankedResult {
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
            rerank_score: None,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            metadata: r.metadata,
            vector_rank: r.vector_rank,
            bm25_rank: r.bm25_rank,
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
            section_path: None,
            page: None,
            rerank_score: r.rerank_score,
            tags: Vec::new(),
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
pub chunk_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

impl From<(rag_engine::api::source_rag::ChunkSearchResult, Option<String>, Option<u32>)> for ChunkResult {
    fn from((r, section_path, page): (rag_engine::api::source_rag::ChunkSearchResult, Option<String>, Option<u32>)) -> Self {
        ChunkResult {
            chunk_id: r.chunk_id,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            content: r.content,
            score: r.similarity,
            metadata: r.metadata,
chunk_type: r.chunk_type,
            section_path,
            page,
        }
    }
}

// --- Benchmark types ---

/// A single entry in the golden dataset JSON file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GoldenEntry {
    pub question: String,
    pub expected_keywords: Vec<String>,
    pub relevant_source_ids: Vec<i64>,
}

/// Result of a single query within the benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDetail {
    pub query: String,
    pub expected_source_ids: Vec<i64>,
    pub found_source_ids: Vec<i64>,
    pub score: f64,
    pub is_hit: bool,
}

/// Aggregate result of running a benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub total_queries: usize,
    pub hits: usize,
    pub misses: usize,
    pub avg_score: f64,
    pub details: Vec<BenchmarkDetail>,
}

// --- Graph types ---

/// Graph data for document similarity visualization.
#[derive(Debug, Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: i64,
    pub name: String,
    pub collection: String,
    pub chunk_count: i32,
}

#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: i64,
    pub target: i64,
    pub similarity: f32,
}
