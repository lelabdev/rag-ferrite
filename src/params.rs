//! Shared types — parameter structs used by both MCP tools and HTTP API.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Ingest config (shared by main.rs and api.rs) ─────────────────────

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub max_concurrent: usize,
    pub relevance_scoring: bool,
    pub min_relevance_score: f32,
    pub chunk_size: usize,
    pub context_batch_size: usize,
    pub context_max_retries: usize,
    pub chunk_overlap_ratio: f64,
    pub merge_last_chunk_threshold: usize,
    pub chunking_strategy: String,
    pub parent_max_chars: usize,
    pub child_max_chars: usize,
    pub child_overlap: usize,
    pub auto_threshold: usize,
    pub child_min_chars: usize,
    pub defer_index_rebuild: bool,
    pub wal_checkpoint_interval: usize,
    pub ingested_dir: String,
}

impl IngestConfig {
    pub fn to_engine_options(&self) -> crate::engine::IngestOptions {
        crate::engine::IngestOptions {
            max_concurrent: self.max_concurrent,
            relevance_scoring: self.relevance_scoring,
            min_relevance_score: self.min_relevance_score as f64,
            chunk_size: self.chunk_size,
            context_batch_size: self.context_batch_size,
            context_max_retries: self.context_max_retries,
            chunk_overlap_ratio: self.chunk_overlap_ratio,
            merge_last_chunk_threshold: self.merge_last_chunk_threshold,
            chunking_strategy: self.chunking_strategy.clone(),
            parent_max_chars: self.parent_max_chars,
            child_max_chars: self.child_max_chars,
            child_overlap: self.child_overlap,
            auto_threshold: self.auto_threshold,
            child_min_chars: self.child_min_chars,
            defer_index_rebuild: self.defer_index_rebuild,
            wal_checkpoint_interval: self.wal_checkpoint_interval,
        }
    }
}

// ── Shared parameter structs ─────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct QueryParams {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
    /// Filter by source IDs (document IDs)
    #[serde(default)]
    pub source_ids: Option<Vec<i64>>,
    /// Filter by metadata using SQL LIKE pattern (e.g. "%.pdf")
    #[serde(default)]
    pub metadata_like: Option<String>,
    /// Filter results by tags (chunks must have at least one of these tags)
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

pub fn default_limit() -> Option<usize> {
    Some(10)
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct IngestFileParams {
    pub file_path: String,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct IngestDataParams {
    pub content: String,
    pub source: String,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct DeleteParams {
    pub source: String,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ChunkNeighborsParams {
    pub source_id: i64,
    pub chunk_index: i64,
    #[serde(default = "default_before")]
    pub before: Option<i64>,
    #[serde(default = "default_after")]
    pub after: Option<i64>,
}

pub fn default_before() -> Option<i64> {
    Some(2)
}
pub fn default_after() -> Option<i64> {
    Some(2)
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct CheckIngestionParams {
    /// Path to the file to check
    pub file_path: Option<String>,
    /// Raw content to check (alternative to file_path)
    pub content: Option<String>,
    /// Source name for duplicate detection (used with content)
    pub source_name: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct BenchmarkParams {
    /// Path to the golden dataset JSON file
    pub file_path: String,
    /// Number of top results to consider per query (default: 10)
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct NoParams {}
