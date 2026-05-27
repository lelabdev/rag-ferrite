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
    /// Filter by collection name
    #[serde(default)]
    pub collection: Option<String>,
}

pub fn default_limit() -> Option<usize> {
    Some(10)
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct IngestFileParams {
    pub file_path: String,
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct IngestDataParams {
    pub content: String,
    pub source: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub collection: Option<String>,
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
    /// Optional collection to filter queries against
    #[serde(default)]
    pub collection: Option<String>,
    /// Number of top results to consider per query (default: 10)
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct NoParams {}
