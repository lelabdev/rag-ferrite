use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Global heat config — set once at startup, read by heat module.
static HEAT_CONFIG: OnceLock<HeatConfig> = OnceLock::new();

/// Store heat config globally so the heat module can access decay factors.
pub fn set_global_heat(config: HeatConfig) {
    let _ = HEAT_CONFIG.set(config);
}

/// Get the global heat config (if set).
pub fn get_heat_config() -> Option<&'static HeatConfig> {
    HEAT_CONFIG.get()
}

#[derive(Debug, Deserialize)]
pub struct Config {
    /// Directory for SQLite databases and indexes
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Embedding API configuration
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// LLM configuration for contextual retrieval
    #[serde(default)]
    pub llm: LlmConfig,

    /// Named LLM profiles — when defined, [llm] action fields reference profiles by name.
    /// If empty, the legacy single-provider [llm] config is used for all actions.
    #[serde(default)]
    pub llm_profile: Vec<LlmProfile>,

    /// Reranker configuration
    #[serde(default)]
    pub reranker: RerankerConfig,

    /// Advanced configuration (tunable parameters)
    #[serde(default)]
    pub advanced: AdvancedConfig,

    /// Chunking strategy configuration
    #[serde(default)]
    pub chunking: ChunkingConfig,

    /// Query fallback LLM — used for queries during active ingestion
    /// to avoid saturating the main LLM provider
    #[serde(default)]
    pub query_fallback: Option<FallbackLlmConfig>,

    /// HTTP server port (0 = disabled, stdio-only mode)
    #[serde(default)]
    pub http_port: u16,

    /// Query classification keywords and thresholds
    #[serde(default)]
    pub query_classification: QueryClassificationConfig,

    /// Heat tracking configuration (decay factors for collection + chunk level)
    #[serde(default)]
    pub heat: HeatConfig,
}

/// Heat tracking configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct HeatConfig {
    /// Daily decay factor for collection-level heat (0-1).
    /// Higher = slower decay. 0.99 = 74% after 30 days.
    #[serde(default = "default_collection_decay")]
    pub collection_decay: f64,

    /// Daily decay factor for chunk-level QA (0-1).
    /// Slower than collection: 0.999 = 69% after 1 year.
    #[serde(default = "default_chunk_decay")]
    pub chunk_decay: f64,
}

fn default_collection_decay() -> f64 { 0.99 }
fn default_chunk_decay() -> f64 { 0.999 }

impl Default for HeatConfig {
    fn default() -> Self {
        Self {
            collection_decay: default_collection_decay(),
            chunk_decay: default_chunk_decay(),
        }
    }
}

/// Query classification config — controls how queries are routed
/// to Simple / Standard / Complex pipelines.
#[derive(Debug, Deserialize, Clone)]
pub struct QueryClassificationConfig {
    /// Words that mark a query as Complex (question markers)
    #[serde(default = "default_question_markers")]
    pub question_markers: Vec<String>,

    /// Boolean operators that mark a query as Complex
    #[serde(default = "default_boolean_operators")]
    pub boolean_operators: Vec<String>,

    /// Word count above which a query is Complex (> threshold)
    #[serde(default = "default_complex_word_threshold")]
    pub complex_word_threshold: usize,

    /// Word count at or below which a query is Simple (<= threshold)
    #[serde(default = "default_simple_word_threshold")]
    pub simple_word_threshold: usize,
}

impl Default for QueryClassificationConfig {
    fn default() -> Self {
        Self {
            question_markers: default_question_markers(),
            boolean_operators: default_boolean_operators(),
            complex_word_threshold: default_complex_word_threshold(),
            simple_word_threshold: default_simple_word_threshold(),
        }
    }
}

fn default_question_markers() -> Vec<String> {
    vec![
        "what".into(), "how".into(), "why".into(), "when".into(),
        "where".into(), "which".into(), "who".into(), "whom".into(),
        "whose".into(), "whether".into(),
        "comment".into(), "pourquoi".into(), "quand".into(), "où".into(),
        "quel".into(), "quelle".into(), "quels".into(), "quelles".into(),
        "qui".into(),
    ]
}

fn default_boolean_operators() -> Vec<String> {
    vec!["AND".into(), "OR".into(), "et".into(), "ou".into()]
}

/// Dictionary file for query classification keywords.
#[derive(Debug, Deserialize, Clone)]
struct QueryClassificationDictionary {
    question_markers: Vec<String>,
    boolean_operators: Vec<String>,
}

/// Try to load query classification dictionaries from well-known locations.
///
/// Search order:
/// 1. `<data_dir>/dictionaries/query_classification.toml`
/// 2. `<config_dir>/dictionaries/query_classification.toml` (directory containing config.toml)
///
/// Returns `None` if no dictionary file is found (hardcoded defaults used).
fn load_query_classification_dictionary(
    data_dir: &std::path::Path,
    config_path: Option<&std::path::Path>,
) -> Option<QueryClassificationDictionary> {
    let mut candidates: Vec<PathBuf> = vec![
        data_dir.join("dictionaries").join("query_classification.toml"),
    ];

    if let Some(cfg) = config_path {
        if let Some(parent) = cfg.parent() {
            candidates.push(parent.join("dictionaries").join("query_classification.toml"));
        }
    }

    for path in &candidates {
        if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(content) => match toml::from_str::<QueryClassificationDictionary>(&content) {
                    Ok(dict) => {
                        tracing::info!(
                            "Loaded query classification dictionary from {}",
                            path.display()
                        );
                        return Some(dict);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse dictionary {}: {e}",
                            path.display()
                        );
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "Failed to read dictionary {}: {e}",
                        path.display()
                    );
                }
            }
        }
    }

    None
}

fn default_complex_word_threshold() -> usize { 8 }
fn default_simple_word_threshold() -> usize { 2 }

/// A named LLM profile with its own provider, model, base_url, and optional API key env var.
#[derive(Debug, Deserialize, Clone)]
pub struct LlmProfile {
    /// Profile name — referenced by ingestion_profile, query_profile, reranker_profile.
    pub name: String,
    /// LLM provider: "ollama", "openai_compatible", etc.
    pub provider: String,
    /// Model name (e.g. "gemma4:31b", "ministral-3:3b").
    pub model: String,
    /// API base URL.
    pub base_url: String,
    /// Environment variable name holding the API key (defaults to "LLM_API_KEY").
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
}

fn default_api_key_env() -> String {
    "LLM_API_KEY".into()
}

impl Config {
    /// Look up an LLM profile by name. Returns None if no profiles are defined
    /// or the name doesn't match.
    pub fn get_profile(&self, name: &str) -> Option<&LlmProfile> {
        self.llm_profile.iter().find(|p| p.name == name)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider: "openai", "cohere", "ollama"
    #[serde(default = "default_provider")]
    pub provider: String,

    /// API key (for cloud providers)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model name
    #[serde(default = "default_model")]
    pub model: String,

    /// Embedding dimensions (auto-detected from API if not set)
    #[serde(default)]
    pub dimensions: Option<usize>,

    /// API base URL (for Ollama or custom endpoints)
    #[serde(default)]
    pub base_url: Option<String>,
}

fn default_data_dir() -> PathBuf {
    dirs_data_dir().unwrap_or_else(|| PathBuf::from("./data"))
}

fn default_provider() -> String {
    "ollama".into()
}

fn default_model() -> String {
    "qwen3-embedding:0.6b".into()
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            base_url: None,
            api_key: None,
            dimensions: None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct LlmConfig {
    /// LLM provider: "ollama", "zai", "openai"
    #[serde(default = "default_llm_provider")]
    pub provider: String,

    /// Model name
    #[serde(default = "default_llm_model")]
    pub model: String,

    /// API key (for cloud providers)
    #[serde(default)]
    pub api_key: Option<String>,

    /// API base URL
    #[serde(default)]
    pub base_url: Option<String>,

    /// Enable contextual retrieval (LLM context prefix)
    #[serde(default = "default_context_enabled")]
    pub context_enabled: bool,

    /// Max concurrent LLM requests for contextual retrieval
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Enable relevance scoring during ingestion — LLM rates each chunk 1-10
    /// and chunks below min_relevance_score are filtered out before embedding.
    #[serde(default)]
    pub relevance_scoring: bool,

    /// Minimum relevance score to keep a chunk (1-10). Only used when relevance_scoring = true.
    /// Default: 5.0 (filters out noise like TOC, index, legal mentions).
    #[serde(default = "default_min_relevance_score")]
    pub min_relevance_score: f32,

    /// Default temperature for scoring/tagging calls
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Default max tokens for scoring/tagging calls
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for query expansion and reformulation
    #[serde(default = "default_expansion_temperature")]
    pub expansion_temperature: f64,

    /// Max tokens for query expansion and reformulation
    #[serde(default = "default_expansion_max_tokens")]
    pub expansion_max_tokens: usize,

    /// Max generated expansion queries per original query
    #[serde(default = "default_max_expansion_queries")]
    pub max_expansion_queries: usize,

    /// Max document chars sent to LLM in prompt (truncation)
    #[serde(default = "default_max_document_prompt_chars")]
    pub max_document_prompt_chars: usize,

    /// Max chunk chars sent to LLM in prompt (truncation)
    #[serde(default = "default_max_chunk_prompt_chars")]
    pub max_chunk_prompt_chars: usize,

    /// Batch size for context generation during ingestion
    #[serde(default = "default_context_batch_size")]
    pub context_batch_size: usize,

    /// Max retries for failed context generation per chunk
    #[serde(default = "default_context_max_retries")]
    pub context_max_retries: usize,

    /// Fallback LLM config — used when primary fails (rate limit, network, etc.)
    #[serde(default)]
    pub fallback: Option<FallbackLlmConfig>,

    // ── Profile-based action assignment ──
    // When [[llm_profile]] entries exist, these reference profile names.
    // When no profiles are defined, the legacy single-provider fields above are used.

    /// Profile name for ingestion (contextualisation during ingestion).
    /// If set and the profile exists, overrides provider/model/base_url/api_key.
    #[serde(default)]
    pub ingestion_profile: Option<String>,

    /// Profile name for query (expansion + reformulation).
    #[serde(default)]
    pub query_profile: Option<String>,

    /// Profile name for reranking results.
    #[serde(default)]
    pub reranker_profile: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FallbackLlmConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

fn default_llm_provider() -> String {
    "ollama".into()
}

fn default_llm_model() -> String {
    "gemma4:31b".into()
}

fn default_context_enabled() -> bool {
    true
}

fn default_relevance_scoring() -> bool {
    false
}

fn default_max_concurrent() -> usize {
    3
}

fn default_min_relevance_score() -> f32 {
    5.0
}
fn default_temperature() -> f64 {
    0.3
}
fn default_max_tokens() -> usize {
    150
}
fn default_expansion_temperature() -> f64 {
    0.7
}
fn default_expansion_max_tokens() -> usize {
    200
}
fn default_max_expansion_queries() -> usize {
    4
}
fn default_max_document_prompt_chars() -> usize {
    8000
}
fn default_max_chunk_prompt_chars() -> usize {
    2000
}
fn default_context_batch_size() -> usize {
    3
}
fn default_context_max_retries() -> usize {
    3
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            model: default_llm_model(),
            api_key: None,
            base_url: None,
            context_enabled: default_context_enabled(),
            max_concurrent: 3,
            relevance_scoring: default_relevance_scoring(),
            min_relevance_score: default_min_relevance_score(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            expansion_temperature: default_expansion_temperature(),
            expansion_max_tokens: default_expansion_max_tokens(),
            max_expansion_queries: default_max_expansion_queries(),
            max_document_prompt_chars: default_max_document_prompt_chars(),
            max_chunk_prompt_chars: default_max_chunk_prompt_chars(),
            context_batch_size: default_context_batch_size(),
            context_max_retries: default_context_max_retries(),
            fallback: None,
            ingestion_profile: None,
            query_profile: None,
            reranker_profile: None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RerankerConfig {
    /// Reranker type: "disabled", "llm", "cohere"
    #[serde(default = "default_reranker_type")]
    pub reranker_type: String,

    /// Model to use for LLM reranking (defaults to llm.model)
    #[serde(default)]
    pub model: Option<String>,

    /// API key (defaults to llm.api_key)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL (defaults to llm.base_url)
    #[serde(default)]
    pub base_url: Option<String>,

    /// Number of top results to rerank (default 10)
    #[serde(default = "default_rerank_top_k")]
    pub top_k: usize,

    /// Max chars of chunk content sent to reranker
    #[serde(default = "default_rerank_preview_chars")]
    pub preview_chars: usize,
}

fn default_reranker_type() -> String { "disabled".into() }
fn default_rerank_top_k() -> usize { 10 }
fn default_rerank_preview_chars() -> usize { 300 }

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            reranker_type: default_reranker_type(),
            model: None,
            api_key: None,
            base_url: None,
            top_k: default_rerank_top_k(),
            preview_chars: default_rerank_preview_chars(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChunkingConfig {
    /// Chunking strategy: "recursive" (default), "parent_child", or "auto"
    #[serde(default = "default_chunking_strategy")]
    pub strategy: String,

    /// Parent chunk size in characters (for parent_child mode)
    #[serde(default = "default_parent_max_chars")]
    pub parent_max_chars: usize,

    /// Child chunk size in characters (for parent_child mode)
    #[serde(default = "default_child_max_chars")]
    pub child_max_chars: usize,

    /// Child chunk overlap in characters
    #[serde(default = "default_child_overlap")]
    pub child_overlap: usize,

    /// Auto-switch threshold: docs >= this size use parent_child (for "auto" mode)
    #[serde(default = "default_auto_threshold")]
    pub auto_threshold: usize,

    /// Min child chars — consecutive children below this are merged into one chunk
    #[serde(default = "default_child_min_chars")]
    pub child_min_chars: usize,
}

fn default_chunking_strategy() -> String { "auto".into() }
fn default_parent_max_chars() -> usize { 2000 }
fn default_child_max_chars() -> usize { 200 }
fn default_child_overlap() -> usize { 20 }
fn default_child_min_chars() -> usize { 100 }
fn default_auto_threshold() -> usize { 5000 }

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: default_chunking_strategy(),
            parent_max_chars: default_parent_max_chars(),
            child_max_chars: default_child_max_chars(),
            child_overlap: default_child_overlap(),
            auto_threshold: default_auto_threshold(),
            child_min_chars: default_child_min_chars(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AdvancedConfig {
    /// Chunk size in characters for document splitting
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Chunk overlap ratio (0.0-0.5, fraction of chunk_size)
    #[serde(default = "default_chunk_overlap_ratio")]
    pub chunk_overlap_ratio: f64,

    /// Merge last chunk if smaller than this (chars)
    #[serde(default = "default_merge_last_chunk_threshold")]
    pub merge_last_chunk_threshold: usize,

    /// Cache TTL in seconds for query results
    #[serde(default = "default_cache_ttl_secs")]
    pub cache_ttl_secs: u64,

    /// Maximum cache entries before eviction
    #[serde(default = "default_cache_max_entries")]
    pub cache_max_entries: usize,

    /// Default query result limit
    #[serde(default = "default_query_limit")]
    pub default_query_limit: usize,

    /// Maximum query result limit (upper bound)
    #[serde(default = "default_max_query_limit")]
    pub max_query_limit: usize,

    /// Quality threshold for corrective RAG (0.0-1.0)
    #[serde(default = "default_quality_threshold")]
    pub quality_threshold: f64,

    /// Max retries for query reformulation
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// High confidence threshold — skip reranking if top score exceeds this
    #[serde(default = "default_high_confidence_threshold")]
    pub high_confidence_threshold: f64,

    /// Embedding batch size (number of texts per API call)
    #[serde(default = "default_embedding_batch_size")]
    pub embedding_batch_size: usize,

    /// Database connection pool size
    #[serde(default = "default_db_pool_size")]
    pub db_pool_size: usize,

    /// SQLite page cache size in MB
    #[serde(default = "default_db_cache_size_mb")]
    pub db_cache_size_mb: usize,

    /// SQLite busy timeout in milliseconds
    #[serde(default = "default_db_busy_timeout_ms")]
    pub db_busy_timeout_ms: usize,

    /// Log file path (relative to working directory)
    #[serde(default = "default_log_file")]
    pub log_file: String,

    /// Log filter (tracing syntax)
    #[serde(default = "default_log_filter")]
    pub log_filter: String,

    /// HTTP bind address
    #[serde(default = "default_http_bind_address")]
    pub http_bind_address: String,

    /// Defer HNSW + BM25 index rebuild to explicit flush (saves RAM during batch ingestion)
    #[serde(default = "default_defer_index_rebuild")]
    pub defer_index_rebuild: bool,

    /// WAL checkpoint every N parents committed (0 = disabled)
    #[serde(default = "default_wal_checkpoint_interval")]
    pub wal_checkpoint_interval: usize,

    /// Move files to ingested_dir after successful ingestion (default: true)
    #[serde(default = "default_move_after_ingest")]
    pub move_after_ingest: bool,

    /// Directory name for ingested files (default: "ingested")
    #[serde(default = "default_ingested_dir")]
    pub ingested_dir: String,
}

fn default_chunk_size() -> usize { 800 }
fn default_chunk_overlap_ratio() -> f64 { 0.1 }
fn default_merge_last_chunk_threshold() -> usize { 200 }
fn default_cache_ttl_secs() -> u64 { 300 }
fn default_cache_max_entries() -> usize { 1000 }
fn default_query_limit() -> usize { 10 }
fn default_max_query_limit() -> usize { 100 }
fn default_quality_threshold() -> f64 { 0.3 }
fn default_max_retries() -> usize { 1 }
fn default_high_confidence_threshold() -> f64 { 0.7 }
fn default_embedding_batch_size() -> usize { 20 }
fn default_db_pool_size() -> usize { 4 }
fn default_db_cache_size_mb() -> usize { 256 }
fn default_db_busy_timeout_ms() -> usize { 5000 }
fn default_log_file() -> String { "rag-ferrite.log".into() }
fn default_log_filter() -> String { "rag_ferrite=debug".into() }
fn default_http_bind_address() -> String { "0.0.0.0".into() }
fn default_defer_index_rebuild() -> bool { true }
fn default_wal_checkpoint_interval() -> usize { 50 }
fn default_move_after_ingest() -> bool { true }
fn default_ingested_dir() -> String { "ingested".into() }

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap_ratio: default_chunk_overlap_ratio(),
            merge_last_chunk_threshold: default_merge_last_chunk_threshold(),
            cache_ttl_secs: default_cache_ttl_secs(),
            cache_max_entries: default_cache_max_entries(),
            default_query_limit: default_query_limit(),
            max_query_limit: default_max_query_limit(),
            quality_threshold: default_quality_threshold(),
            max_retries: default_max_retries(),
            high_confidence_threshold: default_high_confidence_threshold(),
            embedding_batch_size: default_embedding_batch_size(),
            db_pool_size: default_db_pool_size(),
            db_cache_size_mb: default_db_cache_size_mb(),
            db_busy_timeout_ms: default_db_busy_timeout_ms(),
            log_file: default_log_file(),
            log_filter: default_log_filter(),
            http_bind_address: default_http_bind_address(),
            defer_index_rebuild: default_defer_index_rebuild(),
            wal_checkpoint_interval: default_wal_checkpoint_interval(),
            move_after_ingest: default_move_after_ingest(),
            ingested_dir: default_ingested_dir(),
        }
    }
}

fn dirs_data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|p| p.join("rag-ferrite"))
}

impl Config {
    pub fn load() -> Result<Self> {
        // Try config.toml in current dir, then ~/.config/rag-ferrite/config.toml
        let paths = vec![
            PathBuf::from("config.toml"),
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("rag-ferrite")
                .join("config.toml"),
        ];

        for path in &paths {
            if path.exists() {
                let content = std::fs::read_to_string(path)?;
                let mut config: Config = toml::from_str(&content)?;
                // Validate min_relevance_score
                if config.llm.min_relevance_score.is_nan()
                    || config.llm.min_relevance_score < 0.0
                    || config.llm.min_relevance_score > 10.0
                {
                    anyhow::bail!(
                        "min_relevance_score must be between 0.0 and 10.0, got {}",
                        config.llm.min_relevance_score
                    );
                }

                // Try loading query classification dictionaries (optional override)
                if let Some(dict) =
                    load_query_classification_dictionary(&config.data_dir, Some(path))
                {
                    config.query_classification.question_markers = dict.question_markers;
                    config.query_classification.boolean_operators = dict.boolean_operators;
                }

                tracing::info!("Loaded config from {}", path.display());
                return Ok(config);
            }
        }

        tracing::info!("No config file found, using defaults");
        let mut config = Config::default();

        // Even without a config file, try loading dictionaries from data_dir
        if let Some(dict) =
            load_query_classification_dictionary(&config.data_dir, None)
        {
            config.query_classification.question_markers = dict.question_markers;
            config.query_classification.boolean_operators = dict.boolean_operators;
        }

        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            embedding: EmbeddingConfig::default(),
            llm: LlmConfig::default(),
            llm_profile: Vec::new(),
            reranker: RerankerConfig::default(),
            advanced: AdvancedConfig::default(),
            chunking: ChunkingConfig::default(),
            query_fallback: None,
            http_port: 0,
            query_classification: QueryClassificationConfig::default(),
            heat: HeatConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.llm.provider, "ollama");
        assert_eq!(config.llm.model, "gemma4:31b");
        assert!(config.llm.context_enabled);
        assert!(!config.llm.relevance_scoring);
        assert_eq!(config.llm.min_relevance_score, 5.0);
        assert_eq!(config.llm.max_concurrent, 3);
        assert!(config.llm.api_key.is_none());
        assert!(config.llm.base_url.is_none());
        assert!(config.llm.fallback.is_none());
    }

    #[test]
    fn test_default_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.model, "qwen3-embedding:0.6b");
        assert_eq!(config.dimensions, None);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_default_reranker_config() {
        let config = RerankerConfig::default();
        assert_eq!(config.reranker_type, "disabled");
        assert_eq!(config.top_k, 10);
        assert!(config.model.is_none());
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
data_dir = "/tmp/rag-test"

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536
base_url = "https://api.openai.com/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0
max_concurrent = 3

[reranker]
reranker_type = "disabled"
top_k = 10
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/rag-test"));
        assert_eq!(config.embedding.provider, "openai");
        assert_eq!(config.embedding.model, "text-embedding-3-small");
        assert_eq!(config.embedding.dimensions, Some(1536));
        assert_eq!(config.llm.provider, "ollama");
        assert_eq!(config.llm.model, "gemma4:31b");
        assert_eq!(config.llm.base_url.as_deref(), Some("https://api.ollama.com"));
        assert!(config.llm.context_enabled);
        assert!(config.llm.relevance_scoring);
        assert_eq!(config.llm.min_relevance_score, 5.0);
        assert_eq!(config.reranker.reranker_type, "disabled");
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
data_dir = "/tmp/test"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/test"));
        // Everything else should be defaults
        assert_eq!(config.llm.provider, "ollama");
        assert!(!config.llm.relevance_scoring);
        assert_eq!(config.embedding.dimensions, None);
    }

    #[test]
    fn test_parse_config_with_fallback() {
        let toml = r#"
data_dir = "/tmp/test"

[llm]
provider = "zai"
model = "glm-4.5-flash"
base_url = "https://api.z.ai/api/coding/paas/v4"
context_enabled = false

[llm.fallback]
provider = "ollama"
model = "gemma4:31b"
base_url = "http://localhost:11434"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.llm.provider, "zai");
        assert_eq!(config.llm.model, "glm-4.5-flash");
        assert!(!config.llm.context_enabled);
        let fb = config.llm.fallback.unwrap();
        assert_eq!(fb.provider, "ollama");
        assert_eq!(fb.model, "gemma4:31b");
    }

    #[test]
    fn test_parse_config_invalid_toml() {
        let toml = r#"
this is not valid toml
"#;
        assert!(toml::from_str::<Config>(toml).is_err());
    }
}
