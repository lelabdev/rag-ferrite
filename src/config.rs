use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;

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

    /// Reranker configuration
    #[serde(default)]
    pub reranker: RerankerConfig,

    /// HTTP server port (0 = disabled, stdio-only mode)
    #[serde(default)]
    pub http_port: u16,
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

    /// Embedding dimensions
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,

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

fn default_dimensions() -> usize {
    1024
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            api_key: None,
            base_url: None,
            dimensions: default_dimensions(),
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

    /// Fallback LLM config — used when primary fails (rate limit, network, etc.)
    #[serde(default)]
    pub fallback: Option<FallbackLlmConfig>,
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
    "gemma4:e4b".into()
}

fn default_context_enabled() -> bool {
    true
}

fn default_max_concurrent() -> usize {
    3
}

fn default_min_relevance_score() -> f32 {
    5.0
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            model: default_llm_model(),
            api_key: None,
            base_url: None,
            context_enabled: true,
            max_concurrent: 3,
            relevance_scoring: false,
            min_relevance_score: default_min_relevance_score(),
            fallback: None,
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
}

fn default_reranker_type() -> String { "disabled".into() }
fn default_rerank_top_k() -> usize { 10 }

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            reranker_type: default_reranker_type(),
            model: None,
            api_key: None,
            base_url: None,
            top_k: default_rerank_top_k(),
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
            dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("rag-ferrite").join("config.toml"),
        ];

        for path in &paths {
            if path.exists() {
                let content = std::fs::read_to_string(path)?;
                let config: Config = toml::from_str(&content)?;
                tracing::info!("Loaded config from {}", path.display());
                return Ok(config);
            }
        }

        tracing::info!("No config file found, using defaults");
        Ok(Config::default())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            embedding: EmbeddingConfig::default(),
            llm: LlmConfig::default(),
            reranker: RerankerConfig::default(),
            http_port: 0,
        }
    }
}
