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
    "openai".into()
}

fn default_model() -> String {
    "text-embedding-3-small".into()
}

fn default_dimensions() -> usize {
    1536
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

fn dirs_data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|p| p.join("rag-lab"))
}

impl Config {
pub fn load() -> Result<Self> {
        // Try config.toml in current dir, then ~/.config/rag-lab/config.toml
        let paths = vec![
            PathBuf::from("config.toml"),
            dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("rag-lab").join("config.toml"),
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
        }
    }
}
