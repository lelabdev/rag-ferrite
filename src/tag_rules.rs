use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct TagRules {
    #[serde(default)]
    pub synonyms: HashMap<String, String>,
    #[serde(default)]
    pub stop_words: StopWords,
    #[serde(default)]
    pub rules: TagRulesConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StopWords {
    #[serde(default)]
    pub words: Vec<String>,
    #[serde(default)]
    pub meta: Vec<String>,
    #[serde(default)]
    pub technical: Vec<String>,
    #[serde(default)]
    pub emotional: Vec<String>,
    #[serde(default)]
    pub noise: Vec<String>,
}

impl StopWords {
    /// All stop words combined into a single set for O(1) lookup.
    pub fn all(&self) -> Vec<&str> {
        let mut all: Vec<&str> = Vec::new();
        all.extend(self.words.iter().map(|s| s.as_str()));
        all.extend(self.meta.iter().map(|s| s.as_str()));
        all.extend(self.technical.iter().map(|s| s.as_str()));
        all.extend(self.emotional.iter().map(|s| s.as_str()));
        all.extend(self.noise.iter().map(|s| s.as_str()));
        all
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct TagRulesConfig {
    #[serde(default = "default_min_length")]
    pub min_length: usize,
    #[serde(default = "default_max_words")]
    pub max_words: usize,
    #[serde(default = "default_strip_chars")]
    pub strip_chars: String,
}

fn default_min_length() -> usize { 3 }
fn default_max_words() -> usize { 3 }
fn default_strip_chars() -> String { "*$`\"<>|={}[]/".to_string() }

impl Default for TagRules {
    fn default() -> Self {
        Self {
            synonyms: HashMap::new(),
            stop_words: StopWords::default(),
            rules: TagRulesConfig::default(),
        }
    }
}

impl Default for StopWords {
    fn default() -> Self {
        Self {
            words: Vec::new(),
            meta: Vec::new(),
            technical: Vec::new(),
            emotional: Vec::new(),
            noise: Vec::new(),
        }
    }
}

impl Default for TagRulesConfig {
    fn default() -> Self {
        Self {
            min_length: 3,
            max_words: 3,
            strip_chars: "*$`\"<>|={}[]/".to_string(),
        }
    }
}

impl TagRules {
    pub fn load() -> Result<Self> {
        let paths = vec![
            PathBuf::from("tag-rules.toml"),
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("rag-ferrite")
                .join("tag-rules.toml"),
        ];

        for path in &paths {
            if path.exists() {
                let content = std::fs::read_to_string(path)?;
                let rules: TagRules = toml::from_str(&content)?;
                tracing::info!(
                    "Loaded tag rules from {} ({} synonyms, {} stop words)",
                    path.display(),
                    rules.synonyms.len(),
                    rules.stop_words.all().len(),
                );
                return Ok(rules);
            }
        }

        tracing::info!("No tag-rules.toml found, using defaults (no synonyms, no stop words)");
        Ok(TagRules::default())
    }
}
