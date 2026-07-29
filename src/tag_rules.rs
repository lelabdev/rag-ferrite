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

/// Global tag rules — set once at startup, read by sanitize_tags()
static TAG_RULES: std::sync::OnceLock<TagRules> = std::sync::OnceLock::new();

/// Initialize tag rules. Call once at startup.
pub fn init_tag_rules(rules: TagRules) {
    let _ = TAG_RULES.set(rules);
}

/// Get the global tag rules (if initialized).
pub fn get_tag_rules() -> TagRules {
    TAG_RULES.get().cloned().unwrap_or_default()
}

/// Sanitize raw tags from LLM output using tag-rules.toml.
/// Multi-stage pipeline: strip → lowercase → synonyms → stop words → length → dedup.
pub fn sanitize_tags(raw_tags: Vec<String>) -> Vec<String> {
    let rules = TAG_RULES.get().cloned().unwrap_or_default();
    let all_stops = rules.stop_words.all();

    raw_tags.into_iter()
        // Stage 1: Strip special chars
        .map(|t| {
            let mut cleaned = t;
            for c in rules.rules.strip_chars.chars() {
                cleaned = cleaned.replace(c, "");
            }
            cleaned.replace('_', " ").replace('/', " ").trim().to_lowercase()
        })
        // Stage 2: Synonym normalization
        .map(|t| {
            if let Some(canonical) = rules.synonyms.get(&t) {
                canonical.clone()
            } else {
                t
            }
        })
        // Stage 3: Filter
        .filter(|t| {
            if t.is_empty() { return false; }
            if t.len() < rules.rules.min_length { return false; }
            if t.split(|c: char| c == ' ' || c == '-' || c == '_').count() > rules.rules.max_words { return false; }
            if all_stops.contains(&t.as_str()) { return false; }
            if t.chars().all(|c| c.is_numeric()) { return false; }
            true
        })
        // Stage 4: Simple singular normalization
        .map(|mut t| {
            if !t.contains(' ') && !t.contains('-') && t.len() > 4
                && t.ends_with('s')
                && !t.ends_with("ss") && !t.ends_with("us") && !t.ends_with("is")
                && !t.ends_with("as") && !t.ends_with("os")
            {
                t = t[..t.len()-1].to_string();
            }
            t
        })
        // Stage 5: Deduplicate
        .fold(Vec::new(), |mut acc, t| {
            if !acc.contains(&t) {
                acc.push(t);
            }
            acc
        })
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
