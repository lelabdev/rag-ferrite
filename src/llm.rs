use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Result of contextual retrieval: optional context prefix, optional relevance score (1-10),
/// optional extracted metadata, and auto-generated tags.
#[derive(Debug, Clone)]
pub struct ContextResult {
    pub context: Option<String>,
    pub relevance_score: Option<f32>,
    pub extracted_metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
}

/// LLM provider for contextual retrieval and other text generation tasks.
/// Supports a primary provider with an optional fallback for resilience.
#[derive(Debug, Clone)]
pub struct LlmProvider {
    provider: String,
    model: String,
    api_key: Option<String>,
    base_url: String,
    client: reqwest::Client,
    /// Fallback provider used when primary fails (rate limit, network, etc.)
    fallback: Option<Box<LlmProvider>>,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

impl LlmProvider {
    pub fn new(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let api_key = api_key
            .or_else(|| std::env::var("LLM_API_KEY").ok())
            .or_else(|| std::env::var("FALLBACK_API_KEY").ok());

        Self::build(provider, model, api_key, base_url)
    }

    pub fn new_fallback(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let api_key = api_key
            .or_else(|| std::env::var("FALLBACK_API_KEY").ok())
            .or_else(|| std::env::var("LLM_API_KEY").ok());

        Self::build(provider, model, api_key, base_url)
    }

    fn build(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let base_url = base_url.unwrap_or_else(|| {
            tracing::warn!("No base_url configured for LLM provider '{}', using http://localhost:11434", provider);
            "http://localhost:11434".into()
        });

        Self {
            provider,
            model,
            api_key,
            base_url,
            client: reqwest::Client::new(),
            fallback: None,
        }
    }

    /// Set a fallback LLM provider. Used when primary fails.
    pub fn with_fallback(mut self, fallback: LlmProvider) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }

    /// Generate a context prefix for a chunk using the LLM.
    /// Takes the whole document and a specific chunk, returns 1-2 sentences
    /// situating the chunk within the document.
    pub async fn generate_context(
        &self,
        whole_document: &str,
        chunk_content: &str,
        metadata_fields: Option<&[crate::config::MetadataField]>,
    ) -> Result<ContextResult> {
        let metadata_instructions = match metadata_fields {
            Some(fields) if !fields.is_empty() => {
                let field_descriptions: Vec<String> = fields.iter().map(|f| {
                    match &f.description {
                        Some(desc) => format!("  - {} ({}): {}", f.name, f.field_type, desc),
                        None => format!("  - {} ({})", f.name, f.field_type),
                    }
                }).collect();
                format!(
                    "\n\n\
             Additionally, extract the following metadata fields from the chunk:\n\
             {}\n\
             Also include a line with the extracted metadata as JSON:\n\
             METADATA: <json object with field names as keys>",
                    field_descriptions.join("\n")
                )
            }
            _ => String::new(),
        };

        let prompt = format!(
            "<document>\n{}\n</document>\n\n\
             Here is the chunk we want to situate within the whole document:\n\
             <chunk>\n{}\n</chunk>\n\n\
             Assess the relevance of this chunk for informative retrieval on a scale of 1 to 10, \
             where 1 is noise (TOC, index, legal mentions, boilerplate) and 10 is highly informative content.\n\
             Also give a short succinct context to situate this chunk within the overall document \
             for the purposes of improving search retrieval of the chunk.{}\n\n\
             Answer ONLY in this exact format:\n\
             SCORE: <number 1-10>\n\
             CONTEXT: <short succinct context, same language as document>\n\
             TAGS: <2-3 short tags describing the topic, comma-separated>\n\
             METADATA: <json object>{}",
            truncate_for_prompt(whole_document, 8000),
            truncate_for_prompt(chunk_content, 2000),
            &metadata_instructions,
            if metadata_fields.is_some() { "" } else { " (omit if not requested)" },
        );

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt,
        }];

        let response_text = match self.chat(messages.clone()).await {
            Ok(text) => text,
            Err(e) => {
                if let Some(ref fb) = self.fallback {
                    tracing::warn!("Primary LLM ({}/{}) failed: {}. Trying fallback ({}/{})",
                        self.provider, self.model, e, fb.provider, fb.model);
                    fb.chat(messages).await?
                } else {
                    return Err(e);
                }
            }
        };

        let trimmed = response_text.trim();
        let (score, context, extracted_metadata, tags) = parse_context_response(trimmed);
        Ok(ContextResult {
            context,
            relevance_score: score,
            extracted_metadata,
            tags,
        })
    }

    /// Generate context prefixes for multiple chunks in a single batch.
    /// Processes chunks concurrently (up to 10 at a time).
    pub async fn generate_context_batch(
        &self,
        whole_document: &str,
        chunks: &[String],
        max_concurrent: usize,
    ) -> Vec<Result<ContextResult>> {
        let sem = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let sem = sem.clone();
            let provider = self.clone();
            let doc = whole_document.to_string();
            let chunk_content = chunk.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                provider.generate_context(&doc, &chunk_content, None).await
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(anyhow!("Task join error: {}", e))),
            }
        }
        results
    }

    /// Send a chat completion request with default temperature and max_tokens.
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        self.chat_with_options(messages, 0.7, 4096).await
    }

    /// Expand a short or ambiguous query into 2-3 reformulations.
    /// Returns the original query + reformulations for broader retrieval.
    /// Gracefully degrades to just the original query if LLM is unavailable.
    pub async fn expand_query(&self, query: &str) -> Result<Vec<String>> {
        // If no API key and not using Ollama, skip expansion gracefully
        if self.provider.as_str() != "ollama" && self.api_key.is_none() {
            tracing::debug!(
                "Skipping query expansion: no API key for provider '{}'",
                self.provider
            );
            return Ok(vec![query.to_string()]);
        }

        let prompt = format!(
            "Generate 2 alternative reformulations of this search query to improve document retrieval. \
             Each reformulation should use different wording but seek the same information. \
             Return ONLY the reformulations, one per line, no numbering, no explanation.\n\n\
             Query: {}",
            query
        );

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt,
        }];

        let response = match self.chat_with_options(messages, 0.7, 200).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("Query expansion failed, using original query: {}", e);
                return Ok(vec![query.to_string()]);
            }
        };

        let mut expansions = vec![query.to_string()]; // original first

        for line in response.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed != query {
                expansions.push(trimmed.to_string());
            }
        }

        // Cap at 4 total (original + 3 reformulations)
        expansions.truncate(4);
        Ok(expansions)
    }

    /// Reformulate a query for corrective RAG — used when retrieval confidence is low.
    /// Returns a single reformulated query that approaches the information need differently.
    pub async fn reformulate_query(&self, query: &str) -> Result<String> {
        let prompt = format!(
            "The following search query returned poor results. Reformulate it to be more specific \
             and find better matches. Use different keywords and phrasing. \
             Return ONLY the reformulated query, nothing else.\n\n\
             Original query: {}",
            query
        );

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt,
        }];

        let response = self.chat_with_options(messages, 0.7, 200).await?;
        Ok(response.trim().to_string())
    }

    /// Chat with custom temperature and max_tokens.
    async fn chat_with_options(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        max_tokens: u32,
    ) -> Result<String> {
        match self.provider.as_str() {
            "ollama" => {
                let url = format!("{}/api/chat", self.base_url);

                #[derive(Debug, Serialize)]
                struct OllamaChatRequest {
                    model: String,
                    messages: Vec<ChatMessage>,
                    stream: bool,
                    options: OllamaOptions,
                }

                #[derive(Debug, Serialize)]
                struct OllamaOptions {
                    temperature: f32,
                    num_predict: u32,
                }

                let body = OllamaChatRequest {
                    model: self.model.clone(),
                    messages,
                    stream: false,
                    options: OllamaOptions {
                        temperature,
                        num_predict: max_tokens,
                    },
                };

                let mut req = self.client.post(&url).json(&body);
                if let Some(ref api_key) = self.api_key {
                    req = req.header("Authorization", format!("Bearer {}", api_key));
                }
                let resp = req.send().await?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await?;
                    return Err(anyhow!("Ollama API error {}: {}", status, text));
                }

                #[derive(Debug, Deserialize)]
                struct OllamaChatResponse {
                    message: ChatMessage,
                }

                let data: OllamaChatResponse = resp.json().await?;
                Ok(data.message.content)
            }
            _ => {
                let api_key = self.api_key.as_ref()
                    .ok_or_else(|| anyhow!("API key required for {}. Set LLM_API_KEY or FALLBACK_API_KEY.", self.provider))?;

                let url = format!("{}/chat/completions", self.base_url);

                let body = ChatRequest {
                    model: self.model.clone(),
                    messages,
                    temperature,
                    max_tokens,
                    thinking: Some(serde_json::json!({"type": "disabled"})),
                };

                let resp = self.client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .json(&body)
                    .send()
                    .await?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await?;
                    return Err(anyhow!("{} API error {}: {}", self.provider, status, text));
                }

                let data: ChatResponse = resp.json().await?;
                data.choices
                    .into_iter()
                    .next()
                    .map(|c| c.message.content)
                    .ok_or_else(|| anyhow!("No response from {}", self.provider))
            }
        }
    }


}

/// Truncate text to fit within token limits (rough: ~4 chars per token).
fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        // Try to cut at a sentence boundary
        let truncated = &text[..max_chars];
        if let Some(pos) = truncated.rfind('.') {
            format!("{}...", &text[..=pos])
        } else {
            format!("{}...", truncated)
        }
    }
}

use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_response() {
        let response = "SCORE: 8\nCONTEXT: This chunk describes Svelte runes and their usage in Svelte 5.\nMETADATA: {\"topic\": \"svelte\", \"version\": 5}";
        let (score, context, metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(8.0));
        assert_eq!(context, Some("This chunk describes Svelte runes and their usage in Svelte 5.".to_string()));
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta["topic"], "svelte");
        assert_eq!(meta["version"], 5);
    }

    #[test]
    fn test_parse_score_only() {
        let response = "SCORE: 3\nCONTEXT: Low relevance content";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(3.0));
        assert_eq!(context, Some("Low relevance content".to_string()));
    }

    #[test]
    fn test_parse_max_score() {
        let response = "SCORE: 10\nCONTEXT: Excellent chunk";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(10.0));
        assert_eq!(context, Some("Excellent chunk".to_string()));
    }

    #[test]
    fn test_parse_min_score() {
        let response = "SCORE: 1\nCONTEXT: This is noise or boilerplate";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(1.0));
    }

    #[test]
    fn test_parse_no_score_no_context() {
        // Backward compat: no SCORE/CONTEXT → whole response becomes context
        let response = "This is just some text without any structured format.";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        assert_eq!(score, None);
        assert_eq!(context, Some(response.to_string()));
    }

    #[test]
    fn test_parse_empty_response() {
        let response = "";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        assert_eq!(score, None);
        assert_eq!(context, None);
    }

    #[test]
    fn test_parse_multiline_context() {
        let response = "SCORE: 7\nCONTEXT: This is the first line\nand this is the second line\nand a third line\nMETADATA: {\"type\": \"api\"}";
        let (score, context, metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(7.0));
        assert!(context.is_some());
        let ctx = context.unwrap();
        assert!(ctx.contains("first line"));
        assert!(ctx.contains("second line"));
        assert!(ctx.contains("third line"));
        assert!(metadata.is_some());
    }

    #[test]
    fn test_parse_invalid_score() {
        // Non-numeric score should be ignored → backward compat
        let response = "SCORE: abc\nCONTEXT: Some context";
        let (score, context, _metadata, _tags) = parse_context_response(response);
        // "SCORE: abc" fails to parse, so found_score stays false
        // but "CONTEXT:" is found, so found_context is true
        assert_eq!(score, None);
        assert_eq!(context, Some("Some context".to_string()));
    }

    #[test]
    fn test_parse_metadata_invalid_json() {
        let response = "SCORE: 5\nCONTEXT: Test\nMETADATA: {not valid json}";
        let (score, _context, metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(5.0));
        // Invalid JSON metadata should be None
        assert!(metadata.is_none());
    }

    #[test]
    fn test_parse_metadata_only() {
        let response = "SCORE: 9\nCONTEXT: Good chunk\nMETADATA: {\"domain\": \"frontend\", \"framework\": \"svelte\"}";
        let (score, _context, metadata, _tags) = parse_context_response(response);
        assert_eq!(score, Some(9.0));
        let meta = metadata.unwrap();
        assert_eq!(meta["domain"], "frontend");
        assert_eq!(meta["framework"], "svelte");
    }

    #[test]
    fn test_parse_tags_basic() {
        let response = "SCORE: 8\nCONTEXT: Rust programming language features.\nTAGS: rust, programming, systems";
        let (score, context, _metadata, tags) = parse_context_response(response);
        assert_eq!(score, Some(8.0));
        assert_eq!(context, Some("Rust programming language features.".to_string()));
        assert_eq!(tags, vec!["rust", "programming", "systems"]);
    }

    #[test]
    fn test_parse_tags_with_extra_whitespace() {
        let response = "SCORE: 7\nCONTEXT: Some context\nTAGS:  machine-learning ,  neural-networks , ai ";
        let (_score, _context, _metadata, tags) = parse_context_response(response);
        assert_eq!(tags, vec!["machine-learning", "neural-networks", "ai"]);
    }

    #[test]
    fn test_parse_tags_single_tag() {
        let response = "SCORE: 5\nCONTEXT: Chunk about databases.\nTAGS: database";
        let (_score, _context, _metadata, tags) = parse_context_response(response);
        assert_eq!(tags, vec!["database"]);
    }

    #[test]
    fn test_parse_tags_empty() {
        let response = "SCORE: 6\nCONTEXT: Some content.\nTAGS:";
        let (_score, _context, _metadata, tags) = parse_context_response(response);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_no_tags() {
        let response = "SCORE: 6\nCONTEXT: Some content.";
        let (_score, _context, _metadata, tags) = parse_context_response(response);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_tags_with_metadata() {
        let response = "SCORE: 9\nCONTEXT: A chunk about web development.\nTAGS: web, frontend, javascript\nMETADATA: {\"difficulty\": \"intermediate\"}";
        let (score, _context, metadata, tags) = parse_context_response(response);
        assert_eq!(score, Some(9.0));
        assert_eq!(tags, vec!["web", "frontend", "javascript"]);
        assert!(metadata.is_some());
    }

    #[test]
    fn test_parse_full_response_with_tags() {
        let response = "SCORE: 8\nCONTEXT: Describes Svelte 5 runes.\nTAGS: svelte, frontend, runes\nMETADATA: {\"topic\": \"svelte\", \"version\": 5}";
        let (score, context, metadata, tags) = parse_context_response(response);
        assert_eq!(score, Some(8.0));
        assert_eq!(context, Some("Describes Svelte 5 runes.".to_string()));
        assert_eq!(tags, vec!["svelte", "frontend", "runes"]);
        let meta = metadata.unwrap();
        assert_eq!(meta["topic"], "svelte");
        assert_eq!(meta["version"], 5);
    }

    #[test]
    fn test_llm_provider_new_reads_env() {
        // LlmProvider::new falls back to LLM_API_KEY env var
        // We just test construction, not actual API calls
        let provider = LlmProvider::new(
            "ollama".to_string(),
            "gemma4:31b".to_string(),
            Some("test-api-key".to_string()),
            Some("http://localhost:11434".to_string()),
        );
        // Provider should be created successfully (fields are private, just verify no panic)
        let _ = &provider;
    }

    #[test]
    fn test_llm_provider_default_url() {
        // When no base_url is provided, it defaults to localhost:11434
        let provider = LlmProvider::new(
            "ollama".to_string(),
            "test-model".to_string(),
            None,
            None,
        );
        let _ = &provider;
    }
}

/// Parse the LLM response for SCORE, CONTEXT, TAGS and METADATA lines.
/// Returns (relevance_score, context, metadata, tags). If parsing fails, uses the whole
/// response as context and returns score = None (backward compat).
fn parse_context_response(response: &str) -> (Option<f32>, Option<String>, Option<serde_json::Value>, Vec<String>) {
    let mut score: Option<f32> = None;
    let mut context_lines: Vec<&str> = Vec::new();
    let mut extracted_metadata: Option<serde_json::Value> = None;
    let mut tags: Vec<String> = Vec::new();
    let mut found_score = false;
    let mut found_context = false;

    for line in response.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with("SCORE:") && !found_score {
            let score_str = trimmed_line["SCORE:".len()..].trim();
            if let Ok(s) = score_str.parse::<f32>() {
                score = Some(s);
                found_score = true;
                continue;
            }
        }
        if trimmed_line.starts_with("TAGS:") {
            let tags_str = trimmed_line["TAGS:".len()..].trim();
            tags = tags_str
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            continue;
        }
        if trimmed_line.starts_with("METADATA:") {
            let json_str = trimmed_line["METADATA:".len()..].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                extracted_metadata = Some(val);
            }
            continue;
        }
        if trimmed_line.starts_with("CONTEXT:") && !found_context {
            let ctx = trimmed_line["CONTEXT:".len()..].trim();
            if !ctx.is_empty() {
                context_lines.push(ctx);
            }
            found_context = true;
            continue;
        }
        // After CONTEXT: line, collect remaining lines as part of context
        if found_context {
            if !trimmed_line.is_empty() {
                context_lines.push(trimmed_line);
            }
        }
    }

    if found_score || found_context {
        let context = if context_lines.is_empty() {
            None
        } else {
            Some(context_lines.join(" "))
        };
        (score, context, extracted_metadata, tags)
    } else {
        // Parsing failed — backward compat: use whole response as context
        if response.is_empty() {
            (None, None, extracted_metadata, tags)
        } else {
            (None, Some(response.to_string()), extracted_metadata, tags)
        }
    }
}
