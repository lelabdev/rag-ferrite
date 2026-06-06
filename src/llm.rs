use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use crate::tag_rules;

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
    /// Default temperature for scoring/tagging
    pub temperature: f64,
    /// Default max tokens for scoring/tagging
    pub max_tokens: usize,
    /// Temperature for expansion/reformulation
    pub expansion_temperature: f64,
    /// Max tokens for expansion/reformulation
    pub expansion_max_tokens: usize,
    /// Max expansion queries per original query
    pub max_expansion_queries: usize,
    /// Max document chars in prompt
    pub max_document_prompt_chars: usize,
    /// Max chunk chars in prompt
    pub max_chunk_prompt_chars: usize,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: usize,
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
    /// Create an LlmProvider from a named profile (reads API key from the profile's env var).
    pub fn from_profile(profile: &crate::config::LlmProfile) -> Self {
        let api_key = std::env::var(&profile.api_key_env).ok();
        Self::build(
            profile.provider.clone(),
            profile.model.clone(),
            api_key,
            Some(profile.base_url.clone()),
        )
    }

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

    pub fn new_query_fallback(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let api_key = api_key
            .or_else(|| std::env::var("QUERY_FALLBACK_API_KEY").ok())
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

        tracing::info!(
            "LLM provider: {} / {}, base_url: {}, api_key: {}",
            provider,
            model,
            base_url,
            if api_key.is_some() { format!("{}***", &api_key.as_ref().unwrap()[..8.min(api_key.as_ref().unwrap().len())]) } else { "NONE".into() }
        );

        Self {
            provider,
            model,
            api_key,
            base_url,
            client: reqwest::Client::builder().timeout(Duration::from_secs(120)).build().unwrap(),
            fallback: None,
            temperature: 0.3,
            max_tokens: 150,
            expansion_temperature: 0.7,
            expansion_max_tokens: 200,
            max_expansion_queries: 4,
            max_document_prompt_chars: 8000,
            max_chunk_prompt_chars: 2000,
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
    ) -> Result<ContextResult> {
        let prompt = format!(
            "<document>\n{}\n</document>\n\n\
             Here is the chunk we want to situate within the whole document:\n\
             <chunk>\n{}\n</chunk>\n\n\
             Assess the relevance of this chunk for informative retrieval on a scale of 1 to 10, \
             where 1 is noise (TOC, index, legal mentions, boilerplate) and 10 is highly informative content.\
             Also give a short succinct context to situate this chunk within the overall document \
             for the purposes of improving search retrieval of the chunk.\n\n\
             Answer ONLY in this exact format:\n\
             SCORE: <number 1-10>\n\
             CONTEXT: <short succinct context, same language as document>\n\
             TAGS: <1-3 tags, noun phrases only, no adjectives alone, lowercase, hyphenated multi-word>",
            truncate_for_prompt(whole_document, self.max_document_prompt_chars),
            truncate_for_prompt(chunk_content, self.max_chunk_prompt_chars),
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

    /// Generate context prefixes for multiple chunks of the SAME parent in a single LLM call.
    /// Much faster than one call per child — the document is sent once, context for all chunks is returned.
    pub async fn generate_context_for_parent(
        &self,
        whole_document: &str,
        child_chunks: &[String],
    ) -> Vec<Result<ContextResult>> {
        if child_chunks.is_empty() {
            return vec![];
        }
        if child_chunks.len() == 1 {
            // Single child — use the existing single-call method
            return vec![self.generate_context(whole_document, &child_chunks[0]).await];
        }

        // Build numbered chunks for the prompt
        let numbered_chunks: String = child_chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("CHUNK {}:\n{}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "<document>\n{}\n</document>\n\n\
             Here are {} chunks from the same section. For EACH chunk, assess its relevance \
             for informative retrieval on a scale of 1 to 10 (1=noise, 10=highly informative) \
             and give a short context to situate it within the document.\n\n\
             Chunks:\n{}\n\n\
             Answer ONLY in this exact format, one block per chunk:\n\
             CHUNK 1:\n\
             SCORE: <number 1-10>\n\
             CONTEXT: <short succinct context, same language as document>\n\
             TAGS: <1-3 tags, noun phrases only, no adjectives alone, lowercase, hyphenated multi-word>\n\
             CHUNK 2:\n\
             SCORE: <number 1-10>\n\
             CONTEXT: <short succinct context>\n\
             TAGS: <1-3 tags, noun phrases only, no adjectives alone, lowercase, hyphenated multi-word>\n\
             (etc.)",
            truncate_for_prompt(whole_document, self.max_document_prompt_chars),
            child_chunks.len(),
            numbered_chunks,
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
                    match fb.chat(messages).await {
                        Ok(text) => text,
                        Err(e2) => {
                            let err = e2.to_string();
                            return child_chunks.iter().map(|_| Err(anyhow!("{}", err))).collect();
                        }
                    }
                } else {
                    let err = e.to_string();
                    return child_chunks.iter().map(|_| Err(anyhow!("{}", err))).collect();
                }
            }
        };

        // Parse multi-chunk response
        parse_multi_chunk_response(&response_text, child_chunks.len())
    }

    /// Generate context prefixes for multiple chunks in a single batch.
    /// Processes chunks concurrently (up to 10 at a time).
    pub async fn generate_context_batch(
        &self,
        whole_document: &str,
        chunks: &[String],
        max_concurrent: usize,
    ) -> Vec<Result<ContextResult>> {
        let sem = Arc::new(tokio::sync::Semaphore::new(max_concurrent.max(1)));
        let mut handles = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let sem = sem.clone();
            let provider = self.clone();
            let doc = whole_document.to_string();
            let chunk_content = chunk.clone();

            handles.push(tokio::spawn(async move {
                let _permit = match sem.acquire().await {
                    Ok(p) => p,
                    Err(_) => return Err(anyhow::anyhow!("Semaphore closed")),
                };
                provider.generate_context(&doc, &chunk_content).await
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
        self.chat_with_options(messages, self.temperature, self.max_tokens).await
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

        let response = match self.chat_with_options(messages, self.expansion_temperature, self.expansion_max_tokens).await {
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
        expansions.truncate(self.max_expansion_queries);
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

        let response = self.chat_with_options(messages, self.expansion_temperature, self.expansion_max_tokens).await?;
        Ok(response.trim().to_string())
    }

    /// Chat with custom temperature and max_tokens.
    async fn chat_with_options(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f64,
        max_tokens: usize,
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
                    temperature: f64,
                    num_predict: usize,
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
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        // Try to cut at a sentence boundary
        let truncated: String = text.chars().take(max_chars).collect();
        if let Some(pos) = truncated.rfind('.') {
            format!("{}...", &truncated[..=pos])
        } else {
            format!("{}...", truncated)
        }
    }
}

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
        assert_eq!(tags, vec!["rust", "programming", "system"]);
    }

    #[test]
    fn test_parse_tags_with_extra_whitespace() {
        let response = "SCORE: 7\nCONTEXT: Some context\nTAGS:  machine-learning ,  neural-networks , ai ";
        let (_score, _context, _metadata, tags) = parse_context_response(response);
        assert_eq!(tags, vec!["machine-learning", "neural-networks"]);
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
        assert_eq!(tags, vec!["svelte", "frontend", "rune"]);
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
/// Parse a multi-chunk LLM response into individual ContextResults.
/// Expected format: CHUNK N: / SCORE: / CONTEXT: / TAGS: blocks separated by blank lines.
fn parse_multi_chunk_response(response: &str, expected_count: usize) -> Vec<Result<ContextResult>> {
    let mut results: Vec<Result<ContextResult>> = Vec::with_capacity(expected_count);
    
    // Split response by "CHUNK N:" markers
    let chunk_pattern = regex::Regex::new(r"(?m)^CHUNK\s+\d+\s*:").unwrap_or_else(|_| {
        tracing::warn!("Failed to compile chunk pattern regex");
        regex::Regex::new(r"nevermatch").unwrap()
    });
    
    let mut chunks: Vec<&str> = Vec::new();
    let mut last_end = 0;
    
    for mat in chunk_pattern.find_iter(response) {
        if (last_end > 0 || mat.start() > 0)
            && last_end > 0 {
                chunks.push(&response[last_end..mat.start()]);
            }
        last_end = mat.start();
    }
    if last_end < response.len() {
        chunks.push(&response[last_end..]);
    }
    
    if chunks.is_empty() {
        // Fallback: couldn't parse chunks, try parsing as single response
        tracing::warn!("Could not parse multi-chunk response, falling back to single parse");
        let (score, context, metadata, tags) = parse_context_response(response);
        results.push(Ok(ContextResult {
            context, relevance_score: score, extracted_metadata: metadata, tags,
        }));
        // Fill rest with empty results
        while results.len() < expected_count {
            results.push(Ok(ContextResult {
                context: None, relevance_score: None, extracted_metadata: None, tags: Vec::new(),
            }));
        }
        return results;
    }
    
    for chunk_text in &chunks {
        let (score, context, metadata, tags) = parse_context_response(chunk_text);
        results.push(Ok(ContextResult {
            context, relevance_score: score, extracted_metadata: metadata, tags,
        }));
    }
    
    // Pad if we got fewer results than expected
    while results.len() < expected_count {
        results.push(Ok(ContextResult {
            context: None, relevance_score: None, extracted_metadata: None, tags: Vec::new(),
        }));
    }
    
    results.truncate(expected_count);
    results
}

/// Global tag rules — set once at startup, read by sanitize_tags()
static TAG_RULES: std::sync::OnceLock<tag_rules::TagRules> = std::sync::OnceLock::new();

/// Initialize tag rules. Call once at startup.
pub fn init_tag_rules(rules: tag_rules::TagRules) {
    let _ = TAG_RULES.set(rules);
}

/// Sanitize raw tags from LLM output using tag-rules.toml.
/// Multi-stage pipeline: strip → lowercase → synonyms → stop words → length → dedup.
fn sanitize_tags(raw_tags: Vec<String>) -> Vec<String> {
    let rules = TAG_RULES.get().cloned().unwrap_or_default();
    let all_stops = rules.stop_words.all();

    raw_tags.into_iter()
        // Stage 1: Strip special chars
        .map(|t| {
            let mut cleaned = t;
            for c in rules.rules.strip_chars.chars() {
                cleaned = cleaned.replace(c, "");
            }
            cleaned.replace('/', " ").trim().to_lowercase()
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
                score = Some(s.clamp(1.0, 10.0));
                found_score = true;
                continue;
            }
        }
        if trimmed_line.starts_with("TAGS:") {
            let tags_str = trimmed_line["TAGS:".len()..].trim();
            let raw_tags: Vec<String> = tags_str
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            tags = sanitize_tags(raw_tags);
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
        if found_context
            && !trimmed_line.is_empty() {
                context_lines.push(trimmed_line);
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
