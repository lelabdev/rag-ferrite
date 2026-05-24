use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Result of contextual retrieval: optional context prefix, optional relevance score (1-10),
/// and optional extracted metadata.
#[derive(Debug, Clone)]
pub struct ContextResult {
    pub context: Option<String>,
    pub relevance_score: Option<f32>,
    pub extracted_metadata: Option<serde_json::Value>,
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
struct ChatMessage {
    role: String,
    content: String,
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
        let (score, context, extracted_metadata) = parse_context_response(trimmed);
        Ok(ContextResult {
            context,
            relevance_score: score,
            extracted_metadata,
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

    /// Send a chat completion request.
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        match self.provider.as_str() {
            "ollama" => self.chat_ollama(messages).await,
            _ => self.chat_openai_compatible(messages).await,
        }
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

                let resp = self.client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await?;

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

    /// OpenAI-compatible chat endpoint (Z.ai, OpenAI, etc.)
    async fn chat_openai_compatible(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow!("API key required for {}. Set LLM_API_KEY or FALLBACK_API_KEY.", self.provider))?;

        let url = format!("{}/chat/completions", self.base_url);

        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.3,
            max_tokens: 150,
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

    /// Ollama chat endpoint
    async fn chat_ollama(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);

        #[derive(Debug, Serialize)]
        struct OllamaChatRequest {
            model: String,
            messages: Vec<ChatMessage>,
            stream: bool,
        }

        let body = OllamaChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
        };

        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?;

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

/// Parse the LLM response for SCORE and CONTEXT lines.
/// Returns (relevance_score, context). If parsing fails, uses the whole
/// response as context and returns score = None (backward compat).
fn parse_context_response(response: &str) -> (Option<f32>, Option<String>, Option<serde_json::Value>) {
    let mut score: Option<f32> = None;
    let mut context_lines: Vec<&str> = Vec::new();
    let mut extracted_metadata: Option<serde_json::Value> = None;
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
        (score, context, extracted_metadata)
    } else {
        // Parsing failed — backward compat: use whole response as context
        if response.is_empty() {
            (None, None, extracted_metadata)
        } else {
            (None, Some(response.to_string()), extracted_metadata)
        }
    }
}
