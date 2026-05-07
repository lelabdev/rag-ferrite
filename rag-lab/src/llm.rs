use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// LLM provider for contextual retrieval and other text generation tasks.
#[derive(Debug, Clone)]
pub struct LlmProvider {
    provider: String,
    model: String,
    api_key: Option<String>,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
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
            .or_else(|| std::env::var("ZAI_API_KEY").ok());

        let base_url = base_url.unwrap_or_else(|| match provider.as_str() {
            "zai" => "https://api.z.ai/api/paas/v4".into(),
            "openai" => "https://api.openai.com/v1".into(),
            "ollama" => "http://localhost:11434".into(),
            _ => "https://api.openai.com/v1".into(),
        });

        Self {
            provider,
            model,
            api_key,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    /// Generate a context prefix for a chunk using the LLM.
    /// Takes the whole document and a specific chunk, returns 1-2 sentences
    /// situating the chunk within the document.
    pub async fn generate_context(
        &self,
        whole_document: &str,
        chunk_content: &str,
    ) -> Result<String> {
        let prompt = format!(
            "<document>\n{}\n</document>\n\n\
             Here is the chunk we want to situate within the whole document:\n\
             <chunk>\n{}\n</chunk>\n\n\
             Please give a short succinct context to situate this chunk within the overall document \
             for the purposes of improving search retrieval of the chunk. \
             Answer only with the succinct context, nothing else. \
             Use the same language as the document.",
            truncate_for_prompt(whole_document, 8000),
            truncate_for_prompt(chunk_content, 2000),
        );

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt,
        }];

        let response_text = self.chat(messages).await?;
        Ok(response_text.trim().to_string())
    }

    /// Generate context prefixes for multiple chunks in a single batch.
    /// Processes chunks concurrently (up to 10 at a time).
    pub async fn generate_context_batch(
        &self,
        whole_document: &str,
        chunks: &[String],
    ) -> Vec<Result<String>> {
        let sem = Arc::new(tokio::sync::Semaphore::new(10));
        let mut handles = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let sem = sem.clone();
            let provider = self.clone();
            let doc = whole_document.to_string();
            let chunk_content = chunk.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
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

    /// Send a chat completion request.
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        match self.provider.as_str() {
            "zai" | "openai" => self.chat_openai_compatible(messages).await,
            "ollama" => self.chat_ollama(messages).await,
            _ => Err(anyhow!("Unknown LLM provider: {}", self.provider)),
        }
    }

    /// Expand a short or ambiguous query into 2-3 reformulations.
    /// Returns the original query + reformulations for broader retrieval.
    pub async fn expand_query(&self, query: &str) -> Result<Vec<String>> {
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

        let response = self.chat_with_options(messages, 0.7, 200).await?;
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
            "zai" | "openai" => {
                let api_key = self.api_key.as_ref()
                    .ok_or_else(|| anyhow!("API key required for {}. Set LLM_API_KEY or ZAI_API_KEY.", self.provider))?;

                let url = format!("{}/chat/completions", self.base_url);

                let body = ChatRequest {
                    model: self.model.clone(),
                    messages,
                    temperature,
                    max_tokens,
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
            _ => Err(anyhow!("Unknown LLM provider: {}", self.provider)),
        }
    }

    /// OpenAI-compatible chat endpoint (Z.ai, OpenAI, etc.)
    async fn chat_openai_compatible(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow!("API key required for {}. Set LLM_API_KEY or ZAI_API_KEY.", self.provider))?;

        let url = format!("{}/chat/completions", self.base_url);

        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.3,
            max_tokens: 150,
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
