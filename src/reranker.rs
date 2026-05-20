use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Reranker for post-retrieval quality improvement.
#[derive(Debug, Clone)]
pub struct Reranker {
    reranker_type: RerankerType,
    client: reqwest::Client,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RerankerType {
    /// LLM-based reranking using scoring prompt
    #[allow(dead_code)]
    Llm { provider: String, model: String, api_key: Option<String>, base_url: String },
    /// Cohere Rerank API
    #[allow(dead_code)]
    Cohere { api_key: String },
    /// No reranking
    Disabled,
}

#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub doc_id: i64,
    pub content: String,
    pub initial_score: f64,
    pub source_id: i64,
    pub chunk_index: u32,
    pub metadata: Option<String>,
    pub vector_rank: u32,
    pub bm25_rank: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankedResult {
    pub doc_id: i64,
    pub content: String,
    pub score: f64,
    pub source_id: i64,
    pub chunk_index: u32,
    pub metadata: Option<String>,
    pub vector_rank: u32,
    pub bm25_rank: u32,
}

impl Reranker {
    pub fn new(reranker_type: RerankerType) -> Self {
        Self {
            reranker_type,
            client: reqwest::Client::new(),
        }
    }

    pub fn disabled() -> Self {
        Self::new(RerankerType::Disabled)
    }

    pub fn is_enabled(&self) -> bool {
        self.reranker_type != RerankerType::Disabled
    }

    /// Rerank candidates against a query. Returns sorted results.
    pub async fn rerank(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> Result<Vec<RerankedResult>> {
        if candidates.is_empty() || !self.is_enabled() {
            return Ok(candidates
                .into_iter()
                .map(|c| RerankedResult {
                    doc_id: c.doc_id,
                    content: c.content,
                    score: c.initial_score,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata,
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                })
                .collect());
        }

        match &self.reranker_type {
            RerankerType::Llm { provider, model, api_key, base_url } => {
                self.rerank_llm(query, candidates, provider, model, api_key, base_url).await
            }
            RerankerType::Cohere { api_key } => {
                self.rerank_cohere(query, candidates, api_key).await
            }
            RerankerType::Disabled => unreachable!(),
        }
    }

    /// LLM-based reranking: score each candidate's relevance to the query.
    async fn rerank_llm(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
        _provider: &str,
        model: &str,
        api_key: &Option<String>,
        base_url: &str,
    ) -> Result<Vec<RerankedResult>> {
        let key = api_key.as_ref()
            .ok_or_else(|| anyhow!("API key required for LLM reranker"))?;

        // Build a prompt that scores each candidate
        let candidates_text: Vec<String> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i, &c.content[..c.content.len().min(300)]))
            .collect();

        let prompt = format!(
            "Score each passage's relevance to the query on a scale of 0.0 to 1.0.\n\
             Return ONLY a JSON array of objects with \"index\" and \"score\" fields.\n\
             No explanation, no other text.\n\n\
             Query: {}\n\n\
             Passages:\n{}",
            query,
            candidates_text.join("\n")
        );

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

        let body = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage { role: "user".into(), content: prompt }],
            temperature: 0.1,
            max_tokens: 1000,
        };

        let url = format!("{}/chat/completions", base_url);
        let resp = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await?;
            return Err(anyhow!("LLM rerank API error {}: {}", status, text));
        }

        #[derive(Debug, Deserialize)]
        struct ChatResponse {
            choices: Vec<ChatChoice>,
        }
        #[derive(Debug, Deserialize)]
        struct ChatChoice {
            message: ChatMessage,
        }

        let data: ChatResponse = resp.json().await?;
        let content = data.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("No LLM response for reranking"))?;

        // Parse scores from LLM response
        let scores: Vec<(usize, f64)> = if let Some(json_start) = content.find('[') {
            let json_str = &content[json_start..];
            if let Some(end) = json_str.rfind(']') {
                let json_str = &json_str[..=end];
                #[derive(Debug, Deserialize)]
                struct ScoreEntry {
                    index: usize,
                    score: f64,
                }
                if let Ok(entries) = serde_json::from_str::<Vec<ScoreEntry>>(json_str) {
                    entries.into_iter().map(|e| (e.index, e.score)).collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Merge scores with candidates
        let mut results: Vec<RerankedResult> = candidates
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let score = scores.iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, s)| *s)
                    .unwrap_or(c.initial_score);
                RerankedResult {
                    doc_id: c.doc_id,
                    content: c.content,
                    score,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata,
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                }
            })
            .collect();

        // Sort by reranked score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Cohere Rerank API.
    async fn rerank_cohere(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
        api_key: &str,
    ) -> Result<Vec<RerankedResult>> {
        // Extract text for Cohere, keep candidates for later mapping
        let _n = candidates.len();
        let doc_texts: Vec<String> = candidates.iter()
            .map(|c| c.content.chars().take(500).collect())
            .collect();

        #[derive(Debug, Serialize)]
        struct RerankRequest {
            model: String,
            query: String,
            documents: Vec<String>,
        }

        let body = RerankRequest {
            model: "rerank-v3.5".into(),
            query: query.to_string(),
            documents: doc_texts,
        };

        let resp = self.client
            .post("https://api.cohere.ai/v2/rerank")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await?;
            return Err(anyhow!("Cohere rerank error {}: {}", status, text));
        }

        #[derive(Debug, Deserialize)]
        struct RerankResponse {
            results: Vec<RerankEntry>,
        }
        #[derive(Debug, Deserialize)]
        struct RerankEntry {
            index: usize,
            relevance_score: f64,
        }

        let data: RerankResponse = resp.json().await?;

        // Now consume candidates into a map by original index
        let cand_map: std::collections::HashMap<usize, RerankCandidate> = candidates
            .into_iter()
            .enumerate()
            .collect();

        let mut results: Vec<RerankedResult> = data.results
            .into_iter()
            .filter_map(|entry| {
                cand_map.get(&entry.index).map(|c| RerankedResult {
                    doc_id: c.doc_id,
                    content: c.content.clone(),
                    score: entry.relevance_score,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata.clone(),
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}
