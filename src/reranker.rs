use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::llm::LlmProvider;

/// Truncate a string to at most `max_chars` Unicode characters (byte-safe).
fn truncate_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Sort results by rerank_score (falling back to original score) and truncate to top_k.
fn sort_and_truncate(results: &mut Vec<RerankedResult>, top_k: usize) {
    results.sort_by(|a, b| {
        let sa = a.rerank_score.unwrap_or(a.score);
        let sb = b.rerank_score.unwrap_or(b.score);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
}

/// Reranker for post-retrieval quality improvement.
#[derive(Debug, Clone)]
pub struct Reranker {
    reranker_type: RerankerType,
    llm: Option<Arc<LlmProvider>>,
    client: reqwest::Client,
    top_k: usize,
    preview_chars: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RerankerType {
    /// LLM-based reranking via LlmProvider
    Llm,
    /// Cohere Rerank API
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
    /// Hybrid search score (original)
    pub score: f64,
    /// LLM relevance score (0.0-1.0). None = not reranked.
    pub rerank_score: Option<f64>,
    pub source_id: i64,
    pub chunk_index: u32,
    pub metadata: Option<String>,
    pub vector_rank: u32,
    pub bm25_rank: u32,
}

impl Reranker {
    pub fn new_llm(llm: Arc<LlmProvider>, top_k: usize, preview_chars: usize) -> Self {
        Self {
            reranker_type: RerankerType::Llm,
            llm: Some(llm),
            client: reqwest::Client::new(),
            top_k,
            preview_chars,
        }
    }

    pub fn new_cohere(api_key: String, top_k: usize, preview_chars: usize) -> Self {
        Self {
            reranker_type: RerankerType::Cohere { api_key },
            llm: None,
            client: reqwest::Client::new(),
            top_k,
            preview_chars,
        }
    }

    pub fn disabled() -> Self {
        Self {
            reranker_type: RerankerType::Disabled,
            llm: None,
            client: reqwest::Client::new(),
            top_k: 10,
            preview_chars: 300,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.reranker_type != RerankerType::Disabled
    }

    /// Convenience method: convert `HybridSearchResult`s to candidates and rerank.
    pub async fn rerank_hybrid(
        &self,
        query: &str,
        results: Vec<rag_engine::api::hybrid_search::HybridSearchResult>,
    ) -> Vec<RerankedResult> {
        if !self.is_enabled() || results.is_empty() {
            return results.into_iter().map(|r| r.into()).collect();
        }

        // Convert to RerankedResult first as fallback on error
        let fallback: Vec<RerankedResult> = results.iter().map(|r| r.clone().into()).collect();

        let candidates: Vec<RerankCandidate> = results
            .into_iter()
            .map(|r| RerankCandidate {
                doc_id: r.doc_id,
                content: r.content,
                initial_score: r.score,
                source_id: r.source_id,
                chunk_index: r.chunk_index,
                metadata: r.metadata,
                vector_rank: r.vector_rank,
                bm25_rank: r.bm25_rank,
            })
            .collect();

        match self.rerank(query, candidates).await {
            Ok(reranked) => reranked,
            Err(e) => {
                tracing::warn!("Reranking failed: {}, using initial scores", e);
                fallback
            }
        }
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
                    rerank_score: None,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata,
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                })
                .collect());
        }

        match &self.reranker_type {
            RerankerType::Llm => {
                self.rerank_llm(query, candidates).await
            }
            RerankerType::Cohere { api_key } => {
                self.rerank_cohere(query, candidates, api_key).await
            }
            RerankerType::Disabled => unreachable!(),
        }
    }

    /// LLM-based reranking via LlmProvider (handles ollama/openai/zai transparently).
    async fn rerank_llm(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> Result<Vec<RerankedResult>> {
        let provider = self.llm.as_ref()
            .ok_or_else(|| anyhow!("No LLM provider for reranker"))?;

        // Build a prompt that scores each candidate
        let candidates_text: Vec<String> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i, truncate_chars(&c.content, self.preview_chars)))
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

        let messages = vec![crate::llm::ChatMessage { role: "user".into(), content: prompt }];
        let content = provider.chat(messages).await?;

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
                    entries.into_iter().map(|e| (e.index, e.score.clamp(0.0, 1.0))).collect()
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
                let llm_score = scores.iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, s)| *s);
                RerankedResult {
                    doc_id: c.doc_id,
                    content: c.content,
                    score: c.initial_score,
                    rerank_score: llm_score,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata,
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                }
            })
            .collect();

        // Sort by rerank_score (fall back to original score)
        sort_and_truncate(&mut results, self.top_k);

        Ok(results)
    }

    /// Cohere Rerank API.
    async fn rerank_cohere(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
        api_key: &str,
    ) -> Result<Vec<RerankedResult>> {
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
                    score: c.initial_score,
                    rerank_score: Some(entry.relevance_score),
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    metadata: c.metadata.clone(),
                    vector_rank: c.vector_rank,
                    bm25_rank: c.bm25_rank,
                })
            })
            .collect();

        // Sort by rerank_score (fall back to original score)
        sort_and_truncate(&mut results, self.top_k);

        Ok(results)
    }
}
