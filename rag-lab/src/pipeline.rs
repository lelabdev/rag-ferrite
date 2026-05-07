use anyhow::Result;
use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;
use crate::reranker::{Reranker, RerankedResult};

/// Full query pipeline: expansion → search → rerank → corrective RAG.
#[derive(Debug, Clone)]
pub struct QueryPipeline {
    pub embedder: EmbeddingProvider,
    pub llm: Option<LlmProvider>,
    pub reranker: Reranker,
    pub quality_threshold: f64,
    pub max_retries: u32,
}

/// Final output of the query pipeline.
#[derive(Debug, serde::Serialize)]
pub struct QueryOutput {
    pub results: Vec<RerankedResult>,
    pub confidence: Confidence,
    pub retry_count: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl QueryPipeline {
    /// Run the full query pipeline.
    pub async fn query(
        &self,
        query: &str,
        limit: usize,
        filter: Option<rag_engine::api::hybrid_search::SearchFilter>,
    ) -> Result<QueryOutput> {
        let mut current_query = query.to_string();
        let mut retry_count = 0;

        loop {
            // Step 1: Hybrid search with query expansion
            let results = crate::engine::search_hybrid_with_expansion(
                &self.embedder,
                self.llm.as_ref(),
                &current_query,
                limit,
                filter.clone(),
            )
            .await?;

            // Step 2: Rerank if enabled
            let reranked = if self.reranker.is_enabled() && !results.is_empty() {
                let candidates: Vec<crate::reranker::RerankCandidate> = results
                    .into_iter()
                    .map(|r| crate::reranker::RerankCandidate {
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

                match self.reranker.rerank(&current_query, candidates).await {
                    Ok(reranked) => reranked,
                    Err(e) => {
                        tracing::warn!("Reranking failed: {}, using initial scores", e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            // Step 3: Quality gate — check top score
            let top_score = reranked
                .first()
                .map(|r| r.score)
                .unwrap_or(0.0);

            let (confidence, should_retry) = if top_score >= 0.7 {
                (Confidence::High, false)
            } else if top_score >= self.quality_threshold {
                (Confidence::Medium, false)
            } else if retry_count < self.max_retries {
                (Confidence::Low, true)
            } else {
                (Confidence::Low, false)
            };

            // Step 4: Corrective RAG — reformulate and retry
            if should_retry {
                if let Some(llm) = &self.llm {
                    tracing::info!(
                        "Low confidence ({:.3}) for query '{}', retrying with reformulation ({}/{})",
                        top_score,
                        current_query,
                        retry_count + 1,
                        self.max_retries
                    );

                    match llm.reformulate_query(&current_query).await {
                        Ok(reformulated) => {
                            tracing::info!("Reformulated query: {}", reformulated);
                            current_query = reformulated;
                            retry_count += 1;
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!("Query reformulation failed: {}, returning low-confidence results", e);
                        }
                    }
                }
            }

            // Step 5: Return results
            // If reranked is empty (disabled or failed), return from raw results
            return Ok(QueryOutput {
                results: reranked,
                confidence,
                retry_count,
            });
        }
    }
}
