use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;
use crate::reranker::{Reranker, RerankedResult};

/// Default cache TTL in seconds.
pub const CACHE_TTL_SECS: u64 = 300;

/// Simple in-memory query result cache with TTL expiry.
#[derive(Debug)]
pub struct Cache {
    /// Map from cache key → (insertion time, cached output).
    entries: Mutex<HashMap<String, (Instant, QueryOutput)>>,
    /// Time-to-live for each cache entry.
    ttl: Duration,
}

impl Cache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// Build a cache key from query, limit, and optional collection.
    fn make_key(query: &str, limit: usize, collection: Option<&str>) -> String {
        match collection {
            Some(c) => format!("{}|{}|{}", query, limit, c),
            None => format!("{}|{}|", query, limit),
        }
    }

    /// Look up a cached result. Returns `None` on miss or if the entry has expired.
    fn get(&self, key: &str) -> Option<QueryOutput> {
        let map = self.entries.lock().unwrap();
        if let Some((inserted_at, output)) = map.get(key) {
            if inserted_at.elapsed() < self.ttl {
                return Some(output.clone());
            }
        }
        None
    }

    /// Store a result in the cache.
    fn put(&self, key: String, output: QueryOutput) {
        let mut map = self.entries.lock().unwrap();
        map.insert(key, (Instant::now(), output));
    }
}

/// Query complexity classification for adaptive pipeline routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryComplexity {
    Simple,
    Standard,
    Complex,
}

/// Classify a query into Simple / Standard / Complex.
///
/// Rules:
/// - **Simple**: 1–2 words, no question markers, no boolean operators.
/// - **Complex**: >8 words **OR** contains question markers **OR** contains
///   boolean operators (AND / OR / et / ou).
/// - **Standard**: everything else (3–8 words, no question/boolean signals).
pub fn classify_query(query: &str) -> QueryComplexity {
    let words: Vec<&str> = query.split_whitespace().collect();
    let word_count = words.len();

    // Question markers in English and French
    let question_markers = [
        "what", "how", "why", "when", "where", "which", "who", "whom",
        "whose", "whether",
        "comment", "pourquoi", "quand", "où", "quel", "quelle", "quels",
        "quelles", "qui", "comment",
    ];

    // Boolean operators
    let boolean_operators = ["AND", "OR", "et", "ou"];


    let has_question_marker = words.iter().any(|w| {
        let wl = w.to_lowercase()
            .trim_end_matches(|c: char| c == '?' || c == ',' || c == '.' || c == '!' || c == ';')
            .to_string();
        question_markers.iter().any(|m| wl == *m)
    });

    let has_boolean_op = words.iter().any(|w| {
        boolean_operators.iter().any(|op| *w == *op)
    });

    // Complex: >8 words OR question markers OR boolean operators
    if word_count > 8 || has_question_marker || has_boolean_op {
        return QueryComplexity::Complex;
    }

    // Simple: 1–2 words, no signals
    if word_count <= 2 {
        return QueryComplexity::Simple;
    }

    // Standard: 3–8 words, no signals
    QueryComplexity::Standard
}

/// Full query pipeline: router → expansion → search → rerank → corrective RAG.
#[derive(Debug)]
pub struct QueryPipeline {
    pub embedder: EmbeddingProvider,
    pub llm: Option<LlmProvider>,
    pub reranker: Reranker,
    pub quality_threshold: f64,
    pub max_retries: u32,
    cache: std::sync::Arc<Cache>,
}

impl QueryPipeline {
    pub fn new(
        embedder: EmbeddingProvider,
        llm: Option<LlmProvider>,
        reranker: Reranker,
        quality_threshold: f64,
        max_retries: u32,
    ) -> Self {
        Self {
            embedder,
            llm,
            reranker,
            quality_threshold,
            max_retries,
            cache: std::sync::Arc::new(Cache::new(
                std::time::Duration::from_secs(CACHE_TTL_SECS),
            )),
        }
    }
}

impl Clone for QueryPipeline {
    fn clone(&self) -> Self {
        Self {
            embedder: self.embedder.clone(),
            llm: self.llm.clone(),
            reranker: self.reranker.clone(),
            quality_threshold: self.quality_threshold,
            max_retries: self.max_retries,
            cache: self.cache.clone(),
        }
    }
}

/// Final output of the query pipeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryOutput {
    pub results: Vec<RerankedResult>,
    pub confidence: Confidence,
    pub retry_count: u32,
    pub query_complexity: QueryComplexity,
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl QueryPipeline {
    /// Run the adaptive query pipeline.
    pub async fn query(
        &self,
        query: &str,
        limit: usize,
        filter: Option<rag_engine::api::hybrid_search::SearchFilter>,
    ) -> Result<QueryOutput> {
        let collection = filter.as_ref().and_then(|f| f.collection_id.as_deref());
        let cache_key = Cache::make_key(query, limit, collection);

        // Check cache before executing the query
        if let Some(cached) = self.cache.get(&cache_key) {
            tracing::info!(query = query, "Cache hit for query");
            return Ok(cached);
        }

        let complexity = classify_query(query);
        tracing::info!(query = query, complexity = ?complexity, "Classified query");

        let result = match complexity {
            QueryComplexity::Simple => self.query_simple(query, limit, filter, complexity).await?,
            QueryComplexity::Standard => self.query_standard(query, limit, filter, complexity).await?,
            QueryComplexity::Complex => self.query_complex(query, limit, filter, complexity).await?,
        };

        // Store result in cache
        self.cache.put(cache_key, result.clone());
        tracing::debug!(query = query, "Cached query result");

        Ok(result)
    }

    /// Simple path: no expansion, no reranking, no corrective retry.
    /// Optimised for fast keyword lookups.
    async fn query_simple(
        &self,
        query: &str,
        limit: usize,
        filter: Option<rag_engine::api::hybrid_search::SearchFilter>,
        complexity: QueryComplexity,
    ) -> Result<QueryOutput> {
        let results = crate::engine::search_hybrid(
            &self.embedder,
            query,
            limit,
            filter,
        )
        .await?;

        let top_score = results.first().map(|r| r.score).unwrap_or(0.0);
        let confidence = if top_score >= 0.7 {
            Confidence::High
        } else if top_score >= self.quality_threshold {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        // Convert HybridSearchResult → RerankedResult (passthrough, no reranking)
        let reranked: Vec<RerankedResult> = results
            .into_iter()
            .map(|r| RerankedResult {
                doc_id: r.doc_id,
                content: r.content,
                score: r.score,
                rerank_score: None,
                source_id: r.source_id,
                chunk_index: r.chunk_index,
                metadata: r.metadata,
                vector_rank: r.vector_rank,
                bm25_rank: r.bm25_rank,
            })
            .collect();

        Ok(QueryOutput {
            results: reranked,
            confidence,
            retry_count: 0,
            query_complexity: complexity,
        })
    }

    /// Standard path: expansion if short (≤5 words), reranking if enabled.
    /// No corrective retry.
    async fn query_standard(
        &self,
        query: &str,
        limit: usize,
        filter: Option<rag_engine::api::hybrid_search::SearchFilter>,
        complexity: QueryComplexity,
    ) -> Result<QueryOutput> {
        let word_count = query.split_whitespace().count();

        // Use expansion for short queries (≤5 words)
        let results = if word_count <= 5 {
            crate::engine::search_hybrid_with_expansion(
                &self.embedder,
                self.llm.as_ref(),
                query,
                limit,
                filter,
            )
            .await?
        } else {
            crate::engine::search_hybrid(
                &self.embedder,
                query,
                limit,
                filter,
            )
            .await?
        };

        // Rerank if enabled, otherwise pass through results directly
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

            match self.reranker.rerank(query, candidates).await {
                Ok(reranked) => reranked,
                Err(e) => {
                    tracing::warn!("Reranking failed: {}, using initial scores", e);
                    Vec::new()
                }
            }
        } else {
            // No reranker — pass through results directly
            results.into_iter().map(|r| RerankedResult {
                doc_id: r.doc_id,
                content: r.content,
                score: r.score,
                rerank_score: None,
                source_id: r.source_id,
                chunk_index: r.chunk_index,
                metadata: r.metadata,
                vector_rank: r.vector_rank,
                bm25_rank: r.bm25_rank,
            }).collect()
        };

        let top_score = reranked.first().map(|r| r.score).unwrap_or(0.0);
        let confidence = if top_score >= 0.7 {
            Confidence::High
        } else if top_score >= self.quality_threshold {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        Ok(QueryOutput {
            results: reranked,
            confidence,
            retry_count: 0,
            query_complexity: complexity,
        })
    }

    /// Complex path: full pipeline with expansion + reranking + corrective retry.
    async fn query_complex(
        &self,
        query: &str,
        limit: usize,
        filter: Option<rag_engine::api::hybrid_search::SearchFilter>,
        complexity: QueryComplexity,
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
                // No reranker — pass through results directly
                results.into_iter().map(|r| RerankedResult {
                    doc_id: r.doc_id,
                    content: r.content,
                    score: r.score,
                    rerank_score: None,
                    source_id: r.source_id,
                    chunk_index: r.chunk_index,
                    metadata: r.metadata,
                    vector_rank: r.vector_rank,
                    bm25_rank: r.bm25_rank,
                }).collect()
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

            return Ok(QueryOutput {
                results: reranked,
                confidence,
                retry_count,
                query_complexity: complexity,
            });
        }
    }
}
