use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;
use crate::reranker::{Reranker, RerankedResult};

/// Simple in-memory query result cache with TTL expiry.
#[derive(Debug)]
pub struct Cache {
    /// Map from cache key → (insertion time, cached output).
    entries: Mutex<HashMap<String, (Instant, QueryOutput)>>,
    /// Time-to-live for each cache entry.
    ttl: Duration,
    /// Max entries before eviction.
    max_entries: usize,
}

impl Cache {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            max_entries,
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
        let map = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((inserted_at, output)) = map.get(key) {
            if inserted_at.elapsed() < self.ttl {
                return Some(output.clone());
            }
        }
        None
    }

    /// Store a result in the cache. Evicts expired entries when the cache exceeds 1000 items.
    fn put(&self, key: String, output: QueryOutput) {
        let mut map = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(key, (Instant::now(), output));
        if map.len() > self.max_entries {
            let ttl = self.ttl;
            map.retain(|_, (inserted_at, _)| inserted_at.elapsed() < ttl);
        }
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
        // Strip trailing punctuation without allocating
        let trimmed = w.trim_end_matches(|c: char| c == '?' || c == ',' || c == '.' || c == '!' || c == ';');
        question_markers.iter().any(|m| trimmed.eq_ignore_ascii_case(m))
    });

    let has_boolean_op = words.iter().any(|w| {
        boolean_operators.iter().any(|op| w.eq(op))
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

/// Maximum word count for a query to trigger expansion.
pub const EXPANSION_WORD_THRESHOLD: usize = 5;

/// Full query pipeline: router → expansion → search → rerank → corrective RAG.
#[derive(Debug)]
pub struct QueryPipeline {
    pub embedder: EmbeddingProvider,
    pub llm: Option<LlmProvider>,
    pub reranker: Reranker,
    pub quality_threshold: f64,
    pub max_retries: u32,
    pub high_confidence_threshold: f64,
    cache: std::sync::Arc<Cache>,
}

impl QueryPipeline {
    pub fn new(
        embedder: EmbeddingProvider,
        llm: Option<LlmProvider>,
        reranker: Reranker,
        quality_threshold: f64,
        max_retries: u32,
        cache_ttl_secs: u64,
        cache_max_entries: usize,
        high_confidence_threshold: f64,
    ) -> Self {
        Self {
            embedder,
            llm,
            reranker,
            quality_threshold,
            max_retries,
            high_confidence_threshold,
            cache: std::sync::Arc::new(Cache::new(
                std::time::Duration::from_secs(cache_ttl_secs),
                cache_max_entries,
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
            high_confidence_threshold: self.high_confidence_threshold,
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

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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
        let confidence = classify_confidence(top_score, self.quality_threshold, self.high_confidence_threshold);

        // Convert HybridSearchResult → RerankedResult (passthrough, no reranking)
        let reranked: Vec<RerankedResult> = results
            .into_iter().map(|r| r.into()).collect();

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

        // Use expansion for short queries
        let results = if word_count <= EXPANSION_WORD_THRESHOLD {
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
        let reranked = self.reranker.rerank_hybrid(query, results).await;

        let top_score = reranked.first().map(|r| r.score).unwrap_or(0.0);
        let confidence = classify_confidence(top_score, self.quality_threshold, self.high_confidence_threshold);

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
            let reranked = self.reranker.rerank_hybrid(&current_query, results).await;

            // Step 3: Quality gate — check top score
            let top_score = reranked
                .first()
                .map(|r| r.score)
                .unwrap_or(0.0);

            let confidence = classify_confidence(top_score, self.quality_threshold, self.high_confidence_threshold);
            let should_retry = confidence == Confidence::Low && retry_count < self.max_retries;

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

/// Classify confidence based on top score vs quality threshold.
fn classify_confidence(top_score: f64, threshold: f64, high_confidence: f64) -> Confidence {
    if top_score >= high_confidence {
        Confidence::High
    } else if top_score >= threshold {
        Confidence::Medium
    } else {
        Confidence::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_queries() {
        assert_eq!(classify_query("rust"), QueryComplexity::Simple);
        assert_eq!(classify_query("hello world"), QueryComplexity::Simple);
        assert_eq!(classify_query("API"), QueryComplexity::Simple);
    }

    #[test]
    fn test_standard_queries() {
        assert_eq!(classify_query("search my documents for rust"), QueryComplexity::Standard);
        assert_eq!(classify_query("find relevant chunks about machine learning"), QueryComplexity::Standard);
        assert_eq!(classify_query("three word query"), QueryComplexity::Standard);
    }

    #[test]
    fn test_complex_by_word_count() {
        // > 8 words
        assert_eq!(classify_query("this is a very long query with many words in it"), QueryComplexity::Complex);
        assert_eq!(classify_query("one two three four five six seven eight nine"), QueryComplexity::Complex);
    }

    #[test]
    fn test_complex_by_question_markers() {
        // English question markers
        assert_eq!(classify_query("what is rust"), QueryComplexity::Complex);
        assert_eq!(classify_query("how does this work"), QueryComplexity::Complex);
        assert_eq!(classify_query("why is this happening"), QueryComplexity::Complex);
        assert_eq!(classify_query("where are my documents"), QueryComplexity::Complex);

        // French question markers
        assert_eq!(classify_query("comment faire"), QueryComplexity::Complex);
        assert_eq!(classify_query("pourquoi ça marche"), QueryComplexity::Complex);
        assert_eq!(classify_query("quand partir"), QueryComplexity::Complex);
        assert_eq!(classify_query("où suis-je"), QueryComplexity::Complex);
        assert_eq!(classify_query("quel est le problème"), QueryComplexity::Complex);
        assert_eq!(classify_query("quelle est la réponse"), QueryComplexity::Complex);
        assert_eq!(classify_query("qui est là"), QueryComplexity::Complex);
    }

    #[test]
    fn test_complex_by_boolean_operators() {
        assert_eq!(classify_query("rust AND python"), QueryComplexity::Complex);
        assert_eq!(classify_query("cat OR dog"), QueryComplexity::Complex);
        assert_eq!(classify_query("chat et chien"), QueryComplexity::Complex);
        assert_eq!(classify_query("chat ou chien"), QueryComplexity::Complex);
    }

    #[test]
    fn test_question_mark_punctuation() {
        // Question mark is stripped from markers
        assert_eq!(classify_query("what?"), QueryComplexity::Complex);
        assert_eq!(classify_query("how does this work?"), QueryComplexity::Complex);
        assert_eq!(classify_query("comment?"), QueryComplexity::Complex);
    }

    #[test]
    fn test_edge_cases() {
        // Empty string → 0 words → Simple (≤2)
        assert_eq!(classify_query(""), QueryComplexity::Simple);
        // Single char
        assert_eq!(classify_query("x"), QueryComplexity::Simple);
        // Exactly 2 words → Simple
        assert_eq!(classify_query("two words"), QueryComplexity::Simple);
        // Exactly 3 words → Standard (no markers)
        assert_eq!(classify_query("three word test"), QueryComplexity::Standard);
        // Exactly 8 words → Standard (no markers)
        assert_eq!(classify_query("one two three four five six seven eight"), QueryComplexity::Standard);
        // 9 words → Complex
        assert_eq!(classify_query("one two three four five six seven eight nine"), QueryComplexity::Complex);
    }

    #[test]
    fn test_classify_confidence_high() {
        assert_eq!(classify_confidence(0.8, 0.3, 0.7), Confidence::High);
        assert_eq!(classify_confidence(0.7, 0.3, 0.7), Confidence::High); // exactly at threshold
    }

    #[test]
    fn test_classify_confidence_medium() {
        assert_eq!(classify_confidence(0.5, 0.3, 0.7), Confidence::Medium);
        assert_eq!(classify_confidence(0.3, 0.3, 0.7), Confidence::Medium); // exactly at threshold
    }

    #[test]
    fn test_classify_confidence_low() {
        assert_eq!(classify_confidence(0.1, 0.3, 0.7), Confidence::Low);
        assert_eq!(classify_confidence(0.0, 0.3, 0.7), Confidence::Low);
    }

    #[test]
    fn test_classify_confidence_custom_thresholds() {
        assert_eq!(classify_confidence(0.9, 0.5, 0.8), Confidence::High);
        assert_eq!(classify_confidence(0.6, 0.5, 0.8), Confidence::Medium);
        assert_eq!(classify_confidence(0.2, 0.5, 0.8), Confidence::Low);
    }
}


