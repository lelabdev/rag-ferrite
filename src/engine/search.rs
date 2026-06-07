use anyhow::Result;
use rag_engine::api::{hybrid_search, source_rag};

use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;

use super::data_dir;

/// Search with hybrid fusion (BM25 + vector + RRF)
/// Optionally expands short queries via LLM for better retrieval.
pub async fn search_hybrid(
    embedder: &EmbeddingProvider,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    search_hybrid_with_expansion(embedder, None, query, limit, filter).await
}

/// Search with optional query expansion for short/ambiguous queries.
pub async fn search_hybrid_with_expansion(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    // Soft tag routing: try to route to the best-matching collection
    let routed_collection = if filter.as_ref().and_then(|f| f.collection_id.as_ref()).is_none() {
        // Only route if no explicit collection filter was provided
        match super::tag_routing::route_query(query) {
            Ok(route) => {
                if let Some(ref coll) = route.collection {
                    tracing::info!(
                        "Tag routing: query '{}' → collection '{}' (keywords: {:?}, matches: {:?})",
                        query, coll, route.keywords, route.matches
                    );
                    Some(coll.clone())
                } else if !route.matches.is_empty() {
                    tracing::debug!(
                        "Tag routing: ambiguous for query '{}' (matches: {:?})",
                        query, route.matches
                    );
                    None
                } else {
                    None
                }
            }
            Err(e) => {
                tracing::warn!("Tag routing failed: {}, searching all collections", e);
                None
            }
        }
    } else {
        None
    };

    // Activate ALL collection indexes before searching (fixes #154)
    // Without this, HNSW was never activated unless a collection_id filter was passed
    {
        let conn = crate::engine::get_conn()?;
        let collections: Vec<String> = conn
            .prepare("SELECT DISTINCT collection_id FROM sources")?
            .query_map([], |row| row.get(0))?
            .filter_map(Result::ok)
            .collect();
        for coll_name in &collections {
            let coll = super::sanitize_collection(coll_name)?;
            let index_path = format!("{}/hnsw_{}.index", data_dir(), coll);
            if let Err(e) = source_rag::activate_collection_for_hybrid_search(coll.clone(), index_path) {
                tracing::warn!("Failed to activate collection '{}': {}", coll, e);
            }
        }
    }

    // If routing suggested a collection, inject it into the filter
    let filter = if let Some(coll) = routed_collection {
        let mut f = filter.unwrap_or(hybrid_search::SearchFilter {
            source_ids: None,
            metadata_like: None,
            collection_id: None,
        });
        f.collection_id = Some(coll);
        Some(f)
    } else {
        filter
    };

    // Expand short queries (< 5 words) if LLM is available
    let queries = if let Some(llm_provider) = llm {
        let word_count = query.split_whitespace().count();
        if word_count <= crate::pipeline::EXPANSION_WORD_THRESHOLD {
            match llm_provider.expand_query(query).await {
                Ok(expansions) => {
                    tracing::info!("Query expansion: {:?}", expansions);
                    expansions
                }
                Err(e) => {
                    tracing::warn!("Query expansion failed: {}, using original", e);
                    vec![query.to_string()]
                }
            }
        } else {
            vec![query.to_string()]
        }
    } else {
        vec![query.to_string()]
    };

    // Run hybrid search for each query variant
    let mut all_results: Vec<hybrid_search::HybridSearchResult> = Vec::new();
    let mut seen_doc_ids = std::collections::HashSet::new();

    for q in &queries {
        let query_embedding = embedder.embed(q).await?;
        let filter_clone = filter.clone();

        if let Ok(results) = hybrid_search::search_hybrid(
            q.to_string(),
            query_embedding,
            limit as u32,
            None,
            filter_clone,
        ) {
            for result in results {
                // Deduplicate by doc_id
                if seen_doc_ids.insert(result.doc_id) {
                    all_results.push(result);
                }
            }
        }
    }

    // Sort by score descending
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(limit);

    Ok(all_results)
}
