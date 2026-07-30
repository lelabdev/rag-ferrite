use crate::types::{SearchFilter, SearchResult};
use anyhow::Result;

use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;

/// Search with hybrid fusion (BM25 + vector + RRF)
pub async fn search_hybrid(
    embedder: &EmbeddingProvider,
    query: &str,
    limit: usize,
    filter: Option<SearchFilter>,
) -> Result<Vec<SearchResult>> {
    search_hybrid_with_expansion(embedder, None, query, limit, filter).await
}

/// Search with optional query expansion for short/ambiguous queries.
///
/// Collection selection: tag routing picks the best collection based on
/// query keywords. Falls back to the first available collection.
pub async fn search_hybrid_with_expansion(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    query: &str,
    limit: usize,
    filter: Option<SearchFilter>,
) -> Result<Vec<SearchResult>> {
    // ── 1. Tag routing — pick which collection to search ──
    let routed_collection = if filter
        .as_ref()
        .and_then(|f| f.collection_id.as_ref())
        .is_none()
    {
        match super::tag_routing::route_query(query) {
            Ok(route) => {
                if let Some(ref coll) = route.collection {
                    tracing::info!(
                        "Tag routing: query '{}' → collection '{}' (keywords: {:?})",
                        query,
                        coll,
                        route.keywords
                    );
                    Some(coll.clone())
                } else {
                    None
                }
            }
            Err(e) => {
                tracing::warn!("Tag routing failed: {}, using default collection", e);
                None
            }
        }
    } else {
        filter.as_ref().and_then(|f| f.collection_id.clone())
    };

    // ── 2. Build filter ──
    let filter = if let Some(coll) = routed_collection {
        let mut f = filter.unwrap_or(SearchFilter::default());
        f.collection_id = Some(coll);
        Some(f)
    } else {
        filter
    };

    // ── 3. Expand short queries ──
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

    // ── 4. Search ──
    let mut all_results: Vec<SearchResult> = Vec::new();
    let mut seen_doc_ids = std::collections::HashSet::new();

    for q in &queries {
        let query_embedding = embedder.embed(q).await?;
        let filter_clone = filter.clone();
        let query_text = q.clone();

        let results = tokio::task::spawn_blocking(move || {
            crate::storage::search_hybrid(query_text, query_embedding, limit, filter_clone)
        })
        .await??;
        for result in results {
            if seen_doc_ids.insert(result.doc_id) {
                all_results.push(result);
            }
        }
    }

    // Sort by score descending
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(limit);

    Ok(all_results)
}
