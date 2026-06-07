use anyhow::Result;
use rag_engine::api::{hybrid_search, source_rag};

use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;

use super::data_dir;

/// Search with hybrid fusion (BM25 + vector + RRF)
pub async fn search_hybrid(
    embedder: &EmbeddingProvider,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    search_hybrid_with_expansion(embedder, None, query, limit, filter).await
}

/// Search with optional query expansion for short/ambiguous queries.
///
/// Collection selection: tag routing picks the best collection based on
/// query keywords. Falls back to the first available collection.
/// Memory is managed by the OS via mmap — no explicit load/unload needed.
pub async fn search_hybrid_with_expansion(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    // ── 1. Tag routing — pick which collection to search ──
    let routed_collection = if filter.as_ref().and_then(|f| f.collection_id.as_ref()).is_none() {
        match super::tag_routing::route_query(query) {
            Ok(route) => {
                if let Some(ref coll) = route.collection {
                    tracing::info!(
                        "Tag routing: query '{}' → collection '{}' (keywords: {:?})",
                        query, coll, route.keywords
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

    // ── 2. Determine which collection to activate ──
    let collection = routed_collection.clone().unwrap_or_else(|| {
        // Fallback: first collection in DB
        if let Ok(conn) = crate::engine::get_conn() {
            if let Ok(mut stmt) = conn.prepare("SELECT DISTINCT collection_id FROM sources LIMIT 1") {
                if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                    for row in rows.flatten() {
                        return row;
                    }
                }
            }
        }
        "general".to_string()
    });

    let coll_sanitized = super::sanitize_collection(&collection)?;
    let index_path = format!("{}/hnsw_{}.index", data_dir(), coll_sanitized);
    if let Err(e) = source_rag::activate_collection_for_hybrid_search(coll_sanitized.clone(), index_path) {
        tracing::warn!("Failed to activate collection '{}': {}", coll_sanitized, e);
    }

    // ── 3. Build filter ──
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

    // ── 4. Expand short queries ──
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

    // ── 5. Search ──
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
