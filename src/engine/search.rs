use anyhow::Result;
use rag_engine::api::{hybrid_search, source_rag};

use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;

use super::collection_registry;
use super::data_dir;

/// Default heat threshold for lazy loading: collections below this are skipped.
const DEFAULT_HEAT_THRESHOLD: f64 = 5.0;

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
/// Lazy loading: only activates hot collections (heat_score >= threshold).
/// Cold collections are skipped unless explicitly routed to.
pub async fn search_hybrid_with_expansion(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    query: &str,
    limit: usize,
    filter: Option<hybrid_search::SearchFilter>,
) -> Result<Vec<hybrid_search::HybridSearchResult>> {
    // ── 1. Tag routing ──
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
                tracing::warn!("Tag routing failed: {}, searching all hot collections", e);
                None
            }
        }
    } else {
        filter.as_ref().and_then(|f| f.collection_id.clone())
    };

    // ── 2. Determine which collections to search ──
    let all_collections: Vec<String> = {
        let conn = crate::engine::get_conn()?;
        conn.prepare("SELECT DISTINCT collection_id FROM sources")?
            .query_map([], |row| row.get(0))?
            .filter_map(Result::ok)
            .collect()
    };

    let collections_to_search = decide_collections(
        &all_collections,
        routed_collection.as_deref(),
        DEFAULT_HEAT_THRESHOLD,
    );

    tracing::info!(
        "Lazy loading: searching {} / {} collections: {:?}",
        collections_to_search.len(),
        all_collections.len(),
        collections_to_search
    );

    // ── 3. Activate collections (only hot ones) ──
    for coll_name in &collections_to_search {
        let coll = super::sanitize_collection(coll_name)?;
        let index_path = format!("{}/hnsw_{}.index", data_dir(), coll);
        if let Err(e) = source_rag::activate_collection_for_hybrid_search(coll.clone(), index_path) {
            tracing::warn!("Failed to activate collection '{}': {}", coll, e);
        }
        collection_registry::mark_loaded(&coll);
    }

    // ── 4. Build filter ──
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

    // ── 5. Expand short queries ──
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

    // ── 6. Search (uses last-activated collection's index) ──
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

/// Decide which collections to search based on heat score and routing.
fn decide_collections(
    all: &[String],
    routed: Option<&str>,
    heat_threshold: f64,
) -> Vec<String> {
    let statuses = match collection_registry::get_all_statuses(heat_threshold) {
        Ok(s) => s,
        Err(_) => return all.to_vec(),
    };

    let mut to_search: Vec<String> = Vec::new();

    if let Some(routed_coll) = routed {
        if all.iter().any(|c| c == routed_coll) {
            to_search.push(routed_coll.to_string());
        }
    }

    for status in &statuses {
        if status.is_hot && !to_search.contains(&status.collection) {
            to_search.push(status.collection.clone());
        }
    }

    if to_search.is_empty() && !statuses.is_empty() {
        let hottest = statuses
            .iter()
            .max_by(|a, b| {
                a.heat_score
                    .partial_cmp(&b.heat_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        to_search.push(hottest.collection.clone());
    }

    if to_search.is_empty() {
        return all.to_vec();
    }

    to_search
}
