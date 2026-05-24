use anyhow::Result;
use rag_engine::api::{
    db_pool,
    hybrid_search,
    simple,
    source_rag::{self, ChunkData, ChunkSearchResult},
};
use rag_engine::api::source_rag::DEFAULT_COLLECTION_ID;
use crate::chunker;
use crate::config::RerankerConfig;
use crate::embedding::EmbeddingProvider;
use crate::extractor;
use crate::llm::{ContextResult, LlmProvider};
use crate::reranker::{Reranker, RerankCandidate, RerankerType};

/// Stored DB path so list_sources/stats can query across all collections.
static DB_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static RERANKER: std::sync::OnceLock<Reranker> = std::sync::OnceLock::new();

/// Initialize rag_engine: logger + DB pool + schema + reranker
pub fn init(data_dir: &std::path::Path, config: &crate::config::Config) -> Result<()> {
    simple::init_core();
    let db_path = data_dir.join("rag.sqlite3");
    std::fs::create_dir_all(data_dir)?;
    let db_path_str = db_path.to_string_lossy().to_string();
    db_pool::init_db_pool(db_path_str.clone(), 4)?;
    source_rag::init_source_db()?;

    // Migration: add section_path column to chunks (backward-compatible)
    let conn = rusqlite::Connection::open(&db_path_str)?;
    let has_section_path: bool = conn.prepare("SELECT section_path FROM chunks LIMIT 1").is_ok();
    if !has_section_path {
        tracing::info!("Migrating: adding section_path column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN section_path TEXT DEFAULT NULL")?;
    }

    // Migration: add page column to chunks (backward-compatible)
    let has_page: bool = conn.prepare("SELECT page FROM chunks LIMIT 1").is_ok();
    if !has_page {
        tracing::info!("Migrating: adding page column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN page INTEGER DEFAULT NULL")?;
    }
    drop(conn);

    let _ = DB_PATH.set(db_path_str);
    tracing::info!("rag_engine DB initialized at {}", db_path.display());

    // Initialize reranker from config
    let reranker = build_reranker(&config.reranker, &config.llm);
    let _ = RERANKER.set(reranker);
    Ok(())
}

/// Build a Reranker from config, falling back to LLM config for missing fields.
fn build_reranker(cfg: &RerankerConfig, llm: &crate::config::LlmConfig) -> Reranker {
    match cfg.reranker_type.as_str() {
        "llm" => {
            let model = cfg.model.clone().unwrap_or_else(|| llm.model.clone());
            let api_key = cfg.api_key.clone().or_else(|| llm.api_key.clone());
            let base_url = cfg.base_url.clone().unwrap_or_else(|| {
                llm.base_url.clone().unwrap_or_else(|| "https://openrouter.ai/api/v1".into())
            });
            let provider = llm.provider.clone();
            tracing::info!("Reranker: LLM ({}, {})", provider, model);
            Reranker::new(RerankerType::Llm { provider, model, api_key, base_url })
        }
        "cohere" => {
            let api_key = cfg.api_key.clone()
                .expect("Cohere reranker requires reranker.api_key");
            tracing::info!("Reranker: Cohere");
            Reranker::new(RerankerType::Cohere { api_key })
        }
        _ => {
            tracing::info!("Reranker: disabled");
            Reranker::disabled()
        }
    }
}

/// Ingest a text document into the RAG
pub async fn ingest_text(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    content: &str,
    source_name: &str,
    metadata: Option<&str>,
    collection: Option<&str>,
    max_concurrent: usize,
    relevance_scoring: bool,
    min_relevance_score: f32,
) -> Result<i64> {
    let collection_id = collection.unwrap_or(DEFAULT_COLLECTION_ID).to_string();
    let meta = metadata.map(|m| m.to_string()).unwrap_or_default();
    let source = source_rag::add_source_in_collection(
        collection_id.clone(),
        content.to_string(),
        if meta.is_empty() { None } else { Some(meta) },
        Some(source_name.to_string()),
    )?;

    // Custom recursive character chunker (faster, no freeze on large docs)
    let chunk_size = 800;

    let single_section = if content.len() < chunk_size {
        // Even for single-chunk docs, extract section path from the beginning
        let sections = chunker::extract_sections(content);
        chunker::find_section_for_position(&sections, 0)
    } else {
        None
    };

    // Skip chunking for small sources — single chunk is better than 2 tiny ones
    let chunks = if content.len() < chunk_size {
        tracing::info!("Source below chunk size ({} chars), ingesting as single chunk", content.len());
        vec![chunker::Chunk {
            content: content.trim().to_string(),
            index: 0,
            start_pos: 0,
            end_pos: content.len() as i32,
chunk_type: chunker::detect_chunk_type(content.trim(), true),
            section_path: single_section,
            page: None,
        }]
    } else {
        chunker::chunk_text(content, chunk_size)
    };
    tracing::info!("Chunked into {} chunks (size={})", chunks.len(), chunk_size);

    // Contextual retrieval: generate context prefixes + relevance scores via LLM
    let context_results: Vec<ContextResult> = if let Some(llm_provider) = llm {
        tracing::info!("Generating context prefixes for {} chunks via LLM...", chunks.len());
        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        // Process in batches of 20 for rate limiting
        let mut all_results: Vec<ContextResult> = Vec::with_capacity(chunks.len());
        for batch in chunk_texts.chunks(20) {
            let results = llm_provider.generate_context_batch(content, batch, max_concurrent).await;
            for result in results {
                match result {
                    Ok(ctx_result) => all_results.push(ctx_result),
                    Err(e) => {
                        tracing::warn!("Context generation failed for chunk: {}, using raw content", e);
                        all_results.push(ContextResult { context: None, relevance_score: None });
                    }
                }
            }
        }
        let with_ctx = all_results.iter().filter(|c| c.context.is_some()).count();
        tracing::info!("Generated {}/{} context prefixes", with_ctx, chunks.len());
        all_results
    } else {
        vec![ContextResult { context: None, relevance_score: None }; chunks.len()]
    };

    // Filter chunks by relevance score if enabled
    let mut filtered_count = 0usize;
    let kept: Vec<(usize, &chunker::Chunk, &ContextResult)> = chunks
        .iter()
        .enumerate()
        .zip(context_results.iter())
        .filter(|((_, _), ctx_result)| {
            if relevance_scoring {
                if let Some(score) = ctx_result.relevance_score {
                    if score < min_relevance_score {
                        filtered_count += 1;
                        tracing::info!("Filtered chunk (score={:.1} < threshold={:.1})", score, min_relevance_score);
                        return false;
                    }
                }
            }
            true
        })
        .map(|((idx, chunk), ctx_result)| (idx, chunk, ctx_result))
        .collect();

    if filtered_count > 0 {
        tracing::info!("Relevance scoring: filtered {}/{} chunks (threshold={:.1})", filtered_count, chunks.len(), min_relevance_score);
    }

    if kept.is_empty() {
        tracing::warn!("All chunks filtered by relevance scoring, ingesting anyway");
    }

    // Build final texts for embedding: context prefix + chunk content
    let final_texts: Vec<String> = kept
        .iter()
        .map(|(_, chunk, ctx_result)| {
            match &ctx_result.context {
                Some(context) => format!("{}\n\n{}", context, chunk.content),
                None => chunk.content.clone(),
            }
        })
        .collect();

    // Batch embed all chunks (with context prefixes)
    let embeddings = embedder.embed_batch(&final_texts).await?;

    // Store original chunk content (not the prefixed version)
    // Collect section_paths and pages for post-insert UPDATE (rag_engine ChunkData doesn't have these)
    let section_paths: Vec<Option<String>> = kept
        .iter()
        .map(|(_, chunk, _)| chunk.section_path.clone())
        .collect();
    let pages: Vec<Option<u32>> = kept
        .iter()
        .map(|(_, chunk, _)| chunk.page)
        .collect();

    let chunk_data: Vec<ChunkData> = kept
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|((_, chunk, _), emb)| ChunkData {
            content: chunk.content.clone(),
            chunk_index: chunk.index,
            start_pos: chunk.start_pos,
            end_pos: chunk.end_pos,
            chunk_type: chunk.chunk_type.as_str().to_string(),
            embedding: emb,
        })
        .collect();

    let count = source_rag::add_chunks(source.source_id, chunk_data)?;
    tracing::info!("Ingested {} chunks for source {} ({})", count, source.source_id, source_name);

    // Store section_path for each chunk (separate UPDATE since rag_engine doesn't know about it)
    update_chunk_section_paths(source.source_id, &section_paths)?;
    update_chunk_pages(source.source_id, &pages)?;

    // Mark source as completed
    if let Err(e) = source_rag::update_source_status(source.source_id, "completed".to_string()) {
        tracing::warn!("Failed to update source status: {}", e);
    }

    // Rebuild indexes for the target collection
    if let Err(e) = source_rag::rebuild_chunk_hnsw_index_for_collection(collection_id.clone()) {
        tracing::warn!("Failed to rebuild HNSW index for {}: {}", collection_id, e);
    }
    if let Err(e) = source_rag::rebuild_chunk_bm25_index_for_collection(collection_id.clone()) {
        tracing::warn!("Failed to rebuild BM25 index for {}: {}", collection_id, e);
    }

    // Persist HNSW index to disk for fast startup
    let index_path = format!("/home/loops/services/rag-ferrite/data/hnsw_{}.index", collection_id);
    if let Err(e) = source_rag::save_collection_hnsw_index(collection_id.clone(), index_path) {
        tracing::warn!("Failed to save HNSW index: {}", e);
    }

    Ok(source.source_id)
}

/// Ingest a file (PDF, TXT, MD)
/// Uses our custom extractor for reliable text extraction
pub async fn ingest_file(
    embedder: &EmbeddingProvider,
    llm: Option<&LlmProvider>,
    file_path: &str,
    collection: Option<&str>,
    max_concurrent: usize,
    relevance_scoring: bool,
    min_relevance_score: f32,
) -> Result<i64> {
    // Use our custom extractor instead of rag_engine's document_parser
    let text = extractor::extract_text(file_path)?;

    let name = std::path::Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.to_string());
    ingest_text(
        embedder,
        llm,
        &text,
        &name,
        Some(&format!("{{\"path\":\"{}\"}}", file_path)),
        collection,
        max_concurrent,
        relevance_scoring,
        min_relevance_score,
    )
    .await
}

/// Update section_path for all chunks of a source, matched by chunk_index.
fn update_chunk_section_paths(source_id: i64, section_paths: &[Option<String>]) -> Result<()> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    for (idx, path) in section_paths.iter().enumerate() {
        if let Some(sp) = path {
            conn.execute(
                "UPDATE chunks SET section_path = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![sp, source_id, idx as i32],
            )?;
        }
    }

    Ok(())
}

/// Update page for all chunks of a source, matched by chunk_index.
fn update_chunk_pages(source_id: i64, pages: &[Option<u32>]) -> Result<()> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;
    for (idx, page) in pages.iter().enumerate() {
        if let Some(p) = page {
            conn.execute(
                "UPDATE chunks SET page = ?1 WHERE source_id = ?2 AND chunk_index = ?3",
                rusqlite::params![*p as i64, source_id, idx as i32],
            )?;
        }
    }
    Ok(())
}

/// Fetch section_path for a batch of chunk IDs.
pub fn get_section_paths_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<String>>> {
    if chunk_ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    let mut map = std::collections::HashMap::new();
    for &id in chunk_ids {
        let sp: Option<String> = conn
            .query_row(
                "SELECT section_path FROM chunks WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        map.insert(id, sp);
    }
    Ok(map)
}

/// Delete a source by ID
pub fn delete_source(source_id: i64) -> Result<()> {
    source_rag::delete_source(source_id)?;
    // Rebuild all indexes (don't know which collection the source was in)
    let _ = source_rag::rebuild_chunk_hnsw_index();
    let _ = source_rag::rebuild_chunk_bm25_index();
    Ok(())
}

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
    // Activate the correct collection's indexes before searching
    if let Some(ref f) = filter {
        if let Some(ref coll) = f.collection_id {
            let index_path = format!("/home/loops/services/rag-ferrite/data/hnsw_{}.index", coll);
            if let Err(e) = source_rag::activate_collection_for_hybrid_search(coll.clone(), index_path) {
                tracing::warn!("Failed to activate collection '{}': {}", coll, e);
            }
        }
    }

    // Expand short queries (< 5 words) if LLM is available
    let queries = if let Some(llm_provider) = llm {
        let word_count = query.split_whitespace().count();
        if word_count <= 5 {
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

    // Rerank if enabled
    if let Some(reranker) = RERANKER.get() {
        if reranker.is_enabled() && !all_results.is_empty() {
            let candidates: Vec<RerankCandidate> = all_results.iter().map(|r| RerankCandidate {
                doc_id: r.doc_id,
                content: r.content.clone(),
                initial_score: r.score,
                source_id: r.source_id,
                chunk_index: r.chunk_index,
                metadata: r.metadata.clone(),
                vector_rank: r.vector_rank,
                bm25_rank: r.bm25_rank,
            }).collect();
            match reranker.rerank(query, candidates).await {
                Ok(reranked) => {
                    tracing::info!("Reranked {} results", reranked.len());
                    all_results = reranked.into_iter().take(limit).map(|r| hybrid_search::HybridSearchResult {
                        doc_id: r.doc_id,
                        content: r.content,
                        score: r.score,
                        source_id: r.source_id,
                        chunk_index: r.chunk_index,
                        metadata: r.metadata,
                        vector_rank: r.vector_rank,
                        bm25_rank: r.bm25_rank,
                    }).collect();
                }
                Err(e) => {
                    tracing::warn!("Reranking failed, using hybrid scores: {}", e);
                }
            }
        }
    }

    Ok(all_results)
}

/// Fetch page for a batch of chunk IDs.
pub fn get_pages_for_chunk_ids(chunk_ids: &[i64]) -> Result<std::collections::HashMap<i64, Option<u32>>> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;
    let mut map = std::collections::HashMap::new();
    for &id in chunk_ids {
        let page: Option<u32> = conn.query_row(
            "SELECT page FROM chunks WHERE id = ?1",
            rusqlite::params![id],
            |row| row.get(0),
        ).unwrap_or(None);
        map.insert(id, page);
    }
    Ok(map)
}

/// Get chunks adjacent to a given chunk, enriched with section_path and page.
pub fn get_neighbors(source_id: i64, chunk_index: i64, before: i64, after: i64) -> Result<Vec<(ChunkSearchResult, Option<String>, Option<u32>)>> {
    let min_index = (chunk_index - before).max(0);
    let max_index = chunk_index + after;
    let chunks = source_rag::get_adjacent_chunks(source_id, min_index as i32, max_index as i32)?;

    // Fetch section_paths and pages for all chunks
    let chunk_ids: Vec<i64> = chunks.iter().map(|c| c.chunk_id).collect();
    let section_map = get_section_paths_for_chunk_ids(&chunk_ids)?;
    let page_map = get_pages_for_chunk_ids(&chunk_ids)?;

    let enriched = chunks
        .into_iter()
        .map(|c| {
            let sp = section_map.get(&c.chunk_id).cloned().flatten();
            let pg = page_map.get(&c.chunk_id).cloned().flatten();
            (c, sp, pg)
        })
        .collect();

    Ok(enriched)
}

/// List all sources across all collections.
///
/// Queries the `sources` table directly instead of using
/// `source_rag::list_sources()` which hardcodes the `__default__` collection.
pub fn list_sources() -> Result<Vec<source_rag::SourceEntry>> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT id, name, created_at, metadata, status, collection_id
         FROM sources
         ORDER BY id DESC",
    )?;

    let entries: Vec<source_rag::SourceEntry> = stmt.query_map([], |row| {
        Ok(source_rag::SourceEntry {
            id: row.get(0)?,
            name: row.get(1)?,
            created_at: row.get(2)?,
            metadata: row.get(3)?,
            status: row.get(4)?,
            collection_id: row.get(5)?,
        })
    })?.filter_map(|e| e.ok()).collect();

    Ok(entries)
}

/// Get stats across all collections.
pub fn stats() -> Result<Stats> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    let count: usize = conn.query_row(
        "SELECT COUNT(*) FROM sources",
        [],
        |row| row.get::<_, i64>(0),
    )? as usize;

    Ok(Stats {
        document_count: count,
    })
}

pub struct Stats {
    pub document_count: usize,
}

/// Graph data for document similarity visualization.
pub fn get_graph_data(
    collection: Option<&str>,
    threshold: f32,
    max_edges: usize,
) -> Result<crate::types::GraphData> {
    let db_path = DB_PATH.get().ok_or_else(|| anyhow::anyhow!("DB not initialized"))?;
    let conn = rusqlite::Connection::open(db_path)?;

    // 1. Get sources, optionally filtered by collection
    let sources: Vec<(i64, Option<String>, String, i32)> = if let Some(coll) = collection {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s WHERE s.collection_id = ?1 ORDER BY s.id",
        )?;
        stmt.query_map(rusqlite::params![coll], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s ORDER BY s.id",
        )?;
        stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    };

    if sources.is_empty() {
        return Ok(crate::types::GraphData {
            nodes: vec![],
            edges: vec![],
        });
    }

    // Build nodes
    let nodes: Vec<crate::types::GraphNode> = sources
        .iter()
        .map(|(id, name, collection_id, chunk_count)| crate::types::GraphNode {
            id: *id,
            name: name.clone().unwrap_or_else(|| format!("doc_{}", id)),
            collection: collection_id.clone(),
            chunk_count: *chunk_count,
        })
        .collect();

    let source_ids: Vec<i64> = sources.iter().map(|(id, _, _, _)| *id).collect();

    // 2. Load chunk embeddings per source and compute centroids
    let mut centroids: std::collections::HashMap<i64, Vec<f32>> = std::collections::HashMap::new();

    for source_id in &source_ids {
        let mut stmt = conn.prepare(
            "SELECT embedding FROM chunks WHERE source_id = ?1",
        )?;
        let embeddings: Vec<Vec<f32>> = stmt
            .query_map(rusqlite::params![source_id], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(decode_f32_embedding(&blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|v| if v.is_empty() { None } else { Some(v) })
            .collect();

        if embeddings.is_empty() {
            continue;
        }

        // Compute centroid (average of all chunk embeddings)
        let dims = embeddings[0].len();
        let mut centroid = vec![0.0f32; dims];
        let count = embeddings.len() as f32;
        for emb in &embeddings {
            for (i, val) in emb.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in centroid.iter_mut() {
            *val /= count;
        }
        centroids.insert(*source_id, centroid);
    }

    // 3. Compute pairwise cosine similarity
    let mut edges: Vec<crate::types::GraphEdge> = Vec::new();
    let ids_with_centroids: Vec<i64> = source_ids
        .iter()
        .filter(|id| centroids.contains_key(id))
        .copied()
        .collect();

    for i in 0..ids_with_centroids.len() {
        for j in (i + 1)..ids_with_centroids.len() {
            let id_a = ids_with_centroids[i];
            let id_b = ids_with_centroids[j];
            let a = &centroids[&id_a];
            let b = &centroids[&id_b];
            let sim = cosine_similarity(a, b);
            if sim >= threshold {
                edges.push(crate::types::GraphEdge {
                    source: id_a,
                    target: id_b,
                    similarity: (sim * 10000.0).round() / 10000.0, // 4 decimal places
                });
            }
        }
    }

    // 4. Sort by similarity desc, keep max_edges
    edges.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges.truncate(max_edges);

    // Only keep nodes that have edges or have centroids (still visible as isolated nodes)
    let _active_ids: std::collections::HashSet<i64> = ids_with_centroids.into_iter().collect();

    Ok(crate::types::GraphData { nodes, edges })
}

/// Decode a BLOB of native-endian f32 bytes into a Vec<f32>.
fn decode_f32_embedding(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 {
        return Vec::new();
    }
    blob.chunks(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Compute cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
