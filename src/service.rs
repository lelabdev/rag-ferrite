//! Shared service layer — business logic called by both MCP tools (main.rs) and HTTP handlers (api.rs).

use crate::engine;
use crate::llm;
use crate::params::IngestConfig;
use crate::pipeline::QueryPipeline;
use crate::types::{ChunkResult, HybridResult, SourceInfo};
use crate::engine::{HeatTracker, ChunkHeatTracker};
use serde_json::json;

// ── Query ──────────────────────────────────────────────────────────────

pub async fn query_service(
    pipeline: &QueryPipeline,
    query: &str,
    limit: usize,
    source_ids: Option<Vec<i64>>,
    metadata_like: Option<String>,
    tags: Option<Vec<String>>,
    heat_tracker: Option<&HeatTracker>,
    chunk_heat_tracker: Option<&ChunkHeatTracker>,
) -> serde_json::Value {
    let filter = if source_ids.is_some() || metadata_like.is_some() {
        Some(rag_engine::api::hybrid_search::SearchFilter {
            source_ids,
            metadata_like,
            collection_id: None,
        })
    } else {
        None
    };

    match pipeline.query(query, limit, filter).await {
        Ok(output) => {
            let mut doc_ids: Vec<i64> = output.results.iter().map(|r| r.doc_id).collect();
            let section_map = engine::get_section_paths_for_chunk_ids(&doc_ids).unwrap_or_default();
            let tags_map = engine::get_tags_for_chunk_ids(&doc_ids).unwrap_or_default();

            // ── Tag filtering (#149): keep only chunks that have at least one requested tag ──
            let filtered_results = if let Some(ref filter_tags) = tags {
                if filter_tags.is_empty() {
                    output.results
                } else {
                    output.results.into_iter().filter(|r| {
                        let chunk_tags = tags_map.get(&r.doc_id);
                        match chunk_tags {
                            Some(ct) => ct.iter().any(|t| filter_tags.iter().any(|ft| ft.eq_ignore_ascii_case(t))),
                            None => false,
                        }
                    }).collect()
                }
            } else {
                output.results
            };

            // Update doc_ids after tag filtering
            doc_ids = filtered_results.iter().map(|r| r.doc_id).collect();

            // ── Chunk heat tracking (#177): async batched, non-blocking ──
            if !doc_ids.is_empty() {
                if let Some(tracker) = chunk_heat_tracker {
                    tracker.record_chunks(&doc_ids);
                }
            }

            // ── Collection heat tracking (#159 Phase 1): async, non-blocking ──
            if let Some(tracker) = heat_tracker {
                let result_source_ids: Vec<i64> = filtered_results.iter().map(|r| r.source_id).collect();
                match engine::collections_for_sources(&result_source_ids) {
                    Ok(collections) => {
                        tracker.record_collections(&collections);
                    }
                    Err(e) => tracing::debug!("Failed to map sources to collections: {}", e),
                }
            }

            // Parent resolution: for child chunks, replace content with parent's
            let parent_map = engine::query::resolve_parents(&doc_ids).unwrap_or_default();

            let out: Vec<HybridResult> = filtered_results.into_iter().map(|r| {
                let sp = section_map.get(&r.doc_id).cloned().flatten();
                let tags = tags_map.get(&r.doc_id).cloned().unwrap_or_default();
                let parent_info = parent_map.get(&r.doc_id);

                HybridResult {
                    doc_id: r.doc_id,
                    content: parent_info.map(|p| p.content.clone()).unwrap_or(r.content),
                    score: r.score,
                    source_id: r.source_id,
                    chunk_index: r.chunk_index,
                    metadata: r.metadata,
                    vector_rank: r.vector_rank,
                    bm25_rank: r.bm25_rank,
                    section_path: parent_info.and_then(|p| p.section_path.clone()).or(sp),
                    page: parent_info.and_then(|p| p.page),
                    rerank_score: r.rerank_score,
                    tags,
                }
            }).collect();

            json!({
                "results": out,
                "confidence": output.confidence,
                "retries": output.retry_count
            })
        }
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Ingest file ────────────────────────────────────────────────────────

pub async fn ingest_file_service(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    ingestion_llm: Option<&llm::LlmProvider>,
    file_path: &str,
) -> serde_json::Value {
    // Use ingestion_llm if available, otherwise fall back to pipeline.llm
    let llm = ingestion_llm.or(pipeline.llm.as_ref());
    match engine::ingest_file(
        &pipeline.embedder,
        llm,
        file_path,
        Some("general"),
        cfg.clone(),
    )
    .await
    {
        Ok((id, report)) => json!({
            "status": "ok",
            "source_id": id,
            "file_path": file_path,
            "report": report
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Ingest data ────────────────────────────────────────────────────────

pub async fn ingest_data_service(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    ingestion_llm: Option<&llm::LlmProvider>,
    content: &str,
    source: &str,
) -> serde_json::Value {
    // Use ingestion_llm if available, otherwise fall back to pipeline.llm
    let llm = ingestion_llm.or(pipeline.llm.as_ref());
    match engine::ingest_text(
        &pipeline.embedder,
        llm,
        content,
        source,
        None,
        Some("general"),
        cfg.clone(),
    )
    .await
    {
        Ok((id, report)) => json!({
            "status": "ok",
            "source_id": id,
            "source": source,
            "content_length": content.len(),
            "report": report
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Delete ─────────────────────────────────────────────────────────────

pub fn delete_service(source: &str) -> serde_json::Value {
    match source.parse::<i64>() {
        Ok(id) => match engine::delete_source(id) {
            Ok(()) => json!({ "status": "ok", "source_id": id }),
            Err(e) => json!({ "error": e.to_string() }),
        },
        Err(_) => json!({ "error": "source must be a numeric source_id" }),
    }
}

// ── List sources ───────────────────────────────────────────────────────

pub fn list_sources_service() -> serde_json::Value {
    match engine::list_sources() {
        Ok(sources) => {
            // Get chunk counts per source
            let chunk_counts = engine::query::count_chunks_per_source().unwrap_or_default();
            let out: Vec<SourceInfo> = sources
                .into_iter()
                .map(|s| {
                    let mut info = SourceInfo::from(s);
                    info.chunk_count = chunk_counts.get(&info.id).copied().unwrap_or(0);
                    info
                })
                .collect();
            json!({ "files": out })
        }
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Status ─────────────────────────────────────────────────────────────

pub fn status_service() -> serde_json::Value {
    match engine::stats() {
        Ok(s) => json!({
            "document_count": s.document_count,
            "version": env!("CARGO_PKG_VERSION")
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Chunk neighbors ────────────────────────────────────────────────────

pub fn neighbors_service(
    source_id: i64,
    chunk_index: i64,
    before: i64,
    after: i64,
) -> serde_json::Value {
    match engine::get_neighbors(source_id, chunk_index, before, after) {
        Ok(chunks) => {
            let out: Vec<ChunkResult> = chunks
                .into_iter()
                .map(|(chunk, section_path, page)| ChunkResult {
                    chunk_id: chunk.chunk_id,
                    source_id: chunk.source_id,
                    chunk_index: chunk.chunk_index,
                    content: chunk.content,
                    score: chunk.similarity,
                    metadata: chunk.metadata,
                    chunk_type: chunk.chunk_type,
                    section_path,
                    page,
                })
                .collect();
            json!({
                "source_id": source_id,
                "chunk_index": chunk_index,
                "chunks": out
            })
        }
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Collection heat (#159 Phase 1) ─────────────────────────────────────

pub fn collection_heat_service() -> serde_json::Value {
    match engine::get_all_heat() {
        Ok(heat) => json!({
            "collections": heat,
            "total": heat.len()
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Chunk QA (#159 Phase 5) ────────────────────────────────────────────

pub fn chunk_qa_service() -> serde_json::Value {
    match engine::get_chunk_qa_report() {
        Ok(report) => json!({
            "sources": report,
            "total_sources": report.len()
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Tag routing (#163) ────────────────────────────────────────────────

pub fn suggest_collection_service(query: &str) -> serde_json::Value {
    match engine::tag_routing::route_query(query) {
        Ok(route) => json!({
            "query": query,
            "keywords": route.keywords,
            "suggested_collection": route.collection,
            "all_matches": route.matches.iter().map(|(c, s)| {
                json!({"collection": c, "score": s})
            }).collect::<Vec<_>>(),
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

pub fn tag_collection_map_service() -> serde_json::Value {
    match engine::tag_routing::get_tag_collection_map() {
        Ok(entries) => json!({
            "entries": entries.iter().map(|(tag, coll, count)| {
                json!({"tag": tag, "collection": coll, "chunk_count": count})
            }).collect::<Vec<_>>(),
            "total": entries.len()
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Multi-collection (#164) ────────────────────────────────────────────

pub fn reassign_collection_service(source_id: i64, new_collection: &str) -> serde_json::Value {
    match engine::reassign_source_collection(source_id, new_collection) {
        Ok(msg) => json!({ "success": true, "message": msg }),
        Err(e) => json!({ "success": false, "error": e.to_string() }),
    }
}

// ── Reload config (#191) ──────────────────────────────────────────────

pub fn reload_config_service() -> serde_json::Value {
    match crate::config::Config::load() {
        Ok(new_config) => {
            let mut reloaded = Vec::new();
            let mut requires_restart = Vec::new();

            // Report what CAN be hot-reloaded
            reloaded.push(format!("log_filter: {}", new_config.advanced.log_filter));
            reloaded.push(format!("min_relevance_score: {}", new_config.llm.min_relevance_score));
            reloaded.push(format!("reranker_type: {}", new_config.reranker.reranker_type));
            reloaded.push(format!("rerank_top_k: {}", new_config.reranker.top_k));
            reloaded.push(format!("rerank_preview_chars: {}", new_config.reranker.preview_chars));

            // Report what CANNOT be hot-reloaded
            requires_restart.push("http_port".to_string());
            requires_restart.push("data_dir".to_string());
            requires_restart.push("embedding.provider".to_string());
            requires_restart.push("embedding.model".to_string());
            requires_restart.push("embedding.base_url".to_string());
            requires_restart.push("http_bind_address".to_string());

            tracing::info!("Config reloaded from disk (hot-reload requested via API)");

            json!({
                "status": "ok",
                "message": "Config re-read from disk. Some settings take effect on next batch/query.",
                "reloaded": reloaded,
                "requires_restart": requires_restart,
            })
        }
        Err(e) => json!({
            "error": format!("Failed to reload config: {}", e)
        }),
    }
}
