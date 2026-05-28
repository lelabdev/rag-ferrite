//! Shared service layer — business logic called by both MCP tools (main.rs) and HTTP handlers (api.rs).

use crate::engine;
use crate::params::IngestConfig;
use crate::pipeline::QueryPipeline;
use crate::types::{ChunkResult, HybridResult, SourceInfo};
use serde_json::json;

// ── Query ──────────────────────────────────────────────────────────────

pub async fn query_service(
    pipeline: &QueryPipeline,
    query: &str,
    limit: usize,
    source_ids: Option<Vec<i64>>,
    metadata_like: Option<String>,
    collection: Option<String>,
) -> serde_json::Value {
    let filter = if source_ids.is_some() || collection.is_some() || metadata_like.is_some() {
        Some(rag_engine::api::hybrid_search::SearchFilter {
            source_ids,
            metadata_like,
            collection_id: collection,
        })
    } else {
        None
    };

    match pipeline.query(query, limit, filter).await {
        Ok(output) => {
            let doc_ids: Vec<i64> = output.results.iter().map(|r| r.doc_id).collect();
            let section_map = engine::get_section_paths_for_chunk_ids(&doc_ids).unwrap_or_default();
            let tags_map = engine::get_tags_for_chunk_ids(&doc_ids).unwrap_or_default();

            // Parent resolution: for child chunks, replace content with parent's
            let parent_map = engine::query::resolve_parents(&doc_ids).unwrap_or_default();

            let out: Vec<HybridResult> = output.results.into_iter().map(|r| {
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
    file_path: &str,
    collection: Option<&str>,
) -> serde_json::Value {
    match engine::ingest_file(
        &pipeline.embedder,
        pipeline.llm.as_ref(),
        file_path,
        collection,
        cfg.to_engine_options(),
    )
    .await
    {
        Ok((id, report)) => json!({
            "status": "ok",
            "source_id": id,
            "file_path": file_path,
            "collection": collection,
            "report": report
        }),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

// ── Ingest data ────────────────────────────────────────────────────────

pub async fn ingest_data_service(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    content: &str,
    source: &str,
    collection: Option<&str>,
) -> serde_json::Value {
    match engine::ingest_text(
        &pipeline.embedder,
        pipeline.llm.as_ref(),
        content,
        source,
        None,
        collection,
        cfg.to_engine_options(),
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

pub fn list_sources_service(collection: Option<&str>) -> serde_json::Value {
    match engine::list_sources() {
        Ok(sources) => {
            let mut out: Vec<SourceInfo> = sources.into_iter().map(SourceInfo::from).collect();
            if let Some(coll) = collection {
                out.retain(|s| &s.collection_id == coll);
            }
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
