use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::engine;
use crate::RagFerriteServer;

// --- Request types ---

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub source_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub metadata_like: Option<String>,
    #[serde(default)]
    pub collection: Option<String>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
pub struct IngestFileRequest {
    pub file_path: String,
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IngestDataRequest {
    pub content: String,
    pub source: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListDocumentsQuery {
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NeighborsPath {
    pub source_id: i64,
    pub chunk_index: i64,
}

// --- Handlers ---

#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    #[serde(default)]
    pub collection: Option<String>,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_max_edges")]
    pub max_edges: usize,
}

fn default_threshold() -> f32 {
    0.5
}

fn default_max_edges() -> usize {
    50
}

async fn get_graph(
    State(_server): State<Arc<RagFerriteServer>>,
    Query(params): Query<GraphQuery>,
) -> impl IntoResponse {
    match engine::get_graph_data(params.collection.as_deref(), params.threshold, params.max_edges) {
        Ok(graph) => (StatusCode::OK, Json(serde_json::json!(graph))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn status(State(_server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
    match engine::stats() {
        Ok(s) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "document_count": s.document_count,
                "version": env!("CARGO_PKG_VERSION"),
                "db_size": serde_json::Value::Null
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn list_documents(
    State(_server): State<Arc<RagFerriteServer>>,
    Query(params): Query<ListDocumentsQuery>,
) -> impl IntoResponse {
    match engine::list_sources() {
        Ok(sources) => {
            let mut out: Vec<crate::types::SourceInfo> =
                sources.into_iter().map(crate::types::SourceInfo::from).collect();

            // Filter by collection if specified
            if let Some(ref coll) = params.collection {
                out.retain(|s| &s.collection_id == coll);
            }

            (StatusCode::OK, Json(serde_json::json!({ "files": out })))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn get_document(
    State(_server): State<Arc<RagFerriteServer>>,
    Path(source_id): Path<i64>,
) -> impl IntoResponse {
    match engine::list_sources() {
        Ok(sources) => {
            let found = sources.into_iter().find(|s| s.id == source_id);
            match found {
                Some(s) => {
                    let info = crate::types::SourceInfo::from(s);
                    (
                        StatusCode::OK,
                        Json(serde_json::json!(info)),
                    )
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "error": format!("Document {} not found", source_id) })),
                ),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn get_chunk_neighbors(
    State(_server): State<Arc<RagFerriteServer>>,
    Path(params): Path<NeighborsPath>,
) -> impl IntoResponse {
    match engine::get_neighbors(params.source_id, params.chunk_index, 2, 2) {
        Ok(chunks) => {
            let out: Vec<crate::types::ChunkResult> =
                chunks.into_iter().map(crate::types::ChunkResult::from).collect();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "source_id": params.source_id,
                    "chunk_index": params.chunk_index,
                    "chunks": out
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn query_documents(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let filter = if req.source_ids.is_some()
        || req.collection.is_some()
        || req.metadata_like.is_some()
    {
        Some(rag_engine::api::hybrid_search::SearchFilter {
            source_ids: req.source_ids,
            metadata_like: req.metadata_like,
            collection_id: req.collection,
        })
    } else {
        None
    };

    match server.pipeline.query(&req.query, req.limit, filter).await {
        Ok(output) => {
            let out: Vec<crate::types::HybridResult> = output
                .results
                .into_iter()
                .map(crate::types::HybridResult::from)
                .collect();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "results": out,
                    "confidence": output.confidence,
                    "retries": output.retry_count
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn ingest_data(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestDataRequest>,
) -> impl IntoResponse {
    let coll = req.collection.as_deref();
    match engine::ingest_text(
        &server.pipeline.embedder,
        server.pipeline.llm.as_ref(),
        &req.content,
        &req.source,
        None,
        coll,
        server.max_concurrent,
        server.relevance_scoring,
        server.min_relevance_score,
    )
    .await
    {
        Ok((id, report)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "source_id": id,
                "source": req.source,
                "content_length": req.content.len(),
                "report": report
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn ingest_file(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestFileRequest>,
) -> impl IntoResponse {
    let coll = req.collection.as_deref();
    match engine::ingest_file(
        &server.pipeline.embedder,
        server.pipeline.llm.as_ref(),
        &req.file_path,
        coll,
        server.max_concurrent,
        server.relevance_scoring,
        server.min_relevance_score,
    )
    .await
    {
        Ok((id, report)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "source_id": id,
                "file_path": req.file_path,
                "collection": req.collection,
                "report": report
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn delete_document(
    State(_server): State<Arc<RagFerriteServer>>,
    Path(source_id): Path<String>,
) -> impl IntoResponse {
    match source_id.parse::<i64>() {
        Ok(id) => match engine::delete_source(id) {
            Ok(()) => (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "ok", "source_id": id })),
            ),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            ),
        },
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "source must be a numeric source_id" })),
        ),
    }
}

// --- Server startup ---

pub async fn serve(server: Arc<RagFerriteServer>, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/api/status", get(status))
        .route("/api/documents", get(list_documents))
        .route("/api/documents/{source_id}", get(get_document))
        .route(
            "/api/documents/{source_id}/chunks/{chunk_index}/neighbors",
            get(get_chunk_neighbors),
        )
        .route("/api/query", post(query_documents))
        .route("/api/ingest/data", post(ingest_data))
        .route("/api/ingest/file", post(ingest_file))
        .route("/api/documents/{source_id}", delete(delete_document))
        .route("/api/graph", get(get_graph))
        .layer(CorsLayer::permissive())
        .with_state(server);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind HTTP server on {}: {}", addr, e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;

    Ok(())
}
