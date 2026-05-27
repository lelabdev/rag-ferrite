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
use crate::params::*;
use crate::RagFerriteServer;

// --- HTTP-only request types (differ from MCP params) ---

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

// --- Helpers ---

fn json_response(val: serde_json::Value) -> (StatusCode, Json<serde_json::Value>) {
    let code = if val.get("error").is_some() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    };
    (code, Json(val))
}

// --- Handlers ---

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
    json_response(crate::service::status_service())
}

async fn list_documents(
    State(_server): State<Arc<RagFerriteServer>>,
    Query(params): Query<ListDocumentsQuery>,
) -> impl IntoResponse {
    json_response(crate::service::list_sources_service(params.collection.as_deref()))
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
    json_response(crate::service::neighbors_service(params.source_id, params.chunk_index, 2, 2))
}

async fn query_documents(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<QueryParams>,
) -> impl IntoResponse {
    let val = crate::service::query_service(
        &server.pipeline,
        &req.query,
        req.limit.unwrap_or(server.default_query_limit).clamp(1, server.max_query_limit),
        req.source_ids,
        req.metadata_like,
        req.collection,
    )
    .await;
    json_response(val)
}

async fn ingest_data(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestDataParams>,
) -> impl IntoResponse {
    let val = crate::service::ingest_data_service(
        &server.pipeline,
        &server.ingest_config,
        &req.content,
        &req.source,
        req.collection.as_deref(),
    )
    .await;
    json_response(val)
}

async fn ingest_file(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestFileParams>,
) -> impl IntoResponse {
    let val = crate::service::ingest_file_service(
        &server.pipeline,
        &server.ingest_config,
        &req.file_path,
        req.collection.as_deref(),
    )
    .await;
    json_response(val)
}

async fn delete_document(
    State(_server): State<Arc<RagFerriteServer>>,
    Path(source_id): Path<String>,
) -> impl IntoResponse {
    let val = crate::service::delete_service(&source_id);
    let code = if val.get("error").is_some() {
        if source_id.parse::<i64>().is_err() {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    } else {
        StatusCode::OK
    };
    (code, Json(val))
}

// --- Server startup ---

pub async fn serve(server: Arc<RagFerriteServer>, port: u16, bind_address: String) -> anyhow::Result<()> {
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

    let addr = format!("{}:{}", bind_address, port);
    tracing::info!("HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind HTTP server on {}: {}", addr, e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;

    Ok(())
}
