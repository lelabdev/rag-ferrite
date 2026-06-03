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
use tower::ServiceExt;

use crate::engine;
use crate::params::*;
use crate::RagFerriteServer;

// --- HTTP-only request types (differ from MCP params) ---

#[derive(Debug, Deserialize)]
pub struct NeighborsPath {
    pub source_id: i64,
    pub chunk_index: i64,
}

#[derive(Debug, Deserialize)]
pub struct GraphQuery {
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
    match engine::get_graph_data(None, params.threshold, params.max_edges) {
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

async fn ingest_progress(State(server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
    let progress = server.ingestion_manager.get_progress();
    (StatusCode::OK, Json(serde_json::json!(progress)))
}

async fn list_documents(
    State(_server): State<Arc<RagFerriteServer>>,
) -> impl IntoResponse {
    json_response(crate::service::list_sources_service())
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
    // Use fallback pipeline during active ingestion (if configured)
    let pipeline = if server.ingestion_manager.get_progress().status == crate::ingestion::IngestStatus::Running {
        server.query_fallback_pipeline.as_ref().unwrap_or(&server.pipeline)
    } else {
        &server.pipeline
    };
    let val = crate::service::query_service(
        pipeline,
        &req.query,
        req.limit.unwrap_or(server.default_query_limit).clamp(1, server.max_query_limit),
        req.source_ids,
        req.metadata_like,
    )
    .await;
    json_response(val)
}

async fn ingest_data(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestDataParams>,
) -> impl IntoResponse {
    let val = server.ingestion_manager.ingest_data(
        req.content,
        req.source,
    );
    json_response(val)
}

async fn ingest_file(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestFileParams>,
) -> impl IntoResponse {
    let val = server.ingestion_manager.ingest_file(
        req.file_path,
    );
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

async fn rebuild_indexes(
    State(_server): State<Arc<RagFerriteServer>>,
) -> impl IntoResponse {
    tokio::task::spawn_blocking(|| {
        engine::rebuild_and_save_indexes("general");
        engine::wal_checkpoint();
    });
    (StatusCode::OK, Json(serde_json::json!({"status": "rebuilding + WAL checkpoint"})))
}

async fn flush_indexes(
    State(server): State<Arc<RagFerriteServer>>,
) -> impl IntoResponse {
    let val = server.ingestion_manager.flush_indexes();
    json_response(val)
}

// --- Server startup ---

pub async fn serve(server: Arc<RagFerriteServer>, port: u16, bind_address: String) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService,
        session::local::LocalSessionManager,
    };

    // Create MCP Streamable HTTP service
    let mcp_config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default()
        .with_allowed_hosts(vec![
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            &bind_address,
        ]);

    let mcp_service = StreamableHttpService::new(
        {
            let server = server.clone();
            move || Ok((*server).clone())
        },
        Arc::new(LocalSessionManager::default()),
        mcp_config,
    );

    let app = Router::new()
        // REST API
        .route("/api/status", get(status))
        .route("/api/ingest/progress", get(ingest_progress))
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
        .route("/api/rebuild-indexes", post(rebuild_indexes))
        .route("/api/flush-indexes", post(flush_indexes))
        .layer(CorsLayer::permissive())
        .with_state(server);

    // Nest MCP Streamable HTTP under /mcp
    let mcp_router = axum::Router::new()
        .route("/mcp", axum::routing::any(move |req| {
            let mcp = mcp_service.clone();
            async move {
                tower::ServiceExt::oneshot(mcp, req)
                    .await
                    .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "MCP service error"))
            }
        }));

    let app = app
        .merge(mcp_router);

    let addr = format!("{}:{}", bind_address, port);
    tracing::info!("HTTP + MCP Streamable HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind HTTP server on {}: {}", addr, e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;

    Ok(())
}
