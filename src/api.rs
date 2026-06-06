use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
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

// --- API Key authentication middleware ---

/// Check the Authorization header against the configured API key.
/// Returns Ok(()) if authorized, Err((StatusCode, &str)) otherwise.
/// If no api_key is configured, all requests are allowed.
pub fn check_api_key(
    headers: &HeaderMap,
    expected_key: &Option<String>,
) -> Result<(), (StatusCode, &'static str)> {
    let Some(expected) = expected_key else {
        return Ok(()); // No auth configured
    };
    let Some(auth_header) = headers.get("authorization") else {
        return Err((StatusCode::UNAUTHORIZED, "Missing Authorization header"));
    };
    let auth_str = auth_header.to_str().map_err(|_| {
        (StatusCode::BAD_REQUEST, "Invalid Authorization header")
    })?;
    if let Some(token) = auth_str.strip_prefix("Bearer ") {
        if token == expected {
            return Ok(());
        }
    }
    Err((StatusCode::UNAUTHORIZED, "Invalid API key"))
}

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

#[derive(serde::Deserialize)]
struct IngestParams {
    /// Single file path (legacy, convenience)
    file_path: Option<String>,
    /// Multiple file paths (batch)
    paths: Option<Vec<String>>,
    /// Override config default for moving files after ingestion
    #[serde(default)]
    move_after_ingest: Option<bool>,
}

async fn ingest(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestParams>,
) -> impl IntoResponse {
    // Merge file_path into paths — unified single/batch endpoint
    let mut all_paths: Vec<String> = req.paths.unwrap_or_default();
    if let Some(fp) = req.file_path {
        all_paths.insert(0, fp);
    }
    if all_paths.is_empty() {
        return json_response(serde_json::json!({ "error": "No files provided. Use 'file_path' or 'paths'." }));
    }
    // Use API override or fall back to config default
    let move_after = req.move_after_ingest.unwrap_or(true);
    let val = server.ingestion_manager.ingest_batch(all_paths, move_after);
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

pub async fn serve(server: Arc<RagFerriteServer>, port: u16, bind_address: String, api_key: Option<String>) -> anyhow::Result<()> {
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
            // Tailscale IPs for remote MCP access
            "100.90.185.42",  // aether
            "100.97.67.73",   // nova
            "100.88.8.1",     // tuftux
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
        .route("/api/ingest", post(ingest))
        .route("/api/ingest/file", post(ingest))  // alias — same endpoint
        .route("/api/ingest/batch", post(ingest))  // alias — same endpoint
        .route("/api/documents/{source_id}", delete(delete_document))
        .route("/api/graph", get(get_graph))
        .route("/api/rebuild-indexes", post(rebuild_indexes))
        .route("/api/flush-indexes", post(flush_indexes))
        .layer(CorsLayer::permissive())
        .with_state(server);

    // Apply API key auth on all routes if configured
    let app = if let Some(ref key) = api_key {
        tracing::info!("API key authentication enabled");
        let key = key.clone();
        app.layer(axum::middleware::from_fn(move |req: axum::extract::Request, next: axum::middleware::Next| {
            let key = key.clone();
            async move {
                let headers = req.headers().clone();
                if let Err((status, msg)) = check_api_key(&headers, &Some(key)) {
                    return (status, msg).into_response();
                }
                next.run(req).await
            }
        }))
    } else {
        tracing::info!("API key authentication disabled (no key configured)");
        app
    };

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
