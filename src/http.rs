use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

use crate::engine;
use crate::embedding::EmbeddingProvider;
use crate::llm::LlmProvider;

/// Shared state for HTTP handlers
#[derive(Clone)]
pub struct AppState {
    pub embedder: EmbeddingProvider,
    pub llm: Option<LlmProvider>,
    pub api_key: Option<String>,
    pub max_concurrent: usize,
}

// --- Request/Response types ---

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

fn default_limit() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub document_count: usize,
    pub version: String,
}

// --- Handlers ---

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

pub async fn status() -> Result<Json<StatusResponse>, (StatusCode, Json<serde_json::Value>)> {
    match engine::stats() {
        Ok(s) => Ok(Json(StatusResponse {
            document_count: s.document_count,
            version: env!("CARGO_PKG_VERSION").into(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

pub async fn query(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<QueryRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let filter = if req.source_ids.is_some() || req.collection.is_some() || req.metadata_like.is_some() {
        Some(rag_engine::api::hybrid_search::SearchFilter {
            source_ids: req.source_ids,
            metadata_like: req.metadata_like,
            collection_id: req.collection,
        })
    } else {
        None
    };

    match engine::search_hybrid_with_expansion(&state.embedder, state.llm.as_ref(), &req.query, req.limit, filter).await {
        Ok(results) => {
            let out: Vec<crate::types::HybridResult> = results.into_iter().map(crate::types::HybridResult::from).collect();
            Ok(Json(serde_json::json!({ "results": out })))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct IngestFileRequest {
    pub file_path: String,
    #[serde(default)]
    pub collection: Option<String>,
}

pub async fn ingest_file(
    State(state): State<AppState>,
    Json(req): Json<IngestFileRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let coll = req.collection.as_deref();
    match engine::ingest_file(&state.embedder, state.llm.as_ref(), &req.file_path, coll, state.max_concurrent).await {
        Ok(id) => Ok(Json(serde_json::json!({
            "status": "ok",
            "source_id": id,
            "file_path": req.file_path
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct IngestDataRequest {
    pub content: String,
    pub source: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub collection: Option<String>,
}

pub async fn ingest_data(
    State(state): State<AppState>,
    Json(req): Json<IngestDataRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let coll = req.collection.as_deref();
    match engine::ingest_text(&state.embedder, state.llm.as_ref(), &req.content, &req.source, None, coll, state.max_concurrent).await {
        Ok(id) => Ok(Json(serde_json::json!({
            "status": "ok",
            "source_id": id,
            "source": req.source,
            "content_length": req.content.len()
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

// --- Auth middleware ---

pub fn check_api_key(headers: &HeaderMap, expected: &str) -> Result<(), StatusCode> {
    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.strip_prefix("Bearer ").unwrap_or(v));

    match provided {
        Some(key) if key == expected => Ok(()),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// --- SSE endpoint for MCP over SSE ---

#[derive(Debug, Deserialize)]
pub struct McpMessage {
    pub jsonrpc: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    #[serde(default)]
    pub id: Option<serde_json::Value>,
}

pub async fn mcp_sse() -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    // SSE stream that sends endpoint info and keeps alive
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(1);

    tokio::spawn(async move {
        // Send initial endpoint event
        let _ = tx.send(Ok(Event::default().event("endpoint").data("/messages"))).await;
        // Keep alive
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            if tx.send(Ok(Event::default().event("ping").data(""))).await.is_err() {
                break;
            }
        }
    });

    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx)).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(30)),
    )
}

// --- Server startup ---

pub async fn start_server(state: AppState, port: u16) -> Result<()> {
    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/query", post(query))
        .route("/ingest/file", post(ingest_file))
        .route("/ingest/data", post(ingest_data))
        .route("/sse", get(mcp_sse))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
