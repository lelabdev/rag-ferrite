use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::RagFerriteServer;
use crate::engine;
use crate::params::*;

// --- API Key authentication middleware ---

/// Check the Authorization header against configured API keys.
///
/// Access tiers:
/// - Admin key (RAG_API_KEY) → full access (read + write)
/// - Guest key (RAG_GUEST_API_KEY) → read-only (GET + query)
/// - No keys configured → no auth (local dev)
pub fn check_api_key(
    headers: &HeaderMap,
    admin_key: &Option<String>,
    guest_key: &Option<String>,
    method: &axum::http::Method,
    path: &str,
) -> Result<(), (StatusCode, &'static str)> {
    // No keys configured → local dev, everything open
    if admin_key.is_none() && guest_key.is_none() {
        return Ok(());
    }

    // Extract Bearer token
    let Some(auth_header) = headers.get("authorization") else {
        return Err((StatusCode::UNAUTHORIZED, "Missing Authorization header"));
    };
    let auth_str = auth_header
        .to_str()
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid Authorization header"))?;
    let token = auth_str.strip_prefix("Bearer ").unwrap_or("");

    // Admin key → full access
    if let Some(admin) = admin_key {
        if token == admin {
            return Ok(());
        }
    }

    // Guest key → read-only (excluding key management endpoints)
    if let Some(guest) = guest_key {
        if token == guest {
            let is_read = method == axum::http::Method::GET || path == "/api/query"; // POST query is read-only
            // Key management requires admin key regardless of method
            if path.starts_with("/api/keys") {
                return Err((StatusCode::FORBIDDEN, "Key management requires admin key"));
            }
            if is_read {
                return Ok(());
            }
            return Err((StatusCode::FORBIDDEN, "Guest key: read-only access"));
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
    let code = match val.get("error_code").and_then(|v| v.as_str()) {
        Some("queue_full") => StatusCode::TOO_MANY_REQUESTS,
        Some("content_too_large") => StatusCode::PAYLOAD_TOO_LARGE,
        Some(_) => StatusCode::INTERNAL_SERVER_ERROR,
        None => StatusCode::OK,
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

async fn list_documents(State(_server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
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
                    (StatusCode::OK, Json(serde_json::json!(info)))
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(
                        serde_json::json!({ "error": format!("Document {} not found", source_id) }),
                    ),
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
    json_response(crate::service::neighbors_service(
        params.source_id,
        params.chunk_index,
        2,
        2,
    ))
}

async fn query_documents(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<QueryParams>,
) -> impl IntoResponse {
    // Use fallback pipeline during active ingestion (if configured)
    let pipeline = if server.ingestion_manager.get_progress().status
        == crate::ingestion::IngestStatus::Running
    {
        server
            .query_fallback_pipeline
            .as_ref()
            .unwrap_or(&server.pipeline)
    } else {
        &server.pipeline
    };
    let val = crate::service::query_service(
        pipeline,
        &req.query,
        req.limit
            .unwrap_or(server.default_query_limit)
            .clamp(1, server.max_query_limit),
        req.source_ids,
        req.metadata_like,
        req.tags,
        Some(&server.heat_tracker),
        Some(&server.chunk_heat_tracker),
    )
    .await;
    json_response(val)
}

async fn ingest_data(
    State(server): State<Arc<RagFerriteServer>>,
    Json(req): Json<IngestDataParams>,
) -> impl IntoResponse {
    let val = server
        .ingestion_manager
        .ingest_data(req.content, req.source);
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
        return json_response(
            serde_json::json!({ "error": "No files provided. Use 'file_path' or 'paths'." }),
        );
    }
    // Use API override or fall back to config default
    let move_after = req.move_after_ingest.unwrap_or(server.move_after_ingest);
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

async fn rebuild_indexes(State(server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
    json_response(server.ingestion_manager.rebuild_indexes())
}

async fn flush_indexes(State(server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
    let val = server.ingestion_manager.flush_indexes();
    json_response(val)
}

async fn cancel_batch(State(server): State<Arc<RagFerriteServer>>) -> impl IntoResponse {
    let val = server.ingestion_manager.cancel_batch();
    json_response(val)
}

async fn stop_service() -> impl IntoResponse {
    tracing::info!("Stop requested via API. Shutting down...");
    // Graceful shutdown: the caller (script wrapper) will restart the process
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        std::process::exit(0);
    });
    json_response(serde_json::json!({
        "status": "stopping",
        "message": "Server is shutting down."
    }))
}

async fn reload_config() -> impl IntoResponse {
    json_response(crate::service::reload_config_service())
}

async fn get_history() -> impl IntoResponse {
    let entries = crate::engine::history::snapshot();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "history": entries,
            "total": entries.len()
        })),
    )
}

// --- API Key management handlers ---

/// Generate a new random hex API key (64 chars = 32 bytes).
fn generate_random_key() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bytes: [u8; 32] = rng.random();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Find the .env file path used by the server.
/// The server loads from the directory of the running executable.
fn server_env_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let env = dir.join(".env");
            if env.exists() {
                return env;
            }
        }
    }
    // Fallback: ~/.config/ragfer/.env
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(format!("{}/.config/ragfer/.env", home))
}

/// Read current RAG_API_KEY from .env file (disk).
fn read_key_from_env(env_path: &PathBuf) -> Option<String> {
    if env_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(env_path) {
            for line in contents.lines() {
                if let Some(val) = line.strip_prefix("RAG_API_KEY=") {
                    let key = val.trim().trim_matches(|c: char| c == '"').to_string();
                    if !key.is_empty() {
                        return Some(key);
                    }
                }
            }
        }
    }
    None
}

/// Write RAG_API_KEY=<key> to the .env file, preserving other lines.
fn write_key_to_env(env_path: &PathBuf, new_key: &str) -> Result<(), String> {
    let mut lines = Vec::new();
    let mut found = false;

    if env_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(env_path) {
            for line in contents.lines() {
                if line.starts_with("RAG_API_KEY=") {
                    lines.push(format!("RAG_API_KEY={}", new_key));
                    found = true;
                } else {
                    lines.push(line.to_string());
                }
            }
        }
    }
    if !found {
        lines.push(format!("RAG_API_KEY={}", new_key));
    }

    if let Some(parent) = env_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
    }
    std::fs::write(env_path, lines.join("\n") + "\n")
        .map_err(|e| format!("Failed to write .env: {}", e))?;
    Ok(())
}

async fn keys_generate() -> impl IntoResponse {
    let env_path = server_env_path();
    let new_key = generate_random_key();

    match write_key_to_env(&env_path, &new_key) {
        Ok(()) => {
            tracing::info!("API key regenerated via /api/keys/generate");
            // Also update the env var so the NEXT request uses the new key
            // SAFETY: single-threaded admin operation during key rotation
            unsafe {
                std::env::set_var("RAG_API_KEY", &new_key);
            }
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "key": new_key,
                    "message": "New API key generated. The next request must use this key."
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e })),
        ),
    }
}

async fn keys_list() -> impl IntoResponse {
    let env_path = server_env_path();
    let mut keys = Vec::new();

    if let Some(key) = read_key_from_env(&env_path) {
        if key.len() > 16 {
            let masked = format!("{}...{}", &key[..8], &key[key.len() - 8..]);
            keys.push(serde_json::json!({ "masked": masked, "type": "admin" }));
        } else {
            keys.push(serde_json::json!({ "masked": format!("{}...{}", &key[..4], &key[key.len()-4..]), "type": "admin" }));
        }
    }

    // Also check guest key
    if let Ok(guest) = std::env::var("RAG_GUEST_API_KEY") {
        if !guest.is_empty() {
            if guest.len() > 16 {
                let masked = format!("{}...{}", &guest[..8], &guest[guest.len() - 8..]);
                keys.push(serde_json::json!({ "masked": masked, "type": "guest" }));
            } else {
                keys.push(serde_json::json!({ "masked": format!("{}...{}", &guest[..4], &guest[guest.len()-4..]), "type": "guest" }));
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "keys": keys,
            "total": keys.len()
        })),
    )
}

async fn keys_current() -> impl IntoResponse {
    let env_path = server_env_path();
    match read_key_from_env(&env_path) {
        Some(key) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "key": key,
                "type": "admin"
            })),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "No API key found in .env file"
            })),
        ),
    }
}

// --- Server startup ---

pub async fn serve(
    server: Arc<RagFerriteServer>,
    port: u16,
    bind_address: String,
    admin_key: Option<String>,
    guest_key: Option<String>,
    body_limit: usize,
) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    };

    // Create MCP Streamable HTTP service
    let mcp_config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default()
        .with_allowed_hosts(vec![
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            &bind_address,
            // Tailscale IPs for remote MCP access
            "100.90.185.42", // aether
            "100.97.67.73",  // nova
            "100.88.8.1",    // tuftux
            // Docker bridge IPs for LobeChat MCP access
            "10.0.1.1",       // host on lobe-network bridge
            "lobe-rag-proxy", // socat proxy in lobe-network
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
        .route("/api/ingest/file", post(ingest)) // alias — same endpoint
        .route("/api/ingest/batch", post(ingest)) // alias — same endpoint
        .route("/api/documents/{source_id}", delete(delete_document))
        .route("/api/graph", get(get_graph))
        .route("/api/rebuild-indexes", post(rebuild_indexes))
        .route("/api/flush-indexes", post(flush_indexes))
        .route("/api/service/cancel-batch", post(cancel_batch))
        .route("/api/service/stop", post(stop_service))
        .route("/api/reload", post(reload_config))
        .route("/api/history", get(get_history))
        // API key management
        .route("/api/keys/generate", post(keys_generate))
        .route("/api/keys", get(keys_list))
        .route("/api/keys/current", get(keys_current))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(body_limit.max(1)))
        .with_state(server);

    // Apply API key auth on all routes if configured
    let app = if admin_key.is_some() || guest_key.is_some() {
        let admin = admin_key.clone();
        let guest = guest_key.clone();
        if admin.is_some() {
            tracing::info!("Admin API key authentication enabled");
        }
        if guest.is_some() {
            tracing::info!("Guest API key enabled (read-only access)");
        }
        app.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let admin = admin.clone();
                let guest = guest.clone();
                async move {
                    let method = req.method().clone();
                    let path = req.uri().path().to_string();
                    let headers = req.headers().clone();
                    if let Err((status, msg)) =
                        check_api_key(&headers, &admin, &guest, &method, &path)
                    {
                        return (status, msg).into_response();
                    }
                    next.run(req).await
                }
            },
        ))
    } else {
        tracing::info!("API key authentication disabled (no keys configured — local dev)");
        app
    };

    // Nest MCP Streamable HTTP under /mcp
    let mcp_router = axum::Router::new().route(
        "/mcp",
        axum::routing::any(move |req| {
            let mcp = mcp_service.clone();
            async move {
                tower::ServiceExt::oneshot(mcp, req)
                    .await
                    .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "MCP service error"))
            }
        }),
    );

    let app = app.merge(mcp_router);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authentication_allows_admin_everywhere_and_guest_only_reads() {
        let admin = Some("admin-secret".to_string());
        let guest = Some("guest-secret".to_string());
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer admin-secret".parse().unwrap());
        assert!(
            check_api_key(
                &headers,
                &admin,
                &guest,
                &axum::http::Method::POST,
                "/api/ingest"
            )
            .is_ok()
        );

        headers.insert("authorization", "Bearer guest-secret".parse().unwrap());
        assert!(
            check_api_key(
                &headers,
                &admin,
                &guest,
                &axum::http::Method::GET,
                "/api/documents"
            )
            .is_ok()
        );
        assert_eq!(
            check_api_key(
                &headers,
                &admin,
                &guest,
                &axum::http::Method::POST,
                "/api/ingest"
            )
            .unwrap_err()
            .0,
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn authentication_rejects_invalid_or_missing_credentials() {
        let admin = Some("admin-secret".to_string());
        let guest = None;
        let headers = HeaderMap::new();
        assert_eq!(
            check_api_key(
                &headers,
                &admin,
                &guest,
                &axum::http::Method::GET,
                "/api/status"
            )
            .unwrap_err()
            .0,
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn generated_keys_are_unique_32_byte_hex_values() {
        let first = generate_random_key();
        let second = generate_random_key();
        assert_eq!(first.len(), 64);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }
}
