use anyhow::Result;
use std::sync::{Arc, Mutex};
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod config;

#[derive(Debug, Clone)]
struct RagLabServer {
    db: Arc<Mutex<rusqlite::Connection>>,
}

// --- Tool parameter structs ---

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct NoParams {}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct QueryParams {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
}

fn default_limit() -> Option<usize> {
    Some(10)
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct IngestFileParams {
    pub file_path: String,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct IngestDataParams {
    pub content: String,
    pub source: String,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct DeleteParams {
    pub source: String,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct ChunkNeighborsParams {
    pub chunk_index: usize,
    #[serde(default = "default_before")]
    pub before: Option<usize>,
    #[serde(default = "default_after")]
    pub after: Option<usize>,
}

fn default_before() -> Option<usize> { Some(2) }
fn default_after() -> Option<usize> { Some(2) }

// --- Tool result structs ---

#[derive(Debug, Serialize)]
struct SearchResult {
    pub content: String,
    pub score: f64,
    pub chunk_index: usize,
    pub source_id: String,
    pub metadata: Option<String>,
}

#[derive(Debug, Serialize)]
struct StatusResult {
    pub document_count: usize,
    pub chunk_count: usize,
    pub db_size_bytes: u64,
}

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagLabServer {
    #[tool(name = "query_documents", description = "Search documents using hybrid search (BM25 + vector). Returns relevant chunks with scores.")]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        let limit = p.limit.unwrap_or(10);
        // TODO: integrate rag_engine hybrid search
        serde_json::json!({
            "results": [],
            "query": p.query,
            "limit": limit,
            "note": "rag_engine integration pending"
        }).to_string()
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, DOCX, TXT, MD) into the RAG.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        let p = params.0;
        // TODO: integrate rag_engine document_parser + chunking + embedding
        serde_json::json!({
            "status": "pending",
            "file_path": p.file_path,
            "note": "rag_engine integration pending"
        }).to_string()
    }

    #[tool(name = "ingest_data", description = "Index content directly (text, HTML, or markdown) with a source identifier.")]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        // TODO: integrate rag_engine chunking + embedding
        serde_json::json!({
            "status": "pending",
            "source": p.source,
            "content_length": p.content.len(),
            "note": "rag_engine integration pending"
        }).to_string()
    }

    #[tool(name = "delete_file", description = "Remove a document and all its chunks by source identifier.")]
    async fn delete_file(&self, params: Parameters<DeleteParams>) -> String {
        let p = params.0;
        // TODO: integrate rag_engine source_rag::delete_source
        serde_json::json!({
            "status": "pending",
            "source": p.source,
            "note": "rag_engine integration pending"
        }).to_string()
    }

    #[tool(name = "list_files", description = "List all indexed documents with their metadata.")]
    async fn list_files(&self, _params: Parameters<NoParams>) -> String {
        // TODO: integrate rag_engine source_rag::list_sources
        serde_json::json!({
            "files": [],
            "note": "rag_engine integration pending"
        }).to_string()
    }

    #[tool(name = "status", description = "Get RAG engine status: document count, chunk count, database size.")]
    async fn status(&self, _params: Parameters<NoParams>) -> String {
        // TODO: integrate rag_engine source_rag::get_source_stats
        serde_json::json!({
            "document_count": 0,
            "chunk_count": 0,
            "db_size_bytes": 0,
            "status": "scaffold",
            "version": env!("CARGO_PKG_VERSION")
        }).to_string()
    }

    #[tool(name = "read_chunk_neighbors", description = "Get chunks adjacent to a specific chunk for context expansion.")]
    async fn read_chunk_neighbors(&self, params: Parameters<ChunkNeighborsParams>) -> String {
        let p = params.0;
        let before = p.before.unwrap_or(2);
        let after = p.after.unwrap_or(2);
        // TODO: integrate rag_engine source_rag::get_adjacent_chunks
        serde_json::json!({
            "chunk_index": p.chunk_index,
            "before": before,
            "after": after,
            "chunks": [],
            "note": "rag_engine integration pending"
        }).to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("rag_lab=debug")
        .init();

    let config = config::Config::load()?;
    tracing::info!("rag-lab v{} starting — data: {}", env!("CARGO_PKG_VERSION"), config.data_dir.display());

    // Init SQLite
    let db_path = config.data_dir.join("rag.sqlite3");
    std::fs::create_dir_all(&config.data_dir)?;
    let conn = rusqlite::Connection::open(&db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    // TODO: rag_engine schema init

    let server = RagLabServer {
        db: Arc::new(Mutex::new(conn)),
    };

    tracing::info!("Starting MCP server on stdio...");
    let service = server.serve(rmcp::transport::io::stdio()).await?;
    service.waiting().await?;

    Ok(())
}
