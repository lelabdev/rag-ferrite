use anyhow::Result;
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod config;
mod embedding;
mod engine;

#[derive(Debug, Clone)]
struct RagLabServer {
    embedder: embedding::EmbeddingProvider,
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

fn default_limit() -> Option<usize> { Some(10) }

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
    pub source_id: i64,
    pub chunk_index: i64,
    #[serde(default = "default_before")]
    pub before: Option<i64>,
    #[serde(default = "default_after")]
    pub after: Option<i64>,
}

fn default_before() -> Option<i64> { Some(2) }
fn default_after() -> Option<i64> { Some(2) }

// --- Serialization helpers ---

#[derive(Debug, Serialize)]
struct HybridResult {
    doc_id: i64,
    content: String,
    score: f64,
    vector_rank: u32,
    bm25_rank: u32,
    source_id: i64,
    chunk_index: u32,
    metadata: Option<String>,
}

impl From<rag_engine::api::hybrid_search::HybridSearchResult> for HybridResult {
    fn from(r: rag_engine::api::hybrid_search::HybridSearchResult) -> Self {
        HybridResult {
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
            vector_rank: r.vector_rank,
            bm25_rank: r.bm25_rank,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            metadata: r.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChunkResult {
    chunk_id: i64,
    source_id: i64,
    chunk_index: i32,
    content: String,
    score: f64,
    metadata: Option<String>,
}

impl From<rag_engine::api::source_rag::ChunkSearchResult> for ChunkResult {
    fn from(r: rag_engine::api::source_rag::ChunkSearchResult) -> Self {
        ChunkResult {
            chunk_id: r.chunk_id,
            source_id: r.source_id,
            chunk_index: r.chunk_index,
            content: r.content,
            score: r.similarity,
            metadata: r.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
struct SourceInfo {
    id: i64,
    name: Option<String>,
    created_at: i64,
    metadata: Option<String>,
    status: Option<String>,
}

impl From<rag_engine::api::source_rag::SourceEntry> for SourceInfo {
    fn from(s: rag_engine::api::source_rag::SourceEntry) -> Self {
        SourceInfo {
            id: s.id,
            name: s.name,
            created_at: s.created_at,
            metadata: s.metadata,
            status: s.status,
        }
    }
}

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagLabServer {
    #[tool(name = "query_documents", description = "Search documents using hybrid search (BM25 + vector with RRF fusion). Returns relevant chunks with scores.")]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        let limit = p.limit.unwrap_or(10);

        match engine::search_hybrid(&self.embedder, &p.query, limit).await {
            Ok(results) => {
                let out: Vec<HybridResult> = results.into_iter().map(HybridResult::from).collect();
                serde_json::json!({ "results": out }).to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, DOCX, TXT, MD) into the RAG.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        let p = params.0;
        match engine::ingest_file(&self.embedder, &p.file_path).await {
            Ok(id) => serde_json::json!({
                "status": "ok",
                "source_id": id,
                "file_path": p.file_path
            }).to_string(),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "ingest_data", description = "Index content directly (text, HTML, or markdown) with a source identifier.")]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        match engine::ingest_text(&self.embedder, &p.content, &p.source, None).await {
            Ok(id) => serde_json::json!({
                "status": "ok",
                "source_id": id,
                "source": p.source,
                "content_length": p.content.len()
            }).to_string(),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "delete_file", description = "Remove a document and all its chunks by source ID.")]
    async fn delete_file(&self, params: Parameters<DeleteParams>) -> String {
        let p = params.0;
        match p.source.parse::<i64>() {
            Ok(id) => match engine::delete_source(id) {
                Ok(()) => serde_json::json!({ "status": "ok", "source_id": id }).to_string(),
                Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
            },
            Err(_) => serde_json::json!({ "error": "source must be a numeric source_id" }).to_string(),
        }
    }

    #[tool(name = "list_files", description = "List all indexed documents with their metadata.")]
    async fn list_files(&self, _params: Parameters<NoParams>) -> String {
        match engine::list_sources() {
            Ok(sources) => {
                let out: Vec<SourceInfo> = sources.into_iter().map(SourceInfo::from).collect();
                serde_json::json!({ "files": out }).to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "status", description = "Get RAG engine status: document count.")]
    async fn status(&self, _params: Parameters<NoParams>) -> String {
        match engine::stats() {
            Ok(s) => serde_json::json!({
                "document_count": s.document_count,
                "version": env!("CARGO_PKG_VERSION")
            }).to_string(),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "read_chunk_neighbors", description = "Get chunks adjacent to a specific chunk for context expansion.")]
    async fn read_chunk_neighbors(&self, params: Parameters<ChunkNeighborsParams>) -> String {
        let p = params.0;
        let before = p.before.unwrap_or(2);
        let after = p.after.unwrap_or(2);

        match engine::get_neighbors(p.source_id, p.chunk_index, before, after) {
            Ok(chunks) => {
                let out: Vec<ChunkResult> = chunks.into_iter().map(ChunkResult::from).collect();
                serde_json::json!({
                    "source_id": p.source_id,
                    "chunk_index": p.chunk_index,
                    "chunks": out
                }).to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("rag_lab=debug")
        .init();

    let config = config::Config::load()?;
    tracing::info!("rag-lab v{} starting — data: {}", env!("CARGO_PKG_VERSION"), config.data_dir.display());

    // Init rag_engine
    engine::init(&config.data_dir)?;

    // Init embedding provider
    let embedder = embedding::EmbeddingProvider::new(
        config.embedding.provider.clone(),
        config.embedding.model.clone(),
        config.embedding.dimensions,
        config.embedding.api_key.clone(),
        config.embedding.base_url.clone(),
    );
    tracing::info!("Embedding provider: {} / {}", config.embedding.provider, config.embedding.model);

    let server = RagLabServer { embedder };

    tracing::info!("Starting MCP server on stdio...");
    let service = server.serve(rmcp::transport::io::stdio()).await?;
    service.waiting().await?;

    Ok(())
}
