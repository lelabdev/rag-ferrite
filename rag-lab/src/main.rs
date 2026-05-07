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
mod http;
mod llm;
mod reranker;
mod types;

#[derive(Debug, Clone)]
struct RagLabServer {
    embedder: embedding::EmbeddingProvider,
    llm: Option<llm::LlmProvider>,
    reranker: reranker::Reranker,
}

// --- Tool parameter structs ---

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct NoParams {}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct QueryParams {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
    /// Filter by source IDs (document IDs)
    #[serde(default)]
    pub source_ids: Option<Vec<i64>>,
    /// Filter by metadata using SQL LIKE pattern (e.g. "%.pdf")
    #[serde(default)]
    pub metadata_like: Option<String>,
    /// Filter by collection name
    #[serde(default)]
    pub collection: Option<String>,
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

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagLabServer {
    #[tool(name = "query_documents", description = "Search documents using hybrid search (BM25 + vector with RRF fusion). Returns relevant chunks with scores.")]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        let limit = p.limit.unwrap_or(10);

        let filter = if p.source_ids.is_some() || p.metadata_like.is_some() || p.collection.is_some() {
            Some(rag_engine::api::hybrid_search::SearchFilter {
                source_ids: p.source_ids,
                metadata_like: p.metadata_like,
                collection_id: p.collection,
            })
        } else {
            None
        };

        match engine::search_hybrid_with_expansion(&self.embedder, self.llm.as_ref(), &p.query, limit, filter).await {
            Ok(results) => {
                // Rerank if enabled
                let reranked = if self.reranker.is_enabled() && !results.is_empty() {
                    let candidates: Vec<reranker::RerankCandidate> = results.clone().into_iter().map(|r| reranker::RerankCandidate {
                        doc_id: r.doc_id,
                        content: r.content,
                        initial_score: r.score,
                        source_id: r.source_id,
                        chunk_index: r.chunk_index,
                        metadata: r.metadata,
                        vector_rank: r.vector_rank,
                        bm25_rank: r.bm25_rank,
                    }).collect();

                    match self.reranker.rerank(&p.query, candidates).await {
                        Ok(reranked) => reranked,
                        Err(e) => {
                            tracing::warn!("Reranking failed: {}, using initial scores", e);
                            // Can't use candidates (moved), use the reranker's passthrough
                            Vec::new() // Will be handled by the else branch below
                        }
                    }
                } else {
                    Vec::new()
                };

                // If reranking produced no results (disabled or failed), convert from original results
                let final_results = if reranked.is_empty() && !results.is_empty() {
                    results.into_iter().map(|r| reranker::RerankedResult {
                        doc_id: r.doc_id,
                        content: r.content,
                        score: r.score,
                        source_id: r.source_id,
                        chunk_index: r.chunk_index,
                        metadata: r.metadata,
                        vector_rank: r.vector_rank,
                        bm25_rank: r.bm25_rank,
                    }).collect()
                } else {
                    reranked
                };

                let out: Vec<types::HybridResult> = final_results.into_iter().map(|r| types::HybridResult {
                    doc_id: r.doc_id,
                    content: r.content,
                    score: r.score,
                    source_id: r.source_id,
                    chunk_index: r.chunk_index,
                    metadata: r.metadata,
                    vector_rank: r.vector_rank,
                    bm25_rank: r.bm25_rank,
                }).collect();
                serde_json::json!({ "results": out }).to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, DOCX, TXT, MD) into the RAG.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        let p = params.0;
        match engine::ingest_file(&self.embedder, self.llm.as_ref(), &p.file_path).await {
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
        match engine::ingest_text(&self.embedder, self.llm.as_ref(), &p.content, &p.source, None).await {
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
                let out: Vec<types::SourceInfo> = sources.into_iter().map(types::SourceInfo::from).collect();
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
                let out: Vec<types::ChunkResult> = chunks.into_iter().map(types::ChunkResult::from).collect();
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

    // Init LLM provider (optional, for contextual retrieval)
    let llm = if config.llm.context_enabled {
        let provider = llm::LlmProvider::new(
            config.llm.provider.clone(),
            config.llm.model.clone(),
            config.llm.api_key.clone(),
            config.llm.base_url.clone(),
        );
        tracing::info!("LLM provider: {} / {} (contextual retrieval enabled)", config.llm.provider, config.llm.model);
        Some(provider)
    } else {
        tracing::info!("Contextual retrieval disabled");
        None
    };

    let server = RagLabServer { embedder: embedder.clone(), llm: llm.clone(), reranker: reranker::Reranker::disabled() };

    // Mode: stdio-only or dual (stdio + HTTP)
    if config.http_port > 0 {
        let http_state = http::AppState {
            embedder,
            llm,
            api_key: None,
        };

        tracing::info!("Starting dual mode: MCP stdio + HTTP on port {}", config.http_port);
        let http_port = config.http_port;

        tokio::select! {
            r = async {
                let service = server.serve(rmcp::transport::io::stdio()).await?;
                service.waiting().await?;
                Ok::<(), anyhow::Error>(())
            } => r?,
            r = http::start_server(http_state, http_port) => r?,
        }
    } else {
        tracing::info!("Starting MCP server on stdio...");
        let service = server.serve(rmcp::transport::io::stdio()).await?;
        service.waiting().await?;
    }

    Ok(())
}
