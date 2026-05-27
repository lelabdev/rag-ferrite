use anyhow::Result;
use std::sync::Arc;
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod api;
mod config;
mod chunker;
mod embedding;
mod engine;
mod extractor;
mod llm;
mod pipeline;
mod reranker;
mod types;

#[derive(Debug, Clone)]
struct RagFerriteServer {
    pub pipeline: pipeline::QueryPipeline,
    pub max_concurrent: usize,
    pub relevance_scoring: bool,
    pub min_relevance_score: f32,
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
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct IngestDataParams {
    pub content: String,
    pub source: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub collection: Option<String>,
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

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct CheckIngestionParams {
    /// Path to the file to check
    pub file_path: Option<String>,
    /// Raw content to check (alternative to file_path)
    pub content: Option<String>,
    /// Source name for duplicate detection (used with content)
    pub source_name: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
struct BenchmarkParams {
    /// Path to the golden dataset JSON file
    pub file_path: String,
    /// Optional collection to filter queries against
    #[serde(default)]
    pub collection: Option<String>,
    /// Number of top results to consider per query (default: 10)
    #[serde(default = "default_limit")]
    pub limit: Option<usize>,
}

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagFerriteServer {
    #[tool(name = "query_documents", description = "Search documents using hybrid search (BM25 + vector with RRF fusion). Returns relevant chunks with scores.")]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        let limit = p.limit.unwrap_or(10);

        let filter = if p.source_ids.is_some() || p.collection.is_some() || p.metadata_like.is_some() {
            Some(rag_engine::api::hybrid_search::SearchFilter {
                source_ids: p.source_ids,
                metadata_like: p.metadata_like,
                collection_id: p.collection,
            })
        } else {
            None
        };

        match self.pipeline.query(&p.query, limit, filter).await {
            Ok(output) => {
                // Fetch section_paths and tags for all result doc_ids
                let doc_ids: Vec<i64> = output.results.iter().map(|r| r.doc_id).collect();
                let section_map = engine::get_section_paths_for_chunk_ids(&doc_ids).unwrap_or_default();
                let tags_map = engine::get_tags_for_chunk_ids(&doc_ids).unwrap_or_default();

                let out: Vec<types::HybridResult> = output.results.into_iter().map(|r| {
                    let sp = section_map.get(&r.doc_id).cloned().flatten();
                    let tags = tags_map.get(&r.doc_id).cloned().unwrap_or_default();
                    types::HybridResult {
                        doc_id: r.doc_id,
                        content: r.content,
                        score: r.score,
                        source_id: r.source_id,
                        chunk_index: r.chunk_index,
                        metadata: r.metadata,
                        vector_rank: r.vector_rank,
                        bm25_rank: r.bm25_rank,
                        section_path: sp,
                        page: None,
                        rerank_score: r.rerank_score,
                        tags,
                    }
                }).collect();
                serde_json::json!({
                    "results": out,
                    "confidence": output.confidence,
                    "retries": output.retry_count
                }).to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, DOCX, TXT, MD) into the RAG. Optionally specify a collection.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        let p = params.0;
        let coll = p.collection.as_deref();
        match engine::ingest_file(&self.pipeline.embedder, self.pipeline.llm.as_ref(), &p.file_path, coll, self.max_concurrent, self.relevance_scoring, self.min_relevance_score).await {
            Ok((id, report)) => serde_json::json!({
                "status": "ok",
                "source_id": id,
                "file_path": p.file_path,
                "collection": p.collection,
                "report": report
            }).to_string(),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "ingest_data", description = "Index content directly (text, HTML, or markdown) with a source identifier. Optionally specify a collection.")]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        let coll = p.collection.as_deref();
        match engine::ingest_text(&self.pipeline.embedder, self.pipeline.llm.as_ref(), &p.content, &p.source, None, coll, self.max_concurrent, self.relevance_scoring, self.min_relevance_score).await {
            Ok((id, report)) => serde_json::json!({
                "status": "ok",
                "source_id": id,
                "source": p.source,
                "content_length": p.content.len(),
                "report": report
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

    #[tool(name = "check_ingestion", description = "Pre-ingestion quality check: analyze a document before indexing. Returns char count, estimated chunks, language, duplicate detection, and warnings.")]
    async fn check_ingestion(&self, params: Parameters<CheckIngestionParams>) -> String {
        let p = params.0;
        let (content, filename) = if let Some(ref file_path) = p.file_path {
            match crate::extractor::extract_text(file_path) {
                Ok(text) => {
                    let name = std::path::Path::new(file_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| file_path.clone());
                    (text, name)
                }
                Err(e) => return serde_json::json!({ "error": format!("Failed to extract text: {}", e) }).to_string(),
            }
        } else if let Some(ref content) = p.content {
            let name = p.source_name.unwrap_or_else(|| "inline_content".to_string());
            (content.clone(), name)
        } else {
            return serde_json::json!({ "error": "Provide either file_path or content" }).to_string();
        };

        let report = engine::pre_check_document(&content, &filename);
        serde_json::json!({ "pre_check": report }).to_string()
    }

    #[tool(name = "benchmark", description = "Evaluate retrieval quality against a golden dataset JSON file (array of {question, expected_keywords, relevant_source_ids}). Returns hit rate and per-query details.")]
    async fn benchmark(&self, params: Parameters<BenchmarkParams>) -> String {
        let p = params.0;
        let content = match std::fs::read_to_string(&p.file_path) {
            Ok(c) => c,
            Err(e) => return serde_json::json!({ "error": format!("Failed to read golden dataset: {}", e) }).to_string(),
        };
        let entries: Vec<types::GoldenEntry> = match serde_json::from_str(&content) {
            Ok(e) => e,
            Err(e) => return serde_json::json!({ "error": format!("Invalid JSON: {}", e) }).to_string(),
        };
        if entries.is_empty() {
            return serde_json::json!({ "error": "Golden dataset is empty" }).to_string();
        }
        let limit = p.limit.unwrap_or(10);
        match engine::run_benchmark(&self.pipeline.embedder, entries, p.collection, limit).await {
            Ok(result) => serde_json::to_string(&result).unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }).to_string()),
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
                let out: Vec<types::ChunkResult> = chunks.into_iter().map(|(chunk, section_path, page)| {
                    types::ChunkResult {
                        chunk_id: chunk.chunk_id,
                        source_id: chunk.source_id,
                        chunk_index: chunk.chunk_index,
                        content: chunk.content,
                        score: chunk.similarity,
                        metadata: chunk.metadata,
                        chunk_type: chunk.chunk_type,
                        section_path,
                        page,
                    }
                }).collect();
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
    // Log to file for debugging MCP issues
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("rag-ferrite.log")?;
    tracing_subscriber::fmt()
        .with_env_filter("rag_ferrite=debug,rag_engine=debug")
        .with_writer(std::sync::Mutex::new(log_file))
        .init();

    let config = config::Config::load()?;
    tracing::info!("rag-ferrite v{} starting — data: {}", env!("CARGO_PKG_VERSION"), config.data_dir.display());

    // Init rag_engine
    engine::init(&config.data_dir, &config)?;

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
        let mut provider = llm::LlmProvider::new(
            config.llm.provider.clone(),
            config.llm.model.clone(),
            config.llm.api_key.clone(),
            config.llm.base_url.clone(),
        );

        // Set up fallback if configured
        if let Some(ref fb) = config.llm.fallback {
            let fb_provider = llm::LlmProvider::new_fallback(
                fb.provider.clone(),
                fb.model.clone(),
                fb.api_key.clone(),
                fb.base_url.clone(),
            );
            provider = provider.with_fallback(fb_provider);
            tracing::info!("LLM provider: {} / {} → fallback: {} / {} (contextual retrieval enabled)",
                config.llm.provider, config.llm.model, fb.provider, fb.model);
        } else {
            tracing::info!("LLM provider: {} / {} (contextual retrieval enabled)",
                config.llm.provider, config.llm.model);
        }

        Some(provider)
    } else {
        tracing::info!("Contextual retrieval disabled");
        None
    };

    // Build reranker from LLM provider
    let reranker = match config.reranker.reranker_type.as_str() {
        "llm" => {
            if let Some(ref llm_provider) = llm {
                tracing::info!("Reranker: LLM (reusing main LLM provider)");
                reranker::Reranker::new_llm(Arc::new(llm_provider.clone()))
            } else {
                tracing::warn!("Reranker: LLM requested but no LLM provider available, disabling");
                reranker::Reranker::disabled()
            }
        }
        "cohere" => {
            let key = config.reranker.api_key.clone()
                .expect("Cohere reranker requires reranker.api_key");
            tracing::info!("Reranker: Cohere");
            reranker::Reranker::new_cohere(key)
        }
        _ => {
            tracing::info!("Reranker: disabled");
            reranker::Reranker::disabled()
        }
    };

    let server = RagFerriteServer {
        pipeline: pipeline::QueryPipeline::new(
            embedder.clone(),
            llm.clone(),
            reranker,
            0.3,
            1,
        ),
        max_concurrent: config.llm.max_concurrent,
        relevance_scoring: config.llm.relevance_scoring,
        min_relevance_score: config.llm.min_relevance_score,
    };

    let server = Arc::new(server);

    if config.http_port > 0 {
        let http_server = server.clone();
        let http_port = config.http_port;
        tracing::info!("Starting dual mode: MCP stdio + HTTP on port {}", http_port);

        tokio::select! {
            r = async {
                let service = server.serve(rmcp::transport::io::stdio()).await?;
                service.waiting().await?;
                Ok::<(), anyhow::Error>(())
            } => r?,
            r = api::serve(http_server, http_port) => r?,
        }
    } else {
        tracing::info!("Starting MCP server on stdio...");
        let service = server.serve(rmcp::transport::io::stdio()).await?;
        service.waiting().await?;
    }

    Ok(())
}
