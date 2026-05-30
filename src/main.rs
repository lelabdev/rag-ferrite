use anyhow::Result;
use std::sync::Arc;
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    tool, tool_router,
};

mod api;
mod config;
mod chunker;
mod embedding;
mod engine;
mod extractor;
mod ingestion;
mod llm;
mod params;
mod pipeline;
mod reranker;
mod service;
mod types;

use params::*;

#[derive(Clone)]
struct RagFerriteServer {
    pub pipeline: pipeline::QueryPipeline,
    pub query_fallback_pipeline: Option<pipeline::QueryPipeline>,
    pub ingest_config: params::IngestConfig,
    pub ingestion_manager: ingestion::IngestionManager,
    pub default_query_limit: usize,
    pub max_query_limit: usize,
}

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagFerriteServer {
    #[tool(name = "query_documents", description = "Search documents using hybrid search (BM25 + vector with RRF fusion). Returns relevant chunks with scores.")]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        // Use fallback pipeline during active ingestion (if configured)
        let pipeline = if self.ingestion_manager.get_progress().status == ingestion::IngestStatus::Running {
            self.query_fallback_pipeline.as_ref().unwrap_or(&self.pipeline)
        } else {
            &self.pipeline
        };
        service::query_service(
            pipeline,
            &p.query,
            p.limit.unwrap_or(self.default_query_limit).clamp(1, self.max_query_limit),
            p.source_ids,
            p.metadata_like,
            p.collection,
        )
        .await
        .to_string()
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, TXT, MD) into the RAG. Optionally specify a collection.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        let p = params.0;
        service::ingest_file_service(
            &self.pipeline,
            &self.ingest_config,
            &p.file_path,
            p.collection.as_deref(),
        )
        .await
        .to_string()
    }

    #[tool(name = "ingest_data", description = "Index content directly (text, HTML, or markdown) with a source identifier. Optionally specify a collection.")]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        service::ingest_data_service(
            &self.pipeline,
            &self.ingest_config,
            &p.content,
            &p.source,
            p.collection.as_deref(),
        )
        .await
        .to_string()
    }

    #[tool(name = "delete_file", description = "Remove a document and all its chunks by source ID.")]
    async fn delete_file(&self, params: Parameters<DeleteParams>) -> String {
        service::delete_service(&params.0.source).to_string()
    }

    #[tool(name = "list_files", description = "List all indexed documents with their metadata.")]
    async fn list_files(&self, _params: Parameters<NoParams>) -> String {
        service::list_sources_service(None).to_string()
    }

    #[tool(name = "status", description = "Get RAG engine status: document count.")]
    async fn status(&self, _params: Parameters<NoParams>) -> String {
        service::status_service().to_string()
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

        let report = engine::pre_check_document(&content, &filename, self.ingest_config.chunk_size);
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
        let limit = p.limit.unwrap_or(self.default_query_limit).clamp(1, self.max_query_limit);
        match engine::run_benchmark(&self.pipeline.embedder, entries, p.collection, limit).await {
            Ok(result) => serde_json::to_string(&result).unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }).to_string()),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(name = "read_chunk_neighbors", description = "Get chunks adjacent to a specific chunk for context expansion.")]
    async fn read_chunk_neighbors(&self, params: Parameters<ChunkNeighborsParams>) -> String {
        let p = params.0;
        service::neighbors_service(
            p.source_id,
            p.chunk_index,
            p.before.unwrap_or(2),
            p.after.unwrap_or(2),
        )
        .to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env from executable directory (automatic — no manual source needed)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let env_path = dir.join(".env");
            if env_path.exists() {
                let _ = dotenvy::from_path(&env_path);
            }
        }
    }

    let config = config::Config::load()?;

    // Log to file for debugging MCP issues
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.advanced.log_file)?;
    tracing_subscriber::fmt()
        .with_env_filter(&config.advanced.log_filter)
        .with_writer(std::sync::Mutex::new(log_file))
        .init();

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
        config.advanced.embedding_batch_size,
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
        // Set configurable LLM params
        provider.temperature = config.llm.temperature;
        provider.max_tokens = config.llm.max_tokens;
        provider.expansion_temperature = config.llm.expansion_temperature;
        provider.expansion_max_tokens = config.llm.expansion_max_tokens;
        provider.max_expansion_queries = config.llm.max_expansion_queries;
        provider.max_document_prompt_chars = config.llm.max_document_prompt_chars;
        provider.max_chunk_prompt_chars = config.llm.max_chunk_prompt_chars;

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
    let reranker_top_k = config.reranker.top_k;
    let reranker = match config.reranker.reranker_type.as_str() {
        "llm" => {
            if let Some(ref llm_provider) = llm {
                tracing::info!("Reranker: LLM (reusing main LLM provider, top_k={})", reranker_top_k);
                reranker::Reranker::new_llm(Arc::new(llm_provider.clone()), reranker_top_k, config.reranker.preview_chars)
            } else {
                tracing::warn!("Reranker: LLM requested but no LLM provider available, disabling");
                reranker::Reranker::disabled()
            }
        }
        "cohere" => {
            let key = config.reranker.api_key.clone()
                .expect("Cohere reranker requires reranker.api_key");
            let cohere_model = config.reranker.model.clone()
                .unwrap_or_else(|| "rerank-v3.5".to_string());
            let cohere_url = config.reranker.base_url.clone()
                .unwrap_or_else(|| "https://api.cohere.ai/v2/rerank".to_string());
            tracing::info!("Reranker: Cohere {} (top_k={})", cohere_model, reranker_top_k);
            reranker::Reranker::new_cohere(key, reranker_top_k, config.reranker.preview_chars, cohere_model, cohere_url)
        }
        _ => {
            tracing::info!("Reranker: disabled");
            reranker::Reranker::disabled()
        }
    };

    let ingest_config = params::IngestConfig {
            max_concurrent: config.llm.max_concurrent,
            relevance_scoring: config.llm.relevance_scoring,
            min_relevance_score: config.llm.min_relevance_score,
            chunk_size: config.advanced.chunk_size,
            context_batch_size: config.llm.context_batch_size,
            context_max_retries: config.llm.context_max_retries,
            chunk_overlap_ratio: config.advanced.chunk_overlap_ratio,
            merge_last_chunk_threshold: config.advanced.merge_last_chunk_threshold,
            chunking_strategy: config.chunking.strategy.clone(),
            parent_max_chars: config.chunking.parent_max_chars,
            child_max_chars: config.chunking.child_max_chars,
            child_overlap: config.chunking.child_overlap,
            auto_threshold: config.chunking.auto_threshold,
            child_min_chars: config.chunking.child_min_chars,
        };

    let reranker_for_fallback = reranker.clone();
    let pipeline = pipeline::QueryPipeline::new(
        embedder.clone(),
        llm.clone(),
        reranker,
        config.advanced.quality_threshold,
        config.advanced.max_retries as u32,
        config.advanced.cache_ttl_secs,
        config.advanced.cache_max_entries,
        config.advanced.high_confidence_threshold,
    );

    let server = RagFerriteServer {
        pipeline: pipeline.clone(),
        query_fallback_pipeline: config.query_fallback.as_ref().map(|fb| {
            tracing::info!("Query fallback LLM: {} / {} (used during ingestion)", fb.provider, fb.model);
            let fb_llm = llm::LlmProvider::new(
                fb.provider.clone(),
                fb.model.clone(),
                fb.api_key.clone(),
                fb.base_url.clone(),
            );
            pipeline::QueryPipeline::new(
                embedder.clone(),
                Some(fb_llm),
                reranker_for_fallback.clone(),
                config.advanced.quality_threshold,
                config.advanced.max_retries as u32,
                config.advanced.cache_ttl_secs,
                config.advanced.cache_max_entries,
                config.advanced.high_confidence_threshold,
            )
        }),
        ingestion_manager: ingestion::IngestionManager::new(pipeline, ingest_config.clone()),
        ingest_config,
        default_query_limit: config.advanced.default_query_limit,
        max_query_limit: config.advanced.max_query_limit,
    };

    let server = Arc::new(server);

    if config.http_port > 0 {
        let http_port = config.http_port;
        let http_bind = config.advanced.http_bind_address.clone();
        tracing::info!("Starting MCP Streamable HTTP on {}:{}", http_bind, http_port);
        api::serve(server, http_port, http_bind).await?;
    } else {
        tracing::info!("Starting MCP server on stdio...");
        let service = server.serve(rmcp::transport::io::stdio()).await?;
        service.waiting().await?;
    }

    Ok(())
}
