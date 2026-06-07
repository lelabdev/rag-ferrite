use anyhow::Result;
use std::sync::Arc;
use std::fmt::Write;
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    tool, tool_router,
};

/// Custom log timer using local timezone (Europe/Paris).
struct LocalTimer;

impl tracing_subscriber::fmt::time::FormatTime for LocalTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(w, "{}", now.format("%Y-%m-%dT%H:%M:%S%.3f%:z"))
    }
}

mod api;
mod config;
mod chunker;
mod embedding;
mod engine;
mod extractor;
mod ingestion;
mod llm;
mod monitor;
mod params;
mod pipeline;
mod tag_rules;
mod reranker;
mod service;
mod types;

use params::*;

#[derive(Clone)]
struct RagFerriteServer {
    pub pipeline: pipeline::QueryPipeline,
    pub query_fallback_pipeline: Option<pipeline::QueryPipeline>,
    /// LLM provider dedicated to ingestion (MCP tool calls that bypass the queue).
    pub ingestion_llm: Option<llm::LlmProvider>,
    pub ingest_config: params::IngestConfig,
    pub ingestion_manager: ingestion::IngestionManager,
    pub heat_tracker: engine::HeatTracker,
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
            p.tags,
            Some(&self.heat_tracker),
        )
        .await
        .to_string()
    }

    #[tool(name = "ingest_file", description = "Parse and index a document file (PDF, TXT, MD) into the RAG.")]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        service::ingest_file_service(
            &self.pipeline,
            &self.ingest_config,
            self.ingestion_llm.as_ref(),
            &params.0.file_path,
        )
        .await
        .to_string()
    }

    #[tool(name = "ingest_data", description = "Index content directly (text, HTML, or markdown) with a source identifier.")]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        service::ingest_data_service(
            &self.pipeline,
            &self.ingest_config,
            self.ingestion_llm.as_ref(),
            &p.content,
            &p.source,
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
        service::list_sources_service().to_string()
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
        match engine::run_benchmark(&self.pipeline.embedder, entries, None, limit).await {
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

    #[tool(name = "collection_heat", description = "Get collection heat tracking data: which collections are queried most/freshly. Returns heat_score, last_queried_at, and query_count per collection.")]
    async fn collection_heat(&self, _params: Parameters<NoParams>) -> String {
        service::collection_heat_service().to_string()
    }
}

#[tokio::main(worker_threads = 12)]
async fn main() -> Result<()> {
    // Check for monitor subcommand
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "monitor" {
        monitor::run(&args[2..]);
        return Ok(());
    }

    // Check for update subcommand
    if args.len() > 1 && args[1] == "update" {
        // Find update.sh next to the binary
        let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let script = exe.parent()
            .unwrap_or(std::path::Path::new("."))
            .join("update.sh");

        if !script.exists() {
            eprintln!("Error: update.sh not found at {}", script.display());
            eprintln!("Expected next to the binary in the same directory.");
            std::process::exit(1);
        }

        println!("Running update.sh...");
        let status = std::process::Command::new("bash")
            .arg(&script)
            .args(&args[2..])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("✓ Update complete");
                std::process::exit(0);
            }
            Ok(s) => {
                eprintln!("✗ Update failed (exit code {})", s.code().unwrap_or(-1));
                std::process::exit(s.code().unwrap_or(1));
            }
            Err(e) => {
                eprintln!("✗ Failed to run update.sh: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Load .env from executable directory (automatic — no manual source needed)
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent() {
            let env_path = dir.join(".env");
            if env_path.exists() {
                let _ = dotenvy::from_path(&env_path);
            }
        }

    let config = config::Config::load()?;
    let tag_rules = tag_rules::TagRules::load()?;
    llm::init_tag_rules(tag_rules);

    // Log to file for debugging MCP issues
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.advanced.log_file)?;
    tracing_subscriber::fmt()
        .with_env_filter(&config.advanced.log_filter)
        .with_writer(std::sync::Mutex::new(log_file))
        .with_timer(LocalTimer)
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

    // Init LLM providers — profile-based or legacy single provider.
    // When [[llm_profile]] entries exist and action profiles are set, create
    // separate LlmProvider instances for ingestion, query, and reranking.
    // Otherwise, fall back to the legacy single [llm] provider for everything.
    let has_profiles = !config.llm_profile.is_empty();

    // Helper: apply configurable LLM params from LlmConfig to a provider.
    let apply_llm_params = |provider: &mut llm::LlmProvider, cfg: &config::LlmConfig| {
        provider.temperature = cfg.temperature;
        provider.max_tokens = cfg.max_tokens;
        provider.expansion_temperature = cfg.expansion_temperature;
        provider.expansion_max_tokens = cfg.expansion_max_tokens;
        provider.max_expansion_queries = cfg.max_expansion_queries;
        provider.max_document_prompt_chars = cfg.max_document_prompt_chars;
        provider.max_chunk_prompt_chars = cfg.max_chunk_prompt_chars;
    };

    // Helper: attach fallback from config if present.
    let apply_fallback = |provider: llm::LlmProvider, cfg: &config::LlmConfig| -> llm::LlmProvider {
        if let Some(ref fb) = cfg.fallback {
            let fb_provider = llm::LlmProvider::new_fallback(
                fb.provider.clone(),
                fb.model.clone(),
                fb.api_key.clone(),
                fb.base_url.clone(),
            );
            provider.with_fallback(fb_provider)
        } else {
            provider
        }
    };

    // --- Resolve ingestion LLM ---
    let ingestion_llm: Option<llm::LlmProvider> = if config.llm.context_enabled {
        if let Some(ref profile_name) = config.llm.ingestion_profile {
            if let Some(profile) = config.get_profile(profile_name) {
                tracing::info!("Ingestion LLM: profile '{}' ({} / {})", profile.name, profile.provider, profile.model);
                let mut provider = llm::LlmProvider::from_profile(profile);
                apply_llm_params(&mut provider, &config.llm);
                Some(provider)
            } else {
                tracing::warn!("ingestion_profile '{}' not found, using legacy config", profile_name);
                let mut provider = llm::LlmProvider::new(
                    config.llm.provider.clone(),
                    config.llm.model.clone(),
                    config.llm.api_key.clone(),
                    config.llm.base_url.clone(),
                );
                apply_llm_params(&mut provider, &config.llm);
                provider = apply_fallback(provider, &config.llm);
                Some(provider)
            }
        } else {
            // Legacy: use single [llm] config
            let mut provider = llm::LlmProvider::new(
                config.llm.provider.clone(),
                config.llm.model.clone(),
                config.llm.api_key.clone(),
                config.llm.base_url.clone(),
            );
            apply_llm_params(&mut provider, &config.llm);
            provider = apply_fallback(provider, &config.llm);
            if has_profiles {
                tracing::info!("Ingestion LLM: using legacy config ({} / {})", config.llm.provider, config.llm.model);
            }
            Some(provider)
        }
    } else {
        tracing::info!("Contextual retrieval disabled");
        None
    };

    // --- Resolve query LLM ---
    let query_llm: Option<llm::LlmProvider> = if let Some(ref profile_name) = config.llm.query_profile {
        if let Some(profile) = config.get_profile(profile_name) {
            tracing::info!("Query LLM: profile '{}' ({} / {})", profile.name, profile.provider, profile.model);
            let mut provider = llm::LlmProvider::from_profile(profile);
            apply_llm_params(&mut provider, &config.llm);
            Some(provider)
        } else {
            tracing::warn!("query_profile '{}' not found, falling back to ingestion LLM", profile_name);
            ingestion_llm.clone()
        }
    } else {
        // Legacy: same as ingestion LLM
        ingestion_llm.clone()
    };

    // Log legacy mode
    if !has_profiles && config.llm.context_enabled {
        tracing::info!("LLM provider: {} / {} (contextual retrieval enabled)", config.llm.provider, config.llm.model);
    }

    // Build reranker from LLM provider
    let reranker_top_k = config.reranker.top_k;
    let reranker = match config.reranker.reranker_type.as_str() {
        "llm" => {
            // Resolve reranker LLM: dedicated profile > query LLM > ingestion LLM
            let reranker_llm = if let Some(ref profile_name) = config.llm.reranker_profile {
                if let Some(profile) = config.get_profile(profile_name) {
                    tracing::info!("Reranker LLM: profile '{}' ({} / {})", profile.name, profile.provider, profile.model);
                    Some(Arc::new(llm::LlmProvider::from_profile(profile)))
                } else {
                    tracing::warn!("reranker_profile '{}' not found, falling back", profile_name);
                    query_llm.as_ref().map(|l| Arc::new(l.clone()))
                }
            } else {
                // Legacy: reuse query LLM (or ingestion LLM)
                query_llm.as_ref().map(|l| Arc::new(l.clone()))
            };

            if let Some(provider) = reranker_llm {
                tracing::info!("Reranker: LLM (top_k={})", reranker_top_k);
                reranker::Reranker::new_llm(provider, reranker_top_k, config.reranker.preview_chars)
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
            defer_index_rebuild: config.advanced.defer_index_rebuild,
            wal_checkpoint_interval: config.advanced.wal_checkpoint_interval,
            ingested_dir: config.advanced.ingested_dir.clone(),
        };

    let reranker_for_fallback = reranker.clone();
    let pipeline = pipeline::QueryPipeline::new(
        embedder.clone(),
        query_llm.clone(),
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
            let fb_llm = llm::LlmProvider::new_query_fallback(
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
        ingestion_llm: ingestion_llm.clone(),
        ingestion_manager: ingestion::IngestionManager::new(pipeline, ingest_config.clone(), ingestion_llm),
        ingest_config,
        heat_tracker: engine::HeatTracker::new(),
        default_query_limit: config.advanced.default_query_limit,
        max_query_limit: config.advanced.max_query_limit,
    };

    let server = Arc::new(server);

    if config.http_port > 0 {
        let http_port = config.http_port;
        let http_bind = config.advanced.http_bind_address.clone();
        // Read API keys from environment variables
        let admin_key = std::env::var("RAG_API_KEY").ok().filter(|k| !k.is_empty());
        let guest_key = std::env::var("RAG_GUEST_API_KEY").ok().filter(|k| !k.is_empty());
        if admin_key.is_some() {
            tracing::info!("Admin API key enabled (RAG_API_KEY)");
        }
        if guest_key.is_some() {
            tracing::info!("Guest API key enabled (RAG_GUEST_API_KEY — read-only)");
        }
        if admin_key.is_none() && guest_key.is_none() {
            tracing::info!("API key authentication disabled (no keys — local dev)");
        }
        tracing::info!("Starting MCP Streamable HTTP on {}:{}", http_bind, http_port);
        api::serve(server, http_port, http_bind, admin_key, guest_key).await?;
    } else {
        tracing::info!("Starting MCP server on stdio...");
        let service = server.serve(rmcp::transport::io::stdio()).await?;
        service.waiting().await?;
    }

    Ok(())
}
