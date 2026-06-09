use anyhow::Result;
use std::sync::Arc;
use rmcp::ServiceExt;

/// Custom log timer using local timezone (Europe/Paris).
struct LocalTimer;

impl tracing_subscriber::fmt::time::FormatTime for LocalTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(w, "{}", now.format("%Y-%m-%dT%H:%M:%S%.3f%:z"))
    }
}

mod api;
mod client;
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
mod server;
mod tag_rules;
mod reranker;
mod service;
mod types;

use params::*;
pub(crate) use server::RagFerriteServer;

#[tokio::main(worker_threads = 12)]
async fn main() -> Result<()> {
    // Parse CLI args first — client commands exit early, server commands proceed.
    let cli_args = client::parse_args()?;

    match cli_args.command {
        // Server-mode commands handled here
        client::CliCommand::Serve => { /* proceed to server init below */ }
        client::CliCommand::Monitor => {
            monitor::run(&[]);
            return Ok(());
        }
        client::CliCommand::Update => {
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
        // Client-mode commands — hit API and exit
        _ => {
            if let Err(e) = client::execute(cli_args) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            return Ok(());
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
    // Store config globally before it's consumed by server init
    let heat_config = config.heat.clone();
    config::set_global_heat(heat_config);
    let tag_rules = tag_rules::TagRules::load()?;
    tag_rules::init_tag_rules(tag_rules);

    // Log to file AND stderr so systemd journal captures errors
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.advanced.log_file)?;
    use tracing_subscriber::fmt::writer::Tee;
    let file_writer = std::sync::Mutex::new(log_file);
    let stderr_writer = std::io::stderr;
    tracing_subscriber::fmt()
        .with_env_filter(&config.advanced.log_filter)
        .with_writer(Tee::new(file_writer, stderr_writer))
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

    // --- Health check: verify ingestion LLM is reachable ---
    if let Some(ref llm) = ingestion_llm {
        tracing::info!("LLM health check: testing ingestion LLM connection...");
        let test_messages = vec![llm::ChatMessage { role: "user".into(), content: "ping".into() }];
        match llm.chat(test_messages).await {
            Ok(_) => tracing::info!("LLM health check: OK"),
            Err(e) => tracing::error!("LLM health check FAILED — ingestion will not work: {}", e),
        }
    }

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
            min_relevance_score: config.llm.min_relevance_score as f64,
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
        config.query_classification.clone(),
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
                config.query_classification.clone(),
            )
        }),
        ingestion_llm: ingestion_llm.clone(),
        ingestion_manager: ingestion::IngestionManager::new(pipeline, ingest_config.clone(), ingestion_llm),
        ingest_config,
        heat_tracker: engine::HeatTracker::new(),
        chunk_heat_tracker: engine::ChunkHeatTracker::new(),
        default_query_limit: config.advanced.default_query_limit,
        max_query_limit: config.advanced.max_query_limit,
        move_after_ingest: config.advanced.move_after_ingest,
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
