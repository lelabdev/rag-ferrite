use rmcp::{ServiceExt, handler::server::wrapper::Parameters, tool, tool_router};

use crate::params::*;
use crate::{engine, ingestion, llm, pipeline, service, types};

#[derive(Clone)]
pub(crate) struct RagFerriteServer {
    pub pipeline: pipeline::QueryPipeline,
    pub query_fallback_pipeline: Option<pipeline::QueryPipeline>,
    /// LLM provider dedicated to ingestion. All transports use ingestion_manager.
    pub ingestion_llm: Option<llm::LlmProvider>,
    pub ingest_config: IngestConfig,
    pub ingestion_manager: ingestion::IngestionManager,
    pub heat_tracker: engine::HeatTracker,
    pub chunk_heat_tracker: engine::ChunkHeatTracker,
    pub default_query_limit: usize,
    pub max_query_limit: usize,
    pub move_after_ingest: bool,
}

// --- MCP Tools ---

#[tool_router(server_handler)]
impl RagFerriteServer {
    #[tool(
        name = "query_documents",
        description = "Search documents using hybrid search (BM25 + vector with RRF fusion). Returns relevant chunks with scores. Tags use AND logic: 1 tag = broad results, 2 tags = precise intersection. Use 1-2 tags max."
    )]
    async fn query_documents(&self, params: Parameters<QueryParams>) -> String {
        let p = params.0;
        // Use fallback pipeline during active ingestion (if configured)
        let pipeline =
            if self.ingestion_manager.get_progress().status == ingestion::IngestStatus::Running {
                self.query_fallback_pipeline
                    .as_ref()
                    .unwrap_or(&self.pipeline)
            } else {
                &self.pipeline
            };
        service::query_service(
            pipeline,
            &p.query,
            p.limit
                .unwrap_or(self.default_query_limit)
                .clamp(1, self.max_query_limit),
            p.source_ids,
            p.metadata_like,
            p.tags,
            Some(&self.heat_tracker),
            Some(&self.chunk_heat_tracker),
        )
        .await
        .to_string()
    }

    #[tool(
        name = "ingest_file",
        description = "Parse and index a document file (PDF, TXT, MD) into the RAG."
    )]
    async fn ingest_file(&self, params: Parameters<IngestFileParams>) -> String {
        self.ingestion_manager
            .ingest_file(params.0.file_path)
            .to_string()
    }

    #[tool(
        name = "ingest_data",
        description = "Index content directly (text, HTML, or markdown) with a source identifier."
    )]
    async fn ingest_data(&self, params: Parameters<IngestDataParams>) -> String {
        let p = params.0;
        self.ingestion_manager
            .ingest_data(p.content, p.source)
            .to_string()
    }

    #[tool(
        name = "delete_file",
        description = "Remove a document and all its chunks by source ID."
    )]
    async fn delete_file(&self, params: Parameters<DeleteParams>) -> String {
        service::delete_service(&params.0.source).to_string()
    }

    #[tool(
        name = "list_files",
        description = "List all indexed documents with their metadata."
    )]
    async fn list_files(&self, _params: Parameters<NoParams>) -> String {
        service::list_sources_service().to_string()
    }

    #[tool(
        name = "status",
        description = "Get RAG engine status: document count."
    )]
    async fn status(&self, _params: Parameters<NoParams>) -> String {
        service::status_service().to_string()
    }

    #[tool(
        name = "check_ingestion",
        description = "Pre-ingestion quality check: analyze a document before indexing. Returns char count, estimated chunks, language, duplicate detection, and warnings."
    )]
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
                Err(e) => {
                    return serde_json::json!({ "error": format!("Failed to extract text: {}", e) })
                        .to_string();
                }
            }
        } else if let Some(ref content) = p.content {
            let name = p
                .source_name
                .unwrap_or_else(|| "inline_content".to_string());
            (content.clone(), name)
        } else {
            return serde_json::json!({ "error": "Provide either file_path or content" })
                .to_string();
        };

        let report = engine::pre_check_document(&content, &filename, self.ingest_config.chunk_size);
        serde_json::json!({ "pre_check": report }).to_string()
    }

    #[tool(
        name = "benchmark",
        description = "Evaluate retrieval quality against a versioned golden dataset JSON file. Supports the legacy array format and {version, entries}; returns Recall@k, precision, MRR, nDCG, empty-result rate, latency percentiles, and per-query details."
    )]
    async fn benchmark(&self, params: Parameters<BenchmarkParams>) -> String {
        let p = params.0;
        let content = match std::fs::read_to_string(&p.file_path) {
            Ok(c) => c,
            Err(e) => return serde_json::json!({ "error": format!("Failed to read golden dataset: {}", e) }).to_string(),
        };
        let dataset_value: serde_json::Value = match serde_json::from_str(&content) {
            Ok(value) => value,
            Err(e) => {
                return serde_json::json!({ "error": format!("Invalid JSON: {}", e) }).to_string();
            }
        };
        let (dataset_version, entries): (u32, Vec<types::GoldenEntry>) = if dataset_value.is_array()
        {
            match serde_json::from_value(dataset_value) {
                Ok(entries) => (1, entries),
                Err(e) => {
                    return serde_json::json!({ "error": format!("Invalid golden dataset: {}", e) })
                        .to_string();
                }
            }
        } else {
            match serde_json::from_value::<types::GoldenDataset>(dataset_value) {
                Ok(dataset) => (dataset.version, dataset.entries),
                Err(e) => {
                    return serde_json::json!({ "error": format!("Invalid golden dataset: {}", e) })
                        .to_string();
                }
            }
        };
        if entries.is_empty() {
            return serde_json::json!({ "error": "Golden dataset is empty" }).to_string();
        }
        let limit = p
            .limit
            .unwrap_or(self.default_query_limit)
            .clamp(1, self.max_query_limit);
        match engine::run_benchmark(
            &self.pipeline.embedder,
            dataset_version,
            entries,
            None,
            limit,
        )
        .await
        {
            Ok(result) => serde_json::to_string(&result)
                .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }).to_string()),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }

    #[tool(
        name = "read_chunk_neighbors",
        description = "Get chunks adjacent to a specific chunk for context expansion."
    )]
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

    #[tool(
        name = "collection_heat",
        description = "Get collection heat tracking data: which collections are queried most/freshly. Returns heat_score, last_queried_at, and query_count per collection."
    )]
    async fn collection_heat(&self, _params: Parameters<NoParams>) -> String {
        service::collection_heat_service().to_string()
    }

    #[tool(
        name = "chunk_qa",
        description = "Get chunk-level QA report: identify dead chunks (never queried) and cold chunks. Grouped by source with heat scores calculated on-the-fly. Useful for cleaning up noise."
    )]
    async fn chunk_qa(&self, _params: Parameters<NoParams>) -> String {
        service::chunk_qa_service().to_string()
    }

    #[tool(
        name = "suggest_collection",
        description = "Given a query, extract keywords and suggest the best-matching collection based on tag routing. Returns suggested collection, matched keywords, and all candidate collections with scores."
    )]
    async fn suggest_collection(&self, params: Parameters<SuggestCollectionParams>) -> String {
        let query = &params.0.query;
        service::suggest_collection_service(query).to_string()
    }

    #[tool(
        name = "tag_map",
        description = "Show the full tag → collection mapping with chunk counts. Useful to understand which tags belong to which collections."
    )]
    async fn tag_map(&self, _params: Parameters<NoParams>) -> String {
        service::tag_collection_map_service().to_string()
    }

    #[tool(
        name = "reassign_collection",
        description = "Move a source (document) and all its chunks to a different collection. Rebuilds HNSW + BM25 indexes for both old and new collections. Use this to organize documents into thematic collections."
    )]
    async fn reassign_collection(&self, params: Parameters<ReassignCollectionParams>) -> String {
        service::reassign_collection_service(params.0.source_id, &params.0.collection).to_string()
    }

    #[tool(
        name = "rebuild_indexes",
        description = "Rebuild HNSW + BM25 indexes for the general collection and run a WAL checkpoint. Use after bulk deletes or if search quality seems degraded."
    )]
    async fn rebuild_indexes(&self, _params: Parameters<NoParams>) -> String {
        self.ingestion_manager.rebuild_indexes().to_string()
    }

    #[tool(
        name = "flush_indexes",
        description = "Flush the incremental HNSW buffer to disk. Makes recently ingested chunks fully persistent and searchable."
    )]
    async fn flush_indexes(&self, _params: Parameters<NoParams>) -> String {
        let val = self.ingestion_manager.flush_indexes();
        val.to_string()
    }
}
