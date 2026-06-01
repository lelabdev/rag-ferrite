//! Ingestion queue — processes ingestions in the background without blocking HTTP.
//!
//! Problem: chunking large files (1M+ chars) is CPU-bound and blocks the tokio
//! runtime, making the server unresponsive to any request during ingestion.
//!
//! Solution: ingestion requests are pushed to an mpsc channel. A background task
//! processes them one at a time using `spawn_blocking` for CPU-heavy work. The
//! HTTP handlers return immediately with a job ID, and progress can be queried
//! via GET /api/ingest/progress.

use crate::engine;
use crate::llm::LlmProvider;
use crate::params::IngestConfig;
use crate::pipeline::QueryPipeline;
use serde::Serialize;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::mpsc;

// ── Job types ──────────────────────────────────────────────────────────

#[derive(Debug)]
enum IngestJob {
    File {
        file_path: String,
    },
    Data {
        content: String,
        source: String,
    },
}

// ── Progress tracking ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Default)]
pub struct IngestProgress {
    /// Currently ingesting (source name)
    pub current_source: Option<String>,
    /// Parent progress: X done out of Y total
    pub parents_done: usize,
    pub parents_total: usize,
    /// Overall job status
    pub status: IngestStatus,
    /// Last completed source
    pub last_completed: Option<String>,
    /// Last error
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum IngestStatus {
    #[default]
    Idle,
    Running,
}


// ── Manager ────────────────────────────────────────────────────────────

pub struct IngestionManager {
    pub progress: Arc<Mutex<IngestProgress>>,
    sender: mpsc::UnboundedSender<IngestJob>,
    /// LLM provider dedicated to ingestion (contextual retrieval, scoring, tagging).
    /// Separate from the query pipeline's LLM so different profiles can be used.
    ingestion_llm: Option<LlmProvider>,
}

impl Clone for IngestionManager {
    fn clone(&self) -> Self {
        IngestionManager {
            progress: self.progress.clone(),
            sender: self.sender.clone(),
            ingestion_llm: self.ingestion_llm.clone(),
        }
    }
}

impl IngestionManager {
    /// Create a new ingestion manager and spawn the background worker.
    /// `ingestion_llm` is the LLM provider dedicated to ingestion tasks.
    pub fn new(
        pipeline: QueryPipeline,
        ingest_config: IngestConfig,
        ingestion_llm: Option<LlmProvider>,
    ) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let progress = Arc::new(Mutex::new(IngestProgress::default()));

        let worker_progress = progress.clone();
        let worker_llm = ingestion_llm.clone();
        tokio::spawn(async move {
            background_worker(receiver, pipeline, ingest_config, worker_progress, worker_llm).await;
        });

        IngestionManager { progress, sender, ingestion_llm }
    }

    /// Queue a file ingestion. Returns immediately.
    pub fn ingest_file(&self, file_path: String) -> serde_json::Value {
        match self.sender.send(IngestJob::File { file_path: file_path.clone() }) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "file_path": file_path,
                "message": "Ingestion queued. Check GET /api/ingest/progress for status."
            }),
            Err(_) => serde_json::json!({ "error": "Failed to queue ingestion" }),
        }
    }

    /// Queue a data ingestion. Returns immediately.
    pub fn ingest_data(&self, content: String, source: String) -> serde_json::Value {
        match self.sender.send(IngestJob::Data { content, source: source.clone() }) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "source": source,
                "message": "Ingestion queued. Check GET /api/ingest/progress for status."
            }),
            Err(_) => serde_json::json!({ "error": "Failed to queue ingestion" }),
        }
    }

    /// Get current progress.
    pub fn get_progress(&self) -> IngestProgress {
        self.progress.lock().unwrap().clone()
    }
}

// ── Background worker ──────────────────────────────────────────────────

async fn background_worker(
    mut receiver: mpsc::UnboundedReceiver<IngestJob>,
    pipeline: QueryPipeline,
    ingest_config: IngestConfig,
    progress: Arc<Mutex<IngestProgress>>,
    ingestion_llm: Option<LlmProvider>,
) {
    while let Some(job) = receiver.recv().await {
        match job {
            IngestJob::File { file_path } => {
                process_file_job(&pipeline, &ingest_config, &progress, &ingestion_llm, &file_path).await;
            }
            IngestJob::Data { content, source } => {
                process_data_job(&pipeline, &ingest_config, &progress, &ingestion_llm, &content, &source).await;
            }
        }
    }

    // Channel closed — all senders dropped. Reset progress to Idle so the
    // API never reports a stale "Running" status after shutdown.
    {
        let mut p = progress.lock().unwrap();
        p.status = IngestStatus::Idle;
        p.current_source = None;
    }

    tracing::info!("Ingestion worker shut down.");
}

async fn process_file_job(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    progress: &Arc<Mutex<IngestProgress>>,
    ingestion_llm: &Option<LlmProvider>,
    file_path: &str,
) {
    {
        let mut p = progress.lock().unwrap();
        p.status = IngestStatus::Running;
        p.current_source = Some(file_path.to_string());
        p.parents_done = 0;
        p.parents_total = 0;
        p.last_error = None;
    }

    tracing::info!("Ingestion queue: processing file {}", file_path);

    // Use ingestion_llm if available, otherwise fall back to pipeline.llm
    let llm = ingestion_llm.as_ref().or(pipeline.llm.as_ref());
    let result = engine::ingest_file(
        &pipeline.embedder,
        llm,
        file_path,
        Some("general"),
        cfg.to_engine_options(),
    )
    .await;

    let mut p = progress.lock().unwrap();
    match result {
        Ok((_id, report)) => {
            tracing::info!("Ingestion complete: {} — {} chunks", file_path, report.total_chunks);
            p.parents_total = report.total_chunks;
            p.parents_done = report.total_chunks;
            p.last_completed = Some(file_path.to_string());
            p.last_error = None;
        }
        Err(e) => {
            tracing::error!("Ingestion failed: {} — {}", file_path, e);
            p.last_error = Some(e.to_string());
        }
    }
    p.status = IngestStatus::Idle;
    p.current_source = None;
}

async fn process_data_job(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    progress: &Arc<Mutex<IngestProgress>>,
    ingestion_llm: &Option<LlmProvider>,
    content: &str,
    source: &str,
) {
    {
        let mut p = progress.lock().unwrap();
        p.status = IngestStatus::Running;
        p.current_source = Some(source.to_string());
        p.parents_done = 0;
        p.parents_total = 0;
        p.last_error = None;
    }

    tracing::info!("Ingestion queue: processing data source {}", source);

    // Use ingestion_llm if available, otherwise fall back to pipeline.llm
    let llm = ingestion_llm.as_ref().or(pipeline.llm.as_ref());
    let result = engine::ingest_text(
        &pipeline.embedder,
        llm,
        content,
        source,
        None,
        Some("general"),
        cfg.to_engine_options(),
    )
    .await;

    let mut p = progress.lock().unwrap();
    match result {
        Ok((_id, report)) => {
            tracing::info!("Ingestion complete: {} — {} chunks", source, report.total_chunks);
            p.parents_total = report.total_chunks;
            p.parents_done = report.total_chunks;
            p.last_completed = Some(source.to_string());
            p.last_error = None;
        }
        Err(e) => {
            tracing::error!("Ingestion failed: {} — {}", source, e);
            p.last_error = Some(e.to_string());
        }
    }
    p.status = IngestStatus::Idle;
    p.current_source = None;
}
