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
        collection: Option<String>,
    },
    Data {
        content: String,
        source: String,
        collection: Option<String>,
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
pub enum IngestStatus {
    Idle,
    Running,
}

impl Default for IngestStatus {
    fn default() -> Self {
        IngestStatus::Idle
    }
}

// ── Manager ────────────────────────────────────────────────────────────

pub struct IngestionManager {
    pub progress: Arc<Mutex<IngestProgress>>,
    sender: mpsc::UnboundedSender<IngestJob>,
}

impl Clone for IngestionManager {
    fn clone(&self) -> Self {
        IngestionManager {
            progress: self.progress.clone(),
            sender: self.sender.clone(),
        }
    }
}

impl IngestionManager {
    /// Create a new ingestion manager and spawn the background worker.
    pub fn new(pipeline: QueryPipeline, ingest_config: IngestConfig) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let progress = Arc::new(Mutex::new(IngestProgress::default()));

        let worker_progress = progress.clone();
        tokio::spawn(async move {
            background_worker(receiver, pipeline, ingest_config, worker_progress).await;
        });

        IngestionManager { progress, sender }
    }

    /// Queue a file ingestion. Returns immediately.
    pub fn ingest_file(&self, file_path: String, collection: Option<String>) -> serde_json::Value {
        match self.sender.send(IngestJob::File { file_path: file_path.clone(), collection: collection.clone() }) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "file_path": file_path,
                "collection": collection,
                "message": "Ingestion queued. Check GET /api/ingest/progress for status."
            }),
            Err(_) => serde_json::json!({ "error": "Failed to queue ingestion" }),
        }
    }

    /// Queue a data ingestion. Returns immediately.
    pub fn ingest_data(&self, content: String, source: String, collection: Option<String>) -> serde_json::Value {
        match self.sender.send(IngestJob::Data { content, source: source.clone(), collection: collection.clone() }) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "source": source,
                "collection": collection,
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
) {
    while let Some(job) = receiver.recv().await {
        match job {
            IngestJob::File { file_path, collection } => {
                process_file_job(&pipeline, &ingest_config, &progress, &file_path, collection.as_deref()).await;
            }
            IngestJob::Data { content, source, collection } => {
                process_data_job(&pipeline, &ingest_config, &progress, &content, &source, collection.as_deref()).await;
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
    file_path: &str,
    collection: Option<&str>,
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

    let result = engine::ingest_file(
        &pipeline.embedder,
        pipeline.llm.as_ref(),
        file_path,
        collection,
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
    content: &str,
    source: &str,
    collection: Option<&str>,
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

    let result = engine::ingest_text(
        &pipeline.embedder,
        pipeline.llm.as_ref(),
        content,
        source,
        None,
        collection,
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
