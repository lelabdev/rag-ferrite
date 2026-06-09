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
    /// Batch: multiple files with a shared batch_id
    Batch {
        batch_id: String,
        files: Vec<String>,
        move_after_ingest: bool,
    },
    /// Flush indexes: rebuild HNSW + BM25 + WAL checkpoint.
    /// Sent at the end of a batch ingestion to defer expensive index operations.
    FlushIndexes,
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
    /// Active batch info (if running a batch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchProgress>,
    /// Recent activity events (ring buffer, last 20)
    #[serde(default)]
    pub activity_log: engine::activity_log::ActivityLog,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum IngestStatus {
    #[default]
    Idle,
    Running,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BatchProgress {
    pub batch_id: String,
    pub status: BatchStatus,
    pub total_files: usize,
    pub completed_files: usize,
    pub failed_files: usize,
    pub current_file: Option<CurrentFileProgress>,
    pub total_chunks: usize,
    pub completed_chunks: usize,
    pub errors: Vec<BatchError>,
    pub started_at: u64,
    pub elapsed_seconds: u64,
    pub speed_chunks_per_min: f64,
    pub eta_seconds: u64,
    /// Total bytes of files processed
    pub total_size_mb: f64,
    /// Average time per file in seconds
    pub avg_time_per_file_seconds: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Estimated total chunks (based on file sizes at batch start)
    #[serde(default)]
    pub total_estimated_chunks: usize,
    /// Completed files with details
    pub files: Vec<FileResult>,
    /// Pending files (not yet processed)
    #[serde(default)]
    pub pending_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct FileResult {
    pub name: String,
    pub chunks: usize,
    pub size_mb: f64,
    pub duration_seconds: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum BatchStatus {
    #[default]
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct CurrentFileProgress {
    pub name: String,
    pub chunks_done: usize,
    pub chunks_total: usize,
    pub phase: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchError {
    pub file: String,
    pub error: String,
}


// ── Manager ────────────────────────────────────────────────────────────

pub struct IngestionManager {
    pub progress: Arc<Mutex<IngestProgress>>,
    sender: mpsc::UnboundedSender<IngestJob>,
    /// LLM provider dedicated to ingestion (contextual retrieval, scoring, tagging).
    /// Separate from the query pipeline's LLM so different profiles can be used.
    pub ingestion_llm: Option<LlmProvider>,
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
        let worker_sender = sender.clone();
        tokio::spawn(async move {
            background_worker(receiver, pipeline, ingest_config, worker_progress, worker_llm, worker_sender).await;
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

    /// Queue a batch of files. Returns immediately with batch_id.
    pub fn ingest_batch(&self, files: Vec<String>, move_after_ingest: bool) -> serde_json::Value {
        let batch_id = format!("batch-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis());
        let total = files.len();
        match self.sender.send(IngestJob::Batch {
            batch_id: batch_id.clone(),
            files,
            move_after_ingest,
        }) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "batch_id": batch_id,
                "total_files": total,
                "message": "Batch queued. Check GET /api/ingest/batch/{batch_id}/progress for status."
            }),
            Err(_) => serde_json::json!({ "error": "Failed to queue batch" }),
        }
    }

    /// Get current progress.
    pub fn get_progress(&self) -> IngestProgress {
        let mut p = self.progress.lock().unwrap().clone();
        // Inject real-time chunk count from the global counter (incremented by engine per parent commit)
        let live_chunks = engine::chunk_counter::get();
        if let Some(ref mut batch) = p.batch {
            batch.completed_chunks = live_chunks;
            // Recalculate elapsed from started_at (live, not stored)
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            batch.elapsed_seconds = now_secs.saturating_sub(batch.started_at);
            // Recalculate speed and ETA with live data
            if batch.elapsed_seconds > 0 {
                batch.speed_chunks_per_min = (live_chunks as f64 / batch.elapsed_seconds as f64) * 60.0;
                if batch.speed_chunks_per_min > 0.0 && batch.total_estimated_chunks > 0 {
                    let remaining = batch.total_estimated_chunks.saturating_sub(live_chunks);
                    batch.eta_seconds = (remaining as f64 / batch.speed_chunks_per_min * 60.0) as u64;
                }
            }
            // Update current file progress
            if let Some(ref mut cf) = batch.current_file {
                let chunks_before_file = batch.total_chunks;
                cf.chunks_done = live_chunks.saturating_sub(chunks_before_file);
            }
        }
        // Inject activity log snapshot
        p.activity_log = engine::activity_log::snapshot();
        p
    }

    /// Queue a flush: rebuild HNSW + BM25 indexes + WAL checkpoint.
    /// Call after a batch of ingestion jobs to finalize indexes.
    pub fn flush_indexes(&self) -> serde_json::Value {
        match self.sender.send(IngestJob::FlushIndexes) {
            Ok(()) => serde_json::json!({
                "status": "queued",
                "message": "Index rebuild + WAL checkpoint queued."
            }),
            Err(_) => serde_json::json!({ "error": "Failed to queue flush" }),
        }
    }

    /// Cancel the running batch. The worker will stop after the current file.
    pub fn cancel_batch(&self) -> serde_json::Value {
        engine::cancel::request();
        serde_json::json!({
            "status": "cancelling",
            "message": "Batch cancellation requested. Worker will stop after current file."
        })
    }
}

// ── Background worker ──────────────────────────────────────────────────

async fn background_worker(
    mut receiver: mpsc::UnboundedReceiver<IngestJob>,
    pipeline: QueryPipeline,
    ingest_config: IngestConfig,
    progress: Arc<Mutex<IngestProgress>>,
    ingestion_llm: Option<LlmProvider>,
    sender: mpsc::UnboundedSender<IngestJob>,
) {
    while let Some(job) = receiver.recv().await {
        match job {
            IngestJob::File { file_path } => {
                process_file_job(&pipeline, &ingest_config, &progress, &ingestion_llm, &file_path).await;
                // Auto-flush index after single-file ingestion so content is immediately searchable
                let _ = sender.send(IngestJob::FlushIndexes);
            }
            IngestJob::Data { content, source } => {
                process_data_job(&pipeline, &ingest_config, &progress, &ingestion_llm, &content, &source).await;
                let _ = sender.send(IngestJob::FlushIndexes);
            }
            IngestJob::Batch { batch_id, files, move_after_ingest } => {
                process_batch_job(&pipeline, &ingest_config, &progress, &ingestion_llm, &batch_id, &files, move_after_ingest).await;
            }
            IngestJob::FlushIndexes => {
                tracing::info!("FlushIndexes: rebuilding indexes + WAL checkpoint...");
                engine::rebuild_and_save_indexes("general");
                engine::wal_checkpoint();
                tracing::info!("FlushIndexes complete.");
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
        cfg.clone(),
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
        cfg.clone(),
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

async fn process_batch_job(
    pipeline: &QueryPipeline,
    cfg: &IngestConfig,
    progress: &Arc<Mutex<IngestProgress>>,
    ingestion_llm: &Option<LlmProvider>,
    batch_id: &str,
    files: &[String],
    move_after_ingest: bool,
) {
    let total_files = files.len();
    let batch_start = std::time::Instant::now();
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Reset global chunk counter for this batch
    engine::chunk_counter::reset();

    // Reset activity log for this batch
    engine::activity_log::clear();

    // Estimate total chunks based on file sizes (bytes / 800 ≈ chunk count)
    // This gives a realistic ETA instead of avg_time_per_file which is skewed
    // by mixing large books and small articles.
    const CHUNK_SIZE: usize = 800;
    let total_estimated_chunks: usize = files.iter().map(|f| {
        std::fs::metadata(f).map(|m| {
            let bytes = m.len() as usize;
            if bytes == 0 { 0 } else { (bytes / CHUNK_SIZE).max(1) }
        }).unwrap_or(0)
    }).sum();

    tracing::info!("Batch {} started: {} files, ~{} estimated chunks", batch_id, total_files, total_estimated_chunks);

    {
        let mut p = progress.lock().unwrap();
        p.status = IngestStatus::Running;
        p.batch = Some(BatchProgress {
            batch_id: batch_id.to_string(),
            status: BatchStatus::Running,
            total_files,
            completed_files: 0,
            failed_files: 0,
            current_file: None,
            total_chunks: 0,
            completed_chunks: 0,
            errors: Vec::new(),
            started_at,
            elapsed_seconds: 0,
            speed_chunks_per_min: 0.0,
            eta_seconds: 0,
            total_size_mb: 0.0,
            avg_time_per_file_seconds: 0.0,
            error_rate: 0.0,
            total_estimated_chunks,
            files: Vec::new(),
            pending_files: files.iter().map(|f| {
                std::path::Path::new(f)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| f.clone())
            }).collect(),
        });
    }

    let llm = ingestion_llm.as_ref().or(pipeline.llm.as_ref());

    for (i, file_path) in files.iter().enumerate() {
        // ── Check cancellation between files ──
        if engine::cancel::check_and_reset() {
            tracing::info!("Batch {} cancelled by user after {}/{} files", batch_id, i, total_files);
            let mut p = progress.lock().unwrap();
            if let Some(ref mut b) = p.batch {
                b.status = BatchStatus::Cancelled;
            }
            p.status = IngestStatus::Idle;
            return;
        }

        let file_name = std::path::Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.clone());

        // ── Dedup: skip if already ingested ──
        if engine::check_duplicate_source(&file_name) {
            tracing::info!("Batch {} file {}/{}: SKIP {} (already ingested)", batch_id, i + 1, total_files, file_name);
            let mut p = progress.lock().unwrap();
            if let Some(ref mut b) = p.batch {
                b.completed_files += 1;
                b.pending_files.retain(|f| f != &file_name);
                let total_done = b.completed_files + b.failed_files;
                if total_done > 0 {
                    b.avg_time_per_file_seconds = b.elapsed_seconds as f64 / total_done as f64;
                }
                if b.elapsed_seconds > 0 && b.speed_chunks_per_min > 0.0 && b.total_estimated_chunks > 0 {
                    let remaining_chunks = b.total_estimated_chunks.saturating_sub(b.completed_chunks);
                    b.eta_seconds = (remaining_chunks as f64 / b.speed_chunks_per_min * 60.0) as u64;
                }
            }
            continue;
        }

        // Get file size
        let file_size_mb = std::fs::metadata(file_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);

        let file_start = std::time::Instant::now();

        {
            let mut p = progress.lock().unwrap();
            if let Some(ref mut b) = p.batch {
                b.current_file = Some(CurrentFileProgress {
                    name: file_name.clone(),
                    chunks_done: 0,
                    chunks_total: 0,
                    phase: "parsing".to_string(),
                });
            }
            p.current_source = Some(file_path.clone());
            p.last_error = None;
        }

        tracing::info!("Batch {} file {}/{}: {} ({:.1} MB)", batch_id, i + 1, total_files, file_name, file_size_mb);

        // Phase update: embedding
        {
            let mut p = progress.lock().unwrap();
            if let Some(ref mut b) = p.batch {
                if let Some(ref mut cf) = b.current_file {
                    cf.phase = "embedding+llm".to_string();
                }
            }
        }

        let result = engine::ingest_file(
            &pipeline.embedder,
            llm,
            file_path,
            Some("general"),
            cfg.clone(),
        )
        .await;

        let file_duration = file_start.elapsed().as_secs_f64();
        let done = result.is_ok();
        let _chunks = result.as_ref().map(|(_, r)| r.total_chunks).unwrap_or(0);

        {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut p = progress.lock().unwrap();
            if let Some(ref mut b) = p.batch {
                b.elapsed_seconds = now_secs.saturating_sub(started_at);
                b.total_size_mb += file_size_mb;
            }
            match result {
                Ok((_id, report)) => {
                    tracing::info!("Batch {} file {}/{} done: {} — {} chunks in {:.1}s", batch_id, i + 1, total_files, file_name, report.total_chunks, file_duration);
                    if let Some(ref mut b) = p.batch {
                        b.completed_files += 1;
                        b.completed_chunks += report.total_chunks;
                        b.total_chunks += report.total_chunks;
                        b.files.push(FileResult {
                            name: file_name.clone(),
                            chunks: report.total_chunks,
                            size_mb: file_size_mb,
                            duration_seconds: file_duration,
                            status: "ok".to_string(),
                        });
                        b.pending_files.retain(|f| f != &file_name);
                        let total_done = b.completed_files + b.failed_files;
                        if total_done > 0 {
                            b.avg_time_per_file_seconds = b.elapsed_seconds as f64 / total_done as f64;
                            b.error_rate = (b.failed_files as f64 / total_done as f64) * 100.0;
                        }
                        if b.elapsed_seconds > 0 {
                            b.speed_chunks_per_min = (b.completed_chunks as f64 / b.elapsed_seconds as f64) * 60.0;
                            if b.speed_chunks_per_min > 0.0 && b.total_estimated_chunks > 0 {
                                let remaining_chunks = b.total_estimated_chunks.saturating_sub(b.completed_chunks);
                                b.eta_seconds = (remaining_chunks as f64 / b.speed_chunks_per_min * 60.0) as u64;
                            }
                        }
                    }
                    p.last_completed = Some(file_path.clone());
                }
                Err(e) => {
                    tracing::error!("Batch {} file {}/{} FAILED: {} — {}", batch_id, i + 1, total_files, file_name, e);
                    if let Some(ref mut b) = p.batch {
                        b.failed_files += 1;
                        b.errors.push(BatchError {
                            file: file_name.clone(),
                            error: e.to_string(),
                        });
                        b.files.push(FileResult {
                            name: file_name.clone(),
                            chunks: 0,
                            size_mb: file_size_mb,
                            duration_seconds: file_duration,
                            status: format!("error: {}", e),
                        });
                        b.pending_files.retain(|f| f != &file_name);
                        let total_done = b.completed_files + b.failed_files;
                        if total_done > 0 {
                            b.avg_time_per_file_seconds = b.elapsed_seconds as f64 / total_done as f64;
                            b.error_rate = (b.failed_files as f64 / total_done as f64) * 100.0;
                        }
                    }
                    p.last_error = Some(e.to_string());
                }
            }
        }

        if done && move_after_ingest {
            if let Err(e) = move_file_after_ingest(file_path, &cfg.ingested_dir) {
                tracing::warn!("Failed to move {}: {}", file_path, e);
            }
        }
    }

    {
        let mut p = progress.lock().unwrap();
        if let Some(ref mut b) = p.batch {
            b.status = BatchStatus::Completed;
            b.current_file = None;
            b.elapsed_seconds = batch_start.elapsed().as_secs();
            tracing::info!("Batch {} complete: {}/{} files, {} chunks, {:.1} MB, {} errors ({:.1}%), {}s",
                batch_id, b.completed_files, b.total_files, b.completed_chunks,
                b.total_size_mb, b.errors.len(), b.error_rate, b.elapsed_seconds);
        }
        p.status = IngestStatus::Idle;
        p.current_source = None;
    }
}

/// Move a file from inbox/ to ingested/ after successful ingestion.
fn move_file_after_ingest(file_path: &str, ingested_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(file_path);
    if !path.exists() {
        return Err("File does not exist".into());
    }
    let parent = path.parent().ok_or("No parent dir")?;
    let parent_str = parent.to_string_lossy();
    // Try to replace the last segment matching "inbox" with the configured ingested_dir
    let dest_dir = if parent_str.contains("/inbox/") {
        parent_str.replace("/inbox/", &format!("/{}/", ingested_dir))
    } else {
        format!("{}/{}", parent_str, ingested_dir)
    };
    std::fs::create_dir_all(&dest_dir)?;
    let dest = format!("{}/{}", dest_dir, path.file_name().unwrap().to_string_lossy());
    std::fs::rename(file_path, &dest)?;
    tracing::info!("Moved {} to {}", file_path, dest);
    Ok(())
}