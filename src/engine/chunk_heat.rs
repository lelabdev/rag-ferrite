//! Chunk heat tracking — async batched.
//!
//! Follows the same pattern as collection heat (engine/heat.rs):
//! - Events are sent via mpsc channel (non-blocking on query hot path)
//! - Background worker buffers in memory, flushes to SQLite every 30s
//! - Single transaction for all pending updates

use anyhow::Result;
use std::time::Duration;
use tokio::sync::mpsc;

/// Flush interval in seconds.
const FLUSH_INTERVAL_SECS: u64 = 30;

// ── Types ────────────────────────────────────────────────────────────

/// Heat event sent from the query hot path. One per queried chunk.
#[derive(Debug, Clone)]
struct ChunkHeatEvent {
    chunk_id: i64,
}

// ── Tracker ──────────────────────────────────────────────────────────

/// Chunk heat tracker — owns an mpsc channel and a background flush worker.
///
/// Clone-safe: cloning shares the same sender, so all clones push to the same worker.
#[derive(Clone)]
pub struct ChunkHeatTracker {
    sender: mpsc::UnboundedSender<ChunkHeatEvent>,
}

impl ChunkHeatTracker {
    /// Create a new tracker and spawn the background flush worker.
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            background_worker(receiver).await;
        });
        tracing::info!(
            "Chunk heat tracker worker spawned (flush every {}s)",
            FLUSH_INTERVAL_SECS
        );
        ChunkHeatTracker { sender }
    }

    /// Record that chunk_ids were returned in query results. Non-blocking, fire-and-forget.
    pub fn record_chunks(&self, chunk_ids: &[i64]) {
        for &id in chunk_ids {
            let _ = self.sender.send(ChunkHeatEvent { chunk_id: id });
        }
    }
}

impl Default for ChunkHeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Background worker ────────────────────────────────────────────────

async fn background_worker(mut receiver: mpsc::UnboundedReceiver<ChunkHeatEvent>) {
    let mut buffer: std::collections::HashMap<i64, i32> = std::collections::HashMap::new();
    let mut interval = tokio::time::interval(Duration::from_secs(FLUSH_INTERVAL_SECS));
    // First tick completes immediately; skip it so we don't flush an empty buffer.
    interval.tick().await;

    loop {
        tokio::select! {
            // Drain events as they arrive
            Some(event) = receiver.recv() => {
                *buffer.entry(event.chunk_id).or_insert(0) += 1;
            }
            // Periodic flush
            _ = interval.tick() => {
                if !buffer.is_empty() {
                    if let Err(e) = flush(&buffer) {
                        tracing::warn!("Chunk heat flush failed: {}", e);
                    }
                }
                buffer.clear();
            }
        }
    }
}

/// Flush buffered chunk heat counts to SQLite in a single transaction.
fn flush(buffer: &std::collections::HashMap<i64, i32>) -> Result<()> {
    let conn = super::get_conn()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as f64;

    let tx = conn.unchecked_transaction()?;
    {
        for (&chunk_id, &count) in buffer {
            tx.execute(
                "UPDATE chunks SET query_count = query_count + ?1, last_queried_at = ?2 WHERE id = ?3",
                rusqlite::params![count, now, chunk_id],
            )?;
        }
    }
    tx.commit()?;

    tracing::debug!("Chunk heat flush: {} chunks updated", buffer.len());
    Ok(())
}
