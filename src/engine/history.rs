/// In-memory ring buffer of completed batch history.
/// Tracks the last N completed batches for monitoring via GET /api/history.
use std::collections::VecDeque;
use std::sync::Mutex;

/// Maximum number of completed batches retained.
const MAX_HISTORY: usize = 20;

/// A completed batch entry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchHistoryEntry {
    /// Batch identifier (e.g. "batch-1718012345678")
    pub batch_id: String,
    /// Unix epoch seconds when the batch completed
    pub timestamp: u64,
    /// Number of files processed
    pub file_count: usize,
    /// Number of chunks produced
    pub chunk_count: usize,
    /// Duration in seconds
    pub duration_secs: u64,
    /// Number of failed files
    pub errors: usize,
}

static HISTORY: std::sync::OnceLock<Mutex<VecDeque<BatchHistoryEntry>>> =
    std::sync::OnceLock::new();

fn history_instance() -> &'static Mutex<VecDeque<BatchHistoryEntry>> {
    HISTORY.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_HISTORY)))
}

/// Push a completed batch entry, evicting the oldest if at capacity.
pub fn push(entry: BatchHistoryEntry) {
    let mut buf = history_instance().lock().unwrap();
    if buf.len() >= MAX_HISTORY {
        buf.pop_front();
    }
    buf.push_back(entry);
}

/// Take a snapshot of the batch history (most recent last).
pub fn snapshot() -> Vec<BatchHistoryEntry> {
    history_instance().lock().unwrap().iter().cloned().collect()
}
