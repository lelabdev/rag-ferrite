/// Global chunk counter for real-time progress tracking.
/// Set by the ingestion manager before each batch, incremented by the engine
/// each time a parent chunk is committed to the DB.
use std::sync::atomic::{AtomicUsize, Ordering};

static CHUNK_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Set the chunk counter to a specific value (called at batch start).
pub fn reset() {
    CHUNK_COUNTER.store(0, Ordering::Relaxed);
}

/// Increment by n chunks (called after each parent commit).
pub fn add(n: usize) {
    CHUNK_COUNTER.fetch_add(n, Ordering::Relaxed);
}

/// Read the current count (called by progress API).
pub fn get() -> usize {
    CHUNK_COUNTER.load(Ordering::Relaxed)
}
