/// Global cancel flag for batch ingestion.
/// Set by the API handler, checked by the batch worker between files.
use std::sync::atomic::{AtomicBool, Ordering};

static CANCELLED: AtomicBool = AtomicBool::new(false);

/// Request batch cancellation.
pub fn request() {
    CANCELLED.store(true, Ordering::Relaxed);
}

/// Check if cancellation was requested. Resets the flag.
pub fn check_and_reset() -> bool {
    let v = CANCELLED.load(Ordering::Relaxed);
    if v {
        CANCELLED.store(false, Ordering::Relaxed);
    }
    v
}
