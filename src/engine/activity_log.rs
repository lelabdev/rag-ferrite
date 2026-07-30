/// Global activity log for ingestion progress monitoring.
/// Engine pushes events here; ingestion.rs reads them when building progress responses.
use std::sync::Mutex;

/// Maximum number of events retained in the ring buffer.
const MAX_EVENTS: usize = 20;

/// A single activity event during ingestion.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActivityEvent {
    /// Unix epoch millis
    pub timestamp: u64,
    /// Human-readable description
    pub message: String,
    /// "embedding" | "llm" | "chunking" | "error" | "info"
    pub event_type: String,
}

/// Ring buffer of recent activity events.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ActivityLog {
    pub events: Vec<ActivityEvent>,
}

static ACTIVITY_LOG: std::sync::OnceLock<Mutex<ActivityLog>> = std::sync::OnceLock::new();

fn log_instance() -> &'static Mutex<ActivityLog> {
    ACTIVITY_LOG.get_or_init(|| Mutex::new(ActivityLog::new()))
}

impl ActivityLog {
    pub fn new() -> Self {
        Self {
            events: Vec::with_capacity(MAX_EVENTS),
        }
    }

    /// Push an event, evicting the oldest if the buffer is full.
    pub fn push(&mut self, event: ActivityEvent) {
        if self.events.len() >= MAX_EVENTS {
            self.events.remove(0);
        }
        self.events.push(event);
    }
}

/// Push a new activity event to the global log.
pub fn push(event_type: &str, message: String) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut log = log_instance().lock().unwrap();
    log.push(ActivityEvent {
        timestamp,
        message,
        event_type: event_type.to_string(),
    });
}

/// Take a snapshot of the current activity log.
pub fn snapshot() -> ActivityLog {
    log_instance().lock().unwrap().clone()
}

/// Clear the activity log (called at batch start / reset).
pub fn clear() {
    log_instance().lock().unwrap().events.clear();
}
