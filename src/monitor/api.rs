//! API interaction: fetch progress data, post server actions.

use std::time::Duration;

// ── Data structs (serde from API JSON) ──

#[derive(serde::Deserialize, Default)]
pub(crate) struct ProgressResponse {
    pub status: Option<String>,
    pub batch: Option<BatchProgress>,
    pub current_source: Option<String>,
    pub last_error: Option<String>,
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct BatchProgress {
    pub batch_id: Option<String>,
    pub status: Option<String>,
    pub total_files: usize,
    pub completed_files: usize,
    pub failed_files: usize,
    pub total_chunks: usize,
    pub completed_chunks: usize,
    pub total_size_mb: Option<f64>,
    pub speed_chunks_per_min: Option<f64>,
    pub avg_time_per_file_seconds: Option<f64>,
    pub elapsed_seconds: Option<f64>,
    pub eta_seconds: Option<f64>,
    pub error_rate: Option<f64>,
    #[serde(default)]
    pub errors: Vec<ErrorEntry>,
    pub current_file: Option<CurrentFile>,
    #[serde(default)]
    pub files: Vec<FileResult>,
    #[serde(default)]
    pub pending_files: Vec<String>,
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct CurrentFile {
    pub name: Option<String>,
    pub phase: Option<String>,
    pub chunks_done: Option<usize>,
    pub chunks_total: Option<usize>,
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct FileResult {
    pub name: Option<String>,
    pub chunks: Option<usize>,
    pub size_mb: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub status: Option<String>,
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct ErrorEntry {
    pub file: Option<String>,
    pub error: Option<String>,
}

/// Status endpoint response (version + document count).
#[derive(serde::Deserialize, Default)]
pub(crate) struct StatusResponse {
    pub version: Option<String>,
    pub document_count: Option<u64>,
    #[allow(dead_code)]
    pub error: Option<String>,
}

#[derive(serde::Deserialize, Default, Clone)]
pub(crate) struct Document {
    pub id: i64,
    pub name: Option<String>,
    pub collection_id: Option<String>,
    pub status: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct DocumentsResponse {
    #[serde(default)]
    files: Vec<Document>,
}

// ── HTTP fetch ──

pub(crate) fn fetch_documents(url: &str) -> Result<Vec<Document>, String> {
    let endpoint = format!("{}/api/documents", url.trim_end_matches('/'));
    let req = ureq::get(&endpoint).timeout(Duration::from_secs(5));
    let req = if let Some(key) = crate::client::resolve_api_key() {
        req.set("Authorization", &format!("Bearer {}", key))
    } else {
        req
    };
    req.call()
        .map_err(|e| format!("{}: {}", endpoint, e))?
        .into_json::<DocumentsResponse>()
        .map(|response| response.files)
        .map_err(|e| e.to_string())
}

/// Fetch batch progress from the rag-ferrite server.
pub(crate) fn fetch_progress(url: &str) -> Result<ProgressResponse, String> {
    let endpoint = format!("{}/api/ingest/progress", url.trim_end_matches('/'));
    let req = ureq::get(&endpoint).timeout(Duration::from_secs(5));
    // Use client config for API key
    let req = if let Some(key) = crate::client::resolve_api_key() {
        req.set("Authorization", &format!("Bearer {}", key))
    } else {
        req
    };
    match req.call() {
        Ok(resp) => resp
            .into_json::<ProgressResponse>()
            .map_err(|e| e.to_string()),
        Err(_) => Err(format!(
            "Cannot connect to {} — is ragfer serve running?",
            url
        )),
    }
}

/// POST an action to the rag-ferrite server (cancel, stop, rebuild, flush).
pub(crate) fn post_action(url: &str, path: &str) -> Result<String, String> {
    let full_url = format!("{}{}", url.trim_end_matches('/'), path);
    let req = ureq::post(&full_url).timeout(Duration::from_secs(5));
    let req = if let Some(key) = crate::client::resolve_api_key() {
        req.set("Authorization", &format!("Bearer {}", key))
    } else {
        req
    };
    req.call()
        .map_err(|e| format!("{}: {}", full_url, e))?
        .into_string()
        .map_err(|e| e.to_string())
}

// ── Formatting helpers ──

/// Format seconds into a human-readable duration string (e.g. "1h23m", "5m02s", "42s").
pub(crate) fn fmt_duration(secs: Option<f64>) -> String {
    match secs {
        Some(s) if s > 0.0 => {
            let h = (s / 3600.0) as u64;
            let m = ((s % 3600.0) / 60.0) as u64;
            let sec = (s % 60.0) as u64;
            if h > 0 {
                format!("{}h{:02}m", h, m)
            } else if m > 0 {
                format!("{}m{:02}s", m, sec)
            } else {
                format!("{}s", sec)
            }
        }
        _ => "—".to_string(),
    }
}
