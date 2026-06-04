//! rag-ferrite batch monitor — TUI subcommand
//! Usage: rag-ferrite monitor [refresh_seconds] [url]

use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::thread;

#[derive(serde::Deserialize, Default)]
struct ProgressResponse {
    status: Option<String>,
    batch: Option<BatchProgress>,
}

#[derive(serde::Deserialize, Default)]
struct BatchProgress {
    #[serde(rename = "batch_id")]
    _batch_id: Option<String>,
    status: Option<String>,
    total_files: usize,
    completed_files: usize,
    failed_files: usize,
    current_file: Option<CurrentFile>,
    completed_chunks: usize,
    total_chunks: usize,
    total_size_mb: f64,
    speed_chunks_per_min: f64,
    eta_seconds: u64,
    elapsed_seconds: u64,
    avg_time_per_file_seconds: f64,
    error_rate: f64,
    #[serde(default)]
    errors: Vec<BatchError>,
    #[serde(default)]
    files: Vec<FileResult>,
}

#[derive(serde::Deserialize)]
struct CurrentFile {
    name: String,
    #[allow(dead_code)]
    chunks_done: usize,
    #[allow(dead_code)]
    chunks_total: usize,
    phase: Option<String>,
}

#[derive(serde::Deserialize)]
struct BatchError {
    file: String,
    error: String,
}

#[derive(serde::Deserialize)]
struct FileResult {
    name: String,
    chunks: usize,
    size_mb: f64,
    duration_seconds: f64,
    status: String,
}

// ANSI
const HIDE_CURSOR: &str = "\x1b[?25l";
const SHOW_CURSOR: &str = "\x1b[?25h";
const CURSOR_HOME: &str = "\x1b[H";
const CLEAR_BELOW: &str = "\x1b[J";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BRAILLE: &[&str] = &["⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽", "⣾"];

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n - 3])
    }
}

fn fmt_duration(s: u64) -> String {
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        let (m, sec) = (s / 60, s % 60);
        format!("{}m{:02}s", m, sec)
    } else {
        let (h, m) = (s / 3600, (s % 3600) / 60);
        format!("{}h{:02}m", h, m)
    }
}

fn fetch_progress(url: &str) -> Option<ProgressResponse> {
    ureq::get(&format!("{}/api/ingest/progress", url))
        .timeout(Duration::from_secs(5))
        .call()
        .ok()
        .and_then(|r| r.into_json::<ProgressResponse>().ok())
}

fn render(data: &ProgressResponse, spinner_idx: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    match &data.batch {
        None => {
            lines.push(String::new());
            lines.push(format!("  {}─ No active batch ─{}", DIM, RESET));
            lines.push(String::new());
            lines.push(format!("  {}Refreshing... Ctrl+C to quit{}", DIM, RESET));
        }
        Some(batch) => {
            let status = batch.status.as_deref().unwrap_or("?");
            let total = batch.total_files;
            let done = batch.completed_files;
            let pct = if total > 0 { done as f64 / total as f64 * 100.0 } else { 0.0 };

            // Badge
            let badge = match status {
                "running" => format!("{}{} RUNNING{}", YELLOW, SPINNER[spinner_idx % SPINNER.len()], RESET),
                "completed" => format!("{}✓ DONE{}", GREEN, RESET),
                _ => status.to_uppercase(),
            };

            // Bar — █ full cells, ▓▒░ fade on last 3 cells before frontier,
            // ⡀→⣿ braille transition, empty + wave
            let bar_len = 50usize;
            let fill_chars: [&str; 8] = ["⡀", "⡄", "⡆", "⡇", "⣇", "⣧", "⣷", "⣿"];
            let num_fill = fill_chars.len(); // 8
            let total_states = bar_len * num_fill;
            let current_state = if total > 0 {
                (pct / 100.0 * total_states as f64) as usize
            } else {
                0
            };

            let front_cell = current_state / num_fill;
            let empty_start = front_cell + 1;
            let empty_len = (bar_len - empty_start) as f64;
            let fade_chars: [&str; 5] = ["░", "░", "▒", "▓", "▓"]; // 5 cellules, du + clair au + foncé

            // Wave — demi-blocs, doux et sinusoïdal
            let height_chars: [&str; 8] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"];

            let mut bar = String::new();
            for i in 0..bar_len {
                let cell_state = current_state.saturating_sub(i * num_fill);
                if cell_state >= num_fill {
                    // Full — fade last 5 cells before frontier
                    let dist_to_front = front_cell.saturating_sub(i);
                    if dist_to_front < 5 && front_cell >= 5 {
                        bar.push_str(fade_chars[dist_to_front]);
                    } else {
                        bar.push('█');
                    }
                } else if cell_state > 0 {
                    // Braille 1→8
                    bar.push_str(fill_chars[cell_state - 1]);
                } else {
                    // Empty — spinner grisé (DIM), décalé par cellule
                    let rel = i - empty_start;
                    let idx = (spinner_idx + rel) % SPINNER.len();
                    bar.push_str(DIM);
                    bar.push_str(SPINNER[idx]);
                    bar.push_str(RESET);
                }
            }

            let bid = match &data.batch {
                Some(b) => match b._batch_id.as_deref() {
                    Some(id) if id.len() >= 8 => &id[id.len()-8..],
                    _ => "????????",
                },
                None => "????????",
            };

            lines.push(String::new());
            lines.push(format!("  {} {}— batch {}{}", badge, DIM, bid, RESET));
            lines.push(format!("  {}[{}]{}", CYAN, bar, RESET));
            lines.push(format!("  {}{}/{} files — {:.0}%{}", DIM, done, total, pct, RESET));
            lines.push(String::new());

            // Stats
            let err_color = if batch.error_rate > 0.0 { RED } else { DIM };
            lines.push(format!(
                "  Chunks   {:>6} / {:<6}    Size      {:>8.1} MB",
                batch.completed_chunks, batch.total_chunks, batch.total_size_mb
            ));
            lines.push(format!(
                "  Speed    {:>5.0} chunks/min      Avg/file  {:>7.1}s",
                batch.speed_chunks_per_min, batch.avg_time_per_file_seconds
            ));
            lines.push(format!(
                "  Elapsed  {:>6}          ETA       {:>8}",
                fmt_duration(batch.elapsed_seconds), fmt_duration(batch.eta_seconds)
            ));
            lines.push(format!(
                "  Errors   {:>6} ({}{:.1}%{})",
                batch.failed_files, err_color, batch.error_rate, RESET
            ));
            lines.push(String::new());

            // Current file
            if let Some(cf) = &batch.current_file {
                let name = truncate(&cf.name, 55);
                let phase = cf.phase.as_deref().unwrap_or("?");
                lines.push(format!("  {}▶{} {}{}{}", CYAN, RESET, BOLD, name, RESET));
                lines.push(format!("    {}phase: {}{}", DIM, phase, RESET));
                lines.push(String::new());
            }

            // Recent files
            if !batch.files.is_empty() {
                lines.push(format!("  {}Recent files:{}", DIM, RESET));
                for f in batch.files.iter().rev().take(5) {
                    let fname = truncate(&f.name, 45);
                    let mark = if f.status == "ok" {
                        format!("{}✓{}", GREEN, RESET)
                    } else {
                        format!("{}✗{}", RED, RESET)
                    };
                    lines.push(format!(
                        "    {} {:<45} {:>4} ch  {:>5.1}MB  {:>5.1}s",
                        mark, fname, f.chunks, f.size_mb, f.duration_seconds
                    ));
                }
                lines.push(String::new());
            }

            // Errors
            if !batch.errors.is_empty() {
                let count = batch.errors.len();
                lines.push(format!("  {}Errors ({}):{}", RED, count, RESET));
                for e in batch.errors.iter().rev().take(3) {
                    let ef = truncate(&e.file, 40);
                    let er = truncate(&e.error, 60);
                    lines.push(format!("    {}{}: {}{}", DIM, ef, er, RESET));
                }
                lines.push(String::new());
            }

            lines.push(format!("  {}Ctrl+C to quit{}", DIM, RESET));
        }
    }

    lines
}

pub fn run(args: &[String]) {
    let url = args.get(1).cloned().unwrap_or_else(|| "http://localhost:4242".to_string());
    let refresh: f64 = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    let mut stdout = io::stdout();
    let _ = stdout.write_all(format!("{}{}{}", CURSOR_HOME, CLEAR_BELOW, HIDE_CURSOR).as_bytes());
    let _ = stdout.flush();

    let fetch_dur = Duration::from_secs_f64(refresh);
    let anim_dur = Duration::from_millis(150);
    let mut last_fetch = Instant::now() - fetch_dur; // fetch immediately
    let mut last_data: Option<ProgressResponse> = None;
    let mut spinner_idx: usize = 0;
    let mut prev_lines: usize = 0;

    // Handle Ctrl+C cleanly
    let _ = ctrlc_handler();

    loop {
        if last_fetch.elapsed() >= fetch_dur {
            last_data = fetch_progress(&url);
            last_fetch = Instant::now();
        }

        let lines = match &last_data {
            Some(data) => render(data, spinner_idx),
            None => {
                vec![
                    String::new(),
                    format!("  {}Cannot reach rag-ferrite on {}{}", RED, url, RESET),
                    String::new(),
                    format!("  {}Retrying...{}", DIM, RESET),
                ]
            }
        };

        // Pad
        let mut output = format!("{}{}", CURSOR_HOME, CLEAR_BELOW);
        for line in &lines {
            output.push_str(line);
            output.push('\n');
        }
        // Pad with empty lines to cover previous
        while lines.len() < prev_lines {
            output.push('\n');
        }
        prev_lines = lines.len();

        let _ = stdout.write_all(output.as_bytes());
        let _ = stdout.flush();

        spinner_idx = spinner_idx.wrapping_add(1);
        thread::sleep(anim_dur);
    }
}

fn ctrlc_handler() -> Result<(), ()> {
    Ok(())
}
