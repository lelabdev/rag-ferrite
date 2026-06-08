//! rag-monitor — Standalone TUI that connects to a rag-ferrite HTTP API
//!
//! Config via env vars:
//!   RAG_MONITOR_URL     — base URL (default: http://100.97.67.73:4242)
//!   RAG_MONITOR_KEY     — API key, sent as Authorization: Bearer header
//!   RAG_MONITOR_REFRESH — poll interval in seconds (default: 5)

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
    Frame, Terminal,
};

// ── Spinner ──
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── Pendulum animation (braille wave) ──
fn generate_pendulum_frames() -> Vec<String> {
    const WIDTH: usize = 50;
    const MAX_SPREAD: f64 = 6.0;
    const TOTAL_FRAMES: usize = 120;
    const PIXEL_COLS: usize = WIDTH * 2;
    const BRAILLE_BASE: u32 = 0x2800;
    const DOT_BITS: [[u32; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

    let mut frames = Vec::with_capacity(TOTAL_FRAMES);
    for t in 0..TOTAL_FRAMES {
        let mut codes = vec![BRAILLE_BASE; WIDTH];
        let progress = t as f64 / TOTAL_FRAMES as f64;
        let spread = (std::f64::consts::PI * progress).sin() * MAX_SPREAD;
        let base_phase = progress * std::f64::consts::PI * 8.0;

        for pc in 0..PIXEL_COLS {
            let swing = (base_phase + pc as f64 * spread).sin();
            let center = (1.0 - swing) * 1.5;
            for row in 0..4usize {
                if (row as f64 - center).abs() < 0.7 {
                    codes[pc / 2] |= DOT_BITS[row][pc % 2];
                }
            }
        }
        let frame: String = codes
            .iter()
            .map(|&c| char::from_u32(c).unwrap_or(' '))
            .collect();
        frames.push(frame);
    }
    frames
}

// ── API response types ──

#[derive(serde::Deserialize, Default)]
struct StatusResponse {
    version: Option<String>,
    document_count: Option<u64>,
    #[allow(dead_code)]
    error: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct ProgressResponse {
    #[allow(dead_code)]
    status: Option<String>,
    batch: Option<BatchProgress>,
    #[allow(dead_code)]
    current_source: Option<String>,
    #[allow(dead_code)]
    last_error: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct BatchProgress {
    batch_id: Option<String>,
    status: Option<String>,
    total_files: usize,
    completed_files: usize,
    failed_files: usize,
    #[allow(dead_code)]
    total_chunks: usize,
    completed_chunks: usize,
    total_size_mb: Option<f64>,
    speed_chunks_per_min: Option<f64>,
    avg_time_per_file_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
    eta_seconds: Option<f64>,
    error_rate: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    errors: Vec<ErrorEntry>,
    current_file: Option<CurrentFile>,
    #[serde(default)]
    files: Vec<FileResult>,
    #[serde(default)]
    pending_files: Vec<String>,
}

#[derive(serde::Deserialize, Default)]
struct CurrentFile {
    name: Option<String>,
    phase: Option<String>,
    chunks_done: Option<usize>,
    chunks_total: Option<usize>,
}

#[derive(serde::Deserialize, Default)]
struct FileResult {
    name: Option<String>,
    chunks: Option<usize>,
    #[allow(dead_code)]
    size_mb: Option<f64>,
    duration_seconds: Option<f64>,
    status: Option<String>,
}

#[derive(serde::Deserialize, Default)]
#[allow(dead_code)]
struct ErrorEntry {
    file: Option<String>,
    error: Option<String>,
}

// ── App state ──

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Completed,
    Queue,
}

impl Panel {
    fn next(self) -> Self {
        match self {
            Panel::Completed => Panel::Queue,
            Panel::Queue => Panel::Completed,
        }
    }
}

struct App {
    progress: Option<ProgressResponse>,
    status: Option<StatusResponse>,
    error: Option<String>,
    focus: Panel,
    scroll_completed: usize,
    scroll_pending: usize,
    spinner_idx: usize,
    pendulum_frames: Vec<String>,
    show_help: bool,
    action_msg: Option<(String, Instant)>,
}

impl App {
    fn new() -> Self {
        Self {
            progress: None,
            status: None,
            error: None,
            focus: Panel::Completed,
            scroll_completed: 0,
            scroll_pending: 0,
            spinner_idx: 0,
            pendulum_frames: generate_pendulum_frames(),
            show_help: false,
            action_msg: None,
        }
    }
}

// ── HTTP helper ──

fn fetch_json<T: serde::de::DeserializeOwned>(
    base_url: &str,
    api_key: Option<&str>,
    path: &str,
) -> Result<T, String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let mut req = ureq::get(&url).timeout(Duration::from_secs(5));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }
    req.call()
        .map_err(|e| format!("{}: {}", url, e))?
        .into_json::<T>()
        .map_err(|e| e.to_string())
}

fn post_action(
    base_url: &str,
    api_key: Option<&str>,
    path: &str,
) -> Result<String, String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let mut req = ureq::post(&url).timeout(Duration::from_secs(5));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }
    req.call()
        .map_err(|e| format!("{}: {}", url, e))?
        .into_string()
        .map_err(|e| e.to_string())
}

// ── Helpers ──

fn fmt_duration(secs: Option<f64>) -> String {
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

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ── Progress bar ──

fn render_progress_bar<'a>(
    pct: f64,
    spinner_idx: usize,
    bar_len: usize,
    pendulum_frames: &[String],
) -> Vec<Span<'a>> {
    let fill_chars: [&str; 8] = ["⡀", "⡄", "⡆", "⡇", "⣇", "⣧", "⣷", "⣿"];
    let fade_chars: Vec<&str> = vec!["▓", "▓", "░", "░"];
    let num_fill = fill_chars.len();
    let num_fade = fade_chars.len();
    let states_per_cell = num_fill + num_fade;
    let total_states = bar_len * states_per_cell;
    let current_state = (pct / 100.0 * total_states as f64).round() as usize;
    let front_cell = current_state / states_per_cell;
    let cell_remainder = current_state % states_per_cell;
    let empty_start = front_cell + 1;

    let mut spans: Vec<Span<'a>> = Vec::with_capacity(bar_len);
    for i in 0..bar_len {
        if i < front_cell {
            let dist_to_front = front_cell.saturating_sub(i);
            if dist_to_front <= num_fade {
                let fade_idx = num_fade - dist_to_front;
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(Color::Cyan),
                ));
            } else {
                spans.push(Span::styled("█", Style::default().fg(Color::Cyan)));
            }
        } else if i == front_cell && cell_remainder > 0 {
            if cell_remainder <= num_fill {
                spans.push(Span::styled(
                    fill_chars[cell_remainder - 1],
                    Style::default().fg(Color::Cyan),
                ));
            } else {
                let fade_idx = cell_remainder - num_fill - 1;
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(Color::Cyan),
                ));
            }
        } else if i >= empty_start {
            let empty_count = bar_len - empty_start;
            if empty_count > 0 && !pendulum_frames.is_empty() {
                let rel = i - empty_start;
                let frame_idx = spinner_idx % pendulum_frames.len();
                let frame = &pendulum_frames[frame_idx];
                let frame_chars: Vec<char> = frame.chars().collect();
                let ch = frame_chars.get(rel).copied().unwrap_or(' ');
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(Color::Cyan),
                ));
            } else {
                spans.push(Span::raw(" "));
            }
        } else {
            spans.push(Span::raw(" "));
        }
    }
    spans
}

// ── UI ──

fn ui(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let batch = app.progress.as_ref().and_then(|p| p.batch.as_ref());

    // Layout: header | lists | footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12),   // header
            Constraint::Min(10),   // lists
            Constraint::Length(1), // footer
        ])
        .split(size);

    // ── Header ──
    let header = Block::default().borders(Borders::TOP).title(Span::styled(
        " rag-monitor ",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ));
    f.render_widget(header, chunks[0]);

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // status line + bar
            Constraint::Length(1), // pct
            Constraint::Length(1), // blank
            Constraint::Length(4), // stats
            Constraint::Length(1), // server info
        ])
        .margin(1)
        .split(chunks[0]);

    // Status line
    let status_str = batch
        .map(|b| b.status.as_deref().unwrap_or("?"))
        .unwrap_or("connecting...");
    let total = batch.map(|b| b.total_files).unwrap_or(0);
    let done = batch.map(|b| b.completed_files).unwrap_or(0);
    let pct = if total > 0 {
        done as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let bid = batch
        .and_then(|b| b.batch_id.as_deref())
        .map(|id| &id[id.len().saturating_sub(8)..])
        .unwrap_or("????????");

    let spinner_char = SPINNER[app.spinner_idx % SPINNER.len()];
    let badge = match status_str {
        "running" => Span::styled(
            format!("{} RUNNING", spinner_char),
            Style::default().fg(Color::Yellow),
        ),
        "completed" => Span::styled("✓ DONE", Style::default().fg(Color::Green)),
        "failed" => Span::styled("✗ FAILED", Style::default().fg(Color::Red)),
        _ => Span::styled(
            status_str.to_uppercase(),
            Style::default().fg(Color::DarkGray),
        ),
    };

    let status_line = Line::from(vec![
        Span::raw("  "),
        badge,
        Span::raw(" "),
        Span::styled(
            format!("— batch {}", bid),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            if total > 0 {
                format!("    {:.0}% ({}/{})", pct, done, total)
            } else {
                "    waiting…".to_string()
            },
            Style::default().fg(Color::White),
        ),
    ]);
    f.render_widget(Paragraph::new(status_line), inner[0]);

    // Progress bar
    let bar_spans = render_progress_bar(pct, app.spinner_idx, 50, &app.pendulum_frames);
    let mut bar_line_spans = vec![Span::raw("  [")];
    bar_line_spans.extend(bar_spans);
    bar_line_spans.push(Span::raw("]"));
    f.render_widget(Paragraph::new(Line::from(bar_line_spans)), inner[1]);

    // Stats
    if let Some(b) = batch {
        let speed = b.speed_chunks_per_min.unwrap_or(0.0);
        let avg_file = b.avg_time_per_file_seconds.unwrap_or(0.0);
        let elapsed = fmt_duration(b.elapsed_seconds);
        let eta = fmt_duration(b.eta_seconds);
        let err_count = b.failed_files;
        let err_rate = b.error_rate.unwrap_or(0.0);
        let size_mb = b.total_size_mb.unwrap_or(0.0);
        let chunks_done = b.completed_chunks;

        let speed_color = if speed >= 100.0 {
            Color::Green
        } else {
            Color::Yellow
        };
        let stats_lines = vec![
            Line::from(vec![
                Span::styled(
                    format!("  Chunks   {:>6}", chunks_done),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw("          "),
                Span::styled(
                    format!("Size      {:>8.1} MB", size_mb),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("  Speed    {:>6.0} chunks/min", speed),
                    Style::default().fg(speed_color),
                ),
                Span::raw("     "),
                Span::styled(
                    format!("Avg/file  {:>8.1}s", avg_file),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("  Elapsed  {:>6}", elapsed),
                    Style::default().fg(Color::White),
                ),
                Span::raw("       "),
                Span::styled(
                    format!("ETA       {:>8}", eta),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
            Line::from(vec![Span::styled(
                format!("  Errors   {:>6} ({:.1}%)", err_count, err_rate),
                Style::default().fg(if err_count > 0 {
                    Color::Red
                } else {
                    Color::White
                }),
            )]),
        ];
        f.render_widget(Paragraph::new(stats_lines), inner[3]);
    } else if let Some(e) = &app.error {
        let err_line = Line::from(vec![Span::styled(
            format!("  ⚠ Connection error: {}", e),
            Style::default().fg(Color::Red),
        )]);
        f.render_widget(Paragraph::new(err_line), inner[3]);
    }

    // Server status line (version + doc count)
    if let Some(s) = &app.status {
        let version = s.version.as_deref().unwrap_or("?");
        let doc_count = s.document_count.unwrap_or(0);
        let server_line = Line::from(vec![Span::styled(
            format!("  Server v{} • {} documents", version, doc_count),
            Style::default().fg(Color::DarkGray),
        )]);
        f.render_widget(Paragraph::new(server_line), inner[4]);
    }

    // ── Lists ──
    let list_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[1]);

    // Completed files
    let completed_items: Vec<ListItem> = batch
        .map(|b| {
            b.files
                .iter()
                .rev()
                .map(|f| {
                    let name = f.name.as_deref().unwrap_or("?");
                    let chunks = f.chunks.unwrap_or(0);
                    let dur = f.duration_seconds.unwrap_or(0.0);
                    let status_icon = match f.status.as_deref() {
                        Some("ok") | Some("completed") => "✓",
                        Some("error") | Some("failed") => "✗",
                        _ => "?",
                    };
                    let color_raw = match f.status.as_deref() {
                        Some("ok") | Some("completed") => Color::Green,
                        Some("error") | Some("failed") => Color::Red,
                        _ => Color::Yellow,
                    };
                    let text = format!(" {:<48} {:>4}ch {:>6.1}s", truncate(name, 48), chunks, dur);
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{} ", status_icon), Style::default().fg(color_raw)),
                        Span::styled(text, Style::default().fg(color_raw)),
                    ]))
                })
                .collect()
        })
        .unwrap_or_default();

    let completed_title = format!(
        " Completed ({}) ",
        batch.map(|b| b.completed_files).unwrap_or(0)
    );
    let completed_block = Block::default().borders(Borders::ALL).title(Span::styled(
        completed_title,
        Style::default()
            .fg(if app.focus == Panel::Completed {
                Color::Cyan
            } else {
                Color::DarkGray
            })
            .add_modifier(if app.focus == Panel::Completed {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
    ));
    let mut completed_state = ListState::default();
    if app.focus == Panel::Completed && !completed_items.is_empty() {
        completed_state.select(Some(app.scroll_completed));
    }
    let completed_list = List::new(completed_items)
        .block(completed_block)
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(completed_list, list_area[0], &mut completed_state);

    // Right panel: current file + queue
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(5)])
        .split(list_area[1]);

    // Current file
    let current_lines = batch
        .and_then(|b| b.current_file.as_ref())
        .map(|cf| {
            let name = cf.name.as_deref().unwrap_or("?");
            let phase = cf.phase.as_deref().unwrap_or("?");
            let chunks_done = cf.chunks_done.unwrap_or(0);
            let chunks_total = cf.chunks_total.unwrap_or(0);
            vec![
                Line::from(vec![
                    Span::styled(" ▶ ", Style::default().fg(Color::Yellow)),
                    Span::styled(truncate(name, 38), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![Span::styled(
                    format!("   phase: {}", phase),
                    Style::default().fg(Color::DarkGray),
                )]),
                Line::from(vec![Span::styled(
                    format!("   chunks: {}/{}", chunks_done, chunks_total),
                    Style::default().fg(Color::DarkGray),
                )]),
            ]
        })
        .unwrap_or_else(|| {
            vec![Line::from(Span::styled(
                "  idle",
                Style::default().fg(Color::DarkGray),
            ))]
        });

    let current_block = Block::default().borders(Borders::ALL).title(Span::styled(
        " Current ",
        Style::default().fg(Color::Yellow),
    ));
    f.render_widget(
        Paragraph::new(current_lines).block(current_block),
        right_chunks[0],
    );

    // Queue (pending files)
    let queue_items: Vec<ListItem> = batch
        .map(|b| {
            b.pending_files
                .iter()
                .map(|name| {
                    ListItem::new(Line::from(vec![Span::styled(
                        format!(" {}", truncate(name, 42)),
                        Style::default().fg(Color::Gray),
                    )]))
                })
                .collect()
        })
        .unwrap_or_default();

    let queue_title = format!(
        " Queue ({}) ",
        batch.map(|b| b.pending_files.len()).unwrap_or(0)
    );
    let queue_block = Block::default().borders(Borders::ALL).title(Span::styled(
        queue_title,
        Style::default()
            .fg(if app.focus == Panel::Queue {
                Color::Cyan
            } else {
                Color::DarkGray
            })
            .add_modifier(if app.focus == Panel::Queue {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
    ));
    let mut queue_state = ListState::default();
    if app.focus == Panel::Queue && !queue_items.is_empty() {
        queue_state.select(Some(app.scroll_pending));
    }
    let queue_list = List::new(queue_items)
        .block(queue_block)
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(queue_list, right_chunks[1], &mut queue_state);

    // ── Footer ──
    let key_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let dim_style = Style::default().fg(Color::DarkGray);
    // Footer: action feedback or key hints
    if let Some((msg, ts)) = &app.action_msg {
        if ts.elapsed() < Duration::from_secs(5) {
            let color = if msg.starts_with('✓') { Color::Green } else { Color::Red };
            let footer = Line::from(vec![
                Span::styled(msg.clone(), Style::default().fg(color)),
            ]);
            f.render_widget(Paragraph::new(footer), chunks[2]);
        } else {
            app.action_msg = None;
        }
    }
    if app.action_msg.is_none() {
        let footer = Line::from(vec![
            Span::styled("TAB", key_style),
            Span::styled(" switch ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" ↑↓", key_style),
            Span::styled(" scroll ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" c", key_style),
            Span::styled(" cancel ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" x", key_style),
            Span::styled(" stop ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" r", key_style),
            Span::styled(" rebuild ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" f", key_style),
            Span::styled(" flush ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" ?", key_style),
            Span::styled(" help ", dim_style),
            Span::styled("•", dim_style),
            Span::styled(" q", key_style),
            Span::styled(" quit", dim_style),
        ]);
        f.render_widget(Paragraph::new(footer), chunks[2]);
    }

    // ── Help popup ──
    if app.show_help {
        let area = centered_rect(50, 50, f.area());
        let block = Block::default().borders(Borders::ALL).title(Span::styled(
            " Help — press ? or Esc to close ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        let hk = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let hd = Style::default().fg(Color::Gray);
        let help_lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  TAB    ", hk),
                Span::styled("Switch panel", hd),
            ]),
            Line::from(vec![
                Span::styled("  ↑ ↓    ", hk),
                Span::styled("Scroll list", hd),
            ]),
            Line::from(vec![
                Span::styled("  ?      ", hk),
                Span::styled("Toggle this help", hd),
            ]),
            Line::from(vec![
                Span::styled("  q/Esc  ", hk),
                Span::styled("Quit", hd),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "  Env vars:",
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![
                Span::styled("  RAG_MONITOR_URL     ", hk),
                Span::styled("base URL", hd),
            ]),
            Line::from(vec![
                Span::styled("  RAG_MONITOR_KEY     ", hk),
                Span::styled("API key", hd),
            ]),
            Line::from(vec![
                Span::styled("  RAG_MONITOR_REFRESH ", hk),
                Span::styled("poll interval (s)", hd),
            ]),
        ];
        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(help_lines).block(block), area);
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ── Entry point ──

fn main() {
    // Config from env
    let base_url =
        std::env::var("RAG_MONITOR_URL").unwrap_or_else(|_| "http://100.97.67.73:4242".to_string());
    let api_key = std::env::var("RAG_MONITOR_KEY").ok();
    let refresh_secs: f64 = std::env::var("RAG_MONITOR_REFRESH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);

    // Setup terminal
    if enable_raw_mode().is_err() {
        eprintln!("Error: rag-monitor requires a real terminal (TTY).");
        eprintln!(
            "Usage: rag-monitor  (config via RAG_MONITOR_URL, RAG_MONITOR_KEY, RAG_MONITOR_REFRESH)"
        );
        return;
    }
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap_or(());
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut app = App::new();
    let fetch_dur = Duration::from_secs_f64(refresh_secs);
    let mut last_fetch = Instant::now() - fetch_dur; // fetch immediately

    loop {
        // Poll API
        if last_fetch.elapsed() >= fetch_dur {
            let progress_result =
                fetch_json::<ProgressResponse>(&base_url, api_key.as_deref(), "/api/ingest/progress");
            let status_result =
                fetch_json::<StatusResponse>(&base_url, api_key.as_deref(), "/api/status");

            match progress_result {
                Ok(data) => {
                    app.progress = Some(data);
                    app.error = None;
                }
                Err(e) => {
                    app.error = Some(e);
                }
            }

            match status_result {
                Ok(s) => app.status = Some(s),
                Err(_) => {} // don't overwrite progress error for status failures
            }

            last_fetch = Instant::now();
        }

        app.spinner_idx += 1;

        // Render
        terminal.draw(|f| ui(f, &mut app)).unwrap();

        // Poll input
        if event::poll(Duration::from_millis(150)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if app.show_help {
                    if key.code == KeyCode::Char('?') || key.code == KeyCode::Esc {
                        app.show_help = false;
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Tab => app.focus = app.focus.next(),
                    KeyCode::Down => match app.focus {
                        Panel::Completed => {
                            let max = app
                                .progress
                                .as_ref()
                                .and_then(|p| p.batch.as_ref())
                                .map(|b| b.files.len().saturating_sub(1))
                                .unwrap_or(0);
                            if app.scroll_completed < max {
                                app.scroll_completed += 1;
                            }
                        }
                        Panel::Queue => {
                            let max = app
                                .progress
                                .as_ref()
                                .and_then(|p| p.batch.as_ref())
                                .map(|b| b.pending_files.len().saturating_sub(1))
                                .unwrap_or(0);
                            if app.scroll_pending < max {
                                app.scroll_pending += 1;
                            }
                        }
                    },
                    KeyCode::Up => match app.focus {
                        Panel::Completed => {
                            if app.scroll_completed > 0 {
                                app.scroll_completed -= 1;
                            }
                        }
                        Panel::Queue => {
                            if app.scroll_pending > 0 {
                                app.scroll_pending -= 1;
                            }
                        }
                    },
                    KeyCode::Char('?') => app.show_help = true,
                    KeyCode::Char('c') => {
                        match post_action(&base_url, api_key.as_deref(), "/api/service/cancel-batch") {
                            Ok(msg) => app.action_msg = Some((format!("✓ Cancel: {}", msg), Instant::now())),
                            Err(e) => app.action_msg = Some((format!("✗ Cancel failed: {}", e), Instant::now())),
                        }
                    }
                    KeyCode::Char('x') => {
                        match post_action(&base_url, api_key.as_deref(), "/api/service/stop") {
                            Ok(msg) => app.action_msg = Some((format!("✓ Stop: {}", msg), Instant::now())),
                            Err(e) => app.action_msg = Some((format!("✗ Stop failed: {}", e), Instant::now())),
                        }
                    }
                    KeyCode::Char('r') => {
                        match post_action(&base_url, api_key.as_deref(), "/api/rebuild-indexes") {
                            Ok(msg) => app.action_msg = Some((format!("✓ Rebuild: {}", msg), Instant::now())),
                            Err(e) => app.action_msg = Some((format!("✗ Rebuild failed: {}", e), Instant::now())),
                        }
                    }
                    KeyCode::Char('f') => {
                        match post_action(&base_url, api_key.as_deref(), "/api/flush-indexes") {
                            Ok(msg) => app.action_msg = Some((format!("✓ Flush: {}", msg), Instant::now())),
                            Err(e) => app.action_msg = Some((format!("✗ Flush failed: {}", e), Instant::now())),
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode().unwrap_or(());
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap_or(());
    terminal.show_cursor().unwrap_or(());
}
