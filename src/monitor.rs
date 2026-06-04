//! rag-ferrite batch monitor — TUI with ratatui
//! Usage: rag-ferrite monitor [refresh_seconds] [url]

use std::io::{self, Stdout};
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
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
    Terminal, Frame,
};

// ── ANSI colors (reused in progress bar) ──
const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";

// ── Braille spinner (10 frames) ──
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── Data structs (serde from API JSON) ──

#[derive(serde::Deserialize, Default)]
struct ProgressResponse {
    status: Option<String>,
    batch: Option<BatchProgress>,
    current_source: Option<String>,
    last_error: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct BatchProgress {
    batch_id: Option<String>,
    status: Option<String>,
    total_files: usize,
    completed_files: usize,
    failed_files: usize,
    total_chunks: usize,
    completed_chunks: usize,
    total_size_mb: Option<f64>,
    speed_chunks_per_min: Option<f64>,
    avg_time_per_file_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
    eta_seconds: Option<f64>,
    error_rate: Option<f64>,
    #[serde(default)]
    errors: Vec<String>,
    current_file: Option<CurrentFile>,
    #[serde(default)]
    files: Vec<FileResult>,
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
    size_mb: Option<f64>,
    duration_seconds: Option<f64>,
    status: Option<String>,
}

// ── App state ──

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Stats,
    Completed,
}

impl Panel {
    fn next(self) -> Self {
        match self {
            Panel::Stats => Panel::Completed,
            Panel::Completed => Panel::Stats,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ColorMode {
    Full,
    StatsOnly,
    Mono,
}

struct App {
    data: Option<ProgressResponse>,
    error: Option<String>,
    focus: Panel,
    scroll: usize,
    spinner_idx: usize,
    show_lists: bool,
    show_stats: bool,
    show_help: bool,
    color_mode: ColorMode,
}

impl App {
    fn new() -> Self {
        Self {
            data: None,
            error: None,
            focus: Panel::Stats,
            scroll: 0,
            spinner_idx: 0,
            show_lists: true,
            show_stats: true,
            show_help: false,
            color_mode: ColorMode::Full,
        }
    }

    fn scroll_up(&mut self) {
        if self.scroll > 0 {
            self.scroll -= 1;
        }
    }

    fn scroll_down(&mut self, max: usize) {
        if self.scroll < max {
            self.scroll += 1;
        }
    }
}

// ── HTTP fetch ──

fn fetch_progress(url: &str) -> Result<ProgressResponse, String> {
    let endpoint = format!("{}/api/ingest/progress", url.trim_end_matches('/'));
    match ureq::get(&endpoint).timeout(Duration::from_secs(5)).call() {
        Ok(resp) => {
            resp.into_json::<ProgressResponse>().map_err(|e| e.to_string())
        }
        Err(e) => Err(format!("HTTP error: {}", e)),
    }
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

/// Resolve a "full" color to the color to actually render, respecting `color_mode`.
/// `Full` and `StatsOnly` keep the original color (bar is overridden separately
/// inside `render_progress_bar`). `Mono` collapses everything to Gray.
fn color_for(mode: ColorMode, full_color: Color) -> Color {
    match mode {
        ColorMode::Full | ColorMode::StatsOnly => full_color,
        ColorMode::Mono => Color::Gray,
    }
}

// ── Progress bar renderer — returns colored Spans ──

fn render_progress_bar<'a>(
    pct: f64,
    spinner_idx: usize,
    bar_len: usize,
    color_mode: ColorMode,
) -> Vec<Span<'a>> {
    // In the bar specifically, StatsOnly collapses to monochrome cyan,
    // and Mono to gray.
    let bar_color = |c: Color| -> Color {
        match color_mode {
            ColorMode::Full => c,
            ColorMode::StatsOnly => Color::Cyan,
            ColorMode::Mono => Color::Gray,
        }
    };

    let fill_chars: [&str; 8] = ["⡀", "⡄", "⡆", "⡇", "⣇", "⣧", "⣷", "⣿"];
    let fade_chars: [&str; 5] = ["░", "░", "▒", "▓", "▓"];
    let num_fill = fill_chars.len();
    let num_fade = fade_chars.len();
    let states_per_cell = num_fill + num_fade; // 13
    let total_states = bar_len * states_per_cell;
    let current_state = (pct / 100.0 * total_states as f64).round() as usize;
    let front_cell = current_state / states_per_cell;
    let cell_remainder = current_state % states_per_cell;

    let empty_start = front_cell + 1;

    // Color palette for spinner wave — travels through cells
    let wave = [
        Color::DarkGray,
        Color::DarkGray,
        Color::Blue,
        Color::Cyan,
        Color::LightCyan,
        Color::White,
        Color::LightCyan,
        Color::Cyan,
        Color::Blue,
        Color::DarkGray,
    ];

    let mut spans: Vec<Span<'a>> = Vec::with_capacity(bar_len);

    for i in 0..bar_len {
        if i < front_cell {
            // Full — but fade last 5 cells before frontier
            let dist_to_front = front_cell.saturating_sub(i);
            if dist_to_front <= num_fade {
                let fade_idx = num_fade - dist_to_front;
                let color = match fade_idx {
                    0 => Color::Cyan,
                    1 => Color::Cyan,
                    2 => Color::LightBlue,
                    3 => Color::Green,
                    _ => Color::Green,
                };
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(bar_color(color)),
                ));
            } else {
                spans.push(Span::styled(
                    "█",
                    Style::default().fg(bar_color(Color::Green)),
                ));
            }
        } else if i == front_cell && cell_remainder > 0 {
            if cell_remainder <= num_fill {
                // Braille transition — yellow
                spans.push(Span::styled(
                    fill_chars[cell_remainder - 1],
                    Style::default().fg(bar_color(Color::Yellow)),
                ));
            } else {
                // Fade part of transition cell
                let fade_idx = cell_remainder - num_fill - 1;
                let color = match fade_idx {
                    0 => Color::Cyan,
                    1 => Color::LightBlue,
                    _ => Color::Green,
                };
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(bar_color(color)),
                ));
            }
        } else if i >= empty_start {
            // Spinner — color wave traveling
            let rel = i - empty_start;
            let char_idx = (spinner_idx + rel) % SPINNER.len();
            let color_idx = (spinner_idx + rel) % wave.len();
            spans.push(Span::styled(
                SPINNER[char_idx],
                Style::default().fg(bar_color(wave[color_idx])),
            ));
        } else {
            spans.push(Span::raw(" "));
        }
    }

    spans
}

// ── UI ──

fn ui(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let data = app.data.as_ref();
    let cm = app.color_mode;

    // Layout: header (bar+stats) | lists
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12),                                          // header
            if app.show_lists { Constraint::Min(10) } else { Constraint::Length(0) }, // lists
            Constraint::Length(1),                                        // footer
        ])
        .split(size);

    // ── Header ──
    let header = Block::default().borders(Borders::TOP).title(Span::styled(
        " rag-ferrite monitor ",
        Style::default()
            .fg(color_for(cm, Color::Cyan))
            .add_modifier(Modifier::BOLD),
    ));
    f.render_widget(header, chunks[0]);

    // When stats are hidden, collapse the blank+stats rows.
    let stats_constraint = if app.show_stats {
        Constraint::Length(4)
    } else {
        Constraint::Length(0)
    };
    let blank_constraint = if app.show_stats {
        Constraint::Length(1)
    } else {
        Constraint::Length(0)
    };
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // status + bar
            Constraint::Length(1), // pct
            blank_constraint,      // blank
            stats_constraint,      // stats
        ])
        .margin(1)
        .split(chunks[0]);

    // Status line
    let batch = data.and_then(|d| d.batch.as_ref());
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
            Style::default().fg(color_for(cm, Color::Yellow)),
        ),
        "completed" => Span::styled(
            "✓ DONE",
            Style::default().fg(color_for(cm, Color::Green)),
        ),
        "failed" => Span::styled(
            "✗ FAILED",
            Style::default().fg(color_for(cm, Color::Red)),
        ),
        _ => Span::styled(
            status_str.to_uppercase(),
            Style::default().fg(color_for(cm, Color::DarkGray)),
        ),
    };

    let status_line = Line::from(vec![
        Span::raw("  "),
        badge,
        Span::raw(" "),
        Span::styled(
            format!("— batch {}", bid),
            Style::default().fg(color_for(cm, Color::DarkGray)),
        ),
        Span::styled(
            format!("    {:.0}% ({}/{})", pct, done, total),
            Style::default().fg(color_for(cm, Color::White)),
        ),
    ]);
    f.render_widget(Paragraph::new(status_line), inner[0]);

    // Progress bar — colored spans
    let bar_spans = render_progress_bar(pct, app.spinner_idx, 50, cm);
    let mut bar_line_spans = vec![Span::raw("  [")];
    bar_line_spans.extend(bar_spans);
    bar_line_spans.push(Span::raw("]"));
    f.render_widget(Paragraph::new(Line::from(bar_line_spans)), inner[1]);

    // Stats
    if app.show_stats {
        if let Some(b) = batch {
            let speed = b.speed_chunks_per_min.unwrap_or(0.0);
            let avg_file = b.avg_time_per_file_seconds.unwrap_or(0.0);
            let elapsed = fmt_duration(b.elapsed_seconds);
            let eta = fmt_duration(b.eta_seconds);
            let err_count = b.failed_files;
            let err_rate = b.error_rate.unwrap_or(0.0);
            let size_mb = b.total_size_mb.unwrap_or(0.0);
            let chunks_done = b.completed_chunks;
            let chunks_total = b.total_chunks;

            let speed_color_raw = if speed >= 100.0 {
                Color::Green
            } else {
                Color::Yellow
            };
            let stats_lines = vec![
                Line::from(vec![
                    Span::styled(
                        format!("  Chunks   {:>6}", chunks_done),
                        Style::default().fg(color_for(cm, Color::Cyan)),
                    ),
                    Span::styled(
                        format!(" / {:<6}", chunks_total),
                        Style::default().fg(color_for(cm, Color::DarkGray)),
                    ),
                    Span::raw("     "),
                    Span::styled(
                        format!("Size      {:>8.1} MB", size_mb),
                        Style::default().fg(color_for(cm, Color::Gray)),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("  Speed    {:>6.0} chunks/min", speed),
                        Style::default().fg(color_for(cm, speed_color_raw)),
                    ),
                    Span::raw("     "),
                    Span::styled(
                        format!("Avg/file  {:>8.1}s", avg_file),
                        Style::default().fg(color_for(cm, Color::Gray)),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("  Elapsed  {:>6}", elapsed),
                        Style::default().fg(color_for(cm, Color::Gray)),
                    ),
                    Span::raw("       "),
                    Span::styled(
                        format!("ETA       {:>8}", eta),
                        Style::default().fg(color_for(cm, Color::Magenta)),
                    ),
                ]),
                Line::from(vec![Span::styled(
                    format!("  Errors   {:>6} ({:.1}%)", err_count, err_rate),
                    Style::default().fg(color_for(cm, if err_count > 0 {
                        Color::Red
                    } else {
                        Color::DarkGray
                    })),
                )]),
            ];
            f.render_widget(Paragraph::new(stats_lines), inner[3]);
        } else if let Some(e) = &app.error {
            let err_line = Line::from(vec![Span::styled(
                format!("  ⚠ Connection error: {}", e),
                Style::default().fg(color_for(cm, Color::Red)),
            )]);
            f.render_widget(Paragraph::new(err_line), inner[3]);
        }
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
                    let text = format!(
                        " {} {:<48} {:>4}ch {:>6.1}s",
                        status_icon,
                        truncate(name, 48),
                        chunks,
                        dur
                    );
                    ListItem::new(Line::from(vec![Span::styled(
                        text,
                        Style::default().fg(color_for(cm, color_raw)),
                    )]))
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
            .fg(color_for(
                cm,
                if app.focus == Panel::Completed {
                    Color::Cyan
                } else {
                    Color::DarkGray
                },
            ))
            .add_modifier(if app.focus == Panel::Completed {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
    ));
    let completed_list = List::new(completed_items).block(completed_block);
    f.render_widget(completed_list, list_area[0]);

    // Right panel: current file + pending info
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
                    Span::styled(" ▶ ", Style::default().fg(color_for(cm, Color::Yellow))),
                    Span::styled(
                        truncate(name, 38),
                        Style::default().fg(color_for(cm, Color::White)),
                    ),
                ]),
                Line::from(vec![Span::styled(
                    format!("   phase: {}", phase),
                    Style::default().fg(color_for(cm, Color::DarkGray)),
                )]),
                Line::from(vec![Span::styled(
                    format!("   chunks: {}/{}", chunks_done, chunks_total),
                    Style::default().fg(color_for(cm, Color::DarkGray)),
                )]),
            ]
        })
        .unwrap_or_else(|| {
            vec![Line::from(Span::styled(
                "  idle",
                Style::default().fg(color_for(cm, Color::DarkGray)),
            ))]
        });

    let current_block = Block::default().borders(Borders::ALL).title(Span::styled(
        " Current ",
        Style::default().fg(color_for(cm, Color::Yellow)),
    ));
    f.render_widget(
        Paragraph::new(current_lines).block(current_block),
        right_chunks[0],
    );

    // Pending count + errors
    let pending_count = batch
        .map(|b| {
            b.total_files
                .saturating_sub(b.completed_files + b.failed_files)
        })
        .unwrap_or(0);
    let pending_lines = vec![
        Line::from(vec![Span::styled(
            format!(" {} files pending", pending_count),
            Style::default().fg(color_for(cm, Color::DarkGray)),
        )]),
        Line::from(""),
    ];

    let mut all_lines = pending_lines;

    // Show errors if any
    if let Some(b) = batch {
        if !b.errors.is_empty() {
            all_lines.push(Line::from(Span::styled(
                " Errors:",
                Style::default().fg(color_for(cm, Color::Red)),
            )));
            for err in b.errors.iter().take(5) {
                all_lines.push(Line::from(Span::styled(
                    format!(" • {}", truncate(err, 40)),
                    Style::default().fg(color_for(cm, Color::Red)),
                )));
            }
        }
    }

    let pending_block = Block::default().borders(Borders::ALL).title(Span::styled(
        " Queue ",
        Style::default().fg(color_for(cm, Color::DarkGray)),
    ));
    f.render_widget(
        Paragraph::new(all_lines).block(pending_block),
        right_chunks[1],
    );

    // ── Footer ──
    let key_style = Style::default()
        .fg(color_for(cm, Color::Cyan))
        .add_modifier(Modifier::BOLD);
    let dim_style = Style::default().fg(color_for(cm, Color::DarkGray));
    let footer = Line::from(vec![
        Span::styled("TAB", key_style),
        Span::styled(" switch ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" ↑↓", key_style),
        Span::styled(" scroll ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" l", key_style),
        Span::styled(" list ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" c", key_style),
        Span::styled(" color ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" s", key_style),
        Span::styled(" stats ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" o", key_style),
        Span::styled(" open ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" ?", key_style),
        Span::styled(" help ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" q", key_style),
        Span::styled(" quit", dim_style),
    ]);
    f.render_widget(Paragraph::new(footer), chunks[2]);

    // ── Help popup (drawn last so it sits on top) ──
    if app.show_help {
        let area = centered_rect(60, 70, f.area());
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
            Line::from(vec![Span::styled("  TAB    ", hk), Span::styled("Switch panel", hd)]),
            Line::from(vec![Span::styled("  ↑ ↓    ", hk), Span::styled("Scroll list", hd)]),
            Line::from(vec![
                Span::styled("  l      ", hk),
                Span::styled("Toggle lists visibility", hd),
            ]),
            Line::from(vec![
                Span::styled("  c      ", hk),
                Span::styled("Cycle color modes (Full → StatsOnly → Mono)", hd),
            ]),
            Line::from(vec![Span::styled("  s      ", hk), Span::styled("Toggle stats", hd)]),
            Line::from(vec![
                Span::styled("  o      ", hk),
                Span::styled("Open selected file in less  (Completed panel)", hd),
            ]),
            Line::from(vec![
                Span::styled("  ?      ", hk),
                Span::styled("Toggle this help", hd),
            ]),
            Line::from(vec![Span::styled("  q/Esc  ", hk), Span::styled("Quit", hd)]),
        ];

        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(help_lines).block(block), area);
    }
}

/// Compute a centered rect of `percent_x`% width and `percent_y`% height inside `r`.
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

/// Open the currently-selected Completed file in `less`. Suspends the TUI
/// (leaves alternate screen + disables raw mode) while less runs, then
/// restores it and forces a full redraw.
fn open_selected_file(app: &mut App, terminal: &mut Terminal<CrosstermBackend<Stdout>>) {
    let files = app
        .data
        .as_ref()
        .and_then(|d| d.batch.as_ref())
        .map(|b| &b.files);
    let files = match files {
        Some(f) => f,
        None => return,
    };
    if files.is_empty() {
        return;
    }

    // Mirror the reversed order used in the Completed list display.
    let idx = files.len().saturating_sub(1).saturating_sub(app.scroll);
    if idx >= files.len() {
        return;
    }
    let name = match files[idx].name.as_deref() {
        Some(n) => n,
        None => return,
    };

    let home = std::env::var("HOME").unwrap_or_default();
    let path = format!("{}/library/youtube/ingested/{}", home, name);

    if !std::path::Path::new(&path).exists() {
        app.error = Some(format!("File not found: {}", path));
        return;
    }

    // Suspend TUI
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();

    // Run less (blocks until user exits)
    let _ = std::process::Command::new("less").arg(&path).status();

    // Restore TUI and force a redraw
    let _ = enable_raw_mode();
    let _ = execute!(terminal.backend_mut(), EnterAlternateScreen);
    let _ = terminal.draw(|f| ui(f, app));
}

// ── Entry point ──

pub fn run(args: &[String]) {
    let url = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "http://localhost:4242".to_string());
    let refresh: f64 = args
        .get(0)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);

    // Setup terminal
    let mut stdout = io::stdout();
    if enable_raw_mode().is_err() {
        eprintln!("Error: monitor requires a real terminal (TTY).");
        eprintln!("Usage: rag-ferrite monitor [refresh_seconds] [url]");
        return;
    }
    execute!(stdout, EnterAlternateScreen).unwrap_or(());
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut app = App::new();
    let fetch_dur = Duration::from_secs_f64(refresh);
    let mut last_fetch = Instant::now() - fetch_dur; // fetch immediately

    loop {
        // Fetch API if needed
        if last_fetch.elapsed() >= fetch_dur {
            match fetch_progress(&url) {
                Ok(data) => {
                    app.data = Some(data);
                    app.error = None;
                }
                Err(e) => {
                    app.error = Some(e);
                }
            }
            last_fetch = Instant::now();
        }

        app.spinner_idx += 1;

        // Render
        terminal.draw(|f| ui(f, &mut app)).unwrap();

        // Poll input (150ms for smooth animation)
        if event::poll(Duration::from_millis(150)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                // Help popup swallows every key except ? and Esc
                if app.show_help {
                    if key.code == KeyCode::Char('?') || key.code == KeyCode::Esc {
                        app.show_help = false;
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Tab => app.focus = app.focus.next(),
                    KeyCode::Down => {
                        let max = app
                            .data
                            .as_ref()
                            .and_then(|d| d.batch.as_ref())
                            .map(|b| b.files.len().saturating_sub(1))
                            .unwrap_or(0);
                        app.scroll_down(max);
                    }
                    KeyCode::Up => app.scroll_up(),
                    KeyCode::Char('l') => app.show_lists = !app.show_lists,
                    KeyCode::Char('s') => app.show_stats = !app.show_stats,
                    KeyCode::Char('c') => {
                        app.color_mode = match app.color_mode {
                            ColorMode::Full => ColorMode::StatsOnly,
                            ColorMode::StatsOnly => ColorMode::Mono,
                            ColorMode::Mono => ColorMode::Full,
                        };
                    }
                    KeyCode::Char('?') => app.show_help = true,
                    KeyCode::Char('o') | KeyCode::Enter => {
                        if app.focus == Panel::Completed {
                            open_selected_file(&mut app, &mut terminal);
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
