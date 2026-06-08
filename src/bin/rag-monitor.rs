//! rag-monitor — Standalone TUI that connects to a rag-ferrite HTTP API
//!
//! Config via env vars:
//!   RAG_MONITOR_URL     — base URL (default: http://100.97.67.73:4242)
//!   RAG_MONITOR_KEY     — API key, sent as Authorization: Bearer ***
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
struct ActivityEvent {
    #[allow(dead_code)]
    timestamp: u64,
    message: String,
    event_type: String,
}

#[derive(serde::Deserialize, Default)]
struct ActivityLogResponse {
    events: Vec<ActivityEvent>,
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
    #[serde(default)]
    activity_log: ActivityLogResponse,
}

#[derive(serde::Deserialize, Default)]
struct BatchProgress {
    #[allow(dead_code)]
    batch_id: Option<String>,
    #[allow(dead_code)]
    status: Option<String>,
    total_files: usize,
    completed_files: usize,
    #[allow(dead_code)]
    failed_files: usize,
    #[allow(dead_code)]
    total_chunks: usize,
    completed_chunks: usize,
    total_size_mb: Option<f64>,
    speed_chunks_per_min: Option<f64>,
    avg_time_per_file_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
    eta_seconds: Option<f64>,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    chunks_done: Option<usize>,
    #[allow(dead_code)]
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
    show_files: bool,
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
            show_files: false,
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

// ── Activity event color ──

fn event_color(event_type: &str) -> Color {
    match event_type {
        "embedding" => Color::Cyan,
        "llm" => Color::Yellow,
        "chunking" => Color::Green,
        "error" => Color::Red,
        _ => Color::Gray,
    }
}

// ── UI ──

fn ui(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let batch = app.progress.as_ref().and_then(|p| p.batch.as_ref());
    let has_batch = batch.is_some();

    // ── Compute section heights ──
    // Top: header(1) + blank(1) + filename(1) + bar(1) + phase(1) + blank(1) + stats(2) + blank(1) = 9
    // Idle: header(1) + idle line(1) = 2
    let top_h: u16 = if has_batch { 9 } else { 2 };
    let fl_h: u16 = if app.show_files && has_batch {
        std::cmp::min(10, size.height.saturating_sub(top_h + 4))
    } else {
        0
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Length(fl_h),
            Constraint::Min(2), // activity log (separator + at least 1 event)
            Constraint::Length(1), // footer
        ])
        .split(size);

    // ══════════════════════════════════════════════════════════════════
    // ── Header (always rendered) ──
    // ══════════════════════════════════════════════════════════════════
    let version = app
        .status
        .as_ref()
        .and_then(|s| s.version.as_deref())
        .unwrap_or("?");
    let doc_count = app
        .status
        .as_ref()
        .and_then(|s| s.document_count)
        .unwrap_or(0);
    let spinner_char = SPINNER[app.spinner_idx % SPINNER.len()];

    // Header uses first line of top section
    let top_lines = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if has_batch {
            vec![
                Constraint::Length(1), // 0: header
                Constraint::Length(1), // 1: blank
                Constraint::Length(1), // 2: file name
                Constraint::Length(1), // 3: progress bar
                Constraint::Length(1), // 4: phase
                Constraint::Length(1), // 5: blank
                Constraint::Length(1), // 6: stats line 1
                Constraint::Length(1), // 7: stats line 2
                Constraint::Length(1), // 8: blank
            ]
        } else {
            vec![
                Constraint::Length(1), // 0: header
                Constraint::Length(1), // 1: idle/error line
            ]
        })
        .split(outer[0]);

    // Header line: rag-ferrite v5.0.0 • 132 docs  [or spinner if batch running]
    {
        let mut header_spans = vec![Span::styled(
            format!(" rag-ferrite v{} • {} docs", version, doc_count),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        if has_batch {
            header_spans.push(Span::styled(
                format!("  {}", spinner_char),
                Style::default().fg(Color::Yellow),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(header_spans)), top_lines[0]);
    }

    if has_batch {
        if let Some(b) = batch {
            // ── File name (bold) ──
            let name = b
                .current_file
                .as_ref()
                .and_then(|cf| cf.name.as_deref())
                .unwrap_or("?");
            let max_name = top_lines[2].width as usize;
            f.render_widget(
                Paragraph::new(Line::from(vec![Span::styled(
                    format!(" {}", truncate(name, max_name.saturating_sub(1))),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )])),
                top_lines[2],
            );

            // ── Progress bar + file progress ──
            let done = b.completed_files;
            let total = b.total_files;
            let pct = if total > 0 {
                done as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            let file_info = if total > 0 {
                format!(" {:>3.0}%  {}/{}", pct, done, total)
            } else {
                "  waiting…".to_string()
            };
            let bar_len = (top_lines[3].width as usize)
                .saturating_sub(3 + file_info.len()) // " [" + "]" + file_info
                .max(10);
            let bar_spans = render_progress_bar(pct, app.spinner_idx, bar_len, &app.pendulum_frames);
            let mut bar_line = vec![Span::raw(" [")];
            bar_line.extend(bar_spans);
            bar_line.push(Span::raw("]"));
            bar_line.push(Span::styled(
                file_info,
                Style::default().fg(Color::White),
            ));
            f.render_widget(Paragraph::new(Line::from(bar_line)), top_lines[3]);

            // ── Phase + chunks + speed ──
            let phase = b
                .current_file
                .as_ref()
                .and_then(|cf| cf.phase.as_deref())
                .unwrap_or("processing");
            let chunks = b.completed_chunks;
            let speed = b.speed_chunks_per_min.unwrap_or(0.0);
            let speed_fmt = if speed >= 1.0 {
                format!("{:.0}", speed)
            } else {
                "—".to_string()
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" {}", phase),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(" • "),
                    Span::styled(
                        format!("{} chunks", chunks),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw(" • "),
                    Span::styled(
                        format!("{}/min", speed_fmt),
                        Style::default().fg(Color::Green),
                    ),
                ])),
                top_lines[4],
            );

            // ── Stats line 1: Speed + Avg/file ──
            let speed = b.speed_chunks_per_min.unwrap_or(0.0);
            let avg_file = b.avg_time_per_file_seconds.unwrap_or(0.0);
            let speed_str = if speed >= 1.0 {
                format!("{:.0}/min", speed)
            } else {
                "—".to_string()
            };
            let avg_str = if avg_file > 0.0 {
                format!("{:.1}s", avg_file)
            } else {
                "—".to_string()
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(" Speed  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:>8}", speed_str),
                        Style::default().fg(Color::White),
                    ),
                    Span::raw("    "),
                    Span::styled("Avg/file  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:>6}", avg_str),
                        Style::default().fg(Color::White),
                    ),
                ])),
                top_lines[6],
            );

            // ── Stats line 2: Elapsed + ETA + Size ──
            let elapsed = fmt_duration(b.elapsed_seconds);
            let eta = fmt_duration(b.eta_seconds);
            let size_mb = b.total_size_mb.unwrap_or(0.0);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(" Elapsed ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:>6}", elapsed),
                        Style::default().fg(Color::White),
                    ),
                    Span::raw("   "),
                    Span::styled("ETA ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:>6}", eta),
                        Style::default().fg(Color::Magenta),
                    ),
                    Span::raw("   "),
                    Span::styled("Size ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:.1} MB", size_mb),
                        Style::default().fg(Color::White),
                    ),
                ])),
                top_lines[7],
            );
        }
    } else {
        // ── Idle / error state ──
        if let Some(e) = &app.error {
            f.render_widget(
                Paragraph::new(Line::from(vec![Span::styled(
                    format!(" ⚠ {}", e),
                    Style::default().fg(Color::Red),
                )])),
                top_lines[1],
            );
        } else if app.progress.is_some() {
            f.render_widget(
                Paragraph::new(Line::from(vec![Span::styled(
                    " Idle — no batch running",
                    Style::default().fg(Color::DarkGray),
                )])),
                top_lines[1],
            );
        } else {
            f.render_widget(
                Paragraph::new(Line::from(vec![Span::styled(
                    format!(" {} Connecting…", spinner_char),
                    Style::default().fg(Color::Yellow),
                )])),
                top_lines[1],
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // ── File lists (toggled with 'l') ──
    // ══════════════════════════════════════════════════════════════════
    if app.show_files && has_batch && fl_h > 0 {
        let list_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(outer[1]);

        // ── Completed files ──
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
                            " {:<36} {:>4}ch {:>6.1}s",
                            truncate(name, 36),
                            chunks,
                            dur
                        );
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("{} ", status_icon),
                                Style::default().fg(color_raw),
                            ),
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

        // ── Queue (pending files) ──
        let queue_items: Vec<ListItem> = batch
            .map(|b| {
                b.pending_files
                    .iter()
                    .map(|name| {
                        ListItem::new(Line::from(vec![Span::styled(
                            format!(" {}", truncate(name, 36)),
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
        f.render_stateful_widget(queue_list, list_area[1], &mut queue_state);
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Activity log ──
    // ══════════════════════════════════════════════════════════════════
    {
        let act_area = outer[2];
        if act_area.height < 1 {
            // skip if no space
        } else {
            let act_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1), // separator
                    Constraint::Min(0),    // events
                ])
                .split(act_area);

            // Separator: "── Activity ──────────────────────────"
            let label = " Activity ";
            let sep_total = act_layout[0].width as usize;
            let sep_dashes = sep_total.saturating_sub(label.len() + 1); // " " + label + dashes
            let sep_line = format!(
                " {}{}{}",
                "─".repeat(2),
                label,
                "─".repeat(sep_dashes.saturating_sub(2)),
            );
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    sep_line,
                    Style::default().fg(Color::DarkGray),
                ))),
                act_layout[0],
            );

            // Events
            let events = app
                .progress
                .as_ref()
                .map(|p| &p.activity_log.events)
                .map(|e| e.as_slice())
                .unwrap_or(&[]);

            let max_events = act_layout[1].height as usize;
            let start = events.len().saturating_sub(max_events);

            if events.is_empty() {
                f.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        " No activity",
                        Style::default().fg(Color::DarkGray),
                    ))),
                    act_layout[1],
                );
            } else {
                let lines: Vec<Line> = events[start..]
                    .iter()
                    .map(|ev| {
                        let color = event_color(&ev.event_type);
                        Line::from(vec![
                            Span::styled(" → ", Style::default().fg(color)),
                            Span::styled(&ev.message, Style::default().fg(color)),
                        ])
                    })
                    .collect();
                f.render_widget(Paragraph::new(lines), act_layout[1]);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Footer ──
    // ══════════════════════════════════════════════════════════════════
    {
        let footer_area = outer[3];
        // Action feedback overlay (5 seconds)
        if let Some((msg, ts)) = &app.action_msg {
            if ts.elapsed() < Duration::from_secs(5) {
                let color = if msg.starts_with('✓') {
                    Color::Green
                } else {
                    Color::Red
                };
                f.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        msg.clone(),
                        Style::default().fg(color),
                    ))),
                    footer_area,
                );
            } else {
                app.action_msg = None;
            }
        }
        if app.action_msg.is_none() {
            let k = Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD);
            let d = Style::default().fg(Color::DarkGray);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("[c]", k),
                    Span::styled("ancel ", d),
                    Span::styled("[r]", k),
                    Span::styled("ebuild ", d),
                    Span::styled("[f]", k),
                    Span::styled("lush ", d),
                    Span::styled("[x]", k),
                    Span::styled("top ", d),
                    Span::styled("[l]", k),
                    Span::styled("files ", d),
                    Span::styled("[?]", k),
                    Span::styled("help ", d),
                    Span::styled("[q]", k),
                    Span::styled("uit", d),
                ])),
                footer_area,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // ── Help popup ──
    // ══════════════════════════════════════════════════════════════════
    if app.show_help {
        let area = centered_rect(50, 55, f.area());
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
                Span::styled("  l      ", hk),
                Span::styled("Toggle file lists", hd),
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
                "  Actions:",
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![
                Span::styled("  c      ", hk),
                Span::styled("Cancel batch", hd),
            ]),
            Line::from(vec![
                Span::styled("  x      ", hk),
                Span::styled("Stop server", hd),
            ]),
            Line::from(vec![
                Span::styled("  r      ", hk),
                Span::styled("Rebuild indexes", hd),
            ]),
            Line::from(vec![
                Span::styled("  f      ", hk),
                Span::styled("Flush indexes", hd),
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
    // Config from env, falling back to same locations as CLI `rag`
    let base_url =
        std::env::var("RAG_MONITOR_URL").unwrap_or_else(|_| "http://100.97.67.73:4242".to_string());
    let api_key = std::env::var("RAG_MONITOR_KEY")
        .ok()
        .or_else(|| std::env::var("RAG_API_KEY").ok())
        .or_else(|| std::env::var("RAG_API_KEY_NOVA").ok())
        .or_else(|| {
            let path = std::path::PathBuf::from(
                std::env::var("HOME").unwrap_or_else(|_| "/home/loops".to_string()),
            )
            .join(".config/rag/api_key_nova");
            std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
        });
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
                    KeyCode::Char('l') => {
                        app.show_files = !app.show_files;
                    }
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
                            Ok(msg) => {
                                app.action_msg =
                                    Some((format!("✓ Cancel: {}", msg), Instant::now()))
                            }
                            Err(e) => {
                                app.action_msg =
                                    Some((format!("✗ Cancel failed: {}", e), Instant::now()))
                            }
                        }
                    }
                    KeyCode::Char('x') => {
                        match post_action(&base_url, api_key.as_deref(), "/api/service/stop") {
                            Ok(msg) => {
                                app.action_msg =
                                    Some((format!("✓ Stop: {}", msg), Instant::now()))
                            }
                            Err(e) => {
                                app.action_msg =
                                    Some((format!("✗ Stop failed: {}", e), Instant::now()))
                            }
                        }
                    }
                    KeyCode::Char('r') => {
                        match post_action(
                            &base_url,
                            api_key.as_deref(),
                            "/api/rebuild-indexes",
                        ) {
                            Ok(msg) => {
                                app.action_msg =
                                    Some((format!("✓ Rebuild: {}", msg), Instant::now()))
                            }
                            Err(e) => {
                                app.action_msg =
                                    Some((format!("✗ Rebuild failed: {}", e), Instant::now()))
                            }
                        }
                    }
                    KeyCode::Char('f') => {
                        match post_action(
                            &base_url,
                            api_key.as_deref(),
                            "/api/flush-indexes",
                        ) {
                            Ok(msg) => {
                                app.action_msg =
                                    Some((format!("✓ Flush: {}", msg), Instant::now()))
                            }
                            Err(e) => {
                                app.action_msg =
                                    Some((format!("✗ Flush failed: {}", e), Instant::now()))
                            }
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
