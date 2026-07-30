//! rag-ferrite batch monitor — TUI with ratatui
//!
//! Usage: rag-ferrite monitor [refresh_seconds] [url]
//!
//! Environment variables (legacy names, prefer RAGFER_*):
//!   RAGFER_URL      — base URL (overrides client config)
//!   RAGFER_KEY      — API key, sent as Authorization: Bearer ***
//!   RAGFER_REFRESH  — poll interval in seconds (default: 2)

mod api;
mod ui;

use std::collections::BTreeMap;
use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use chrono::Local;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use api::{
    BatchProgress, CurrentFile, Document, FileResult, ProgressResponse, fetch_documents,
    fetch_progress, post_action,
};
use ui::{build_folder_map, generate_pendulum_frames, parent_folder, ui};

// ── Braille spinner (10 frames) ──
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── App state ──

#[derive(Clone, Copy, PartialEq)]
enum View {
    Dashboard,
    Library,
    Query,
    Ingest,
    Admin,
}

impl View {
    fn label(self) -> &'static str {
        match self {
            View::Dashboard => "Dashboard",
            View::Library => "Library",
            View::Query => "Query",
            View::Ingest => "Ingest",
            View::Admin => "Admin",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Completed,
    Current,
    Queue,
}

impl Panel {
    fn next(self) -> Self {
        match self {
            Panel::Completed => Panel::Current,
            Panel::Current => Panel::Queue,
            Panel::Queue => Panel::Completed,
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
    documents: Vec<Document>,
    error: Option<String>,
    view: View,
    focus: Panel,
    scroll_completed: usize,
    scroll_pending: usize,
    spinner_idx: usize,
    pendulum_frames: Vec<String>,
    fade_len: usize,
    show_lists: bool,
    show_stats: bool,
    show_help: bool,
    color_mode: ColorMode,
    folder_view: bool,
    expanded_folder: Option<String>,
    scroll_folders_completed: usize,
    scroll_folders_queue: usize,
    action_msg: Option<(String, Instant)>,
    tick_count: usize,
}

impl App {
    fn new() -> Self {
        Self {
            data: None,
            documents: Vec::new(),
            error: None,
            view: View::Dashboard,
            focus: Panel::Completed,
            scroll_completed: 0,
            scroll_pending: 0,
            spinner_idx: 0,
            pendulum_frames: generate_pendulum_frames(),
            fade_len: 5,
            show_lists: true,
            show_stats: true,
            show_help: false,
            color_mode: ColorMode::Full,
            folder_view: false,
            expanded_folder: None,
            scroll_folders_completed: 0,
            scroll_folders_queue: 0,
            action_msg: None,
            tick_count: 0,
        }
    }
}

/// Open the currently-selected file in `less`. Suspends the TUI
/// (leaves alternate screen + disables raw mode) while less runs, then
/// restores it and forces a full redraw.
///
/// Behaviour depends on the focused panel:
/// - `Completed`: opens `~/library/youtube/ingested/{name}`
/// - `Queue`:     opens `~/library/youtube/inbox/{name}`
fn open_selected_file(app: &mut App, terminal: &mut Terminal<CrosstermBackend<Stdout>>) {
    let batch = match app.data.as_ref().and_then(|d| d.batch.as_ref()) {
        Some(b) => b,
        None => return,
    };

    let home = std::env::var("HOME").unwrap_or_default();
    let (base_dir, name) = match app.focus {
        Panel::Completed => {
            let files = &batch.files;
            if files.is_empty() {
                return;
            }
            let idx = files
                .len()
                .saturating_sub(1)
                .saturating_sub(app.scroll_completed);
            if idx >= files.len() {
                return;
            }
            let name = match files[idx].name.as_deref() {
                Some(n) => n.to_string(),
                None => return,
            };
            (format!("{}/library/youtube/ingested", home), name)
        }
        Panel::Queue => {
            let pending = &batch.pending_files;
            if pending.is_empty() {
                return;
            }
            let idx = app.scroll_pending;
            if idx >= pending.len() {
                return;
            }
            let name = pending[idx].clone();
            (format!("{}/library/youtube/inbox", home), name)
        }
        Panel::Current => return,
    };

    // Search for the file recursively in subdirectories
    let find_output = std::process::Command::new("find")
        .arg(&base_dir)
        .arg("-name")
        .arg(&name)
        .arg("-type")
        .arg("f")
        .output();

    let path = match find_output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let found = stdout.lines().next();
            match found {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => {
                    app.error = Some(format!("File not found: {}", name));
                    return;
                }
            }
        }
        _ => {
            app.error = Some(format!("Search failed: {}", name));
            return;
        }
    };

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
    // Load .env from the binary's directory (same folder as the server's .env)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let env_path = dir.join(".env");
            let _ = dotenvy::from_path(&env_path);
        }
    }

    // Check for --demo flag
    let demo_mode = args.iter().any(|a| a == "--demo" || a == "demo");

    // Parse --fade N (default 5)
    let fade_len: usize = {
        let mut f = 5;
        let mut iter = args.iter();
        while let Some(a) = iter.next() {
            if a == "--fade" {
                if let Some(val) = iter.next() {
                    if let Ok(n) = val.parse::<usize>() {
                        f = n;
                    }
                }
            }
        }
        f
    };

    let non_flag_args: Vec<&String> = args
        .iter()
        .filter(|a| *a != "--demo" && *a != "demo" && *a != "--fade" && !a.starts_with("monitor"))
        .collect();
    // Use client config for URL, allow override via positional arg
    let default_url = crate::client::get_server_url();
    let url = non_flag_args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or(&default_url);
    let refresh: f64 = non_flag_args
        .get(0)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);

    // Setup terminal
    let mut stdout = io::stdout();
    if enable_raw_mode().is_err() {
        eprintln!("Error: monitor requires a real terminal (TTY).");
        eprintln!("Usage: rag-ferrite monitor [refresh_seconds] [url] [--demo] [--fade N]");
        return;
    }
    execute!(stdout, EnterAlternateScreen).unwrap_or(());
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut app = App::new();
    app.fade_len = fade_len;
    let fetch_dur = Duration::from_secs_f64(refresh);
    let mut last_fetch = Instant::now() - fetch_dur; // fetch immediately
    let mut demo_pct: f64 = 0.0;

    loop {
        if demo_mode {
            // Fake progress: slowly fill from 0 to 100, then reset
            demo_pct += 0.3;
            if demo_pct > 100.0 {
                demo_pct = 0.0;
            }
            let done = (demo_pct / 100.0 * 220.0).round() as usize;
            let demo_folders = [
                "@AlexHormozi",
                "@CodieSanchezCT",
                "@naval",
                "@SahilBloom",
                "@TheKitchen",
            ];
            // Generate demo files spread across folders
            let mut demo_files: Vec<FileResult> = Vec::new();
            let mut demo_pending: Vec<String> = Vec::new();
            for i in 0..220 {
                let folder = demo_folders[i % demo_folders.len()];
                let file_name = format!("{}/video_{:03}.txt", folder, i);
                if i < done {
                    let is_error = done > 50 && done < 55 && (i % 20 == 0);
                    demo_files.push(FileResult {
                        name: Some(file_name),
                        chunks: Some(if is_error { 0 } else { 8 + i % 5 }),
                        size_mb: Some(1.2 + (i as f64 % 3.0)),
                        duration_seconds: Some(if is_error {
                            0.1
                        } else {
                            3.0 + (i as f64 % 5.0)
                        }),
                        status: Some(if is_error {
                            "error".into()
                        } else {
                            "ok".into()
                        }),
                    });
                } else {
                    demo_pending.push(file_name);
                }
            }
            app.data = Some(ProgressResponse {
                status: Some("running".into()),
                current_source: Some(format!("demo_file_{:03}.txt", done % 220)),
                last_error: None,
                batch: Some(BatchProgress {
                    batch_id: Some("demo-batch".into()),
                    status: Some("running".into()),
                    total_files: 220,
                    completed_files: done,
                    failed_files: if done > 50 && done < 55 { 2 } else { 0 },
                    total_chunks: done * 850,
                    completed_chunks: done * 800,
                    total_size_mb: Some(done as f64 * 1.5),
                    speed_chunks_per_min: Some(847.3),
                    avg_time_per_file_seconds: Some(4.2),
                    elapsed_seconds: Some(done as f64 * 4.2),
                    eta_seconds: Some((220 - done) as f64 * 4.2),
                    error_rate: Some(if done > 50 && done < 55 { 3.6 } else { 0.0 }),
                    errors: Vec::new(),
                    current_file: Some(CurrentFile {
                        name: Some(format!(
                            "{}/video_{:03}.txt",
                            demo_folders[(done + 1) % demo_folders.len()],
                            done + 1
                        )),
                        phase: Some("embedding".into()),
                        chunks_done: Some(3),
                        chunks_total: Some(12),
                    }),
                    files: demo_files,
                    pending_files: demo_pending,
                }),
            });
            app.error = None;
        } else if last_fetch.elapsed() >= fetch_dur {
            match fetch_progress(url) {
                Ok(data) => {
                    app.data = Some(data);
                    app.error = None;
                }
                Err(e) => {
                    app.error = Some(e);
                }
            }
            if app.view == View::Library {
                match fetch_documents(url) {
                    Ok(documents) => app.documents = documents,
                    Err(e) => app.error = Some(e),
                }
            }
            last_fetch = Instant::now();
        }

        app.spinner_idx += 1;
        app.tick_count += 1;

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
                    KeyCode::Char('1') => app.view = View::Dashboard,
                    KeyCode::Char('2') => app.view = View::Library,
                    KeyCode::Char('3') => app.view = View::Query,
                    KeyCode::Char('4') => app.view = View::Ingest,
                    KeyCode::Char('5') => app.view = View::Admin,
                    KeyCode::Char('q') => break,
                    KeyCode::Esc => {
                        // Esc collapses folder if expanded, otherwise quits
                        if app.expanded_folder.is_some() {
                            app.expanded_folder = None;
                            app.scroll_completed = 0;
                            app.scroll_pending = 0;
                        } else {
                            break;
                        }
                    }
                    KeyCode::Tab => app.focus = app.focus.next(),
                    KeyCode::Char('g') => {
                        app.folder_view = !app.folder_view;
                        app.expanded_folder = None;
                        app.scroll_completed = 0;
                        app.scroll_pending = 0;
                        app.scroll_folders_completed = 0;
                        app.scroll_folders_queue = 0;
                    }
                    KeyCode::Down => match app.focus {
                        Panel::Completed => {
                            if app.folder_view && app.expanded_folder.is_none() {
                                // Scroll folder list
                                let max = app
                                    .data
                                    .as_ref()
                                    .and_then(|d| d.batch.as_ref())
                                    .map(|b| {
                                        build_folder_map(&b.files, &b.pending_files)
                                            .len()
                                            .saturating_sub(1)
                                    })
                                    .unwrap_or(0);
                                if app.scroll_folders_completed < max {
                                    app.scroll_folders_completed += 1;
                                }
                            } else {
                                let max = if app.folder_view {
                                    // Expanded folder: count filtered files
                                    app.data
                                        .as_ref()
                                        .and_then(|d| d.batch.as_ref())
                                        .map(|b| {
                                            b.files
                                                .iter()
                                                .filter(|f| {
                                                    let name = f.name.as_deref().unwrap_or("?");
                                                    let folder = if name.contains('/') {
                                                        parent_folder(name)
                                                    } else {
                                                        ""
                                                    };
                                                    Some(folder) == app.expanded_folder.as_deref()
                                                })
                                                .count()
                                                .saturating_sub(1)
                                        })
                                        .unwrap_or(0)
                                } else {
                                    app.data
                                        .as_ref()
                                        .and_then(|d| d.batch.as_ref())
                                        .map(|b| b.files.len().saturating_sub(1))
                                        .unwrap_or(0)
                                };
                                if app.scroll_completed < max {
                                    app.scroll_completed += 1;
                                }
                            }
                        }
                        Panel::Queue => {
                            if app.folder_view && app.expanded_folder.is_none() {
                                // Scroll folder list
                                let max = app
                                    .data
                                    .as_ref()
                                    .and_then(|d| d.batch.as_ref())
                                    .map(|b| {
                                        let mut m = BTreeMap::new();
                                        for pf in &b.pending_files {
                                            let folder = if pf.contains('/') {
                                                parent_folder(pf).to_string()
                                            } else {
                                                "(root)".to_string()
                                            };
                                            *m.entry(folder).or_insert(0usize) += 1;
                                        }
                                        m.len().saturating_sub(1)
                                    })
                                    .unwrap_or(0);
                                if app.scroll_folders_queue < max {
                                    app.scroll_folders_queue += 1;
                                }
                            } else {
                                let max = if app.folder_view {
                                    // Expanded folder: count filtered pending files
                                    app.data
                                        .as_ref()
                                        .and_then(|d| d.batch.as_ref())
                                        .map(|b| {
                                            b.pending_files
                                                .iter()
                                                .filter(|pf| {
                                                    let folder = if pf.contains('/') {
                                                        parent_folder(pf)
                                                    } else {
                                                        ""
                                                    };
                                                    Some(folder) == app.expanded_folder.as_deref()
                                                })
                                                .count()
                                                .saturating_sub(1)
                                        })
                                        .unwrap_or(0)
                                } else {
                                    app.data
                                        .as_ref()
                                        .and_then(|d| d.batch.as_ref())
                                        .map(|b| b.pending_files.len().saturating_sub(1))
                                        .unwrap_or(0)
                                };
                                if app.scroll_pending < max {
                                    app.scroll_pending += 1;
                                }
                            }
                        }
                        Panel::Current => {}
                    },
                    KeyCode::Up => match app.focus {
                        Panel::Completed => {
                            if app.folder_view && app.expanded_folder.is_none() {
                                if app.scroll_folders_completed > 0 {
                                    app.scroll_folders_completed -= 1;
                                }
                            } else if app.scroll_completed > 0 {
                                app.scroll_completed -= 1;
                            }
                        }
                        Panel::Queue => {
                            if app.folder_view && app.expanded_folder.is_none() {
                                if app.scroll_folders_queue > 0 {
                                    app.scroll_folders_queue -= 1;
                                }
                            } else if app.scroll_pending > 0 {
                                app.scroll_pending -= 1;
                            }
                        }
                        Panel::Current => {}
                    },
                    KeyCode::Left | KeyCode::Backspace => {
                        // Collapse expanded folder
                        if app.expanded_folder.is_some() {
                            app.expanded_folder = None;
                            app.scroll_completed = 0;
                            app.scroll_pending = 0;
                        }
                    }
                    KeyCode::Right | KeyCode::Enter => {
                        if app.folder_view && app.expanded_folder.is_none() {
                            // Expand the selected folder
                            let folder_name = match app.focus {
                                Panel::Completed => app
                                    .data
                                    .as_ref()
                                    .and_then(|d| d.batch.as_ref())
                                    .map(|b| {
                                        let map = build_folder_map(&b.files, &b.pending_files);
                                        let keys: Vec<&String> = map.keys().collect();
                                        keys.get(app.scroll_folders_completed).map(|k| (*k).clone())
                                    })
                                    .flatten(),
                                Panel::Queue => app
                                    .data
                                    .as_ref()
                                    .and_then(|d| d.batch.as_ref())
                                    .map(|b| {
                                        let mut m = BTreeMap::new();
                                        for pf in &b.pending_files {
                                            let folder = if pf.contains('/') {
                                                parent_folder(pf).to_string()
                                            } else {
                                                "(root)".to_string()
                                            };
                                            *m.entry(folder).or_insert(0usize) += 1;
                                        }
                                        let keys: Vec<String> = m.keys().cloned().collect();
                                        keys.get(app.scroll_folders_queue).cloned()
                                    })
                                    .flatten(),
                                Panel::Current => None,
                            };
                            if let Some(name) = folder_name {
                                app.expanded_folder = Some(name);
                                app.scroll_completed = 0;
                                app.scroll_pending = 0;
                            }
                        } else if !app.folder_view {
                            // Normal mode: open file in less
                            if app.focus == Panel::Completed || app.focus == Panel::Queue {
                                open_selected_file(&mut app, &mut terminal);
                            }
                        }
                    }
                    KeyCode::Char('l') => app.show_lists = !app.show_lists,
                    KeyCode::Char('s') => app.show_stats = !app.show_stats,
                    KeyCode::Char('c') => {
                        app.color_mode = match app.color_mode {
                            ColorMode::Full => ColorMode::StatsOnly,
                            ColorMode::StatsOnly => ColorMode::Mono,
                            ColorMode::Mono => ColorMode::Full,
                        };
                    }
                    KeyCode::Char('C') => {
                        // Cancel batch via server API
                        let msg = match post_action(url, "/api/service/cancel-batch") {
                            Ok(msg) => format!("✓ Cancel: {}", msg),
                            Err(e) => format!("✗ Cancel failed: {}", e),
                        };
                        app.action_msg = Some((msg, Instant::now()));
                    }
                    KeyCode::Char('x') => {
                        // Stop server
                        let msg = match post_action(url, "/api/service/stop") {
                            Ok(msg) => format!("✓ Stop: {}", msg),
                            Err(e) => format!("✗ Stop failed: {}", e),
                        };
                        app.action_msg = Some((msg, Instant::now()));
                    }
                    KeyCode::Char('r') => {
                        // Rebuild indexes
                        let msg = match post_action(url, "/api/rebuild-indexes") {
                            Ok(msg) => format!("✓ Rebuild: {}", msg),
                            Err(e) => format!("✗ Rebuild failed: {}", e),
                        };
                        app.action_msg = Some((msg, Instant::now()));
                    }
                    KeyCode::Char('f') => {
                        // Flush indexes
                        let msg = match post_action(url, "/api/flush-indexes") {
                            Ok(msg) => format!("✓ Flush: {}", msg),
                            Err(e) => format!("✗ Flush failed: {}", e),
                        };
                        app.action_msg = Some((msg, Instant::now()));
                    }
                    KeyCode::Char('o') => {
                        if !app.folder_view
                            && (app.focus == Panel::Completed || app.focus == Panel::Queue)
                        {
                            open_selected_file(&mut app, &mut terminal);
                        }
                    }
                    KeyCode::Char('?') => app.show_help = true,
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
