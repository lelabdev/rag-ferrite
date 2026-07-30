//! UI rendering: all ratatui drawing, progress bar, help popup, folder maps.

use std::char;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
};

use super::{App, ColorMode, Panel, SPINNER, View};

// ── Pendulum frames (ported from unicode_animations Dart package) ──
pub(super) fn generate_pendulum_frames() -> Vec<String> {
    // Width matches bar_len directly (1 frame char = 1 bar cell)
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

/// Resolve a "full" color to the color to actually render, respecting `color_mode`.
/// `Full` and `StatsOnly` keep the original color (bar is overridden separately
/// inside `render_progress_bar`). `Mono` collapses everything to Gray.
pub(super) fn color_for(mode: ColorMode, full_color: Color) -> Color {
    match mode {
        ColorMode::Full | ColorMode::StatsOnly => full_color,
        ColorMode::Mono => Color::Gray,
    }
}

/// Truncate a string to `max` characters, appending an ellipsis if needed.
pub(super) fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

/// Extract the parent folder from a file path (e.g. "@AlexHormozi/video.txt" → "@AlexHormozi").
pub(super) fn parent_folder(path: &str) -> &str {
    match path.rfind('/') {
        Some(idx) => &path[..idx],
        None => "",
    }
}

/// Folder group stats: (ok_count, failed_count, pending_count)
pub(super) struct FolderStats {
    pub ok: usize,
    pub failed: usize,
    pub pending: usize,
}

/// Build a BTreeMap of folder → stats from completed files and pending files.
pub(super) fn build_folder_map(
    completed_files: &[super::api::FileResult],
    pending_files: &[String],
) -> std::collections::BTreeMap<String, FolderStats> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, FolderStats> = BTreeMap::new();

    for f in completed_files {
        let name = f.name.as_deref().unwrap_or("?");
        let folder = if name.contains('/') {
            parent_folder(name).to_string()
        } else {
            "(root)".to_string()
        };
        let entry = map.entry(folder).or_insert(FolderStats {
            ok: 0,
            failed: 0,
            pending: 0,
        });
        match f.status.as_deref() {
            Some("ok") | Some("completed") => entry.ok += 1,
            Some("error") | Some("failed") => entry.failed += 1,
            _ => {}
        }
    }

    for pf in pending_files {
        let folder = if pf.contains('/') {
            parent_folder(pf).to_string()
        } else {
            "(root)".to_string()
        };
        let entry = map.entry(folder).or_insert(FolderStats {
            ok: 0,
            failed: 0,
            pending: 0,
        });
        entry.pending += 1;
    }

    map
}

// ── Progress bar renderer — returns colored Spans ──

fn render_progress_bar<'a>(
    pct: f64,
    spinner_idx: usize,
    bar_len: usize,
    color_mode: ColorMode,
    pendulum_frames: &[String],
    fade_len: usize,
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
    let fade_chars: Vec<&str> = match fade_len {
        0 => vec![],
        2 => vec!["▓", "░"],
        3 => vec!["▓", "▒", "░"],
        4 => vec!["▓", "▓", "░", "░"],
        _ => vec!["▓", "▓", "▓", "░", "░"],
    };
    let num_fill = fill_chars.len();
    let num_fade = fade_chars.len();
    let states_per_cell = num_fill + num_fade; // 13
    let total_states = bar_len * states_per_cell;
    let current_state = (pct / 100.0 * total_states as f64).round() as usize;
    let front_cell = current_state / states_per_cell;
    let cell_remainder = current_state % states_per_cell;

    let empty_start = front_cell + 1;

    let mut spans: Vec<Span<'a>> = Vec::with_capacity(bar_len);

    for i in 0..bar_len {
        if i < front_cell {
            // Full — but fade last 5 cells before frontier
            let dist_to_front = front_cell.saturating_sub(i);
            if dist_to_front <= num_fade {
                let fade_idx = num_fade - dist_to_front;
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(bar_color(Color::Cyan)),
                ));
            } else {
                spans.push(Span::styled(
                    "█",
                    Style::default().fg(bar_color(Color::Cyan)),
                ));
            }
        } else if i == front_cell && cell_remainder > 0 {
            if cell_remainder <= num_fill {
                spans.push(Span::styled(
                    fill_chars[cell_remainder - 1],
                    Style::default().fg(bar_color(Color::Cyan)),
                ));
            } else {
                let fade_idx = cell_remainder - num_fill - 1;
                spans.push(Span::styled(
                    fade_chars[fade_idx],
                    Style::default().fg(bar_color(Color::Cyan)),
                ));
            }
        } else if i >= empty_start {
            // Pendulum — braille wave from unicode_animations package
            let empty_count = bar_len - empty_start;
            if empty_count > 0 && !pendulum_frames.is_empty() {
                let rel = i - empty_start;
                let frame_idx = spinner_idx % pendulum_frames.len();
                let frame = &pendulum_frames[frame_idx];
                let frame_chars: Vec<char> = frame.chars().collect();
                let ch = frame_chars.get(rel).copied().unwrap_or(' ');

                // Color: all cyan
                let color = Color::Cyan;
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(bar_color(color)),
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

pub(super) fn ui(f: &mut Frame, app: &mut App) {
    if app.view != View::Dashboard {
        render_workspace(f, app);
        return;
    }
    let size = f.area();
    let data = app.data.as_ref();
    let cm = app.color_mode;

    // Layout: header (bar+stats) | lists
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12), // header
            if app.show_lists {
                Constraint::Min(10)
            } else {
                Constraint::Length(0)
            }, // lists
            Constraint::Length(1), // footer
        ])
        .split(size);

    // ── Header ──
    let header = Block::default().borders(Borders::TOP).title(Span::styled(
        format!(" rag-ferrite · {} ", app.view.label()),
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
        "completed" => Span::styled("✓ DONE", Style::default().fg(color_for(cm, Color::Green))),
        "failed" => Span::styled("✗ FAILED", Style::default().fg(color_for(cm, Color::Red))),
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
            if total > 0 {
                format!("    {:.0}% ({}/{})", pct, done, total)
            } else {
                match status_str {
                    "queued" => "    queued…".to_string(),
                    _ => "    waiting…".to_string(),
                }
            },
            Style::default().fg(color_for(cm, Color::White)),
        ),
    ]);
    f.render_widget(Paragraph::new(status_line), inner[0]);

    // Progress bar — only show when a batch is running
    if batch.is_some() {
        let bar_spans = render_progress_bar(
            pct,
            app.spinner_idx,
            50,
            cm,
            &app.pendulum_frames,
            app.fade_len,
        );
        let mut bar_line_spans = vec![Span::raw("  [")];
        bar_line_spans.extend(bar_spans);
        bar_line_spans.push(Span::raw("]"));
        f.render_widget(Paragraph::new(Line::from(bar_line_spans)), inner[1]);
    } else {
        // Idle: show connected status
        let idle_line = Line::from(vec![Span::styled(
            "  idle — no batch running",
            Style::default().fg(color_for(cm, Color::DarkGray)),
        )]);
        f.render_widget(Paragraph::new(idle_line), inner[1]);
    }

    // Stats
    if app.show_stats {
        if let Some(b) = batch {
            let speed = b.speed_chunks_per_min.unwrap_or(0.0);
            let avg_file = b.avg_time_per_file_seconds.unwrap_or(0.0);
            let elapsed = super::api::fmt_duration(b.elapsed_seconds);
            let eta = super::api::fmt_duration(b.eta_seconds);
            let err_count = b.failed_files;
            let err_rate = b.error_rate.unwrap_or(0.0);
            let size_mb = b.total_size_mb.unwrap_or(0.0);
            let chunks_done = b.completed_chunks;
            let _chunks_total = b.total_chunks;

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
                    Span::raw("          "),
                    Span::styled(
                        format!("Size      {:>8.1} MB", size_mb),
                        Style::default().fg(color_for(cm, Color::White)),
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
                        Style::default().fg(color_for(cm, Color::White)),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("  Elapsed  {:>6}", elapsed),
                        Style::default().fg(color_for(cm, Color::White)),
                    ),
                    Span::raw("       "),
                    Span::styled(
                        format!("ETA       {:>8}", eta),
                        Style::default().fg(color_for(cm, Color::Magenta)),
                    ),
                ]),
                Line::from(vec![Span::styled(
                    format!("  Errors   {:>6} ({:.1}%)", err_count, err_rate),
                    Style::default().fg(color_for(
                        cm,
                        if err_count > 0 {
                            Color::Red
                        } else {
                            Color::White
                        },
                    )),
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

    // Completed files (or folder groups)
    let completed_items: Vec<ListItem> = if app.folder_view {
        if let Some(ref expanded) = app.expanded_folder {
            // Expanded folder: show individual completed files for this folder
            batch
                .map(|b| {
                    b.files
                        .iter()
                        .rev()
                        .filter(|f| {
                            let name = f.name.as_deref().unwrap_or("?");
                            let folder = if name.contains('/') {
                                parent_folder(name)
                            } else {
                                ""
                            };
                            folder == expanded.as_str()
                        })
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
                            let file_name = name.rfind('/').map(|i| &name[i + 1..]).unwrap_or(name);
                            let text = format!(
                                " {:<48} {:>4}ch {:>6.1}s",
                                truncate(file_name, 48),
                                chunks,
                                dur
                            );
                            ListItem::new(Line::from(vec![
                                Span::styled(
                                    format!("{} ", status_icon),
                                    Style::default().fg(color_for(cm, color_raw)),
                                ),
                                Span::styled(text, Style::default().fg(color_for(cm, color_raw))),
                            ]))
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            // Folder view: group by parent directory with combined stats
            let folder_map = batch
                .map(|b| build_folder_map(&b.files, &b.pending_files))
                .unwrap_or_default();
            folder_map
                .iter()
                .map(|(folder, stats)| {
                    let total = stats.ok + stats.failed + stats.pending;
                    let done = stats.ok + stats.failed;
                    let icon = if stats.failed > 0 {
                        "✗"
                    } else if stats.pending > 0 {
                        "⏳"
                    } else {
                        "✓"
                    };
                    let color = if stats.failed > 0 {
                        Color::Red
                    } else if stats.pending > 0 {
                        Color::Yellow
                    } else {
                        Color::Green
                    };
                    let text = format!(
                        " {:<28} {:>3}/{:<3} {}",
                        truncate(folder, 28),
                        done,
                        total,
                        icon
                    );
                    ListItem::new(Line::from(vec![Span::styled(
                        text,
                        Style::default().fg(color_for(cm, color)),
                    )]))
                })
                .collect()
        }
    } else {
        // Normal view: individual files
        batch
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
                        let text =
                            format!(" {:<48} {:>4}ch {:>6.1}s", truncate(name, 48), chunks, dur);
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("{} ", status_icon),
                                Style::default().fg(color_for(cm, color_raw)),
                            ),
                            Span::styled(text, Style::default().fg(color_for(cm, color_raw))),
                        ]))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let completed_title = if app.folder_view && app.expanded_folder.is_none() {
        let folder_count = batch
            .map(|b| build_folder_map(&b.files, &b.pending_files).len())
            .unwrap_or(0);
        format!(" Folders ({}) ", folder_count)
    } else {
        format!(
            " Completed ({}) ",
            batch.map(|b| b.completed_files).unwrap_or(0)
        )
    };
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
    let mut completed_state = ListState::default();
    if app.focus == Panel::Completed && !completed_items.is_empty() {
        if app.folder_view && app.expanded_folder.is_none() {
            completed_state.select(Some(app.scroll_folders_completed));
        } else {
            completed_state.select(Some(app.scroll_completed));
        }
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
        Style::default()
            .fg(color_for(
                cm,
                if app.focus == Panel::Current {
                    Color::Cyan
                } else {
                    Color::Yellow
                },
            ))
            .add_modifier(if app.focus == Panel::Current {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
    ));
    f.render_widget(
        Paragraph::new(current_lines).block(current_block),
        right_chunks[0],
    );

    // Queue — scrollable list of pending files (or folder groups)
    let queue_items: Vec<ListItem> = if app.folder_view {
        if let Some(ref expanded) = app.expanded_folder {
            // Expanded folder: show individual pending files for this folder
            batch
                .map(|b| {
                    b.pending_files
                        .iter()
                        .filter(|pf| {
                            let folder = if pf.contains('/') {
                                parent_folder(pf)
                            } else {
                                ""
                            };
                            folder == expanded.as_str()
                        })
                        .map(|pf| {
                            let file_name =
                                pf.rfind('/').map(|i| &pf[i + 1..]).unwrap_or(pf.as_str());
                            ListItem::new(Line::from(vec![Span::styled(
                                format!(" {}", truncate(file_name, 42)),
                                Style::default().fg(color_for(cm, Color::Gray)),
                            )]))
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            // Folder view: group pending files by parent directory
            let mut pending_folders: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            if let Some(b) = batch {
                for pf in &b.pending_files {
                    let folder = if pf.contains('/') {
                        parent_folder(pf).to_string()
                    } else {
                        "(root)".to_string()
                    };
                    *pending_folders.entry(folder).or_insert(0) += 1;
                }
            }
            pending_folders
                .iter()
                .map(|(folder, count)| {
                    let text = format!(" {:<28} {:>3} ⏳", truncate(folder, 28), count);
                    ListItem::new(Line::from(vec![Span::styled(
                        text,
                        Style::default().fg(color_for(cm, Color::Yellow)),
                    )]))
                })
                .collect()
        }
    } else {
        // Normal view: individual pending files
        let pending_files: Vec<&String> = batch
            .map(|b| b.pending_files.iter().collect())
            .unwrap_or_default();
        pending_files
            .iter()
            .map(|name| {
                ListItem::new(Line::from(vec![Span::styled(
                    format!(" {}", truncate(name, 42)),
                    Style::default().fg(color_for(cm, Color::Gray)),
                )]))
            })
            .collect()
    };

    let queue_title = if app.folder_view && app.expanded_folder.is_none() {
        let pending_folders: std::collections::BTreeMap<String, usize> = batch
            .map(|b| {
                let mut m = std::collections::BTreeMap::new();
                for pf in &b.pending_files {
                    let folder = if pf.contains('/') {
                        parent_folder(pf).to_string()
                    } else {
                        "(root)".to_string()
                    };
                    *m.entry(folder).or_insert(0) += 1;
                }
                m
            })
            .unwrap_or_default();
        format!(" Queue Folders ({}) ", pending_folders.len())
    } else {
        let pending_count = batch.map(|b| b.pending_files.len()).unwrap_or(0);
        format!(" Queue ({}) ", pending_count)
    };
    let queue_block = Block::default().borders(Borders::ALL).title(Span::styled(
        queue_title,
        Style::default()
            .fg(color_for(
                cm,
                if app.focus == Panel::Queue {
                    Color::Cyan
                } else {
                    Color::DarkGray
                },
            ))
            .add_modifier(if app.focus == Panel::Queue {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
    ));

    let mut queue_state = ListState::default();
    if app.focus == Panel::Queue && !queue_items.is_empty() {
        if app.folder_view && app.expanded_folder.is_none() {
            queue_state.select(Some(app.scroll_folders_queue));
        } else {
            queue_state.select(Some(app.scroll_pending));
        }
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
        .fg(color_for(cm, Color::Cyan))
        .add_modifier(Modifier::BOLD);
    let dim_style = Style::default().fg(color_for(cm, Color::DarkGray));
    let footer = Line::from(vec![
        Span::styled("1-5", key_style),
        Span::styled(" views ", dim_style),
        Span::styled("•", dim_style),
        Span::styled("TAB", key_style),
        Span::styled(" panels ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" ↑↓", key_style),
        Span::styled(" scroll ", dim_style),
        Span::styled("•", dim_style),
        Span::styled(" g", key_style),
        Span::styled(" folders ", dim_style),
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
            Line::from(vec![
                Span::styled("  TAB    ", hk),
                Span::styled("Switch panel", hd),
            ]),
            Line::from(vec![
                Span::styled("  ↑ ↓    ", hk),
                Span::styled("Scroll list", hd),
            ]),
            Line::from(vec![
                Span::styled("  g      ", hk),
                Span::styled("Toggle folder grouping view", hd),
            ]),
            Line::from(vec![
                Span::styled("  Enter→ ", hk),
                Span::styled("Expand folder / open file in less", hd),
            ]),
            Line::from(vec![
                Span::styled("  ← Esc  ", hk),
                Span::styled("Collapse folder (in folder view)", hd),
            ]),
            Line::from(vec![
                Span::styled("  l      ", hk),
                Span::styled("Toggle lists visibility", hd),
            ]),
            Line::from(vec![
                Span::styled("  c      ", hk),
                Span::styled("Cycle color modes (Full → StatsOnly → Mono)", hd),
            ]),
            Line::from(vec![
                Span::styled("  s      ", hk),
                Span::styled("Toggle stats", hd),
            ]),
            Line::from(vec![
                Span::styled("  o      ", hk),
                Span::styled("Open selected file in less (Completed/Queue)", hd),
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
            Line::from(vec![
                Span::styled("  C      ", hk),
                Span::styled("Cancel batch", hd),
            ]),
            Line::from(vec![
                Span::styled("  ?      ", hk),
                Span::styled("Toggle this help", hd),
            ]),
            Line::from(vec![
                Span::styled("  q/Esc  ", hk),
                Span::styled("Quit", hd),
            ]),
        ];

        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(help_lines).block(block), area);
    }
}

/// Compute a centered rect of `percent_x`% width and `percent_y`% height inside `r`.
fn render_workspace(f: &mut Frame, app: &App) {
    let title = format!(" {} ", app.view.label());
    let block = Block::default().borders(Borders::ALL).title(title);
    let area = f.area();
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines: Vec<Line> = match app.view {
        View::Library => {
            if app.documents.is_empty() {
                vec![Line::from("No documents loaded. Press r to refresh.")]
            } else {
                app.documents
                    .iter()
                    .enumerate()
                    .map(|(index, document)| {
                        Line::from(format!(
                            "{}#{:<6} {:<42} {:<18} {}",
                            if index == app.library_cursor {
                                "▶ "
                            } else {
                                "  "
                            },
                            document.id,
                            document.name.as_deref().unwrap_or("(unnamed)"),
                            document.collection_id.as_deref().unwrap_or("-"),
                            document.status.as_deref().unwrap_or("-")
                        ))
                    })
                    .collect()
            }
        }
        View::Query => {
            if app.query_results.is_empty() {
                vec![Line::from("Query workspace — press Q to enter a query")]
            } else {
                app.query_results
                    .iter()
                    .map(|line| Line::from(line.clone()))
                    .collect()
            }
        }
        View::Ingest => vec![Line::from(
            "Ingestion workspace — press i to submit a file to the shared queue",
        )],
        View::Admin => vec![Line::from(
            "Administration workspace — e edits config, r rebuilds, f flushes, C cancels",
        )],
        View::Dashboard => Vec::new(),
    };
    f.render_widget(Paragraph::new(lines), inner);
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
