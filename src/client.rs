/// ragfer client - CLI subcommands that hit the rag-ferrite HTTP API.
/// Config: ~/.config/ragfer/config.toml (url = ...)
/// API key: ~/.config/ragfer/.env or RAG_API_KEY env var

use anyhow::{bail, Result};
use serde_json::Value;
use std::io::{self, Write};
use std::path::PathBuf;

// --- Config ---

fn config_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(format!("{}/.config/ragfer", home))
}
fn config_path() -> PathBuf { config_dir().join("config.toml") }
fn env_path() -> PathBuf { config_dir().join(".env") }

fn get_url() -> Result<String> {
    let path = config_path();
    if !path.exists() { run_setup()?; }
    let content = std::fs::read_to_string(&path)?;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        if let Some(rest) = line.strip_prefix("url") {
            let val = rest.trim().trim_start_matches('=').trim();
            let val = val.trim_matches(|c: char| c == char::from(34));
            if !val.is_empty() { return Ok(val.to_string()); }
        }
    }
    bail!("No 'url' in {}. Run 'ragfer setup'.", path.display())
}

fn get_api_key() -> Result<String> {
    if let Ok(key) = std::env::var("RAG_API_KEY") {
        if !key.is_empty() { return Ok(key); }
    }
    let ef = env_path();
    if ef.exists() {
        if let Ok(contents) = std::fs::read_to_string(&ef) {
            for line in contents.lines() {
                if let Some(val) = line.strip_prefix("RAG_API_KEY=") {
                    let key = val.trim().trim_matches(|c: char| c == char::from(34)).to_string();
                    if !key.is_empty() { return Ok(key); }
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let local = dir.join(".env");
            if local.exists() {
                if let Ok(contents) = std::fs::read_to_string(&local) {
                    for line in contents.lines() {
                        if let Some(val) = line.strip_prefix("RAG_API_KEY=") {
                            let key = val.trim().trim_matches(|c: char| c == char::from(34)).to_string();
                            if !key.is_empty() { return Ok(key); }
                        }
                    }
                }
            }
        }
    }
    bail!("No API key. Set RAG_API_KEY or run 'ragfer setup'.")
}

fn run_setup() -> Result<()> {
    eprintln!("ragfer - first run setup\n");
    std::fs::create_dir_all(config_dir())?;
    eprint!("Server URL [http://localhost:4242]: ");
    io::stderr().flush()?;
    let mut url = String::new();
    io::stdin().read_line(&mut url)?;
    let url = url.trim();
    let url = if url.is_empty() { "http://localhost:4242" } else { url };
    std::fs::write(config_path(), format!("# ragfer client config\nurl = \"{}\"\n", url))?;
    eprintln!("Config written to {}", config_path().display());
    eprint!("API key (leave empty to skip): ");
    io::stderr().flush()?;
    let mut key = String::new();
    io::stdin().read_line(&mut key)?;
    let key = key.trim();
    if !key.is_empty() {
        std::fs::write(env_path(), format!("RAG_API_KEY={}", key))?;
        eprintln!("API key written to {}", env_path().display());
    } else {
        eprintln!("Skipped. Set RAG_API_KEY or edit {} later.", env_path().display());
    }
    eprintln!("\nDone! Run 'ragfer -s' to test.");
    std::process::exit(0);
}

pub fn get_server_url() -> String { get_url().unwrap_or_else(|_| "http://localhost:4242".into()) }
pub fn resolve_api_key() -> Option<String> { get_api_key().ok() }
pub fn cmd_setup() -> Result<()> { run_setup() }

fn api_call(method: &str, path: &str, body: Option<Value>) -> Result<Value> {
    let url = get_url()?;
    let key = get_api_key()?;
    let full = format!("{}{}", url.trim_end_matches('/'), path);
    let req = ureq::AgentBuilder::new()
        .timeout_read(std::time::Duration::from_secs(30))
        .timeout_write(std::time::Duration::from_secs(30))
        .build()
        .request(method, &full)
        .set("Authorization", &format!("Bearer {}", key))
        .set("Content-Type", "application/json");
    let resp = match body {
        Some(data) => req.send_json(data)?,
        None => req.call()?,
    };
    Ok(resp.into_json()?)
}

// ─── Commands ──────────────────────────────────────────────────────────────

pub fn cmd_status(json: bool) -> Result<()> {
    let r = api_call("GET", "/api/status", None)?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); return Ok(()); }
    println!("rag-ferrite v{}", r.get("version").and_then(|v| v.as_str()).unwrap_or("?"));
    println!("Documents: {}", r.get("document_count").and_then(|v| v.as_u64()).unwrap_or(0));
    Ok(())
}

pub fn cmd_progress(json: bool) -> Result<()> {
    let r = api_call("GET", "/api/ingest/progress", None)?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); return Ok(()); }
    let status = r.get("status").and_then(|v| v.as_str()).unwrap_or("idle");
    if status == "idle" || !r.get("batch").is_some() { println!("No batch running."); return Ok(()); }
    let b = r.get("batch").unwrap();
    let total = b.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
    let done = b.get("completed_files").and_then(|v| v.as_u64()).unwrap_or(0);
    let pct = if total > 0 { done as f64 / total as f64 * 100.0 } else { 0.0 };
    let chunks = b.get("completed_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
    let speed = b.get("speed_chunks_per_min").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let elapsed = fmt_dur(b.get("elapsed_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0));
    let eta = fmt_dur(b.get("eta_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0));
    let errors = b.get("failed_files").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("Batch {}", b.get("batch_id").and_then(|v| v.as_u64()).unwrap_or(0));
    println!("  Status:    {}", b.get("status").and_then(|v| v.as_str()).unwrap_or("?"));
    println!("  Progress:  {}/{} files ({:.0}%)", done, total, pct);
    println!("  Chunks:    {} done", chunks);
    println!("  Speed:     {:.0} chunks/min", speed);
    println!("  Elapsed:   {}", elapsed);
    println!("  ETA:       {}", eta);
    println!("  Errors:    {}", errors);
    if let Some(cf) = b.get("current_file") {
        let name = cf.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let phase = cf.get("phase").and_then(|v| v.as_str()).unwrap_or("?");
        let fp = cf.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
        println!("  Current:   {} ({:.0}% — {})", name, fp, phase);
    }
    if errors > 0 {
        if let Some(el) = b.get("errors").and_then(|v| v.as_array()) {
            for e in el.iter().take(5) {
                println!("  x {}: {}", e.get("file").and_then(|v| v.as_str()).unwrap_or("?"),
                    e.get("error").and_then(|v| v.as_str()).unwrap_or("?"));
            }
        }
    }
    Ok(())
}

/// Poll batch progress until complete, then print summary with errors.
fn poll_batch_result() -> Result<()> {
    let max_wait_secs = 120;
    let start = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let r = api_call("GET", "/api/ingest/progress", None)?;
        let status = r.get("status").and_then(|v| v.as_str()).unwrap_or("idle");
        let batch = r.get("batch");
        if status == "idle" || batch.is_none() {
            // Batch finished — progress reset to idle
            println!("  Done.");
            return Ok(());
        }
        let b = batch.unwrap();
        let batch_status = b.get("status").and_then(|v| v.as_str()).unwrap_or("?");
        if batch_status == "Completed" || batch_status == "Cancelled" {
            let done = b.get("completed_files").and_then(|v| v.as_u64()).unwrap_or(0);
            let failed = b.get("failed_files").and_then(|v| v.as_u64()).unwrap_or(0);
            let chunks = b.get("completed_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  Result: {} files done, {} failed, {} chunks", done, failed, chunks);
            if failed > 0 {
                if let Some(errors) = b.get("errors").and_then(|v| v.as_array()) {
                    for e in errors.iter().take(5) {
                        let file = e.get("file").and_then(|v| v.as_str()).unwrap_or("?");
                        let err = e.get("error").and_then(|v| v.as_str()).unwrap_or("?");
                        eprintln!("  ERROR — {}: {}", file, err);
                    }
                    if errors.len() > 5 {
                        eprintln!("  ... and {} more errors", errors.len() - 5);
                    }
                }
            }
            return Ok(());
        }
        if start.elapsed().as_secs() > max_wait_secs {
            eprintln!("  Timeout waiting for batch to complete ({}s). Check 'ragfer progress'.", max_wait_secs);
            return Ok(());
        }
    }
}

fn fmt_dur(secs: f64) -> String {
    if secs <= 0.0 { return String::from("—"); }
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    if h > 0 { format!("{}h{:02}m", h, m) } else if m > 0 { format!("{}m{:02}s", m, s) } else { format!("{}s", s) }
}

pub fn cmd_query(json: bool, text: &str, collection: Option<&str>, limit: usize, tags: Option<&str>) -> Result<()> {
    let mut data = serde_json::json!({"query": text, "limit": limit});
    if let Some(c) = collection { data["collection"] = Value::String(c.to_string()); }
    if let Some(t) = tags { data["tags"] = Value::String(t.to_string()); }
    let r = api_call("POST", "/api/query", Some(data))?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); return Ok(()); }
    let empty = Vec::new();
    let chunks = r.get("chunks").and_then(|v| v.as_array()).unwrap_or(&empty);
    if chunks.is_empty() { println!("No results found."); return Ok(()); }
    for (i, ch) in chunks.iter().enumerate() {
        let src = ch.get("source_name").or_else(|| ch.get("source")).and_then(|v| v.as_str()).unwrap_or("?");
        let content = ch.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let tags_list = ch.get("tags").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let score = ch.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        println!("--- Result {} (score: {:.2}) ---", i + 1, score);
        println!("Source: {}", src);
        if !tags_list.is_empty() { println!("Tags: {}", tags_list); }
        println!("{}
", content);
    }
    Ok(())
}

pub fn cmd_list(json: bool, collection: Option<&str>) -> Result<()> {
    let path = if let Some(c) = collection { format!("/api/documents?collection={}", c) } else { "/api/documents".to_string() };
    let r = api_call("GET", &path, None)?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); return Ok(()); }
    let empty = Vec::new();
    let files = r.get("files").and_then(|v| v.as_array()).unwrap_or(&empty);
    if files.is_empty() { println!("No documents found."); return Ok(()); }
    println!("{:<5} {:<50} {:<10}", "ID", "Source", "Chunks");
    println!("{}", "─".repeat(67));
    for f in files {
        let id = f.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let name = f.get("source_name").or_else(|| f.get("name")).and_then(|v| v.as_str()).unwrap_or("?");
        let chunks = f.get("chunk_count").and_then(|v| v.as_u64()).unwrap_or(0);
        println!("{:<5} {:<50} {:<10}", id, name, chunks);
    }
    println!("\n{} documents", files.len());
    Ok(())
}

pub fn cmd_ingest_file(json: bool, path: &str, collection: Option<&str>, force: bool) -> Result<()> {
    let full_path = PathBuf::from(shellexpand::tilde(path).to_string());
    if !full_path.exists() { bail!("File not found: {}", full_path.display()); }
    if force {
        let filename = full_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let docs_r = api_call("GET", "/api/documents", None)?;
        if let Some(files) = docs_r.get("files").and_then(|v| v.as_array()) {
            for f in files {
                let name = f.get("source_name").or_else(|| f.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                if name.contains(&filename) {
                    if let Some(sid) = f.get("id") { let _ = api_call("DELETE", &format!("/api/documents/{}", sid), None); }
                }
            }
        }
    }
    let mut data = serde_json::json!({"file_path": full_path.to_string_lossy()});
    if let Some(c) = collection { data["collection"] = Value::String(c.to_string()); }
    let r = api_call("POST", "/api/ingest", Some(data))?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); return Ok(()); }
    println!("Ingestion started. Batch ID: {}", r.get("batch_id").unwrap_or(&Value::Null));

    // Poll progress until batch completes, then report result
    poll_batch_result()?;
    Ok(())
}

pub fn cmd_ingest_batch(json: bool, paths: &[String], collection: Option<&str>) -> Result<()> {
    let expanded: Vec<String> = paths.iter().map(|p| shellexpand::tilde(p).to_string()).collect();
    let valid: Vec<&String> = expanded.iter().filter(|p| PathBuf::from(p).exists()).collect();
    if valid.is_empty() { bail!("None of the specified files exist."); }
    let mut data = serde_json::json!({"paths": expanded});
    if let Some(c) = collection { data["collection"] = Value::String(c.to_string()); }
    let r = api_call("POST", "/api/ingest/batch", Some(data))?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); } else { println!("Batch started ({} files). ID: {}", valid.len(), r.get("batch_id").unwrap_or(&Value::Null)); }
    Ok(())
}

pub fn cmd_ingest_data(json: bool, name: &str, collection: Option<&str>) -> Result<()> {
    let mut content = String::new();
    io::stdin().read_line(&mut content)?;
    use std::io::Read;
    io::stdin().read_to_string(&mut content)?;
    let mut data = serde_json::json!({"content": content, "source_name": name, "format": "text"});
    if let Some(c) = collection { data["collection"] = Value::String(c.to_string()); }
    let r = api_call("POST", "/api/ingest/data", Some(data))?;
    if json { println!("{}", serde_json::to_string_pretty(&r)?); } else { println!("Ingested as '{}'.", name); }
    Ok(())
}

pub fn cmd_delete(json: bool, source_id: &str) -> Result<()> {
    let r = api_call("DELETE", &format!("/api/documents/{}", source_id), None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

pub fn cmd_flush(_json: bool) -> Result<()> { let r = api_call("POST", "/api/flush-indexes", None)?; println!("{}", serde_json::to_string_pretty(&r)?); Ok(()) }
pub fn cmd_rebuild(_json: bool) -> Result<()> { let r = api_call("POST", "/api/rebuild-indexes", None)?; println!("{}", serde_json::to_string_pretty(&r)?); Ok(()) }
pub fn cmd_cancel(_json: bool) -> Result<()> { let r = api_call("POST", "/api/service/cancel-batch", None)?; println!("{}", serde_json::to_string_pretty(&r)?); Ok(()) }
pub fn cmd_stop(_json: bool) -> Result<()> { let r = api_call("POST", "/api/service/stop", None)?; println!("{}", serde_json::to_string_pretty(&r)?); Ok(()) }

// ─── CLI ───────────────────────────────────────────────────────────────────

pub struct CliArgs { pub json: bool, pub command: CliCommand }

pub enum CliCommand {
    Serve, Status, Progress, Monitor, Update, Setup,
    Query { text: String, collection: Option<String>, limit: usize, tags: Option<String> },
    List { collection: Option<String> },
    IngestFile { path: String, collection: Option<String>, force: bool },
    IngestBatch { paths: Vec<String>, collection: Option<String> },
    IngestData { name: String, collection: Option<String> },
    Delete { source_id: String },
    Flush, Rebuild, Cancel, Stop,
}

pub fn print_usage() {
    eprintln!(r#"ragfer — rag-ferrite CLI client

Usage:
    ragfer                            Launch monitor (default)
    ragfer serve  (-d)                Launch server (daemon)
    ragfer status (-s)                Engine status
    ragfer progress (-p)             Batch ingestion progress
    ragfer query (-q) "text"         Semantic search
    ragfer list (-l)                 List indexed documents
    ragfer ingest-file <path>        Ingest a file
    ragfer ingest-batch <paths...>   Ingest multiple files
    ragfer ingest-data <name>        Ingest stdin as text
    ragfer delete <source_id>        Delete a document
    ragfer flush                     Flush HNSW buffer to disk
    ragfer rebuild                   Rebuild all indexes
    ragfer cancel                    Cancel running batch
    ragfer stop                      Stop the server
    ragfer setup                     Configure server URL + API key
    ragfer help (-h)                 Show this help

Options:
    --json    Raw JSON output
    -c <col>  Target collection
    -n <num>  Query result limit (default: 10)
    -t <tags> Filter by tags
    --force   Re-ingest (delete existing first)

Config:
    ~/.config/ragfer/config.toml  Server URL
    ~/.config/ragfer/.env         API key
    RAG_API_KEY env var           Or set as environment variable
"#);
}

pub fn parse_args() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut json = false;
    let mut collection: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut tags: Option<String> = None;
    let mut force = false;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--json" => json = true,
            "-c" => { i += 1; if i >= raw.len() { bail!("-c requires a collection name"); } collection = Some(raw[i].clone()); }
            "-n" => { i += 1; if i >= raw.len() { bail!("-n requires a number"); } limit = Some(raw[i].parse()?); }
            "-t" => { i += 1; if i >= raw.len() { bail!("-t requires tag values"); } tags = Some(raw[i].clone()); }
            "--force" => force = true,
            _ => positional.push(raw[i].clone()),
        }
        i += 1;
    }
    let subcmd = positional.first().map(|s| s.as_str()).unwrap_or("monitor");
    let args = if positional.len() > 1 { &positional[1..] } else { &[] };
    let command = match subcmd {
        "serve" | "-d" | "--serve" => CliCommand::Serve,
        "status" | "-s" => CliCommand::Status,
        "progress" | "-p" => CliCommand::Progress,
        "query" | "-q" => { if args.is_empty() { bail!("query requires text"); } CliCommand::Query { text: args.join(" "), collection, limit: limit.unwrap_or(10), tags } }
        "list" | "-l" => CliCommand::List { collection },
        "ingest-file" | "ingest_file" => { if args.is_empty() { bail!("ingest-file requires a path"); } CliCommand::IngestFile { path: args[0].clone(), collection, force } }
        "ingest-batch" | "ingest_batch" => { if args.is_empty() { bail!("ingest-batch requires paths"); } CliCommand::IngestBatch { paths: args.to_vec(), collection } }
        "ingest-data" | "ingest_data" => { if args.is_empty() { bail!("ingest-data requires a name"); } CliCommand::IngestData { name: args[0].clone(), collection } }
        "delete" => { if args.is_empty() { bail!("delete requires a source_id"); } CliCommand::Delete { source_id: args[0].clone() } }
        "flush" => CliCommand::Flush,
        "rebuild" => CliCommand::Rebuild,
        "cancel" => CliCommand::Cancel,
        "stop" => CliCommand::Stop,
        "monitor" | "-m" => CliCommand::Monitor,
        "update" => CliCommand::Update,
        "setup" => CliCommand::Setup,
        "setup" => CliCommand::Setup,
        "help" | "-h" | "--help" => { print_usage(); std::process::exit(0); }
        _ => { eprintln!("Unknown command: {}", subcmd); print_usage(); std::process::exit(1); }
    };
    Ok(CliArgs { json, command })
}

pub fn execute(args: CliArgs) -> Result<()> {
    match args.command {
        CliCommand::Status => cmd_status(args.json),
        CliCommand::Progress => cmd_progress(args.json),
        CliCommand::Query { text, collection, limit, tags } => cmd_query(args.json, &text, collection.as_deref(), limit, tags.as_deref()),
        CliCommand::List { collection } => cmd_list(args.json, collection.as_deref()),
        CliCommand::IngestFile { path, collection, force } => cmd_ingest_file(args.json, &path, collection.as_deref(), force),
        CliCommand::IngestBatch { paths, collection } => cmd_ingest_batch(args.json, &paths, collection.as_deref()),
        CliCommand::IngestData { name, collection } => cmd_ingest_data(args.json, &name, collection.as_deref()),
        CliCommand::Delete { source_id } => cmd_delete(args.json, &source_id),
        CliCommand::Flush => cmd_flush(args.json),
        CliCommand::Rebuild => cmd_rebuild(args.json),
        CliCommand::Cancel => cmd_cancel(args.json),
        CliCommand::Stop => cmd_stop(args.json),
        CliCommand::Setup => cmd_setup(),
        CliCommand::Serve | CliCommand::Monitor | CliCommand::Update => unreachable!("handled by main.rs"),
    }
}
