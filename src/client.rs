/// ragfer client — CLI subcommands that hit the rag-ferrite HTTP API.
///
/// Ported from the Python CLI (cli/rag). Same behaviour, same formatters,
/// but now compiled into the main binary.

use anyhow::{bail, Result};
use serde_json::Value;
use std::path::PathBuf;

// ─── Instance config ────────────────────────────────────────────────────────

struct Instance {
    url: &'static str,
    key_env: &'static str,
    key_file: &'static str,
}

fn get_instance(env: &str) -> &'static Instance {
    // Could use a static map but two instances is fine.
    match env {
        "test" => &Instance {
            url: "http://100.90.185.42:4242",
            key_env: "RAG_API_KEY_AETHER",
            key_file: "~/.config/rag/api_key_aether",
        },
        _ => &Instance {
            url: "http://100.97.67.73:4242",
            key_env: "RAG_API_KEY_NOVA",
            key_file: "~/.config/rag/api_key_nova",
        },
    }
}

fn get_api_key(inst: &Instance) -> Result<String> {
    // 1. Env var
    if let Ok(key) = std::env::var(inst.key_env) {
        if !key.is_empty() {
            return Ok(key);
        }
    }
    // 2. Key file (~/.config/rag/)
    let expanded = shellexpand::tilde(inst.key_file).to_string();
    let path = PathBuf::from(&expanded);
    if path.exists() {
        let key = std::fs::read_to_string(&path)?;
        let key = key.trim().to_string();
        if !key.is_empty() {
            return Ok(key);
        }
    }
    // 3. Key file next to binary (Nova setup: ~/services/rag-ferrite/.env)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let env_path = dir.join(".env");
            if env_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&env_path) {
                    for line in contents.lines() {
                        if let Some(val) = line.strip_prefix("RAG_API_KEY=") {
                            let key = val.trim().trim_matches('"').to_string();
                            if !key.is_empty() {
                                return Ok(key);
                            }
                        }
                    }
                }
            }
        }
    }
    // 4. Inline fallback (prod/Nova — same as Python CLI)
    if inst.url.contains("100.97.67.73") {
        return Ok("e521d0ef391b719af8773857c912a9bd2fdf86e27d77c906".into());
    }
    bail!(
        "No API key for instance. Set ${} or write to {}",
        inst.key_env,
        inst.key_file
    )
}

// ─── HTTP helpers ────────────────────────────────────────────────────────────

fn api_call(env: &str, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
    let inst = get_instance(env);
    let key = get_api_key(inst)?;
    let url = format!("{}{}", inst.url, path);

    let mut req = match method {
        "POST" => ureq::post(&url),
        "DELETE" => ureq::delete(&url),
        _ => ureq::get(&url),
    };

    req = req.set("Authorization", &format!("Bearer {}", key));

    let resp = if let Some(data) = body {
        req.send_json(data)?
    } else {
        req.call()?
    };

    let val: Value = resp.into_json()?;
    Ok(val)
}

// ─── Formatters ──────────────────────────────────────────────────────────────

fn fmt_duration(seconds: f64) -> String {
    if seconds <= 0.0 {
        return "—".into();
    }
    let s = seconds as u64;
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        format!("{}h{}m", s / 3600, (s % 3600) / 60)
    }
}

// ─── Subcommands ─────────────────────────────────────────────────────────────

pub fn cmd_status(env: &str, json: bool) -> Result<()> {
    let r = api_call(env, "GET", "/api/status", None)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }
    println!("rag-ferrite v{}", r.get("version").and_then(|v| v.as_str()).unwrap_or("?"));
    println!(
        "Documents: {}",
        r.get("document_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    Ok(())
}

pub fn cmd_progress(env: &str, json: bool) -> Result<()> {
    let r = api_call(env, "GET", "/api/ingest/progress", None)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }

    let status = r
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("idle");
    if status == "idle" || !r.get("batch").is_some() {
        println!("No batch running.");
        return Ok(());
    }

    let b = r.get("batch").unwrap();
    let total: u64 = b.get("total_files").and_then(|v| v.as_u64()).unwrap_or(0);
    let completed: u64 = b
        .get("completed_files")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let pct = if total > 0 {
        completed as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let speed = b
        .get("speed_chunks_per_min")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let eta = fmt_duration(
        b.get("eta_seconds")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
    );
    let elapsed = fmt_duration(
        b.get("elapsed_seconds")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
    );

    println!("Batch {}", b.get("batch_id").and_then(|v| v.as_u64()).unwrap_or(0));
    println!("  Status:    {}", b.get("status").and_then(|v| v.as_str()).unwrap_or("?"));
    println!("  Progress:  {}/{} files ({:.0}%)", completed, total, pct);
    println!(
        "  Chunks:    {} done",
        b.get("completed_chunks")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!("  Speed:     {:.0} chunks/min", speed);
    println!("  Elapsed:   {}", elapsed);
    println!("  ETA:       {}", eta);
    println!(
        "  Errors:    {}",
        b.get("failed_files")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );

    if let Some(cf) = b.get("current_file") {
        let name = cf.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        println!("  Current:   {}", &name[..name.len().min(70)]);
        if let Some(phase) = cf.get("phase").and_then(|v| v.as_str()) {
            println!("    Phase:   {}", phase);
        }
    }

    if let Some(errors) = b.get("errors").and_then(|v| v.as_array()) {
        if !errors.is_empty() {
            println!("\n  Recent errors:");
            for e in errors.iter().rev().take(3) {
                println!("    - {}", e);
            }
        }
    }

    Ok(())
}

pub fn cmd_query(env: &str, json: bool, text: &str, collection: Option<&str>, limit: usize, tags: Option<&str>) -> Result<()> {
    let mut data = serde_json::json!({
        "query": text,
        "limit": limit
    });
    if let Some(c) = collection {
        data["collection"] = Value::String(c.into());
    }
    if let Some(t) = tags {
        data["tags"] = serde_json::from_str(&format!(
            "[{}]",
            t.split(',')
                .map(|s| format!("\"{}\"", s.trim()))
                .collect::<Vec<_>>()
                .join(",")
        ))
        .unwrap_or(Value::Null);
    }

    let r = api_call(env, "POST", "/api/query", Some(data))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }

    // Results can be at several paths depending on API version
    let results = if r.is_array() {
        r.as_array().unwrap().clone()
    } else if let Some(arr) = r.get("results").and_then(|v| v.as_array()) {
        arr.clone()
    } else if let Some(arr) = r.get("chunks").and_then(|v| v.as_array()) {
        arr.clone()
    } else {
        vec![]
    };

    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }

    for (i, chunk) in results.iter().enumerate() {
        let score = chunk
            .get("score")
            .or_else(|| chunk.get("rerank_score"))
            .map(|v| v.to_string())
            .unwrap_or_default();
        let source = chunk
            .get("source_name")
            .or_else(|| chunk.get("source"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let content = chunk
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let tags_list = chunk
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        println!("{}", "─".repeat(70));
        println!("[{}] {}  (score: {})", i + 1, source, score);
        if !tags_list.is_empty() {
            println!("    tags: {}", tags_list.join(", "));
        }
        let truncated = &content[..content.len().min(300)];
        println!("    {}...", truncated);
    }
    println!("\n{} result(s)", results.len());
    Ok(())
}

pub fn cmd_list(env: &str, json: bool, collection: Option<&str>) -> Result<()> {
    let path = match collection {
        Some(c) => format!("/api/documents?collection={}", c),
        None => "/api/documents".into(),
    };

    let r = api_call(env, "GET", &path, None)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }

    let docs = if r.is_array() {
        r.as_array().unwrap().clone()
    } else {
        r.get("files")
            .or_else(|| r.get("documents"))
            .or_else(|| r.get("sources"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    };

    if docs.is_empty() {
        println!("No documents.");
        return Ok(());
    }

    for d in &docs {
        let name = d.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let coll = d
            .get("collection_id")
            .or_else(|| d.get("collection"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let chunks = d
            .get("chunk_count")
            .or_else(|| d.get("chunks"))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into());

        let display_name = if name.len() > 60 {
            format!("{}...", &name[..57])
        } else {
            name.to_string()
        };
        println!("  {:<60} [{}] {} chunks", display_name, coll, chunks);
    }
    println!("\n{} document(s)", docs.len());
    Ok(())
}

pub fn cmd_ingest_file(env: &str, json: bool, path: &str, collection: Option<&str>, force: bool) -> Result<()> {
    let full_path = PathBuf::from(shellexpand::tilde(path).to_string());
    if !full_path.exists() {
        bail!("File not found: {}", full_path.display());
    }

    if force {
        let fname = full_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let docs_r = api_call(env, "GET", "/api/documents", None)?;
        let docs = if docs_r.is_array() {
            docs_r.as_array().unwrap().clone()
        } else {
            docs_r
                .get("files")
                .or_else(|| docs_r.get("documents"))
                .or_else(|| docs_r.get("sources"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        };
        for d in &docs {
            if d.get("name").and_then(|v| v.as_str()) == Some(&fname) {
                let sid = d
                    .get("id")
                    .or_else(|| d.get("source_id"))
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                eprintln!("  Deleting existing source {} ({})...", sid, fname);
                let _ = api_call(env, "DELETE", &format!("/api/documents/{}", sid), None);
                break;
            }
        }
    }

    let mut data = serde_json::json!({
        "file_path": full_path.to_string_lossy()
    });
    if let Some(c) = collection {
        data["collection"] = Value::String(c.into());
    }

    let r = api_call(env, "POST", "/api/ingest/file", Some(data))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&r)?);
    }
    Ok(())
}

pub fn cmd_ingest_batch(env: &str, json: bool, paths: &[String], collection: Option<&str>) -> Result<()> {
    let expanded: Vec<String> = paths
        .iter()
        .map(|p| {
            let full = PathBuf::from(shellexpand::tilde(p).to_string());
            full.to_string_lossy().to_string()
        })
        .collect();

    // Verify all files exist
    for (i, p) in expanded.iter().enumerate() {
        if !PathBuf::from(p).exists() {
            bail!("File not found: {}", paths[i]);
        }
    }

    let mut data = serde_json::json!({
        "paths": expanded
    });
    if let Some(c) = collection {
        data["collection"] = Value::String(c.into());
    }

    let r = api_call(env, "POST", "/api/ingest/batch", Some(data))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&r)?);
    }
    Ok(())
}

pub fn cmd_ingest_data(env: &str, json: bool, name: &str, collection: Option<&str>) -> Result<()> {
    let content = {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    };
    if content.trim().is_empty() {
        bail!("No data on stdin.");
    }

    let mut data = serde_json::json!({
        "content": content,
        "source_name": name
    });
    if let Some(c) = collection {
        data["collection"] = Value::String(c.into());
    }

    let r = api_call(env, "POST", "/api/ingest/data", Some(data))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&r)?);
    }
    Ok(())
}

pub fn cmd_delete(env: &str, _json: bool, source_id: &str) -> Result<()> {
    let r = api_call(env, "DELETE", &format!("/api/documents/{}", source_id), None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

pub fn cmd_flush(env: &str, _json: bool) -> Result<()> {
    eprintln!("Flushing HNSW indexes... (this may take a while)");
    let r = api_call(env, "POST", "/api/flush-indexes", None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

pub fn cmd_rebuild(env: &str, _json: bool) -> Result<()> {
    eprintln!("Rebuilding indexes... (this may take a while)");
    let r = api_call(env, "POST", "/api/rebuild-indexes", None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

pub fn cmd_cancel(env: &str, _json: bool) -> Result<()> {
    let r = api_call(env, "POST", "/api/service/cancel-batch", None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

pub fn cmd_stop(env: &str, _json: bool) -> Result<()> {
    let r = api_call(env, "POST", "/api/service/stop", None)?;
    println!("{}", serde_json::to_string_pretty(&r)?);
    Ok(())
}

// ─── Usage / Help ────────────────────────────────────────────────────────────

pub fn print_usage() {
    eprintln!(
        r#"ragfer — rag-ferrite CLI client

Usage:
    ragfer                            Show this help
    ragfer serve  (-d)                Launch server (daemon)
    ragfer status (-s)                Engine status
    ragfer progress (-p)             Batch ingestion progress
    ragfer query (-q) "text"         Search documents
    ragfer list (-l)                  List documents
    ragfer monitor (-m)              Launch TUI monitor
    ragfer ingest-file <path>       Ingest a file
    ragfer ingest-batch <paths...>  Ingest multiple files
    ragfer ingest-data <name>       Ingest from stdin
    ragfer delete <source_id>       Delete a document
    ragfer flush                     Flush HNSW indexes
    ragfer rebuild                   Rebuild indexes
    ragfer cancel                    Cancel running batch
    ragfer stop                      Stop the server
    ragfer update                    Download latest + restart

Options:
    --env <env>      Instance: prod (default) | test
    --json           Raw JSON output
    -c <collection>  Collection name
    -n <limit>       Result limit (default 10)
    -t <tags>        Tag filter (comma-separated)
    --force          Force reingest (delete existing first)
"#
    );
}

// ─── Arg parsing & dispatch ─────────────────────────────────────────────────

/// Parsed global options + subcommand.
pub struct CliArgs {
    pub env: String,
    pub json: bool,
    pub command: CliCommand,
}

pub enum CliCommand {
    Serve,
    Status,
    Progress,
    Query {
        text: String,
        collection: Option<String>,
        limit: usize,
        tags: Option<String>,
    },
    List {
        collection: Option<String>,
    },
    IngestFile {
        path: String,
        collection: Option<String>,
        force: bool,
    },
    IngestBatch {
        paths: Vec<String>,
        collection: Option<String>,
    },
    IngestData {
        name: String,
        collection: Option<String>,
    },
    Delete {
        source_id: String,
    },
    Flush,
    Rebuild,
    Cancel,
    Stop,
    Monitor,
    Update,
}

pub fn parse_args() -> Result<CliArgs> {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    // Collect --env/--json/-c/-l/-t/--force from anywhere, everything else is positional
    let mut env = "prod".to_string();
    let mut json = false;
    let mut collection: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut tags: Option<String> = None;
    let mut force = false;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--env" => {
                i += 1;
                if i >= raw.len() {
                    bail!("--env requires a value (prod|test)");
                }
                env = raw[i].clone();
            }
            "--json" => json = true,
            "-c" => {
                i += 1;
                if i >= raw.len() {
                    bail!("-c requires a collection name");
                }
                collection = Some(raw[i].clone());
            }
            "-n" => {
                i += 1;
                if i >= raw.len() {
                    bail!("-l requires a number");
                }
                limit = Some(raw[i].parse()?);
            }
            "-t" => {
                i += 1;
                if i >= raw.len() {
                    bail!("-t requires tag values");
                }
                tags = Some(raw[i].clone());
            }
            "--force" => force = true,
            _ => positional.push(raw[i].clone()),
        }
        i += 1;
    }

    let subcmd = positional.first().map(|s| s.as_str()).unwrap_or("help");
    let args = if positional.len() > 1 { &positional[1..] } else { &[] };

    let command = match subcmd {
        "serve" | "-d" | "--serve" => CliCommand::Serve,
        "status" | "-s" => CliCommand::Status,
        "progress" | "-p" => CliCommand::Progress,
        "query" | "-q" => {
            if args.is_empty() {
                bail!("query requires a text argument");
            }
            CliCommand::Query {
                text: args.join(" "),
                collection,
                limit: limit.unwrap_or(10),
                tags,
            }
        }
        "list" | "-l" => CliCommand::List { collection },
        "ingest-file" | "ingest_file" => {
            if args.is_empty() {
                bail!("ingest-file requires a file path");
            }
            CliCommand::IngestFile {
                path: args[0].clone(),
                collection,
                force,
            }
        }
        "ingest-batch" | "ingest_batch" => {
            if args.is_empty() {
                bail!("ingest-batch requires at least one file path");
            }
            CliCommand::IngestBatch {
                paths: args.to_vec(),
                collection,
            }
        }
        "ingest-data" | "ingest_data" => {
            if args.is_empty() {
                bail!("ingest-data requires a source name");
            }
            CliCommand::IngestData {
                name: args[0].clone(),
                collection,
            }
        }
        "delete" => {
            if args.is_empty() {
                bail!("delete requires a source_id");
            }
            CliCommand::Delete {
                source_id: args[0].clone(),
            }
        }
        "flush" => CliCommand::Flush,
        "rebuild" => CliCommand::Rebuild,
        "cancel" => CliCommand::Cancel,
        "stop" => CliCommand::Stop,
        "monitor" | "-m" => CliCommand::Monitor,
        "update" => CliCommand::Update,
        "help" | "-h" | "--help" => {
            print_usage();
            std::process::exit(0);
        }
        _ => {
            eprintln!("Unknown command: {}", subcmd);
            print_usage();
            std::process::exit(1);
        }
    };

    Ok(CliArgs { env, json, command })
}

/// Execute a client subcommand. Returns None if the subcommand is not a client
/// command (serve, monitor, update) and should be handled by the caller.
pub fn execute(args: CliArgs) -> Result<()> {
    let env = &args.env;
    let json = args.json;

    match args.command {
        CliCommand::Status => cmd_status(env, json),
        CliCommand::Progress => cmd_progress(env, json),
        CliCommand::Query {
            text,
            collection,
            limit,
            tags,
        } => cmd_query(env, json, &text, collection.as_deref(), limit, tags.as_deref()),
        CliCommand::List { collection } => cmd_list(env, json, collection.as_deref()),
        CliCommand::IngestFile {
            path,
            collection,
            force,
        } => cmd_ingest_file(env, json, &path, collection.as_deref(), force),
        CliCommand::IngestBatch { paths, collection } => {
            cmd_ingest_batch(env, json, &paths, collection.as_deref())
        }
        CliCommand::IngestData { name, collection } => {
            cmd_ingest_data(env, json, &name, collection.as_deref())
        }
        CliCommand::Delete { source_id } => cmd_delete(env, json, &source_id),
        CliCommand::Flush => cmd_flush(env, json),
        CliCommand::Rebuild => cmd_rebuild(env, json),
        CliCommand::Cancel => cmd_cancel(env, json),
        CliCommand::Stop => cmd_stop(env, json),
        // These are handled by main.rs directly
        CliCommand::Serve | CliCommand::Monitor | CliCommand::Update => {
            unreachable!("serve/monitor/update are handled by main.rs")
        }
    }
}
