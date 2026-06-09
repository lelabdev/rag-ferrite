use anyhow::Result;
use rusqlite::Connection;
use rag_engine::api::{
    db_pool,
    simple,
    source_rag,
};

pub mod search;
pub mod query;
pub mod benchmark;
pub mod tags;
pub mod chunk_counter;
pub mod cancel;
pub mod activity_log;
pub mod heat;
pub mod tag_routing;
pub mod chunk_heat;
pub mod precheck;
pub mod ingest;
pub mod indexes;

// Re-export public items from sub-modules
pub use search::{search_hybrid, search_hybrid_with_expansion};
pub use query::{get_section_paths_for_chunk_ids, get_neighbors, delete_source, list_sources};
pub use benchmark::{run_benchmark, get_graph_data};
pub use tags::{create_chunk_tags_table, create_collection_tags_table, insert_chunk_tags, update_collection_tags, get_tags_for_chunk_ids};
pub use heat::{create_collection_heat_table, HeatTracker, get_all_heat, collections_for_sources, get_chunk_qa_report};

// Re-export from chunk_heat
pub use chunk_heat::ChunkHeatTracker;

// Re-export from ingest
pub use ingest::{ingest_text, ingest_file};

// Re-export from indexes
pub use indexes::{add_embeddings_to_buffer, rebuild_and_save_indexes, reassign_source_collection, wal_checkpoint};

// Re-export from precheck
pub use precheck::{pre_check_document, check_duplicate_source, verify_chunks};

/// Stored DB path so list_sources/stats can query across all collections.
static DB_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Centralized SQLite connection — avoids ad-hoc `Connection::open` calls.
static DB_CONN: std::sync::OnceLock<std::sync::Mutex<rusqlite::Connection>> = std::sync::OnceLock::new();

/// Obtain a locked handle to the shared SQLite connection.
pub fn get_conn() -> Result<std::sync::MutexGuard<'static, rusqlite::Connection>> {
    DB_CONN
        .get()
        .ok_or_else(|| anyhow::anyhow!("DB not initialized"))?
        .lock()
        .map_err(|e| anyhow::anyhow!("DB connection lock poisoned: {}", e))
}

/// Get the data directory from DB_PATH.
pub(crate) fn data_dir() -> String {
    DB_PATH.get()
        .map(|p| std::path::Path::new(p).parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_else(|| ".".to_string()))
        .unwrap_or_else(|| ".".to_string())
}

/// Initialize rag_engine: logger + DB pool + schema + reranker
pub fn init(data_dir: &std::path::Path, config: &crate::config::Config) -> Result<()> {
    simple::init_core();
    let db_path = data_dir.join("rag.sqlite3");
    std::fs::create_dir_all(data_dir)?;
    let db_path_str = db_path.to_string_lossy().to_string();
    db_pool::init_db_pool(db_path_str.clone(), config.advanced.db_pool_size as u32)?;
    source_rag::init_source_db()?;

    // Migration: add section_path column to chunks (backward-compatible)
    let conn = rusqlite::Connection::open(&db_path_str)?;
    let has_section_path: bool = conn.prepare("SELECT section_path FROM chunks LIMIT 1").is_ok();
    if !has_section_path {
        tracing::info!("Migrating: adding section_path column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN section_path TEXT DEFAULT NULL")?;
    }

    // Migration: add page column to chunks (backward-compatible)
    let has_page: bool = conn.prepare("SELECT page FROM chunks LIMIT 1").is_ok();
    if !has_page {
        tracing::info!("Migrating: adding page column to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN page INTEGER DEFAULT NULL")?;
    }

    // Migration: add parent_id and chunk_type columns for parent-child chunking
    let has_parent_id: bool = conn.prepare("SELECT parent_id FROM chunks LIMIT 1").is_ok();
    if !has_parent_id {
        tracing::info!("Migrating: adding parent_id and chunk_type columns to chunks");
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN parent_id INTEGER DEFAULT NULL")?;
        conn.execute_batch("ALTER TABLE chunks ADD COLUMN chunk_role TEXT DEFAULT NULL")?;
    }

    // Migration: make embedding nullable (parents don't have embeddings)
    let embedding_notnull: bool = {
        let mut stmt = conn.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'")?;
        let sql: String = stmt.query_row([], |row| row.get(0))?;
        sql.contains("embedding BLOB NOT NULL")
    };
    if embedding_notnull {
        tracing::info!("Migrating: making embedding column nullable for parent-child support");
        conn.execute_batch(
            "CREATE TABLE chunks_new (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                collection_id TEXT NOT NULL DEFAULT '__default__',
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                chunk_type TEXT DEFAULT 'general',
                embedding BLOB,
                embedding_i8 BLOB,
                embedding_scale REAL,
                section_path TEXT DEFAULT NULL,
                page INTEGER DEFAULT NULL,
                parent_id INTEGER DEFAULT NULL,
                chunk_role TEXT DEFAULT NULL
            );
            INSERT INTO chunks_new SELECT * FROM chunks;
            DROP TABLE chunks;
            ALTER TABLE chunks_new RENAME TO chunks;"
        )?;
    }
    drop(conn);

    // Create chunk_tags table for auto-tagging
    create_chunk_tags_table(&db_path_str)?;

    // Create collection_tags table for tag routing (v5 design)
    create_collection_tags_table(&db_path_str)?;

    // Create collection_heat table for heat tracking (v5 design, Phase 1)
    create_collection_heat_table(&db_path_str)?;

    // Add heat tracking columns to chunks table (v5 design)
    let conn = Connection::open(&db_path_str)?;
    let has_query_count = conn
        .query_row("SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='query_count'", [], |row| row.get::<_, i64>(0))
        .unwrap_or(0) > 0;
    if !has_query_count {
        conn.execute_batch(
            "ALTER TABLE chunks ADD COLUMN query_count INTEGER NOT NULL DEFAULT 0;
             ALTER TABLE chunks ADD COLUMN last_queried_at REAL;"
        )?;
        tracing::info!("Added heat tracking columns to chunks table");
    }
    drop(conn);

    let _ = DB_PATH.set(db_path_str.clone());

    // Store a shared connection for all subsequent get_conn() calls
    let shared_conn = rusqlite::Connection::open(&db_path_str)?;
    shared_conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL; PRAGMA busy_timeout={}; PRAGMA cache_size=-{};",
        config.advanced.db_busy_timeout_ms, config.advanced.db_cache_size_mb
    ))?;
    let _ = DB_CONN.set(std::sync::Mutex::new(shared_conn));

    // Check embedding dimension mismatch: compare DB vectors with configured dimensions
    if let Some(config_dims) = config.embedding.dimensions {
        let conn = get_conn()?;
        let db_dims: Option<usize> = conn.query_row(
            "SELECT vector FROM chunks WHERE vector IS NOT NULL LIMIT 1",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob.len() / 4) // f32 = 4 bytes
            },
        ).ok();
        drop(conn);

        if let Some(stored_dims) = db_dims {
            if stored_dims != config_dims {
                anyhow::bail!(
                    "Embedding dimension mismatch: DB has {} but config says {}. Re-ingest all documents or update config.",
                    stored_dims, config_dims
                );
            }
            tracing::info!("Embedding dimensions verified: {} (DB matches config)", stored_dims);
        }
    }

    tracing::info!("rag_engine DB initialized at {}", db_path.display());

    Ok(())
}

/// Sanitize a collection ID: only allow alphanumeric, underscore, and hyphen.
/// Returns an error if the result is empty after sanitization.
pub fn sanitize_collection(collection: &str) -> Result<String> {
    let sanitized: String = collection
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized.is_empty() {
        anyhow::bail!("Invalid collection ID: '{}' contains no valid characters", collection);
    }
    Ok(sanitized)
}

/// Get stats across all collections.
pub fn stats() -> Result<Stats> {
    let conn = get_conn()?;

    let count: usize = conn.query_row(
        "SELECT COUNT(*) FROM sources",
        [],
        |row| row.get::<_, i64>(0),
    )? as usize;

    Ok(Stats {
        document_count: count,
    })
}

pub struct Stats {
    pub document_count: usize,
}
