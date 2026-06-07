//! Collection heat tracking — async batched, EMA decay.
//!
//! Phase 1 of v5 architecture (#159). Tracks which collections are queried most
//! and most recently, so future phases can make lazy-loading decisions.
//!
//! Design:
//! - Events are sent via mpsc channel (non-blocking on query hot path)
//! - Background worker buffers in memory, flushes to SQLite every 30s
//! - heat_score is on [0, 100]: 100 = just queried, decays by 0.99/day
//!   → 93% after 7 days, 74% after 30 days, 48% after 90 days, ~2.5% after 1 year
//! - query_count is a separate accumulator (total queries since start, never decays)
//! - Collections not queried this cycle still get decayed

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Flush interval in seconds.
const FLUSH_INTERVAL_SECS: u64 = 30;

/// Daily decay factor (×0.99 per day).
/// 7d→93%, 30d→74%, 90d→48%, 180d→23%, 365d→2.5%
const DAILY_DECAY: f64 = 0.99;

// ── Table creation ───────────────────────────────────────────────────

/// Create the collection_heat table if it doesn't exist.
pub fn create_collection_heat_table(db_path: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS collection_heat (
            collection      TEXT PRIMARY KEY,
            heat_score      REAL NOT NULL DEFAULT 0.0,
            last_queried_at REAL,
            query_count     INTEGER NOT NULL DEFAULT 0
        );",
    )?;
    tracing::info!("collection_heat table ready");
    Ok(())
}

// ── Types ────────────────────────────────────────────────────────────

/// Heat event sent from the query hot path. One per queried collection.
#[derive(Debug, Clone)]
struct HeatEvent {
    collection: String,
}

/// Serializable heat snapshot for the MCP endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionHeat {
    pub collection: String,
    pub heat_score: f64,
    pub last_queried_at: Option<f64>,
    pub query_count: i64,
}

// ── Tracker ──────────────────────────────────────────────────────────

/// Heat tracker — owns an mpsc channel and a background flush worker.
///
/// Clone-safe: cloning shares the same sender, so all clones push to the same worker.
#[derive(Clone)]
pub struct HeatTracker {
    sender: mpsc::UnboundedSender<HeatEvent>,
}

impl HeatTracker {
    /// Create a new tracker and spawn the background flush worker.
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            background_worker(receiver).await;
        });
        tracing::info!("Heat tracker worker spawned (flush every {}s)", FLUSH_INTERVAL_SECS);
        HeatTracker { sender }
    }

    /// Record that a collection was queried. Non-blocking, fire-and-forget.
    pub fn record(&self, collection: &str) {
        let _ = self.sender.send(HeatEvent {
            collection: collection.to_string(),
        });
    }

    /// Record multiple collections at once (e.g. from a set of source_ids).
    pub fn record_collections(&self, collections: &[String]) {
        for coll in collections {
            self.record(coll);
        }
    }
}

impl Default for HeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Background worker ────────────────────────────────────────────────

async fn background_worker(mut receiver: mpsc::UnboundedReceiver<HeatEvent>) {
    let mut buffer: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut interval = tokio::time::interval(Duration::from_secs(FLUSH_INTERVAL_SECS));
    // First tick completes immediately; skip it so we don't flush an empty buffer.
    interval.tick().await;

    loop {
        tokio::select! {
            // Drain events as they arrive
            Some(event) = receiver.recv() => {
                *buffer.entry(event.collection).or_insert(0) += 1;
            }
            // Periodic flush + decay
            _ = interval.tick() => {
                if let Err(e) = flush_and_decay(&buffer) {
                    tracing::warn!("Heat flush failed: {}", e);
                }
                buffer.clear();
            }
        }
    }
}

/// Flush buffered counts to SQLite and apply EMA decay to all collections.
fn flush_and_decay(buffer: &std::collections::HashMap<String, i32>) -> Result<()> {
    let conn = super::get_conn()?;
    let now = now_secs();

    // 1. Apply decay to ALL collections first
    //    decay = DAILY_DECAY^(elapsed_days since last_queried_at)
    //    For collections never queried, last_queried_at is NULL → skip (heat stays 0).
    {
        let mut stmt = conn.prepare(
            "SELECT collection, heat_score, last_queried_at FROM collection_heat",
        )?;
        let rows: Vec<(String, f64, Option<f64>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);

        for (coll, score, last_q) in &rows {
            let decayed = if let Some(last) = last_q {
                let elapsed_days = (now as f64 - last) / 86400.0;
                let factor = DAILY_DECAY.powf(elapsed_days.max(0.0));
                score * factor
            } else {
                *score
            };
            conn.execute(
                "UPDATE collection_heat SET heat_score = ?1 WHERE collection = ?2",
                rusqlite::params![decayed, coll],
            )?;
        }
    }

    // 2. Upsert buffered counts — reset heat to 100, accumulate query_count separately
    for (collection, &count) in buffer {
        // Check if collection exists
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM collection_heat WHERE collection = ?1",
                rusqlite::params![collection],
                |_| Ok(()),
            )
            .is_ok();

        if exists {
            conn.execute(
                "UPDATE collection_heat
                 SET heat_score = 100.0,
                     last_queried_at = ?1,
                     query_count = query_count + ?2
                 WHERE collection = ?3",
                rusqlite::params![now, count, collection],
            )?;
        } else {
            conn.execute(
                "INSERT INTO collection_heat (collection, heat_score, last_queried_at, query_count)
                 VALUES (?1, 100.0, ?2, ?3)",
                rusqlite::params![collection, now, count],
            )?;
        }
    }

    tracing::debug!("Heat flush: {} collections updated", buffer.len());
    Ok(())
}

// ── Query helpers ────────────────────────────────────────────────────

/// Get heat snapshot for all collections, ordered by heat_score descending.
pub fn get_all_heat() -> Result<Vec<CollectionHeat>> {
    let conn = super::get_conn()?;
    let mut stmt = conn.prepare(
        "SELECT collection, heat_score, last_queried_at, query_count
         FROM collection_heat
         ORDER BY heat_score DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(CollectionHeat {
            collection: row.get(0)?,
            heat_score: row.get(1)?,
            last_queried_at: row.get(2)?,
            query_count: row.get(3)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| anyhow::anyhow!(e))
}

/// Map a set of source_ids to their collection names.
/// Used by the pipeline to know which collections were touched by a query.
pub fn collections_for_sources(source_ids: &[i64]) -> Result<Vec<String>> {
    if source_ids.is_empty() {
        return Ok(Vec::new());
    }
    let conn = super::get_conn()?;
    let placeholders: Vec<String> = (0..source_ids.len())
        .map(|_| "?".to_string())
        .collect();
    let sql = format!(
        "SELECT DISTINCT collection_id FROM sources WHERE id IN ({})",
        placeholders.join(",")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = source_ids
        .iter()
        .map(|id| id as &dyn rusqlite::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        let coll: String = row.get(0)?;
        Ok(coll)
    })?;
    rows.filter_map(Result::ok).collect::<Vec<_>>()
        .into_iter()
        .try_fold(Vec::new(), |mut acc, coll| {
            acc.push(coll);
            Ok::<_, anyhow::Error>(acc)
        })
}

// ── Utils ────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
