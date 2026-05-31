# 0015: Fix SQLite Deadlock with Single Connection Path

## Status
Accepted

## Context
During ingestion, the server froze completely. No queries, no status endpoint, no logs. Process alive but 0% CPU. Required `kill -9` to recover.

## Root Cause
Two compounding issues:

1. **Non-reentrant Mutex deadlock**: `commit_parent_to_db()` held `get_conn()` Mutex, then called `insert_chunk_tags()` which tried to re-lock the same `std::sync::Mutex`. Rust's `std::sync::Mutex` is NOT reentrant — same thread locking twice = deadlock. This froze the entire tokio runtime because `std::sync::Mutex` blocks the OS thread, not the async task.

2. **Dual connection system**: Code mixed `get_conn()` (single `Mutex<Connection>`) with `source_rag::add_chunks()` (rag_engine crate's connection pool). Two systems competing for SQLite write access.

## Decision
- Replace `source_rag::add_chunks()` calls with direct INSERT statements via `get_conn()` connection
- Make `insert_chunk_tags()` accept `Option<&Connection>` to reuse the caller's existing lock
- All parent-child writes go through a single connection — no pool contention, no deadlock

## Consequences
- Ingestion completes without freezing
- Queries respond in <1s during ingestion
- Single connection path means slightly less write throughput (one writer at a time), but SQLite WAL mode allows concurrent reads
- Future: consider `rusqlite::Connection` with `busy_timeout` instead of `Mutex` for cleaner async integration
