# ADR 0014: Non-blocking ingestion queue

**Status:** Accepted
**Date:** 2026-05-30
**Issue:** #140

## Context

During ingestion of large files (1M+ chars), the HTTP server became completely unresponsive. Users could not query the RAG, check status, or do anything until ingestion finished. For files taking 30+ minutes, this was unacceptable.

## Decision

Implement an ingestion queue using `tokio::sync::mpsc` channel with a background worker task spawned via `tokio::spawn` on the main runtime.

- HTTP handlers return immediately with `{"status": "queued"}`
- Background worker processes jobs sequentially
- Progress available via `GET /api/ingest/progress`
- Queries work during ingestion

### Rejected: Separate tokio runtime

Also tried running the worker on a separate `tokio::Runtime` via `std::thread::spawn + Runtime::new()`. This degraded ingestion speed 10x (2.5 min/parent vs 5-15s). Cause: HTTP clients (reqwest) created on the main runtime performed poorly when used from a different runtime.

## Consequences

- Server stays responsive during ingestion
- Ingestion speed unchanged (same runtime, same performance)
- Ollama Cloud can still be slow during concurrent ingestion + queries (provider-side contention)
- Sequential ingestion only — one file at a time (acceptable for single-user system)
