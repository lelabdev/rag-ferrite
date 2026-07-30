# ADR-0017: SQLite storage with sqlite-vec and FTS5

- **Status:** Accepted
- **Date:** 2026-06-29
- **Supersedes:** ADR-0002

## Context

The original design used rag_engine and persisted HNSW files. The production storage implementation now keeps source and chunk data in SQLite, uses sqlite-vec for vector search when available, and uses FTS5 for BM25 retrieval.

## Decision

Use one SQLite database as the source index:

- sqlite-vec stores vector rows and is queried through the SQLite connection;
- FTS5 stores searchable content and is rebuilt from `chunks` when needed;
- hybrid retrieval fuses vector and BM25 rankings with RRF;
- brute-force cosine search remains the vector fallback when sqlite-vec is unavailable;
- WAL and explicit checkpointing remain enabled for durability and maintenance.

There are no `.hnsw.data` or `.hnsw.graph` runtime files. Index rebuild and flush operations rebuild FTS5 and checkpoint SQLite; they do not persist an HNSW buffer.

## Consequences

SQLite backups contain the source data and derived retrieval indexes in one file. sqlite-vec and FTS5 migrations must be tested together with ingestion, deletion, and recovery. ADR-0002 remains historical context only.
