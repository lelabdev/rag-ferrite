# Changelog

All notable changes to rag-ferrite are documented here.

## [4.6.0] - 2025-06-03

### Added
- **Incremental HNSW buffer** — Chunks are immediately searchable after ingestion via dual-index strategy (in-memory buffer + HNSW). No flush needed to search new content. Uses `rag_engine::incremental_index` module.
- **Deferred index rebuild** (`defer_index_rebuild = true`) — New chunks go to incremental buffer instead of triggering full HNSW rebuild. Reduces RAM from ~2 GB to ~30 MB during ingestion.
- **WAL checkpoint** (`wal_checkpoint_interval = 50`) — Periodic SQLite WAL checkpoint during ingestion.
- **Tag rules** (`tag-rules.toml`) — External config for tag normalization: synonym mappings, stop words, filtering rules. No recompilation needed.
- **Tag sanitization pipeline** — Multi-stage: strip chars → lowercase → synonym lookup → stop word filter → length filter → singular normalization → dedup.
- **`POST /api/flush-indexes`** — Trigger HNSW + BM25 rebuild + WAL checkpoint. Merges incremental buffer into persistent index.
- `add_embeddings_to_buffer()` — Adds new embeddings to buffer at ingestion time.

### Changed
- `defer_index_rebuild = true` now means "use incremental buffer" instead of "skip entirely". Chunks are searchable immediately.
- `rebuild_and_save_indexes()` merges incremental buffer before rebuild and clears it after.
- Tag generation prompt improved: "noun phrases only, no adjectives alone, lowercase, hyphenated multi-word"
- `ingest-library.sh` calls `/api/flush-indexes` at end of batch

## [4.5.0] - 2025-06-01

### Changed
- **Removed collections** — Switched to a single flat index with tags-only classification. All multi-collection logic removed from codebase, API, and config.
- **Simplified API** — Removed `collection` parameter from all MCP tools and HTTP endpoints. Queries run against a single unified index.

### Added
- **`/api/rebuild-indexes` endpoint** — Async, non-blocking endpoint to trigger full HNSW + BM25 index rebuild from existing SQLite data.

## [4.4.1] - 2025-05-31

### Fixed
- **Embedding batch failure on large files** — `embed_openai()` sent all chunks in a single request, causing reqwest to fail on files with 1000+ texts. Added configurable batching (default: 20 texts per batch).
- **No retry on embedding failures** — Added 3-attempt retry with 2s delay on embedding request failures.

### Added
- **Embedding batching** — `embed_openai()` now splits texts into configurable batch sizes before sending to the API.
- **Embedding retry logic** — Failed embedding requests retry up to 3 times with delay.
- **Application logging** — `[advanced] log_file` and `log_filter` in config.toml for persistent file logging via tracing.

## [4.4.0] - 2025-05-31

### Fixed
- **SQLite deadlock during ingestion (#140)** — Server froze permanently when `commit_parent_to_db()` held a non-reentrant Mutex and called `insert_chunk_tags()` which tried to re-lock it. Replaced dual connection system (pool + Mutex) with single connection path using direct INSERTs.
- **Worker thread starvation (#140)** — Queries timed out during ingestion because 4 default tokio threads were saturated by parallel LLM calls. Bumped to 12 worker threads.
- **Reqwest timeout (#141)** — Added 120s timeout to all HTTP clients to prevent hanging on unresponsive LLM/embedding APIs.
- **Error feedback in ingestion progress (#144)** — Ingestion errors now appear in `/api/ingest/progress` instead of being swallowed silently.

### Added
- **Modular LLM profiles (#146)** — `[[llm_profile]]` array allows different providers/models for ingestion, query, and reranking. Each action runs independently without provider contention.
- **Query fallback provider (#143)** — `QUERY_FALLBACK_API_KEY` env var for separate query endpoint. Queries use a different provider than ingestion.
- **Parallel parents with JoinSet (#136)** — Parents processed concurrently via `tokio::task::JoinSet` with configurable `max_concurrent`. ~2.5x faster ingestion.
- **Context retry (#107, #138)** — Failed LLM context generation retries individually instead of failing the entire parent.
- **Skip small chunks before LLM call** — Chunks < 200 chars are embedded directly without wasting an LLM call.
- **Merge consecutive small chunks (#139)** — Adjacent tiny children merged for better context quality.

### Changed
- Default `context_batch_size` reduced from 20 to 3 to prevent oversized prompts.
- Separate tokio runtime approach reverted — caused 10x slowdown. Single runtime with more worker threads instead.

### Docs
- ADR 0012-0016 for parallel processing, skip small chunks, non-blocking queue, deadlock fix, worker threads.
- AGENTS.md updated with LLM profiles, architecture decisions.

## [4.3.0] - 2025-05-28

### Added
- Parent-child chunking with contextual retrieval (#135)
- HNSW vector search via rag_engine v0.8
- Hybrid BM25 + vector search with RRF fusion
- MCP server via rmcp (stdio + Streamable HTTP)
- SQLite storage with WAL mode
- Embedding via OpenRouter (Qwen3 8B, 4096 dims)

## [4.2.0] - 2025-05-25

### Added
- Initial chunking pipeline
- Basic ingestion API

## [4.0.0] - 2025-05-20

### Added
- Rewrite from scratch in Rust
- rag_engine integration
