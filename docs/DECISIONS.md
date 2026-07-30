# RAG Ferrite — Architecture Decision Records

This index lists architectural decisions only. Work planning and prioritization belong to the GitHub issue tracker:

https://github.com/lelabdev/rag-ferrite/issues

Decisions are documented as individual ADR files in `docs/adr/`.

## Active decisions

| ADR | Title | Status |
|-----|-------|--------|
| [0001](adr/0001-single-binary-no-external-dependencies.md) | Single binary, no external dependencies | Accepted |
| [0003](adr/0003-no-python-ecosystem.md) | No Python ecosystem | Accepted |
| [0004](adr/0004-no-enterprise-features.md) | No enterprise features | Accepted |
| [0005](adr/0005-parent-child-chunking-with-contextual-retrieval.md) | Parent-child chunking with contextual retrieval | Accepted |
| [0006](adr/0006-merge-consecutive-small-children.md) | Merge consecutive small children | Accepted |
| [0007](adr/0007-no-external-vector-databases.md) | No external vector databases | Accepted |
| [0008](adr/0008-no-graphrag-or-multi-hop-reasoning.md) | No GraphRAG or multi-hop reasoning | Accepted |
| [0012](adr/0012-parallel-parents-with-joinset-for-ingestion-speed.md) | Parallel parents with JoinSet | Accepted |
| [0013](adr/0013-skip-small-chunks-before-llm.md) | Skip small chunks before LLM call | Accepted |
| [0014](adr/0014-non-blocking-ingestion-queue.md) | Non-blocking ingestion queue | Accepted |
| [0015](adr/0015-single-connection-path-for-sqlite-writes.md) | Single connection path for SQLite writes | Accepted |
| [0016](adr/0016-increase-tokio-worker-threads-for-io-workloads.md) | Increase Tokio worker threads for I/O workloads | Accepted |
| [0017](adr/0017-sqlite-vec-and-fts5-storage.md) | SQLite storage with sqlite-vec and FTS5 | Accepted |

## Superseded decisions

| ADR | Superseded by | Title |
|-----|--------------|-------|
| [0002](adr/0002-hybrid-bm25-+-hnsw-vector-search-via-rag_engine.md) | [0017](adr/0017-sqlite-vec-and-fts5-storage.md) | Hybrid BM25 + HNSW vector search via rag_engine |

## Won't fix

| ADR | Title | Issue |
|-----|-------|-------|
| [0009](adr/0009-delete-endpoint-uses-different-response-pattern.md) | DELETE endpoint uses different response pattern | #128 |
| [0010](adr/0010-chunker-uses-vec<char>-indexing-not-char_indices.md) | Chunker uses Vec<char> indexing | #122 |
| [0011](adr/0011-http-graph-defaults-are-per-request-query-params.md) | HTTP graph defaults are per-request query params | #130 |
