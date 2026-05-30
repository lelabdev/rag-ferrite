# RAG Ferrite — Architecture Decision Records

Decisions are documented as individual ADR files in `docs/adr/`.

## Active decisions

| ADR | Title | Status |
|-----|-------|--------|
| [0001](adr/0001-single-binary-no-external-dependencies.md) | Single binary, no external dependencies | Accepted |
| [0002](adr/0002-hybrid-bm25-+-hnsw-vector-search-via-rag_engine.md) | Hybrid BM25 + HNSW vector search via rag_engine | Accepted |
| [0003](adr/0003-no-python-ecosystem.md) | No Python ecosystem | Accepted |
| [0004](adr/0004-no-enterprise-features.md) | No enterprise features | Accepted |
| [0005](adr/0005-parent-child-chunking-with-contextual-retrieval.md) | Parent-child chunking with contextual retrieval | Accepted |
| [0006](adr/0006-merge-consecutive-small-children.md) | Merge consecutive small children | Accepted |
| [0007](adr/0007-no-external-vector-databases.md) | No external vector databases | Accepted |
| [0008](adr/0008-no-graphrag-or-multi-hop-reasoning.md) | No GraphRAG or multi-hop reasoning | Accepted |
| [0012](adr/0012-parallel-parents-with-joinset-for-ingestion-speed.md) | Parallel parents with JoinSet | Accepted |

## Won't fix

| ADR | Title | Issue |
|-----|-------|-------|
| [0009](adr/0009-delete-endpoint-uses-different-response-pattern.md) | DELETE endpoint uses different response pattern | #128 |
| [0010](adr/0010-chunker-uses-vecchar-indexing-not-char_indices.md) | Chunker uses Vec<char> indexing | #122 |
| [0011](adr/0011-http-graph-defaults-are-per-request-query-params.md) | HTTP graph defaults are per-request query params | #130 |

## Scale reference

| Chunks | Books (est.) | Search speed (HNSW) | Notes |
|--------|-------------|---------------------|-------|
| 1K | ~2 | Instant (<5ms) | Current scale |
| 10K | ~20 | Fast (<10ms) | HNSW handles easily |
| 50K | ~100 | Fast (<20ms) | No problem |
| 100K | ~200 | Fast (<50ms) | HNSW scales well |
| 500K | ~1000 | OK (<100ms) | May need tuning |

## Performance targets
- Ingestion: <10 min for a 500K document (with parallel parents)
- Search: <50ms for any query
- Storage: SQLite + HNSW, single file + index files, portable
- Scale: comfortable up to 500K+ chunks (HNSW via rag_engine)
