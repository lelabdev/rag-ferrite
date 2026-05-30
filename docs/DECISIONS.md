# RAG Ferrite — Decisions

## What we do
- Personal/small-team RAG (PMI, freelancer, small business)
- Single binary, Rust, SQLite, no external dependencies
- Hybrid BM25 + vector search
- Parent-child chunking with contextual retrieval
- Progressive commit with resume
- MCP + HTTP API

## Scale reference

Average book (~300 pages) ≈ 500 chunks
- Short book (~150 pages) ≈ 250 chunks
- Technical docs (framework API reference) ≈ 1000-3000 chunks

| Chunks | Books (est.) | Search speed (HNSW) | Notes |
|--------|-------------|---------------------|-------|
| 1K | ~2 | Instant (<5ms) | Current scale |
| 10K | ~20 | Fast (<10ms) | HNSW handles easily |
| 50K | ~100 | Fast (<20ms) | No problem |
| 100K | ~200 | Fast (<50ms) | HNSW scales well |
| 500K | ~1000 | OK (<100ms) | May need tuning |

## What we don't do (and why)

### GraphRAG / Multi-hop reasoning
**Why not:** Adds a knowledge graph layer. Complex to implement, heavy to maintain. At our scale, the LLM does multi-hop naturally when given multiple relevant chunks.
**Revisit if:** User needs complex cross-document reasoning that the LLM can't handle.

### External vector databases (Qdrant, Milvus, Weaviate)
**Why not:** Breaks the single-binary model. Adds operational complexity (Docker, separate process, config). At our scale (<50K chunks), SQLite is sufficient.
**Revisit if:** Performance degrades above 50K chunks. sqlite-vec can extend SQLite's capacity significantly before needing this.

### HNSW / dedicated vector index
**Decision:** Already implemented via rag_engine v0.8 (`hnsw_index` module). Index built after each ingestion, persisted as `.hnsw.data`/`.hnsw.graph` files, loaded on search.
**Status:** Active. No need for sqlite-vec — HNSW covers the same use case with better performance.

### sqlite-vec
**Decision:** Not needed. rag_engine provides HNSW indexing out of the box. sqlite-vec would be redundant.
**Revisit if:** HNSW index doesn't scale past 500K+ chunks.

### Python ecosystem (LlamaIndex, LangChain, Haystack)
**Why not:** We're Rust. Python adds a runtime, virtualenv, dependency hell, and slower performance. Our target users don't want to install Python.
**Never revisit.**

### Large-scale enterprise features (RBAC, multi-tenant, compliance)
**Why not:** Not our target. Personal/small-team RAG. Enterprise features would bloat the binary and complexity.
**Never revisit.**

### Multi-modal (images, audio, video)
**Why not:** Out of scope for now. Text-only keeps it simple.
**Revisit if:** Clear user need emerges.

## Performance targets
- Ingestion: <10 min for a 500K document (with parallel parents)
- Search: <50ms for any query
- Storage: SQLite + HNSW, single file + index files, portable
- Scale: comfortable up to 500K+ chunks (HNSW handles indexing natively via rag_engine)

## What we might do later (not now)

### sqlite-vec integration
**Decision:** Not needed. rag_engine v0.8 already provides HNSW vector indexing. Index is built after each ingestion and persisted to disk.
**Closed:** Issue #137. HNSW covers all scaling needs at our scale.

### #107 — ingest_text is a 214-line mega-function
**Status:** Done. Refactored into shared functions: `generate_contexts`, `compute_relevance_stats`, `process_parent`, `commit_parent_to_db`.

### #128 — api.rs delete_document bypasses json_response
**Decision:** Won't fix. DELETE endpoints have different response semantics (no body, status-only). Forcing the same json_response pattern adds complexity for no functional gain.

### #122 — Chunker byte position allocates intermediate Strings
**Decision:** Won't fix. Entire chunker is built on Vec<char> indexing. Switching to char_indices() would require rewriting most of the chunker (splits, overlaps, positions) for negligible perf gain (~ms on 1MB docs).

### #130 — HTTP graph defaults should be config
**Decision:** Won't fix. Graph defaults are per-request query params by design, not global config.
