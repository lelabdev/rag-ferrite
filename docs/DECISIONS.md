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

| Chunks | Books (est.) | Search speed (linear scan) | sqlite-vec needed? |
|--------|-------------|---------------------------|-------------------|
| 1K | ~2 | Instant (<5ms) | No |
| 10K | ~20 | Fast (<20ms) | No |
| 50K | ~100 | Noticeable (~100ms) | Starting to make sense |
| 100K | ~200 | Slow (~200ms+) | Yes |
| 500K | ~1000 | Very slow (seconds) | Absolutely needed |

## What we don't do (and why)

### GraphRAG / Multi-hop reasoning
**Why not:** Adds a knowledge graph layer. Complex to implement, heavy to maintain. At our scale, the LLM does multi-hop naturally when given multiple relevant chunks.
**Revisit if:** User needs complex cross-document reasoning that the LLM can't handle.

### External vector databases (Qdrant, Milvus, Weaviate)
**Why not:** Breaks the single-binary model. Adds operational complexity (Docker, separate process, config). At our scale (<50K chunks), SQLite is sufficient.
**Revisit if:** Performance degrades above 50K chunks. sqlite-vec can extend SQLite's capacity significantly before needing this.

### HNSW / dedicated vector index
**Why not:** Would need an external crate (instant-distance, hnswlib) or migration to sqlite-vec. Linear scan is instant at our current scale (~1000 chunks).
**Revisit when:** Approaching 50K+ chunks. sqlite-vec is the planned path — same SQLite, just add an index.
**Status:** Researched. sqlite-vec integrates easily with our BLOB embeddings. Mode 1 (scalar functions, no migration) is ~15min. Mode 2 (vec0 virtual table) for full indexing when needed.

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
- Storage: SQLite, single file, portable
- Scale: comfortable up to 50K chunks without sqlite-vec, 500K+ with sqlite-vec

## What we might do later (not now)

### sqlite-vec integration
**What:** Add vector index to SQLite for faster search at scale.
**Why not now:** Not needed at current scale (~1000 chunks). Research shows it's easy — same BLOB format, no migration. Mode 1 (scalar distance functions) is ~15min.
**When:** Approaching 50K+ chunks.

### #107 — ingest_text is a 214-line mega-function
**Why not now:** Refactoring would help with parallel parents but it's a big rewrite with risk of breaking things.
**When:** As part of parallel parents implementation.
