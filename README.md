# rag-ferrite

A personal RAG engine that does one thing well: **turn your documents into queryable knowledge, fast.**

Single binary, single database, multi-collection. Built in Rust because your personal knowledge base shouldn't need a Kubernetes cluster.

## Why this exists

Most RAG frameworks are built for enterprise. LangChain, LlamaIndex — they're powerful, but for a personal knowledge base they're overkill. You don't need 47 abstractions, a vector database, and 12 microservices to search through your PDFs.

rag-ferrite is the opposite approach:

- **One binary** — no containers, no orchestration, no YAML files
- **One database** — SQLite, backup with `cp`
- **One model** — bge-m3 on a local GPU, no API keys needed
- **Collections** — not separate databases, not separate services, just a column

Think of it like Obsidian's graph — you organize notes into folders, link them together, search across everything. rag-ferrite does the same for any document: books, docs, papers, whatever. You throw text at it, it makes it searchable with semantic understanding.

## Why not mcp-local-rag?

We used [mcp-local-rag](https://github.com/nicholasgriffintn/mcp-local-rag) before building this. It's a fine project, but we hit walls:

| | mcp-local-rag | rag-ferrite |
|---|---|---|
| **Language** | TypeScript | Rust |
| **Collections** | 3 separate instances (3× RAM, 3× processes) | 1 instance, native collection routing |
| **Chunking** | Basic line-based | Custom recursive character splitter (800 chars, 20% overlap) |
| **PDF extraction** | pdf-extract crate (75% empty pages on complex PDFs) | pdftotext via poppler-utils (gold standard) |
| **Embeddings** | Local or API | bge-m3 on GPU (46ms/embedding) |
| **Search** | Vector only | Hybrid BM25 + HNSW + RRF fusion |
| **Index persistence** | Rebuild on every restart | HNSW saved to disk, lazy-loaded |
| **Memory** | ~500 MB × 3 instances | ~15 MB idle, single process |
| **Scalability** | Max ~3 collections before RAM runs out | Collections created on-the-fly, no limit |

The main difference: **mcp-local-rag runs 3 separate services for 3 collections.** rag-ferrite runs 1 service with native collection support. On a home server with 8 GB RAM, that matters.

## Stack

| Component | Choice | Why |
|---|---|---|
| **RAG core** | [`rag_engine`](https://lib.rs/crates/rag_engine) v0.8.1 | HNSW, BM25, hybrid RRF fusion, SQLite — does the heavy lifting |
| **MCP server** | [`rmcp`](https://github.com/anthropics/rmcp-rust-sdk) | Standard MCP protocol — works with Claude, Hermes, any MCP client |
| **Embeddings** | BAAI/bge-m3 via Ollama | SOTA multilingual model, 1024 dims, GPU-accelerated |
| **Storage** | SQLite + HNSW | One file, one backup, zero ops |
| **HTTP bridge** | `axum` | REST API on port 3456 for non-MCP clients |

## Architecture

```
rag-ferrite/
├── config.toml
├── data/
│   ├── rag.sqlite3          ← all collections, one DB
│   ├── hnsw_rpg.index       ← persisted HNSW indexes
│   ├── hnsw_growth.index
│   ├── hnsw_code.index
│   └── hnsw_general.index
└── rag-ferrite.log
```

Collections are first-class. Each document belongs to one collection. The HNSW index is built and persisted per-collection — loaded from disk on first query (instant), rebuilt only after new ingestion.

Collections are created on-the-fly during ingestion. No setup, no schema, just start ingesting.

## Pipeline

```
Document → pdftotext extractor (PDFs) or raw text
         → Recursive character chunker (800 chars, 20% overlap)
         → Batch embedding (bge-m3 on GPU)
         → SQLite + HNSW + BM25 indexes
         → Persist HNSW to disk

Query → MCP tool call
      → Activate collection (load index from disk)
      → Hybrid retrieval (BM25 + HNSW + RRF fusion)
      → Top-k chunks with optional neighbor expansion
```

## Performance

| Metric | Value |
|---|---|
| Embedding | bge-m3 on RTX 4050 GPU — 46ms/embedding |
| Ingestion | ~3.5 chunks/sec including embedding |
| Query (warm) | ~300ms (index loaded from disk) |
| Query (cold) | ~4s (first-time index build) |
| Memory idle | ~15 MB |
| 3 Burning Wheel books (1.4M chars) | 4,699 chunks in 200s |
| n8n hosting docs (33 pages) | 88 chunks in 30s |

## MCP Tools

| Tool | Description |
|---|---|
| `query_documents(query, collection?, limit?)` | Hybrid search with optional collection filter |
| `ingest_file(file_path, collection?)` | Ingest PDF/TXT/MD/DOCX |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text or markdown |
| `delete_file(source)` | Remove document by source identifier |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |

## What we built on top of rag_engine

rag_engine provides the core (HNSW, BM25, SQLite). We added:

- **`src/extractor.rs`** — PDF extraction via `pdftotext` (poppler-utils). rag_engine's built-in `pdf-extract` crate returns 75% empty pages on complex PDFs. pdftotext is the gold standard.
- **`src/chunker.rs`** — Recursive character text splitter with UTF-8 boundary safety. rag_engine's semantic chunker freezes on documents >100K chars and over-splits short paragraphs.
- **`src/engine.rs`** — Collection-aware ingestion with proper `collection_id` routing, status tracking, and HNSW index persistence per collection.
- **`src/pipeline.rs`** — Adaptive query routing (simple/standard/complex) with reranker passthrough when disabled.
- **`src/embedding.rs`** — Batch Ollama embedding with correct API format (`input` not `prompt`, `embeddings` not `embedding`).
- **`src/main.rs`** — Dual MCP + HTTP mode with file logging for debugging.

## Configuration

```toml
# config.toml
data_dir = "/home/loops/services/rag-ferrite/data"
http_port = 3456

[embedding]
provider = "ollama"
model = "bge-m3:latest"
dimensions = 1024
base_url = "http://192.168.1.111:11434"   # Ollama on GPU machine

[llm]
context_enabled = false   # Disable for bulk ingestion (rate limit avoidance)
```

## Requirements

- Rust toolchain (edition 2021)
- Ollama with bge-m3 model (GPU recommended, CPU works)
- `poppler-utils` for PDF extraction (`apt install poppler-utils`)

## License

MIT
