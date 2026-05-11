# rag-ferrite

A personal RAG engine that does one thing well: **turn your documents into queryable knowledge, fast.**

Single binary, single database, multi-collection. Built in Rust because your personal knowledge base shouldn't need a Kubernetes cluster.

## Quick start

```bash
# 1. Build
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite && cargo build --release

# 2. Configure
cp config.example.toml config.toml
# Edit config.toml — set your embedding provider (Ollama, OpenAI, OpenRouter...)

# 3. Run
./target/release/rag-ferrite
# → MCP server on stdin, HTTP API on http://localhost:3456
```

That's it. Ingest your first document:

```bash
# Via HTTP API
curl -X POST http://localhost:3456/ingest/data \
  -H "Content-Type: application/json" \
  -d '{"content": "Your text here", "source": "my-first-doc"}'

# Or ingest a PDF
curl -X POST http://localhost:3456/ingest/file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf", "collection": "my-docs"}'

# Query it
curl -X POST http://localhost:3456/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what did I write about?"}'
```

For MCP integration (Claude, Hermes, etc.), see [Configuration](#configuration) below.

## Why this exists

The RAG space is dominated by business solutions. LangChain, LlamaIndex, Pinecone, Weaviate — they're built for teams, for scale, for enterprise. And that's fine. But when you want to search through your own books, docs, and notes? You end up with 47 abstractions, a managed vector database subscription, and 12 microservices to do what amounts to: *put text in, get text out.*

**rag-ferrite is the personal take.**

- **One binary** — `cargo build --release`, done. No containers, no orchestration.
- **One database** — SQLite. Backup with `cp`. No Pinecone, no Weaviate, no subscription.
- **Any embedding provider** — Ollama (local, free), OpenAI, OpenRouter, whatever. Change one URL in config.
- **Collections** — not separate databases, not separate services. Just a column. Create on-the-fly.

Think of it as a semantic search engine for your personal library. Books, papers, API docs, RPG manuals — throw text at it, it makes it searchable.

**Simple doesn't mean dumb.** Hybrid BM25 + HNSW search with RRF fusion. Custom chunker. GPU-accelerated embeddings. Persistent indexes. All in 15 MB of RAM at idle.

## Why not just use LangChain?

You absolutely can. LangChain is great — if you need multi-tenant isolation, pluggable retriever chains, agent orchestration, and a team to maintain the pipeline. For personal use, that's a lot of machinery for what boils down to: chunk text → embed → store → query. rag-ferrite does exactly that, no more, no less.

## Why not mcp-local-rag?

We used [mcp-local-rag](https://github.com/nicholasgriffintn/mcp-local-rag) before building this. It works, it's simple, and it got us started. But we wanted more control over the pipeline:

| | mcp-local-rag | rag-ferrite |
|---|---|---|
| **Language** | TypeScript | Rust |
| **Chunking** | Basic line-based | Custom recursive character splitter (800 chars, 10% overlap) |
| **PDF extraction** | pdf-extract crate (75% empty pages on complex PDFs) | pdftotext via poppler-utils (gold standard) |
| **Search** | Vector only | Hybrid BM25 + HNSW + RRF fusion |
| **Index persistence** | Rebuild on every restart | HNSW saved to disk, lazy-loaded |
| **Embeddings** | Local or API | Configurable — any OpenAI-compatible provider |

## Stack

| Component | Choice | Why |
|---|---|---|
| **RAG core** | [`rag_engine`](https://lib.rs/crates/rag_engine) v0.8.1 | HNSW, BM25, hybrid RRF fusion, SQLite — does the heavy lifting |
| **MCP server** | [`rmcp`](https://github.com/anthropics/rmcp-rust-sdk) | Standard MCP protocol — works with Claude, Hermes, any MCP client |
| **Embeddings** | Configurable | Defaults to bge-m3 via Ollama, but any OpenAI-compatible API works |
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
         → Recursive character chunker (800 chars, 10% overlap)
         → Batch embedding (any OpenAI-compatible provider)
         → SQLite + HNSW + BM25 indexes
         → Persist HNSW to disk

Query → MCP tool call or HTTP POST
      → Activate collection (load index from disk)
      → Hybrid retrieval (BM25 + HNSW + RRF fusion)
      → Top-k chunks with optional neighbor expansion
```

## Performance

| Metric | Value |
|---|---|
| Embedding | bge-m3 via Ollama on RTX 4050 — 46ms/embedding |
| Ingestion | ~3.5 chunks/sec including embedding (GPU) |
| Query (warm) | ~300ms (index loaded from disk) |
| Query (cold) | ~4s (first-time index build) |
| Memory idle | ~15 MB |
| 3 books (1.4M chars) | 4,699 chunks in 200s |

## MCP Tools

rag-ferrite runs as a native MCP server via stdio. Add it to your MCP client config:

```json
{
  "mcpServers": {
    "rag-ferrite": {
      "command": "/path/to/rag-ferrite",
      "args": [],
      "env": {
        "EMBEDDING_API_KEY": "sk-..." 
      }
    }
  }
}
```

Available tools:

| Tool | Description |
|---|---|
| `query_documents(query, collection?, limit?)` | Hybrid search with optional collection filter |
| `ingest_file(file_path, collection?)` | Ingest PDF/TXT/MD/DOCX |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text or markdown |
| `delete_file(source)` | Remove document by source identifier |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |

## HTTP API

For non-MCP clients, rag-ferrite exposes a REST API:

```
GET  /health       # Health check
GET  /status       # Document count, version
POST /query        # Hybrid search — {"query": "...", "collection": "...", "limit": 10}
POST /ingest/file  # Ingest PDF/DOCX — {"file_path": "...", "collection": "..."}
POST /ingest/data  # Ingest raw text — {"content": "...", "source": "...", "collection": "..."}
```

## What we built on top of rag_engine

rag_engine provides the core (HNSW, BM25, SQLite). We added:

- **`src/extractor.rs`** — PDF extraction via `pdftotext` (poppler-utils). rag_engine's built-in `pdf-extract` crate returns 75% empty pages on complex PDFs. pdftotext is the gold standard.
- **`src/chunker.rs`** — Recursive character text splitter with UTF-8 boundary safety. rag_engine's semantic chunker freezes on documents >100K chars and over-splits short paragraphs.
- **`src/engine.rs`** — Collection-aware ingestion with proper `collection_id` routing, status tracking, and HNSW index persistence per collection.
- **`src/pipeline.rs`** — Adaptive query routing (simple/standard/complex) with reranker passthrough when disabled.
- **`src/embedding.rs`** — Batch embedding with proper API format (`input` not `prompt`, `embeddings` not `embedding`).
- **`src/main.rs`** — Dual MCP + HTTP mode with file logging for debugging.

## Configuration

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "ollama"             # "ollama" or "openai" (any OpenAI-compatible API)
model = "bge-m3:latest"
dimensions = 1024
base_url = "http://localhost:11434"
# api_key = "sk-..."           # For OpenAI/OpenRouter — or set EMBEDDING_API_KEY env var

[http]
port = 3456

[llm]
context_enabled = false         # Enable contextual retrieval (needs LLM API key)
```

## Requirements

- Rust toolchain (edition 2024)
- An embedding provider — Ollama with bge-m3 (GPU recommended, CPU works), or any OpenAI-compatible API
- `poppler-utils` for PDF extraction (`apt install poppler-utils`)

## License

MIT
