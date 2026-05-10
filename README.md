# rag-ferrite

Custom RAG engine in Rust. Single binary, multi-collection, hybrid search (BM25 + HNSW + RRF fusion).

Built on [rag_engine](https://lib.rs/crates/rag_engine) + [rmcp](https://github.com/anthropics/rmcp-rust-sdk) for MCP server support.

## Stack

| Component | Choice | Purpose |
|---|---|---|
| **RAG core** | `rag_engine` v0.8.1 | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage |
| **MCP server** | `rmcp` | Expose as MCP server (stdio + HTTP) |
| **Embeddings** | BAAI/bge-m3 via Ollama | Multilingual SOTA model, 1024 dims, GPU-accelerated |
| **Storage** | SQLite + HNSW | Single DB file, backup = cp |
| **HTTP bridge** | `axum` | SSE endpoint on port 3456 |

## Architecture

Single binary, single database, **collections** for routing:

```
rag-ferrite/
├── config.toml
├── data/
│   ├── rag.sqlite3          ← all collections in one DB
│   ├── hnsw_rpg.index       ← persisted HNSW indexes
│   ├── hnsw_growth.index
│   ├── hnsw_code.index
│   └── hnsw_general.index
└── rag-ferrite.log
```

**Collections** are first-class — each document belongs to one. The HNSW index is loaded on-demand per collection (`activate_collection_for_hybrid_search`), and persisted to disk after ingestion for fast startup.

## Collections

| Collection | Content |
|---|---|
| `rpg` | Tabletop RPG rules (Burning Wheel) |
| `growth` | Business, marketing, psychology, self-help, health |
| `code` | Development, AI/ML, security, architecture |
| `general` | Everything else (cooking, culture, philosophy, misc) |

Collections are created on-the-fly during ingestion — no setup needed.

## Pipeline

```
Document → Custom pdftotext extractor (for PDFs)
         → Recursive character chunker (800 chars, 160 overlap)
         → Batch embedding (bge-m3 via Ollama on GPU)
         → SQLite + HNSW + BM25 indexes
         → Persist HNSW to disk

Query → MCP tool call
      → Activate target collection (load or rebuild index)
      → Hybrid retrieval (BM25 + HNSW + RRF)
      → Adaptive pipeline (simple/standard/complex routing)
      → Top-k chunks + neighbors
```

## Performance

| Metric | Value |
|---|---|
| Embedding model | bge-m3 on RTX 4050 (TufTux) |
| Chunk size | 800 chars, 160 overlap |
| Ingestion speed | ~3.5 chunks/sec (including embedding) |
| Query latency | ~300ms (index load from disk) or ~4s (first-time rebuild) |
| Memory | ~15 MB idle, spikes during index rebuild |
| 3 Burning Wheel books (1.4M chars) | 4,699 chunks in ~200s |
| 8 general books (4.7 MB) | 9,198 chunks in ~20 min |

## MCP Tools

| Tool | Description |
|---|---|
| `query_documents` | Hybrid search with optional collection filter |
| `ingest_file` | Ingest PDF/TXT/MD with optional collection |
| `ingest_data` | Ingest raw text with source identifier |
| `delete_file` | Remove document by source ID |
| `list_files` | List all indexed documents |
| `status` | Document count + version |
| `read_chunk_neighbors` | Expand context around a chunk |

## Custom Code (on top of rag_engine)

- `src/extractor.rs` — PDF text extraction via `pdftotext` (10x better than pdf-extract crate)
- `src/chunker.rs` — Recursive character text splitter with UTF-8 boundary safety
- `src/engine.rs` — Collection-aware ingestion, status tracking, index persistence
- `src/pipeline.rs` — Adaptive query routing (simple/standard/complex)
- `src/embedding.rs` — Ollama batch embedding with proper API format
- `src/main.rs` — MCP server + HTTP dual mode, file logging

## Configuration

```toml
# config.toml
data_dir = "/home/loops/services/rag-ferrite/data"
http_port = 3456

[embedding]
provider = "ollama"
model = "bge-m3:latest"
dimensions = 1024
base_url = "http://192.168.1.111:11434"

[llm]
provider = "zai"
model = "glm-4.7-flash"
context_enabled = false
```

## Requirements

- Rust toolchain (edition 2021)
- Ollama with bge-m3 model (on GPU for performance)
- `poppler-utils` for pdftotext PDF extraction

## License

MIT
