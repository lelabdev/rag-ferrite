<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">
</div>

# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

A personal RAG engine that does one thing well: **turn your documents into queryable knowledge, fast.**

Single binary, single database, multi-collection. MCP-native. Built in Rust because your personal knowledge base shouldn't need a Kubernetes cluster.

## Quick start

**Option A — Download the binary** (recommended):

Grab the latest release for your platform from the [releases page](https://github.com/lelabdev/rag-ferrite/releases/latest).

**Option B — Build from source:**

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite && cargo build --release
```

Then:

```bash
# Configure
cp config.example.toml config.toml
# Edit config.toml — set your embedding and LLM providers

# Run
./rag-ferrite
# → MCP server on stdin, ready for Hermes / Claude / any MCP client
```

Use it from your MCP client:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

## Why this exists

The RAG space is dominated by business solutions. LangChain, LlamaIndex, Pinecone, Weaviate — they're built for teams, for scale, for enterprise. But when you want to search through your own books, docs, and notes? You end up with 47 abstractions, a managed vector database subscription, and 12 microservices to do what amounts to: *put text in, get text out.*

**rag-ferrite is the personal take.**

- **One binary** — download or `cargo build --release`, done. No containers, no orchestration.
- **One database** — SQLite. Backup with `cp`. No Pinecone, no Weaviate, no subscription.
- **Any embedding provider** — Ollama (local, free), OpenAI, OpenRouter, whatever. Change one URL in config.
- **Collections** — not separate databases, not separate services. Just a column. Create on-the-fly.
- **MCP-native** — runs as a stdio MCP server. No HTTP overhead, no auth layer, no port to expose.

Think of it as a semantic search engine for your personal library. Books, papers, API docs, RPG manuals — throw text at it, it makes it searchable.

**Simple doesn't mean dumb.** Hybrid BM25 + HNSW search with RRF fusion. Contextual retrieval. Custom chunker. Persistent indexes. All in 15 MB of RAM at idle.

## Recommended setup

For the LLM used in contextual retrieval, we recommend **Qwen models via OpenRouter** — cheap, fast, and excellent multilingual support. Something like `qwen/qwen3-32b` works great for generating context prefixes in any language without breaking the bank.

For embeddings, **bge-m3 via Ollama** (local, free, GPU recommended) or any OpenAI-compatible provider.

```toml
[embedding]
provider = "ollama"
model = "bge-m3:latest"
dimensions = 1024
base_url = "http://localhost:11434"

[llm]
provider = "openrouter"
model = "qwen/qwen3-32b"
base_url = "https://openrouter.ai/api/v1"
context_enabled = true
# api_key loaded from LLM_API_KEY env var
```

## What rag-ferrite is (and isn't)

rag-ferrite is a **textual document retrieval engine**. It excels at making dense, structured content searchable — books, technical documentation, research papers, RPG manuals, API docs.

**It is not:**

- **A knowledge graph.** No entity extraction, no relationship mapping. If your use case is connecting fragmented notes via semantic links (à la Obsidian), you need a different tool.
- **A multimodal processor.** Images, tables, equations — rag-ferrite manages text. Extract first, then feed it.
- **An Obsidian plugin.** Pointing a vector search at a vault of 500 three-line notes won't give you a "second brain." The value of Obsidian is the graph of links — and that graph is lost the moment you chunk files into vectors.

## Why not just use LangChain?

You absolutely can. LangChain is great — if you need multi-tenant isolation, pluggable retriever chains, agent orchestration, and a team to maintain the pipeline. For personal use, that's a lot of machinery for what boils down to: chunk text → embed → store → query.

## Why not mcp-local-rag?

| | mcp-local-rag | rag-ferrite |
|---|---|---|
| **Language** | TypeScript | Rust |
| **Chunking** | Basic line-based | Custom recursive character splitter (800 chars, 10% overlap) |
| **PDF extraction** | pdf-extract (75% empty pages on complex PDFs) | pdftotext via poppler-utils (gold standard) |
| **Search** | Vector only | Hybrid BM25 + HNSW + RRF fusion |
| **Index persistence** | Rebuild on every restart | HNSW saved to disk, lazy-loaded |
| **Embeddings** | Local or API | Configurable — any OpenAI-compatible provider |

## Architecture

```
Document → Pre-ingestion check (quality, duplicates, language)
         → pdftotext (PDFs) or raw text
         → Recursive chunker (800 chars, 10% overlap)
         → Contextual retrieval (LLM prefix + metadata)
         → Relevance filtering (optional)
         → Batch embedding → SQLite + HNSW + BM25
         → Persist HNSW to disk

Query → MCP tool call
      → Hybrid retrieval (BM25 + HNSW + RRF fusion)
      → LLM reranking (optional)
      → Top-k chunks with neighbor expansion
```

```
rag-ferrite/
├── config.toml
├── .env                 ← LLM_API_KEY
├── data/
│   ├── rag.sqlite3      ← all collections, one DB
│   ├── hnsw_*.hnsw.data ← persisted HNSW indexes
│   └── hnsw_*.hnsw.graph
└── rag-ferrite.log
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

| Tool | Description |
|---|---|
| `query_documents(query, collection?, limit?)` | Hybrid search with optional collection filter |
| `ingest_file(file_path, collection?)` | Ingest PDF/TXT/MD/DOCX |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text or markdown |
| `delete_file(source)` | Remove document by source identifier |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, collection?, limit?)` | Evaluate retrieval quality against a golden dataset |

## Configuration

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "ollama"
model = "bge-m3:latest"
dimensions = 1024
base_url = "http://localhost:11434"

[llm]
provider = "openrouter"
model = "qwen/qwen3-32b"
base_url = "https://openrouter.ai/api/v1"
context_enabled = true
max_concurrent = 3
# api_key loaded from LLM_API_KEY env var
```

For advanced options (reranking, metadata extraction, golden dataset benchmarking, relevance scoring), see [docs/advanced.md](docs/advanced.md).

## Requirements

- An embedding provider — Ollama with bge-m3 (GPU recommended, CPU works), or any OpenAI-compatible API
- An LLM provider for contextual retrieval — OpenRouter recommended (cheap, multilingual)
- `poppler-utils` for PDF extraction (`apt install poppler-utils`)

## License

MIT
