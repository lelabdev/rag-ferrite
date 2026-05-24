# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

A personal RAG engine that does one thing well: **turn your documents into queryable knowledge, fast.**

Single binary, single database, multi-collection. MCP-native. Built in Rust because your personal knowledge base shouldn't need a Kubernetes cluster.

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
# → MCP server on stdin, ready for Hermes / Claude / any MCP client
```

That's it. Use it from your MCP client, or ingest your first document via the MCP tools:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

## Why this exists

The RAG space is dominated by business solutions. LangChain, LlamaIndex, Pinecone, Weaviate — they're built for teams, for scale, for enterprise. And that's fine. But when you want to search through your own books, docs, and notes? You end up with 47 abstractions, a managed vector database subscription, and 12 microservices to do what amounts to: *put text in, get text out.*

**rag-ferrite is the personal take.**

- **One binary** — `cargo build --release`, done. No containers, no orchestration.
- **One database** — SQLite. Backup with `cp`. No Pinecone, no Weaviate, no subscription.
- **Any embedding provider** — Ollama (local, free), OpenAI, OpenRouter, whatever. Change one URL in config.
- **Collections** — not separate databases, not separate services. Just a column. Create on-the-fly.
- **MCP-native** — runs as a stdio MCP server. No HTTP overhead, no auth layer, no port to expose.

Think of it as a semantic search engine for your personal library. Books, papers, API docs, RPG manuals — throw text at it, it makes it searchable.

**Simple doesn't mean dumb.** Hybrid BM25 + HNSW search with RRF fusion. Contextual retrieval. Custom chunker. Persistent indexes. All in 15 MB of RAM at idle.

## What rag-ferrite is (and isn't)

rag-ferrite is a **textual document retrieval engine**. It excels at making dense, structured content searchable — books, technical documentation, research papers, RPG manuals, API docs.

**It is not:**

- **A knowledge graph.** There's no entity extraction, no relationship mapping, no graph traversal. If your use case is connecting hundreds of fragmented notes via semantic links (à la Obsidian), you need a different tool — one that builds and queries a graph, not just a vector index.
- **A multimodal processor.** Images, tables, equations, charts — rag-ferrite doesn't process them. It manages text. Extract text first, then feed it to the engine. This is by design: separation of concerns keeps the pipeline predictable and debuggable.
- **An Obsidian plugin.** Pointing a vector search at a vault of 500 three-line notes won't give you a "second brain." It'll give you 500 poorly-embedded chunks with no structure. The value of Obsidian is the graph of links between notes — and that graph is lost the moment you chunk files into vectors.

**Why no knowledge graph?** Because for structured documents (books, papers, docs), vector + keyword search is simpler, faster, and often more accurate than graph-based retrieval. A knowledge graph becomes essential when you're dealing with fragmented, interconnected notes — a fundamentally different problem that deserves its own tool.

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

## Architecture

```
rag-ferrite/
├── config.toml
├── .env                 ← LLM_API_KEY for contextual retrieval
├── data/
│   ├── rag.sqlite3      ← all collections, one DB
│   ├── hnsw_*.hnsw.data ← persisted HNSW indexes
│   └── hnsw_*.hnsw.graph
└── rag-ferrite.log
```

Collections are first-class. Each document belongs to one collection. The HNSW index is built and persisted per-collection — loaded from disk on first query (instant), rebuilt only after new ingestion.

Collections are created on-the-fly during ingestion. No setup, no schema, just start ingesting.

## Pipeline

```
Document → Pre-ingestion check (quality preview, duplicates, language)
         → pdftotext extractor (PDFs) or raw text
         → Recursive character chunker (800 chars, 10% overlap)
         → Post-chunking verification (coverage ratio, empty chunk detection)
         → Contextual retrieval (LLM context prefix + domain metadata extraction)
         → Relevance filtering (discard low-quality chunks, optional)
         → Batch embedding (any OpenAI-compatible provider)
         → SQLite + HNSW + BM25 indexes
         → Ingestion report (timing, relevance stats, filtered chunks)
         → Persist HNSW to disk

Query → MCP tool call
      → Query expansion (multi-query for complex questions)
      → Hybrid retrieval (BM25 + HNSW + RRF fusion)
      → LLM reranking (score results by relevance)
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
        "LLM_API_KEY": "sk-..."
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
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, collection?, limit?)` | Evaluate retrieval quality against a golden dataset |

## What we built on top of rag_engine

rag_engine provides the core (HNSW, BM25, SQLite). We added:

- **`src/extractor.rs`** — PDF extraction via `pdftotext` (poppler-utils). rag_engine's built-in `pdf-extract` crate returns 75% empty pages on complex PDFs. pdftotext is the gold standard.
- **`src/chunker.rs`** — Recursive character text splitter with UTF-8 boundary safety, content-type detection, section path tracking, and page-aware chunking.
- **`src/engine.rs`** — Collection-aware ingestion with chunk metadata persistence, HNSW index management, pre-ingestion checks, post-chunking verification, and ingestion reporting.
- **`src/pipeline.rs`** — Adaptive query routing (simple/standard/complex) with LLM reranking and corrective RAG.
- **`src/reranker.rs`** — Post-retrieval reranking via LLM scoring or Cohere API. Keeps hybrid score + rerank score separate.
- **`src/embedding.rs`** — Batch embedding with proper API format (`input` not `prompt`, `embeddings` not `embedding`).
- **`src/main.rs`** — MCP stdio server with HTTP API (Axum) and file logging for debugging.

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
provider = "your-provider"              # Any OpenAI-compatible provider
model = "your-model-here"
base_url = "https://your-provider-url/v1"
context_enabled = true              # Enable contextual retrieval (Anthropic technique)
max_concurrent = 3                 # Max parallel LLM requests (lower for rate-limited APIs)
# relevance_scoring = true         # Enable ingestion-time quality filter (requires context_enabled)
# min_relevance_score = 5.0         # Discard chunks rated below this (1–10, default 5.0)
# api_key loaded from LLM_API_KEY env var

# Optional: fallback LLM when primary is unavailable
# [llm.fallback]
# provider = "your-provider"
# model = "your-fallback-model"
# base_url = "https://your-provider-url/v1"
# api_key loaded from FALLBACK_API_KEY env var
```

### Reranking

Post-retrieval reranking improves precision by scoring the top-k results with an LLM. When enabled, results carry both the **hybrid score** (BM25 + HNSW fusion) and a **rerank_score** (0.0–1.0 LLM relevance). Results are sorted by rerank_score when available.

**Two backends:**
- `llm` — Uses your configured LLM to score each passage. Inherits provider/model/api_key/base_url from `[llm]` if not specified.
- `cohere` — Cohere Rerank API (requires `api_key`).

```toml
[reranker]
reranker_type = "llm"    # "disabled" (default), "llm", or "cohere"
top_k = 10               # Number of top results to rerank
# model = "..."          # Override LLM model (defaults to llm.model)
# api_key = "..."        # Override API key (defaults to llm.api_key)
# base_url = "..."       # Override base URL (defaults to llm.base_url)
```

When reranking fails (API error, rate limit), results fall back to hybrid scores with a warning log. No data loss, no silent degradation.


### Domain metadata extraction

Extract structured metadata fields from each chunk during ingestion. The LLM identifies domain-specific attributes alongside the contextual prefix — stored as JSON in chunk metadata for filtering and enrichment.

```toml
[metadata]
fields = [
  { name = "topic", field_type = "string" },
  { name = "difficulty", field_type = "string", description = "beginner, intermediate, or advanced" },
  { name = "author", field_type = "string", required = false },
]
```

Fields are extracted during contextual retrieval (no extra LLM calls). Use `metadata_like` in search filters to query by extracted values.

### Golden Dataset benchmarking

Measure retrieval quality objectively with a golden dataset — a JSON file of question → expected source mappings.

```json
[
  {
    "question": "What are the rules for grappling?",
    "relevant_source_ids": [1, 5],
    "expected_keywords": ["grapple", "strength", "obstacle"]
  }
]
```

Run `benchmark(file_path: "golden.json")` to get hit rate, average score, and per-query details. Use it to catch regressions after config changes, embedding model switches, or pipeline tweaks.

### Contextual retrieval

The LLM is used for **contextual retrieval** — generating a 1-2 sentence context prefix for each chunk before embedding. This dramatically improves search quality (Anthropic's technique). Any instruction-following model works. Popular free options on OpenRouter include Qwen3 Next 80B, GLM 4.5 Air, and Gemma 4 31B. For local inference, Gemma 4 E2B Q4 fits in 6GB VRAM.

Set your API key in the `.env` file (see `.env.example`).

### Relevance scoring

Optional ingestion-time quality filter. When enabled, the LLM rates each chunk on a 1–10 relevance scale during contextual retrieval. Chunks scoring below the threshold are **discarded before embedding** — they never enter the vector space.

**What it filters out:** table-of-contents entries, index pages, legal mentions / copyright notices, blank or near-blank pages, and transition text ("Chapter 3 begins on the next page").

**Why use it:** cleaner vector space, less RAM usage, better retrieval precision. Junk chunks that would dilute search results are simply never indexed.

**Cost:** zero extra LLM calls. The relevance score is produced alongside the contextual retrieval prefix in the same prompt — it's a single additional line in the output.

**Backward compatible:** off by default. Existing configs work without any changes.

**How to enable:**

```toml
[llm]
context_enabled = true
relevance_scoring = true              # Enable relevance filtering
min_relevance_score = 5.0             # Discard chunks rated below this (1–10, default 5.0)
```

Both `relevance_scoring` and `context_enabled` must be true. If contextual retrieval is off, there's no LLM call to piggyback on, so relevance scoring has nothing to attach to.

## Requirements

- Rust toolchain (edition 2024)
- An embedding provider — Ollama with bge-m3 (GPU recommended, CPU works), or any OpenAI-compatible API
- `poppler-utils` for PDF extraction (`apt install poppler-utils`)

## License

MIT
