<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">
</div>

# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

A personal RAG engine that does one thing well: **turn your documents into queryable knowledge, fast.**

Single binary, single database, multi-collection. MCP-native. Built in Rust.

## Quick start

**Option A — Download the binary** (recommended):

Grab the latest release from the [releases page](https://github.com/lelabdev/rag-ferrite/releases/latest).

**Option B — Build from source:**

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite && cargo build --release
```

Then configure and run:

```bash
cp config.example.toml config.toml
# Edit config.toml — set your providers (see below)
./rag-ferrite
# → MCP server on stdin, ready for Hermes / Claude / any MCP client
```

**Prerequisites:** `poppler-utils` for PDF support (`apt install poppler-utils`), and API keys for your providers.

## Recommended setup

The simplest setup uses **OpenRouter** for embeddings and **Ollama Cloud** for LLM — no local GPU needed.

1. Create an [OpenRouter](https://openrouter.ai) account and get an API key
2. Create an [Ollama Cloud](https://ollama.com) account (unlimited subscription available)
3. Set up your config:

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0

[reranker]
reranker_type = "llm"
top_k = 10
```

4. Set your API keys: `export LLM_API_KEY=... EMBEDDING_API_KEY=sk-...`
5. Run: `./rag-ferrite`

That's it. Ingest your first document and search:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

**Local alternative:** bge-m3 via Ollama works for embeddings (free, GPU recommended). Just change the `[embedding]` block to point to your local Ollama instance.

## Features

- **Hybrid search** — BM25 + vector (HNSW) + RRF fusion for best of both worlds
- **Relevance scoring** — LLM filters junk chunks (TOC, index, legal) at ingestion
- **Contextual retrieval** — LLM adds context prefix to each chunk for better semantic matching
- **Auto-tagging** — LLM generates tags per chunk for cross-collection filtering
- **LLM reranking** — Post-retrieval scoring for higher precision
- **Query expansion** — Short queries automatically expanded by LLM
- **Query reformulation** — Failed queries auto-retried with reformulated wording
- **Query caching** — Repeated queries return instantly from in-memory cache (300s TTL)
- **Adaptive pipeline** — Queries classified as simple/standard/complex with different strategies
- **Multi-collection** — Organize documents by topic, filter queries by collection
- **Golden dataset benchmarking** — Measure retrieval quality objectively

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

For advanced configuration (reranking, metadata extraction, golden dataset benchmarking, relevance scoring, architecture details), see [docs/advanced.md](docs/advanced.md).

## Acknowledgements

The ingestion pipeline was heavily inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — specifically contextual retrieval, pre-ingestion quality checks, post-chunking verification, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
