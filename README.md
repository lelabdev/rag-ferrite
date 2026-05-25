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

The simplest setup uses **OpenRouter** for both embedding and contextual retrieval — no local GPU needed.

1. Create an [OpenRouter](https://openrouter.ai) account and get an API key
2. Set up your config:

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "openrouter"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "openrouter"
model = "qwen/qwen3-32b"
base_url = "https://openrouter.ai/api/v1"
context_enabled = true
```

3. Set your API key: `export LLM_API_KEY=sk-...`
4. Run: `./rag-ferrite`

That's it. Ingest your first document and search:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

**Local alternative:** bge-m3 via Ollama works for embeddings (free, GPU recommended). Just change the `[embedding]` block to point to your local Ollama instance.

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
