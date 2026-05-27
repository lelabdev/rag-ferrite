<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">
</div>

# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

**Your documents, searchable with meaning — not just keywords.**

Single binary, single database, multi-collection. MCP-native. Built in Rust.

---

## Why rag-ferrite?

Most note-taking tools (Obsidian, Notion) search by keywords. That works for 10 notes. It falls apart with 500 PDFs, books, and technical docs. Results are noisy — tables of contents, index pages, legal notices all pollute your search.

rag-ferrite understands what your documents **mean**. It filters the noise, tags everything automatically, and keeps searching until it finds the right answer.

**What makes it different:**

- 🎯 **Understands meaning, not just words** — semantic search finds relevant passages even without exact keyword matches
- 🧹 **Filters the noise** — automatically removes junk chunks (TOC, index pages, boilerplate) at ingestion
- 🏷️ **Tags everything automatically** — each chunk gets smart tags for cross-collection filtering
- 🔄 **Self-correcting** — if results are weak, it reformulates and retries automatically
- ⚡ **Fast** — hybrid search combines keyword + semantic for the best of both worlds
- 📦 **Single binary** — no Docker, no GPU, no cloud dependency. Runs on a $5 VPS
- 🔌 **MCP-native** — works with Hermes, Claude Desktop, or any MCP client out of the box

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
# Edit config.toml — set your LLM and embedding providers
./rag-ferrite
# → MCP server on stdin, ready for any MCP client
```

**Prerequisites:** `poppler-utils` for PDF support (`apt install poppler-utils`), and API keys for your providers.

## Configuration

You need two things: an **LLM provider** (for understanding, scoring, tagging) and an **embedding provider** (for vector search). Any OpenAI-compatible API works.

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"  # or your local Ollama, etc.

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

Set your API keys: `export LLM_API_KEY=... EMBEDDING_API_KEY=...`

Then ingest and search:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

**Local alternative:** any local model via Ollama works for embeddings. Just point `[embedding]` to your local instance.

## How it works

### Ingestion pipeline

```
Document → Extract text → Chunk (800 chars)
         → Relevance scoring (LLM filters junk)
         → Context prefix (LLM adds context to each chunk)
         → Auto-tag (LLM generates 2-3 tags per chunk)
         → Embed → Store in SQLite + HNSW + BM25
```

### Query pipeline

```
Query → Classify (simple / standard / complex)
      → [standard/complex] Expand query (LLM multi-query)
      → Hybrid search (BM25 + vector + RRF fusion)
      → [standard/complex] Rerank (LLM scores top results)
      → Quality gate → [weak?] Reformulate + retry
      → Return top chunks with tags
```

### MCP Tools

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

For advanced configuration details, see [docs/advanced.md](docs/advanced.md).

## Acknowledgements

The ingestion pipeline was heavily inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — specifically contextual retrieval, pre-ingestion quality checks, post-chunking verification, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
