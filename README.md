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

**Option A — One-line install** (recommended):

```bash
curl -fsSL https://raw.githubusercontent.com/lelabdev/rag-ferrite/main/install.sh | bash
```

Downloads the binary, generates a default config, sets up PATH. Optionally installs a systemd user service.

**Option B — Build from source:**

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite && cargo build --release
```

### Setup

You need **two API keys** — one for the LLM (scoring, tagging, expansion) and one for embeddings (vector search). Any OpenAI-compatible provider works.

**1. Set your API keys:**

```bash
export LLM_API_KEY="your-llm-api-key"           # Ollama Cloud, OpenRouter, OpenAI, etc.
export EMBEDDING_API_KEY="your-embedding-api-key"  # OpenRouter, OpenAI, etc.
```

Or create a `.env` file next to the binary:

```
LLM_API_KEY=your-llm-api-key
EMBEDDING_API_KEY=your-embedding-api-key
```

**2. Edit the config** (`~/.config/rag-ferrite/config.toml`):

```toml
data_dir = "./data"

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"     # or text-embedding-3-small, etc.
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"  # or https://api.openai.com/v1

[llm]
provider = "ollama"                    # or "openai", "openrouter"
model = "gemma4:31b"                   # or gpt-4o, llama3, etc.
base_url = "https://api.ollama.com"    # or https://api.openai.com/v1
```

**3. Run:**

```bash
rag-ferrite
# → MCP server on stdin, ready for any MCP client
```

**Prerequisites:** `poppler-utils` for PDF support (`apt install poppler-utils`).

## Configuration

You need two things: an **LLM provider** (for understanding, scoring, tagging) and an **embedding provider** (for vector search). Any OpenAI-compatible API works.

Here's a complete config with all available options:

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

### Quick usage

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
| `ingest_file(file_path, collection?)` | Ingest PDF/TXT/MD |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text or markdown |
| `delete_file(source)` | Remove document by source identifier |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, collection?, limit?)` | Evaluate retrieval quality against a golden dataset |

---

## Advanced configuration

Everything below has sensible defaults — you only need to set these if you want to fine-tune behavior for your models or hardware.

### LLM tuning

These control how the LLM generates responses during ingestion (scoring, tagging, context) and querying (expansion, reformulation).

```toml
[llm]
# How creative the LLM is for scoring/tagging (0.0 = deterministic, 1.0 = creative)
# Lower = more consistent, higher = more varied
temperature = 0.3

# Max tokens the LLM can generate per call for scoring/tagging
# 150 is enough for scores and tags. Increase if your model needs more room.
max_tokens = 150

# Temperature for query expansion and reformulation
# Higher than scoring because you want diverse rephrasings
expansion_temperature = 0.7

# Max tokens for expansion/reformulation calls
expansion_max_tokens = 200

# Max alternative queries generated per original query
# More = broader search, but costs more tokens
max_expansion_queries = 4

# How many characters of the full document to include in context prompts
# Higher = better context, but uses more tokens per chunk
max_document_prompt_chars = 8000

# How many characters of each chunk to include in prompts
max_chunk_prompt_chars = 2000

# Number of chunks processed in parallel during ingestion
# Lower = fewer API calls at once (good for rate-limited providers)
context_batch_size = 20
```

### Reranker

```toml
[reranker]
# "disabled" = no reranking, "llm" = uses your configured LLM, "cohere" = Cohere Rerank API
reranker_type = "llm"
top_k = 10
preview_chars = 300

# Cohere-specific (only used when reranker_type = "cohere")
# model = "rerank-v3.5"              # defaults to rerank-v3.5
# base_url = "https://api.cohere.ai/v2/rerank"  # defaults to Cohere API
# api_key = "your-cohere-api-key"
```

### Chunking & ingestion

These control how documents are split into searchable chunks.

```toml
[advanced]
# Chunk size in characters
# 800 is a good balance for most documents. Larger = more context per chunk,
# smaller = more precise matching but less context.
chunk_size = 800

# How much consecutive chunks overlap (0.0–1.0)
# 0.1 = 10% overlap. Prevents information split across chunk boundaries.
chunk_overlap_ratio = 0.1

# If the last chunk is smaller than this, merge it with the previous one
# Avoids tiny orphan chunks at the end of documents
merge_last_chunk_threshold = 200
```

### Query pipeline

These control the search quality and retry behavior.

```toml
[advanced]
# Minimum quality score to accept search results (0.0–1.0)
# Below this threshold, the pipeline reformulates and retries.
# Lower = accept weaker results, higher = stricter quality.
quality_threshold = 0.3

# How many times to retry with reformulated queries
max_retries = 1

# Score above which a chunk is considered high-confidence during reranking (0.0–1.0)
high_confidence_threshold = 0.7
```

### Database & performance

```toml
[advanced]
# SQLite connection pool size
# Increase if you see "database locked" errors under heavy concurrent load
db_pool_size = 4

# How long to wait (ms) if the database is busy before giving up
db_busy_timeout_ms = 5000

# Number of embeddings sent per batch to the embedding API
# Lower = gentler on rate limits, higher = faster ingestion
embedding_batch_size = 20
```

### Logging & HTTP

```toml
[advanced]
# Log file path (relative to working directory)
log_file = "rag-ferrite.log"

# Log filter (Rust tracing syntax)
# Increase to "rag_ferrite=trace" for verbose debugging
log_filter = "rag_ferrite=debug,rag_engine=debug"

# HTTP API bind address (for the REST API, if http_port > 0)
http_bind_address = "0.0.0.0"
```

## Acknowledgements

The ingestion pipeline was heavily inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — specifically contextual retrieval, pre-ingestion quality checks, post-chunking verification, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
