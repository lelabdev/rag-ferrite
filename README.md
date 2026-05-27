<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">
</div>

# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

**Your documents, searchable with meaning — not just keywords.**

Single binary (15 MB). Single database. Multi-collection. MCP-native. Built in Rust.

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/lelabdev/rag-ferrite/main/install.sh | bash
```

Or build from source:

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite && cargo build --release
```

**Prerequisites:** `poppler-utils` for PDF support (`apt install poppler-utils`).

### Configuration

You need **two API keys** — one for the LLM, one for embeddings. Any OpenAI-compatible provider works.

Set them as environment variables or in a `.env` file:

```bash
export LLM_API_KEY="your-llm-api-key"
export EMBEDDING_API_KEY="your-embedding-api-key"
```

Then edit `~/.config/rag-ferrite/config.toml`:

```toml
data_dir = "./data"

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"         # or text-embedding-3-small, nomic-embed-text, etc.
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"  # or https://api.openai.com/v1, http://localhost:11434

[llm]
provider = "ollama"                          # or "openai", "openrouter"
model = "gemma4:31b"                         # or gpt-4o, llama3, etc.
base_url = "https://api.ollama.com"          # or https://api.openai.com/v1, http://localhost:11434
```

Run:

```bash
rag-ferrite
# → MCP server on stdin, ready for any MCP client
```

Usage:

- `ingest_file("/path/to/document.pdf", collection: "my-docs")`
- `query_documents("what did I write about?", collection: "my-docs")`

---

## Why rag-ferrite?

Most search tools match keywords. That works for 10 notes. It falls apart with 500 PDFs, books, and technical docs. Results are noisy — tables of contents, index pages, legal notices all pollute your search.

rag-ferrite understands what your documents **mean**. It filters the noise, tags everything automatically, and keeps searching until it finds the right answer.

| Feature | How |
|---|---|
| **Semantic search** | Finds relevant passages even without exact keyword matches |
| **Noise filtering** | Automatically removes junk chunks (TOC, boilerplate) at ingestion |
| **Auto-tagging** | Each chunk gets smart tags for cross-collection filtering |
| **Self-correcting** | Weak results trigger automatic reformulation and retry |
| **Hybrid search** | BM25 + vector search combined with RRF fusion |
| **15 MB binary** | No Docker, no GPU required. Cloud or local — your choice |
| **MCP-native** | Works with Hermes, Claude Desktop, or any MCP client |

---

## How it works

### Ingestion

```
Document → Extract text → Chunk (800 chars)
         → Relevance scoring (LLM filters junk)
         → Context prefix (LLM adds context to each chunk)
         → Auto-tag (LLM generates 2-3 tags per chunk)
         → Embed → Store in SQLite + HNSW + BM25
```

### Query

```
Query → Classify (simple / standard / complex)
      → [standard/complex] Expand query (LLM multi-query)
      → Hybrid search (BM25 + vector + RRF fusion)
      → [standard/complex] Rerank (LLM scores top results)
      → Quality gate → [weak?] Reformulate + retry
      → Return top chunks with tags
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `query_documents(query, collection?, limit?)` | Hybrid search with optional collection filter |
| `ingest_file(file_path, collection?)` | Ingest PDF, DOCX, TXT, or MD |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text, HTML, or markdown |
| `delete_file(source)` | Remove document and all its chunks |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, collection?, limit?)` | Evaluate retrieval quality against a golden dataset |

### MCP client setup

**Hermes:**

```yaml
mcp_servers:
  rag-ferrite:
    command: /path/to/rag-ferrite
    env:
      LLM_API_KEY: "..."
      EMBEDDING_API_KEY: "..."
```

**Claude Desktop:**

```json
{
  "mcpServers": {
    "rag-ferrite": {
      "command": "/path/to/rag-ferrite",
      "env": {
        "LLM_API_KEY": "...",
        "EMBEDDING_API_KEY": "..."
      }
    }
  }
}
```

---

## Advanced configuration

Everything below has sensible defaults. Only change these to fine-tune for your models or hardware.

### Full config reference

```toml
data_dir = "./data"

[embedding]
provider = "openai"                          # openai-compatible API
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"                          # ollama, openai, or openrouter
model = "gemma4:31b"
base_url = "https://api.ollama.com"
context_enabled = true                       # add context prefix to each chunk
relevance_scoring = true                     # LLM filters junk at ingestion
min_relevance_score = 5.0                    # chunks below 5/10 are discarded
temperature = 0.3                            # scoring/tagging consistency
max_tokens = 150                             # per LLM call
expansion_temperature = 0.7                  # query expansion creativity
expansion_max_tokens = 200                   # per expansion call
max_expansion_queries = 4                    # alternative queries per original
max_document_prompt_chars = 8000             # context window for prompts
max_chunk_prompt_chars = 2000
context_batch_size = 20                      # parallel chunks during ingestion
max_concurrent = 3                           # concurrent LLM calls

[reranker]
reranker_type = "llm"                        # disabled, llm, or cohere
top_k = 10
preview_chars = 300
# Cohere-specific (only when reranker_type = "cohere"):
# model = "rerank-v3.5"
# base_url = "https://api.cohere.ai/v2/rerank"
# api_key = "your-cohere-api-key"

[advanced]
chunk_size = 800                             # characters per chunk
chunk_overlap_ratio = 0.1                    # 10% overlap between chunks
merge_last_chunk_threshold = 200             # merge tiny last chunk
quality_threshold = 0.3                      # minimum confidence to accept results
max_retries = 1                              # corrective RAG retries
high_confidence_threshold = 0.7              # above = high confidence
query_limit = 10                             # default result count
db_pool_size = 4                             # SQLite connection pool
db_busy_timeout_ms = 5000
embedding_batch_size = 20                    # embeddings per API call
log_file = "rag-ferrite.log"
log_filter = "rag_ferrite=debug,rag_engine=debug"
http_bind_address = "0.0.0.0"               # for REST API (if http_port > 0)
```

---

## Acknowledgements

The ingestion pipeline was inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — contextual retrieval, pre-ingestion quality checks, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
