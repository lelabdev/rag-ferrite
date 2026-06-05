<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">
</div>

# rag-ferrite

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)

**Your documents, searchable with meaning — not just keywords.**

Single binary (15 MB). Single database. Tag-based classification. MCP-native. Built in Rust.

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

```bash
export LLM_API_KEY="your-llm-api-key"
export EMBEDDING_API_KEY="your-embedding-api-key"
```

Minimal `~/.config/rag-ferrite/config.toml`:

```toml
data_dir = "./data"
http_port = 4242                     # 0 = stdio-only, >0 = also serve HTTP

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
```

Run:

```bash
rag-ferrite
# → MCP on stdio + HTTP API on http://0.0.0.0:4242
```

That's it. See [Configuration](#configuration) for LLM profiles, tag rules, and advanced tuning.

---

## Features

| Feature | How |
|---|---|
| **Semantic search** | Finds relevant passages even without exact keyword matches |
| **Noise filtering** | Automatically removes junk chunks (TOC, boilerplate) at ingestion |
| **Auto-tagging** | Each chunk gets smart tags for filtering and classification |
| **Hybrid chunking** | Parent-child chunking for long docs — precise matching + full context |
| **Self-correcting** | Weak results trigger automatic reformulation and retry |
| **Hybrid search** | BM25 + vector search combined with RRF fusion |
| **Batch ingestion** | HTTP API for multi-file ingestion with real-time progress monitoring |
| **15 MB binary** | No Docker, no GPU required. Cloud or local — your choice |
| **MCP-native** | Works with Hermes, Claude Desktop, or any MCP client |

---

## Usage

### MCP Tools

| Tool | Description |
|---|---|
| `query_documents(query, limit?)` | Hybrid search with optional limit |
| `ingest_file(file_path)` | Ingest PDF, DOCX, TXT, or MD |
| `ingest_data(content, source, format?)` | Ingest raw text, HTML, or markdown |
| `delete_file(source)` | Remove document and all its chunks |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, limit?)` | Evaluate retrieval quality against a golden dataset |

### MCP client setup

**Option A — stdio** (local, simple):

```yaml
# Hermes
mcp_servers:
  rag-ferrite:
    command: /path/to/rag-ferrite
    timeout: 9999
    env:
      LLM_API_KEY: "..."
      EMBEDDING_API_KEY: "..."
```

```json
// Claude Desktop
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

**Option B — Streamable HTTP** (recommended for production):

Set `http_port = 4242` in `config.toml`, then run as a systemd service:

```bash
sudo cp rag-ferrite.service /etc/systemd/system/
sudo systemctl enable rag-ferrite
```

Connect from any MCP client:

```yaml
# Hermes — local or remote
mcp_servers:
  rag-ferrite:
    url: "http://localhost:4242/mcp"    # or http://100.x.x.x:4242/mcp
    timeout: 9999
```

**Why Streamable HTTP?**
- Runs as an **independent service** — survives client restarts
- Works over the **network** — run rag-ferrite on any server
- **Multiple clients** can connect simultaneously
- Long ingestions are **decoupled** from client lifecycle

### Batch ingestion

```bash
# Single file
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.txt"}'

# Multiple files (batch)
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["file1.txt", "file2.txt"]}'

# Disable auto-move for this batch only
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["file1.txt"], "move_after_ingest": false}'
```

**Auto-move after ingestion:** By default, files are moved from `inbox/` to `ingested/` after successful ingestion. This prevents accidental re-ingestion of the same files.

- `inbox/@channel/video.txt` → `ingested/@channel/video.txt`
- Configurable via `[advanced]` section in `config.toml`:

```toml
[advanced]
move_after_ingest = true    # default: true
ingested_dir = "ingested"   # default: "ingested"
```

- Override per-request with `"move_after_ingest": false` in the API call.

The batch runs in the background — the API returns immediately with a `batch_id`.

### Real-time monitor

```bash
# Default (2s refresh, localhost)
rag-ferrite monitor

# Custom refresh + remote server
rag-ferrite monitor 1 http://100.x.x.x:4242
```

The monitor is a full TUI with a colored progress bar, real-time stats, file lists, and keyboard controls:

```
  ⠹ RUNNING — batch 82446255                    28%
  [████████████▓▓▒░⣷⣧⣇⡇⡆▁ ⠹⠸⠼⠴⠦⠧⠇⠏⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⠋⠙⠹⠸⠼⠴]
  70/247 files

  Chunks    12439 / 12439     Size           1.7 MB
  Speed      115 chunks/min      Avg/file     92.5s
  Elapsed   1h47m          ETA          4h32m
  Errors        0 (0.0%)

┌─ Completed (70) ──────────────┬─ Current ──────────────┐
│ ✓ file1.txt        41 ch  34s│ ▶ prompt-engineering...│
│ ✓ file2.txt       237 ch 101s│   phase: embed+llm     │
│ ✓ file3.txt       101 ch 126s├─ Queue ────────────────┤
│ ↑↓ scroll • TAB switch       │ 127 files pending      │
└──────────────────────────────┴────────────────────────┘
TAB switch • ↑↓ scroll • l list • c color • s stats • o open • ? help • q quit
```

**Progress bar zones:**
- `█` **Green** — completed files (with `▓▒░` cyan/blue fade near frontier)
- `⡀→⣿` **Yellow** — current file (braille 1-8 dots = per-file progress)
- `⠋⠙⠹...` **Color wave** — pending files (animated, traveling gradient)

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `TAB` | Switch panel |
| `↑↓` | Scroll file list |
| `l` | Toggle lists (full-screen progress bar) |
| `c` | Cycle color modes (Full → Stats only → Mono) |
| `s` | Toggle stats |
| `o` | Open selected file in `less` |
| `?` | Show/hide help popup |
| `q` / `Esc` | Quit |

---

## How it works

### Ingestion pipeline

```
Document → Extract text → Chunk (auto/recursive/parent-child)
         → Relevance scoring (LLM filters junk)
         → Context prefix (LLM adds context to each chunk)
         → Auto-tag (LLM generates 2-3 tags per chunk)
         → Embed → Store in SQLite + HNSW + BM25
```

**Chunking strategies** (configurable via `[chunking]`):

| Strategy | How | Best for |
|---|---|---|
| `recursive` | Fixed-size chunks (~800 chars) with overlap | Short docs, notes, FAQ |
| `parent_child` | Large parents (~2000 chars) → small children (~200 chars). Children are embedded for search, parents returned for context | Books, manuals, long-form docs |
| `auto` (default) | Uses parent_child for docs ≥ 5000 chars, recursive for smaller ones | Mixed document sizes — best of both |

### Query pipeline

```
Query → Classify (simple / standard / complex)
      → [standard/complex] Expand query (LLM multi-query)
      → Hybrid search (BM25 + vector + RRF fusion)
      → [standard/complex] Rerank (LLM scores top results)
      → Quality gate → [weak?] Reformulate + retry
      → Return top chunks with tags
```

> **Note:** Ingestion with contextual retrieval enabled can take 5–15 minutes per document. The monitor (`rag-ferrite monitor`) shows real-time progress. If your MCP client has a request timeout, set it high (e.g. 9999 seconds).

---

## Configuration

Everything below has sensible defaults. Only change these to fine-tune for your models or hardware.

### Modular LLM profiles

Assign different models to different actions — ingestion, queries, and reranking can each use their own provider.

```toml
[[llm_profile]]
name = "fast"
provider = "ollama"
model = "ministral-3:3b"
base_url = "https://api.ollama.com"

[[llm_profile]]
name = "smart"
provider = "openai_compatible"
model = "glm-5.1"
base_url = "https://api.z.ai/api/coding/paas/v4"
api_key_env = "GLM_API_KEY"

[llm]
ingestion_profile = "fast"       # contextualisation during ingestion
query_profile = "smart"           # query expansion + reformulation
reranker_profile = "fast"         # reranking search results
context_enabled = true
```

| Action | Priority | Recommended models |
|--------|----------|--------------------|
| **Ingestion** | Speed + cost (thousands of calls) | ministral-3:3b, gemma-3-4b |
| **Query** | Quality (one call per query) | gemma4:31b, glm-5.1, gpt-4o-mini |
| **Reranker** | Speed + cost (mechanical scoring) | Same as ingestion |

Without profiles, all actions use the single `[llm]` provider (backward compatible).

### Full config reference

```toml
data_dir = "./data"
http_port = 4242                             # 0 = stdio only

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
context_enabled = true                       # add context prefix to each chunk
relevance_scoring = true                     # LLM filters junk at ingestion
min_relevance_score = 5.0                    # chunks below 5/10 are discarded
temperature = 0.3
max_tokens = 150                             # per LLM call
expansion_temperature = 0.7                  # query expansion creativity
expansion_max_tokens = 200
max_expansion_queries = 4                    # alternative queries per original
max_document_prompt_chars = 8000
max_chunk_prompt_chars = 2000
context_batch_size = 3                       # chunks per batch LLM call
max_concurrent = 3                           # concurrent LLM calls

[reranker]
reranker_type = "llm"                        # disabled, llm, or cohere
top_k = 10
preview_chars = 300

[chunking]
strategy = "auto"                            # recursive, parent_child, or auto
parent_max_chars = 2000
child_max_chars = 200
child_overlap = 20
auto_threshold = 5000                        # switch to parent_child above this

[advanced]
chunk_size = 800
chunk_overlap_ratio = 0.1
quality_threshold = 0.3                      # minimum confidence
max_retries = 1                              # corrective RAG retries
high_confidence_threshold = 0.7
query_limit = 10                             # default result count
db_pool_size = 4
db_busy_timeout_ms = 5000
embedding_batch_size = 20
log_file = "rag-ferrite.log"
log_filter = "rag_ferrite=debug,rag_engine=debug"
http_bind_address = "0.0.0.0"
defer_index_rebuild = true                   # incremental HNSW buffer (low RAM)
wal_checkpoint_interval = 50                 # WAL checkpoint frequency
```

### Tag rules (`tag-rules.toml`)

Auto-generated tags are cleaned through a configurable pipeline:

```toml
[synonyms]
"advertising" = "copywriting"
"social media" = "social media strategy"
"props" = "svelte"

[stop_words]
words = ["creative", "general", "basic", "success"]
meta = ["introduction", "conclusion", "references"]
technical = ["syntax", "configuration", "installation"]

[rules]
min_length = 3
max_words = 3
strip_chars = "*$`\"<>|={}[]/"
```

**Pipeline:** strip chars → lowercase → synonym lookup → stop word filter → length filter → singular normalization → dedup.

No recompilation needed — edit the file and restart.

---

## Performance

### Ingestion speed benchmarks

With embedding + contextual retrieval + auto-tagging + relevance scoring (3-4 LLM calls per chunk):

| Speed | Level |
|-------|-------|
| < 200 chunks/min | Slow (embedding-only pipelines) |
| 200–1,000 chunks/min | Good for enriched pipelines |
| 1,000–3,000 chunks/min | Fast |
| 3,000+ | Very fast (minimal per-chunk LLM work) |

A typical rag-ferrite setup with a small LLM (e.g. `ministral-3b`) runs at **100–150 chunks/min** = **400–600 LLM calls/min** — solid throughput for an enriched pipeline.

---

## Acknowledgements

The ingestion pipeline was inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — contextual retrieval, pre-ingestion quality checks, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
