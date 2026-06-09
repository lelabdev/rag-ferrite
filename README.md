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
# → produces target/release/ragfer
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
dimensions = 512
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
```

Run:

```bash
ragfer serve
# → MCP on stdio + HTTP API on http://0.0.0.0:4242
```

Just `ragfer` without args launches the TUI monitor (see [CLI](#cli)).

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

### MCP Tools (16)

| Tool | Description |
|---|---|
| `query_documents(query, collection?, limit?)` | Hybrid search with filters, reranking, expansion, cache |
| `ingest_file(file_path, collection?)` | Ingest PDF, DOCX, TXT, or MD |
| `ingest_data(content, source, collection?, format?)` | Ingest raw text, HTML, or markdown |
| `delete_file(source)` | Remove document and all its chunks (instant, no synchronous index rebuild) |
| `list_files()` | List indexed documents |
| `status()` | Engine status and document count |
| `read_chunk_neighbors(source_id, chunk_index)` | Expand context around a chunk |
| `check_ingestion(file_path?, content?, source_name?)` | Preview document quality before ingestion |
| `benchmark(file_path, collection?, limit?)` | Evaluate retrieval quality against a golden dataset |
| `collection_heat()` | Collection heat tracking: heat_score, last_queried_at, query_count per collection |
| `chunk_qa()` | Chunk-level QA: dead/cold chunks grouped by source, heat calculated on-the-fly |
| `suggest_collection(query)` | Tag routing: extract keywords, match against collection_tags, suggest best collection |
| `tag_map()` | Full tag → collection mapping with chunk counts |
| `reassign_collection(source_id, collection)` | Move a source and its chunks to a different collection, rebuilds indexes |
| `rebuild_indexes()` | Rebuild HNSW + BM25 indexes + WAL checkpoint |
| `flush_indexes()` | Flush incremental HNSW buffer to disk |

### MCP client setup

**Option A — stdio** (local, simple):

```yaml
# Hermes
mcp_servers:
  rag-ferrite:
    command: /path/to/ragfer
    args: ["serve"]
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
      "command": "/path/to/ragfer",
      "args": ["serve"],
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
sudo systemctl enable --now rag-ferrite
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
- Works over the **network** — run ragfer on any server
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

### HTTP API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | Engine status, version, document count |
| `POST` | `/api/ingest` | Unified ingest — `file_path` (single) or `paths` (batch), `move_after_ingest` |
| `POST` | `/api/ingest/data` | Ingest raw text content |
| `GET` | `/api/ingest/progress` | Batch progress: files, chunks, speed, ETA, errors, per-file results, `activity_log.events[]` |
| `POST` | `/api/query` | Hybrid search with reranking |
| `GET` | `/api/documents` | List all sources |
| `GET` | `/api/documents/{id}` | Get document details |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `GET` | `/api/graph` | Source relationship graph |
| `POST` | `/api/flush-indexes` | Rebuild HNSW + BM25 + WAL checkpoint |
| `POST` | `/api/rebuild-indexes` | Full index rebuild |
| `POST` | `/api/service/cancel-batch` | Cancel running batch (stops after current file) |
| `POST` | `/api/service/stop` | Graceful server shutdown |

### CLI

The `ragfer` binary includes a built-in CLI client. All commands hit the HTTP API of a running server.

```bash
ragfer                            # Launch TUI monitor (default)
ragfer serve            (-d)      # Launch server (daemon)
ragfer status           (-s)      # Engine status
ragfer progress         (-p)      # Batch ingestion progress
ragfer query (-q) "text"         # Search documents
ragfer list             (-l)      # List documents
ragfer monitor          (-m)      # Launch TUI monitor
ragfer ingest-file <path>        # Ingest a file
ragfer ingest-batch <paths...>   # Ingest multiple files
ragfer ingest-data <name>        # Ingest from stdin
ragfer delete <source_id>        # Delete a document
ragfer flush                     # Flush HNSW indexes
ragfer rebuild                   # Rebuild indexes
ragfer cancel                    # Cancel running batch
ragfer stop                      # Stop the server
ragfer update                    # Download latest + restart
```

| Short flag | Long form | Description |
|:----------:|-----------|-------------|
| `-d` | `serve` | Start the server daemon (MCP stdio + HTTP) |
| `-s` | `status` | Show engine status and document count |
| `-l` | `list` | List indexed documents |
| `-q` | `query` | Search documents (requires query text) |
| `-p` | `progress` | Show batch ingestion progress |
| `-m` | `monitor` | Launch TUI monitor |

**Common options:**

| Option | Description |
|--------|-------------|
| `--env <env>` | Instance: `prod` (default) or `test` |
| `--json` | Raw JSON output |
| `-c <collection>` | Target collection name |
| `-n <limit>` | Result limit (default 10) |
| `-t <tags>` | Tag filter (comma-separated) |
| `--force` | Force re-ingest (delete existing first) |

> **Note:** The Cargo package name is `rag-ferrite`; the compiled binary is `ragfer`. This is set via `[[bin]] name = "ragfer"` in `Cargo.toml`.

### Real-time monitor

**Built-in TUI** (default — just run `ragfer` with no args):
```bash
ragfer                              # launches monitor (default)
ragfer monitor [refresh_seconds] [url] [--demo] [--fade N]
```

**Standalone monitor** (separate binary, connects via HTTP, no SSH needed):
```bash
rag-monitor [refresh_seconds] [url]
```

Configuration via environment:
- `RAG_MONITOR_URL` — server URL (default: `http://localhost:4242`)
- `RAG_API_KEY` or `RAG_MONITOR_KEY` — API key
- `RAG_MONITOR_REFRESH` — refresh interval in seconds

API key lookup order: env vars → `~/.config/rag/api_key_nova`

The standalone `rag-monitor` is a full TUI with a colored progress bar, real-time stats, activity log with timestamps, and always-visible file lists (same UI as the built-in `ragfer` monitor):

```
 rag-ferrite v5.1.0 • 132 docs  ⠹

 prompt-engineering.txt
  28%  70/247
 [████████████▓▓▒░⣷⣧⣇⡇⡆▁ ⠹⠸⠼⠴⠦⠧⠇⠏⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⠋⠙⠹⠸⠼⠴]
   phase: embed+llm

 Chunks  12439/12439   Size   1.7 MB   Elapsed  1h47m
 Speed   115 ch/min    Avg    92.5s    ETA      4h32m   Errors  0

─── Activity ─────────────────────────────────────────────
 14:32:07 Embedding 48 texts...
 14:32:09 Embedding done in 2340ms (48 texts)
 14:32:09 Parent 3/8: contextual retrieval for 12 children
 14:32:15 Parent 3/8: context done in 5812ms (12 ok, 0 skip, 0 fail)

┌─ Completed (70) ───────────────┬─ Queue ─────────────────┐
│ ✓ file1.txt       41 ch   34s │ 127 files pending       │
│ ✓ file2.txt      237 ch  101s │                         │
│ ✓ file3.txt      101 ch  126s │                         │
└────────────────────────────────┴─────────────────────────┘
[c]ancel [r]ebuild [f]lush [x]top [?]help [q]uit
```

**Progress bar zones:**
- `█` **Green** — completed files (with `▓▒░` cyan/blue fade near frontier)
- `⡀→⣿` **Yellow** — current file (braille 1-8 dots = per-file progress)
- `⠋⠙⠹...` **Color wave** — pending files (animated, traveling gradient)

**Activity log:** Shows the last 20 ingestion events (embedding, LLM, chunking, error, info) with timestamps, sourced from the progress API's `activity_log.events[]` ring buffer. Elapsed time, speed, and ETA are recalculated live from `started_at` on every refresh — no stale counters.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `TAB` | Switch panel (Completed ↔ Queue) |
| `↑↓` | Scroll file list |
| `c` | Cancel running batch |
| `r` | Rebuild indexes |
| `f` | Flush indexes |
| `x` | Stop server |
| `?` | Show/hide help popup |
| `q` / `Esc` | Quit |

The built-in monitor (`ragfer` / `ragfer monitor`) has additional shortcuts: `l` toggle lists, `s` toggle stats, `c` cycle color modes, `o` open file in `less`.

Modes (built-in only):
- `--demo`: simulate a batch without ingestion (for testing animations)
- `--fade N`: fade length (0 = no fade, default 5)

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

**Query classification:**

- **Simple**: direct keyword match (e.g. "what is X?")
- **Standard**: needs expansion + reranking (e.g. "compare X and Y")
- **Complex**: multi-step reasoning (e.g. "how does X relate to Y given Z?")

Classification is automatic. Keywords and thresholds are configurable via `[query_classification]` in config.toml. Keywords can also be loaded from an external dictionary file `dictionaries/query_classification.toml` (optional — falls back to hardcoded defaults if absent).

```toml
[query_classification]
question_markers = ["what", "how", "why", "comment", "pourquoi", ...]
boolean_operators = ["AND", "OR", "et", "ou"]
complex_word_threshold = 8   # >8 words → complex
simple_word_threshold = 2    # ≤2 words → simple
```

> **Note:** Ingestion with contextual retrieval enabled can take 5–15 minutes per document. The monitor (`ragfer`) shows real-time progress. If your MCP client has a request timeout, set it high (e.g. 9999 seconds).

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
dimensions = 512
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

[query_classification]
# Keywords can also be loaded from dictionaries/query_classification.toml
question_markers = ["what", "how", "why", "comment", "pourquoi"]
boolean_operators = ["AND", "OR", "et", "ou"]
complex_word_threshold = 8
simple_word_threshold = 2

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
