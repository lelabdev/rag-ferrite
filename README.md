<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">

# rag-ferrite

**A lightweight, self-hosted document memory for AI assistants.**

Give Claude Code, Hermes, Claude Desktop, and other MCP clients fast access to your notes, documentation, books, transcripts, and research.

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)
[![CI](https://github.com/lelabdev/rag-ferrite/actions/workflows/ci.yml/badge.svg)](https://github.com/lelabdev/rag-ferrite/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit)

**Hybrid retrieval · SQLite storage · Native MCP · Single Rust binary**

</div>

---

## Overview

`rag-ferrite` indexes your files and exposes them to the assistants you already use.

```text
Markdown · PDF · text · transcripts · documentation
                         │
                         ▼
                    rag-ferrite
          FTS5 + sqlite-vec + rank fusion
                         │
             MCP · REST · CLI · TUI
                         │
                         ▼
       Claude Code · Hermes · Claude Desktop · agents
```

It is deliberately not a chatbot, note editor, hosted AI platform, or distributed vector database. Your files remain the source of truth; SQLite is a derived search index that can be rebuilt.

### Why use it?

- Search by exact wording and semantic meaning.
- Share one knowledge base between several assistants.
- Ingest raw reference material before manually organizing it.
- Keep the deployment small: one binary and one SQLite database.
- Use local Ollama models or hosted OpenAI-compatible providers.
- Access the same business logic through MCP, REST, CLI, and the terminal UI.

---

## Features

- **MCP-native:** stdio and Streamable HTTP transports.
- **Hybrid retrieval:** SQLite FTS5 keyword search plus sqlite-vec semantic search.
- **Reciprocal rank fusion:** combines lexical and vector rankings.
- **Adaptive query pipeline:** query classification, expansion, quality gate, and corrective retry.
- **Optional reranking:** disabled, LLM, or Cohere modes.
- **Parent-child chunking:** precise child matches with broader parent context.
- **Contextual ingestion:** optional LLM-generated context, relevance scoring, and automatic tags.
- **Atomic tags and AND filtering:** narrow retrieval with one or more topic tags.
- **Bounded ingestion queue:** asynchronous jobs with backpressure, cancellation, progress, and history.
- **Safe resumability:** parent groups are committed transactionally and interrupted ingestion resumes missing groups.
- **Retrieval evaluation:** versioned golden datasets with Recall@k, precision@k, MRR, nDCG, empty-result rate, and latency percentiles.
- **Interactive TUI:** dashboard, library, query, ingestion, and administration workspaces.
- **Optional web console:** library inspection and ingestion through the shared REST authentication policy, with untrusted document metadata rendered as text and without a bundled chatbot.
- **Single binary:** no Python runtime and no mandatory container stack.

---

## Requirements

- Linux for the prebuilt release; building from source may work on other Rust-supported targets.
- An embedding provider. Ollama and OpenAI-compatible APIs are supported.
- Optional LLM access for contextual retrieval, relevance scoring, query expansion, and reranking.
- Poppler's `pdftotext` command for PDF ingestion.

Install PDF support:

```bash
# Debian / Ubuntu
sudo apt install poppler-utils

# Fedora
sudo dnf install poppler-utils

# Arch Linux
sudo pacman -S poppler
```

---

## Installation

### Download the latest release

```bash
mkdir -p ~/.local/bin
curl -fL https://github.com/lelabdev/rag-ferrite/releases/latest/download/ragfer \
  -o ~/.local/bin/ragfer
chmod +x ~/.local/bin/ragfer
```

Ensure `~/.local/bin` is in your `PATH`.

### Build from source

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite
cargo build --release --bin ragfer
install -Dm755 target/release/ragfer ~/.local/bin/ragfer
```

---

## Local code graph with Graphify

The repository maintains a shared, machine-readable architecture graph with [Graphify](https://github.com/safishamsi/graphifyy). `graphify-out/graph.json` is committed so AI agents and contributors use the same graph. Human-oriented reports and visualizations remain local.

```bash
# Install Graphify, then activate the repository hook once
# (hooks are local to each clone)
git config core.hooksPath .githooks

# Refresh manually at any time
graphify update .

# Query the shared graph
graphify query "How does ingestion connect to retrieval?"
```

The `pre-commit` hook runs `graphify update .` and stages `graphify-out/graph.json` before each commit. This incrementally refreshes the code graph; rerun the full `/graphify .` workflow when documentation or images change and their semantic layer must be regenerated.

---

## Quick start

This minimal configuration uses Ollama locally for embeddings and disables optional LLM processing.

1. Pull an embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

2. Create `~/.config/rag-ferrite/config.toml`:

```toml
http_port = 4242

[embedding]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimensions = 512
base_url = "http://localhost:11434"

[llm]
context_enabled = false

[reranker]
reranker_type = "disabled"

[advanced]
http_bind_address = "127.0.0.1"
allowed_ingest_roots = ["/home/your-user/Documents", "/home/your-user/Notes"]
move_after_ingest = false
```

3. Start the server:

```bash
ragfer serve
```

4. In another terminal, configure the CLI and ingest a file:

```bash
ragfer setup
ragfer ingest-file ~/Documents/example.md
ragfer query "What does this document explain?"
```

With `http_port = 4242`, the endpoints are:

```text
MCP:  http://localhost:4242/mcp
REST: http://localhost:4242/api
```

Running `ragfer` without arguments opens the TUI monitor. To use MCP over stdio instead, set `http_port = 0` and configure the client to launch `ragfer serve`.

---

## Configuration

The server searches for configuration in this order:

1. `./config.toml`
2. `~/.config/rag-ferrite/config.toml`
3. built-in defaults

Secrets can be set in the environment or in a `.env` file beside the executable.

### Hosted provider example

```toml
data_dir = "./data"
http_port = 4242

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 512
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "openai"
model = "your-model"
base_url = "https://your-provider.example/v1"
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0

[reranker]
reranker_type = "llm"
top_k = 10

[chunking]
strategy = "auto"
parent_max_chars = 2000
child_max_chars = 200
child_overlap = 20
child_min_chars = 100
auto_threshold = 5000

[advanced]
http_bind_address = "127.0.0.1"
allowed_hosts = ["localhost", "127.0.0.1", "[::1]"]
ingestion_queue_capacity = 32
max_inline_content_bytes = 10485760
http_body_limit_bytes = 12582912
ingestion_timeout_secs = 900
allowed_ingest_roots = ["/srv/library"]
move_after_ingest = false
web_ui_enabled = false
```

Set provider credentials without committing them:

```bash
export EMBEDDING_API_KEY="..."
export LLM_API_KEY="..."
```

The embedding provider also accepts `OPENAI_API_KEY`. Named LLM profiles can use a custom `api_key_env` per profile; see [`llms.txt`](llms.txt) for the complete configuration reference.

### Important advanced settings

| Setting | Default | Purpose |
| --- | ---: | --- |
| `http_bind_address` | `127.0.0.1` | HTTP listener address |
| `allowed_hosts` | loopback hosts | Accepted Host headers for Streamable HTTP MCP |
| `unsafe_bind_without_auth` | `false` | Explicitly permit an unauthenticated non-loopback bind |
| `allowed_ingest_roots` | empty | Filesystem roots accepted by path-based ingestion |
| `ingestion_queue_capacity` | `32` | Maximum queued ingestion jobs |
| `max_inline_content_bytes` | 10 MiB | Maximum inline MCP/REST content |
| `http_body_limit_bytes` | 12 MiB | Maximum HTTP request body |
| `ingestion_timeout_secs` | `900` | Per-job timeout |
| `web_ui_enabled` | `false` | Serve the optional console at `/` |

Non-loopback HTTP binds are rejected unless authentication is configured or `unsafe_bind_without_auth = true` is explicitly set.

---

## Authentication

REST and Streamable HTTP MCP use the same Bearer authentication middleware.

```bash
export RAG_API_KEY="admin-secret"
export RAG_GUEST_API_KEY="read-only-secret" # optional
ragfer serve
```

Access levels:

- **Admin key (`RAG_API_KEY`):** read and write access.
- **Guest key (`RAG_GUEST_API_KEY`):** GET endpoints and `POST /api/query` only.
- **No keys:** open access, intended for loopback development. The authentication middleware remains installed, so generating an admin key immediately protects subsequent requests without restarting the server.

Generate and inspect keys through the CLI:

```bash
ragfer key generate
ragfer key list
ragfer key show
```

Remote deployments should use authentication, a restrictive `allowed_hosts` list, and a private network such as Tailscale. Host allowlisting protects against unwanted Host headers; it is not a substitute for authentication.

---

## Connect an MCP client

### Streamable HTTP

Start `ragfer serve` with `http_port > 0`, then connect to:

```text
http://localhost:4242/mcp
```

Example client configuration:

```yaml
mcp_servers:
  rag-ferrite:
    url: "http://localhost:4242/mcp"
    headers:
      Authorization: "Bearer ${RAG_API_KEY}"
    timeout: 9999
```

### stdio

Set `http_port = 0` or omit it:

```json
{
  "mcpServers": {
    "rag-ferrite": {
      "command": "/home/user/.local/bin/ragfer",
      "args": ["serve"],
      "env": {
        "EMBEDDING_API_KEY": "...",
        "LLM_API_KEY": "..."
      }
    }
  }
}
```

The exact configuration shape depends on the MCP client.

---

## MCP tools

### Retrieval and inspection

| Tool | Purpose |
| --- | --- |
| `query_documents` | Hybrid retrieval with source, metadata, and AND-tag filters |
| `read_chunk_neighbors` | Read surrounding chunks for a result |
| `list_files` | List indexed sources |
| `status` | Show engine status |
| `suggest_collection` | Suggest a collection from query keywords |
| `tag_map` | Return tags, collections, and chunk counts |
| `collection_heat` | Show collection query activity |
| `chunk_qa` | Find cold or unused chunks |

### Ingestion and administration

| Tool | Purpose |
| --- | --- |
| `ingest_file` | Queue a local PDF, TXT, or Markdown file |
| `ingest_data` | Queue inline text, HTML, or Markdown |
| `check_ingestion` | Inspect quality and duplication before ingestion |
| `delete_file` | Delete a source and its chunks |
| `reassign_collection` | Move a source to another collection |
| `rebuild_indexes` | Rebuild derived indexes |
| `flush_indexes` | Refresh derived indexes and checkpoint SQLite |
| `benchmark` | Evaluate retrieval against a golden dataset |

Path-based tools are restricted by `advanced.allowed_ingest_roots`. Inline ingestion is preferable when the MCP server cannot access a client's local filesystem.

---

## CLI

```text
ragfer                         Open the terminal UI
ragfer serve                   Start the server
ragfer status                  Show engine status
ragfer progress                Show active ingestion progress
ragfer query "text"            Search documents
ragfer list                    List indexed documents
ragfer ingest-file <path>      Ingest one file
ragfer ingest-batch <paths...> Ingest several files as one batch
ragfer ingest-data <name>      Ingest standard input
ragfer delete <source_id>      Delete a document
ragfer history                 Show recent ingestion batches
ragfer cancel                  Cancel the active batch
ragfer flush                   Refresh derived indexes and checkpoint SQLite
ragfer rebuild                 Rebuild all indexes
ragfer reload                  Reload supported configuration
ragfer stop                    Stop the server
ragfer restart                 Stop and wait for service restart
ragfer key generate|list|show  Manage API keys
ragfer setup                   Configure the HTTP client
ragfer update                  Run update.sh beside the binary
```

Common options:

| Option | Purpose |
| --- | --- |
| `--json` | Print raw JSON |
| `-c <collection>` | Select or filter a collection |
| `-n <limit>` | Set query result count |
| `-t <tag1,tag2>` | Apply AND-tag filtering |
| `--force` | Replace an existing source during file ingestion |

Examples:

```bash
ragfer ingest-batch book.pdf notes.md transcript.txt
cat article.md | ragfer ingest-data "article-name"
ragfer query "SQLite concurrency" -n 5
ragfer query "MCP authentication" -t security,mcp --json
```

The CLI client stores its server URL and key separately from the server configuration:

```text
~/.config/ragfer/config.toml  # url = "http://localhost:4242"
~/.config/ragfer/.env         # RAG_API_KEY=...
```

Run `ragfer setup` to create them interactively.

---

## Terminal UI

Launch the built-in monitor with `ragfer` or `ragfer monitor`.

| Key | Action |
| --- | --- |
| `1` | Dashboard |
| `2` | Library |
| `3` | Query workspace |
| `4` | Ingestion workspace |
| `5` | Administration |
| `j` / `k` | Select a document in Library |
| `d` | Delete the selected document after confirmation |
| `Q` | Enter a query |
| `i` | Submit a file for ingestion |
| `e` | Edit configuration from Admin and request reload |
| `?` | Show help |
| `q` | Exit |

The dashboard shows server state, active jobs, file and chunk progress, speed, ETA, errors, and recent activity.

---

## Optional web console

Set:

```toml
[advanced]
web_ui_enabled = true
```

The server then exposes a small console at `/`. It uses the same REST API and authentication policy for:

- inline text and Markdown ingestion;
- ingestion progress;
- document listing and confirmed deletion;
- retrieval inspection;
- tags and source relationships.

The console renders document metadata through DOM text nodes; indexed source names and collection IDs are not interpreted as HTML. It is intentionally not a chatbot, model selector, or source editor. See [`docs/web-interface-plan.md`](docs/web-interface-plan.md) for its scope.

---

## REST API

All protected requests use:

```http
Authorization: Bearer <key>
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Engine status |
| `POST` | `/api/query` | Hybrid retrieval |
| `GET` | `/api/documents` | List sources |
| `GET` | `/api/documents/{id}` | Get one source |
| `DELETE` | `/api/documents/{id}` | Delete one source |
| `GET` | `/api/documents/{id}/chunks/{index}/neighbors` | Read surrounding chunks |
| `POST` | `/api/ingest` | Queue one or several filesystem paths |
| `POST` | `/api/ingest/data` | Queue inline content |
| `GET` | `/api/ingest/progress` | Active batch progress |
| `GET` | `/api/history` | Recent batches |
| `GET` | `/api/tags` | Tag and collection map |
| `GET` | `/api/graph` | Source relationship data |
| `POST` | `/api/service/cancel-batch` | Cancel the active batch |
| `POST` | `/api/reload` | Reload supported configuration |
| `POST` | `/api/flush-indexes` | Refresh derived indexes |
| `POST` | `/api/rebuild-indexes` | Rebuild indexes |
| `POST` | `/api/service/stop` | Stop the server |
| `POST` | `/api/keys/generate` | Rotate the admin key |
| `GET` | `/api/keys` | List masked keys |
| `GET` | `/api/keys/current` | Show the current admin key |

Example query:

```bash
curl http://localhost:4242/api/query \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does hybrid retrieval work?","limit":5,"tags":["rag"]}'
```

Example inline ingestion:

```bash
curl http://localhost:4242/api/ingest/data \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source":"manual-note","content":"Text to index"}'
```

Errors include an `error_code` and use meaningful HTTP statuses, including `400` for invalid input, `401/403` for authentication and authorization, `404` for missing resources, `409` for conflicts, `413` for oversized content, `429` for queue backpressure, and `500` for internal failures.

---

## Ingestion model

```text
source
  → extraction
  → pre-ingestion checks and deduplication
  → recursive or parent-child chunking
  → optional relevance scoring and contextualization
  → automatic tags
  → batched embeddings
  → transactional SQLite commit
  → FTS5 and sqlite-vec indexes
```

Supported file extensions are `.pdf`, `.txt`, and `.md`. PDF extraction uses `pdftotext`. HTML and other textual formats can be submitted as inline content.

REST, MCP, CLI, and TUI submissions all use the same bounded worker queue. A full queue returns backpressure instead of retaining unbounded payloads in memory. Each parent and its children are committed atomically; stable logical parent identifiers let an interrupted ingestion resume without duplicating completed groups.

Automatic movement after successful ingestion is enabled by default. Files under an `inbox` directory retain their nested layout under its sibling `ingested` directory. Existing destinations are never overwritten. For a permanent library, set:

```toml
[advanced]
move_after_ingest = false
```

---

## Retrieval model

```text
query
  → simple / standard / complex classification
  → optional query expansion
  → FTS5 candidates + sqlite-vec candidates
  → reciprocal rank fusion
  → optional LLM or Cohere reranking
  → confidence gate
  → optional corrective reformulation and retry
  → ranked chunks with source metadata and tags
```

Keyword search preserves exact identifiers, commands, names, and errors. Vector search finds related concepts and paraphrases. Filters are applied before candidate limits, with adaptive over-fetching where necessary, so selective filters do not silently lose matches outside an initial candidate window.

Tags are atomic and combined with AND logic:

```text
["security"]        → broad security matches
["security", "mcp"] → chunks tagged with both terms
```

Query cache keys include source, metadata, tag, and chunk filters; mutations invalidate cached retrieval results.

---

## Retrieval benchmarks

The `benchmark` MCP tool accepts a legacy array or a versioned dataset:

```json
{
  "version": 1,
  "entries": [
    {
      "question": "How does hybrid retrieval work?",
      "expected_keywords": ["BM25", "vector", "RRF"],
      "relevant_source_ids": [7, 19]
    }
  ]
}
```

Reported metrics include Recall@k, precision@k, MRR, nDCG, empty-result rate, p50/p95 latency, and per-query details. The benchmark evaluates retrieval, not final assistant answer faithfulness.

See:

- [`examples/benchmark-golden.json`](examples/benchmark-golden.json)
- [`docs/semantic-chunking-evaluation.md`](docs/semantic-chunking-evaluation.md)
- [`docs/llm-cache-evaluation.md`](docs/llm-cache-evaluation.md)

---

## Storage and architecture

```text
┌────────────────────────────────────────────┐
│ MCP stdio / MCP HTTP / REST / CLI / TUI   │
├────────────────────────────────────────────┤
│ Shared service and ingestion queue         │
├────────────────────────────────────────────┤
│ Query pipeline / chunking / LLM / reranker │
├────────────────────────────────────────────┤
│ SQLite metadata + FTS5 + sqlite-vec         │
└────────────────────────────────────────────┘
```

SQLite runs in WAL mode with a serialized shared connection. Blocking database work is dispatched through Tokio's blocking pool. There is no external vector database and no separate worker service.

The database and indexes live under `data_dir`. Back up the service while it is stopped or use an SQLite-safe backup procedure; do not treat copied WAL files as a consistent backup.

Architecture decisions are indexed in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Development

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo audit
```

CI runs formatting, linting, all tests, and the security audit on pull requests and `main`. Dependabot checks Cargo dependencies weekly.

---

## Scope

`rag-ferrite` is designed for personal and trusted-team knowledge bases. It prioritizes retrieval quality, operational simplicity, local storage, provider independence, and MCP compatibility over multi-tenant enterprise features.

A normal folder and keyword search may be enough for a small collection. `rag-ferrite` becomes useful when the corpus grows, the wording of a question differs from the source, several assistants need shared retrieval, or relevant context is spread across many documents.

---

## License

MIT
