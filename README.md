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

Prerequisites: `poppler-utils` for PDF extraction (`apt install poppler-utils`), and accounts for your embedding & LLM providers.

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

For embeddings, we recommend **Qwen3 Embedding 8B via OpenRouter** (4096 dims) — excellent multilingual support, cheap, and no local GPU required. Alternatively, bge-m3 via Ollama works locally for free.

For the LLM used in contextual retrieval, **Qwen models via OpenRouter** — cheap, fast, great multilingual support. Something like `qwen/qwen3-32b` works well.

```toml
[embedding]
provider = "openrouter"
model = "qwen/qwen3-embedding-8b"
dimensions = 4096
base_url = "https://openrouter.ai/api/v1"
# api_key loaded from EMBEDDING_API_KEY env var

[llm]
provider = "openrouter"
model = "qwen/qwen3-32b"
base_url = "https://openrouter.ai/api/v1"
context_enabled = true
max_concurrent = 3
# api_key loaded from LLM_API_KEY env var
```

For advanced options (reranking, metadata extraction, golden dataset benchmarking, relevance scoring), see [docs/advanced.md](docs/advanced.md).

## Acknowledgements

The ingestion pipeline was heavily inspired by [Jonas Roman's video on production RAG workflows](https://www.youtube.com/watch?v=phZ_iqu1gN0) — specifically contextual retrieval, pre-ingestion quality checks, post-chunking verification, query expansion, LLM reranking, and golden dataset benchmarking.

## License

MIT
