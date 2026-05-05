# rag-lab

Custom RAG engine in Rust — built on `rag_engine` + `rmcp` + `axum`.

## What it does

Self-hosted RAG pipeline with hybrid search (BM25 + vector), semantic chunking, contextual retrieval, cross-encoder reranking, and MCP server support. One binary, multiple isolated RAG instances.

## Why SQLite

SQLite + HNSW handles the vast majority of RAG use cases without the complexity of a dedicated vector database. One file per RAG, backup = copy, query metadata with plain SQL.

**Examples of what fits comfortably in SQLite:**

| Use case | Docs | Chunks | Works? |
|---|---|---|---|
| Personal knowledge base | ~500 | ~50k | ✅ Trivial |
| Dev team docs & books (our lab) | ~150 | ~97k | ✅ Easy |
| Medical practice (1 doctor, 1500 patients) | ~20k | ~80k | ✅ Easy |
| SME internal wiki | ~10k | ~400k | ✅ Fine |
| Multi-practice clinic (20 doctors) | ~600k | ~2.5M | ✅ Still good |
| Full enterprise search (500+ employees) | ~2M+ | ~10M+ | ⚠️ Consider PostgreSQL + pgvector |
| Web-scale search (Wikipedia FR) | ~2.4M articles | ~50M+ | ❌ Need Qdrant/Milvus |

**Rule of thumb:** Under ~1M chunks (~200k documents), SQLite is the right choice. It's simpler, portable, zero-dependency, and lets you filter with plain SQL.

**When to move to something bigger:**
- **PostgreSQL + pgvector** — distributed teams sharing one DB, 1M-10M chunks
- **Qdrant / Milvus** — web-scale, 10M+ chunks, distributed search
- **LanceDB** — heavy multimodal workloads (images, video frames)

## Stack

| Component | Choice | Purpose |
|---|---|---|
| RAG core | `rag_engine` 0.8.1 | HNSW, BM25, hybrid RRF, SQLite, chunking, doc parsing |
| MCP server | `rmcp` | stdio + SSE MCP protocol |
| Embeddings | OpenAI / Cohere / Ollama | Pluggable provider |
| Storage | SQLite + HNSW | One file per RAG instance |
| HTTP bridge | `axum` | SSE endpoint (replaces dojo) |

## Architecture

```
rag-lab (single binary, N instances)
├── config.toml          ← Which RAG, port, embedding provider
├── data/
│   ├── rag-code.sqlite3
│   ├── rag-business.sqlite3
│   ├── rag-general.sqlite3
│   └── rag-rpg.sqlite3
└── indexes/             ← Persisted HNSW indexes
```

Each instance runs isolated — same binary, different data.

## Roadmap

See [Issues](../../issues) for the prioritized backlog.

## License

MIT
