# rag-ferrite

Self-hosted RAG engine in Rust — hybrid search, contextual retrieval, reranking, and MCP server in a single binary.

Built for teams and individuals who want serious retrieval quality without the overhead of a vector database. Uses SQLite + HNSW under the hood — zero dependencies, one file per RAG instance, backup is a copy.

## Features

**Search pipeline:**
- Hybrid search (BM25 + vector HNSW + RRF fusion)
- Contextual retrieval — LLM-generated context prefix before embedding
- Cross-encoder reranking (LLM-based or Cohere API)
- Query expansion for short/ambiguous queries
- Corrective RAG — quality gate with automatic retry on low confidence
- Adaptive query routing — Simple / Standard / Complex classification

**Infrastructure:**
- MCP server (stdio + HTTP SSE) — drop-in replacement for mcp-local-rag
- REST API (health, status, query, ingest, SSE)
- Metadata filtering via SQL
- Pluggable embedding providers (Ollama, OpenAI, Cohere)
- Pluggable LLM providers for pipeline features (Z.ai, OpenAI, Ollama)
- Semantic chunking + markdown-aware chunking
- Evaluation metrics (precision, recall, NDCG, MRR)

## Why SQLite

SQLite + HNSW handles the vast majority of RAG use cases without the complexity of a dedicated vector database.

| Use case | Docs | Chunks | Fits? |
|---|---|---|---|
| Personal knowledge base | ~500 | ~50k | ✅ |
| Dev team docs & books | ~150 | ~100k | ✅ |
| SME internal wiki | ~10k | ~400k | ✅ |
| Multi-practice clinic | ~600k | ~2.5M | ✅ |
| Enterprise search (500+ employees) | ~2M+ | ~10M+ | ⚠️ Consider pgvector |
| Web-scale (Wikipedia FR) | ~2.4M articles | ~50M+ | ❌ Need Qdrant/Milvus |

**Rule of thumb:** Under ~1M chunks, SQLite is the right call. Simpler, portable, zero-dependency, SQL filtering.

## Stack

| Component | Choice | Purpose |
|---|---|---|
| RAG core | `rag_engine` 0.8.1 | HNSW, BM25, hybrid RRF, SQLite, semantic chunking |
| MCP server | `rmcp` | stdio + SSE protocol |
| Embeddings | Ollama / OpenAI / Cohere | Pluggable provider |
| LLM | Z.ai (GLM-4.7-Flash) / OpenAI / Ollama | Context generation, query expansion, reranking |
| Storage | SQLite + HNSW | One file per RAG instance |
| HTTP bridge | `axum` | REST + SSE endpoint |

## Quick Start

```bash
# Install Ollama and pull an embedding model
ollama pull qwen3-embedding:0.6b

# Set your LLM API key (for contextual retrieval)
export ZAI_API_KEY=your-key

# Build and run
cargo build --release
./target/release/rag-ferrite
```

Create a `config.toml` in the working directory:

```toml
data_dir = "./data"
http_port = 3456    # 0 = stdio-only

[embedding]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimensions = 1024

[llm]
provider = "zai"
model = "glm-4.7-flash"
context_enabled = true
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/status` | Document count + version |
| POST | `/query` | Hybrid search with filters |
| POST | `/ingest/file` | Ingest a file (PDF, DOCX) |
| POST | `/ingest/data` | Ingest text/HTML/markdown |
| GET | `/sse` | MCP over SSE stream |

### Query example

```bash
curl -X POST http://localhost:3456/query \
  -H "Content-Type: application/json" \
  -d '{"query": "marketing strategy mistakes", "limit": 5}'
```

### Filter example

```bash
curl -X POST http://localhost:3456/query \
  -H "Content-Type: application/json" \
  -d '{"query": "marketing", "limit": 5, "metadata_like": "%.pdf"}'
```

## Architecture

```
rag-ferrite (single binary, N instances)
├── config.toml          ← Per-instance config
├── data/
│   ├── rag-code.sqlite3
│   ├── rag-business.sqlite3
│   └── rag-general.sqlite3
└── indexes/             ← Persisted HNSW indexes
```

Each instance runs isolated — same binary, different data. Routing is handled by Hermes MCP config or the HTTP API.

## Query Pipeline

```
Query → classify (Simple / Standard / Complex)
     → [Complex] Query expansion (multi-query)
     → Hybrid retrieval (BM25 + HNSW + RRF)
     → [Standard+] Cross-encoder reranking
     → [Complex] Quality gate (confidence score)
     → [Low confidence] Reformulate + retry
     → Top-k results with confidence flag
```

## License

MIT
