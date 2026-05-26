# Advanced Configuration

## Architecture

```
Document → Pre-ingestion check (quality, duplicates, language)
         → pdftotext (PDFs) or raw text
         → Recursive chunker (800 chars, 10% overlap)
         → Relevance scoring (LLM 1-10, filters junk)
         → Contextual retrieval (LLM context prefix)
         → Auto-tagging (LLM generates tags per chunk)
         → Batch embedding → SQLite + HNSW + BM25
         → Persist HNSW to disk

Query → MCP tool call
      → Classify (simple / standard / complex)
      → Query cache check (300s TTL)
      → Hybrid retrieval (BM25 + HNSW + RRF fusion)
      → LLM reranking (optional, scores top-k results)
      → Quality gate (confidence: High / Medium / Low)
      → [If Low] Corrective RAG (reformulate + retry)
      → Top-k chunks with tags
```

```
rag-ferrite/
├── config.toml
├── .env                 ← LLM_API_KEY
├── data/
│   ├── rag.sqlite3      ← all collections, one DB
│   ├── hnsw_*.hnsw.data ← persisted HNSW indexes
│   └── hnsw_*.hnsw.graph
└── rag-ferrite.log
```

## Performance

| Metric | Value |
|---|---|
| Embedding | Qwen3 Embedding 8B via OpenRouter — ~100ms/embedding |
| Query (warm) | ~300ms (index loaded from disk) |
| Query (cold) | ~4s (first-time index build) |
| Memory idle | ~15 MB |
| 3 books (1.4M chars) | 4,699 chunks in 200s |

## Full Configuration

```toml
# config.toml
data_dir = "./data"

[embedding]
provider = "openrouter"        # "ollama", "openai", "openrouter", etc.
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

# Optional: relevance scoring (filters junk chunks at ingestion)
# [llm]
# relevance_scoring = true
# min_relevance_score = 5.0
```

## Reranking

Post-retrieval reranking improves precision by scoring the top-k results with an LLM. When enabled, results carry both the **hybrid score** (BM25 + HNSW fusion) and a **rerank_score** (0.0–1.0 LLM relevance). Results are sorted by rerank_score when available.

**Two backends:**
- `llm` — Uses your configured LLM to score each passage. Inherits provider/model/api_key/base_url from `[llm]` if not specified.
- `cohere` — Cohere Rerank API (requires `api_key`).

```toml
[reranker]
reranker_type = "llm"    # "disabled" (default), "llm", or "cohere"
top_k = 10               # Number of top results to rerank
# model = "..."          # Override LLM model (defaults to llm.model)
# api_key = "..."        # Override API key (defaults to llm.api_key)
# base_url = "..."       # Override base URL (defaults to llm.base_url)
```

When reranking fails (API error, rate limit), results fall back to hybrid scores with a warning log. No data loss, no silent degradation.

## Domain Metadata Extraction

Extract structured metadata fields from each chunk during ingestion. The LLM identifies domain-specific attributes alongside the contextual prefix — stored as JSON in chunk metadata for filtering and enrichment.

```toml
[metadata]
fields = [
  { name = "topic", field_type = "string" },
  { name = "difficulty", field_type = "string", description = "beginner, intermediate, or advanced" },
  { name = "author", field_type = "string", required = false },
]
```

Fields are extracted during contextual retrieval (no extra LLM calls). Use `metadata_like` in search filters to query by extracted values.

## Golden Dataset Benchmarking

Measure retrieval quality objectively with a golden dataset — a JSON file of question → expected source mappings.

```json
[
  {
    "question": "What are the rules for grappling?",
    "relevant_source_ids": [1, 5],
    "expected_keywords": ["grapple", "strength", "obstacle"]
  }
]
```

Run `benchmark(file_path: "golden.json")` to get hit rate, average score, and per-query details. Use it to catch regressions after config changes, embedding model switches, or pipeline tweaks.

## Relevance Scoring

Optional ingestion-time quality filter. When enabled, the LLM rates each chunk on a 1–10 relevance scale during contextual retrieval. Chunks scoring below the threshold are **discarded before embedding** — they never enter the vector space.

**What it filters out:** table-of-contents entries, index pages, legal mentions / copyright notices, blank or near-blank pages, and transition text ("Chapter 3 begins on the next page").

**Why use it:** cleaner vector space, less RAM usage, better retrieval precision.

**Cost:** zero extra LLM calls. The relevance score is produced alongside the contextual retrieval prefix in the same prompt.

```toml
[llm]
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0   # Discard chunks rated below this (1–10, default 5.0)
```

Both `relevance_scoring` and `context_enabled` must be true.

## Auto-Tagging

When contextual retrieval is enabled, the LLM generates 2-3 descriptive tags per chunk alongside the score and context. Tags are stored in a `chunk_tags` table and returned in query results.

**What it enables:** cross-collection search by topic, fine-grained filtering, metadata-aware retrieval.

**Format:** the LLM returns `TAGS: tag1, tag2, tag3` alongside `SCORE:` and `CONTEXT:`.

**Cost:** zero extra LLM calls. Tags are generated in the same prompt as contextual retrieval.

Tags are automatically generated for all new ingestions. No configuration needed.

## Query Caching

Repeated queries return instantly from an in-memory cache with 300s TTL. The cache stores the full query result (chunks + scores + tags).

**Cache key:** `query|limit|collection`

**When it helps:** interactive exploration where users re-query similar terms, agents making repeated lookups.

**When it invalidates:** after 300 seconds (TTL). Cache is not persisted across restarts.

## Query Pipeline

The query pipeline automatically classifies each query and adapts the strategy:

- **Simple** (1-2 words): direct hybrid search, no expansion, no reranking. Fast.
- **Standard** (3-8 words): query expansion if ≤5 words, LLM reranking. Balanced.
- **Complex** (>8 words or question markers): full pipeline — expansion + reranking + corrective RAG (auto-retry with reformulation if confidence is low).

## Why not just use LangChain?

You absolutely can. LangChain is great — if you need multi-tenant isolation, pluggable retriever chains, agent orchestration, and a team to maintain the pipeline. For personal use, that's a lot of machinery for what boils down to: chunk text → embed → store → query.

## Why not mcp-local-rag?

| | mcp-local-rag | rag-ferrite |
|---|---|---|
| **Language** | TypeScript | Rust |
| **Chunking** | Basic line-based | Custom recursive character splitter (800 chars, 10% overlap) |
| **PDF extraction** | pdf-extract (75% empty pages on complex PDFs) | pdftotext via poppler-utils (gold standard) |
| **Search** | Vector only | Hybrid BM25 + HNSW + RRF fusion |
| **Index persistence** | Rebuild on every restart | HNSW saved to disk, lazy-loaded |
| **Embeddings** | Local or API | Configurable — any OpenAI-compatible provider |
