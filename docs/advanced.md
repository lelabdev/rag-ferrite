# Advanced Configuration

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

**Why use it:** cleaner vector space, less RAM usage, better retrieval precision. Junk chunks that would dilute search results are simply never indexed.

**Cost:** zero extra LLM calls. The relevance score is produced alongside the contextual retrieval prefix in the same prompt — it's a single additional line in the output.

**How to enable:**

```toml
[llm]
context_enabled = true
relevance_scoring = true
min_relevance_score = 5.0   # Discard chunks rated below this (1–10, default 5.0)
```

Both `relevance_scoring` and `context_enabled` must be true.
