# rag-ferrite — Custom RAG Engine (Rust)

Moteur RAG personnel, en Rust. MCP server exposé via stdio ou Streamable HTTP.

## Stack

| Composant | Choix | Rôle |
|---|---|---|
| Coeur RAG | rag_engine v0.8 | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking |
| MCP Server | rmcp | Exposition stdio + Streamable HTTP |
| Embeddings | OpenRouter (Qwen3 8B) | Vecteurs 4096 dims |
|| LLM | Ollama Cloud (Gemma4 31B) | Scoring, contextual retrieval, tagging, reranking |
|| LLM Profiles | Modular per action | Ingestion, query, reranker can use different providers/models |
| Stockage | SQLite + HNSW | 1 fichier DB, backup = cp |

## Architecture

Binaire unique. Deux modes :

- **stdio** (défaut) : MCP sur stdin/stdout, Hermes spawn le process
- **Streamable HTTP** (`http_port > 0`) : MCP sur HTTP `/mcp`, service indépendant, accessible à distance

### Architecture decisions

All technical decisions are documented as ADR files in `docs/adr/`.
See `docs/DECISIONS.md` for the index.

Key decisions:
- Single binary, Rust, SQLite + HNSW (no external DB, no Python)
- Parent-child chunking with contextual retrieval
- Parallel parents (JoinSet) + batch children for ingestion speed
- Modular LLM profiles: different models for ingestion, query, reranker (see `[[llm_profile]]` in config)
- Non-blocking ingestion queue (mpsc channel + background worker)
- Merge consecutive small children (<100 chars) for technical docs
- Skip small chunks before LLM call (saves tokens, accurate stats)
- Progress endpoint for monitoring active ingestions
- Won't fix: #122 (chunker rewrite), #128 (DELETE pattern), #130 (graph config)

~/services/rag-ferrite/
  rag-ferrite          <- binaire
  rag-ferrite-mcp      <- wrapper (cd + exec)
  config.toml          <- config runtime
  .env                 <- LLM_API_KEY, EMBEDDING_API_KEY
  data/
    rag.sqlite3
    hnsw_*.hnsw.data   <- index vectoriels persistés
    hnsw_*.hnsw.graph
  rag-ferrite.log

Code source : ~/dev/rag-ferrite/rag-ferrite/
Déploiement : copie du binaire compilé vers ~/services/rag-ferrite/

### Structure du code

src/
  main.rs        — MCP server (rmcp), initialise pipeline + reranker
  service.rs     — Couche service partagée MCP + HTTP
  ingestion.rs   — Queue d'ingestion non-bloquante (mpsc + background worker)
  api.rs         — HTTP endpoints (axum, optionnel)
  pipeline.rs    — Orchestration query (simple/standard/complex), cache
  engine/
    mod.rs       — init(), ingest_text(), ingest_file(), get_conn(), data_dir()
    search.rs    — search_hybrid(), search_hybrid_with_expansion()
    query.rs     — get_section_paths, get_neighbors, delete_source, list_sources
    benchmark.rs — run_benchmark(), get_graph_data()
    tags.rs      — create_chunk_tags_table, insert_chunk_tags, get_tags_for_chunk_ids
  llm.rs         — LlmProvider (ollama + openai), contextual retrieval, scoring, tagging, profile builder
  reranker.rs    — Reranker (LLM + passthrough), rerank_hybrid()
  embedding.rs   — EmbeddingProvider (openai-compatible)
  config.rs      — Config TOML parsing, LlmProfile struct, profile lookup
  chunker.rs     — Recursive chunking, section extraction, language detection
  extractor.rs   — PDF/DOCX/text extraction
  types.rs       — Structs partagés + From impls

## Config actuelle

config.toml :

  data_dir = "/home/loops/services/rag-ferrite/data"

  [embedding]
  provider = "openai"
  model = "qwen/qwen3-embedding-8b"
  dimensions = 4096
  base_url = "https://openrouter.ai/api/v1"

  [llm]
  provider = "ollama"
  model = "gemma4:31b"
  base_url = "https://api.ollama.com"
  context_enabled = true
  relevance_scoring = true
  min_relevance_score = 5.0
  max_concurrent = 3

  [reranker]
  reranker_type = "llm"
  top_k = 10

API keys via .env : LLM_API_KEY (Ollama Cloud), EMBEDDING_API_KEY (OpenRouter).

## Pipeline d'ingestion

  Document → Pre-ingestion check (qualité, doublons, langue)
           → Extraction texte (pdftotext / docx-lite / raw)
           → Chunking parent-child ou récursif (auto-détecté)
           → Merge consecutive small children (<100 chars)
           → Skip chunks below child_min_chars (no LLM call)
           → Relevance scoring LLM (1-10, filtre le bruit)
           → Contextual retrieval (LLM context prefix, batch + retry)
           → Auto-tagging (2-3 tags par chunk)
           → Embedding batch → SQLite + HNSW + BM25

Ingestion is queued via mpsc channel — HTTP returns immediately.
Progress: GET /api/ingest/progress

## Pipeline de query

  Query → MCP tool call
        → Classification (simple / standard / complex)
        → [Si standard/complex] Query expansion (LLM multi-query)
        → Hybrid retrieval (BM25 + HNSW + RRF)
        → LLM reranking (scoring 0-1 des top-k résultats)
        → Query caching (résultats en cache 300s)
        → [Quality gate] Score de confiance
        → [Si faible] Corrective RAG (reformulation + retry)
        → Top-k chunks avec tags

## Outils MCP (9)

| Outil | Description |
|---|---|
| query_documents | Recherche hybride avec filtres, reranking, expansion, cache |
| ingest_file | Ingest un fichier (PDF, DOCX, TXT, MD) |
| ingest_data | Ingest du contenu brut (texte, HTML, markdown) |
| delete_file | Supprime un document + ses chunks + tags |
| list_files | Liste les documents indexés |
| status | Stats : nombre de documents |
| read_chunk_neighbors | Chunks adjacents pour expansion de contexte |
| check_ingestion | Preview qualité avant ingestion |
| benchmark | Évaluation qualité vs dataset golden |

## Ce que rag_engine fournit (crate externe v0.8)

| Feature | Module |
|---|---|
| Vector search (HNSW) | hnsw_index |
| BM25 keyword search | bm25_search |
| Hybrid search + RRF | hybrid_search |
| Semantic chunking | semantic_chunker |
| Markdown-aware chunking | markdown_chunk |
| PDF / DOCX parsing | pdf-extract / docx-lite |
| SQLite storage | db_pool + source_rag |
| Chunk neighbors | get_adjacent_chunks |
| Metadata filtering | SearchFilter |
| Tokenization | HuggingFace tokenizers |

## Ce qu'on code par-dessus

- Contextual retrieval (LLM context prefix)
- Relevance scoring (filtrage ingestion)
- Auto-tagging (tags LLM par chunk)
- LLM reranking (via LlmProvider partagé)
- Query expansion (multi-query)
- Query reformulation (corrective RAG)
- Query caching (in-memory TTL 300s)
- Embedding provider abstraction
- Ollama Cloud auth (Bearer token)
- Évaluation (benchmark vs golden dataset)

## Build & Deploy

  cd ~/dev/rag-ferrite/rag-ferrite
  cargo build --release
  cp target/release/rag-ferrite ~/services/rag-ferrite/rag-ferrite-new
  # Puis reload MCP pour swaper

Pas de docker, pas de systemd. Hermes lance le wrapper rag-ferrite-mcp en stdio.

## Tests

  cargo test    # 34 tests unitaires (chunker, extractor, config, llm, tags)

## Rôles

- Ludo : priorisation, go/no-go
- Mako : implémentation Rust, tests, PRs

## License

MIT
