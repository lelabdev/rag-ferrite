# rag-ferrite — Custom RAG Engine (Rust)

Moteur RAG personnel, en Rust. MCP server unique exposé via stdio a Hermes.

## Stack

| Composant | Choix | Role |
|---|---|---|
| Coeur RAG | rag_engine v0.8.1 | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking, doc parsing |
| MCP Server | rmcp | Exposition stdio via Hermes |
| Embeddings | Ollama (bge-m3) | Vecteurs locaux |
| Stockage | SQLite + HNSW | 1 fichier DB, backup = cp |
| LLM | Z.AI (glm-4.5-flash) | Contextual retrieval |

## Architecture

Binaire unique, mode MCP stdio uniquement. Pas de HTTP, pas de SSE, pas de bridge.

~/services/rag-ferrite/
  rag-ferrite          <- binaire
  rag-ferrite-mcp      <- wrapper (cd + exec)
  config.toml          <- config runtime
  .env                 <- LLM_API_KEY
  data/
    rag.sqlite3
    rag.sqlite3-shm
    rag.sqlite3-wal
    hnsw_*.hnsw.data
    hnsw_*.hnsw.graph
  rag-ferrite.log

Code source : ~/dev/rag-ferrite/rag-ferrite/
Deploiement : copie du binaire compile vers ~/services/rag-ferrite/

## Config

config.toml (dans le cwd au lancement) :

  data_dir = "/home/loops/services/rag-ferrite/data"

  [embedding]
  provider = "ollama"
  model = "bge-m3:latest"
  dimensions = 1024
  base_url = "http://100.88.8.1:11434"   # Tailscale TufTux

  [llm]
  provider = "zai"
  model = "glm-4.5-flash"
  base_url = "https://api.z.ai/api/coding/paas/v4"
  context_enabled = true
  max_concurrent = 2

API key via .env -> LLM_API_KEY.

## Collections

Multi-collections dans la meme DB. Chaque outil MCP accepte un parametre collection optionnel.

## Pipeline

  Document -> Document-aware chunking (markdown_chunk / semantic_chunk)
           -> Contextual retrieval (LLM context prefix)
           -> Metadata enrichment (section, source, type, date)
           -> Embedding (Ollama bge-m3)
           -> SQLite + HNSW index

  Query -> MCP tool call
        -> [Si complexe] Query expansion (multi-query)
        -> Hybrid retrieval (BM25 + HNSW + RRF)
        -> Cross-encoder reranking
        -> [Quality gate] Score de confiance
        -> [Si faible] Corrective RAG (retry / fallback)
        -> Top-k chunks + neighbors

## Outils MCP (7)

| Outil | Description |
|---|---|
| query_documents | Recherche hybride (BM25 + vector + RRF) avec filtres |
| ingest_file | Ingest un fichier (PDF, DOCX, TXT, MD) |
| ingest_data | Ingest du contenu brut (texte, HTML, markdown) |
| delete_file | Supprime un document par source_id |
| list_files | Liste les documents indexes |
| status | Stats : nombre de documents |
| read_chunk_neighbors | Chunks adjacents pour expansion de contexte |

## Ce que rag_engine fournit

| Feature | Module |
|---|---|
| Vector search (HNSW) | hnsw_index |
| BM25 keyword search | bm25_search |
| Hybrid search + RRF | hybrid_search |
| Semantic chunking | semantic_chunker |
| Markdown-aware chunking | markdown_chunk |
| PDF / DOCX parsing | pdf-extract / docx-lite |
| SQLite storage | db_pool + source_rag |
| Multi-collections | source_rag |
| Chunk neighbors | get_adjacent_chunks |
| Metadata filtering | SearchFilter |
| Tokenization | HuggingFace tokenizers |
| Compression | compression_utils |

## Ce qu on code par-dessus

- Contextual retrieval (LLM context prefix)
- Cross-encoder reranking
- Query expansion (multi-query)
- Corrective RAG (quality gate)
- Embedding provider abstraction
- Evaluation (metrics: recall, MRR, nDCG)

## Build & Deploy

  cd ~/dev/rag-ferrite/rag-ferrite
  cargo build --release
  cp target/release/rag-ferrite ~/services/rag-ferrite/rag-ferrite

Pas de docker, pas de systemd. Hermes lance le wrapper rag-ferrite-mcp en stdio.

## Roles

- Ludo : priorisation, go/no-go
- Mako : implementation Rust, tests, PRs

## License

MIT
