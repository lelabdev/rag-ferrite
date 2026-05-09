# rag-ferrite — Custom RAG Engine (Rust)

Moteur RAG custom du lab, en Rust. Buildé sur `rag_engine` + `rmcp` + couches différenciantes.

## Stack technique

| Composant | Choix | Rôle |
|---|---|---|
| **Cœur RAG** | `rag_engine` v0.8.1 | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking, doc parsing |
| **MCP Server** | `rmcp` | Exposition en tant que MCP server (stdio + SSE) |
| **Embeddings** | **BAAI/bge-m3 (Ollama)** | Modèle d'embedding multilingue SOTA, 1024 dimensions, 100+ langues |
| **Compute** | **Ollama on TufTux GPU (RTX 4050)** | Acceleration GPU pour embeddings rapides (~230ms/embedding) |
| **Stockage** | SQLite + HNSW | 1 fichier par RAG, backup = cp |
| **HTTP Bridge** | `axum` | SSE endpoint (remplace le dojo) |

## Pourquoi bge-m3 ?

Après comparaison directe (mêmes queries, mêmes sources) entre **bge-m3** et **qwen3-embedding:0.6b** :

- **bge-m3** est clairement supérieur en qualité de retrieval — il trouve les bons chunks en top 1 plus souvent, surtout sur les queries en français et les questions précises.
- qwen3 a tendance à remonter des résultats hors sujet en top 1.
- bge-m3 supporte **multilingue** (FR + EN), **long-context** (8192 tokens), et **hybrid retrieval** (dense + sparse).
- qwen3 n'est que anglais-focused et n'a pas l'hybrid search.

**Verdict** : bge-m3 est le meilleur choix pour notre RAG multilingue Burning Wheel.

## Performances (benchmark avec bge-m3 sur RTX 4050)

| Fichier | Taille | Chunks | Temps ingest |
|---|---|---|---|
| Anthology | 191K | 154 | 5.2s |
| Codex | 991K | ~800 | 24.2s |
| Gold | 1.2M | ~950 | 25.7s |
| **Total** | **2.4M** | **~1900** | **~55s** |

Comparé à la config initiale (bge-m3 sur CPU aether) : **7m30s → 55s** = **~8x plus rapide**.

## Architecture

```
rag-ferrite (binaire unique)
├── config.toml          ← Quel RAG, quel port, quel embedding provider
├── data/
│   ├── rag-code.sqlite3
│   ├── rag-business.sqlite3
│   ├── rag-general.sqlite3
│   └── rag-rpg.sqlite3
└── indexes/             ← HNSW indexes persistés
```

**4 instances du même binaire**, chacune avec son dossier de données. Comme des workers — même code, différentes données.

Pourquoi séparé :
- Routing clair — une instance Hermes utilise UN RAG ciblé
- Partage sélectif — le dojo expose rag-business, pas rag-code
- Cycle de vie indépendant — réindexer un RAG sans toucher les autres
- Isolation — un RAG corrompu n'en touche pas un autre

## Pipeline

```
Document → Paragraph-based chunking (custom, ~1500 chars)
         → Embedding (bge-m3 via Ollama on TufTux GPU)
         → SQLite + HNSW index
         → BM25 keyword index

Query → MCP tool call
      → Hybrid retrieval (BM25 + HNSW + RRF)
      → Top-k chunks + neighbors
```

## Ce que rag_engine donne GRATUITEMENT

| Feature | Module | Statut |
|---|---|---|
| Vector search (HNSW) | `hnsw_index` | ✅ |
| BM25 keyword search | `bm25_search` | ✅ |
| Hybrid search + RRF | `hybrid_search` | ✅ |
| SQLite storage | `db_pool` + `source_rag` | ✅ |
| Multi-collections | `source_rag` collections | ✅ |
| Chunk neighbors | `get_adjacent_chunks` | ✅ |
| Metadata filtering | `SearchFilter` | ✅ |

## Ce qu'on CODE par-dessus

| Feature | Effort |
|---|---|
| Paragraph-based chunker (fix pour les petits paragraphes) | ⭐⭐ |
| Cross-encoder reranking | ⭐⭐ |
| MCP server (rmcp) | ⭐⭐ |
| Corrective RAG (quality gate) | ⭐⭐ |
| Adaptive RAG (query router) | ⭐⭐ |
| Évaluation (metrics: recall, MRR, nDCG) | ⭐⭐⭐ |
| HTTP SSE bridge (axum, remplace dojo) | ⭐⭐ |

## Comparaison avec l'existant

| | Avant (mcp-local-rag) | Après (rag-ferrite) |
|---|---|---|
| Runtime | Node.js, ~60 MB × 4 instances | Rust, ~10-15 MB × 4 |
| Stockage | LanceDB (4 dossiers) | SQLite (4 fichiers) |
| Recherche | Hybride basique | Hybride RRF + poids custom |
| Chunking | Fixe, boîte noire | Paragraph-based, ~1500 chars |
| Metadata filtering | ❌ | ✅ SQL |
| Reranking | ❌ | ✅ |
| MCP server | Via Hermes config | Natif (rmcp) |
| Bridge HTTP | Dojo séparé | Intégré |
| Évaluation | ❌ | ✅ Metrics intégrées |

## Sources de recherche

- *RAG with Python Cookbook* (Deepak Dhyani)
- *Agentic Architectural Patterns* (Arsanjani & Bustos)
- Anthropic — Contextual Retrieval (2024)
- Hub France IA — Évaluation des Chaînes de RAG (2025)
- rag_engine crate — https://lib.rs/crates/rag_engine
- rmcp — Rust MCP SDK
- Perplexity Research — "Best Embedding Models for RAG (2025)" comparison

## Rôles

- **Ludo** : priorisation, cas clients, go/no-go
- **Mako** : implémentation Rust, benchmarks, tests, PRs

## License

MIT
