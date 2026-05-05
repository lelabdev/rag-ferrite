# rag-lab — Custom RAG Engine (Rust)

Moteur RAG custom du lab, en Rust. Buildé sur `rag_engine` + `rmcp` + couches différenciantes.

## Stack technique

| Composant | Choix | Rôle |
|---|---|---|
| **Cœur RAG** | `rag_engine` v0.8.1 | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking, doc parsing |
| **MCP Server** | `rmcp` | Exposition en tant que MCP server (stdio + SSE) |
| **Embeddings** | API externe (OpenAI/Cohere/Qwen) ou Ollama local | Vecteurs — plug notre propre provider |
| **Stockage** | SQLite + HNSW | 1 fichier par RAG, backup = cp |
| **HTTP Bridge** | `axum` | SSE endpoint (remplace le dojo) |
| **LLM calls** | `rig` ou custom | Contextual retrieval, query expansion, reranking |

## Architecture

```
rag-lab (binaire unique)
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

## Pipeline cible

```
Document → Document-aware chunking (markdown_chunk / semantic_chunk)
         → Contextual retrieval (LLM context prefix)
         → Metadata enrichment (section, source, type, date)
         → Embedding (API ou local)
         → SQLite + HNSW index

Query → MCP tool call
      → [Router] simple / standard / complex
      → [Si complexe] Query expansion (multi-query)
      → Hybrid retrieval (BM25 + HNSW + RRF)
      → Cross-encoder reranking
      → [Quality gate] Score de confiance
      → [Si faible] Corrective RAG (retry / fallback)
      → Top-k chunks + neighbors
```

## Ce que rag_engine donne GRATUITEMENT

| Feature | Module | Statut |
|---|---|---|
| Vector search (HNSW) | `hnsw_index` | ✅ |
| BM25 keyword search | `bm25_search` | ✅ |
| Hybrid search + RRF | `hybrid_search` + `RrfConfig` | ✅ |
| Semantic chunking | `semantic_chunker` | ✅ |
| Markdown-aware chunking | `markdown_chunk` avec header path | ✅ |
| PDF parsing | `pdf-extract` | ✅ |
| DOCX parsing | `docx-lite` | ✅ |
| SQLite storage | `db_pool` + `source_rag` | ✅ |
| Multi-collections | `source_rag` collections | ✅ |
| Chunk neighbors | `get_adjacent_chunks` | ✅ |
| Metadata filtering | `SearchFilter` | ✅ (partiel) |
| Tokenization | HuggingFace tokenizers | ✅ |
| Compression | `compression_utils` | ✅ Bonus |

## Ce qu'on CODE par-dessus

| Feature | Effort |
|---|---|
| Contextual retrieval (LLM context prefix) | ⭐⭐ |
| Cross-encoder reranking | ⭐⭐ |
| Query expansion (multi-query) | ⭐⭐ |
| MCP server (rmcp) | ⭐⭐ |
| Corrective RAG (quality gate) | ⭐⭐ |
| Adaptive RAG (query router) | ⭐⭐ |
| Évaluation (metrics: recall, MRR, nDCG) | ⭐⭐⭐ |
| Embedding provider abstraction | ⭐⭐ |
| HTTP SSE bridge (axum, remplace dojo) | ⭐⭐ |

## Positionnement produit

### Cas d'usage cibles

**Cabinet médical** — Le médecin pose une question, l'IA trouve la réponse dans ses propres dossiers patients. Tout reste local, rien ne sort.
- 1 500 patients × ~15 docs = ~22 500 docs = ~80 000 chunks → SQLite largement suffisant
- Argument clé : données sensibles → local, pas de cloud, RGPD friendly
- Pas besoin d'internet, backup = copier un fichier

**PME / Startup** — Base de connaissances interne accessible via IA. Docs, wiki, procédures, onboarding.
- ~1 000-10 000 docs → ~400 000 chunks → SQLite suffisant
- Argument clé : auto-hébergé, pas d'abonnement, contrôle total

**Dev / Lab** (notre cas) — Documentation technique, livres, code, references indexés et recherchables.
- ~150 docs → ~55 000 chunks → SQLite suffisant
- Argument clé : chunking intelligent, reranking, multi-domaine

### Limites — Quand SQLite ne suffit plus

| Scale | Solution |
|---|---|
| < 1M chunks (~200k docs) | **SQLite + HNSW** ← notre sweet spot |
| 1M-10M chunks | PostgreSQL + pgvector |
| 10M+ chunks | Distributed vector DB (Qdrant, Milvus) |
| Imagerie multimodale masse | LanceDB ou spécialisé |

### Le pitch

> "Vos documents restent sur votre machine. L'IA les lit localement, rien ne sort. Vous posez une question, elle trouve la réponse dans vos propres docs. Un seul fichier à sauvegarder. Pas d'abonnement cloud."

## Comparaison avec l'existant

| | Avant (mcp-local-rag) | Après (rag-lab) |
|---|---|---|
| Runtime | Node.js, ~60 MB × 4 instances | Rust, ~10-15 MB × 4 |
| Stockage | LanceDB (4 dossiers) | SQLite (4 fichiers) |
| Recherche | Hybride basique | Hybride RRF + poids custom + filtres SQL |
| Chunking | Fixe, boîte noire | Sémantique + Markdown-aware + overlap |
| Metadata filtering | ❌ | ✅ SQL |
| Contextual retrieval | ❌ | ✅ |
| Reranking | ❌ | ✅ |
| Query expansion | ❌ | ✅ |
| Corrective RAG | ❌ | ✅ |
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
- rig — https://github.com/0xPlaygrounds/rig (4.6k stars)

## Rôles

- **Ludo** : priorisation, cas clients, go/no-go
- **Mako** : implémentation Rust, benchmarks, tests, PRs

## License

MIT
