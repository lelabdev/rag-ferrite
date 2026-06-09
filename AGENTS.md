# rag-ferrite — Custom RAG Engine (Rust)

Moteur RAG personnel, en Rust. MCP server exposé via stdio ou Streamable HTTP.

## Stack

| Composant | Choix | Rôle |
|---|---|---|
| Coeur RAG | rag_engine (fork lelabdev/rag-engine) | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking |
| MCP Server | rmcp | Exposition stdio + Streamable HTTP |
| Embeddings | OpenRouter (Qwen3 8B) | 512 dims (sweet spot perf/RAM) |
| LLM | Ollama Cloud (Gemma4 31B) | Scoring, contextual retrieval, tagging, reranking |
| LLM Profiles | Modular per action | Ingestion, query, reranker can use different providers/models |
| Stockage | SQLite + HNSW | 1 fichier DB, backup = cp |

## Architecture

Binaire `ragfer` (Cargo.toml: `name = "ragfer"`). CLI intégré (`src/client.rs`) + daemon en un seul binaire.

Deux modes d'exécution :

- **`ragfer`** (sans args) = monitor TUI (défaut)
- **`ragfer serve`** / **`ragfer -d`** = daemon MCP (stdio ou HTTP)
- **`ragfer <commande>`** = commande client (status, list, query, progress, ingest, delete, flush, rebuild, cancel, stop, monitor, update, help)

Flags courts : `-s` status, `-l` list, `-q` query, `-p` progress, `-m` monitor, `-d` serve.

Le daemon expose :
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
- Embedding dimensions: 512. Content is broad topics (books, transcripts, tech docs) where BM25 + tag routing compensate the minimal accuracy loss. Keeps RAM low and scales to 4× the data without OOM.
- Fork of rag_engine under `lelabdev/rag-engine` — enables mmap for vector data loading (OS manages page cache, cold collections naturally evicted from RAM)
- Async batched chunk heat (mpsc channel + 30s flush) vs old synchronous N UPDATEs
- Global atomic patterns for cross-cutting signals (chunk_counter, cancel)
- Delete instant (no synchronous index rebuild)
- External query classification dictionary (optional TOML, `dictionaries/query_classification.toml`)
- CLI intégré dans le binaire `ragfer` (src/client.rs) — remplace l'ancien CLI Python
- Standalone rag-monitor binary (client-side TUI via HTTP polling)
- Activity log: global ring buffer (last 20 events) with OnceLock + Mutex; engine pushes events during ingestion (embedding, llm, chunking, error, info); exposed in progress API for real-time monitoring
- Live elapsed/speed/ETA: `get_progress()` recalculates `elapsed_seconds` from `started_at` timestamp on every call — no stale counters, always up-to-date

### Structure du code

```
src/
  main.rs        — MCP server (rmcp), initialise pipeline + reranker
  client.rs      — CLI intégré : commandes client (status, list, query, progress, ingest, delete, flush, rebuild, cancel, stop, monitor, update, help)
  service.rs     — Couche service partagée MCP + HTTP
  ingestion.rs   — Queue d'ingestion non-bloquante (mpsc + background worker)
  api.rs         — HTTP endpoints (axum, optionnel)
  pipeline.rs    — Orchestration query (simple/standard/complex), cache
  engine/
    mod.rs       — init(), config(), stats(), sanitize(), IngestOptions
    ingest.rs    — ingest_text(), ingest_file(), pipeline parent-child
    indexes.rs   — HNSW/BM25 rebuild, buffer, WAL checkpoint
    precheck.rs  — pre-check, langue, doublons, vérification chunks
    chunk_heat.rs — Chunk heat async batched (mpsc + 30s flush)
    chunk_counter.rs — Compteur atomique global pour progression temps réel
    cancel.rs    — Flag d'annulation global pour batch cancellation
    activity_log.rs — Ring buffer global (20 events), push/snapshot/clear, exposé dans progress API
    search.rs    — search_hybrid(), search_hybrid_with_expansion()
    query.rs     — get_section_paths, get_neighbors, delete_source, list_sources
    benchmark.rs — run_benchmark(), get_graph_data()
    tags.rs      — chunk_tags + collection_tags tables, insert/update/get tags
    tag_routing.rs — Keyword extraction, collection_tags matching, route_query()
    heat.rs      — Collection heat tracking: HeatTracker, EMA decay, chunk QA
  bin/
    rag-monitor.rs — Binaire TUI standalone, polling HTTP pour monitoring
  llm.rs         — LlmProvider (ollama + openai), contextual retrieval, scoring, tagging, profile builder
  reranker.rs    — Reranker (LLM + passthrough), rerank_hybrid()
  embedding.rs   — EmbeddingProvider (openai-compatible)
  config.rs      — Config TOML parsing, LlmProfile struct, profile lookup, dictionary loader
  chunker.rs     — Recursive chunking, section extraction, language detection
  extractor.rs   — PDF/DOCX/text extraction
  types.rs       — Structs partagés + From impls
dictionaries/
  query_classification.toml — Dictionnaire externe de mots-clés pour classification de query
```

## Config actuelle

Config **serveur** : `~/services/rag-ferrite/config.toml` (inchangé, voir exemple ci-dessous).

Config **client** (`ragfer` CLI) : `~/.config/ragfer/config.toml` :

```toml
server_url = "http://127.0.0.1:8080"
```

Clé API client : `~/.config/ragfer/.env` avec `RAG_API_KEY=…`, ou variable d'environnement `RAG_API_KEY`.

Premier lancement : `ragfer setup` — prompt interactif qui crée `~/.config/ragfer/config.toml` + `.env`.

Le flag `--env` et les instances codées en dur ont été supprimés ; tout passe par le fichier config.

Exemple config serveur (`~/services/rag-ferrite/config.toml`) :

```toml
data_dir = "/home/loops/services/rag-ferrite/data"

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 512
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
```

API keys serveur via `.env` : `LLM_API_KEY` (Ollama Cloud), `EMBEDDING_API_KEY` (OpenRouter).

## Pipeline d'ingestion

```
Document → Pre-ingestion check (qualité, doublons, langue)
         → Extraction texte (pdftotext / docx-lite / raw)
         → Chunking parent-child ou récursif (auto-détecté)
         → Merge consecutive small children (<100 chars)
         → Skip chunks below child_min_chars (no LLM call)
         → Relevance scoring LLM (1-10, filtre le bruit)
         → Contextual retrieval (LLM context prefix, batch + retry)
         → Auto-tagging (2-3 tags par chunk)
         → Embedding batch → SQLite + HNSW + BM25
```

Ingestion is queued via mpsc channel — HTTP returns immediately.
Progress: GET /api/ingest/progress

**v5.0**: Le pipeline intègre désormais un compteur atomique global (`chunk_counter`) pour la progression temps réel et un check d'annulation (`cancel`) entre chaque fichier traité. Le suivi est consultable via le binaire `rag-monitor`.

**v5.1**: Instrumentation activity log — chaque étape du pipeline pousse un événement typé (`embedding`, `llm`, `chunking`, `error`, `info`) dans un ring buffer global (`activity_log`). `get_progress()` recalcule `elapsed_seconds` depuis `started_at` à chaque appel (live elapsed, speed, ETA). Le TUI `rag-monitor` affiche l'historique d'activité avec timestamps et les fichiers traités en bas permanent.

## Pipeline de query

```
Query → MCP tool call
      → Classification (simple / standard / complex)
      → [Si standard/complex] Query expansion (LLM multi-query)
      → Tag routing (sélection collection via collection_tags)
      → Hybrid retrieval (BM25 + HNSW + RRF)
      → LLM reranking (scoring 0-1 des top-k résultats)
      → Query caching (résultats en cache 300s)
      → [Quality gate] Score de confiance
      → [Si faible] Corrective RAG (reformulation + retry)
      → Top-k chunks avec tags
```

## Outils MCP (16)

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
| collection_heat | Heat tracking par collection (heat_score, last_queried_at, query_count) |
| chunk_qa | QA chunk-level : chunks morts/froids par source (heat calculé à la volée) |
| suggest_collection | Tag routing : suggère la meilleure collection pour une query |
| tag_map | Mapping complet tag → collection avec chunk counts |
| reassign_collection | Déplace un source (et ses chunks) vers une autre collection |
| rebuild_indexes | Reconstruction des indexes HNSW + BM25 + WAL checkpoint |
| flush_indexes | Flush du buffer HNSW incrémental vers le disque |

## Build & Deploy

```bash
cd ~/dev/rag-ferrite-hub/rag-ferrite
cargo build --release --bin ragfer
```

### Déploiement via GitHub Releases

1. Créer une release avec les binaires :
   ```bash
   gh release create vX.Y.Z target/release/ragfer target/release/rag-monitor
   ```
2. Sur la machine cible (Nova ou aether) :
   ```bash
   ragfer update
   # ou via le wrapper :
   ~/services/rag-ferrite/rag-ferrite update
   ```
3. Sur chaque machine cliente, créer le config client :
   ```bash
   ragfer setup   # crée ~/.config/ragfer/config.toml + .env
   ```

Le binaire appelle `update.sh` (à côté de lui dans `~/services/rag-ferrite/`).
Le script : stop service → vérifie arrêt → télécharge depuis GitHub Releases → remplace binaire → restart.

### Symlinks

| Machine | Chemin |
|---|---|
| aether | `~/bin/ragfer` → `~/services/rag-ferrite/ragfer` |
| TufTux | `~/.local/bin/ragfer` → `~/services/rag-ferrite/ragfer` |
| Nova | `rag-ferrite-mcp` (wrapper) appelle `exec ./ragfer serve` |

Chaque machine cliente doit aussi avoir `~/.config/ragfer/config.toml` + `.env` (créés via `ragfer setup`).

### Fichiers de déploiement

```
~/services/rag-ferrite/
  ragfer              ← binaire unique (CLI + daemon)
  rag-monitor          ← binaire TUI monitoring standalone
  rag-ferrite-mcp      ← wrapper Nova (exec ./ragfer serve)
  update.sh            ← script de mise à jour (appelé par `ragfer update`)
  config.toml          ← config runtime
  .env                 ← LLM_API_KEY, EMBEDDING_API_KEY, RAG_API_KEY
  data/
    rag.sqlite3
    hnsw_*.hnsw.data   ← index vectoriels persistés
    hnsw_*.hnsw.graph
  rag-ferrite.log
```

## Tests

```bash
cargo test    # 50 tests unitaires (chunker, extractor, config, llm, tags, pipeline, tag_routing)
```

## ⚠️ Mise à jour des docs — OBLIGATOIRE

Après chaque changement de fonctionnalité, mettre à jour CES 3 FICHIERS :

1. **llms.txt** — doc publique (API, config, features)
2. **AGENTS.md** — conventions, architecture, structure du code (ce fichier)
3. **README.md** — si changements visibles pour l'utilisateur final

Pas d'exception. Si on ajoute/supprime/modifie une feature → on update les docs dans le même commit.

## Rôles

- Ludo : priorisation, go/no-go
- Mako : implémentation Rust, tests, PRs

## License

MIT
