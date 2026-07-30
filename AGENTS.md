# rag-ferrite — Moteur RAG personnel (Rust)

Moteur RAG personnel, en Rust. Binaire unique `rag-ferrite`, MCP server (stdio ou Streamable HTTP), CLI intégré, TUI monitor intégré.

## Stack

| Composant | Choix | Rôle |
|---|---|---|
| Coeur RAG | rag_engine (fork lelabdev/rag-engine) | HNSW vector search, BM25, hybrid fusion (RRF), SQLite storage, semantic chunking |
| MCP Server | rmcp | Exposition stdio + Streamable HTTP |
| Embeddings | OpenRouter (Qwen3 8B) | 512 dims (sweet spot perf/RAM) |
| LLM | Ollama Cloud (Gemma4 31B) | Scoring, contextual retrieval, tagging, reranking |
| LLM Profiles | Modular per action | Ingestion, query, reranker peuvent utiliser des providers/models différents |
| Stockage | SQLite + HNSW | 1 fichier DB, backup = cp |
| TUI Monitor | ratatui | Intégré dans le binaire (src/monitor/) |

## Architecture

Binaire unique `ragfer` (Cargo.toml: package name `rag-ferrite`).

Trois modes d'exécution :

- **`ragfer`** (sans args) = monitor TUI (défaut)
- **`ragfer serve`** / **`-d`** = daemon MCP (stdio ou HTTP)
- **`ragfer <commande>`** = commande client (status, list, query, progress, ingest, delete, flush, rebuild, cancel, stop, monitor, update, setup, help)

Flags courts : `-s` status, `-l` list, `-q` query, `-p` progress, `-m` monitor, `-d` serve.

Le daemon expose :
- **stdio** (défaut) : MCP sur stdin/stdout, Hermes spawn le process
- **Streamable HTTP** (`http_port > 0`) : MCP sur HTTP `/mcp`, service indépendant, accessible à distance

### Décisions d'architecture

ADR dans `docs/adr/`. Index dans `docs/DECISIONS.md`.

Décisions clés :
- Binaire unique, Rust, SQLite + HNSW (pas de DB externe, pas de Python)
- Parent-child chunking avec contextual retrieval
- Parents parallèles (JoinSet) + enfants en batch pour la vitesse d'ingestion
- LLM profiles modulaires : modèles différents pour ingestion, query, reranker (`[[llm_profile]]` dans config)
- Bounded ingestion queue (mpsc channel + single background worker): REST and MCP share the queue; rebuild/flush operations are serialized with writes, with inline-size, HTTP-body, and per-job timeout limits.
- Merge des children consécutifs trop petits (<100 chars) pour les docs techniques
- Skip des petits chunks avant l'appel LLM (économise les tokens, stats précises)
- Endpoint de progression pour monitorer les ingestions actives
- Dimensions embedding : 512. Contenu broad (livres, transcriptions, docs tech) où BM25 + tag routing compensent la perte minime. RAM basse, 4× plus de données sans OOM
- Fork de rag_engine sous `lelabdev/rag-engine` — mmap pour le chargement vectoriel (l'OS gère le page cache, collections froides naturellement évitées de la RAM)
- Chunk heat async batched (canal mpsc + flush 30s) vs anciens N UPDATEs synchrones
- Patterns atomiques globaux pour signaux transversaux (chunk_counter, cancel)
- Delete instantané (pas de rebuild synchrone des indexes)
- Dictionnaire externe de classification de query (TOML optionnel, `dictionaries/query_classification.toml`)
- CLI intégré dans le binaire (src/client.rs) — remplace l'ancien CLI Python
- Monitor TUI intégré au binaire (src/monitor/) — remplace l'ancien binaire standalone rag-monitor
- Server struct extraite de main.rs → src/server.rs
- Tags consolidés dans src/tag_rules.rs (sanitize + global state)
- IngestOptions éliminé — le moteur utilise params::IngestConfig directement
- Activity log : ring buffer global (20 derniers events) avec OnceLock + Mutex ; le moteur pousse des événements pendant l'ingestion (embedding, llm, chunking, error, info) ; exposé dans la progress API
- Live elapsed/speed/ETA : `get_progress()` recalcule `elapsed_seconds` depuis `started_at` à chaque appel — pas de compteurs périmés
- Atomic tags + AND filtering (v5.2.0) : tags normalisés en minuscules (les tags hyphenés sont préservés), filtrage par intersection (AND) pour la précision. Pre-filter SQL INTERSECT pour petits sets, post-filter avec over-fetch pour grands sets (>2000 chunks). Intersection vide → retour immédiat avec `tag_filter_note`.
- Vague 1 search correctness : filtered vector/BM25 searches resolve one shared allowed-ID set per hybrid query and adaptively over-fetch sqlite-vec/FTS5 candidates; parent-child commits synchronize child rows with FTS5 + sqlite-vec and populate `chunk_role`; query-cache keys include all search filters and invalidate on data mutations.
- Storage integrity : source deletion is transactional across chunks, tags, FTS5, and sqlite-vec; deduplication uses SHA-256 with lazy legacy-hash migration; parent-child resume uses persisted logical parent indices rather than completion counts.
- Retrieval evaluation : `engine::benchmark::run_benchmark` accepts legacy arrays and versioned golden datasets, and reports Recall@k, precision@k, MRR, nDCG, empty-result rate, plus p50/p95 latency.

### Structure du code

```
src/
├── main.rs (359 lines) — entry point, CLI dispatch, server startup
├── server.rs (191 lines) — RagFerriteServer struct + MCP tool methods
├── client.rs (384 lines) — CLI client (subcommands via ureq HTTP)
├── config.rs (884 lines) — TOML config loading
├── llm.rs (802 lines) — LLM provider, context generation, response parsing
├── chunker.rs (837 lines) — text chunking (recursive + parent-child)
├── ingestion.rs (680 lines) — ingestion queue + progress
├── pipeline.rs (470 lines) — query pipeline, classification, cache
├── api.rs (392 lines) — HTTP routes (axum)
├── service.rs (302 lines) — shared business logic (MCP + HTTP)
├── reranker.rs (330 lines) — reranker (LLM, Cohere, Disabled)
├── embedding.rs (246 lines) — embedding provider
├── types.rs (228 lines) — shared types
├── params.rs (141 lines) — parameter structs (IngestConfig, QueryParams, etc.)
├── extractor.rs (119 lines) — PDF/DOCX/text extraction
├── tag_rules.rs (181 lines) — tag rules + sanitize + global state
├── monitor/
│   ├── mod.rs (630 lines) — App struct, event loop, key handling
│   ├── ui.rs (947 lines) — ratatui rendering
│   └── api.rs (128 lines) — HTTP fetch for monitor
├── engine/
│   ├── mod.rs (221 lines) — DB init, schema migrations, stats
│   ├── ingest.rs (829 lines) — core ingestion
│   ├── search.rs (137 lines) — hybrid search
│   ├── query.rs (199 lines) — query helpers
│   ├── benchmark.rs (225 lines) — benchmarks
│   ├── indexes.rs (129 lines) — HNSW/BM25 index management
│   ├── heat.rs (348 lines) — collection heat tracking
│   ├── tag_routing.rs (163 lines) — tag-based routing
│   ├── tags.rs (129 lines) — tag DB operations
│   ├── precheck.rs (145 lines) — pre-ingestion check
│   ├── chunk_heat.rs (109 lines) — chunk heat tracking
│   ├── activity_log.rs (68 lines) — activity log ring buffer
│   ├── chunk_counter.rs (22 lines) — atomic chunk counter
│   └── cancel.rs (20 lines) — cancel flag
```

Pas de `bin/`. Pas de `rag-monitor` standalone. Monitor = `src/monitor/` (3 fichiers).

## Config actuelle

Config **serveur** : `~/services/rag-ferrite/config.toml`.

Config **client** (`ragfer` CLI) : `~/.config/ragfer/config.toml` :

```toml
server_url = "http://127.0.0.1:8080"
```

Clé API client : `~/.config/ragfer/.env` avec `RAG_API_KEY=…`, ou variable d'environnement `RAG_API_KEY`.

Premier lancement : `ragfer setup` — prompt interactif qui crée `~/.config/ragfer/config.toml` + `.env`.

Pas de flag `--env`, pas d'instances codées en dur ; tout passe par le fichier config et les env vars.

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

Ingestion mise en queue via canal mpsc — HTTP retourne immédiatement.
Progression : GET /api/ingest/progress

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

## Build & Deploy

```bash
cd ~/dev/rag-ferrite-hub/rag-ferrite
cargo build --release
```

### Déploiement via GitHub Releases

1. Créer une release avec le binaire :
   ```bash
   gh release create vX.Y.Z target/release/ragfer
   ```
2. Sur la machine cible (Nova ou aether) :
   ```bash
   ragfer update
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

Chaque machine cliente doit avoir `~/.config/ragfer/config.toml` + `.env` (créés via `ragfer setup`).

### Fichiers de déploiement

```
~/services/rag-ferrite/
  ragfer              ← binaire unique (CLI + daemon + monitor)
  rag-ferrite-mcp     ← wrapper Nova (exec ./ragfer serve)
  update.sh           ← script de mise à jour (appelé par `ragfer update`)
  config.toml         ← config runtime
  .env                ← LLM_API_KEY, EMBEDDING_API_KEY, RAG_API_KEY
  data/
    rag.sqlite3
    hnsw_*.hnsw.data   ← index vectoriels persistés
    hnsw_*.hnsw.graph
  rag-ferrite.log
```

## Tests

```bash
cargo test    # tests unitaires (chunker, extractor, config, llm, tags, pipeline, tag_routing)
```

## ⚠️ Mise à jour des docs — OBLIGATOIRE

Après chaque changement de fonctionnalité, mettre à jour CES 3 FICHIERS :

1. **README.md** — si changements visibles pour l'utilisateur final
2. **llms.txt** — doc technique complète (API, config, features, architecture)
3. **AGENTS.md** — conventions, architecture, structure du code (ce fichier)

Pas d'exception. Si on ajoute/supprime/modifie une feature → on update les docs dans le même commit.

## Rôles

- Ludo : priorisation, go/no-go
- Mako : implémentation Rust, tests, PRs

## License

MIT
