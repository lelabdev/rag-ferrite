# rag-lab — RAG Engine Improvements

Backlog et implémentation des améliorations du RAG du lab (mcp-local-rag).

## Contexte

Le lab utilise `mcp-local-rag` (Node.js) avec 4 instances:

| Instance | Documents | Chunks | Mémoire |
|---|---|---|---|
| rag-code | 147 | 55 579 | 69.5 MB |
| rag-business | 17 | 36 791 | 62.8 MB |
| rag-general | 5 | 5 212 | 54.0 MB |
| rag-rpg | — | — | — |

Toutes les instances utilisent la recherche hybride (BM25 + vector) avec FTS activé.

## État des lieux

**Ce qui fonctionne:**
- Recherche hybride (keyword + vector)
- Chunk neighbors (expansion post-retrieval)
- Bases spécialisées par domaine
- Support multi-format (PDF, DOCX, TXT, MD, HTML, web)
- Ré-ingestion pour mise à jour

**Les 3 gaps majeurs:**
1. Pas de reranking → plus gros manque, plus gros gain potentiel
2. Pas de contextual retrieval → chunks nus, sans contexte structurel
3. Pas d'évaluation → on ne mesure pas objectivement la qualité

## Roadmap

Voir les [Issues](../../issues) — 10 améliorations classées par impact/effort.

### Priorité recommandée (sprint order)

1. **#1 — Contextual Retrieval** (Impact ⭐⭐⭐⭐⭐ | Effort ⭐⭐)
2. **#2 — Reranking Cross-Encoder** (Impact ⭐⭐⭐⭐⭐ | Effort ⭐⭐)
3. **#9 — Évaluation Continue** (Impact ⭐⭐⭐⭐ | Effort ⭐⭐⭐)
4. **#3 — Query Expansion** (Impact ⭐⭐⭐⭐ | Effort ⭐⭐)
5. **#4 — Document-Aware Chunking** (Impact ⭐⭐⭐⭐ | Effort ⭐⭐⭐)

## Pipeline cible

```
Document → Document-aware chunking + contextual retrieval
         → Metadata enrichment (section, source, type)
         → Embedding (Qwen3-Embedding ou BGE-M3)
         → Index BM25 + Vector (hybride)

Query → Router (simple / standard / complexe)
      → [Si complexe] Query expansion → reformulations
      → Hybrid retrieval (BM25 + dense)
      → Cross-encoder reranking
      → [Quality gate] Score de confiance
      → [Si faible] Corrective RAG (retry / fallback)
      → Top-k chunks → LLM
```

## Sources de recherche

- *RAG with Python Cookbook* (Deepak Dhyani) — dans rag-code
- *Agentic Architectural Patterns* (Arsanjani & Bustos) — dans rag-code
- Anthropic — Contextual Retrieval (2024)
- Hub France IA — Évaluation des Chaînes de RAG (2025)
- Hoko Team — 11 Stratégies RAG pour 2026
- Firecrawl — Best Chunking Strategies for RAG 2025/2026
- MTEB Leaderboard (HuggingFace)
- Rapport interne — session 2026-05-05 (Perplexity + livres RAG)

## Rôles

- **Ludo** : priorisation, décisions tech, go/no-go
- **Mako** : implémentation, benchmarks, PRs
