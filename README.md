# rag-lab

RAG engine improvements for LeLabDev — contextual retrieval, reranking, evaluation, and advanced patterns.

## Context

LeLabDev uses `mcp-local-rag` (Node.js) as its RAG engine across 4 instances:

| Instance | Documents | Chunks | Memory |
|---|---|---|---|
| rag-code | 147 | 55,579 | 69.5 MB |
| rag-business | 17 | 36,791 | 62.8 MB |
| rag-general | 5 | 5,212 | 54.0 MB |
| rag-rpg | — | — | — |

All instances run hybrid search (BM25 + vector) with FTS enabled.

## Roadmap

See [Issues](../../issues) for the prioritized improvement backlog.

## License

MIT
