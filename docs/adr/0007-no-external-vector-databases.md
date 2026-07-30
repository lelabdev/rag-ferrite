# ADR-0007: No external vector databases

## Status
Accepted

## Context
Single binary model. External DBs break this.

## Decision
SQLite with sqlite-vec and FTS5. No Qdrant, Milvus, Weaviate.

## Alternatives considered
- Qdrant (Docker)
- Milvus (Docker)
- Weaviate (Docker)

## Consequences
Scale limited to single machine. Sufficient for target use case (<500K chunks). Revisit if scale requirements change.
