# ADR-0002: Hybrid BM25 + HNSW vector search via rag_engine

> **Superseded by ADR-0017.** This document records the historical design and is not the current storage architecture.

## Status
Accepted

## Context
Need both keyword precision and semantic matching. rag_engine v0.8 provides HNSW indexing, BM25, and hybrid fusion (RRF) out of the box.

## Decision
Use rag_engine crate for all search operations. HNSW index built after each ingestion, persisted as .hnsw.data/.hnsw.graph files per collection. Brute force cosine as fallback.

## Alternatives considered
- sqlite-vec for vector indexing
- External vector DB (Qdrant, Milvus)
- Custom HNSW implementation

## Consequences
No need for sqlite-vec (#137 closed). HNSW scales to 500K+ chunks. Index files are portable. Closes #137.
