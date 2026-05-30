# ADR-0008: No GraphRAG or multi-hop reasoning

## Status
Accepted

## Context
Adds knowledge graph layer. Complex, heavy to maintain. At our scale, LLM does multi-hop naturally with multiple relevant chunks.

## Decision
Rely on LLM for reasoning over retrieved chunks. No knowledge graph.

## Alternatives considered
- GraphRAG with entity extraction
- Knowledge graph in SQLite
- Neo4j integration

## Consequences
Simpler system. May miss complex cross-document patterns. Revisit if users need it.
