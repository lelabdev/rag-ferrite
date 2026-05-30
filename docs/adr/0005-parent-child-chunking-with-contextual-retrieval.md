# ADR-0005: Parent-child chunking with contextual retrieval

## Status
Accepted

## Context
Documents need precise matching (small chunks) AND sufficient context (large chunks). Research shows parent-child with contextual retrieval improves quality 35-49%.

## Decision
Parent chunks (~2000 chars) for LLM context. Child chunks (~200 chars) for embedding and search. LLM generates context prefix for each child. Progressive commit with resume support.

## Alternatives considered
- Fixed-size chunks only
- Semantic chunking only
- No contextual retrieval

## Consequences
Slower ingestion (LLM calls) but higher retrieval quality. Parallel parents (~3x speedup) mitigates the cost.
