# ADR-0012: Parallel parents with JoinSet for ingestion speed

## Status
Accepted

## Context
Sequential parent processing was the bottleneck. Batch children gave 1.7x speedup. Parents still processed one at a time.

## Decision
Use tokio::JoinSet to process up to max_concurrent parents in parallel. LLM + embedding happen concurrently. DB writes remain sequential (SQLite).

## Alternatives considered
- Sequential with larger batches
- Full async with connection pooling
- Separate ingestion workers

## Consequences
~3x speedup on top of batch children (~5x total). Requires cloning LLM/embedder per task. Pipeline stays full by spawning next task before processing result. Closes #136.
