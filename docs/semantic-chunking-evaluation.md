# Semantic chunking evaluation

## Decision

Semantic chunking is not enabled in rag-ferrite. The current production strategies remain `recursive`, `parent_child`, and `auto`. No repeatable quality gain has been demonstrated that justifies an additional embedding pass during ingestion.

## Evaluation protocol

Use the same corpus snapshot, embedding profile, database settings, and golden dataset for every strategy:

1. Ingest a clean copy of each corpus with one strategy.
2. Record ingestion wall time, embedding request count, chunk count, database size, and index size.
3. Run the versioned retrieval benchmark from `examples/benchmark-golden.json`.
4. Record Recall@k, precision@k, MRR, nDCG, empty-result rate, and p50/p95 query latency.
5. Report results separately for technical documentation, transcripts, books, and short notes.
6. Repeat each run after rebuilding indexes and compare the median of at least three runs.

A semantic strategy should only be added if it improves retrieval ranking consistently for a corpus class without unacceptable ingestion cost or storage growth. The benchmark harness measures retrieval quality; ingestion telemetry must be collected by the deployment runner so provider calls and storage growth are measured without changing the production pipeline.

## Reproducibility

- Pin the corpus revision and golden dataset revision.
- Keep embedding model, dimensions, chunk limits, overlap, and reranker settings identical.
- Store raw benchmark JSON and ingestion telemetry next to the corpus revision.
- Do not make semantic chunking the default based on a single corpus or a single query set.
