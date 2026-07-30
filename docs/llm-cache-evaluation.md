# Persistent LLM cache evaluation

## Decision

No persistent per-operation LLM cache is enabled. The existing in-memory query-result cache remains separate. A persistent cache would need model-aware invalidation, privacy controls, bounded storage, and prompt-version discipline; the current code does not yet provide evidence that repeated LLM operations justify that complexity.

## Measurement protocol

Instrument contextual enrichment, tagging, query expansion, and reranking separately. For each operation record:

- provider and model identity;
- prompt/template version and response format;
- request count, token/cost estimate, latency, and failures;
- exact duplicate prompt count and potential hit rate;
- invalidations caused by configuration, prompt, or indexed-data changes.

Run the measurement during normal ingestion, deliberate re-ingestion, repeated queries, retries, and reranking-heavy workloads. Keep operation classes separate; an aggregate hit rate can hide that only one operation benefits.

A persistent cache should only be implemented when repeated identical operations represent a material share of cost or latency. Any future key must include provider, model, prompt version, parameters, input hash, and response format, with a bounded store, TTL, explicit invalidation, privacy policy, and exposed hit/miss statistics.
