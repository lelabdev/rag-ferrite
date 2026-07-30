# Optional web interface plan

## Positioning

The web interface is an optional library console for ingestion and retrieval inspection. It is not a chatbot, source editor, model selector, or LLM-generated knowledge graph. Existing files remain the source of truth and SQLite remains a derived index.

## Delivery order

1. Upload supported documents and submit them through the shared bounded ingestion queue.
2. Show live progress, history, errors, queue capacity, and cancellation.
3. List sources and inspect chunks, metadata, tags, and parent-child relationships.
4. Search through the existing retrieval API and show ranked passages with filter state.
5. Add a graph view using existing source/chunk/tag/collection relationships only.

## API requirements

The UI must use the existing authenticated REST API and never access SQLite directly. It must surface structured `error_code` responses and distinguish validation, authentication, not-found, conflict, backpressure, and internal errors. Uploads must respect the configured HTTP body limit and allowed ingestion roots; browser uploads should use the inline ingestion endpoint rather than sending arbitrary server paths.

## Security and operations

- Keep the interface disabled unless explicitly enabled by deployment configuration.
- Reuse REST authentication and Host-header policy.
- Do not expose API keys, `.env` contents, or server filesystem paths.
- Use the existing ingestion progress/history endpoints rather than duplicating workers.
- Keep graph rendering bounded and derived from paginated API responses.

Implementation should begin after storage, retrieval, error semantics, and the shared ingestion queue are stable.
