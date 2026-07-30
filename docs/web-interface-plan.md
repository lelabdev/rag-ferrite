# Optional web console

## Status

The optional library console is implemented and disabled by default. Enable it with:

```toml
[advanced]
web_ui_enabled = true
```

When enabled, the server exposes the console at `/`.

## Scope

The console is for library ingestion and retrieval inspection. It supports:

- inline text and Markdown ingestion;
- ingestion progress, history, errors, queue capacity, and cancellation;
- source listing and confirmed deletion;
- chunk, metadata, tag, and parent-child inspection;
- retrieval inspection with filter state;
- source, chunk, tag, and collection relationships.

It is intentionally not a chatbot, source editor, model selector, or LLM-generated knowledge graph. Existing files remain the source of truth; SQLite remains a derived index.

## API and security constraints

- Use the authenticated REST API; never access SQLite directly.
- Reuse REST authentication and Host-header policy.
- Surface structured `error_code` responses.
- Respect configured HTTP body limits and allowed ingestion roots.
- Use inline ingestion for browser uploads; never accept arbitrary server paths from the browser.
- Do not expose API keys, `.env` contents, or server filesystem paths.
- Keep graph rendering bounded and derived from paginated API responses.

Changes to this scope are tracked through GitHub issues, which is the project’s single source of truth for planning and prioritization.
