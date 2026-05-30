# ADR-0006: Merge consecutive small children

## Status
Accepted

## Context
Technical docs produce many fragments under 100 chars (table rows, code lines). 72% context failure rate on Rust Book because LLM can't contextualize '| ident::ident | Namespace path |'.

## Decision
After chunking, consecutive children below child_min_chars (default: 100) are merged into single chunks. Configurable via [chunking] config.

## Alternatives considered
- Skip short chunks entirely
- Merge into parent
- Increase child_max_chars globally

## Consequences
Fewer LLM calls, higher context quality on technical docs, fewer failures. Merged chunks may exceed child_max_chars slightly. Closes #139.
