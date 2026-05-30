# ADR 0013: Skip small chunks before LLM call

**Status:** Accepted
**Date:** 2026-05-30
**Issue:** #139 (partial)

## Context

Chunks below `child_min_chars` (default: 100 chars) were sent to the LLM for context generation. The LLM consistently failed on these tiny fragments, producing "context failures" in the report. These were not real failures — they were expected for fragments like table cells or short headings.

## Decision

Skip chunks below `child_min_chars` entirely. No LLM call, no context prefix. Store with raw content only. The chunks remain searchable via embeddings.

Report now shows: `contextualized / skipped / failed / filtered` instead of just `context_failures`.

## Consequences

- Saves LLM tokens on technical docs (up to 70% fewer calls on docs like Rust Book)
- Accurate stats: skipped ≠ failed
- Slightly less context for very short chunks, but they're too short to benefit anyway
