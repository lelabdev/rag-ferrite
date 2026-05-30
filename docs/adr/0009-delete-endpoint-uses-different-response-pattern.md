# ADR-0009: DELETE endpoint uses different response pattern

## Status
Accepted

## Context
api.rs delete_document bypasses the json_response helper and handles errors manually. Issue #128 suggested unifying.

## Decision
Won't fix. DELETE endpoints have different semantics (no body, status-only). Forcing json_response adds complexity for no gain.

## Alternatives considered
- Unify with json_response helper
- Create separate delete_response helper

## Consequences
Slightly inconsistent API code but correct HTTP semantics. Closes #128.
