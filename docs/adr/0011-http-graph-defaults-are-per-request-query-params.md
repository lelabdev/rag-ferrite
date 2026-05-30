# ADR-0011: HTTP graph defaults are per-request query params

## Status
Accepted

## Context
api.rs hardcodes graph defaults. Issue #130 suggested making them global config.

## Decision
Won't fix. Graph parameters are per-request by design — different queries need different visualization settings.

## Alternatives considered
- Global config defaults
- Per-collection defaults

## Consequences
Users pass params on each request. Simple and flexible. Closes #130.
