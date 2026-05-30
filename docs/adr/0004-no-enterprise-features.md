# ADR-0004: No enterprise features

## Status
Accepted

## Context
Target is personal/small-team RAG, not enterprise.

## Decision
No RBAC, multi-tenant, compliance, or audit features.

## Alternatives considered
- Add RBAC layer
- Multi-tenant SQLite
- Enterprise SaaS model

## Consequences
Simpler codebase, smaller binary. Never revisit.
