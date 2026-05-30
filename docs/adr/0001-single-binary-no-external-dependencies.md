# ADR-0001: Single binary, no external dependencies

## Status
Accepted

## Context
Personal/small-team RAG needs to be simple to deploy and operate. No Docker, no separate services, no Python runtime.

## Decision
Single Rust binary with embedded SQLite. All dependencies compiled in. Copy binary + config file = ready.

## Alternatives considered
- Docker compose with separate services
- Python package (pip install)
- Microservices architecture

## Consequences
Limited to what fits in a single binary. No horizontal scaling. But: zero ops, instant deploy, portable.
