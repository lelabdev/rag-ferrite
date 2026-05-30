# ADR-0003: No Python ecosystem

## Status
Accepted

## Context
Target users are developers and small teams who want simplicity. Python adds runtime, virtualenv, dependency management.

## Decision
Rust only. No LlamaIndex, LangChain, Haystack, or any Python dependency.

## Alternatives considered
- Python SDK wrapper
- LangChain integration
- Hybrid Rust+Python architecture

## Consequences
Smaller ecosystem but zero Python ops. Never revisit.
