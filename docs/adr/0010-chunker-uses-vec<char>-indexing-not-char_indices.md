# ADR-0010: Chunker uses Vec<char> indexing (not char_indices)

## Status
Accepted

## Context
chunker.rs collects chars into String for byte positions, allocating intermediate Strings. Issue #122 suggested using char_indices() for zero-alloc.

## Decision
Won't fix. Entire chunker (splits, overlaps, positions) is built on Vec<char>. Switching requires rewriting most of the chunker for ~ms gain on 1MB docs.

## Alternatives considered
- char_indices() based chunker
- rope data structure
- Byte-based chunking

## Consequences
Minor allocation overhead during chunking. Not noticeable at our document sizes. Closes #122.
