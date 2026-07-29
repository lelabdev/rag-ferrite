<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">

# rag-ferrite

**A lightweight personal knowledge base for AI assistants.**

Give Claude Code, Codex, Hermes, Claude Desktop, and other MCP-compatible clients fast access to your documents, notes, transcripts, and technical knowledge.

**Collect first. Retrieve when needed. Organize only what matters.**

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release\&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid search · Local storage · Native MCP · Single Rust binary

</div>

---

## What is rag-ferrite?

`rag-ferrite` is a self-hosted knowledge server for AI assistants.

You already have an assistant. You already have documents. `rag-ferrite` gives the assistant a reliable way to search those documents whenever it needs context.

```text
Your files
Markdown · PDF · DOCX · TXT · transcripts · documentation
                              │
                              ▼
                         rag-ferrite
                keyword + semantic retrieval
                              │ MCP
                              ▼
        Claude Code · Codex · Hermes · other AI clients
```

`rag-ferrite` does not provide a chatbot, replace your assistant, or impose a note-taking application. Your existing AI client calls it through MCP and receives relevant passages from your own knowledge base.

Your original files remain the source of truth. They can be carefully organized notes, raw documents, books, project documentation, transcripts, research papers, or anything in between.

### One knowledge base, two ways to use it

A personal library can act as a second brain in two complementary ways:

* **For you:** files remain readable, editable, portable, and usable with any editor or note-taking tool.
* **For your assistants:** `rag-ferrite` turns the same files into a retrieval layer they can search by exact wording or meaning.

You keep the human-readable knowledge you control. Your assistants gain a machine-oriented way to find the right parts without requiring you to locate, open, and paste them into every conversation.

---

## Quick start

### Install

```bash
curl -fsSL https://raw.githubusercontent.com/lelabdev/rag-ferrite/main/install.sh | bash
```

Or build from source:

```bash
git clone https://github.com/lelabdev/rag-ferrite.git
cd rag-ferrite
cargo build --release
```

The compiled binary is:

```text
target/release/ragfer
```

### PDF support

PDF extraction requires Poppler.

Debian or Ubuntu:

```bash
sudo apt install poppler-utils
```

Fedora:

```bash
sudo dnf install poppler-utils
```

Arch Linux:

```bash
sudo pacman -S poppler
```

---

## Configuration

`rag-ferrite` uses:

* an embedding model for semantic retrieval;
* an LLM for contextual processing, tagging, query recovery, and optional reranking.

Set the required API keys:

```bash
export LLM_API_KEY="your-llm-api-key"
export EMBEDDING_API_KEY="your-embedding-api-key"
```

On its first server run, `ragfer` creates a default configuration file when none exists.

Minimal example:

```toml
data_dir = "./data"
http_port = 4242

[embedding]
provider = "openai"
model = "qwen/qwen3-embedding-8b"
dimensions = 512
base_url = "https://openrouter.ai/api/v1"

[llm]
provider = "ollama"
model = "gemma4:31b"
base_url = "https://api.ollama.com"
```

Start the server:

```bash
ragfer serve
```

When HTTP is enabled:

```text
MCP Streamable HTTP: http://localhost:4242/mcp
REST API:            http://localhost:4242/api
```

Running the binary without arguments opens the terminal monitor:

```bash
ragfer
```

---

## Connect MCP clients

### Hermes over Streamable HTTP

```yaml
mcp_servers:
  rag-ferrite:
    url: "http://localhost:4242/mcp"
    timeout: 9999
```

Streamable HTTP is useful when:

* several assistants use the same knowledge base;
* the service runs continuously;
* the server is located on another machine;
* ingestion should continue after a client closes;
* one persistent index is shared by multiple clients.

### Hermes over stdio

```yaml
mcp_servers:
  rag-ferrite:
    command: /path/to/ragfer
    args: ["serve"]
    timeout: 9999
    env:
      LLM_API_KEY: "..."
      EMBEDDING_API_KEY: "..."
```

### Claude Desktop

```json
{
  "mcpServers": {
    "rag-ferrite": {
      "command": "/path/to/ragfer",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "...",
        "EMBEDDING_API_KEY": "..."
      }
    }
  }
}
```

Claude Code and other MCP clients can connect through stdio or Streamable HTTP depending on their supported configuration.

---

## Why I built it

Personal knowledge rarely arrives in a clean and organized form.

It may be:

* a Markdown note;
* a PDF;
* a course;
* a video transcript;
* technical documentation;
* a game guide;
* an article;
* a research paper;
* a document that may become useful months later.

Traditional knowledge management often expects you to process everything before it becomes useful:

```text
read
→ summarize
→ classify
→ link
→ remember where it was stored
```

That works well for carefully maintained notes, but it creates a large amount of work when you collect more information than you have time to organize.

`rag-ferrite` enables a different workflow:

```text
collect
→ ingest
→ retrieve when needed
→ organize only what matters
```

Your sources can be raw, structured, or somewhere in between.

Once indexed, they become searchable by your AI assistants through MCP.

You can:

* ask a question immediately;
* recover something you forgot;
* compare information across several sources;
* retrieve advice while working or playing;
* generate a structured note later;
* keep the source without manually summarizing it first.

The goal is not to replace your files, editor, or note-taking system.

The goal is to make everything you may need later easier to retrieve.

---

## A practical example

You may collect several video transcripts, guides, forum exports, and notes about a game such as Victoria 3.

Instead of manually turning every source into a polished note, you can ingest them directly into `rag-ferrite`.

Later, while playing, you can ask your assistant:

```text
What is the best way to increase construction capacity
without destabilizing my economy?
```

The assistant can search the indexed guides and transcripts, retrieve the relevant passages, compare the advice, and help you make a decision.

The same workflow applies to:

* programming documentation;
* courses;
* research;
* personal projects;
* books;
* hobbies;
* technical references;
* professional knowledge.

The information becomes useful before it has been perfectly organized.

---

## More than ordinary file search

A folder of documents is already a useful personal knowledge base.

It is:

* portable;
* readable;
* easy to edit;
* easy to back up;
* independent from a particular application.

However, traditional file search is mostly lexical.

It works best when you already know:

* the exact filename;
* the exact term used in the document;
* the folder containing the information;
* the wording of the original note.

It works less well when:

* the query uses different vocabulary;
* the relevant information is spread across several documents;
* two sources express the same idea differently;
* you want to compare several viewpoints;
* you do not remember where something was written;
* a relevant passage does not contain your exact keywords;
* you want an AI assistant to explore the knowledge base autonomously.

For example, a search for:

```text
database corruption during concurrent indexing
```

may fail to find a note containing:

```text
parallel index rebuilds can damage stored search data
```

The meaning is related, but the wording is different.

`rag-ferrite` combines lexical and semantic retrieval so both kinds of matches can be found.

---

## Why not use a complete RAG platform?

Many RAG solutions are designed as full applications.

They may require:

```text
Python
+ a Web application
+ a vector database
+ background workers
+ several containers
+ an ingestion service
+ an embedding service
+ a chat interface
+ user management
```

These platforms can be powerful, but they are often unnecessarily complex for a personal knowledge base.

`rag-ferrite` takes a smaller and more focused approach:

```text
one binary
+ one local database
+ your preferred model providers
+ an MCP connection
```

It does not impose another chat interface.

Instead, it connects the assistants you already use to the documents you already have.

---

## Core principles

* **Your files remain the source of truth.**
* **MCP is the primary interface.**
* **One knowledge base can be shared by several assistants.**
* **Exact terms and semantic meaning both matter.**
* **Knowledge should be useful before it is perfectly organized.**
* **The system should remain simple enough for personal use.**
* **No external vector database should be required.**
* **Local and hosted model providers should both be supported.**
* **The service should remain understandable and maintainable by one person.**

---

## Common use cases

### Personal documentation

Give your assistant access to procedures, references, project notes, and technical documentation.

```text
Search my documentation for the backup restoration procedure.
```

### Coding assistants

Allow Claude Code or another coding agent to retrieve architecture decisions and project conventions.

```text
Before changing the storage layer, search for previous architecture decisions.
```

### Courses and learning material

Index courses, books, notes, and transcripts without summarizing every source manually.

```text
Explain the differences between these approaches using my course material.
```

### Video transcripts

Collect YouTube or podcast transcripts and search them later.

```text
Find the videos that discussed hybrid retrieval and summarize the key differences.
```

### Personal research

Search papers, articles, and books by meaning rather than only by title or keywords.

```text
Find the sources discussing the limitations of semantic chunking.
```

### Hobbies and games

Build a knowledge base from guides, transcripts, and reference documents.

```text
Based on my Victoria 3 guides, what should I prioritize in this economic situation?
```

### Cross-document comparison

Retrieve several passages covering the same subject.

```text
Compare the recommendations about local vector databases.
```

### Notes and knowledge libraries

Index notes from any editor or Markdown-based knowledge library.

```text
Find my previous notes about authentication, even if they use different terms.
```

### Note or document generation

Use retrieved context to create a report, checklist, documentation page, or synthesis note.

```text
Use several relevant sources to create a structured reference note.
```

Generating a note is optional. The knowledge base remains useful even when no new note is created.

---

## Supported sources

`rag-ferrite` can ingest:

* Markdown;
* plain text;
* PDF;
* DOCX;
* raw text supplied through the API;
* HTML or Markdown content supplied directly.

Possible source directories include:

```text
~/library/
~/Documents/
~/Projects/*/docs/
~/Notes/
```

The system does not depend on a particular editor or note-taking application.

---

## Ingest documents

### One file

```bash
ragfer ingest-file "/path/to/document.md"
```

### Several files

```bash
ragfer ingest-batch \
  "/path/to/book.pdf" \
  "/path/to/documentation.md" \
  "/path/to/transcript.txt"
```

### Raw content

```bash
cat note.md | ragfer ingest-data "manual-note"
```

### Select a collection

```bash
ragfer ingest-file "/path/to/rust-book.pdf" -c programming
```

### Force re-ingestion

```bash
ragfer ingest-file "/path/to/document.md" --force
```

---

## Search from the CLI

Basic search:

```bash
ragfer query "How does hybrid retrieval work?"
```

Limit the number of results:

```bash
ragfer query "SQLite vector search" -n 5
```

Filter by tags:

```bash
ragfer query "MCP authentication" -t security,mcp
```

Select a collection:

```bash
ragfer query "async Rust runtime" -c programming
```

Return raw JSON:

```bash
ragfer query "embedding dimensions" --json
```

---

## MCP tools

### Search and reading

| Tool                                           | Description                                                                             |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `query_documents(query, tags?, limit?)`        | Search indexed documents using hybrid retrieval, filters, query recovery, and reranking |
| `read_chunk_neighbors(source_id, chunk_index)` | Retrieve passages surrounding a specific result                                         |
| `list_files()`                                 | List indexed documents                                                                  |
| `status()`                                     | Return server and index status                                                          |
| `suggest_collection(query)`                    | Suggest the most relevant collection                                                    |
| `tag_map()`                                    | Show tags, collections, and chunk counts                                                |

### Ingestion and quality

| Tool                                                  | Description                                        |
| ----------------------------------------------------- | -------------------------------------------------- |
| `ingest_file(file_path, collection?)`                 | Ingest a PDF, DOCX, TXT, or Markdown file          |
| `ingest_data(content, source, collection?, format?)`  | Ingest raw text, HTML, or Markdown                 |
| `check_ingestion(file_path?, content?, source_name?)` | Inspect document quality before indexing           |
| `benchmark(file_path, collection?, limit?)`           | Evaluate retrieval against a golden dataset        |
| `collection_heat()`                                   | Show frequently and recently queried collections   |
| `chunk_qa()`                                          | Identify cold, unused, or potentially noisy chunks |

### Administration

| Tool                                         | Description                                        |
| -------------------------------------------- | -------------------------------------------------- |
| `delete_file(source)`                        | Remove a document and its chunks                   |
| `reassign_collection(source_id, collection)` | Move a source to another collection                |
| `rebuild_indexes()`                          | Rebuild search indexes and checkpoint the database |
| `flush_indexes()`                            | Persist recently indexed vector data               |

---

## HTTP ingestion

### One file

```bash
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.md"
  }'
```

### Several files

```bash
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "paths": [
      "/path/to/book.pdf",
      "/path/to/article.md",
      "/path/to/transcript.txt"
    ]
  }'
```

Batch ingestion returns immediately with a batch identifier.

Monitor progress:

```bash
ragfer progress
```

Or:

```bash
curl http://localhost:4242/api/ingest/progress
```

---

## Auto-move after ingestion

By default, files can be moved after successful ingestion.

This supports inbox-style workflows:

```text
inbox/
└── article.md

        ↓ ingestion

ingested/
└── article.md
```

Configuration:

```toml
[advanced]
move_after_ingest = true
ingested_dir = "ingested"
```

Disable it for a specific request:

```json
{
  "paths": ["/path/to/article.md"],
  "move_after_ingest": false
}
```

For a permanent library directory, disabling automatic movement may be preferable.

---

## Client configuration

The CLI reads its configuration from:

```text
~/.config/ragfer/
```

| File          | Purpose    |
| ------------- | ---------- |
| `config.toml` | Server URL |
| `.env`        | API key    |

Interactive setup:

```bash
ragfer setup
```

The default server URL is:

```text
http://localhost:4242
```

---

## Terminal monitor

Launch the built-in TUI:

```bash
ragfer
```

Or:

```bash
ragfer monitor
```

The monitor displays:

* server status;
* indexed document count;
* active ingestion batch;
* current document;
* processed chunks;
* ingestion speed;
* estimated completion time;
* recent errors;
* activity events;
* ingestion history.

Environment variables:

```text
RAGFER_URL       Server URL
RAG_API_KEY      Server API key
RAGFER_KEY       Alternative key variable
RAGFER_REFRESH   Refresh interval
```

---

## Authentication and network use

Generate an API key:

```bash
ragfer key generate
```

Provide it to clients:

```bash
export RAG_API_KEY="your-key"
```

Or store it in:

```text
~/.config/ragfer/.env
```

`rag-ferrite` is primarily designed for trusted personal environments.

Recommended deployments:

* localhost;
* a private workstation;
* a home server;
* a trusted local network;
* a private Tailscale network.

Avoid exposing the service directly to the public Internet without reviewing the current authentication and security configuration.

A dedicated operating-system user is recommended when the service can ingest local filesystem paths.

---

## Features

| Feature                      | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| **MCP-native**               | Direct integration with Claude Code, Hermes, Claude Desktop, and other MCP clients |
| **Hybrid retrieval**         | Combines full-text and vector search                                               |
| **FTS5 keyword search**      | Preserves exact names, identifiers, commands, and error messages                   |
| **Semantic search**          | Finds related concepts and paraphrases                                             |
| **sqlite-vec**               | Local vector retrieval inside SQLite                                               |
| **Reciprocal rank fusion**   | Combines lexical and semantic rankings                                             |
| **Optional reranking**       | Improves final result precision                                                    |
| **Parent-child chunking**    | Balances precise matching with broader context                                     |
| **Context expansion**        | Retrieves neighboring passages around a result                                     |
| **Query recovery**           | Reformulates weak queries and retries                                              |
| **Noise filtering**          | Removes low-value and boilerplate chunks                                           |
| **Automatic tagging**        | Adds fine-grained topic metadata                                                   |
| **Collections**              | Organizes documents into broad knowledge domains                                   |
| **Batch ingestion**          | Indexes multiple files asynchronously                                              |
| **Ingestion quality checks** | Inspects documents before adding them                                              |
| **Retrieval benchmarks**     | Evaluates search against golden datasets                                           |
| **Collection heat tracking** | Shows frequently and recently queried collections                                  |
| **Chunk-level QA**           | Identifies cold, unused, or potentially noisy chunks                               |
| **REST API**                 | Allows integration outside MCP                                                     |
| **CLI and TUI**              | Manages and monitors the service from a terminal                                   |
| **Local database**           | Uses SQLite without a separate database server                                     |
| **Provider flexibility**     | Supports local and hosted OpenAI-compatible APIs                                   |
| **Single binary**            | No Python runtime or mandatory Docker stack                                        |

---

## Architecture

```text
              ┌─────────────────────────────┐
              │ Markdown · PDF · DOCX · TXT │
              │ Notes · docs · transcripts  │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Extraction and cleaning     │
              │ Noise filtering             │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Parent-child chunking       │
              │ Context and auto-tagging    │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ SQLite                      │
              │ FTS5 + sqlite-vec           │
              │ Metadata and tags           │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Hybrid retrieval            │
              │ Reciprocal rank fusion      │
              │ Query recovery              │
              │ Optional reranking          │
              └──────────────┬──────────────┘
                             │
                             ▼
          ┌─────────────────────────────────────┐
          │ MCP · REST API · CLI · Terminal UI │
          └─────────────────────────────────────┘
```

---

## How retrieval works

```text
Documents
    │
    ▼
Extraction and cleaning
    │
    ▼
Parent-child chunking
    │
    ▼
Embeddings + full-text index
    │
    ├──▶ Keyword search
    │
    └──▶ Vector search
             │
             ▼
      Reciprocal rank fusion
             │
             ▼
       Optional reranking
             │
             ▼
       Context expansion
             │
             ▼
        MCP search result
```

Each stage solves a different retrieval problem.

---

## Hybrid search

Hybrid search combines lexical and semantic retrieval. Filtered searches use the same active SQLite connection for candidate filtering, avoiding recursive connection-lock deadlocks. Query-cache entries include the complete source, collection, metadata, and chunk filters, are canonicalized for unordered IDs, and are invalidated after data mutations.

```text
User query
    │
    ├──▶ Full-text search
    │      Exact words, names, identifiers
    │
    └──▶ Vector search
           Meaning, concepts, paraphrases
                 │
                 ▼
          Rank fusion and reranking
                 │
                 ▼
             Final results
```

### Full-text search

Full-text search is effective for precise terms.

Examples:

```text
RAG_API_KEY
sqlite-vec
ECONNREFUSED
src/storage/sqlite.rs
ADR-0015
```

A purely semantic system may treat these tokens as unimportant or confuse them with related concepts.

Keyword retrieval preserves exact matching.

### Vector search

Vector search represents queries and passages as embeddings.

This makes it possible to retrieve related content even when the wording differs.

```text
Query:
How can agents access my documentation?

Possible matching passage:
The MCP server exposes indexed knowledge to external AI clients.
```

The exact words are different, but the meaning is related.

### Why combine them?

Neither approach works best for every query.

| Query type                  | Keyword search | Vector search |
| --------------------------- | -------------: | ------------: |
| Exact command or identifier |      Excellent |      Variable |
| Error message               |      Excellent |      Variable |
| General concept             |        Limited |     Excellent |
| Paraphrased question        |        Limited |     Excellent |
| Proper name                 |      Excellent |          Good |
| Related explanation         |        Limited |     Excellent |
| Mixed technical query       |           Good |          Good |

Hybrid retrieval improves the probability that the right passage reaches the candidate set.

---

## Reciprocal rank fusion

Keyword and vector searches return scores with different meanings.

A lexical relevance score cannot be compared directly with a vector similarity score.

`rag-ferrite` uses reciprocal rank fusion to combine the rankings instead of comparing their raw scores.

```text
Keyword ranking       Vector ranking
---------------       --------------
1. Document A         1. Document B
2. Document C         2. Document A
3. Document B         3. Document D
       │                     │
       └─────────┬───────────┘
                 ▼
          Fused ranking
          --------------
          1. Document A
          2. Document B
          3. Document C
          4. Document D
```

A passage that ranks well in both retrieval methods receives a stronger final position.

This prevents the system from depending too heavily on either keywords or embeddings.

---

## Reranking

Initial retrieval is optimized for recall.

Its job is to find a broad set of potentially relevant passages quickly.

However, the initial ranking is not always perfect.

A passage may contain many matching words without answering the actual question. Another passage may be semantically similar but not practically useful.

Reranking adds a second relevance evaluation:

```text
Initial retrieval
20 candidate passages
        │
        ▼
Detailed relevance evaluation
        │
        ▼
Best passages moved to the top
```

Without reranking, an assistant may receive:

* repeated passages;
* documents that mention the topic only briefly;
* results matching the wording but not the intent;
* semantically similar but irrelevant text.

Reranking evaluates the candidates against the exact query and improves their final order.

| Stage             | Objective                               |
| ----------------- | --------------------------------------- |
| Hybrid retrieval  | Avoid missing relevant information      |
| Rank fusion       | Combine lexical and semantic candidates |
| Reranking         | Improve precision and final ordering    |
| Context expansion | Recover surrounding explanations        |

---

## Parent-child chunking

Large documents cannot be searched efficiently as a single block.

They need to be split into passages.

Very small chunks improve precision but may lose context.

Very large chunks preserve context but reduce retrieval precision.

`rag-ferrite` uses a parent-child approach:

```text
Parent section
┌─────────────────────────────────────┐
│ Broader topic and explanation       │
│                                     │
│  ┌──────────┐  ┌──────────┐         │
│  │ Child 1  │  │ Child 2  │  ...    │
│  └──────────┘  └──────────┘         │
└─────────────────────────────────────┘
```

Small child chunks are used for precise matching.

Broader parent context can then be returned so the assistant receives a coherent explanation rather than an isolated sentence.

This is especially useful for:

* technical documentation;
* books;
* long articles;
* research papers;
* architecture documents;
* courses;
* transcripts.

---

## Context expansion

A search result may identify the correct passage without containing the full explanation.

The MCP tool `read_chunk_neighbors` lets an assistant retrieve the surrounding chunks:

```text
Previous chunk
      │
Matched chunk
      │
Next chunk
```

This allows agents to search precisely first and expand the context only when necessary.

It avoids returning large amounts of text for every query while still making the full explanation available.

---

## Query recovery

Users do not always use the same terminology as the indexed documents.

Initial results may therefore be weak.

`rag-ferrite` can detect weak retrieval, reformulate the query, and search again.

```text
Original query
      │
      ▼
Weak results detected
      │
      ▼
Query reformulation
      │
      ▼
Second retrieval attempt
```

This is useful when:

* the user uses informal vocabulary;
* the sources use technical terminology;
* a concept has several names;
* the first query is too broad;
* the wording is ambiguous.

---

## Automatic tagging and collections

Documents can be organized into broad collections, while individual chunks receive more specific tags.

Example collections:

```text
programming
research
personal
games
projects
documentation
courses
```

Example tags:

```text
rust
authentication
economy
victoria-3
database
mcp
performance
```

Tags passed to `query_documents` use AND logic:

```text
1 tag  → broad topic filtering
2 tags → precise intersection
```

For example:

```text
security
```

may return all security-related passages, while:

```text
security + mcp
```

focuses on passages related to both topics.

---

## Comparing complementary and conflicting sources

`rag-ferrite` does not decide by itself whether two sources contradict each other.

Its role is retrieval.

Semantic and hybrid search can surface passages that discuss the same topic using different vocabulary.

An AI assistant can then compare the passages and identify:

* agreements;
* complementary explanations;
* alternative approaches;
* outdated decisions;
* conflicting recommendations;
* differences between sources.

```text
Source A:
Use a full index rebuild after every ingestion.

Source B:
Incremental insertion avoids expensive rebuilds.

                    │
                    ▼
       AI assistant compares both
```

A simple keyword search may fail to place these passages together when they use different terminology.

---

## REST API reference

| Method   | Path                        | Description                      |
| -------- | --------------------------- | -------------------------------- |
| `GET`    | `/api/status`               | Server status and document count |
| `POST`   | `/api/query`                | Search indexed knowledge         |
| `POST`   | `/api/ingest`               | Ingest one or several files      |
| `POST`   | `/api/ingest/data`          | Ingest raw content               |
| `GET`    | `/api/ingest/progress`      | Show ingestion progress          |
| `GET`    | `/api/documents`            | List indexed documents           |
| `GET`    | `/api/documents/{id}`       | Get document details             |
| `DELETE` | `/api/documents/{id}`       | Delete a document                |
| `GET`    | `/api/graph`                | Return source relationship data  |
| `POST`   | `/api/flush-indexes`        | Persist pending index data       |
| `POST`   | `/api/rebuild-indexes`      | Rebuild indexes                  |
| `POST`   | `/api/service/cancel-batch` | Cancel the active batch          |
| `POST`   | `/api/service/stop`         | Stop the server                  |

---

## CLI reference

```text
ragfer                         Open the terminal monitor
ragfer serve                   Start the server
ragfer status                  Show server status
ragfer progress                Show ingestion progress
ragfer query "text"            Search documents
ragfer list                    List indexed documents
ragfer monitor                 Open the terminal monitor
ragfer ingest-file <path>      Ingest one file
ragfer ingest-batch <paths>    Ingest several files
ragfer ingest-data <name>      Ingest standard input
ragfer delete <source_id>      Delete a document
ragfer flush                   Persist pending index data
ragfer rebuild                 Rebuild indexes
ragfer cancel                  Cancel the active batch
ragfer stop                    Stop the server
ragfer restart                 Restart the service
ragfer reload                  Reload supported configuration
ragfer history                 Show ingestion history
ragfer setup                   Configure the CLI client
ragfer key generate            Generate a server API key
ragfer key show                Display the current API key
ragfer key list                List configured keys
ragfer update                  Install the latest release
```

Common options:

| Option            | Description                       |
| ----------------- | --------------------------------- |
| `--json`          | Return raw JSON                   |
| `-c <collection>` | Select a collection               |
| `-n <limit>`      | Set the result limit              |
| `-t <tags>`       | Filter with comma-separated tags  |
| `--force`         | Replace an already indexed source |

---

## What rag-ferrite is not

`rag-ferrite` is not:

* a replacement for Markdown;
* a replacement for your editor or note-taking application;
* a complete chat application;
* a hosted AI platform;
* an enterprise document-management system;
* a regulated archive;
* a public multi-tenant RAG service;
* a framework requiring you to assemble the retrieval pipeline yourself.

It is a focused personal knowledge service that gives AI assistants better access to your documents.

---

## When a simple folder is enough

You may not need `rag-ferrite` when:

* you have only a small number of documents;
* filenames and folders are sufficient;
* exact keyword search finds everything you need;
* you do not use AI assistants;
* you already remember where information is stored;
* you rarely search across several sources.

A simple collection of Markdown files remains one of the best formats for personal knowledge.

`rag-ferrite` becomes useful when your collection grows and you want assistants to retrieve information by meaning, not only by exact words.

---

## When rag-ferrite is useful

`rag-ferrite` is especially useful when:

* your knowledge is spread across many files;
* some sources are raw or poorly organized;
* you collect more information than you can summarize;
* you use several MCP-compatible assistants;
* you want one shared knowledge base;
* you frequently forget where information was written;
* your query uses different wording from the documents;
* you need exact technical search and semantic search;
* you want to compare several sources;
* you want a capable RAG without maintaining a large software stack.

---

## Scope and scaling

`rag-ferrite` is designed for personal knowledge bases ranging from a few documents to hundreds of thousands of searchable passages.

Its recommended default of 512 embedding dimensions offers a practical balance between:

* semantic quality;
* storage usage;
* memory usage;
* retrieval speed.

The project prioritizes personal-scale simplicity over enterprise-scale distributed infrastructure.

For very large corpora, collection routing, filtering, chunk quality, and retrieval configuration become increasingly important.

---

## Project status

`rag-ferrite` is developed primarily for personal and trusted-network use.

The project prioritizes:

* retrieval quality;
* operational simplicity;
* MCP compatibility;
* local storage;
* provider independence;
* maintainability;
* low infrastructure requirements.

Current improvement areas include:

* stronger MCP authentication;
* read-only access profiles;
* more integration tests;
* continuous integration;
* improved source citations;
* watched-folder synchronization;
* additional retrieval benchmarks.

See the GitHub issues for the latest implementation status.

---

## License

MIT