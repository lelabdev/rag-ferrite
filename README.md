<div align="center">
  <img src="assets/logo.svg" alt="rag-ferrite" width="128" height="128">

# rag-ferrite

**A lightweight personal knowledge base for AI assistants.**

Give Claude Code, Hermes, Claude Desktop, and other MCP-compatible clients fast access to your documents, notes, and technical knowledge.

[![Release](https://img.shields.io/github/v/release/lelabdev/rag-ferrite?label=release\&color=cyan)](https://github.com/lelabdev/rag-ferrite/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid search · Local storage · Native MCP · Single Rust binary

</div>

---

## What is rag-ferrite?

`rag-ferrite` is a self-hosted knowledge server designed for personal AI assistants.

It indexes your documents and exposes them through MCP, allowing tools such as Claude Code, Hermes, Claude Desktop, and other compatible clients to search your personal knowledge.

```text
Markdown · PDF · DOCX · TXT · documentation
                     │
                     ▼
                rag-ferrite
       keyword + semantic retrieval
                     │
                     ▼
      Claude Code · Hermes · MCP clients
```

It can be used with:

* a documentation directory;
* personal Markdown notes;
* an Obsidian vault;
* technical references;
* books and PDF files;
* project documentation;
* research papers;
* exported conversations;
* video transcripts;
* specifications and architecture decisions.

Your original files remain the source of truth.

`rag-ferrite` creates a searchable index on top of them so AI assistants can retrieve useful context without requiring you to manually find and paste the right document into every conversation.

---

## Why I built it

A folder of Markdown files is already a good personal knowledge base.

It is:

* portable;
* readable;
* easy to edit;
* easy to back up;
* independent from a particular application;
* compatible with tools such as Obsidian.

However, plain files and traditional vault search have important limitations.

They work well when you already know:

* the exact filename;
* the exact term used in the document;
* the folder containing the information;
* the wording of the original note.

They work less well when:

* the query uses different vocabulary;
* the relevant information is spread across several documents;
* two sources express the same concept differently;
* you want to compare several viewpoints;
* you need to recover an old decision without remembering its exact wording;
* a relevant passage does not contain the keywords you searched for;
* you want an AI assistant to explore the knowledge base autonomously.

`rag-ferrite` was created to keep the simplicity of a personal document collection while adding the retrieval capabilities expected from a modern RAG system.

The goal is not to replace your editor, your Markdown files, or Obsidian.

The goal is to give your AI assistants a fast and reliable way to search them.

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

These systems can be powerful, but they are often unnecessarily complex for a personal knowledge base.

`rag-ferrite` takes a smaller and more focused approach:

```text
One binary
+ one local database
+ your preferred model providers
+ an MCP connection
```

It does not provide another mandatory chat interface.

Instead, it connects the assistants you already use to the documents you already have.

---

## Core principles

* **Your files remain the source of truth.**
* **MCP is the primary interface.**
* **The knowledge base should be shared by several assistants.**
* **Search should combine exact terms and semantic meaning.**
* **The system should remain simple enough for personal use.**
* **No external vector database should be required.**
* **Local and hosted models should both be supported.**
* **The service should remain understandable and maintainable by one person.**

---

## More than a simple vault search

Traditional search is usually lexical.

It finds documents containing the words from your query.

For example, a search for:

```text
database corruption during concurrent indexing
```

may fail to find a document containing:

```text
parallel index rebuilds can damage stored search data
```

The two passages discuss a related problem, but they do not use the same vocabulary.

Vector search helps retrieve text by semantic similarity.

Keyword search remains valuable for exact technical information such as:

* function names;
* filenames;
* error messages;
* command-line options;
* product names;
* acronyms;
* identifiers.

`rag-ferrite` combines both approaches instead of choosing only one.

---

## Hybrid search

Hybrid search combines lexical and semantic retrieval.

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

A purely semantic search can sometimes treat these tokens as unimportant or confuse them with related concepts.

Keyword retrieval preserves this precision.

### Vector search

Vector search represents queries and document passages as embeddings.

This makes it possible to retrieve semantically related content even when the wording differs.

For example:

```text
Query:
How can agents access my documentation?

Possible matching passage:
The MCP server exposes indexed knowledge to external AI clients.
```

The exact words are different, but the meaning is related.

### Why combine them?

Neither method is sufficient for every query.

| Query type                  | Keyword search | Vector search |
| --------------------------- | -------------: | ------------: |
| Exact command or identifier |      Excellent |      Variable |
| Error message               |      Excellent |      Variable |
| General concept             |        Limited |     Excellent |
| Paraphrased question        |        Limited |     Excellent |
| Proper name                 |      Excellent |          Good |
| Related explanation         |        Limited |     Excellent |
| Mixed technical query       |           Good |          Good |

Hybrid retrieval improves the probability that the right passage appears in the candidate set.

---

## Reciprocal rank fusion

Keyword and vector searches produce different scores that cannot always be compared directly.

A lexical relevance score and a vector similarity score do not represent the same thing.

`rag-ferrite` uses rank fusion to combine their result lists.

Instead of trusting the raw scores, the fusion process considers the position of each result in each ranking.

A passage that ranks well in both searches receives a stronger combined position.

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

This avoids depending too heavily on one retrieval method.

---

## Reranking

The initial search stage is optimized for recall.

Its job is to find a broad set of potentially useful passages quickly.

However, the first retrieved results are not always ordered perfectly.

A passage may contain many matching terms without truly answering the question. Another may be semantically close but only loosely related.

Reranking adds a second relevance evaluation after retrieval.

```text
Initial retrieval
20 possible passages
        │
        ▼
Detailed relevance evaluation
        │
        ▼
Best passages moved to the top
```

### Why reranking matters

Without reranking, an assistant may receive:

* repeated passages;
* documents that mention the topic only briefly;
* results matching the wording but not the intent;
* semantically similar but practically irrelevant text.

Reranking attempts to answer a more precise question:

> Given this exact user query, which of these retrieved passages are the most useful?

This improves the context eventually sent to the AI assistant.

### Retrieval and reranking have different roles

| Stage             | Objective                                    |
| ----------------- | -------------------------------------------- |
| Hybrid retrieval  | Avoid missing relevant information           |
| Rank fusion       | Combine lexical and semantic candidates      |
| Reranking         | Improve precision and final ordering         |
| Context expansion | Recover surrounding explanations when needed |

---

## Parent-child chunking

Large documents cannot be searched efficiently as a single block.

They must be divided into smaller passages.

Very small chunks improve precision, but they can lose context.

Very large chunks preserve context, but they can make retrieval less precise.

`rag-ferrite` uses a parent-child approach:

```text
Parent section
┌─────────────────────────────────────┐
│ Full topic with broader context     │
│                                     │
│  ┌──────────┐  ┌──────────┐         │
│  │ Child 1  │  │ Child 2  │  ...    │
│  └──────────┘  └──────────┘         │
└─────────────────────────────────────┘
```

Small child chunks are used for precise matching.

Broader parent context can then be returned so the assistant receives a coherent explanation rather than an isolated sentence.

This is useful for:

* technical documentation;
* books;
* long articles;
* research papers;
* architecture documents;
* transcripts.

---

## Context expansion

A search result may identify the correct passage without containing the complete explanation.

The MCP tool `read_chunk_neighbors` allows an assistant to retrieve the chunks before and after a result.

```text
Previous chunk
      │
Matched chunk
      │
Next chunk
```

This allows agents to search precisely first and expand context only when necessary.

It avoids returning very large amounts of text for every query while still making the surrounding explanation available.

---

## Query recovery

A user query is not always written using the terminology found in the documents.

Initial search results may therefore be weak.

`rag-ferrite` can detect weak retrieval and reformulate the query before trying again.

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
* the documents use technical terminology;
* a concept has several names;
* the first query is too broad;
* the original wording is ambiguous.

---

## Automatic tagging and collections

Documents can be organized into collections, while chunks can receive more specific tags.

Collections provide broad separation:

```text
programming
research
personal
projects
documentation
```

Tags provide more precise filtering:

```text
rust
authentication
security
database
mcp
performance
```

An assistant can search broadly or filter results when it knows the relevant topic.

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

## Identifying complementary and conflicting sources

`rag-ferrite` does not decide by itself whether two ideas contradict each other.

Its role is retrieval.

Hybrid and semantic search can surface passages that discuss the same topic using different wording.

An AI assistant can then compare those passages and identify:

* agreements;
* complementary explanations;
* alternative approaches;
* outdated decisions;
* conflicting recommendations;
* differences between sources.

```text
Source A: Use a full index rebuild after every ingestion.
Source B: Incremental insertion avoids expensive rebuilds.
                           │
                           ▼
             AI assistant compares both
```

A simple keyword search may fail to place these passages together if they use different terminology.

Semantic retrieval makes such cross-document comparison more practical.

---

## Features

| Feature                   | Description                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------- |
| **MCP-native**            | Direct integration with Claude Code, Hermes, Claude Desktop, and other MCP clients |
| **Hybrid retrieval**      | Combines full-text and vector search                                               |
| **FTS5 keyword search**   | Retrieves exact terms, identifiers, and error messages                             |
| **Semantic search**       | Finds related concepts and paraphrases                                             |
| **sqlite-vec**            | Local vector retrieval inside SQLite                                               |
| **Rank fusion**           | Combines lexical and semantic rankings                                             |
| **Optional reranking**    | Improves final result precision                                                    |
| **Parent-child chunking** | Balances precise search with broader context                                       |
| **Context expansion**     | Retrieves neighboring passages around a result                                     |
| **Query recovery**        | Reformulates weak queries and retries                                              |
| **Noise filtering**       | Removes low-value and boilerplate chunks                                           |
| **Automatic tagging**     | Adds fine-grained topic metadata                                                   |
| **Collections**           | Organizes documents into broad knowledge domains                                   |
| **Batch ingestion**       | Indexes multiple files asynchronously                                              |
| **Quality checks**        | Inspects documents before indexing                                                  |
| **Retrieval benchmarks**  | Evaluates search against golden datasets                                           |
| **Heat tracking**         | Identifies frequently queried collections and chunks                               |
| **CLI and TUI**           | Manages and monitors the service from a terminal                                   |
| **Local database**         | Uses SQLite without a separate database server                                     |
| **Provider flexibility**  | Supports local and hosted model APIs                                               |
| **Single binary**         | No Python runtime or mandatory container stack                                     |

---

## Supported sources

`rag-ferrite` can ingest:

* Markdown;
* plain text;
* PDF;
* DOCX;
* HTML or Markdown content supplied directly through the API.

Possible document collections include:

```text
~/library/
~/Documents/
~/Projects/*/docs/
~/Notes/
~/Obsidian/Vault/
```

An Obsidian vault works because its notes are Markdown files.

However, Obsidian is only one possible source. The system does not depend on Obsidian and does not require an Obsidian installation.

---

## Common use cases

### Personal documentation

Give an AI assistant access to your own procedures, references, and technical notes.

```text
Search my documentation for the backup restoration procedure.
```

### Coding assistants

Allow Claude Code or another coding agent to retrieve architecture decisions, project conventions, and internal documentation.

```text
Before changing the storage layer, search for previous architecture decisions.
```

### Research library

Search books, articles, papers, and transcripts by meaning rather than only by title or keywords.

```text
Find the sources discussing the limitations of semantic chunking.
```

### Cross-document comparison

Retrieve several passages covering the same subject so an assistant can compare them.

```text
Compare the different recommendations about local vector databases.
```

### Obsidian vault search

Index Markdown notes from an Obsidian vault and make them accessible to MCP clients.

```text
Find my previous notes about authentication, even if they use different terms.
```

### Synthesis and note generation

Use retrieved context to create a report, documentation page, or synthesis note.

```text
Use several sources from the knowledge base to create a structured summary.
```

This is an optional workflow, not a requirement.

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
              │ Rank fusion                 │
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

## Configure model providers

`rag-ferrite` uses:

* an embedding model for vector retrieval;
* an LLM for contextual processing, tagging, query recovery, and optional reranking.

Set the corresponding API keys:

```bash
export LLM_API_KEY="your-llm-api-key"
export EMBEDDING_API_KEY="your-embedding-api-key"
```

Any compatible local or hosted provider can be used.

Minimal configuration:

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
* ingestion should continue after the client closes;
* you want one persistent index shared by all clients.

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

## MCP tools

### Search and reading

| Tool                   | Description                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `query_documents`      | Search indexed documents using hybrid retrieval, filters, query recovery, and reranking |
| `read_chunk_neighbors` | Retrieve passages surrounding a specific result                                         |
| `list_files`           | List indexed source documents                                                           |
| `status`               | Return server and index status                                                          |
| `suggest_collection`   | Suggest the most relevant collection for a query                                        |
| `tag_map`              | Show tags, collections, and chunk counts                                                |

### Ingestion and quality

| Tool              | Description                                        |
| ----------------- | -------------------------------------------------- |
| `ingest_file`     | Ingest a PDF, DOCX, TXT, or Markdown file          |
| `ingest_data`     | Ingest raw text, HTML, or Markdown content         |
| `check_ingestion` | Inspect document quality before indexing           |
| `benchmark`       | Evaluate retrieval against a golden dataset        |
| `collection_heat` | Show frequently and recently queried collections   |
| `chunk_qa`        | Identify cold, unused, or potentially noisy chunks |

### Administration

| Tool                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| `delete_file`         | Remove a document and its chunks                   |
| `reassign_collection` | Move a document to another collection              |
| `rebuild_indexes`     | Rebuild search indexes and checkpoint the database |
| `flush_indexes`       | Persist recently indexed vector data               |

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

Limit results:

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

Return JSON:

```bash
ragfer query "embedding dimensions" --json
```

---

## HTTP API

### Search

```bash
curl -X POST http://localhost:4242/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does hybrid search improve retrieval?",
    "limit": 5
  }'
```

### Ingest one file

```bash
curl -X POST http://localhost:4242/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.md"
  }'
```

### Ingest several files

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

### Monitor ingestion

```bash
curl http://localhost:4242/api/ingest/progress
```

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
| `POST`   | `/api/reload`               | Reload supported configuration   |
| `GET`    | `/api/history`              | Return recent ingestion history  |

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

Provide it to clients with:

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

## What rag-ferrite is not

`rag-ferrite` is not:

* a replacement for Markdown;
* a replacement for Obsidian;
* a complete chat application;
* a hosted AI platform;
* an enterprise document-management system;
* a public multi-tenant RAG service;
* a framework requiring you to assemble the retrieval pipeline yourself.

It is a focused personal knowledge service that gives AI assistants better access to your documents.

---

## When a simple vault is enough

You may not need `rag-ferrite` when:

* you have only a small number of notes;
* filenames and folders are sufficient;
* exact keyword search finds everything you need;
* you do not use AI assistants;
* you rarely search across several documents;
* you already remember the terminology used in your notes.

A simple Markdown collection remains one of the best formats for personal knowledge.

`rag-ferrite` becomes useful when the collection grows and you want assistants to retrieve information by meaning, not only by exact words.

---

## When rag-ferrite is useful

`rag-ferrite` is especially useful when:

* your documentation is spread across many files;
* you use several MCP-compatible assistants;
* you want one shared knowledge base;
* you frequently forget where information was written;
* your queries use different wording from your documents;
* you need both exact technical search and semantic search;
* you compare information across several sources;
* you want a capable RAG without maintaining a large software stack.

---

## Project status

`rag-ferrite` is developed primarily for personal and trusted-network use.

The project prioritizes:

* retrieval quality;
* operational simplicity;
* MCP compatibility;
* local and provider-independent storage;
* maintainability;
* low infrastructure requirements.

Current improvement areas include:

* stronger authentication and permission levels for MCP;
* read-only access profiles;
* more integration tests;
* continuous integration;
* improved source citations;
* watched-folder synchronization;
* additional retrieval benchmarks.

See the GitHub issues for the latest status.

---

## License

MIT