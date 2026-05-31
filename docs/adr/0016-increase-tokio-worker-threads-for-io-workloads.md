# 0016: Increase Tokio Worker Threads Beyond CPU Count

## Status
Accepted

## Context
With default tokio worker threads (4, matching CPU count), HTTP queries timed out at 30s during ingestion. The server was running but couldn't process incoming connections.

## Root Cause
During ingestion, parallel LLM+embedding tasks consumed all 4 worker threads. Each `.await` point needs a worker thread to resume on. With all threads occupied by ingestion I/O, no thread was available for HTTP handlers.

## Decision
Set `worker_threads = 12` in `#[tokio::main]`.

## Rationale
Async I/O threads are cheap — they spend most time sleeping on epoll/kqueue waiting for network responses. Having 12 threads for 4 CPUs is fine because they're not doing CPU work. The default `worker_threads = num_cpus` is optimized for CPU-bound workloads, not I/O-heavy servers doing parallel HTTP calls.

## Consequences
- Query response during ingestion: 30s timeout → <1s
- Negligible memory overhead (each thread stack ~8MB)
- Works well with `max_concurrent = 3` for parallel parent processing
