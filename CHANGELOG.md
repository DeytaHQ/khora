# Changelog

## [0.2.1] - 2026-02-07

### Performance

- **Doubled all LLM and embedding concurrency defaults** across the entire stack for higher throughput during ingestion and extraction.

#### Khora concurrency changes (10 source files)

| File | Parameter | Old | New |
|------|-----------|-----|-----|
| `src/khora/config/llm.py` | `max_concurrent_llm_calls` | 10 | 20 |
| `src/khora/config/llm.py` | `embed_concurrency` | 25 | 50 |
| `src/khora/extraction/extractors/llm.py` | `max_concurrent` | 5 | 10 |
| `src/khora/extraction/embedders/litellm.py` | `batch_size` | 100 | 200 |
| `src/khora/extraction/embedders/litellm.py` | `embed_concurrency` | 10 | 20 |
| `src/khora/pipelines/flows/ingest.py` | `max_concurrent_extractions` | 10 | 20 |
| `src/khora/pipelines/flows/ingest.py` | `embedding_batch_size` | 50 | 100 |
| `src/khora/pipelines/flows/ingest.py` | `max_concurrent_documents` | 5 | 10 |
| `src/khora/pipelines/flows/expansion.py` | entity semaphores (x2) | 20 | 40 |
| `src/khora/memory_lake.py` | `max_concurrent` (remember_batch) | 5 | 10 |
| `src/khora/memory_lake.py` | `max_concurrent` (remember_batch_legacy) | 5 | 10 |
| `src/khora/engines/protocol.py` | `max_concurrent` | 5 | 10 |
| `src/khora/engines/graphrag/engine.py` | `max_concurrent` | 5 | 10 |
| `src/khora/engines/skeleton/engine.py` | `max_concurrent` | 10 | 20 |
| `src/khora/engines/vectorcypher/engine.py` | `max_concurrent` | 10 | 20 |

#### Genesis concurrency changes (8 files)

| File | Parameter | Old | New |
|------|-----------|-----|-----|
| `src/genesis/config.py` | `max_concurrent_llm_calls` | 20 | 40 |
| `src/genesis/config.py` | `embed_concurrency` | 10 | 20 |
| `src/genesis/__init__.py` | CLI `--batch-size` default | 10 | 20 |
| `config/graphrag/litellm.yaml` | `max_concurrent_llm_calls` | 100 | 200 |
| `config/graphrag/litellm.yaml` | `embed_concurrency` | 100 | 200 |
| `config/skeleton/litellm.yaml` | `max_concurrent_llm_calls` | 100 | 200 |
| `config/skeleton/litellm.yaml` | `embed_concurrency` | 100 | 200 |
| `config/vectorcypher/litellm.yaml` | `max_concurrent_llm_calls` | 100 | 200 |
| `config/vectorcypher/litellm.yaml` | `embed_concurrency` | 100 | 200 |
| `config/graphrag/genesis.yaml` | `max_concurrent_documents` | 50 | 100 |
| `config/graphrag/genesis.yaml` | `max_concurrent_chunks` | 100 | 200 |
| `config/skeleton/genesis.yaml` | `max_concurrent_documents` | 50 | 100 |
| `config/skeleton/genesis.yaml` | `max_concurrent_chunks` | 100 | 200 |
| `config/vectorcypher/genesis.yaml` | `max_concurrent_documents` | 50 | 100 |
| `config/vectorcypher/genesis.yaml` | `max_concurrent_chunks` | 100 | 200 |

### Chores

- **Removed REPOMIX** — deleted `REPOMIX.md`, `repomix.config.json`, `scripts/update_repomix.py`, and the `update-repomix` pre-commit hook. Removed `REPOMIX.md` exclusion patterns from `trailing-whitespace`, `end-of-file-fixer`, and `check-added-large-files` hooks.
- **Removed stale planning docs** — deleted `docs/OPTIMIZATION_PLAN.md` and `docs/RUST_ACCELERATION_PLAN.md` (completed work, no longer needed).
- **Excluded `docs/REFERENCES.md` from version control** — kept locally for reference, added to `.gitignore`.
- **Bumped version** from `0.2.0` to `0.2.1`.

## [0.2.0] - 2026-02-07

### Features

- **Rust acceleration layer** (`khora-accel`) for CPU-intensive operations — entity resolution, BM25 search, PageRank, cosine similarity, and RRF fusion via PyO3/maturin with 3-tier fallback (Rust, numpy/rapidfuzz, pure Python).
- Log active acceleration backend on import.

### Chores

- Improved upsert result mismatch diagnostics.
- Downgraded extraction log to debug level.
