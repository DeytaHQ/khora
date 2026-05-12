# Khora

[![CI](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml)
[![Release](https://github.com/DeytaHQ/khora/actions/workflows/release.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/DeytaHQ/khora/branch/main/graph/badge.svg)](https://codecov.io/gh/DeytaHQ/khora)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> *"Khora is the receptacle, the space, the matrix in which all things come to be."* — Plato, *Timaeus*

Khora is a knowledge memory library for long-horizon AI agents, with pluggable retrieval engines and storage backends to fit different workloads. It stores knowledge as documents, embeddings, and graph relationships, and retrieves it through hybrid search (vector + graph + keyword), reranking, and temporal context.

Khora is a **library, not an application**. Tooling lives in sibling packages (coming soon...):

- khora-cli (to be released soon) — CLI tooling for extraction and search.
- khora-explorer (to be released soon) — tooling for ontology construction and exploration.

## Install

```bash
pip install khora                 # core (PostgreSQL + pgvector)
pip install khora[sqlite-lance]   # [experimental] embedded SQLite + LanceDB
pip install khora[surrealdb]      # [experimental] unified SurrealDB (single store)
pip install khora[all-backends]   # everything: Neo4j, SurrealDB, SQLite+LanceDB, Weaviate, AGE
```

See [docs/configuration.md](docs/configuration.md) for the full extras list.

## Production stack

The production-ready combination in v0.9.0 is **PostgreSQL + pgvector + Neo4j**:

- **VectorCypher** (default engine) — runs on PostgreSQL + pgvector + Neo4j.
- **Chronicle** — runs on PostgreSQL + pgvector (no graph DB required).
- **Skeleton** — available; PostgreSQL + pgvector (no graph DB required).

Set `KHORA_DATABASE_URL` and `KHORA_NEO4J_URL`, run `uv run alembic upgrade head`, then instantiate `Khora()` with no arguments:

```python
import asyncio
from khora import Khora

async def main() -> None:
    async with Khora() as kb:  # reads KHORA_DATABASE_URL / KHORA_NEO4J_URL
        ns = await kb.create_namespace("demo")
        await kb.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903.",
            namespace=ns.namespace_id,
        )
        result = await kb.recall("What did Curie win?", namespace=ns.namespace_id)
        print(result.context_text)

asyncio.run(main())
```

## Batch processing

`submit_batch()` stages documents as PENDING and returns a `BatchHandle` immediately. A background processor picks them up and calls `on_result` per document as each completes.

**The processor is opt-in.** Call `kb.start_pending_processor()` after `connect()` on services that write documents. Read-only services do not need it. The processor can be stopped with `await kb.stop_pending_processor()` and restarted at any time.

```python
async with Khora() as kb:
    kb.start_pending_processor()   # opt-in; write-path services only
    handle = await kb.submit_batch(
        [{"content": "doc 1"}, {"content": "doc 2"}],
        on_result=lambda completed, total, result: print(result),
        namespace=ns_id,
    )
    await handle.wait()
```

## Embedded options (experimental)

Khora ships two zero-infrastructure paths. Both are marked **experimental** in v0.9.0 — fine for demos, evaluation, tests, and small single-user CLIs; not yet stamped as a deployment story.

- **SQLite + LanceDB** (`pip install khora[sqlite-lance]`, set `KHORA_STORAGE_BACKEND=sqlite_lance`) — recommended embedded stack. Covers VectorCypher, Skeleton, and Chronicle via dialect-aware Alembic migrations and LanceDB-backed vector search. Documented scale ceiling: **~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3**. Known gaps: no point-in-time queries (DYT-3550), partial atomicity in `coordinator.transaction()`, FTS on chunks only. See [configuration.md](docs/configuration.md#embedded-backends-experimental).
- **SurrealDB** (`pip install khora[surrealdb]`) — unified relational + vector + graph in one store. Python SDK is on the alpha track (`>=2.0.0a1`), and KNN (`<|K|>`) is unreliable in embedded mode (uses brute-force cosine + HNSW fallback). Suitable for experimentation; not recommended for production.

> **Quickstart caveat.** A literal `Khora("memory://")` call passes `"memory://"` as the PostgreSQL URL, not as a backend selector — there is no `memory://` URL scheme parsed by khora itself today. To use the embedded path, set `KHORA_STORAGE_BACKEND=sqlite_lance` (or `surrealdb`) and the corresponding `db_path` / connection settings. Routing a true `memory://` URI to the SQLite+LanceDB stack is tracked for v0.10.

## Observability

khora emits OpenTelemetry spans and metrics via [Logfire](https://logfire.pydantic.dev/) and records structured `LLMEvent` / `StorageEvent` / `PipelineEvent` rows to PostgreSQL when a collector is configured. Both integrations are opt-in — without them, all instrumentation is a zero-cost no-op.

- **Public surface is documented in [`docs/telemetry-contract.json`](docs/telemetry-contract.json)** (with explainer at [`docs/telemetry-contract.md`](docs/telemetry-contract.md)). It lists every public span, metric, pipeline stage, event-type field, and `khora.telemetry.__all__` export. Items tagged `stability: public` are part of khora's API surface and follow standard semver — breaking changes require a major version bump. Drift is enforced in CI via `tests/unit/telemetry/test_contract.py`.
- **OTel semantic conventions** apply to attributes: `gen_ai.*` for LLM calls, `db.*` for storage, `code.*` for stack info. Vendor-neutral over the OTel exporter chain.
- **Logfire integration is opt-in via the `[logfire]` extra:**

  ```bash
  pip install khora[logfire]
  ```

  ```python
  import logfire
  from khora import Khora

  logfire.configure(service_name="my-service")
  # khora's @trace decorators and trace_span() context managers
  # now emit spans automatically; metrics like khora.memory.recall.duration,
  # khora.llm.tokens, khora.llm.cost_usd, khora.chronicle.abstention_signal
  # are exported on the standard OTel cadence.
  ```

  Without the `logfire` extra installed, `trace_span()` yields a no-op and `metric_*` registrations short-circuit.
- **Structured event recording is opt-in via `KHORA_TELEMETRY_DATABASE_URL`** (PostgreSQL). When set, `TelemetryCollector` writes `LLMEvent` / `StorageEvent` / `PipelineEvent` rows for downstream cost tracking and incident reconstruction. Without it, `NoOpCollector` is used (zero cost).
- **Async logging caveat.** Library consumers that import khora without configuring loguru sinks inherit the default sync stderr sink, which blocks the event loop on every log call inside `async def`. Either call `khora.logging_config.setup_logging()` (which configures sinks with `enqueue=True` and registers an `atexit` drain) or configure your own loguru sinks with `enqueue=True` explicitly.

## Documentation

Start at [docs/README.md](docs/README.md). Key entry points:

- [API reference](docs/api-reference.md) — public `Khora` surface.
- [Configuration](docs/configuration.md) — `KHORA_*` env vars and `KhoraConfig`.
- [Architecture](docs/architecture/overview.md) — how the pieces fit.
- [Engines](docs/engines/engine-comparison.md) — VectorCypher, Skeleton, Chronicle.
- [Migrations](docs/migrations.md) — Alembic workflow for library users.
- [Downstream consumers](docs/consumers.md) — sibling packages and integration guide.

## Development

```bash
make dev         # start PostgreSQL + Neo4j (Docker)
make test        # pytest with coverage
make format      # ruff format + isort
make lint        # ruff + ty typecheck
```

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Copyright 2026 AllTheData Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
