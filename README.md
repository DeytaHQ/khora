# Khora

[![CI](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml)
[![Release](https://github.com/DeytaHQ/khora/actions/workflows/release.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/DeytaHQ/khora/branch/main/graph/badge.svg)](https://codecov.io/gh/DeytaHQ/khora)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> *"Khora is the receptacle, the space, the matrix in which all things come to be."* — Plato, *Timaeus*

Khora is a **Memory Lake** library for Python 3.13+. It stores knowledge as a mix of documents, vectors, and graph relationships and retrieves it through hybrid search (vector + graph + keyword), reranking, and temporal context.

Khora is a **library, not an application**. CLI tooling lives in sibling packages:

- [khora-cli](https://github.com/DeytaHQ/khora-cli) — `extract` / `search` commands for ingesting files and querying namespaces.
- [khora-explorer](https://github.com/DeytaHQ/khora-explorer) — ontology construction (`construct` / `validate` / `preview`).

## Install

```bash
pip install khora                 # core (PostgreSQL + pgvector)
pip install khora[sqlite-lance]   # [experimental] embedded SQLite + LanceDB
pip install khora[surrealdb]      # [experimental] unified SurrealDB (single store)
pip install khora[all-backends]   # everything: Neo4j, SurrealDB, SQLite+LanceDB, Weaviate, AGE
```

See [docs/configuration.md](docs/configuration.md) for the full extras list. The `kuzu` extra is **deprecated in 0.9.0** and scheduled for removal in 0.10.

## Production stack

The production-ready combination in v0.9.0 is **PostgreSQL + pgvector + Neo4j**:

- **VectorCypher** (default engine) — runs on PostgreSQL + pgvector + Neo4j.
- **Chronicle** — runs on PostgreSQL + pgvector (no graph DB required).
- **GraphRAG** and **Skeleton** — available; same PG+Neo4j (or PG-only for Skeleton) shape.

Set `KHORA_DATABASE_URL` and `KHORA_NEO4J_URL`, run `uv run alembic upgrade head`, then instantiate `MemoryLake()` with no arguments:

```python
import asyncio
from khora import MemoryLake

async def main() -> None:
    async with MemoryLake() as lake:  # reads KHORA_DATABASE_URL / KHORA_NEO4J_URL
        ns = await lake.create_namespace("demo")
        await lake.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903.",
            namespace=ns.namespace_id,
        )
        result = await lake.recall("What did Curie win?", namespace=ns.namespace_id)
        print(result.context_text)

asyncio.run(main())
```

## Embedded options (experimental)

Khora ships two zero-infrastructure paths. Both are marked **experimental** in v0.9.0 — fine for demos, evaluation, tests, and small single-user CLIs; not yet stamped as a deployment story.

- **SQLite + LanceDB** (`pip install khora[sqlite-lance]`, set `KHORA_STORAGE_BACKEND=sqlite_lance`) — recommended embedded stack. Covers VectorCypher, GraphRAG, Skeleton, and Chronicle via dialect-aware Alembic migrations and LanceDB-backed vector search. Documented scale ceiling: **~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3**. Known gaps: no point-in-time queries (DYT-3550), partial atomicity in `coordinator.transaction()`, FTS on chunks only. See [configuration.md](docs/configuration.md#embedded-backends-experimental).
- **SurrealDB** (`pip install khora[surrealdb]`) — unified relational + vector + graph in one store. Python SDK is on the alpha track (`>=2.0.0a1`), and KNN (`<|K|>`) is unreliable in embedded mode (uses brute-force cosine + HNSW fallback). Suitable for experimentation; not recommended for production.

> **Quickstart caveat.** A literal `MemoryLake("memory://")` call passes `"memory://"` as the PostgreSQL URL, not as a backend selector — there is no `memory://` URL scheme parsed by the lake itself today. To use the embedded path, set `KHORA_STORAGE_BACKEND=sqlite_lance` (or `surrealdb`) and the corresponding `db_path` / connection settings. Routing a true `memory://` URI to the SQLite+LanceDB stack is tracked for v0.10.

## Documentation

Start at [docs/README.md](docs/README.md). Key entry points:

- [API reference](docs/api-reference.md) — public `MemoryLake` surface (ADR-024).
- [Configuration](docs/configuration.md) — `KHORA_*` env vars and `KhoraConfig`.
- [Architecture](docs/architecture/overview.md) — how the pieces fit.
- [Engines](docs/engines/engine-comparison.md) — VectorCypher, GraphRAG, Skeleton, Chronicle.
- [Migrations](docs/migrations.md) — Alembic workflow for library users.
- [Downstream consumers](docs/consumers.md) — how genesis, khora-cli, khora-explorer, khora-benchmarks consume khora.

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
