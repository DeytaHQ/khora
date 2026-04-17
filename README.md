# Khora

> *"Khora is the receptacle, the space, the matrix in which all things come to be."* — Plato, *Timaeus*

Khora is a **Memory Lake** library for Python 3.13+. It stores knowledge as a mix of documents, vectors, and graph relationships and retrieves it through hybrid search (vector + graph + keyword), reranking, and temporal context.

Khora is a **library, not an application**. CLI tooling lives in sibling packages:

- [khora-cli](https://github.com/DeytaHQ/khora-cli) — `extract` / `search` commands for ingesting files and querying namespaces.
- [khora-explorer](https://github.com/DeytaHQ/khora-explorer) — ontology construction (`construct` / `validate` / `preview`).

## Install

```bash
pip install khora                 # core (PostgreSQL + pgvector)
pip install khora[surrealdb]      # embedded SurrealDB, zero infrastructure
pip install khora[all-backends]   # Neo4j, Kuzu, SurrealDB, SQLite, Weaviate
```

See [docs/configuration.md](docs/configuration.md) for the full extras list.

## Quickstart

Zero-infrastructure — SurrealDB runs in-process:

```python
import asyncio
from khora import MemoryLake

async def main() -> None:
    async with MemoryLake("memory://") as lake:
        ns = await lake.create_namespace("demo")
        await lake.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903.",
            namespace=ns.namespace_id,
        )
        result = await lake.recall("What did Curie win?", namespace=ns.namespace_id)
        print(result.context_text)

asyncio.run(main())
```

For production (PostgreSQL + pgvector + Neo4j), set `KHORA_DATABASE_URL` and `KHORA_NEO4J_URL`, run `uv run alembic upgrade head`, then instantiate `MemoryLake()` with no arguments. See [docs/](docs/) for details.

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

Copyright (c) 2024-2026 Deyta. All rights reserved.
