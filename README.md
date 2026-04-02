# Khora

> *"Khora is the receptacle, the space, the matrix in which all things come to be."* — Plato, *Timaeus*

Khora is a **Memory Lake** — a Python library that stores knowledge as graphs, vectors, and relational data, and retrieves it through semantic search, graph traversal, and temporal context.

```python
from khora import MemoryLake

async with MemoryLake("memory://") as lake:                     # embedded SurrealDB, zero infra
    ns = await lake.create_namespace("demo")
    await lake.remember("Einstein developed relativity in 1905.", namespace=ns.namespace_id)
    result = await lake.recall("Who developed relativity?", namespace=ns.namespace_id)
    print(result.context_text)
```

## Installation

```bash
pip install khora                   # core (PostgreSQL + pgvector)
pip install khora[surrealdb]        # embedded SurrealDB (zero infrastructure)
pip install khora[lancedb]          # embedded LanceDB vector store
pip install khora[all]              # everything
```

**Requirements:** Python 3.13+

## Getting Started

### Quickest path: embedded SurrealDB

No Docker, no databases — SurrealDB runs in-process:

```python
import asyncio
from khora import MemoryLake

async def main():
    async with MemoryLake("memory://", engine="skeleton") as lake:
        ns = await lake.create_namespace("quickstart")

        # Store knowledge
        result = await lake.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903 "
            "and the Nobel Prize in Chemistry in 1911.",
            namespace=ns.namespace_id,
        )
        print(f"Stored: {result.chunks_created} chunks, {result.entities_extracted} entities")

        # Retrieve knowledge
        answer = await lake.recall("What prizes did Curie win?", namespace=ns.namespace_id)
        print(answer.context_text)

asyncio.run(main())
```

### Recommended: PostgreSQL + pgvector + Neo4j

For production workloads with full knowledge graph support:

```bash
# Start databases
make dev  # ports: PostgreSQL=5434, Neo4j Bolt=7688

# Run migrations
uv run alembic upgrade head
```

```bash
# .env
KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora
KHORA_NEO4J_URL=bolt://neo4j:pleaseletmein@localhost:7688
OPENAI_API_KEY=sk-...          # for embeddings (text-embedding-3-small)
```

```python
import asyncio
from khora import MemoryLake

async def main():
    async with MemoryLake(engine="graphrag", run_migrations=True) as lake:
        ns = await lake.create_namespace("research")

        await lake.remember(
            "Einstein developed relativity at the patent office in Bern.",
            namespace=ns.namespace_id,
        )

        # Hybrid search: vector + graph + keyword with RRF fusion
        result = await lake.recall("Where did Einstein work?", namespace=ns.namespace_id)
        print(result.context_text)

        # Explore the knowledge graph
        entities = await lake.list_entities(namespace=ns.namespace_id, entity_type="PERSON")
        for entity in entities:
            related = await lake.find_related_entities(entity.id)
            print(f"{entity.name}: {len(related)} connections")

asyncio.run(main())
```

## Pluggable Engines

| Engine | Best For | Graph DB | LLM Cost |
|--------|----------|----------|----------|
| **[GraphRAG](docs/engines/engine-comparison.md)** (default) | Knowledge bases, entity exploration | Required | Higher |
| **[VectorCypher](docs/engines/vectorcypher-engine.md)** | Multi-hop queries, complex relationships | Required | Medium |
| **[Skeleton](docs/engines/skeleton-engine.md)** | Chat logs, events, cost-sensitive apps | Optional | Lower |
| **[Chronicle](docs/engines/chronicle-engine.md)** | Temporal memory, conversational recall | Not needed | Medium |

Select an engine at initialization:

```python
async with MemoryLake("memory://", engine="chronicle") as lake:
    ...
```

See [Engine Comparison](docs/engines/engine-comparison.md) for detailed guidance. Engines are pluggable — you can also [register custom engines](docs/engines/engine-comparison.md#custom-engines).

## Key Features

- **`remember()` / `recall()` / `forget()`** — simple API for storing and retrieving knowledge
- **Hybrid Search** — vector + graph + keyword with [Reciprocal Rank Fusion](docs/query-engine/fusion.md)
- **4 Engines** — GraphRAG, VectorCypher, Skeleton, Chronicle
- **Multi-Tenancy** — namespace-level isolation
- **Event Sourcing** — immutable event log for audit trails
- **Semantic Hooks** — subscribe to extraction events with [3-level semantic filtering](docs/hooks/semantic-hooks.md)
- **3-Phase Ingestion** — stage, enrich, expand with [domain expertise](docs/extraction/expertise-system.md)
- **Rust Acceleration** — optional native extensions for cosine, PageRank, entity resolution
- **LiteLLM** — unified access to OpenAI, Anthropic, Google, and other providers

## Configuration

All settings use the `KHORA_` prefix (e.g., `KHORA_LLM_MODEL=gpt-4o`).

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_DATABASE_URL` | PostgreSQL or SurrealDB URL | Required |
| `KHORA_NEO4J_URL` | Neo4j connection URL | — |
| `KHORA_STORAGE_BACKEND` | `postgres` or `surrealdb` | `postgres` |
| `KHORA_LLM_MODEL` | Primary LLM model | `gpt-4o-mini` |
| `KHORA_LLM_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `KHORA_QUERY_DEFAULT_MODE` | Search mode: `vector`, `graph`, `hybrid` | `hybrid` |
| `KHORA_QUERY_ENABLE_HYDE` | HyDE query expansion: `auto`, `always`, `never` | `auto` |
| `KHORA_DEBUG` | Debug logging | `false` |

See [full configuration reference](docs/architecture/storage-backends.md) for all options.

## Documentation

| Topic | Pages |
|-------|-------|
| **[Engines](docs/engines/)** | [Comparison](docs/engines/engine-comparison.md) · [GraphRAG](docs/engines/engine-comparison.md#graphrag) · [VectorCypher](docs/engines/vectorcypher-engine.md) · [Skeleton](docs/engines/skeleton-engine.md) · [Chronicle](docs/engines/chronicle-engine.md) |
| **[Architecture](docs/architecture/)** | [Overview](docs/architecture/overview.md) · [Storage Backends](docs/architecture/storage-backends.md) · [Multi-Tenancy](docs/architecture/multi-tenancy.md) · [Event Sourcing](docs/architecture/event-sourcing.md) |
| **[Extraction](docs/extraction/)** | [Pipeline](docs/extraction/ingestion-pipeline.md) · [Chunkers](docs/extraction/chunkers.md) · [Expertise](docs/extraction/expertise-system.md) · [Expansion](docs/extraction/semantic-expansion.md) |
| **[Query Engine](docs/query-engine/)** | [Overview](docs/query-engine/overview.md) · [Search Modes](docs/query-engine/search-modes.md) · [Fusion](docs/query-engine/fusion.md) · [Temporal](docs/query-engine/temporal-queries.md) |
| **[Performance](docs/architecture/)** | [Rust Acceleration](docs/architecture/rust-acceleration.md) · [Optimization](docs/architecture/performance-optimization.md) |
| **[Hooks](docs/hooks/)** | [Semantic Hooks](docs/hooks/semantic-hooks.md) |

## Development

```bash
make test              # pytest with coverage
make format            # black + isort + ruff
make lint              # ruff + ty typecheck
make dev               # start PostgreSQL + Neo4j (Docker)
make down              # stop databases
```

```bash
uv run alembic upgrade head                       # run migrations
uv run alembic revision --autogenerate -m "desc"  # create migration
```

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Copyright (c) 2024-2026 Deyta. All rights reserved.
