# Khora

[![CI](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/ci.yml)
[![Release](https://github.com/DeytaHQ/khora/actions/workflows/release.yml/badge.svg)](https://github.com/DeytaHQ/khora/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/DeytaHQ/khora/branch/main/graph/badge.svg)](https://codecov.io/gh/DeytaHQ/khora)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Python library for creating knowledge repositories that ingest unstructured and structured multi-source data and expose a single query substrate, built for integrating into long-horizon AI agents.

## Quickstart

Two supported stacks. Pick by deployment shape - the public API is identical for both.

```bash
uv add khora              # Docker: PostgreSQL + pgvector + Neo4j
uv add khora[embedded]    # Embedded: SQLite + LanceDB, no external services
```

```python
import asyncio
from khora import Khora, context_text

async def main() -> None:
    async with Khora() as kb:  # reads KHORA_DATABASE_URL / KHORA_NEO4J_URL
        ns = await kb.create_namespace()
        await kb.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903.",
            namespace=ns.namespace_id,
            entity_types=["PERSON", "AWARD"],
            relationship_types=["WON"],
        )
        result = await kb.recall("What did Curie win?", namespace=ns.namespace_id)
        print(context_text(result))

asyncio.run(main())
```

For the Docker stack, set `KHORA_DATABASE_URL` and `KHORA_NEO4J_URL`, run `uv run alembic upgrade head`, then run the snippet above. For the embedded stack, set `KHORA_STORAGE_BACKEND=sqlite_lance` and the corresponding `db_path` - no external services required.

See the [docs](https://docs.deyta.ai/khora) for the full extras list and env-var reference.

## Why khora?

Knowledge repositories for long-horizon agents - copilots, customer-support bots, research assistants - hit two problems that pure vector search doesn't solve:

1. **Ingest is more than chunking.** A useful repository needs entities, relationships, and temporal anchors extracted from the raw text. Khora runs a 3-phase ingest pipeline (stage → enrich → expand) with selective LLM extraction (default 70% of chunks, configurable) and cross-batch entity resolution.
2. **Retrieval is more than cosine.** Real queries mix semantic similarity, multi-hop entity reasoning, freshness, and keyword precision. Khora combines vector + Cypher graph traversal + BM25 + RRF fusion + temporal-anchored reranking, routed per query.

## Storage stacks

| Stack | Install | Use when |
|---|---|---|
| **PostgreSQL + pgvector + Neo4j** | `uv add khora` | Default. Docker-friendly, multi-tenant ready, scales horizontally. |
| **SQLite + LanceDB** (experimental) | `uv add khora[embedded]` | Demos, evaluation, tests, single-user CLIs. Documented ceiling: ~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3. Known gaps: no point-in-time queries, partial atomicity in `coordinator.transaction()`, FTS on chunks only. |

## Retrieval engines

Khora's retrieval layer is pluggable. More than one engine can sit on the same storage substrate. The default is **VectorCypher**: it handles structured and unstructured data from multiple sources by combining a knowledge graph with a vector store, then dispatching each query to the best retrieval method via a query-aware router.

The router selects between vector similarity, graph traversal, BM25 keyword, or a fused blend. RRF fusion, optional PPR (Personalized PageRank), and optional cross-encoder reranking shape the result set. Selective per-chunk LLM extraction (KET-RAG style) bounds ingest cost.

## Batch processing

`submit_batch()` stages documents as PENDING and returns a `BatchHandle` immediately. A background processor picks them up and calls `on_result` per document as each completes.

The processor is opt-in - call `kb.start_pending_processor()` after `connect()` on services that write documents. Read-only services do not need it.

```python
async with Khora() as kb:
    kb.start_pending_processor()
    handle = await kb.submit_batch(
        [{"content": "doc 1"}, {"content": "doc 2"}],
        on_result=lambda completed, total, result: print(result),
        namespace=ns_id,
        entity_types=["PERSON", "ORG"],
        relationship_types=["WORKS_AT"],
    )
    await handle.wait()
```

## Integrations

Opt-in adapters for the major agentic frameworks. Each adapter is in its own extra; the framework is imported lazily so `import khora` never pulls in a framework you don't use.

| Framework | Install | Khora surface |
|---|---|---|
| [CrewAI](https://docs.deyta.ai/khora/integrations/crewai) | `uv add khora[crewai]` | `KhoraMemory` - drop-in storage backend for CrewAI's unified `Memory`. |
| [LangGraph](https://docs.deyta.ai/khora/integrations/langgraph) | `uv add khora[langgraph]` | `KhoraStore` - `BaseStore` implementation for `StateGraph` semantic long-term memory. |
| [Google ADK](https://docs.deyta.ai/khora/integrations/google_adk) | `uv add khora[google-adk]` | `KhoraMemoryService` - `BaseMemoryService` drop-in for ADK `Runner`. |
| [OpenAI Agents SDK](https://docs.deyta.ai/khora/integrations/openai_agents) | `uv add khora[openai-agents]` | `KhoraSession`, `khora_recall_tool`, `KhoraMemoryHooks`. |
| [LlamaIndex](https://docs.deyta.ai/khora/integrations/llamaindex) | `uv add khora[llamaindex]` | `KhoraRetriever`, `KhoraMemoryBlock`. |

See the [docs](https://docs.deyta.ai/khora) for per-adapter guides and the "write your own" Protocol surface.

## Examples

Runnable demos under `examples/`:

- **`00_quickstart/`** - remember + recall, grounded answers, forget, namespaces.
- **`10_core_apis/`** - batch ingest, recall filters, ontology config, entities + relationships, graph traversal.
- **`20_integrations/`** - LangGraph, OpenAI Agents SDK, CrewAI.
- **`30_workloads/`** - per-user preferences with temporal decay, document Q&A with multi-signal abstention, support-ticket knowledge graphs, agent conversation history, namespace versioning, resume search with cross-document entity resolution.

Run any demo from the repo root, e.g. `uv run python examples/30_workloads/01_per_user_preferences.py`. The embedded backend (`examples/khora.embedded.yaml`) needs no external services; pass `--config examples/khora.standard.yaml` to target PostgreSQL + Neo4j.

## Rust acceleration (optional)

Khora ships an optional Rust extension (`khora-accel`) that speeds up MMR, cosine similarity, PageRank, entity resolution, community detection, and temporal operators. Pure-Python fallbacks ship in the base package; the Rust path is opt-in.

```bash
uv add khora[rust]
```

Prebuilt wheels cover the common platforms (macOS arm64/x86_64, Linux x86_64/aarch64, Windows x86_64), so most users won't need a toolchain. Building from source requires **Rust 1.85+**.

## Observability

Khora emits OpenTelemetry spans and metrics through the OTel API. Export path is your choice: vanilla OTel SDK (`uv add khora[otel]`), [Logfire](https://logfire.pydantic.dev/) (`uv add khora[logfire]`), or nothing (zero-cost no-op). Credential fields on `KhoraConfig` are `pydantic.SecretStr`; free-text never leaks into span attributes.

See the [docs](https://docs.deyta.ai/khora) for the env-var contract, vendor recipes, and the telemetry surface reference.

## Documentation

Full documentation at **[docs.deyta.ai/khora](https://docs.deyta.ai/khora)** - API reference, configuration, architecture, migrations, integrations, and the downstream-consumer guide.

## Development

```bash
make dev         # start PostgreSQL + Neo4j (Docker); alias: make db-up
make test        # pytest with coverage
make format      # ruff format + isort
make lint        # ruff + ty typecheck
```

See [CHANGELOG.md](CHANGELOG.md) for release history.

## The name

> *"Khora is the receptacle, the space, the matrix in which all things come to be."* - Plato, *Timaeus*

Khora (χώρα) is Plato's term for the receptacle that holds and gives place to everything that comes into being - a fitting name for a substrate that takes in what an agent learns and makes it retrievable.

## License

Copyright 2026 AllTheData Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
