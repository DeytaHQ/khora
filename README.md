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

The production stack is **PostgreSQL + pgvector + Neo4j**:

- **VectorCypher** (default engine) — runs on PostgreSQL + pgvector + Neo4j.
- **Chronicle** — runs on PostgreSQL + pgvector (no graph DB required).
- **Skeleton** — available; PostgreSQL + pgvector (no graph DB required).

Set `KHORA_DATABASE_URL` and `KHORA_NEO4J_URL`, run `uv run alembic upgrade head`, then instantiate `Khora()` with no arguments:

```python
import asyncio
from khora import Khora

async def main() -> None:
    async with Khora() as kb:  # reads KHORA_DATABASE_URL / KHORA_NEO4J_URL
        ns = await kb.create_namespace()  # keyword-only kwargs; no positional name
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

Khora ships two zero-infrastructure paths. Both are marked **experimental** — fine for demos, evaluation, tests, and small single-user CLIs; not yet stamped as a deployment story.

- **SQLite + LanceDB** (`pip install khora[sqlite-lance]`, set `KHORA_STORAGE_BACKEND=sqlite_lance`) — recommended embedded stack. Covers VectorCypher, Skeleton, and Chronicle via dialect-aware Alembic migrations and LanceDB-backed vector search. Documented scale ceiling: **~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3**. Known gaps: no point-in-time queries, partial atomicity in `coordinator.transaction()`, FTS on chunks only. See [configuration.md](docs/configuration.md#embedded-backends-experimental).
- **SurrealDB** (`pip install khora[surrealdb]`) — unified relational + vector + graph in one store. Python SDK is on the alpha track (`>=2.0.0a1`), and KNN (`<|K|>`) is unreliable in embedded mode (uses brute-force cosine + HNSW fallback). Remote (WebSocket) mode supports atomic multi-statement transactions via `conn.transaction()` (v0.12.0); embedded / memory modes still operate per-statement-atomic. Suitable for experimentation; not recommended for production.

> **Quickstart caveat.** A literal `Khora("memory://")` call passes `"memory://"` as the PostgreSQL URL, not as a backend selector — there is no `memory://` URL scheme parsed by khora itself today. To use the embedded path, set `KHORA_STORAGE_BACKEND=sqlite_lance` (or `surrealdb`) and the corresponding `db_path` / connection settings.

## Integrations

khora ships ready-made adapters for the major agentic frameworks. Each adapter is an opt-in optional extra — install only what you use, and the framework itself is imported lazily so importing `khora` never pulls in a framework you don't need.

| Framework | Install | Khora surface |
|---|---|---|
| [CrewAI](docs/integrations/crewai.md) | `pip install khora[crewai]` | `KhoraMemory` — drop-in storage backend for CrewAI's unified `Memory`. |
| [LangGraph](docs/integrations/langgraph.md) | `pip install khora[langgraph]` | `KhoraStore` — `BaseStore` implementation for `StateGraph` semantic long-term memory. |
| [Google ADK](docs/integrations/google_adk.md) | `pip install khora[google-adk]` | `KhoraMemoryService` — `BaseMemoryService` drop-in for ADK `Runner`. |
| [OpenAI Agents SDK](docs/integrations/openai_agents.md) | `pip install khora[openai-agents]` | `KhoraSession` (`SessionABC`), `khora_recall_tool`, `KhoraMemoryHooks` — compose for session memory, recall-as-tool, and auto-persist. |
| [LlamaIndex](docs/integrations/llamaindex.md) | `pip install khora[llamaindex]` | `KhoraRetriever` (async `BaseRetriever`), `KhoraMemoryBlock`, and the deprecated `KhoraChatStore`. |

See [docs/integrations/](docs/integrations/index.md) for the full per-adapter docs and the "write your own" Protocol surface.

## Maintenance: dream phase

khora ships an **offline maintenance pass** ("dream phase") that audits an accumulated namespace and plans consolidation work — entity dedupe, fact compaction, event clustering. Run it on a schedule (cron, Temporal, k8s CronJob) and consume the structured reports through three independently-togglable sinks: file, semantic-event, or telemetry collector.

```python
from khora import Khora, KhoraConfig, DreamConfig

kb = Khora(config=KhoraConfig(dream=DreamConfig(enabled=True)))

# Dry-run — pure observation/planning, zero writes.
result = await kb.dream(namespace_id, mode="dry-run")

for op in result.ops:
    print(op.op_type, op.decision, op.outputs)
```

Ten operations ship in v0.15.0 across both engines:

| Phase | Engine | Operation | Surfaces |
|---|---|---|---|
| 1 audit | chronicle | abstention-threshold drift | Configured thresholds vs observed p50/p90/p99 |
| 1 audit | chronicle | tombstone audit | Active / inactive / invalidated fact ratios + age distribution |
| 1 audit | vectorcypher | schema drift | New / unused / frequency-changed types vs `ExpertiseConfig` |
| 1 audit | vectorcypher | orphan PageRank | Bottom-5% PR entities flagged as `archive_candidate` |
| 1 audit | vectorcypher | source_chunk_ids audit | Dead UUID counts + array-length distribution |
| 2 planner | vectorcypher | cross-batch entity dedupe | Pairs above the per-type cosine threshold, planned merges |
| 2 planner | vectorcypher | centroid recompute | Per-cluster `centroid` / `re_embed` / `skip_multimodal` decisions |
| 2 planner | vectorcypher | source_chunk_ids GC | Per-entity rewrites dropping dead chunk UUIDs |
| 2 planner | chronicle | memory_facts compaction | Tombstoned rows past `retention_days` |
| 2 planner | chronicle | event clustering | Near-duplicate `chronicle_events` within a sliding window |

Default is `DreamConfig(enabled=False)` — the master switch is opt-in. **Both modes are live in v0.15.0**: `mode="dry-run"` emits the plan only; `mode="apply"` runs the matching apply handler under a per-op transaction with the pre-state snapshotted into `undo.json` before each mutation. Five guardrails protect the apply path (7-day hard retention floor, `KHORA_DREAM_DISABLE_APPLY` kill-switch, chunk_id runtime assertion, snapshot-before-delete, advisory-lock-held-through-apply).

See [docs/dream-phase.md](docs/dream-phase.md) for the full operator guide: research lineage, configuration surface, sink wiring, telemetry contract, storage substrate, and stability tags.

## Observability

khora emits OpenTelemetry spans and metrics through the OTel API.
The export path is your choice: vanilla OTel SDK (`pip install
khora[otel]`), [Logfire](https://logfire.pydantic.dev/)
(`pip install khora[logfire]`), or nothing (zero-cost no-op). Khora
never installs a `TracerProvider` at import time and never sets
`service.name` — those belong to the host application.

```bash
pip install khora[otel]
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_SERVICE_NAME="my-app"
```

```python
from khora.telemetry import configure_telemetry
configure_telemetry()      # honors OTEL_* env vars
```

See [docs/observability.md](docs/observability.md) for the full env-var
contract, the precedence rules, vendor recipes (Honeycomb, Datadog,
Tempo, etc.), sampling guidance, and the troubleshooting checklist.
The complete telemetry surface lives in
[`docs/telemetry-contract.json`](docs/telemetry-contract.json) with the
drift gate enforced by `tests/unit/telemetry/test_contract.py`.

Two separate observability channels live in `khora.telemetry`:

- **Spans + metrics** via the OTel API (this section).
- **Structured `LLMEvent` / `StorageEvent` / `PipelineEvent` rows** to
  a dedicated PostgreSQL database when `KHORA_TELEMETRY_DATABASE_URL`
  is set. Without it, a `NoOpCollector` is used (zero cost). Wired by
  `init_telemetry()`, independent of `configure_telemetry()`.

Credential fields on `KhoraConfig` (DSNs, passwords) are
`pydantic.SecretStr` — `repr()` and config dumps render as
`'**********'`. Callers that need the cleartext must call
`.get_secret_value()` explicitly.

**Async logging caveat.** Library consumers that import khora without
configuring loguru sinks inherit the default sync stderr sink, which
blocks the event loop on every log call inside `async def`. Either
call `khora.logging_config.setup_logging()` (which configures sinks
with `enqueue=True` and registers an `atexit` drain) or configure
your own loguru sinks with `enqueue=True` explicitly.

## Documentation

Start at [docs/README.md](docs/README.md). Key entry points:

- [API reference](docs/api-reference.md) — public `Khora` surface.
- [Configuration](docs/configuration.md) — `KHORA_*` env vars and `KhoraConfig`.
- [Observability](docs/observability.md) — OTel spans/metrics, `[otel]`/`[logfire]` paths, `configure_telemetry()`.
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
