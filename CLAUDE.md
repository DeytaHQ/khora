# Khora

Memory Lake library: knowledge graphs + vector search + PostgreSQL for unified knowledge storage. **Library, not an application.**

## Commands

```bash
make test              # pytest, coverage ≥30%
make format            # black, isort, ruff
make lint              # ruff + ty typecheck
make dev               # Start postgres + neo4j
uv run alembic upgrade head                       # Run migrations
uv run khora ontology construct --source <path>   # AI ontology generation
uv run khora ontology validate <file.yaml>        # Validate ontology YAML
uv run khora ontology preview <file.yaml>         # Rich preview
```

## Architecture

```
MemoryLake → Engine (graphrag | skeleton | vectorcypher) → StorageCoordinator
                                        ├── PostgreSQL (documents, tenancy)
                                        ├── pgvector (embeddings)
                                        └── Graph backend (Neo4j | SurrealDB | ...)

Traditional stack: PostgreSQL + pgvector + Neo4j  (three databases)
Unified stack:     SurrealDB                      (one database, all roles)
```

- **Engines:** implement `MemoryEngineProtocol` in `engines/protocol.py`
- **Graph backends:** Neo4j, SurrealDB, Memgraph, ArcadeDB — implement `GraphBackend` in `storage/backends/base.py`
- **SurrealDB:** unified backend (graph + vector + relational). Modes: `memory://`, `surrealkv://` (embedded), `ws://` (remote). Set `backend: surrealdb` in config
- **Extraction skills:** YAML-defined in `extraction/skills/builtin/`. Generate with `khora ontology construct`
- **Config:** env vars with `KHORA_` prefix, `__` nesting (e.g., `KHORA_QUERY__ENABLE_HYDE=true`)

## Key Entry Points

- `memory_lake.py` — `remember()`, `recall()`, `forget()`, `remember_batch()`. Accepts `expertise: ExpertiseConfig`
- `storage/coordinator.py` — `transaction()` for atomic multi-backend ops
- `storage/backends/base.py` — `GraphBackend` protocol (implement for new backends)
- `storage/backends/surrealdb/` — Unified SurrealDB backend
- `db/models.py` — SQLAlchemy ORM (UUID columns use `as_uuid=True`)
- `_accel.py` — Rust/NumPy acceleration (MMR, cosine, BM25, temporal detection)
- `cli/ontology/` — Ontology construction: `commands.py`, `flow.py`, `inference/`, `sources/`, `sampling/`
- `pipelines/flows/ingest.py` — Document ingestion pipeline
- `db/migrations/env.py` — Alembic with advisory locking

## Engine Selection

| Use Case | Engine | Requires |
|----------|--------|----------|
| Knowledge bases, entity exploration | `graphrag` | Neo4j or SurrealDB |
| Multi-hop queries, complex relationships | `vectorcypher` | Neo4j or SurrealDB |
| Chat history, cost-sensitive | `skeleton` | Graph backend optional |

## Testing

```bash
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest -k "test_remember" -v         # By name
uv run pytest tests/unit/test_memory_lake.py  # Single file
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`. Async: `asyncio_mode = "auto"`.

## Releasing

Versions from git tags — no manual bumps. `git tag vX.Y.Z && git push origin vX.Y.Z` triggers publish workflows. See [`docs/RELEASE.md`](docs/RELEASE.md).

## Gotchas

### Migrations & Schema
- **Never use `create_tables()`** — deprecated, bypasses Alembic. Use `run_migrations()` or `MemoryLake(run_migrations=True)`. Create new migrations with `uv run alembic revision --autogenerate -m "desc"`
- **Version table:** `khora_alembic_version` (not `alembic_version`) — avoids conflicts with downstream apps
- **Advisory lock:** `run_migrations()` uses `pg_advisory_xact_lock` (ID `6001515088189075507`), 60s timeout
- **Migrations bundled** in `src/khora/db/migrations/`, not `alembic/`. Root `alembic.ini` is dev-only
- **Skip-ahead:** When multiple services share a DB with different Khora versions, `run_migrations()` detects if the DB revision is unknown (ahead) and skips gracefully — returns `MigrationResult(success=True, skipped=True)`. Signaled via `_DatabaseAheadError` from `env.py` to `session.py`

### UUID & Type Handling
- **ORM:** all 52 UUID columns use `as_uuid=True` — native `uuid.UUID`, never `str()` wrap
- **Graph boundary:** Neo4j/Memgraph need `str(uuid)` at the driver boundary only
- **SurrealDB:** `RecordID` accepts UUID objects directly (no `str()` needed since SDK 2.0)

### Backend Specifics
- **Shared engine pools:** `StorageFactory` caches by URL. Shared-engine backends skip `dispose()`
- **Transactions:** `async with coordinator.transaction() as txn:` for atomic multi-backend ops
- **SurrealDB unified:** all four adapters share one `SurrealDBConnection`. Coordinator skips duplicate writes
- **SurrealDB schema:** declarative (`DEFINE IF NOT EXISTS`), auto-initializes on `connect()`. No Alembic
- **SurrealDB SDK:** pinned `>=2.0.0a1` for 3.x support. Install: `pip install khora[surrealdb]`
- **SurrealDB KNN broken:** `<|K|>` unreliable in embedded mode. Uses brute-force cosine + HNSW instead
- **SurrealDB entity gate:** `_SurrealDBEntityKeyGate` serializes concurrent upserts by (ns, name, type) key

### Extraction & Search
- **Pre-normalized embeddings** — L2-normalized at ingest. Uses `batch_dot_product` (3x faster than cosine)
- **Entity unique constraint** — `(namespace_id, name, entity_type)` UNIQUE in both PostgreSQL and SurrealDB
- **Namespace versioning** — dual IDs: `id` (row-level) vs `namespace_id` (stable). Public API uses `namespace_id`, resolves to `id` via indexed lookup

### Optional Dependencies
- **spaCy:** `_HAS_SPACY` flag, falls back to regex sentence splitting
- **Logfire:** `_HAS_LOGFIRE` flag, `trace_span()` yields no-op when absent. Install: `pip install khora[logfire]`
- **`@trace` decorator:** `from khora.telemetry import trace`. Zero overhead when logfire absent

### Downstream
- `genesis` and `khora-benchmarks` depend on khora. `lake.storage` is a stable public API
- `scripts/` vendored from TTOJ — skip in audits
