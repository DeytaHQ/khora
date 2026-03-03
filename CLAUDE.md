# Khora

Memory Lake library combining knowledge graphs, vector database (pgvector), and PostgreSQL for unified knowledge storage and retrieval. **This is a library, not a deployable application.**

## Commands

```bash
make test              # Run tests (pytest, coverage ≥30%)
make format            # Format code (black, isort, ruff)
make lint              # Lint + typecheck (ruff, ty)
make dev               # Start local databases (postgres + neo4j)
uv run khora serve --reload  # Dev server
uv run alembic upgrade head  # Run migrations
```

## Architecture

```
MemoryLake (facade) → Engine (graphrag | skeleton | vectorcypher) → StorageCoordinator
                                                    ├── PostgreSQL (documents, tenancy)
                                                    ├── pgvector (embeddings)
                                                    └── Graph backend (entities, relationships)
```

- **Engines are pluggable** — implement `MemoryEngineProtocol` in `engines/protocol.py`
- **Graph backends are interchangeable** — all implement `GraphBackend` in `storage/backends/base.py`
- **Extraction skills are YAML-defined** — see `extraction/skills/builtin/`
- **Multi-tenancy:** Organization → Workspace → MemoryNamespace
- **Config via env vars** — prefix `KHORA_`, use `__` for nesting (e.g., `KHORA_QUERY__ENABLE_HYDE=true`)

## Key Entry Points

- `memory_lake.py` — Public API: `remember()`, `recall()`, `forget()`, `remember_batch()`
- `storage/coordinator.py` — Backend orchestration, `TransactionContext`, `transaction()`
- `storage/factory.py` — Backend creation with shared engine pools
- `db/session.py` — `DatabaseManager` class for session/engine lifecycle
- `db/models.py` — SQLAlchemy ORM (all UUID columns use `as_uuid=True`)
- `engines/` — GraphRAG (default), Skeleton Construction, VectorCypher
- `query/engine.py` — `HybridQueryEngine` search pipeline
- `_accel.py` — Rust/NumPy/Python acceleration facade (MMR, cosine, temporal, BM25, etc.)
- `pipelines/flows/ingest.py` — Document ingestion pipeline with entity ID mapping

## Engine Selection

| Use Case | Engine | Key Trait |
|----------|--------|-----------|
| Knowledge bases, entity exploration | `graphrag` | Full graph extraction, requires Neo4j/Kuzu |
| Multi-hop queries, complex relationships | `vectorcypher` | Vector + Cypher hybrid, requires Neo4j |
| Chat history, event streams, cost-sensitive | `skeleton` | Temporal-first, 5-10x fewer LLM calls, Neo4j optional |

## Testing

```bash
uv run pytest tests/unit/ -v               # Unit tests only
uv run pytest -k "test_remember" -v         # By name
uv run pytest tests/unit/test_memory_lake.py  # Single file
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`. Async tests use `asyncio_mode = "auto"`.

## Version Bumps

IMPORTANT: When bumping the version, always update **all four files** and regenerate lockfiles:
1. `pyproject.toml` — khora version
2. `src/khora/__init__.py` — `__version__`
3. `rust/khora-accel/Cargo.toml` — khora-accel version
4. `rust/khora-accel/pyproject.toml` — khora-accel version
5. Run `uv lock` and `cargo generate-lockfile` in `rust/khora-accel/`

## Gotchas

- **No Docker in CI** — khora is a library; CI only runs tests, linting, and type checking
- **UUID columns use `as_uuid=True`** — all 52 UUID columns in `db/models.py` map to native Python `uuid.UUID` objects. Never use `str()` wrapping when building ORM models
- **Graph backends need `str()` at boundary** — Neo4j/Kuzu/Memgraph don't support native UUIDs, so convert at the graph DB boundary only
- **Shared engine pools** — `StorageFactory` caches engines by normalized URL. Backends sharing the same URL reuse one `AsyncEngine`. Shared-engine backends must skip `dispose()` on disconnect
- **Transactions** — use `async with coordinator.transaction() as txn:` for atomic multi-backend operations. Backend write methods accept optional `session` parameter to join an existing transaction
- **spaCy is optional** — `_HAS_SPACY` flag controls sentence splitting. Uses blank model with `sentencizer` pipe (no model download needed). Falls back to regex when spaCy is not installed
- **Logfire is optional** — `_HAS_LOGFIRE` flag in `telemetry/logfire_integration.py` controls OTEL span emission. Install with `pip install khora[logfire]`. When absent, `trace_span()` yields a no-op `Span` singleton that silently discards attribute writes (zero-cost). Consumers import `trace_span` from `khora.telemetry`, not from `logfire_integration` directly. Custom telemetry (`collector.record_*`) fires regardless of logfire presence. Khora never calls `logfire.configure()` or `logfire.instrument_*()` — that's the consumer's responsibility
- **@trace decorator** — Use `from khora.telemetry import trace` for automatic span creation. Decorates sync/async functions, auto-captures arguments as span attributes (UUID→str, list/tuple/set→count, enum→value, complex objects skipped). Supports `include`/`exclude` filters and `result` extractor for return values. When logfire is absent, short-circuits to direct function call (zero overhead). Use `@trace` for simple span-per-function patterns; use `trace_span()` context manager for complex methods needing mid-function attributes. Example: `@trace("khora.search", exclude={"query"}, result=lambda r: {"count": len(r)})`
- **Downstream consumers** — `genesis` and `khora-benchmarks` depend on khora. Check compatibility when changing public APIs. `lake.storage` is a stable public API used by both
- **Entity unique constraint** — `entities(namespace_id, name, entity_type)` has a UNIQUE constraint (migration 008). Entity upserts use `ON CONFLICT` on this constraint. Dedup migration is irreversible
- **Pre-normalized embeddings** — All embeddings are L2-normalized at ingest time. Scoring uses `batch_dot_product` instead of `batch_cosine_similarity` for ~3x speedup. Dot product of unit vectors = cosine similarity
- **MMR diversity enabled by default** — `enable_diversity=True` in `QuerySettings`. The MMR stage runs in Rust via `_accel.mmr_diversity_select` with NumPy and pure-Python fallbacks
- **`ty` type checker** — Pre-commit hook runs `ty check src/` which passes clean (`All checks passed!`). If ty fails on your changes, fix the diagnostics before committing
