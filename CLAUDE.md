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

- **Engines:** implement `MemoryEngineProtocol` in `engines/protocol.py`
- **Graph backends:** Neo4j, SurrealDB, Memgraph — implement `GraphBackend` in `storage/backends/base.py`
- **SurrealDB:** unified backend (graph + vector + relational). Modes: `memory://`, `surrealkv://` (embedded), `ws://` (remote). Set `backend: surrealdb` in config
- **Extraction skills:** YAML-defined in `extraction/skills/builtin/`. Generate with `khora ontology construct`
- **Config:** env vars with `KHORA_` prefix and single underscore (e.g., `KHORA_QUERY_ENABLE_HYDE=true`, `KHORA_LLM_MODEL=gpt-4o`). Legacy `__` nesting also supported

### Key Entry Points

- `memory_lake.py` — `remember()`, `recall()`, `forget()`, `remember_batch()`. Accepts `expertise: ExpertiseConfig`
- `extraction/skills/base.py` — `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` (ADR-022 stable)
- `storage/coordinator.py` — `transaction()` for atomic multi-backend ops
- `storage/backends/base.py` — `GraphBackend` protocol (implement for new backends)
- `storage/backends/surrealdb/` — Unified SurrealDB backend
- `db/models.py` — SQLAlchemy ORM (UUID columns use `as_uuid=True`)
- `_accel.py` — Rust/NumPy acceleration (MMR, cosine, pagerank, entity resolution, community detection, temporal)
- `cli/ontology/` — Ontology construction: `commands.py`, `flow.py`, `inference/`, `sources/`, `sampling/`
- `pipelines/flows/ingest.py` — Document ingestion pipeline (3-phase: stage → enrich → expand)
- `db/migrations/env.py` — Alembic with advisory locking
- `config/schema.py` — `KhoraConfig` Pydantic settings (storage, LLM, pipeline, query, tenancy)
- `telemetry/` — Optional PostgreSQL-backed telemetry collector + `@trace` decorator

## Conventions

### Version Bumps

Khora uses `hatch-vcs` — the package version comes from git tags (`git tag vX.Y.Z`). Only khora-accel needs a manual version in source:

1. `rust/khora-accel/Cargo.toml` — update `version = "X.Y.Z"`
2. Run `cargo generate-lockfile` in `rust/khora-accel/` to update `rust/Cargo.lock`
3. Commit both `Cargo.toml` and `Cargo.lock` in the same PR
4. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`

### Before Creating PRs

Always run `make format && make test` before committing. CI will reject PRs that fail formatting or tests.

## Gotchas

### Migrations & Schema
- **Never use `create_tables()`** — deprecated, bypasses Alembic. Use `run_migrations()` or `MemoryLake(run_migrations=True)`. Create new migrations with `uv run alembic revision --autogenerate -m "desc"`
- **Version table:** `khora_alembic_version` (not `alembic_version`) — avoids conflicts with downstream apps
- **Advisory lock:** `run_migrations()` uses `pg_advisory_xact_lock` (ID `6001515088189075507`), 60s timeout
- **Migrations bundled** in `src/khora/db/migrations/`, not `alembic/`. Root `alembic.ini` is dev-only
- **Skip-ahead:** When multiple services share a DB with different Khora versions, `run_migrations()` detects if the DB revision is unknown (ahead) and skips gracefully — returns `MigrationResult(success=True, skipped=True)`. Signaled via `_DatabaseAheadError` from `env.py` to `session.py`
- **Fresh-DB behavior:** On a PostgreSQL database with no `khora_alembic_version` table yet, `run_migrations()` / `MemoryLake(run_migrations=True)` correctly creates all tables from scratch. Prior to v0.6.6 (DYT-1447), querying the missing version table inside an explicit transaction caused `InFailedSQLTransactionError`. The fix uses `information_schema.tables` to check table existence before querying it — never issuing a statement that could abort the transaction.

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
- **Selective extraction** — KET-RAG style: scores chunk importance, sends top 70% to LLM, rest get co-occurrence edges only
- **Entity resolution** — multi-strategy dedup with per-type thresholds (PERSON 0.92, DATE 0.95, default 0.85)
- **Semantic expansion** — optional cross-tool entity unification + relationship inference (4 modes: smart/batch/incremental/none)

### Optional Dependencies
- **spaCy:** `_HAS_SPACY` flag, falls back to regex sentence splitting
- **Logfire:** `_HAS_LOGFIRE` flag, `trace_span()` yields no-op when absent. Install: `pip install khora[logfire]`
- **`@trace` decorator:** `from khora.telemetry import trace`. Zero overhead when logfire absent
- **Telemetry collector:** `KHORA_TELEMETRY_DATABASE_URL` enables PostgreSQL-backed event recording. Without it, `NoOpCollector` is used (zero cost)

### Downstream
- `genesis` and `khora-benchmarks` depend on khora. `lake.storage` is a stable public API
- **LLMUsage contract:** `LLMUsage` fields are consumed by Poros/Peras for cost tracking (DYT-645) — changes require coordination
- **ExpertiseConfig contract:** ADR-022 stable API — `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` changes require coordination
- `scripts/` vendored from TTOJ — skip in audits

@.claude/docs/workflow.md
