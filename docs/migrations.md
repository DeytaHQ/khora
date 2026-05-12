# Migrations

Khora ships its own Alembic migrations bundled inside the package at `src/khora/db/migrations/`. This applies to **PostgreSQL-backed** deployments only — SurrealDB uses a declarative schema (`DEFINE … IF NOT EXISTS`) that is applied automatically on `connect()`.

## Who runs migrations?

Library consumers (genesis, khora-cli, khora-benchmarks, custom services) need Khora's schema to exist before calling `Khora()`. Two options:

### 1. Let Khora run them for you

```python
async with Khora(run_migrations=True) as kb:
    ...
```

Khora takes a PostgreSQL advisory lock (ID `6001515088189075507`, 60 s timeout), runs any pending migrations, and releases the lock. Safe under concurrent startup — only one process runs the migrations at a time, the others wait and then no-op.

### 2. Run them out-of-band

Pre-deploy with the alembic CLI:

```bash
uv run alembic upgrade head
```

The repo's root `alembic.ini` is dev-only; in CI or production, Khora consumers typically call `alembic upgrade head` against the packaged migration directory. Example invocation from a downstream package:

```bash
uv run alembic -c path/to/your/alembic.ini upgrade head
```

The migration directory is resolved via `script_location = khora:db/migrations` when you install Khora as a dependency.

## Creating a new migration

From the Khora repo (not a downstream consumer):

```bash
uv run alembic revision --autogenerate -m "add widget table"
```

Review the generated file in `src/khora/db/migrations/versions/` before committing. Autogenerate catches 90 % of schema diffs; index changes, enum alterations, and PostgreSQL extensions often need manual tweaks.

## Version table

Khora's migrations live in `khora_alembic_version`, **not** `alembic_version`. This avoids collisions with a downstream app that has its own alembic history. If your app has a separate migration system, point it at `alembic_version` and keep Khora's table untouched.

## Skip-ahead behaviour

A downstream service at Khora v0.7 may run against a database already migrated to v0.8 (by another service). Khora detects that the current DB revision is unknown to the installed package and skips gracefully:

```python
result = await run_migrations(database_url)
# MigrationResult(success=True, skipped=True, current_revision="ab1c2d3e…")
```

This is signalled internally by a `_DatabaseAheadError` from `env.py` to `session.py` — library code does not need to handle it explicitly. The takeaway: **do not pin different services to different Khora major versions that share a PostgreSQL database**. Use the same major across services; the skip-ahead is a safety net, not a coordination tool.

## Fresh-database behaviour

On a PostgreSQL database with no `khora_alembic_version` table yet, `run_migrations()` / `Khora(run_migrations=True)` creates every table from scratch. The implementation checks for the table's existence via `information_schema.tables` rather than issuing a raw query that would abort the transaction (fixed in v0.6.6, DYT-1447).

## What about `create_tables()`?

Removed — it bypassed Alembic and left the version table in an inconsistent state. If you find old docs or sample code referencing `create_tables()`, replace it with `run_migrations()` or `Khora(run_migrations=True)`.

## SurrealDB

SurrealDB doesn't use Alembic. The schema is defined with idempotent `DEFINE … IF NOT EXISTS` statements that execute on `SurrealDBBackend.connect()`. There's no migration flag; the schema is always current. See [architecture/storage-backends.md](architecture/storage-backends.md#surrealdb) for the schema layout.
