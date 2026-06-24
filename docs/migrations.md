# Migrations

Khora ships its own Alembic migrations bundled inside the package at `src/khora/db/migrations/`. This applies to **PostgreSQL-backed** deployments only - SurrealDB uses a declarative schema (`DEFINE … IF NOT EXISTS`) that is applied automatically on `connect()`.

## Who runs migrations?

Library consumers need Khora's schema to exist before calling `Khora()`. Two options:

### 1. Let Khora run them for you

```python
async with Khora(run_migrations=True) as kb:
    ...
```

Khora takes a session-scoped PostgreSQL advisory lock (`pg_advisory_lock`, ID `6001515088189075507`, 60 s timeout) before the migration transaction, runs any pending migrations, and releases the lock explicitly in a `finally` block. The lock is session-scoped (rather than transaction-scoped) because each migration commits its own transaction, which would otherwise release the lock mid-chain. Safe under concurrent startup - only one process runs the migrations at a time, the others wait and then no-op.

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

This is signalled internally by a `_DatabaseAheadError` from `env.py` to `session.py` - library code does not need to handle it explicitly. The takeaway: **do not pin different services to different Khora major versions that share a PostgreSQL database**. Use the same major across services; the skip-ahead is a safety net, not a coordination tool.

## Fresh-database behaviour

On a PostgreSQL database with no `khora_alembic_version` table yet, `run_migrations()` / `Khora(run_migrations=True)` creates every table from scratch. The implementation checks for the table's existence via `information_schema.tables` rather than issuing a raw query that would abort the transaction (fixed in v0.6.6).

## What about `create_tables()`?

Removed - it bypassed Alembic and left the version table in an inconsistent state. If you find old docs or sample code referencing `create_tables()`, replace it with `run_migrations()` or `Khora(run_migrations=True)`.

## Dialect-conditional migrations

A few migrations execute only on PostgreSQL - they use Postgres-specific features that have no SQLite analogue. The migration scripts gate on `op.get_bind().dialect.name == "postgresql"` and skip silently on `sqlite_lance` so the embedded test stack runs the same chain without errors.

Current dialect-gated migrations:

- **`029_chunks_created_at_brin` (v0.12.0)** - BRIN index on `chunks.created_at` (`pages_per_range = 32`), built with `CREATE INDEX CONCURRENTLY` inside an Alembic autocommit block so online traffic is not blocked. BRIN indexes are tiny (KB-sized) and well-suited to time-correlated columns like `created_at`; they don't compete with HNSW vector indexes or the existing B-trees on chunks. The index helps long-range archive / export queries (months of data) that today sequential-scan the table. No effect on point queries or HNSW similarity search. SQLite-backed embedded stacks skip this migration entirely - see Issue #593 for the rationale.
- **`030_session_id_columns`** - adds a nullable `session_id UUID` column to `documents`, `chunks`, `memory_events`, `chronicle_events`, and `memory_facts`. Runs on both PostgreSQL and SQLite (the column itself is portable). Existing rows naturally carry `NULL`; adapters that don't track sessions can keep ignoring the field. Companion to migration 031.
- **`031_session_id_indexes`** - Postgres-only. Adds two partial B-tree indexes on `(namespace_id, session_id) WHERE session_id IS NOT NULL` for `chunks` and `documents`, and a BRIN index on `chunks (session_id, created_at)` (`pages_per_range = 32`). All created with `CREATE INDEX CONCURRENTLY` inside an autocommit block. The partial indexes cover session-scoped recall (`WHERE namespace_id = ? AND session_id = ?`); the BRIN accelerates time-bounded session replay and `gc.expire_sessions(before=…)`. SQLite-backed embedded stacks skip silently - point lookups on `chunks` are fine at SQLite scale and `CREATE INDEX CONCURRENTLY` is Postgres-specific. See Issue #620.
- **`032_dream_runs`** - Postgres-only via the same dialect gate as migration 029. Adds `khora_dream_runs`, a per-namespace checkpoint table so the dream-phase orchestrator can resume a crashed apply pass against the last committed op-seq rather than restarting from scratch. The embedded `sqlite_lance` stack mirrors checkpoint state to a `dream_runs.jsonl` file sink instead. See Issue #651 (dream Phase 0.2).
- **`033_bitemporal_columns`** - Adds nullable `valid_to`, `invalidated_at`, `invalidated_by` to `relationships` and `memory_facts`. The column adds run on both dialects (columns are portable); the partial indexes `ix_relationships_live` and `ix_memory_facts_live` (`WHERE invalidated_at IS NULL`) are Postgres-only via `CREATE INDEX CONCURRENTLY` inside an autocommit block. Existing rows backfill to all-NULL (= "still valid"). See Issue #653 (dream Phase 0.3).
- **`034_chronicle_events_bitemporal`** - Adds `invalidated_at`, `invalidated_by`, and `merged_into_event_id` (self-FK with `ON DELETE SET NULL`, created via `use_alter=True` mirroring migration 004) to `chronicle_events`. The partial composite index `ix_chronicle_events_live` on `(namespace_id, occurred_at) WHERE invalidated_at IS NULL` is Postgres-only. See Issue #669 (dream Phase 4 event clustering).
- **`035_dream_communities`** - Postgres-only via the dialect gate. Adds `khora_dream_communities` for community-summary persistence with bi-temporal validity; the apply path writes grounded LLM summaries (uncited claims dropped before INSERT). The `sqlite_lance` stack mirrors community state to the JSONL undo sink instead. See Issue #670 (dream Phase 5.1).
- **`036_dream_conflicts`** - Postgres-only via the dialect gate. Adds `dream_conflicts` for the vectorcypher contradiction-detection op's report-only findings (the op never mutates `relationships` - Phase 5.4 / Issue #673 will own that). See Issue #672 (dream Phase 5.3).
- **`037_recall_response_format`** - Cross-dialect. Adds `documents.source_name VARCHAR(64)` (backfilled from `nango://<provider>/...` patterns in `source`), `documents.source_url VARCHAR(2048)`, and `chunks.chunker_info JSONB NOT NULL DEFAULT '{}'::jsonb`. Also flips six `documents` columns (`source`, `content_type`, `title`, `author`, `language`, `checksum`) to nullable-with-no-default; legacy `create_tables()`-created rows that were `NOT NULL DEFAULT ''` need their `NOT NULL` dropped before the empty-string normalisation runs (see PR #819).
- **`038_khora_chunks_chunker_info`** - Mirrors 037's `chunker_info` onto the vectorcypher temporal-store table `khora_chunks`. Postgres-specific: asserts `server_version_num >= 110000` so the fast `ADD COLUMN NOT NULL DEFAULT` path is available (avoids a multi-hour table rewrite on PG < 11), and issues `SET lock_timeout = '5s'` before DDL so the `AccessExclusiveLock` acquisition is bounded; on lock-timeout, logs `khora.migration.applied` with `lock_timeout_tripped=True` and SQLSTATE `55P03` for dashboard correlation.
- **`039_khora_chunks_content_tsv_gin`** - Postgres-only. Adds a GIN index on `khora_chunks.content_tsv` (BM25 / `ts_rank` queries against vectorcypher's temporal-store chunks) via `CREATE INDEX CONCURRENTLY ... IF NOT EXISTS` in an autocommit block. Converges cleanly with the runtime `CREATE INDEX IF NOT EXISTS` in `PgVectorTemporalStore.connect()` - whichever runs first wins. SQLite uses an FTS5 virtual table instead.
- **`040_chunks_last_accessed_at`** - Cross-dialect. Adds a nullable `last_accessed_at TIMESTAMPTZ` column to the `chunks` table. Stamped by the engine when `KHORA_QUERY_CHRONICLE_ENABLE_RECALL_REINFORCEMENT=true`; the temporal-decay path reads `max(source_timestamp, last_accessed_at)` so frequently-recalled chunks stay fresh. The partial index skips NULL rows on both dialects (`postgresql_where` / `sqlite_where`).
- **`041_khora_chunks_denormalized_columns`** - Cross-dialect. Adds eight nullable denormalized document columns to the `khora_chunks` temporal-store table (`source_type`, `source_name`, `source_url`, `source_timestamp`, `external_id`, `content_type`, `source`, `title`). Enables recall-filter pushdown directly on chunk rows without a join.
- **`042_widen_khora_chunks_source_external_id`** - Postgres-only DDL; cross-dialect schema notes. Widens `khora_chunks.external_id` and related VARCHAR columns.
- **`043_khora_chunks_metadata_backfill`** - Cross-dialect. Backfills the denormalized columns (migration 041) from the parent `documents` row for all existing chunks.
- **`044_khora_chunks_backfill_denormalized`** - Postgres-only. Adds supplemental backfill covering document-level columns not caught by migration 043.
- **`045_khora_try_timestamptz`** - Postgres-only. Adds the PL/pgSQL function `khora_try_timestamptz(text)` - a safe `text → TIMESTAMPTZ` cast that returns `NULL` rather than aborting the query on a malformed value. Used by the recall-filter `$date` operator. Skips silently on SQLite.
- **`046_chunks_occurred_at`** - Cross-dialect. Adds a nullable `occurred_at TIMESTAMPTZ` column to the `chunks` table for real-world event time, distinct from ingestion `created_at`. Enables time-bounded session recall and temporal-filter pushdown.
- **`047_dream_runs_graph_mirror_pending`** - Cross-dialect. Adds a nullable `graph_mirror_pending JSONB` column to `khora_dream_runs`. Holds the pending-ops list for the post-commit dream-on-graph mirror reconciler (#1274). NULL = no ops awaiting mirror; existing rows are untouched.
- **`048_dream_conflicts_reconcile`** - Cross-dialect. Extends `dream_conflicts` (migration 036) with reconcile-outcome columns: `resolution VARCHAR(16)`, `loser_relationship_id UUID`, `winner_relationship_id UUID`, `judge_rationale_hash VARCHAR(16)`, `resolved_by_op_id UUID`. The contradiction-detection op can now record soft-delete judgements (#1281) alongside detection findings.
- **`049_hook_subscriptions`** - Cross-dialect. Adds `khora_hook_subscriptions` for durable semantic-hook persistence (#599). Nine columns including `id`, `namespace_id`, `event_type`, `filter (JSONB)`, `delivery (JSONB)`, `created_at`, `last_delivered_at`. Enables `HookDispatcher.register_persistent` / `load_persistent` so subscriptions survive restarts.

## SurrealDB

SurrealDB doesn't use Alembic. The schema is defined with idempotent `DEFINE … IF NOT EXISTS` statements that execute on `SurrealDBBackend.connect()`. There's no migration flag; the schema is always current. See [architecture/storage-backends.md](architecture/storage-backends.md#surrealdb-the-unified-backend) for the schema layout.

In v0.16.0 the SurrealDB schema gained `DEFINE FIELD rel_id ON relates_to TYPE string` plus a `UNIQUE INDEX` on it, as part of the `table:⟨$var⟩` interpolation repair (PR #770 / issue #750) - see the v0.16.0 entry in [CHANGELOG.md](../CHANGELOG.md) for the full surface. The schema is reapplied automatically on `connect()`; no action is required on existing SurrealDB stores.

## v0.16.0 - API migration: namespace kwarg required everywhere

v0.16.0 closed out the cross-namespace IDOR family (PRs #761 / #765 / #766 / #769). This is an API migration, not a schema migration - no Alembic revisions are involved - but downstream code that pokes at the storage substrate needs to be updated. See the [Security exception entry in consumers.md](consumers.md#versioning-policy) for the policy rationale.

### What changed

Every read, exists-check, and mutation on every storage backend (`RelationalBackend`, `VectorBackend`, `GraphBackend`, `EventStore`) now requires `*, namespace_id: UUID` (kwarg-only) and filters at the SQL / Cypher / SurrealQL layer.

- **Top-level facade**: `Khora.get_document(doc_id, *, namespace=…)` requires the `namespace=` kwarg. Cross-tenant lookups by id return `None`.
- **Coordinator getters** (`StorageCoordinator.{relational,vector,graph,event_store}`) are now `NamespaceRequiredProxy` instances. Reading them emits one `DeprecationWarning` per role per process; calling a method on the proxy with no `namespace_id=` raises `TypeError`. Public attributes are removed in **v0.17** - internal canonical references use `self._{relational,vector,graph,event_store}` instead.
- **Backend methods tightened**:
  - *Reads*: `RelationalBackend.get_document` / `get_documents_batch` / `get_document_sources_batch` / `get_document_projections_batch` / `get_document_by_external_id` / `get_documents_by_external_ids`; `VectorBackend.entity_exists` plus pgvector-specific `get_entity` / `get_entities_batch`; `GraphBackend.get_entity` / `get_entities_batch` / `get_relationship` / `get_episode` / `get_entity_relationships` / `get_neighborhood` / `get_neighborhoods_batch` / `find_paths` / `get_temporal_neighbors`; `EventStore.get_events_for_resource` / `get_latest_event`.
  - *Writes*: `RelationalBackend.delete_document`; `VectorBackend.delete_chunks_by_document` / `update_entity` / `update_entity_embedding` / `update_entity_embeddings_batch` / `delete_entities_batch` / `delete_relationships_batch` / `supersede_fact`; `GraphBackend.update_entity` / `delete_entity` / `delete_relationship` / `delete_entities_batch` / `delete_relationships_batch` / Neo4j-specific `retire_orphaned_relationships_batch` / `remap_source_document_ids_batch`.
- **Cross-namespace reads** return `None` / `{}` / `[]`; cross-namespace writes silently no-op (raising would expose row existence). Graph traversal filters at every hop so a traversal seeded inside namespace A cannot visit a node in namespace B.
- **Regression gate**: `tests/security/test_cross_namespace_idor_signatures.py` walks every concrete backend at collection time and asserts the contract. CI fails on any future signature drift.

### Migration recipe

If your code calls these methods directly:

```python
# Before (v0.15.x) - implicit namespace was allowed
# doc = await kb.get_document(doc_id)
# ent = await kb.storage.graph.get_entity(entity_id)
# await kb.storage.relational.delete_document(doc_id)

# After (v0.16.0+)
doc = await kb.get_document(doc_id, namespace=ns_id)
ent = await kb.storage.get_entity(entity_id, namespace_id=ns_id)  # via coordinator
await kb.storage.delete_document(doc_id, namespace_id=ns_id)      # coordinator method
```

Code paths that already routed through the `Khora` facade are unaffected - none touch `kb.storage.{relational,vector,graph,event_store}` directly. Code paths that reach into `kb.storage.<role>` will see a `DeprecationWarning` on first access per role per process; calling a method without `namespace_id=` raises `TypeError`. Migrate to coordinator-level methods at your earliest convenience - the proxy wrappers (`NamespaceRequiredProxy`) are currently deprecated and will be removed in a future major release.

## Replacing the removed GraphRAG engine

The `graphrag` engine is no longer available. Replace it with `vectorcypher`:

```python
# Old (graphrag - no longer available)
# async with Khora(db_url, engine="graphrag") as kb:
#     await kb.remember(content, namespace=ns_id, ...)

# Drop-in replacement: vectorcypher with full extraction
async with Khora(db_url, engine="vectorcypher",
                 engine_kwargs={"skeleton_core_ratio": 1.0}) as kb:
    await kb.remember(
        content,
        namespace=ns_id,
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
    )

# Or accept default selective extraction (recommended - 30% cheaper):
async with Khora(db_url, engine="vectorcypher") as kb:
    await kb.remember(
        content,
        namespace=ns_id,
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
    )
```

Existing graphrag-ingested data remains queryable via `vectorcypher` against the same database - the table shapes are identical.

## v0.8.0 - CLI extraction

The CLI commands (`khora extract`, `khora search`, `khora ontology …`) were removed from the `khora` package so the library has no CLI dependencies (no `click`, no `rich`, no PDF / Excel readers by default). The `khora` top-level imports (`Khora`, `KhoraConfig`, `SearchMode`, `ExpertiseConfig`, etc.) are unchanged - call them directly from your service or notebook.

Companion CLI packages (`khora-cli`, `khora-explorer`) are planned for a later release and are not available today. Until they ship, there is no in-library CLI replacement.

The one piece of CLI functionality available inside the library today is binary-document text extraction. Install with `pip install khora[binary-readers]` and use it directly:

```python
from khora.extraction.binary_readers import extract_if_needed
```

**Failure contract:** `extract_if_needed` raises `ExtractionError` on genuine parse/open failures (xlsx, docx, parquet). Passing a `.pdf` path raises `NotImplementedError` - preprocess PDFs upstream or use `khora-cli`'s PDF preprocessing.

Old `from khora.discovery …` and `from khora.cli …` imports have no in-library replacement - call the public `khora` API instead.
