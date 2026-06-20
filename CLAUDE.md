# Khora

Khora library: knowledge graphs + vector search + PostgreSQL for unified knowledge storage. **Library, not an application.**

## Commands

```bash
make install           # uv sync --all-extras --no-extra google-adk (crewai combo)
make install-adk       # uv sync --all-extras --no-extra crewai     (google-adk combo)
make test              # pytest, coverage ≥65%
make format            # black, isort, ruff
make lint              # ruff + ty typecheck
make dev               # Start postgres + neo4j
uv run alembic upgrade head                       # Run migrations
```

**Installing all extras at once doesn't work.** `crewai` and `google-adk` declare mutually-incompatible `opentelemetry-api` ranges (crewai pins `<1.35`, google-adk pins `>=1.36`) and `[tool.uv].conflicts` makes the conflict explicit, so `uv sync --all-extras` is rejected. Pick one combo: `make install` for crewai (matches CI's `test` job), `make install-adk` for google-adk. CI also keeps `UV_NO_SYNC=1` set so subsequent `uv run` calls don't silently re-resolve the venv to the other combo.

CLI tooling (`extract`, `search`, ontology tools) was removed from the `khora` package. khora is a Python library.

## Test Commands

```bash
make test                                          # Full test suite (unit + integration + e2e), coverage ≥65%
uv run pytest -m integration                       # Integration tests only
uv run pytest -m e2e                               # End-to-end tests only
```

Docker Compose is always available. Always run `make test` before opening a PR. Never skip tests.

## Test Infrastructure Isolation

**Never reuse running Docker containers from other projects.** Integration tests must use their own Docker Compose stack (compose file in this repo), not containers from other worktrees or other developer projects. Before running integration tests:

1. Ensure your test databases are started from THIS repo's compose file
2. If port conflicts arise, stop your own containers or use different ports - never repurpose another project's infrastructure

## Architecture

- **Engines:** implement `MemoryEngineProtocol` in `engines/protocol.py`. Default engine is `vectorcypher`
- **Graph backends:** Neo4j, SurrealDB, Memgraph, Neptune, AGE - implement `GraphBackend` in `storage/backends/base.py`
- **SurrealDB:** unified backend (graph + vector + relational). Modes: `memory://`, `surrealkv://` (embedded), `ws://` (remote). Set `backend: surrealdb` in config
- **Extraction skills:** YAML-defined in `extraction/skills/builtin/`.
- **Config:** env vars with `KHORA_` prefix and single underscore (e.g., `KHORA_QUERY_ENABLE_HYDE=true`, `KHORA_LLM_MODEL=gpt-4o`). The legacy `__` nesting form is kept working silently for backward compatibility; single underscore is the documented form

### Key Entry Points

- `khora.py` - `remember()`, `recall()`, `forget()`, `remember_batch()`. Accepts `expertise: ExpertiseConfig`
- `extraction/skills/base.py` - `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig`
- `storage/coordinator.py` - `transaction()` for atomic multi-backend ops
- `storage/backends/base.py` - `GraphBackend` protocol (implement for new backends)
- `storage/backends/surrealdb/` - Unified SurrealDB backend
- `db/models.py` - SQLAlchemy ORM (UUID columns use `as_uuid=True`)
- `_accel.py` - Rust/NumPy acceleration (MMR, cosine, pagerank, entity resolution, community detection, temporal)
- `extraction/binary_readers.py` - xlsx/docx/parquet readers; stable public boundary (`.pdf` raises `NotImplementedError` — preprocess upstream or use khora-cli).
- `pipelines/flows/ingest.py` - Document ingestion pipeline (3-phase: stage → enrich → expand)
- `db/migrations/env.py` - Alembic with advisory locking
- `config/schema.py` - `KhoraConfig` Pydantic settings (storage, LLM, pipeline, query, tenancy)
- `exceptions.py` - `KhoraError` hierarchy with domain-specific exceptions
- `telemetry/` - Optional PostgreSQL-backed telemetry collector + `@trace` decorator

## Issue tracking & workflow

khora is open source. **All khora work is tracked in GitHub Issues** at https://github.com/DeytaHQ/khora/issues. Use `gh issue` from the CLI or the GitHub web UI.

Workflow for any change:

1. Create or pick a GitHub issue describing the work.
2. Create a feature branch off `main` (`<initials>/<short-desc>`).
3. Open a PR against `main`. Include `Fixes #<n>` in the body to auto-close the issue on merge.
4. CI must be green before merge. Squash-merge by default.
5. The `release.yml` workflow publishes to PyPI on `v*` tag push (see `docs/RELEASE.md`).

**Do not maintain `docs/AI_CHANGELOG.md` in this repo.** Commit messages and merged-PR titles are the changelog of record.

## Conventions

### Version Bumps

Khora uses `hatch-vcs` - khora's version comes from git tags (`git tag vX.Y.Z`). khora-accel has its version in source. **khora and khora-accel are always released at the same version (lockstep contract)** - the matching pin in `pyproject.toml`'s `rust` extra enforces this for installers.

Per release:

1. `rust/khora-accel/Cargo.toml` - update `version = "X.Y.Z"`
2. `pyproject.toml` (root) - update `khora-accel == X.Y.Z` in the `rust` extra to match
3. Run `cargo generate-lockfile` in `rust/khora-accel/` to update `rust/Cargo.lock`
4. `CHANGELOG.md` - prepend a `## [X.Y.Z] - <one-line headline>` entry above the previous version with `### Fixed` / `### Changed` / `### Added` / `### Removed` sections as appropriate
5. Commit all four in the same PR
6. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`. The release pipeline publishes to PyPI and auto-creates a GitHub release at `github.com/DeytaHQ/khora/releases/tag/vX.Y.Z` with notes generated from merged PRs since the previous tag.

Why all four together? The release pipeline does NOT modify `pyproject.toml` at runtime - that would dirty the working tree and confuse hatch-vcs into producing a `.devN` version. The lockstep pin must already be correct in the committed source. The CHANGELOG entry must also be present in the tagged commit so users browsing PyPI or the source tarball can see what changed.

### Before Creating PRs

Always run `make format && make test` before committing. CI will reject PRs that fail formatting or tests. Docker Compose is always available - never skip tests by claiming infrastructure is unavailable.

### Integration examples

Every adapter ships `examples/integrations/<name>/example.py` that runs without external services (sqlite_lance fixture + mock LLM helpers under `examples/_helpers/`). The `python title="example.py"` block in `docs/integrations/<name>.md` must be byte-identical to that file. The `examples-smoke` CI job gates drift via `tools/check_examples_drift.py` and smoke-runs each example under a 30s timeout.

### Coding Principles

#### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These principles are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Gotchas

### Failure observability (ADR-001)
- **Functions that catch exceptions and return defaults MUST record a `Degradation` / `ErrorRecord` / `SkipReason` per ADR-001 convention.** TypedDicts live in `khora.core.diagnostics`; the convention doc with reference impls (`#871` storage partial-failure counter, `#880` `DreamResult.metadata["skip_reasons"]`, `#901` + `#906` chronicle channel degradations) is `docs/architecture/failure-observability-contract.md`. Attach the list to whichever observability dict the result already carries (`metadata` for `DreamResult` / `RememberResult`, `engine_info` for `RecallResult`). Log levels: `WARNING` for degradation (with `exc_info=True` when triggered by a caught exception), `ERROR` for unrecoverable, `INFO` (or `DEBUG` on cold-start) for skip. Emit `khora.{engine}.{component}.degraded_total{channel, reason}` - no `namespace_id` label (cardinality rule). Tests can guard happy paths with `tests/test_helpers/diagnostics.py::assert_no_silent_degradation(result)`.

### Migrations & Schema
- **Never use `create_tables()`** - deprecated, bypasses Alembic. Use `run_migrations()` or `Khora(run_migrations=True)`. Create new migrations with `uv run alembic revision --autogenerate -m "desc"`
- **Version table:** `khora_alembic_version` (not `alembic_version`) - avoids conflicts with downstream apps
- **Advisory lock:** `run_migrations()` uses a session-scoped `pg_advisory_lock` (ID `6001515088189075507`), 60s timeout. Acquired before the migration transaction and released explicitly in a `finally` block. Session-scoped (not the transaction-scoped `pg_advisory_xact_lock`) because migrations run with `transaction_per_migration=True` — each migration commits its own transaction, which would release a transaction-scoped lock mid-chain
- **Migrations bundled** in `src/khora/db/migrations/`, not `alembic/`. Root `alembic.ini` is dev-only
- **Skip-ahead:** When multiple services share a DB with different Khora versions, `run_migrations()` detects if the DB revision is unknown (ahead) and skips gracefully - returns `MigrationResult(success=True, skipped=True)`. Signaled via `_DatabaseAheadError` from `env.py` to `session.py`
- **Fresh-DB behavior:** On a PostgreSQL database with no `khora_alembic_version` table yet, `run_migrations()` / `Khora(run_migrations=True)` correctly creates all tables from scratch. Prior to v0.6.6, querying the missing version table inside an explicit transaction caused `InFailedSQLTransactionError`. The fix uses `information_schema.tables` to check table existence before querying it - never issuing a statement that could abort the transaction.
- **Dialect-gated migrations:** Some migrations are Postgres-only and skip silently on SQLite (the `sqlite_lance` test fixture stack runs the full chain). They check `op.get_bind().dialect.name == "postgresql"` before issuing Postgres-specific SQL. Current example: `029_chunks_created_at_brin` (v0.12.0) creates a BRIN index on `chunks.created_at` via `CREATE INDEX CONCURRENTLY` inside an autocommit block - Postgres-only feature, KB-sized, helps archive-side range scans.
- **Bi-temporal columns are ACTIVE on read (#888 -> #970 -> #1272):** `valid_to` / `invalidated_at` / `invalidated_by` on `entities` / `relationships` / `memory_facts` / `chronicle_events` (migrations 033/034) WERE reserved (write-only, never filtered) until the Neo4j tombstone-mirror landed in #1272. Now dream-apply stamps the PG soft-delete columns, a post-commit mirror (`DreamOrchestrator._mirror_dream_op`) folds them onto the graph `valid_until` via the #1271 verbs (eventual-consistency, reconciler-backed via `khora_dream_runs.graph_mirror_pending`), and the recall read paths filter both stores in lockstep: pgvector `list_entities`/`list_relationships` apply `_entity_live_filter()` (`valid_until` window) / `_relationship_live_filter()` (`valid_to` + `invalidated_at` + `valid_until`), and Neo4j `list_entities`/`list_relationships` filter `valid_until` unconditionally. A mirror failure increments `khora.dream.graph_mirror.partial_failure` (no namespace_id label) + appends a `Degradation` to `DreamResult.metadata`. Cross-store invariant guarded by `tests/integration/dream/test_neo4j_dream_mirror_integration.py`; the old reserve tripwire (`tests/unit/test_bitemporal_columns_reserved.py`) was widened then inverted. The endpoint-rewrite leg of dedupe (incident-edge re-pointing) is still #1273 - NOT mirrored here. See `docs/architecture/storage-backends.md`.
- **Embedded point-in-time (entity-version) recall degrades, it no longer raises:** the embedded backends (`sqlite_lance`, embedded SurrealDB) lack the `version_valid_from/to` columns, so a point-in-time `target_date` query cannot narrow entity versions. The VectorCypher retriever no longer fail-fasts with `NotImplementedError` - `_version_filter_entities` returns current-state entities unfiltered, records a structured `Degradation` (`component="vectorcypher.version_filter"`, `reason="embedded_no_version_columns"`, WARNING) on `RecallResult.engine_info["degradations"]`, and increments the internal `khora.vectorcypher.version_filter.degraded_total` counter. An **occurred-bounds** filter (`start_time`/`end_time` → `occurred_at`) is NOT point-in-time: it narrows chunks normally on the embedded path and records no degradation. The embedded VectorCypher temporal e2e lane (`tests/e2e/`, the `vc_embedded` matrix leg) is the regression guard.

### UUID & Type Handling
- **ORM:** all 52 UUID columns use `as_uuid=True` - native `uuid.UUID`, never `str()` wrap
- **Graph boundary:** Neo4j/Memgraph need `str(uuid)` at the driver boundary only
- **SurrealDB:** `RecordID` accepts UUID objects directly (no `str()` needed since SDK 2.0)

### Backend Specifics
- **Shared engine pools:** `StorageFactory` caches by URL. Shared-engine backends skip `dispose()`
- **Transactions:** `async with coordinator.transaction() as txn:` for atomic multi-backend ops
- **SurrealDB unified:** all four adapters share one `SurrealDBConnection`. Coordinator skips duplicate writes
- **SurrealDB schema:** declarative (`DEFINE IF NOT EXISTS`), auto-initializes on `connect()`. No Alembic
- **SurrealDB SDK:** pinned `>=2.0.0,<3.0` for 3.x support. Install: `pip install khora[surrealdb]`
- **SurrealDB KNN broken:** `<|K|>` unreliable in embedded mode. Uses brute-force cosine + HNSW instead
- **SurrealDB entity gate:** `_SurrealDBEntityKeyGate` serializes concurrent upserts by (ns, name, type) key
- **SurrealDB transactions (v0.12.0):** `SurrealDBConnection.transaction()` wraps the body in `BEGIN`/`COMMIT`/`CANCEL` on remote (`ws://`) mode and is a no-op on embedded / memory (surrealkv raises on `BEGIN`). `execute_batch([(sql, bindings), ...])` joins statements with `;` for an embedded-mode batched alternative; rejects parameter-name collisions. The coordinator's session-shaped `transaction()` does NOT yet route to SurrealDB - callers needing remote-mode atomicity use the connection-level primitive directly. See `docs/architecture/storage-backends.md#surrealdb-transactions-and-batching`.
- **Graph-less stacks list entities via vector backend (v0.12.0):** `StorageCoordinator.list_entities` / `list_relationships` fall back to the vector backend (`PgvectorBackend.list_entities` / `list_relationships`) when no graph backend is configured (chronicle on PG-only). Pre-#587 the expansion pipeline crashed with `AttributeError` on graph-less stacks; that path now returns the entity set from the pgvector tables. Relationships table on chronicle+PG-only is not actively written, so the relationships fallback returns `[]` rather than crashing.

### Extraction & Search
- **Session ID is a first-class column (v0.13.0+).** Migration 030 adds nullable `session_id UUID` to `documents`, `chunks`, `memory_events`, `chronicle_events`, and `memory_facts`. Migration 031 adds Postgres-only partial composite indexes `ix_chunks_ns_session` / `ix_documents_ns_session (namespace_id, session_id) WHERE session_id IS NOT NULL` plus a BRIN `ix_chunks_session_created_brin (session_id, created_at)` for time-bounded session replay. `Khora.remember(..., session_id=…)` and `Khora.submit_batch(..., session_id=…)` stamp the field; `Khora.forget_session(ns, sid)` is the cascade-delete API (FK + per-doc engine cleanup) and `khora.gc.expire_sessions(before=…)` is the opt-in TTL helper. See #620.
- **Pre-normalized embeddings** - L2-normalized at ingest. Uses `batch_dot_product` (3x faster than cosine)
- **Entity unique constraint** - `(namespace_id, name, entity_type)` UNIQUE in both PostgreSQL and SurrealDB
- **Namespace versioning** - dual IDs: `id` (row-level) vs `namespace_id` (stable). Public API uses `namespace_id`, resolves to `id` via indexed lookup
- **Cross-encoder reranking** - optional reranking via cross-encoder models (cached, runs in asyncio.to_thread)
- **Temporal SQL pushdown** - relative date queries ("last 7 days") pushed to SQL WHERE clauses
- **Selective extraction** - KET-RAG style: scores chunk importance, sends top 70% to LLM, rest get co-occurrence edges only
- **Entity resolution** - multi-strategy dedup with per-type thresholds (PERSON 0.92, DATE 0.95, default 0.85)
- **Semantic expansion** - optional cross-tool entity unification + relationship inference (4 modes: smart/batch/incremental/none)
- **Chronicle abstention signals** - `RecallResult.metadata["abstention_signals"]` exposes 4 boolean flags (`entities_empty`, `chunks_empty`, `chunks_below_min`, `top_score_low`), a weighted `combined_score` (0.0 high-confidence → 1.0 should-abstain), and a `should_abstain` convenience flag for downstream LLM answer-generation. Passive signals - chronicle still returns chunks even when they trip. Tunable via `ChronicleEngine` kwargs `abstention_min_chunks`, `abstention_min_top_score`, `abstention_combined_threshold`.
- **Temporal-anchored HyDE (v0.12.0)** - when HyDE fires (`enable_hyde` in `auto`/`always`) on a query the temporal detector flags as RECENCY / STATE_QUERY / CHANGE, `HyDEExpander` selects a system prompt that anchors the hypothetical to today's date with explicit dates / weekdays / relative markers. Other categories use the generic time-blind prompt. Zero additional LLM calls - only the prompt string changes. Category detection runs in Rust Aho-Corasick (sub-ms).
- **HyDE-Cypher (v0.12.0, opt-in)** - `khora.query.hyde_cypher` module: LLM picks a parameterized Cypher template (`recent_by_type`, `entity_relationships`, `cooccurrence`) and fills slots. Cypher source is static; slot values bind via Neo4j `$placeholder` parameters and are validated against `ExpertiseConfig` whitelists. Default OFF behind `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`. Failures (timeout, hallucinated id, validation error) degrade to text-HyDE - never crashes the query.
- **Cross-encoder date-prefix experiment (v0.12.0, opt-in)** - `CrossEncoderReranker(include_date_prefix=True)` prepends `[YYYY-MM-DD] ` to each candidate before scoring. Source priority: `metadata.custom.occurred_at` → `metadata.custom.sent_at` → `metadata.created_at`. Default OFF. The reranker cache is keyed by `(model, include_date_prefix)` so the two variants coexist without a 500ms model reload.

### Optional Dependencies
- **spaCy:** `_HAS_SPACY` flag, falls back to regex sentence splitting
- **Logfire:** `_HAS_LOGFIRE` flag, `trace_span()` yields no-op when absent. Install: `pip install khora[logfire]`
- **`@trace` decorator:** `from khora.telemetry import trace`. Zero overhead when logfire absent
- **Telemetry collector:** `KHORA_TELEMETRY_DATABASE_URL` enables PostgreSQL-backed event recording. Without it, `NoOpCollector` is used (zero cost)
- **Neo4j pool metrics:** When logfire is installed, `Neo4jBackend` emits OTel metrics. **Use `khora.neo4j.pool.acquire_duration`** (histogram, seconds) for alerting - it records the real time until a connection is bound to the session, wrapped around `AsyncSession._connect` so retries and queries don't inflate it. Also emitted: `khora.neo4j.pool.timeout` (counter, increments on every `ConnectionAcquisitionTimeoutError` from any entry path), `khora.neo4j.pool.connections.{active,idle,total,creating}` + `khora.neo4j.pool.utilization` (observable gauges, ~60s export cadence - best-effort, unlocked reads), and `khora.neo4j.session.duration` (histogram, total session hold time). **Legacy:** `khora.neo4j.pool.acquisition_time` records session-object construction only (near-zero) - kept for dashboard back-compat; prefer `acquire_duration`. **Opt-in high-frequency sampler:** set `KHORA_STORAGE_GRAPH_POOL_SAMPLER_ENABLED=true` (+ optional `KHORA_STORAGE_GRAPH_POOL_SAMPLER_INTERVAL_MS=500`, clamped to [50, 60000]) to emit `khora.neo4j.pool.sampled.{active,idle,total,creating,utilization}` histograms for sub-minute burst/ramp investigation - sampler takes `pool.lock` for a short critical section. Zero cost without logfire, zero cost when sampler disabled. Pool internals (`driver._pool.connections`, `connections_reservations`, `in_use_connection_count`, `lock`) verified stable neo4j 5.x–6.2; all reads degrade gracefully via `getattr` fallback if internals shift.
- **Hooks Level 2 LLM cost controls (v0.12.0):** the Level 2 evaluator is **default-OFF** (`KHORA_HOOKS_LLM_EVALUATION_ENABLED=false`) and gated by two independent rolling-hour token budgets: per-namespace (`llm_max_tokens_per_namespace_per_hour`, default 10k) and per-subscription (`llm_max_tokens_per_subscription_per_hour`, default 0 = disabled). Either breach fails open and fires `khora.hooks.llm.throttled_total`. Cross-batch decision cache keyed on `(filter_id, bounded_text_hash(event_summary))` short-circuits repeats; intra-batch coalescing dedupes by event_summary_hash before building the prompt so a burst of 50 identical events spends 1 prompt slot. Cache + budget knobs: `llm_cache_size`, `llm_cache_ttl_seconds`, `llm_max_tokens_per_subscription_per_hour`. Metric names: `khora.hooks.llm.{evaluations,tokens,throttled,cache_hits,cache_misses}_total` (see `docs/telemetry-contract.json`).
- **khora.diagnostics (v0.12.0):** one-shot reporter package - currently houses `compute_graph_stats` / `GraphStats` for the PPR decision gate (#598). **Explicitly NOT stable public API** - may be renamed or removed without a major-version bump. Use through `scripts/audit_graph_density.py`.
- **khora.integrations (v0.13, #619):** adapter foundation for agentic frameworks. Three runtime-checkable Protocols (`MemoryAdapter`, `RetrieverAdapter`, marker `KhoraIntegration`) + entry-point registry (group `khora.integrations`, with `register()` test escape hatch) + `_sync.run_sync` bridge (raises if called from inside a running loop - that's the deadlock surface) + `Khora.shared()` process-wide singleton (cached by config hash, `await Khora.shared.clear()` for tests). Adapter submodules MUST NOT import their framework at top level - enforced by `tools/check_optional_imports.py` (AST lint, run in the CI lint job and via `make lint`).
- **Fork-safety contract (#790):** the integrations subsystem is fork-safe. Both `khora.integrations._sync` (the daemon-thread bridge) and `Khora.shared()` (the cached-instance dict) register `os.register_at_fork(after_in_child=...)` handlers that drop parent state in forked children - the next `run_sync` / `Khora.shared()` call from the child transparently rebuilds the bridge / re-instantiates Khora. Standard production deployment shapes that fork after import (gunicorn pre-fork, multiprocessing.Pool, Celery prefork, uvicorn `--workers N` on platforms that pre-fork) are safe. The handler does NOT close parent asyncpg connections from the child (same fds; would corrupt the parent) - it just discards references. POSIX-only; Windows is a no-op (no `os.register_at_fork`).
- **CrewAI adapter:** `pip install khora[crewai]` enables `khora.integrations.crewai.KhoraMemory`. See `docs/integrations/crewai.md`.
- **LangGraph adapter:** `pip install khora[langgraph]` enables `khora.integrations.langgraph.KhoraStore` (semantic long-term memory). See `docs/integrations/langgraph.md`.
- **Google ADK adapter:** `pip install khora[google-adk]` enables `khora.integrations.google_adk.KhoraMemoryService` (drop-in for ADK's `BaseMemoryService` - namespace is UUID5 of `adk:{app_name}:{user_id}`, session-id round-trips via #620, Memory only, no `KhoraSessionService` in v1). See `docs/integrations/google_adk.md`.
- **OpenAI Agents SDK adapter (#625):** `pip install khora[openai-agents]` enables `khora.integrations.openai_agents.KhoraSession` (full `agents.memory.session.SessionABC`), `khora_recall_tool` (`FunctionTool` factory), and `KhoraMemoryHooks` (`RunHooks`-shaped, auto-persists tool results). Items round-trip via `Document.metadata.custom["oai_item"]` (verbatim JSON); ordering via `oai_seq`. Pinned tight (`openai-agents>=0.17,<0.18`) - SDK is pre-1.0 with 17 releases in 7 months; bump in a deliberate PR per upstream minor. See `docs/integrations/openai_agents.md`.
- **LlamaIndex adapter:** `pip install khora[llamaindex]` enables `khora.integrations.llamaindex.KhoraRetriever` (async-only `BaseRetriever`), `KhoraMemoryBlock` (long-term memory block for `llama_index.core.memory.Memory`), and `KhoraChatStore` (deprecated `BaseChatStore` for legacy `ChatMemoryBuffer` users). Pin is narrow (`llama-index-core>=0.14,<0.15`) - LlamaIndex breaks on minor bumps. See `docs/integrations/llamaindex.md`.

### Logging
- **loguru sinks are sync by default** - `logger.add(...)` has `enqueue=False`. In async code, `logger.*` calls then block the event loop on each format+write. Khora's `setup_logging()` enables `enqueue=True` on all sinks it installs, and registers `atexit.register(logger.complete)` so the queue drains on clean exit.
- **Downstream code MUST either** (a) call `khora.logging_config.setup_logging()`, OR (b) configure its own loguru sinks with `enqueue=True` explicitly. If a downstream service imports khora without doing either, it inherits loguru's default sync stderr sink and silently pays event-loop-blocking cost on every `logger.*` call inside an `async def`.
- **Graceful shutdown drains via `logger.complete()`** - setup_logging registers this via atexit. Downstream code that configures its own sinks must do the same, otherwise in-flight queue entries are lost on exit.
- **Abrupt termination (SIGKILL, crash) drops in-flight log records.** This is inherent to the enqueue model - the queue is drained by a background thread that can't run during a kill.
- **loguru queue is unbounded in 0.7.3.** Under sustained burst (DEBUG mode + error storm + slow sink), records accumulate faster than the background thread can drain them and eventually OOM the process. Napkin math: ~1 KB/record × ~9k records/s net accumulation → 512 MB pod OOMs in ~60s (INFO) or ~3s (DEBUG with cascading errors). loguru 0.7.3 does not expose a `maxsize` kwarg on `logger.add()`. Mitigation: keep log volume bounded by request rate (avoid unbounded DEBUG in prod); watch for `MemoryError` in low-memory containers. Revisit if loguru exposes `maxsize` upstream.
- **`enqueue=True` is not a free latency win.** See `scripts/bench_logger_enqueue.py`: on fast buffered sinks, the pickle + IPC overhead dominates a userspace memcpy, so enqueue is ~5× slower per call than sync. On slow sinks with sustained throughput (no idle between bursts), enqueue does not help either - the kernel pipe fills and the producer blocks. Enqueue wins only in the realistic case: slow sink + idle between bursts (a request handler doing async I/O between log calls), where p99 event-loop stalls drop ~25-40% (≈15-18 ms → ≈11 ms, run-dependent) and wall time drops ~23% (2058 ms → 1593 ms) on the handler-shaped scenario. Keep this in mind when evaluating logging overhead on hot paths.

### Telemetry
- **Public contract lives at `docs/telemetry-contract.json`** (with sibling explainer `docs/telemetry-contract.md`). When you add a span (`trace_span`), pipeline stage (`pipeline_stage` / `record_pipeline_stage`), metric (`metric_counter` / `metric_histogram` / `metric_gauge_callback`), event-type field, or new public export to `khora.telemetry.__all__`, you MUST update the contract JSON in the same PR. CI fails otherwise via `tests/unit/telemetry/test_contract.py` (10-test drift gate that walks the codebase with ripgrep).
- **Public vs internal stability tags.** Items tagged `stability: public` in the contract are part of the OSS API surface - renaming or removing them requires a major version bump and prior coordination through the CHANGELOG / release-notes process. Items tagged `internal` may be renamed freely as long as the JSON is updated. Top-level engine entry points (`khora.recall`, `khora.remember`, `khora.vectorcypher.retrieve`) and operator-facing metrics (`khora.memory.recall.duration`, `khora.llm.tokens`, etc.) are public. Inner-loop spans (`khora.vectorcypher.coherence_boost`, `khora.vectorcypher.rrf_fusion`, etc.) are internal.
- **Cardinality rule - never put `namespace_id` on a metric.** It is a span attribute and a log field only. Phase-0 audit measured 438 distinct namespace IDs over the production retention window in one deployment; Logfire and Prometheus bill per series, so a `namespace_id` label produces an unbounded cost curve. The same rule applies to any other attribute with cardinality ~O(tenants).
- **Free-text span attributes.** Use `khora.telemetry.bounded_text_hash` (added in #504) for any free-text value (raw user query, document content, chunk text) - it returns a SHA1[:8] hash. Never put raw text on a span attribute: it is both a privacy hazard and a cardinality bomb.
- **OTel semconv adopted for new attributes.** `gen_ai.*` for LLM (model, prompt tokens, completion tokens), `db.*` for storage backends, `code.*` for stack info. Keeps khora vendor-neutral over the OTel exporter chain.
- **Two existing LLM instrumentation patterns - pick whichever the surrounding code uses, do not introduce a third.** (a) Pass `_telemetry_op="<op>"` through `khora.config.llm.acompletion`; the wrapper records the call automatically. (b) Inline `record_llm_call(...)` after a direct `litellm.acompletion` call. The 9 call sites that existed before #508 plus the 6 added in #508 (HyDE, listwise rerank, fact extraction, fact reconcile, event extraction; chat was already wired) all follow one of these two patterns.
- **Rate-limit + connector span attributes.** `khora.embedder.litellm_request` and `khora.extraction.llm_call` carry provider rate-limit attributes (`ratelimit.remaining_requests`, `ratelimit.remaining_tokens`, `ratelimit.reset_requests`, `ratelimit.reset_tokens`, `retry_after`) read off the response *after* it returns, and connector-contention attributes (`connector.in_use`, `connector.available`, `connector.queued`, `connector.limit`, `connector.limit_per_host`) snapshotted off the shared aiohttp session *before* the request awaits (so they reflect acquisition-time contention, not idle post-release state). Both families live in `extraction/embedders/_request_telemetry.py` and use the same "best-effort, unlocked reads" pattern as the Neo4j pool gauges: defensive `getattr`/`isinstance` reads that log at DEBUG on any error and never raise into the request path, degrading gracefully if litellm header shapes or aiohttp connector internals (`_acquired`/`_conns`/`_waiters`) shift. Span attributes only - no new span name and no metric, so the contract drift gate is unaffected. `x-ratelimit-limit-requests` is parsed for the DEBUG `{remaining}/{limit}` log line but kept off the span via an allowlist.
- **`khora.log.queue.depth` is a proxy.** It exports the loguru-handler-error count, not the real enqueue-queue size - `loguru>=0.7.3` does not expose `qsize()`. The metric *name* is in the public contract because dashboards depend on it; the implementation can switch to a real reading when loguru exposes one. Do not "fix" this by removing the metric.
- **`coordinator.transaction()` cross-store atomicity remains partial on embedded.** Span instrumentation does not promise transactional semantics it does not have - do not annotate spans in a way that implies all-or-nothing writes across SQL + LanceDB on the embedded path.
- **Telemetry collector is opt-in.** `KHORA_TELEMETRY_DATABASE_URL` enables PostgreSQL-backed event recording; without it, `NoOpCollector` is used (zero cost). Logfire integration is gated by `_HAS_LOGFIRE` - `trace_span()` yields a no-op when the optional `logfire` extra is absent.

### Downstream
- `kb.storage` is a stable public API.
- **LLMUsage contract:** `LLMUsage` fields are part of the stable public API; breaking changes are recorded in CHANGELOG.md.
- **ExpertiseConfig contract:** stable API - changes to `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig`, `ConfidenceConfig`, `ExpansionConfig`, `CorrelationRule`, `InferenceRule` are recorded in CHANGELOG.md. `__all__` in `src/khora/extraction/skills/base.py` is the machine-readable contract.
- Any breaking change to the stable public API is recorded in CHANGELOG.md. `__all__` in `src/khora/__init__.py` is the machine-readable contract for the top-level surface.
