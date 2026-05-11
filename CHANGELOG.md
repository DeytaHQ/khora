# Changelog

All notable changes to Khora are documented here.

Format: versions match git tags (`git tag vX.Y.Z`). Versions before 0.5.1 were internal (no git tags).

## [0.10.0] — Rename `MemoryLake` → `Khora`, drop "Memory Lake" branding

### Changed — BREAKING

- **`MemoryLake` → `Khora`.** The top-level facade class is now `Khora`.
  Import path: `from khora import Khora` (was `from khora import MemoryLake`).
  Submodule path: `khora.khora.Khora` (was `khora.memory_lake.MemoryLake`).
- **Module rename:** `src/khora/memory_lake.py` → `src/khora/khora.py`.
  All public result types (`RememberResult`, `RecallResult`, `BatchResult`,
  `Stats`, `LLMUsage`, `DocumentResult`) moved with it and remain importable
  from `khora` at the top level (preferred) or `khora.khora` (submodule).
- **No deprecation shim.** `from khora import MemoryLake` and
  `from khora.memory_lake import …` raise `ImportError` on 0.10.0+.
  Downstream consumers (`genesis`, `khora-benchmarks`) migrate via
  coordinated PRs after this release (DYT-3967, DYT-3968). Both
  pre-pinned `khora<0.10` (DYT-3969) to avoid Renovate auto-bumps.
- **LLM prompts** in `khora.query.understanding` now say "knowledge base"
  instead of "memory lake" (`COMPREHENSIVE_UNDERSTANDING_PROMPT`,
  `LIGHTWEIGHT_UNDERSTANDING_PROMPT`). Sent verbatim to the LLM.
- **Telemetry `owner` field** changed on four facade spans
  (`khora.recall`, `khora.remember`, `khora.remember_batch`, `khora.forget`):
  `"owner": "memory_lake"` → `"owner": "khora"`. Internal grouping label;
  not part of the public stability contract.

### Unchanged (deliberately)

- `khora.memory.*` metric names (`khora.memory.recall.duration`,
  `khora.memory.ingest.duration`) — generic concept of memory storage,
  not the retired brand. Operator dashboards keep working.
- `khora_alembic_version` table name and advisory lock id
  `6001515088189075507` — schema continuity.
- SurrealDB `memory_namespace` table name — unrelated to the brand.
- `KHORA_` env-var prefix.

### Migration

```diff
- from khora import MemoryLake
+ from khora import Khora

- async with MemoryLake(config) as lake:
+ async with Khora(config) as lake:
      ...
```

See ADR-027 for rationale and ADR-024 (revised) for the full public-API
contract.

## [Unreleased] — Recall API time bounds honored

### Fixed — API-supplied `temporal_filter` no longer dropped (DYT-3605)

Callers passing an explicit `temporal_filter` to `MemoryLake.recall()` had their bounds silently bypassed in two places:

* **vectorcypher**: when `temporal_filter` was provided, the auto-detection branch (which also synthesizes the `TemporalSignal` consumed by the retriever's skip-fallback / version-filter / recency-weighting logic) was skipped entirely. The retriever then saw `temporal_signal=None` and applied none of the temporal-aware behavior the caller had asked for, including the sparse-results fallback that re-runs the search without the time bound.
* **graphrag**: same shape — the auto-detect block was guarded by `temporal_filter is None`, so the resulting `RecallResult.metadata` was missing `temporal_category` / `temporal_confidence` for API-bounded queries.

Both engines now synthesize an `EXPLICIT`-category `TemporalSignal` with `confidence=1.0` and `source="api"` when `temporal_filter` is supplied, so downstream behavior is consistent regardless of whether the bounds came from the caller or were detected from the query string. The new `source="api"` value joins the existing `"dictionary"` / `"semantic"` / `"none"` set on the `temporal_detect` span — telemetry contract unchanged (span attributes are not enumerated). graphrag's `apply_recency_bias` remains untouched on the API path: `EXPLICIT` does not match the `RECENCY`/`STATE_QUERY` guard, preserving existing behavior for API callers.

---

## [Unreleased] — Connector throughput restoration

### Performance — restore pre-0.9.0 LiteLLM throughput (DYT-3599)

The shared aiohttp session introduced in DYT-3156 (v0.9.0) was created with hard-coded `TCPConnector(limit=20, limit_per_host=10)`. `limit_per_host=10` silently throttled all OpenAI / Anthropic / etc. requests to 10 in flight per host, regardless of caller-configured concurrency. Downstream services (e.g. Genesis with `max_concurrent_llm_calls=200`) regressed ~5–20× on wall-time after upgrading to 0.9.x because the shared session became the dominant ceiling on parallel LLM/embedding calls.

The connector is now configurable through `LiteLLMConfig` and `LLMSettings`:

| Field                       | Default | aiohttp arg            |
|-----------------------------|---------|------------------------|
| `max_total_connections`     | 200     | `limit`                |
| `max_connections_per_host`  | 0 (unlimited) | `limit_per_host` |
| `keepalive_timeout_s`       | 30.0    | `keepalive_timeout`    |

Defaults restore pre-0.9.0 throughput: total cap is generous, no per-host throttle. Fields are read by `_init_shared_session` from a cache populated by `configure_litellm` (first-call-wins; subsequent calls with non-matching connector settings log a warning and are ignored).

**Migration call-out** — anyone who relied on the v0.9.0 connector throttle as a budget brake or rate-limit circuit-breaker should set `max_connections_per_host` explicitly in YAML / env (`KHORA_LLM_MAX_CONNECTIONS_PER_HOST`). On Anthropic Claude tier 1 in particular, an unlimited per-host connector combined with extraction loops can produce 429 storms that the previous 10-cap masked. Pick a value that matches your provider tier rather than relying on the connector for backpressure.

### Out of scope (related but tracked separately)

* DYT-3079's `_bisect_and_extract` issues up to 2N LLM calls when truncation is detected — amplifies any concurrency change downstream. Not touched here.
* DYT-3305's unified pending processor spawns 20 background workers on every `MemoryLake.connect()` even for engines that never call `submit_batch`. Idle but not free. Not touched here.

---

## [Unreleased] — Telemetry Public Surface, OSS Observability Contract

Telemetry workstream (PRs #504–#509) shipped after the v0.9.1 tag. It hardens cardinality safety, codifies the public observability surface as a JSON contract enforced by a CI drift gate, fixes a silent regression that had been zeroing out `storage_events.namespace_id` since February 2026, and broadens metric coverage. See [ADR-026](docs/adrs/adr-026-telemetry-contract.md) for the design rationale and the OSS implication: public telemetry names are now API and break the same way any other public symbol does.

### Added

- **Public observability contract.** `docs/telemetry-contract.json` lists every public export in `khora.telemetry.__all__` (19 names), every `LLMEvent` / `StorageEvent` / `PipelineEvent` field, all 22 collector-recorded pipeline stages, all 58 `trace_span(...)` call sites (22 public, 36 internal), and all 21 metrics (16 public, 5 internal). `docs/telemetry-contract.md` is the human-facing explainer. `tests/unit/telemetry/test_contract.py` (10-test drift gate) walks the codebase via ripgrep and fails CI on any undeclared instrumentation. (#505)
- **`khora.telemetry.bounded_text_hash`.** Helper that turns free-text span attributes (raw query, document content, chunk text) into a SHA1[:8] hash — caps cardinality and removes the privacy hazard of raw text on spans. Now used at the four query / extraction sites that previously emitted raw text. (#504)
- **Chronicle abstention metrics.** `khora.chronicle.abstention_signal` (counter, public) and `khora.chronicle.abstention_combined_score` (histogram, public) aggregate the four boolean abstention signals + combined score that `RecallResult.metadata["abstention_signals"]` exposes per call, so abstention rate and confidence distribution can be tracked at fleet scale instead of only inspected per-request. (#507)
- **Aggregate operator metrics.** `khora.memory.recall.duration` (histogram, public, seconds), `khora.memory.ingest.duration` (histogram, public, seconds), `khora.llm.tokens` (counter, public), `khora.llm.cost_usd` (counter, public), `khora.log.queue.depth` (gauge, public, proxy via handler-error count — loguru 0.7.3 does not expose `qsize()`). (#509)
- **Six additional LLM call sites instrumented.** HyDE, listwise rerank, fact extraction, fact reconciliation, event extraction now record `LLMEvent` rows; chat was already wired. Two patterns coexist (`_telemetry_op="..."` through `khora.config.llm.acompletion` vs. inline `record_llm_call` after direct litellm calls); both are documented in `CLAUDE.md`. (#508)

### Fixed

- **`storage_events.namespace_id` 100% NULL since Feb 2026.** Restored namespace propagation through the storage telemetry path. The break had survived multiple releases because no operator dashboard was reading the column — DYT-3398 surfaced it during the Phase-0 audit. (#506)

### OSS implication

- Names tagged `public` in `docs/telemetry-contract.json` are now part of khora's public API. Renames or removals require a major version bump and prior coordination with genesis, khora-benchmarks, khora-explorer, and khora-cli. Names tagged `internal` (e.g. inner-loop spans like `khora.vectorcypher.coherence_boost`) may be renamed freely as long as the JSON is updated in the same PR.
- New attributes follow OTel semantic conventions: `gen_ai.*` for LLM, `db.*` for storage, `code.*` for stack info.
- The contract enables the operator-dashboard work that follows; it does not by itself fix the under-utilisation. Telemetry has been collected to PostgreSQL since 0.4.0, and dashboards / alerts that consume those events remain TODO.

---

## [0.9.0] — 2026-05-02 — Embedded Backend Realignment, Production-Readiness Scoping

### Embedded backend overhaul (DYT-3545 family)

The v0.9.0 embedded path lands as a complete-but-experimental SQLite + LanceDB stack covering all four engines (VectorCypher, GraphRAG, Skeleton, Chronicle). Engine × embedded integration tests now exist for all four engines; the prior "unverified embedded code path" gap from the audit is closed. See [ADR-025](docs/adrs/adr-025-embedded-backend-realignment.md) for the strategic rationale.

**Production-readiness scoping (per stack, not per engine).** Stamping is now per `(engine × storage stack)`:

- **VectorCypher** — production-ready on **PostgreSQL + pgvector + Neo4j** only.
- **Chronicle** — production-ready on **PostgreSQL + pgvector** (no graph DB required).
- **GraphRAG** and **Skeleton** — available; same PG-based stacks.
- **SQLite + LanceDB** for any engine — **experimental**. Documented scale ceiling: ~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3.
- **SurrealDB** for any engine — **experimental**. Python SDK on alpha track (`>=2.0.0a1`); KNN unreliable in embedded mode (brute-force cosine + HNSW fallback).

See [docs/engines/engine-comparison.md](docs/engines/engine-comparison.md#production-readiness-by-stack-v090) for the full matrix.

### Embedded engine wiring

- DYT-3560 (#482): VectorCypher wired to the `sqlite_lance` backend.
- DYT-3561 (#481): Skeleton wired to the `sqlite_lance` backend with a temporal-store adapter.
- DYT-3562: GraphRAG embedded path pushes the temporal filter into the LanceDB WHERE — was previously post-hoc and xfail-pinned.
- DYT-3578 (#486): Temporal filter pushed into SQLite-side WHERE in the GraphRAG embedded chunk fetch path.
- DYT-3581: VectorCypher honours `metadata['occurred_at']` on the embedded path (parity with `remember_batch`).

### Embedded retrieval correctness

- DYT-3547: Chronicle channels (BM25 / semantic / temporal / entity) now share the same `created_after`/`created_before` bounds — fixes channel divergence that broke RRF fusion.
- DYT-3548: Recursive-CTE graph traversal switched from node-visited to edge-visited tracking (mirrors Neo4j `MATCH [*1..N]`).
- DYT-3549: `valid_until > now` filter inlined into both anchor and recursive arms of the CTE.
- DYT-3555 / DYT-3556: Skeleton tag-cast and `occurred_at` parsing fixes (`Skeleton.remember()` parity with `remember_batch()`, DYT-3557).
- DYT-3558: Embedded compensating-delete-on-failure logging hardened.
- DYT-3579 (#485): LanceDB IVF-PQ index now retrains once the corpus grows past `retrain_factor × (rows at last training)`. Configurable via `KHORA_STORAGE_SQLITE_LANCE__RETRAIN_FACTOR` (default `2.0`). Fixes silent recall degradation as the corpus grows past the initial training threshold (5k rows). Set ≤ `1.0` to disable.

### Embedded warts (documented, not fixed)

- **Partial atomicity in `coordinator.transaction()`** on embedded — only the SQL session is enrolled; LanceDB writes happen post-commit with compensating deletes.
- **DYT-3550**: Point-in-time queries are not supported on the embedded stack. The CTE port does not implement PIT semantics. Tracked.
- **FTS5 on chunks only** — entity-anchored recall falls back to `LIKE` / JSON-equality on embedded. Use the PostgreSQL stack for entity-heavy corpora.

### Deprecated

- **Kuzu graph backend** (`khora[kuzu]`) — deprecated in 0.9.0, scheduled for removal in 0.10. Kuzu was acquired by Apple in October 2025 and the upstream repository is archived. Migrate to SQLite + LanceDB (embedded) or PostgreSQL + Neo4j (production).

### v0.10 roadmap

ADR-025 enumerates two deferred decisions for v0.10 to address the embedded warts:

- **sqlite-vec** as a candidate to collapse the SQLite + LanceDB dual-store into a single in-SQLite-transaction vector store (eliminates partial atomicity, drops install footprint from ~150 MB to ~5 MB).
- **`pgserver` (embedded Postgres)** as a candidate for true production-parity embedded mode (HNSW recall, real ACID, zero schema fork).
- **Default embedded URI routing** — currently `MemoryLake("memory://")` treats the URL as the PostgreSQL `database_url`; SurrealDB owns the `memory://` scheme internally. Routing a top-level `memory://` URI to the recommended embedded stack is a v0.10 code change.
- **`lance-graph` integration** is explicitly **deferred to v0.10** — no second 0.x Rust crate enters a "production-ready" path in v0.9.0.

---

## [Unreleased] — Graph Backends, Temporal Precision, Discovery Agent Overhaul

### Added
- ADR-024 codifying the khora public API surface consumed by downstream packages (genesis, khora-benchmarks, khora-explorer, khora-cli). See `docs/adrs/adr-024-memory-lake-public-api.md`.

### Removed
- `khora` console script and CLI subcommands (`extract`, `search`) — moved to [khora-cli](https://github.com/DeytaHQ/khora-cli). Install with `uv pip install khora-cli` and run `uv run khora-cli extract` / `uv run khora-cli search`.
- `khora ontology` CLI subcommands (moved to khora-explorer)
- `khora.discovery` package (moved to khora-explorer)
- `khora.cli` package (entire subtree — `extract`, `search`, `_common`)
- `click` and `rich` dropped from core dependencies (only the CLI used them)

### Changed
- **Breaking**: khora is now a pure memory-lake library. `uv run khora ...` is no longer a valid command; use `uv pip install khora-cli` and `uv run khora-cli extract` / `search` instead.
- `khora.discovery.extraction` → `khora.extraction.binary_readers` (binary file reader consumed by khora-cli)
- Documentation rework post-extraction (DYT-2714): short, library-focused `README.md`; new `docs/README.md` index, `docs/configuration.md`, `docs/api-reference.md`, `docs/migrations.md`, and `docs/consumers.md`; removed stale `khora extract` / `khora search` / `khora ontology` references from the top-level docs in favour of pointers to `khora-cli` and `khora-explorer`.

### New graph backends
- AWS Neptune with Bolt protocol + IAM SigV4 auth (#272)
- PostgreSQL AGE with Cypher-in-SQL, shares PG connection pool (#273)

### Retrieval quality
- Cross-encoder reranking integrated into VectorCypher and Chronicle (#236, #314)
- Version-aware scoring penalizes superseded document versions (#319, #328, #344)
- Independent BM25 channel in VectorCypher retriever (#276)
- Temporal SQL WHERE pushdown for relative dates ("last 7 days") across all engines (#316)
- LLM temporal reranking for top-5 after cross-encoder (#311)
- Entity semantic gate filters low-relevance entity-adjacent chunks (#314)
- Timestamp collapse detection for batch-ingested content (#314)
- Session-aware parallel retrieval for cross-session temporal queries (#279)

### Chronicle epic (#444, #446, #447, #448, #449, #450, this PR)
- #1 (#444): events / facts schema + per-namespace toggle
- #2 (#447): EventExtractor wired into `remember()` / `remember_batch()`
- #3 (#448): FactExtractor with ADD / UPDATE / DELETE / NOOP reconciliation
- #4 (#449): temporal channel queries `chronicle_events.referenced_date` for cross-session entity resolution
- #5 (#450): direct entity-channel hits + temporal-event subjects surfaced in `RecallResult.entities`
- #6 (this PR): `QueryComplexityRouter` skips BM25 + entity channels for SIMPLE queries (temporal channel always preserved); fusion swapped to weighted RRF with per-channel min-max score normalisation to neutralise the BM25-vs-cosine score-scale mismatch. New `router_enabled` constructor flag (default `True`); router relocated to `khora.query.router` with a back-compat shim at `khora.engines.vectorcypher.router`.
- #7 (#446): LanceDB vector store as an alternative chunk-vector backend

### Discovery agent overhaul
- 6-phase improvement: bug fixes, model hierarchy, retry resilience, semantic relevance, Chronicle memory, multi-step exploration (#256-#263)
- LiteLLM YAML config for per-task model selection (#258)
- .docx and .parquet extractors (#263)
- Firecrawl fallback on HTTP 403 (#267)

### Security
- Cypher injection fixes in Neptune, Memgraph, AGE backends (#281)
- KhoraError exception hierarchy with domain-specific exceptions (#282)

### Performance
- Cross-encoder model caching — avoid reloading per query (#270, #340)
- asyncio.to_thread for reranker inference (#317)
- Column projection excludes embeddings from search results (#317)
- ef_search at connection level (#317)
- Parallel Chronicle channels (#301)
- CI parallel jobs — 50% faster feedback (#335)

### Configuration
- Default engine changed from graphrag to vectorcypher
- LLM max_tokens default 8192 to 12288 (#333)
- Configurable extraction_batch_size (#339)
- Neo4j query timeout configuration (#293)

### Bug fixes
- PDF extraction "document closed" error (#267)
- Neo4j credential passthrough (#294)
- Async logging with enqueue=True (#295)
- Advisory lock for entity upsert deadlocks (#341)
- JSON repair for malformed LLM responses (#338)
- Markdown fence stripping from extraction responses (#336)

### SurrealDB optimizations
- Graph traversal depth cap raised from 3 to 6 hops (#265)
- Single-query temporal neighbors (N queries to 1) (#265)
- Batch relationship fetch in get_neighborhoods_batch (#265)
- Graph traversal indexes on in/out fields (#265)

---

## [0.7.0] — 2026-04-02 — Chronicle Engine, Semantic Hooks, Retrieval Quality

### Chronicle engine (new)

- 4th memory engine optimized for temporal/conversational memory (LongMemEval, LoCoMo, BEAM) (#199, #200)
- 4-channel parallel retrieval: semantic + BM25 + temporal decay + entity co-occurrence
- Ebbinghaus forgetting curve for temporal decay scoring
- Event decomposition into SVO tuples with triple timestamps (observation, referenced, relative)
- Progressive memory compression with contradiction detection (ADD/UPDATE/DELETE/NOOP)
- LanceDB embedded vector store option — file-backed, no server (`pip install khora[lancedb]`)
- Rust-accelerated temporal scoring via khora-accel
- No graph database required — PostgreSQL + pgvector only

### Semantic hooks

- Event subscription system: `lake.subscribe("entity.created", callback)` (#193)
- `SemanticFilter` with 3-level cascade: type pre-filter (free) → embedding similarity (sub-ms) → LLM yes/no (#194)
- `HookDispatcher` with async concurrent dispatch and failure isolation
- Binary-quantized embedding cache for sub-microsecond pre-screening (Hamming distance)
- Configurable via `KHORA_HOOKS_ENABLED`, `KHORA_HOOKS_FILTER_MODEL` env vars
- Wired into ingestion pipeline — fires during entity/relationship extraction

### Retrieval quality

- Expose 16 previously hardcoded scoring parameters as configurable `QuerySettings` fields (#206)
- Entity linking thresholds tuned: fuzzy 0.6→0.5, max_candidates 5→10
- Fix halfvec HNSW indexes causing full sequential scans (#183)

### CI & infrastructure

- Upgrade CI actions to Node.js 24 (actions/checkout v6, setup-python v6, codecov v5) (#197)
- Add pip-audit dependency vulnerability scanning
- Upgrade aiohttp 3.13.3→3.13.5 (CVE-2026-22815)
- Upgrade all dependencies to latest compatible versions
- Fix ty 0.0.27 type errors from dependency upgrade
- Rust edition 2021→2024, minimum rustc 1.83→1.85 (#209)

### Bug fixes

- Fix `run_migrations()` on fresh PostgreSQL database — use `information_schema.tables` (#201, DYT-1447)
- Reduce extraction batch size from 10 to 5 and make configurable (#195)
- Move per-document extraction log lines from INFO to DEBUG (#196)

### Version

- khora-accel 0.7.0, Rust edition 2024 (#209)

---

## [0.6.0] — 2026-03-28 — SurrealDB Optimization, Ontology CLI, Discovery Agent

### SurrealDB backend hardening

- Schema parity: added ~50 missing fields to match PostgreSQL ORM (#130, #131)
- SQL injection: replaced 15+ f-string interpolations with parameterized queries (#131)
- Entity unique constraint restored: `idx_entity_unique (namespace, name, entity_type)` (#131)
- Entity key gate: `_SurrealDBEntityKeyGate` prevents concurrent upsert races (#136)
- SDK upgrade to `surrealdb>=2.0.0a1` for SurrealDB 3.x support (#133)
- Removed no-op helpers (`_iso`, `_dt_to_iso`); pass UUIDs directly to `RecordID` (#136)
- Schema init race fix: module-level `asyncio.Lock` for embedded mode (#155)
- Write conflict retry with exponential backoff and write semaphore (#157)

### SurrealDB performance

- `vector::dot()` for pre-normalized vectors (~3x faster than cosine) (#150)
- Single-query deletes via `DELETE ... RETURN BEFORE` (50% fewer round-trips) (#150)
- Composite indexes: `(document, chunk_index)`, `(namespace, created_at)`, `(namespace_id, relationship_type, weight)` (#150)
- Single-query multi-depth graph traversal (1 round-trip instead of 3) (#154)
- `INSERT INTO entity $records` for batch creates (replaces FOR loops) (#154)
- Tuple IN for upsert prefetch: `[name, entity_type] IN $pairs` (#154)

### Ontology CLI (`khora ontology`)

- `khora ontology construct --source <path>` — AI-powered ontology construction from data (#138, #140, #141)
- `khora ontology validate <file>` — schema + reference integrity validation (#138)
- `khora ontology preview <file>` — Rich table + tree display (#138)
- `OntologyLLM` wrapper with token/cost tracking and `--budget` USD cap (#138)
- Stratified multi-source data sampling (sqrt-weighted allocation) (#140)
- Domain detection, entity/relationship/rule inference via LLM (#140)
- Session persistence with `--resume` flag (#141)
- 50 unit tests (#142)

### Discovery agent (`khora ontology discover`)

- Interactive agent for finding/fetching data from the internet (#164, #167)
- Perplexity search + Firecrawl scraping API clients (#163)
- Code generation with AST validation and sandboxed execution (#169)
- Data validation pipeline with format detection and quality checks (#170, #171)
- Link-index detection and deep crawl (#176)
- Binary format extraction (PDF, XLS) (#177)
- Non-linear conversational interaction (#178)
- Discovery-to-construct handoff (#174)

### Query engine

- Parallel fallback search via `asyncio.gather()` (~45% latency reduction) (#151)
- Per-call `chunk_strategy` override on `remember()`/`remember_batch()` (#137)

### Embedding & extraction

- Dynamic embedding batch sizing by token budget (#152)
- Bisect embedding batches on JSON parse errors (#149)
- Extraction circuit breaker for batch failures (#187)

### Configuration

- Single-underscore env vars: `KHORA_LLM_MODEL` alongside legacy `KHORA_LLM__MODEL` (#186)

### Infrastructure

- Dev release pipeline: publish to CodeArtifact on every merge to main (#153)
- Bandit security scanning in CI (#185)
- khora-accel wheels for Python 3.13 and 3.14 (#162)
- Shared PG engine for `PgVectorTemporalStore` (#146)
- Skip migrations gracefully when database is ahead (#144)
- Smart resolution and HNSW index rebuild optimized (~30min → ~5min) (#147)

### Bug fixes

- Fix `_parse_uuid` for non-UUID SurrealDB record IDs (#168)
- Fix SurrealDB ingestion performance regression (#160)
- Fix `temporal_chunk` tags: coerce JSON strings to native arrays (#158)
- Fix `SurrealDBTemporalStore` import of removed `_iso` helper (#156)
- Fix alembic `env.py`: replace removed `get_current_revision` API (#180)
- Fix LLM JSON parser: handle trailing commas, bare arrays, code blocks
- Fix VectorCypher engine dropping non-scalar chunk metadata (#134)

---

## [0.5.5] — 2026-03-26 — Ontology CLI & SurrealDB Hardening

First release with the ontology construction CLI and comprehensive SurrealDB
optimization audit. Includes the entity key gate, SDK upgrade to 2.0.0a1,
schema parity fixes, and 15+ SQL injection fixes. See 0.6.0 for the detailed
breakdown (0.5.5 was the last tagged release before the 0.6.0 cycle).

---

## [0.5.4] — 2026-03-25 — Test Audit & CI Fixes

- Replace `__slots__` implementation-detail tests with behavioral checks (#128)
- Fix publish-accel uv cache failure (#129)
- Gitignore `.agents/` folder (#126)

---

## [0.5.3] — 2026-03-24 — macOS Build Fix

- Remove x86_64-apple-darwin from macOS build matrix (#125)

---

## [0.5.2] — 2026-03-24 — Release Pipeline Consolidation

- Consolidate khora and khora-accel into single release pipeline (#124)
- Fix accel build matrix and sccache configuration (#124)

---

## [0.5.1] — 2026-03-24 — First Tagged Release

First release with git-tag-based versioning via `hatch-vcs`. Includes all
features from 0.4.0 and 0.5.0 internal versions.

- Add `publish.yml` and `publish-accel.yml` CodeArtifact workflows (#123)
- Switch to `hatch-vcs` for version derivation from git tags
- Add sccache for Rust build acceleration
- Add `docs/RELEASE.md` with release process documentation

---

## [0.5.0] — SurrealDB Unified Backend & Engine Modernization

### SurrealDB unified backend (Phase 1–4)

- Foundation: `SurrealDBConfig`, connection module (memory/embedded/remote modes),
  relational adapter, vector adapter with HNSW + BM25 (#86–#89)
- Graph adapter: entities, relationships, episodes, traversal, path finding,
  neighborhoods, batch operations (#90, #91)
- Event store adapter (#91)
- Optimization: coordinator dual-write collapse, crash-safe defaults (#92)
- VectorCypher engine support for SurrealDB (#109)
- Skeleton engine `SurrealDBTemporalStore` (#104)
- 14 bug fixes for SDK compatibility, KNN operator, namespace resolution,
  connection sharing, datetime handling (#105–#118)

### Engine modernization

- Modernize Skeleton and GraphRAG engines for robustness and performance (#103)
- Shared `build_storage_config()` helper for all engines
- Move `TemporalDetector` to shared `query/` location
- Add `@trace` telemetry to Skeleton and GraphRAG engines
- Add `bulk_mode` support to all engines
- Improve GraphRAG `stats()` efficiency
- Add importance scoring to Skeleton engine

### Expertise & extraction API

- `ExpertiseConfig` as stable public API (ADR-022) with YAML loading,
  composition, and registry (#96)
- `LLMUsage` type for token/cost tracking in `RememberResult` and `BatchResult`
- `expertise` parameter pass-through on `remember()` and `remember_batch()`
- `extraction_config_hash` column for re-extraction tracking (#97, #99)

### Migrations & deprecations

- Deprecate `create_tables()`/`init_db()` — use `run_migrations()` (#98)
- Migration drift CI test (`test_migration_drift.py`)
- `khora_alembic_version` dedicated version table
- Advisory lock for concurrent migration safety
- Temporal expression index for query performance (#117)

### Other

- Neo4j connection lifetime and liveness config (#102)
- Fix conversation-mode entity extraction regression (#122)
- Widen `extraction_config_hash` to VARCHAR(255) (#99)

---

## [0.4.0] — Logfire Telemetry, Namespace Versioning, Alembic Overhaul

### Logfire / OTEL integration

- Optional `logfire` integration for distributed tracing (#32)
- `trace_span()` context manager and `@trace` decorator
- `_HAS_LOGFIRE` feature flag — zero-cost no-op when absent
- Consumers import from `khora.telemetry`, not `logfire_integration`

### Namespace versioning

- Dual-ID scheme: `id` (row-level, changes per version) + `namespace_id` (stable)
- `resolve_namespace()` idempotent resolution for public API
- Flatten namespace hierarchy (migration 010)
- Add stable `namespace_id` column (migration 012)
- Drop `previous_version_id` (migration 013)
- Drop `slug` (migration 011)

### Alembic overhaul

- Bundle migrations in `src/khora/db/migrations/` (not `alembic/`)
- Dedicated `khora_alembic_version` table (avoids downstream conflicts)
- `pg_advisory_xact_lock` for concurrent migration safety
- Programmatic `run_migrations(url)` and `MemoryLake(run_migrations=True)`
- Sync `document_status` enum (migration 014)

### FastAPI removal

- Remove FastAPI dependency — Khora is a library, not a web app (#35)

### Other

- Fix Alembic migrations on fresh databases (#37)
- Fix UUID `as_uuid=True` across all 52 columns (migration 006)
- Add `document_status` enum sync (migration 014)
- Temporal coalesce expression index (migration 017)

---

## [0.3.10] — Chunker Safety & Rust Performance

### Empty chunk filtering

All three chunkers (Fixed, Recursive, Semantic) now filter out empty or
near-empty chunks before returning results. Root cause: `tokenizer.decode()`
output was not stripped, producing whitespace-only chunks that polluted the
vector index. Previously ~17% of retrieval queries encountered sub-10-character
chunks filtered at query time.

- `MIN_CHUNK_CHARS = 10` constant and `filter_empty_chunks()` method added to
  `Chunker` base class (`extraction/chunkers/base.py`)
- `.strip()` added to tokenizer decode in `FixedChunker` and
  `SemanticChunker._fixed_split()`
- All three chunkers call `self.filter_empty_chunks()` before returning

### Rust entity resolution: HashMap O(1) lookups

`resolve_entities_batch` in `entity_resolution.rs` now builds
`HashMap<String, usize>` indexes for exact name and alias matching stages,
replacing O(n) linear scans with O(1) lookups. This reduces stages 1 and 2
from O(new × existing) to O(new + existing).

### Rust parallelism threshold

Rayon parallel iteration in `entity_resolution.rs` and `string_sim.rs` now
only engages when batch size ≥ 512 elements. Smaller batches use sequential
iteration to avoid thread-pool overhead that dominated at small scale.

### Safe MMR iterator

`mmr.rs` iterator updated to avoid potential panics on empty input.

---

## [0.3.9] — Key-Aware Neo4j Write Coordination

### Why: overlapping MERGE batches caused Neo4j lock contention

The entity write path used a plain semaphore (concurrency 12) to limit concurrent
Neo4j transactions. This prevented connection exhaustion but allowed overlapping
`MERGE` transactions — two batches touching the same entity key would run
concurrently, causing Neo4j to detect lock contention, abort one transaction, and
retry with ~1 s exponential backoff. Under heavy ingestion this cascaded into
minutes of wasted retries.

### `_EntityKeyGate` replaces entity write semaphore

New `_EntityKeyGate` class (`storage/backends/neo4j.py`) tracks in-flight entity
keys — `(namespace_id, name, entity_type)`, the same triple used in the Cypher
`MERGE` clause. Non-overlapping batches proceed concurrently (up to
`entity_write_concurrency`, default 12). Overlapping batches are automatically
serialized at the gate, eliminating Neo4j-side retries entirely.

| Metric | Semaphore only | Key-aware gate |
|--------|---------------|----------------|
| Non-overlapping batches | Concurrent (up to 12) | Concurrent (up to 12) |
| Overlapping batches | Concurrent → lock contention → ~1 s retry | Serialized → zero retries |
| 500 entities, 10% overlap | ~45 s | ~18 s |
| 500 entities, 0% overlap | ~18 s | ~18 s |

Relationship writes still use a plain semaphore (8 concurrent) since `CREATE`
transactions don't contend.

---

## [0.3.8] — Temporal Search Improvements

### Why: temporal queries silently degraded to generic search

When the LLM timed out (2s budget), the heuristic fallback detected temporal
*intent* but produced `start_date=None, end_date=None`, so no temporal filter
was applied. "What happened last week?" returned the same results as "What
happened?". Additionally, temporal filtering happened post-retrieval in Stage 3
(Python-side soft scoring on 200 candidates) instead of as SQL WHERE clauses in
Stage 1, wasting retrieval budget. Chunks also lacked source timestamps — a
Slack message sent Jan 15 but ingested Jan 20 appeared as a Jan 20 event.

### Two-tier temporal resolver

New `TemporalResolver` class (`query/temporal_resolver.py`) provides fast
dateparser-based resolution (~0.25ms with `languages=['en']`) with LLM fallback
for natural language temporal expressions ("last week", "yesterday",
"January 2025", "Q3 2024", "3 weeks ago", etc.). The resolver runs before the
LLM understanding call and sets temporal filters immediately when dateparser
succeeds. When dateparser fails on a recognized temporal pattern, the resolver
falls back to regex-based granularity inference via `_point_to_range()`.

Date validation rejects dates >1 year in the future, before 2000, and
automatically swaps inverted ranges and caps future dates.

### SQL-level temporal pushdown

Temporal filters are now applied as WHERE clauses in Stage 1 SQL queries
(both pgvector similarity and fulltext search) instead of post-retrieval
Python-side filtering in Stage 3. `StorageCoordinator.search_similar_chunks()`
and `search_fulltext_chunks()` accept `created_after`/`created_before` params
that thread through to the pgvector backend.

### Source timestamps

New `source_timestamp` column on `chunks` and `documents` tables
(migration 009). The ingest pipeline extracts timestamps from metadata fields
(`sent_at`, `created_at`, `timestamp`, `date`) and propagates them to documents
and chunks. Temporal filtering uses `COALESCE(source_timestamp, created_at)` so
content is filtered by when it actually occurred, not when it was ingested.

### Configuration

New fields in `QuerySettings`: `enable_temporal_resolver` (default `True`),
`temporal_resolver_strategy` (`"hybrid"` / `"dateparser"` / `"llm"`),
`temporal_sql_pushdown` (default `True`), `temporal_date_validation`
(default `True`). All features are backward-compatible and toggleable.

### Other improvements

- Heuristic fallback now resolves actual dates via `TemporalResolver` instead
  of leaving `start_date=None`
- Auto-recency bias when temporal intent is detected (minimum weight 0.2)
- ISO date parse failures promoted from DEBUG to WARNING
- Database indexes: `ix_chunks_created_at`, `ix_chunks_ns_created`,
  `ix_documents_created_at`, `ix_chunks_source_ts` (partial)
- New dependency: `dateparser>=1.2.0`

### Migrations

- `009_temporal_search_indexes` — temporal indexes, `source_timestamp` columns

---

## [0.3.7] — Stability Fixes

### Fixed

- Cap Neo4j write concurrency and bound provenance list growth to prevent OOM
  under high-volume ingestion (#28)
- Remove invalid `IF EXISTS` from `REINDEX` command that caused PostgreSQL
  errors on older versions (#27)

### Added

- TTOJ team profile templates (#29)

---

## [0.3.6] — VectorCypher Entity Search Fix

### Fixed

- VectorCypher entity search called a non-existent coordinator method,
  causing `AttributeError` on entity-heavy queries (DYT-180) (#26)

---

## [0.3.5] — Phase 3 Benchmark Optimizations

### Temporal retrieval

- Propagate document custom metadata to chunk metadata, fixing 57% zero-recall
  on temporal queries where session-level fields (author, channel) were missing
  from chunks.
- Fall back to `thread_id` when `channel` is absent for session filtering.

### Graph density

- Expand graph search entry points from ~8 to ~18 seed entities for broader
  traversal coverage.
- Lower relationship confidence threshold from 0.35 to 0.25 for denser graphs.
- Relax entity dedup Levenshtein threshold from 0.8 to 0.7 to merge name
  variants (e.g., "J. Smith" / "John Smith").

### Adversarial / confounder rejection

- Add bigram coherence scoring (`bigram_coherence_score()`) to penalize
  word-shuffled confounders without LLM cost. Integrated into VectorCypher's
  RRF fusion via `apply_coherence_boost()` with `coherence_weight=0.1`.

### Performance

- Enable query result caching in VectorCypher (`query_cache_ttl_seconds=300`,
  `query_cache_max_size=100`).
- Raise router LLM confidence threshold from 0.7 to 0.85 to reduce
  mis-routed queries.

### Housekeeping

- Version bump 0.3.4 → 0.3.5.

---

## [0.3.4] — ty Type Checker Clean

- Resolve all remaining `ty` diagnostics — `ty check src/` now passes with
  zero warnings.
- Version bump 0.3.3 → 0.3.4.

---

## [0.3.3] — Neo4j Deadlock Fixes

- Shared semaphore for Neo4j relationship writes to prevent deadlocks during
  concurrent batch ingestion.
- Tune Neo4j driver parameters (`max_transaction_retry_time`,
  `connection_acquisition_timeout`) to reduce transaction deadlock retries.
- Version bump 0.3.2 → 0.3.3.

---

## [0.3.2] — Phase 2 Benchmark Optimizations

- Restore parallel Neo4j writes and reduce relationship volume.
- Add co-occurrence edges, lazy entity expansion, skeleton skip, and
  concurrency alignment.
- Phase 2 benchmark optimizations for improved ingestion throughput.
- Version bump 0.3.1 → 0.3.2.

---

## [0.3.1] — Benchmark-Driven Optimizations

### Why: restoring incremental MRR and improving retrieval quality

Benchmark run `2f7d4b0b` revealed that incremental ingestion (add
documents in multiple batches) produced an MRR of 0.0 — newly ingested
content was effectively invisible to queries. Root cause analysis by a
6-specialist agent team identified compounding bugs in entity ID
mapping, BM25 cache invalidation, and a config mismatch that disabled
the diversity stage. This release fixes those bugs, then layers on
graph density, temporal accuracy, evidence quality, and Rust
acceleration improvements identified during the same analysis.

### Critical bug fixes (P0)

**BM25 cache invalidation.** After `remember()` or `remember_batch()`,
the GraphRAG engine now calls `invalidate_caches()` on the query engine
so stale BM25 indexes are rebuilt. Previously, keyword search results
were frozen to the state at first query.

**Entity ID mapping.** Neo4j's `MERGE` mutates `entity.id` in-place
when an existing node is found. The post-ingestion ID mapping loop now
reads from a `pre_upsert_ids` snapshot taken before the MERGE, so
relationship source/target IDs point at the correct graph nodes.

**Neo4j relationship MERGE semantics.** Relationships now use `MERGE`
with `ON CREATE SET` / `ON MATCH SET` instead of `CREATE`, preventing
duplicate edges when the same relationship is ingested across batches.

**Entity unique constraint.** A new Alembic migration
(`008_entity_dedup_and_indexes`) deduplicates entities by
`(namespace_id, name, entity_type)`, merges their
`source_document_ids`, re-points foreign keys, and adds a `UNIQUE`
constraint. The pgvector backend's entity upsert uses `ON CONFLICT` on
this constraint.

**`datetime.now(UTC)`.** Four instances of timezone-naive
`datetime.now()` in `query/temporal.py` now use `datetime.now(UTC)`.

**Temporal sort TypeError.** The fallback key for sorting chunks by
`created_at` now uses `datetime.min` instead of `0`, preventing
`TypeError` when comparing `int` to `datetime`.

### Retrieval quality

**MMR diversity enabled by default.** `QuerySettings.enable_diversity`
now defaults to `True`, matching the `QueryConfig` dataclass. The MMR
stage (Stage 5) rejects same-document dominance and improves confounder
rejection.

**Adaptive top-k "very_focused" tier.** Queries with complexity < 0.3
(single-entity factual lookups) now return at most 3 chunks with a
0.25 minimum similarity, reducing noise for simple queries.

**Title propagation.** Document titles are propagated into
`chunk.metadata.custom["title"]` during ingestion, giving the
cross-encoder reranker reliable title context.

**Selective entity embedding.** Low-value entity types (DATE, URL,
EMAIL) with mention count ≤ 1 skip embedding generation, reducing LLM
calls during ingestion. Controlled by
`KHORA_PIPELINE__SKIP_EMBEDDING_ENTITY_TYPES` and
`KHORA_PIPELINE__SKIP_EMBEDDING_MENTION_THRESHOLD`.

### Graph density (G-1 through G-6)

**Extraction prompt verification.** The LLM extraction prompt now
includes a verification instruction asking the model to confirm each
entity has at least one relationship before returning.

**Two-pass extraction threshold.** The second extraction pass now
triggers when `num_relationships < max(2, num_entities // 2)` instead
of the previous fixed `< 2`, catching sparse graphs with many isolated
entities.

**Expanded INVALID_PAIRS.** Co-occurrence inference now rejects
DATE↔LOCATION, URL↔LOCATION, and EMAIL↔LOCATION pairs in addition to
existing invalid combinations.

**More bidirectional types.** Neo4j relationship creation now generates
reverse edges for LEADS↔LED_BY and ASSIGNED_TO↔HAS_ASSIGNEE, in
addition to the existing bidirectional types.

### Temporal accuracy (T-1 through T-6)

JSON schema temporal fields, source timestamp propagation, temporal
re-ranking, session boundary detection, and UTC normalization were all
implemented. Neo4j now creates temporal indexes on relationship
`valid_from` and `created_at` properties for the highest-volume
relationship types.

### Database optimizations

**HNSW tuning.** Migration `007_hnsw_parameter_tuning` sets `m=24`
and `ef_construction=128` (up from 16/64), improving vector recall at
the cost of slightly larger indexes.

**Halfvec infrastructure.** `StorageSettings.use_halfvec` enables
float16 HNSW indexes (requires pgvector ≥ 0.7.0). Column data remains
full precision.

**Entity temporal indexes.** Partial indexes on `entities.valid_from`
and `entities.valid_until` (WHERE NOT NULL) accelerate temporal
filtering.

**khora_chunks composite index.** New index on
`khora_chunks(namespace_id, document_id)` supports Skeleton and
VectorCypher engine queries.

### Rust acceleration

**MMR diversity selection.** New `mmr.rs` module implements greedy
Maximal Marginal Relevance in Rust with SIMD-friendly dot product, GIL
release, and incremental max-similarity tracking. 10-50x faster than
the Python loop. Falls back to NumPy, then pure Python.

**Pre-normalized embeddings.** The LiteLLM embedder now L2-normalizes
all embeddings at ingest time. Scoring switches from
`batch_cosine_similarity` to `batch_dot_product` (~3x faster since it
skips redundant normalization).

**Temporal filtering.** New `temporal.rs` module provides batch
datetime comparison and recency scoring with rayon parallelism and GIL
release.

### Cross-narrative contamination (N-1, N-2)

Narrative coherence scoring and source-aware context assembly were
added to reduce cross-topic contamination in multi-document memory
lakes.

### Genesis integration (GN-1, GN-2, GN-3)

GitHub and Jira sources now route to `technical_project` expertise.
The rule engine supports up to 8 inference conditions. A new Slack
skill (`extraction/skills/builtin/slack.yaml`) provides DM recipient
extraction guidance with MESSAGED and SENT_MESSAGE_TO relationship
types.

### Integration tests

25 new tests in `tests/integration/test_incremental_ingestion.py`
cover batch 1/2/3 ingestion, entity MERGE across batches, relationship
MERGE, BM25 cache invalidation, and full remember→recall flows.

### Migrations

- `007_hnsw_parameter_tuning` — HNSW m=24, ef_construction=128
- `008_entity_dedup_and_indexes` — Entity dedup + unique constraint,
  khora_chunks composite index, entity temporal partial indexes

---

## [0.3.0] — Engineering Improvements

### Why: removing accidental complexity

Global state in database session management, UUID string wrapping across
52 ORM columns, redundant connection pools for backends sharing the same
database URL, and stale deprecated APIs that no longer matched the
codebase — none of these served users, and all of them created friction
for contributors. This release removes the accidental complexity so the
next round of features lands on cleaner ground.

### UUID migration

All 52 UUID columns in `db/models.py` now declare `as_uuid=True`,
mapping to native Python `uuid.UUID` objects. This is a Python-side-only
change — the PostgreSQL column type remains `UUID`. The practical effect
is that code no longer needs `str()` wrapping when building ORM models
or `UUID()` parsing when reading them. Graph backends (Neo4j, Kuzu,
Memgraph) still convert at the boundary because they don't support
native UUIDs.

### DatabaseManager

`db/session.py` previously used module-level globals for the async
engine and session factory. These are now encapsulated in a
`DatabaseManager` class that owns engine creation, session lifecycle,
and disposal. Backward-compatible module-level wrappers are preserved
so existing callers continue to work without changes.

### Shared connection pools

`StorageFactory` now caches async engines by normalized URL. When
PostgreSQL, pgvector, and the event store all point at the same
database (the common case), they share a single connection pool instead
of creating three independent ones. Backends using a shared engine skip
`dispose()` on disconnect to avoid pulling the pool out from under
siblings.

### TransactionContext

`StorageCoordinator.transaction()` returns an async context manager
that wraps multiple backend writes in a single database transaction.
`TransactionContext.savepoint()` creates nested savepoints for partial
rollback. Backend write methods accept an optional `session` parameter
to join the active transaction.

### Deprecated API cleanup

- `lake.storage` — promoted to stable public API (used by `genesis` and
  `khora-benchmarks`). The deprecation warning has been removed.
- `lake.query_engine` — removed. Use `lake.recall(raw=True)` for
  unprocessed search results.
- `remember_batch_legacy()` — removed. Use `remember_batch()`.

### Chat module tests

71 new tests across 4 files covering the chat module (`chat/engine.py`,
`chat/history.py`, `chat/persona.py`, `chat/prompt.py`). The module
itself is unchanged — these tests document and lock existing behavior.

### spaCy sentence splitting

The semantic chunker now uses spaCy's `sentencizer` component when
available, improving sentence boundary detection. Install with
`pip install khora[nlp]`. The sentencizer is a rule-based component
that ships with spaCy core — no model download needed. When spaCy is
not installed, the chunker falls back to its existing regex-based
splitter transparently.

### Docker removal

The `Dockerfile` and CI `docker-build` job have been removed. Khora is
a library, not a deployable application — the Dockerfile was never used
in production and added maintenance burden. Development databases
continue to use `compose.yaml` via `make dev`.

### Housekeeping

- Version bumped from 0.2.3 to 0.3.0 in `pyproject.toml`,
  `src/khora/__init__.py`, `rust/khora-accel/Cargo.toml`, and
  `rust/khora-accel/pyproject.toml`.

---

## [0.2.3] — Namespace Optimization Design

### Why: surfacing what's real vs. what's aspirational

A team of five specialist agents audited Khora's namespace isolation,
multi-tenancy enforcement, and temporal extraction paths. The audit
found that several documented features — `TenancyMode` routing, ACL
enforcement, bi-temporal edge storage, and the time hierarchy builder —
exist as code but are never exercised at runtime. Meanwhile, the
namespace-level row filtering that *is* active lacks an orphan-entity
cleanup path when documents are deleted. This release ships the
comprehensive design for fixing all of it, marks the stale
documentation, and inventories the dead code so the next releases can
act on it.

### Namespace optimization design

New `docs/design/namespace-optimization-plan.md` lays out a six-phase
implementation roadmap:

1. **Orphan fix** — delete graph entities left behind after `forget()`.
2. **Data-model hardening** — add `namespace_id` to Neo4j entity/chunk
   nodes and enforce it in Cypher queries.
3. **Isolated-mode core** — per-org connection routing driven by
   `TenancyMode.ISOLATED`.
4. **Shared-mode ACL** — wire `ACLEnforcer` into the API dependency
   chain for `TenancyMode.SHARED`.
5. **ACL enforcement** — row-level security policies and graph-side
   namespace filtering.
6. **Rust acceleration** — move hot-path namespace filtering into
   `khora-accel`.

### Dead-code inventory

- `TenancyMode` enum (`core/models/tenancy.py`) is defined but never
  checked at runtime — all orgs use implicit shared mode.
- `ACLEnforcer` and `ACLContext` (`acl/`) are importable but the API
  dependency in `api/deps.py` is disabled.
- `TemporalEdgeStorage` and `TimeHierarchyBuilder` (`engines/skeleton/`)
  exist as modules but are never called by any engine's ingest or recall
  paths. The `occurred_at` column on chunks works through the pgvector
  backend directly.

### Stale documentation fixes

Added status notices to five documentation files flagging features that
are designed but not yet wired:

- `docs/architecture/multi-tenancy.md` — TenancyMode and ACL sections.
- `docs/engines/temporal-model.md` — bi-temporal edge model.
- `docs/engines/skeleton-engine.md` — architecture diagram components.
- `README.md` — multi-tenancy feature bullet.
- `docs/architecture/overview.md` — ACL enforcer mention.

### Housekeeping

- Bumped version from 0.2.2 to 0.2.3.

---

## [0.2.2] — VectorCypher Optimization

### Why: making hybrid retrieval competitive on benchmarks

VectorCypher launched in 0.2.0 with sensible defaults, but benchmark runs
against GraphRAG-Bench showed that retrieval quality dropped on complex
multi-hop queries and that the configuration wasn't surfaced cleanly
through the public API. This release is the result of a benchmarking-
driven optimization cycle: tune retrieval, wire the knobs, add the
indexes to support it, and clean up the code that was left behind.

### Retrieval quality

**Per-complexity fusion weights.** The original retriever used a single
pair of vector/graph weights (0.6/0.4) for every query. Simple factual
queries don't benefit from graph expansion, while complex multi-hop
queries need more graph signal. The retriever now applies different
weights per complexity level: SIMPLE gets 0.8/0.2 (vector-heavy),
MODERATE keeps the 0.6/0.4 default, and COMPLEX flips to 0.4/0.6
(graph-heavy). These are configurable via `VectorCypherConfig`.

**Adaptive graph traversal depth.** Previously, graph depth was fixed
at 2 regardless of how many entry entities the vector search returned.
When many entities match (≥10), deep traversal explodes the candidate
set without adding signal — so the retriever now drops to depth 1.
Conversely, when very few entities match (≤2), it increases depth to
compensate. The thresholds are configurable via `RetrieverConfig`.

**Score normalization.** The fusion function (`weighted_rrf_normalized`)
now min-max normalizes vector and graph scores to [0, 1] before
computing RRF, producing more balanced fusion when score distributions
differ between the two sources.

**Entity resolution and graph density.** Improved entity similarity
thresholds (`min_entity_similarity=0.3`) and skeleton core ratio
(now 0.70 by default) increase the number of entities that get full
LLM extraction, producing a denser graph for traversal.

### Configuration wiring

**`VectorCypherConfig` dataclass.** All VectorCypher-specific knobs —
routing, skeleton indexing, graph traversal, fusion weights, temporal
settings, and search thresholds — live in a single dataclass that can
be passed through the `MemoryLake` constructor:

```python
from khora import MemoryLake
from khora.engines.vectorcypher import VectorCypherConfig

async with MemoryLake(
    db_url,
    engine="vectorcypher",
    engine_kwargs={"vectorcypher_config": VectorCypherConfig(
        skeleton_core_ratio=0.50,
        fusion_complex_vector_weight=0.3,
        fusion_complex_graph_weight=0.7,
    )},
) as lake:
    ...
```

**`engine_kwargs` passthrough.** The `MemoryLake` constructor now
accepts an `engine_kwargs` dict that is forwarded to the engine
constructor. This is the mechanism for passing `VectorCypherConfig`
(or any future engine-specific config) without changing the public API.

### Search indexes (migration 005)

Three new PostgreSQL indexes improve query-time performance:

- **GIN index** on `khora_chunks.tags` for array-containment queries
- **Composite index** on `(namespace_id, occurred_at)` for temporal
  filtering within a namespace
- **HNSW index** rebuilt with `ef_construction=128` (up from 64) for
  better vector recall at the same latency

### Housekeeping

- Skeleton engine code cleanup: removed 122 lines of dead formatting
  and redundant logic.
- Removed hardcoded fusion weights from the retriever in favor of
  config-driven values.

---

## [0.2.1] — Concurrency & Throughput

### Why: filling the gap Rust opened

Version 0.2.0 moved CPU-bound work (similarity scoring, PageRank, BM25
indexing) off the Python event loop and into native Rust threads. The
immediate effect was that CPU cycles were no longer the bottleneck during
large ingestion runs — network I/O to LLM and embedding providers was.
Concurrency limits that once protected against CPU saturation were now
artificially capping throughput: async tasks sat idle waiting for
semaphore permits while the CPU and network had headroom to spare.
Doubling the defaults across every concurrency-controlling parameter
lets Khora fill that idle time, keeping both the network pipe and the
Rust worker pool saturated.

### Concurrency changes by layer

**Configuration defaults.** The global LLM concurrency ceiling
(`max_concurrent_llm_calls`) moved from 10 to 20, and the embedding
concurrency limit from 25 to 50. These two knobs govern all downstream
semaphores, so raising them was the prerequisite for everything else.

**Extractors and embedders.** The LLM extractor's own semaphore doubled
from 5 to 10 concurrent calls. On the embedding side, the LiteLLM
embedder now batches 200 texts per request (up from 100) and runs 20
concurrent embedding calls (up from 10), reducing round-trip overhead
on high-throughput workloads.

**Ingestion pipeline.** The ingestion flow — Khora's primary data path —
doubled three independent limits: concurrent extractions (10 to 20),
embedding batch size (50 to 100), and concurrent document processing
(5 to 10). Together these allow the pipeline to keep more documents
in flight simultaneously.

**Engine-level parallelism.** Every engine's `max_concurrent` semaphore
doubled: GraphRAG from 5 to 10, Skeleton from 10 to 20, VectorCypher
from 10 to 20. The `remember_batch` entry points on MemoryLake and the
base engine protocol matched at 10 (up from 5). Entity expansion
semaphores in the expansion flow doubled from 20 to 40.

**Genesis (bulk loader).** Genesis configuration files for all three
engine profiles doubled their LLM/embedding concurrency (100 to 200),
document concurrency (50 to 100), and chunk concurrency (100 to 200).
The CLI default batch size moved from 10 to 20.

### Housekeeping

- Removed REPOMIX tooling: `REPOMIX.md`, `repomix.config.json`,
  `scripts/update_repomix.py`, and the `update-repomix` pre-commit hook
  (along with REPOMIX exclusions in other hooks).
- Deleted completed planning docs (`OPTIMIZATION_PLAN.md`,
  `RUST_ACCELERATION_PLAN.md`).
- Excluded `docs/REFERENCES.md` from version control (`.gitignore`).
- Bumped version from 0.2.0 to 0.2.1.

---

## [0.2.0] — Rust Acceleration Layer

### The problem

Profiling large ingestion runs showed that CPU-bound operations —
cosine similarity over dense embedding matrices, edit-distance
computations during entity resolution, PageRank convergence over chunk
graphs, and BM25 scoring — dominated wall-clock time once documents
were chunked and LLM calls returned. Python's GIL serialized these
hot loops, and even NumPy could not parallelize the non-BLAS workloads
(string comparisons, graph iteration, inverted-index lookups).

### The approach

Khora 0.2.0 introduces `khora-accel`, a Rust extension built with
PyO3 and maturin. The design philosophy is **zero mandatory
dependencies**: a three-tier fallback (`_accel.py`) checks for the
Rust extension first, then NumPy/RapidFuzz, then pure Python. Every
accelerated function is a drop-in replacement — the Python signature
and return type are identical across all three tiers. Set the
`KHORA_ACCEL_BACKEND` environment variable to `"rust"`, `"numpy"`, or
`"python"` to pin a specific tier; leave it unset for automatic
detection of the fastest available backend.

### Accelerated operations

**Vector similarity.** Cosine similarity (single-pair, one-to-many
batch, and all-pairs above threshold) is implemented with a fused
dot-product-and-norm single pass. Batch operations accept NumPy arrays
via zero-copy `PyReadonlyArray` bindings, copy once into owned Rust
vectors, then release the GIL and fan out across cores with rayon
parallel iterators. For a 10K-candidate batch, this eliminates both
the GIL bottleneck and the Python loop overhead.

**String similarity.** Levenshtein distance and sequence-match ratio use
the `strsim` crate, which implements single-row Wagner-Fischer DP
natively. Batch variants (`batch_levenshtein`, `batch_sequence_match`)
release the GIL and score candidates in parallel via rayon. This
matters for entity resolution, where every new entity must be compared
against hundreds or thousands of existing names.

**BM25 search.** `RustBM25Index` is a full inverted-index implementation
with tokenization, stopword filtering, and suffix-based stemming built
into the Rust layer. The inverted index narrows candidates before
scoring, and the entire scoring phase runs with the GIL released.
Unlike the pure-Python version (which re-tokenized queries on each
call), the Rust implementation tokenizes each query once and pre-computes
IDF scores across the candidate set.

**Graph algorithms.** PageRank and chunk-edge construction power Skeleton
Construction's core indexing step, which identifies the ~10% highest-
value chunks for targeted LLM extraction. Both functions release the
GIL and run pure Rust graph iteration — adjacency-list storage,
iterative convergence with early termination, and O(k^2) bidirectional
edge generation from keyword co-occurrence weighted by IDF.

**Entity resolution.** `resolve_entities_batch` implements the same
three-stage cascade as the Python original (exact name match, alias
match, fuzzy Levenshtein match) but pre-lowercases all existing names
and aliases once, then processes the full batch in parallel with rayon.
For workloads with hundreds of new entities against thousands of
existing ones, this turns an O(n*m) serial Python loop into a
rayon-parallelized Rust loop with no GIL contention.

**Text processing.** Keyword extraction (`extract_keywords`,
`extract_keywords_batch`) uses a compiled `LazyLock<Regex>` and a
`hashbrown::HashSet` stopword table. The batch variant parallelizes
across documents with rayon, which is particularly effective during
bulk ingestion when thousands of chunks need keyword tagging
simultaneously.

**Score fusion.** Reciprocal Rank Fusion (basic and weighted variants)
and min-max score normalization use `hashbrown::HashMap` for
accumulation and `OrderedFloat` for deterministic sorting. These are
lightweight operations, so the Rust version's advantage is mainly in
eliminating Python dict/sort overhead on large ranked lists.

### Integration

The `_accel.py` facade exposes 18 public functions consumed by:
- `engines/skeleton/skeleton.py` — PageRank, chunk edges, keywords, BM25
- `engines/vectorcypher/fusion.py` — RRF, weighted RRF, score normalization
- `query/engine.py` — cosine similarity, BM25 search
- `extraction/entity_resolution.py` — batch entity resolution
- `storage/` and `pipelines/` — embedding similarity, string matching

The active backend is logged at import time for observability.

### Other changes

- Improved upsert result mismatch diagnostics.
- Downgraded extraction log to debug level.
