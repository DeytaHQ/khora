# Changelog

All notable changes to Khora are documented here.

Format: versions match git tags (`git tag vX.Y.Z`). Versions before 0.5.1 were internal (no git tags).

## [Unreleased]

### Fixed

- Stale results from `Khora.recall()` after `remember`/`forget`. The in-process query result cache in the vectorcypher retriever held results for up to 5 minutes without invalidation on writes.
- **Entity-upsert advisory lock collided at ~65K namespaces** ([#738](https://github.com/DeytaHQ/khora/issues/738)). `_namespace_lock_key` folded the 128-bit `namespace_id` UUID down to a single signed `int4` via 4-way XOR, used as `key2` in `pg_advisory_xact_lock(KHOR, key2)`. Deployments with more than ~65K distinct namespaces (per-user / per-agent patterns under `khora.integrations.openai_agents`, `google_adk`, `crewai`, `langgraph`) hit birthday-paradox collisions — empirically observed at 120K in the issue's repro. Two namespaces sharing a folded key would serialize their entity upserts behind each other, producing tail-latency spikes on a random subset of namespaces. No data loss — the lock auto-released on commit and `_retry_on_deadlock` covered the contention. Replaced with `_namespace_lock_keys(...)` which fills both 32-bit slots of Postgres's two-int advisory-lock form from the full 128 bits of the UUID, giving ~2^64 effective lock-id entropy (birthday-safe at billions of namespaces). **Operators:** the legacy `0x4B484F52` ("KHOR") `classid` is no longer set on these locks — update any `pg_locks` dashboards that filter on it.
- **`PgVectorBackend.upsert_entities_batch` reported `is_new=True` for every upsert** ([#719](https://github.com/DeytaHQ/khora/issues/719)), even when the row already existed in the namespace. The implementation hardcoded `[(entity, True) for entity in sorted_entities]` despite the "`is_new` is approximate" docstring claim — the pgvector half of a postgres+neo4j dual-write disagreed with the Neo4j half's `MERGE` semantics, silently inflating any coordinator / telemetry counter keyed on `is_new=True`. The `ON CONFLICT DO UPDATE` statement now adds `RETURNING (xmax = 0) AS is_new` and maps results back to inputs by `(name, entity_type)`; `xmax = 0` is the canonical Postgres marker for "freshly inserted" vs "matched + updated" in an upsert. Verified end-to-end against real Postgres in `tests/integration/test_pgvector_upsert_is_new.py`.

### Removed

- `RetrieverConfig.query_cache_ttl_seconds` and `RetrieverConfig.query_cache_max_size` (also removed from `VectorCypherEngineConfig`). **Breaking:** passing these kwargs now raises `TypeError`. Drop them from your config.

## [0.15.1] — Security patch release

### Security

- **Cypher / SQL injection via `Entity.attributes` / `Entity.metadata` (and the equivalent `Relationship` and `Episode` fields) on the AGE graph backend.** A document submitted through `Khora.remember` whose extracted entity attributes or metadata contained a single quote was JSON-serialised unescaped into the AGE Cypher template, letting the payload close the Cypher string literal and execute Cypher of the attacker's choice. Because AGE wrapped Cypher inside a PostgreSQL `$$ … $$` dollar-quoted string, a payload containing `$$` further escalated to SQL injection on the host. Fixed by:
  1. New `AGEBackend._serialize_dict_literal()` helper that JSON-encodes and then runs the result through the existing `_escape` (single quotes, backslashes, control characters). Applied at every Cypher-template site that interpolates a dict (entity create / update, relationship create, episode create — 7 call sites total).
  2. `AGEBackend._cypher()` now wraps the inner Cypher in a uniquely-tagged dollar-quote `$khora_age$ … $khora_age$`, defanging the `$$`-breakout escalation. Inputs containing the literal tag are refused with a `ValueError` as defense in depth.

  Reachable from any caller that can submit a document to `remember()` in a deployment where `backend=age` is configured; the attacker-controlled value reaches the AGE template through the LLM extractor's `attributes` / `metadata` output.

- **Cross-namespace IDOR on `StorageCoordinator.get_entity` / `get_relationship` / `get_episode`.** The public storage-facade getters took only an ID and returned whatever the graph backend held under that ID. A caller scoped to namespace B that knew an entity ID from namespace A received the namespace-A entity verbatim, violating the per-tenant isolation invariant. `Khora.get_entity()` (top-level), the engine `get_entity` methods (`vectorcypher`, `chronicle`, `skeleton`), and the `MemoryEngineProtocol.get_entity` had the same shape. The facade now requires a `namespace_id` keyword argument and returns `None` whenever the persisted row's `namespace_id` does not match the caller's. The underlying graph-backend `get_entity` / `get_relationship` / `get_episode` methods retain their ID-only shape (they sit below the trust boundary); filtering happens at the facade.

- **Cross-namespace chunk access via `kb.storage.get_chunk` / `get_chunks_batch` / `get_chunks_by_document`.** The three chunk-getter facade methods (and their underlying vector-backend implementations in pgvector, sqlite, sqlite+lance, and surrealdb) previously accepted only an id and did not filter by namespace, allowing a caller scoped to namespace B to retrieve chunks belonging to namespace A by id (an IDOR primitive on multi-tenant deployments). The methods now require a `namespace_id` keyword argument and apply a namespace predicate at SQL level; cross-namespace ids are silently dropped from the result. All in-tree callers have been updated.

### Changed (breaking)

- **`khora.Khora.get_entity(entity_id)` now requires `namespace=...`.** Resolution mirrors `list_entities` / `find_related_entities` — accepts `str | UUID`. Calling without it raises `TypeError`. Downstream consumers (`khora-cli`, `khora-explorer`) must be updated in lockstep.
- **`StorageCoordinator.get_entity(entity_id)` / `get_relationship(relationship_id)` / `get_episode(episode_id)` now require keyword-only `namespace_id: UUID`.** Calls without it raise `TypeError`.
- **`MemoryEngineProtocol.get_entity` and its three implementations (`VectorCypherEngine`, `ChronicleEngine`, `SkeletonEngine`) gained a required `namespace_id` kwarg.**
- **`StorageCoordinator.get_chunk(chunk_id)` / `get_chunks_batch(chunk_ids)` / `get_chunks_by_document(document_id)` now require keyword-only `namespace_id: UUID`.** Same shape on the four vector-backend implementations (pgvector, sqlite, sqlite+lance, surrealdb). Calls without it raise `TypeError`; cross-namespace ids in `get_chunks_batch` are silently dropped from the returned dict; `get_chunks_by_document` returns `[]` if the document doesn't belong to the namespace.

## [0.15.0] — Dream-phase Phase 2 + Phase 4, PPR retrieval, kuzu removed

Minor release. Lands Phase 2 (planner ops) and Phase 4 (apply mode) of the [Dream Phase umbrella (#649)](https://github.com/DeytaHQ/khora/issues/649) — `Khora.dream(namespace, mode="apply")` is now end-to-end functional with bi-temporal soft-delete, per-op transactions, and snapshotted undo records. Also: Personalized PageRank retrieval for VectorCypher (#542), the kuzu backend is removed, and the README + dream-phase docs are rewritten.

### Added

- **Dream phase Phase 2 — five planner operations.** Each emits a `DreamOp` describing what `apply` mode would do; dry-run is free of side effects.
  - `vectorcypher_dedupe_entities` ([#658](https://github.com/DeytaHQ/khora/issues/658) → [PR #691](https://github.com/DeytaHQ/khora/pull/691)) — cross-batch entity resolution against the full namespace, per-type cosine thresholds (default 0.90; tighter than the online 0.85), skip-collision reporting.
  - `vectorcypher_centroid_recompute` ([#660](https://github.com/DeytaHQ/khora/issues/660) → [PR #690](https://github.com/DeytaHQ/khora/pull/690)) — three-decision planner (`centroid` / `re_embed` / `skip_multimodal`) for post-merge canonical embeddings.
  - `vectorcypher_source_chunk_ids_gc` ([#662](https://github.com/DeytaHQ/khora/issues/662) → [PR #689](https://github.com/DeytaHQ/khora/pull/689)) — plans per-entity rewrites that drop dead chunk UUIDs from `Entity.source_chunk_ids`.
  - `chronicle_fact_compaction` ([#664](https://github.com/DeytaHQ/khora/issues/664) → [PR #688](https://github.com/DeytaHQ/khora/pull/688)) — plans hard-deletes of tombstoned `memory_facts` rows past `fact_compaction_retention_days`.
  - `chronicle_event_clustering` ([#665](https://github.com/DeytaHQ/khora/issues/665) → [PR #692](https://github.com/DeytaHQ/khora/pull/692)) — clusters near-duplicate `chronicle_events` within a sliding `referenced_date` window.
- **Dream phase Phase 4 — apply mode** ([#667](https://github.com/DeytaHQ/khora/issues/667) / [#668](https://github.com/DeytaHQ/khora/issues/668) / [#669](https://github.com/DeytaHQ/khora/issues/669) → PRs [#698](https://github.com/DeytaHQ/khora/pull/698) / [#699](https://github.com/DeytaHQ/khora/pull/699) / [#700](https://github.com/DeytaHQ/khora/pull/700) / [#701](https://github.com/DeytaHQ/khora/pull/701)). `Khora.dream(ns, mode="apply")` now executes the plan. The orchestrator's `_apply_phase` calls per-op apply handlers under per-op transactions, persists `UndoRecord.before` snapshots to `undo.json` (schema `dream-undo/1`) **before** any mutation, and updates the `khora_dream_runs.last_committed_op_seq` checkpoint between ops. Five guardrails protect the path:
  - **Hard 7-day retention floor** on `fact_compaction_retention_days` — config validator rejects sub-floor values; the apply handler re-checks defense-in-depth.
  - **`KHORA_DREAM_DISABLE_APPLY` env-var kill-switch** — `mode="apply"` raises `DreamApplyDisabled` immediately without touching the DB.
  - **`chunk_id` runtime assertion** — any `UndoRecord.before` payload carrying a top-level `"chunk_id"` key aborts the run with `DreamForbiddenOpError`. The "never mutate `chronicle_events.chunk_id`" architectural promise is now a runtime guarantee.
  - **Snapshot-before-mutate** for `fact_compaction` — the only hard-delete op. `SELECT *` per target row runs before any `DELETE`; mid-snapshot failure rolls back with zero rows touched. TDD-pinned by `test_snapshot_captured_before_delete_executes`.
  - **Advisory lock held through apply** — the per-namespace `pg_advisory_xact_lock` covers planning *and* application. Concurrent dream runs against the same namespace fast-fail with `DreamLockUnavailable`.
- **Migration 034 — `chronicle_events` bi-temporal columns** ([PR #701](https://github.com/DeytaHQ/khora/pull/701)). Adds `invalidated_at`, `invalidated_by`, and `merged_into_event_id` (self-FK, `ON DELETE SET NULL` on Postgres) to `chronicle_events`. Postgres-only partial composite index `ix_chronicle_events_live` over `(namespace_id, referenced_date) WHERE invalidated_at IS NULL`. Required substrate for `apply_chronicle_event_clustering`.
- **Personalized PageRank retrieval for VectorCypher** ([#542](https://github.com/DeytaHQ/khora/issues/542) → [PR #693](https://github.com/DeytaHQ/khora/pull/693)). Opt-in via `KHORA_QUERY_ENABLE_PPR_RETRIEVAL=true` (default off). Wired into the retriever as a separate channel; degrades to vector-only on empty entry entities / empty graph / no seed overlap. Tuning knobs in `KhoraConfig.query`.
- **Public surface additions.**
  - `UndoRecord(op_id, op_type, before, applied_at)` — returned by every apply handler; persisted into `undo.json`.
  - `DreamApplyDisabled` exception — raised by the env-var kill-switch.
  - `OpSummary` — the shape of items in `DreamResult.ops` (aggregate counters per op kind).
- **Apply functions** registered in `khora.dream.engines.registry._APPLY_HANDLER_NAMES`: `apply_vectorcypher_dedupe_entities`, `apply_vectorcypher_centroid_recompute`, `apply_vectorcypher_source_chunk_ids_gc`, `apply_chronicle_fact_compaction`, `apply_chronicle_event_clustering`. All honor the caller-owned-transaction contract (no commit, no log, no telemetry from the handler).
- **`SECURITY.md`** ([PR #707](https://github.com/DeytaHQ/khora/pull/707)). Project security policy: scope (credential leakage, SQL/Cypher injection, path traversal, cross-tenant leakage), reporting routes (GitHub Private Vulnerability Reporting first, `security@deytahq.com` fallback), supported-versions policy (latest minor + n-1), response targets (ack 2 BD, triage 5 BD, fix 30 days high/critical or 90 days medium/low), and a "What khora already does" section listing existing controls (`SecretStr` on credentials, `bounded_text_hash` on telemetry, namespace-cardinality rule, secret-typing semgrep, pip-audit in CI, parameterized SQL/Cypher, bi-temporal soft-delete).
- **Test coverage** for Phase 4 surfaces: 37 apply-handler tests across 5 modules (chronicle + vectorcypher), 14 orchestrator-apply tests (covering kill-switch, retention floor, chunk_id assertion, resume-from semantics, undo.json incremental write + fsync), plus 5 integration tests for migration 034 (Postgres-gated).

### Changed

- **README rewritten as an evaluation-flow entry point** ([PR #708](https://github.com/DeytaHQ/khora/pull/708) + follow-ups [#709](https://github.com/DeytaHQ/khora/pull/709) / [#710](https://github.com/DeytaHQ/khora/pull/710)). New "Why khora?" section names the four problems pure vector search doesn't solve (ingest depth, recall complexity, drift, observability). New "Engines" section gives VectorCypher and Chronicle full descriptions with when-to-pick guidance; Skeleton is explicitly marked experimental. Inline 3-engine comparison table so the choice is visible without a docs hop.
- **`docs/dream-phase.md` rewritten** ([PR #697](https://github.com/DeytaHQ/khora/pull/697) + cleanup [PR #702](https://github.com/DeytaHQ/khora/pull/702)). Adds Phase 2 + Phase 4 operations alongside Phase 1 audits, full apply-mode contract, "Apply functions" API reference section, "Research & Prior Art" section with paper citations (McClelland 1995 on CLS, Schaul 2016 on prioritized experience replay, Kreps 2011 on Kafka log compaction, MemGPT / GraphRAG / Self-RAG with arXiv IDs, Köpcke & Rahm 2010 on entity resolution, Snodgrass 1999 on bi-temporal modeling), and an explicit "LLM usage" section calling out that dream phase makes **zero LLM calls** in v0.15.
- **CI coverage floor 30% → 65%** ([PR #696](https://github.com/DeytaHQ/khora/pull/696)). Matches actual main coverage (65.88% at the time of the bump) — the floor was previously misleading. Roadmap to 85% tracked under [#695](https://github.com/DeytaHQ/khora/issues/695) as staged PRs.
- **Coverage push: +~980 statements across three modules** ([PR #703](https://github.com/DeytaHQ/khora/pull/703) `pipelines/flows/ingest.py` 18% → 63%, [PR #704](https://github.com/DeytaHQ/khora/pull/704) `query/engine.py` 47% → 72%, [PR #705](https://github.com/DeytaHQ/khora/pull/705) `engines/vectorcypher/retriever.py` 48% → 69%). Step 2 of the #695 ladder.
- **codecov.yml gradient** `30...85` → `65...85`. Badge now renders neutral-to-green starting at the actual floor instead of red.
- **Default file-sink base directory documented.** With `report_file_sink_enabled=True`, reports land under `<system temp dir>/khora-dream-reports`. On Linux that's `/tmp/...`, which is wiped on reboot — documented with operator guidance to set a persistent path.
- **`codecov.yml` ignores `examples/**`** ([PR #694](https://github.com/DeytaHQ/khora/pull/694)). Pre-emptive — the adapter examples are smoke-tested but not coverage-measured.
- **Loosened the planner-`mode="apply"` contradiction.** v0.14 planners raised `NotImplementedError` on `mode="apply"`; the orchestrator now routes apply through dedicated `apply_<op>` functions instead. Direct callers of `plan_<op>(..., mode="apply")` still hit the raise — that path is reserved for testing. Tracked for follow-up cleanup.

### Removed

- **kuzu backend** ([PR #706](https://github.com/DeytaHQ/khora/pull/706)). Deprecated in v0.9.0 with a v0.10.0 removal target that never landed. The upstream Kùzu project has been archived since the October 2025 acquisition — the dependency is dead code. The `kuzu` extra (`pip install khora[kuzu]`) is gone; the `graph-all` and `all-backends` extras no longer pull in the kuzu wheel; the `KuzuBackend` / `KuzuConfig` symbols are removed from `khora.storage.backends` and `khora.config.schema`; 909 LOC of unmaintained backend code deleted. Migration path: `pip install khora[sqlite-lance]` for embedded, `pip install khora[neo4j]` for graph DB.

## [0.14.0] — Dream-phase audit foundation

Minor release. Lands Phase 0 (foundation), Phase 1 (read-only audit operations), and Phase 3 (Rust acceleration) of the [Dream Phase umbrella (#649)](https://github.com/DeytaHQ/khora/issues/649). `Khora.dream(namespace, mode="dry-run")` is live end-to-end: operators can plan a consolidation pass over their graph, see exactly what every audit op would surface (drift thresholds, tombstone ratios, schema mismatches, orphan candidates, dead chunk references), and have those decisions emitted through three independently-togglable sinks (file, semantic-event, telemetry collector). No mutation operations ship in v0.14.0 — Phase 2 (mutation-planning ops, dry-run only) and Phase 4 (apply mode) land in a follow-up release. Audit-only is the deliberate "validate demand before committing engineering" gate.

### Added

#### Phase 0 — Foundation

- **`khora.dream` module scaffolding + `DreamConfig` ([#650](https://github.com/DeytaHQ/khora/issues/650) → PR #675).** New top-level `khora.dream` subpackage with `DreamConfig` (Pydantic settings, env-var prefix `KHORA_DREAM_*`, master switch defaults to `False`), `DreamResult`, `DreamRunInfo`, `DreamMode`, `DreamScope`, `OpKind`, and internal `DreamOp` / `DreamPlan` / `DreamReport` dataclasses. `Khora.dream()` / `Khora.dream_status()` / `Khora.dream_history()` stubbed on the public `Khora` class. Stability: top-level surface is **public** (`khora.__all__`); op-kind values and sub-dataclasses are **internal** and may evolve through Phase 1 / 2 without a major bump.
- **Migration 032 `khora_dream_runs` ([#651](https://github.com/DeytaHQ/khora/issues/651) → PR #676).** Postgres-only checkpoint table for crash-resume semantics. Records `run_id`, `namespace_id`, `mode`, `state` (init/planning/applying/completed/cancelled/crashed), `plan_hash`, `last_committed_op_seq`, `heartbeat_at`, `report_path`, and error JSONB. Indexed `(namespace_id, started_at DESC)` for `Khora.dream_history()`. Dialect-gated; sqlite_lance fixture path mirrors checkpoint state via the file sink instead.
- **Migration 033 bi-temporal columns ([#653](https://github.com/DeytaHQ/khora/issues/653) → PR #674).** Adds `valid_to`, `invalidated_at`, `invalidated_by` (UUID) to both `relationships` and `memory_facts`. Backfill is null (= "still valid"). Postgres-only partial composite indexes `ix_relationships_live` / `ix_memory_facts_live` accelerate the live-fact retrieval path. Soft-delete substrate for the Phase 4 apply-mode rollout; coexists with the legacy `memory_facts.is_active` flag.
- **Advisory lock + `DreamCapable` Protocol ([#656](https://github.com/DeytaHQ/khora/issues/656) → PR #677).** `acquire_namespace_dream_lock(session, namespace_id, timeout_seconds=60)` async context manager using `pg_advisory_xact_lock` with namespace-derived lock IDs (blake2b, domain-separated from the migration lock). Embedded fallback uses an in-process `asyncio.Lock` (cross-process safety not promised on sqlite_lance). `DreamCapable` Protocol (`plan_dream` + `apply_dream` + `dream_capabilities` property, `runtime_checkable`) — engines opt in by implementing it; the orchestrator runtime-checks before scheduling.
- **Three reporting sinks + `EventType.DREAM_*` family + telemetry contract ([#666](https://github.com/DeytaHQ/khora/issues/666) → PR #678).** File sink writes `{base_dir}/{namespace_id}/{date}/{run_id}.{summary.md,events.jsonl,manifest.json,undo.json}`. Event sink bridges into the existing `HookDispatcher` with six new `EventType.DREAM_*` values (`DREAM_RUN_STARTED`, `DREAM_PHASE_STARTED`, `DREAM_OP_DECIDED`, `DREAM_PHASE_COMPLETED`, `DREAM_RUN_COMPLETED`, `DREAM_RUN_FAILED`) — reuses `SemanticFilter` cascade with two new low-cost level-0 filter fields. Collector sink emits OTel spans + metrics declared in `docs/telemetry-contract.json` (4 public top-level spans, 4 internal inner spans, 7 public metrics — none labelled by `namespace_id` per the cardinality rule). Free-text span attributes go through `khora.telemetry.bounded_text_hash`. `redact_text` config knob (`"none"|"summary"|"all"`, default `"summary"`).
- **Orchestrator state machine + `Khora.dream()` wiring ([#661](https://github.com/DeytaHQ/khora/issues/661) → PR #684).** `DreamOrchestrator` implements INIT → PLAN → REPORT (dry-run) / APPLY → FINALIZE. Acquires the per-namespace advisory lock; calls into the registered engine plugin's `plan_dream()`; fans out `DreamOp` results through the three sinks; persists checkpoint state to `khora_dream_runs` for crash-resume; cancels between ops (never mid-op); enforces the safety floor (no Document delete, no UNIQUE-invariant break, no read-only namespace). `Khora.dream(ns, mode="dry-run")` is now reachable; `Khora.dream_status(run_id)` and `Khora.dream_history(ns, limit=...)` round-trip against `khora_dream_runs`.

#### Phase 1 — Read-only audit operations

All five ops are pure-SELECT / pure-observation. Zero LLM calls, zero mutations, zero risk to production graphs. Each returns a `DreamOp` with `decision="audit_complete"` (or `"insufficient_data"` / `"empty_namespace"`) and a structured `outputs` dict; the orchestrator routes those through the three sinks.

- **Chronicle abstention-threshold drift report ([#652](https://github.com/DeytaHQ/khora/issues/652) → PR #681).** Reads the OpenTelemetry histogram of `combined_score` / `top_score` values emitted by chronicle's existing `_compute_abstention_signals` (plus a bounded in-process ring buffer for the no-logfire path) and recommends — never applies — a threshold recalibration: e.g. "p90 `top_score` is 0.18 but `abstention_min_top_score` is 0.3 — most recalls fire `top_score_low` even on good answers; consider lowering to 0.15". Refuses below `abstention_drift_min_samples` (default 1000) with `decision="insufficient_data"`.
- **Chronicle `memory_facts` tombstone audit ([#654](https://github.com/DeytaHQ/khora/issues/654) → PR #683).** Counts active / inactive (legacy `is_active`) / invalidated (`invalidated_at IS NOT NULL`) rows. Reports `tombstone_ratio`, oldest-tombstone age, p50/p90 age of inactive facts, and top-K offenders by age. Recommends a retention threshold; the Phase 2 compaction op (#664) is the actual reclaimer.
- **VectorCypher schema-drift report against `ExpertiseConfig` ([#655](https://github.com/DeytaHQ/khora/issues/655) → PR #679).** Diffs the multiset of observed `entity_type` / `relationship_type` strings against the active `ExpertiseConfig`, with four buckets: types in data but not in config, types in config but unused, frequency-delta >50% since previous run, and the relationship-type variant of each. Never auto-normalizes — `ExpertiseConfig` is declarative user intent; normalization is operator policy (deferred to Phase 5).
- **VectorCypher PageRank-based orphan-entity report ([#657](https://github.com/DeytaHQ/khora/issues/657) → PR #682).** Builds the namespace's entity-relationship graph, down-weights `ASSOCIATED_WITH` co-occurrence edges to 0.2 so they don't dominate, runs the existing `_accel.pagerank` Rust kernel, and flags entities matching all of: PR score in the bottom 5th percentile AND `mention_count <= 1` AND no recent recall hits. Returns `archive_candidate=true` flags; the operator decides what to do next.
- **VectorCypher `source_chunk_ids` array-length audit ([#659](https://github.com/DeytaHQ/khora/issues/659) → PR #680).** Joins entities × chunks (Postgres `unnest`; SQLite Python-side) to count dead chunk-UUID references per entity, then reports the array-length distribution (p50/p90/p99) and top-K offenders. Surfaces the GC candidates for the Phase 2 source-chunk-ids GC op (#662).

#### Phase 3 — Rust acceleration

- **`khora._accel.block_and_score_pairs` ([#663](https://github.com/DeytaHQ/khora/issues/663) → PR #685).** New Rust kernel in `khora-accel` for pairwise cosine similarity with optional token-prefix name blocking. Powers the Phase 2 cross-batch entity-resolution op (#658) at namespace scale: at N≈100k entities, naive pairwise is ~5×10⁹ cross-products (~30s wall); token-blocking cuts the candidate set ~10× on realistic name distributions. **Benchmark: 6.4× speedup** vs `pairwise_cosine_above_threshold` at N=10k, D=128, threshold=0.85 (72.9 ms vs 463.9 ms). Falls back to pure-Python when the `khora[rust]` extra isn't installed. Same module pattern as the existing `cosine.rs` kernels — rayon-parallel, GIL released via `py.detach()`.

### Changed

- **`khora-accel` version bumped from 0.13.0 to 0.14.0** (lockstep with khora itself per the version-bump contract). Root `pyproject.toml`'s `rust` extra pin updated to `khora-accel==0.14.0`; `rust/Cargo.lock` regenerated.
- **Telemetry contract additions** (`docs/telemetry-contract.json`): four public top-level `khora.dream.*` spans (`run`, `phase`, `llm_call`, `undo`), four internal per-op spans (`op`, `entity_merge`, `edge_prune`, `community_summary`) plus five per-Phase-1-op spans, and seven public aggregate metrics (`khora.dream.runs_total`, `khora.dream.run.duration`, `khora.dream.phase.duration`, `khora.dream.ops_total`, `khora.dream.llm.tokens`, `khora.dream.undo_invocations_total`, `khora.dream.report.write_failures_total`). None labelled by `namespace_id` (cardinality rule).

### Not yet shipped (planned for follow-up releases)

- **Phase 2 — mutation-planning ops (dry-run only).** Cross-batch entity resolution, centroid recompute, source_chunk_ids GC, chronicle fact compaction, chronicle event clustering. All five will land in a follow-up release; their tickets are written (#658, #660, #662, #664, #665) and link the umbrella.
- **Phase 4 — apply mode.** Flips Phase 2 ops to actually mutate state. Requires the audit-only release to bake first; the kill criterion is whether operators actually invoke `kb.dream()` in production within ~30 days.
- **Phase 5 — advanced operations.** Community detection + summaries (opt-in, LLM-heavy), edge pruning by weight × recency, contradiction detection, schema-drift normalization (operator-supplied mapping).

## [0.13.0] — Agentic framework adapters; session_id first-class; SurrealDB 2.0 stable

Minor release. Five new opt-in adapters for agentic frameworks (CrewAI, LangGraph, Google ADK, OpenAI Agents SDK, LlamaIndex), a new `session_id` first-class column with cascade-delete and TTL helpers, and the SurrealDB 2.0 stable pin. No public API removals.

### Added

- **`khora.integrations` adapter foundation ([#619](https://github.com/DeytaHQ/khora/issues/619) → PR #631).** New subpackage exposing three runtime-checkable Protocols (`MemoryAdapter`, `RetrieverAdapter`, marker `KhoraIntegration`), an entry-point registry (group `khora.integrations`, with `register()` test escape hatch), the `_sync.run_sync` cross-thread bridge, and a config-hash-keyed `Khora.shared()` process-wide singleton. Adapter submodules MUST NOT import their framework at top level — enforced by `tools/check_optional_imports.py` (AST lint).
- **CrewAI adapter ([#623](https://github.com/DeytaHQ/khora/issues/623) → PR #633).** `khora.integrations.crewai.KhoraMemory` is a drop-in `StorageBackend` for CrewAI's unified `Memory`. Install with `pip install khora[crewai]`. Tz-naive recency math (`_strip_tz`) keeps CrewAI's `datetime.now() - record.created_at` happy against khora's tz-aware UTC timestamps.
- **LangGraph adapter ([#624](https://github.com/DeytaHQ/khora/issues/624) → PR #634).** `khora.integrations.langgraph.KhoraStore` implements `BaseStore` for semantic long-term memory inside `StateGraph` runners. Install with `pip install khora[langgraph]`.
- **Google ADK adapter ([#626](https://github.com/DeytaHQ/khora/issues/626) → PR #642).** `khora.integrations.google_adk.KhoraMemoryService` implements `BaseMemoryService` (`add_session_to_memory` + `search_memory`). Namespace is UUID5 of `adk:{app_name}:{user_id}`; `Session.id` round-trips via `session_id`. Memory only — no `KhoraSessionService` in v1 (ADK's `DatabaseSessionService` already covers turn state). Install with `pip install khora[google-adk]`.
- **OpenAI Agents SDK adapter ([#625](https://github.com/DeytaHQ/khora/issues/625) → PR #643).** `khora.integrations.openai_agents.KhoraSession` implements `agents.memory.session.SessionABC`; `khora_recall_tool()` is a `FunctionTool` factory; `KhoraMemoryHooks` is a `RunHooks`-shaped auto-persist callback. Items round-trip via `Document.metadata.custom["oai_item"]` (verbatim JSON); ordering via `oai_seq`. Pinned tight (`openai-agents>=0.17,<0.18`). Install with `pip install khora[openai-agents]`.
- **LlamaIndex adapter ([#627](https://github.com/DeytaHQ/khora/issues/627) → PR #644).** `khora.integrations.llamaindex.KhoraRetriever` is an async-only `BaseRetriever`; `KhoraMemoryBlock` is a `BaseMemoryBlock` factory for long-term memory; `KhoraChatStore` is a deprecated `BaseChatStore` shim for legacy `ChatMemoryBuffer` users (emits `DeprecationWarning`). Pin is narrow (`llama-index-core>=0.14,<0.15`). Install with `pip install khora[llamaindex]`.
- **`session_id` is a first-class column ([#620](https://github.com/DeytaHQ/khora/issues/620) → PR #632).** Migration 030 adds nullable `session_id UUID` to `documents`, `chunks`, `memory_events`, `chronicle_events`, and `memory_facts`. Migration 031 adds Postgres-only partial composite indexes `ix_chunks_ns_session` / `ix_documents_ns_session (namespace_id, session_id) WHERE session_id IS NOT NULL` plus a BRIN `ix_chunks_session_created_brin (session_id, created_at)` for time-bounded session replay. New public API: `Khora.remember(..., session_id=…)`, `Khora.submit_batch(..., session_id=…)`, `Khora.forget_session(ns, sid)` cascade-delete, and the opt-in `khora.gc.expire_sessions(before=…)` TTL helper.
- **Adapter examples CI infrastructure ([#622](https://github.com/DeytaHQ/khora/issues/622) → PR #630).** Every adapter ships `examples/integrations/<name>/example.py` that runs without external services (sqlite_lance fixture + mock LLM helpers under `examples/_helpers/`). The `python title="example.py"` block in `docs/integrations/<name>.md` must be byte-identical to that file. The `examples-smoke` CI job gates drift via `tools/check_examples_drift.py` and smoke-runs each example under a 30s timeout.
- **Integrations roll-up docs (PR #645, PR #647).** README "Integrations" section + new `docs/integrations/index.md` landing page covering all five adapters. New `make install` / `make install-adk` Makefile targets pick the correct extras combo (see Changed → CI section).
- **Contributor Covenant Code of Conduct ([PR #641](https://github.com/DeytaHQ/khora/pull/641)).** Standard `CODE_OF_CONDUCT.md` for the OSS repository.

### Changed

- **SurrealDB SDK pin bumped to GA stable (PR #646).** `surrealdb>=2.0.0a1` → `surrealdb>=2.0.0,<3.0` at all three sites (`khora[surrealdb]`, `khora[embedded]`, `khora[all-backends]`). Drops the alpha caveat from CLAUDE.md.
- **Direct-dependency floor pins raised (PR #646).** 25 floor pins in `[project] dependencies` and `[project.optional-dependencies]` bumped to match the resolved versions in `uv.lock`, so the declared minimums reflect the actually-tested versions. Notable: `sqlalchemy 2.0.47 → 2.0.49`, `litellm 1.81.15 → 1.84.0`, `neo4j 6.1.0 → 6.2.0` (7 sites), `sentence-transformers 5.2.3 → 5.4.1`, `opentelemetry-{api,sdk,exporters} 1.27.0 → 1.34.1`, `pyarrow 18.0.0 → 24.0.0`, `lancedb 0.25 → 0.30.0`, `pytest-xdist 3.6 → 3.8.0`, `ruff 0.15.2 → 0.15.12`, `ty 0.0.18 → 0.0.34`, `hypothesis 6.140.0 → 6.152.4`, `weaviate-client 4.20.1 → 4.21.0`, `logfire 4.0 → 4.6.0`. For dual-version packages caused by the crewai/google-adk extras conflict (`pydantic-settings`, `opentelemetry-api`, `aiosqlite`, `lancedb`, `logfire`), the lower of the two resolved versions is used as the floor so both combos still satisfy.
- **`crewai` and `google-adk` declared as mutually-exclusive extras (PR #642).** The two extras pin incompatible `opentelemetry-api` ranges (crewai `<1.35`, google-adk `>=1.36`). Declared in `[tool.uv].conflicts`. `uv sync --all-extras` is rejected with an explicit error; use `--no-extra google-adk` (crewai combo, CI default) or `--no-extra crewai` (google-adk combo). The new `make install` / `make install-adk` targets (PR #647) pick a combo for you.
- **`UV_NO_SYNC=1` in Makefile + CI (PR #642).** Prevents `uv run` from silently re-resolving the venv to the other extras-combo branch of the lockfile mid-test-run, which otherwise breaks logfire's in-tree imports when otel-sdk version flips.
- **Dedicated `test-google-adk` CI job (PR #642).** The default `test` job uses the crewai combo (`--no-extra google-adk`); a separate job builds the google-adk combo and runs the adapter's unit tests + the no-eager-imports probe.
- **`khora-accel` lockstep pin → 0.13.0.** Matches the `rust/khora-accel/Cargo.toml` version in this release.

### Fixed

- **SurrealDB UUID quoting in `get_neighborhood` ([#635](https://github.com/DeytaHQ/khora/issues/635) → PR #636).** `get_neighborhood` was string-interpolating bare UUIDs into Cypher-like SurQL, which failed parsing on UUIDs containing characters that need escaping (notably hyphens treated as subtraction in some surrealdb-rust versions). Routed through parameter binding so RecordIDs are quoted by the SDK.
- **Exception chain no longer leaks unredacted DSN via `__cause__` ([PR #640](https://github.com/DeytaHQ/khora/pull/640)).** Connection-pool error reraises preserved the original asyncpg exception's `__cause__`, which contained the raw DSN with credentials. The wrapper now scrubs `__cause__` before re-raising, so a `KhoraConnectionError` in operator logs no longer pastes secrets into the trace.
- **`StorageConfig` `__repr__` no longer exposes plaintext credentials (DYT-4171, DYT-4300).** `neo4j_user` and related fields are now marked `repr=False`. `DSN` fields with no password but a userinfo are also properly redacted (DYT-4122). Affects Pydantic settings dumps and any `print(config)` output in service start-up logs.

### Removed

- Nothing public removed. Existing extras and APIs continue to work; this release is additive.

## [0.12.1] — Chunk source_timestamp propagation

Patch release. Single bug fix.

### Fixed

- **Chunk `source_timestamp` is no longer lost during ingest ([#615](https://github.com/DeytaHQ/khora/issues/615) → PR #616).** `chunk_document()` built `Chunk` objects with only `created_at` set, never `source_timestamp`. The ingest pipeline already parses doc-level `source_timestamp` from connector metadata (`occurred_at` / `sent_at` / etc.) and stamps it on `Document`, but the field never reached the chunks — so every recall on every (engine × backend) cell returned `chunk.source_timestamp=None`. This silently broke date-bounded recalls: the query layer's temporal scoring fell back to `chunk.created_at` (ingest time), so "last week" queries surfaced older historical rows that happened to be ingested recently. Fix is a one-line propagation in `pipelines/tasks/chunk.py` — upstream of every storage adapter, so it lights up all six (engine × backend) cells. Two regression tests cover propagation when the document has a `source_timestamp` and `None`-preservation when it doesn't (no invented timestamps).

## [0.12.0] — Temporal Phase D: HyDE + reranker + BRIN; hooks LLM cost controls; PPR enabler; SurrealDB remote transactions

Minor release. Substantial additive surface — five new feature-flagged retrieval channels and one new Python subpackage — plus two correctness bug fixes that change scoring/behaviour on graph-only and graph-less stacks. No public API removals; the `enable_hyde` flag accepts a new string shape (`auto`/`always`/`never`) but the legacy boolean form normalizes transparently.

### Fixed

- **`unify_entities` no longer crashes on graph-less stacks ([#587](https://github.com/DeytaHQ/khora/issues/587) → PR #588).** `expansion.py:load_entities` / `load_relationships` were calling `storage.graph.get_entities_by_namespace` and `storage.relational.get_entities_by_namespace` — methods that exist on **no** backend. Every call into the expansion pipeline (chronicle, vectorcypher, skeleton; PG-only and PG+Neo4j and sqlite+lancedb) crashed with `AttributeError`. Routed through `StorageCoordinator.list_entities` / `list_relationships` instead, and extended the coordinator to fall back to the vector backend when no graph backend is configured. `PgvectorBackend` gains `list_entities` / `list_relationships` so chronicle+PG-only stacks now serve the entity table without a Neo4j requirement.
- **`find_related_entities` score decay restored on graph-only backends ([#581](https://github.com/DeytaHQ/khora/issues/581) → PR #589).** VectorCypher's `find_related_entities` applies `score = 1 / (1 + distance)` on the Neo4j (dual-nodes) path, but the graph-only path (sqlite_lance, surrealdb) hard-coded `1.0` because `get_neighborhood` did not expose per-entity distance. The engine now BFS-walks the returned relationships as an undirected adjacency to recover min hop-distance and applies the same decay; results are sorted descending. No backend protocol changes — works across `sqlite_lance` and `surrealdb` without further adapter work.
- **Release pipeline no longer races the merge-commit CI ([#554](https://github.com/DeytaHQ/khora/issues/554) → PR #591).** Tag pushes that land within ~30s of the bump-PR merge raced the merge commit's `ci.yml` run; `verify-ci-green` read the runs index, found no successful run yet, and failed. We worked around this manually on v0.10.6 / v0.11.0 / v0.11.1 via `workflow_dispatch`. Replaced the one-shot check with a bounded 10-minute poll that classifies the latest run into `success` / `failed` / `running` and sleeps on `running`. Window tunable via the `VERIFY_CI_GREEN_TIMEOUT_SECONDS` repo variable.

### Added

- **Temporal-anchored HyDE for RECENCY queries ([#592](https://github.com/DeytaHQ/khora/issues/592) → PR #602, Phase D1).** When HyDE fires on a query the temporal detector classifies as `RECENCY` / `STATE_QUERY` / `CHANGE`, `HyDEExpander` selects a system prompt that anchors the hypothetical to today's ISO date with explicit dates / weekdays / relative-time markers. Other categories keep the generic time-blind prompt. Zero additional LLM calls; only the prompt string changes. Category detection runs in Rust Aho-Corasick (sub-ms). See [`docs/query-engine/temporal-queries.md`](docs/query-engine/temporal-queries.md#temporal-anchored-hyde).
- **HyDE-Cypher templated graph queries ([#595](https://github.com/DeytaHQ/khora/issues/595) → PR #610, Phase D2, opt-in).** New module `khora.query.hyde_cypher` with three parameterized Cypher templates (`recent_by_type`, `entity_relationships`, `cooccurrence`). An LLM picks a template and fills slots; slot values are bound via Neo4j `$placeholder` parameters (never string-interpolated) and validated against `ExpertiseConfig` whitelists. Failures (timeout, hallucinated id, validation error) degrade to text-HyDE — never crashes the query. Enable via `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`. **Default OFF** pending an A/B on a hand-curated structured-query set.
- **BRIN index on `chunks.created_at` ([#593](https://github.com/DeytaHQ/khora/issues/593) → PR #603, Phase D4).** New migration `029_chunks_created_at_brin` adds `CREATE INDEX CONCURRENTLY` (in an autocommit block) on the time-correlated `created_at` column, `pages_per_range = 32`. KB-sized footprint that doesn't compete with HNSW or B-trees; helps long-range archive / export scans. Postgres-only — SQLite-backed sqlite_lance stacks skip the migration silently via a dialect gate.
- **Cross-encoder reranker date-prefix experiment ([#594](https://github.com/DeytaHQ/khora/issues/594) → PR #609, Phase D5, opt-in).** `CrossEncoderReranker(include_date_prefix=True)` prepends `[YYYY-MM-DD] ` to each candidate's content. Source priority: `metadata.custom.occurred_at` → `metadata.custom.sent_at` → `metadata.created_at`. `create_reranker` cache key now includes the flag so the two variants coexist without a 500ms model reload. **Default OFF** pending A/B.
- **PPR enabler: `pagerank(personalization=...)` ([#597](https://github.com/DeytaHQ/khora/issues/597) → PR #604).** Both the Rust impl (`rust/khora-accel/src/pagerank.rs`) and the Python fallback (`khora._accel.pagerank`) accept an optional L1-normalizable personalization vector. PPR formula: `r = (1 - d) * p + d * Mᵀ r`. When `personalization` is `None` or uniform-equivalent, this reduces to standard PageRank — every existing call site continues to receive identical scores. Validation: negatives clipped to 0, length mismatch falls back to uniform, all-zero falls back to uniform. The Python wrapper only forwards the new kwarg when set so older `khora-accel` wheels keep working until rebuilt. **Enabler only** — the BFS+RRF → PPR swap in VectorCypher is still gated on the graph-density audit (#598).
- **PPR graph-density audit reporter ([#598](https://github.com/DeytaHQ/khora/issues/598) → PR #605).** New `khora.diagnostics.graph_density.compute_graph_stats()` returns per-namespace `|V|`, `|E|`, mean + median degree, connected-component count, largest-CC fraction, mean degree restricted to the largest CC, plus a `meets_ppr_threshold` flag applying the #598 decision rule (≥3 connected components OR mean degree ≥5 in the largest CC). Operator script: `scripts/audit_graph_density.py` emits CSV or JSON plus a stderr verdict. **`khora.diagnostics` is explicitly NOT stable public API** and may be renamed without a major-version bump.
- **Hooks Level 2 LLM cache + per-subscription budget ([#601](https://github.com/DeytaHQ/khora/issues/601) → PR #607).** Cross-batch decision cache keyed on `(filter_id, bounded_text_hash(event_summary))` — repeat bulk-upsert events that share `name`/`type`/`description` short-circuit the LLM. TTL + LRU eviction, configurable via `KHORA_HOOKS_LLM_CACHE_SIZE` (default `2048`) and `KHORA_HOOKS_LLM_CACHE_TTL_SECONDS` (default `3600`). Per-subscription rolling-hour token budget alongside the namespace cap so a single noisy filter cannot drain its namespace allowance: `KHORA_HOOKS_LLM_MAX_TOKENS_PER_SUBSCRIPTION_PER_HOUR` (default `0` = disabled). New telemetry metrics `khora.hooks.llm.cache_hits_total` (label `category={match,no_match}`) and `khora.hooks.llm.cache_misses_total`.
- **Hooks Level 2 intra-batch event-summary coalescing ([#608](https://github.com/DeytaHQ/khora/issues/608) → PR #611).** Within a single LLM batch, `_evaluate_bucket` now deduplicates pending pairs by `event_summary_hash` before building the prompt and fans the decision out to every awaiting future. Combined with the #607 cross-batch cache, a burst of 50 identical events spends exactly **1 LLM call** even on a cold cache (the original acceptance bar was ≤2). When events differ this is a no-op.
- **SurrealDB remote-mode transactions ([#541](https://github.com/DeytaHQ/khora/issues/541) → PR #612).** New `SurrealDBConnection.transaction()` async context manager wraps the body in `BEGIN TRANSACTION` / `COMMIT TRANSACTION` on remote (`ws://`) mode, with `CANCEL TRANSACTION` on exception (original error preserved if `CANCEL` itself fails). Embedded (`surrealkv://`) and memory (`memory://`) modes are no-ops — surrealkv raises on `BEGIN`, so the existing per-statement-atomicity contract is preserved. Companion `execute_batch([(sql, bindings), ...])` joins statements with `;` for an embedded-mode batched alternative; rejects parameter-name collisions across statements. New `supports_transactions` property surfaces the per-mode capability without reading internal state.

### Changed

- **`enable_hyde` flag value shape.** `KHORA_QUERY_ENABLE_HYDE` now accepts string values `auto` (default), `always`, or `never`. Legacy booleans (`True` / `False`) are still accepted and normalize to `always` / `never` respectively — no breaking change at the API boundary, but operators reading the config docs in v0.11.x and earlier will see a different field shape now.
- **Pre-v0.12.0 documentation pass (PR #613).** Top-level README quickstart fixed (the v0.11.x example called `kb.create_namespace("demo")` which is a `TypeError` — `create_namespace` is keyword-only). Hooks doc rewritten to cover Phase 2 + v0.12.0 surface: EventBridge `match` DSL, `CHUNK_ENTITIES_RESOLVED`, default-OFF Level 2 cost warning, per-namespace + per-subscription budgets, cache + intra-batch coalescing, co-occurrence example. Fixed three wrong default values in `docs/hooks/semantic-hooks.md` (`gpt-4o-mini` → `gpt-4.1-nano`, `0.7` similarity → `0.5`). `docs/api-reference.md` `create_namespace` and `register_engine` signatures corrected; `BatchHandle` / `DocumentResult` added; new "Advanced (opt-in, v0.12.0)" section. CLAUDE.md gotchas extended with bullets for dialect-gated migrations, SurrealDB transactions, graph-less-stack entity listing, temporal HyDE, HyDE-Cypher, reranker date-prefix, hooks Level 2 cost controls, and the `khora.diagnostics` non-stability disclaimer.

### Removed

Nothing. Every v0.11.x symbol still exists with the same signature.

## [0.11.1] — Semantic hooks Phase 1: bugfixes + opt-in Level 2 LLM evaluator

Patch release covering [#576](https://github.com/DeytaHQ/khora/issues/576)
(PR #577). Closes the trust gap between docs and code in
`khora.hooks`. Every claim in `docs/hooks/semantic-hooks.md` is now
true; no public API removals.

### Fixed

- **Level 1 (embedding similarity) was structurally unreachable.**
  `HookDispatcher.dispatch()` reads `event.data["embedding"]` but the
  ingest pipeline never populated it because entity events fired
  *before* the parallel `_embed_entities` branch completed. Operators
  who set `SemanticFilter(description=..., similarity_threshold=0.7)`
  per the docs were silently getting Level 0 (type filter) only. Now
  ingest defers entity-event dispatch until both gather phases
  complete, backfilling the freshly-computed embedding into each
  pending event's payload.
- **`callback_timeout_seconds` config field was never honored.** The
  field existed on `SemanticHooksConfig` but `_safe_invoke` had no
  `asyncio.wait_for`. A misbehaving callback could hang ingest
  indefinitely. Now timed-out callbacks log a warning and release the
  concurrency semaphore.
- **`SemanticFilter.description` was never auto-embedded on
  subscribe.** Filters registered with a description but no
  precomputed embedding are now queued and drained by
  `Khora.connect()` via the engine's embedder. Closes the silent
  degradation where operators following the docs (no manual embed
  step) got Level 0 only.
- **`EventType.ENTITY_MERGED` was defined but never dispatched.** The
  cross-tool unifier merged entities silently. Now emits
  `entity.merged` with `{merged_from, surviving_id, source_tools,
  strategy, namespace_id}` — the dedup signal corporate-data
  customers care most about.

### Added

- **`RECALL_REQUESTED` / `RECALL_RESULTS_READY` / `RECALL_COMPLETED`
  events** fired from `Khora.recall()`. Shared `recall_id` (UUID)
  across the three events so subscribers can correlate. Payload caps
  (top 20 entity/chunk IDs) bound event size. Hook dispatch wrapped
  in try/except — failures never break `recall()`.
- **Optional Level 2 LLM filter evaluator** (`khora.hooks.llm_evaluator.LLMFilterEvaluator`).
  Default OFF (`KHORA_HOOKS_LLM_EVALUATION_ENABLED=false`).
  Micro-batched (10 pairs / 100 ms window), JSON-schema output via
  `khora.config.llm.acompletion` with `_telemetry_op="hooks.filter_eval"`,
  per-namespace token-budget cap (default 10k tokens/hour ≈ 100
  evaluations — deliberately conservative; operator tunes up).
  Fail-open on LLM errors / budget breach. Only runs for filters
  that set `examples=[...]`.
- **New telemetry contract entries**: `khora.hooks.llm.evaluations_total`
  (counter, labels `category={match,no_match,timeout,budget_exceeded}`),
  `khora.hooks.llm.tokens_total` (labels `direction={input,output}`),
  `khora.hooks.llm.throttled_total`.

### Changed

- **`KhoraConfig.hooks: Any = None` → `SemanticHooksConfig`** (typed).
  The `Any` placeholder dated to a circular-import worry; lazy
  module-level import resolves it cleanly. Operators can now read
  `cfg.hooks.enabled`, `cfg.hooks.callback_timeout_seconds`, etc.
  Type-narrowing, not breaking — anyone reading `cfg.hooks` and
  getting `None` today now gets a populated config.
- **`CrossToolUnifier.unify()` is now async.** Required to dispatch
  the new `ENTITY_MERGED` event. Both production call sites
  (`SemanticExpander.expand`, `pipelines/flows/expansion.unify_entities`)
  updated; the 5 existing unifier tests converted to async.

## [0.11.0] — Temporal retrieval overhaul: scoring fixes + ingestion contract + entity-anchored fast path

Three-phase rework of khora's temporal retrieval, addressing the production
complaint that queries like *"what are the latest action items from recent
meetings?"* returned stale records on corporate-data corpora (Slack, email,
calendar, Salesforce). All new behavior is **feature-flagged off by
default**; operators opt in per-namespace via `KHORA_QUERY_TEMPORAL_*` env
vars. Existing consumers see no behavior change unless they enable flags.

Closes [#567](https://github.com/DeytaHQ/khora/issues/567) (Phase A — scoring),
[#568](https://github.com/DeytaHQ/khora/issues/568) (Phase B — ingestion),
[#569](https://github.com/DeytaHQ/khora/issues/569) (Phase C — entity-anchored).

### Added

- **Phase A: scoring & retrieval (#567 / PR #571).**
  - `_calculate_recency_scores(reference_mode="wall_clock" | "relative")` — wall-clock
    is production-correct (was a relative max-in-set heuristic). `KHORA_BENCH_MODE=true`
    forces `relative` for benchmark replay.
  - Synthetic date floor for RECENCY/CHANGE queries with no parseable date
    (e.g., bare "latest", "recent"). Defaults: RECENCY=30d, CHANGE=60d.
  - `ANTI_RECENCY_TOKENS` veto list — historical / counterfactual queries
    ("ever", "history of", "would have", "if we had", "back when", "at one
    point", etc.) suppress the synthesized floor.
  - **LLM disambiguation tier** — when Aho-Corasick fires RECENCY/CHANGE
    AND the query contains an ambiguity-trigger token (`would`, `could`,
    `if `, `previously`, etc.), a short LLM call classifies the query as
    RECENT / HISTORICAL / COUNTERFACTUAL / NEUTRAL. Floor vetoed for
    non-RECENT outputs. Per-query cache; bounded cost.
  - Parallel "recency channel" — pure `ORDER BY occurred_at DESC LIMIT N`
    SQL fused via RRF pool augmentation. Cosine relevance floor (0.40
    default) gates which fresh-but-irrelevant chunks can enter.
  - Per-source decay dict — Slack 3d / email 7d / calendar 14d / Salesforce
    180d / `_default` 14d. Looked up via `chunk.metadata.custom["source_system"]`.
  - pgvector `hnsw.iterative_scan = strict_order` capability-probed and
    enabled when a temporal filter is set (avoids HNSW recall collapse
    under selective filters).
  - New public exports: `TemporalIntent`, `classify_temporal_intent_llm`,
    `has_anti_recency_token`, `has_ambiguity_trigger`, `ANTI_RECENCY_TOKENS`.
  - Seven new `QuerySettings.temporal_*` knobs gating each behavior.

- **Phase B: ingestion contract (#568 / PR #573).**
  - `khora.pipelines.ConnectorMetadata` — public `TypedDict(total=False)`
    documenting the canonical metadata-field surface for connector authors.
  - `khora.pipelines.SourceSystem` — `Literal["slack","email","calendar","salesforce","jira","linear","manual"]`.
  - `khora.pipelines.validate_connector_metadata(metadata, source_type) -> list[str]` —
    advisory warnings; connector CI runs it pre-ingest.
  - `khora.pipelines.CANONICAL_TIMESTAMP_FIELDS` — tuple matching the extractor priority.
  - Source-type-aware timestamp priority: `source_type in {calendar,meeting,event}`
    now prefers `occurred_at` over `sent_at`.
  - New span attribute `chunk.occurred_at.source = "metadata" | "ingest_fallback"`
    on chunk construction. Operators can detect silent connector breakage.
  - New metric `khora.ingest.source_timestamp.fallback_count` — counter,
    bounded `source_type` label. Throttled WARN log when a canonical-source
    connector misses the timestamp.
  - `docs/extraction/ingestion-pipeline.md` § "Canonical metadata fields
    per source" — full mapping table per source system.

- **Phase C: entity-anchored fast path (#569 / PR #574).**
  - New `QueryComplexity.TYPED_ENTITY_RECENT` routes queries matching
    `(latest|most recent|newest|recent) (action items|decisions|blockers|risks|...)`
    through a single Cypher fast path.
  - New retriever method `_typed_entity_recent_retrieve()` — single
    `MATCH ... MENTIONED_IN ... max(c.occurred_at) ORDER BY DESC` query
    with status filter for ACTION_ITEM / COMMITMENT / OPEN_QUESTION
    (excludes done / cancelled / completed / closed). Graceful fallback
    when no typed entities exist or Neo4j is unavailable.
  - New opt-in extraction skill `builtin:meetings` with 4 typed entity
    types: **ACTION_ITEM** (assignee, due_by, status), **DECISION**
    (decided_on, rationale), **BLOCKER** (blocking_for, severity), **RISK**
    (likelihood, impact). High-precision prompt with anti-fabrication
    guardrails.
  - Composite index migration `028_typed_entity_recency_index` —
    `(namespace_id, entity_type, created_at DESC)` on `entities`. Plus
    matching Neo4j `entity_type_recency` index.
  - Measurement scaffold for action-item extraction precision/recall
    (`scripts/measure_action_item_extraction.py` + 5-example stub
    fixture). 50-example expansion gates GA per the Devil's-Advocate
    review.

### Changed

- **`RetrievalParams.prefer_current` decoupled from `temporal_sort`** (#569).
  Regression fix: ORDINAL queries ("which came first") no longer filter
  out historical entities by accident. Per-category: RECENCY / STATE_QUERY
  / CHANGE → True; NONE / EXPLICIT / ORDINAL / AGGREGATE → False.

### Telemetry contract

Contract version 1.1 → 1.2 (additive only):
- New spans: `khora.vectorcypher.recency_floor_synthesis`,
  `khora.vectorcypher.recency_channel`,
  `khora.vectorcypher.typed_entity_recent`,
  `khora.ingest.chunk_temporal_attribution`.
- New metrics: `khora.query.temporal.floor_applied_total`,
  `khora.query.temporal.recency_channel_fired_total`,
  `khora.recall.recency.query_to_top1_age_days`,
  `khora.ingest.source_timestamp.fallback_count`.
- All bounded labels per the cardinality rule; no `namespace_id` on any metric.

### Benchmark validation (LoCoMo `--small`, 92 questions, 4 iterations)

Final iteration (v4: floor=0.40, recency channel skipped on anti-recency/LLM
veto) vs baseline:

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| counterfactual_accuracy | 0.6667 | 0.6667 | ±0 |
| abstention_accuracy | 0.4583 | 0.4583 | ±0 |
| answer_accuracy | 0.5382 | 0.5481 | +1.0pp |
| implicit_inference_accuracy | 0.7917 | 0.7917 | ±0 |
| mrr | 0.6327 | 0.6401 | +0.7pp |
| ndcg@15 | 0.6829 | 0.6909 | +0.8pp |
| recall@15 | 0.8478 | 0.8587 | +1.1pp |
| mean latency | 342.6ms | 363.3ms | +6% (LLM tier) |

No regressions on counterfactual or abstention; meaningful gains on IR
metrics. Production-shape benchmark with mixed-era timestamps tracked
at [DeytaHQ/khora-benchmarks#301](https://github.com/DeytaHQ/khora-benchmarks/issues/301).

### Acknowledgements

Devil's-Advocate review predictions all came true and were addressed:
- 14d floor regressing counterfactual (#10 in the original review) → tightened
  defaults + anti-recency veto + LLM disambiguation.
- Wall-clock switch tanking benchmark replay (#3) → `KHORA_BENCH_MODE` gate.
- Bare-stopword false positives (#2) → narrowed `ANTI_RECENCY_TOKENS`.

## [0.10.8] — OTel-first telemetry, vanilla OpenTelemetry SDK extras, observability docs

Closes [#564](https://github.com/DeytaHQ/khora/issues/564) — make khora's
observability stack vendor-neutral. Logfire moves from "the only path"
to "one of several supported backends." Backward-compatible for
existing `khora[logfire]` users.

### Added

- **`pip install khora[otel]`** — pulls `opentelemetry-sdk` +
  `opentelemetry-exporter-otlp-proto-http`. Honors the standard
  `OTEL_*` env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`,
  `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_HEADERS`,
  `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`,
  `OTEL_TRACES_SAMPLER`, etc.). Ships spans/metrics to any
  OTLP-compatible collector or vendor (Tempo, Jaeger, Honeycomb,
  Datadog, New Relic, Dynatrace).
- **`pip install khora[otel-grpc]`** — composes `khora[otel]` with
  the gRPC OTLP exporter for sites that prefer Bolt-style transport.
- **`khora.telemetry.configure_telemetry()`** — single in-code entry
  point with precedence: caller-supplied providers → existing host
  provider → `LOGFIRE_TOKEN` env → `OTEL_*` env → no-op. Idempotent;
  returns a `TelemetryHandle` for inspection and explicit shutdown.
- **`khora.telemetry.diagnostics()`** — prints active provider class,
  endpoint, contract version, and the OTel env. Run this first when
  spans appear to be missing.
- **`khora.telemetry.shutdown_telemetry_providers()`** — force-flush
  and shutdown only the providers khora installed; host-owned
  providers are left untouched.
- **Resource attribute `khora.telemetry.contract.version`** — exported
  alongside SDK defaults so dashboards can filter by telemetry-schema
  version independently of the package version.
- **`docs/observability.md`** — single canonical observability page.
  Covers install paths, env-var contract, programmatic config,
  precedence, sampling/cost guidance, vendor recipes (Honeycomb,
  Grafana Cloud, Datadog, local Jaeger), and the "I see no spans"
  troubleshooting checklist.
- **`tests/unit/telemetry/test_otel_parity.py`** — vanilla-OTel parity
  gate. Runs against `InMemorySpanExporter` + `InMemoryMetricReader`
  with no logfire path and asserts every public span/metric is emitted
  through the OTel API.
- **`scripts/bench_telemetry_overhead.py`** — regression gate for the
  no-provider trace/decorator hot path; current baseline ~3 µs/call,
  budget 5 µs/call.
- **`tests/integration/test_otel_smoke.py`** — binds an in-process
  OTLP/HTTP listener, runs `configure_telemetry()`, and asserts spans
  actually leave the process. Complements the unit-level parity gate.

### Changed

- **`khora.telemetry.trace_span()` yields the real OTel
  `opentelemetry.trace.Span`** rather than a wrapper. Call-site shape
  is unchanged (`set_attribute` / `set_attributes` on the OTel Span
  protocol are identical to the old facade). The `Span` /
  `LogfireSpan` / `NoOpSpan` ABCs in the previous
  `logfire_integration.py` module have been removed.
- **`khora.telemetry.metric_counter` / `metric_histogram` /
  `metric_gauge_callback`** delegate to the OTel `Meter` directly. No
  more `_HAS_LOGFIRE` branching; when no MeterProvider is set, OTel's
  proxy instruments quietly swallow calls.
- **`@trace` decorator** no longer short-circuits on missing logfire.
  Spans always open; the OTel API's `NonRecordingSpan` keeps the
  no-provider path effectively free (~3 µs/call vs. 0.4 µs for a noop
  context manager — measured, well under any meaningful threshold).
- **`docs/architecture/overview.md` "Observability" section** rewritten
  as vendor-neutral OTel-first. Points at `docs/observability.md`
  instead of inlining Logfire-specific guidance.
- **`README.md`** "Production stack" and "Embedded options" stamps no
  longer say "v0.9.0"; "Observability" section condensed to a
  pointer at `docs/observability.md` and a one-line `[otel]` recipe.
- **`docs/configuration.md`**: drops "experimental in v0.9.0" / "Added
  in v0.9.0" stamps; expands the Secrets section with the
  `SecretStr` / `.get_secret_value()` contract (#553 follow-up);
  documents the `uv exclude-newer = "7 days"` lockfile policy (#548).
- **`docs/telemetry-contract.json`** bumped from `1.0` → `1.1`
  (additive only). Adds `instrumentation_scope`, `resource_attributes`,
  `backends`, and the new public exports
  (`configure_telemetry`, `shutdown_telemetry_providers`,
  `TelemetryHandle`, `diagnostics`, `install_neo4j_log_bridge`).
- **`release.yml` smoke-install** now also installs `khora[otel]`
  and verifies the SDK + HTTP exporter import alongside the
  existing remember/recall round-trip.

### Removed

- **`src/khora/telemetry/logfire_integration.py`** — replaced by
  `_otel.py` (tracer/meter + `trace_span` + `install_neo4j_log_bridge`),
  `_attrs.py` (`bounded_text_hash`), and `bootstrap.py`
  (`configure_telemetry` and friends). All call sites updated.
- **The `Span` / `LogfireSpan` / `NoOpSpan` ABC hierarchy.** Callers
  now use `opentelemetry.trace.Span` directly. The migration is
  source-compatible — the `.set_attribute` / `.set_attributes` shape
  is identical.
- **"SurrealDB Phase 1" status note** in
  `docs/architecture/storage-backends.md` — the backend has been
  feature-complete since the 2026-03-25 audit.
- **Last user-facing `kuzu` mention** in
  `docs/engines/engine-comparison.md`. The `kuzu` extra still ships
  for downstream callers but is no longer advertised in docs.

### Deprecated

- **`khora.telemetry.install_neo4j_logfire_handler`** — renamed to
  `install_neo4j_log_bridge` (now picks the OTel logs SDK handler
  when logfire is absent). The old name is kept as a
  `DeprecationWarning`-emitting alias for one minor release; will be
  removed in khora 0.12.

### Internal

- Base install now depends on `opentelemetry-api>=1.27.0` (the wheel
  is small; the SDK and exporters remain optional). This is what makes
  `khora.telemetry.metric_counter` / `metric_histogram` /
  `metric_gauge_callback` callable from any code path without an
  availability check.

## [0.10.7] — PyPI README rewriting, SecretStr config fields, release-pipeline fixes

### Changed

- **Credential and DSN config fields re-typed as `pydantic.SecretStr`** (#553). Affects `KhoraConfig.storage.*` connection URLs/passwords (Postgres, Neo4j, Memgraph, Neptune, AGE, SurrealDB graph/relational/vector/event-store) and the telemetry collector DSN. Each backend's engine/driver factory unwraps the secret exactly once at the driver edge via the new `khora.config._secrets._secret_value()` helper. Config dumps and log lines now render these fields as `'**********'`; values written to fields still accept `str` for back-compat, but reads return `SecretStr` — callers that previously did `str(cfg.storage.neo4j.password)` and expected the cleartext now need `.get_secret_value()`. Pre-1.0 patch; flagging it explicitly here so downstream consumers can audit.
- **README rendering on PyPI** (#556). The on-disk README keeps relative links (`docs/configuration.md`) for GitHub readers; `hatch-fancy-pypi-readme` substitution rewrites them to `https://github.com/DeytaHQ/khora/blob/main/...` at wheel/sdist build time so PyPI's project page renders working links. Also adds `[project.urls]` for the PyPI sidebar (Documentation, Source, Issues, Changelog, Releases).

### Fixed

- **`release.yml` smoke-install** (#552). The smoke-install step installed `khora[sqlite-lance]==${VERSION}` and then asserted `pip show khora-accel`, which always failed because `[sqlite-lance]` doesn't pull khora-accel — that's the `[rust]` extra. Tripped the gate on every release since the assertion landed; v0.10.6 had its `github-release` job skipped because of it (workaround: manual `gh release create`). Now installs `khora[sqlite-lance,rust]` so both wheels are verified together.

### OSS prep

- **`CLAUDE.md` scrub** (#555) — final pass for OSS readiness.

## [0.10.6] — Test-infra expansion, dependency staging window, docs cleanup

### Added

- **Embedded test footprint** (PR-A / #536) — matrix tests for the SQLite+LanceDB stack covering VectorCypher, Skeleton, and Chronicle.
- **Embedded backend hardening** (PR-B / #537) — typed `EmbeddingError` validation in `khora.storage.backends.sqlite_lance.vector` so malformed inputs surface at the boundary instead of failing deep inside LanceDB.
- **Property-based tests** (PR-C / #538) — Hypothesis tests pinning Chronicle abstention `combined_score`, FTS5 escape parseability, MMR λ-direction (khora's λ=1 ⇒ pure relevance convention), and SQLite FTS5 `bm25()` sign handling.
- **Test infrastructure** (PR-D / #540) — `embedded` pytest marker, `scripts/check_coverage_floors.py` per-path coverage gate wired into CI, `make test-embedded` / `make test-soak` targets, and a top-level `CONTRIBUTING.md`.
- **Codecov split flags** (#539) — unit and integration uploads land separately so PRs that only touch unit-tested paths still get a Codecov diff comment. `codecov.yml` carries auto-target project status with sensible thresholds.
- **`uv exclude-newer = "7 days"` policy** (#548) — `uv lock` ignores PyPI releases uploaded within the last 7 days, re-evaluated every sync. `exclude-newer-package = { urllib3 = "0 days" }` lets same-week CVE fixes through; pattern reusable for future security-critical updates.

### Changed

- **Docs cleanup** — README, `docs/configuration.md`, `docs/README.md`, and `docs/architecture/{overview,storage-backends}.md` no longer reference the deprecated `kuzu` extra (#550). README sibling-package list trimmed; `khora-service` linked to its GitHub repo (#531). README description language: "library, not an application; tooling lives in sibling packages."
- **Docstring scrub** (#547) — `LiteLLMConfig.max_total_connections` description now uses vendor-neutral phrasing ("typical high-throughput ingestion concurrency"). Internal-service name removed from public API surface.
- **Release pipeline** (#532) — `release.yml` auto-creates GitHub releases on tag push with auto-generated notes from merged PRs.

### Fixed

- **`find_related_entities` fallback** (#535) — VectorCypher's graph-only fallback called a non-existent backend method on `sqlite_lance` / `surrealdb`. Falls back gracefully now.

### Removed

- **Memory Lake branding residue** (#534) — final pass on examples, tests, and inline strings. No public API changes.

### Infrastructure

- **kuzu still ships as an optional extra**, but is no longer advertised in user-facing docs (the upstream repository was archived after Apple's acquisition in October 2025).

## [0.10.5] — FTS5 escape, Chronicle sqlite_lance persistence, loguru placeholders

### Fixed

- **FTS5 syntax error on punctuated queries** (#526). `Khora.recall("What did Curie win?", …)` against `sqlite_lance` raised `sqlite3.OperationalError: fts5: syntax error near "?"`. New `escape_fts5_query` helper at `khora.storage.backends._fts5` is now wired into all three FTS5 sites (`engines/skeleton/backends/sqlite_lance.py`, `storage/backends/sqlite_lance/vector.py`, `storage/backends/sqlite.py`). Tokenizes on whitespace, wraps each token as a quoted FTS5 phrase, caps at 64 tokens. Recall semantics preserved.
- **Chronicle persistence on `sqlite_lance`** (#529). `Khora.remember()` previously dropped every event and fact (`0 events, 0 facts` in logs) because the coordinator dispatched chronicle methods exclusively to `self.vector`, and `sqlite_lance`'s vector adapter is LanceDB (no SQL session). `write_events`, `write_facts`, `query_events`, `query_active_facts_for_subject`, `supersede_fact` now live on `SQLiteLanceRelationalAdapter` using dedicated SQLite `Table` objects to bypass the ORM's Postgres-only `Vector(1536)` and `ARRAY(UUID)` column types. Coordinator falls back from vector → relational; pgvector path unchanged.
- **Loguru `%s` placeholder format** (#530). 25 logger format strings in `chronicle/engine.py` used stdlib-`logging`-style `%s/%r/%d` placeholders that loguru doesn't substitute (they printed literally). Converted to loguru's `{}` / `{!r}` / `{:.2f}` placeholders.

### Release tooling

- `verify-ci-green` no longer trips on the GitHub workflow-runs index lag; checks `.conclusion == "success"` via jq instead of `?status=success` URL filter (~30-60s stale window after CI completion was observed on the v0.10.2 tag push).
- `release.yml` now creates a GitHub release automatically on tag push with auto-generated notes.

### OSS prep

- All khora work tracked in **GitHub Issues**, not Linear (`CLAUDE.md` inlines a short GitHub workflow).
- Public-surface docs (`CLAUDE.md`, `README.md`, `docs/consumers.md`) no longer reference internal Deyta projects.
- PyPI long description: "Knowledge memory library for long-horizon AI agents — hybrid retrieval over documents, embeddings, and graph relationships."
- Sibling packages on PyPI: `khora-cli`, `khora-explorer`, `khora-service` (coming soon).

## [0.10.4] — First clean PyPI release after the migration

### Fixed

- **Lockstep version computation** (PR #527). The release pipeline sed'd `pyproject.toml` on the runner before `python -m build`, leaving the working tree dirty. `setuptools_scm` (via hatch-vcs) treated `dirty` at a tag as "ahead of tag" → bumped the patch → produced `0.10.4.dev0` instead of `0.10.3`. Removed the runtime sed; the lockstep `khora-accel == X.Y.Z` pin in `pyproject.toml`'s `rust` extra is now committed alongside the `khora-accel/Cargo.toml` version bump.

### Status

- First post-migration release where **both** `khora` and `khora-accel` wheels published cleanly to PyPI at the same version with the lockstep contract verified end-to-end (wheel METADATA contains `Requires-Dist: khora-accel==0.10.4`).
- Updated `CLAUDE.md` → Version Bumps to require updating all three files (`Cargo.toml`, `Cargo.lock`, `pyproject.toml` pin) in the same PR.

## [0.10.3] — Partial release (khora-accel only)

### Fixed

- **`actions/checkout@v6` did not fetch tag refs** (PR #525). `fetch-depth: 0` controls history depth, not tags; without `fetch-tags: true`, `git describe` saw no tag on the runner and `hatch-vcs` fell back to "next-dev". The v0.10.2 release published khora as `0.10.3.dev0` for this reason. Added `fetch-tags: true` to the khora checkout step; added `skip-existing: true` to both PyPI publish steps for safer re-runs.

### Status

- `khora-accel 0.10.3` published to PyPI; `khora 0.10.3` was **not** published (the lockstep-sed bug fixed in 0.10.4 produced `0.10.4.dev0` for khora). Use 0.10.4+ for the matched-pair install.

## [0.10.2] — Publishing migrated to PyPI

### Changed

- **Publishing target**: moved from AWS CodeArtifact to **public PyPI** under the Deyta organization (PR #524). Uses PyPI Trusted Publishing via GitHub OIDC — no API tokens, no AWS, no secrets in the repo. `pypa/gh-action-pypi-publish@release/v1` with an environment-bound trusted publisher per project.
- **khora-accel** now ships as an **sdist only** (no platform-wheel matrix). Users compile the Rust extension at install time via maturin's PEP 517 backend; requires a Rust toolchain (`rustup`) on the install host.
- **Version lockstep**: khora and khora-accel are always released at identical versions. The published khora wheel pins `khora-accel == X.Y.Z` exact.
- **Publish order**: serialized `publish-accel → publish-khora` so khora's wheel can only land on PyPI if accel is already resolvable.
- `ci.yml` dev-publish jobs removed; only tag pushes publish.

### Status

- `khora-accel 0.10.2` published to PyPI; `khora 0.10.2` was **not** published (the `fetch-tags` bug fixed in 0.10.3 produced `0.10.3.dev0` for khora). Use 0.10.4+ for the matched-pair install.

## [0.10.1] — Remove `graphrag` engine

### Removed — BREAKING

- **`graphrag` engine.** `engine="graphrag"` no longer accepted by `Khora(...)` or
  `create_engine(...)`; raises `ValueError: Unknown engine: graphrag` on 0.10.1+.
  The engine module (`src/khora/engines/graphrag/`) and its dedicated tests have
  been deleted.

### Migration

```diff
- async with Khora(db_url, engine="graphrag") as lake:
+ async with Khora(db_url, engine="vectorcypher") as lake:
```

For graphrag-equivalent **100% chunk extraction** (vs vectorcypher's default
selective 70%):

```python
async with Khora(
    db_url,
    engine="vectorcypher",
    engine_kwargs={"skeleton_core_ratio": 1.0},
) as lake:
    ...
```

### Data portability

Graphrag and vectorcypher wrote to the same tables (`documents`, `chunks`,
`entities`, `relationships`) on the same Postgres/Neo4j stack. **Existing
graphrag-ingested data remains queryable via vectorcypher** with no migration.
New ingest under vectorcypher uses KET-RAG selectivity unless overridden.

### Why no deprecation cycle?

The engine had no external dependencies and only one internal consumer
(`genesis`, which has migrated in lockstep). The breaking-change cost of a
deprecation shim was higher than the breaking-change cost of a hard removal.

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
  Downstream consumers migrate via coordinated PRs after this release.
  Both pre-pinned `khora<0.10` to avoid Renovate auto-bumps.
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


## [Unreleased] — Recall API time bounds honored

### Fixed — API-supplied `temporal_filter` no longer dropped

Callers passing an explicit `temporal_filter` to `MemoryLake.recall()` had their bounds silently bypassed in two places:

* **vectorcypher**: when `temporal_filter` was provided, the auto-detection branch (which also synthesizes the `TemporalSignal` consumed by the retriever's skip-fallback / version-filter / recency-weighting logic) was skipped entirely. The retriever then saw `temporal_signal=None` and applied none of the temporal-aware behavior the caller had asked for, including the sparse-results fallback that re-runs the search without the time bound.
* **graphrag**: same shape — the auto-detect block was guarded by `temporal_filter is None`, so the resulting `RecallResult.metadata` was missing `temporal_category` / `temporal_confidence` for API-bounded queries.

Both engines now synthesize an `EXPLICIT`-category `TemporalSignal` with `confidence=1.0` and `source="api"` when `temporal_filter` is supplied, so downstream behavior is consistent regardless of whether the bounds came from the caller or were detected from the query string. The new `source="api"` value joins the existing `"dictionary"` / `"semantic"` / `"none"` set on the `temporal_detect` span — telemetry contract unchanged (span attributes are not enumerated). graphrag's `apply_recency_bias` remains untouched on the API path: `EXPLICIT` does not match the `RECENCY`/`STATE_QUERY` guard, preserving existing behavior for API callers.

---

## [Unreleased] — Connector throughput restoration

### Performance — restore pre-0.9.0 LiteLLM throughput

The shared aiohttp session introduced in v0.9.0 was created with hard-coded `TCPConnector(limit=20, limit_per_host=10)`. `limit_per_host=10` silently throttled all OpenAI / Anthropic / etc. requests to 10 in flight per host, regardless of caller-configured concurrency. Downstream services (e.g. Genesis with `max_concurrent_llm_calls=200`) regressed ~5–20× on wall-time after upgrading to 0.9.x because the shared session became the dominant ceiling on parallel LLM/embedding calls.

The connector is now configurable through `LiteLLMConfig` and `LLMSettings`:

| Field                       | Default | aiohttp arg            |
|-----------------------------|---------|------------------------|
| `max_total_connections`     | 200     | `limit`                |
| `max_connections_per_host`  | 0 (unlimited) | `limit_per_host` |
| `keepalive_timeout_s`       | 30.0    | `keepalive_timeout`    |

Defaults restore pre-0.9.0 throughput: total cap is generous, no per-host throttle. Fields are read by `_init_shared_session` from a cache populated by `configure_litellm` (first-call-wins; subsequent calls with non-matching connector settings log a warning and are ignored).

**Migration call-out** — anyone who relied on the v0.9.0 connector throttle as a budget brake or rate-limit circuit-breaker should set `max_connections_per_host` explicitly in YAML / env (`KHORA_LLM_MAX_CONNECTIONS_PER_HOST`). On Anthropic Claude tier 1 in particular, an unlimited per-host connector combined with extraction loops can produce 429 storms that the previous 10-cap masked. Pick a value that matches your provider tier rather than relying on the connector for backpressure.

### Out of scope (related but tracked separately)

* `_bisect_and_extract` issues up to 2N LLM calls when truncation is detected — amplifies any concurrency change downstream. Not touched here.
* unified pending processor spawns 20 background workers on every `MemoryLake.connect()` even for engines that never call `submit_batch`. Idle but not free. Not touched here.

---

## [Unreleased] — Telemetry Public Surface, OSS Observability Contract

Telemetry workstream (PRs #504–#509) shipped after the v0.9.1 tag. It hardens cardinality safety, codifies the public observability surface as a JSON contract enforced by a CI drift gate, fixes a silent regression that had been zeroing out `storage_events.namespace_id` since February 2026, and broadens metric coverage. The OSS implication: public telemetry names are now API and break the same way any other public symbol does.

### Added

- **Public observability contract.** `docs/telemetry-contract.json` lists every public export in `khora.telemetry.__all__` (19 names), every `LLMEvent` / `StorageEvent` / `PipelineEvent` field, all 22 collector-recorded pipeline stages, all 58 `trace_span(...)` call sites (22 public, 36 internal), and all 21 metrics (16 public, 5 internal). `docs/telemetry-contract.md` is the human-facing explainer. `tests/unit/telemetry/test_contract.py` (10-test drift gate) walks the codebase via ripgrep and fails CI on any undeclared instrumentation. (#505)
- **`khora.telemetry.bounded_text_hash`.** Helper that turns free-text span attributes (raw query, document content, chunk text) into a SHA1[:8] hash — caps cardinality and removes the privacy hazard of raw text on spans. Now used at the four query / extraction sites that previously emitted raw text. (#504)
- **Chronicle abstention metrics.** `khora.chronicle.abstention_signal` (counter, public) and `khora.chronicle.abstention_combined_score` (histogram, public) aggregate the four boolean abstention signals + combined score that `RecallResult.metadata["abstention_signals"]` exposes per call, so abstention rate and confidence distribution can be tracked at fleet scale instead of only inspected per-request. (#507)
- **Aggregate operator metrics.** `khora.memory.recall.duration` (histogram, public, seconds), `khora.memory.ingest.duration` (histogram, public, seconds), `khora.llm.tokens` (counter, public), `khora.llm.cost_usd` (counter, public), `khora.log.queue.depth` (gauge, public, proxy via handler-error count — loguru 0.7.3 does not expose `qsize()`). (#509)
- **Six additional LLM call sites instrumented.** HyDE, listwise rerank, fact extraction, fact reconciliation, event extraction now record `LLMEvent` rows; chat was already wired. Two patterns coexist (`_telemetry_op="..."` through `khora.config.llm.acompletion` vs. inline `record_llm_call` after direct litellm calls); both are documented in `CLAUDE.md`. (#508)

### Fixed

- **`storage_events.namespace_id` 100% NULL since Feb 2026.** Restored namespace propagation through the storage telemetry path. The break had survived multiple releases because no operator dashboard was reading the column — surfaced during the Phase-0 audit. (#506)

### OSS implication

- Names tagged `public` in `docs/telemetry-contract.json` are now part of khora's public API. Renames or removals require a major version bump and prior coordination with genesis, khora-benchmarks, khora-explorer, and khora-cli. Names tagged `internal` (e.g. inner-loop spans like `khora.vectorcypher.coherence_boost`) may be renamed freely as long as the JSON is updated in the same PR.
- New attributes follow OTel semantic conventions: `gen_ai.*` for LLM, `db.*` for storage, `code.*` for stack info.
- The contract enables the operator-dashboard work that follows; it does not by itself fix the under-utilisation. Telemetry has been collected to PostgreSQL since 0.4.0, and dashboards / alerts that consume those events remain TODO.

---

## [0.9.0] — 2026-05-02 — Embedded Backend Realignment, Production-Readiness Scoping

### Embedded backend overhaul

The v0.9.0 embedded path lands as a complete-but-experimental SQLite + LanceDB stack covering all four engines (VectorCypher, GraphRAG, Skeleton, Chronicle). Engine × embedded integration tests now exist for all four engines; the prior "unverified embedded code path" gap from the audit is closed.

**Production-readiness scoping (per stack, not per engine).** Stamping is now per `(engine × storage stack)`:

- **VectorCypher** — production-ready on **PostgreSQL + pgvector + Neo4j** only.
- **Chronicle** — production-ready on **PostgreSQL + pgvector** (no graph DB required).
- **GraphRAG** and **Skeleton** — available; same PG-based stacks.
- **SQLite + LanceDB** for any engine — **experimental**. Documented scale ceiling: ~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3.
- **SurrealDB** for any engine — **experimental**. Python SDK on alpha track (`>=2.0.0a1`); KNN unreliable in embedded mode (brute-force cosine + HNSW fallback).

See [docs/engines/engine-comparison.md](docs/engines/engine-comparison.md#production-readiness-by-stack-v090) for the full matrix.

### Embedded engine wiring

- (#482): VectorCypher wired to the `sqlite_lance` backend.
- (#481): Skeleton wired to the `sqlite_lance` backend with a temporal-store adapter.
- GraphRAG embedded path pushes the temporal filter into the LanceDB WHERE — was previously post-hoc and xfail-pinned.
- (#486): Temporal filter pushed into SQLite-side WHERE in the GraphRAG embedded chunk fetch path.
- VectorCypher honours `metadata['occurred_at']` on the embedded path (parity with `remember_batch`).

### Embedded retrieval correctness

- Chronicle channels (BM25 / semantic / temporal / entity) now share the same `created_after`/`created_before` bounds — fixes channel divergence that broke RRF fusion.
- Recursive-CTE graph traversal switched from node-visited to edge-visited tracking (mirrors Neo4j `MATCH [*1..N]`).
- `valid_until > now` filter inlined into both anchor and recursive arms of the CTE.
- Skeleton tag-cast and `occurred_at` parsing fixes (`Skeleton.remember()` parity with `remember_batch()`).
- Embedded compensating-delete-on-failure logging hardened.
- (#485): LanceDB IVF-PQ index now retrains once the corpus grows past `retrain_factor × (rows at last training)`. Configurable via `KHORA_STORAGE_SQLITE_LANCE__RETRAIN_FACTOR` (default `2.0`). Fixes silent recall degradation as the corpus grows past the initial training threshold (5k rows). Set ≤ `1.0` to disable.

### Embedded warts (documented, not fixed)

- **Partial atomicity in `coordinator.transaction()`** on embedded — only the SQL session is enrolled; LanceDB writes happen post-commit with compensating deletes.
- **Point-in-time queries** are not supported on the embedded stack. The CTE port does not implement PIT semantics. Tracked.
- **FTS5 on chunks only** — entity-anchored recall falls back to `LIKE` / JSON-equality on embedded. Use the PostgreSQL stack for entity-heavy corpora.

### Deprecated

- **Kuzu graph backend** (`khora[kuzu]`) — deprecated in 0.9.0, scheduled for removal in 0.10. Kuzu was acquired by Apple in October 2025 and the upstream repository is archived. Migrate to SQLite + LanceDB (embedded) or PostgreSQL + Neo4j (production).

### v0.10 roadmap

Two deferred decisions for v0.10 address the embedded warts:

- **sqlite-vec** as a candidate to collapse the SQLite + LanceDB dual-store into a single in-SQLite-transaction vector store (eliminates partial atomicity, drops install footprint from ~150 MB to ~5 MB).
- **`pgserver` (embedded Postgres)** as a candidate for true production-parity embedded mode (HNSW recall, real ACID, zero schema fork).
- **Default embedded URI routing** — currently `MemoryLake("memory://")` treats the URL as the PostgreSQL `database_url`; SurrealDB owns the `memory://` scheme internally. Routing a top-level `memory://` URI to the recommended embedded stack is a v0.10 code change.
- **`lance-graph` integration** is explicitly **deferred to v0.10** — no second 0.x Rust crate enters a "production-ready" path in v0.9.0.

---

## [Unreleased] — Graph Backends, Temporal Precision, Discovery Agent Overhaul

### Added
- Codified the khora public API surface consumed by downstream packages (genesis, khora-benchmarks, khora-explorer, khora-cli).

### Removed
- `khora` console script and CLI subcommands (`extract`, `search`) — moved to khora-cli. Install with `uv pip install khora-cli` and run `uv run khora-cli extract` / `uv run khora-cli search`.
- `khora ontology` CLI subcommands (moved to khora-explorer)
- `khora.discovery` package (moved to khora-explorer)
- `khora.cli` package (entire subtree — `extract`, `search`, `_common`)
- `click` and `rich` dropped from core dependencies (only the CLI used them)

### Changed
- **Breaking**: khora is now a pure memory-lake library. `uv run khora ...` is no longer a valid command; use `uv pip install khora-cli` and `uv run khora-cli extract` / `search` instead.
- `khora.discovery.extraction` → `khora.extraction.binary_readers` (binary file reader consumed by khora-cli)
- Documentation rework post-extraction: short, library-focused `README.md`; new `docs/README.md` index, `docs/configuration.md`, `docs/api-reference.md`, `docs/migrations.md`, and `docs/consumers.md`; removed stale `khora extract` / `khora search` / `khora ontology` references from the top-level docs in favour of pointers to `khora-cli` and `khora-explorer`.

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

- Fix `run_migrations()` on fresh PostgreSQL database — use `information_schema.tables` (#201)
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

- `ExpertiseConfig` as stable public API with YAML loading,
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
  causing `AttributeError` on entity-heavy queries (#26)

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
