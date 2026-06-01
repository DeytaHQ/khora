# Changelog

All notable changes to Khora are documented here.

Format: versions match git tags (`git tag vX.Y.Z`). Versions before 0.5.1 were internal (no git tags).

## [0.18.1] - silent-config and observability fixes

Patch release. Fixes a batch of "accepted but silently ignored / silently dropped" config and observability gaps from Damir Krstanovic's reports. No breaking changes.

### Fixed

- **Chronicle surfaces dropped per-namespace event/fact overrides when `get_namespace` fails** (#893): a transient error fetching the namespace previously fell back to defaults silently (re-enabling events/facts a namespace had disabled). It now logs a WARNING and records a `Degradation` on the result.
- **`pending_processor_max_concurrent` rejects non-positive values** (#933): `0` or a negative value previously started a processor that drained nothing; it now raises a `ValidationError` (mirrors the `ge=1` convention used by other concurrency settings).
- **Skeleton `remember_batch(max_concurrent=...)` is honored** (#935): the kwarg was declared but ignored; Skeleton now bounds its batch fan-out with a semaphore, matching VectorCypher.
- **`linked_entity_boost` config is applied** (#918): the knob was loaded but never read; query scoring now uses the configured value instead of a hardcoded `1.5`.
- **Telemetry buffer flushes when `flush_threshold` is reached** (#934): the threshold was stored but never triggered a flush, so the buffer only drained on the timer; it now schedules a flush once the buffer hits the threshold.

## [0.18.0] - Chronicle reinforcement-on-recall + silent-failure observability + breaking Skeleton/config changes

Minor release. Adds Chronicle reinforcement-on-recall and a project-wide failure-observability convention (ADR-001), with a large batch of silent-failure and partial-failure fixes. Contains breaking changes: Skeleton now rejects unsupported kwargs instead of silently ignoring them, and the dead `auth_enabled` config field is removed.

### Breaking changes

- **Skeleton `remember()` now raises `UnsupportedEngineKwargError` on non-empty `entity_types` / `relationship_types`** (#890, in #917). Previously a silent no-op - Skeleton does not extract entities, so the kwargs gave a false impression of graph extraction.
- **Skeleton `recall(recency_bias=...)` now raises on a non-`None` value** (#891, in #917). Previously a silent no-op - Skeleton has no recency scoring.
- **`auth_enabled` config field removed from `KhoraConfig`** (#908, in #941). The field was dead; with `extra='forbid'`, passing it now raises a Pydantic `ValidationError`.
- **Postgres backend rejects `embedding_dimension != 1536`** (#925, guard). A config-time `ValidationError` now fires instead of silently writing wrong-width vectors into the hardcoded `Vector(1536)` columns. Arbitrary dimensions on Postgres require a parameterized migration (tracked in #925 for a future release); `sqlite_lance` already supports any dimension.

### Added

- **Chronicle reinforcement-on-recall** (#855): recalling a chunk stamps `chunk.last_accessed_at`, so frequently-accessed memories resist temporal decay. Schema + storage round-trip plus telemetry and task tracking.
- **Failure-observability convention (ADR-001)** (#912, #916): `Degradation` / `ErrorRecord` / `SkipReason` TypedDicts surfaced on result objects, plus `khora.*.degraded_total{channel, reason}` metrics. Reference impl is the chronicle channel degradations.
- **Claim-based orphan recovery** (#885, #886): recovery now covers `PROCESSING` rows and uses `FOR UPDATE SKIP LOCKED` so concurrent workers do not double-claim.
- **sqlite_lance parity** (#896, #905): `dream_history` persistence and `get_chunks_by_document` now behave the same on sqlite_lance as on Postgres.
- **`source_timestamp` propagation through VectorCypher** ingest and recall paths (#859).
- **`unify_entities` threshold kwargs + per-strategy candidate counters** (#865).
- **Dream skip-reasons** exposed on `DreamResult.metadata["skip_reasons"]` for empty results (#876).
- **Vector-first dual-write ordering + partial-failure counters** for `upsert_entities_batch` and `replace_document` (#868, #884, #915).
- **Stats counter-failure surfacing** via ADR-001 degradations (#878).

### Changed

- **Chronicle temporal decay is now applied AFTER cross-encoder rerank** so `chronicle_decay_weight` is actually honored (#866). Previously rerank ran last and overwrote the decay-adjusted ordering.
- **`count_*` routing by data ownership** (#878): count calls route to the backend that owns the data rather than assuming a fixed backend.
- **`from_yaml` now honors `KHORA_DATABASE_URL` / `KHORA_NEO4J_URL` env vars** with correct precedence (#897).
- **Per-document batch progress callbacks** on Skeleton and Chronicle batch paths (#898).
- **Embedding dimension is honored end-to-end and validated** (#926, #931): the ingest / batch paths now pass the configured `embedding_dimension` (they previously defaulted to 1536), and `LiteLLMEmbedder` raises `EmbeddingError` on a provider dimension mismatch instead of silently overwriting its configured dimension.

### Fixed

- **`BatchHandle.wait()` no longer hangs on a pre-try fault**; worker-level fallback handles faults raised before the per-item try block (#869).
- **`replace_document_extraction` stamps status inside the SQL transaction** and no longer marks a document FAILED when Postgres committed but the graph-mirror phase raised (#887).
- **VectorCypher `remember()` no longer reports a document as completed on total LLM extraction failure** (#889); extraction errors now surface on `RememberResult.metadata`.
- **Chronicle fact-reconcile no longer fails open to ADD on an LLM error** (#892); an LLM failure no longer silently appends a duplicate fact.
- **Chronicle extraction, VectorCypher relationship fetch, and ingest relationship-drop silent failures** now surface via ADR-001 degradations (#903, #904, #907).
- **Rust accel raises `ValueError` instead of `PanicException`** on mismatched array lengths in MMR and entity resolution (#902).
- **Misleading Neo4j / event-store WARNINGs suppressed** when an engine has opted out of the graph / event store (#877).
- **Empty-name entities are skipped** rather than persisted (#894).
- **`update_document` partial update preserves unset columns** instead of NULL-overwriting them (#895).
- **Dialect-gated dream apply on sqlite_lance** plus a warning on Postgres + Neo4j divergence (#875).
- **`gc.expire_sessions(before=...)` no longer crashes on a naive datetime** (#930): a naive `before` (e.g. `datetime.utcnow()`) is normalized to UTC instead of raising `TypeError` and aborting the sweep.
- **`forget()` now cleans entities and relationships on all graph backends, not just Neo4j** (#923): the cascade is re-anchored on `source_document_ids` refcounting (orphans deleted, shared entities survive with the document id stripped), so SurrealDB / Memgraph / Neptune / AGE / sqlite_lance no longer silently retain orphaned entities while reporting success. Also fixes SurrealDB silently dropping `source_document_ids` (SCHEMAFULL array element field was undeclared).
- **`delete_entity` removes the pgvector mirror, not just the graph node** (#928), so a deleted entity no longer keeps surfacing in vector entity search on Postgres + Neo4j.
- **`remember_batch` session episodes are paired by document id** (#929), fixing episode misattribution and a dropped final document when one document in the batch fails.

## [0.17.4] - VectorCypher entity projection on graph-less backends

Patch release on top of v0.17.3. One bug fix from Damir Krstanovic.

### Fixed

- **VectorCypher `recall()` now populates `result.entities` and `result.relationships` on sqlite_lance and other graph-less backends** (#857). Previously `_simple_retrieve` (the fallback used when Neo4j is unavailable or the storage backend has no graph driver) returned `entities=[]` hardcoded - so `kb.recall()` on `vectorcypher + sqlite_lance` returned chunks correctly but always empty entities and relationships, even though the graph was correctly populated at ingest time (`kb.list_entities()` and `kb.find_related_entities()` worked fine on the same backend). Fix: in `_simple_retrieve`, after building `chunk_results`, fetch entities via `storage.list_entities(namespace_id, limit=1000)` and filter Python-side to those whose `source_chunk_ids` overlap the recalled chunk set. Relationships are filtered to those where both endpoints are in the recalled entity set. Scoring is `overlap_count / len(source_chunk_ids)` for entities, `1.0` constant for relationships (no traversal-derived weight available without a graph driver). Storage failures degrade to empty lists with a warning, matching the surrounding defensive style.

### Behavior change

- On production `postgres + neo4j` deployments, the `_vector_only_fallback` path (used when Neo4j is unavailable) previously returned `entities=[]`. After this patch, it returns the entities-via-relational-storage projection. The existing `engine_info["graph_unavailable"]=True` flag still surfaces the degradation. This is the intended product behavior - degraded > empty.

### Migration

- **No migration required.** The fix only affects callers using `engine="vectorcypher"` with a graph-less storage backend (sqlite_lance), or a postgres deployment when Neo4j is temporarily unavailable. Production `postgres + neo4j` callers in steady state are unaffected.

## [0.17.3] - Chronicle bi-temporal + Skeleton source_timestamp + decay-default unification

Patch release on top of v0.17.2. Six bug reports from Damir Krstanovic - one of them critical to Chronicle's bi-temporal promise. The Chronicle temporal-decay code path was reading ingest time instead of user-supplied event time, the half-life default disagreed in three places (24h field default vs 168h function defaults vs 168h docs), the canonical formula was written one way in the docstring and another in the implementation, and Skeleton silently dropped `source_timestamp` on the way to `chunk.occurred_at`. Plus a tuning bump on the chronicle decay weight default.

### Changed

- **Chronicle temporal decay now reads `chunk.source_timestamp or chunk.created_at`** instead of `chunk.created_at` (#848). The user-supplied event time was being persisted on `Chunk.source_timestamp` since v0.13 but never reached the scoring path - decay differentiated chunks only by ingestion time. Backfilling a year of historical events with their true `source_timestamp` collapsed to "all events equally fresh" from decay's perspective.
- **`QuerySettings.temporal_half_life_hours` default raised from 24.0 to 168.0** (#851). Three places disagreed: the field default was 24h, the function-level defaults in `_ebbinghaus_decay` / `_apply_temporal_decay` were 168h, and the docs claimed 168h. Real users got 24h - aggressive enough that a memory was half-faded after one day. Unified on 168h (7 days) as the single source of truth via new module constants `DEFAULT_CHRONICLE_HALF_LIFE_HOURS` and `DEFAULT_CHRONICLE_DECAY_WEIGHT` in `khora.engines.chronicle.engine`.
- **`QuerySettings.chronicle_decay_weight` default raised from 0.10 to 0.30** (#853). Under the multiplicative decay formula, the prior 0.10 default capped the max age penalty at 10% - relevance dominated, ancient facts routinely outranked recent ones. The new 0.30 default keeps fresh memories at 100% of relevance and fully-faded memories at 70%, giving recency enough weight to matter in personal-assistant and incident-memory use cases.
- **Canonical Chronicle decay formula documented as multiplicative** (#852): `score = relevance * ((1 - w) + w * retention)`, matching what the implementation always ran (and what Elasticsearch / Mem0 use). The function docstring at `_apply_temporal_decay:140` previously claimed an additive form (`(1 - w) * relevance + w * retention`) that the code did not match; both the docstring and `docs/engines/chronicle-engine.md` now show the multiplicative form with a worked example.
- **VectorCypher `_soft_temporal_score` decoupled from its hardcoded 24h half-life**. Both call sites in `src/khora/query/engine.py` (around lines 1225 and 2758) now pass `cfg.temporal_half_life_hours` and `cfg.temporal_hard_cutoff_days` through explicitly. Previously VectorCypher's soft temporal scoring ran at 24h regardless of the configured value, so the chronicle batch's 168h bump above also flips VectorCypher behavior - this is the intentional cross-engine alignment.

### Fixed

- **Skeleton `Khora.remember(source_timestamp=t)` now propagates the timestamp to `chunk.occurred_at`** (#856). Prior resolution order in `engines/skeleton/engine.py:remember()` was `occurred_at kwarg > metadata["occurred_at"] > now()`, dropping the user-supplied `source_timestamp` entirely. New order: `occurred_at kwarg > metadata["occurred_at"] > source_timestamp > now()`. Same fix applied to the batch path, with the batch-level `source_timestamp` kwarg used as the per-doc fallback when `doc_data["source_timestamp"]` isn't set.
- **Chronicle docs reference the real tuning knobs** (#850). `docs/engines/chronicle-engine.md` previously claimed decay was "Configurable via `recency_weight` and `recency_decay_days`" - those exist on `QuerySettings` but are VectorCypher's knobs and have no effect on Chronicle. Replaced with `chronicle_decay_weight` and `temporal_half_life_hours` (and their env-var equivalents `KHORA_QUERY_CHRONICLE_DECAY_WEIGHT` / `KHORA_QUERY_TEMPORAL_HALF_LIFE_HOURS`).
- **`_accel.batch_recency_scores` clamps future timestamps to age=0** in the pure-Python fallback. The chronicle-engine variant of decay already clamped, but the accel shim's fallback did not - a forward-dated `source_timestamp` produced negative `age_days`, `math.exp(positive)` returned > 1.0, and the resulting score could exceed the relevance ceiling. Now matches the chronicle clamp.

### Migration

- **Anyone running with the implicit `temporal_half_life_hours=24.0` default** will see decay run 7x slower under v0.17.3. Set `KHORA_QUERY_TEMPORAL_HALF_LIFE_HOURS=24` (or the YAML equivalent) to restore the prior behavior.
- **Anyone relying on `chronicle_decay_weight=0.10`** will see recency weighted 3x more under v0.17.3. Set `KHORA_QUERY_CHRONICLE_DECAY_WEIGHT=0.10` to restore.
- **Code that hardcoded `chunk.created_at` as "when the event happened"** should switch to `chunk.source_timestamp or chunk.created_at` for chronicle-style use cases.
- **Skeleton callers passing both `source_timestamp` and `occurred_at`**: explicit `occurred_at` still wins. Callers passing only `source_timestamp` now get bi-temporal behavior that they probably expected all along.

### Out of scope (filed separately)

- **#855** (Chronicle reinforcement-on-recall) is a feature request, not a bug fix. Decay-only behavior continues to be the contract in v0.17.3; reinforcement is queued for a future minor.

## [0.17.2] - Extraction + batch hardening (Damir follow-ups)

Patch release on top of v0.17.1. Two more bug reports from Damir Krstanovic surfaced after the 0.17.1 batch: a `submit_batch` kwarg that did nothing, and a silent entity-type drop when the configured LLM was off the strict-JSON-schema allowlist.

### Added

- **`gpt-5`, `gpt-5-mini`, `gpt-5.4`, `gpt-5.4-mini` join `MODELS_REQUIRING_JSON_SCHEMA`** in `src/khora/extraction/extractors/llm.py`. These OpenAI models support strict `json_schema` response format - they belong on the same fast path as `gpt-4o*` / `gpt-4.1*` / `o1*`, not in the lenient `json_object` fallback.
- **One-shot warning when the configured extraction model is not on the allowlist**, telling operators that fallback parsing is in play and pointing at `MODELS_REQUIRING_JSON_SCHEMA`. Deduped per-process via class-level `_WARNED_NON_ALLOWLIST_MODELS` so CI logs don't drown.

### Changed

- **LLM response parser now accepts both long and short JSON keys** (#839). Pre-0.17.2, the parser at `src/khora/extraction/extractors/llm.py` read only `entity_type` / `relationship_type` / `event_type` / `source_entity` / `target_entity`. Off-allowlist models (e.g. local llama.cpp, Anthropic in some configurations) fell back to `{"type": "json_object"}` with no schema enforcement and emitted the short forms (`type`, `source`, `target`). The parser's `.get(...)` calls returned `None` and the dataclass defaults (`"CONCEPT"`, `"RELATES_TO"`, `"EVENT"`, `""`) kicked in - so users saw every entity stored as `CONCEPT` and every relationship as `ASSOCIATED_WITH` despite the LLM having returned the correct types. Six parse sites in `llm.py` (lines around 981, 1847, 1894, 2013, 2045, 2068) now read `dict.get("long") or dict.get("short") or "<default>"`.

### Fixed

- **`submit_batch(max_concurrent=N)` is now honored as a per-batch concurrency cap** (#838). Previously declared in the signature but never branched on - this release wires it to a per-batch `asyncio.Semaphore` that caps in-flight document processing for that batch. Defaults to 20. Bounded above by the global `pending_processor_max_concurrent` pool size, so effective per-batch concurrency is `min(pool_size, max_concurrent)`. Concurrent batches each carry their own semaphore - they do not share state and their `max_concurrent` values do not stack.

### Migration

- **No migration needed for #839**: the parser now accepts both key shapes, so existing models continue to work and previously-broken off-allowlist models (most notably gpt-5.x) now produce correctly-typed entities and relationships.

## [0.17.1] - Recall-API contract repair (Damir feedback batch)

Patch release on top of v0.17.0. Six bug reports from an external user (Damir Krstanovic) exposed `Khora.recall(...)` as a leaky abstraction: parameters documented but silently ignored, parameters that meant different things on different engines, behavior that contradicted docs. This release bundles a coherent contract repair across all three engines.

### Added

- **`EngineCapabilityError`** (`src/khora/exceptions.py`) - new exception raised when a caller passes a `SearchMode` an engine doesn't support. Exported from `khora` and `khora.exceptions`. Each engine now declares `supported_modes: ClassVar[frozenset[SearchMode]]` on the class.
- **`khora.core.recall_scoring.min_max_normalize`** - shared helper used by Chronicle and Skeleton to normalize recall scores. Handles tied/single/empty inputs; returns `[1.0] * n` when all scores tie.

### Changed

- **`RecallChunk.score` is now min-max normalized rank in [0, 1] on every engine** (#834). Top chunk = 1.0, bottom = 0.0 (with 2+ chunks); single chunk = 1.0; tied = 1.0. Previously the field meant three different things: VectorCypher already returned this shape; Chronicle returned a post-rerank fused score (arbitrary scale); Skeleton returned raw cosine or BM25 (engine-internal). Callers writing `if chunk.score > 0.3` now get consistent semantics across engines.
- **VectorCypher: `min_similarity` is now honored** (#830). The parameter was declared on `Khora.recall(min_similarity=T)` and forwarded into the VectorCypher engine but dropped at the retriever boundary, making it a no-op on the flagship engine while Skeleton and Chronicle honored it. The threshold is now applied as a raw-cosine floor on the vector channel before RRF fusion, matching Skeleton/Chronicle semantics (`engine.py` -> `retriever.retrieve()` -> `_vector_search_chunks()` -> `vector_store.search(min_similarity=T)`).
- **VectorCypher: real channel-skip support for `mode=VECTOR / GRAPH / KEYWORD`** (#833). Previously VC collapsed every mode except `ALL` to `HYBRID`. The retriever now honors each mode by skipping the unused channels: VECTOR skips graph + BM25; GRAPH skips chunk-level vector + BM25 (entity-vector seeding still runs as graph entry point, by design); KEYWORD bypasses RRF fusion and surfaces BM25 chunks directly. `ALL` retains its existing balanced-RRF behavior (hybrid_alpha=0.5); `HYBRID` is vector-weighted hybrid (hybrid_alpha=0.7). Documented at the top of `engine.recall()`.
- **Chronicle and Skeleton raise `EngineCapabilityError` on unsupported modes** (#833). Chronicle supports VECTOR, HYBRID, ALL (KEYWORD and GRAPH were silently returning empty results). Skeleton supports VECTOR, HYBRID, KEYWORD (GRAPH and ALL were silently mapping to HYBRID). Mode validation runs at the top of each engine's `recall()` before any I/O.

### Removed

- **`raw: bool` kwarg on `Khora.recall()`** (#831). Documented to "skip query understanding, entity linking, reranking, and HyDE" but never branched on in any engine body - HyDE / query understanding live in a separate `QueryEngine` not invoked from `engine.recall()` at all. For benchmark / LLM-free recall, use the config flags `enable_llm_reranking`, `enable_hyde`, and `temporal_llm_disambiguation_enabled` on `KhoraConfig.query`. Parameter removed from `Khora.recall()`, `MemoryEngineProtocol.recall()`, and all three engine signatures.
- **`agentic: bool` kwarg on `Khora.recall()`** (#832). Documented to enable multi-step exploration with follow-up queries, but never branched on - implementation was removed when graphrag engine was deprecated, though the docstring was never updated. `AgenticSearchAgent` still lives at `khora.query.agentic` and can be invoked directly via `HybridQueryEngine.query(agentic=True)`. Re-introducing as a recall-level feature is queued for a future minor.
- **`src/khora/__main__.py`** (#835). The file imported `khora.cli` which was removed when the CLI moved to a separate `khora-cli` package. `python -m khora` now produces the standard `No module named khora.__main__` Python message.

### Migration

- **Callers passing `raw=True` or `agentic=True`** to `Khora.recall(...)` will get a `TypeError: unexpected keyword argument`. The parameters never did what their docstrings claimed; deletion makes the documented contract honest. Switch LLM-free benchmark paths to the `KhoraConfig.query` flags listed above.
- **Callers comparing `RecallChunk.score` against absolute thresholds** should re-tune. Pre-0.17.1 thresholds calibrated on Chronicle (post-rerank scores ~0.7) or Skeleton (raw cosine ~0.01) will not transfer to the new normalized [0, 1] rank scale.
- **Callers passing `mode=KEYWORD` or `mode=GRAPH` to Chronicle, or `mode=GRAPH` or `mode=ALL` to Skeleton** will now hit `EngineCapabilityError` instead of getting silently-empty results. The error message lists `supported_modes` for the engine.

## [0.17.0] - Turbopuffer Skeleton backend + relationship-FK remap + API time-bound fix

Minor release on top of v0.16.4. Headline is the new opt-in Turbopuffer backend for the Skeleton engine (serverless scale tier) alongside a vectorcypher correctness fix where entity ID canonicalization during upsert left relationships pointing at throwaway IDs - a sqlite_lance FK crash and a silent Neo4j edge drop. Plus a recall-API fix where explicit `temporal_filter` was bypassed on vectorcypher and graphrag, and two smaller ingest / vectorcypher hardening items.

### Added

- **Skeleton engine: opt-in Turbopuffer backend** (#827, closes #824). New `src/khora/engines/skeleton/backends/turbopuffer.py` (~440 LOC) ships `TurbopufferBackendConfig` and `TurbopufferTemporalStore`, mapping one Turbopuffer namespace per khora `namespace_id` (`khora_<hex>`). Hybrid retrieval is client-side RRF over a multi-query batch, so `hybrid_alpha` is a documented no-op; ALL-tags filters fold into N `Contains` clauses. 42 new unit tests run against an injected fake SDK. New `[turbopuffer]` pyproject extra pins `turbopuffer>=2.1.0,<3.0`. Out of scope for this release: a real-cluster CI integration job (needs a sandbox API key) and Chronicle / VectorCypher bindings.

### Changed

- **VectorCypher engine: `engine_info.mode` is now the lowercase mode string** (#822). `RecallResult.engine_info["mode"]` is emitted as `mode.name.lower()` (one of `vector` | `graph` | `hybrid` | `all` | `keyword`) instead of the `SearchMode` enum integer, matching the documented recall `mode` vocabulary. Builds on the canonical `engine_info` keys from #805 (v0.16.4). `engine_info` is "free-form engine telemetry" per `docs/api-reference.md` (only the `"engine"` key is contractual), so this is not a breaking change to a documented public surface - flagged here for any consumer who happened to parse the field as an int.

### Fixed

- **vectorcypher: remap entity IDs after upsert so relationships don't crash on shared entities** (#825, closes #806). The engine built `Relationship.source_entity_id` / `target_entity_id` from extraction-time UUIDs, but `upsert_entities_batch` then canonicalised `entity.id` to the persisted row's UUID on match - leaving relationships pointing at throwaway IDs. On sqlite_lance the FK fired and crashed the write; on Neo4j the `MATCH`-by-id silently dropped the edge. Three-place fix: (1) `sqlite_lance/graph.py:upsert_entities_batch` now mutates input `entity.id` on match (mirrors the Neo4j path), (2) `vectorcypher/engine._run_skeleton_extraction` snapshots pre-upsert IDs and remaps relationships after the batch, and (3) the streaming batch path composes dedup + upsert remaps. Regression test in `tests/integration/matrix/test_vectorcypher_sqlite_lance.py`.
- **recall: API-supplied `temporal_filter` no longer silently dropped on vectorcypher + graphrag.** Both engines guarded the auto-detect block with `temporal_filter is None`, so an explicit caller-supplied bound bypassed `TemporalSignal` synthesis - the retriever then saw `temporal_signal=None` and skipped version-aware scoring, recency weighting, and the sparse-results fallback. Both engines now synthesize an `EXPLICIT`-category `TemporalSignal` with `confidence=1.0` and `source="api"` when `temporal_filter` is supplied; graphrag's `apply_recency_bias` is untouched on the API path because `EXPLICIT` does not match the `RECENCY`/`STATE_QUERY` guard. The new `source="api"` value joins the existing `"dictionary" / "semantic" / "none"` set on the `temporal_detect` span.
- **ingest: coerce string `source_timestamp` to `datetime` instead of letting it reach `Document(...)`** (PR #823). The public `source_timestamp` kwarg on `remember` / `remember_batch` / `submit_batch` is typed `datetime`, but upstream connectors and adapters routinely pass ISO-8601 strings (trailing `Z`, explicit offset, or date-only `YYYY-MM-DD`). A new shared `coerce_source_timestamp(value)` helper returns an existing `datetime` unchanged, parses those string forms, and returns `None` for empty / unparseable input without raising; `_extract_source_timestamp` now delegates to it. Applied at every `Document(...)` construction site that takes an explicit or per-doc `source_timestamp` (three ingest staging paths plus the `submit_batch` insert and re-queue paths), so a stray string can no longer be persisted as-is or crash ingestion.
- **document ingestion: stop logging the raw DB exception repr on persist failure** (PR #823 sibling). Several failure paths interpolated `{exc}` directly into a log line or surfaced `str(exc)` as `DocumentResult.error`; on a SQLAlchemy `DBAPIError` that string embeds the failed statement *and* its bind-parameter tuple - i.e. the full document content and metadata - leaking it into logs and to callers. The `submit_batch` create / update-record warnings *and* the pending / staged-document processing failure surfaces (process error log, status-update-failure warning, and the `DocumentResult.error` returned to callers) now use a bounded `_safe_exc_summary` that prefers the underlying driver message (via `.orig` / `__cause__`), strips the `[SQL: ...]` / `[parameters: ...]` tail, truncates, and prefixes the exception class name, while keeping the `external_id` / `doc_id` context. The persisted `error_message` column is unchanged.

## [0.16.4] - Abstention-signal correctness + BM25 routing + Weaviate hardening

Patch release on top of v0.16.3. Three recall-quality fixes — Chronicle stops wiping `entity_hits` before packaging recall (so `RecallResult.entities` and the `entities_empty` abstention signal stop reporting steady-state false alarms), the `top_score_low` abstention signal now reads the raw vector cosine instead of the post-rerank display score (it had become a steady-state false negative on rerank-enabled queries), and the BM25 channel is routed to the temporal-store chunk table so `enable_bm25_channel` is no longer a silent no-op on `remember_batch`-ingested namespaces (adds Postgres-only migration 039). Plus canonical `engine_info` keys on the vectorcypher path, Weaviate Skeleton-backend hardening (async client + auth + Weaviate Cloud), and faster CI.

### Added

- **vectorcypher: emit canonical `engine_info` keys** (#805). `RecallResult.engine_info` from `VectorCypherEngine.recall()` now includes five engine-agnostic canonical keys alongside the existing engine-specific telemetry: `mode` (from the `SearchMode` argument), `channels_used` (subset of `{"vector", "graph", "bm25"}` derived from per-channel chunk counts), `rrf_k` (from `vc_config.fusion_rrf_k`), `temporal_signal` (`{category, source}` dict, defaults to `{"category": "none", "source": "none"}` when no signal detected), and `abstention_signals` (4 boolean flags + `combined_score` + `should_abstain`, same shape as chronicle). Two new metrics: `khora.vectorcypher.abstention_signal` (counter) and `khora.vectorcypher.abstention_combined_score` (histogram). Chronicle's `_compute_abstention_signals` refactored to delegate to the shared `khora.core.recall_abstention.compute_abstention_signals` helper — chronicle's output and metric emission are unchanged.
- **Weaviate Skeleton backend hardened: async client + auth + Weaviate Cloud** (#803, refs #783). The experimental Skeleton-engine Weaviate backend switches from the sync `weaviate.connect_to_local` to the v4 `WeaviateAsyncClient` family (`use_async_with_local` / `use_async_with_custom` / `use_async_with_weaviate_cloud`), so the engine event loop no longer blocks on Weaviate I/O. New `WeaviateBackendConfig` frozen dataclass carries `url` / `cluster_url` / `api_key` (`str` or `SecretStr`) / `grpc_port` / TLS + header knobs, validating that exactly one of `url` / `cluster_url` is set and that Weaviate Cloud is never addressed anonymously; the legacy `WeaviateTemporalStore(config, "http://...")` URL contract is preserved. Read-path hardened for the v4 client's parsed-`datetime` and polymorphic vector shapes. New `weaviate-integration` CI job exercises a real cluster behind `WEAVIATE_INTEGRATION_TEST=1`. Chronicle / VectorCypher Weaviate support remain out of scope.

### Fixed

- **chronicle: stop wiping `entity_hits` before recall packaging** (#812, closes #808). `ChronicleEngine.recall()` populated `entity_hits` via `_collect_entities()` and then immediately overwrote it with `[]` one block later, so `RecallResult.entities` was always empty and `abstention_signals["entities_empty"]` was permanently `True` — a constant +0.3 contribution to the abstention `combined_score` on every query regardless of the actual retrieval outcome. Dropped the shadow reassignment so the collected entities flow through to the abstention computation, the log line, and the final `RecallResult.entities` field.
- **chronicle + vectorcypher: `top_score_low` reads the raw vector cosine, not the post-rerank score** (#815, closes #809). The abstention helper read `chunks[0][1]`, which is the score *after* weighted RRF fusion, temporal decay, version-aware scoring, cross-encoder reranking (default-on), and cross-session expansion. Cross-encoder reranking compresses scores into a narrow high-side band (~0.6–0.8) regardless of topical overlap, so `top_score_low` became a steady-state false negative on every rerank-enabled query (an off-topic query scoring 0.715 never tripped abstention). `compute_abstention_signals`' kwarg is renamed `top_chunk_score` → `top_vector_score`; both engines now forward the pre-rerank, pre-fusion raw cosine (`max_raw_cosine` / `result.metadata["max_raw_vector_score"]`).
- **vectorcypher: route the BM25 channel to the temporal-store chunk table** (#816, originally #813). The streaming ingest path writes chunks to `khora_chunks` (the temporal-store table), but the retriever's BM25 channel queried the relational `chunks` table — so `enable_bm25_channel=True` was a silent no-op for any namespace ingested via `remember_batch`. Adds a `search_fulltext(...)` method on the pgvector / sqlite_lance / surrealdb temporal stores plus a `temporal_chunk_to_chunk` adapter (preserving `chunker_info`, `created_at`, `source_timestamp`, `session_id`); `VectorCypherRetriever._bm25_search_chunks` now prefers the temporal store's reader and falls back to the coordinator-level relational reader otherwise. Emits a one-shot WARNING per namespace when the enabled BM25 channel returns 0 chunks. Postgres-only, dialect-gated, idempotent Alembic migration 039 adds `CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_content_tsv` for deployments that run migrations before any process opens the temporal store.
- **migration `037_recall_response_format`: reorder nullability flip ahead of empty-string normalization** (#819). On the six `documents` columns (`source`, `content_type`, `title`, `author`, `language`, `checksum`) the upgrade ran `UPDATE ... SET <col> = NULL WHERE <col> = ''` before the `DROP NOT NULL`. On databases created via the legacy `create_tables()` / `Base.metadata.create_all` path — where the old `Mapped[str]` ORM declared these columns `NOT NULL` — writing NULL violated the still-present constraint and rolled back the whole revision, leaving the database stuck at the prior revision. Alembic-chain-created databases already had the columns nullable, so the bug never fired there. Fixed by dropping NOT NULL (and the default) first, then running the empty-string → NULL normalization.
- **vectorcypher: stamp co-occurrence edges with chunk + document provenance** (#810). `_build_cooccurrence_relationships` now takes the originating `chunks: list[Chunk]` argument and sets `source_chunk_ids=[chunk_id]` / `source_document_ids=[chunk.document_id]` on every ASSOCIATED_WITH edge it emits, matching the sibling builder in `pipelines/flows/ingest.py`. Previously the four call sites in `engines/vectorcypher/engine.py` (standard create-path, skeleton-deferred, multi-stage extraction, batch path) persisted co-occurrence relationships to Neo4j with empty provenance arrays, leaving downstream provenance/expansion queries unable to trace these edges back to their source chunks or documents. If an entity's `source_chunk_ids` references a chunk that is not in the passed list (shouldn't happen via normal extraction paths), the builder logs a warning and falls back to an empty `source_document_ids` list rather than crashing.

### Chores

- **Parallelize integration tests via `pytest-xdist` (`-n auto`)** (#817). Integration tests ran serial while unit tests already ran parallel (162 integration tests in ~230s serial). Real-Postgres / real-Neo4j tests stay gated behind their own env vars and are skipped in this job, so xdist worker isolation is not a concern; `--cov-append` already combines per-worker `.coverage.<worker_id>` files.

## [0.16.3] - Recall response shape fixes + chunker_info persistence

Patch release on top of v0.16.2. Two end-to-end recall-contract fixes (chunker self-identification now round-trips on vectorcypher; unset optional strings serialize as `null` rather than `""`) plus a CVE-allowlist trim as `litellm` SSTI is no longer load-bearing on the pinned version.

### Fixed

- **vectorcypher: persist `chunker_info` end-to-end** (#800). `Chunk.chunker_info` now surfaces the chunker self-identification dict on recall instead of `{}`. Wires the field through every `TemporalChunk(...)` write site (single-document, replace-document, batch), every `Chunk(...)` retriever construction (Neo4j JSON string or `TemporalChunk` field), and the Cypher `RETURN` clauses that feed retriever sites. New Alembic migration `038_khora_chunks_chunker_info` adds the column to production deployments that predate it; fresh deploys and the sqlite_lance test fixture pick it up via `metadata.create_all` / runtime DDL.
- **recall: surface unset optional strings as `null` on recall response** (#801). `title`, `external_id`, `source`, `source_name`, `source_url`, and `content_type` now serialize as `None` (→ `null`) rather than `""` when unset, on all three relational backends (postgresql, sqlite_lance, surrealdb). The ingest pipeline's `text/plain` default on `content_type` is dropped at all three call sites so unset content types stay `None` end-to-end. Behavior of `source_type` (still defaults to `"library"`), `source_timestamp`, and `metadata` is unchanged.

### Chores

- **Drop `GHSA-xqmj-j6mv-4862` from CVE allowlist** (#778, #799). `litellm` SSTI fix landed in 1.83.7; khora pins `litellm` 1.84.0, so the allowlist entry was no longer load-bearing. Verified via `pip-audit`: trimmed allowlist still reports zero known vulnerabilities.

## [0.16.2] - Fork-safe integrations + vectorcypher proxy fix + Hermes adapter

Patch release on top of v0.16.1. Two production-impacting fixes (fork-safe integration globals; `Khora.recall()` `AttributeError` on session-less corpora introduced by the v0.16.0 namespace proxy), plus the Hermes memory-provider adapter and the env-var naming consolidation that already landed on `main` since v0.16.1.

### Added

- **Hermes memory provider adapter** (#628). `khora.integrations.hermes.KhoraMemoryProvider` wires a `Khora` instance into the NousResearch/hermes-agent `MemoryProvider` ABC. Backs `initialize`, `prefetch` / `queue_prefetch` (with per-session cache), `sync_turn`, `on_pre_compress`, `on_session_end`, and two LLM-callable tools (`memory_search`, `memory_recall`). One `concurrent.futures.ThreadPoolExecutor(max_workers=1)` per provider gives strict FIFO ordering of writes; all async work goes through the shared `khora.integrations._sync.run_sync` bridge - no per-provider asyncio loop. Stability: experimental. Distribution: (a) - adapter in the khora repo; plugin directory shipped under `examples/integrations/hermes/plugin/` (copy into `$HERMES_HOME/plugins/khora/`). The `[hermes]` extra is intentionally NOT declared because `hermes-agent==0.13.0` exact-pins `requests==2.33.0`, which conflicts with khora's CVE-2026-25645 constraint (`requests>=2.33.1`); install `hermes-agent` manually until upstream loosens its pin. New telemetry: 3 internal spans (`khora.integrations.hermes.{initialize,prefetch,sync_turn}`) and 4 public counters (`khora.hermes.tool_call_total`, `khora.hermes.remember.{success,failed}_total`, `khora.hermes.queue.shed_total`). See `docs/integrations/hermes.md`.
- **Configurable Neo4j entity-provenance caps with observability** (#777). `Neo4jConfig.entity_source_document_ids_max` (default 100, env `KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX`) and `Neo4jConfig.entity_source_chunk_ids_max` (default 250, env `KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX`) replace the hard-coded `[-100..]` / `[-250..]` Cypher slices on `Entity.source_document_ids` / `Entity.source_chunk_ids` in `Neo4jBackend.upsert_entities_batch`. Defaults preserve pre-#777 behavior; deep-provenance workloads can raise either knob. New public telemetry counter `khora.neo4j.entity.source_id_truncated{field, kind}` increments by the number of entries dropped whenever (existing + incoming) exceeds the cap, paired with a `logger.warning(...)` that names the field, dropped count, rows affected, and the actionable env var. The drop count is computed in Python from the existing prefetch pass (no extra round-trip). Mirror of #737 (relationship side).

### Changed

- **Env var naming consolidated to single underscore everywhere** (#789). Canonical form is now `KHORA_STORAGE_GRAPH_URL`, `KHORA_STORAGE_SQLITE_LANCE_DB_PATH`, `KHORA_DREAM_OPS_DEDUPE_ENTITIES`, etc. - matches the existing `KHORA_LLM_*` / `KHORA_QUERY_*` / `KHORA_PIPELINES_*` flat style. The legacy double-underscore form (`KHORA_STORAGE__GRAPH__URL`) is kept working silently for backward compatibility via per-field `AliasChoices` - no deprecation warnings; existing `.env` files do not need to change. New docs, error messages, and examples use only the single-underscore form. Same PR pins `pydantic-settings>=2.10.1,<3` to lock the behavior, adds `extra="forbid"` to all discriminated-union members (closes a latent silent-drop footgun where backend-mismatched env vars were dropped without error), and adds a conflict-detection model_validator that raises if both forms are set with different values for the same field.
- **Neo4j entity-provenance truncation is no longer silent** (#777). Prior to this change, over-cap entries on entity MERGE were dropped with no log line and no metric - operator-visible only via downstream provenance gaps. The bulk-mode (`--rewrite`) path is untouched because ON MATCH cannot fire there (new namespace, no existing entities to merge against). Only the Neo4j backend is affected - pgvector / sqlite_lance / surrealdb / memgraph / age / neptune use `CREATE` rather than `MERGE`-with-tail-slice for entities and have no equivalent silent path.

### Fixed

- **`khora.integrations._sync` and `Khora.shared()` are now fork-safe** (#790). Both surfaces own process-global state that became stale in a forked child: `_sync` owns a daemon-thread asyncio loop that does not survive fork (next `run_coroutine_threadsafe(...)` would hang forever on the orphaned parent loop), and `Khora.shared()` caches connected `Khora` instances by config hash with asyncpg pools whose fds the child also has open (reuse races the parent; asyncpg protocol machinery is not fork-safe). Fix registers `os.register_at_fork(after_in_child=...)` on both modules to drop parent state - the next call from the child rebuilds the bridge / re-instantiates `Khora` lazily. Handlers deliberately do NOT try to disconnect parent pools from the child (the same fds would close in the parent). POSIX-only; Windows is a no-op. Unblocks fork-after-import deployment shapes: gunicorn pre-fork, `multiprocessing.Pool`, Celery prefork, `uvicorn --workers N`. Affects every adapter under `khora.integrations.*` - hermes, crewai, google_adk, langgraph, llamaindex, openai_agents.
- **`VectorCypherRetriever` no longer raises `AttributeError('_session')` on session-less corpora** (#793, closes #792). After #765 wrapped `StorageCoordinator.graph` in `NamespaceRequiredProxy`, the proxy's `__getattr__` rejected underscore-prefixed lookups, so the first call to `DualNodeManager._session()` raised. The session-aware retrieval path swallowed it as "falling back to global search", but the very next `_fetch_chunks_from_entities` call hit the same proxy attribute and bubbled the error out of `Khora.recall()` - affecting roughly 90% of queries against corpora ingested without session metadata. Fix: `VectorCypherRetriever` now reads `storage._graph` (matches the internal-access pattern already documented in `khora.storage._namespace_proxy` and used by `VectorCypherEngine.connect()`).

### Chores

- **`ruff` bumped to `~=0.15.12` and the pre-commit hook id renamed to `ruff-check`** (DYT-4538). Aligns with ruff's upstream rename of the hook id; lockstep with the dependency bump avoids the "unknown hook id" pre-commit failure on fresh clones.
- **Opt-in Weaviate service added to `compose.yaml`** (#786). Helps local triage of Weaviate-backed integrations without touching the default `make dev` stack.
- **Orphan `fly.toml` removed** (#785). Leftover from a deployment that never shipped; khora is a library, not an app.

## [0.16.1] - Bug-fix sweep + CI security allowlist

This is a bug-fix release on top of v0.16.0. No new breaking changes; one new config knob (Neo4j source-id truncation caps); one new kwarg (`Khora.remember(..., source_timestamp=…)`); two formatters promoted to the `khora.core.recall_context` public surface. Several CHANGELOG entries below reconstruct work that landed since v0.16.0 but lost its `[Unreleased]` notes to squash-merge collisions.

### Added

- **Explicit `source_timestamp` kwarg on `Khora.remember()`, `Khora.remember_batch()`, and `Khora.submit_batch()`** (PR #779). Callers can now pass `source_timestamp: datetime | None` directly instead of relying on the metadata-derived fallback. Per-doc dict key (`source_timestamp`) on `remember_batch` / `submit_batch` overrides the top-level kwarg for that document, mirroring the existing `source_type` / `source_name` / `source_url` pattern. When the kwarg is provided it wins over the metadata-based fallback (`sent_at` / `occurred_at` / `created_at` / …); when omitted, the existing fallback in `_extract_source_timestamp` is preserved unchanged.
- **Configurable Neo4j relationship-provenance caps with observability** (PR #775, closes #737). `Neo4jConfig.relationship_source_document_ids_max` (default 100, env `KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX`) and `Neo4jConfig.relationship_source_chunk_ids_max` (default 250, env `KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX`) replace the hard-coded `[-100..]` / `[-250..]` Cypher slices on `Relationship.source_document_ids` / `Relationship.source_chunk_ids`. Defaults preserve pre-#737 behavior; deep-provenance workloads can raise either knob. New public telemetry counter `khora.neo4j.relationship.source_id_truncated{field, kind}` increments by the number of entries dropped whenever (existing + incoming) exceeds the cap. The OPTIONAL MATCH that captures the pre-MERGE union size adds one indexed lookup per row on `create_relationships_batch` and `create_relationship`; cost is negligible against the MERGE itself. Same shape exists on the entity-side `upsert_entities_batch` at `neo4j.py:1343-1344` - tracked separately in #777.

### Changed

- **Neo4j relationship-provenance truncation is no longer silent** (PR #775, closes #737). Both `Neo4jBackend.create_relationships_batch` and `create_relationship` now emit a `logger.warning(...)` on every truncation event with the field name, dropped-entry count, rows affected, configured limit, and the relationship type - alongside the counter above. Prior to this change, over-cap entries were dropped with no log line and no metric, masking provenance loss until operators manually queried `size(r.source_document_ids)` on suspect rows. Only the Neo4j backend was affected - Memgraph / AGE / Neptune / SurrealDB / sqlite_lance use `CREATE` rather than `MERGE`-with-tail-slice and have no equivalent silent path.
- **`context_text` formatters consolidated to `khora.core.recall_context`** (PR #780). `format_entity_section` and `format_relationship_section` are now public on `khora.core.recall_context` (added to its `__all__`). The chunk-grouping block of `context_text(result)` is extracted into a module-private `_group_chunks_by_title(chunks, documents, max_chunks)` helper. Output of `khora.context_text(result)` is byte-identical to prior behavior (golden-string tests verify).

### Fixed

- **`Neo4jBackend._record_to_relationship` resilient to edges missing `id` / `namespace_id`** (PR #767, closes #763). The mapper hard-subscripted `rel["id"]` / `rel["namespace_id"]` while every other field used `rel.get(...)` with a default. A single relationship whose property map lacked either key raised `KeyError`, which propagated out of the comprehension and broke entire neighbourhood / relationship-batch reads. Now every field is read with a safe default; rows with no recoverable id are dropped with a debug-level log line.
- **Relationship-type sanitization now consistent across every graph backend** (PR #774, closes #749). Pre-#749 the Cypher backends (Neo4j, Memgraph, Neptune, AGE, sqlite_lance) UPPER_SNAKE_CASEd `Relationship.relationship_type` while SurrealDB stored the raw string verbatim - feeding `"lives in"` to both adapters then reading back via `get_entity_relationships` returned `"LIVES_IN"` from sqlite_lance and `"lives in"` from SurrealDB. Same input, semantically different output, broke any cross-backend filter or replay code. Every backend now funnels `relationship_type` through the shared `sanitize_cypher_label` helper at write time and mirrors the sanitized form back onto the caller's `Relationship` object so the in-memory model matches what is persisted. AGE additionally moves from its bespoke case-preserving regex to the shared helper (the previous AGE regex also had a latent crash on empty input - `[r:]` is invalid Cypher - now handled via the shared `RELATES_TO` fallback). `_sanitize_neo4j_label` is preserved as an internal alias for `sanitize_cypher_label` so the coordinator and vectorcypher engine keep their existing imports.
- **v0.16.0 CHANGELOG cosmetic corrections** (PRs #773 + this release):
  - The v0.16.0 `### Added` entry for `ChunkResult` self-identification originally referenced `ChunkResult.metadata["chunker_strategy"]`. The actual key every chunker stamps is `"chunker"` (with values `"fixed"` / `"recursive"` / `"semantic"` / `"conversation"`); the same key propagates to `RecallChunk.chunker_info["chunker"]` on recall (PR #773).
  - The v0.16.0 `### Fixed` entry on `Khora.recall().documents[]` implied the `Khora._upgrade_recall_documents` pass was closing a tuple→typed gap. It isn't - the architect's investigation while removing the docs-TODO confirmed `_upgrade_recall_documents` is a separate enrichment step (batched `DocumentProjection` source-metadata lookup + reverse-index of chunk→entity edges) operating on already-typed engine output. `documents[]` is producer-enforced by every engine (vectorcypher / chronicle / skeleton) before the upgrade pass runs.

### Security

- **CI security allowlist for 20 unfixable upstream CVEs** (PR #781, tracked in #778). pip-audit surfaced 20 vulnerabilities in pinned ML dependencies (joblib 1.5.3, transformers 5.8.0, torch 2.11.0) with no upstream fix versions. PyPI confirms no patches: joblib 1.5.3 is the latest; transformers 5.8.1 is a Deepseek-V4 integration patch with no security mentions; torch 2.12.0's "Security" section is a single bullet on GitHub Actions SHA pinning. Each CVE is now explicitly allowlisted in `.github/workflows/ci.yml` with a per-CVE justification of why khora's usage doesn't reach the vulnerable surface. khora's only direct ML usage is `sentence_transformers.CrossEncoder(model_name)` in `khora.query.reranking` against vetted reranker model IDs - `torch` / `transformers` / `joblib` are never imported directly by khora source. The vulnerable codepaths (checkpoint deserialization for Perceiver / Transformer-XL / megatron_gpt2 / X-CLIP / GLM4 / SEW / HuBERT, plus `torch.jit` / `torch.profiler` / quantization / RNN packing / CUDA caching-allocator / `.pt2` archives / `joblib.numpy_pickle`) are not reachable through khora's public API. The allowlist will shrink as upstream fixes ship; #778 tracks the weekly-revisit operating procedure.

### Removed

- `khora.query.engine.QueryResult.get_context_text` - use the public `khora.context_text(result)` helper instead (same output) (PR #780).
- `khora.query.engine.format_entity_section` and `format_relationship_section` - moved to `khora.core.recall_context` (PR #780).

### Chores

- **Source-tree scrub of internal ticket IDs** (PR #773). Removed every `IGR-*` reference from `src/` and `tests/` docstrings, comments, and test fixtures (212 substitutions across 43 files). `IGR-NNN:` comment prefixes → `Security:`; ` - IGR-NNN` trailing tags dropped; `(IGR-NNN)` standalone → `(IDOR family)`. Mirrors the prior PR-body and CHANGELOG scrubs; preserves all surrounding security context while removing internal-tracker IDs that were never meant for the public source surface.
- **Documentation: HybridQueryEngine layering reframe** (PR #776). `docs/query-engine/overview.md` previously carried a TODO claiming `HybridQueryEngine.query()` and `Khora.recall()` were divergent shapes of the same retrieval path. They aren't - `Khora.recall()` goes through the typed engines (vectorcypher / chronicle / skeleton) that build `RecallResult` directly, while `HybridQueryEngine.query()` is consumed only by `khora.query.agentic.AgenticSearchAgent`. The doc now documents both as independent retrieval surfaces with intentionally different result shapes (richer per-method telemetry on the agentic path vs. typed projection on the public-API path).

## [0.16.0] - Cross-namespace IDOR family close-out + SurrealDB interpolation repair

### Security

- **Cross-namespace IDOR family - full close-out across reads, writes, and the coordinator surface** (PRs #761 / #765 / #766 / #769). Earlier facade-only fixes in PRs #721 / #722 hardened three coordinator methods by post-fetch namespace comparison. The vulnerability surface was actually much wider: every `StorageCoordinator.{graph,vector,relational,event_store}` sub-backend (public dataclass fields) exposed un-namespaced `get_*` / `entity_exists` / traversal / write / delete methods that operated across tenants given just an id. The `GraphBackendProtocol` and `RelationalBackendProtocol` themselves declared read- and write-by-id methods without a `namespace_id` argument - every backend implementation correctly mirrored the Protocol, so the bug was uniform across surrealdb / pgvector / postgresql / sqlite_lance / neo4j / age / memgraph / neptune. **Tightened the Protocol contracts** in `src/khora/storage/backends/base.py`: every read, exists-check, and mutation method now requires `*, namespace_id: UUID` (kwarg-only). Every backend implementation filters at the SQL / Cypher / SurrealQL layer - not post-fetch (post-fetch leaks timing as an existence oracle and still touches foreign rows in the DB). Cross-namespace reads return `None` / `{}` / `[]`; cross-namespace writes silently no-op (raising would expose row existence). Graph traversal (`get_neighborhood*`, `find_paths`) filters at every hop so a traversal seeded inside namespace A cannot visit a node in namespace B. **Methods tightened (reads):** `RelationalBackend.get_document` / `get_documents_batch` / `get_document_sources_batch` / `get_document_projections_batch` / `get_document_by_external_id` / `get_documents_by_external_ids`; `VectorBackend.entity_exists` plus pgvector-specific `get_entity` / `get_entities_batch`; `GraphBackend.get_entity` / `get_entities_batch` / `get_relationship` / `get_episode` / `get_entity_relationships` / `get_neighborhood` / `get_neighborhoods_batch` / `find_paths` / `get_temporal_neighbors`; `EventStore.get_events_for_resource` / `get_latest_event`; plus the `SurrealDB.vector.get_entity` silent-accept bug (kwarg present, docstring admitted it was ignored, SQL didn't filter - now actually filters). **Methods tightened (writes):** `RelationalBackend.delete_document`; `VectorBackend.delete_chunks_by_document` / `update_entity` / `update_entity_embedding` / `update_entity_embeddings_batch` / `delete_entities_batch` / `delete_relationships_batch` / `supersede_fact`; `GraphBackend.update_entity` / `delete_entity` / `delete_relationship` / `delete_entities_batch` / `delete_relationships_batch` / `retire_orphaned_relationships_batch` (Neo4j) / `remap_source_document_ids_batch` (Neo4j). **Coordinator privatized:** `StorageCoordinator.{relational,vector,graph,event_store}` are now wrapped in a `NamespaceRequiredProxy` that emits a `DeprecationWarning` on first access per role per process and refuses dispatch on read methods missing `namespace_id=` - public attribute kept as a deprecation surface, removed in v0.17. **AGE Cypher hardening:** every UUID f-string interpolation in `storage/backends/age.py` (entities, relationships, episodes, traversal seeds, attribute search) routes through a validated `_uuid_lit(...)` helper. Type-safe today (UUIDs only), injection-resistant if a duck-typed caller appears. **sqlite_lance defense-in-depth:** `vector.search_similar` SQLite re-fetch now filters by `namespace_id` in the SQL `WHERE` clause; no longer trusts LanceDB's filter alone. **Regression gate:** `tests/security/test_cross_namespace_idor_signatures.py` walks every concrete backend, enumerates every `get_*` / `entity_exists` / `find_paths` / `get_neighborhood*` / `delete_*` / `update_entity*` / `supersede_*` method with a required id-typed parameter, and asserts `*, namespace_id: UUID` is in the signature kwarg-only. Fails at collection time on any backend method that violates the contract. **Breaking:** `Khora.get_document(doc_id, *, namespace=…)` now requires the namespace kwarg; the same applies to every coordinator/backend method listed above. Per `docs/consumers.md` §"Security exception", this carve-out is sanctioned for the minor bump. `kb.storage` and the documented public surface (khora-cli, khora-explorer) are unaffected - none touch `kb.storage.{graph,vector,relational,event_store}` directly.

### Fixed

- **SurrealDB `table:⟨$var⟩` interpolation across read/write paths** (PR #770, issue #750). SurrealDB does **not** substitute parameters inside the `table:⟨$var⟩` RecordID-literal shorthand - the `$var` is parsed as a literal string. 19 sites in `src/khora/storage/backends/surrealdb/{graph,vector}.py` used this pattern, silently corrupting reads and writes. The visible symptom was three get-by-id methods (`graph.get_relationship`, `vector.get_chunk`, `vector.get_chunks_by_document`) returning `None` / `[]` for inputs that demonstrably existed in storage. The audit surfaced a wider blast radius: every relationship row created via `create_relationships_batch` had bogus `in` / `out` endpoints (stored as the literal string `entity:⟨$rel.source_rid⟩` instead of a real entity RecordID) and `rel_id = None` (silently dropped by `SCHEMAFULL` because the schema didn't declare the field). Every `table:⟨$var⟩` interpolation now binds via either (a) parameter binding via `_rid("table", uuid)` for single-record queries or (b) `(type::thing($var))` with a full `_rid()` RecordID object for loop-body cases (`FOR $x IN $xs`). Schema gains `DEFINE FIELD rel_id ON relates_to TYPE string` plus a `UNIQUE` index. New `tests/integration/storage/backends/surrealdb/test_get_by_id_roundtrip.py` covers every fixed get-by-id method against in-memory SurrealDB.
- **`Khora.recall()` always populates `RecallResult.documents[]`** (PR #760). Every `chunks[i].document_id` and every id in `entities[i].source_document_ids` / `relationships[i].source_document_ids` is guaranteed to appear in `documents[]` (producer-enforced invariant). Engines emit deduplicated `DocumentProjection` rows for every referenced document; the top-level `Khora._upgrade_recall_documents` pass batches the lookup through `storage.get_document_projections_batch`.

### Added

- **`khora.context_text(result)` helper** (PR #762). The legacy `RecallResult.context_text` attribute was removed in v0.15.3 along with the broader typed-projection refactor; this public function returns the formatted context string for adapters that need one without forcing them to roll the join logic themselves. Re-exported from the top-level `khora` namespace.
- **`ChunkResult` self-identification** (PR #759). Chunkers now stamp `ChunkResult.metadata["chunker"]` on every emitted chunk (values: `"fixed"`, `"recursive"`, `"semantic"`, `"conversation"`); downstream consumers can route on strategy identity rather than infer from chunker config. The same field is propagated to `RecallChunk.chunker_info["chunker"]` on recall. Replaces the stale `context_text` references at the chunker boundary.

## [0.15.3] - Typed recall projection; supersedes broken 0.15.2

Supersedes 0.15.2, which shipped a `khora-accel==0.15.1` pin in the `[rust]` extra pointing at a never-published accel version, breaking `pip install khora[rust]==0.15.2`. v0.15.3 restores lockstep with the matching `khora-accel==0.15.3` pin.

### Changed

- **`Khora.recall()` returns a typed projection - BREAKING.** `RecallResult` is rewritten as a JSON-serializable response projection at `khora.core.models.recall` (re-exported from `khora`). Migration:
  - `result.chunks: list[tuple[Chunk, float]]` → `list[RecallChunk]`. Read `chunk.score` / `chunk.content` / `chunk.id` directly; the per-chunk tuple is gone.
  - `result.entities: list[tuple[Entity, float]]` → `list[RecallEntity]`. Read `entity.score` / `entity.source_document_ids` / `entity.source_chunk_ids`. The new projection no longer carries the full `Entity` ORM object.
  - `result.relationships: list[tuple[Relationship, float]]` → `list[RecallRelationship]`. Same shape change. Always present (possibly empty) on every engine; previously populated only by VectorCypher.
  - **Renames:** `result.metadata` → `result.engine_info`; `result.llm_usage` → `result.usage`.
  - **Removed:** `result.context_text` is gone from the public surface - adapters that need a context string build it locally from `result.chunks[i].content`. A `khora.context_text(result)` helper will return in a follow-up.
  - **New top-level field:** `result.documents: list[DocumentProjection]` - deduplicated source documents referenced by any chunk/entity/relationship. Every `chunks[i].document_id` and every id in `entities[i].source_document_ids` / `relationships[i].source_document_ids` is guaranteed to appear in `documents[]` (producer-enforced invariant).
  - **New mandatory engine telemetry key:** every engine emits `engine_info["engine"] = "<strategy-name>"` (`vectorcypher` / `chronicle` / `skeleton`) so consumers can route on producer identity.
  - **`Khora.recall(..., include_sources=True)`** is now a documented no-op kept for API stability - the prior implementation mutated `Chunk.source_document` / `Entity.source_documents` in place, which is incompatible with frozen projections. Full source population returns with the recall-method rewrite.
  - **New public exports** from `khora` and `khora.core.models`: `DocumentProjection`, `RecallChunk`, `RecallEntity`, `RecallRelationship` (alongside the rewritten `RecallResult`).
  - **Downstream consumers** (`khora-cli`, `khora-explorer`) must be updated in lockstep - `__all__` in `khora/__init__.py` and `khora/core/models/__init__.py` is the machine-readable contract.
- **Coverage floor lifted 72% → 77%** ([#695](https://github.com/DeytaHQ/khora/issues/695) step 3+). ~500 new unit tests across 9 modules with the largest remaining gaps. Unit-only coverage rose from 73.15% to **76.87%**; combined unit+integration on CI projected ≥78%. Per-module before → after: `query/engine.py` 52→84%, `query/router.py` 52→96%, `query/reranking.py` 12→96%, `engines/vectorcypher/engine.py` 50→83%, `engines/vectorcypher/retriever.py` 71→85%, `engines/skeleton/engine.py` 49→63%, `engines/skeleton/backends/pgvector.py` 18→72%, `storage/backends/pgvector.py` (unit-only) 39→67%, `storage/backends/neo4j.py` (unit-only) 15→46%. Next ladder step is 80%.
- **Coverage floor lifted 65% → 72%** ([#695](https://github.com/DeytaHQ/khora/issues/695) step 2). `--cov-fail-under` raised in `pyproject.toml`, `Makefile`, and `.github/workflows/ci.yml`. Backed by 500+ new unit tests across 15 previously under-covered modules (query/{normalization,agentic,understanding,hyde,linking,temporal_detection}, storage/{optimize,expertise_store,event_store}, storage/backends/postgresql, pipelines/tasks/extract, pipelines/flows/ingest, extraction/{entity_resolution,expansion/rule_engine,extractors/llm}, integrations/crewai/storage). Unit-only coverage rose from ~68% to **73.15%**; combined unit+integration projected at ~75%. Next ladder steps remain at 75% / 80% / 85% per the issue plan.

### Fixed

- **`pip install khora[rust]==0.15.2` failed to resolve.** The published 0.15.2 wheel hard-required `khora-accel==0.15.1`, but accel `0.15.1` was never published to PyPI (the lockstep `pyproject.toml` pin was not bumped in the 0.15.2 release commit). v0.15.3 ships matching `khora==0.15.3` + `khora-accel==0.15.3`. Yank 0.15.2 on PyPI.
- Stale results from `Khora.recall()` after `remember`/`forget`. The in-process query result cache in the vectorcypher retriever held results for up to 5 minutes without invalidation on writes.
- **vectorcypher entity-chunk fetch crashed on SurrealDB-only deployments** ([#754](https://github.com/DeytaHQ/khora/issues/754)). `VectorCypherRetriever._fetch_chunks_from_entities` falls back to the unified `self._storage` backend when no graph (`_dual_nodes`) is wired - the case for `backend=surrealdb` with the vectorcypher engine. The fallback built each `chunk_record` dict without a `document_id` key, but the downstream result-building loop unconditionally did `UUID(record["document_id"])`, producing an unhandled `KeyError` whenever any entity had source chunks. The fallback's `try/except` only wrapped the storage calls, not the consumer loop, so the error propagated upward and crashed recall calls routed through the entity-anchored channel. Now stamps `document_id` from the `Chunk.document_id` field returned by `storage.get_chunks_batch`. Regression covered in `tests/unit/engines/vectorcypher/test_fetch_chunks_surrealdb_fallback.py`.
- **Entity-upsert advisory lock collided at ~65K namespaces** ([#738](https://github.com/DeytaHQ/khora/issues/738)). `_namespace_lock_key` folded the 128-bit `namespace_id` UUID down to a single signed `int4` via 4-way XOR, used as `key2` in `pg_advisory_xact_lock(KHOR, key2)`. Deployments with more than ~65K distinct namespaces (per-user / per-agent patterns under `khora.integrations.openai_agents`, `google_adk`, `crewai`, `langgraph`) hit birthday-paradox collisions - empirically observed at 120K in the issue's repro. Two namespaces sharing a folded key would serialize their entity upserts behind each other, producing tail-latency spikes on a random subset of namespaces. No data loss - the lock auto-released on commit and `_retry_on_deadlock` covered the contention. Replaced with `_namespace_lock_keys(...)` which fills both 32-bit slots of Postgres's two-int advisory-lock form from the full 128 bits of the UUID, giving ~2^64 effective lock-id entropy (birthday-safe at billions of namespaces). **Operators:** the legacy `0x4B484F52` ("KHOR") `classid` is no longer set on these locks - update any `pg_locks` dashboards that filter on it.
- **`PgVectorBackend.upsert_entities_batch` reported `is_new=True` for every upsert** ([#719](https://github.com/DeytaHQ/khora/issues/719)), even when the row already existed in the namespace. The implementation hardcoded `[(entity, True) for entity in sorted_entities]` despite the "`is_new` is approximate" docstring claim - the pgvector half of a postgres+neo4j dual-write disagreed with the Neo4j half's `MERGE` semantics, silently inflating any coordinator / telemetry counter keyed on `is_new=True`. The `ON CONFLICT DO UPDATE` statement now adds `RETURNING (xmax = 0) AS is_new` and maps results back to inputs by `(name, entity_type)`; `xmax = 0` is the canonical Postgres marker for "freshly inserted" vs "matched + updated" in an upsert. Verified end-to-end against real Postgres in `tests/integration/test_pgvector_upsert_is_new.py`.

### Removed

- `RetrieverConfig.query_cache_ttl_seconds` and `RetrieverConfig.query_cache_max_size` (also removed from `VectorCypherEngineConfig`). **Breaking:** passing these kwargs now raises `TypeError`. Drop them from your config.

## [0.15.1] - Security patch release

### Security

- **Cypher / SQL injection via `Entity.attributes` / `Entity.metadata` (and the equivalent `Relationship` and `Episode` fields) on the AGE graph backend.** A document submitted through `Khora.remember` whose extracted entity attributes or metadata contained a single quote was JSON-serialised unescaped into the AGE Cypher template, letting the payload close the Cypher string literal and execute Cypher of the attacker's choice. Because AGE wrapped Cypher inside a PostgreSQL `$$ … $$` dollar-quoted string, a payload containing `$$` further escalated to SQL injection on the host. Fixed by:
  1. New `AGEBackend._serialize_dict_literal()` helper that JSON-encodes and then runs the result through the existing `_escape` (single quotes, backslashes, control characters). Applied at every Cypher-template site that interpolates a dict (entity create / update, relationship create, episode create - 7 call sites total).
  2. `AGEBackend._cypher()` now wraps the inner Cypher in a uniquely-tagged dollar-quote `$khora_age$ … $khora_age$`, defanging the `$$`-breakout escalation. Inputs containing the literal tag are refused with a `ValueError` as defense in depth.

  Reachable from any caller that can submit a document to `remember()` in a deployment where `backend=age` is configured; the attacker-controlled value reaches the AGE template through the LLM extractor's `attributes` / `metadata` output.

- **Cross-namespace IDOR on `StorageCoordinator.get_entity` / `get_relationship` / `get_episode`.** The public storage-facade getters took only an ID and returned whatever the graph backend held under that ID. A caller scoped to namespace B that knew an entity ID from namespace A received the namespace-A entity verbatim, violating the per-tenant isolation invariant. `Khora.get_entity()` (top-level), the engine `get_entity` methods (`vectorcypher`, `chronicle`, `skeleton`), and the `MemoryEngineProtocol.get_entity` had the same shape. The facade now requires a `namespace_id` keyword argument and returns `None` whenever the persisted row's `namespace_id` does not match the caller's. The underlying graph-backend `get_entity` / `get_relationship` / `get_episode` methods retain their ID-only shape (they sit below the trust boundary); filtering happens at the facade.

- **Cross-namespace chunk access via `kb.storage.get_chunk` / `get_chunks_batch` / `get_chunks_by_document`.** The three chunk-getter facade methods (and their underlying vector-backend implementations in pgvector, sqlite, sqlite+lance, and surrealdb) previously accepted only an id and did not filter by namespace, allowing a caller scoped to namespace B to retrieve chunks belonging to namespace A by id (an IDOR primitive on multi-tenant deployments). The methods now require a `namespace_id` keyword argument and apply a namespace predicate at SQL level; cross-namespace ids are silently dropped from the result. All in-tree callers have been updated.

### Changed (breaking)

- **`khora.Khora.get_entity(entity_id)` now requires `namespace=...`.** Resolution mirrors `list_entities` / `find_related_entities` - accepts `str | UUID`. Calling without it raises `TypeError`. Downstream consumers (`khora-cli`, `khora-explorer`) must be updated in lockstep.
- **`StorageCoordinator.get_entity(entity_id)` / `get_relationship(relationship_id)` / `get_episode(episode_id)` now require keyword-only `namespace_id: UUID`.** Calls without it raise `TypeError`.
- **`MemoryEngineProtocol.get_entity` and its three implementations (`VectorCypherEngine`, `ChronicleEngine`, `SkeletonEngine`) gained a required `namespace_id` kwarg.**
- **`StorageCoordinator.get_chunk(chunk_id)` / `get_chunks_batch(chunk_ids)` / `get_chunks_by_document(document_id)` now require keyword-only `namespace_id: UUID`.** Same shape on the four vector-backend implementations (pgvector, sqlite, sqlite+lance, surrealdb). Calls without it raise `TypeError`; cross-namespace ids in `get_chunks_batch` are silently dropped from the returned dict; `get_chunks_by_document` returns `[]` if the document doesn't belong to the namespace.

## [0.15.0] - Dream-phase Phase 2 + Phase 4, PPR retrieval, kuzu removed

Minor release. Lands Phase 2 (planner ops) and Phase 4 (apply mode) of the [Dream Phase umbrella (#649)](https://github.com/DeytaHQ/khora/issues/649) - `Khora.dream(namespace, mode="apply")` is now end-to-end functional with bi-temporal soft-delete, per-op transactions, and snapshotted undo records. Also: Personalized PageRank retrieval for VectorCypher (#542), the kuzu backend is removed, and the README + dream-phase docs are rewritten.

### Added

- **Dream phase Phase 2 - five planner operations.** Each emits a `DreamOp` describing what `apply` mode would do; dry-run is free of side effects.
  - `vectorcypher_dedupe_entities` ([#658](https://github.com/DeytaHQ/khora/issues/658) → [PR #691](https://github.com/DeytaHQ/khora/pull/691)) - cross-batch entity resolution against the full namespace, per-type cosine thresholds (default 0.90; tighter than the online 0.85), skip-collision reporting.
  - `vectorcypher_centroid_recompute` ([#660](https://github.com/DeytaHQ/khora/issues/660) → [PR #690](https://github.com/DeytaHQ/khora/pull/690)) - three-decision planner (`centroid` / `re_embed` / `skip_multimodal`) for post-merge canonical embeddings.
  - `vectorcypher_source_chunk_ids_gc` ([#662](https://github.com/DeytaHQ/khora/issues/662) → [PR #689](https://github.com/DeytaHQ/khora/pull/689)) - plans per-entity rewrites that drop dead chunk UUIDs from `Entity.source_chunk_ids`.
  - `chronicle_fact_compaction` ([#664](https://github.com/DeytaHQ/khora/issues/664) → [PR #688](https://github.com/DeytaHQ/khora/pull/688)) - plans hard-deletes of tombstoned `memory_facts` rows past `fact_compaction_retention_days`.
  - `chronicle_event_clustering` ([#665](https://github.com/DeytaHQ/khora/issues/665) → [PR #692](https://github.com/DeytaHQ/khora/pull/692)) - clusters near-duplicate `chronicle_events` within a sliding `referenced_date` window.
- **Dream phase Phase 4 - apply mode** ([#667](https://github.com/DeytaHQ/khora/issues/667) / [#668](https://github.com/DeytaHQ/khora/issues/668) / [#669](https://github.com/DeytaHQ/khora/issues/669) → PRs [#698](https://github.com/DeytaHQ/khora/pull/698) / [#699](https://github.com/DeytaHQ/khora/pull/699) / [#700](https://github.com/DeytaHQ/khora/pull/700) / [#701](https://github.com/DeytaHQ/khora/pull/701)). `Khora.dream(ns, mode="apply")` now executes the plan. The orchestrator's `_apply_phase` calls per-op apply handlers under per-op transactions, persists `UndoRecord.before` snapshots to `undo.json` (schema `dream-undo/1`) **before** any mutation, and updates the `khora_dream_runs.last_committed_op_seq` checkpoint between ops. Five guardrails protect the path:
  - **Hard 7-day retention floor** on `fact_compaction_retention_days` - config validator rejects sub-floor values; the apply handler re-checks defense-in-depth.
  - **`KHORA_DREAM_DISABLE_APPLY` env-var kill-switch** - `mode="apply"` raises `DreamApplyDisabled` immediately without touching the DB.
  - **`chunk_id` runtime assertion** - any `UndoRecord.before` payload carrying a top-level `"chunk_id"` key aborts the run with `DreamForbiddenOpError`. The "never mutate `chronicle_events.chunk_id`" architectural promise is now a runtime guarantee.
  - **Snapshot-before-mutate** for `fact_compaction` - the only hard-delete op. `SELECT *` per target row runs before any `DELETE`; mid-snapshot failure rolls back with zero rows touched. TDD-pinned by `test_snapshot_captured_before_delete_executes`.
  - **Advisory lock held through apply** - the per-namespace `pg_advisory_xact_lock` covers planning *and* application. Concurrent dream runs against the same namespace fast-fail with `DreamLockUnavailable`.
- **Migration 034 - `chronicle_events` bi-temporal columns** ([PR #701](https://github.com/DeytaHQ/khora/pull/701)). Adds `invalidated_at`, `invalidated_by`, and `merged_into_event_id` (self-FK, `ON DELETE SET NULL` on Postgres) to `chronicle_events`. Postgres-only partial composite index `ix_chronicle_events_live` over `(namespace_id, referenced_date) WHERE invalidated_at IS NULL`. Required substrate for `apply_chronicle_event_clustering`.
- **Personalized PageRank retrieval for VectorCypher** ([#542](https://github.com/DeytaHQ/khora/issues/542) → [PR #693](https://github.com/DeytaHQ/khora/pull/693)). Opt-in via `KHORA_QUERY_ENABLE_PPR_RETRIEVAL=true` (default off). Wired into the retriever as a separate channel; degrades to vector-only on empty entry entities / empty graph / no seed overlap. Tuning knobs in `KhoraConfig.query`.
- **Public surface additions.**
  - `UndoRecord(op_id, op_type, before, applied_at)` - returned by every apply handler; persisted into `undo.json`.
  - `DreamApplyDisabled` exception - raised by the env-var kill-switch.
  - `OpSummary` - the shape of items in `DreamResult.ops` (aggregate counters per op kind).
- **Apply functions** registered in `khora.dream.engines.registry._APPLY_HANDLER_NAMES`: `apply_vectorcypher_dedupe_entities`, `apply_vectorcypher_centroid_recompute`, `apply_vectorcypher_source_chunk_ids_gc`, `apply_chronicle_fact_compaction`, `apply_chronicle_event_clustering`. All honor the caller-owned-transaction contract (no commit, no log, no telemetry from the handler).
- **`SECURITY.md`** ([PR #707](https://github.com/DeytaHQ/khora/pull/707)). Project security policy: scope (credential leakage, SQL/Cypher injection, path traversal, cross-tenant leakage), reporting routes (GitHub Private Vulnerability Reporting first, `security@deytahq.com` fallback), supported-versions policy (latest minor + n-1), response targets (ack 2 BD, triage 5 BD, fix 30 days high/critical or 90 days medium/low), and a "What khora already does" section listing existing controls (`SecretStr` on credentials, `bounded_text_hash` on telemetry, namespace-cardinality rule, secret-typing semgrep, pip-audit in CI, parameterized SQL/Cypher, bi-temporal soft-delete).
- **Test coverage** for Phase 4 surfaces: 37 apply-handler tests across 5 modules (chronicle + vectorcypher), 14 orchestrator-apply tests (covering kill-switch, retention floor, chunk_id assertion, resume-from semantics, undo.json incremental write + fsync), plus 5 integration tests for migration 034 (Postgres-gated).

### Changed

- **README rewritten as an evaluation-flow entry point** ([PR #708](https://github.com/DeytaHQ/khora/pull/708) + follow-ups [#709](https://github.com/DeytaHQ/khora/pull/709) / [#710](https://github.com/DeytaHQ/khora/pull/710)). New "Why khora?" section names the four problems pure vector search doesn't solve (ingest depth, recall complexity, drift, observability). New "Engines" section gives VectorCypher and Chronicle full descriptions with when-to-pick guidance; Skeleton is explicitly marked experimental. Inline 3-engine comparison table so the choice is visible without a docs hop.
- **`docs/dream-phase.md` rewritten** ([PR #697](https://github.com/DeytaHQ/khora/pull/697) + cleanup [PR #702](https://github.com/DeytaHQ/khora/pull/702)). Adds Phase 2 + Phase 4 operations alongside Phase 1 audits, full apply-mode contract, "Apply functions" API reference section, "Research & Prior Art" section with paper citations (McClelland 1995 on CLS, Schaul 2016 on prioritized experience replay, Kreps 2011 on Kafka log compaction, MemGPT / GraphRAG / Self-RAG with arXiv IDs, Köpcke & Rahm 2010 on entity resolution, Snodgrass 1999 on bi-temporal modeling), and an explicit "LLM usage" section calling out that dream phase makes **zero LLM calls** in v0.15.
- **CI coverage floor 30% → 65%** ([PR #696](https://github.com/DeytaHQ/khora/pull/696)). Matches actual main coverage (65.88% at the time of the bump) - the floor was previously misleading. Roadmap to 85% tracked under [#695](https://github.com/DeytaHQ/khora/issues/695) as staged PRs.
- **Coverage push: +~980 statements across three modules** ([PR #703](https://github.com/DeytaHQ/khora/pull/703) `pipelines/flows/ingest.py` 18% → 63%, [PR #704](https://github.com/DeytaHQ/khora/pull/704) `query/engine.py` 47% → 72%, [PR #705](https://github.com/DeytaHQ/khora/pull/705) `engines/vectorcypher/retriever.py` 48% → 69%). Step 2 of the #695 ladder.
- **codecov.yml gradient** `30...85` → `65...85`. Badge now renders neutral-to-green starting at the actual floor instead of red.
- **Default file-sink base directory documented.** With `report_file_sink_enabled=True`, reports land under `<system temp dir>/khora-dream-reports`. On Linux that's `/tmp/...`, which is wiped on reboot - documented with operator guidance to set a persistent path.
- **`codecov.yml` ignores `examples/**`** ([PR #694](https://github.com/DeytaHQ/khora/pull/694)). Pre-emptive - the adapter examples are smoke-tested but not coverage-measured.
- **Loosened the planner-`mode="apply"` contradiction.** v0.14 planners raised `NotImplementedError` on `mode="apply"`; the orchestrator now routes apply through dedicated `apply_<op>` functions instead. Direct callers of `plan_<op>(..., mode="apply")` still hit the raise - that path is reserved for testing. Tracked for follow-up cleanup.

### Removed

- **kuzu backend** ([PR #706](https://github.com/DeytaHQ/khora/pull/706)). Deprecated in v0.9.0 with a v0.10.0 removal target that never landed. The upstream Kùzu project has been archived since the October 2025 acquisition - the dependency is dead code. The `kuzu` extra (`pip install khora[kuzu]`) is gone; the `graph-all` and `all-backends` extras no longer pull in the kuzu wheel; the `KuzuBackend` / `KuzuConfig` symbols are removed from `khora.storage.backends` and `khora.config.schema`; 909 LOC of unmaintained backend code deleted. Migration path: `pip install khora[sqlite-lance]` for embedded, `pip install khora[neo4j]` for graph DB.

## [0.14.0] - Dream-phase audit foundation

Minor release. Lands Phase 0 (foundation), Phase 1 (read-only audit operations), and Phase 3 (Rust acceleration) of the [Dream Phase umbrella (#649)](https://github.com/DeytaHQ/khora/issues/649). `Khora.dream(namespace, mode="dry-run")` is live end-to-end: operators can plan a consolidation pass over their graph, see exactly what every audit op would surface (drift thresholds, tombstone ratios, schema mismatches, orphan candidates, dead chunk references), and have those decisions emitted through three independently-togglable sinks (file, semantic-event, telemetry collector). No mutation operations ship in v0.14.0 - Phase 2 (mutation-planning ops, dry-run only) and Phase 4 (apply mode) land in a follow-up release. Audit-only is the deliberate "validate demand before committing engineering" gate.

### Added

#### Phase 0 - Foundation

- **`khora.dream` module scaffolding + `DreamConfig` ([#650](https://github.com/DeytaHQ/khora/issues/650) → PR #675).** New top-level `khora.dream` subpackage with `DreamConfig` (Pydantic settings, env-var prefix `KHORA_DREAM_*`, master switch defaults to `False`), `DreamResult`, `DreamRunInfo`, `DreamMode`, `DreamScope`, `OpKind`, and internal `DreamOp` / `DreamPlan` / `DreamReport` dataclasses. `Khora.dream()` / `Khora.dream_status()` / `Khora.dream_history()` stubbed on the public `Khora` class. Stability: top-level surface is **public** (`khora.__all__`); op-kind values and sub-dataclasses are **internal** and may evolve through Phase 1 / 2 without a major bump.
- **Migration 032 `khora_dream_runs` ([#651](https://github.com/DeytaHQ/khora/issues/651) → PR #676).** Postgres-only checkpoint table for crash-resume semantics. Records `run_id`, `namespace_id`, `mode`, `state` (init/planning/applying/completed/cancelled/crashed), `plan_hash`, `last_committed_op_seq`, `heartbeat_at`, `report_path`, and error JSONB. Indexed `(namespace_id, started_at DESC)` for `Khora.dream_history()`. Dialect-gated; sqlite_lance fixture path mirrors checkpoint state via the file sink instead.
- **Migration 033 bi-temporal columns ([#653](https://github.com/DeytaHQ/khora/issues/653) → PR #674).** Adds `valid_to`, `invalidated_at`, `invalidated_by` (UUID) to both `relationships` and `memory_facts`. Backfill is null (= "still valid"). Postgres-only partial composite indexes `ix_relationships_live` / `ix_memory_facts_live` accelerate the live-fact retrieval path. Soft-delete substrate for the Phase 4 apply-mode rollout; coexists with the legacy `memory_facts.is_active` flag.
- **Advisory lock + `DreamCapable` Protocol ([#656](https://github.com/DeytaHQ/khora/issues/656) → PR #677).** `acquire_namespace_dream_lock(session, namespace_id, timeout_seconds=60)` async context manager using `pg_advisory_xact_lock` with namespace-derived lock IDs (blake2b, domain-separated from the migration lock). Embedded fallback uses an in-process `asyncio.Lock` (cross-process safety not promised on sqlite_lance). `DreamCapable` Protocol (`plan_dream` + `apply_dream` + `dream_capabilities` property, `runtime_checkable`) - engines opt in by implementing it; the orchestrator runtime-checks before scheduling.
- **Three reporting sinks + `EventType.DREAM_*` family + telemetry contract ([#666](https://github.com/DeytaHQ/khora/issues/666) → PR #678).** File sink writes `{base_dir}/{namespace_id}/{date}/{run_id}.{summary.md,events.jsonl,manifest.json,undo.json}`. Event sink bridges into the existing `HookDispatcher` with six new `EventType.DREAM_*` values (`DREAM_RUN_STARTED`, `DREAM_PHASE_STARTED`, `DREAM_OP_DECIDED`, `DREAM_PHASE_COMPLETED`, `DREAM_RUN_COMPLETED`, `DREAM_RUN_FAILED`) - reuses `SemanticFilter` cascade with two new low-cost level-0 filter fields. Collector sink emits OTel spans + metrics declared in `docs/telemetry-contract.json` (4 public top-level spans, 4 internal inner spans, 7 public metrics - none labelled by `namespace_id` per the cardinality rule). Free-text span attributes go through `khora.telemetry.bounded_text_hash`. `redact_text` config knob (`"none"|"summary"|"all"`, default `"summary"`).
- **Orchestrator state machine + `Khora.dream()` wiring ([#661](https://github.com/DeytaHQ/khora/issues/661) → PR #684).** `DreamOrchestrator` implements INIT → PLAN → REPORT (dry-run) / APPLY → FINALIZE. Acquires the per-namespace advisory lock; calls into the registered engine plugin's `plan_dream()`; fans out `DreamOp` results through the three sinks; persists checkpoint state to `khora_dream_runs` for crash-resume; cancels between ops (never mid-op); enforces the safety floor (no Document delete, no UNIQUE-invariant break, no read-only namespace). `Khora.dream(ns, mode="dry-run")` is now reachable; `Khora.dream_status(run_id)` and `Khora.dream_history(ns, limit=...)` round-trip against `khora_dream_runs`.

#### Phase 1 - Read-only audit operations

All five ops are pure-SELECT / pure-observation. Zero LLM calls, zero mutations, zero risk to production graphs. Each returns a `DreamOp` with `decision="audit_complete"` (or `"insufficient_data"` / `"empty_namespace"`) and a structured `outputs` dict; the orchestrator routes those through the three sinks.

- **Chronicle abstention-threshold drift report ([#652](https://github.com/DeytaHQ/khora/issues/652) → PR #681).** Reads the OpenTelemetry histogram of `combined_score` / `top_score` values emitted by chronicle's existing `_compute_abstention_signals` (plus a bounded in-process ring buffer for the no-logfire path) and recommends - never applies - a threshold recalibration: e.g. "p90 `top_score` is 0.18 but `abstention_min_top_score` is 0.3 - most recalls fire `top_score_low` even on good answers; consider lowering to 0.15". Refuses below `abstention_drift_min_samples` (default 1000) with `decision="insufficient_data"`.
- **Chronicle `memory_facts` tombstone audit ([#654](https://github.com/DeytaHQ/khora/issues/654) → PR #683).** Counts active / inactive (legacy `is_active`) / invalidated (`invalidated_at IS NOT NULL`) rows. Reports `tombstone_ratio`, oldest-tombstone age, p50/p90 age of inactive facts, and top-K offenders by age. Recommends a retention threshold; the Phase 2 compaction op (#664) is the actual reclaimer.
- **VectorCypher schema-drift report against `ExpertiseConfig` ([#655](https://github.com/DeytaHQ/khora/issues/655) → PR #679).** Diffs the multiset of observed `entity_type` / `relationship_type` strings against the active `ExpertiseConfig`, with four buckets: types in data but not in config, types in config but unused, frequency-delta >50% since previous run, and the relationship-type variant of each. Never auto-normalizes - `ExpertiseConfig` is declarative user intent; normalization is operator policy (deferred to Phase 5).
- **VectorCypher PageRank-based orphan-entity report ([#657](https://github.com/DeytaHQ/khora/issues/657) → PR #682).** Builds the namespace's entity-relationship graph, down-weights `ASSOCIATED_WITH` co-occurrence edges to 0.2 so they don't dominate, runs the existing `_accel.pagerank` Rust kernel, and flags entities matching all of: PR score in the bottom 5th percentile AND `mention_count <= 1` AND no recent recall hits. Returns `archive_candidate=true` flags; the operator decides what to do next.
- **VectorCypher `source_chunk_ids` array-length audit ([#659](https://github.com/DeytaHQ/khora/issues/659) → PR #680).** Joins entities × chunks (Postgres `unnest`; SQLite Python-side) to count dead chunk-UUID references per entity, then reports the array-length distribution (p50/p90/p99) and top-K offenders. Surfaces the GC candidates for the Phase 2 source-chunk-ids GC op (#662).

#### Phase 3 - Rust acceleration

- **`khora._accel.block_and_score_pairs` ([#663](https://github.com/DeytaHQ/khora/issues/663) → PR #685).** New Rust kernel in `khora-accel` for pairwise cosine similarity with optional token-prefix name blocking. Powers the Phase 2 cross-batch entity-resolution op (#658) at namespace scale: at N≈100k entities, naive pairwise is ~5×10⁹ cross-products (~30s wall); token-blocking cuts the candidate set ~10× on realistic name distributions. **Benchmark: 6.4× speedup** vs `pairwise_cosine_above_threshold` at N=10k, D=128, threshold=0.85 (72.9 ms vs 463.9 ms). Falls back to pure-Python when the `khora[rust]` extra isn't installed. Same module pattern as the existing `cosine.rs` kernels - rayon-parallel, GIL released via `py.detach()`.

### Changed

- **`khora-accel` version bumped from 0.13.0 to 0.14.0** (lockstep with khora itself per the version-bump contract). Root `pyproject.toml`'s `rust` extra pin updated to `khora-accel==0.14.0`; `rust/Cargo.lock` regenerated.
- **Telemetry contract additions** (`docs/telemetry-contract.json`): four public top-level `khora.dream.*` spans (`run`, `phase`, `llm_call`, `undo`), four internal per-op spans (`op`, `entity_merge`, `edge_prune`, `community_summary`) plus five per-Phase-1-op spans, and seven public aggregate metrics (`khora.dream.runs_total`, `khora.dream.run.duration`, `khora.dream.phase.duration`, `khora.dream.ops_total`, `khora.dream.llm.tokens`, `khora.dream.undo_invocations_total`, `khora.dream.report.write_failures_total`). None labelled by `namespace_id` (cardinality rule).

### Not yet shipped (planned for follow-up releases)

- **Phase 2 - mutation-planning ops (dry-run only).** Cross-batch entity resolution, centroid recompute, source_chunk_ids GC, chronicle fact compaction, chronicle event clustering. All five will land in a follow-up release; their tickets are written (#658, #660, #662, #664, #665) and link the umbrella.
- **Phase 4 - apply mode.** Flips Phase 2 ops to actually mutate state. Requires the audit-only release to bake first; the kill criterion is whether operators actually invoke `kb.dream()` in production within ~30 days.
- **Phase 5 - advanced operations.** Community detection + summaries (opt-in, LLM-heavy), edge pruning by weight × recency, contradiction detection, schema-drift normalization (operator-supplied mapping).

## [0.13.0] - Agentic framework adapters; session_id first-class; SurrealDB 2.0 stable

Minor release. Five new opt-in adapters for agentic frameworks (CrewAI, LangGraph, Google ADK, OpenAI Agents SDK, LlamaIndex), a new `session_id` first-class column with cascade-delete and TTL helpers, and the SurrealDB 2.0 stable pin. No public API removals.

### Added

- **`khora.integrations` adapter foundation ([#619](https://github.com/DeytaHQ/khora/issues/619) → PR #631).** New subpackage exposing three runtime-checkable Protocols (`MemoryAdapter`, `RetrieverAdapter`, marker `KhoraIntegration`), an entry-point registry (group `khora.integrations`, with `register()` test escape hatch), the `_sync.run_sync` cross-thread bridge, and a config-hash-keyed `Khora.shared()` process-wide singleton. Adapter submodules MUST NOT import their framework at top level - enforced by `tools/check_optional_imports.py` (AST lint).
- **CrewAI adapter ([#623](https://github.com/DeytaHQ/khora/issues/623) → PR #633).** `khora.integrations.crewai.KhoraMemory` is a drop-in `StorageBackend` for CrewAI's unified `Memory`. Install with `pip install khora[crewai]`. Tz-naive recency math (`_strip_tz`) keeps CrewAI's `datetime.now() - record.created_at` happy against khora's tz-aware UTC timestamps.
- **LangGraph adapter ([#624](https://github.com/DeytaHQ/khora/issues/624) → PR #634).** `khora.integrations.langgraph.KhoraStore` implements `BaseStore` for semantic long-term memory inside `StateGraph` runners. Install with `pip install khora[langgraph]`.
- **Google ADK adapter ([#626](https://github.com/DeytaHQ/khora/issues/626) → PR #642).** `khora.integrations.google_adk.KhoraMemoryService` implements `BaseMemoryService` (`add_session_to_memory` + `search_memory`). Namespace is UUID5 of `adk:{app_name}:{user_id}`; `Session.id` round-trips via `session_id`. Memory only - no `KhoraSessionService` in v1 (ADK's `DatabaseSessionService` already covers turn state). Install with `pip install khora[google-adk]`.
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

## [0.12.1] - Chunk source_timestamp propagation

Patch release. Single bug fix.

### Fixed

- **Chunk `source_timestamp` is no longer lost during ingest ([#615](https://github.com/DeytaHQ/khora/issues/615) → PR #616).** `chunk_document()` built `Chunk` objects with only `created_at` set, never `source_timestamp`. The ingest pipeline already parses doc-level `source_timestamp` from connector metadata (`occurred_at` / `sent_at` / etc.) and stamps it on `Document`, but the field never reached the chunks - so every recall on every (engine × backend) cell returned `chunk.source_timestamp=None`. This silently broke date-bounded recalls: the query layer's temporal scoring fell back to `chunk.created_at` (ingest time), so "last week" queries surfaced older historical rows that happened to be ingested recently. Fix is a one-line propagation in `pipelines/tasks/chunk.py` - upstream of every storage adapter, so it lights up all six (engine × backend) cells. Two regression tests cover propagation when the document has a `source_timestamp` and `None`-preservation when it doesn't (no invented timestamps).

## [0.12.0] - Temporal Phase D: HyDE + reranker + BRIN; hooks LLM cost controls; PPR enabler; SurrealDB remote transactions

Minor release. Substantial additive surface - five new feature-flagged retrieval channels and one new Python subpackage - plus two correctness bug fixes that change scoring/behaviour on graph-only and graph-less stacks. No public API removals; the `enable_hyde` flag accepts a new string shape (`auto`/`always`/`never`) but the legacy boolean form normalizes transparently.

### Fixed

- **`unify_entities` no longer crashes on graph-less stacks ([#587](https://github.com/DeytaHQ/khora/issues/587) → PR #588).** `expansion.py:load_entities` / `load_relationships` were calling `storage.graph.get_entities_by_namespace` and `storage.relational.get_entities_by_namespace` - methods that exist on **no** backend. Every call into the expansion pipeline (chronicle, vectorcypher, skeleton; PG-only and PG+Neo4j and sqlite+lancedb) crashed with `AttributeError`. Routed through `StorageCoordinator.list_entities` / `list_relationships` instead, and extended the coordinator to fall back to the vector backend when no graph backend is configured. `PgvectorBackend` gains `list_entities` / `list_relationships` so chronicle+PG-only stacks now serve the entity table without a Neo4j requirement.
- **`find_related_entities` score decay restored on graph-only backends ([#581](https://github.com/DeytaHQ/khora/issues/581) → PR #589).** VectorCypher's `find_related_entities` applies `score = 1 / (1 + distance)` on the Neo4j (dual-nodes) path, but the graph-only path (sqlite_lance, surrealdb) hard-coded `1.0` because `get_neighborhood` did not expose per-entity distance. The engine now BFS-walks the returned relationships as an undirected adjacency to recover min hop-distance and applies the same decay; results are sorted descending. No backend protocol changes - works across `sqlite_lance` and `surrealdb` without further adapter work.
- **Release pipeline no longer races the merge-commit CI ([#554](https://github.com/DeytaHQ/khora/issues/554) → PR #591).** Tag pushes that land within ~30s of the bump-PR merge raced the merge commit's `ci.yml` run; `verify-ci-green` read the runs index, found no successful run yet, and failed. We worked around this manually on v0.10.6 / v0.11.0 / v0.11.1 via `workflow_dispatch`. Replaced the one-shot check with a bounded 10-minute poll that classifies the latest run into `success` / `failed` / `running` and sleeps on `running`. Window tunable via the `VERIFY_CI_GREEN_TIMEOUT_SECONDS` repo variable.

### Added

- **Temporal-anchored HyDE for RECENCY queries ([#592](https://github.com/DeytaHQ/khora/issues/592) → PR #602, Phase D1).** When HyDE fires on a query the temporal detector classifies as `RECENCY` / `STATE_QUERY` / `CHANGE`, `HyDEExpander` selects a system prompt that anchors the hypothetical to today's ISO date with explicit dates / weekdays / relative-time markers. Other categories keep the generic time-blind prompt. Zero additional LLM calls; only the prompt string changes. Category detection runs in Rust Aho-Corasick (sub-ms). See [`docs/query-engine/temporal-queries.md`](docs/query-engine/temporal-queries.md#temporal-anchored-hyde).
- **HyDE-Cypher templated graph queries ([#595](https://github.com/DeytaHQ/khora/issues/595) → PR #610, Phase D2, opt-in).** New module `khora.query.hyde_cypher` with three parameterized Cypher templates (`recent_by_type`, `entity_relationships`, `cooccurrence`). An LLM picks a template and fills slots; slot values are bound via Neo4j `$placeholder` parameters (never string-interpolated) and validated against `ExpertiseConfig` whitelists. Failures (timeout, hallucinated id, validation error) degrade to text-HyDE - never crashes the query. Enable via `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`. **Default OFF** pending an A/B on a hand-curated structured-query set.
- **BRIN index on `chunks.created_at` ([#593](https://github.com/DeytaHQ/khora/issues/593) → PR #603, Phase D4).** New migration `029_chunks_created_at_brin` adds `CREATE INDEX CONCURRENTLY` (in an autocommit block) on the time-correlated `created_at` column, `pages_per_range = 32`. KB-sized footprint that doesn't compete with HNSW or B-trees; helps long-range archive / export scans. Postgres-only - SQLite-backed sqlite_lance stacks skip the migration silently via a dialect gate.
- **Cross-encoder reranker date-prefix experiment ([#594](https://github.com/DeytaHQ/khora/issues/594) → PR #609, Phase D5, opt-in).** `CrossEncoderReranker(include_date_prefix=True)` prepends `[YYYY-MM-DD] ` to each candidate's content. Source priority: `metadata.custom.occurred_at` → `metadata.custom.sent_at` → `metadata.created_at`. `create_reranker` cache key now includes the flag so the two variants coexist without a 500ms model reload. **Default OFF** pending A/B.
- **PPR enabler: `pagerank(personalization=...)` ([#597](https://github.com/DeytaHQ/khora/issues/597) → PR #604).** Both the Rust impl (`rust/khora-accel/src/pagerank.rs`) and the Python fallback (`khora._accel.pagerank`) accept an optional L1-normalizable personalization vector. PPR formula: `r = (1 - d) * p + d * Mᵀ r`. When `personalization` is `None` or uniform-equivalent, this reduces to standard PageRank - every existing call site continues to receive identical scores. Validation: negatives clipped to 0, length mismatch falls back to uniform, all-zero falls back to uniform. The Python wrapper only forwards the new kwarg when set so older `khora-accel` wheels keep working until rebuilt. **Enabler only** - the BFS+RRF → PPR swap in VectorCypher is still gated on the graph-density audit (#598).
- **PPR graph-density audit reporter ([#598](https://github.com/DeytaHQ/khora/issues/598) → PR #605).** New `khora.diagnostics.graph_density.compute_graph_stats()` returns per-namespace `|V|`, `|E|`, mean + median degree, connected-component count, largest-CC fraction, mean degree restricted to the largest CC, plus a `meets_ppr_threshold` flag applying the #598 decision rule (≥3 connected components OR mean degree ≥5 in the largest CC). Operator script: `scripts/audit_graph_density.py` emits CSV or JSON plus a stderr verdict. **`khora.diagnostics` is explicitly NOT stable public API** and may be renamed without a major-version bump.
- **Hooks Level 2 LLM cache + per-subscription budget ([#601](https://github.com/DeytaHQ/khora/issues/601) → PR #607).** Cross-batch decision cache keyed on `(filter_id, bounded_text_hash(event_summary))` - repeat bulk-upsert events that share `name`/`type`/`description` short-circuit the LLM. TTL + LRU eviction, configurable via `KHORA_HOOKS_LLM_CACHE_SIZE` (default `2048`) and `KHORA_HOOKS_LLM_CACHE_TTL_SECONDS` (default `3600`). Per-subscription rolling-hour token budget alongside the namespace cap so a single noisy filter cannot drain its namespace allowance: `KHORA_HOOKS_LLM_MAX_TOKENS_PER_SUBSCRIPTION_PER_HOUR` (default `0` = disabled). New telemetry metrics `khora.hooks.llm.cache_hits_total` (label `category={match,no_match}`) and `khora.hooks.llm.cache_misses_total`.
- **Hooks Level 2 intra-batch event-summary coalescing ([#608](https://github.com/DeytaHQ/khora/issues/608) → PR #611).** Within a single LLM batch, `_evaluate_bucket` now deduplicates pending pairs by `event_summary_hash` before building the prompt and fans the decision out to every awaiting future. Combined with the #607 cross-batch cache, a burst of 50 identical events spends exactly **1 LLM call** even on a cold cache (the original acceptance bar was ≤2). When events differ this is a no-op.
- **SurrealDB remote-mode transactions ([#541](https://github.com/DeytaHQ/khora/issues/541) → PR #612).** New `SurrealDBConnection.transaction()` async context manager wraps the body in `BEGIN TRANSACTION` / `COMMIT TRANSACTION` on remote (`ws://`) mode, with `CANCEL TRANSACTION` on exception (original error preserved if `CANCEL` itself fails). Embedded (`surrealkv://`) and memory (`memory://`) modes are no-ops - surrealkv raises on `BEGIN`, so the existing per-statement-atomicity contract is preserved. Companion `execute_batch([(sql, bindings), ...])` joins statements with `;` for an embedded-mode batched alternative; rejects parameter-name collisions across statements. New `supports_transactions` property surfaces the per-mode capability without reading internal state.

### Changed

- **`enable_hyde` flag value shape.** `KHORA_QUERY_ENABLE_HYDE` now accepts string values `auto` (default), `always`, or `never`. Legacy booleans (`True` / `False`) are still accepted and normalize to `always` / `never` respectively - no breaking change at the API boundary, but operators reading the config docs in v0.11.x and earlier will see a different field shape now.
- **Pre-v0.12.0 documentation pass (PR #613).** Top-level README quickstart fixed (the v0.11.x example called `kb.create_namespace("demo")` which is a `TypeError` - `create_namespace` is keyword-only). Hooks doc rewritten to cover Phase 2 + v0.12.0 surface: EventBridge `match` DSL, `CHUNK_ENTITIES_RESOLVED`, default-OFF Level 2 cost warning, per-namespace + per-subscription budgets, cache + intra-batch coalescing, co-occurrence example. Fixed three wrong default values in `docs/hooks/semantic-hooks.md` (`gpt-4o-mini` → `gpt-4.1-nano`, `0.7` similarity → `0.5`). `docs/api-reference.md` `create_namespace` and `register_engine` signatures corrected; `BatchHandle` / `DocumentResult` added; new "Advanced (opt-in, v0.12.0)" section. CLAUDE.md gotchas extended with bullets for dialect-gated migrations, SurrealDB transactions, graph-less-stack entity listing, temporal HyDE, HyDE-Cypher, reranker date-prefix, hooks Level 2 cost controls, and the `khora.diagnostics` non-stability disclaimer.

### Removed

Nothing. Every v0.11.x symbol still exists with the same signature.

## [0.11.1] - Semantic hooks Phase 1: bugfixes + opt-in Level 2 LLM evaluator

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
  strategy, namespace_id}` - the dedup signal corporate-data
  customers care most about.

### Added

- **`RECALL_REQUESTED` / `RECALL_RESULTS_READY` / `RECALL_COMPLETED`
  events** fired from `Khora.recall()`. Shared `recall_id` (UUID)
  across the three events so subscribers can correlate. Payload caps
  (top 20 entity/chunk IDs) bound event size. Hook dispatch wrapped
  in try/except - failures never break `recall()`.
- **Optional Level 2 LLM filter evaluator** (`khora.hooks.llm_evaluator.LLMFilterEvaluator`).
  Default OFF (`KHORA_HOOKS_LLM_EVALUATION_ENABLED=false`).
  Micro-batched (10 pairs / 100 ms window), JSON-schema output via
  `khora.config.llm.acompletion` with `_telemetry_op="hooks.filter_eval"`,
  per-namespace token-budget cap (default 10k tokens/hour ≈ 100
  evaluations - deliberately conservative; operator tunes up).
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
  Type-narrowing, not breaking - anyone reading `cfg.hooks` and
  getting `None` today now gets a populated config.
- **`CrossToolUnifier.unify()` is now async.** Required to dispatch
  the new `ENTITY_MERGED` event. Both production call sites
  (`SemanticExpander.expand`, `pipelines/flows/expansion.unify_entities`)
  updated; the 5 existing unifier tests converted to async.

## [0.11.0] - Temporal retrieval overhaul: scoring fixes + ingestion contract + entity-anchored fast path

Three-phase rework of khora's temporal retrieval, addressing the production
complaint that queries like *"what are the latest action items from recent
meetings?"* returned stale records on corporate-data corpora (Slack, email,
calendar, Salesforce). All new behavior is **feature-flagged off by
default**; operators opt in per-namespace via `KHORA_QUERY_TEMPORAL_*` env
vars. Existing consumers see no behavior change unless they enable flags.

Closes [#567](https://github.com/DeytaHQ/khora/issues/567) (Phase A - scoring),
[#568](https://github.com/DeytaHQ/khora/issues/568) (Phase B - ingestion),
[#569](https://github.com/DeytaHQ/khora/issues/569) (Phase C - entity-anchored).

### Added

- **Phase A: scoring & retrieval (#567 / PR #571).**
  - `_calculate_recency_scores(reference_mode="wall_clock" | "relative")` - wall-clock
    is production-correct (was a relative max-in-set heuristic). `KHORA_BENCH_MODE=true`
    forces `relative` for benchmark replay.
  - Synthetic date floor for RECENCY/CHANGE queries with no parseable date
    (e.g., bare "latest", "recent"). Defaults: RECENCY=30d, CHANGE=60d.
  - `ANTI_RECENCY_TOKENS` veto list - historical / counterfactual queries
    ("ever", "history of", "would have", "if we had", "back when", "at one
    point", etc.) suppress the synthesized floor.
  - **LLM disambiguation tier** - when Aho-Corasick fires RECENCY/CHANGE
    AND the query contains an ambiguity-trigger token (`would`, `could`,
    `if `, `previously`, etc.), a short LLM call classifies the query as
    RECENT / HISTORICAL / COUNTERFACTUAL / NEUTRAL. Floor vetoed for
    non-RECENT outputs. Per-query cache; bounded cost.
  - Parallel "recency channel" - pure `ORDER BY occurred_at DESC LIMIT N`
    SQL fused via RRF pool augmentation. Cosine relevance floor (0.40
    default) gates which fresh-but-irrelevant chunks can enter.
  - Per-source decay dict - Slack 3d / email 7d / calendar 14d / Salesforce
    180d / `_default` 14d. Looked up via `chunk.metadata.custom["source_system"]`.
  - pgvector `hnsw.iterative_scan = strict_order` capability-probed and
    enabled when a temporal filter is set (avoids HNSW recall collapse
    under selective filters).
  - New public exports: `TemporalIntent`, `classify_temporal_intent_llm`,
    `has_anti_recency_token`, `has_ambiguity_trigger`, `ANTI_RECENCY_TOKENS`.
  - Seven new `QuerySettings.temporal_*` knobs gating each behavior.

- **Phase B: ingestion contract (#568 / PR #573).**
  - `khora.pipelines.ConnectorMetadata` - public `TypedDict(total=False)`
    documenting the canonical metadata-field surface for connector authors.
  - `khora.pipelines.SourceSystem` - `Literal["slack","email","calendar","salesforce","jira","linear","manual"]`.
  - `khora.pipelines.validate_connector_metadata(metadata, source_type) -> list[str]` -
    advisory warnings; connector CI runs it pre-ingest.
  - `khora.pipelines.CANONICAL_TIMESTAMP_FIELDS` - tuple matching the extractor priority.
  - Source-type-aware timestamp priority: `source_type in {calendar,meeting,event}`
    now prefers `occurred_at` over `sent_at`.
  - New span attribute `chunk.occurred_at.source = "metadata" | "ingest_fallback"`
    on chunk construction. Operators can detect silent connector breakage.
  - New metric `khora.ingest.source_timestamp.fallback_count` - counter,
    bounded `source_type` label. Throttled WARN log when a canonical-source
    connector misses the timestamp.
  - `docs/extraction/ingestion-pipeline.md` § "Canonical metadata fields
    per source" - full mapping table per source system.

- **Phase C: entity-anchored fast path (#569 / PR #574).**
  - New `QueryComplexity.TYPED_ENTITY_RECENT` routes queries matching
    `(latest|most recent|newest|recent) (action items|decisions|blockers|risks|...)`
    through a single Cypher fast path.
  - New retriever method `_typed_entity_recent_retrieve()` - single
    `MATCH ... MENTIONED_IN ... max(c.occurred_at) ORDER BY DESC` query
    with status filter for ACTION_ITEM / COMMITMENT / OPEN_QUESTION
    (excludes done / cancelled / completed / closed). Graceful fallback
    when no typed entities exist or Neo4j is unavailable.
  - New opt-in extraction skill `builtin:meetings` with 4 typed entity
    types: **ACTION_ITEM** (assignee, due_by, status), **DECISION**
    (decided_on, rationale), **BLOCKER** (blocking_for, severity), **RISK**
    (likelihood, impact). High-precision prompt with anti-fabrication
    guardrails.
  - Composite index migration `028_typed_entity_recency_index` -
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

## [0.10.8] - OTel-first telemetry, vanilla OpenTelemetry SDK extras, observability docs

Closes [#564](https://github.com/DeytaHQ/khora/issues/564) - make khora's
observability stack vendor-neutral. Logfire moves from "the only path"
to "one of several supported backends." Backward-compatible for
existing `khora[logfire]` users.

### Added

- **`pip install khora[otel]`** - pulls `opentelemetry-sdk` +
  `opentelemetry-exporter-otlp-proto-http`. Honors the standard
  `OTEL_*` env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`,
  `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_HEADERS`,
  `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`,
  `OTEL_TRACES_SAMPLER`, etc.). Ships spans/metrics to any
  OTLP-compatible collector or vendor (Tempo, Jaeger, Honeycomb,
  Datadog, New Relic, Dynatrace).
- **`pip install khora[otel-grpc]`** - composes `khora[otel]` with
  the gRPC OTLP exporter for sites that prefer Bolt-style transport.
- **`khora.telemetry.configure_telemetry()`** - single in-code entry
  point with precedence: caller-supplied providers → existing host
  provider → `LOGFIRE_TOKEN` env → `OTEL_*` env → no-op. Idempotent;
  returns a `TelemetryHandle` for inspection and explicit shutdown.
- **`khora.telemetry.diagnostics()`** - prints active provider class,
  endpoint, contract version, and the OTel env. Run this first when
  spans appear to be missing.
- **`khora.telemetry.shutdown_telemetry_providers()`** - force-flush
  and shutdown only the providers khora installed; host-owned
  providers are left untouched.
- **Resource attribute `khora.telemetry.contract.version`** - exported
  alongside SDK defaults so dashboards can filter by telemetry-schema
  version independently of the package version.
- **`docs/observability.md`** - single canonical observability page.
  Covers install paths, env-var contract, programmatic config,
  precedence, sampling/cost guidance, vendor recipes (Honeycomb,
  Grafana Cloud, Datadog, local Jaeger), and the "I see no spans"
  troubleshooting checklist.
- **`tests/unit/telemetry/test_otel_parity.py`** - vanilla-OTel parity
  gate. Runs against `InMemorySpanExporter` + `InMemoryMetricReader`
  with no logfire path and asserts every public span/metric is emitted
  through the OTel API.
- **`scripts/bench_telemetry_overhead.py`** - regression gate for the
  no-provider trace/decorator hot path; current baseline ~3 µs/call,
  budget 5 µs/call.
- **`tests/integration/test_otel_smoke.py`** - binds an in-process
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
  context manager - measured, well under any meaningful threshold).
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

- **`src/khora/telemetry/logfire_integration.py`** - replaced by
  `_otel.py` (tracer/meter + `trace_span` + `install_neo4j_log_bridge`),
  `_attrs.py` (`bounded_text_hash`), and `bootstrap.py`
  (`configure_telemetry` and friends). All call sites updated.
- **The `Span` / `LogfireSpan` / `NoOpSpan` ABC hierarchy.** Callers
  now use `opentelemetry.trace.Span` directly. The migration is
  source-compatible - the `.set_attribute` / `.set_attributes` shape
  is identical.
- **"SurrealDB Phase 1" status note** in
  `docs/architecture/storage-backends.md` - the backend has been
  feature-complete since the 2026-03-25 audit.
- **Last user-facing `kuzu` mention** in
  `docs/engines/engine-comparison.md`. The `kuzu` extra still ships
  for downstream callers but is no longer advertised in docs.

### Deprecated

- **`khora.telemetry.install_neo4j_logfire_handler`** - renamed to
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

## [0.10.7] - PyPI README rewriting, SecretStr config fields, release-pipeline fixes

### Changed

- **Credential and DSN config fields re-typed as `pydantic.SecretStr`** (#553). Affects `KhoraConfig.storage.*` connection URLs/passwords (Postgres, Neo4j, Memgraph, Neptune, AGE, SurrealDB graph/relational/vector/event-store) and the telemetry collector DSN. Each backend's engine/driver factory unwraps the secret exactly once at the driver edge via the new `khora.config._secrets._secret_value()` helper. Config dumps and log lines now render these fields as `'**********'`; values written to fields still accept `str` for back-compat, but reads return `SecretStr` - callers that previously did `str(cfg.storage.neo4j.password)` and expected the cleartext now need `.get_secret_value()`. Pre-1.0 patch; flagging it explicitly here so downstream consumers can audit.
- **README rendering on PyPI** (#556). The on-disk README keeps relative links (`docs/configuration.md`) for GitHub readers; `hatch-fancy-pypi-readme` substitution rewrites them to `https://github.com/DeytaHQ/khora/blob/main/...` at wheel/sdist build time so PyPI's project page renders working links. Also adds `[project.urls]` for the PyPI sidebar (Documentation, Source, Issues, Changelog, Releases).

### Fixed

- **`release.yml` smoke-install** (#552). The smoke-install step installed `khora[sqlite-lance]==${VERSION}` and then asserted `pip show khora-accel`, which always failed because `[sqlite-lance]` doesn't pull khora-accel - that's the `[rust]` extra. Tripped the gate on every release since the assertion landed; v0.10.6 had its `github-release` job skipped because of it (workaround: manual `gh release create`). Now installs `khora[sqlite-lance,rust]` so both wheels are verified together.

### OSS prep

- **`CLAUDE.md` scrub** (#555) - final pass for OSS readiness.

## [0.10.6] - Test-infra expansion, dependency staging window, docs cleanup

### Added

- **Embedded test footprint** (PR-A / #536) - matrix tests for the SQLite+LanceDB stack covering VectorCypher, Skeleton, and Chronicle.
- **Embedded backend hardening** (PR-B / #537) - typed `EmbeddingError` validation in `khora.storage.backends.sqlite_lance.vector` so malformed inputs surface at the boundary instead of failing deep inside LanceDB.
- **Property-based tests** (PR-C / #538) - Hypothesis tests pinning Chronicle abstention `combined_score`, FTS5 escape parseability, MMR λ-direction (khora's λ=1 ⇒ pure relevance convention), and SQLite FTS5 `bm25()` sign handling.
- **Test infrastructure** (PR-D / #540) - `embedded` pytest marker, `scripts/check_coverage_floors.py` per-path coverage gate wired into CI, `make test-embedded` / `make test-soak` targets, and a top-level `CONTRIBUTING.md`.
- **Codecov split flags** (#539) - unit and integration uploads land separately so PRs that only touch unit-tested paths still get a Codecov diff comment. `codecov.yml` carries auto-target project status with sensible thresholds.
- **`uv exclude-newer = "7 days"` policy** (#548) - `uv lock` ignores PyPI releases uploaded within the last 7 days, re-evaluated every sync. `exclude-newer-package = { urllib3 = "0 days" }` lets same-week CVE fixes through; pattern reusable for future security-critical updates.

### Changed

- **Docs cleanup** - README, `docs/configuration.md`, `docs/README.md`, and `docs/architecture/{overview,storage-backends}.md` no longer reference the deprecated `kuzu` extra (#550). README sibling-package list trimmed; `khora-service` linked to its GitHub repo (#531). README description language: "library, not an application; tooling lives in sibling packages."
- **Docstring scrub** (#547) - `LiteLLMConfig.max_total_connections` description now uses vendor-neutral phrasing ("typical high-throughput ingestion concurrency"). Internal-service name removed from public API surface.
- **Release pipeline** (#532) - `release.yml` auto-creates GitHub releases on tag push with auto-generated notes from merged PRs.

### Fixed

- **`find_related_entities` fallback** (#535) - VectorCypher's graph-only fallback called a non-existent backend method on `sqlite_lance` / `surrealdb`. Falls back gracefully now.

### Removed

- **Memory Lake branding residue** (#534) - final pass on examples, tests, and inline strings. No public API changes.

### Infrastructure

- **kuzu still ships as an optional extra**, but is no longer advertised in user-facing docs (the upstream repository was archived after Apple's acquisition in October 2025).

## [0.10.5] - FTS5 escape, Chronicle sqlite_lance persistence, loguru placeholders

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
- PyPI long description: "Knowledge memory library for long-horizon AI agents - hybrid retrieval over documents, embeddings, and graph relationships."
- Sibling packages on PyPI: `khora-cli`, `khora-explorer`, `khora-service` (coming soon).

## [0.10.4] - First clean PyPI release after the migration

### Fixed

- **Lockstep version computation** (PR #527). The release pipeline sed'd `pyproject.toml` on the runner before `python -m build`, leaving the working tree dirty. `setuptools_scm` (via hatch-vcs) treated `dirty` at a tag as "ahead of tag" → bumped the patch → produced `0.10.4.dev0` instead of `0.10.3`. Removed the runtime sed; the lockstep `khora-accel == X.Y.Z` pin in `pyproject.toml`'s `rust` extra is now committed alongside the `khora-accel/Cargo.toml` version bump.

### Status

- First post-migration release where **both** `khora` and `khora-accel` wheels published cleanly to PyPI at the same version with the lockstep contract verified end-to-end (wheel METADATA contains `Requires-Dist: khora-accel==0.10.4`).
- Updated `CLAUDE.md` → Version Bumps to require updating all three files (`Cargo.toml`, `Cargo.lock`, `pyproject.toml` pin) in the same PR.

## [0.10.3] - Partial release (khora-accel only)

### Fixed

- **`actions/checkout@v6` did not fetch tag refs** (PR #525). `fetch-depth: 0` controls history depth, not tags; without `fetch-tags: true`, `git describe` saw no tag on the runner and `hatch-vcs` fell back to "next-dev". The v0.10.2 release published khora as `0.10.3.dev0` for this reason. Added `fetch-tags: true` to the khora checkout step; added `skip-existing: true` to both PyPI publish steps for safer re-runs.

### Status

- `khora-accel 0.10.3` published to PyPI; `khora 0.10.3` was **not** published (the lockstep-sed bug fixed in 0.10.4 produced `0.10.4.dev0` for khora). Use 0.10.4+ for the matched-pair install.

## [0.10.2] - Publishing migrated to PyPI

### Changed

- **Publishing target**: moved from AWS CodeArtifact to **public PyPI** under the Deyta organization (PR #524). Uses PyPI Trusted Publishing via GitHub OIDC - no API tokens, no AWS, no secrets in the repo. `pypa/gh-action-pypi-publish@release/v1` with an environment-bound trusted publisher per project.
- **khora-accel** now ships as an **sdist only** (no platform-wheel matrix). Users compile the Rust extension at install time via maturin's PEP 517 backend; requires a Rust toolchain (`rustup`) on the install host.
- **Version lockstep**: khora and khora-accel are always released at identical versions. The published khora wheel pins `khora-accel == X.Y.Z` exact.
- **Publish order**: serialized `publish-accel → publish-khora` so khora's wheel can only land on PyPI if accel is already resolvable.
- `ci.yml` dev-publish jobs removed; only tag pushes publish.

### Status

- `khora-accel 0.10.2` published to PyPI; `khora 0.10.2` was **not** published (the `fetch-tags` bug fixed in 0.10.3 produced `0.10.3.dev0` for khora). Use 0.10.4+ for the matched-pair install.

## [0.10.1] - Remove `graphrag` engine

### Removed - BREAKING

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

## [0.10.0] - Rename `MemoryLake` → `Khora`, drop "Memory Lake" branding

### Changed - BREAKING

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
  `khora.memory.ingest.duration`) - generic concept of memory storage,
  not the retired brand. Operator dashboards keep working.
- `khora_alembic_version` table name and advisory lock id
  `6001515088189075507` - schema continuity.
- SurrealDB `memory_namespace` table name - unrelated to the brand.
- `KHORA_` env-var prefix.

### Migration

```diff
- from khora import MemoryLake
+ from khora import Khora

- async with MemoryLake(config) as lake:
+ async with Khora(config) as lake:
      ...
```


## [Unreleased] - Telemetry Public Surface, OSS Observability Contract

Telemetry workstream (PRs #504–#509) shipped after the v0.9.1 tag. It hardens cardinality safety, codifies the public observability surface as a JSON contract enforced by a CI drift gate, fixes a silent regression that had been zeroing out `storage_events.namespace_id` since February 2026, and broadens metric coverage. The OSS implication: public telemetry names are now API and break the same way any other public symbol does.

### Added

- **Public observability contract.** `docs/telemetry-contract.json` lists every public export in `khora.telemetry.__all__` (19 names), every `LLMEvent` / `StorageEvent` / `PipelineEvent` field, all 22 collector-recorded pipeline stages, all 58 `trace_span(...)` call sites (22 public, 36 internal), and all 21 metrics (16 public, 5 internal). `docs/telemetry-contract.md` is the human-facing explainer. `tests/unit/telemetry/test_contract.py` (10-test drift gate) walks the codebase via ripgrep and fails CI on any undeclared instrumentation. (#505)
- **`khora.telemetry.bounded_text_hash`.** Helper that turns free-text span attributes (raw query, document content, chunk text) into a SHA1[:8] hash - caps cardinality and removes the privacy hazard of raw text on spans. Now used at the four query / extraction sites that previously emitted raw text. (#504)
- **Chronicle abstention metrics.** `khora.chronicle.abstention_signal` (counter, public) and `khora.chronicle.abstention_combined_score` (histogram, public) aggregate the four boolean abstention signals + combined score that `RecallResult.metadata["abstention_signals"]` exposes per call, so abstention rate and confidence distribution can be tracked at fleet scale instead of only inspected per-request. (#507)
- **Aggregate operator metrics.** `khora.memory.recall.duration` (histogram, public, seconds), `khora.memory.ingest.duration` (histogram, public, seconds), `khora.llm.tokens` (counter, public), `khora.llm.cost_usd` (counter, public), `khora.log.queue.depth` (gauge, public, proxy via handler-error count - loguru 0.7.3 does not expose `qsize()`). (#509)
- **Six additional LLM call sites instrumented.** HyDE, listwise rerank, fact extraction, fact reconciliation, event extraction now record `LLMEvent` rows; chat was already wired. Two patterns coexist (`_telemetry_op="..."` through `khora.config.llm.acompletion` vs. inline `record_llm_call` after direct litellm calls); both are documented in `CLAUDE.md`. (#508)

### Fixed

- **`storage_events.namespace_id` 100% NULL since Feb 2026.** Restored namespace propagation through the storage telemetry path. The break had survived multiple releases because no operator dashboard was reading the column - surfaced during the Phase-0 audit. (#506)

### OSS implication

- Names tagged `public` in `docs/telemetry-contract.json` are now part of khora's public API. Renames or removals require a major version bump and prior coordination with genesis, khora-benchmarks, khora-explorer, and khora-cli. Names tagged `internal` (e.g. inner-loop spans like `khora.vectorcypher.coherence_boost`) may be renamed freely as long as the JSON is updated in the same PR.
- New attributes follow OTel semantic conventions: `gen_ai.*` for LLM, `db.*` for storage, `code.*` for stack info.
- The contract enables the operator-dashboard work that follows; it does not by itself fix the under-utilisation. Telemetry has been collected to PostgreSQL since 0.4.0, and dashboards / alerts that consume those events remain TODO.

---

## [0.9.0] - 2026-05-02 - Embedded Backend Realignment, Production-Readiness Scoping

### Embedded backend overhaul

The v0.9.0 embedded path lands as a complete-but-experimental SQLite + LanceDB stack covering all four engines (VectorCypher, GraphRAG, Skeleton, Chronicle). Engine × embedded integration tests now exist for all four engines; the prior "unverified embedded code path" gap from the audit is closed.

**Production-readiness scoping (per stack, not per engine).** Stamping is now per `(engine × storage stack)`:

- **VectorCypher** - production-ready on **PostgreSQL + pgvector + Neo4j** only.
- **Chronicle** - production-ready on **PostgreSQL + pgvector** (no graph DB required).
- **GraphRAG** and **Skeleton** - available; same PG-based stacks.
- **SQLite + LanceDB** for any engine - **experimental**. Documented scale ceiling: ~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3.
- **SurrealDB** for any engine - **experimental**. Python SDK on alpha track (`>=2.0.0a1`); KNN unreliable in embedded mode (brute-force cosine + HNSW fallback).

See [docs/engines/engine-comparison.md](docs/engines/engine-comparison.md#production-readiness-by-stack-v090) for the full matrix.

### Embedded engine wiring

- (#482): VectorCypher wired to the `sqlite_lance` backend.
- (#481): Skeleton wired to the `sqlite_lance` backend with a temporal-store adapter.
- GraphRAG embedded path pushes the temporal filter into the LanceDB WHERE - was previously post-hoc and xfail-pinned.
- (#486): Temporal filter pushed into SQLite-side WHERE in the GraphRAG embedded chunk fetch path.
- VectorCypher honours `metadata['occurred_at']` on the embedded path (parity with `remember_batch`).

### Embedded retrieval correctness

- Chronicle channels (BM25 / semantic / temporal / entity) now share the same `created_after`/`created_before` bounds - fixes channel divergence that broke RRF fusion.
- Recursive-CTE graph traversal switched from node-visited to edge-visited tracking (mirrors Neo4j `MATCH [*1..N]`).
- `valid_until > now` filter inlined into both anchor and recursive arms of the CTE.
- Skeleton tag-cast and `occurred_at` parsing fixes (`Skeleton.remember()` parity with `remember_batch()`).
- Embedded compensating-delete-on-failure logging hardened.
- (#485): LanceDB IVF-PQ index now retrains once the corpus grows past `retrain_factor × (rows at last training)`. Configurable via `KHORA_STORAGE_SQLITE_LANCE__RETRAIN_FACTOR` (default `2.0`). Fixes silent recall degradation as the corpus grows past the initial training threshold (5k rows). Set ≤ `1.0` to disable.

### Embedded warts (documented, not fixed)

- **Partial atomicity in `coordinator.transaction()`** on embedded - only the SQL session is enrolled; LanceDB writes happen post-commit with compensating deletes.
- **Point-in-time queries** are not supported on the embedded stack. The CTE port does not implement PIT semantics. Tracked.
- **FTS5 on chunks only** - entity-anchored recall falls back to `LIKE` / JSON-equality on embedded. Use the PostgreSQL stack for entity-heavy corpora.

### Deprecated

- **Kuzu graph backend** (`khora[kuzu]`) - deprecated in 0.9.0, scheduled for removal in 0.10. Kuzu was acquired by Apple in October 2025 and the upstream repository is archived. Migrate to SQLite + LanceDB (embedded) or PostgreSQL + Neo4j (production).

### v0.10 roadmap

Two deferred decisions for v0.10 address the embedded warts:

- **sqlite-vec** as a candidate to collapse the SQLite + LanceDB dual-store into a single in-SQLite-transaction vector store (eliminates partial atomicity, drops install footprint from ~150 MB to ~5 MB).
- **`pgserver` (embedded Postgres)** as a candidate for true production-parity embedded mode (HNSW recall, real ACID, zero schema fork).
- **Default embedded URI routing** - currently `MemoryLake("memory://")` treats the URL as the PostgreSQL `database_url`; SurrealDB owns the `memory://` scheme internally. Routing a top-level `memory://` URI to the recommended embedded stack is a v0.10 code change.
- **`lance-graph` integration** is explicitly **deferred to v0.10** - no second 0.x Rust crate enters a "production-ready" path in v0.9.0.

---

## [Unreleased] - Graph Backends, Temporal Precision, Discovery Agent Overhaul

### Added
- Codified the khora public API surface consumed by downstream packages (genesis, khora-benchmarks, khora-explorer, khora-cli).

### Removed
- `khora` console script and CLI subcommands (`extract`, `search`) - moved to khora-cli. Install with `uv pip install khora-cli` and run `uv run khora-cli extract` / `uv run khora-cli search`.
- `khora ontology` CLI subcommands (moved to khora-explorer)
- `khora.discovery` package (moved to khora-explorer)
- `khora.cli` package (entire subtree - `extract`, `search`, `_common`)
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
- Cross-encoder model caching - avoid reloading per query (#270, #340)
- asyncio.to_thread for reranker inference (#317)
- Column projection excludes embeddings from search results (#317)
- ef_search at connection level (#317)
- Parallel Chronicle channels (#301)
- CI parallel jobs - 50% faster feedback (#335)

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

## [0.7.0] - 2026-04-02 - Chronicle Engine, Semantic Hooks, Retrieval Quality

### Chronicle engine (new)

- 4th memory engine optimized for temporal/conversational memory (LongMemEval, LoCoMo, BEAM) (#199, #200)
- 4-channel parallel retrieval: semantic + BM25 + temporal decay + entity co-occurrence
- Ebbinghaus forgetting curve for temporal decay scoring
- Event decomposition into SVO tuples with triple timestamps (observation, referenced, relative)
- Progressive memory compression with contradiction detection (ADD/UPDATE/DELETE/NOOP)
- LanceDB embedded vector store option - file-backed, no server (`pip install khora[lancedb]`)
- Rust-accelerated temporal scoring via khora-accel
- No graph database required - PostgreSQL + pgvector only

### Semantic hooks

- Event subscription system: `lake.subscribe("entity.created", callback)` (#193)
- `SemanticFilter` with 3-level cascade: type pre-filter (free) → embedding similarity (sub-ms) → LLM yes/no (#194)
- `HookDispatcher` with async concurrent dispatch and failure isolation
- Binary-quantized embedding cache for sub-microsecond pre-screening (Hamming distance)
- Configurable via `KHORA_HOOKS_ENABLED`, `KHORA_HOOKS_FILTER_MODEL` env vars
- Wired into ingestion pipeline - fires during entity/relationship extraction

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

- Fix `run_migrations()` on fresh PostgreSQL database - use `information_schema.tables` (#201)
- Reduce extraction batch size from 10 to 5 and make configurable (#195)
- Move per-document extraction log lines from INFO to DEBUG (#196)

### Version

- khora-accel 0.7.0, Rust edition 2024 (#209)

---

## [0.6.0] - 2026-03-28 - SurrealDB Optimization, Ontology CLI, Discovery Agent

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

- `khora ontology construct --source <path>` - AI-powered ontology construction from data (#138, #140, #141)
- `khora ontology validate <file>` - schema + reference integrity validation (#138)
- `khora ontology preview <file>` - Rich table + tree display (#138)
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

## [0.5.5] - 2026-03-26 - Ontology CLI & SurrealDB Hardening

First release with the ontology construction CLI and comprehensive SurrealDB
optimization audit. Includes the entity key gate, SDK upgrade to 2.0.0a1,
schema parity fixes, and 15+ SQL injection fixes. See 0.6.0 for the detailed
breakdown (0.5.5 was the last tagged release before the 0.6.0 cycle).

---

## [0.5.4] - 2026-03-25 - Test Audit & CI Fixes

- Replace `__slots__` implementation-detail tests with behavioral checks (#128)
- Fix publish-accel uv cache failure (#129)
- Gitignore `.agents/` folder (#126)

---

## [0.5.3] - 2026-03-24 - macOS Build Fix

- Remove x86_64-apple-darwin from macOS build matrix (#125)

---

## [0.5.2] - 2026-03-24 - Release Pipeline Consolidation

- Consolidate khora and khora-accel into single release pipeline (#124)
- Fix accel build matrix and sccache configuration (#124)

---

## [0.5.1] - 2026-03-24 - First Tagged Release

First release with git-tag-based versioning via `hatch-vcs`. Includes all
features from 0.4.0 and 0.5.0 internal versions.

- Add `publish.yml` and `publish-accel.yml` CodeArtifact workflows (#123)
- Switch to `hatch-vcs` for version derivation from git tags
- Add sccache for Rust build acceleration
- Add `docs/RELEASE.md` with release process documentation

---

## [0.5.0] - SurrealDB Unified Backend & Engine Modernization

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

- Deprecate `create_tables()`/`init_db()` - use `run_migrations()` (#98)
- Migration drift CI test (`test_migration_drift.py`)
- `khora_alembic_version` dedicated version table
- Advisory lock for concurrent migration safety
- Temporal expression index for query performance (#117)

### Other

- Neo4j connection lifetime and liveness config (#102)
- Fix conversation-mode entity extraction regression (#122)
- Widen `extraction_config_hash` to VARCHAR(255) (#99)

---

## [0.4.0] - Logfire Telemetry, Namespace Versioning, Alembic Overhaul

### Logfire / OTEL integration

- Optional `logfire` integration for distributed tracing (#32)
- `trace_span()` context manager and `@trace` decorator
- `_HAS_LOGFIRE` feature flag - zero-cost no-op when absent
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

- Remove FastAPI dependency - Khora is a library, not a web app (#35)

### Other

- Fix Alembic migrations on fresh databases (#37)
- Fix UUID `as_uuid=True` across all 52 columns (migration 006)
- Add `document_status` enum sync (migration 014)
- Temporal coalesce expression index (migration 017)

---

## [0.3.10] - Chunker Safety & Rust Performance

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

## [0.3.9] - Key-Aware Neo4j Write Coordination

### Why: overlapping MERGE batches caused Neo4j lock contention

The entity write path used a plain semaphore (concurrency 12) to limit concurrent
Neo4j transactions. This prevented connection exhaustion but allowed overlapping
`MERGE` transactions - two batches touching the same entity key would run
concurrently, causing Neo4j to detect lock contention, abort one transaction, and
retry with ~1 s exponential backoff. Under heavy ingestion this cascaded into
minutes of wasted retries.

### `_EntityKeyGate` replaces entity write semaphore

New `_EntityKeyGate` class (`storage/backends/neo4j.py`) tracks in-flight entity
keys - `(namespace_id, name, entity_type)`, the same triple used in the Cypher
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

## [0.3.8] - Temporal Search Improvements

### Why: temporal queries silently degraded to generic search

When the LLM timed out (2s budget), the heuristic fallback detected temporal
*intent* but produced `start_date=None, end_date=None`, so no temporal filter
was applied. "What happened last week?" returned the same results as "What
happened?". Additionally, temporal filtering happened post-retrieval in Stage 3
(Python-side soft scoring on 200 candidates) instead of as SQL WHERE clauses in
Stage 1, wasting retrieval budget. Chunks also lacked source timestamps - a
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

- `009_temporal_search_indexes` - temporal indexes, `source_timestamp` columns

---

## [0.3.7] - Stability Fixes

### Fixed

- Cap Neo4j write concurrency and bound provenance list growth to prevent OOM
  under high-volume ingestion (#28)
- Remove invalid `IF EXISTS` from `REINDEX` command that caused PostgreSQL
  errors on older versions (#27)

### Added

- TTOJ team profile templates (#29)

---

## [0.3.6] - VectorCypher Entity Search Fix

### Fixed

- VectorCypher entity search called a non-existent coordinator method,
  causing `AttributeError` on entity-heavy queries (#26)

---

## [0.3.5] - Phase 3 Benchmark Optimizations

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

## [0.3.4] - ty Type Checker Clean

- Resolve all remaining `ty` diagnostics - `ty check src/` now passes with
  zero warnings.
- Version bump 0.3.3 → 0.3.4.

---

## [0.3.3] - Neo4j Deadlock Fixes

- Shared semaphore for Neo4j relationship writes to prevent deadlocks during
  concurrent batch ingestion.
- Tune Neo4j driver parameters (`max_transaction_retry_time`,
  `connection_acquisition_timeout`) to reduce transaction deadlock retries.
- Version bump 0.3.2 → 0.3.3.

---

## [0.3.2] - Phase 2 Benchmark Optimizations

- Restore parallel Neo4j writes and reduce relationship volume.
- Add co-occurrence edges, lazy entity expansion, skeleton skip, and
  concurrency alignment.
- Phase 2 benchmark optimizations for improved ingestion throughput.
- Version bump 0.3.1 → 0.3.2.

---

## [0.3.1] - Benchmark-Driven Optimizations

### Why: restoring incremental MRR and improving retrieval quality

Benchmark run `2f7d4b0b` revealed that incremental ingestion (add
documents in multiple batches) produced an MRR of 0.0 - newly ingested
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

- `007_hnsw_parameter_tuning` - HNSW m=24, ef_construction=128
- `008_entity_dedup_and_indexes` - Entity dedup + unique constraint,
  khora_chunks composite index, entity temporal partial indexes

---

## [0.3.0] - Engineering Improvements

### Why: removing accidental complexity

Global state in database session management, UUID string wrapping across
52 ORM columns, redundant connection pools for backends sharing the same
database URL, and stale deprecated APIs that no longer matched the
codebase - none of these served users, and all of them created friction
for contributors. This release removes the accidental complexity so the
next round of features lands on cleaner ground.

### UUID migration

All 52 UUID columns in `db/models.py` now declare `as_uuid=True`,
mapping to native Python `uuid.UUID` objects. This is a Python-side-only
change - the PostgreSQL column type remains `UUID`. The practical effect
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

- `lake.storage` - promoted to stable public API (used by `genesis` and
  `khora-benchmarks`). The deprecation warning has been removed.
- `lake.query_engine` - removed. Use `lake.recall(raw=True)` for
  unprocessed search results.
- `remember_batch_legacy()` - removed. Use `remember_batch()`.

### Chat module tests

71 new tests across 4 files covering the chat module (`chat/engine.py`,
`chat/history.py`, `chat/persona.py`, `chat/prompt.py`). The module
itself is unchanged - these tests document and lock existing behavior.

### spaCy sentence splitting

The semantic chunker now uses spaCy's `sentencizer` component when
available, improving sentence boundary detection. Install with
`pip install khora[nlp]`. The sentencizer is a rule-based component
that ships with spaCy core - no model download needed. When spaCy is
not installed, the chunker falls back to its existing regex-based
splitter transparently.

### Docker removal

The `Dockerfile` and CI `docker-build` job have been removed. Khora is
a library, not a deployable application - the Dockerfile was never used
in production and added maintenance burden. Development databases
continue to use `compose.yaml` via `make dev`.

### Housekeeping

- Version bumped from 0.2.3 to 0.3.0 in `pyproject.toml`,
  `src/khora/__init__.py`, `rust/khora-accel/Cargo.toml`, and
  `rust/khora-accel/pyproject.toml`.

---

## [0.2.3] - Namespace Optimization Design

### Why: surfacing what's real vs. what's aspirational

A team of five specialist agents audited Khora's namespace isolation,
multi-tenancy enforcement, and temporal extraction paths. The audit
found that several documented features - `TenancyMode` routing, ACL
enforcement, bi-temporal edge storage, and the time hierarchy builder -
exist as code but are never exercised at runtime. Meanwhile, the
namespace-level row filtering that *is* active lacks an orphan-entity
cleanup path when documents are deleted. This release ships the
comprehensive design for fixing all of it, marks the stale
documentation, and inventories the dead code so the next releases can
act on it.

### Namespace optimization design

New `docs/design/namespace-optimization-plan.md` lays out a six-phase
implementation roadmap:

1. **Orphan fix** - delete graph entities left behind after `forget()`.
2. **Data-model hardening** - add `namespace_id` to Neo4j entity/chunk
   nodes and enforce it in Cypher queries.
3. **Isolated-mode core** - per-org connection routing driven by
   `TenancyMode.ISOLATED`.
4. **Shared-mode ACL** - wire `ACLEnforcer` into the API dependency
   chain for `TenancyMode.SHARED`.
5. **ACL enforcement** - row-level security policies and graph-side
   namespace filtering.
6. **Rust acceleration** - move hot-path namespace filtering into
   `khora-accel`.

### Dead-code inventory

- `TenancyMode` enum (`core/models/tenancy.py`) is defined but never
  checked at runtime - all orgs use implicit shared mode.
- `ACLEnforcer` and `ACLContext` (`acl/`) are importable but the API
  dependency in `api/deps.py` is disabled.
- `TemporalEdgeStorage` and `TimeHierarchyBuilder` (`engines/skeleton/`)
  exist as modules but are never called by any engine's ingest or recall
  paths. The `occurred_at` column on chunks works through the pgvector
  backend directly.

### Stale documentation fixes

Added status notices to five documentation files flagging features that
are designed but not yet wired:

- `docs/architecture/multi-tenancy.md` - TenancyMode and ACL sections.
- `docs/engines/temporal-model.md` - bi-temporal edge model.
- `docs/engines/skeleton-engine.md` - architecture diagram components.
- `README.md` - multi-tenancy feature bullet.
- `docs/architecture/overview.md` - ACL enforcer mention.

### Housekeeping

- Bumped version from 0.2.2 to 0.2.3.

---

## [0.2.2] - VectorCypher Optimization

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
set without adding signal - so the retriever now drops to depth 1.
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

**`VectorCypherConfig` dataclass.** All VectorCypher-specific knobs -
routing, skeleton indexing, graph traversal, fusion weights, temporal
settings, and search thresholds - live in a single dataclass that can
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

## [0.2.1] - Concurrency & Throughput

### Why: filling the gap Rust opened

Version 0.2.0 moved CPU-bound work (similarity scoring, PageRank, BM25
indexing) off the Python event loop and into native Rust threads. The
immediate effect was that CPU cycles were no longer the bottleneck during
large ingestion runs - network I/O to LLM and embedding providers was.
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

**Ingestion pipeline.** The ingestion flow - Khora's primary data path -
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

## [0.2.0] - Rust Acceleration Layer

### The problem

Profiling large ingestion runs showed that CPU-bound operations -
cosine similarity over dense embedding matrices, edit-distance
computations during entity resolution, PageRank convergence over chunk
graphs, and BM25 scoring - dominated wall-clock time once documents
were chunked and LLM calls returned. Python's GIL serialized these
hot loops, and even NumPy could not parallelize the non-BLAS workloads
(string comparisons, graph iteration, inverted-index lookups).

### The approach

Khora 0.2.0 introduces `khora-accel`, a Rust extension built with
PyO3 and maturin. The design philosophy is **zero mandatory
dependencies**: a three-tier fallback (`_accel.py`) checks for the
Rust extension first, then NumPy/RapidFuzz, then pure Python. Every
accelerated function is a drop-in replacement - the Python signature
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
GIL and run pure Rust graph iteration - adjacency-list storage,
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
- `engines/skeleton/skeleton.py` - PageRank, chunk edges, keywords, BM25
- `engines/vectorcypher/fusion.py` - RRF, weighted RRF, score normalization
- `query/engine.py` - cosine similarity, BM25 search
- `extraction/entity_resolution.py` - batch entity resolution
- `storage/` and `pipelines/` - embedding similarity, string matching

The active backend is logged at import time for observability.

### Other changes

- Improved upsert result mismatch diagnostics.
- Downgraded extraction log to debug level.
