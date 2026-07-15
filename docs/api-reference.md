# API Reference

The public Khora surface is pinned by the machine-readable `__all__` in `src/khora/__init__.py`. The symbols documented here are stable. Additional stable-but-not-yet-fully-documented symbols are noted in [Additional stable exports](#additional-stable-exports) below. Symbols absent from `__all__` may change without notice.

## Top-level imports

```python
from khora import (
    Khora,
    KhoraConfig,
    KhoraError,
    EngineCapabilityError,  # raised when a mode/feature is unsupported by the active engine
    SearchMode,
    RememberResult,
    RecallResult,
    BatchResult,
    BatchHandle,        # submit_batch() return value - has .wait() and .batch_id
    DocumentResult,     # per-document callback payload from submit_batch
    Stats,
    LLMUsage,
    UsageSummary,       # aggregate view over a list of LLMUsage
    DocumentSource,
    EventType,
    SemanticFilter,
    context_text,       # render a RecallResult as a flat LLM context string
    # Engines
    create_engine,
    list_engines,
    register_engine,
    # Expertise types
    ExpertiseConfig,
    EntityTypeConfig,
    RelationshipTypeConfig,
    # Recall filter DSL
    RecallFilter,
    StringOps,
    DateOps,
    Op,
    SYSTEM_KEYS,
    RecallFilterValidationError,
    RecallFilterUnsupportedError,
    # Dream phase
    DreamConfig,
    DreamMode,
    DreamScope,
    DreamResult,
    DreamRunInfo,
    OpKind,
    # Filter pushdown reporting
    FilterPushdownReport,
    FilterChannelReport,
)
```

## `Khora`

Primary facade. Delegates to a pluggable engine (default: `vectorcypher`).

```python
Khora(
    database_url: str | KhoraConfig | None = None,
    *,
    engine: str = "vectorcypher",
    graph_url: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    storage_config: StorageConfig | None = None,
    engine_kwargs: dict[str, Any] | None = None,
    run_migrations: bool = False,
)
```

- Pass a PostgreSQL URL, a SurrealDB URL (`memory://`, `surrealkv://…`, `ws://…`), or a full `KhoraConfig`.
- Pass nothing to read from `KHORA_DATABASE_URL` / `KHORA_NEO4J_URL`.
- `run_migrations=True` runs Alembic under an advisory lock on connect. See [migrations.md](migrations.md).
- Credential fields on `KhoraConfig` (DSNs, passwords, API keys) are `pydantic.SecretStr` - `repr()` and `model_dump()` render `'**********'`. Code that reads the cleartext must call `.get_secret_value()`. See the [Secrets section of configuration.md](configuration.md#secretstr-typed-credential-fields).

### Connection

```python
async with Khora(...) as kb:
    ...                       # connect() / disconnect() called automatically

# or manually
kb = Khora(...)
await kb.connect()
try:
    ...
finally:
    await kb.disconnect()
```

### Namespaces

```python
ns = await kb.create_namespace(*, config_overrides=None, metadata=None)  # returns MemoryNamespace
ns = await kb.get_namespace(namespace_id: UUID)                    # returns MemoryNamespace | None
ns = await kb.get_namespace_by_stable_id(namespace_id: str | UUID) # stable-ID lookup
```

`create_namespace` is keyword-only; there is no positional name argument. The optional `config_overrides` dict layers per-namespace settings on top of the global `KhoraConfig`.

Namespaces are the sole tenancy boundary. Use `ns.namespace_id` (the stable public ID) everywhere below - not the row-level `ns.id`. See [architecture/multi-tenancy.md](architecture/multi-tenancy.md).

### `remember`

```python
result: RememberResult = await kb.remember(
    content: str,
    *,
    namespace: str | UUID,
    title: str = "",
    source: str = "",
    source_type: str = "library",
    source_name: str | None = None,
    source_url: str | None = None,
    source_timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    skill_name: str = "general_entities",
    entity_types: list[str],
    relationship_types: list[str],
    expertise: ExpertiseConfig | str | None = None,
    extraction_config_hash: str | None = None,
    chunk_strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    external_id: str | None = None,
    session_id: UUID | None = None,
)
```

Ingests content through the 3-phase pipeline (stage → enrich → expand). `expertise` also accepts a `str`: it is resolved as a registered expertise name or a YAML file path via `load_expertise`. `chunk_strategy` accepts `"fixed"`, `"semantic"`, `"recursive"`, or `"conversation"`. `chunk_size` overrides the target chunk size in tokens for this call only; `None` (default) uses the configured pipeline default (512). `external_id` must be `None` or a non-blank string (≤ 512 chars); otherwise `ValueError` is raised. `session_id` is propagated to `Document.session_id` and every chunk's `Chunk.session_id` so session-scoped recall hits the partial composite index (#620).

`source_type` / `source_name` / `source_url` populate the typed provenance columns on the documents table. `source_type` defaults to `"library"` for direct callers; producers that ingest from a connector (API, database, object store, …) override it. `source_name` is the connector/provider identifier (e.g. `"slack"`, `"linear"`, `"s3"`); `source_url` is the canonical URL for the source row, when one exists.

### `remember_batch`

```python
result: BatchResult = await kb.remember_batch(
    documents: list[dict[str, Any]],
    *,
    namespace: str | UUID,
    skill_name: str = "general_entities",
    source_type: str = "library",
    source_name: str | None = None,
    source_url: str | None = None,
    source_timestamp: datetime | None = None,
    max_concurrent: int = 10,
    deduplicate: bool = True,
    infer_relationships: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    entity_types: list[str],
    relationship_types: list[str],
    expertise: ExpertiseConfig | str | None = None,
    extraction_config_hash: str | None = None,
    chunk_strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    extraction_batch_size: int | None = None,
    extraction_max_tokens: int | None = None,
)
```

Concurrent ingestion with per-document deduplication and optional expansion. Each dict in `documents` accepts the same per-document fields as `remember()` - including `source_type`, `source_name`, `source_url` at the top level of the doc dict (siblings of `content`, `title`, `source`, `external_id`). **Per-doc dict values override the top-level kwargs** for that document; absent keys fall back to the kwarg, which itself defaults to `source_type="library"` / `source_name=None` / `source_url=None`.

### `submit_batch`

```python
handle: BatchHandle = await kb.submit_batch(
    documents: list[dict[str, Any]],
    *,
    on_result: Callable[[int, int, DocumentResult], None],
    namespace: str | UUID,
    skill_name: str = "general_entities",
    source_type: str = "library",
    source_name: str | None = None,
    source_url: str | None = None,
    source_timestamp: datetime | None = None,
    entity_types: list[str],
    relationship_types: list[str],
    expertise: ExpertiseConfig | None = None,
    extraction_config_hash: str | None = None,
    chunk_strategy: ChunkStrategy | None = None,
    max_chunks_in_flight: int | None = None,
    max_concurrent: int = 20,
    reprocess_archived: bool = False,
    session_id: UUID | None = None,
)
```

Deferred sibling of `remember_batch()`: persists every document as `PENDING` and returns a `BatchHandle` immediately; the pending processor picks each document up and fires `on_result` as it completes. Note: unlike `remember()` / `remember_batch()`, `submit_batch()` accepts only `ExpertiseConfig | None` for `expertise` - the string (name / YAML-path) form is not resolved here. Accepts the same provenance kwargs and per-doc dict shape as `remember_batch()` - per-doc dict values override the top-level kwargs. See [`BatchHandle`](#batchhandle) below for the wait/identity surface.

> **The pending processor is opt-in.** `submit_batch()` raises if `kb.start_pending_processor()` has not been called first (after `connect()`), so call it on each write-path service. The check runs after the documents are staged as `PENDING`, so a failure leaves rows in the queue.

`max_concurrent` caps concurrency for this batch's documents specifically; the global processor pool size still applies as a ceiling. The pool is sized via `KhoraConfig.pipelines.pending_processor_max_concurrent` (env: `KHORA_PIPELINES_PENDING_PROCESSOR_MAX_CONCURRENT`); effective per-batch concurrency is `min(pool_size, max_concurrent)`. Two batches submitted concurrently are rate-limited independently - their `max_concurrent` values do not stack.

### `recall`

```python
result: RecallResult = await kb.recall(
    query: str,
    *,
    namespace: str | UUID,
    limit: int = 10,
    mode: SearchMode = SearchMode.HYBRID,
    min_similarity: float = 0.0,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    filter: RecallFilter | dict[str, Any] | None = None,
)
```

- `mode` - one of `SearchMode.VECTOR`, `GRAPH`, `HYBRID`, `ALL`, or `KEYWORD`.
- `min_similarity` - a hard cosine floor honored by every mode (VECTOR / GRAPH / HYBRID / ALL / KEYWORD) as of #1438/#1445: candidates below the floor are dropped before fusion. BM25/keyword evidence can still boost a chunk that clears the floor, but cannot rescue one below it. Default `0.0` disables the floor (falling back to the configured `min_chunk_similarity`, itself `0.0` by default). See [Recall semantics](query-engine/recall-semantics.md).
- `start_time` / `end_time` - **Deprecated.** Explicit temporal filter; bypasses NLP temporal detection. Both-naive or both-aware datetimes are required. Honored on all three engines (chronicle, vectorcypher, skeleton). Prefer the `filter` form: `filter={"occurred_at": {"$gte": ..., "$lt": ...}}`. Cannot be combined with `filter=`.
- To skip LLM-side work: LLM listwise reranking is off by default (enable via `query.enable_llm_reranking=True` / `KHORA_QUERY_ENABLE_LLM_RERANKING`), and set `enable_hyde="never"` on `KhoraConfig.query` (env: `KHORA_QUERY_ENABLE_HYDE`) to disable HyDE expansion. Note the cross-encoder reranker is on by default and runs locally (not an LLM call); disable it with `KHORA_QUERY_ENABLE_RERANKING=false`.

### `context_text`

```python
from khora import context_text

text: str = context_text(result: RecallResult, *, max_chunks: int = 5)
```

Render a `RecallResult` as a flat text context string suitable for an LLM prompt. Groups the first `max_chunks` chunks by document title (`DocumentProjection.title`) joined with `\n\n---\n\n`, then appends an `--- Entities ---` section (deduplicated by entity id) and a `--- Relationships ---` section (deduplicated by `(source_entity_id, target_entity_id, relationship_type)`, with endpoint names resolved from `result.entities` and `str(uuid)` fallback). Returns the empty string when there are no chunks, entities, or relationships to render.

```python
result = await kb.recall("query", namespace=ns_id)
print(context_text(result, max_chunks=3))
```

### `forget`

```python
removed: bool = await kb.forget(document_id: UUID, *, namespace: str | UUID)
```

### `forget_session`

```python
deleted: int = await kb.forget_session(namespace_id: UUID, session_id: UUID)
```

Delete every document in `namespace_id` tagged with `session_id`. Cascade-deletes chunks (via the FK `ON DELETE CASCADE`) and routes per-document graph cleanup through the engine's `forget()` so Neo4j Chunk nodes and extracted entities/relationships are tidied up. Returns the count of documents deleted. See [migrations.md](migrations.md) for the `session_id` column and indexes (migrations 030 + 031).

### `gc.expire_sessions`

```python
from khora import gc

expired_count: int = await gc.expire_sessions(
    *,
    kb: Khora,
    before: datetime,
    namespace_id: UUID | None = None,
)
```

Background-coroutine-friendly TTL cleanup. Calls `forget_session()` for each `session_id` whose newest document predates `before` (using `COALESCE(source_timestamp, created_at)` as the comparison time). **Opt-in** - Khora does not run a scheduler. Adapters / downstream services call this from their own background loop. Pass `namespace_id` for tenant-scoped sweeps; omit to scan every active namespace.

### `list_entities` / `find_related_entities`

Convenience accessors over the underlying engine's graph-view API. Signatures are stable but return types are engine-specific; consult the type hints in `src/khora/khora.py`.

### `get_entity` and `get_document`

```python
entity = await kb.get_entity(entity_id, namespace=ns.namespace_id)
# Entity | None  - returns None for cross-namespace lookups.

document = await kb.get_document(document_id, namespace=ns.namespace_id)
# Document | None - namespace kwarg required.
```

`namespace` is **required** (accepts `str | UUID`, mirrors `list_entities` / `find_related_entities`). The facade fetches the row and verifies its `namespace_id` matches - cross-namespace ids resolve to `None` rather than the foreign entity. Calling without `namespace=` raises `TypeError`. `get_entity` also accepts `include_sources: bool = False`; when `True`, source-document metadata is populated on the returned entity.

This shape applies to the whole `kb.storage` getter surface - namespace is the trust boundary, never derivable from the id alone:

| Method | Required keyword |
|---|---|
| `kb.storage.get_entity(entity_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_relationship(relationship_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_episode(episode_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_chunk(chunk_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_chunks_batch(chunk_ids, *, namespace_id)` | `namespace_id: UUID` - cross-namespace ids silently dropped from the returned dict |
| `kb.storage.get_chunks_by_document(document_id, *, namespace_id)` | `namespace_id: UUID` - returns `[]` if the document doesn't belong to the namespace |

**The contract applies to every backend method** (PRs #761 / #765 / #766 / #769). Every read, exists-check, and mutation on `RelationalBackend` / `VectorBackend` / `GraphBackend` / `EventStore` requires `*, namespace_id: UUID` (kwarg-only) and filters at the SQL / Cypher / SurrealQL layer - not post-fetch. Cross-namespace reads return `None` / `{}` / `[]`; cross-namespace writes silently no-op (raising would expose row existence). The full list of tightened methods is in [migrations.md](migrations.md#v0160---api-migration-namespace-kwarg-required-everywhere).

> **Deprecation.** `StorageCoordinator.{relational,vector,graph,event_store}` are `NamespaceRequiredProxy` wrappers. Accessing them emits one `DeprecationWarning` per role per process; calling a method without `namespace_id=` raises `TypeError`. Call coordinator-level methods (`kb.storage.<method>`) instead.

### `stats`

```python
stats: Stats = await kb.stats(namespace=ns.namespace_id)
# Stats(documents=..., chunks=..., entities=..., relationships=..., last_activity_at=...)
```

## Result dataclasses

All result types are frozen slotted dataclasses.

### `RememberResult`

| Field | Type |
|---|---|
| `document_id` | `UUID` |
| `namespace_id` | `UUID` |
| `chunks_created` | `int` |
| `entities_extracted` | `int` |
| `relationships_created` | `int` |
| `relationships_skipped` | `int` - un-remappable relationships dropped during ingest because a source/target entity could not be canonicalized (ADR-001, #907); mirrored as a `Degradation` in `metadata["degradations"]` when > 0. Always `0` on engines that skip the shared pipeline. |
| `metadata` | `dict[str, Any]` |
| `llm_usage` | `list[LLMUsage]` |

### `BatchResult`

| Field | Type |
|---|---|
| `total` / `processed` / `skipped` / `failed` | `int` |
| `chunks` / `entities` / `relationships` | `int` |
| `metadata` | `dict[str, Any]` |
| `llm_usage` | `list[LLMUsage]` |
| `per_document` | `list[dict]` - one entry per submitted document (input order): `document_id` / `source` / `chunks` / `entities` / `skipped`. Checksum-skipped duplicates carry the *existing* document's id. Populated by VectorCypher; may be empty on other engines. |

`metadata["extraction_errors"]` (int) and `metadata["degradations"]` (ADR-001 list) are present only when one or more chunks failed LLM extraction; `metadata` is an empty dict on the happy path.

### `BatchHandle`

Returned by `kb.submit_batch(...)` (the async-staging path that returns immediately and is processed by the pending-processor task - which must already be running via `kb.start_pending_processor()`; `submit_batch` raises otherwise). Use `await handle.wait()` to block until all documents finish. `submit_batch` also accepts an optional `session_id: UUID | None = None` kwarg that is stamped onto every staged document (per-document `metadata["session_id"]` wins if both are set) - see #620 and the [`session_id` column](migrations.md) for retention/forget semantics.

| Field / method | Type | Description |
|---|---|---|
| `batch_id` | `UUID` | Batch identifier (also surfaced in per-document `DocumentResult`). |
| `total` | `int` | Number of documents staged. |
| `await handle.wait()` | `coroutine` | Resolves when the worker has called `on_result` for every document. |

### `DocumentResult`

Delivered to the `on_result` callback per document as `submit_batch` work completes.

| Field | Type | Description |
|---|---|---|
| `document_id` | `UUID` | Row-level id of the staged document. |
| `namespace_id` | `UUID` | Namespace the document was ingested into. |
| `success` | `bool` | `False` indicates a processing failure; `error` holds the message. |
| `error` | `str \| None` | Populated when `success=False`. |
| `chunks_created` / `entities_extracted` / `relationships_created` | `int` | Per-document counts. |
| `llm_usage` | `list[LLMUsage]` | Costs incurred for this document. |
| `skipped` | `bool` | `True` when the document was in `COMPLETED` / `PROCESSING` / `ARCHIVED` state and `reprocess_archived=False`. |

### `RecallResult`

JSON-serializable response projection. Lives at `khora.core.models.recall.RecallResult` and is re-exported from `khora` for back-compat.

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The original query string. |
| `namespace_id` | `UUID` | Namespace the recall was executed against. |
| `documents` | `list[DocumentProjection]` | Deduplicated set of source documents referenced by any chunk, entity, or relationship in the result. |
| `chunks` | `list[RecallChunk]` | Scored chunks. Score is a typed field, not a tuple position. |
| `entities` | `list[RecallEntity]` | Scored entities with document/chunk provenance ids. |
| `relationships` | `list[RecallRelationship]` | Scored relationships. Always present; populated by graph-backed engines (VectorCypher), empty list for others (Chronicle, Skeleton). |
| `llm_usage` | `list[LLMUsage]` | Token usage incurred during the recall. |
| `communities` | `list[CommunityNode]` | Dream-phase community summaries surfaced by graph-backed engines. Empty list when no communities are materialized or the engine doesn't support them. See [`CommunityNode`](#communitynode). |
| `engine_info` | `dict[str, Any]` | Free-form engine telemetry. **Every engine emits the mandatory key `"engine": "<strategy-name>"`** (`vectorcypher` / `chronicle` / `skeleton`) so consumers can route on producer identity. |

**Producer invariant:** every `chunks[i].document_id` and every id in `entities[i].source_document_ids` / `relationships[i].source_document_ids` is guaranteed to appear as some `documents[j].id`.

#### `engine_info` well-known keys

`engine` is always present. `filter` carries a filter-pushdown report on the standard recall path - it reports what the caller filter narrowed, or "nothing narrowed" when no `filter=` was passed. The remaining keys are best-effort telemetry. Both default engines also emit (since #1331, on `engine_info`, NOT `.metadata`):

| Key | Emitted by | Description |
|---|---|---|
| `abstention_signals` | vectorcypher, chronicle | Four boolean flags (`entities_empty`, `chunks_empty`, `chunks_below_min`, `top_score_low`) plus `combined_score` (0.0 = confident, 1.0 = should abstain) and a `should_abstain` convenience flag. Passive - recall still returns chunks when they trip. |
| `confidence` | vectorcypher, chronicle | Calibrated `0.8 * cosine + 0.2 * gap` score in [0, 1]. |
| `degradations` | chronicle (vectorcypher on degraded paths) | ADR-001 degradation records; empty/absent when nothing degraded. |
| `channels_used` | vectorcypher | Channel provenance for this recall. |
| `channels` | chronicle | Per-channel hit counts (`semantic` / `bm25` / `temporal` / `entity`). Note the key name differs from vectorcypher's `channels_used`. |
| `routing` | vectorcypher, chronicle | Query-complexity routing decision. |

See [Recall semantics](query-engine/recall-semantics.md) for how these interact with the score contract.

#### `engine_info["filter"]` — honest filter-pushdown report

When `recall(filter=...)` is given a structured [`RecallFilter`](#recallfilter), the engine populates `engine_info["filter"]` with a **`FilterPushdownReport`** — a backend-agnostic, honest account of what happened to the filter's constraint leaves. Both `FilterPushdownReport` and the per-channel `FilterChannelReport` are public (exported from `khora` and `khora.filter`). The report is a Pydantic model; engines serialize it (e.g. `model_dump()`) into the free-form `engine_info` dict.

| Field | Type | Description |
|---|---|---|
| `pushed_down` | `bool` | `True` only when the filter is *fully* pushed: `post_filtered_keys` is empty AND `pushed_keys` covers every constraint leaf. `False` for a constraint-free / no-filter recall — nothing was narrowed. |
| `post_filtered` | `bool` | `True` when any constraint leaf was re-checked in memory on a gating channel, OR when a channel ran a defensive full-predicate re-check even though every leaf compiled down. |
| `pushed_keys` | `list[str]` | Dotted constraint-leaf keys pushed into the backend query on *every* gating channel (sorted, JSON-stable). |
| `post_filtered_keys` | `list[str]` | Dotted constraint-leaf keys re-checked in memory on at least one gating channel (sorted). |
| `unenforced_keys` | `list[str]` | Dotted constraint-leaf keys that **no** channel pushed or post-filtered — i.e. no channel enforced them on this recall (sorted). Empty on a correct recall. A non-empty value is the in-band signal that a result-producing path returned candidates without enforcing these leaves (silent under-enforcement). |
| `channels` | `dict[str, FilterChannelReport]` | Per-channel breakdown keyed by channel name. One entry per channel the engine reported on — a single-channel engine emits one entry on every recall (even a no-filter recall, with empty key lists). |

`pushed_keys`, `post_filtered_keys`, and `unenforced_keys` together form a **total, disjoint** partition of the filter's constraint leaves. A leaf is in `pushed_keys` only when every channel that gates it pushed it into the backend query; it is in `post_filtered_keys` when at least one gating channel re-checked it in memory; and it is in `unenforced_keys` when no channel gates it at all — the in-band signal that nothing enforced it on this recall. On a correct recall `unenforced_keys == []` (for a single-channel engine like `skeleton`, whose one channel gates every leaf, it is always empty; multi-channel engines populate it only when a constraint leaf slips past every channel). A *defensive* full-predicate re-check (the `sqlite_lance` path: `compile_lance` fully pushes the predicate to SQL, but a `compile_python` post-filter re-checks the whole AST as a safety net) sets `post_filtered=True` **without demoting** any fully-pushed leaf — so `pushed_down` stays `True` while `post_filtered` is also `True`.

Each `FilterChannelReport` carries that channel's own `pushed_keys` / `post_filtered_keys` (sorted dotted leaf keys). The serialized wire shape (`model_dump(mode="json")`) is:

```json
{
  "pushed_down": true,
  "post_filtered": true,
  "pushed_keys": ["metadata.tier"],
  "post_filtered_keys": [],
  "unenforced_keys": [],
  "channels": {
    "sqlite_lance": {"pushed_keys": ["metadata.tier"], "post_filtered_keys": []}
  }
}
```

Prior to #1069 the skeleton engine derived `pushed_down` from a hardcoded `backend == "pgvector"` check, so it under-reported on every non-pgvector backend (e.g. `sqlite_lance`) whose compiler actually pushed the predicate down. The report is now derived from what each channel's compiler consumed.

#### `DocumentProjection`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Document id. |
| `created_at` | `datetime` | Document creation timestamp. |
| `source_type` | `str` | Category; defaults to `"library"` for direct library calls. Free-form - Khora does not validate or enumerate. |
| `title` | `str \| None` | Optional title. |
| `external_id` | `str \| None` | Caller-supplied opaque identifier. |
| `source` | `str \| None` | Optional connector URI. |
| `source_name` | `str \| None` | SaaS-tool / connector identifier. |
| `source_url` | `str \| None` | Addressable doc URL. |
| `content_type` | `str \| None` | MIME / content type. |
| `source_timestamp` | `datetime \| None` | Source-system timestamp (e.g., message sent-at), distinct from ingest `created_at`. ISO-8601 strings are accepted on input and coerced via `coerce_source_timestamp`. |
| `metadata` | `dict[str, Any]` | Free-form user metadata. |

#### `RecallChunk`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Chunk id. |
| `document_id` | `UUID` | Foreign key into `RecallResult.documents`. |
| `content` | `str` | Chunk text. |
| `score` | `float` | Absolute relevance signal (#1433/#1441): the raw query-to-chunk cosine when available, else `0.0` meaning "no vector measurement" (e.g. graph-only hits), NOT "irrelevant". Chunk **order** is the authoritative ranking (fusion + boost + rerank) - re-sorting by `score` discards graph/rerank evidence. See [Recall semantics](query-engine/recall-semantics.md). |
| `created_at` | `datetime` | Chunk creation timestamp. |
| `occurred_at` | `datetime \| None` | Event-time anchor when applicable (e.g., chat message sent-at). |
| `connected_entity_ids` | `list[UUID]` | Engine-populated entity ids linked to this chunk. |
| `chunker_info` | `dict[str, Any]` | Chunker self-identification dict (min `{"chunker": "<name>"}`). |

#### `RecallEntity`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Entity id. |
| `name` | `str` | Entity name. |
| `entity_type` | `str` | Entity type label (e.g., `PERSON`). |
| `description` | `str` | Description (may be empty). |
| `score` | `float` | Retrieval score. |
| `attributes` | `dict[str, Any]` | Free-form entity attributes. |
| `mention_count` | `int` | Number of mentions seen. |
| `source_document_ids` | `list[UUID]` | Source document ids; all entries appear in `RecallResult.documents`. |
| `source_chunk_ids` | `list[UUID]` | Source chunk ids. |

#### `RecallRelationship`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Relationship id. |
| `source_entity_id` | `UUID` | Source entity. |
| `target_entity_id` | `UUID` | Target entity. |
| `relationship_type` | `str` | Relationship type label. |
| `description` | `str` | Description (may be empty). |
| `score` | `float` | Retrieval score. |
| `valid_from` | `datetime \| None` | Validity window start. |
| `valid_until` | `datetime \| None` | Validity window end. |
| `source_document_ids` | `list[UUID]` | Source document ids; all entries appear in `RecallResult.documents`. |

### `Stats`

| Field | Type |
|---|---|
| `documents` / `chunks` / `entities` / `relationships` | `int` |
| `last_activity_at` | `datetime \| None` |
| `metadata` | `dict[str, Any]` - ADR-001 failure records; `metadata["errors"]` holds an `ErrorRecord` when a counter could not run (the int field stays `0`), so callers can tell "couldn't count" from "counted zero". |

### `LLMUsage`

`LLMUsage` fields are part of the stable public API and are consumed by external cost-tracking integrations. Do not mutate instances; they are `frozen`.

| Field | Type | Description |
|---|---|---|
| `operation` | `str` | Logical operation name (e.g. `"entity_extraction"`, `"embedding"`). |
| `model` | `str` | Model identifier (e.g. `"gpt-4o"`, `"text-embedding-3-small"`). |
| `prompt_tokens` | `int` | Input tokens consumed. |
| `completion_tokens` | `int` | Output tokens (0 for embeddings). |
| `total_tokens` | `int` | Sum of prompt and completion tokens. |
| `latency_ms` | `float` | Round-trip latency in milliseconds. |
| `batch_size` | `int` | Batch size (`>1` for embedding batches). |
| `cost_usd` | `float` | Estimated USD cost via litellm pricing tables; `0.0` when unknown. |

### `UsageSummary`

`UsageSummary` is a stable public export (`from khora import UsageSummary`). It aggregates a `list[LLMUsage]` into totals and per-operation / per-model breakdowns. See `Khora.recall()`'s return value or build one directly with `UsageSummary.from_usage(usages)`.

| Field | Type | Description |
|---|---|---|
| `total_prompt_tokens` | `int` | Sum of all `LLMUsage.prompt_tokens`. |
| `total_completion_tokens` | `int` | Sum of all `LLMUsage.completion_tokens`. |
| `total_tokens` | `int` | Sum of all `LLMUsage.total_tokens`. |
| `total_cost_usd` | `float` | Sum of all `LLMUsage.cost_usd`. |
| `total_latency_ms` | `float` | Sum of all `LLMUsage.latency_ms`. |
| `by_operation` | `dict[str, _OperationUsage]` | Totals keyed by `LLMUsage.operation`. |
| `by_model` | `dict[str, _OperationUsage]` | Totals keyed by `LLMUsage.model`. |

### `CommunityNode`

Returned in `RecallResult.communities`. A materialized dream-phase community summary node, produced by the `community_summary` dream op and mirrored to the graph as a `:Community` node.

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Community node id. |
| `namespace_id` | `UUID` | Namespace the community belongs to. |
| `summary` | `str` | LLM-grounded summary text (uncited claims dropped before insert). |
| `member_ids` | `list[UUID]` | Entity ids that are members of this community. |
| `summary_depth` | `int` | Hierarchical depth (1 = leaf community). |
| `embedding` | `list[float] \| None` | Optional embedding of the summary text. |

## `SearchMode`

```python
from khora import SearchMode

SearchMode.VECTOR    # pgvector / HNSW only
SearchMode.GRAPH     # Cypher / graph traversal only
SearchMode.HYBRID    # vector + graph + keyword, fused via RRF
SearchMode.ALL       # every available channel (slower, more context)
SearchMode.KEYWORD   # BM25 / keyword-only; skips vector search, returns BM25 results on all backends
```

See [query-engine/search-modes.md](query-engine/search-modes.md).

## Engines

Engines are discovered through the `khora.engines` registry. The default is `vectorcypher`.

```python
from khora import create_engine, list_engines, register_engine

list_engines()                                              # ['skeleton', 'vectorcypher', 'chronicle']
engine = create_engine("chronicle", ...)                    # low-level - prefer Khora(engine="chronicle")
register_engine("my_engine", "my.module", "MyEngineClass")  # lazy: module path + class name
```

A custom engine **must** implement the full `MemoryEngineProtocol` from `src/khora/engines/protocol.py`. See [engines/engine-comparison.md](engines/engine-comparison.md) for selection guidance.

## Expertise

`ExpertiseConfig` is a stable public API.

```python
from khora import ExpertiseConfig, EntityTypeConfig, RelationshipTypeConfig

expertise = ExpertiseConfig(
    name="medical_research",
    entity_types=[
        EntityTypeConfig(type="DRUG", description="Pharmaceutical compound", ...),
        EntityTypeConfig(type="CONDITION", description="Medical condition", ...),
    ],
    relationship_types=[
        RelationshipTypeConfig(type="TREATS", description="Drug treats condition", ...),
    ],
)

await kb.remember(content, namespace=ns.namespace_id, expertise=expertise,
                    entity_types=[t.type for t in expertise.entity_types],
                    relationship_types=[t.type for t in expertise.relationship_types])
```

See [extraction/expertise-system.md](extraction/expertise-system.md). The machine-readable contract is `__all__` in `src/khora/extraction/skills/base.py`.

## Hooks

```python
from khora import SemanticFilter, EventType
```

Subscribe to extraction and recall events. The full Phase 2 surface (EventBridge-style `match` DSL, `CHUNK_ENTITIES_RESOLVED` for co-occurrence filtering, Level 2 LLM evaluator with cache + per-subscription budget) is documented in [hooks/semantic-hooks.md](hooks/semantic-hooks.md).

## Advanced (opt-in)

These surfaces are documented for completeness but are **default-OFF** behind config flags pending A/B validation.

```python
from khora.query import hyde_cypher                     # parameterized graph queries
from khora.diagnostics import compute_graph_stats, GraphStats  # PPR decision-gate reporter
```

- `khora.query.hyde_cypher` - `select_template()`, `generate_cypher()`, `validate_selection()`, `TEMPLATES`, `HyDECypherTemplate`, `HyDECypherSelection`, `HyDECypherValidationError`. Default OFF; enable via `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`. See [query-engine/retrieval-tuning.md](query-engine/retrieval-tuning.md).
- `khora.diagnostics.graph_density` - one-shot reporter for the PPR audit (Issue #598). Operator script: `scripts/audit_graph_density.py`. This module is intentionally **not stable public API** - it may be renamed or removed without a major-version bump.

## RecallFilter

The `RecallFilter` DSL provides deterministic, pre-retrieval filtering of recall results. It is a stable public export.

```python
from datetime import datetime, timezone
from khora import RecallFilter, StringOps, DateOps, Op, SYSTEM_KEYS
from khora import RecallFilterValidationError, RecallFilterUnsupportedError

# Match documents from a specific source (kwargs form, recommended)
f = RecallFilter(source_name="slack")

# Operator form - exclude a source_type
f = RecallFilter(source_name=StringOps(**{"$ne": "internal"}))

# Date range filter using DateOps
f = RecallFilter(occurred_at=DateOps(**{"$gte": datetime(2024, 1, 1, tzinfo=timezone.utc),
                                        "$lt": datetime(2024, 7, 1, tzinfo=timezone.utc)}))

# Wire/dict form (useful when deserializing HTTP bodies or YAML configs)
f = RecallFilter.model_validate({"source_name": "slack", "source_type": {"$ne": "internal"}})

result = await kb.recall("query", namespace=ns_id, filter=f)
```

Pass a `RecallFilter` (or a raw `dict` that will be coerced to one) to `recall(filter=...)`. The recommended form is Pydantic kwargs (`RecallFilter(source_name="slack")`); the wire form (`RecallFilter.model_validate({...})`) is useful for deserializing HTTP bodies or YAML configs. The engine reports what it enforced in `engine_info["filter"]` as a [`FilterPushdownReport`](#recallresult) (see above). Validation errors raise `RecallFilterValidationError`; unsupported operations on the active backend raise `RecallFilterUnsupportedError`. `SYSTEM_KEYS` is a frozenset of the ten filterable system keys (`occurred_at`, `created_at`, `source_timestamp`, `source_type`, `source_name`, `source_url`, `external_id`, `content_type`, `source`, `title`).

## Dream phase

The dream phase is Khora's background knowledge-consolidation cycle. It is documented in [dream-phase.md](dream-phase.md).

Stable public exports for dream orchestration:

```python
from khora import DreamConfig, DreamMode, DreamScope, DreamResult, DreamRunInfo, OpKind
```

- `DreamConfig` - per-run configuration (which ops to run, concurrency, LLM budget).
- `DreamMode` - enum controlling which dream ops fire (`FULL`, `QUICK`, `CUSTOM`).
- `DreamScope` - namespace(s) the dream run targets.
- `DreamResult` - return value of `Khora.dream()` (per-op results, metadata, partial-failure diagnostics).
- `DreamRunInfo` - lightweight summary of a completed or in-progress dream run.
- `OpKind` - enum of available dream ops (`DEDUPE_ENTITIES`, `COMMUNITY_SUMMARY`, `CONTRADICTION_DETECT`, …).

## Additional stable exports

The following symbols are in `__all__` and therefore stable but are not yet individually documented on this page.

| Symbol | Notes |
|---|---|
| `FilterPushdownReport` / `FilterChannelReport` | Documented under [`RecallResult`](#recallresult) above. |
| `RecallFilter` / `StringOps` / `DateOps` / `Op` / `SYSTEM_KEYS` / `RecallFilterValidationError` / `RecallFilterUnsupportedError` | Documented under [`RecallFilter`](#recallfilter) above. |
| `DreamConfig` / `DreamMode` / `DreamScope` / `DreamResult` / `DreamRunInfo` / `OpKind` | Documented under [`Dream phase`](#dream-phase) above. |
| `UsageSummary` | Documented under [`UsageSummary`](#usagesummary) above. |
| `EngineCapabilityError` | Documented under [`Errors`](#errors) below. |

## Errors

All domain errors subclass `KhoraError`. Catch it at system boundaries; internal code uses specific subclasses. See `src/khora/exceptions.py` for the hierarchy.

```python
from khora import KhoraError

try:
    await kb.remember(...)
except KhoraError as exc:
    ...
```

`EngineCapabilityError` (a `KhoraError` subclass, stable public export) is raised when the active engine does not support a requested mode or feature - for example, asking the `skeleton` engine for `SearchMode.GRAPH` traversal. Catch it to implement graceful fallback between engines.

## Stability guarantee

Symbols imported from the top-level `khora` namespace and `khora.extraction.skills` are stable. Additive changes land in minor releases; breaking changes require a major version bump.

Private imports (`khora.engines.vectorcypher`, `khora.query.engine`, `khora.pipelines.flows`, etc.) are **not** stable and can change between minor versions.
