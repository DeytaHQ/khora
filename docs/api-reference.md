# API Reference

The public Khora surface is pinned by the machine-readable `__all__` in `src/khora/__init__.py`. Everything on this page is stable. Symbols not listed here may change without notice.

## Top-level imports

```python
from khora import (
    Khora,
    KhoraConfig,
    KhoraError,
    SearchMode,
    RememberResult,
    RecallResult,
    BatchResult,
    BatchHandle,        # submit_batch() return value — has .wait() and .id
    DocumentResult,     # per-document callback payload from submit_batch
    Stats,
    LLMUsage,
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
- Credential fields on `KhoraConfig` (DSNs, passwords, API keys) are `pydantic.SecretStr` — `repr()` and `model_dump()` render `'**********'`. Code that reads the cleartext must call `.get_secret_value()`. See the [Secrets section of configuration.md](configuration.md#secretstr-typed-credential-fields).

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
ns = await kb.create_namespace(*, config_overrides=None)           # returns MemoryNamespace
ns = await kb.get_namespace(namespace_id: UUID)                    # returns MemoryNamespace | None
ns = await kb.get_namespace_by_stable_id(namespace_id: str | UUID) # stable-ID lookup
```

`create_namespace` is keyword-only; there is no positional name argument. The optional `config_overrides` dict layers per-namespace settings on top of the global `KhoraConfig`.

Namespaces are the sole tenancy boundary. Use `ns.namespace_id` (the stable public ID) everywhere below — not the row-level `ns.id`. See [architecture/multi-tenancy.md](architecture/multi-tenancy.md).

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
    metadata: dict[str, Any] | None = None,
    skill_name: str = "general_entities",
    entity_types: list[str],
    relationship_types: list[str],
    expertise: ExpertiseConfig | None = None,
    extraction_config_hash: str | None = None,
    chunk_strategy: ChunkStrategy | None = None,
    external_id: str | None = None,
    session_id: UUID | None = None,
)
```

Ingests content through the 3-phase pipeline (stage → enrich → expand). `chunk_strategy` accepts `"fixed"`, `"semantic"`, `"recursive"`, or `"conversation"`. `external_id` must be `None` or a non-blank string (≤ 512 chars); otherwise `ValueError` is raised. `session_id` is propagated to `Document.session_id` and every chunk's `Chunk.session_id` so session-scoped recall hits the partial composite index (#620).

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
    max_concurrent: int = 10,
    deduplicate: bool = True,
    infer_relationships: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    entity_types: list[str],
    relationship_types: list[str],
    expertise: ExpertiseConfig | None = None,
    extraction_config_hash: str | None = None,
    chunk_strategy: ChunkStrategy | None = None,
    extraction_batch_size: int | None = None,
    extraction_max_tokens: int | None = None,
)
```

Concurrent ingestion with per-document deduplication and optional expansion. Each dict in `documents` accepts the same per-document fields as `remember()` — including `source_type`, `source_name`, `source_url` at the top level of the doc dict (siblings of `content`, `title`, `source`, `external_id`). **Per-doc dict values override the top-level kwargs** for that document; absent keys fall back to the kwarg, which itself defaults to `source_type="library"` / `source_name=None` / `source_url=None`.

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

Deferred sibling of `remember_batch()`: persists every document as `PENDING` and returns a `BatchHandle` immediately; processing continues in the background and fires `on_result` per document as it completes. Accepts the same provenance kwargs and per-doc dict shape as `remember_batch()` — per-doc dict values override the top-level kwargs. See [`BatchHandle`](#batchhandle) below for the wait/identity surface.

### `recall`

```python
result: RecallResult = await kb.recall(
    query: str,
    *,
    namespace: str | UUID,
    limit: int = 10,
    mode: SearchMode = SearchMode.HYBRID,
    min_similarity: float = 0.0,
    agentic: bool = False,
    raw: bool = False,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
)
```

- `mode` — one of `SearchMode.VECTOR`, `GRAPH`, `HYBRID`, or `ALL`.
- `agentic=True` — multi-step exploration with follow-up queries.
- `raw=True` — skips query understanding, reranking, HyDE, and entity linking (useful for benchmarks).
- `start_time` / `end_time` — explicit temporal filter; bypasses NLP temporal detection. Both-naive or both-aware datetimes are required.

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

Background-coroutine-friendly TTL cleanup. Calls `forget_session()` for each `session_id` whose newest document predates `before` (using `COALESCE(source_timestamp, created_at)` as the comparison time). **Opt-in** — Khora does not run a scheduler. Adapters / downstream services call this from their own background loop. Pass `namespace_id` for tenant-scoped sweeps; omit to scan every active namespace.

### `list_entities` / `find_related_entities`

Convenience accessors over the underlying engine's graph-view API. Signatures are stable but return types are engine-specific; consult the type hints in `src/khora/khora.py`.

### `get_entity`

```python
entity = await kb.get_entity(entity_id, namespace=ns.namespace_id)
# Entity | None  — returns None for cross-namespace lookups.
```

`namespace` is **required** (accepts `str | UUID`, mirrors `list_entities` / `find_related_entities`). The facade fetches the row and verifies its `namespace_id` matches — cross-namespace ids resolve to `None` rather than the foreign entity. Calling without `namespace=` raises `TypeError`.

This shape applies to the whole `kb.storage` getter surface — namespace is the trust boundary, never derivable from the id alone:

| Method | Required keyword |
|---|---|
| `kb.storage.get_entity(entity_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_relationship(relationship_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_episode(episode_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_chunk(chunk_id, *, namespace_id)` | `namespace_id: UUID` |
| `kb.storage.get_chunks_batch(chunk_ids, *, namespace_id)` | `namespace_id: UUID` — cross-namespace ids silently dropped from the returned dict |
| `kb.storage.get_chunks_by_document(document_id, *, namespace_id)` | `namespace_id: UUID` — returns `[]` if the document doesn't belong to the namespace |

The underlying graph-backend / vector-backend `get_*` methods retain their id-only shape; they sit below the trust boundary. Filtering happens at the facade.

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
| `metadata` | `dict[str, Any]` |
| `llm_usage` | `list[LLMUsage]` |

### `BatchResult`

| Field | Type |
|---|---|
| `total` / `processed` / `skipped` / `failed` | `int` |
| `chunks` / `entities` / `relationships` | `int` |
| `metadata` | `dict[str, Any]` |
| `llm_usage` | `list[LLMUsage]` |

### `BatchHandle`

Returned by `kb.submit_batch(...)` (the async-staging path that returns immediately and processes via a background worker). Use `await handle.wait()` to block until all documents finish. `submit_batch` also accepts an optional `session_id: UUID | None = None` kwarg that is stamped onto every staged document (per-document `metadata["session_id"]` wins if both are set) — see #620 and the [`session_id` column](migrations.md) for retention/forget semantics.

| Field / method | Type | Description |
|---|---|---|
| `id` | `UUID` | Batch identifier (also surfaced in per-document `DocumentResult`). |
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
| `usage` | `list[LLMUsage]` | Token usage incurred during the recall. |
| `engine_info` | `dict[str, Any]` | Free-form engine telemetry. **Every engine emits the mandatory key `"engine": "<strategy-name>"`** (`vectorcypher` / `chronicle` / `skeleton`) so consumers can route on producer identity. |

**Producer invariant:** every `chunks[i].document_id` and every id in `entities[i].source_document_ids` / `relationships[i].source_document_ids` is guaranteed to appear as some `documents[j].id`.

#### `DocumentProjection`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Document id. |
| `created_at` | `datetime` | Document creation timestamp. |
| `source_type` | `str` | Category; defaults to `"library"` for direct library calls. Free-form — Khora does not validate or enumerate. |
| `title` | `str \| None` | Optional title. |
| `external_id` | `str \| None` | Caller-supplied opaque identifier. |
| `source` | `str \| None` | Optional connector URI. |
| `source_name` | `str \| None` | SaaS-tool / connector identifier. |
| `source_url` | `str \| None` | Addressable doc URL. |
| `content_type` | `str \| None` | MIME / content type. |
| `source_timestamp` | `datetime \| None` | Source-system timestamp (e.g., message sent-at), distinct from ingest `created_at`. |
| `metadata` | `dict[str, Any]` | Free-form user metadata. |

#### `RecallChunk`

| Field | Type | Description |
|---|---|---|
| `id` | `UUID` | Chunk id. |
| `document_id` | `UUID` | Foreign key into `RecallResult.documents`. |
| `content` | `str` | Chunk text. |
| `score` | `float` | Retrieval score. |
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

### `LLMUsage`

`LLMUsage` fields are part of the stable public API and are consumed by external cost-tracking integrations. Do not mutate instances; they are `frozen`.

## `SearchMode`

```python
from khora import SearchMode

SearchMode.VECTOR    # pgvector / HNSW only
SearchMode.GRAPH     # Cypher / graph traversal only
SearchMode.HYBRID    # vector + graph + keyword, fused via RRF
SearchMode.ALL       # every available channel (slower, more context)
```

See [query-engine/search-modes.md](query-engine/search-modes.md).

## Engines

Engines are discovered through the `khora.engines` registry. The default is `vectorcypher`.

```python
from khora import create_engine, list_engines, register_engine

list_engines()                                              # ['skeleton', 'vectorcypher', 'chronicle']
engine = create_engine("chronicle", ...)                    # low-level — prefer Khora(engine="chronicle")
register_engine("my_engine", "my.module", "MyEngineClass")  # lazy: module path + class name
```

A custom engine **must** implement the full `MemoryEngineProtocol` from `src/khora/engines/protocol.py`. See [engines/engine-comparison.md](engines/engine-comparison.md) for selection guidance.

## Expertise

`ExpertiseConfig` is a stable sibling API maintained in coordination with khora-explorer.

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

## Advanced (opt-in, v0.12.0)

These surfaces are documented for completeness but are **default-OFF** behind config flags pending A/B validation.

```python
from khora.query import hyde_cypher                     # parameterized graph queries
from khora.diagnostics import compute_graph_stats, GraphStats  # PPR decision-gate reporter
```

- `khora.query.hyde_cypher` — `select_template()`, `generate_cypher()`, `validate_selection()`, `TEMPLATES`, `HyDECypherTemplate`, `HyDECypherSelection`, `HyDECypherValidationError`. Default OFF; enable via `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`. See [query-engine/retrieval-tuning.md](query-engine/retrieval-tuning.md).
- `khora.diagnostics.graph_density` — one-shot reporter for the PPR audit (Issue #598). Operator script: `scripts/audit_graph_density.py`. This module is intentionally **not stable public API** — it may be renamed or removed without a major-version bump.

## Errors

All domain errors subclass `KhoraError`. Catch it at system boundaries; internal code uses specific subclasses. See `src/khora/exceptions.py` for the hierarchy.

```python
from khora import KhoraError

try:
    await kb.remember(...)
except KhoraError as exc:
    ...
```

## Stability guarantee

Symbols imported from the top-level `khora` namespace and `khora.extraction.skills` are stable. Additive changes land in minor releases; breaking changes require a major version bump and coordinated release with khora-cli and khora-explorer.

Private imports (`khora.engines.vectorcypher`, `khora.query.engine`, `khora.pipelines.flows`, etc.) are **not** stable and can change between minor versions.
