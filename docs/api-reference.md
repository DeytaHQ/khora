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

### `remember_batch`

```python
result: BatchResult = await kb.remember_batch(
    documents: list[dict[str, Any]],
    *,
    namespace: str | UUID,
    skill_name: str = "general_entities",
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

Concurrent ingestion with per-document deduplication and optional expansion. Each dict in `documents` has the same fields you'd pass to `remember()`.

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
    include_sources: bool = False,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
)
```

- `mode` — one of `SearchMode.VECTOR`, `GRAPH`, `HYBRID`, or `ALL`.
- `agentic=True` — multi-step exploration with follow-up queries.
- `raw=True` — skips query understanding, reranking, HyDE, and entity linking (useful for benchmarks).
- `start_time` / `end_time` — explicit temporal filter; bypasses NLP temporal detection. Both-naive or both-aware datetimes are required.

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

| Field | Type |
|---|---|
| `query` | `str` |
| `namespace_id` | `UUID` |
| `chunks` | `list[tuple[Chunk, float]]` |
| `entities` | `list[tuple[Entity, float]]` |
| `relationships` | `list[tuple[Relationship, float]]` — only populated by VectorCypher |
| `context_text` | `str` — pre-formatted for LLM context |
| `metadata` | `dict[str, Any]` |
| `llm_usage` | `list[LLMUsage]` |

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
