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
ns = await kb.create_namespace(namespace_name)                     # returns MemoryNamespace
ns = await kb.get_namespace(namespace_id: UUID)                    # returns MemoryNamespace | None
ns = await kb.get_namespace_by_stable_id(namespace_id: str | UUID) # stable-ID lookup
```

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
)
```

Ingests content through the 3-phase pipeline (stage → enrich → expand). `chunk_strategy` accepts `"fixed"`, `"semantic"`, `"recursive"`, or `"conversation"`. `external_id` must be `None` or a non-blank string (≤ 512 chars); otherwise `ValueError` is raised.

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

list_engines()                              # ['skeleton', 'vectorcypher', 'chronicle']
engine = create_engine("chronicle", ...)    # low-level — prefer Khora(engine="chronicle")
register_engine("my_engine", MyEngineClass) # must implement MemoryEngineProtocol
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

Subscribe to extraction events. See [hooks/semantic-hooks.md](hooks/semantic-hooks.md).

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
