# Documents and Chunks

Documents are the primary input to Khora, representing source content that gets processed, chunked, and indexed. This document covers the Document and Chunk data models.

## Document Model

Located at `src/khora/core/models/document.py`.

```python
@dataclass
class Document:
    id: UUID
    namespace_id: UUID
    content: str
    external_id: str | None = None  # Caller-supplied source identity
    status: DocumentStatus = DocumentStatus.PENDING

    # Source / provenance (flat fields)
    title: str | None = None
    source: str | None = None           # "slack/channel-123", "notion/page-id"
    source_type: str = "library"
    source_name: str | None = None
    source_url: str | None = None
    content_type: str | None = None
    author: str | None = None
    language: str | None = None
    checksum: str | None = None         # SHA-256 for deduplication
    size_bytes: int = 0

    # Free-form metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Processing stats
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0

    # Extraction config tracking
    extraction_config_hash: str | None = None
    extraction_params: dict[str, Any] | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
    source_timestamp: datetime | None = None

    # Session attribution (#620)
    session_id: UUID | None = None

    # Replace-path graph-mirror queue (migration 051, #1430)
    graph_mirror_pending: dict[str, Any] | None = None
```

A non-NULL `graph_mirror_pending` means the `external_id` replace path committed Postgres (new chunks + `COMPLETED` status) but the post-commit graph mirror failed; **status stays `COMPLETED`** and the payload carries the computed graph plan for the replace-mirror reconciler to replay. NULL/absent = the graph is in lockstep with PG for that document. The payload shape comes from `khora.storage.replace_mirror.build_replace_mirror_payload`; the column is `JSONB(none_as_null=True)`, so clearing it writes SQL `NULL` and the partial index `ix_documents_graph_mirror_pending WHERE graph_mirror_pending IS NOT NULL` stays drainable. Modeled on the dream reconciler's `khora_dream_runs.graph_mirror_pending` (migration 047).

### Document Status

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"        # Awaiting processing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"    # Successfully processed
    FAILED = "failed"          # Processing failed
    ARCHIVED = "archived"      # Archived, not actively used
```

### Status Transitions

```
PENDING → PROCESSING → COMPLETED → ARCHIVED
                     → FAILED
```

```python
# Mark as processing
document.mark_processing()
# status → PROCESSING, updated_at → now

# Mark as completed
document.mark_completed(chunk_count=5, entity_count=10, relationship_count=3)
# status → COMPLETED, chunk_count=5, entity_count=10, relationship_count=3
# processed_at → now, updated_at → now

# Mark as failed
document.mark_failed("Error: LLM timeout")
# status → FAILED, error_message = "Error: LLM timeout"
# updated_at → now
```

### Checksum Deduplication

Documents are deduplicated by content checksum, but only when all caller-supplied identity fields also match the existing row. Same content with a new `external_id` or `session_id` produces a **new** document.

```python
# During ingestion
checksum = hashlib.sha256(content.encode()).hexdigest()

# Check if identical document exists
existing = await storage.get_document_by_checksum(namespace_id, checksum)
if existing and _checksum_dedup_applies(existing, external_id=external_id, session_id=session_id):
    # Skip - same content AND same caller identity
    return None
# Otherwise ingest as a new document (new external_id or session_id = new document)
```

The `_checksum_dedup_applies` guard (added in #1171) returns `False` when the caller supplies an `external_id` that differs from the existing row, or a `session_id` that differs. Callers that supply neither field keep the checksum-only behavior.

## Chunk Model

Chunks are segments of documents optimized for embedding and retrieval.

```python
@dataclass(slots=True)
class Chunk:
    id: UUID
    document_id: UUID
    namespace_id: UUID
    content: str

    # Position in document (flat)
    chunk_index: int = 0       # 0-based chunk index
    start_char: int = 0        # Character offset start
    end_char: int = 0          # Character offset end
    token_count: int = 0       # Token count (for chunking decisions)

    # Free-form metadata propagated from the parent document
    metadata: dict[str, Any] = field(default_factory=dict)
    # Chunker output (strategy, overlap, etc.) - distinct from metadata
    chunker_info: dict[str, Any] = field(default_factory=dict)

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_timestamp: datetime | None = None
    occurred_at: datetime | None = None        # Chunk event-time
    last_accessed_at: datetime | None = None   # Set on recall with reinforcement

    # Session attribution propagated from the parent document (#620)
    session_id: UUID | None = None

    # Populated by Khora when include_sources=True
    source_document: DocumentSource | None = None
```

### Chunk Metadata vs chunker_info

``Chunk.metadata`` is the propagated document-level free-form dict (callers'
own keys). ``Chunk.chunker_info`` carries chunker output (strategy, overlap
settings, etc.) plus the four position-bookkeeping keys ``chunk_index`` /
``start_char`` / ``end_char`` / ``token_count``, which the writers stamp last
so the engine's computed values win any collision with chunker-emitted keys.
The two dicts are kept separate so chunker keys never shadow document
metadata:

```python
chunk.metadata = {
    # Propagated from document.metadata
    "source": "slack/channel-123",
    "title": "Team Discussion",
}
chunk.chunker_info = {
    "chunker": "semantic",
    "overlap_tokens": 50,
    # Position bookkeeping, stamped by the engine on every persisted chunk
    "chunk_index": 3,
    "start_char": 2048,
    "end_char": 3071,
    "token_count": 256,
}
```

The separation also means the namespaces never mix: a caller metadata key
that happens to be named ``chunk_index`` is user data — it is preserved in
``metadata`` verbatim and may legitimately differ from
``chunker_info["chunk_index"]``, which is the authoritative source for chunk
position.

## Timestamp Axes

khora keeps two distinct time axes (see `docs/engines/temporal-model.md`). They
must never be conflated:

- **Real-world / valid time** - when the underlying fact was true in the world.
  Carried by `Document.source_timestamp` and `Chunk.source_timestamp` (projected
  as `RecallChunk.occurred_at`).
- **khora-ops / transaction time** - when khora persisted the row. Carried by
  `created_at` / `updated_at`. `created_at` records when the row was created and
  never changes; it is *not* derived from `source_timestamp`.

```python
# The caller-supplied event time lands on its own column. created_at /
# updated_at fall to the model default datetime.now(UTC) - ingest time.
document = Document(
    content=content,
    source_timestamp=source_timestamp,  # e.g., email sent_at (real-world)
)

# Chunks carry the same real-world event time via source_timestamp; their
# created_at stays ingest time (khora-ops).
chunk = Chunk(
    document_id=document.id,
    content=chunk_content,
    source_timestamp=document.source_timestamp,
)
```

Temporal recall windows over the original event time (`occurred_at` /
`start_time` / `end_time`) read `source_timestamp`, so accurate
"when did this happen" queries work without overwriting the ingest-time
`created_at`.

## Document-Chunk Relationship

```text
Document
    │
    ├── Chunk 0 (chunk_index=0, start_char=0, end_char=512)
    │       │
    │       └── embedding: [0.12, -0.34, ...]
    │
    ├── Chunk 1 (chunk_index=1, start_char=462, end_char=974)  ← overlap
    │       │
    │       └── embedding: [0.08, -0.22, ...]
    │
    └── Chunk 2 (chunk_index=2, start_char=924, end_char=1250)
            │
            └── embedding: [0.15, -0.41, ...]
```

### Chunk Overlap

Chunks typically overlap to preserve context at boundaries:

```python
# Example: 512 token chunks with 50 token overlap
chunk_size = 512
chunk_overlap = 50

# Chunk 0: tokens 0-511
# Chunk 1: tokens 462-973 (50 token overlap)
# Chunk 2: tokens 924-1435 (50 token overlap)
```

## API Usage

### Creating Documents

```python
from khora import Khora

async with Khora() as kb:
    ns = await kb.create_namespace()
    result = await kb.remember(
        "Einstein developed the theory of relativity in 1905.",
        namespace=ns.namespace_id,
        title="Physics History",
        source="wikipedia",
        metadata={"topic": "physics"},
        entity_types=["PERSON", "ORG"],
        relationship_types=["MENTIONS"],
    )

    print(f"Document ID: {result.document_id}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Entities extracted: {result.entities_extracted}")
```

### Retrieving Documents

```python
# Get document by ID. `namespace_id` is required and kwarg-only -
# returns None if the document is not in this namespace.
doc = await kb.storage.get_document(document_id, namespace_id=namespace_id)
print(f"Status: {doc.status}")
print(f"Chunk count: {doc.chunk_count}")

# Get chunks for a document, scoped to the caller's namespace.
chunks = await kb.storage.get_chunks_by_document(
    document_id,
    namespace_id=namespace_id,
)
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.content[:100]}...")
```

### Batch Document Creation

```python
documents = [
    {"content": "First document...", "title": "Doc 1"},
    {"content": "Second document...", "title": "Doc 2"},
    {"content": "Third document...", "title": "Doc 3"},
]

result = await kb.remember_batch(
    documents,
    namespace=ns.namespace_id,
    skill_name="general_entities",
    max_concurrent=10,
    entity_types=["PERSON", "ORG"],
    relationship_types=["MENTIONS"],
)

print(f"Processed: {result.processed}")
print(f"Skipped (duplicates): {result.skipped}")
```

### Forgetting Documents

```python
# Remove document and all associated data
await kb.forget(document_id, namespace=ns.namespace_id)

# This removes:
# - The document
# - All chunks
# - Entity source references (entities with no sources are removed)
# - Relationship source references
```

## Chunking Strategies

Documents are chunked using configurable strategies:

| Strategy | Description |
|----------|-------------|
| `fixed` | Fixed token size with overlap |
| `semantic` | Sentence/paragraph boundaries |
| `recursive` | Hierarchical splitting |
| `conversation` | Groups messages into coherent conversation chunks (Slack and similar) |

See [Chunkers](../extraction/chunkers.md) for details.

## Storage

Documents are stored in PostgreSQL:

The DDL below is illustrative and abbreviated. The `DocumentModel` ORM (`db/models.py`) carries additional columns: typed provenance (`source_type`, `source_name`, `source_url`, `content_type`, `title`, `author`, `language`, `size_bytes`), `error_message`, `extraction_config_hash`, `extraction_params`, `relationship_count`, `source_timestamp`, `session_id` (#620), and `graph_mirror_pending JSONB` (migration 051, #1430 - the replace-path graph plan; NULL when the graph is in lockstep with PG).

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    content TEXT,
    external_id VARCHAR(512),      -- Caller-supplied source identity
    status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB,
    checksum VARCHAR(64),          -- SHA-256 for deduplication
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    session_id UUID,               -- #620 session-scoped recall
    graph_mirror_pending JSONB,    -- migration 051 (#1430); NULL = graph in lockstep
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_documents_namespace ON documents(namespace_id);
CREATE INDEX idx_documents_status ON documents(namespace_id, status);
CREATE INDEX ix_documents_namespace_checksum ON documents(namespace_id, checksum);
CREATE UNIQUE INDEX ix_documents_namespace_external_id_unique
    ON documents(namespace_id, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX ix_documents_graph_mirror_pending
    ON documents(namespace_id) WHERE graph_mirror_pending IS NOT NULL;
```

Chunks with embeddings are stored in pgvector. Again abbreviated - the `ChunkModel` ORM also has `chunker_info JSONB` (migration 038), `embedding_model`, a generated `content_tsv TSVECTOR` + GIN index (039), `source_timestamp`, `last_accessed_at` (040), `occurred_at` (046), and `session_id` (030):

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    namespace_id UUID NOT NULL,
    content TEXT,
    embedding vector(1536),
    chunk_index INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    token_count INTEGER,
    metadata JSONB,
    chunker_info JSONB,            -- migration 038
    occurred_at TIMESTAMPTZ,       -- migration 046 event-time anchor
    session_id UUID,               -- migration 030
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
-- Full-precision HNSW index (migrations 002/007):
CREATE INDEX ix_chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);
-- The operative index at recall time is the halfvec (float16) expression
-- index from migration 018, which the ORDER BY casts to match (#1423):
CREATE INDEX ix_chunks_embedding_halfvec_hnsw ON chunks
    USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
    WITH (m = 24, ef_construction = 128);
```

> **Two `chunks`-shaped tables exist.** The `chunks` table above is the pgvector-backend ORM table. Migrations named `NNN_khora_chunks_*` (038, 039, 041-044) target a *separate*, runtime-created (not Alembic-managed) `khora_chunks` table owned by the VectorCypher `PgVectorTemporalStore`, which carries denormalized document-grained columns (`source_type` / `source_name` / `source_url` / `source_timestamp` / `external_id` / `content_type` / `source` / `title`, migration 041). The `Chunk` dataclass and the `chunks` ORM table do not carry those denormalized fields.

## Next Steps

- [Knowledge Graph](knowledge-graph.md) - Entities and relationships
- [Chunkers](../extraction/chunkers.md) - Chunking strategies
- [Embedders](../extraction/embedders.md) - Embedding generation
