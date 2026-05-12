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
    metadata: DocumentMetadata
    status: DocumentStatus = DocumentStatus.PENDING

    # Processing stats
    chunk_count: int = 0
    entity_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
```

### Document Metadata

```python
@dataclass
class DocumentMetadata:
    # Source information
    source: str = ""           # "slack/channel-123", "notion/page-id"
    source_type: str = "manual"  # manual, api, sync
    content_type: str = "text/plain"

    # Document info
    title: str = ""
    author: str = ""
    language: str = "en"

    # Integrity
    checksum: str = ""         # SHA-256 for deduplication
    size_bytes: int = 0

    # Custom metadata
    custom: dict[str, Any] = field(default_factory=dict)
```

### Document Status

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"        # Awaiting processing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"    # Successfully processed
    FAILED = "failed"          # Processing failed
```

### Status Transitions

```python
# Mark as processing
document.mark_processing()
# status → PROCESSING, updated_at → now

# Mark as completed
document.mark_completed(chunk_count=5, entity_count=10)
# status → COMPLETED, chunk_count=5, entity_count=10
# processed_at → now, updated_at → now

# Mark as failed
document.mark_failed("Error: LLM timeout")
# status → FAILED, metadata.custom["error"] = "Error: LLM timeout"
# updated_at → now
```

### Checksum Deduplication

Documents are deduplicated by content checksum:

```python
# During ingestion
checksum = hashlib.sha256(content.encode()).hexdigest()

# Check if identical document exists
existing = await storage.get_document_by_checksum(namespace_id, checksum)
if existing:
    # Skip - document already exists
    return None
```

## Chunk Model

Chunks are segments of documents optimized for embedding and retrieval.

```python
@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    namespace_id: UUID
    content: str

    # Embedding
    embedding: list[float] | None = None

    # Position in document
    index: int = 0             # 0-based chunk index
    start_char: int = 0        # Character offset start
    end_char: int = 0          # Character offset end

    # Size info
    token_count: int = 0       # Token count (for chunking decisions)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Chunk Metadata

Chunks inherit document metadata and add chunk-specific info:

```python
chunk.metadata = {
    # Inherited from document
    "source": "slack/channel-123",
    "title": "Team Discussion",

    # Chunk-specific
    "chunker": "semantic",
    "overlap_tokens": 50,
}
```

## Timestamp Inheritance

Chunks inherit timestamps from their parent documents:

```python
# When document is created with source timestamp
document = Document(
    content=content,
    created_at=source_timestamp,  # e.g., email sent_at
    updated_at=source_timestamp,
)

# Chunks inherit the document's timestamp
chunk = Chunk(
    document_id=document.id,
    content=chunk_content,
    created_at=document.created_at,  # Same as document
    updated_at=document.created_at,
)
```

This enables accurate temporal queries based on when content was originally created, not when it was ingested.

## Document-Chunk Relationship

```
Document
    │
    ├── Chunk 0 (index=0, start_char=0, end_char=512)
    │       │
    │       └── embedding: [0.12, -0.34, ...]
    │
    ├── Chunk 1 (index=1, start_char=462, end_char=974)  ← overlap
    │       │
    │       └── embedding: [0.08, -0.22, ...]
    │
    └── Chunk 2 (index=2, start_char=924, end_char=1250)
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
    result = await kb.remember(
        "Einstein developed the theory of relativity in 1905.",
        title="Physics History",
        source="wikipedia",
        metadata={"topic": "physics"},
    )

    print(f"Document ID: {result.document_id}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Entities extracted: {result.entities_extracted}")
```

### Retrieving Documents

```python
# Get document by ID
doc = await kb.storage.get_document(document_id)
print(f"Status: {doc.status}")
print(f"Chunk count: {doc.chunk_count}")

# Get chunks for a document
chunks = await kb.storage.get_document_chunks(document_id)
for chunk in chunks:
    print(f"Chunk {chunk.index}: {chunk.content[:100]}...")
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
    skill_name="general_entities",
    max_concurrent=10,
)

print(f"Processed: {result['processed_documents']}")
print(f"Skipped (duplicates): {result['skipped_documents']}")
```

### Forgetting Documents

```python
# Remove document and all associated data
await kb.forget(document_id)

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

See [Chunkers](../extraction/chunkers.md) for details.

## Storage

Documents are stored in PostgreSQL:

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    content TEXT,
    external_id VARCHAR(512),      -- Caller-supplied source identity
    status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB,
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_documents_namespace ON documents(namespace_id);
CREATE INDEX idx_documents_status ON documents(namespace_id, status);
CREATE INDEX idx_documents_checksum ON documents(namespace_id, (metadata->>'checksum'));
CREATE UNIQUE INDEX ix_documents_namespace_external_id_unique
    ON documents(namespace_id, external_id) WHERE external_id IS NOT NULL;
```

Chunks with embeddings are stored in pgvector:

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    namespace_id UUID NOT NULL,
    content TEXT,
    embedding vector(1536),
    index INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    token_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

## Next Steps

- [Knowledge Graph](knowledge-graph.md) - Entities and relationships
- [Chunkers](../extraction/chunkers.md) - Chunking strategies
- [Embedders](../extraction/embedders.md) - Embedding generation
