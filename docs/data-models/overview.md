# Data Models Overview

Khora's data models represent the things you care about: documents you've stored, concepts you've extracted, relationships you've discovered, and the history of how it all evolved.

## The Model Landscape

Everything in Khora fits into one of three layers:

```text
+-----------------------------------------------------------+
|                      TENANCY LAYER                        |
|                                                           |
|   Namespace A  (your data lives here)                     |
|   Namespace B  (another dataset)                          |
|   Namespace A' (version 2 of A, for zero-downtime swaps)  |
|                                                           |
|   Each namespace has two IDs:                             |
|     namespace_id - stable across versions                 |
|     id - row-level, changes per version                   |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|                      CONTENT LAYER                        |
|                                                           |
|   Document ----+---- Chunk ----+---- Entity               |
|   (the source) |    (pieces)   |       (concepts)         |
|                |               |                          |
|                +---- Chunk     +---- Relationship         |
|                      |               (connections)        |
|                      +---- Chunk                          |
|                                +---- Episode              |
|                                      (events)             |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|                       EVENT LAYER                         |
|                                                           |
|   MemoryEvent  (immutable log of everything that happens) |
+-----------------------------------------------------------+
```

## Content Models

### Document

A document is the raw content you store - the starting point for everything.

```python
from khora.core.models import Document

Document(
    id=UUID("..."),
    namespace_id=UUID("..."),
    content="Einstein published his theory...",
    title="Physics History",
    source="upload",
    content_type="text/plain",
    checksum="sha256:abc123...",
    status=DocumentStatus.COMPLETED,
    chunk_count=5,
    entity_count=12,
    created_at=datetime(2024, 1, 15, 10, 30),
)
```

**Key fields:**
- `content` - The actual text
- `checksum` - SHA-256 hash for deduplication
- `status` - Where it is in processing (PENDING → PROCESSING → COMPLETED or FAILED)
- `chunk_count`, `entity_count` - Summary stats after processing

**Lifecycle:**
```
PENDING        Just created, waiting for processing
    |
    v
PROCESSING     Being chunked, embedded, extracted
    |
    +---> COMPLETED    Successfully processed
    |
    +---> FAILED       Error occurred (message in metadata)
```

### Chunk

Chunks are document pieces optimized for embedding and retrieval.

```python
Chunk(
    id=UUID("..."),
    document_id=UUID("..."),    # Parent document
    namespace_id=UUID("..."),
    content="...portion of text...",
    embedding=[0.021, -0.156, ...],  # 1536 floats
    embedding_model="text-embedding-3-small",
    index=2,                    # Third chunk in document
    start_char=1024,            # Character offset
    end_char=1536,
    token_count=512,
    created_at=datetime(2024, 1, 15, 10, 31)
)
```

**Key fields:**
- `embedding` - Vector representation for similarity search
- `index` - Position in parent document (for context)
- `start_char`, `end_char` - Character offsets (for highlighting)
- `token_count` - Useful for understanding chunk sizes

### Entity

Entities are named concepts extracted from your content.

```python
Entity(
    id=UUID("..."),
    namespace_id=UUID("..."),
    name="Albert Einstein",
    entity_type="PERSON",
    description="German-born theoretical physicist",
    attributes={
        "birth_year": 1879,
        "known_for": ["relativity", "quantum mechanics"]
    },
    embedding=[...],           # For similarity search
    confidence=0.95,           # Extraction confidence
    valid_from=datetime(1879, 3, 14),
    valid_until=datetime(1955, 4, 18),
    source_document_ids=[...], # Where we learned this
    source_chunk_ids=[...],
    mention_count=15           # How often mentioned
)
```

**Built-in entity types:**
- `PERSON` - People
- `ORGANIZATION` - Companies, institutions
- `LOCATION` - Places
- `PRODUCT` - Products, services
- `CONCEPT` - Abstract ideas
- `EVENT` - Named events
- `TECHNOLOGY` - Technologies, tools
- `CUSTOM` - Your own types

**Temporal validity**: Entities can have time bounds. Albert Einstein is valid from birth to death. A company's name might change. This enables temporal queries.

### Relationship

Relationships connect entities with typed edges.

```python
Relationship(
    id=UUID("..."),
    namespace_id=UUID("..."),
    source_entity_id=UUID("..."),  # From entity
    target_entity_id=UUID("..."),  # To entity
    relationship_type="WORKS_FOR",
    properties={
        "role": "Chief Scientist",
        "start_date": "2020-01-15"
    },
    weight=0.9,                    # Strength (0-1)
    confidence=0.85,               # Extraction confidence
    valid_from=datetime(2020, 1, 15),
    source_document_ids=[...],
    source_chunk_ids=[...]
)
```

**Built-in relationship types:**
- `WORKS_FOR` - Employment
- `KNOWS` - Personal connection
- `PART_OF` - Membership, containment
- `RELATED_TO` - General association
- `CREATED` - Authorship, invention
- `LOCATED_IN` - Physical location
- `OWNS` - Ownership
- `COLLABORATES_WITH` - Collaboration

### Episode

Episodes are events with temporal extent.

```python
Episode(
    id=UUID("..."),
    namespace_id=UUID("..."),
    name="Product Launch",
    description="Launch of version 2.0 at the annual conference",
    occurred_at=datetime(2024, 3, 15, 9, 0),
    duration_seconds=7200,  # 2 hours
    entity_ids=[product_id, conference_id, ceo_id],
    source_document_ids=[...],
    source_chunk_ids=[...]
)
```

Episodes connect multiple entities to a point (or span) in time.

## The Source Chain

One crucial feature: everything tracks where it came from.

```text
Document "Meeting Notes"
     |
     +-- Chunk #1 ----+
     |                |
     +-- Chunk #2 ----+-- Entity "Alice" (mentioned in chunks 1, 2, 3)
     |                |
     +-- Chunk #3 ----+-- Relationship "Alice WORKS_FOR Acme"
```

Every entity and relationship remembers:
- `source_document_ids` - Which documents mentioned it
- `source_chunk_ids` - Which specific chunks

This enables:
- **Provenance** - "Where did we learn this?"
- **Citation** - "Here's the source for this claim"
- **Cascading deletes** - Delete a document, its entities/relationships update

## Event Models

### MemoryEvent

Every change is recorded as an immutable event:

```python
MemoryEvent(
    id=UUID("..."),
    namespace_id=UUID("..."),
    event_type="document.created",
    resource_type="document",
    resource_id=UUID("..."),
    data={
        "title": "Meeting Notes",
        "source": "upload",
        "size_bytes": 15234
    },
    actor_id="user:alice",
    actor_type="user",
    correlation_id=UUID("..."),  # Links related events
    timestamp=datetime(2024, 1, 15, 10, 30)
)
```

**Event types span the lifecycle:**

| Category | Event Types |
|----------|-------------|
| Document | `document.created`, `document.updated`, `document.deleted`, `document.processing_started`, `document.processing_completed`, `document.processing_failed` |
| Chunk | `chunk.created`, `chunk.deleted`, `chunk.embedding_generated` |
| Entity | `entity.created`, `entity.updated`, `entity.deleted`, `entity.merged` |
| Relationship | `relationship.created`, `relationship.updated`, `relationship.deleted`, `relationship.inferred` |
| Namespace | `namespace.created`, `namespace.activated`, `namespace.archived` |

**Correlation IDs** link related events. When you call `remember()`, a single correlation ID ties together all the events it generates.

## Model Relationships Summary

```text
Namespace
    |
    |-- has many --> Document
    |                    |
    |                    +-- has many --> Chunk
    |
    |-- has many --> Entity
    |                    |
    |                    +-- source --> Relationship --> target --> Entity
    |                    |
    |                    +-- participates in --> Episode
    |
    |-- has many --> MemoryEvent
```

## Working with Models

### Creating and Storing

```python
from khora.core.models import Document, DocumentStatus

doc = Document(
    id=uuid4(),
    namespace_id=namespace_id,
    content="Your content here",
    title="My Doc",
    status=DocumentStatus.PENDING,
)

await storage.create_document(doc)
```

### Querying

```python
# Get a document. `namespace_id` is required and kwarg-only - the lookup
# returns None if the id belongs to a different namespace.
doc = await storage.get_document(doc_id, namespace_id=namespace_id)

# Find entities by type (list-style scan; namespace_id is positional).
entities = await storage.list_entities(
    namespace_id,
    entity_type="PERSON",
    limit=50,
)

# Get relationships for an entity, scoped to the caller's namespace.
relationships = await storage.get_entity_relationships(
    entity_id, namespace_id=namespace_id,
)
```

### Timestamps

All models track time:
- `created_at` - When created (never changes)
- `updated_at` - When last modified

Entities and relationships can also have:
- `valid_from` - When this became true in the real world
- `valid_until` - When it stopped being true

## What's Next?

- **[Documents & Chunks](documents-chunks.md)** - Content storage in depth
- **[Knowledge Graph](knowledge-graph.md)** - Entities, relationships, episodes
- **[Events](events.md)** - The immutable audit trail
