# Data Models Overview

Khora's data models represent the core domain concepts for knowledge management. This document provides an overview of model relationships and their purposes.

## Model Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Domain Models                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Tenancy Layer                                   ││
│  │                                                                          ││
│  │   Organization ──┬── Workspace ──┬── Namespace                          ││
│  │                  │               │                                       ││
│  │                  └── Workspace   └── Namespace (versioned)              ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Content Layer                                   ││
│  │                                                                          ││
│  │   Document ──┬── Chunk ──────────────────────────┐                       ││
│  │   (source)   │   (text + embedding)              │                       ││
│  │              │                                   │                       ││
│  │              └── Chunk                           │                       ││
│  │                                                  │                       ││
│  │              ┌───────────────────────────────────┘                       ││
│  │              ▼                                                           ││
│  │   Entity ───┬── Relationship ─── Entity                                  ││
│  │   (node)    │   (edge)            (node)                                 ││
│  │             │                                                            ││
│  │             └── Episode ────────── Entity                                ││
│  │                 (temporal event)                                         ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Event Layer                                     ││
│  │                                                                          ││
│  │   MemoryEvent ─── (append-only log of all changes)                       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Models

### Document

The source content container. Documents are the primary input to the memory lake.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `namespace_id` | UUID | Owning namespace |
| `content` | str | Source text content |
| `metadata` | DocumentMetadata | Source, title, type, checksum |
| `status` | DocumentStatus | pending, processing, completed, failed |
| `chunk_count` | int | Number of chunks created |
| `entity_count` | int | Number of entities extracted |
| `created_at` | datetime | When document was added |

See [Documents & Chunks](documents-chunks.md) for details.

### Chunk

A segment of a document, sized for embedding and retrieval.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `document_id` | UUID | Parent document |
| `namespace_id` | UUID | Owning namespace |
| `content` | str | Chunk text |
| `embedding` | list[float] | Vector embedding |
| `index` | int | Position in document |
| `token_count` | int | Token count for sizing |
| `created_at` | datetime | When chunk was created |

See [Documents & Chunks](documents-chunks.md) for details.

### Entity

A named concept extracted from content (person, organization, concept, etc.).

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `namespace_id` | UUID | Owning namespace |
| `name` | str | Entity name |
| `entity_type` | EntityType | PERSON, ORGANIZATION, etc. |
| `description` | str | Entity description |
| `attributes` | dict | Arbitrary key-value attributes |
| `embedding` | list[float] | Vector embedding |
| `confidence` | float | Extraction confidence (0-1) |
| `valid_from` / `valid_until` | datetime | Temporal validity |

See [Knowledge Graph](knowledge-graph.md) for details.

### Relationship

A typed edge between two entities.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `namespace_id` | UUID | Owning namespace |
| `source_entity_id` | UUID | Source entity |
| `target_entity_id` | UUID | Target entity |
| `relationship_type` | RelationshipType | WORKS_FOR, KNOWS, etc. |
| `weight` | float | Relationship strength (0-1) |
| `properties` | dict | Edge properties |
| `valid_from` / `valid_until` | datetime | Temporal validity |

See [Knowledge Graph](knowledge-graph.md) for details.

### Episode

A temporal event with associated entities and duration.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `namespace_id` | UUID | Owning namespace |
| `name` | str | Episode name |
| `description` | str | What happened |
| `occurred_at` | datetime | When it happened |
| `duration_seconds` | int | Event duration |
| `entity_ids` | list[UUID] | Participating entities |

See [Knowledge Graph](knowledge-graph.md) for details.

### MemoryEvent

An immutable record of a change to the memory lake.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique event ID |
| `namespace_id` | UUID | Namespace scope |
| `event_type` | EventType | document.created, entity.merged, etc. |
| `resource_type` | str | Affected resource type |
| `resource_id` | UUID | Affected resource ID |
| `data` | dict | Event payload |
| `actor_id` | str | Who triggered the event |
| `correlation_id` | UUID | Links related events |

See [Events](events.md) for details.

## Model Relationships

```
Namespace (1)
    │
    ├──────────────── Document (n)
    │                      │
    │                      └── Chunk (n per doc)
    │                            │
    │                            └── (generates embeddings)
    │
    ├──────────────── Entity (n)
    │                      │
    │                      ├── Relationship (n) ── Entity
    │                      │
    │                      └── (participates in) Episode
    │
    └──────────────── MemoryEvent (n)
                           │
                           └── (tracks all changes)
```

### Source Tracking

All extracted entities and relationships track their sources:

```python
@dataclass
class Entity:
    # ...
    source_document_ids: list[UUID]  # Documents that mention this entity
    source_chunk_ids: list[UUID]     # Specific chunks

@dataclass
class Relationship:
    # ...
    source_document_ids: list[UUID]
    source_chunk_ids: list[UUID]
```

This enables:
- Provenance tracking
- Source citation in responses
- Cascading deletes when documents are removed

## Status Lifecycle

### Document Status

```
PENDING → PROCESSING → COMPLETED
              │
              └──────── FAILED
```

| Status | Description |
|--------|-------------|
| `PENDING` | Awaiting processing |
| `PROCESSING` | Currently being chunked/embedded/extracted |
| `COMPLETED` | Successfully processed |
| `FAILED` | Processing error (error message in metadata) |

## Timestamps

All models track temporal information:

```python
created_at: datetime   # When created (immutable)
updated_at: datetime   # When last modified

# For entities and relationships (optional):
valid_from: datetime   # When this became true
valid_until: datetime  # When this stopped being true
```

## Next Steps

- [Documents & Chunks](documents-chunks.md) - Content storage
- [Knowledge Graph](knowledge-graph.md) - Entities and relationships
- [Events](events.md) - Audit trail
