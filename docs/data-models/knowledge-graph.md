# Knowledge Graph

Khora extracts entities and relationships from documents to build a knowledge graph. This graph enables relationship-based queries, entity exploration, and contextual search enhancement.

## Storage Architecture

Entities are stored in **two backends** for different purposes:

| Backend | Purpose | Data |
|---------|---------|------|
| **Neo4j** | Graph traversal, relationship queries | Entity nodes, relationship edges |
| **PostgreSQL/pgvector** | Embedding similarity search | Entity records with vector embeddings |

This dual storage enables:
- **Graph queries**: Traverse relationships efficiently in Neo4j
- **Entity similarity**: Find semantically related entities via embedding search
- **Hybrid search**: Combine both methods for comprehensive retrieval

## Entity Model

Located at `src/khora/core/models/entity.py`.

```python
@dataclass
class Entity:
    id: UUID
    namespace_id: UUID
    name: str
    entity_type: str = "CONCEPT"
    description: str = ""

    # Attributes from extraction
    attributes: dict[str, Any] = field(default_factory=dict)

    # Embedding for similarity matching (stored in pgvector)
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Confidence and mentions
    confidence: float = 1.0
    mention_count: int = 1

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Temporal validity
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Entity Types

Entity types are plain strings. Callers define their own ontology - Khora does not prescribe a fixed set. Common examples:

- `"PERSON"` - Individual people
- `"ORGANIZATION"` - Companies, institutions
- `"LOCATION"` - Places, addresses
- `"CONCEPT"` - Abstract ideas, theories
- `"EVENT"` - Occurrences, incidents
- `"TECHNOLOGY"` - Tools, platforms, languages

### Entity Attributes

Entities store arbitrary attributes as key-value pairs:

```python
entity = Entity(
    name="Albert Einstein",
    entity_type="PERSON",
    attributes={
        "role": "Physicist",
        "email": "einstein@princeton.edu",
        "birth_year": 1879,
        "nationality": "German-American",
    },
)
```

### Entity Merging

When duplicate entities are detected, they are merged:

```python
def merge_with(self, other: Entity) -> None:
    """Merge another entity into this one (deduplication)."""
    # Combine source references
    for doc_id in other.source_document_ids:
        if doc_id not in self.source_document_ids:
            self.source_document_ids.append(doc_id)
    for chunk_id in other.source_chunk_ids:
        if chunk_id not in self.source_chunk_ids:
            self.source_chunk_ids.append(chunk_id)

    # Update mention count
    self.mention_count += other.mention_count

    # Merge attributes (prefer existing; only fill in missing keys)
    for key, value in other.attributes.items():
        if key not in self.attributes:
            self.attributes[key] = value

    # Update confidence (take max)
    self.confidence = max(self.confidence, other.confidence)

    # Update description if empty
    if not self.description and other.description:
        self.description = other.description

    self.updated_at = datetime.now(UTC)
```

### Temporal Validity

Entities can have temporal bounds indicating when they were valid:

```python
# Entity valid from 2020 to 2023
entity = Entity(
    name="Acme CEO",
    entity_type="PERSON",
    valid_from=datetime(2020, 1, 1),
    valid_until=datetime(2023, 12, 31),
)
```

This enables temporal queries like "Who was the CEO in 2022?"

### Entity Embeddings

Entities have vector embeddings for similarity search:

```python
entity = Entity(
    name="Albert Einstein",
    entity_type="PERSON",
    description="Theoretical physicist known for relativity",
    embedding=[0.1, 0.2, ...],  # Generated from name + description
    embedding_model="text-embedding-3-small",
)
```

Embeddings are:
- **Generated automatically** during ingestion from `{name}: {description}`
- **Stored in pgvector** (PostgreSQL) for cosine similarity search
- **Used by graph search** to find relevant entities before traversal

To find similar entities:

```python
# Search for entities similar to query
similar_entities = await storage.search_similar_entities(
    namespace_id,
    query_embedding,
    limit=10,
    min_similarity=0.3,
)
```

## Relationship Model

```python
@dataclass
class Relationship:
    id: UUID
    namespace_id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str = "RELATES_TO"

    # Relationship metadata
    description: str = ""
    weight: float = 1.0           # Relationship strength (0-1)
    properties: dict[str, Any] = field(default_factory=dict)

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Temporal validity
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadata
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Populated at recall time (denormalized endpoint names for display)
    source_entity_name: str | None = None
    target_entity_name: str | None = None
```

`source_entity_name` / `target_entity_name` are populated by the recall path so callers can render an edge without a second entity lookup.

### Bi-temporal soft-delete columns (read-active since #1272)

The `RelationshipModel` ORM (not the dataclass above) carries three tombstone columns added in migration 033 (#653): `valid_to`, `invalidated_at`, `invalidated_by`. These are distinct from the `valid_until` temporal window - non-NULL means the edge is dead. They are **active on read**: pgvector's `_relationship_live_filter()` filters `valid_to IS NULL AND invalidated_at IS NULL AND (valid_until IS NULL OR valid_until > now())`, and Neo4j filters `valid_until` in lockstep (dream-apply mirrors the three PG tombstones onto the graph's single `valid_until`). Entities use only the `valid_until` window (`_entity_live_filter()`), not the tombstone triple. These live on the ORM model; the lean `Relationship` dataclass above intentionally does not carry them.

### Relationship Types

Relationship types are plain strings. Callers define their own ontology. Common examples:

- `"WORKS_FOR"`, `"MANAGES"`, `"REPORTS_TO"` - Organizational
- `"KNOWS"`, `"COLLABORATES_WITH"` - Social
- `"PART_OF"`, `"CONTAINS"` - Composition
- `"LOCATED_IN"` - Location
- `"DEPENDS_ON"`, `"IMPLEMENTS"` - Technical
- `"RELATES_TO"`, `"ASSOCIATED_WITH"` - Generic

### Relationship Properties

```python
relationship = Relationship(
    source_entity_id=person_id,
    target_entity_id=company_id,
    relationship_type="WORKS_FOR",
    properties={
        "start_date": "2020-01-15",
        "title": "Senior Engineer",
        "department": "Engineering",
    },
    weight=0.95,
    valid_from=datetime(2020, 1, 15),
)
```

## Episode Model

Episodes represent temporal events with associated entities.

```python
@dataclass
class Episode:
    id: UUID
    namespace_id: UUID
    name: str
    description: str = ""

    # Temporal
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: int | None = None

    # Participants
    entity_ids: list[UUID] = field(default_factory=list)

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Embedding for similarity search
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Episode Examples

```python
# Meeting episode
meeting = Episode(
    name="Q4 Planning Meeting",
    description="Quarterly planning discussion",
    occurred_at=datetime(2024, 10, 15, 14, 0),
    duration_seconds=3600,  # 1 hour
    entity_ids=[person1_id, person2_id, project_id],
    metadata={"meeting_type": "planning"},
)

# Event episode
launch = Episode(
    name="Product Launch",
    description="Launch of Product X",
    occurred_at=datetime(2024, 12, 1),
    entity_ids=[product_id, company_id],
    metadata={"event_type": "launch"},
)
```

## Graph Structure

### Neo4j Representation

Entities become nodes, relationships become edges:

```cypher
// Entity nodes
(:Entity {
    id: "uuid-1",
    namespace_id: "ns-uuid",
    name: "Einstein",
    entity_type: "PERSON",
    description: "Theoretical physicist",
    confidence: 0.95,
    mention_count: 5
})

// Relationships as edges
(:Entity {name: "Einstein"})-[:WORKS_FOR {
    weight: 0.9,
    valid_from: datetime("1933-10-17")
}]->(:Entity {name: "Princeton University"})

// Episode nodes
(:Episode {
    id: "episode-uuid",
    name: "Nobel Prize Award",
    occurred_at: datetime("1921-11-09")
})-[:INVOLVES]->(:Entity {name: "Einstein"})
```

The Cypher above is illustrative. In practice `:Entity` nodes and their edges also carry a `valid_until` property that the graph-side recall filter uses (`neo4j.py`) - nodes/edges shown without it read as always-live. Beyond the caller-defined semantic verbs, the graph carries several system verbs and node labels:

- `CO_OCCURS_WITH` - the generic co-occurrence edge written by KET-RAG selective extraction and by expansion-time relationship inference.
- `[:SUPERSEDES]` edges pointing at `:EntityVersion` snapshot nodes - traversed by CHANGE / point-in-time recall to reconstruct an entity's history.
- `:Community` nodes with `[:HAS_MEMBER]` edges - written by the dream community-summary op (#1276).

## Source Tracking

All graph elements track their sources:

```python
# Entity extracted from multiple chunks
entity = Entity(
    name="Einstein",
    source_document_ids=[doc1_id, doc2_id],
    source_chunk_ids=[chunk1_id, chunk5_id, chunk12_id],
)

# Relationship from specific chunk
relationship = Relationship(
    source_entity_id=einstein_id,
    target_entity_id=princeton_id,
    relationship_type="WORKS_FOR",
    source_document_ids=[doc1_id],
    source_chunk_ids=[chunk3_id],
)
```

This enables:
- Citing sources in responses
- Cascading deletes when documents are removed
- Confidence based on number of mentions

## API Usage

### Listing Entities

```python
from khora import Khora

async with Khora() as kb:
    # `namespace` is required - there is no default namespace.
    entities = await kb.list_entities(namespace=namespace_id)

    # Filter by type
    people = await kb.list_entities(namespace=namespace_id, entity_type="PERSON")

    # Larger result page
    entities = await kb.list_entities(namespace=namespace_id, limit=100)
```

### Finding Related Entities

```python
# Get entities related to a specific entity. `namespace` is required -
# the traversal is scoped to the caller's namespace.
related = await kb.find_related_entities(
    entity_id,
    namespace=namespace_id,
    max_depth=2,      # Traverse up to 2 hops
    limit=20,
)

for entity, score in related:
    print(f"{entity.name}: {score:.2f}")
```

### Graph Traversal

```python
# Get entity neighborhood (graph context). namespace_id is required and
# kwarg-only - the traversal does not cross into other namespaces.
neighborhood = await kb.storage.get_neighborhood(
    entity_id,
    namespace_id=namespace_id,
    depth=2,
    relationship_types=["WORKS_FOR", "MANAGES"],
    limit=50,
)

# neighborhood contains:
# - entities: list of related entities
# - relationships: edges between them
# - paths: traversal paths from source entity
```

### Creating Entities Manually

```python
from khora.core.models import Entity

entity = Entity(
    namespace_id=namespace_id,
    name="Acme Corporation",
    entity_type="ORGANIZATION",
    attributes={"industry": "Technology"},
)

await kb.storage.create_entity(entity)
```

## Entity Unique Constraint

Entities have a UNIQUE constraint on `(namespace_id, name, entity_type)` (added in migration 008). This enables efficient `ON CONFLICT` upserts:

```sql
INSERT INTO entities (id, namespace_id, name, entity_type, ...)
VALUES (...)
ON CONFLICT (namespace_id, name, entity_type)
DO UPDATE SET description = EXCLUDED.description, ...
```

The dedup migration is irreversible. All entity upserts use this constraint for atomic create-or-update semantics.

**Canonical-id sync on re-mention (#1429).** A re-mentioned entity arrives with a throwaway extraction-time UUID that never lands in the table: `ON CONFLICT (namespace_id, name, entity_type) DO UPDATE` keeps the *existing* row's `id`, and `upsert_entities_batch` syncs each input entity's `id` in place to the persisted canonical id (derived from the `RETURNING` row on pgvector; Neo4j and sqlite_lance apply the same remap). This matters because callers build relationship endpoints from `entity.id` - without the sync those FKs would point at the throwaway UUID and abort ingest. `is_new` is read from Postgres `xmax = 0` in the same `RETURNING`.

## Deduplication

Entities are deduplicated during ingestion. The approach depends on the inference mode.

### Per-Entity Dedup (Incremental/Batch Modes)

```python
existing = await storage.get_entity_by_name(
    namespace_id,
    entity.name,
    entity.entity_type,
)

if existing:
    existing.merge_with(entity)
    await storage.update_entity(existing)
else:
    await storage.create_entity(entity)
```

### Index-Based Dedup (Smart Mode)

In smart mode (the default), a shared in-memory `EntityIndex` handles dedup without database round-trips:

```python
from khora.extraction.expansion import EntityIndex

index = EntityIndex()

for entity in extracted_entities:
    existing = index.add(entity)      # O(1) exact match
    if existing is not None:
        existing.merge_with(entity)   # Merge in-memory
    # No database call needed per entity
```

After all documents are processed, entities are written to storage in batches using `upsert_entities_batch()`, and cross-document fuzzy/embedding resolution runs via token-blocked candidate matching.

For batch deduplication, the Rust `resolve_entities_batch` function provides a 3-stage cascade: exact name match → alias match → fuzzy Levenshtein, with Rayon parallelism for large batches (≥512 entities).

See [Semantic Expansion](../extraction/semantic-expansion.md) for full details on the `EntityIndex`, token blocking, and the smart mode resolution pipeline.

## Next Steps

- [Events](events.md) - Event sourcing for graph changes
- [Extractors](../extraction/extractors.md) - Entity extraction
- [Semantic Expansion](../extraction/semantic-expansion.md) - Entity unification
