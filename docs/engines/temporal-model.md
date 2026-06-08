# Temporal Model

> **Status:** The bi-temporal edge storage (`TemporalEdgeStorage`) and time hierarchy (`TimeHierarchyBuilder`) described here exist as code but are **never called** by any engine's ingest or recall paths. The `occurred_at` column on chunks is populated and filterable via the pgvector backend, but the full bi-temporal edge model is not yet active.
>
> **VectorCypher temporal detection:** The VectorCypher engine classifies queries into 7 temporal categories via `TemporalDetector` (see [temporal-queries.md](../query-engine/temporal-queries.md)) and applies category-specific retrieval parameters (recency boost, sorting, decay override). The `occurred_at` column on chunks is also used for temporal sorting during Neo4j graph traversal (`ORDER BY c.occurred_at DESC`) when the temporal signal indicates it.

The Skeleton Construction engine implements a bi-temporal model inspired by [Graphiti](https://github.com/getzep/graphiti), with a hierarchical time graph inspired by [TG-RAG](https://arxiv.org/abs/2410.15149). This document explains the theory and implementation.

## Bi-Temporal Theory

Traditional databases track a single timestamp (typically `created_at`). Bi-temporal databases track two independent time dimensions:

1. **Transaction Time (System Time)**: When the fact was recorded in the system
2. **Valid Time (Application Time)**: When the fact was true in the real world

In the Skeleton Construction engine, we use:

| Field | Temporal Concept | Description |
|-------------|------------------|-------------|
| `ingested_at` | Transaction time | When we learned about it |
| `occurred_at` | Valid time | When the event actually happened |

### Why Bi-Temporal?

**Problem with single timestamp:**

```
Day 1: Alice joins Engineering team
Day 3: System records "Alice joined Engineering on Day 1"

With single timestamp:
- created_at = Day 3
- Query "Who was on Engineering on Day 2?" → Alice not found (wrong!)

With bi-temporal:
- ingested_at = Day 3 (when we learned)
- occurred_at = Day 1 (when it happened)
- Query "Who was on Engineering on Day 2?" → Alice found (correct!)
```

**Use Cases:**

| Scenario | occurred_at | ingested_at |
|----------|-------------|-------------|
| Live chat message | Now | Now |
| Historical import | Original date | Import date |
| Backdated correction | Corrected date | Now |
| Late-arriving data | Event date | Arrival date |

## TemporalEdge Data Model

```python
@dataclass
class TemporalEdge:
    id: UUID
    namespace_id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str = ""

    # Bi-temporal fields
    occurred_at: datetime      # When the fact happened
    ingested_at: datetime      # When we learned about it
    valid_from: datetime       # Validity window start
    valid_until: datetime      # Validity window end (None = still valid)

    # Conflict tracking
    is_valid: bool = True
    invalidated_by_id: UUID | None = None
    invalidation_reason: str | None = None

    # Provenance
    confidence: float = 1.0
    properties: dict[str, Any] = {}
    source_document_ids: list[UUID] = []
    source_chunk_ids: list[UUID] = []
```

### Validity Windows

Edges can have explicit validity windows:

```python
# Alice worked at Acme from Jan 2020 to Dec 2023
edge = TemporalEdge(
    namespace_id=ns_id,
    source_entity_id=alice_id,
    target_entity_id=acme_id,
    relationship_type="WORKS_FOR",
    occurred_at=datetime(2020, 1, 15),
    valid_from=datetime(2020, 1, 15),
    valid_until=datetime(2023, 12, 31),
    is_valid=True
)

# Query: "Where did Alice work in 2022?"
edges = await storage.get_valid_at(
    entity_id=alice_id,
    namespace_id=ns_id,
    point_in_time=datetime(2022, 6, 1)
)
# Returns: WORKS_FOR → Acme
```

## Conflict Resolution

### Exclusive Relationships

Some relationships are mutually exclusive - a person can only have one at a time. The set is a local variable (`exclusive_types`) inside `_handle_conflicts`, not a module-level constant:

```python
# Local to TemporalEdgeStorage._handle_conflicts
exclusive_types = {
    "WORKS_FOR",
    "REPORTS_TO",
    "MANAGES",
    "MARRIED_TO",
    "CEO_OF",
    "PRESIDENT_OF",
    "LOCATED_AT",
    "HEADQUARTERED_IN",
}
```

### Automatic Invalidation

When a new exclusive relationship is created, conflicting older edges are automatically invalidated:

```python
# January 2024: Alice works for Acme
edge1 = await storage.create_edge(
    TemporalEdge(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=alice_id,
        target_entity_id=acme_id,
        relationship_type="WORKS_FOR",
        occurred_at=datetime(2024, 1, 15),
    ),
)
# edge1.is_valid = True

# March 2024: Alice now works for Beta Corp
edge2 = await storage.create_edge(
    TemporalEdge(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=alice_id,
        target_entity_id=beta_id,
        relationship_type="WORKS_FOR",
        occurred_at=datetime(2024, 3, 1),
    ),
)
# edge2.is_valid = True
# edge1.is_valid = False (automatically invalidated)
# edge1.invalidated_by_id = edge2.id
# edge1.invalidation_reason = "Superseded by newer WORKS_FOR relationship"
```

### Conflict Detection Algorithm

```python
async def _handle_conflicts(self, new_edge: TemporalEdge) -> None:
    """Check for and handle conflicting edges."""
    # Find existing valid edges of the same type between the same pair
    existing = await self.get_edges_by_entity_pair(
        new_edge.source_entity_id,
        new_edge.target_entity_id,
        new_edge.namespace_id,
        relationship_type=new_edge.relationship_type,
        include_invalid=False,
    )

    exclusive_types = {
        "WORKS_FOR",
        "REPORTS_TO",
        "MANAGES",
        "MARRIED_TO",
        "CEO_OF",
        "PRESIDENT_OF",
        "LOCATED_AT",
        "HEADQUARTERED_IN",
    }

    if new_edge.relationship_type.upper() in exclusive_types:
        for old_edge in existing:
            # If new edge is more recent, invalidate the old one
            if new_edge.occurred_at > old_edge.occurred_at:
                await self.invalidate_edge(
                    old_edge.id,
                    invalidated_by=new_edge.id,
                    reason=f"Superseded by newer {new_edge.relationship_type} edge",
                )
```

## Hierarchical Time Graph

Inspired by TG-RAG, the Skeleton Construction engine maintains a hierarchical time structure for efficient temporal queries:

```
2024 (year)
├── Q1 2024 (quarter)
│   ├── January 2024 (month)
│   │   ├── Week 1 2024 (week, ISO week number)
│   │   │   ├── 2024-01-01 (day)
│   │   │   ├── 2024-01-02 (day)
│   │   │   ├── 2024-01-03 (day)
│   │   │   └── ...
│   │   ├── Week 2 2024 (week)
│   │   └── ...
│   ├── February 2024 (month)
│   └── March 2024 (month)
├── Q2 2024 (quarter)
└── ...
```

### TimeNode Structure

```python
@dataclass
class TimeNode:
    id: UUID
    namespace_id: UUID
    granularity: str        # DAY, WEEK, MONTH, QUARTER, YEAR
    start_time: datetime    # Start of period (inclusive)
    end_time: datetime      # End of period (exclusive)
    parent_id: UUID | None  # Link to parent node
    name: str               # Human-readable: "January 2024", "Q1 2024"
    edge_count: int = 0     # Edges linked to this node
    entity_count: int = 0   # Entities linked to this node
```

### Automatic Hierarchy Creation

When storing an edge with a timestamp, the time hierarchy is automatically created:

```python
# Store edge that occurred on 2024-01-15
await storage.create_edge(
    TemporalEdge(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type="MENTIONS",
        occurred_at=datetime(2024, 1, 15),
    ),
)

# Automatically creates (if not existing):
# - 2024 (year)
# - Q1 2024 (quarter)
# - January 2024 (month)
# - Week 3 2024 (week containing Jan 15)
# - 2024-01-15 (day)
```

### Time Range Queries

The hierarchy enables efficient range queries:

```python
# "What happened in Q1 2024?"
# Instead of scanning all edges, query the Q1 node and its children

nodes = await hierarchy.get_nodes_in_range(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 31),
    granularity="MONTH"  # Or DAY for fine-grained
)
# Returns: [January 2024, February 2024, March 2024]

# Get covering node for a range
covering_node = await hierarchy.get_covering_node(
    start=datetime(2024, 1, 15),
    end=datetime(2024, 2, 28)
)
# Returns: Q1 2024 (smallest node covering the range)
```

### Drill-Down Navigation

```python
# Start at year level
year_node = await hierarchy.get_or_create_day_node(datetime(2024, 1, 1))

# Navigate to quarters
quarters = await hierarchy.get_children(year_node.id)
# Returns: [Q1 2024, Q2 2024, Q3 2024, Q4 2024]

# Drill into Q1
months = await hierarchy.get_children(quarters[0].id)
# Returns: [January 2024, February 2024, March 2024]

# Further drill into January
weeks = await hierarchy.get_children(months[0].id)
# Returns: [Week 1, Week 2, Week 3, Week 4, Week 5]
```

## Database Schema

### PostgreSQL Implementation

```sql
-- Temporal chunks with BRIN index
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),

    -- Bi-temporal fields
    occurred_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Structured metadata
    source_system VARCHAR(255),
    author VARCHAR(255),
    channel VARCHAR(255),
    tags TEXT[],
    metadata JSONB
);

-- BRIN index: 99% space savings vs B-tree for time-series data
CREATE INDEX idx_chunks_occurred_at_brin
ON khora_chunks USING BRIN (occurred_at)
WITH (pages_per_range = 128);

-- Time hierarchy table
CREATE TABLE khora_time_nodes (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    granularity VARCHAR(20) NOT NULL,  -- DAY, WEEK, MONTH, QUARTER, YEAR
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    parent_id UUID REFERENCES khora_time_nodes(id),
    name VARCHAR(100) NOT NULL,
    edge_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,

    UNIQUE (namespace_id, granularity, start_time)
);

-- Temporal edges
CREATE TABLE khora_temporal_edges (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    source_entity_id UUID NOT NULL,
    target_entity_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    description TEXT,

    -- Bi-temporal fields
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_from TIMESTAMP WITH TIME ZONE,
    valid_until TIMESTAMP WITH TIME ZONE,

    -- Conflict tracking
    is_valid BOOLEAN DEFAULT TRUE,
    invalidated_by_id UUID REFERENCES khora_temporal_edges(id),
    invalidation_reason TEXT,

    -- Metadata
    confidence FLOAT DEFAULT 1.0,
    properties JSONB,
    source_document_ids UUID[],
    source_chunk_ids UUID[],

    -- Link to time hierarchy
    day_node_id UUID REFERENCES khora_time_nodes(id)
);

-- Index for temporal queries
CREATE INDEX idx_edges_occurred_at ON khora_temporal_edges (occurred_at);
CREATE INDEX idx_edges_valid ON khora_temporal_edges (is_valid) WHERE is_valid = TRUE;
CREATE INDEX idx_edges_source ON khora_temporal_edges (source_entity_id);
CREATE INDEX idx_edges_target ON khora_temporal_edges (target_entity_id);
```

### Why BRIN Indexes?

BRIN (Block Range Index) is ideal for time-series data:

| Index Type | Size for 10M rows | Range Query Speed |
|------------|-------------------|-------------------|
| B-tree | ~200 MB | Fast |
| BRIN | ~2 MB | Fast (for sorted data) |

BRIN works because:
1. Data is naturally ordered by time (append-only)
2. Queries are typically range-based ("last 30 days")
3. 99% space savings with similar performance

## Query Patterns

### Point-in-Time Queries

```python
# What was true at a specific moment?
edges = await storage.get_valid_at(
    entity_id=alice_id,
    namespace_id=ns_id,
    point_in_time=datetime(2023, 6, 15),
)
```

### Range Queries

```python
# What happened in Q1 2024?
edges = await storage.get_edges_by_time_range(
    namespace_id=ns_id,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 31),
)
```

### Temporal Evolution

```python
# How has Alice's employment changed over time?
all_edges = await storage.get_edges_by_entity(
    alice_id,
    namespace_id=ns_id,
    relationship_type="WORKS_FOR",
    direction="outgoing",
    valid_only=False  # Include invalidated edges
)

for edge in sorted(all_edges, key=lambda e: e.occurred_at):
    status = "✓" if edge.is_valid else "✗"
    print(f"{status} {edge.occurred_at}: {edge.relationship_type} → {edge.target_entity_id}")

# Output:
# ✗ 2020-01-15: WORKS_FOR → Acme Inc
# ✗ 2022-03-01: WORKS_FOR → Beta Corp
# ✓ 2024-01-10: WORKS_FOR → Gamma Tech
```

### As-Of Queries

The `ingested_at` column stores transaction time, but `get_edges_by_entity` does not expose an `ingested_before` filter - it only filters on `occurred_at` (via `time_start` / `time_end`). An "as-of" query over transaction time would need a new query method or a direct `TemporalEdgeModel` select on `ingested_at`.

## Related Documentation

- [Skeleton Construction Engine](skeleton-engine.md) - Overview of the Skeleton Construction engine
- [Query Engine](../query-engine/temporal-queries.md) - Temporal query patterns
