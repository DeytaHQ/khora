# Temporal Queries

Khora supports time-based filtering and recency bias for temporal queries. This document covers temporal filter configuration and usage.

## Overview

Temporal features enable:
- **Time filtering**: Restrict results to specific time ranges
- **Recency bias**: Boost newer content in rankings
- **Temporal validity**: Respect entity/relationship time bounds

## TemporalFilter

Located at `src/khora/query/temporal.py`.

```python
@dataclass
class TemporalFilter:
    operator: TemporalOperator
    start: datetime | None = None
    end: datetime | None = None
```

### Operators

```python
class TemporalOperator(str, Enum):
    BEFORE = "before"     # Documents before date
    AFTER = "after"       # Documents after date
    BETWEEN = "between"   # Documents within range
    DURING = "during"     # Active during period (for entities)
    OVERLAPS = "overlaps" # Any overlap with period
```

## Factory Methods

### Last N Days

```python
from khora.query.temporal import TemporalFilter

# Last 7 days
filter = TemporalFilter.last_days(7)

# Last 30 days
filter = TemporalFilter.last_days(30)
```

### Last N Hours

```python
# Last 24 hours
filter = TemporalFilter.last_hours(24)

# Last 6 hours
filter = TemporalFilter.last_hours(6)
```

### Between Dates

```python
from datetime import datetime

filter = TemporalFilter.between(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 31),
)
```

### Before/After

```python
# Before specific date
filter = TemporalFilter.before(datetime(2024, 6, 1))

# After specific date
filter = TemporalFilter.after(datetime(2024, 1, 1))
```

## Usage

### Via MemoryLake

```python
results = await lake.recall(
    "product updates",
    temporal_filter=TemporalFilter.last_days(7),
)
```

### Via QueryConfig

```python
from khora.query import QueryConfig

results = await lake.recall(
    "engineering discussions",
    config=QueryConfig(
        temporal_filter=TemporalFilter.between(
            datetime(2024, 1, 1),
            datetime(2024, 3, 31),
        ),
    ),
)
```

## Recency Bias

Apply exponential decay to boost newer content:

```python
results = await lake.recall(
    "team updates",
    temporal_filter=TemporalFilter.last_days(30),
    recency_bias=0.3,  # 0 = disabled, 0.1-1.0 = strength
)
```

### Decay Formula

```
recency_score = exp(-decay_rate * age_in_days)
final_score = base_score * (1 + recency_bias * recency_score)
```

### Recency Bias Strengths

| Value | Effect |
|-------|--------|
| 0.0 | No recency bias |
| 0.1 | Subtle boost for recent |
| 0.3 | Moderate boost |
| 0.5 | Strong boost |
| 1.0 | Very strong boost |

### Example

Document from 1 day ago vs 30 days ago, with `recency_bias=0.3`:

```
1-day-old:  recency_score ≈ 0.97  →  boost = 1.29x
30-day-old: recency_score ≈ 0.37  →  boost = 1.11x
```

## Temporal Info in Results

Query results include temporal information:

```python
result = await lake.recall(query, temporal_filter=filter)

if result.temporal_info:
    print(f"Filter applied: {result.temporal_info.filter_applied}")
    print(f"Time start: {result.temporal_info.time_start}")
    print(f"Time end: {result.temporal_info.time_end}")
```

```python
@dataclass
class TemporalInfo:
    filter_applied: bool
    time_start: datetime | None
    time_end: datetime | None
```

## Document Timestamps

Filtering applies to document timestamps:

```python
# Documents store creation time
document.created_at = datetime(2024, 1, 15)

# Chunks inherit document timestamp
chunk.created_at = document.created_at
```

### Source Timestamp Inheritance

Documents can preserve source system timestamps:

```python
# When ingesting from Slack, use message sent_at
document = Document(
    content=message.text,
    created_at=message.sent_at,  # Original timestamp
)
```

This enables accurate "last week" queries even for recently ingested historical data.

## Entity Temporal Validity

Entities have optional validity periods:

```python
entity = Entity(
    name="Acme CEO",
    entity_type="PERSON",
    valid_from=datetime(2020, 1, 1),
    valid_until=datetime(2023, 12, 31),
)
```

### DURING Operator

Find entities active during a period:

```python
# Find who was CEO in 2022
filter = TemporalFilter(
    operator=TemporalOperator.DURING,
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31),
)
```

### OVERLAPS Operator

Find entities with any overlap:

```python
# Entities active any time in 2023
filter = TemporalFilter(
    operator=TemporalOperator.OVERLAPS,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
)
```

## Query Understanding Integration

Temporal references are extracted automatically:

```python
# Query: "Updates from last week"
understanding = await understand_query(query)

# Extracted temporal reference
temporal_ref = understanding.temporal_references[0]
# TemporalReference(
#     text="last week",
#     temporal_type="relative",
#     iso_start="2024-01-20T00:00:00Z",
#     iso_end="2024-01-27T00:00:00Z",
# )

# Converted to filter automatically
filter = TemporalFilter.between(
    datetime.fromisoformat(temporal_ref.iso_start),
    datetime.fromisoformat(temporal_ref.iso_end),
)
```

## SQL Implementation

Temporal filtering is applied at the database level:

```sql
-- BETWEEN filter
SELECT * FROM chunks
WHERE namespace_id = $1
  AND created_at >= $2
  AND created_at <= $3
ORDER BY created_at DESC;

-- With recency bias
SELECT *,
    (1 + 0.3 * EXP(-0.03 * EXTRACT(EPOCH FROM NOW() - created_at) / 86400)) as recency_boost
FROM chunks
WHERE namespace_id = $1
  AND created_at >= $2
ORDER BY base_score * recency_boost DESC;
```

## Event Sourcing for Historical State

Query historical state via event sourcing:

```python
# Get events before a date to reconstruct state
events = await storage.get_events(
    namespace_id,
    before=datetime(2024, 1, 1),
)

# Filter to created events
created_ids = {e.resource_id for e in events if "created" in e.event_type}
deleted_ids = {e.resource_id for e in events if "deleted" in e.event_type}

# Active at that time
active_at_date = created_ids - deleted_ids
```

See [Event Sourcing](../architecture/event-sourcing.md) for details.

## Examples

### Recent Updates

```python
results = await lake.recall(
    "team updates",
    temporal_filter=TemporalFilter.last_days(7),
    recency_bias=0.3,
)
```

### Q4 2024

```python
results = await lake.recall(
    "quarterly planning",
    temporal_filter=TemporalFilter.between(
        datetime(2024, 10, 1),
        datetime(2024, 12, 31),
    ),
)
```

### Historical State

```python
# Find entities valid in 2022
results = await lake.recall(
    "company leadership",
    temporal_filter=TemporalFilter(
        operator=TemporalOperator.DURING,
        start=datetime(2022, 1, 1),
        end=datetime(2022, 12, 31),
    ),
)
```

### Real-Time with Recency

```python
# Last 24 hours, strongly prefer newer
results = await lake.recall(
    "incident reports",
    temporal_filter=TemporalFilter.last_hours(24),
    recency_bias=0.5,
)
```

## Next Steps

- [Query Understanding](query-understanding.md) - Automatic temporal extraction
- [Event Sourcing](../architecture/event-sourcing.md) - Historical state
- [Overview](overview.md) - Full query pipeline
