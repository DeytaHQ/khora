# Temporal Queries

Time matters. "What did we discuss last week?" is different from "What have we ever discussed?" Khora lets you filter results by time and optionally boost newer content.

## Two Time Features

**Temporal Filtering** - Only return results from a specific time range:
```python
# Only content from the last 7 days
results = await lake.recall(
    "product updates",
    temporal_filter=TemporalFilter.last_days(7)
)
```

**Recency Bias** - Rank newer content higher:
```python
# All content, but prefer recent
results = await lake.recall(
    "team updates",
    recency_bias=0.3  # Boost recent content
)
```

You can use both together:
```python
# Last 30 days, newer is better
results = await lake.recall(
    "incident reports",
    temporal_filter=TemporalFilter.last_days(30),
    recency_bias=0.5
)
```

## Temporal Filters

### Quick Filters

The most common patterns have shortcuts:

```python
from khora.query.temporal import TemporalFilter

# Last N days
TemporalFilter.last_days(7)
TemporalFilter.last_days(30)

# Last N hours (for real-time)
TemporalFilter.last_hours(24)
TemporalFilter.last_hours(6)

# Specific date range
TemporalFilter.between(
    datetime(2024, 1, 1),
    datetime(2024, 3, 31)
)

# Before/after a date
TemporalFilter.before(datetime(2024, 6, 1))
TemporalFilter.after(datetime(2024, 1, 1))
```

### The Full Filter Object

For more control:

```python
from khora.query.temporal import TemporalFilter, TemporalOperator

# BETWEEN: start <= timestamp <= end
filter = TemporalFilter(
    operator=TemporalOperator.BETWEEN,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 31)
)

# AFTER: timestamp > start
filter = TemporalFilter(
    operator=TemporalOperator.AFTER,
    start=datetime(2024, 1, 1)
)

# BEFORE: timestamp < end
filter = TemporalFilter(
    operator=TemporalOperator.BEFORE,
    end=datetime(2024, 6, 1)
)
```

### Entity Validity Filters

Entities can have validity periods - "Alice was CEO from 2020-2023". Two special operators handle this:

```python
# DURING: Find entities that were active throughout a period
# (entity.valid_from <= start AND entity.valid_until >= end)
filter = TemporalFilter(
    operator=TemporalOperator.DURING,
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31)
)
# "Who was CEO for all of 2022?"

# OVERLAPS: Find entities with any overlap
# (entity.valid_from <= end AND entity.valid_until >= start)
filter = TemporalFilter(
    operator=TemporalOperator.OVERLAPS,
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31)
)
# "Who was CEO at any point during 2022?"
```

## Recency Bias

Sometimes you don't want to filter - you want everything, but newer content should rank higher.

### How It Works

Recency bias applies an exponential decay boost:

```
boost = 1 + recency_bias * exp(-decay * age_in_days)
final_score = base_score * boost
```

A document from yesterday gets a bigger boost than one from a month ago.

### Strength Settings

| Value | Effect |
|-------|--------|
| `0.0` | No recency preference (default) |
| `0.1` | Subtle - barely noticeable |
| `0.3` | Moderate - recent content noticeably higher |
| `0.5` | Strong - recent content significantly favored |
| `1.0` | Very strong - recent content dominates |

### Example

With `recency_bias=0.3`:
- 1-day-old document: ~1.29x boost
- 7-day-old document: ~1.24x boost
- 30-day-old document: ~1.11x boost

This means a slightly less relevant document from yesterday can outrank a more relevant document from last month.

### When to Use

**Use recency bias for:**
- News and updates ("What's happening?")
- Active projects ("Current status?")
- Evolving situations ("Latest on the merger?")

**Skip recency bias for:**
- Reference material ("How does X work?")
- Historical research ("What happened in 2020?")
- Timeless concepts ("Explain relativity")

## Source Timestamps

An important detail: temporal queries use the document's **creation timestamp**, which Khora preserves from the source system.

```python
# When ingesting from Slack
document = Document(
    content=message.text,
    created_at=message.sent_at  # Original message time, not ingestion time
)
```

This means if you ingest a year's worth of Slack messages today, "last week" will correctly return messages from last week, not messages you happened to ingest today.

Timestamp priority (first available wins):
1. `sent_at` - Message sent time
2. `created_at` - Creation time
3. `timestamp` - Generic timestamp
4. `date` - Date field
5. `occurred_at` - Event time
6. Current time (fallback)

## Query Understanding Integration

You don't have to specify temporal filters manually. Natural language time references are extracted automatically:

```python
# Query: "What updates were there last week?"
#
# Query understanding extracts:
#   temporal_reference: "last week"
#   iso_start: "2024-01-20T00:00:00Z"
#   iso_end: "2024-01-27T00:00:00Z"
#
# Which becomes:
#   TemporalFilter.between(start, end)
```

This happens during the query understanding step. Supported expressions include:
- "last week", "last month", "last 7 days"
- "yesterday", "today"
- "in January", "in Q4"
- "before March", "after 2023"
- "between January and March"

## Usage Examples

### Recent Updates

```python
# What happened this week?
results = await lake.recall(
    "team updates",
    temporal_filter=TemporalFilter.last_days(7)
)
```

### Real-Time Monitoring

```python
# Last 6 hours, strongly prefer newest
results = await lake.recall(
    "incident reports",
    temporal_filter=TemporalFilter.last_hours(6),
    recency_bias=0.5
)
```

### Quarterly Review

```python
# Q4 2024
results = await lake.recall(
    "quarterly planning",
    temporal_filter=TemporalFilter.between(
        datetime(2024, 10, 1),
        datetime(2024, 12, 31)
    )
)
```

### Historical Research

```python
# What was the team structure in 2022?
results = await lake.recall(
    "team structure",
    temporal_filter=TemporalFilter(
        operator=TemporalOperator.DURING,
        start=datetime(2022, 1, 1),
        end=datetime(2022, 12, 31)
    )
)
```

### Full Config Example

```python
from khora.query import QueryConfig
from khora.query.temporal import TemporalFilter

results = await lake.recall(
    "product decisions",
    config=QueryConfig(
        mode=SearchMode.HYBRID,
        temporal_filter=TemporalFilter.last_days(30),
        recency_bias=0.3,
        limit=20
    )
)
```

## Result Metadata

Query results include information about temporal filtering:

```python
result = await lake.recall(query, temporal_filter=filter)

if result.temporal_info:
    print(f"Filter applied: {result.temporal_info.filter_applied}")
    print(f"Time range: {result.temporal_info.time_start} to {result.temporal_info.time_end}")
```

## Under the Hood

Temporal filtering happens at the database level for efficiency:

```sql
-- BETWEEN filter
SELECT * FROM chunks
WHERE namespace_id = ?
  AND created_at >= ?
  AND created_at <= ?
ORDER BY created_at DESC;

-- With recency bias (simplified)
SELECT *,
    base_score * (1 + ? * EXP(-0.03 * age_days)) as final_score
FROM chunks
WHERE namespace_id = ?
ORDER BY final_score DESC;
```

Entities with validity periods use similar logic on `valid_from` and `valid_until` columns.

## Combining with Event Sourcing

For true time travel - reconstructing the state of the knowledge graph at a past point - use the event store:

```python
# What entities existed on January 1st?
events = await storage.get_events(
    namespace_id,
    before=datetime(2024, 1, 1)
)

created = {e.resource_id for e in events if "created" in e.event_type}
deleted = {e.resource_id for e in events if "deleted" in e.event_type}
existed_then = created - deleted
```

See [Event Sourcing](../architecture/event-sourcing.md) for details.

## What's Next?

- **[Query Understanding](query-understanding.md)** - How time expressions are extracted
- **[Search Modes](search-modes.md)** - Combining temporal with different search types
- **[Event Sourcing](../architecture/event-sourcing.md)** - Full historical state
