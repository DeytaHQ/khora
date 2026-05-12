# Temporal Queries

Time matters. "What did we discuss last week?" is different from "What have we ever discussed?" Khora handles temporal intent automatically in the VectorCypher engine, and also provides explicit API controls for precise time-based filtering.

## How It Works

The VectorCypher engine uses a **three-tier cascade** to detect temporal intent in natural language queries and adapt retrieval behavior accordingly. You don't need to specify temporal parameters manually — the engine classifies your query and tunes recency weighting, sort order, and decay rate automatically.

```
"What instrument does the user currently play?"
  → Detected: STATE_QUERY (recency_weight=0.5, temporal sort ON)

"Which city did they move to first?"
  → Detected: ORDINAL (recency_weight=0.1, temporal sort ON)

"How many jobs has Alice had in total?"
  → Detected: AGGREGATE (recency_weight=0.0, no temporal sort)
```

If you need explicit date-range filtering, you can still pass a `TemporalFilter` object — this acts as a manual override and skips automatic detection.

## Automatic Temporal Detection

### Temporal Categories

Every query is classified into one of seven categories, each driving different retrieval behavior:

| Category | Description | Example Query | Recency Weight | Temporal Sort | Notes |
|-----------|-------------|---------------|----------------|---------------|-------|
| `NONE` | No temporal signal | "Explain how X works" | 0.2 | No | Default behavior |
| `EXPLICIT` | Parseable dates | "Before April 2024" | 0.3 | No | Generates a `TemporalFilter` for date-range pushdown |
| `STATE_QUERY` | Current state | "What does Alice currently do?" | 0.5 | Yes (DESC) | High recency, favors newest facts |
| `ORDINAL` | Ordering / sequence | "Which event happened first?" | 0.1 | Yes (DESC) | Low recency, preserves chronological order |
| `AGGREGATE` | Totals / counts | "How many jobs in total?" | 0.0 | No | No recency — needs broad recall |
| `RECENCY` | Latest results | "Most recent update" | 0.5 | Yes (DESC) | Short decay window (7 days) |
| `CHANGE` | Temporal evolution | "Did they switch jobs?" | 0.3 | Yes (DESC) | Moderate recency for tracking changes |

### Detection Cascade

Detection runs as a three-tier cascade, stopping at the first match:

**Tier 1: Aho-Corasick Dictionary (~1–10μs)**
A dictionary of ~200 categorized keyword patterns is matched against the query using Aho-Corasick automaton (Rust via `khora-accel`, with a Python substring fallback). This catches ~85–90% of temporal queries with 0.9 confidence. Example patterns:

- EXPLICIT: "before", "after", "since", "yesterday", "last week", month names
- STATE_QUERY: "currently", "right now", "at the moment", "these days"
- ORDINAL: "first", "earliest", "which came", "preceding"
- AGGREGATE: "how many times", "in total", "all instances"
- RECENCY: "most recent", "newest", "recently"
- CHANGE: "changed", "used to", "no longer", "still", "switched"

**Tier 2: Model2Vec Embedding Centroid (~40–50μs)** *(planned, not yet implemented)*
For queries that slip past the dictionary, a pre-trained Model2Vec centroid will detect implicit temporal intent via embedding similarity. When implemented, queries above a similarity threshold will be classified as `STATE_QUERY` by default.

**Tier 3: LLM-based Query Understanding** *(existing, separate path)*
The full query understanding pipeline can extract temporal references via LLM. This tier is not invoked in the VectorCypher raw retrieval path but is available for complex query planning.

### TemporalSignal

The detector returns a `TemporalSignal` dataclass:

```python
from khora.engines.vectorcypher.temporal_detection import TemporalSignal

@dataclass(frozen=True)
class TemporalSignal:
    is_temporal: bool               # Whether temporal intent was detected
    category: TemporalCategory      # One of the 7 categories
    confidence: float               # 0.0–1.0
    source: str                     # "dictionary", "semantic", or "none"
    temporal_filter: TemporalFilter | None  # Date-range filter (EXPLICIT only)
```

### Integration in VectorCypher

The engine's `recall()` method runs detection automatically when no explicit `TemporalFilter` is provided:

```python
# Automatic — the engine classifies the query and adapts retrieval
results = await kb.recall("What instrument does Alice currently play?")
# → STATE_QUERY: recency_weight=0.5, temporal_sort=True

# Explicit override — skips automatic detection
results = await kb.recall(
    "product updates",
    temporal_filter=TemporalFilter.last_days(7)
)
```

When temporal sort is enabled, the Neo4j graph traversal orders chunks by `occurred_at DESC` (instead of the default `total_mentions DESC`), ensuring time-sensitive queries surface the most chronologically relevant results.

## Recency Bias

### How It Works

Recency bias applies an exponential decay boost to rank newer content higher:

```
boost = 1 + recency_weight * exp(-decay * age_in_days)
final_score = base_score * boost
```

### Relative Recency

Recency is computed **relative to the result set**, not wall-clock time. The reference point is `max(occurred_at)` across all results in the current query, with a fallback to `datetime.now(UTC)` when no timestamps exist.

**Why this matters:** If your data is from 2024 but you run queries in 2026, wall-clock-based recency would give every result a near-zero score (all ~2 years old). Relative recency ensures meaningful score discrimination within any time range. For live data where `max(occurred_at) ≈ now`, the behavior is equivalent to the old wall-clock approach.

### Category-Specific Behavior

Each temporal category maps to specific retrieval parameters:

| Category | Recency Weight | Decay Override | Effect |
|-----------|---------------|----------------|--------|
| `NONE` | 0.2 | — | Subtle recency preference |
| `EXPLICIT` | 0.3 | — | Moderate, combined with date-range filter |
| `STATE_QUERY` | 0.5 | — | Strong recency, most recent fact wins |
| `ORDINAL` | 0.1 | — | Weak recency, preserves chronological order |
| `AGGREGATE` | 0.0 | — | No recency at all — pure relevance |
| `RECENCY` | 0.5 | 7 days | Strong recency, sharp 7-day decay |
| `CHANGE` | 0.3 | — | Moderate recency for evolution tracking |

### Manual Recency Bias

You can also set recency bias explicitly via the API:

```python
# All content, but prefer recent
results = await kb.recall(
    "team updates",
    recency_bias=0.3  # Boost recent content
)
```

| Value | Effect |
|-------|--------|
| `0.0` | No recency preference |
| `0.1` | Subtle — barely noticeable |
| `0.3` | Moderate — recent content noticeably higher |
| `0.5` | Strong — recent content significantly favored |
| `1.0` | Very strong — recent content dominates |

## Temporal Filters

Temporal filters restrict results to a specific time range. They can be generated automatically (from `EXPLICIT` category detection) or specified manually.

### Quick Filters

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

Entities can have validity periods — "Alice was CEO from 2020–2023". Two special operators handle this:

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

### Automatic Date Extraction

When the dictionary detector classifies a query as `EXPLICIT`, it also attempts to extract dates from the query text and build a `TemporalFilter` automatically:

```python
# Query: "What happened before April 2024?"
# → EXPLICIT category, TemporalFilter(occurred_before=2024-04-01)

# Query: "Updates since January 2024"
# → EXPLICIT category, TemporalFilter(occurred_after=2024-01-01)

# Query: "Events around March 15, 2024"
# → EXPLICIT category, TemporalFilter(occurred_after=2024-02-13, occurred_before=2024-04-14)
```

Supported date formats include `YYYY-MM-DD`, `YYYY/MM/DD`, `Month DD, YYYY`, and `DD Month YYYY`.

## Source Timestamps

Temporal queries use the document's **`occurred_at` timestamp**, which Khora preserves from the source system.

```python
# When ingesting from Slack
document = Document(
    content=message.text,
    created_at=message.sent_at  # Original message time, not ingestion time
)
```

This means if you ingest a year's worth of Slack messages today, "last week" will correctly return messages from last week, not messages you happened to ingest today.

Timestamp priority (first available wins):
1. `sent_at` — Message sent time
2. `created_at` — Creation time
3. `timestamp` — Generic timestamp
4. `date` — Date field
5. `occurred_at` — Event time
6. Current time (fallback)

## Usage Examples

### Automatic Detection (Recommended)

```python
# The engine detects temporal intent and adapts automatically
results = await kb.recall("What team is Alice on currently?")
# → STATE_QUERY: high recency, temporal sort

results = await kb.recall("Which project started first?")
# → ORDINAL: low recency, temporal sort

results = await kb.recall("How many times has the team restructured?")
# → AGGREGATE: no recency, broad recall

results = await kb.recall("What's the latest deployment status?")
# → RECENCY: high recency, 7-day decay window

results = await kb.recall("Did Alice switch teams?")
# → CHANGE: moderate recency, temporal sort
```

### Explicit Temporal Filter

```python
# Override automatic detection with a specific time range
results = await kb.recall(
    "incident reports",
    temporal_filter=TemporalFilter.last_hours(6),
    recency_bias=0.5
)
```

### Quarterly Review

```python
results = await kb.recall(
    "quarterly planning",
    temporal_filter=TemporalFilter.between(
        datetime(2024, 10, 1),
        datetime(2024, 12, 31)
    )
)
```

### Historical Research

```python
results = await kb.recall(
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

results = await kb.recall(
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
result = await kb.recall(query, temporal_filter=filter)

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

When temporal sort is active, the Neo4j graph traversal changes ordering:

```cypher
// Default (no temporal signal):
ORDER BY total_mentions DESC

// With temporal sort (STATE_QUERY, ORDINAL, RECENCY, CHANGE):
ORDER BY c.occurred_at DESC, total_mentions DESC
```

Entities with validity periods use similar logic on `valid_from` and `valid_until` columns.

## Combining with Event Sourcing

For true time travel — reconstructing the state of the knowledge graph at a past point — use the event store:

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

- **[Query Understanding](query-understanding.md)** — How time expressions are extracted
- **[Search Modes](search-modes.md)** — Combining temporal with different search types
- **[Event Sourcing](../architecture/event-sourcing.md)** — Full historical state
- **[Temporal Model](../engines/temporal-model.md)** — Bi-temporal edge model and time hierarchy
