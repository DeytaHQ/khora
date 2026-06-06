# Chronicle Engine

> **Status: experimental.** Chronicle is not yet stamped production-ready on any stack - it is suitable for evaluation, prototypes, and benchmarks. For production, use [VectorCypher](vectorcypher-engine.md), the only production-ready engine today. See [engine-comparison.md](engine-comparison.md#production-readiness-by-stack).

The Chronicle engine is Khora's memory engine designed for **temporal and conversational memory**. It targets high scores on benchmarks like LongMemEval, LoCoMo, and BEAM by combining semantic search with temporal reasoning.

Unlike VectorCypher, Chronicle requires **no graph database** - it runs on PostgreSQL + pgvector only, with an optional embedded LanceDB store for zero-infrastructure deployments.

## When to Use Chronicle

- Conversational memory (chat logs, meeting transcripts, support tickets)
- Temporal queries ("What did Alice say last week about the budget?")
- Long-running interactions where recency and time context matter
- Deployments where you want to avoid running Neo4j

## Architecture

Chronicle uses **4-channel parallel retrieval** fused with Reciprocal Rank Fusion:

| Channel | Method | Purpose |
|---------|--------|---------|
| Semantic | pgvector cosine similarity | Find contextually relevant content |
| BM25 | Keyword search | Exact name/term matches |
| Temporal | Ebbinghaus decay scoring | Boost recent, time-relevant results |
| Entity | Co-occurrence similarity | Find content about the same people/things |

All four channels execute in parallel via `asyncio.gather()`, then results are fused using weighted RRF.

### Ebbinghaus Temporal Decay

Chronicle applies an exponential decay curve inspired by the Ebbinghaus forgetting curve. Retention is folded into the relevance score via a multiplicative blend (matches Elasticsearch / Mem0 industry convention):

```
retention = exp(-ln(2) * age_hours / half_life_hours)
final_score = relevance * ((1 - decay_weight) + decay_weight * retention)
```

The max age penalty is `decay_weight` (when `retention -> 0`): a fully-faded memory keeps `(1 - decay_weight)` of its relevance score, while a fresh memory keeps 100%.

Age is measured against `chunk.source_timestamp` (event time, supplied by the user via `metadata['occurred_at']` etc.), falling back to `chunk.created_at` (ingest time) only when no event time was supplied. This prevents a 6-month-old conversation from being treated as "fresh" because it was just ingested.

Defaults:

- `temporal_half_life_hours = 168` (7 days): a memory retains 50% strength after one week, 25% after two weeks.
- `chronicle_decay_weight = 0.30`: a fully-faded memory keeps 70% of its relevance, a fresh memory keeps 100%.

Configurable via `chronicle_decay_weight` and `temporal_half_life_hours` on `QuerySettings`, or the matching env vars `KHORA_QUERY_CHRONICLE_DECAY_WEIGHT` and `KHORA_QUERY_TEMPORAL_HALF_LIFE_HOURS`.

### Reinforcement on Recall

Chronicle can optionally "freshen" chunks every time they are returned by a recall, so that frequently-accessed memories stay sticky and rarely-accessed ones fade. The implementation follows the Stanford generative-agents pattern: each chunk carries a `last_accessed_at` timestamp that the engine stamps after every recall, and the decay function treats the most recent of `source_timestamp` and `last_accessed_at` as the effective event time.

Enable it by setting `KHORA_QUERY_CHRONICLE_ENABLE_RECALL_REINFORCEMENT=true` (or `chronicle_enable_recall_reinforcement=True` on `QuerySettings`). Default is OFF.

When the flag is on, the decay formula's age input changes from:

```
age = now - (source_timestamp OR created_at)
```

to:

```
age = now - max(source_timestamp, last_accessed_at, fallback=created_at)
```

The blend formula itself does not change.

Trade-offs:

- One extra UPDATE per recall (single-statement, scoped to namespace) - fired as `asyncio.create_task(...)` so the recall response is not delayed.
- Eventual consistency on `last_accessed_at`: a recall that runs concurrently with the prior recall's UPDATE may not yet see the freshened timestamp. The reinforcement is best-effort - failures (DB down, network blip) log a warning and never break recall.
- Reinforcement loss across process restarts on the embedded sqlite_lance backend - if the process exits before the spawned task finishes, the UPDATE is dropped. Acceptable: reinforcement is an optimization, not a correctness property.

## Key Features

### Event Decomposition

Chronicle extracts structured **SVO tuples** (subject-verb-object) from content with triple timestamps:

- **observation_date** - when the content was ingested
- **referenced_date** - when the event actually occurred
- **relative_offset** - temporal distance (e.g., "last week", "in March")

This enables precise temporal queries that distinguish between "when was this stored" and "when did this happen."

### Progressive Compression

For long-running conversations, Chronicle compresses older memories to manage token budgets:

1. Extract atomic **Elementary Discourse Units** (facts) from content
2. Detect contradictions via `FactOperation`: ADD, UPDATE, DELETE, NOOP
3. Merge and compress - achieves 3-6x token reduction while preserving key facts

### LanceDB Embedded Store

For zero-infrastructure deployments, Chronicle can use LanceDB as an embedded vector store instead of pgvector:

```bash
pip install khora[lancedb]
```

LanceDB stores vectors in local files with HNSW indexing - no database server required.

## Usage

```python
import asyncio
from khora import Khora

async def main():
    # Chronicle with PostgreSQL + pgvector
    async with Khora(
        "postgresql://khora:khora@localhost:5434/khora",
        engine="chronicle",
        run_migrations=True,
    ) as kb:
        ns = await kb.create_namespace()

        # Store conversation turns
        await kb.remember(
            "Alice: We should switch to quarterly releases. "
            "Bob: I agree, monthly is too frequent.",
            namespace=ns.namespace_id,
            metadata={"occurred_at": "2026-03-15T10:00:00Z"},
            entity_types=["PERSON", "TOPIC"],
            relationship_types=["DISCUSSES", "AGREES_WITH"],
        )

        await kb.remember(
            "Alice: Actually, let's stick with monthly releases. "
            "The team prefers the faster cadence.",
            namespace=ns.namespace_id,
            metadata={"occurred_at": "2026-03-22T14:00:00Z"},
            entity_types=["PERSON", "TOPIC"],
            relationship_types=["DISCUSSES", "AGREES_WITH"],
        )

        # Temporal query - Chronicle uses recency to find the latest stance
        result = await kb.recall(
            "What is Alice's current position on release cadence?",
            namespace=ns.namespace_id,
        )
        for chunk in result.chunks:
            print(chunk.content)

asyncio.run(main())
```

### With Embedded SurrealDB (Zero Infrastructure)

```python
async with Khora("memory://", engine="chronicle") as kb:
    ns = await kb.create_namespace()
    await kb.remember(
        "...",
        namespace=ns.namespace_id,
        entity_types=["PERSON", "TOPIC"],
        relationship_types=["DISCUSSES", "AGREES_WITH"],
    )
    result = await kb.recall("...", namespace=ns.namespace_id)
```

### Recall response shape

`result.documents` is always populated for every
document referenced by a chunk in the result (#761) - Chronicle relies on the
namespace-scoped coordinator facade to batch-fetch documents. Render a
context string with the public
`khora.context_text(result, max_chunks=…)` helper if you need one.

Chronicle's namespace scoping is enforced at the SQL/SurrealQL layer:
every read filters by `namespace_id` directly rather than post-fetching
and comparing in Python (which would leak existence as a timing oracle).
See the IDOR close-out (#769) for the full Protocol-level
contract on every storage backend.

## Comparison with Other Engines

| Feature | Chronicle | VectorCypher | Skeleton |
|---------|-----------|--------------|----------|
| Graph DB required | No | Yes | Optional |
| Temporal decay | Ebbinghaus | Per-category | Recency bias |
| Retrieval channels | 4 (parallel) | 2 (vector+graph) | 2 (vector+BM25) |
| Event decomposition | SVO tuples | No | No |
| Compression | Progressive | No | No |
| Best for | Conversations, temporal | Multi-hop queries | Cost-sensitive |

## Configuration

Chronicle respects standard `KHORA_QUERY_*` env vars, plus these are particularly relevant:

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_QUERY_CHRONICLE_DECAY_WEIGHT` | Multiplicative decay weight (max age penalty) | `0.30` |
| `KHORA_QUERY_TEMPORAL_HALF_LIFE_HOURS` | Half-life in hours for the exponential decay | `168.0` (7 days) |
| `KHORA_QUERY_CHRONICLE_TEMPORAL_WINDOW_DAYS` | Temporal channel window (0 = unlimited, -1 = disable) | `0.0` |
| `KHORA_QUERY_CHRONICLE_ENABLE_RECALL_REINFORCEMENT` | Stamp `last_accessed_at` on recall and treat `max(source_timestamp, last_accessed_at)` as the effective event time | `false` |

## Related Documentation

- [Engine Comparison](engine-comparison.md) - side-by-side feature matrix
- [Temporal Queries](../query-engine/temporal-queries.md) - time filtering and recency
- [Hybrid Search](hybrid-search.md) - RRF fusion details
