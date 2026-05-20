# Chronicle Engine

The Chronicle engine is Khora's memory engine designed for **temporal and conversational memory**. It targets high scores on benchmarks like LongMemEval, LoCoMo, and BEAM by combining semantic search with temporal reasoning.

Unlike VectorCypher, Chronicle requires **no graph database** — it runs on PostgreSQL + pgvector only, with an optional embedded LanceDB store for zero-infrastructure deployments.

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

Chronicle applies an exponential decay curve inspired by the Ebbinghaus forgetting curve:

```
score = base + weight * exp(-ln(2) * age_days / half_life)
```

Default half-life: 168 hours (7 days). A memory retains 50% strength after one week, 25% after two weeks. Configurable via `recency_weight` and `recency_decay_days`.

## Key Features

### Event Decomposition

Chronicle extracts structured **SVO tuples** (subject-verb-object) from content with triple timestamps:

- **observation_date** — when the content was ingested
- **referenced_date** — when the event actually occurred
- **relative_offset** — temporal distance (e.g., "last week", "in March")

This enables precise temporal queries that distinguish between "when was this stored" and "when did this happen."

### Progressive Compression

For long-running conversations, Chronicle compresses older memories to manage token budgets:

1. Extract atomic **Elementary Discourse Units** (facts) from content
2. Detect contradictions via `FactOperation`: ADD, UPDATE, DELETE, NOOP
3. Merge and compress — achieves 3-6x token reduction while preserving key facts

### LanceDB Embedded Store

For zero-infrastructure deployments, Chronicle can use LanceDB as an embedded vector store instead of pgvector:

```bash
pip install khora[lancedb]
```

LanceDB stores vectors in local files with HNSW indexing — no database server required.

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
        ns = await kb.create_namespace("conversations")

        # Store conversation turns
        await kb.remember(
            "Alice: We should switch to quarterly releases. "
            "Bob: I agree, monthly is too frequent.",
            namespace=ns.namespace_id,
            metadata={"occurred_at": "2026-03-15T10:00:00Z"},
        )

        await kb.remember(
            "Alice: Actually, let's stick with monthly releases. "
            "The team prefers the faster cadence.",
            namespace=ns.namespace_id,
            metadata={"occurred_at": "2026-03-22T14:00:00Z"},
        )

        # Temporal query — Chronicle uses recency to find the latest stance
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
    ns = await kb.create_namespace("demo")
    await kb.remember("...", namespace=ns.namespace_id)
    result = await kb.recall("...", namespace=ns.namespace_id)
```

### Recall response shape

Since v0.16.0 (#761), `result.documents` is always populated for every
document referenced by a chunk in the result — Chronicle relies on the
namespace-scoped coordinator facade to batch-fetch documents, never on
the legacy public sub-backend attrs. The `RecallResult.context_text`
attribute is gone; render a context string with the public
`khora.context_text(result, max_chunks=…)` helper if you need one.

Chronicle's namespace scoping is enforced at the SQL/SurrealQL layer:
every read filters by `namespace_id` directly rather than post-fetching
and comparing in Python (which would leak existence as a timing oracle).
See the v0.16.0 IDOR close-out (#769) for the full Protocol-level
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
| `KHORA_QUERY_APPLY_RECENCY_BIAS` | Enable temporal decay | `false` (Chronicle enables internally) |
| `KHORA_QUERY_RECENCY_WEIGHT` | Decay weight | `0.2` |
| `KHORA_QUERY_RECENCY_DECAY_DAYS` | Half-life in days | `30.0` |

## Related Documentation

- [Engine Comparison](engine-comparison.md) — side-by-side feature matrix
- [Temporal Queries](../query-engine/temporal-queries.md) — time filtering and recency
- [Hybrid Search](hybrid-search.md) — RRF fusion details
