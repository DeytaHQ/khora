# Agentic Search

Khora provides a two-step exploration agent for deep search. The key efficiency: all LLM extraction happens in the initial query understanding call.

## Overview

Agentic search extends regular search with multi-step exploration:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agentic Search                               │
│                                                                  │
│  Step 1: Initial Search                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - Query understanding (single LLM call)                     ││
│  │ - Multi-source search (vector + graph + keyword)            ││
│  │ - Extract follow-up queries (pre-computed)                  ││
│  │ - Extract source priorities                                 ││
│  │ - Extract complexity score                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Step 2+: Follow-Up Exploration (no LLM calls)                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - Execute pre-computed follow-up queries                    ││
│  │ - Explore under-represented sources                         ││
│  │ - Investigate high-scoring entities                         ││
│  │ - Merge results (keep higher scores)                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Final: Merge and Rank                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - Deduplicate across all steps                              ││
│  │ - Sort by score                                              ││
│  │ - Generate summary (no LLM call)                            ││
│  │ - Return with full trace                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Regular vs Agentic Search

| Aspect | Regular Search | Agentic Search |
|--------|----------------|----------------|
| LLM Calls | 1 (understanding) | 1 (understanding) |
| Search Steps | 1 | 1-3+ |
| Follow-ups | None | Pre-computed |
| Source balance | As found | Explored |
| Trace | Minimal | Full |

## AgenticSearchAgent

Located at `src/khora/query/agentic.py`.

```python
from khora.query.agentic import AgenticSearchAgent

agent = AgenticSearchAgent(
    engine=hybrid_query_engine,
    llm_config=llm_config,
)

result = await agent.search(
    "What is our product strategy?",
    namespace_id=namespace_id,
    max_steps=3,
)
```

## AgenticSearchResult

```python
@dataclass
class AgenticSearchResult:
    # Combined results from all steps
    chunks: list[tuple[Chunk, float, str]]  # (chunk, score, source)
    entities: list[tuple[Entity, float]]

    # Summary (generated without LLM)
    summary: str

    # Full exploration trace
    trace: AgenticSearchTrace

    # Query understanding from step 1
    understanding: UnderstandingResult | None

    # Metadata
    metadata: dict[str, Any]
```

## AgenticSearchTrace

Full trace of the exploration:

```python
@dataclass
class AgenticSearchTrace:
    session_id: str
    original_query: str
    started_at: datetime
    completed_at: datetime | None

    # Understanding (from single LLM call)
    understanding_reasoning: str
    complexity_score: float
    source_priority: dict[str, float]

    # Steps
    steps: list[SearchStep]

    # Summary
    summary: str
    total_unique_chunks: int
    total_unique_entities: int
    sources_explored: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
```

## SearchStep

Each exploration step is tracked:

```python
@dataclass
class SearchStep:
    step_number: int
    query: str
    reasoning: str
    timestamp: datetime

    # Results summary
    total_chunks: int
    total_entities: int
    sources_hit: dict[str, int]

    # Search contributions
    vector_hits: int
    graph_hits: int
    keyword_hits: int

    # Graph exploration
    entities_linked: list[str]
    relationships_traversed: list[tuple[str, str, str]]

    # Temporal info
    temporal_filter_applied: bool
    time_range: tuple[datetime | None, datetime | None] | None
```

## Follow-Up Query Generation

Follow-ups come from two sources:

### 1. Pre-Computed (Query Understanding)

Extracted during the single LLM call:

```python
# From understanding result
follow_up_queries = [
    {
        "query": "product roadmap 2024",
        "reasoning": "Find specific roadmap details",
    },
    {
        "query": "competitive analysis",
        "reasoning": "Understand market positioning",
    },
]
```

### 2. Result Analysis (No LLM)

Generated locally based on results:

```python
def _generate_additional_follow_ups(self, result, analysis):
    follow_ups = []

    # Source imbalance detection
    if one_source_dominates(analysis.sources_hit):
        # Target under-represented sources
        follow_ups.append({
            "query": f"{query} {underrepresented_source}",
            "reasoning": f"Targeting: {source}",
        })

    # Entity exploration
    if result.entities:
        top_entity = result.entities[0][0]
        follow_ups.append({
            "query": f"{top_entity.name} context details",
            "reasoning": f"Exploring top entity: {top_entity.name}",
        })

    return follow_ups[:2]
```

## Source Imbalance Detection

If initial results are concentrated in one source, explore others:

```python
# If >80% from one source
if dominant_source_ratio > 0.8:
    # Generate query targeting other sources
    for source in ["linear", "notion", "attio", "gong"]:
        if source not in hit_sources:
            add_follow_up(f"{query} {source}")
```

## Summary Generation (No LLM)

Summary is generated from structured data:

```python
def _generate_summary_fast(self, query, chunks, entities, trace):
    # Count by source
    sources = count_by_source(chunks)
    source_summary = ", ".join(f"{s}: {c}" for s, c in sources.items())

    # Top entities
    top_entities = [e.name for e, _ in sorted(entities.values())[:5]]

    parts = [
        f"Found {len(chunks)} results across {len(sources)} sources ({source_summary}).",
        f"Key entities: {', '.join(top_entities)}.",
        f"Explored in {len(trace.steps)} steps.",
    ]

    if trace.complexity_score > 0.7:
        parts.append("Query was identified as complex, requiring multi-step exploration.")

    return " ".join(parts)
```

## Result Merging

Results from all steps are merged, keeping higher scores:

```python
for chunk, score in step_result.chunks:
    chunk_id = str(chunk.id)
    if chunk_id not in all_chunks or all_chunks[chunk_id][1] < score:
        all_chunks[chunk_id] = (chunk, score, source)
```

## Usage

### Basic Agentic Search

```python
from khora.query.agentic import AgenticSearchAgent

agent = AgenticSearchAgent(engine=engine)

result = await agent.search(
    "What is our product strategy?",
    namespace_id,
    max_steps=3,
)

print(f"Found {len(result.chunks)} unique chunks")
print(f"Summary: {result.summary}")
```

### With QueryConfig

```python
result = await agent.search(
    query,
    namespace_id,
    config=QueryConfig(
        mode=SearchMode.HYBRID,
        temporal_filter=TemporalFilter.last_days(30),
    ),
    max_steps=3,
)
```

### Via MemoryLake

```python
result = await lake.recall(
    "product strategy",
    config=QueryConfig(
        enable_agentic=True,
        max_agentic_steps=3,
    ),
)
```

### Accessing Trace

```python
result = await agent.search(query, namespace_id)

# Full trace as dict
trace_dict = result.trace.to_dict()

# Step-by-step analysis
for step in result.trace.steps:
    print(f"Step {step.step_number}: {step.query}")
    print(f"  Reasoning: {step.reasoning}")
    print(f"  Found: {step.total_chunks} chunks, {step.total_entities} entities")
    print(f"  Sources: {step.sources_hit}")
```

## When to Use Agentic Search

**Use for:**
- Complex queries requiring synthesis
- Exploratory research
- Queries about unknown territory
- When initial results seem incomplete

**Skip for:**
- Simple factual lookups
- Specific entity queries
- Time-sensitive searches
- Cost-constrained scenarios

## Complexity-Based Triggering

The complexity score can trigger agentic search:

```python
# In query understanding
complexity_score = 0.8  # High complexity

if complexity_score > 0.6:
    # Suggest agentic search
    pass
```

## Performance

- **Single LLM call**: All extraction happens upfront
- **Parallel search**: Each step uses parallel multi-source search
- **Batch source lookup**: Document sources fetched in batch
- **No additional LLM**: Follow-ups are pre-computed or local

## Example Output

```python
AgenticSearchResult(
    chunks=[(chunk1, 0.85, "notion"), (chunk2, 0.82, "slack"), ...],
    entities=[(entity1, 0.9), (entity2, 0.8), ...],
    summary="Found 15 results across 3 sources (notion: 8, slack: 5, linear: 2). Key entities: Product Team, Q4 Roadmap, Feature X. Explored in 3 steps.",
    trace=AgenticSearchTrace(
        session_id="abc-123",
        steps=[
            SearchStep(step_number=1, query="product strategy", ...),
            SearchStep(step_number=2, query="product roadmap 2024", ...),
            SearchStep(step_number=3, query="Product Team context details", ...),
        ],
        complexity_score=0.75,
        ...
    ),
    metadata={
        "original_query": "What is our product strategy?",
        "total_steps": 3,
        "sources_explored": {"notion": 8, "slack": 5, "linear": 2},
    },
)
```

## Next Steps

- [Query Understanding](query-understanding.md) - Pre-computed follow-ups
- [Search Modes](search-modes.md) - Multi-source search
- [Overview](overview.md) - Full query pipeline
