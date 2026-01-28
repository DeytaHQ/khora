# Agentic Search

Sometimes one search isn't enough. Complex questions need exploration - following threads, exploring entities, checking different sources. Agentic search does this automatically, but with a twist: all the "thinking" happens upfront.

## The Key Insight

Traditional agentic approaches make multiple LLM calls - one to search, one to decide what to do next, one to search again, etc. This is slow and expensive.

Khora's approach: **one LLM call upfront, multiple searches after.**

```
Regular Search:
  Question → [LLM: understand] → [Search] → Results

Agentic Search:
  Question → [LLM: understand + plan follow-ups] → [Search] → [Search more] → [Search more] → Combined Results
                         |
                         └── All LLM work done here!
```

The initial understanding call extracts not just what you're asking, but what follow-up queries would be useful if initial results aren't enough.

## How It Works

### Step 1: Understand and Plan

The initial LLM call extracts:
- What you're asking
- Entities mentioned
- **Pre-computed follow-up queries** (the key innovation)
- How complex this query is
- Which sources to prioritize

```python
# From a single LLM call
understanding = {
    "intent": "find product strategy information",
    "entities": ["Product Team", "2024 Roadmap"],
    "follow_up_queries": [
        {"query": "product roadmap 2024", "reason": "specific timeline details"},
        {"query": "competitive analysis", "reason": "market positioning context"}
    ],
    "complexity_score": 0.75,  # High - suggests multi-step exploration
    "source_priority": {"notion": 0.4, "linear": 0.3, "slack": 0.3}
}
```

### Step 2: Initial Search

Run the original query across all sources:

```
Query: "What is our product strategy?"
         |
    +----+----+----+
    |    |    |    |
    v    v    v    v
 Vector Graph Keyword  → RRF Fusion → Initial Results
```

Analyze what came back:
- Which sources contributed?
- Any dominant source (>80%)?
- Which entities were found?

### Step 3+: Follow-Up Exploration

Execute pre-computed follow-ups (no additional LLM calls):

```python
# From the understanding step
follow_up_queries = [
    {"query": "product roadmap 2024", "reason": "specific timeline details"},
    {"query": "competitive analysis", "reason": "market positioning context"}
]

# Execute each
for follow_up in follow_up_queries:
    results = await search(follow_up["query"], namespace_id)
    all_results.merge(results)
```

Additional follow-ups may be generated locally (no LLM) based on results:

```python
# If results are 90% from Notion, explore other sources
if dominant_source == "notion" and dominance > 0.8:
    follow_ups.append({"query": f"{query} slack", "reason": "balance sources"})

# Explore top entities
if top_entity := results.entities[0]:
    follow_ups.append({"query": f"{top_entity.name} details", "reason": "entity context"})
```

### Step 4: Merge and Return

Combine results from all steps:
- Deduplicate (same chunk found multiple ways)
- Keep higher scores when duplicates occur
- Generate summary from structured data (no LLM)

## Using Agentic Search

### Basic Usage

```python
from khora.query.agentic import AgenticSearchAgent

agent = AgenticSearchAgent(engine=hybrid_engine)

result = await agent.search(
    "What is our product strategy?",
    namespace_id,
    max_steps=3  # Initial + 2 follow-ups
)

print(f"Found {len(result.chunks)} unique chunks")
print(f"Summary: {result.summary}")
```

### Via MemoryLake

```python
result = await lake.recall(
    "product strategy",
    config=QueryConfig(
        enable_agentic=True,
        max_agentic_steps=3
    )
)
```

### With Other Options

```python
result = await agent.search(
    query,
    namespace_id,
    config=QueryConfig(
        mode=SearchMode.HYBRID,
        temporal_filter=TemporalFilter.last_days(30),
        recency_bias=0.2
    ),
    max_steps=3
)
```

## The Result Object

```python
AgenticSearchResult(
    # Combined results (chunk, score, source)
    chunks=[(chunk1, 0.85, "notion"), (chunk2, 0.82, "slack"), ...],

    # Entities found
    entities=[(entity1, 0.9), (entity2, 0.8), ...],

    # Auto-generated summary (no LLM call)
    summary="Found 15 results across 3 sources (notion: 8, slack: 5, linear: 2). Key entities: Product Team, Q4 Roadmap. Explored in 3 steps.",

    # Full trace for debugging/analysis
    trace=AgenticSearchTrace(...),

    # Original understanding
    understanding=UnderstandingResult(...)
)
```

## The Trace

Every agentic search produces a detailed trace:

```python
result = await agent.search(query, namespace_id)

# See what happened
for step in result.trace.steps:
    print(f"Step {step.step_number}: {step.query}")
    print(f"  Reason: {step.reasoning}")
    print(f"  Found: {step.total_chunks} chunks")
    print(f"  Sources: vector={step.vector_hits}, graph={step.graph_hits}, keyword={step.keyword_hits}")
```

Example trace:

```
Step 1: "What is our product strategy?"
  Reason: Initial query
  Found: 8 chunks
  Sources: vector=5, graph=2, keyword=1

Step 2: "product roadmap 2024"
  Reason: specific timeline details (pre-computed)
  Found: 5 chunks
  Sources: vector=4, graph=1, keyword=0

Step 3: "Product Team context details"
  Reason: exploring top entity (generated)
  Found: 4 chunks
  Sources: vector=1, graph=3, keyword=0
```

### Trace as Dict

For logging or storage:

```python
trace_dict = result.trace.to_dict()
# Contains: session_id, steps, complexity_score, sources_explored, etc.

await save_to_analytics(trace_dict)
```

## When to Use Agentic Search

**Good candidates:**
- Open-ended research questions
- Complex topics requiring synthesis
- Exploratory queries ("tell me about X")
- Questions that might need context from multiple areas

**Skip for:**
- Simple factual lookups ("what's Alice's email?")
- Specific entity queries ("show me the Q4 report")
- Time-sensitive real-time queries
- When you're counting API costs

## How Follow-Ups Are Generated

### Pre-Computed (from LLM)

The understanding step generates these based on:
- What additional context would help
- What aspects the question implies
- Common follow-up patterns for this query type

### Generated Locally

No LLM needed - these come from analyzing results:

**Source Imbalance:**
```python
# 85% from Notion? Try other sources
if notion_ratio > 0.8:
    follow_ups.append(f"{query} slack linear")
```

**Entity Exploration:**
```python
# Found "Product Team" entity? Get more context
if top_entity := results.entities[0]:
    follow_ups.append(f"{top_entity.name} context")
```

## Summary Generation (No LLM)

The summary is built from structured data:

```python
def generate_summary(query, chunks, entities, trace):
    sources = count_by_source(chunks)
    top_entities = [e.name for e, _ in entities[:5]]

    parts = [
        f"Found {len(chunks)} results across {len(sources)} sources.",
        f"Key entities: {', '.join(top_entities)}.",
        f"Explored in {len(trace.steps)} steps."
    ]

    if trace.complexity_score > 0.7:
        parts.append("Query identified as complex.")

    return " ".join(parts)
```

## Performance

Agentic search is designed to be efficient:

| Operation | LLM Calls | Notes |
|-----------|-----------|-------|
| Understanding | 1 | All extraction upfront |
| Follow-up planning | 0 | Pre-computed |
| Each search step | 0 | Pure retrieval |
| Summary generation | 0 | Template-based |
| **Total** | **1** | Same as regular search! |

The cost difference vs. regular search is just the additional search queries (cheap) - not additional LLM calls (expensive).

## Comparison

| Feature | Regular Search | Agentic Search |
|---------|----------------|----------------|
| LLM calls | 1 | 1 |
| Search steps | 1 | 1-3+ |
| Follow-ups | None | Pre-computed + local |
| Source balance | As found | Actively explored |
| Entity exploration | Basic | Deep |
| Trace detail | Minimal | Full |

## What's Next?

- **[Query Understanding](query-understanding.md)** - How follow-ups are pre-computed
- **[Search Modes](search-modes.md)** - The underlying search methods
- **[Fusion](fusion.md)** - How results are combined
