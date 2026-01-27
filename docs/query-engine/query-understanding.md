# Query Understanding

Khora uses LLM-based query understanding to analyze queries before search. A single LLM call extracts multiple aspects of the query for improved retrieval.

## Overview

Query understanding extracts:
- **Intent** - What the user wants to do
- **Entities** - Named entities mentioned
- **Temporal references** - Time-related constraints
- **Query expansion** - Related terms for better recall
- **Source priority** - Which sources to prioritize
- **Search strategy** - Optimal search weights
- **Follow-up queries** - Pre-computed exploration queries

## Single LLM Call

All extraction happens in one LLM call for efficiency:

```python
# Single call extracts everything
understanding = await understand_query(query)

# No additional LLM calls during search
# Follow-up queries are pre-computed
```

## UnderstandingResult

```python
@dataclass
class UnderstandingResult:
    # Original query
    query: str

    # Intent classification
    intent: QueryIntent           # SEARCH, QUESTION, TEMPORAL, etc.
    intent_confidence: float

    # Entity mentions
    entity_mentions: list[EntityMention]

    # Temporal references
    temporal_references: list[TemporalReference]

    # Query expansion
    expanded_terms: list[str]

    # Relationship hints for graph exploration
    relationship_hints: list[str]

    # Source prioritization
    source_priority: dict[str, float]  # {"vector": 0.6, "graph": 0.4}

    # Recommended search weights
    search_strategy: SearchStrategy

    # Pre-computed follow-up queries
    follow_up_queries: list[FollowUpQuery]

    # Complexity assessment
    complexity_score: float       # 0.0-1.0
    reasoning: str                # Explanation

    # Metadata
    metadata: dict[str, Any]
```

## Intent Classification

```python
class QueryIntent(str, Enum):
    SEARCH = "search"           # Find relevant content
    QUESTION = "question"       # Answer a specific question
    TEMPORAL = "temporal"       # Time-based query
    COMPARISON = "comparison"   # Compare entities
    EXPLORATION = "exploration" # Discover related information
    AGGREGATION = "aggregation" # Summarize across multiple sources
```

### Examples

| Query | Intent |
|-------|--------|
| "Einstein papers" | SEARCH |
| "Who founded Acme?" | QUESTION |
| "Updates last week" | TEMPORAL |
| "Compare React vs Vue" | COMPARISON |
| "What is related to Project X?" | EXPLORATION |
| "Summarize Q4 activities" | AGGREGATION |

## Entity Mentions

Extracted entities with confidence and type hints:

```python
@dataclass
class EntityMention:
    text: str                 # "Einstein"
    entity_type: str | None   # "PERSON"
    confidence: float         # 0.95
    start_pos: int            # Position in query
    end_pos: int
```

### Example

Query: "Who did Einstein work with at Princeton?"

```python
entity_mentions = [
    EntityMention(text="Einstein", entity_type="PERSON", confidence=0.98),
    EntityMention(text="Princeton", entity_type="ORGANIZATION", confidence=0.92),
]
```

## Temporal References

Parsed time references with ISO 8601 conversion:

```python
@dataclass
class TemporalReference:
    text: str                 # "last week"
    temporal_type: str        # "relative"
    iso_start: str | None     # "2024-01-20T00:00:00Z"
    iso_end: str | None       # "2024-01-27T00:00:00Z"
    confidence: float
```

### Examples

| Text | Type | ISO Conversion |
|------|------|----------------|
| "last week" | relative | Start: 7 days ago, End: now |
| "in 2023" | absolute | Start: 2023-01-01, End: 2023-12-31 |
| "before December" | relative | End: December 1 |
| "Q4 2024" | absolute | Start: 2024-10-01, End: 2024-12-31 |

## Query Expansion

Related terms for improved recall:

```python
# Query: "machine learning"
expanded_terms = [
    "artificial intelligence",
    "deep learning",
    "neural networks",
    "ML",
    "AI",
]
```

## Relationship Hints

Suggested relationship types for graph exploration:

```python
# Query: "Einstein collaborators"
relationship_hints = [
    "COLLABORATES_WITH",
    "WORKS_FOR",
    "KNOWS",
]
```

## Source Priority

Recommended weight distribution across sources:

```python
# Query: "Who manages the engineering team?"
source_priority = {
    "graph": 0.7,   # Relationship query → prioritize graph
    "vector": 0.2,
    "keyword": 0.1,
}

# Query: "quantum physics concepts"
source_priority = {
    "vector": 0.7,  # Conceptual query → prioritize vector
    "graph": 0.2,
    "keyword": 0.1,
}
```

## Search Strategy

Optimized search configuration:

```python
@dataclass
class SearchStrategy:
    mode: SearchMode              # Recommended mode
    vector_weight: float
    graph_weight: float
    keyword_weight: float
    graph_depth: int              # Recommended traversal depth
    graph_relationship_types: list[str] | None
```

## Follow-Up Queries

Pre-computed queries for multi-step exploration:

```python
@dataclass
class FollowUpQuery:
    query: str                    # "Einstein contributions to physics"
    reasoning: str                # "Explore specific achievements"
    priority: float               # 0.8
```

### Example

Query: "What is our product strategy?"

```python
follow_up_queries = [
    FollowUpQuery(
        query="product roadmap 2024",
        reasoning="Find specific roadmap details",
        priority=0.9,
    ),
    FollowUpQuery(
        query="competitive analysis",
        reasoning="Understand market positioning",
        priority=0.7,
    ),
    FollowUpQuery(
        query="customer feedback features",
        reasoning="Feature requests from customers",
        priority=0.6,
    ),
]
```

These are used by agentic search for multi-step exploration without additional LLM calls.

## Complexity Score

Assessment of query difficulty:

```python
# Simple query
complexity_score = 0.2
reasoning = "Single entity, direct search"

# Complex query
complexity_score = 0.8
reasoning = "Multiple entities, temporal constraints, requires synthesis"
```

Used to determine if agentic exploration is needed.

## Usage

### Via Query Engine

```python
result = await engine.query(
    "Einstein collaborators at Princeton",
    namespace_id,
    config=QueryConfig(
        enable_understanding=True,  # Default
    ),
)

# Access understanding in metadata
understanding = result.metadata.get("understanding")
print(f"Intent: {understanding['intent']}")
print(f"Entities: {understanding['entity_mentions']}")
```

### Direct Understanding

```python
from khora.query.understanding import QueryUnderstanding

understanding = QueryUnderstanding(llm_config=config)

result = await understanding.understand(
    "What meetings did the engineering team have last week?",
)

print(f"Intent: {result.intent}")
print(f"Entities: {[e.text for e in result.entity_mentions]}")
print(f"Temporal: {result.temporal_references}")
print(f"Follow-ups: {[fq.query for fq in result.follow_up_queries]}")
```

## LLM Configuration

```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    model="gpt-4o-mini",          # Fast, cost-effective
    temperature=0.3,              # Low for consistency
    max_tokens=2000,              # Enough for complex queries
)
```

## Disabling Understanding

For simple queries or cost reduction:

```python
result = await engine.query(
    "simple search",
    namespace_id,
    config=QueryConfig(
        enable_understanding=False,
    ),
)
```

Without understanding:
- No entity extraction
- No temporal parsing
- No query expansion
- No follow-up queries
- Uses default search weights

## Next Steps

- [Fusion](fusion.md) - How results are combined
- [Agentic Search](agentic-search.md) - Multi-step exploration using follow-ups
- [Temporal Queries](temporal-queries.md) - Time filtering
