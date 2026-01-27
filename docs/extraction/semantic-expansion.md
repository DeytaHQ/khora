# Semantic Expansion

Semantic expansion enhances the knowledge graph through entity unification and relationship inference. This document covers the expansion components.

## Overview

Expansion runs after initial extraction to:
1. **Unify entities** - Merge duplicates from different sources
2. **Infer relationships** - Discover implicit connections

```
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Expansion                            │
│                                                                  │
│   Extracted           Cross-Tool           Relationship         │
│   Entities      →     Unifier        →     Inferrer             │
│   & Relations                                                    │
│                                                                  │
│   - From multiple      - Exact match       - Pattern rules      │
│     chunks/docs        - Fuzzy match       - Transitive         │
│   - Potential dups     - Embedding sim     - Configurable       │
│                        - Merge entities    - New edges          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## SemanticExpander

Orchestrates unification and inference:

```python
from khora.extraction.expansion import SemanticExpander
from khora.extraction.skills import load_expertise

expertise = load_expertise("saas_expert")

expander = SemanticExpander(
    expertise=expertise,
    enable_unification=True,
    enable_inference=True,
    inference_depth=2,
    embedding_threshold=0.85,
    fuzzy_threshold=0.85,
    min_inference_confidence=0.3,
)

result = await expander.expand(
    entities=extracted_entities,
    relationships=extracted_relationships,
    namespace_id=namespace_id,
)
```

### ExpansionResult

```python
@dataclass
class ExpansionResult:
    # Unified entities (after deduplication)
    entities: list[Entity]

    # Updated relationships (with remapped entity IDs)
    relationships: list[Relationship]

    # Newly inferred relationships
    inferred_relationships: list[Relationship]

    # Statistics
    original_entity_count: int
    merged_entity_count: int
    original_relationship_count: int
    inferred_relationship_count: int

    # Mapping for provenance tracking
    entity_mapping: dict[UUID, UUID]
```

## CrossToolUnifier

Identifies and merges duplicate entities:

```python
from khora.extraction.expansion import CrossToolUnifier

unifier = CrossToolUnifier(
    expertise=expertise,
    embedding_threshold=0.85,
    fuzzy_threshold=0.85,
)

result = unifier.unify(
    entities=entities,
    relationships=relationships,
    use_embeddings=True,
    use_fuzzy=True,
)
```

### Matching Strategies

#### 1. Exact Name Matching

Entities with identical names (case-insensitive) and same type are merged:

```python
# These would be merged:
Entity(name="Acme Corp", entity_type=ORGANIZATION)
Entity(name="acme corp", entity_type=ORGANIZATION)
```

#### 2. Field-Based Matching (Correlation Rules)

Match on specific fields defined in expertise:

```yaml
correlation_rules:
  - name: email_match
    match_fields: [email]
    entity_types: [PERSON]
    confidence: 0.95
```

```python
# Matched by email field:
Entity(name="John Doe", attributes={"email": "john@acme.com"})
Entity(name="John D.", attributes={"email": "john@acme.com"})
```

#### 3. Fuzzy String Matching

Levenshtein distance for similar names:

```python
# Similarity = 1 - (edit_distance / max_length)
fuzzy_threshold = 0.85

# These would match (similarity > 0.85):
"Acme Corporation" ↔ "Acme Corp"
"Jennifer Walsh" ↔ "Jenny Walsh"
```

#### 4. Embedding Similarity

Cosine similarity between entity embeddings:

```python
# Match if cosine similarity >= threshold
embedding_threshold = 0.85

# Entities with similar semantic meaning:
Entity(name="Machine Learning", embedding=[...])
Entity(name="ML", embedding=[...])  # Similar embedding
```

### Merge Algorithm

Uses Union-Find for efficient grouping:

```python
def _find_merge_groups(self, entities, ...):
    # Union-Find structure
    parent = {e.id: e.id for e in entities}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Apply all matching strategies
    for rule in correlation_rules:
        matches = find_field_matches(entities, rule)
        for e1, e2 in matches:
            union(e1.id, e2.id)

    # ... exact matching, fuzzy matching, embedding matching

    # Build groups from union-find
    return groups_with_multiple_entities
```

### Entity Merge Strategy

When merging entities:

```python
def merge_with(self, other: Entity):
    # Sum mention counts
    self.mention_count += other.mention_count

    # Average confidence
    self.confidence = (self.confidence + other.confidence) / 2

    # Merge attributes (non-empty values preferred)
    for key, value in other.attributes.items():
        if value and not self.attributes.get(key):
            self.attributes[key] = value

    # Combine aliases
    self.aliases.extend(a for a in other.aliases if a not in self.aliases)

    # Combine source references
    self.source_document_ids.extend(other.source_document_ids)
    self.source_chunk_ids.extend(other.source_chunk_ids)

    # Expand temporal validity
    if other.valid_from and other.valid_from < self.valid_from:
        self.valid_from = other.valid_from
```

## RelationshipInferrer

Applies inference rules to discover implicit relationships:

```python
from khora.extraction.expansion import RelationshipInferrer

inferrer = RelationshipInferrer(
    expertise=expertise,
    min_confidence=0.3,
)

inferred = inferrer.infer(
    entities=entities,
    relationships=relationships,
    depth=2,  # Number of passes
)
```

### Inference Rules

Rules are defined in expertise configuration:

```yaml
inference_rules:
  # Colleagues work for same company
  - name: colleague_inference
    when:
      - relationship: WORKS_FOR
        source_type: PERSON
      - relationship: WORKS_FOR
        source_type: PERSON
    then:
      relationship: COLLEAGUES_WITH
      source: first.source   # Person 1
      target: second.source  # Person 2
    confidence: 0.6

  # Transitive management
  - name: indirect_reports
    when:
      - relationship: MANAGES
      - relationship: MANAGES
    then:
      relationship: INDIRECTLY_MANAGES
      source: first.source
      target: second.target
    confidence: 0.4
```

### Inference Depth

Multiple passes find transitive relationships:

```
Depth 1:
  A WORKS_FOR Company1
  B WORKS_FOR Company1
  → A COLLEAGUES_WITH B

Depth 2:
  A MANAGES B
  B MANAGES C
  → A INDIRECTLY_MANAGES C
```

### Entity References

Rule `then_source` and `then_target` use reference notation:

| Reference | Meaning |
|-----------|---------|
| `first.source` | Source entity of first matched relationship |
| `first.target` | Target entity of first matched relationship |
| `second.source` | Source entity of second matched relationship |
| `second.target` | Target entity of second matched relationship |

## Inference Modes

Three modes control when inference runs:

### None Mode

Only unification, no inference:

```yaml
expansion:
  enabled: true
  relationship_inference: false
```

### Incremental Mode (Default)

Infer per-document, querying existing graph for context:

```python
if inference_mode == "incremental":
    # Fetch existing entities/relationships from storage
    existing_entities = await storage.list_entities(namespace_id)
    existing_relationships = await storage.list_relationships(namespace_id)

    # Include in expansion context
    expansion_entities.extend(existing_entities)
    expansion_relationships.extend(existing_relationships)
```

Enables cross-document inference during ingestion.

### Batch Mode

Infer only after all documents are processed:

```python
# During ingestion: unification only
expander = SemanticExpander(
    expertise=expertise,
    enable_inference=False,  # Skip inference
)

# After all documents: batch inference
from khora.pipelines.flows import run_batch_inference

result = await run_batch_inference(
    namespace_id=namespace_id,
    storage=storage,
    expertise=expertise,
)
```

Better for large imports where incremental would be slow.

## UnificationResult

```python
@dataclass
class UnificationResult:
    # Unified entities
    unified_entities: list[Entity]

    # Original → unified ID mapping
    entity_mapping: dict[UUID, UUID]

    # Relationships with updated entity references
    updated_relationships: list[Relationship]

    # New relationships from correlation rules
    new_relationships: list[Relationship]

    # Statistics
    entities_merged: int
    merge_groups: list[list[UUID]]
```

## Relationship ID Remapping

After entities are merged, relationship IDs are updated:

```python
def _update_relationships(self, relationships, entity_mapping):
    updated = []
    for rel in relationships:
        # Remap source/target to unified entity IDs
        new_source = entity_mapping.get(rel.source_entity_id, rel.source_entity_id)
        new_target = entity_mapping.get(rel.target_entity_id, rel.target_entity_id)

        # Skip self-referential (emerged from merging)
        if new_source == new_target:
            continue

        rel.source_entity_id = new_source
        rel.target_entity_id = new_target
        updated.append(rel)

    return updated
```

## API Usage

### Via Pipeline

```python
result = await ingest_documents(
    namespace_id,
    documents,
    storage,
    expertise="saas_expert",
    enable_expansion=True,
)
```

### Direct Usage

```python
from khora.extraction.expansion import SemanticExpander

expander = SemanticExpander(expertise=expertise)

result = await expander.expand(
    entities=entities,
    relationships=relationships,
    namespace_id=namespace_id,
)

print(f"Unified {result.original_entity_count} → {len(result.entities)}")
print(f"Inferred {result.inferred_relationship_count} relationships")
```

### From Expertise Name

```python
expander = SemanticExpander.from_expertise_name("saas_expert")
```

## Configuration

### Similarity Thresholds

```python
# Higher thresholds = fewer false merges
embedding_threshold = 0.90  # Stricter
fuzzy_threshold = 0.90      # Stricter

# Lower thresholds = more merges (potential false positives)
embedding_threshold = 0.75
fuzzy_threshold = 0.75
```

### Confidence Minimum

```python
# Only keep inferred relationships with high confidence
min_inference_confidence = 0.5

# Keep more speculative inferences
min_inference_confidence = 0.2
```

## Next Steps

- [Expertise System](expertise-system.md) - Define expansion rules
- [Knowledge Graph](../data-models/knowledge-graph.md) - Entity/relationship models
- [Ingestion Pipeline](ingestion-pipeline.md) - Expansion in pipeline
