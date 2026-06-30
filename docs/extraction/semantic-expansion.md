# Semantic Expansion

Semantic expansion enhances the knowledge graph through entity unification and relationship inference. After extraction turns raw text into entities and relationships, expansion asks: "Are any of these actually the same thing?" and "What connections can we deduce from what we already know?"

## Overview

Expansion runs after initial extraction to:
1. **Unify entities** -- Merge duplicates from different sources
2. **Infer relationships** -- Discover implicit connections

```
+-----------------------------------------------------------------+
|                    Semantic Expansion                            |
|                                                                 |
|   Extracted           Cross-Tool           Relationship         |
|   Entities      ->    Unifier        ->    Inferrer             |
|   & Relations                                                   |
|                                                                 |
|   - From multiple      - Exact match       - Pattern rules      |
|     chunks/docs        - Fuzzy match       - Transitive         |
|   - Potential dups     - Embedding sim     - Configurable       |
|                        - Merge entities    - New edges          |
|                                                                 |
+-----------------------------------------------------------------+
```

## Inference Modes

The expansion system supports four modes that control *when* and *how* entity resolution and relationship inference happen. Choosing the right mode matters: it determines whether you wait seconds or hours on a large corpus.

| Mode | When It Runs | Complexity | Best For |
|------|-------------|------------|----------|
| `smart` (default) | Per-doc dedup during ingestion, full resolution once after all docs | O(n * k) | Large imports, production use |
| `incremental` | Per document, with existing graph context | O(n^2) per document | Small graphs, real-time trickle feeds |
| `batch` | After all documents, on full graph | O(n^2) once | Legacy bulk imports |
| `none` | Never | O(1) | Unification only, no inference |

### Smart Mode (Recommended)

Smart mode separates entity resolution into two phases:

**Phase 1 -- Per-document dedup (during ingestion):**
Each document's entities are checked against a shared in-memory `EntityIndex`. Exact duplicates (same normalized name + same type) are merged immediately in O(1). No database round-trips, no pairwise comparisons. The index grows as documents are processed.

**Phase 2 -- Cross-document resolution (after all documents):**
After every document has been processed, a single resolution pass runs across the full entity set. The `CrossToolUnifier` uses the `EntityIndex` for token-blocked candidate retrieval, reducing fuzzy and embedding matching from O(n^2) to O(n * k) where k is the blocked candidate set size (typically 10-20 entities). Relationship inference also runs once on the fully resolved graph.

```python
from khora.extraction.skills import load_expertise

expertise = load_expertise("saas_expert")
# Set inference_mode via ExpertiseConfig.expansion.inference_mode:
expertise.expansion.inference_mode = "smart"  # default

result = await ingest_documents(
    namespace_id,
    documents,
    storage,
    expertise=expertise,
    enable_expansion=True,
)
```

The practical difference is dramatic: on a corpus of 16,000 documents with 5,000 entities, incremental mode reloads the full entity set from storage *for every document* and runs O(n^2) matching each time. Smart mode does O(1) index lookups during ingestion and a single O(n * k) resolution pass afterward.

### Incremental Mode

Runs full expansion per document, fetching existing entities and relationships from storage each time:

```python
if inference_mode == "incremental":
    # Fetch existing entities/relationships from storage
    existing_entities = await storage.list_entities(namespace_id)
    existing_relationships = await storage.list_relationships(namespace_id)

    # Include in expansion context
    expansion_entities.extend(existing_entities)
    expansion_relationships.extend(existing_relationships)
```

This enables cross-document inference during ingestion but becomes increasingly expensive as the graph grows, since every document triggers O(n^2) pairwise comparisons on the combined entity set.

### Batch Mode

Skips inference during ingestion entirely, then runs it once afterward:

```python
# During ingestion: unification only
expander = SemanticExpander(
    expertise=expertise,
    enable_inference=False,  # Skip inference
)

# After all documents: batch inference
from khora.pipelines.flows.ingest import run_batch_inference

result = await run_batch_inference(
    namespace_id=namespace_id,
    storage=storage,
    expertise=expertise,
)
```

### None Mode

Only unification, no inference:

```yaml
expansion:
  enabled: true
  relationship_inference: false
```

## EntityIndex

The `EntityIndex` is an in-memory blocking index that makes entity resolution efficient. It's the core data structure behind smart mode, though it can be used independently whenever you need fast entity lookup.

### What It Does

The index maintains four internal structures:

| Index | Key | Purpose | Complexity |
|-------|-----|---------|-----------|
| **Exact index** | `(normalized_name, type)` | Instant duplicate detection | O(1) lookup |
| **Token index** | name token -> entity IDs | Candidate blocking for fuzzy matching | O(k) retrieval |
| **Type index** | entity type -> entities | Same-type candidate sets for embedding matching | O(1) per type |
| **ID index** | UUID -> entity | Master lookup | O(1) lookup |

### Token Blocking

The key insight behind the performance improvement. Instead of comparing every entity against every other entity (O(n^2)), token blocking narrows the candidate set by requiring at least one shared name token before computing expensive similarity metrics.

```
"Microsoft Corporation"  ->  tokens: {microsoft, corporation}
"Microsoft Corp"         ->  tokens: {microsoft, corp}
"Apple Inc"              ->  tokens: {apple, inc}

Looking for matches for "Microsoft Corporation":
  Token "microsoft" -> {entity_1, entity_2}    (shared token!)
  Token "corporation" -> {entity_1}

  Candidate set: {entity_2}  (entity_1 is self, excluded)
  Only compute Levenshtein on this small set.

  "Apple Inc" never enters the candidate set.
```

Tokenization normalizes names to lowercase and strips punctuation. Tokens shorter than 2 characters are discarded to avoid noise from initials and articles.

For embedding-based matching, the candidate set also includes all entities of the same type (since embeddings can match entities with completely different names, like "ML" and "Machine Learning"). This is still far fewer comparisons than the full O(n^2) pairwise approach.

### Usage

```python
from khora.extraction.expansion import EntityIndex

index = EntityIndex()

# Add entities - returns existing if exact match found
for entity in extracted_entities:
    existing = index.add(entity)
    if existing is not None:
        existing.merge_with(entity)  # Caller handles merging

# Post-ingestion: find fuzzy candidates via token blocking
candidates = index.find_fuzzy_candidates(entity, threshold=0.85)
# Returns: [(candidate_entity, similarity_score), ...]

# Post-ingestion: find embedding candidates
candidates = index.find_embedding_candidates(entity, threshold=0.85)

# Bulk access
all_entities = index.get_all_entities()
people = index.get_entities_by_type("PERSON")
stats = index.stats()
# {"total_entities": 5000, "exact_keys": 5000, "token_keys": 3200, "type_groups": 6}
```

### Why Not FAISS or ANN Libraries?

At the scale Khora typically operates (1K-50K entities), token blocking reduces candidate sets to 10-20 per entity. Brute-force cosine similarity on 20 candidates is instant. FAISS, Annoy, or other ANN libraries would add heavy native dependencies for no measurable gain at this scale. The blocking itself *is* the optimization.

If you're dealing with millions of entities, you'd want a different approach entirely -- but at that point you're in a dedicated entity resolution pipeline, not a knowledge graph ingestion tool.

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
    entity_index=entity_index,   # Optional: enables token-blocked matching
)
```

When an `entity_index` is passed, the expander forwards it to the `CrossToolUnifier`, which uses blocked candidate retrieval instead of pairwise comparisons. When `entity_index` is `None`, the unifier falls back to the original O(n^2) behavior.

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
    entity_index=entity_index,  # Optional: enables blocked matching
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
"Acme Corporation" <-> "Acme Corp"
"Jennifer Walsh" <-> "Jenny Walsh"
```

When an `EntityIndex` is provided, fuzzy matching only runs on the token-blocked candidate set instead of all entity pairs. This is the main performance win.

#### 4. Embedding Similarity

Cosine similarity between entity embeddings:

```python
# Match if cosine similarity >= threshold
embedding_threshold = 0.85

# Entities with similar semantic meaning:
Entity(name="Machine Learning", embedding=[...])
Entity(name="ML", embedding=[...])  # Similar embedding
```

With blocking, embedding candidates include all same-type entities plus token-sharing entities, then cosine similarity is computed only within that set.

### Merge Algorithm

Uses Union-Find for efficient transitive grouping:

```python
def _find_merge_groups(self, entities, ..., entity_index=None):
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

    # Fuzzy matching (blocked if entity_index provided)
    # Embedding matching (blocked if entity_index provided)

    # Build groups from union-find
    return groups_with_multiple_entities
```

When `entity_index` is provided, `_find_fuzzy_matches` and `_find_embedding_matches` call `entity_index.find_fuzzy_candidates()` and `entity_index.find_embedding_candidates()` respectively, iterating per-entity instead of doing pairwise comparisons.

### Entity Merge Strategy

When merging entities:

```python
def merge_with(self, other: Entity):
    # Combine source references (deduplicated)
    for doc_id in other.source_document_ids:
        if doc_id not in self.source_document_ids:
            self.source_document_ids.append(doc_id)
    for chunk_id in other.source_chunk_ids:
        if chunk_id not in self.source_chunk_ids:
            self.source_chunk_ids.append(chunk_id)

    # Sum mention counts
    self.mention_count += other.mention_count

    # Merge attributes: only add keys not already present (existing values win)
    for key, value in other.attributes.items():
        if key not in self.attributes:
            self.attributes[key] = value

    # Take maximum confidence (not average)
    self.confidence = max(self.confidence, other.confidence)

    # Fill description only if currently empty
    if not self.description and other.description:
        self.description = other.description
```

Key points: confidence uses `max()`, not the average. Attributes are added only when the key does not already exist (existing values are never overwritten). There is no `aliases` field and no `valid_from` expansion -- those were not part of the model.

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
  -> A COLLEAGUES_WITH B

Depth 2:
  A MANAGES B
  B MANAGES C
  -> A INDIRECTLY_MANAGES C
```

### Incremental Context Updates

The `RuleEvaluationContext` supports incremental updates via its `update()` method. When new entities and relationships are discovered during an inference pass, the context's internal indices (entity index, type index, relationship indices) are updated in-place rather than rebuilt from scratch. This avoids redundant O(n) rebuilds between inference depth passes.

```python
context = RuleEvaluationContext.from_data(entities, relationships)

# After inferring new relationships at depth 1:
context.update(new_entities=[], new_relationships=inferred_rels)

# Depth 2 uses the updated context without rebuilding
```

### Entity References

Rule `then_source` and `then_target` use reference notation:

| Reference | Meaning |
|-----------|---------|
| `first.source` | Source entity of first matched relationship |
| `first.target` | Target entity of first matched relationship |
| `second.source` | Source entity of second matched relationship |
| `second.target` | Target entity of second matched relationship |

## UnificationResult

```python
@dataclass
class UnificationResult:
    # Unified entities
    unified_entities: list[Entity]

    # Original -> unified ID mapping
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

## Smart Mode Resolution Pipeline

When smart mode runs its post-ingestion resolution (via `run_smart_resolution`), the steps are:

1. **Unification** -- Run `CrossToolUnifier` with the populated `EntityIndex` for token-blocked matching. Produces a mapping from original entity IDs to unified entity IDs.

2. **Batch upsert** -- Write resolved entities to storage using `upsert_entities_batch()`, which uses `UNWIND + MERGE` in Neo4j and `INSERT ... ON CONFLICT DO UPDATE` in PostgreSQL. Configurable batch size (default 50 entities per batch).

3. **Embedding generation** -- Generate embeddings for any entities that don't have them yet.

4. **Relationship remapping** -- Load all relationships from storage once. Remap source/target entity IDs using the unification mapping.

5. **Relationship inference** -- Run `RelationshipInferrer.infer()` on the full resolved graph. This happens once, not per-document.

6. **Batch store** -- Write inferred relationships to storage using `create_relationships_batch()`.

```python
# Post-ingestion resolution (called automatically in smart mode)
from khora.pipelines.flows.ingest import run_smart_resolution

result = await run_smart_resolution(
    namespace_id,
    storage,
    entity_index,        # Populated during ingestion
    expertise,
    embedding_model="text-embedding-3-small",
)

# result = {
#     "entities_resolved": 5000,
#     "entities_merged": 120,
#     "inferred_relationships": 45,
# }
```

## API Usage

### Via Pipeline (Smart Mode)

```python
from khora.extraction.skills import load_expertise

expertise = load_expertise("saas_expert")
# inference_mode defaults to "smart" (controlled via ExpertiseConfig)

result = await ingest_documents(
    namespace_id,
    documents,
    storage,
    expertise=expertise,
    enable_expansion=True,
)
```

### Via Pipeline (Other Modes)

```python
from khora.extraction.skills import load_expertise

expertise = load_expertise("saas_expert")
expertise.expansion.inference_mode = "incremental"  # or "batch", "none"

result = await ingest_documents(
    namespace_id,
    documents,
    storage,
    expertise=expertise,
    enable_expansion=True,
)
```

### Direct Usage

```python
from khora.extraction.expansion import SemanticExpander, EntityIndex

# Build index
index = EntityIndex()
for entity in all_entities:
    index.add(entity)

# Expand with blocking
expander = SemanticExpander(expertise=expertise)
result = await expander.expand(
    entities=all_entities,
    relationships=relationships,
    namespace_id=namespace_id,
    entity_index=index,
)

print(f"Unified {result.original_entity_count} -> {len(result.entities)}")
print(f"Inferred {result.inferred_relationship_count} relationships")
```

### From Expertise Name

```python
expander = SemanticExpander.from_expertise_name("saas_expert")
```

## Configuration

### Per-Type Merge Thresholds

Entity resolution uses per-type thresholds to balance precision and recall. Types where false merges are most damaging (people, dates) get higher thresholds:

| Entity Type | Threshold | Rationale |
|-------------|-----------|-----------|
| `DATE` | 0.95 | Dates are precise; merging distinct dates is a correctness error |
| `PERSON` | 0.92 | Different people should not be merged |
| `ORGANIZATION` | 0.88 | Company names are fairly unique |
| `LOCATION` | 0.85 | Locations can have aliases (city/municipality) |
| `TECHNOLOGY` | 0.85 | Tech names are specific |
| `PRODUCT` | 0.85 | Product names are specific |
| `CONCEPT` | 0.82 | Concepts overlap more; moderate merge tolerance |
| `EVENT` | 0.80 | Event names are often informal |
| *(default)* | 0.85 | Fallback for unknown types |

These defaults live in `extraction/entity_resolution.py::DEFAULT_MERGE_THRESHOLDS` and can be overridden via expertise configuration.

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

### Smart Mode Configuration

```yaml
expansion:
  enabled: true
  depth: 2
  inference_mode: smart          # "smart", "incremental", "batch", "none"
  preload_existing: true         # Pre-load existing entities into index on start
  batch_storage_size: 50         # Entities per batch upsert
```

`preload_existing` controls whether existing entities from the database are loaded into the `EntityIndex` before processing new documents. This ensures that new entities are deduplicated against what's already stored, not just against other new entities in the same batch. Set to `false` if you're ingesting into a clean namespace.

`batch_storage_size` controls how many entities are sent to the database in a single batch operation during post-ingestion resolution. Larger batches reduce round-trips but use more memory.

## References

The entity resolution approach used in smart mode draws from established techniques in the record linkage and entity resolution literature:

- Papadakis, G., et al. "Blocking and Filtering Techniques for Entity Resolution." *ACM Computing Surveys*, 2020. [doi:10.1145/3377455](https://dl.acm.org/doi/abs/10.1145/3377455) -- Comprehensive survey of blocking techniques including token blocking, the approach used by `EntityIndex`.

- Graphlet AI. "The Rise of Semantic Entity Resolution." 2024. [Blog post](https://blog.graphlet.ai/the-rise-of-semantic-entity-resolution-45c48b5eb00a) -- Discusses combining traditional string matching with embedding-based semantic matching for entity resolution in knowledge graphs.

- Microsoft. "GraphRAG: Default Dataflow." [Documentation](https://microsoft.github.io/graphrag/index/default_dataflow/) -- Microsoft's approach to entity resolution in their GraphRAG pipeline, which similarly separates per-document extraction from cross-document resolution.

- BlockingPy. "ANN-based blocking for entity resolution." [GitHub](https://github.com/ncn-foreigners/BlockingPy) -- Reference implementation of approximate nearest neighbor blocking for entity resolution at larger scales.

- Zhang, Y., et al. "Efficient Knowledge Graph Construction for Retrieval-Augmented Generation." *arXiv:2507.03226*, 2025. [Paper](https://arxiv.org/abs/2507.03226) -- Recent work on efficient KG construction pipelines for RAG systems.

## Next Steps

- [Expertise System](expertise-system.md) -- Define expansion rules and configure smart mode
- [Knowledge Graph](../data-models/knowledge-graph.md) -- Entity/relationship models
- [Ingestion Pipeline](ingestion-pipeline.md) -- How expansion fits into the pipeline
- [Performance Optimization](../architecture/performance-optimization.md) -- Benchmarks and optimization details
