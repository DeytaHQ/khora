# Skeleton Indexing

Skeleton indexing is a cost-optimization technique inspired by [KET-RAG](https://arxiv.org/abs/2502.00596) (Knowledge-Enhanced Text RAG). Instead of extracting entities from every document, Khora identifies a small set of "core" chunks using PageRank and only processes those with LLM calls.

## The Cost Problem

Traditional RAG systems extract entities from every chunk:

```
10,000 documents
× 5 chunks per document (average)
× 1 LLM call per chunk (extraction)
───────────────────────────
= 50,000 LLM calls for extraction alone
```

At $0.01 per 1K tokens (GPT-4o-mini), this becomes:
- ~500 tokens per extraction call
- 50,000 × 0.5 × $0.01 = **$250** just for extraction
- Plus another 50,000 calls for relationship extraction

## Skeleton-Based Solution

Khora uses PageRank to identify ~10% of chunks as "core":

```
10,000 documents
× 5 chunks per document
× 10% core ratio
× 1 LLM call per core chunk
───────────────────────────
= 5,000 LLM calls (10x reduction)
```

The insight: **Not all chunks are equally important.** Core chunks are semantically central and contain most of the important information. Non-core chunks can be retrieved via keywords without full extraction.

## How It Works

### 1. Keyword Extraction (No LLM)

```python
def _extract_keywords(self, content: str) -> set[str]:
    """Extract keywords using TF-based scoring."""
    # Tokenize and normalize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

    # Remove stopwords
    words = [w for w in words if w not in STOPWORDS]

    # Calculate term frequency
    word_counts = Counter(words)
    total_words = len(words)

    # Select top keywords by TF
    tf_scores = {w: count / total_words for w, count in word_counts.items()}
    top_keywords = sorted(tf_scores.items(), key=lambda x: -x[1])[:20]

    return {kw for kw, _ in top_keywords}
```

### 2. Build Bipartite Graph

Create a graph connecting chunks to their keywords:

```
Chunk 1 ──── "machine"
        ├─── "learning"
        └─── "neural"

Chunk 2 ──── "learning"     (shared keyword)
        ├─── "algorithm"
        └─── "optimization"

Chunk 3 ──── "neural"       (shared keyword)
        ├─── "network"
        └─── "architecture"
```

### 3. Calculate IDF Scores

```python
def _calculate_idf_scores(self):
    """Calculate inverse document frequency for keywords."""
    num_chunks = len(self.chunk_nodes)

    for keyword, node in self.keyword_nodes.items():
        doc_freq = len(node.chunk_ids)
        # IDF = log(N / df)
        node.idf_score = math.log(num_chunks / doc_freq) if doc_freq > 0 else 0
```

Keywords that appear in fewer chunks have higher IDF (more discriminative).

### 4. Build Chunk-to-Chunk Edges

Chunks are connected if they share keywords, weighted by IDF:

```python
def _build_chunk_edges(self) -> dict[UUID, dict[UUID, float]]:
    """Build weighted edges between chunks via shared keywords."""
    edges = defaultdict(lambda: defaultdict(float))

    for keyword_node in self.keyword_nodes.values():
        chunk_list = list(keyword_node.chunk_ids)
        idf = keyword_node.idf_score

        # Connect all pairs of chunks sharing this keyword
        for i in range(len(chunk_list)):
            for j in range(i + 1, len(chunk_list)):
                c1, c2 = chunk_list[i], chunk_list[j]
                edges[c1][c2] += idf
                edges[c2][c1] += idf

    return edges
```

### 5. Run PageRank

```python
def _calculate_pagerank(
    self,
    edges: dict[UUID, dict[UUID, float]],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[UUID, float]:
    """Calculate PageRank scores for chunks."""
    chunk_ids = list(self.chunk_nodes.keys())
    n = len(chunk_ids)

    # Initialize scores uniformly
    scores = {cid: 1.0 / n for cid in chunk_ids}

    for iteration in range(max_iterations):
        new_scores = {}

        for cid in chunk_ids:
            # Random jump
            rank = (1 - damping) / n

            # Contribution from neighbors
            for neighbor_id, weight in edges[cid].items():
                neighbor_out_weight = sum(edges[neighbor_id].values())
                if neighbor_out_weight > 0:
                    rank += damping * scores[neighbor_id] * weight / neighbor_out_weight

            new_scores[cid] = rank

        # Check convergence
        diff = sum(abs(new_scores[cid] - scores[cid]) for cid in chunk_ids)
        scores = new_scores

        if diff < tolerance:
            break

    return scores
```

### 6. Select Core Chunks

```python
def _select_core_chunks(self, core_ratio: float = 0.1):
    """Select top N% chunks by PageRank as 'core'."""
    # Sort by PageRank score
    sorted_chunks = sorted(
        self.chunk_nodes.values(),
        key=lambda c: c.pagerank_score,
        reverse=True
    )

    # Select top N%
    num_core = max(1, int(len(sorted_chunks) * core_ratio))

    for i, chunk in enumerate(sorted_chunks):
        chunk.is_core = (i < num_core)
```

## Usage

### During Ingestion

```python
from khora.engines.khora.skeleton import SkeletonIndexer

indexer = SkeletonIndexer()

# Add all chunks (fast, no LLM)
for chunk in chunks:
    indexer.add_chunk(chunk.id, chunk.content)

# Build skeleton - identifies core chunks
core_chunk_ids = indexer.build_skeleton(core_ratio=0.1)

# Only extract entities from core chunks
for chunk_id in core_chunk_ids:
    chunk = get_chunk(chunk_id)
    entities = await extract_entities(chunk.content)  # LLM call
    await store_entities(chunk_id, entities)

# Non-core chunks: store keywords only (no LLM)
for chunk in chunks:
    if chunk.id not in core_chunk_ids:
        keywords = indexer.chunk_nodes[chunk.id].keywords
        await store_keywords(chunk.id, keywords)
```

### During Retrieval

```python
# Search returns a mix of core and non-core chunks
results = await vector_search(query_embedding, limit=20)

for chunk_id, score in results:
    if indexer.is_core_chunk(chunk_id):
        # Core chunk: has pre-extracted entities
        entities = await get_entities(chunk_id)
    else:
        # Non-core chunk: expand lazily if needed
        entities = await lazy_expander.maybe_expand(chunk_id)
```

## Lazy Entity Expansion

Non-core chunks can be expanded on-demand during retrieval:

```python
class LazyEntityExpander:
    def __init__(self, skeleton_indexer: SkeletonIndexer):
        self.indexer = skeleton_indexer
        self.expanded_chunks: set[UUID] = set()

    async def maybe_expand(
        self,
        chunk_id: UUID,
        chunk_content: str | None = None,
    ) -> list[str]:
        """Maybe extract entities for non-core chunk."""
        # Already expanded
        if chunk_id in self.expanded_chunks:
            return await get_stored_entities(chunk_id)

        # Core chunks already have entities
        if self.indexer.is_core_chunk(chunk_id):
            return await get_stored_entities(chunk_id)

        # For non-core: return keywords as pseudo-entities
        # Or trigger full extraction for high-relevance chunks
        chunk_node = self.indexer.chunk_nodes.get(chunk_id)
        if chunk_node:
            return list(chunk_node.keywords)

        return []
```

### Full Expansion (Optional)

For highly relevant non-core chunks, you can trigger full LLM extraction:

```python
async def expand_fully(self, chunk_id: UUID, chunk_content: str):
    """Perform full LLM extraction on a non-core chunk."""
    entities = await extract_entities(chunk_content)  # LLM call
    await store_entities(chunk_id, entities)
    self.expanded_chunks.add(chunk_id)
    return entities
```

## Related Chunk Discovery

The skeleton graph enables finding related chunks:

```python
def get_related_chunks(
    self,
    chunk_id: UUID,
    limit: int = 10,
    min_similarity: float = 0.1,
) -> list[tuple[UUID, float]]:
    """Find chunks related via keyword overlap."""
    chunk = self.chunk_nodes.get(chunk_id)
    if not chunk:
        return []

    similarities = []
    for other_id, other_chunk in self.chunk_nodes.items():
        if other_id == chunk_id:
            continue

        # Jaccard similarity of keyword sets
        intersection = len(chunk.keywords & other_chunk.keywords)
        union = len(chunk.keywords | other_chunk.keywords)
        similarity = intersection / union if union > 0 else 0

        if similarity >= min_similarity:
            similarities.append((other_id, similarity))

    return sorted(similarities, key=lambda x: -x[1])[:limit]
```

## Keyword Search

Search chunks by keywords without embeddings:

```python
def search_by_keywords(
    self,
    keywords: list[str],
    limit: int = 10,
) -> list[tuple[UUID, float]]:
    """Search chunks by keyword match, weighted by IDF."""
    scores = defaultdict(float)

    for keyword in keywords:
        keyword_node = self.keyword_nodes.get(keyword.lower())
        if keyword_node:
            for chunk_id in keyword_node.chunk_ids:
                scores[chunk_id] += keyword_node.idf_score

    return sorted(scores.items(), key=lambda x: -x[1])[:limit]
```

## Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `core_ratio` | 0.1 | Percentage of chunks to mark as core (10%) |
| `damping` | 0.85 | PageRank damping factor |
| `max_iterations` | 100 | Maximum PageRank iterations |
| `tolerance` | 1e-6 | Convergence tolerance |
| `max_keywords` | 20 | Keywords to extract per chunk |

### Tuning Guidelines

**`core_ratio`:**
- Higher (0.2-0.3): Better retrieval quality, higher cost
- Lower (0.05): Lower cost, may miss important content
- Default (0.1): Good balance for most use cases

**When to increase `core_ratio`:**
- Documents are highly interconnected
- Precision is critical
- Budget allows more LLM calls

**When to decrease `core_ratio`:**
- Large corpus with many similar documents
- Cost is primary concern
- Documents are relatively independent

## Cost Analysis

### Extraction Costs

| Corpus Size | Full Extraction | Skeleton (10%) | Savings |
|-------------|-----------------|----------------|---------|
| 1,000 docs | 5,000 calls | 500 calls | 90% |
| 10,000 docs | 50,000 calls | 5,000 calls | 90% |
| 100,000 docs | 500,000 calls | 50,000 calls | 90% |

### Quality Trade-offs

| Aspect | Full Extraction | Skeleton |
|--------|-----------------|----------|
| Entity coverage | 100% | ~70-80% |
| Relationship coverage | 100% | ~50-60% |
| Retrieval recall | Baseline | ~95% of baseline |
| Retrieval precision | Baseline | ~98% of baseline |

*Note: Quality metrics are approximate and depend on corpus characteristics.*

## Implementation Details

### Memory Usage

The skeleton indexer maintains in-memory data structures:

```python
# Per chunk: ~1KB (keywords, scores)
# Per keyword: ~100 bytes (chunk IDs, IDF)
# 10,000 chunks → ~15 MB memory

class SkeletonIndexer:
    chunk_nodes: dict[UUID, ChunkNode]    # O(n) space
    keyword_nodes: dict[str, KeywordNode]  # O(k) space, k = unique keywords
```

### Batch Processing

For large corpora, process in batches:

```python
# Process 1000 chunks at a time
batch_size = 1000
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    for chunk in batch:
        indexer.add_chunk(chunk.id, chunk.content)

# Build skeleton after all chunks added
core_ids = indexer.build_skeleton(core_ratio=0.1)
```

### Persistence

The skeleton can be persisted for incremental updates:

```python
# Save skeleton state
state = {
    "chunk_nodes": {
        str(cid): {
            "keywords": list(node.keywords),
            "pagerank_score": node.pagerank_score,
            "is_core": node.is_core,
        }
        for cid, node in indexer.chunk_nodes.items()
    },
    "keyword_nodes": {
        kw: {
            "chunk_ids": [str(cid) for cid in node.chunk_ids],
            "idf_score": node.idf_score,
        }
        for kw, node in indexer.keyword_nodes.items()
    },
}
```

## Related Documentation

- [Khora Engine](khora-engine.md) - Overview of the Khora engine
- [Engine Comparison](engine-comparison.md) - Cost comparison with GraphRAG
- [References](../REFERENCES.md) - KET-RAG paper citation
