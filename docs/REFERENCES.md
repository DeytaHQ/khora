# References

This document lists research papers, systems, and approaches that inspired Khora's design.

## Research Papers

### Microsoft GraphRAG

> **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**
> Microsoft Research, 2024
> https://microsoft.github.io/graphrag/

The foundational paper for knowledge graph-based RAG. Key contributions:
- Entity and relationship extraction from text
- Community detection for document clustering
- Graph-based query answering

**Khora adoption:**
- GraphRAG engine implements full entity extraction
- Community detection for related content discovery
- Graph traversal for multi-hop queries

---

### TG-RAG (Time-Aware Graph RAG)

> **TG-RAG: Time-Aware Graph Retrieval-Augmented Generation**
> arXiv:2410.15149, October 2024
> https://arxiv.org/abs/2410.15149

Introduces hierarchical time structures for temporal reasoning in RAG:
- Year → Quarter → Month → Week → Day hierarchy
- Time-based edge linking
- Temporal query decomposition

**Khora adoption:**
- `TimeHierarchyBuilder` implements the hierarchical time graph
- Automatic ancestor creation for temporal navigation
- Time range queries via hierarchy traversal

---

### KET-RAG (Knowledge-Enhanced Text RAG)

> **KET-RAG: A Cost-Effective Multi-Granular Indexing for Graph-RAG**
> arXiv:2502.00596, January 2025
> https://arxiv.org/abs/2502.00596

Cost optimization through skeleton-based indexing:
- PageRank identifies semantically central chunks
- Only core chunks (~10%) require LLM extraction
- Non-core chunks use keyword-based retrieval

**Khora adoption:**
- `SkeletonIndexer` implements PageRank-based core selection
- `LazyEntityExpander` for on-demand extraction
- 5-10x reduction in LLM extraction calls

---

### Entity Resolution Survey (Papadakis et al.)

> **Blocking and Filtering Techniques for Entity Resolution**
> Papadakis et al., ACM Computing Surveys, 2020
> doi:10.1145/3377455

Comprehensive survey of entity resolution techniques:
- Blocking strategies for scalability
- String similarity measures
- Machine learning approaches

**Khora adoption:**
- GraphRAG engine's entity deduplication
- Name normalization and matching
- Confidence scoring for entity merging

---

### Graphiti

> **Graphiti: Build Real-time, Temporally-Aware Knowledge Graphs**
> Zep AI, 2024
> https://github.com/getzep/graphiti

Bi-temporal knowledge graph system:
- Transaction time vs. valid time separation
- Edge invalidation for contradicting facts
- Episodic memory with temporal context

**Khora adoption:**
- Bi-temporal model (`occurred_at` vs `ingested_at`)
- `TemporalEdgeStorage` with conflict detection
- Edge validity windows and invalidation tracking

---

## Competing Systems

Khora was designed with awareness of these systems:

### Cognee

> https://github.com/topoteretes/cognee

Knowledge graph memory system:
- Multi-step entity extraction
- Graph-based retrieval
- Modular pipeline architecture

**Comparison:**
- Cognee focuses on knowledge graphs
- Khora offers dual engines (GraphRAG + temporal-first)
- Khora adds bi-temporal support

---

### Mem0

> https://github.com/mem0ai/mem0

Intelligent memory layer for LLMs:
- Automatic memory updates
- Entity-based memory organization
- Cross-session memory

**Comparison:**
- Mem0 targets conversational AI memory
- Khora offers document-level memory with rich search
- Khora supports multi-tenant isolation

---

### LangChain Memory

> https://python.langchain.com/docs/concepts/memory/

Conversation memory for LangChain:
- Buffer memory, summary memory, entity memory
- Redis/PostgreSQL backends
- Conversation-centric design

**Comparison:**
- LangChain memory is conversation-focused
- Khora supports document and event storage
- Khora offers graph-based and temporal-first options

---

## Algorithms & Techniques

### Reciprocal Rank Fusion (RRF)

> **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods**
> Cormack et al., SIGIR 2009

Technique for combining multiple ranked lists:
- Scale-invariant fusion
- No calibration required
- Robust to score distribution differences

**Khora adoption:**
- Hybrid search combines vector + BM25 with RRF
- `hybrid_alpha` controls relative weighting

---

### PageRank

> **The PageRank Citation Ranking: Bringing Order to the Web**
> Page et al., Stanford InfoLab, 1999

Graph centrality algorithm:
- Iterative score propagation
- Damping factor for convergence
- Identifies important nodes

**Khora adoption:**
- Skeleton indexing uses PageRank to find core chunks
- Keyword-weighted edges between chunks
- Core chunks selected by PageRank score

---

### BRIN Indexes

> **BRIN Indexes in PostgreSQL**
> PostgreSQL Documentation

Block Range Indexes for time-series data:
- 99% space savings vs B-tree
- Efficient for sorted, append-only data
- Ideal for temporal range queries

**Khora adoption:**
- `occurred_at` indexed with BRIN
- Efficient temporal filtering
- Low storage overhead

---

### HNSW (Hierarchical Navigable Small World)

> **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs**
> Malkov & Yashunin, IEEE TPAMI, 2020

Approximate nearest neighbor algorithm:
- Logarithmic search complexity
- High recall with low latency
- Memory-efficient graph structure

**Khora adoption:**
- pgvector uses HNSW for vector similarity
- Configurable `m` and `ef_construction` parameters
- Sub-100ms queries on millions of vectors

---

## Implementation Notes

### Paper-to-Code Mapping

| Paper/System | Khora Component | File |
|--------------|-----------------|------|
| TG-RAG | TimeHierarchyBuilder | `engines/khora/time_hierarchy.py` |
| KET-RAG | SkeletonIndexer | `engines/khora/skeleton.py` |
| Graphiti | TemporalEdgeStorage | `engines/khora/temporal_edges.py` |
| GraphRAG | GraphRAGEngine | `engines/graphrag/engine.py` |
| RRF | rrf_fusion() | `engines/khora/backends/pgvector.py` |

### Deviations from Papers

1. **TG-RAG**: Paper uses Neo4j; Khora uses PostgreSQL tables for portability
2. **KET-RAG**: Paper uses community detection; Khora simplifies to PageRank-only
3. **Graphiti**: Paper is Neo4j-native; Khora abstracts to multiple backends

---

## Citation

If you use Khora in research, please cite:

```bibtex
@software{khora2025,
  title = {Khora: A Memory Lake for Knowledge Graphs and Temporal Events},
  author = {Deyta},
  year = {2025},
  url = {https://github.com/DeytaHQ/khora}
}
```

## Further Reading

- [Khora Engine Documentation](engines/khora-engine.md)
- [Engine Comparison](engines/engine-comparison.md)
- [Temporal Model Deep Dive](engines/temporal-model.md)
- [Skeleton Indexing](engines/skeleton-indexing.md)
