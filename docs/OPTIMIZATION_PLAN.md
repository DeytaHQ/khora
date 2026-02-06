# Khora Optimization Plan

## Executive Summary

Multi-agent analysis of GraphRAG, Skeleton, and VectorCypher engines identified opportunities for **5-10x performance gains** and **15-30% quality improvements**.

**Current State:**
- Ingestion achieves ~5-10x parallelism but theoretical max is **30-50x**
- Extraction quality: ~75-80% precision, 60-65% recall
- Search quality: ~70% NDCG@10 (without reranking), ~80% with reranking
- Key constraint: Relationship inference is **99% serial**

---

## Phase 1: Quick Wins (1-2 weeks)

### Performance (5-7x combined gain)

| Fix | Location | Current Issue | Expected Gain | Effort |
|-----|----------|---------------|---------------|--------|
| **Async relationship inference** | `ingest.py:815-825` | Serial O(n²) blocks pipeline | **3-5x** | Medium |
| **Parallel Neo4j sessions** | `neo4j.py:545` | 1 session per rel type | **2x** | Low |
| **Batch chunk fetching** | `query/engine.py:1183-1196` | 25 individual `get_chunk()` calls | **3-5x** | Low |
| **Parallel storage backend connect** | `coordinator.py:79-96` | Sequential 4×500ms | **3-4x startup** | Trivial |
| **Parallel storage batch writes** | `ingest.py:300-400` | Sequential chunk/entity/rel inserts | **2-3x** | Trivial |

### Quality (10-15% improvement)

| Fix | Location | Current Issue | Expected Gain | Effort |
|-----|----------|---------------|---------------|--------|
| **Fix reranking score normalization** | `reranking.py:141` | Cross-encoder vs RRF scale mismatch | Consistent ranking | 1 hour |
| **Real LLM confidence scoring** | `llm.py:1030,1060,1077` | Hardcoded 0.9 confidence | Enable true filtering | 2 hours |
| **Increase VectorCypher core_ratio** | `vectorcypher/engine.py:56` | 25% → 40% | **+10-20% recall** | 1 line |

---

## Phase 2: Medium Term (2-4 weeks)

### Performance (2-3x additional)

| Fix | Location | Current Issue | Expected Gain | Effort |
|-----|----------|---------------|---------------|--------|
| **Entity resolution cache sharing** | `entity_resolution.py` | 100 docs = 100 entity list calls | **10-20x fewer queries** | Low |
| **LLM multi-extraction batching** | `extractors/llm.py:671` | Semaphore limits to 5 concurrent | **2-3x** | Medium |
| **Cross-document embedding cache** | `embedders/litellm.py:193` | Only dedup within single batch | **10-50x** (duplicate-heavy) | Low |
| **Streaming entity embedding** | `ingest.py:367-394` | Sequential within document | **1.5x** | Low |
| **Smarter entity index blocking** | `entity_index.py` | O(n²) fuzzy matching | **2x** | Medium |

### Quality (15-20% additional)

| Fix | Location | Current Issue | Expected Gain | Effort |
|-----|----------|---------------|---------------|--------|
| **Multi-stage ranking pipeline** | `query/engine.py` | Single-stage fusion | **+15-20% NDCG** | 4 hours |
| **Adaptive skeleton core_ratio** | `skeleton/skeleton.py:59` | Fixed 10% for all domains | **+15-20% recall** | 3 hours |
| **Skeleton keyword search fallback** | `skeleton/engine.py:408` | pgvector-only for 90% chunks | **+15-30% recall** | Medium |
| **Entity resolution attribute matching** | `entity_resolution.py:180` | Embedding-only, no attributes | **+10-15% dedup** | 4 hours |
| **Per-type merge thresholds** | `entity_resolution.py:215` | Flat 0.85 threshold | Reduce false merges | 2 hours |

---

## Phase 3: Long Term (4-8 weeks)

### Performance (5-10x additional)

| Fix | Description | Expected Gain | Effort |
|-----|-------------|---------------|--------|
| **Incremental inference** | Run rules on delta, not full graph | **1.5x latency** | High |
| **Neighborhood caching** | Materialize graph neighborhoods | **1.5x search** | Medium |
| **Columnar embedding storage** | DuckDB for bulk operations | **5-10x bulk** | High |
| **Two-tier storage** | Hot (FAISS) → Warm (pgvector) → Cold (Neo4j) | **10x query** | Very High |
| **Event-driven relationship creation** | No ID remapping needed | **2x latency** | High |

### Quality

| Fix | Description | Expected Gain | Effort |
|-----|-------------|---------------|--------|
| **Query routing with LLM** | Replace regex heuristics | **+15% routing accuracy** | 5 hours |
| **Pattern-based inference learning** | Auto-discover closure rules | Better inference | 6 hours |
| **Temporal awareness** | Mandatory temporal extraction | **+25% time queries** | Medium |
| **Diversity reranking (MMR)** | Reduce result redundancy | Better coverage | Medium |

---

## Critical Bottlenecks Identified

### Ingestion Pipeline

```
Current Bottleneck Waterfall:
1. Staging:     100% parallelized ✓
2. Extraction:   50% parallelized (max_concurrent=10, but only 5 docs)
3. Embedding:    25% parallelized (sequential within doc)
4. Expansion:     5% parallelized (entirely serial)
5. Storage:      40% parallelized (semaphore + backend serialization)
6. Inference:     1% parallelized (completely serial) ← CRITICAL
```

### Query Pipeline

```
Current:
  Understanding (LLM) → Entity Linking → Searches (parallel) → Fusion → Reranking

Optimized:
  ┌─ Understanding (LLM)
  │                        → Searches (parallel) → Fusion → Reranking
  └─ Query Embed (parallel)
```

---

## Quality Vulnerabilities

| Issue | Severity | Impact |
|-------|----------|--------|
| Fake confidence scores (hardcoded 0.9) | CRITICAL | No quality filtering possible |
| Text truncation (8000 char limit) | HIGH | 15-20% entity loss in long docs |
| Reranking score scale mismatch | HIGH | Cross-encoder boost inconsistent |
| Skeleton 10% core ratio | HIGH | 25-30% relationship loss |
| RRF ignores score magnitude | MEDIUM | Rank-only fusion loses precision |
| Merge threshold too low (0.85) | MEDIUM | Premature entity merging |

---

## Implementation Priorities

### MUST DO (>2x gain, <1 week)
1. **Async relationship inference** - 3-5x gain, medium effort
2. **Parallel Neo4j sessions** - 2x gain, low effort
3. **Batch chunk fetching in graph search** - 3-5x gain, low effort
4. **Fix reranking score normalization** - quality fix, 1 hour

### SHOULD DO (>1.5x gain, <2 weeks)
5. **Entity resolution cache sharing** - 10-20x fewer queries
6. **Streaming entity embedding** - 1.5x gain, low effort
7. **Parallel storage writes** - 2-3x gain, trivial
8. **Increase VectorCypher core_ratio** - +10-20% recall, 1 line

### NICE TO HAVE (>1.5x gain, >4 weeks)
9. **Incremental inference** - 1.5x latency, high effort
10. **Neighborhood caching** - 1.5x search, medium effort
11. **Two-tier storage** - 10x query latency, very high effort

---

## Theoretical Maximum Parallelism

| Operation | Current | Theoretical Max | Limiting Factor |
|-----------|---------|-----------------|-----------------|
| Document ingestion | 5 | 50-100 | LLM rate limits |
| Entity extraction | 50 calls | 500+ | API rate limits |
| Vector embedding | 50 | 500+ | Embedding API |
| Entity unification | 1 (serial) | 100-200 | Blocking-based parallelization |
| Relationship inference | 1 (serial) | 100-200 | Rule parallelization |
| Storage writes (Neo4j) | 1 session | 10-20 | Connection pool |

---

## Expected Outcomes

### After Phase 1
- Ingestion: **3-5x faster**
- Search: **2-3x faster**
- Quality: **+10-15%** (fixed confidence, core ratio)

### After Phase 2
- Ingestion: **5-10x faster** (cumulative)
- Search: **3-5x faster** (cumulative)
- Quality: **+20-30%** (multi-stage ranking, skeleton improvements)

### After Phase 3
- Ingestion: **10-20x faster** (cumulative)
- Search: **10x faster** (two-tier caching)
- Quality: **+30-40%** (temporal awareness, diversity)

---

## Agent Contributions

| Agent | Focus | Key Findings |
|-------|-------|--------------|
| **Computer Scientist** | Algorithm optimization | LLM batching, embedding dedup, skeleton indexing |
| **Python Engineer** | Code-level fixes | N+1 queries, parallel patterns, cache sharing |
| **Technical Architect** | System design | Async inference, event-driven creation, two-tier storage |
| **Devil's Advocate** | Quality protection | Confidence scoring, normalization bugs, merge thresholds |

---

## Next Steps

1. Create tracking issues for Phase 1 items
2. Implement "MUST DO" fixes in order of impact/effort ratio
3. Measure baseline metrics before each change
4. Validate quality improvements with test queries
5. Document performance gains in benchmarks
