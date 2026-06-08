# Retrieval Tuning

This document explains changes made to Khora's retrieval pipeline in response to benchmark analysis against Cognee, Graphiti, and Mem0. The short version: Khora's query pipeline was too aggressive at filtering results, causing 25% of queries to return nothing. The fixes are mostly about lowering thresholds and adding fallback paths.

## Background: What the Benchmarks Showed

We ran the `retrieval_basic` benchmark (120 documents, 55 queries across 3 difficulty levels) against four systems. Khora had a serious problem: **25.5% of queries returned zero results**. Not low-quality results - literally nothing.

The strangest finding was that Khora performed *worst on the easiest queries*. Simple factual lookups like "wrought-iron tower built for the 1889 World's Fair in Paris" (expecting the Eiffel Tower document) returned nothing, while complex multi-concept queries like "quantum mechanical effects near black hole event horizons" returned perfect results.

Every other system found the Eiffel Tower document. Khora didn't.

## The Three Compounding Problems

The zero-result failures weren't caused by a single bug. Three independent issues compounded to create a retrieval dead zone for paraphrased and descriptive queries.

### Problem 1: Similarity Threshold Too High

The most impactful issue. When you called `kb.recall("some query")`, the default `min_similarity` was `0.5`. This value propagated down to pgvector as a hard `WHERE similarity >= 0.5` filter at the database level. Any chunk with cosine similarity below 0.5 was silently discarded before Khora even had a chance to rank it.

For a descriptive query like "wrought-iron tower built for the 1889 World's Fair in Paris", the embedding similarity to a document about the Eiffel Tower might be 0.35–0.49. That's clearly relevant - a human would call it a match - but the threshold threw it away.

The threshold chain was:

```
Khora.recall(min_similarity=0.5)        # caller-facing default
    → QueryConfig(min_chunk_similarity=0.3)   # internal default
    → pgvector WHERE similarity >= 0.3        # DB-level filter
```

When using `Khora.recall()` without arguments, the `0.5` default overrode the `QueryConfig` default of `0.3`, making things even worse.

For comparison: Cognee applies no threshold at all - it returns whatever pgvector finds, ranked by distance, and lets the caller decide what's "good enough."

### Problem 2: No Keyword Search in HYBRID Mode

The default search mode is `HYBRID`, which runs vector search and graph search in parallel. Keyword search (PostgreSQL full-text search via `tsvector`/`tsquery`) was only activated in `ALL` mode:

```python
# Before: keyword search gated behind ALL mode
if cfg.mode == SearchMode.ALL and cfg.enable_keyword_search:
```

This meant that for a query like "Eiffel Tower 1889 Paris", there was no keyword/full-text fallback. If vector similarity was below the threshold and entity linking found nothing, the query had no other path to find results.

Keyword search is exactly the safety net you want for queries containing proper nouns, dates, and specific terms. It works on a completely different principle (term frequency, not embedding similarity), so it catches cases that vector search misses.

### Problem 3: Graph Search Cascading Failure

Graph search works by finding entities first, then traversing their relationships to discover connected chunks. It finds entities through two paths:

1. **Entity linking**: Matches query mentions to stored entities (exact, fuzzy, or embedding match)
2. **Entity embedding search**: Finds entities with similar embeddings

Both paths had the same high thresholds. Entity linking required 0.8 fuzzy match ratio and 0.7 embedding similarity. Entity embedding search used the same `min_entity_similarity` as vector search.

For paraphrased queries, entity linking often found nothing (no exact or fuzzy match), and entity embedding search filtered out candidates below the threshold. When graph search has zero entities to start from, it returns zero chunks. This meant both vector *and* graph returned nothing simultaneously.

## What Changed

### Similarity Thresholds (P0)

The `Khora.recall()` default `min_similarity` changed from `0.5` to `0.0`. The `QueryConfig` defaults for `min_chunk_similarity` and `min_entity_similarity` changed from `0.3` to `0.05`.

Setting `min_similarity=0.0` at the `Khora` level means: don't filter at the database level, let the ranking pipeline (RRF fusion, reranking) decide what's relevant. The small `0.05` default in `QueryConfig` is a noise floor - it filters out truly random matches without discarding anything a human might consider relevant.

If you have a use case where you want strict filtering (e.g., only returning very confident matches), you can still pass `min_similarity=0.7` explicitly. The change only affects the default behavior.

Files changed:
- `khora.py`: `min_similarity` parameter default `0.5` → `0.0`
- `query/engine.py`: `QueryConfig.min_chunk_similarity` default `0.3` → `0.05`
- `query/engine.py`: `QueryConfig.min_entity_similarity` default `0.3` → `0.05`
- `config/schema.py`: `QuerySettings.min_chunk_similarity` default `0.3` → `0.05`
- `config/schema.py`: `QuerySettings.min_entity_similarity` default `0.3` → `0.05`

### Keyword Search in HYBRID Mode (P0)

Keyword search now runs alongside vector and graph search in `HYBRID` mode, not just `ALL` mode:

```python
# After: keyword search runs in HYBRID and ALL
if cfg.mode in (SearchMode.HYBRID, SearchMode.ALL) and cfg.enable_keyword_search:
```

This means the default `HYBRID` mode now uses all three search methods: vector similarity, graph traversal, and keyword/full-text matching. The existing RRF fusion weights still apply (vector: 0.5, graph: 0.3, keyword: 0.2), so keyword results contribute without dominating.

This is arguably what `HYBRID` should have always meant. The previous behavior (vector + graph only) is now what you'd get with `SearchMode.HYBRID` and `enable_keyword_search=False`.

File changed: `query/engine.py`

### Zero-Result Fallback (P0)

After the fusion step, if no chunks were found, the engine now retries with relaxed parameters:

1. **Vector search with `min_similarity=0.0`** - in case the configured threshold filtered out borderline results
2. **Keyword/full-text search** - if it wasn't already part of the search (e.g., `VECTOR` or `GRAPH` mode)

Fallback results go through the same RRF fusion, so they're properly ranked. This is a safety net, not the primary path - with the lowered thresholds, most queries should find results on the first pass.

The fallback only fires when the initial search returns literally nothing. It doesn't activate for low-quality results or few results.

File changed: `query/engine.py` (in the `query()` method, after RRF fusion)

### Reranking Skip for Small Result Sets (P1)

Neural reranking (cross-encoder) now only runs when there are 5 or more candidate chunks. When fewer than 5 results are available, the existing RRF scores are already a reasonable ranking, and the cross-encoder adds several seconds of latency for minimal benefit.

This matters most for the zero-result fallback path, where recovery might produce only 2-3 chunks. Without this change, those 2-3 chunks would still go through the full cross-encoder pipeline.

File changed: `query/engine.py`

### Entity Linking Thresholds (P2)

The fuzzy matching threshold dropped from 0.8 to 0.6, and the embedding similarity threshold from 0.7 to 0.4.

The previous 0.8 fuzzy threshold required near-exact string matches (e.g., "Einstein" would match "Einstien" but not "Albert Einstein"). At 0.6, more reasonable variations get through to the linking step, where further disambiguation happens.

The 0.7 embedding threshold for entity linking was stricter than the chunk similarity threshold, which made no sense - if you're willing to consider a chunk at 0.3 similarity, you should be willing to consider an entity match at 0.4.

Files changed:
- `query/engine.py`: `QueryConfig` defaults
- `config/schema.py`: `QuerySettings` defaults

### Graph Search Entity Fallback (P2)

When graph search finds no entities via embedding similarity (using the configured threshold), it now retries with `min_similarity=0.0` to find the top 3 closest entities regardless of distance. This prevents the cascading failure where graph search contributes nothing because it couldn't find a starting entity.

The fallback entities will have lower similarity scores, which propagates through to their chunk scores, so they won't unfairly dominate the fusion results.

File changed: `query/engine.py` (in `_graph_search()`)

## Threshold Philosophy

The old approach was: filter aggressively at each stage, trusting that only high-similarity results are relevant. This works when queries closely match document content (exact entity names, similar vocabulary), but fails for paraphrased or descriptive queries.

The new approach is: cast a wide net at the retrieval stage, and rely on the ranking pipeline (RRF fusion, source priority boosting, neural reranking) to surface the best results. This matches what Cognee and Graphiti do - they retrieve broadly and rank carefully.

The ranking pipeline is Khora's strength. Query understanding adjusts fusion weights per query. Entity linking boosts chunks connected to recognized entities. Cross-encoder reranking considers query-document pairs holistically. All of these work *better* when they have more candidates to work with. A 0.35-similarity chunk that's the right answer is better than zero results.

There's a trade-off: lower thresholds mean more candidates pass through the pipeline, which adds some latency. In practice, pgvector returns results in distance order, so you're adding a few more low-scoring candidates to the tail - the fusion step is O(n) and fast. The reranking skip for small result sets also helps offset this.

## Configuration Reference

All thresholds can be overridden per-query or via environment variables:

```python
# Per-query override
result = await kb.recall(
    "specific query",
    namespace=ns_id,
    min_similarity=0.3,  # stricter than default
)

# Or via QueryConfig for full control
config = QueryConfig(
    min_chunk_similarity=0.2,
    min_entity_similarity=0.2,
    entity_linking_fuzzy_threshold=0.7,
    entity_linking_embedding_threshold=0.5,
)
```

Environment variables:
```bash
KHORA_QUERY_MIN_CHUNK_SIMILARITY=0.05     # default
KHORA_QUERY_MIN_ENTITY_SIMILARITY=0.05    # default
KHORA_QUERY_ENTITY_LINKING_FUZZY_THRESHOLD=0.6
KHORA_QUERY_ENTITY_LINKING_EMBEDDING_THRESHOLD=0.4
```

### Adaptive Top-K for Focused Queries

The query engine uses the `complexity_score` from query understanding to trim the evidence set for focused queries. There are two reduction branches, and both only ever *lower* the limit - they never raise it, and they fire only when the configured `max_chunks` is above 8 and the query does not require multi-step reasoning:

| Reason | Condition | Effect |
|------|-----------|--------|
| `very_focused` | `complexity_score < 0.3` and not multi-step and `max_chunks > 8` | Caps `max_chunks` and `max_entities` at 8; raises the chunk/entity similarity floor to at least 0.25 |
| `single_topic` | `complexity_score < 0.5` and not multi-step and `max_chunks > 8` | Caps `max_chunks` and `max_entities` at 8; raises the chunk/entity similarity floor to at least 0.15 |

Neither branch sets a 3/5/10/15 chunk ladder, and there is no tier that raises the limit for complex queries: a complex query keeps the configured `max_chunks` unchanged. At the default `max_chunks=10`, a focused query is reduced to 8. When the firing branch reduces the limit, it records `metadata["adaptive_top_k"] = {"reduced": True, "reason": ...}`. The complexity score is computed during query understanding based on entity count, relationship complexity, and temporal references.

### MMR Diversity Selection

The diversity stage (Stage 5 of the query pipeline) uses Maximal Marginal Relevance to select a diverse set of results from the candidate pool:

1. **Enabled by default**: `enable_diversity` defaults to `True` in both `QueryConfig` and `QuerySettings`.

2. **Rust acceleration**: MMR selection uses a 3-tier fallback (Rust → NumPy → pure Python). The Rust implementation in `khora-accel` uses SIMD-friendly dot product with GIL release, providing ~5x speedup over pure Python for typical result set sizes.

3. **Pre-normalized embeddings**: Embeddings are L2-normalized at ingest time, allowing MMR to use dot product instead of cosine similarity (~3x speedup since normalization is amortized).

Configuration:
```python
config = QueryConfig(
    enable_diversity=True,    # default: True
    diversity_lambda=0.5,     # default; balance: 1.0 = pure relevance, 0.0 = pure diversity
)
```

Environment variable:
```bash
KHORA_QUERY_ENABLE_DIVERSITY=true   # default
KHORA_QUERY_DIVERSITY_LAMBDA=0.5    # default
```

### Coherence Scoring

The VectorCypher retriever applies a lightweight text coherence signal after RRF fusion to penalize word-shuffled confounders. This is particularly effective when LLM reranking is disabled (the default), where confounders would otherwise rank alongside genuine results.

**How it works:** `bigram_coherence_score()` checks function-word transitions (articles → content words, prepositions → noun phrases). Genuine text has predictable bigram patterns; word-shuffled text does not. The score is blended into the RRF score via `apply_coherence_boost()`.

**Configuration:**

```python
config = RetrieverConfig(
    coherence_weight=0.1,  # default; tunable 0.0–0.5
)
```

| Weight | Effect |
|--------|--------|
| 0.0 | Disabled - pure RRF ranking |
| 0.1 | Default - gentle confounder demotion |
| 0.3+ | Aggressive - may over-penalize informal text |

Coherence scoring complements, but does not replace, MMR diversity selection. MMR removes same-document dominance; coherence scoring removes incoherent text. Both can be enabled simultaneously (the default).

> **Note:** Coherence scoring only applies to the VectorCypher retriever pipeline.

### Reranking

After RRF fusion, an optional neural reranker rescores the top candidates as query–document *pairs* (a cross-encoder, unlike the bi-encoder used for the initial embedding search). The reranker is cached across queries and runs in `asyncio.to_thread`, and it [skips small result sets](#reranking-skip-for-small-result-sets-p1) (<5 chunks).

On the default **VectorCypher** engine, reranking is **on by default**: `query.enable_reranking` defaults to `True`, and the engine reconciles the `query.reranking_*` family — and the matching `KHORA_QUERY_RERANKING_*` env vars — onto its config. So the simplest way to configure it is through `KhoraConfig.query` / env vars:

```bash
KHORA_QUERY_ENABLE_RERANKING=true                      # default
KHORA_QUERY_RERANKING_MODEL=BAAI/bge-reranker-v2-m3    # default
KHORA_QUERY_RERANKING_TOP_N=50                         # default
KHORA_QUERY_RERANKING_BLEND_WEIGHT=0.7                 # default
```

| Env var (`KHORA_QUERY_…`) / `query.*` field | Default | Notes |
|---|---|---|
| `ENABLE_RERANKING` / `enable_reranking` | `true` | Master on/off. **On by default.** |
| `RERANKING_MODEL` / `reranking_model` | `BAAI/bge-reranker-v2-m3` | Any model loadable by `CrossEncoder(name)`. |
| `RERANKING_TOP_N` / `reranking_top_n` | `50` | Candidates fed to the reranker. |
| `RERANKING_BLEND_WEIGHT` / `reranking_blend_weight` | `0.7` | Reranker-score weight when blending with the original fused score (`0.7` = 70 % reranker / 30 % original). |

> **Heads-up — reranking is on by default and loads a ~2.3 GB model.** With the defaults, the first `recall()` downloads and loads `BAAI/bge-reranker-v2-m3` (best on GPU). To disable it set `KHORA_QUERY_ENABLE_RERANKING=false` (or `query.enable_reranking=False`); to keep reranking but cut the footprint, pick a lighter `reranking_model` (see below).

**The default reranker.** The default is **[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)** — a 568M-parameter multilingual (XLM-RoBERTa-large) reranker chosen for its markedly stronger discrimination than the legacy MS MARCO MiniLM cross-encoders (on GraphRAG-Bench it lifted medium-difficulty accuracy from ~0.796 to ~0.82). The trade-off is weight: it downloads **~2.3 GB** and is best run on a **GPU** (the reranker auto-detects the device). It emits relevance logits rather than 0–1 scores; the default `reranking_blend_weight=0.7` handles that fine.

**Choosing a lighter model (CPU-only / latency-sensitive).** On CPU-only deployments, or when the ~2.3 GB download and per-query cost of bge are too much, switch to a small MS MARCO MiniLM cross-encoder. The `L-N` suffix is the number of transformer layers ([sentence-transformers pretrained models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)) — more layers means more accurate but slower:

| Model | Notes |
|---|---|
| `BAAI/bge-reranker-v2-m3` (default) | Strongest relevance; ~2.3 GB; GPU recommended. |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Lightweight; ~74.3 NDCG@10 on TREC-DL. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightest; near-identical quality to L-12, ~2× faster. Best CPU default. |

Because the reranker is cached by `(model, include_date_prefix)`, switching the model is a one-line config change:

```bash
KHORA_QUERY_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2   # lighter, CPU-friendly
```

**Per-engine overrides with `VectorCypherConfig`.** For finer control you can pass a `VectorCypherConfig` via `engine_kwargs`. Any field you set there explicitly wins over `query.*`, which in turn wins over the `VectorCypherConfig` defaults:

```python
from khora import Khora
from khora.engines.vectorcypher import VectorCypherConfig

kb = Khora(
    config,
    engine="vectorcypher",
    engine_kwargs={"vectorcypher_config": VectorCypherConfig(reranking_top_n=100)},
)
```

> **Precedence:** explicit `vectorcypher_config` field > `query.*` / `KHORA_QUERY_RERANKING_*` > `VectorCypherConfig` default. One edge case: a `vectorcypher_config` field set *to its own dataclass default* can't be distinguished from "unset", so `query.*` wins for that field — configure those via `query.*` instead.

**LLM listwise reranking (opt-in).** For temporal queries you can chain an LLM reranker *after* the cross-encoder stage. It only fires when the cross-encoder is *not* already confident — i.e. when the rank-1-vs-rank-2 score gap is below the confidence threshold — so most queries never pay the extra LLM call:

| Env var (`KHORA_QUERY_…`) / `query.*` field | Default | Notes |
|---|---|---|
| `ENABLE_LLM_RERANKING` / `enable_llm_reranking` | `false` | Opt-in; runs after the cross-encoder, temporal queries only. |
| `LLM_RERANKING_MODEL` / `llm_reranking_model` | `gpt-4o-mini` | Model for the listwise pass. |
| `LLM_RERANKING_TOP_N` / `llm_reranking_top_n` | `10` | Top candidates sent to the LLM. |
| `LLM_RERANKING_CONFIDENCE_THRESHOLD` / `llm_reranking_confidence_threshold` | `0.1` | Trigger only when the cross-encoder rank-1/rank-2 gap is below this. |
| `llm_reranking_mode` (`VectorCypherConfig` only) | `"auto"` | `"auto"` gates the LLM step on version metadata; `"always"` forces it on every temporal query. |

> **Note:** `KHORA_QUERY_RERANKING_METHOD` and `KHORA_QUERY_RERANKING_FINAL_K` exist on `QuerySettings` but have no `VectorCypherConfig` equivalent, so they are **not** applied on the VectorCypher (`recall()`) path — they affect the separate `HybridQueryEngine` only.

See also the opt-in [cross-encoder date-prefix experiment](#whats-next) below, which prepends `[YYYY-MM-DD]` to each candidate before scoring.

## What's Next

These changes should eliminate the zero-result problem and significantly improve retrieval quality on descriptive/paraphrased queries. The benchmark should be re-run to validate the expected impact:

- Zero-result rate: 25.5% → near 0%
- MRR: 0.736 → estimated 0.85–0.95
- Hit rate: 74.5% → near 100%
- Easy query MRR: 0.433 → estimated 0.90+
- Latency: some improvement from reranking skip, but the main latency contributors (query understanding, reranking) are unchanged

Further improvements to consider:
- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical document for the query to improve embedding similarity for descriptive queries. Mode is controlled by `KHORA_QUERY_ENABLE_HYDE` taking `auto` (default), `always`, or `never` (booleans `True`/`False` are still accepted and normalize to `always`/`never`). In `auto` mode HyDE fires when the query understanding layer flags the query as complex or temporal. RECENCY / STATE_QUERY / CHANGE queries get a time-anchored hypothetical that injects today's ISO date - see [Temporal queries](temporal-queries.md#temporal-anchored-hyde).
- **HyDE-Cypher (opt-in)**: For *structured* RECENCY queries (e.g. "latest action items", "who works for Acme", "Phoenix and security recently"), khora can ask an LLM to pick a parameterized Cypher template and execute it against the graph backend as an additional retrieval channel. Three templates ship: `recent_by_type`, `entity_relationships`, `cooccurrence`. Slot values are validated against `ExpertiseConfig` whitelists and bound via Neo4j parameters - slot strings never reach the Cypher source. Enable via `KHORA_QUERY_ENABLE_HYDE_CYPHER=true`; cap result-set size with `KHORA_QUERY_HYDE_CYPHER_LIMIT` (default 20). **Default OFF - flip after an A/B run on a hand-curated structured-query set.**
- **Cross-encoder date-prefix experiment (opt-in)**: `CrossEncoderReranker(include_date_prefix=True)` prepends `[YYYY-MM-DD] ` to each candidate's content before scoring. Off-the-shelf rerankers tokenize ISO dates fine and the extra ~12 tokens per candidate are negligible vs. the model's forward pass. Source-priority: `metadata.custom.occurred_at` → `metadata.custom.sent_at` → `metadata.created_at`. **Default OFF - A/B required before flipping.**
- **SearchMode.ALL as default**: Now that keyword search runs in HYBRID, the distinction between HYBRID and ALL is smaller - HYBRID is effectively ALL.
