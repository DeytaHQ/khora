# Recall Semantics: Scores, Ordering, Floors

What `kb.recall()` promises about the results it returns: what `chunk.score`
means, what decides the order, how the `min_similarity` floor behaves, and
which confidence signals ride along on `engine_info`. This page consolidates
the #811 / #1404 / #1425 / #1433 / #1441 contract that individual engine pages
reference.

## Order vs. score (#1433 / #1441)

The returned chunk **order** is the authoritative relevance ranking. It is
decided by the engine's full pipeline - RRF fusion across channels, recency
and coherence boosts, cross-encoder / LLM reranking, MMR diversity. The
internal sort key (`rrf_score`) is never exposed.

`chunk.score` is an **absolute** relevance signal, not the sort key:

- It is the raw query-to-chunk cosine similarity when one is available
  (captured pre-fusion, or computed on the fly for graph-only hits that carry
  an embedding), clamped to `>= 0.0`.
- It is `0.0` when no vector measurement exists (e.g. a graph-only hit whose
  stored form carries no embedding). `0.0` means "no vector-relevance
  measurement", **not** "irrelevant".
- Because it is absolute, it is comparable across queries and suitable for
  thresholding: an off-topic top result reads low (e.g. ~0.1) instead of the
  1.0 that per-result-set min-max normalization used to force (#811).

**Do not re-sort chunks by `score`.** That discards the graph, rerank, and
boost evidence baked into the order and may reorder results. Use the order
for ranking; use `score` for thresholds and cross-query comparison.

Source of truth: `RecallChunk` docstring (`src/khora/core/models/recall.py`)
and `attach_relevance_scores()` (`src/khora/engines/vectorcypher/fusion.py`).

## The `min_similarity` floor (#1404 / #1406 / #1425 / #1438 / #1445)

`recall(..., min_similarity=...)` is a hard cosine floor honored by every
mode (VECTOR / GRAPH / HYBRID / ALL / KEYWORD) on all three engines:

- Candidates below the floor are dropped at the **storage layer**, before any
  fusion or reranking.
- A per-call `min_similarity > 0` wins. `0.0` (the default) means "unset" and
  falls back to the configured `min_chunk_similarity`
  (`KHORA_QUERY_MIN_CHUNK_SIMILARITY`, default `0.0` = no floor since #1406).
- An effective floor `> 0` also bounds the lexical channel (#1425):
  lexical-only chunks are excluded from the fused set because BM25 /
  keyword-PPR scores are not cosines, and KEYWORD mode gates its hits against
  a floored vector search - mirroring the temporal stores' #1404 semantics.
  BM25/keyword evidence can still boost a chunk that clears the floor, but
  cannot rescue one below it.

### Why the floor is a post-filter on pgvector (#1407)

pgvector's HNSW index can only serve `ORDER BY embedding <=> $query` ascending.
Ordering by the wrapped similarity (`1 - distance DESC`) or pushing the floor
into `WHERE` would force a sequential scan, so similarity is computed in the
projection and `min_similarity` is applied as a post-filter on the returned
distance. See [storage-backends.md](../architecture/storage-backends.md) for
the full HNSW ordering discussion.

## Confidence signals on `engine_info`

Recall never withholds results - low confidence is reported passively via
`RecallResult.engine_info` (NOT `.metadata`), emitted by both the default
VectorCypher engine and Chronicle since #1331:

- `engine_info["abstention_signals"]` - four boolean flags (`entities_empty`,
  `chunks_empty`, `chunks_below_min`, `top_score_low`), a `combined_score`
  (0.0 = high confidence, 1.0 = should abstain), and a `should_abstain`
  convenience flag. The default derivation mode is `cosine_floor`: abstain
  when `top_score_low` fires or retrieval is genuinely empty (`chunks_empty
  AND entities_empty`). The legacy `weighted` mode thresholds
  `combined_score`. Inputs use the raw pre-fusion cosine
  (`max_raw_vector_score`), consistent with the score contract above.
- `engine_info["confidence"]` - a calibrated
  `0.8 * clip01(top_cosine / target_cosine) + 0.2 * clip01(gap / target_gap)`
  score.
- `engine_info["degradations"]` - ADR-001 degradation records; empty when
  nothing degraded.

Tunables live under `KhoraConfig.query.abstention_*` - see the
[abstention table in configuration.md](../configuration.md#abstention-signals).

### Other well-known `engine_info` keys

`engine` and `filter` (the `FilterPushdownReport`) are the only keys
guaranteed on every engine. The rest are best-effort telemetry:

| Key | Engines | Meaning |
|---|---|---|
| `channels_used` | vectorcypher | Which retrieval channels contributed. |
| `channels` | chronicle | Per-channel hit counts (`semantic` / `bm25` / `temporal` / `entity`). Note the key name differs from vectorcypher's. |
| `routing` | both | Query-complexity routing decision. |
| `temporal_signal` / `temporal_category` | vectorcypher | Temporal-detection outcome. |
| `rrf_k` | vectorcypher | Fusion smoothing constant used. |
| `timings` | chronicle | Per-stage latency breakdown. |

## Related pages

- [Fusion](fusion.md) - how RRF combines the channels.
- [Retrieval tuning](retrieval-tuning.md) - the practical knobs.
- [VectorCypher engine](../engines/vectorcypher-engine.md) - the default
  engine's full pipeline.
