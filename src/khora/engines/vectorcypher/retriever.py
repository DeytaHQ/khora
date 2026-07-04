"""VectorCypher retriever - hybrid vector+graph retrieval.

Implements the VectorCypher retrieval pipeline:
1. Vector search to find entry entities (pgvector)
2. Cypher traversal to expand relationships (Neo4j)
3. Chunk retrieval via MENTIONED_IN relationships
4. RRF fusion to combine vector and graph scores

Performance optimizations:
- Parallel execution of independent operations (vector chunk search + entity path)
- Batch entity neighborhood fetching via UNWIND
- Normalized score fusion for better ranking
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from loguru import logger
from neo4j.exceptions import (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)

try:
    from neo4j.exceptions import ConnectionAcquisitionTimeoutError, ConnectionPoolError
except ImportError:
    ConnectionAcquisitionTimeoutError = ServiceUnavailable  # type: ignore[misc,assignment]
    ConnectionPoolError = ServiceUnavailable  # type: ignore[misc,assignment]

from khora.core.diagnostics import Degradation
from khora.core.models import Chunk, Entity, Relationship
from khora.filter.model import RecallFilterUnsupportedError
from khora.filter.report import ChannelPlan
from khora.filter.telemetry import record_graph_channel_empty
from khora.query import SearchMode
from khora.query.hyde import HyDEExpander
from khora.telemetry import bounded_text_hash, trace_span
from khora.telemetry.metrics import metric_counter

from .dual_nodes import DualNodeManager
from .fusion import (
    FusedResult,
    apply_coherence_boost,
    apply_recency_boost,
    attach_relevance_scores,
    normalize_scores,
    weighted_rrf,
    weighted_rrf_normalized,
)
from .router import QueryComplexity, QueryComplexityRouter, RouterConfig, RoutingDecision
from .temporal_detection import (
    RETRIEVAL_PARAMS,
    RetrievalParams,
    TemporalCategory,
    TemporalSignal,
    get_retrieval_params,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.core.temporal import ChunkTemporalFilter
    from khora.extraction.embedders import EmbedderProtocol  # type: ignore[unresolved-import]
    from khora.filter import FilterNode
    from khora.query.reranking import CrossEncoderReranker, LLMReranker
    from khora.storage import StorageCoordinator
    from khora.storage.temporal import TemporalVectorStore

# Transient Neo4j errors that trigger graceful degradation rather than
# failing the entire request. Excludes ClientError (query bugs) and
# generic Neo4jError.
_NEO4J_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ServiceUnavailable,
    ConnectionPoolError,
    SessionExpired,
    TransientError,
)

# KHORA_BENCH_MODE forces "relative" recency reference (max(occurred_at) in
# the result set) regardless of the temporal_reference_wall_clock config
# flag — used during benchmark replay where the dataset's newest timestamp
# may be years stale and "wall-clock recency" produces uniformly low scores.
# Read once at import to avoid an env lookup per recency-score computation
# on the hot path. Set ``KHORA_BENCH_MODE=true|1|yes`` to enable.
_BENCH_MODE: bool = os.environ.get("KHORA_BENCH_MODE", "").strip().lower() in {"true", "1", "yes"}


# LLM rerank skip counter (issue #814). Tracks how often the LLM rerank
# step is skipped, broken down by reason. No ``namespace_id`` label —
# cardinality rule (see CLAUDE.md). Module-level so the meter provider
# can de-duplicate by name across retriever instances.
_LLM_RERANKING_SKIPPED_COUNTER = metric_counter(
    "khora.vectorcypher.llm_reranking.skipped_total",
    description=(
        "Number of times the VectorCypher LLM rerank step was skipped, "
        "by reason (not_temporal / no_version_metadata / decisive_winner)."
    ),
)

# Relationship-fetch degradation counter (issue #904, ADR-001). The
# vectorcypher rel-fetch arm silently resets to ``raw_rels = []`` on any
# exception; previously this was asymmetric with the cypher-expand fallback
# (which sets ``graph_fallback`` / ``graph_error``) and callers had no
# machine-readable signal that relationships were missing from the response.
# NO namespace_id label - cardinality rule.
_REL_FETCH_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.rel_fetch.degraded_total",
    unit="1",
    description=(
        "Issue #904. VectorCypher relationship-fetch silent fallbacks. "
        "Incremented when the parallel ``rels_task`` raises and the retrieve "
        "path continues without relationships. The same event is also "
        "appended to RecallResult.engine_info['degradations']. Labels: "
        "reason (fetch_failed). NO namespace_id label - cardinality rule."
    ),
)

# Entity-version-filter degradation counter. The embedded
# sqlite_lance schema lacks the ``version_valid_from/to`` columns that
# ``_version_filter_entities`` reads, so point-in-time entity-version
# narrowing is skipped on that backend. Recall continues with occurred-bounds
# chunk filtering only; the same event is appended to
# ``RecallResult.engine_info['degradations']``. NO namespace_id label.
_VERSION_FILTER_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.version_filter.degraded_total",
    unit="1",
    description=(
        "VectorCypher entity-version filtering fallbacks. Incremented when an "
        "EXPLICIT/occurred-bounds recall runs on the embedded sqlite_lance "
        "backend, which lacks version_valid_from/to columns, so point-in-time "
        "entity-version narrowing is skipped (occurred-bounds chunk filtering "
        "still applies). Labels: reason (embedded_no_version_columns). "
        "NO namespace_id label - cardinality rule."
    ),
)

# Cypher-expand neighborhood-normalization degradation counter. The embedded
# sqlite_lance backend returns ``Entity`` domain objects (mapped to dicts);
# any entry that is neither an ``Entity`` nor a dict is dropped from the
# expansion. Incremented per dropped entry so an unexpectedly empty
# graph-channel recall is observable; the same event is appended to
# ``RecallResult.engine_info['degradations']``. NO namespace_id label.
_CYPHER_EXPAND_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.cypher_expand.degraded_total",
    unit="1",
    description=(
        "VectorCypher graph-expansion normalization fallbacks. Incremented when "
        "a neighborhood entry returned by get_neighborhoods_batch has an "
        "unrecognized shape (neither an Entity domain object nor a dict) and is "
        "dropped from the expansion. The same event is appended to "
        "RecallResult.engine_info['degradations']. Labels: reason "
        "(unrecognized_neighborhood_shape). NO namespace_id label - cardinality rule."
    ),
)

# Entry-entity vector-search degradation counter (issue #1158, ADR-001).
# ``_vector_search_entities`` previously caught any exception, logged a bare
# WARNING, and returned ``[]`` - so entry-entity discovery failing silently
# collapsed the entire graph-expansion channel of GRAPH/HYBRID recall to
# vector-only with no machine-readable signal. Now the catch site records a
# Degradation and bumps this counter. NO namespace_id label - cardinality rule.
_ENTITY_VECTOR_SEARCH_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.entity_vector_search.degraded_total",
    unit="1",
    description=(
        "Issue #1158 (ADR-001). VectorCypher entry-entity vector-search silent "
        "fallbacks. Incremented when search_similar_entities raises and the "
        "channel returns [], collapsing graph expansion to vector-only. The "
        "same event is also appended to RecallResult.engine_info['degradations']. "
        "Labels: reason (channel_exception). NO namespace_id label - cardinality rule."
    ),
)

# BM25 channel degradation counter (issue #1158, ADR-001). ``_bm25_search_chunks``
# previously caught any exception, logged a bare WARNING, and returned ``[]`` -
# so the independent lexical channel silently disappeared from RRF fusion. Now
# the catch site records a Degradation and bumps this counter. NO namespace_id
# label - cardinality rule.
_BM25_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.bm25.degraded_total",
    unit="1",
    description=(
        "Issue #1158 (ADR-001). VectorCypher BM25 channel silent fallbacks. "
        "Incremented when the full-text search raises (reason=channel_exception) "
        "or when a >=2-token keyword query matches 0 rows "
        "(reason=empty_multitoken_channel, #1330), dropping the lexical "
        "contribution from RRF fusion. The same event is also appended to "
        "RecallResult.engine_info['degradations']. Labels: reason "
        "(channel_exception, empty_multitoken_channel). NO namespace_id label - "
        "cardinality rule."
    ),
)

# ADR-001. When the recency channel's ``search_recent_chunks`` raises an
# operational fault (DB / network), the channel returns [] and its
# pool-augmentation contribution drops out of RRF. The catch site records a
# Degradation and bumps this counter so the silently-dropped channel is
# observable. (A RecallFilterUnsupportedError is NOT counted here - it is a
# determinism error that propagates, never a degradable fault.) The same event
# is appended to RecallResult.engine_info['degradations']. NO namespace_id
# label - cardinality rule.
_RECENCY_DEGRADED_COUNTER = metric_counter(
    "khora.vectorcypher.recency_channel.degraded_total",
    unit="1",
    description=(
        "ADR-001. VectorCypher recency channel silent fallback. Incremented when "
        "search_recent_chunks raises an operational fault and the channel returns "
        "[], dropping the recency pool-augmentation contribution from RRF fusion. "
        "The same event is also appended to RecallResult.engine_info['degradations']. "
        "Labels: reason (channel_exception). NO namespace_id label - cardinality rule."
    ),
)


def _decode_chunker_info(value: Any) -> dict[str, Any]:
    """Normalize the value of ``chunker_info`` returned by a record dict.

    Neo4j serializes ``chunker_info`` as a JSON string at write time (see
    ``dual_nodes.create_chunk_nodes_batch``); other graph stores return
    a native dict. A corrupted persisted value (direct DB tampering, a
    half-finished migration) must not crash ``recall()`` - fall back to
    an empty dict on ``json.JSONDecodeError``.
    """
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _coerce_occurred_at(value: Any) -> datetime | None:
    """Coerce a persisted ``occurred_at`` value to a ``datetime``.

    Neo4j returns ``occurred_at`` as an ISO-8601 string (see
    ``dual_nodes.create_chunk_nodes_batch``). The SurrealDB fallback
    forwards the native ``datetime`` from ``Chunk.source_timestamp``.
    Used by Chunk-construction sites that need to populate the
    ``source_timestamp`` field so the recall projection in
    ``engine._build_recall_result`` reads back the persisted value
    instead of ``None`` (fixes #859).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass
class VectorCypherResult:
    """Result from VectorCypher retrieval."""

    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    routing_decision: RoutingDecision
    relationships: list[tuple[Relationship, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    # Graph traversal settings
    default_depth: int = 2
    max_depth: int = 4
    max_entry_entities: int = 10

    # Adaptive depth settings
    adaptive_depth_enabled: bool = True
    adaptive_depth_high_entity_threshold: int = 10  # Shallow depth if >= this many entities
    adaptive_depth_low_entity_threshold: int = 2  # Deeper depth if <= this many entities

    # Fusion settings
    rrf_k: int = 60
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Per-complexity fusion overrides (used when routing is enabled)
    simple_vector_weight: float = 0.8
    simple_graph_weight: float = 0.2
    complex_vector_weight: float = 0.4
    complex_graph_weight: float = 0.6

    # Temporal fusion overrides (used when temporal signal is detected)
    temporal_vector_weight: float = 0.3
    temporal_graph_weight: float = 0.7

    # Temporal settings. Defaults canonicalized to QuerySettings' values in
    # #1406; the engine wires them from ``KhoraConfig.query.recency_weight`` /
    # ``recency_decay_days``. Decay restored to 30 in #1421 - the 7d BEAM
    # tuning is a conversational-recency opt-in, not the default.
    # NOTE: at runtime the per-query recency weight comes from the
    # temporal-category RETRIEVAL_PARAMS table (or a per-call ``recency_bias``
    # override); ``recency_weight`` here is not consulted by retrieve().
    # ``recency_decay_days`` IS live: it is the fallback decay window in
    # ``_calculate_recency_scores`` when no category override applies.
    recency_weight: float = 0.35
    recency_decay_days: float = 30.0
    recency_decay_type: str = "exponential"  # "linear" or "exponential"

    # Issue #567 — temporal recency Phase A. Each flag defaults OFF so the
    # behavior of an existing-default retriever is unchanged. Operators opt
    # in by plumbing the matching ``KhoraConfig.query.temporal_*`` field
    # through when constructing ``RetrieverConfig``.
    #
    # ``temporal_reference_wall_clock``: when True, ``_calculate_recency_scores``
    #   uses ``datetime.now(UTC)`` as the reference instead of the newest
    #   ``occurred_at`` in the result set. Production-correct; the legacy
    #   "newest-in-set" reference is preserved only when this flag is False
    #   or when ``KHORA_BENCH_MODE=true`` overrides for benchmark replay.
    # ``temporal_recency_floor_enabled``: when True, RECENCY/CHANGE queries
    #   that lack an explicit date and contain no anti-recency token receive
    #   a synthesized ``ChunkTemporalFilter(occurred_after=now-default_window_days)``.
    # ``temporal_per_source_decay``: when True, ``_calculate_recency_scores``
    #   looks up a per-chunk decay window via
    #   ``chunk.metadata["source_system"]`` against
    #   ``temporal_default_decay_by_source`` (falling back to ``_default``).
    # ``temporal_default_decay_by_source``: dict[source_system, decay_days].
    #   Must include a ``"_default"`` key used when ``source_system`` is
    #   absent, None, empty, or unknown.
    temporal_reference_wall_clock: bool = False
    temporal_recency_floor_enabled: bool = False
    temporal_per_source_decay: bool = False
    temporal_default_decay_by_source: dict[str, int] = field(
        default_factory=lambda: {
            "slack": 3,
            "email": 7,
            "calendar": 14,
            "salesforce": 180,
            "_default": 14,
        }
    )
    # ``temporal_recency_channel_enabled``: when True, RECENCY/CHANGE queries
    #   fuse a parallel "recency channel" (pure ORDER BY occurred_at DESC,
    #   no embedding) alongside the cosine + BM25 channels. Chunks from this
    #   channel only enter fusion when their cosine similarity to the query
    #   embedding exceeds ``temporal_query_relevance_floor`` — prevents
    #   today's irrelevant chunks from muscling into top-K.
    temporal_recency_channel_enabled: bool = False
    # 0.40 default — was 0.30 in the initial Phase A; raised after LoCoMo
    # --small showed a persistent 4.2pp abstention regression from
    # just-above-floor chunks diluting the abstention signal. Operators
    # can override via KHORA_QUERY_TEMPORAL_QUERY_RELEVANCE_FLOOR.
    temporal_query_relevance_floor: float = 0.40
    temporal_recency_channel_limit: int = 50
    # ``temporal_llm_disambiguation_enabled``: when True, queries that
    #   fire RECENCY/CHANGE in the Aho-Corasick tier AND contain
    #   ambiguity-trigger tokens are routed to a small-model LLM for a
    #   final RECENT/HISTORICAL/COUNTERFACTUAL/NEUTRAL classification.
    #   Floor is vetoed for non-RECENT outputs. Cost bounded by query
    #   distinct-count (results cached per-query). Targets the LoCoMo
    #   counterfactual regression seen in PR #571.
    temporal_llm_disambiguation_enabled: bool = False
    temporal_llm_disambiguation_model: str | None = None

    # Coherence scoring (penalizes word-shuffled confounders)
    coherence_weight: float = 0.1

    # Search thresholds
    min_entity_similarity: float = 0.3
    # Cosine floor for the chunk (vector) channel, applied when the per-call
    # ``min_similarity`` kwarg is left at its 0.0 default (#1406). Wired from
    # ``KhoraConfig.query.min_chunk_similarity`` / KHORA_QUERY_MIN_CHUNK_SIMILARITY.
    # Default 0.0 = no floor (the previous effective behavior); opt in via config.
    min_chunk_similarity: float = 0.0
    hybrid_alpha: float = 0.7

    # Lazy entity expansion
    lazy_entity_expansion: bool = False
    skeleton_core_ratio: float = 0.70  # Skip lazy expansion when > 0.6

    # BM25 channel (independent full-text search alongside vector + graph)
    enable_bm25_channel: bool = False
    bm25_weight: float = 0.3
    bm25_top_k: int = 50  # How many BM25 results to fetch

    # Lexical-channel selector (#1391). "bm25" (default) keeps BM25 in the
    # lexical slot; "keyword_ppr" swaps in the experimental keyword-chunk
    # PageRank channel. The keyword_ppr channel feeds the SAME lexical/bm25
    # fusion slot (bm25_weight) so fusion is unchanged.
    lexical_channel: str = "bm25"
    keyword_ppr_damping: float = 0.85
    keyword_ppr_max_edges: int = 50_000

    # Cross-encoder reranking (default BAAI/bge-reranker-v2-m3; see VectorCypherConfig)
    enable_reranking: bool = False
    reranking_model: str = "BAAI/bge-reranker-v2-m3"
    reranking_top_n: int = 50  # Candidates to feed to the cross-encoder
    reranking_blend_weight: float = 0.7  # Rerank vs original score blend

    # LLM reranking (applied after cross-encoder, only for temporal queries).
    #
    # ``enable_llm_reranking=True`` is also gated by ``llm_reranking_mode``
    # (see below): in the default ``"auto"`` mode the LLM rerank step is
    # skipped when no chunk in the top candidates carries
    # ``metadata["version"]`` — PR #364 showed it regresses MRR on
    # conversational benchmarks. Set ``llm_reranking_mode="always"`` to
    # bypass that precondition.
    enable_llm_reranking: bool = False
    llm_reranking_model: str = "gpt-4o-mini"
    llm_reranking_top_n: int = 5
    llm_reranking_confidence_threshold: float = 0.1  # Skip LLM reranking when cross-encoder gap >= this
    # ``"auto"`` (default) — gate LLM rerank on the version-metadata
    # precondition. When the gate fires for a namespace, a one-time
    # WARNING surfaces the skip so users discover why their opt-in did
    # not invoke the LLM (issue #814).
    # ``"always"`` — bypass the version-metadata precondition; LLM rerank
    # runs on every temporal query (subject to the decisive-winner skip,
    # which is a latency optimization not the bug being addressed).
    llm_reranking_mode: Literal["auto", "always"] = "auto"
    # "Decisive winner" gate (Sprint 1 — multihop latency regression).
    # Skip LLM rerank when the cross-encoder's #1 result is BOTH high-scoring
    # AND well-separated from #2 — the LLM call adds 200–400 ms and rarely
    # changes the ranking when the cross-encoder is already confident.
    # Both conditions must hold to skip:
    #   top_score      >= llm_reranking_min_top_score
    #   top - second   >= llm_reranking_decisive_gap
    llm_reranking_min_top_score: float = 0.7
    llm_reranking_decisive_gap: float = 0.1

    # Session-aware parallel retrieval for cross-session temporal queries.
    # When enabled AND the query is temporal AND entry entities span multiple
    # sessions, fans out parallel per-session vector searches instead of a
    # single global search.  Improves session_crossing_recall.
    enable_session_aware_search: bool = True

    # HyDE query-embedding expansion (#1018). Threaded from
    # ``KhoraConfig.query.enable_hyde`` so configuring it actually affects the
    # default recall() path (previously inert on VectorCypher — only the
    # bypassed ``khora.query.QueryEngine`` honored it). "auto" expands for
    # complex / temporal queries, "always" always, "never" disables.
    enable_hyde: str = "auto"
    hyde_num_hypotheticals: int = 1

    # Stage-1 broad-recall breadth (#1018). When reranking or MMR diversity is
    # active, the vector channel over-fetches this many candidates so there is a
    # genuine pool to rerank / diversify across before the final ``limit``
    # truncation. Mirrors ``QueryEngine`` stage-1. ``None`` keeps the historic
    # per-channel ``limit`` fetch (no over-fetch).
    stage1_recall_limit: int = 200

    # MMR diversity selection (#1018). When enabled, the final top-``limit``
    # chunks are chosen by Maximal Marginal Relevance over the candidate pool
    # instead of pure score order. ``diversity_lambda`` trades relevance
    # (1.0) against diversity (0.0). Threaded from ``KhoraConfig.query`` so the
    # setting is no longer inert on the default recall() path.
    enable_diversity: bool = True
    diversity_lambda: float = 0.5

    # Personalized PageRank retrieval path (Issue #542 — HippoRAG 2).
    # Default OFF — when ON, _vectorcypher_retrieve replaces the BFS+RRF
    # graph expansion with query-time PPR seeded from entry entities and
    # scores chunks via PR-weighted sum over their mentioned entities.
    # When the entity graph is empty or no entry entities are found the
    # path falls back to vector-only (degrades, never crashes).
    enable_ppr_retrieval: bool = False
    ppr_damping: float = 0.85
    ppr_max_iter: int = 50
    ppr_tol: float = 1e-5
    ppr_top_entities: int = 30
    # #1373: when the global PPR slice hits its cap (namespace > ~5000
    # entities), augment it with the query seeds + their 1-hop neighborhood so
    # the resolved seeds survive into the graph. Below the cap these are inert.
    ppr_neighborhood_per_seed_limit: int = 64
    ppr_max_neighborhood_entities: int = 2000

    # Limits
    max_chunks: int = 50
    max_entities: int = 30
    max_relationships: int = 90  # ~3x max_entities

    # Over-fetch multiplier for the graph chunk channel when a residual metadata
    # predicate must be applied as an in-memory post-filter (metadata is not
    # pushable to Cypher). Capped at min(limit*multiplier, 200).
    metadata_overfetch_multiplier: int = 3


def _extract_occurred_at(item: Any) -> str | None:
    """Extract occurred_at string from a Chunk or dict item."""
    if isinstance(item, Chunk):
        return (item.metadata or {}).get("occurred_at")
    elif isinstance(item, dict):
        return item.get("occurred_at")
    return None


def _extract_source_system(item: Any) -> str | None:
    """Extract ``source_system`` string from a Chunk or dict item.

    Returns ``None`` when the value is missing, empty, or the item shape
    doesn't carry chunk metadata — callers fall back to ``"_default"`` in
    that case. ``source_system`` is option<string> in the storage schema,
    so we explicitly treat None and empty strings as "unknown source".
    """
    if isinstance(item, Chunk):
        value = (item.metadata or {}).get("source_system")
    elif isinstance(item, dict):
        value = item.get("source_system")
    else:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _candidates_have_versions(candidates: list[Any]) -> bool:
    """Return ``True`` if any candidate carries ``metadata["version"]``.

    Used by :meth:`VectorCypherRetriever._evaluate_llm_rerank_gate` to
    decide whether the version-metadata precondition (PR #364, issue #814)
    is satisfied. Accepts both candidate shapes that the LLM-rerank gate
    sees in practice:

    - ``FusedResult`` (complex path) — metadata lives at ``r.item.metadata``.
    - ``(Chunk, score)`` (simple path) — metadata lives at ``c.metadata``.
    """
    for cand in candidates:
        # Simple-path shape: (Chunk, score)
        if isinstance(cand, tuple) and len(cand) == 2:
            chunk = cand[0]
            meta = getattr(chunk, "metadata", None)
            if isinstance(meta, dict) and meta.get("version"):
                return True
            continue
        # Complex-path shape: FusedResult (has .item with .metadata)
        item = getattr(cand, "item", None)
        meta = getattr(item, "metadata", None) if item is not None else None
        if isinstance(meta, dict) and meta.get("version"):
            return True
    return False


def _extract_top_two_scores(candidates: list[Any]) -> tuple[float | None, float | None]:
    """Return ``(top_score, second_score)`` from candidates, or ``(None, None)``.

    Mirrors ``_candidates_have_versions`` in handling both candidate
    shapes. When fewer than two candidates are available, returns
    ``(None, None)`` so callers can short-circuit the decisive-winner
    check without raising.
    """
    if len(candidates) < 2:
        return None, None
    first, second = candidates[0], candidates[1]
    if isinstance(first, tuple) and len(first) == 2 and isinstance(second, tuple) and len(second) == 2:
        return float(first[1]), float(second[1])
    top = getattr(first, "rrf_score", None)
    runner = getattr(second, "rrf_score", None)
    if top is None or runner is None:
        return None, None
    return float(top), float(runner)


def _has_target_date(
    temporal_filter: Any | None,
    temporal_signal: Any | None,
) -> bool:
    """Return True if the recall request carries a point-in-time target date.

    Used to gate the embedded sqlite_lance backend which lacks
    bi-temporal version columns. Both inputs are duck-typed because the
    same attribute names (``occurred_after`` / ``occurred_before``) are
    shared between the user-facing temporal filter and the
    ``TemporalSignal``-attached filter produced by EXPLICIT detection.
    """
    for tf in (temporal_filter, getattr(temporal_signal, "temporal_filter", None)):
        if tf is None:
            continue
        if getattr(tf, "occurred_before", None) is not None:
            return True
        if getattr(tf, "occurred_after", None) is not None:
            return True
    return False


class VectorCypherRetriever:
    """Hybrid retriever combining vector search with Cypher graph traversal.

    The retrieval pipeline:
    1. Route query to determine search strategy
    2. Vector search for entry entities via pgvector
    3. (If complex) Expand entities via Neo4j Cypher queries
    4. Fetch chunks connected to entities via MENTIONED_IN
    5. Apply RRF fusion to combine results
    6. Apply temporal recency boost
    """

    def __init__(
        self,
        vector_store: TemporalVectorStore,
        neo4j_driver: AsyncDriver | None,
        embedder: EmbedderProtocol,
        *,
        database: str = "neo4j",
        config: RetrieverConfig | None = None,
        router_config: RouterConfig | None = None,
        storage: StorageCoordinator | None = None,
        neo4j_query_timeout: float | None = None,
        backend: str = "postgres",
    ):
        """Initialize the retriever.

        Args:
            vector_store: pgvector temporal store for chunk search
            neo4j_driver: Neo4j async driver for graph traversal
            embedder: Embedder for query embedding
            database: Neo4j database name
            config: Retriever configuration
            router_config: Router configuration (optional, for LLM routing etc.)
            storage: Storage coordinator for entity vector search via pgvector.
                When ``storage._graph`` is a ``Neo4jBackend``, its ``_session``
                helper is forwarded to ``DualNodeManager`` so pool metrics
                observe every Neo4j session opened through this retriever.
                ``_graph`` (not the public ``graph`` proxy) is used because
                the ``NamespaceRequiredProxy`` refuses dunder/private attribute
                lookups — see ``khora.storage._namespace_proxy``.
            neo4j_query_timeout: Optional per-transaction timeout in seconds
                forwarded to the underlying ``DualNodeManager`` to bound
                ``get_entity_neighborhoods``. ``None`` disables the timeout.
            backend: Storage backend identifier (e.g., ``"postgres"``,
                ``"surrealdb"``, ``"sqlite_lance"``). Used to gate features
                that aren't implemented on the embedded backend.
        """
        self._vector_store = vector_store
        self._neo4j_driver = neo4j_driver
        self._embedder = embedder
        self._database = database
        self._config = config or RetrieverConfig()
        self._storage = storage
        self._backend = backend
        self._bm25_empty_warned_ns: set[str] = set()

        # Initialize router with config, syncing adaptive depth settings
        if router_config is None:
            router_config = RouterConfig(
                adaptive_depth_enabled=self._config.adaptive_depth_enabled,
                adaptive_depth_high_entity_threshold=self._config.adaptive_depth_high_entity_threshold,
                adaptive_depth_low_entity_threshold=self._config.adaptive_depth_low_entity_threshold,
                complex_depth=self._config.default_depth,
            )
        self._router = QueryComplexityRouter(router_config)
        # Forward Neo4jBackend when available so pool metrics observe
        # all traversals driven from the retriever (see DualNodeManager).
        # Bypass the NamespaceRequiredProxy on ``storage.graph`` by reading
        # ``_graph`` directly — the proxy refuses underscore-prefixed lookups
        # (DualNodeManager._session would hit AttributeError('_session')).
        pool_backend = getattr(storage, "_graph", None) if storage else None
        self._dual_nodes = (
            DualNodeManager(
                neo4j_driver,
                database,
                query_timeout=neo4j_query_timeout,
                pool_backend=pool_backend,
            )
            if neo4j_driver
            else None
        )

        # Lazy entity expansion cache: chunk_id -> expansion_score (0 = no match)
        self._expansion_cache: dict[UUID, float] = {}

        # Cached cross-encoder reranker (lazy-init on first use, reused across queries).
        # An ``asyncio.Lock`` makes the check-then-set explicit even though today's
        # ``CrossEncoderReranker()`` constructor is synchronous: it survives a future
        # refactor that adds an ``await`` between the check and the assignment, and
        # documents at the call site that this state is shared across concurrent
        # recall tasks (see issue #985).
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_lock = asyncio.Lock()

        # Cached LLM reranker for temporal queries (lazy-init on first use, same guard).
        self._llm_reranker: LLMReranker | None = None
        self._llm_reranker_lock = asyncio.Lock()

        # Issue #814 — one-time WARNING dedupe for LLM-rerank skip reasons.
        # Tuples are ``(namespace_id_or_None, skip_reason)``; once a tuple
        # has been logged we keep silent on subsequent queries for that
        # namespace so a hot recall path doesn't spam the operator log.
        self._warned_rerank_skips: set[tuple[UUID | None, str]] = set()

        # Cached HyDE expander (lazy-init on first use, reused across recalls;
        # #1018). ``None`` until the first query for which HyDE fires.
        self._hyde_expander: HyDEExpander | None = None

    def _should_hyde(
        self,
        routing: RoutingDecision,
        temporal_signal: TemporalSignal | None,
    ) -> bool:
        """Decide whether HyDE should fire for this query (#1018).

        Mirrors ``QueryEngine`` semantics: ``"always"`` always, ``"never"``
        never, ``"auto"`` for complex (non-SIMPLE routing) or temporal queries.
        """
        mode = self._config.enable_hyde
        if mode == "always":
            return True
        if mode == "never":
            return False
        # "auto": expand for complex or temporal queries.
        is_temporal = temporal_signal is not None and temporal_signal.is_temporal
        is_complex = routing.complexity != QueryComplexity.SIMPLE
        return is_temporal or is_complex

    async def _maybe_expand_hyde(
        self,
        query: str,
        query_embedding: list[float],
        *,
        routing: RoutingDecision,
        temporal_signal: TemporalSignal | None,
        out_degradations: list[Degradation] | None = None,
    ) -> list[float]:
        """Apply HyDE query-embedding expansion when configured (#1018).

        Returns the expanded embedding when HyDE fires, otherwise the original.
        The expander degrades to the original embedding on any failure, so this
        never aborts a recall. When ``out_degradations`` is supplied and the
        expander falls back, a :class:`Degradation` is appended (ADR-001,
        issue #1324).
        """
        if not self._should_hyde(routing, temporal_signal):
            return query_embedding
        if self._hyde_expander is None:
            self._hyde_expander = HyDEExpander(
                self._embedder,
                num_hypotheticals=self._config.hyde_num_hypotheticals,
            )
        with trace_span("khora.vectorcypher.hyde"):
            return await self._hyde_expander.expand_query_embedding(
                query, query_embedding, out_diagnostics=out_degradations
            )

    def _vector_fetch_limit(self, limit: int) -> int:
        """Broad-recall vector fetch budget for this call (#1018).

        Over-fetches to ``stage1_recall_limit`` when a narrowing stage
        (reranking or MMR diversity) is active so there is a genuine pool to
        rerank / diversify across; otherwise keeps the historic per-channel
        ``limit`` fetch (no extra work).
        """
        if self._config.enable_reranking or self._config.enable_diversity:
            return max(limit, self._config.stage1_recall_limit)
        return limit

    def _mmr_select_fused(
        self,
        fused_results: list[FusedResult],
        query_embedding: list[float],
        *,
        k: int,
        lambda_param: float,
    ) -> list[FusedResult]:
        """Reorder fused results so the MMR-selected top-``k`` lead (#1018).

        Maximal Marginal Relevance balances relevance to the query against
        diversity among selected chunks. Falls back to score order when chunk
        embeddings are unavailable (the embedded path may not hydrate them).
        Mirrors ``QueryEngine._mmr_diversity_select``.
        """
        from khora._accel import batch_dot_product as _bdp
        from khora._accel import mmr_diversity_select, normalize_embeddings_batch

        if len(fused_results) <= k:
            return fused_results

        chunk_embeddings = [getattr(r.item, "embedding", None) for r in fused_results]
        if all(e is None for e in chunk_embeddings):
            return fused_results

        filled = [emb if emb is not None else query_embedding for emb in chunk_embeddings]
        normalized = normalize_embeddings_batch([query_embedding, *filled])
        norm_query, norm_embeddings = normalized[0], normalized[1:]

        scores = [0.0] * len(norm_embeddings)
        for idx, sim in _bdp(norm_query, norm_embeddings, threshold=0.0):
            scores[idx] = sim

        selected = mmr_diversity_select(norm_embeddings, scores, lambda_param, k)
        selected_set = set(selected)
        # Selected (MMR order) first, then the remaining results in their
        # existing score order so nothing is dropped.
        return [fused_results[i] for i in selected] + [r for i, r in enumerate(fused_results) if i not in selected_set]

    def _effective_min_similarity(self, min_similarity: float) -> float:
        """Resolve the chunk-channel cosine floor (#1406).

        A positive per-call ``min_similarity`` wins (caller intent). ``0.0``
        (the kwarg default) means "unset" and falls back to the configured
        ``min_chunk_similarity`` - the documented
        ``KHORA_QUERY_MIN_CHUNK_SIMILARITY`` knob, which was previously dead
        on this path. Set it to ``0`` to disable the floor entirely.
        """
        if min_similarity > 0.0:
            return min_similarity
        return self._config.min_chunk_similarity

    async def retrieve(
        self,
        query: str,
        namespace_id: UUID,
        *,
        temporal_filter: ChunkTemporalFilter | None = None,
        temporal_signal: TemporalSignal | None = None,
        graph_depth: int | None = None,
        limit: int | None = None,
        min_similarity: float = 0.0,
        mode: SearchMode = SearchMode.HYBRID,
        hybrid_alpha_override: float | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> VectorCypherResult:
        """Retrieve relevant chunks using VectorCypher hybrid approach.

        Args:
            query: User query
            namespace_id: Namespace to search
            temporal_filter: Optional temporal constraints
            temporal_signal: Optional temporal detection signal (drives recency/sort behavior)
            graph_depth: Override for graph traversal depth
            limit: Maximum chunks to return
            min_similarity: Minimum cosine-similarity floor applied to the
                vector channel. Chunks below this threshold are filtered at
                the storage layer before any fusion / reranking happens.
                ``0.0`` (the default) means "unset" and falls back to the
                configured ``min_chunk_similarity`` (#1406).
            hybrid_alpha_override: Per-call vector/BM25 blend factor. Threaded
                explicitly from the engine so the shared ``RetrieverConfig``
                is never mutated (#1116). ``None`` keeps the configured /
                channel-derived alpha; a concrete value overrides it for this
                call only. The BM25 channel still forces pure vector
                (``1.0``) when active to avoid double-counting.
            recency_bias: Per-call recency-boost weight override (0.0-1.0).
                Threaded explicitly from the engine (#1156). ``None`` keeps
                the temporal-signal-derived ``recency_weight``; a concrete
                value overrides the recency-boost weight for this call.
            mode: Search mode contract (#833). ``HYBRID`` and ``ALL`` keep the
                existing multi-channel behaviour. ``VECTOR`` skips the graph
                and BM25 channels (pure vector); ``GRAPH`` skips vector +
                BM25 (chunks come solely from cypher expansion); ``KEYWORD``
                skips vector + graph (BM25 only). The engine is responsible
                for validating this against ``supported_modes`` before
                calling retrieve.
            filter_ast: Canonical recall-filter AST. When set, it is pushed
                down into the vector channel (``_vector_search_chunks`` ->
                ``TemporalVectorStore.search``) and the independent BM25
                channel (``_bm25_search_chunks`` -> ``search_fulltext``),
                where the pgvector backend compiles it to the SAME
                ``khora_chunks`` WHERE predicate. ``None`` leaves every path
                byte-identical to the no-filter behaviour.

        Returns:
            VectorCypherResult with chunks, entities, and metadata
        """
        with trace_span("khora.vectorcypher.retrieve", namespace_id=str(namespace_id)) as span:
            limit = limit or self._config.max_chunks
            min_similarity = self._effective_min_similarity(min_similarity)

            # Resolve retrieval parameters from temporal signal
            params = (
                get_retrieval_params(temporal_signal) if temporal_signal else RETRIEVAL_PARAMS[TemporalCategory.NONE]
            )
            if temporal_signal and temporal_signal.is_temporal:
                span.set_attribute("temporal_category", temporal_signal.category.value)
                logger.debug(
                    f"Temporal signal: {temporal_signal.category.value} (confidence={temporal_signal.confidence:.2f})"
                )

            # Issue #567 A2: synthesize a RECENCY/CHANGE date floor when the
            # category fires but the upstream resolver (dateparser) couldn't
            # turn a bare adjective like "latest" / "recent" into a SQL
            # filter. The veto check happens BEFORE synthesis so historical
            # queries ("all-time", "ever", "history of …") stay unbounded.
            #
            # Synthesis runs at this call site — never inside
            # ``TemporalResolver.resolve_fast`` — so callers on the EXPLICIT
            # path (where ``temporal_filter`` is already populated) are
            # unaffected, and the synthesis decision can use both the
            # ``RetrieverConfig`` feature flag and the original query text.
            synthetic_applied = False
            anti_recency_veto = False
            llm_veto_intent: str | None = None
            with trace_span("khora.vectorcypher.recency_floor_synthesis") as synth_span:
                if (
                    self._config.temporal_recency_floor_enabled
                    and temporal_signal is not None
                    and temporal_signal.is_temporal
                    and temporal_filter is None
                    and params.default_window_days is not None
                ):
                    from khora.query.temporal_detection import (
                        TemporalIntent,
                        classify_temporal_intent_llm,
                        has_ambiguity_trigger,
                        has_anti_recency_token,
                    )

                    veto = False
                    if has_anti_recency_token(query):
                        anti_recency_veto = True
                        veto = True
                    elif self._config.temporal_llm_disambiguation_enabled and has_ambiguity_trigger(query):
                        # Tier-3: ambiguity-trigger token present and LLM
                        # disambiguation is enabled. Defer the final call.
                        # The LLM is cached per-query so repeat queries
                        # cost zero.
                        intent, confidence = await classify_temporal_intent_llm(
                            query,
                            model=self._config.temporal_llm_disambiguation_model,
                        )
                        if confidence > 0 and intent != TemporalIntent.RECENT:
                            # HISTORICAL / COUNTERFACTUAL / NEUTRAL → veto.
                            llm_veto_intent = intent.value
                            veto = True

                    if not veto:
                        from khora.core.temporal import ChunkTemporalFilter as _SynthTF

                        temporal_filter = _SynthTF(
                            occurred_after=datetime.now(UTC) - timedelta(days=params.default_window_days),
                        )
                        synthetic_applied = True
                        logger.debug(
                            "Synthesized RECENCY floor: occurred_after={} (window={}d, category={})",
                            temporal_filter.occurred_after.isoformat(),
                            params.default_window_days,
                            temporal_signal.category.value,
                        )
                synth_span.set_attribute("synthetic_temporal_filter_applied", synthetic_applied)
                synth_span.set_attribute("anti_recency_veto", anti_recency_veto)
                if llm_veto_intent is not None:
                    synth_span.set_attribute("llm_veto_intent", llm_veto_intent)
                if params.default_window_days is not None:
                    synth_span.set_attribute("temporal_floor_days", params.default_window_days)
                if synthetic_applied or anti_recency_veto:
                    # bounded_text_hash keeps the query content out of the
                    # span as raw text (privacy + cardinality), but lets us
                    # correlate synthesis decisions with the request when
                    # debugging.
                    synth_span.set_attribute("query_hash", bounded_text_hash(query))

            # Mirror the synthesis decision onto the parent span too, so
            # operator dashboards filtering at the top-level retrieve span
            # don't need to drill into the child.
            span.set_attribute("synthetic_temporal_filter_applied", synthetic_applied)
            span.set_attribute("anti_recency_veto", anti_recency_veto)
            # Operator-visible metric — bounded labels only (category enum,
            # vetoed bool). See docs/telemetry-contract.json §metrics
            # `khora.query.temporal.floor_applied_total`.
            if (
                self._config.temporal_recency_floor_enabled
                and temporal_signal is not None
                and temporal_signal.is_temporal
                and params.default_window_days is not None
            ):
                from khora.telemetry.temporal_metrics import record_floor_applied

                record_floor_applied(
                    category=temporal_signal.category.value,
                    vetoed=anti_recency_veto,
                )

            # Step 1: Route query to determine strategy
            with trace_span("khora.vectorcypher.route") as route_span:
                routing = await self._router.route(query, temporal_signal=temporal_signal)
                route_span.set_attribute("complexity", routing.complexity.value)
                route_span.set_attribute("use_graph", routing.use_graph)
            logger.debug(f"Query routing: {routing.complexity.value} (use_graph={routing.use_graph})")
            span.set_attribute("routing_complexity", routing.complexity.value)

            # Step 2: Embed the query
            with trace_span("khora.vectorcypher.embed_query") as embed_span:
                embed_span.set_attribute("model", self._embedder.model_name)
                embed_span.set_attribute("dimension", self._embedder.dimension)
                embed_span.set_attribute("text_length", len(query))
                _stats = getattr(self._embedder, "cache_stats", None)
                _pre_hits = _stats["hits"] if isinstance(_stats, dict) else None
                query_embedding = await self._embedder.embed(query)
                if _pre_hits is not None:
                    _post_hits = self._embedder.cache_stats["hits"]
                    embed_span.set_attribute("cache_hit", _post_hits > _pre_hits)

            # Step 2b: HyDE query-embedding expansion (#1018). Honors
            # ``query.enable_hyde`` on the default recall() path (previously only
            # the bypassed QueryEngine applied it). "auto" expands for complex /
            # temporal queries; the expander degrades to the original embedding
            # on any LLM/embed failure.
            # ADR-001 (#1324): collect HyDE degradations here and merge them
            # into the sub-path result's ``metadata["degradations"]`` below.
            _hyde_degradations: list[Degradation] = []
            query_embedding = await self._maybe_expand_hyde(
                query,
                query_embedding,
                routing=routing,
                temporal_signal=temporal_signal,
                out_degradations=_hyde_degradations,
            )

            # #833 mode dispatch: VECTOR / KEYWORD short-circuit to the
            # simple path (no graph traversal). GRAPH forces the vectorcypher
            # path (entity expansion). HYBRID / ALL keep the legacy routing.
            # The sub-paths receive ``mode`` and honor the channel-skip
            # contract internally.
            force_simple = mode in (SearchMode.VECTOR, SearchMode.KEYWORD)
            force_graph = mode == SearchMode.GRAPH

            # Step 3: Vector search for entry points
            if (
                not force_simple
                and not force_graph
                and routing.complexity == QueryComplexity.TYPED_ENTITY_RECENT
                and (filter_ast is None or not filter_ast.children)
            ):
                # Gate (no CONSTRAINING filter): the fast-path Cypher cannot
                # enforce caller filters — chunk metadata is a serialized
                # JSON property on the graph node, not queryable columns — so
                # a filtered recall would silently return unfiltered chunks.
                # A constraint-free filter (``filter={}`` / ``RecallFilter()``)
                # parses to a non-null match-everything ``AND`` with no
                # children: there is nothing to enforce, so it keeps the fast
                # path (#1232). The constraint-free test mirrors
                # ``build_filter_report`` (``filter_ast is None or not
                # filter_ast.children``). Genuinely-constraining filtered
                # recalls take the full _vectorcypher_retrieve path below,
                # which enforces + reports the filter per channel.
                #
                # Phase C fast path (#569): a single Cypher query that
                # finds typed entities (ACTION_ITEM, DECISION, BLOCKER,
                # RISK) ordered by last MENTIONED_IN chunk timestamp.
                # Falls back to _vectorcypher_retrieve on empty rows.
                result = await self._typed_entity_recent_retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    graph_depth=graph_depth,
                    limit=limit,
                    routing=routing,
                    hybrid_alpha_override=hybrid_alpha_override,
                    recency_bias=recency_bias,
                    filter_ast=filter_ast,
                )
            elif force_simple or (not force_graph and routing.complexity == QueryComplexity.SIMPLE):
                # Simple path: direct chunk retrieval. Also the destination
                # for mode=VECTOR / mode=KEYWORD short-circuits.
                result = await self._simple_retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    limit=limit,
                    routing=routing,
                    effective_recency=(recency_bias if recency_bias is not None else params.recency_weight),
                    decay_days_override=params.decay_days_override,
                    temporal_sort=params.temporal_sort,
                    recency_floor=params.recency_floor,
                    temporal_signal=temporal_signal,
                    min_similarity=min_similarity,
                    mode=mode,
                    hybrid_alpha_override=hybrid_alpha_override,
                    filter_ast=filter_ast,
                )
            else:
                # Complex/moderate path: VectorCypher with parallel execution.
                # Also the destination for mode=GRAPH (entity expansion,
                # vector + BM25 channels skipped).
                # Wrap in try/except for graceful fallback on graph failures
                try:
                    result = await self._vectorcypher_retrieve(
                        query=query,
                        query_embedding=query_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=temporal_filter,
                        graph_depth=graph_depth,
                        limit=limit,
                        routing=routing,
                        temporal_params=params,
                        temporal_signal=temporal_signal,
                        min_similarity=min_similarity,
                        mode=mode,
                        hybrid_alpha_override=hybrid_alpha_override,
                        recency_bias=recency_bias,
                        filter_ast=filter_ast,
                    )
                except _NEO4J_TRANSIENT_ERRORS as e:
                    logger.warning(f"Graph search failed, falling back to vector-only: {e}")
                    result = await self._vector_only_fallback(
                        query=query,
                        query_embedding=query_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=temporal_filter,
                        limit=limit,
                        routing=routing,
                        effective_recency=(recency_bias if recency_bias is not None else params.recency_weight),
                        decay_days_override=params.decay_days_override,
                        temporal_sort=params.temporal_sort,
                        recency_floor=params.recency_floor,
                        temporal_signal=temporal_signal,
                        min_similarity=min_similarity,
                        mode=mode,
                        hybrid_alpha_override=hybrid_alpha_override,
                        filter_ast=filter_ast,
                    )

            span.set_attribute("chunk_count", len(result.chunks))
            span.set_attribute("entity_count", len(result.entities))

            # ADR-001 (#1324): fold any HyDE degradations collected before the
            # sub-path dispatch into the result's ``metadata["degradations"]``
            # list so they surface on ``RecallResult.engine_info["degradations"]``
            # via the engine's ``**result.metadata`` spread.
            if _hyde_degradations:
                result.metadata.setdefault("degradations", []).extend(_hyde_degradations)

            # Record top-1 chunk age — feeds the
            # ``khora.recall.recency.query_to_top1_age_days`` histogram so
            # operators can spot temporal-quality drift in production
            # (the dashboard slices it by temporal_category via the span
            # parent context). Only fires when the top chunk carries a
            # parseable occurred_at; skipped on empty result sets.
            if result.chunks:
                try:
                    # result.chunks may be list[Chunk] or list[tuple[Chunk, score]]
                    # depending on the path; unwrap defensively.
                    raw = result.chunks[0]
                    top_chunk = raw[0] if isinstance(raw, tuple) else raw
                    occurred_at_iso = None
                    meta = getattr(top_chunk, "metadata", None)
                    if isinstance(meta, dict):
                        occurred_at_iso = meta.get("occurred_at")
                    if occurred_at_iso:
                        occurred_at = datetime.fromisoformat(occurred_at_iso.replace("Z", "+00:00"))
                        if occurred_at.tzinfo is None:
                            occurred_at = occurred_at.replace(tzinfo=UTC)
                        age_days = (datetime.now(UTC) - occurred_at).total_seconds() / 86400.0
                        from khora.telemetry.temporal_metrics import record_top1_age_days

                        record_top1_age_days(max(age_days, 0.0))
                except (ValueError, AttributeError, TypeError):
                    # Metric is best-effort — never fail the recall path
                    # because top-1 age extraction misfired.
                    pass

            return result

    async def _typed_entity_recent_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        graph_depth: int | None,
        limit: int,
        routing: RoutingDecision,
        *,
        hybrid_alpha_override: float | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> VectorCypherResult:
        """Phase C fast path (#569): typed-entity recency in one Cypher query.

        For queries matching ``(latest|most recent|newest|recent)
        (action items|decisions|blockers|risks|...)`` the router dispatches
        here. We run a single Cypher query that:

        1. Matches Entity nodes of the resolved type (ACTION_ITEM, etc.).
        2. Joins MENTIONED_IN chunks; computes max(c.occurred_at) per entity.
        3. Returns one evidence chunk per entity (the most recent mention).
        4. Orders by last_mention DESC.

        The dispatch in ``retrieve()`` gates this path on ``filter_ast is
        None`` — the fast-path Cypher does not enforce caller filters, so
        filtered recalls take the full ``_vectorcypher_retrieve`` path.

        For entity types that carry a ``status`` attribute (ACTION_ITEM,
        COMMITMENT, OPEN_QUESTION), filters out entries whose status is in
        the "closed" set — we don't want to surface "done" action items as
        "the latest action items" by default.

        Falls back to ``_vectorcypher_retrieve`` when:
        - ``self._dual_nodes`` is None (no Neo4j configured).
        - Cypher returns zero rows (typed entities haven't been extracted
          on this namespace; the namespace may not have opted into the
          ``builtin:meetings`` skill yet).
        """
        from khora.query.router import TYPED_ENTITY_NOUN_MAP

        # Resolve the entity_type from the query phrase.
        entity_type: str | None = None
        lowered = query.lower()
        for noun, type_name in TYPED_ENTITY_NOUN_MAP.items():
            if noun in lowered:
                entity_type = type_name
                break
        if entity_type is None:
            # Pattern matched but our local map missed — fall back.
            return await self._vectorcypher_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                graph_depth=graph_depth,
                limit=limit,
                routing=routing,
                hybrid_alpha_override=hybrid_alpha_override,
                recency_bias=recency_bias,
                filter_ast=filter_ast,
            )

        # No graph layer → fall back.
        if self._dual_nodes is None:
            fallback = await self._vectorcypher_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                graph_depth=graph_depth,
                limit=limit,
                routing=routing,
                hybrid_alpha_override=hybrid_alpha_override,
                recency_bias=recency_bias,
                filter_ast=filter_ast,
            )
            fallback.metadata["typed_entity_fast_path_fallback"] = True
            fallback.metadata["typed_entity_type"] = entity_type
            return fallback

        # Status filter — only for entity types that carry a status.
        status_filter = ""
        types_with_status = {"ACTION_ITEM", "COMMITMENT", "OPEN_QUESTION"}
        if entity_type in types_with_status:
            status_filter = "AND (a.status IS NULL OR NOT a.status IN ['done', 'cancelled', 'completed', 'closed']) "

        cypher = f"""
        MATCH (a:Entity {{entity_type: $entity_type, namespace_id: $namespace_id}})-[:MENTIONED_IN]->(c:Chunk)
        WHERE 1=1 {status_filter}
        WITH a, max(c.occurred_at) AS last_mention, collect(c) AS chunks
        ORDER BY last_mention DESC LIMIT $limit
        RETURN a AS entity,
               last_mention,
               [c IN chunks WHERE c.occurred_at = last_mention][0] AS evidence_chunk
        """

        async def _work(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(
                cypher,
                entity_type=entity_type,
                namespace_id=str(namespace_id),
                limit=limit,
            )
            return [record async for record in result]

        with trace_span(
            "khora.vectorcypher.typed_entity_recent",
            entity_type=entity_type,
            namespace_id=str(namespace_id),
        ) as fp_span:
            try:
                async with self._dual_nodes._session() as session:
                    rows = await session.execute_read(_work)
            except Exception as exc:
                logger.debug("Typed-entity fast path failed: {}; falling back", exc)
                fallback = await self._vectorcypher_retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    graph_depth=graph_depth,
                    limit=limit,
                    routing=routing,
                    hybrid_alpha_override=hybrid_alpha_override,
                    recency_bias=recency_bias,
                    filter_ast=filter_ast,
                )
                fallback.metadata["typed_entity_fast_path_fallback"] = True
                fallback.metadata["typed_entity_type"] = entity_type
                return fallback

            if not rows:
                fp_span.set_attribute("row_count", 0)
                fallback = await self._vectorcypher_retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    graph_depth=graph_depth,
                    limit=limit,
                    routing=routing,
                    hybrid_alpha_override=hybrid_alpha_override,
                    recency_bias=recency_bias,
                    filter_ast=filter_ast,
                )
                fallback.metadata["typed_entity_fast_path_fallback"] = True
                fallback.metadata["typed_entity_type"] = entity_type
                return fallback

            fp_span.set_attribute("row_count", len(rows))

            # Build entities + evidence chunks from rows. Score decays by
            # rank position so downstream consumers can sort.
            entities: list[tuple[Entity, float]] = []
            chunks: list[tuple[Chunk, float]] = []
            for idx, row in enumerate(rows):
                e_data: dict[str, Any] | None = row["entity"] if "entity" in row else None
                c_data: dict[str, Any] | None = row["evidence_chunk"] if "evidence_chunk" in row else None
                if e_data is None:
                    continue
                # Score: linear decay from 1.0 by rank; first row = 1.0.
                rank_score = max(0.0, 1.0 - (idx / max(len(rows), 1)))
                entity = Entity(
                    id=UUID(e_data["id"]),
                    namespace_id=namespace_id,
                    name=e_data["name"],
                    entity_type=e_data["entity_type"],
                    description=e_data.get("description", ""),
                )
                entities.append((entity, rank_score))
                if c_data is not None:
                    chunk = Chunk(
                        id=UUID(c_data["id"]),
                        namespace_id=namespace_id,
                        document_id=UUID(c_data["document_id"]),
                        content=c_data.get("content", ""),
                        metadata={"occurred_at": c_data.get("occurred_at")},
                        chunker_info=_decode_chunker_info(c_data.get("chunker_info")),
                        # The graph store carries only the chunk event-time;
                        # surface it as occurred_at so the recall projection
                        # uses it. Neo4j stores it as an ISO-8601 string;
                        # coerce to datetime. source_timestamp mirrors it as
                        # the projection fallback carrier.
                        occurred_at=_coerce_occurred_at(c_data.get("occurred_at")),
                        source_timestamp=_coerce_occurred_at(c_data.get("occurred_at")),
                    )
                    chunks.append((chunk, rank_score))

            return VectorCypherResult(
                chunks=chunks,
                entities=entities,
                routing_decision=routing,
                metadata={
                    "typed_entity_fast_path": True,
                    "typed_entity_type": entity_type,
                    # This path is now reachable only when no caller filter is
                    # present (the dispatch gates it on filter_ast is None), so
                    # the empty plans dict is the no-filter carrier — there is
                    # no filter to enforce or report here.
                    "_filter_channel_plans": {},
                },
            )

    async def _vectorcypher_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        graph_depth: int | None,
        limit: int,
        routing: RoutingDecision,
        *,
        temporal_params: RetrievalParams | None = None,
        temporal_signal: TemporalSignal | None = None,
        min_similarity: float = 0.0,
        mode: SearchMode = SearchMode.HYBRID,
        hybrid_alpha_override: float | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> VectorCypherResult:
        """Internal VectorCypher retrieval with graph traversal.

        This is the main VectorCypher path that combines vector and graph search.
        Separated from retrieve() to enable clean fallback handling.

        Implements adaptive depth: adjusts graph traversal depth based on the
        number of entry entities found. More entities = shallower depth (to avoid
        explosion), fewer entities = deeper depth (to find more context).

        Bi-temporal versioning:
        - EXPLICIT temporal queries with a date filter narrow entities to those
          valid at the target date via ``version_valid_from``/``version_valid_to``.
        - CHANGE temporal queries traverse ``[:SUPERSEDES]`` edges to surface
          entity version history for comparison.
        """
        _tp = temporal_params or RETRIEVAL_PARAMS[TemporalCategory.NONE]
        base_depth = graph_depth or routing.graph_depth
        entry_limit = routing.suggested_entry_limit

        # ADR-001 (issue #904): silent fallback paths append ``Degradation``
        # entries here; the list is surfaced on the response under
        # ``engine_info["degradations"]`` so callers can detect partial
        # failures without changing signatures.
        degradations: list[Degradation] = []

        # Per-call honest filter-pushdown plans, one per retrieval channel that
        # actually enforces the caller filter this recall. Built from each
        # channel's ACTUAL compile (never a backend-name check) and stashed on
        # the result for the engine to fold into ``engine_info["filter"]``. A
        # fresh per-call dict + a fresh per-call vector sink keep the report
        # race-free under concurrent recalls on the shared retriever (no mutable
        # instance state). All four channels GATE: each independently produces
        # candidates entering RRF fusion, so a leaf only one gating channel
        # post-filters lands in the top-level post_filtered_keys even when the
        # SQL channels pushed it (the report builder's per-leaf partition does
        # this automatically). Channels that did not run this recall are simply
        # absent from the dict.
        filter_channel_plans: dict[str, ChannelPlan] = {}
        vector_filter_plan_sink: list[ChannelPlan] = []
        bm25_filter_plan_sink: list[ChannelPlan] = []

        # When the caller filter has a leaf the Cypher chunk channel cannot push
        # down (any metadata predicate — metadata is a serialized JSON property
        # on the Chunk node, not pushable to Cypher), the graph channel applies
        # an in-memory post-filter after the fetch. Over-fetch the graph channel
        # so the post-filter still has enough candidates to fuse. The probe runs
        # the Cypher compiler in split mode (never raises) and compares its
        # consumed keys to every leaf key; system-key-only and no-filter recalls
        # leave the fetch limit unchanged (those push down exactly).
        graph_overfetch = False
        # Probe compile used ONLY to size ``graph_overfetch`` (does a residual
        # metadata predicate remain after the system-key slice pushes down?). It
        # no longer feeds the graph channel's pushdown plan — that derives from
        # the compile actually spliced into the executed ``WHERE``, threaded out
        # via ``graph_pushed_keys_sink`` below. Defined only under a filter.
        compiled_cypher = None
        if filter_ast is not None:
            from khora.filter.compilers.cypher import compile_cypher
            from khora.filter.execute import build_compile_context, has_residual_metadata

            compiled_cypher = compile_cypher(
                filter_ast,
                build_compile_context("Chunk", table_alias="c", on_unsupported="split"),
            )
            graph_overfetch = has_residual_metadata(filter_ast, compiled_cypher.consumed_keys)
        # Graph + PPR fetch budget: widen to make room for the post-filter when a
        # residual metadata predicate is present; otherwise the historical
        # ``limit * 2`` fetch is preserved. The absolute ``200`` ceiling caps the
        # extra graph work so a large multiplier paired with a large limit cannot
        # blow up the Neo4j fetch (the multiplier is bounded ge=2 le=10).
        graph_fetch_limit = (
            min(limit * self._config.metadata_overfetch_multiplier, 200) if graph_overfetch else limit * 2
        )

        # #833 channel-skip: GRAPH mode drops vector + BM25 chunk channels
        # (entry entities still come from the entity-level vector index,
        # since the cypher expansion needs seeds).
        skip_vector_channel = mode == SearchMode.GRAPH
        skip_bm25_channel = mode == SearchMode.GRAPH

        # #1018: when reranking or MMR diversity will narrow the result set,
        # over-fetch the vector channel to ``stage1_recall_limit`` so there is a
        # genuine broad-recall pool to rerank / diversify across before the final
        # ``limit`` truncation. Mirrors QueryEngine's stage-1 breadth.
        vector_fetch_limit = self._vector_fetch_limit(limit)

        # OPTIMIZATION: Start vector chunk search immediately in parallel
        # This operation doesn't depend on entity search results.
        # When the BM25 channel is active, use pure vector (hybrid_alpha=1.0)
        # to avoid double-counting BM25 (once in the vector blend, once as
        # its own independent channel). Otherwise a per-call
        # ``hybrid_alpha_override`` (threaded explicitly from the engine, #1116)
        # wins over the configured default; ``None`` keeps the config value.
        if self._lexical_channel_active():
            effective_hybrid_alpha = 1.0
        else:
            effective_hybrid_alpha = hybrid_alpha_override
        vector_chunks_task: asyncio.Task[list[tuple[UUID, float, Chunk]]] | None = None
        if not skip_vector_channel:
            vector_chunks_task = asyncio.create_task(
                self._vector_search_chunks(
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    query_text=query,
                    limit=vector_fetch_limit,
                    hybrid_alpha_override=effective_hybrid_alpha,
                    min_similarity=min_similarity,
                    filter_ast=filter_ast,
                    # Capture the vector channel's pushdown plan from the SAME
                    # ``khora_chunks`` compile this search runs. The session
                    # fan-out / CHANGE decomposition vector calls compile the
                    # identical WHERE, so one representative capture is honest.
                    filter_plan_out=vector_filter_plan_sink,
                )
            )

        # Launch the lexical channel (bm25 or keyword_ppr, #1391) in parallel
        # with vector search (independent channel).
        bm25_chunks_task: asyncio.Task[list[tuple[UUID, float, Chunk]]] | None = None
        if not skip_bm25_channel and self._lexical_channel_active() and self._storage:
            bm25_chunks_task = asyncio.create_task(
                self._lexical_search_chunks(
                    query=query,
                    namespace_id=namespace_id,
                    limit=self._config.bm25_top_k,
                    filter_ast=filter_ast,
                    # Capture the BM25 channel's pushdown plan from the temporal
                    # store's actual ``search_fulltext`` compile (same WHERE the
                    # vector channel pushes). Ignored by the keyword_ppr branch.
                    filter_plan_out=bm25_filter_plan_sink,
                    degradations=degradations,
                )
            )

        # Step 3a: Find entry entities via vector search (runs in parallel with vector_chunks_task)
        entry_entities = await self._vector_search_entities(
            query_embedding=query_embedding,
            namespace_id=namespace_id,
            limit=entry_limit,
            degradations=degradations,
        )

        if not entry_entities:
            logger.debug("No entry entities found, falling back to simple retrieval")
            # Cancel the parallel tasks since we're taking a different path
            if vector_chunks_task is not None:
                vector_chunks_task.cancel()
                try:
                    await vector_chunks_task
                except asyncio.CancelledError:
                    pass
            if bm25_chunks_task is not None:
                bm25_chunks_task.cancel()
                try:
                    await bm25_chunks_task
                except asyncio.CancelledError:
                    pass
            return await self._simple_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                limit=limit,
                routing=routing,
                effective_recency=(recency_bias if recency_bias is not None else _tp.recency_weight),
                decay_days_override=_tp.decay_days_override,
                temporal_sort=_tp.temporal_sort,
                recency_floor=_tp.recency_floor,
                temporal_signal=temporal_signal,
                min_similarity=min_similarity,
                mode=mode,
                hybrid_alpha_override=hybrid_alpha_override,
                filter_ast=filter_ast,
                # Carry any degradations recorded before the fallback (e.g. the
                # entry-entity vector search that just collapsed graph expansion
                # to vector-only) so they surface on the result (#1158).
                degradations=degradations,
            )

        # Step 3b: Session-aware parallel retrieval
        # When enabled and the query is temporal, discover which sessions the
        # entry entities belong to. If they span multiple sessions, cancel the
        # single global vector search and fan out parallel per-session searches
        # to improve session_crossing_recall.
        session_aware_activated = False
        _session_aware_chunks: list[tuple[UUID, float, Chunk]] | None = None
        if (
            self._config.enable_session_aware_search
            and self._dual_nodes is not None
            and temporal_signal
            and temporal_signal.is_temporal
            and len(entry_entities) >= 1
        ):
            with trace_span(
                "khora.vectorcypher.session_discovery",
                entity_count=len(entry_entities),
            ) as sa_span:
                try:
                    entity_channels = await self._dual_nodes.get_entity_channels(
                        entity_ids=[str(e[0]) for e in entry_entities],
                        namespace_id=str(namespace_id),
                    )
                    sa_span.set_attribute("channel_count", len(entity_channels))
                except Exception as e:
                    logger.warning(f"Session discovery failed, using global search: {e}")
                    entity_channels = []

            # Best-effort efficiency guard: when the caller pins a
            # ``metadata.channel`` at the top level of its filter, fan out only
            # over the discovered sessions that intersect it. NOTE: the
            # discovered ``channel`` (a Chunk-node column) and the caller's
            # ``metadata.channel`` (a JSONB blob key) are distinct fields — this
            # intersect is purely an efficiency narrowing. Correctness never
            # depends on it: filter_ast is AND-composed into every per-session
            # search, so even when the two fields differ the result row-set is
            # still correct (possibly empty). A disjoint or empty intersect
            # skips the fan-out and keeps the global vector task (which already
            # carries filter_ast).
            from khora.filter.execute import caller_channel_constraint

            caller_channels = caller_channel_constraint(filter_ast) if filter_ast is not None else None
            if caller_channels is not None:
                fanout_channels = [ch for ch in entity_channels if ch in caller_channels]
            else:
                fanout_channels = entity_channels

            if len(fanout_channels) >= 2:
                # Cancel the original global vector search
                if vector_chunks_task is not None:
                    vector_chunks_task.cancel()
                    try:
                        await vector_chunks_task
                    except asyncio.CancelledError:
                        pass

                # Fan out per-session vector searches + one unscoped fallback
                session_aware_activated = True
                per_session_limit = max(3, limit // len(fanout_channels))
                logger.info(
                    f"Session-aware search: {len(fanout_channels)} sessions, {per_session_limit} chunks/session"
                )

                from khora.core.temporal import ChunkTemporalFilter as _TF

                session_tasks: list[asyncio.Task[list[tuple[UUID, float, Chunk]]]] = []
                for ch in fanout_channels:
                    # Build a per-session temporal filter, preserving any existing
                    # time-range constraints from the original filter.
                    if temporal_filter is not None:
                        session_tf = _TF(
                            occurred_after=temporal_filter.occurred_after,
                            occurred_before=temporal_filter.occurred_before,
                            created_after=temporal_filter.created_after,
                            created_before=temporal_filter.created_before,
                            source_system=temporal_filter.source_system,
                            author=temporal_filter.author,
                            channel=ch,
                            tags=temporal_filter.tags,
                            additional=temporal_filter.additional,
                        )
                    else:
                        session_tf = _TF(channel=ch)

                    # Session fan-out is PRIMARY retrieval (per-session
                    # partitioning of the same query — it replaces the cancelled
                    # global vector task), so it carries the caller filter.
                    # filter_ast composes orthogonally with the per-session
                    # ChunkTemporalFilter (both AND in the same conditions list).
                    session_tasks.append(
                        asyncio.create_task(
                            self._vector_search_chunks(
                                query_embedding=query_embedding,
                                namespace_id=namespace_id,
                                temporal_filter=session_tf,
                                query_text=query,
                                limit=per_session_limit,
                                hybrid_alpha_override=effective_hybrid_alpha,
                                min_similarity=min_similarity,
                                filter_ast=filter_ast,
                            )
                        )
                    )

                # Also keep one unscoped search as fallback (in case sessions
                # are incomplete or the query spans non-entity sessions). This
                # unscoped search always runs when fan-out activates, so it is
                # where we capture the vector channel's pushdown plan (the
                # original global task was cancelled above, possibly before its
                # compile ran).
                # INVARIANT (why one capture is faithful): the caller filter is
                # AND-composed orthogonally with each per-session scope, so every
                # per-session search and this unscoped fallback compile the
                # IDENTICAL ``khora_chunks`` WHERE for the filter — only the
                # session-scope conditions differ. The single plan captured here
                # therefore represents the pushdown of all per-session searches.
                fallback_limit = max(3, limit // 3)
                session_tasks.append(
                    asyncio.create_task(
                        self._vector_search_chunks(
                            query_embedding=query_embedding,
                            namespace_id=namespace_id,
                            temporal_filter=temporal_filter,
                            query_text=query,
                            limit=fallback_limit,
                            hybrid_alpha_override=effective_hybrid_alpha,
                            min_similarity=min_similarity,
                            filter_ast=filter_ast,
                            filter_plan_out=vector_filter_plan_sink,
                        )
                    )
                )

                # Gather all per-session results
                all_session_results = await asyncio.gather(*session_tasks, return_exceptions=True)

                # Merge and deduplicate by chunk_id, keeping the best score
                merged: dict[UUID, tuple[UUID, float, Chunk]] = {}
                for i, result in enumerate(all_session_results):
                    if isinstance(result, Exception):
                        ch_label = fanout_channels[i] if i < len(fanout_channels) else "fallback"
                        logger.warning(f"Session search failed for channel={ch_label}: {result}")
                        continue
                    for chunk_id, score, chunk in result:
                        if chunk_id not in merged or score > merged[chunk_id][1]:
                            merged[chunk_id] = (chunk_id, score, chunk)

                # Store merged results; we'll use _session_aware_chunks instead
                # of awaiting vector_chunks_task in Step 6.
                _session_aware_chunks = list(merged.values())
                logger.info(
                    f"Session-aware search merged {len(merged)} unique chunks "
                    f"from {len(fanout_channels)} sessions + fallback"
                )

        # Compute adaptive depth based on entry entity count
        # This prevents explosion when many entities are found
        depth = self._router.compute_adaptive_depth(
            entry_entity_count=len(entry_entities),
            base_depth=base_depth,
        )

        # Step 4: Cypher expand to find related entities
        # For temporal queries (STATE_QUERY/RECENCY/CHANGE), prefer currently-valid
        # entities by filtering out those whose valid_until has passed.
        graph_fallback = False
        graph_error_msg: str | None = None
        try:
            expanded_entities, entity_info_map = await self._cypher_expand(
                entry_entity_ids=[e[0] for e in entry_entities],
                namespace_id=namespace_id,
                depth=depth,
                prefer_current=_tp.prefer_current,
                degradations=degradations,
            )
        except _NEO4J_TRANSIENT_ERRORS as exc:
            logger.warning(
                f"Neo4j unavailable during cypher_expand for namespace={namespace_id}, "
                f"falling back to vector-only results: {type(exc).__name__}: {exc}"
            )
            expanded_entities = {}
            entity_info_map = {}
            graph_fallback = True
            graph_error_msg = type(exc).__name__

        # Step 4b: Bi-temporal version filtering
        # For EXPLICIT temporal queries with a parsed date, narrow entities to
        # those whose version was valid at the target date.
        version_history: list[dict[str, Any]] | None = None
        all_entity_ids = list({e[0] for e in entry_entities} | expanded_entities.keys())

        if temporal_signal and temporal_signal.is_temporal:
            if temporal_signal.category == TemporalCategory.EXPLICIT and temporal_signal.temporal_filter is not None:
                # Derive a target date from the temporal filter
                tf = temporal_signal.temporal_filter
                target_date = getattr(tf, "occurred_before", None) or getattr(tf, "occurred_after", None)
                if target_date is not None:
                    with trace_span("khora.vectorcypher.version_filter", target_date=target_date.isoformat()):
                        try:
                            # On graph-less backends (sqlite_lance / surrealdb)
                            # this is a no-op that returns entities unfiltered and
                            # records a structured degradation — the embedded schema
                            # lacks the version_valid_from/to columns. Occurred-bounds
                            # chunk filtering is still honored via ``temporal_filter``.
                            all_entity_ids = await self._version_filter_entities(
                                entity_ids=all_entity_ids,
                                namespace_id=namespace_id,
                                target_date=target_date,
                                degradations=degradations,
                            )
                        except _NEO4J_TRANSIENT_ERRORS as exc:
                            logger.warning(
                                f"Version filter failed for namespace={namespace_id}, "
                                f"skipping: {type(exc).__name__}: {exc}"
                            )

            elif temporal_signal.category == TemporalCategory.CHANGE:
                # For CHANGE queries, fetch version history via SUPERSEDES edges
                with trace_span("khora.vectorcypher.version_history", entity_count=len(all_entity_ids)):
                    try:
                        version_history = await self._fetch_version_history(
                            entity_ids=all_entity_ids,
                            namespace_id=namespace_id,
                        )
                    except _NEO4J_TRANSIENT_ERRORS as exc:
                        logger.warning(
                            f"Version history fetch failed for namespace={namespace_id}, "
                            f"skipping: {type(exc).__name__}: {exc}"
                        )
                        version_history = None

        # Step 5: Fetch chunks from all entities
        # Skip when graph is unavailable — _fetch_chunks_from_entities uses
        # Neo4j via DualNodeManager and would fail with the same error.
        #
        # Issue #542 — when the PPR retrieval flag is on AND a storage
        # coordinator is wired, replace the BFS-driven chunk fetch with
        # query-time Personalized PageRank.  Falls back to vector-only
        # (graph_chunks=[]) when entities or seed overlap are empty.
        # The vector channel is preserved either way and fusion still
        # runs — so a degenerate PPR result never silently kills recall.
        ppr_path_used = False
        ppr_entity_scores: dict[UUID, float] = {}
        # Sink for the graph channel's pushdown plan. The Neo4j BFS fetch appends
        # the consumed keys of the compile it actually spliced into the executed
        # ``WHERE`` (threaded through ``_fetch_chunks_from_entities`` →
        # ``get_chunks_by_entities``), so the report derives from the executing
        # compile. The PPR path and the storage-fallback (SurrealDB / embedded)
        # paths push nothing and leave it empty, so every leaf falls to
        # ``post_filtered_keys`` for them.
        graph_pushed_keys_sink: list[frozenset[str]] = []
        if graph_fallback:
            graph_chunks: list[tuple[UUID, float, Chunk]] = []
        elif self._config.enable_ppr_retrieval and self._storage is not None:
            from khora.engines.vectorcypher.ppr_retrieval import ppr_retrieve_chunks

            # Build a chunk_id -> cosine-similarity map from the vector channel so
            # PPR can blend graph mass with query relevance (HippoRAG-2 style,
            # mass * (1 + sim)) instead of ordering chunks query-agnostically by
            # pure PR mass. The vector task is awaited here and again at Step 6;
            # asyncio caches the result, so this peek is effectively free.
            chunk_similarity: dict[UUID, float] = {}
            try:
                if _session_aware_chunks is not None:
                    _vc_for_sim = _session_aware_chunks
                elif vector_chunks_task is not None:
                    _vc_for_sim = await vector_chunks_task
                else:
                    _vc_for_sim = []
                chunk_similarity = {cid: score for cid, score, _ in _vc_for_sim}
            except Exception:  # noqa: S110 - sim is optional; PPR still runs without it
                chunk_similarity = {}

            ppr_chunks, ppr_entity_scores = await ppr_retrieve_chunks(
                storage=self._storage,
                namespace_id=namespace_id,
                entry_entities=entry_entities,
                damping=self._config.ppr_damping,
                max_iter=self._config.ppr_max_iter,
                tol=self._config.ppr_tol,
                top_entities=self._config.ppr_top_entities,
                chunk_similarity=chunk_similarity,
                limit=graph_fetch_limit,
                neighborhood_per_seed_limit=self._config.ppr_neighborhood_per_seed_limit,
                max_neighborhood_entities=self._config.ppr_max_neighborhood_entities,
                # ADR-001 (#1373): surface a Degradation on the engine_info list
                # when the graph channel silently returns nothing.
                out_degradations=degradations,
            )
            graph_chunks = ppr_chunks
            ppr_path_used = bool(ppr_chunks)
            logger.debug(
                "PPR retrieval path: {} chunks scored over {} entities",
                len(graph_chunks),
                len(ppr_entity_scores),
            )
        else:
            graph_chunks = await self._fetch_chunks_from_entities(
                entity_ids=all_entity_ids,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                limit=graph_fetch_limit,  # Fetch more for fusion (widened on residual metadata)
                temporal_sort=_tp.temporal_sort,
                prefer_current=_tp.prefer_current,
                filter_ast=filter_ast,
                graph_pushed_keys_out=graph_pushed_keys_sink,
            )

        # Step 6: Wait for parallel vector chunk search to complete
        # This was started at the beginning and may already be done.
        # If session-aware search produced results, use those instead.
        if _session_aware_chunks is not None:
            vector_chunks = _session_aware_chunks
            # The original task was already cancelled; no need to await.
        elif vector_chunks_task is not None:
            vector_chunks = await vector_chunks_task
        else:
            # #833: mode=GRAPH skipped the vector chunk channel entirely.
            vector_chunks = []

        # Fallback: if temporal filter was too restrictive, re-run without it.
        # SKIP fallback when the temporal signal is EXPLICIT with a parsed date —
        # sparse results are the correct signal (the data may not exist for that
        # time window, which is important for abstention on unanswerable queries).
        # Also skip when mode=GRAPH (vector channel is intentionally disabled).
        is_explicit_with_date = (
            temporal_signal
            and temporal_signal.category == TemporalCategory.EXPLICIT
            and temporal_signal.temporal_filter is not None
        )
        if (
            not skip_vector_channel
            and temporal_filter
            and len(vector_chunks) < limit // 2
            and not is_explicit_with_date
            # Skip under ANY caller filter: the re-run intentionally drops the
            # caller filter (it re-searches with temporal_filter=None and does
            # not thread filter_ast), so re-running it would smuggle
            # filter-violating chunks into RRF. Under a caller filter, sparse
            # vector results are the correct (deterministic) signal.
            and filter_ast is None
        ):
            logger.debug(f"Temporal filter too restrictive ({len(vector_chunks)} results), falling back to unfiltered")
            vector_chunks = await self._vector_search_chunks(
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=None,
                query_text=query,
                limit=limit,
                hybrid_alpha_override=effective_hybrid_alpha,
                min_similarity=min_similarity,
                # Reached only when filter_ast is None (gated above); the re-run
                # drops the temporal filter, so there is no caller filter to
                # thread here.
            )

        # Await BM25 results (also launched in parallel at the beginning)
        bm25_chunks: list[tuple[UUID, float, Chunk]] = []
        if bm25_chunks_task is not None:
            try:
                bm25_chunks = await bm25_chunks_task
            except Exception as e:
                logger.warning(f"BM25 channel failed, continuing without: {e}")

        # Step 6a: Temporal query decomposition for CHANGE queries
        # Runs a second vector search focused on the "current state" sub-query
        # to ensure both past and present evidence are retrieved. The original
        # query naturally retrieves past-state chunks ("used to", "previously"),
        # while the decomposed sub-query targets current-state chunks.
        #
        # The CHANGE signal is derived from the query string (not from
        # temporal_filter), so this sub-search can fire under a filtered recall.
        # It runs with temporal_filter=None (current-state intent) but carries
        # the caller filter_ast, which _vector_search_chunks pushes down, so its
        # merged results are filter-correct.
        if temporal_signal and temporal_signal.category == TemporalCategory.CHANGE and version_history:
            current_state_query = self._decompose_change_query(query)
            if current_state_query and current_state_query != query:
                with trace_span(
                    "khora.vectorcypher.change_decomposition",
                    sub_query_hash=bounded_text_hash(current_state_query),
                    sub_query_length=len(current_state_query),
                ):
                    sub_embedding = await self._embedder.embed(current_state_query)
                    sub_vector_chunks = await self._vector_search_chunks(
                        query_embedding=sub_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=None,  # No temporal filter — want current state
                        query_text=current_state_query,
                        limit=limit,
                        min_similarity=min_similarity,
                        filter_ast=filter_ast,
                    )
                    # Merge sub-query results, deduplicating by chunk ID
                    existing_ids = {c[0] for c in vector_chunks}
                    new_chunks = [c for c in sub_vector_chunks if c[0] not in existing_ids]
                    if new_chunks:
                        vector_chunks = vector_chunks + new_chunks
                        logger.debug(
                            f"CHANGE decomposition added {len(new_chunks)} chunks "
                            f"from sub-query: {current_state_query[:60]}"
                        )

        # Step 6a': Recency channel — pool augmentation (Issue #567 A3).
        # When RECENCY/CHANGE fires and the flag is enabled, fetch the most
        # recent N chunks (pure ORDER BY occurred_at DESC, no embedding) and
        # merge those that exceed the cosine relevance floor into the vector
        # pool. The existing RRF + recency boost then scores them alongside
        # the cosine top-K. Three safety properties:
        #   1. Pool-augmentation only — never replaces cosine candidates.
        #   2. Relevance gate prevents today's HR-all-hands from muscling
        #      into the top-K for a niche query (Devil's-Advocate demand #3).
        #   3. Skipped for historical / counterfactual queries — they by
        #      definition don't benefit from injecting recent chunks. PR #571
        #      LoCoMo --small showed running the channel anyway cost ~16.7pp
        #      counterfactual_accuracy on a 6-q subset.
        #
        #      Issue #1227: the veto must depend on the recency-CHANNEL flag,
        #      not the unrelated recency-FLOOR flag. Channel-on / floor-off is
        #      a reachable combo of two independent user settings; gating the
        #      veto on the floor flag left it dead in that combo, re-injecting
        #      today's chunks into the exact counterfactual queries it skips.
        #
        #      Veto signal:
        #        * anti-recency token in the query ("ever", "all-time", "history
        #          of …") — detected directly so the veto works regardless of
        #          the floor flag.
        #        * floor-on + ``temporal_filter is None`` — preserves the
        #          original proxy that also caught the LLM-disambiguation veto:
        #          when floor synthesis ran but was vetoed (anti-recency OR a
        #          non-RECENT LLM intent), the filter stays None. A legit
        #          floor-on recency query synthesizes a filter (non-None) and
        #          is NOT vetoed.
        from khora.query.temporal_detection import has_anti_recency_token

        synthesis_vetoed = (
            self._config.temporal_recency_channel_enabled
            and temporal_signal is not None
            and temporal_signal.is_temporal
            and _tp.default_window_days is not None
            and (
                has_anti_recency_token(query)
                or (self._config.temporal_recency_floor_enabled and temporal_filter is None)
            )
        )
        if (
            self._config.temporal_recency_channel_enabled
            and temporal_signal is not None
            and temporal_signal.is_temporal
            and _tp.default_window_days is not None
            and not synthesis_vetoed
        ):
            # Intentionally pass temporal_filter=None: the recency channel's
            # job is to surface today's chunks even when the cosine channel
            # narrowed by the synthesized 14d floor. The cosine relevance gate
            # (temporal_query_relevance_floor) is the safeguard against
            # irrelevant-but-fresh chunks muscling in (Devil's-Advocate
            # follow-up: decouple channel filter from synthesized floor).
            #
            # The recency channel reads the temporal store's chunk table
            # (``khora_chunks`` on pgvector). The caller filter is pushed into
            # that recency SQL — the store compiles it to the SAME raise-mode
            # ``khora_chunks`` WHERE the vector path uses — so no filter-violating
            # chunk is fetched and a leaf the compiler cannot push raises (matching
            # the vector channel's fail-loud contract). Quality trade-off
            # (accepted): under a restrictive caller filter the recency SQL may
            # return fewer rows, slightly under-recalling the "current state"
            # intent on a tightly date-filtered namespace.
            # The recency channel records its own ChannelPlan internally — and
            # only when it actually produced gating chunks on the real execution
            # path. Backends WITHOUT the capability (e.g. SurrealDB) return the
            # protocol default [] from ``search_recent_chunks``, so the channel
            # honestly never appears in the report rather than being credited with
            # a disposition it never reached.
            recent_chunks = await self._recency_channel_chunks(
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=None,
                filter_ast=filter_ast,
                filter_channel_plans=filter_channel_plans,
                degradations=degradations,
            )
            if recent_chunks:
                existing_ids = {c[0] for c in vector_chunks}
                merged_in = [rc for rc in recent_chunks if rc[0] not in existing_ids]
                if merged_in:
                    vector_chunks = vector_chunks + merged_in
                    logger.debug(
                        "Recency channel merged {} new chunks (relevance>= {}, category={})",
                        len(merged_in),
                        self._config.temporal_query_relevance_floor,
                        temporal_signal.category.value,
                    )
                    from khora.telemetry.temporal_metrics import record_recency_channel_fired

                    record_recency_channel_fired(category=temporal_signal.category.value)

        # Step 6b: Lazy entity expansion for vector-only chunks
        # Recovers graph coverage lost from low skeleton_core_ratio by doing
        # lightweight keyword matching (no LLM) on chunks without MENTIONED_IN edges
        if self._config.lazy_entity_expansion and vector_chunks and self._config.skeleton_core_ratio <= 0.6:
            graph_chunk_ids = {c[0] for c in graph_chunks}
            vector_only = [c for c in vector_chunks if c[0] not in graph_chunk_ids]
            if vector_only:
                expanded = self._lazy_expand_chunks(vector_only, entry_entities, entity_info_map)
                if expanded:
                    graph_chunks = graph_chunks + expanded
                    logger.debug(f"Lazy expansion added {len(expanded)} chunks to graph results")

        # Step 6c: Full-AST in-memory post-filter on the graph chunk channel.
        # The vector + BM25 channels enforce the caller filter via the pgvector
        # pushdown; the graph (and PPR) channel reads chunks from Neo4j, where
        # only the system-key slice pushed down (metadata is a serialized JSON
        # property, not Cypher-expressible). Re-check the WHOLE AST here so a
        # metadata leaf — including one inside an ``$or`` whose Cypher side
        # collapsed to a non-constraining ``true`` — is enforced before fusion.
        # No-filter recalls leave ``graph_chunks`` untouched.
        if filter_ast is not None and graph_chunks:
            from khora.filter.compilers.python import compile_python
            from khora.filter.execute import build_compile_context, filter_leaf_keys

            # Honest graph-channel plan, built at the post-filter site so it
            # covers ALL THREE graph fetch paths uniformly — they all funnel into
            # ``graph_chunks`` and are re-checked by this one full-AST post-filter.
            # The pushed keys come from the compile that actually spliced the
            # ``WHERE`` inside ``get_chunks_by_entities``, threaded out via
            # ``graph_pushed_keys_sink`` (the same "report source = execution
            # input" pattern as ``search_fulltext``'s ``filter_plan_out``). The
            # PPR and storage-fallback (SurrealDB / embedded) paths leave the sink
            # empty, so every leaf falls to ``post_filtered_keys`` for them.
            # ``defensive_recheck=True`` because this channel ALWAYS runs the
            # full-AST in-memory post-filter below — so a fully-pushed system-only
            # filter is reported as post-filtered (flag flips) WITHOUT demoting the
            # pushed leaves out of ``pushed_keys``. Recorded only when the graph
            # channel actually held candidates this recall.
            cypher_pushed = graph_pushed_keys_sink[0] if graph_pushed_keys_sink else frozenset()
            filter_channel_plans["graph"] = ChannelPlan(
                pushed_keys=cypher_pushed,
                post_filtered_keys=filter_leaf_keys(filter_ast) - cypher_pushed,
                defensive_recheck=True,
            )

            graph_post_filter = compile_python(
                filter_ast, build_compile_context("Chunk", on_unsupported="split")
            ).predicate
            graph_chunks_before = len(graph_chunks)
            graph_chunks = [(cid, s, ch) for (cid, s, ch) in graph_chunks if graph_post_filter(ch)]
            if not graph_chunks:
                # The caller filter narrowed the graph channel to empty (it held
                # candidates before the post-filter). Emit the service-level
                # filter counter for every such recall.
                record_graph_channel_empty()
                # When the SQL-pushed vector/BM25 channels still returned filtered
                # rows, the graph side under-recalled relative to the completeness
                # backstop: record one degradation so callers see the dropped channel.
                if vector_chunks or bm25_chunks:
                    logger.warning(
                        f"Graph chunk channel emptied by metadata post-filter "
                        f"({graph_chunks_before} dropped); vector/BM25 channels returned rows"
                    )
                    degradations.append(
                        Degradation(
                            component="vectorcypher.graph_channel",
                            reason="empty_under_filter",
                            detail=f"{graph_chunks_before} graph chunks dropped by metadata post-filter",
                        )
                    )

        # Assemble the SQL-channel filter-pushdown plans from each channel's
        # ACTUAL compile (graph was recorded at its post-filter site; recency
        # records itself inside ``_recency_channel_chunks`` only when it actually
        # produced surviving chunks that GATE in RRF). A channel appears ONLY if it actually
        # executed this recall AND a caller filter is present. All channels GATE
        # (each feeds RRF), so the report builder's per-leaf partition reports a
        # leaf as post-filtered if ANY gating channel re-checked it in memory,
        # even when the SQL channels pushed it.
        if filter_ast is not None:
            # Vector channel: the pgvector store appended the ChannelPlan it
            # built from the SAME ``khora_chunks`` compile its search ran
            # (on_unsupported="raise" — a populated sink means every leaf was
            # consumed). Present whenever the vector channel ran (not skipped in
            # mode=GRAPH) and the sink was populated.
            if not skip_vector_channel and vector_filter_plan_sink:
                filter_channel_plans["vector"] = vector_filter_plan_sink[0]

            # BM25 channel: the plan the temporal-store ``search_fulltext`` path
            # appended to the sink, built from ITS OWN fulltext compile — so it
            # matches the vector channel's split for the same backend (all-pushed
            # on raise-mode pg/surreal, partial on split-mode sqlite_lance).
            # Absent when BM25 fell back to the coordinator (only reached with no
            # filter) or failed before the temporal call.
            if bm25_filter_plan_sink:
                filter_channel_plans["bm25"] = bm25_filter_plan_sink[0]

        # Step 7: RRF fusion with score normalization and dynamic weights
        fused_results = self._fuse_results(
            vector_chunks=vector_chunks,
            graph_chunks=graph_chunks,
            bm25_chunks=bm25_chunks if bm25_chunks else None,
            use_normalization=True,
            routing=routing,
            is_temporal=_tp.recency_weight > 0.2,
        )

        # Step 8: Apply recency boost driven by temporal signal category.
        # A per-call ``recency_bias`` (threaded explicitly from the engine,
        # #1156) overrides the signal-derived weight and the WS4 clamp below.
        if recency_bias is not None:
            effective_recency = recency_bias
        else:
            effective_recency = _tp.recency_weight
            # WS4: Also boost when explicit temporal filter is active
            if temporal_filter is not None and effective_recency > 0:
                effective_recency = max(effective_recency, 0.4)
        if effective_recency > 0:
            with trace_span("khora.vectorcypher.recency_boost", chunk_count=len(fused_results)):
                recency_scores = self._calculate_recency_scores(
                    fused_results, decay_days_override=_tp.decay_days_override
                )
                fused_results = apply_recency_boost(
                    fused_results,
                    recency_scores,
                    recency_weight=effective_recency,
                    recency_floor=_tp.recency_floor,
                )

        # Step 8b: Apply coherence scoring to penalize word-shuffled confounders
        if self._config.coherence_weight > 0:
            with trace_span("khora.vectorcypher.coherence_boost", chunk_count=len(fused_results)):
                # Normalize fused scores to [0, 1] BEFORE the convex blend so
                # coherence_weight behaves as the documented ~w nudge. Without
                # this, the raw weighted-RRF scale (top ~0.02 at k=60) makes the
                # w*coherence term (up to w) dominate ranking and demote the
                # relevance winner (#1056). This is an internal-only normalization
                # feeding the blend; the reported score is set absolutely at the
                # exit via attach_relevance_scores (#811).
                fused_results = normalize_scores(fused_results)
                fused_results = apply_coherence_boost(
                    fused_results,
                    coherence_weight=self._config.coherence_weight,
                )

        # Step 8c: Cross-encoder reranking (after boosts, before version scoring)
        if self._config.enable_reranking:
            with trace_span("khora.vectorcypher.reranking", candidate_count=len(fused_results)):
                fused_results = await self._apply_reranking(query, fused_results, limit, namespace_id=namespace_id)

        # Step 8d: LLM reranking of top-N for temporal queries (after cross-encoder).
        # Gating centralised in ``_evaluate_llm_rerank_gate`` (issue #814) — covers
        # the not-temporal, no-version-metadata, and decisive-winner skips and
        # emits the one-time warning when mode='auto' triggers the version gate.
        if self._config.enable_llm_reranking:
            should_run, skip_reason = self._evaluate_llm_rerank_gate(
                fused_results,
                temporal_signal,
                namespace_id=namespace_id,
            )
            if not should_run and skip_reason is not None:
                _LLM_RERANKING_SKIPPED_COUNTER.add(1, attributes={"reason": skip_reason})
            if should_run:
                with trace_span(
                    "khora.vectorcypher.llm_reranking",
                    candidate_count=len(fused_results),
                    mode=self._config.llm_reranking_mode,
                ):
                    fused_results = await self._apply_llm_reranking(
                        query, fused_results, limit, namespace_id=namespace_id
                    )

        # Step 8e: Version-aware scoring — the FINAL score adjustment.
        # Applied after ALL reranking (cross-encoder + LLM) so nothing can
        # undo the version preference. The LLM reranker provides valuable
        # content understanding but has no temporal awareness; version scoring
        # layers recency preference on top of the LLM's relevance baseline.
        # CHANGE excluded: needs both old and new versions for comparison.
        # ORDINAL excluded: needs full version history for ordering.
        if temporal_signal and temporal_signal.category in (
            TemporalCategory.STATE_QUERY,
            TemporalCategory.RECENCY,
        ):
            from collections import defaultdict

            entity_versions: dict[str, int] = defaultdict(int)  # entity -> max version
            chunk_versions: dict[UUID, int] = {}  # chunk_id -> version

            for r in fused_results:
                meta = r.item.metadata if hasattr(r.item, "metadata") and r.item.metadata else {}
                if isinstance(meta, dict):
                    version = meta.get("version") or meta.get("entity_version", 0)
                    if version:
                        chunk_versions[r.item_id] = int(version)
                        for ref in meta.get("entity_refs") or []:
                            entity_versions[ref] = max(entity_versions[ref], int(version))

            if entity_versions:
                _VERSION_DECAY = 0.7  # Stronger penalty: v1/v5 → 0.44 (was 0.6 with 0.5)
                for r in fused_results:
                    v = chunk_versions.get(r.item_id, 0)
                    if v > 0:
                        meta = r.item.metadata if hasattr(r.item, "metadata") and r.item.metadata else {}
                        if isinstance(meta, dict):
                            for ref in meta.get("entity_refs") or []:
                                max_v = entity_versions.get(ref, v)
                                if max_v > v:
                                    ratio = v / max_v
                                    r.rrf_score *= 1.0 - _VERSION_DECAY * (1.0 - ratio)
                                    break

        # Surface an ABSOLUTE relevance score (raw vector cosine) on each chunk
        # instead of the per-result-set min-max normalized RRF score (#811).
        # Min-max forced the top chunk to 1.0 and the bottom to 0.0 regardless
        # of actual relevance, so off-topic queries still reported score=1.0 and
        # no threshold was meaningful. Ranking is unchanged - order is decided by
        # rrf_score (after fusion + boosts + reranking); this only rewrites the
        # reported VALUE to the raw cosine captured pre-fusion.
        fused_results = attach_relevance_scores(fused_results)

        # Step 8f: MMR diversity selection (#1018). When enabled, choose the
        # top-``limit`` fused results by Maximal Marginal Relevance over the
        # broad-recall pool instead of pure score order, so near-duplicate
        # chunks don't crowd out diverse-but-relevant ones. Honors
        # ``query.enable_diversity`` / ``query.diversity_lambda`` (previously
        # inert on VectorCypher). Reorders ``fused_results`` so the selected
        # results lead; the existing ``[:limit]`` slices below pick them up.
        if self._config.enable_diversity and len(fused_results) > limit:
            with trace_span("khora.vectorcypher.mmr_diversity", candidate_count=len(fused_results), k=limit):
                fused_results = self._mmr_select_fused(
                    fused_results, query_embedding, k=limit, lambda_param=self._config.diversity_lambda
                )

        # Build result
        chunk_results = [(r.item, r.rrf_score) for r in fused_results[:limit]]

        # Classify each chunk by which search method(s) found it
        vector_only_ids: list[UUID] = []
        graph_only_ids: list[UUID] = []
        both_ids: list[UUID] = []
        for r in fused_results[:limit]:
            has_vector = r.vector_rank is not None
            has_graph = r.graph_rank is not None
            if has_vector and has_graph:
                both_ids.append(r.item_id)
            elif has_vector:
                vector_only_ids.append(r.item_id)
            elif has_graph:
                graph_only_ids.append(r.item_id)

        vector_ids = vector_only_ids + both_ids
        graph_ids = graph_only_ids + both_ids

        # Entity IDs are discovered via vector similarity then expanded via graph,
        # so they are attributed to "graph" (the graph expansion is what surfaces them)
        entity_ids_str = [str(eid) for eid, _ in entry_entities[: self._config.max_entities]]

        search_methods = {
            "chunk_overlap": {
                "vector_only": {"ids": [str(id) for id in vector_only_ids], "count": len(vector_only_ids)},
                "graph_only": {"ids": [str(id) for id in graph_only_ids], "count": len(graph_only_ids)},
                "vector_and_graph": {"ids": [str(id) for id in both_ids], "count": len(both_ids)},
            },
            "entity_overlap": {
                "vector_and_graph": {"ids": entity_ids_str, "count": len(entity_ids_str)},
                "vector_only": {"ids": [], "count": 0},
                "graph_only": {"ids": [], "count": 0},
            },
            "by_method": {
                "vector": {"chunk_ids": [str(id) for id in vector_ids], "count": len(vector_ids)},
                "graph": {"chunk_ids": [str(id) for id in graph_ids], "count": len(graph_ids)},
            },
        }

        # Collect all entity IDs: entry entities (score 1.0) + expanded (score from graph distance).
        # Entry entities come first (higher relevance), expanded follow sorted by score desc.
        all_entity_scores: list[tuple[UUID, float]] = []
        seen_ids: set[UUID] = set()
        for eid, score in entry_entities:
            if eid not in seen_ids:
                all_entity_scores.append((eid, score))
                seen_ids.add(eid)
        for eid, score in sorted(expanded_entities.items(), key=lambda x: x[1], reverse=True):
            if eid not in seen_ids:
                all_entity_scores.append((eid, score))
                seen_ids.add(eid)

        # Cap total entities to max_entities
        all_entity_scores = all_entity_scores[: self._config.max_entities]

        # OPTIMIZATION: Fire entity batch-fetch (PostgreSQL) and relationship
        # fetch (Neo4j) in parallel — they hit different databases and both
        # only need the final entity ID list computed above.
        entity_ids_to_fetch = [eid for eid, _ in all_entity_scores]
        entity_ids_str = [str(eid) for eid, _ in all_entity_scores]

        # Start relationship fetch immediately (doesn't need full Entity objects)
        # Skip Neo4j relationship fetch when graph is unavailable to avoid
        # blocking on a second timeout.
        if self._dual_nodes is not None and not graph_fallback:
            rels_task = asyncio.create_task(
                self._dual_nodes.get_relationships_between(
                    entity_ids_str,
                    str(namespace_id),
                    limit=self._config.max_relationships,
                )
            )
        else:
            # SurrealDB: no dual node manager, fetch via storage coordinator
            async def _fetch_rels_from_storage() -> list:
                if not self._storage or not self._storage._graph:
                    return []
                rels = []
                for eid in entity_ids_to_fetch[:10]:
                    try:
                        entity_rels = await self._storage._graph.get_entity_relationships(
                            eid, namespace_id=namespace_id, limit=20
                        )
                        for r in entity_rels:
                            rels.append(
                                {
                                    "id": str(r.id),
                                    "source_entity_id": str(r.source_entity_id),
                                    "target_entity_id": str(r.target_entity_id),
                                    "relationship_type": r.relationship_type,
                                    "description": r.description or "",
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Failed to fetch relationships for entity {eid}: {e}")
                return rels

            rels_task = asyncio.create_task(_fetch_rels_from_storage())

        # Batch-fetch full entities from storage in parallel
        entity_results: list[tuple[Entity, float]] = []

        if entity_ids_to_fetch and self._storage:
            try:
                entities_map = await self._storage.get_entities_batch(entity_ids_to_fetch, namespace_id=namespace_id)
                for eid, score in all_entity_scores:
                    if eid in entities_map:
                        entity_results.append((entities_map[eid], score))
                    else:
                        # Fallback: use info from graph expansion
                        info = entity_info_map.get(str(eid), {})
                        entity = Entity(
                            id=eid,
                            namespace_id=namespace_id,
                            name=info.get("name", ""),
                            entity_type=info.get("entity_type", ""),
                            description=info.get("description", ""),
                            source_tool=info.get("source_tool", ""),
                        )
                        entity_results.append((entity, score))
            except Exception as e:
                logger.warning(f"Failed to batch-fetch entities, using stubs: {e}")
                # Fall back to stub construction
                for eid, score in all_entity_scores:
                    info = entity_info_map.get(str(eid), {})
                    entity = Entity(
                        id=eid,
                        namespace_id=namespace_id,
                        name=info.get("name", ""),
                        entity_type=info.get("entity_type", ""),
                        description=info.get("description", ""),
                        source_tool=info.get("source_tool", ""),
                    )
                    entity_results.append((entity, score))
        else:
            # No storage available or no entities to fetch
            for eid, score in all_entity_scores:
                info = entity_info_map.get(str(eid), {})
                entity = Entity(
                    id=eid,
                    namespace_id=namespace_id,
                    name=info.get("name", ""),
                    entity_type=info.get("entity_type", ""),
                    description=info.get("description", ""),
                    source_tool=info.get("source_tool", ""),
                )
                entity_results.append((entity, score))

        # Await the parallel relationship fetch
        try:
            raw_rels = await rels_task
        except Exception as exc:
            logger.warning("Relationship fetch failed, continuing without relationships", exc_info=True)
            raw_rels = []
            # ADR-001 (issue #904): record the silent fallback so callers
            # can see that relationships are missing from the response.
            # Asymmetric with the _cypher_expand fallback above which sets
            # graph_fallback/graph_error - this site previously left no
            # machine-readable signal at all.
            degradations.append(
                Degradation(
                    component="vectorcypher.relationship_fetch",
                    reason="fetch_failed",
                    detail=str(exc)[:200] or None,
                    exception=type(exc).__name__,
                )
            )
            _REL_FETCH_DEGRADED_COUNTER.add(1, attributes={"reason": "fetch_failed"})
        entity_scores_by_id: dict[UUID, float] = {entity.id: score for entity, score in entity_results}
        entity_names_by_id: dict[UUID, str] = {entity.id: entity.name for entity, _ in entity_results}
        relationships: list[tuple[Relationship, float]] = []
        for raw in raw_rels:
            src_id = UUID(raw["source_entity_id"])
            tgt_id = UUID(raw["target_entity_id"])
            rel_score = (entity_scores_by_id.get(src_id, 0.0) + entity_scores_by_id.get(tgt_id, 0.0)) / 2
            rel = Relationship(
                id=UUID(raw["id"]) if raw.get("id") else uuid4(),
                namespace_id=namespace_id,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=raw.get("relationship_type", "RELATES_TO"),
                description=raw.get("description", "") or "",
                source_entity_name=entity_names_by_id.get(src_id, ""),
                target_entity_name=entity_names_by_id.get(tgt_id, ""),
                source_document_ids=[UUID(d) for d in (raw.get("source_document_ids") or [])],
                source_chunk_ids=[UUID(c) for c in (raw.get("source_chunk_ids") or [])],
                confidence=raw.get("confidence") if raw.get("confidence") is not None else 1.0,
                weight=raw.get("weight") if raw.get("weight") is not None else 1.0,
            )
            relationships.append((rel, rel_score))
        relationships.sort(key=lambda x: x[1], reverse=True)

        return VectorCypherResult(
            chunks=chunk_results,
            entities=entity_results,
            relationships=relationships,
            routing_decision=routing,
            metadata={
                "entry_entities": len(entry_entities),
                "expanded_entities": len(expanded_entities),
                "graph_depth": depth,
                "base_depth": base_depth,
                "adaptive_depth_applied": depth != base_depth,
                "total_chunks_before_fusion": len(graph_chunks) + len(vector_chunks) + len(bm25_chunks),
                "routing_confidence": routing.confidence,
                # Fusion telemetry
                "vector_chunk_count": len(vector_chunks),
                "graph_chunk_count": len(graph_chunks),
                "bm25_chunk_count": len(bm25_chunks),
                "is_temporal": _tp.recency_weight > 0.2,
                "recency_weight": _tp.recency_weight,
                "effective_recency": effective_recency,
                # Max raw cosine similarity from vector search (pre-fusion).
                # Used by abstention system's vector_confidence_override.
                "max_raw_vector_score": max(s for _, s, _ in vector_chunks) if vector_chunks else 0.0,
                # Session-aware search telemetry
                "session_aware_activated": session_aware_activated,
                # Bi-temporal entity version history (populated for CHANGE queries)
                "version_history": version_history,
                # Search provenance: which method(s) found each chunk
                "search_methods": search_methods,
                "graph_fallback": graph_fallback,
                **({"graph_error": graph_error_msg} if graph_error_msg else {}),
                # ADR-001 (issue #904): silent-failure entries from this path
                # (currently: relationship-fetch failure). Forwarded onto
                # ``RecallResult.engine_info["degradations"]`` by the engine
                # via the ``**result.metadata`` spread.
                "degradations": degradations,
                # Issue #542 — PPR retrieval (HippoRAG 2). Surfaced for
                # benchmark scoring and operator dashboards. False when
                # the flag is off or when the path fell back to vector-only.
                "ppr_path_used": ppr_path_used,
                "ppr_entity_count": len(ppr_entity_scores),
                # Private carrier (popped by the engine before the public spread):
                # the per-channel honest filter-pushdown plans the engine folds
                # into ``engine_info["filter"]``. Never leaks into engine_info.
                "_filter_channel_plans": filter_channel_plans,
            },
        )

    async def _vector_only_fallback(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        limit: int,
        routing: RoutingDecision,
        *,
        effective_recency: float = 0.0,
        decay_days_override: int | None = None,
        temporal_sort: bool = False,
        recency_floor: float = 0.5,
        temporal_signal: TemporalSignal | None = None,
        min_similarity: float = 0.0,
        mode: SearchMode = SearchMode.HYBRID,
        hybrid_alpha_override: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> VectorCypherResult:
        """Fallback to vector-only search when graph operations fail.

        This provides graceful degradation when Neo4j is unavailable or
        returns errors. Results are still useful, just without graph expansion.
        Any recall ``filter_ast`` is threaded into the vector-only path so the
        degraded result honors the same filter as the primary path.
        """
        logger.info("Using vector-only fallback due to graph search failure")

        # Use the simple retrieval path which only needs pgvector. When the
        # caller asked for mode=GRAPH and we ended up here because Neo4j is
        # unavailable, downgrade to HYBRID in the fallback so the user gets
        # *something* rather than an empty response - the metadata still
        # tracks the failure via ``graph_unavailable``.
        fallback_mode = SearchMode.HYBRID if mode == SearchMode.GRAPH else mode
        # If filter compilation fails here, the exception propagates intentionally:
        # a filter the vector channel cannot honor must fail loud rather than
        # return filter-violating chunks, matching the graph-path contract.
        result = await self._simple_retrieve(
            query=query,
            query_embedding=query_embedding,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            limit=limit,
            routing=routing,
            effective_recency=effective_recency,
            decay_days_override=decay_days_override,
            temporal_sort=temporal_sort,
            recency_floor=recency_floor,
            temporal_signal=temporal_signal,
            min_similarity=min_similarity,
            mode=fallback_mode,
            hybrid_alpha_override=hybrid_alpha_override,
            filter_ast=filter_ast,
        )

        # Update metadata to indicate fallback was used
        result.metadata["fallback_mode"] = "vector_only"
        result.metadata["graph_unavailable"] = True
        result.metadata["graph_fallback"] = True

        return result

    async def _apply_reranking(
        self,
        query: str,
        fused_results: list[FusedResult],
        limit: int,
        *,
        namespace_id: UUID,
    ) -> list[FusedResult]:
        """Apply cross-encoder reranking to fused results.

        Takes the top-N candidates (configured via reranking_top_n), scores them
        with the cross-encoder model, and returns re-ordered results. The reranker
        blends cross-encoder scores with original RRF scores using
        reranking_blend_weight.

        Falls back to original ordering on any error.

        Args:
            query: Original query text
            fused_results: Fused results after recency/coherence boosts
            limit: Final number of results to return

        Returns:
            Re-ordered FusedResult list (may be shorter than input)
        """
        if not fused_results:
            return fused_results

        from khora.query.reranking import CrossEncoderReranker, RerankCandidate, hydrate_doc_titles

        top_n = min(self._config.reranking_top_n, len(fused_results))
        candidates_to_rerank = fused_results[:top_n]
        remainder = fused_results[top_n:]

        # Normalize original scores to [0,1] before passing to the reranker
        # so the 0.3 original_score blend is meaningful (raw RRF scores are
        # ~0.01-0.02 which would make the blend effectively zero).
        raw_scores = [r.rrf_score for r in candidates_to_rerank]
        score_min = min(raw_scores) if raw_scores else 0.0
        score_max = max(raw_scores) if raw_scores else 1.0
        score_range = score_max - score_min
        candidates = []
        for r in candidates_to_rerank:
            # Build content with optional temporal prefix for cross-encoder context
            chunk_content = r.item.content if hasattr(r.item, "content") else str(r.item)
            if hasattr(r.item, "metadata") and isinstance(r.item.metadata, dict):
                raw = r.item.metadata
                if raw:
                    prefix_parts = []
                    session_id = raw.get("session_id") or raw.get("conversation_id")
                    if session_id:
                        prefix_parts.append(f"Session: {session_id}")
                    occurred_at = raw.get("occurred_at") or raw.get("source_timestamp")
                    if occurred_at:
                        prefix_parts.append(f"Date: {str(occurred_at)[:10]}")
                    if prefix_parts:
                        chunk_content = f"[{', '.join(prefix_parts)}] {chunk_content}"
            candidates.append(
                RerankCandidate(
                    item=r,
                    original_score=(r.rrf_score - score_min) / score_range if score_range > 1e-9 else 0.5,
                    content=chunk_content,
                    metadata=r.item.metadata if hasattr(r.item, "metadata") else {},
                    doc_title="",
                )
            )
        await hydrate_doc_titles(
            candidates,
            self._storage,
            lambda fr: getattr(getattr(fr, "item", None), "document_id", None),
            namespace_id=namespace_id,
        )

        try:
            async with self._reranker_lock:
                if self._reranker is None:
                    self._reranker = CrossEncoderReranker(model_name=self._config.reranking_model)
            results = await self._reranker.rerank(
                query, candidates, top_k=top_n, blend_weight=self._config.reranking_blend_weight
            )

            # Map reranked scores back onto FusedResult objects
            reranked: list[FusedResult] = []
            for rr in results:
                fused = rr.item  # The original FusedResult
                fused.rrf_score = rr.final_score
                reranked.append(fused)

            # Append any remainder that wasn't reranked (already sorted by original score)
            reranked.extend(remainder)
            logger.debug(f"Cross-encoder reranking applied: {top_n} candidates scored, returning top {limit}")
            return reranked[:limit]
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed, keeping original order: {e}")
            return fused_results

    def _should_skip_llm_rerank(self, top_score: float, gap: float) -> bool:
        """Decide whether to skip the LLM rerank step.

        The LLM rerank call adds 200–400 ms per query. Two independent gates
        let us skip when the cross-encoder ranking is already trustworthy:

        1. ``gap`` gate (legacy): the cross-encoder's #1 vs #2 gap is large
           enough that LLM scoring is unlikely to flip them. Tunable via
           ``llm_reranking_confidence_threshold``.

        2. ``decisive winner`` gate (Sprint 1): the cross-encoder's #1 has a
           high absolute score AND a meaningful gap to #2. This catches the
           common case where #1 is a clear answer (top_score ~ 0.85) with a
           solid lead (gap ~ 0.15) — the legacy gap-only gate may have a
           threshold low enough that we still pay LLM cost on these.

        Returns ``True`` when EITHER gate fires.
        """
        if gap >= self._config.llm_reranking_confidence_threshold:
            logger.debug(
                f"Skipping LLM reranking: cross-encoder gap {gap:.4f} >= "
                f"threshold {self._config.llm_reranking_confidence_threshold}"
            )
            return True
        if top_score >= self._config.llm_reranking_min_top_score and gap >= self._config.llm_reranking_decisive_gap:
            logger.debug(
                f"Skipping LLM reranking: decisive winner "
                f"(top={top_score:.4f} >= {self._config.llm_reranking_min_top_score}, "
                f"gap={gap:.4f} >= {self._config.llm_reranking_decisive_gap})"
            )
            return True
        return False

    def _evaluate_llm_rerank_gate(
        self,
        candidates: list[Any],
        temporal_signal: TemporalSignal | None,
        *,
        namespace_id: UUID | None,
    ) -> tuple[bool, str | None]:
        """Centralized gate for the LLM rerank step (issue #814).

        Consolidates the three independent skip reasons that historically
        lived inline in both the complex ``_vectorcypher_retrieve`` path and
        the simple ``_simple_retrieve`` path:

        - ``"not_temporal"``: ``enable_llm_reranking=True`` but the query
          isn't temporal. Documented behavior — never logs a warning.
        - ``"no_version_metadata"``: ``llm_reranking_mode="auto"`` (the
          default) and no chunk in the top candidates carries
          ``metadata["version"]``. Logs a one-time WARNING per
          ``(namespace_id, reason)`` tuple so users who opted in to LLM
          rerank discover why it isn't being invoked.
        - ``"decisive_winner"``: cross-encoder already produced a confident
          #1 (latency optimization — runs in both modes).

        ``candidates`` may be a list of ``FusedResult`` (complex path) or a
        list of ``(Chunk, score)`` tuples (simple path); both shapes are
        handled by ``_extract_candidate_metadata`` / ``_extract_top_scores``.

        Returns ``(should_run, skip_reason)``. ``skip_reason`` is ``None``
        only when ``should_run`` is ``True``.
        """
        if not self._config.enable_llm_reranking:
            # Caller already gated on this — defensive return so the helper
            # can be invoked unconditionally without paying the unused-skip
            # warning cost.
            return False, "not_temporal"

        if not (temporal_signal and temporal_signal.is_temporal):
            return False, "not_temporal"

        if not candidates:
            # Nothing to rerank — treat as a decisive-winner-style skip
            # (silent, no warning).
            return False, "decisive_winner"

        # Version-metadata gate is bypassed in "always" mode (issue #814).
        if self._config.llm_reranking_mode == "auto":
            has_versions = _candidates_have_versions(candidates[:10])
            if not has_versions:
                key = (namespace_id, "no_version_metadata")
                if key not in self._warned_rerank_skips:
                    self._warned_rerank_skips.add(key)
                    logger.warning(
                        "VectorCypher LLM reranking skipped for this namespace: "
                        "enable_llm_reranking=True but no chunks in the top "
                        "candidates carry metadata['version']. Set "
                        "llm_reranking_mode='always' to force LLM rerank on all "
                        "temporal queries, or set enable_llm_reranking=False to "
                        "suppress this warning."
                    )
                return False, "no_version_metadata"

        # Decisive-winner gate — applies in both auto and always modes.
        top_score, second_score = _extract_top_two_scores(candidates)
        if top_score is not None and second_score is not None:
            gap = top_score - second_score
            if self._should_skip_llm_rerank(top_score, gap):
                return False, "decisive_winner"

        return True, None

    async def _apply_llm_reranking(
        self,
        query: str,
        fused_results: list[FusedResult],
        limit: int,
        *,
        namespace_id: UUID,
    ) -> list[FusedResult]:
        """Apply LLM reranking to the top-N fused results for temporal queries.

        Similar to ``_apply_reranking`` but uses the LLM-based reranker which
        understands temporal context better than the cross-encoder.  Only the
        top ``llm_reranking_top_n`` candidates are sent to the LLM; the
        remainder is appended unchanged.

        Args:
            query: Original query text
            fused_results: Fused results (already cross-encoder reranked if enabled)
            limit: Final number of results to return

        Returns:
            Re-ordered FusedResult list
        """
        if not fused_results:
            return fused_results

        from khora.query.reranking import LLMReranker, RerankCandidate, hydrate_doc_titles

        top_n = min(self._config.llm_reranking_top_n, len(fused_results))
        candidates_to_rerank = fused_results[:top_n]
        remainder = fused_results[top_n:]

        # Normalize original scores to [0,1] for blending
        raw_scores = [r.rrf_score for r in candidates_to_rerank]
        score_min = min(raw_scores) if raw_scores else 0.0
        score_max = max(raw_scores) if raw_scores else 1.0
        score_range = score_max - score_min

        candidates = []
        for r in candidates_to_rerank:
            # Build content with temporal metadata prefix
            chunk_content = r.item.content if hasattr(r.item, "content") else str(r.item)
            if hasattr(r.item, "metadata") and r.item.metadata:
                meta = r.item.metadata
                raw = meta.custom if hasattr(meta, "custom") else (meta if isinstance(meta, dict) else {})
                if raw:
                    prefix_parts = []
                    session_id = raw.get("session_id") or raw.get("conversation_id")
                    if session_id:
                        prefix_parts.append(f"Session: {session_id}")
                    occurred_at = raw.get("occurred_at") or raw.get("source_timestamp")
                    if occurred_at:
                        prefix_parts.append(f"Date: {str(occurred_at)[:10]}")
                    if prefix_parts:
                        chunk_content = f"[{', '.join(prefix_parts)}] {chunk_content}"
            candidates.append(
                RerankCandidate(
                    item=r,
                    original_score=(r.rrf_score - score_min) / score_range if score_range > 1e-9 else 0.5,
                    content=chunk_content,
                    metadata=r.item.metadata if hasattr(r.item, "metadata") else {},
                    doc_title="",
                )
            )
        await hydrate_doc_titles(
            candidates,
            self._storage,
            lambda fr: getattr(getattr(fr, "item", None), "document_id", None),
            namespace_id=namespace_id,
        )

        try:
            async with self._llm_reranker_lock:
                if self._llm_reranker is None:
                    self._llm_reranker = LLMReranker(model=self._config.llm_reranking_model)
            results = await self._llm_reranker.rerank(query, candidates, top_k=top_n, blend_weight=0.7)

            reranked: list[FusedResult] = []
            for rr in results:
                fused = rr.item  # The original FusedResult
                fused.rrf_score = rr.final_score
                reranked.append(fused)

            reranked.extend(remainder)
            logger.debug(f"LLM reranking applied: {top_n} candidates scored for temporal query")
            return reranked[:limit]
        except Exception as e:
            logger.warning(f"LLM reranking failed, keeping current order: {e}")
            return fused_results

    async def _simple_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        limit: int,
        routing: RoutingDecision,
        *,
        effective_recency: float = 0.0,
        decay_days_override: int | None = None,
        temporal_sort: bool = False,
        recency_floor: float = 0.5,
        temporal_signal: TemporalSignal | None = None,
        min_similarity: float = 0.0,
        mode: SearchMode = SearchMode.HYBRID,
        hybrid_alpha_override: float | None = None,
        filter_ast: FilterNode | None = None,
        degradations: list[Degradation] | None = None,
    ) -> VectorCypherResult:
        """Simple retrieval path - vector search only.

        For SIMPLE-routed queries, uses a lower hybrid_alpha (0.5) to give
        BM25 equal weight — lexical overlap is stronger for factual queries.

        A per-call ``hybrid_alpha_override`` (threaded explicitly from the
        engine, #1116) wins over both the configured default and the
        SIMPLE-query clamp; ``None`` keeps the existing behaviour.

        When temporal_sort is True, results are re-sorted by occurred_at DESC
        after recency boosting so that the most recent chunks surface first
        (matches the graph-path behaviour for temporal categories).
        """
        with trace_span("khora.vectorcypher.simple_retrieve", namespace_id=str(namespace_id)) as span:
            # Per-call honest filter-pushdown plans for the simple (graph-less)
            # path: vector + optional BM25 only, no graph / recency channels.
            # Both channels GATE and both push the SAME ``khora_chunks`` WHERE.
            # A fresh per-call dict + sinks keep the report race-free under
            # concurrent recalls (no mutable instance state).
            filter_channel_plans: dict[str, ChannelPlan] = {}
            vector_filter_plan_sink: list[ChannelPlan] = []
            bm25_filter_plan_sink: list[ChannelPlan] = []

            # ADR-001 (#1158): carry any degradations the caller already recorded
            # (e.g. the entry-entity vector search that collapsed graph expansion
            # to this fallback) plus any this path records, so they surface on the
            # result. A fresh list keeps direct (non-fallback) callers race-free.
            degradations = degradations if degradations is not None else []

            # #833 channel-skip:
            #   VECTOR  -> pure vector store search, no internal BM25 fusion,
            #              no independent BM25 channel.
            #   KEYWORD -> skip vector store search entirely; results come
            #              solely from the BM25 channel.
            #   HYBRID / ALL -> legacy behaviour (vector + optional BM25).
            skip_vector_in_store = mode == SearchMode.KEYWORD
            run_bm25_channel = mode in (SearchMode.HYBRID, SearchMode.ALL, SearchMode.KEYWORD)

            # When BM25 channel is active, use pure vector (hybrid_alpha=1.0)
            # to avoid double-counting BM25 in both the vector blend and the
            # independent channel. Otherwise, lower alpha for SIMPLE queries
            # to boost the pgvector-internal BM25 signal.
            if mode == SearchMode.VECTOR:
                # Pure-vector: skip the pgvector-internal BM25 fusion entirely.
                effective_alpha = None
            elif self._lexical_channel_active() and run_bm25_channel:
                effective_alpha = 1.0
            else:
                # Per-call override (#1116) replaces the config read; the SIMPLE
                # clamp below still applies, matching the pre-fix behaviour
                # where the engine mutated the shared config before this read.
                effective_alpha = (
                    hybrid_alpha_override if hybrid_alpha_override is not None else self._config.hybrid_alpha
                )
                if routing.complexity == QueryComplexity.SIMPLE:
                    effective_alpha = min(effective_alpha, 0.5)

            # Launch BM25 search in parallel with vector search. KEYWORD mode
            # always launches BM25 (it is the only source of chunks).
            bm25_task: asyncio.Task[list[tuple[UUID, float, Chunk]]] | None = None
            if run_bm25_channel and self._storage and (self._lexical_channel_active() or mode == SearchMode.KEYWORD):
                bm25_task = asyncio.create_task(
                    self._lexical_search_chunks(
                        query=query,
                        namespace_id=namespace_id,
                        limit=self._config.bm25_top_k if mode != SearchMode.KEYWORD else max(limit, 50),
                        filter_ast=filter_ast,
                        filter_plan_out=bm25_filter_plan_sink,
                        degradations=degradations,
                    )
                )

            if skip_vector_in_store:
                results = []
            else:
                results = await self._vector_store.search(
                    namespace_id=namespace_id,
                    query_embedding=query_embedding,
                    limit=limit,
                    min_similarity=min_similarity,
                    temporal_filter=temporal_filter,
                    hybrid_alpha=effective_alpha,
                    query_text=query,
                    filter_ast=filter_ast,
                    filter_plan_out=vector_filter_plan_sink,
                )

            chunk_results: list[tuple[Chunk, float]] = []
            _max_raw_cosine = max((r.similarity for r in results), default=0.0)
            # Per-chunk raw vector cosine, captured pre-fusion. Used at exit to
            # report an ABSOLUTE relevance score instead of a per-result-set
            # min-max normalized one (#811); boosts/reranking only decide ORDER.
            _raw_cosine_by_id: dict[UUID, float] = {}
            for r in results:
                _raw_cosine_by_id[r.chunk.id] = r.similarity
                chunk = Chunk(
                    id=r.chunk.id,
                    namespace_id=r.chunk.namespace_id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    metadata={
                        "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                        **(r.chunk.metadata or {}),
                    },
                    chunker_info=r.chunk.chunker_info or {},
                    created_at=r.chunk.created_at or r.chunk.occurred_at,
                    # Carry the chunk event-time and the producer's verbatim
                    # time as distinct values; the recall projection applies
                    # the fallback.
                    occurred_at=r.chunk.occurred_at,
                    source_timestamp=r.chunk.source_timestamp,
                )
                chunk_results.append((chunk, r.combined_score or r.similarity))

            # Fuse with BM25 results if the channel is active
            simple_bm25_count = 0
            if bm25_task is not None:
                try:
                    bm25_results = await bm25_task
                except Exception as e:
                    logger.warning(f"BM25 channel failed in simple path: {e}")
                    bm25_results = []

                simple_bm25_count = len(bm25_results)
                if mode == SearchMode.KEYWORD:
                    # #833 KEYWORD: BM25-only - sole source of chunks. Skip
                    # the RRF fusion (there are no vector results to merge).
                    chunk_results = [(c, score) for _cid, score, c in bm25_results][:limit]
                elif bm25_results:
                    from khora.query.fusion import reciprocal_rank_fusion as _nlist_rrf

                    bm25_weight = self._config.bm25_weight
                    # Convert vector results to (item, score) format
                    ranked_lists: dict[str, list[tuple[Chunk, float]]] = {
                        "vector": [(c, s) for c, s in chunk_results],
                        "bm25": [(bm25_chunk, bm25_score) for _cid, bm25_score, bm25_chunk in bm25_results],
                    }
                    weights: dict[str, float] = {"vector": 1.0, "bm25": bm25_weight}
                    fused_raw = _nlist_rrf(
                        ranked_lists,
                        weights=weights,
                        k=self._config.rrf_k,
                        id_extractor=lambda c: c.id,
                    )
                    chunk_results = list(fused_raw[:limit])
                    logger.debug(
                        f"Simple path BM25 fusion: {len(bm25_results)} BM25 + "
                        f"{len(chunk_results)} vector -> {len(fused_raw)} fused"
                    )

            # Apply recency boost to simple path (was previously missing)
            if effective_recency > 0 and chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                with trace_span("khora.vectorcypher.recency_boost", chunk_count=len(fused)):
                    recency_scores = self._calculate_recency_scores(fused, decay_days_override=decay_days_override)
                    fused = apply_recency_boost(
                        fused, recency_scores, recency_weight=effective_recency, recency_floor=recency_floor
                    )
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            # Cross-encoder reranking (after recency boost, before version scoring)
            if self._config.enable_reranking and chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                with trace_span("khora.vectorcypher.reranking", candidate_count=len(fused)):
                    fused = await self._apply_reranking(query, fused, limit, namespace_id=namespace_id)
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            # LLM reranking of top-N for temporal queries (after cross-encoder).
            # Gating centralised in ``_evaluate_llm_rerank_gate`` (issue #814) —
            # see the parallel comment in ``_vectorcypher_retrieve``.
            if self._config.enable_llm_reranking and chunk_results:
                should_run, skip_reason = self._evaluate_llm_rerank_gate(
                    chunk_results,
                    temporal_signal,
                    namespace_id=namespace_id,
                )
                if not should_run and skip_reason is not None:
                    _LLM_RERANKING_SKIPPED_COUNTER.add(1, attributes={"reason": skip_reason})
                if should_run:
                    fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                    with trace_span(
                        "khora.vectorcypher.llm_reranking",
                        candidate_count=len(fused),
                        mode=self._config.llm_reranking_mode,
                    ):
                        fused = await self._apply_llm_reranking(query, fused, limit, namespace_id=namespace_id)
                    chunk_results = [(r.item, r.rrf_score) for r in fused]

            # Version-aware scoring — the FINAL score adjustment after ALL reranking.
            # CHANGE/ORDINAL excluded: need full version history.
            if (
                temporal_signal
                and temporal_signal.category
                in (
                    TemporalCategory.STATE_QUERY,
                    TemporalCategory.RECENCY,
                )
                and chunk_results
            ):
                from collections import defaultdict as _defaultdict

                _entity_versions: dict[str, int] = _defaultdict(int)
                _chunk_versions: dict[UUID, int] = {}

                for c, _s in chunk_results:
                    meta = c.metadata if isinstance(c.metadata, dict) else {}
                    if isinstance(meta, dict):
                        version = meta.get("version") or meta.get("entity_version", 0)
                        if version:
                            _chunk_versions[c.id] = int(version)
                            for ref in meta.get("entity_refs") or []:
                                _entity_versions[ref] = max(_entity_versions[ref], int(version))

                if _entity_versions:
                    _VERSION_DECAY = 0.7  # Stronger penalty: v1/v5 → 0.44 (was 0.6 with 0.5)
                    updated = []
                    for c, s in chunk_results:
                        v = _chunk_versions.get(c.id, 0)
                        if v > 0:
                            meta = c.metadata if isinstance(c.metadata, dict) else {}
                            if isinstance(meta, dict):
                                for ref in meta.get("entity_refs") or []:
                                    max_v = _entity_versions.get(ref, v)
                                    if max_v > v:
                                        ratio = v / max_v
                                        s *= 1.0 - _VERSION_DECAY * (1.0 - ratio)
                                        break
                        updated.append((c, s))
                    chunk_results = updated

            # Apply temporal sort: re-order by occurred_at DESC so the most
            # recent chunks rank first. This mirrors the graph path's
            # temporal_sort and is critical for STATE_QUERY/RECENCY/CHANGE.
            #
            # Skip when cross-encoder reranking is active: the reranker already
            # captures semantic relevance order, and the recency boost (above)
            # provides temporal discrimination. A hard re-sort by timestamp
            # would override the reranker's carefully computed ranking.
            if temporal_sort and chunk_results and not self._config.enable_reranking:
                from datetime import datetime as _dt

                # Normalize every key to tz-aware UTC: metadata occurred_at
                # may be a user-supplied naive ISO string (it overrides the
                # column value in the metadata dict), created_at is tz-aware
                # from the DB, and the sentinel must match. Mixing naive and
                # aware datetimes in one sort raises TypeError (#1115). Same
                # normalization as _calculate_recency_scores.
                _ts_sentinel = _dt.min.replace(tzinfo=UTC)

                def _ts(pair: tuple[Chunk, float]) -> _dt:
                    occ = (pair[0].metadata or {}).get("occurred_at") if isinstance(pair[0].metadata, dict) else None
                    if occ:
                        try:
                            parsed = _dt.fromisoformat(occ.replace("Z", "+00:00"))
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=UTC)
                            return parsed
                        except (ValueError, TypeError, AttributeError):
                            pass
                    created = pair[0].created_at
                    if created is None:
                        return _ts_sentinel
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    return created

                # ORDINAL queries ("first", "which came earlier") need ascending
                # order; all other temporal categories use descending (most recent first).
                sort_descending = True
                if temporal_signal and temporal_signal.category == TemporalCategory.ORDINAL:
                    sort_descending = False

                chunk_results.sort(key=_ts, reverse=sort_descending)

            # Report an ABSOLUTE relevance score (raw vector cosine) per chunk
            # instead of a per-result-set min-max normalized one (#811). Order is
            # unchanged - it was already decided by the boost/rerank-adjusted
            # score above. BM25-only chunks have no cosine; keep their current
            # score rather than rescaling the set.
            if chunk_results:
                chunk_results = [(c, _raw_cosine_by_id.get(c.id, s)) for c, s in chunk_results]

            span.set_attribute("chunk_count", len(chunk_results))

            # All chunks come from vector search in simple mode
            all_ids = [str(c.id) for c, _ in chunk_results]
            search_methods = {
                "chunk_overlap": {
                    "vector_only": {"ids": all_ids, "count": len(all_ids)},
                    "graph_only": {"ids": [], "count": 0},
                    "vector_and_graph": {"ids": [], "count": 0},
                },
                "by_method": {
                    "vector": {"chunk_ids": all_ids, "count": len(all_ids)},
                    "graph": {"chunk_ids": [], "count": 0},
                },
            }

            # #833: honest channel-count reporting. KEYWORD mode has 0
            # vector chunks; VECTOR mode has 0 BM25 chunks regardless of
            # whether BM25 was queried.
            if mode == SearchMode.KEYWORD:
                reported_vector_count = 0
                reported_bm25_count = simple_bm25_count
            elif mode == SearchMode.VECTOR:
                reported_vector_count = len(results)
                reported_bm25_count = 0
            else:
                reported_vector_count = len(results)
                reported_bm25_count = simple_bm25_count

            # #857: project entities + relationships from storage when the
            # simple (graph-less) path produced chunks. Prior to this fix the
            # return value hardcoded ``entities=[]`` and ``relationships=[]``,
            # so backends without a Neo4j driver (sqlite_lance, surrealdb,
            # postgres-only) surfaced empty entity lists from ``recall()``
            # even when the graph was populated. We fetch all entities /
            # relationships in the namespace (capped at 1000) and filter by
            # overlap with the recalled chunk ids.
            # TODO: replace the namespace-wide list + Python filter with a
            #       per-backend ``list_entities_by_chunk_ids`` query method
            #       once it lands across sqlite_lance / surrealdb / pgvector
            #       (perf follow-up to #857).
            entities_with_scores: list[tuple[Entity, float]] = []
            relationships_with_scores: list[tuple[Relationship, float]] = []
            if chunk_results and self._storage is not None:
                recalled_chunk_ids = {c.id for c, _ in chunk_results}
                try:
                    all_entities = await self._storage.list_entities(namespace_id, limit=1000)
                except Exception as e:
                    logger.warning(f"#857 simple-path entity projection failed: {e}")
                    all_entities = []
                for entity in all_entities:
                    src_chunks = entity.source_chunk_ids or []
                    overlap = sum(1 for cid in src_chunks if cid in recalled_chunk_ids)
                    if overlap > 0:
                        # Score by mention overlap fraction - entities mentioned
                        # in more recalled chunks rank higher.
                        score = overlap / max(1, len(src_chunks))
                        entities_with_scores.append((entity, score))

                if entities_with_scores:
                    try:
                        all_relationships = await self._storage.list_relationships(namespace_id, limit=1000)
                    except Exception as e:
                        logger.warning(f"#857 simple-path relationship projection failed: {e}")
                        all_relationships = []
                    recalled_entity_ids = {e.id for e, _ in entities_with_scores}
                    for rel in all_relationships:
                        if rel.source_entity_id in recalled_entity_ids and rel.target_entity_id in recalled_entity_ids:
                            relationships_with_scores.append((rel, 1.0))

            # Honest per-channel filter-pushdown plans for the simple path. Each
            # plan is the one its channel appended to the sink from the BACKEND's
            # OWN compile — all-pushed on a raise-mode backend (pgvector /
            # surrealdb), a pushed/post-filtered split on the split-mode
            # sqlite_lance backend (which re-checks the residual in memory). The
            # vector store and BM25's ``search_fulltext`` compile the same WHERE
            # for a given backend, so the two channels report the same split.
            # (BM25 records only when the temporal-store path ran, regardless of
            # row count.)
            if filter_ast is not None:
                if not skip_vector_in_store and vector_filter_plan_sink:
                    filter_channel_plans["vector"] = vector_filter_plan_sink[0]
                if bm25_filter_plan_sink:
                    filter_channel_plans["bm25"] = bm25_filter_plan_sink[0]

            return VectorCypherResult(
                chunks=chunk_results,
                entities=entities_with_scores,
                relationships=relationships_with_scores,
                routing_decision=routing,
                metadata={
                    "search_mode": "simple_vector" if not simple_bm25_count else "simple_vector_bm25",
                    "routing_confidence": routing.confidence,
                    "vector_chunk_count": reported_vector_count,
                    "graph_chunk_count": 0,
                    "bm25_chunk_count": reported_bm25_count,
                    "effective_recency": effective_recency,
                    "temporal_sort": temporal_sort,
                    # Max raw cosine similarity (pre-fusion) for abstention system
                    "max_raw_vector_score": _max_raw_cosine,
                    # Search provenance: all chunks from vector in simple mode
                    "search_methods": search_methods,
                    # ADR-001 (#1158): silent-failure entries from this path
                    # (BM25 channel) plus any carried in from a graph-path
                    # fallback (entry-entity vector search). Forwarded onto
                    # ``RecallResult.engine_info["degradations"]`` by the engine.
                    "degradations": degradations,
                    # Private carrier (popped by the engine before the public
                    # spread): per-channel honest filter-pushdown plans.
                    "_filter_channel_plans": filter_channel_plans,
                },
            )

    async def _vector_search_entities(
        self,
        query_embedding: list[float],
        namespace_id: UUID,
        limit: int,
        *,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Search for entry entities using vector similarity via pgvector HNSW.

        Args:
            degradations: When provided, a structured ``Degradation`` is appended
                if the search raises so the silently-collapsed graph-expansion
                channel is observable (ADR-001, issue #1158). Recall continues
                with no entry seeds (vector-only fallback) rather than crashing.
        """
        if not self._storage:
            logger.warning("Storage coordinator not available for entity vector search")
            return []

        with trace_span("khora.vectorcypher.vector_search_entities", namespace_id=str(namespace_id)) as span:
            try:
                results = await self._storage.search_similar_entities(
                    namespace_id,
                    query_embedding,
                    limit=limit,
                    min_similarity=self._config.min_entity_similarity,
                )
                span.set_attribute("entity_count", len(results))
                return results
            except Exception as e:
                # ADR-001 (issue #1158): entry-entity discovery failing here
                # collapses graph expansion to vector-only. Record a structured
                # Degradation so the dropped channel is observable rather than
                # silent (matches the rel_fetch / cypher_expand convention).
                logger.warning(f"Entity vector search failed: {e}", exc_info=True)
                _ENTITY_VECTOR_SEARCH_DEGRADED_COUNTER.add(1, attributes={"reason": "channel_exception"})
                if degradations is not None:
                    degradations.append(
                        Degradation(
                            component="vectorcypher.entity_vector_search",
                            reason="channel_exception",
                            detail=str(e)[:200] or None,
                            exception=type(e).__name__,
                        )
                    )
                return []

    async def _cypher_expand(
        self,
        entry_entity_ids: list[UUID],
        namespace_id: UUID,
        depth: int,
        *,
        prefer_current: bool = False,
        degradations: list[Degradation] | None = None,
    ) -> tuple[dict[UUID, float], dict[str, dict[str, str]]]:
        """Expand entry entities to find related entities via graph traversal.

        Args:
            entry_entity_ids: Starting entity IDs
            namespace_id: Namespace constraint
            depth: Maximum traversal depth
            prefer_current: When True, filter out expired entities (for temporal queries)
            degradations: When provided, a structured ``Degradation`` is appended
                if a neighborhood entry has an unrecognized shape (neither an
                ``Entity`` domain object nor a dict) so the dropped expansion
                hop is observable.

        Returns:
            Tuple of:
            - Dict mapping entity_id -> relevance score
            - Dict mapping entity_id_str -> {name, entity_type} for all discovered entities
        """
        if not entry_entity_ids:
            return {}, {}

        with trace_span("khora.vectorcypher.cypher_expand", entry_count=len(entry_entity_ids), depth=depth) as span:
            depth = min(max(1, depth), self._config.max_depth)

            # Get neighborhoods from dual node manager (Neo4j) or storage coordinator (SurrealDB)
            if self._dual_nodes is not None:
                neighborhoods = await self._dual_nodes.get_entity_neighborhoods(
                    entity_ids=entry_entity_ids,
                    namespace_id=namespace_id,
                    depth=depth,
                    limit_per_entity=20,
                    prefer_current=prefer_current,
                )
            elif self._storage and self._storage._graph:
                raw_neighborhoods = await self._storage.get_neighborhoods_batch(
                    entry_entity_ids,
                    namespace_id=namespace_id,
                    depth=depth,
                    limit_per_entity=20,
                    prefer_current=prefer_current,
                )
                # Normalize: get_neighborhoods_batch returns
                # {UUID: {"entities": [...], "relationships": [...]}}
                # but the scoring loop expects {UUID: [{"id":..., "distance":..., ...}]}.
                # The embedded sqlite_lance backend returns ``Entity`` domain
                # objects (not dicts), so map those to the dict shape the scoring
                # loop reads before the ``isinstance(..., dict)`` check.
                neighborhoods = {}
                for eid, data in raw_neighborhoods.items():
                    entities_list = data.get("entities", []) if isinstance(data, dict) else data
                    normalized = []
                    for i, entity_data in enumerate(entities_list if isinstance(entities_list, list) else []):
                        if isinstance(entity_data, Entity):
                            mapped: dict[str, Any] = {
                                "id": entity_data.id,
                                "name": entity_data.name,
                                "entity_type": entity_data.entity_type,
                                "description": entity_data.description,
                                "source_tool": entity_data.source_tool,
                            }
                            entity_data = mapped
                        if isinstance(entity_data, dict):
                            entity_data.setdefault("distance", i + 1)
                            normalized.append(entity_data)
                        else:
                            # Unrecognized neighborhood-entry shape (neither an
                            # Entity domain object nor a dict): log + count + record
                            # the dropped expansion hop so an unexpectedly empty
                            # graph-channel recall is observable rather than silent
                            # (matches the version_filter / rel_fetch convention).
                            logger.warning(
                                f"Dropping unrecognized neighborhood entry of type "
                                f"{type(entity_data).__name__} in _cypher_expand; "
                                f"graph-expansion hop skipped."
                            )
                            _CYPHER_EXPAND_DEGRADED_COUNTER.add(
                                1, attributes={"reason": "unrecognized_neighborhood_shape"}
                            )
                            if degradations is not None:
                                degradations.append(
                                    Degradation(
                                        component="vectorcypher.cypher_expand",
                                        reason="unrecognized_neighborhood_shape",
                                        detail=f"dropped neighborhood entry of type {type(entity_data).__name__}",
                                    )
                                )
                    neighborhoods[eid] = normalized
            else:
                neighborhoods = {}

            # Score entities by distance from entry points and collect entity info
            entity_scores: dict[UUID, float] = {}
            entity_info_map: dict[str, dict[str, str]] = {}

            for source_id, related in neighborhoods.items():
                for entity_info in related:
                    # Handle both bare UUIDs (Neo4j) and record IDs like "entity:⟨uuid⟩" (SurrealDB)
                    raw_id = entity_info["id"]
                    try:
                        entity_id = UUID(str(raw_id)) if not isinstance(raw_id, UUID) else raw_id
                    except ValueError:
                        # SurrealDB record ID — extract UUID from "table:⟨uuid⟩"
                        import re

                        m = re.search(r"[0-9a-fA-F\-]{36}", str(raw_id))
                        entity_id = UUID(m.group(0)) if m else UUID(int=0)
                    distance = entity_info.get("distance", 1)
                    # Score decreases with distance
                    score = 1.0 / (1 + distance)

                    if entity_id in entity_scores:
                        # Take max score if entity reached multiple ways
                        entity_scores[entity_id] = max(entity_scores[entity_id], score)
                    else:
                        entity_scores[entity_id] = score

                    # Capture name, type, description, source_tool (zero-cost, data already fetched)
                    eid_str = str(entity_id)
                    if eid_str not in entity_info_map:
                        entity_info_map[eid_str] = {
                            "name": entity_info.get("name", ""),
                            "entity_type": entity_info.get("entity_type", ""),
                            "description": entity_info.get("description", ""),
                            "source_tool": entity_info.get("source_tool", ""),
                        }

            span.set_attribute("expanded_entity_count", len(entity_scores))
            return entity_scores, entity_info_map

    @staticmethod
    def _decompose_change_query(query: str) -> str | None:
        """Decompose a CHANGE query into a current-state sub-query.

        Rewrites temporal-change phrasing into a present-tense question about
        the entity's current state, so a second vector search retrieves
        up-to-date evidence alongside the historical evidence from the
        original query.

        Examples:
            "What did Alice used to play?" → "What does Alice play now?"
            "Does she still work at Google?" → "Where does she work now?"
            "He switched from piano to guitar" → "What instrument does he play now?"
        """
        import re

        q = query.strip()
        ql = q.lower()

        # Pattern: "used to <verb>" → "currently <verb>"
        m = re.search(r"(\w+)\s+used\s+to\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            rest = m.group(2).rstrip("? .")
            return f"What does {subject} {rest} now?"

        # Pattern: "still <verb>" → current state question
        m = re.search(r"(?:does|do|is)\s+(\w+)\s+still\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            rest = m.group(2).rstrip("? .")
            return f"What does {subject} {rest} now?"

        # Pattern: "switched from X to Y" / "changed from X to Y"
        m = re.search(r"(\w+)\s+(?:switched|changed|moved|transitioned)\s+(?:from\s+.+?\s+)?to\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            new_state = m.group(2).rstrip("? .")
            return f"What is {subject} {new_state} now?"

        # Pattern: "no longer" → ask about current state
        m = re.search(r"(\w+)\s+(?:is|was)\s+no\s+longer\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            old_state = m.group(2).rstrip("? .")
            return f"What is {subject} doing instead of {old_state}?"

        # Fallback: prepend "currently" to make it present-focused
        if any(kw in ql for kw in ("used to", "still", "previously", "before", "changed", "switched")):
            # Strip common change keywords and add "currently"
            cleaned = re.sub(
                r"\b(used to|still|previously|formerly|no longer)\b",
                "currently",
                ql,
                count=1,
            )
            return cleaned.strip()

        return None

    async def _version_filter_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        target_date: datetime,
        degradations: list[Degradation] | None = None,
    ) -> list[UUID]:
        """Filter entities to those valid at a specific point in time (bi-temporal).

        Uses ``version_valid_from`` / ``version_valid_to`` properties on
        Entity nodes.  Entities without version properties are treated as
        always-valid (backward-compatible).

        Also checks :EntityVersion snapshot nodes reachable via SUPERSEDES
        edges, returning the snapshot ID when the current entity was not
        yet valid at the target date but a prior version was.

        Args:
            entity_ids: Candidate entity IDs
            namespace_id: Namespace constraint
            target_date: The point-in-time to query for
            degradations: When provided, a structured ``Degradation`` is appended
                on the graph-less no-op path (see below) so callers can detect
                that point-in-time entity-version narrowing was skipped.

        Returns:
            Filtered list of entity IDs (may include EntityVersion IDs)
            that were valid at ``target_date``
        """
        if not entity_ids:
            return []

        # Graph-less backends (sqlite_lance / surrealdb) have no Neo4j driver and
        # no version_valid_from/to columns, so point-in-time entity-version
        # narrowing cannot run — return entities unfiltered (current-state).
        # This is the genuine no-op site: record a structured degradation here
        # (condition-driven, no exc_info) so the skip is observable. A plain
        # occurred-bounds recall never reaches this method, so it stays
        # degradation-free; only a parsed target_date lands here.
        if self._neo4j_driver is None:
            logger.warning(
                "Entity-version filtering is unavailable on this backend (no graph "
                "driver / version columns); recall continues with occurred-bounds "
                "chunk filtering only (no point-in-time entity versioning)."
            )
            if degradations is not None:
                degradations.append(
                    Degradation(
                        component="vectorcypher.version_filter",
                        reason="embedded_no_version_columns",
                        detail=(
                            "sqlite_lance lacks version_valid_from/to; point-in-time "
                            "entity-version filtering skipped — returning current-state entities"
                        ),
                    )
                )
            _VERSION_FILTER_DEGRADED_COUNTER.add(1, attributes={"reason": "embedded_no_version_columns"})
            return list(entity_ids)

        # First: keep current Entity nodes that are valid at target_date,
        # OR that have no version properties (backward-compatible).
        # Second: for entities not valid at target_date, check if a prior
        # EntityVersion was valid via SUPERSEDES edges.
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {id: eid, namespace_id: $namespace_id})
        OPTIONAL MATCH (e)-[:SUPERSEDES]->(ev:EntityVersion)
        WHERE ev.namespace_id = $namespace_id
          AND (ev.version_valid_from IS NULL OR ev.version_valid_from <= $target_date)
          AND (ev.version_valid_to IS NULL OR ev.version_valid_to > $target_date)
        WITH e, collect(ev.id) AS version_ids
        WITH e, version_ids,
             CASE
               WHEN e.version_valid_from IS NULL THEN true
               WHEN e.version_valid_from <= $target_date
                    AND (e.version_valid_to IS NULL OR e.version_valid_to > $target_date)
               THEN true
               ELSE false
             END AS current_valid
        WHERE current_valid OR size(version_ids) > 0
        RETURN CASE WHEN current_valid THEN e.id
                    ELSE version_ids[0]
               END AS id
        """

        async with self._dual_nodes._session() as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=[str(eid) for eid in entity_ids],
                    namespace_id=str(namespace_id),
                    target_date=target_date.isoformat(),
                )
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        filtered = [UUID(r["id"]) for r in records if r["id"]]
        logger.debug(
            f"Version filter at {target_date.isoformat()}: {len(entity_ids)} candidates -> {len(filtered)} valid"
        )
        return filtered

    async def _fetch_version_history(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
    ) -> list[dict[str, Any]]:
        """Traverse SUPERSEDES edges to retrieve version history for entities.

        Used for CHANGE-category temporal queries ("what did X used to be?",
        "how has Y changed?").

        Args:
            entity_ids: Entity IDs to get version history for
            namespace_id: Namespace constraint

        Returns:
            List of dicts with ``current_*`` and ``previous_*`` fields
            representing the version transition chain.
        """
        if not entity_ids or self._neo4j_driver is None:
            return []

        query = """
        UNWIND $entity_ids AS eid
        MATCH (current:Entity {id: eid, namespace_id: $namespace_id})
        OPTIONAL MATCH (current)-[s:SUPERSEDES]->(prev:EntityVersion)
        RETURN current.id AS current_id,
               current.name AS name,
               current.entity_type AS entity_type,
               current.attributes AS current_attributes,
               current.version_valid_from AS current_valid_from,
               current.version_valid_to AS current_valid_to,
               prev.id AS previous_id,
               prev.attributes AS previous_attributes,
               prev.version_valid_from AS previous_valid_from,
               prev.version_valid_to AS previous_valid_to,
               s.superseded_at AS superseded_at
        ORDER BY current.name, s.superseded_at DESC
        """

        async with self._dual_nodes._session() as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=[str(eid) for eid in entity_ids],
                    namespace_id=str(namespace_id),
                )
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        logger.debug(f"Version history: {len(records)} version records for {len(entity_ids)} entities")
        return records

    async def _fetch_chunks_from_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        limit: int,
        *,
        temporal_sort: bool = False,
        prefer_current: bool = False,
        filter_ast: FilterNode | None = None,
        graph_pushed_keys_out: list[frozenset[str]] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Fetch chunks connected to entities via MENTIONED_IN.

        Args:
            entity_ids: Entity IDs to fetch chunks for
            namespace_id: Namespace constraint
            temporal_filter: Optional temporal constraints
            limit: Maximum chunks to return
            temporal_sort: If True, sort by occurred_at DESC (for temporal queries)
            prefer_current: When True, filter out expired entities
            filter_ast: Canonical recall-filter AST. The system-key slice is
                pushed down into the Cypher chunk query; metadata leaves are
                left for the engine's in-memory post-filter.
            graph_pushed_keys_out: Optional sink forwarded to the Neo4j
                ``get_chunks_by_entities`` call; receives the consumed keys of
                the compile actually spliced into the executed ``WHERE``. The
                SurrealDB storage-fallback and empty branches never touch it, so
                they leave it empty (nothing pushed).

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        with trace_span(
            "khora.vectorcypher.fetch_entity_chunks",
            entity_count=len(entity_ids),
            namespace_id=str(namespace_id),
        ) as span:
            if self._dual_nodes is not None:
                chunk_records = await self._dual_nodes.get_chunks_by_entities(
                    entity_ids=entity_ids,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    temporal_sort=temporal_sort,
                    prefer_current=prefer_current,
                    limit=limit,
                    filter_ast=filter_ast,
                    pushed_keys_out=graph_pushed_keys_out,
                )
            elif self._storage:
                # SurrealDB fallback: get chunks via entity source_chunk_ids
                chunk_records = []
                try:
                    entities_map = await self._storage.get_entities_batch(entity_ids, namespace_id=namespace_id)
                    all_chunk_ids: list[UUID] = []
                    for entity in entities_map.values():
                        all_chunk_ids.extend(entity.source_chunk_ids[:5])
                    if all_chunk_ids:
                        chunks_map = await self._storage.get_chunks_batch(all_chunk_ids, namespace_id=namespace_id)
                        for cid, chunk in chunks_map.items():
                            chunk_records.append(
                                {
                                    "chunk_id": str(cid),
                                    # document_id is consumed by the result-building
                                    # loop below (UUID(record["document_id"])).
                                    # Forgetting to include it here makes the
                                    # fallback KeyError out and crash any recall
                                    # query routed through this channel on a
                                    # SurrealDB-only deployment (#754).
                                    "document_id": str(chunk.document_id),
                                    "content": chunk.content,
                                    "embedding": chunk.embedding,
                                    "total_mentions": 1,
                                    "entity_ids": [],
                                    "occurred_at": getattr(chunk, "source_timestamp", None),
                                    "chunker_info": getattr(chunk, "chunker_info", None) or {},
                                }
                            )
                except Exception as e:
                    logger.warning(f"SurrealDB chunk fetch fallback failed: {e}")
                    chunk_records = []
            else:
                chunk_records = []

            results: list[tuple[UUID, float, Chunk]] = []
            for record in chunk_records:
                chunk_id = UUID(record["chunk_id"])
                # Score based on mention count and entity coverage
                score = float(record.get("total_mentions", 1))
                entity_count = len(record.get("entity_ids", []))
                score = score * (1 + 0.1 * entity_count)  # Boost for multiple entity connections

                chunk = Chunk(
                    id=chunk_id,
                    namespace_id=namespace_id,
                    document_id=UUID(record["document_id"]),
                    content=record["content"],
                    metadata={
                        "occurred_at": record.get("occurred_at"),
                        "connected_entities": record.get("entity_ids", []),
                        **(record.get("metadata") or {}),
                    },
                    chunker_info=_decode_chunker_info(record.get("chunker_info")),
                    # The graph store carries only the chunk event-time;
                    # surface it as occurred_at so the recall projection uses
                    # it. record["occurred_at"] is an ISO-8601 string from
                    # Neo4j or a datetime from the SurrealDB fallback above.
                    # source_timestamp mirrors it as the projection fallback
                    # carrier.
                    occurred_at=_coerce_occurred_at(record.get("occurred_at")),
                    source_timestamp=_coerce_occurred_at(record.get("occurred_at")),
                )
                results.append((chunk_id, score, chunk))

            span.set_attribute("chunk_count", len(results))
            return results

    async def _vector_search_chunks(
        self,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        query_text: str,
        limit: int,
        *,
        hybrid_alpha_override: float | None = None,
        min_similarity: float = 0.0,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Direct vector search on chunks via pgvector.

        Args:
            query_embedding: Query embedding
            namespace_id: Namespace to search
            temporal_filter: Temporal constraints
            query_text: Original query text for hybrid search
            limit: Maximum results
            hybrid_alpha_override: If set, overrides the configured hybrid_alpha.
                                   Used to force pure vector (1.0) when the BM25
                                   channel is active to avoid double-counting.
            min_similarity: Per-call cosine-similarity floor applied at the
                storage layer. Forwarded to ``TemporalVectorStore.search``.
            filter_ast: Canonical recall-filter AST. Forwarded to
                ``TemporalVectorStore.search``, where the pgvector backend
                compiles it to a ``khora_chunks`` WHERE predicate.
            filter_plan_out: Optional per-call sink for the honest
                filter-pushdown plan. The vector store appends the
                ``ChannelPlan`` it built from the SAME compile this search ran
                (no re-compile, no backend-name check). A fresh per-call list
                keeps the report race-free under concurrent recalls.

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        effective_alpha = hybrid_alpha_override if hybrid_alpha_override is not None else self._config.hybrid_alpha
        with trace_span("khora.vectorcypher.vector_search_chunks", namespace_id=str(namespace_id)) as span:
            results = await self._vector_store.search(
                namespace_id=namespace_id,
                query_embedding=query_embedding,
                limit=limit,
                min_similarity=min_similarity,
                temporal_filter=temporal_filter,
                hybrid_alpha=effective_alpha,
                query_text=query_text,
                filter_ast=filter_ast,
                filter_plan_out=filter_plan_out,
            )

            span.set_attribute("chunk_count", len(results))
            return [
                (
                    r.chunk.id,
                    r.combined_score or r.similarity,
                    Chunk(
                        id=r.chunk.id,
                        namespace_id=r.chunk.namespace_id,
                        document_id=r.chunk.document_id,
                        content=r.chunk.content,
                        metadata={
                            "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                            **(r.chunk.metadata or {}),
                        },
                        chunker_info=r.chunk.chunker_info or {},
                        created_at=r.chunk.created_at or r.chunk.occurred_at,
                        occurred_at=r.chunk.occurred_at,
                        source_timestamp=r.chunk.source_timestamp,
                    ),
                )
                for r in results
            ]

    async def _recency_channel_chunks(
        self,
        *,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: ChunkTemporalFilter | None,
        filter_ast: FilterNode | None = None,
        filter_channel_plans: dict[str, ChannelPlan] | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Issue #567 A3 — pure-recency candidate pool.

        Pulls the N most-recent chunks in the namespace via the temporal
        store's ``search_recent_chunks`` (on pgvector this reads the
        ``khora_chunks`` table, served by the ``ix_khora_chunks_ns_recency``
        index), then computes per-chunk cosine similarity against
        ``query_embedding`` and drops anything below
        ``temporal_query_relevance_floor``. Pool augmentation only — the
        caller merges results into the existing vector pool.

        ``filter_ast``: the caller recall-filter AST. It is pushed into the
        recency SQL — the store compiles it to the SAME raise-mode
        ``khora_chunks`` WHERE predicate the vector path uses, so no
        filter-violating chunk is ever fetched. (No in-memory post-filter:
        the prior implementation rebuilt each chunk into a provenance-blank
        public ``Chunk`` and could not enforce provenance leaves — GitHub
        issue #1223.)

        ``filter_channel_plans``: when provided, the honest recency ChannelPlan
        (handed back by the store via the per-call plan sink) is recorded here —
        but ONLY under a caller filter (``filter_ast is not None``, symmetric with
        the vector / BM25 channels), on the real execution path, and only when the
        channel actually produced surviving chunks that will gate in RRF. Because
        the SQL pushes every leaf, the plan credits each as pushed. The
        early-returns above (backend without the capability, no rows, no
        embeddings) record nothing, so a channel that never produced a gating
        result is never credited with a disposition.

        ``degradations``: when provided, a structured ``Degradation`` is appended
        if ``search_recent_chunks`` raises an operational fault (DB / network), so
        the silently-dropped pool-augmentation channel is observable (ADR-001).
        A ``RecallFilterUnsupportedError`` is NOT a degradation — it propagates
        (fail-loud), matching the vector channel's contract.

        Returns ``list[tuple[chunk_id, score, Chunk]]`` matching the
        shape of ``_vector_search_chunks`` so RRF can fuse it directly.
        """
        if self._vector_store is None or not hasattr(self._vector_store, "search_recent_chunks"):
            return []

        with trace_span(
            "khora.vectorcypher.recency_channel",
            namespace_id=str(namespace_id),
            limit=self._config.temporal_recency_channel_limit,
        ) as span:
            recency_plan_sink: list[ChannelPlan] = []
            try:
                recent_tuples = await self._vector_store.search_recent_chunks(
                    namespace_id=namespace_id,
                    limit=self._config.temporal_recency_channel_limit,
                    created_after=getattr(temporal_filter, "occurred_after", None),
                    filter_ast=filter_ast,
                    filter_plan_out=recency_plan_sink,
                )
            except RecallFilterUnsupportedError:
                # A filter leaf the khora_chunks compiler cannot push is a hard
                # determinism error, not a degradable runtime fault. The vector
                # channel runs the SAME raise-mode compile and lets it propagate
                # (it is never caught) — the recency channel must match that
                # fail-loud contract rather than silently dropping the channel
                # and masking a filter that was never enforced.
                raise
            except Exception as exc:
                # Operational fault (DB / network). The recency channel is pool
                # augmentation only, so degrade to [] — the vector + BM25
                # channels still enforce the filter and carry the report. ADR-001:
                # record a structured Degradation + bump the counter so the
                # silently-dropped channel is observable (matches the bm25 /
                # rel_fetch / cypher_expand convention).
                logger.warning("Recency channel SQL failed: {}", exc, exc_info=True)
                span.set_attribute("error", str(exc)[:200])
                _RECENCY_DEGRADED_COUNTER.add(1, attributes={"reason": "channel_exception"})
                if degradations is not None:
                    degradations.append(
                        Degradation(
                            component="vectorcypher.recency_channel",
                            reason="channel_exception",
                            detail=str(exc)[:200] or None,
                            exception=type(exc).__name__,
                        )
                    )
                return []

            if not recent_tuples:
                span.set_attribute("raw_count", 0)
                return []

            # Filter by cosine relevance against the query embedding.
            # Devil's-Advocate demand #3: pure-rank fusion without a
            # relevance gate would let today's irrelevant chunks muscle
            # into top-K. Drop anything below the configured floor.
            chunks_with_embedding = [(chunk, getattr(chunk, "embedding", None)) for chunk, _ in recent_tuples]
            chunks_with_embedding = [(c, e) for c, e in chunks_with_embedding if e is not None]
            if not chunks_with_embedding:
                span.set_attribute("raw_count", len(recent_tuples))
                span.set_attribute("filtered_count", 0)
                return []

            from khora._accel import batch_cosine_similarity

            floor = self._config.temporal_query_relevance_floor
            # batch_cosine_similarity returns list[(index, similarity)]
            # already filtered by ``threshold``. We pass the floor directly
            # so the floor gate happens inside Rust/NumPy, not in Python.
            sim_pairs = batch_cosine_similarity(
                query_embedding,
                [emb for _, emb in chunks_with_embedding],
                threshold=floor,
            )
            filtered: list[tuple[UUID, float, Chunk]] = []
            for idx, sim in sim_pairs:
                chunk, _emb = chunks_with_embedding[idx]
                # Re-shape into ``(chunk_id, score, Chunk)`` matching
                # ``_vector_search_chunks``. The Chunk's metadata dict
                # carries ``occurred_at`` for the downstream recency
                # boost (RRF only uses rank position, so the score is
                # informational).
                filtered.append(
                    (
                        chunk.id,
                        float(sim),
                        Chunk(
                            id=chunk.id,
                            namespace_id=chunk.namespace_id,
                            document_id=chunk.document_id,
                            content=chunk.content,
                            metadata={
                                "occurred_at": (
                                    chunk.occurred_at.isoformat() if getattr(chunk, "occurred_at", None) else None
                                ),
                                **(getattr(chunk, "metadata", None) or {}),
                            },
                            chunker_info=getattr(chunk, "chunker_info", None) or {},
                            created_at=getattr(chunk, "created_at", None) or getattr(chunk, "occurred_at", None),
                            # The temporal store carries the chunk event-time
                            # (occurred_at, surfaced into metadata above) and the
                            # producer time (source_timestamp); the projection
                            # uses event-time first, then producer time.
                            source_timestamp=getattr(chunk, "source_timestamp", None),
                        ),
                    )
                )

            # Record the honest recency ChannelPlan from the SQL pushdown the
            # store performed (``recency_plan_sink[0]``). The recency SQL now
            # compiles ``filter_ast`` into the ``khora_chunks`` WHERE clause —
            # the SAME raise-mode compile the vector path uses — so every leaf
            # is pushed and no filter-violating chunk reaches this point. Record
            # only when there IS a caller filter (symmetric with the vector /
            # BM25 channels, which the caller gates on ``filter_ast is not None``)
            # AND the store reported a plan AND the channel produced surviving
            # chunks that will GATE in RRF: a no-filter recall, or a channel that
            # survived to here with zero rows, contributes nothing to gate and is
            # not credited with a disposition.
            if filter_ast is not None and filter_channel_plans is not None and recency_plan_sink and filtered:
                filter_channel_plans["recency"] = recency_plan_sink[0]

            span.set_attribute("raw_count", len(recent_tuples))
            span.set_attribute("filtered_count", len(filtered))
            return filtered

    def _lexical_channel_active(self) -> bool:
        """Whether the lexical recall slot is active (#1391).

        Active when BM25 is enabled (the pre-#1391 gate) OR the lexical channel
        is set to keyword_ppr. The channel selector is itself the opt-in for
        keyword_ppr, so it does not require enable_bm25_channel.
        """
        return self._config.enable_bm25_channel or self._config.lexical_channel == "keyword_ppr"

    async def _lexical_search_chunks(
        self,
        query: str,
        namespace_id: UUID,
        limit: int,
        *,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Run the configured lexical channel (bm25 or keyword_ppr) (#1391).

        Dispatches on ``self._config.lexical_channel``. The keyword_ppr branch
        fills the SAME lexical/bm25 fusion slot so RRF is unchanged. Default
        ("bm25") is byte-identical to the pre-#1391 ``_bm25_search_chunks`` call.
        """
        if self._config.lexical_channel == "keyword_ppr":
            return await self._keyword_ppr_search_chunks(
                query=query,
                namespace_id=namespace_id,
                limit=limit,
                filter_ast=filter_ast,
                degradations=degradations,
            )
        return await self._bm25_search_chunks(
            query=query,
            namespace_id=namespace_id,
            limit=limit,
            filter_ast=filter_ast,
            filter_plan_out=filter_plan_out,
            degradations=degradations,
        )

    async def _keyword_ppr_search_chunks(
        self,
        query: str,
        namespace_id: UUID,
        limit: int,
        *,
        filter_ast: FilterNode | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """KET-RAG keyword-chunk PageRank lexical channel (#1391).

        Runs the per-query personalized PageRank over the namespace
        keyword->chunk bipartite and hydrates the ranked ids to ``(chunk_id,
        score, Chunk)`` triples in the shape fusion expects (matching the BM25
        and PPR channels).

        The channel cannot push a recall-filter down to the bipartite, so under
        a CONSTRAINING ``filter_ast`` it returns an empty channel (records an
        ADR-001 ``Degradation``) rather than smuggling filter-violating chunks
        into RRF - mirroring BM25's filtered-fallback guard. A constraint-free
        filter (``filter={}`` / ``RecallFilter()`` -> non-null match-everything
        ``AND`` with no children) has nothing to enforce, so it passes through;
        the test mirrors the ``filter_ast is None or not filter_ast.children``
        idiom used by the typed-entity fast path and the turbopuffer guard.
        Degrade-safe: no storage / no keywords / no edges / no seed overlap ->
        ``[]``.
        """
        if self._storage is None:
            return []
        if filter_ast is not None and filter_ast.children:
            if degradations is not None:
                degradations.append(
                    Degradation(
                        component="vectorcypher.keyword_ppr",
                        reason="filter_not_pushable",
                        detail="keyword_ppr cannot enforce a recall filter; lexical channel skipped",
                    )
                )
            return []

        from khora.extraction.tokenize import tokenize_multilingual
        from khora.query.keyword_ppr import keyword_ppr_retrieve_chunks

        with trace_span("khora.vectorcypher.keyword_ppr_search_chunks", namespace_id=str(namespace_id)) as span:
            ranked = await keyword_ppr_retrieve_chunks(
                self._storage,
                namespace_id,
                query,
                tokenizer=tokenize_multilingual,
                damping=self._config.keyword_ppr_damping,
                max_iter=self._config.ppr_max_iter,
                tol=self._config.ppr_tol,
                limit=limit,
                max_edges=self._config.keyword_ppr_max_edges,
            )
            span.set_attribute("ranked_count", len(ranked))
            if not ranked:
                return []
            chunk_ids = [cid for cid, _ in ranked]
            chunks_map = await self._storage.get_chunks_batch(chunk_ids, namespace_id=namespace_id)
            results: list[tuple[UUID, float, Chunk]] = []
            for cid, score in ranked:
                chunk = chunks_map.get(cid)
                if chunk is not None:
                    results.append((cid, score, chunk))
            span.set_attribute("chunk_count", len(results))
            return results

    async def _bm25_search_chunks(
        self,
        query: str,
        namespace_id: UUID,
        limit: int,
        *,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Full-text BM25 search on chunks.

        Prefers the temporal vector store's ``search_fulltext`` when
        available — that's where the batch ingest path actually writes
        chunk content. Falls back to the coordinator's
        ``search_fulltext_chunks`` (relational ``chunks`` table) for
        backends that don't expose a fulltext-search method on the
        temporal store.

        Args:
            query: Original query text
            namespace_id: Namespace to search
            limit: Maximum results
            filter_ast: Canonical recall-filter AST. When set, the search is
                pushed down through the temporal store's ``search_fulltext``
                (which compiles it to the SAME ``khora_chunks`` WHERE the
                vector channel uses). The coordinator ``chunks``-table
                fallback is then NOT used: that legacy table cannot honor a
                ``khora_chunks``-compiled predicate, so falling back would
                smuggle filter-violating chunks into RRF. When the temporal
                path is unavailable/empty under a filter, BM25 returns ``[]``
                rather than unfiltered rows. ``None`` keeps the
                temporal-first-then-coordinator behaviour byte-identical.
            filter_plan_out: Optional per-call sink for the honest
                filter-pushdown plan. The temporal-store ``search_fulltext``
                populates it from ITS OWN fulltext compile — so a raise-mode
                backend (pgvector / surrealdb) reports every leaf pushed, while
                the split-mode sqlite_lance backend reports the pushed slice plus
                the residual it re-checks in memory (``defensive_recheck=True``).
                Recorded regardless of row count (empty rows still means the
                filter was enforced). The coordinator fallback never runs under a
                filter, so it never contributes a plan.
            degradations: When provided, a structured ``Degradation`` is appended
                if the search raises so the silently-dropped lexical channel is
                observable (ADR-001, issue #1158). Recall continues with the
                BM25 contribution absent from RRF rather than crashing.

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        if not self._storage and not self._vector_store:
            logger.debug("No storage available for BM25 search")
            return []

        with trace_span("khora.vectorcypher.bm25_search_chunks", namespace_id=str(namespace_id)) as span:
            try:
                results: list[tuple[Chunk, float]] = []
                source = "coordinator"
                temporal_fulltext = getattr(self._vector_store, "search_fulltext", None)
                if callable(temporal_fulltext):
                    # Thread the per-call sink THROUGH to the temporal store so it
                    # records the honest plan from its OWN fulltext compile — a
                    # raise-mode backend reports every leaf pushed, the split-mode
                    # sqlite_lance backend reports the pushed/post-filtered split
                    # its WHERE + in-memory re-check actually produced. The store
                    # appends regardless of row count (an empty result still means
                    # the filter was enforced), so we no longer fabricate an
                    # all-pushed plan here. A backend that doesn't honor the sink
                    # (or a mock) simply leaves it empty, and the caller records no
                    # bm25 channel — honest.
                    raw = await temporal_fulltext(
                        namespace_id,
                        query,
                        limit=limit,
                        filter_ast=filter_ast,
                        filter_plan_out=filter_plan_out,
                    )
                    # Real backends return ``list[tuple[Chunk, float]]``; the
                    # ``isinstance`` check rejects the bare-AsyncMock case
                    # (and any other non-list return) so we fall through to
                    # the coordinator path instead of silently iterating
                    # an unrelated object.
                    if isinstance(raw, list) and raw:
                        results = raw
                        source = "temporal_store"
                # Coordinator fallback reads the legacy ``chunks`` table, whose
                # schema cannot carry the ``khora_chunks``-compiled predicate.
                # Only take it when NO deterministic filter is set — otherwise
                # it would smuggle filter-violating chunks into RRF (the
                # filtered path forbids a PG post-filter backstop).
                if not results and filter_ast is None and self._storage is not None:
                    coord_results = await self._storage.search_fulltext_chunks(
                        namespace_id,
                        query,
                        limit=limit,
                    )
                    if coord_results:
                        results = coord_results
                        source = "coordinator"
                span.set_attribute("chunk_count", len(results))
                span.set_attribute("source", source)
                if not results:
                    self._warn_bm25_empty_once(namespace_id)
                    # ADR-001 (issue #1330): a >=2-token keyword query that
                    # returns 0 BM25 rows silently drops the lexical channel
                    # from RRF without raising. Now that escape_fts5_query OR's
                    # its tokens, an empty multi-token channel is the residual
                    # failure mode worth surfacing. (A single-token / bare-ID
                    # query that finds nothing is expected, not a degradation.)
                    # Gate to the unfiltered path: under a deterministic
                    # ``filter_ast`` an empty result is a legitimate filtered
                    # miss (the predicate excluded every candidate), not a
                    # broken lexical channel - flagging it would inflate the
                    # public degraded_total counter with benign events.
                    if filter_ast is None and len(query.split()) >= 2:
                        _BM25_DEGRADED_COUNTER.add(1, attributes={"reason": "empty_multitoken_channel"})
                        if degradations is not None:
                            degradations.append(
                                Degradation(
                                    component="vectorcypher.bm25",
                                    reason="empty_multitoken_channel",
                                    detail="multi-token keyword query matched 0 chunks",
                                )
                            )
                logger.debug(f"BM25 channel returned {len(results)} chunks via {source}")
                return [
                    (
                        chunk.id,
                        score,
                        chunk,
                    )
                    for chunk, score in results
                ]
            except Exception as e:
                # ADR-001 (issue #1158): a BM25 failure here silently drops the
                # independent lexical channel from RRF fusion. Record a structured
                # Degradation so the missing channel is observable rather than
                # silent (matches the rel_fetch / cypher_expand convention).
                logger.warning(f"BM25 search failed: {e}", exc_info=True)
                _BM25_DEGRADED_COUNTER.add(1, attributes={"reason": "channel_exception"})
                if degradations is not None:
                    degradations.append(
                        Degradation(
                            component="vectorcypher.bm25",
                            reason="channel_exception",
                            detail=str(e)[:200] or None,
                            exception=type(e).__name__,
                        )
                    )
                return []

    def _warn_bm25_empty_once(self, namespace_id: UUID) -> None:
        """Log a one-shot WARNING when the BM25 channel returns 0 chunks.

        Tracks the first miss per namespace so dashboards and operators
        notice a silently-broken hybrid retrieval instead of having the
        signal buried at DEBUG level.
        """
        key = str(namespace_id)
        if key in self._bm25_empty_warned_ns:
            return
        self._bm25_empty_warned_ns.add(key)
        logger.warning(
            "BM25 channel returned 0 chunks for namespace {ns} — verify the "
            "ingest path populated the temporal-store chunk table "
            "(GitHub issue #813 tracked a write/read divergence on this path).",
            ns=key,
        )

    def _lazy_expand_chunks(
        self,
        vector_only_chunks: list[tuple[UUID, float, Chunk]],
        entry_entities: list[tuple[UUID, float]],
        entity_info_map: dict[str, dict[str, str]],
    ) -> list[tuple[UUID, float, Chunk]]:
        """Expand vector-only chunks by keyword matching against known entities.

        For chunks retrieved via vector search that have no MENTIONED_IN edges,
        extract keywords and match them against entity names. This recovers
        graph signal for chunks that weren't covered by skeleton extraction.

        Results are cached per chunk_id so repeated retrievals are fast.
        """
        from khora._accel import extract_keywords

        # Build lowercased entity name set
        entity_names: set[str] = set()
        for eid, _ in entry_entities:
            info = entity_info_map.get(str(eid), {})
            name = info.get("name", "").lower().strip()
            if name:
                entity_names.add(name)

        if not entity_names:
            return []

        results: list[tuple[UUID, float, Chunk]] = []
        for chunk_id, _vec_score, chunk in vector_only_chunks:
            # Check cache first
            if chunk_id in self._expansion_cache:
                cached_score = self._expansion_cache[chunk_id]
                if cached_score > 0:
                    results.append((chunk_id, cached_score, chunk))
                continue

            content = chunk.content
            if not content:
                self._expansion_cache[chunk_id] = 0.0
                continue

            keywords = {kw.lower() for kw in extract_keywords(content)}
            matches = keywords & entity_names
            if matches:
                # Weak signal: 0.5 per matched entity name
                expansion_score = len(matches) * 0.5
                results.append((chunk_id, expansion_score, chunk))
                self._expansion_cache[chunk_id] = expansion_score
            else:
                self._expansion_cache[chunk_id] = 0.0

        return results

    def _fuse_results(
        self,
        vector_chunks: list[tuple[UUID, float, Chunk]],
        graph_chunks: list[tuple[UUID, float, Chunk]],
        *,
        bm25_chunks: list[tuple[UUID, float, Chunk]] | None = None,
        use_normalization: bool = False,
        routing: RoutingDecision | None = None,
        is_temporal: bool = False,
    ) -> list[FusedResult]:
        """Fuse vector, graph, and optionally BM25 results using weighted RRF.

        When ``bm25_chunks`` is provided (BM25 channel active), uses the N-list
        ``reciprocal_rank_fusion`` from :mod:`khora.query.fusion` to fuse all
        three channels. Otherwise falls back to the 2-list
        ``weighted_rrf_normalized`` for vector+graph fusion.

        Args:
            vector_chunks: Results from vector search
            graph_chunks: Results from graph traversal
            bm25_chunks: Results from BM25 full-text search (optional)
            use_normalization: If True, normalize scores before fusion for better ranking
            routing: If provided, adjust weights based on query complexity
            is_temporal: If True, use temporal fusion weights (graph-heavy)

        Returns:
            Fused and sorted results
        """
        with trace_span(
            "khora.vectorcypher.rrf_fusion",
            vector_count=len(vector_chunks),
            graph_count=len(graph_chunks),
        ) as span:
            # Dynamic fusion weights based on query complexity
            vector_weight = self._config.vector_weight
            graph_weight = self._config.graph_weight
            if is_temporal:
                # Temporal queries benefit from graph-heavy fusion:
                # graph traversal surfaces temporally-related entities and their chunks
                vector_weight = self._config.temporal_vector_weight
                graph_weight = self._config.temporal_graph_weight

                # Adapt fusion weights when graph returns zero/few results.
                # When graph retrieval yields nothing (common for short conversational
                # messages without entity extraction), using graph-heavy weights (0.3/0.7)
                # dilutes good vector results and hurts ranking.
                graph_count = len(graph_chunks)
                if graph_count == 0:
                    vector_weight = 0.85
                    graph_weight = 0.15
                    logger.debug(
                        "Adaptive fusion: graph empty, using vector-heavy weights (%.2f/%.2f)",
                        vector_weight,
                        graph_weight,
                    )
                elif graph_count < 3:
                    vector_weight = self._config.vector_weight  # default 0.6
                    graph_weight = self._config.graph_weight  # default 0.4
                    logger.debug(
                        "Adaptive fusion: sparse graph (%d chunks), using moderate weights (%.2f/%.2f)",
                        graph_count,
                        vector_weight,
                        graph_weight,
                    )
            elif routing is not None:
                if routing.complexity == QueryComplexity.SIMPLE:
                    vector_weight = self._config.simple_vector_weight
                    graph_weight = self._config.simple_graph_weight
                elif routing.complexity == QueryComplexity.COMPLEX:
                    vector_weight = self._config.complex_vector_weight
                    graph_weight = self._config.complex_graph_weight

            span.set_attribute("vector_weight", vector_weight)
            span.set_attribute("graph_weight", graph_weight)

            # ── 3-channel fusion (vector + graph + BM25) ────────────────
            if bm25_chunks:
                from khora.query.fusion import reciprocal_rank_fusion as _nlist_rrf

                bm25_weight = self._config.bm25_weight
                span.set_attribute("bm25_weight", bm25_weight)
                span.set_attribute("bm25_count", len(bm25_chunks))

                # Build ranked lists in the (item, score) format expected by
                # the N-list reciprocal_rank_fusion.
                ranked_lists: dict[str, list[tuple[Chunk, float]]] = {}
                if vector_chunks:
                    ranked_lists["vector"] = [(chunk, score) for _cid, score, chunk in vector_chunks]
                if graph_chunks:
                    ranked_lists["graph"] = [(chunk, score) for _cid, score, chunk in graph_chunks]
                ranked_lists["bm25"] = [(chunk, score) for _cid, score, chunk in bm25_chunks]

                weights: dict[str, float] = {
                    "vector": vector_weight,
                    "graph": graph_weight,
                    "bm25": bm25_weight,
                }

                fused_raw: list[tuple[Chunk, float]] = _nlist_rrf(
                    ranked_lists,
                    weights=weights,
                    k=self._config.rrf_k,
                    id_extractor=lambda chunk: chunk.id,
                )

                # Convert (Chunk, rrf_score) tuples to FusedResult objects.
                # The N-list RRF doesn't populate per-source ranks, so we
                # build lookup maps to back-fill vector/graph provenance.
                vector_rank_map: dict[UUID, int] = {}
                vector_score_map: dict[UUID, float] = {}
                for rank, (cid, score, _chunk) in enumerate(vector_chunks, start=1):
                    vector_rank_map[cid] = rank
                    vector_score_map[cid] = score

                graph_rank_map: dict[UUID, int] = {}
                graph_score_map: dict[UUID, float] = {}
                for rank, (cid, score, _chunk) in enumerate(graph_chunks, start=1):
                    graph_rank_map[cid] = rank
                    graph_score_map[cid] = score

                return [
                    FusedResult(
                        item_id=chunk.id,
                        item=chunk,
                        rrf_score=rrf_score,
                        vector_rank=vector_rank_map.get(chunk.id),
                        graph_rank=graph_rank_map.get(chunk.id),
                        vector_score=vector_score_map.get(chunk.id),
                        graph_score=graph_score_map.get(chunk.id),
                    )
                    for chunk, rrf_score in fused_raw
                ]

            # ── 2-channel fusion (vector + graph) ──────────────────────
            if use_normalization:
                return weighted_rrf_normalized(
                    vector_results=vector_chunks,
                    graph_results=graph_chunks,
                    k=self._config.rrf_k,
                    vector_weight=vector_weight,
                    graph_weight=graph_weight,
                )
            return weighted_rrf(
                vector_results=vector_chunks,
                graph_results=graph_chunks,
                k=self._config.rrf_k,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            )

    def _calculate_recency_scores(
        self,
        results: list[FusedResult],
        *,
        decay_days_override: int | None = None,
        reference_mode: Literal["wall_clock", "relative"] | None = None,
    ) -> dict[UUID, float]:
        """Calculate recency scores for temporal boosting.

        Reference time selection (issue #567 A1):
        - ``reference_mode="wall_clock"`` uses ``datetime.now(UTC)`` —
          production-correct: if all retrieved chunks are old, the newest
          stale chunk must NOT receive ``recency=1.0``.
        - ``reference_mode="relative"`` uses ``max(occurred_at)`` over the
          result set — required for benchmark replay where the dataset's
          newest timestamp may be years in the past.
        - ``reference_mode=None`` (default) resolves via:
          ``KHORA_BENCH_MODE`` env var (read once at import) forces
          ``"relative"`` unconditionally; otherwise the
          ``RetrieverConfig.temporal_reference_wall_clock`` flag selects
          ``"wall_clock"`` (True) or ``"relative"`` (False — legacy).

        Decay window selection (issue #567 A4):
        - When ``RetrieverConfig.temporal_per_source_decay`` is True, each
          chunk's decay window is looked up from
          ``temporal_default_decay_by_source[chunk.metadata['source_system']]``
          with fallback to the ``"_default"`` key when ``source_system`` is
          None, empty, or absent from the dict.
        - When the flag is False, behaviour is unchanged:
          ``decay_days_override or self._config.recency_decay_days``.

        Args:
            results: Fused results with items containing occurred_at
            decay_days_override: Override for decay_days (e.g. 7 for RECENCY
                category). Ignored when per-source decay is enabled.
            reference_mode: Explicit override for the wall-clock vs relative
                reference decision; ``None`` resolves from config + env.

        Returns:
            Dict mapping item_id -> recency score (0-1)
        """
        scores: dict[UUID, float] = {}

        # First pass: extract all occurred_at timestamps.
        parsed_times: dict[UUID, datetime] = {}
        for r in results:
            occurred_at_str = _extract_occurred_at(r.item)
            if occurred_at_str:
                try:
                    occurred_at = datetime.fromisoformat(occurred_at_str.replace("Z", "+00:00"))
                    if occurred_at.tzinfo is None:
                        occurred_at = occurred_at.replace(tzinfo=UTC)
                    parsed_times[r.item_id] = occurred_at
                except (ValueError, TypeError):
                    pass

        # Resolve reference mode from explicit arg → bench env → config flag.
        # Bench-mode env wins to keep benchmark replays deterministic even
        # when the production-correct config flag is otherwise True.
        if reference_mode is None:
            if _BENCH_MODE:
                effective_mode: Literal["wall_clock", "relative"] = "relative"
            elif self._config.temporal_reference_wall_clock:
                effective_mode = "wall_clock"
            else:
                effective_mode = "relative"
        else:
            effective_mode = reference_mode

        if effective_mode == "wall_clock":
            now = datetime.now(UTC)
        else:  # "relative"
            now = max(parsed_times.values()) if parsed_times else datetime.now(UTC)

        # Per-source decay lookup.
        per_source = self._config.temporal_per_source_decay
        decay_map = self._config.temporal_default_decay_by_source if per_source else None

        def _decay_for(item: Any) -> float:
            """Resolve the decay window (in days) for a single result item."""
            if not per_source or decay_map is None:
                return float(decay_days_override or self._config.recency_decay_days)
            src = _extract_source_system(item)
            if src is not None and src in decay_map:
                return float(decay_map[src])
            # Fall back to the dict's _default, then to the static config
            # recency_decay_days if _default is somehow absent.
            fallback = decay_map.get("_default")
            if fallback is None:
                return float(self._config.recency_decay_days)
            return float(fallback)

        # Second pass: compute recency scores relative to reference time.
        for r in results:
            if r.item_id in parsed_times:
                decay_days = _decay_for(r.item)
                # Guard against pathological decay=0 (would div-by-zero).
                if decay_days <= 0:
                    decay_days = float(self._config.recency_decay_days)
                # Issue #1230: clamp at 0 so a future-dated chunk (occurred_at
                # ahead of the wall-clock reference) is treated as maximally
                # recent (factor 1.0) rather than producing a negative days_old
                # that pushes the recency factor above 1.0 and inflates the
                # multiplicative boost above the original fused score.
                days_old = max(0.0, (now - parsed_times[r.item_id]).total_seconds() / 86400.0)
                if self._config.recency_decay_type == "exponential":
                    half_life_lambda = math.log(2) / decay_days
                    recency = math.exp(-half_life_lambda * days_old)
                else:
                    recency = max(0.0, 1.0 - (days_old / decay_days))
                scores[r.item_id] = recency
            else:
                scores[r.item_id] = 0.5  # Default for missing/unparseable dates

        return scores


__all__ = [
    "RetrieverConfig",
    "VectorCypherResult",
    "VectorCypherRetriever",
]
