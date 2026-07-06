"""Chronicle engine — temporal-semantic memory for benchmark-optimized recall.

This engine is designed for high-accuracy memory retrieval benchmarks
(LongMemEval, LoCoMo, BEAM). Unlike VectorCypher, it requires no external graph
database. Two storage backends are supported:

* ``"pgvector"`` (default): PostgreSQL + pgvector. Best for shared / managed
  deployments and large corpora.
* ``"lancedb"``: SQLite (relational + FTS5) + LanceDB (vectors). Embedded,
  zero-infrastructure path that reuses the ``sqlite_lance`` storage backend.

Unlike Skeleton, chronicle performs full entity extraction and 4-channel
retrieval with temporal decay scoring on either backend.

Implements:
- Full ingest pipeline (chunking, embedding, entity extraction)
- 4-channel parallel retrieval: semantic + BM25 + temporal + entity co-occurrence
- Reciprocal Rank Fusion across all channels
- Temporal decay scoring (Ebbinghaus forgetting curve)
- Event decomposition (SVO tuples with datetime ranges)
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.diagnostics import Degradation, ErrorRecord
from khora.core.models import Chunk, Document, Entity, MemoryNamespace
from khora.core.models.recall import (
    DocumentProjection,
    RecallChunk,
    RecallEntity,
)
from khora.core.recall_abstention import compute_abstention_signals, compute_confidence
from khora.core.recall_scoring import min_max_normalize
from khora.engines._forget_cascade import _FORGET_DEGRADED_COUNTER, cascade_forget_extraction
from khora.engines._stats import gather_counts
from khora.engines._storage_config import build_storage_config
from khora.engines.chronicle.compression import (
    FactExtractor,
    FactOperation,
    MemoryCompressor,
    MemoryFact,
)
from khora.engines.chronicle.events import ChronicleEvent, EventExtractor
from khora.exceptions import ConfigurationError, EngineCapabilityError
from khora.extraction.embedders import LiteLLMEmbedder
from khora.khora import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.query.router import QueryComplexity, QueryComplexityRouter, RouterConfig
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator
from khora.telemetry import trace
from khora.telemetry.metrics import metric_counter, metric_histogram

if TYPE_CHECKING:
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.filter import FilterNode

ChronicleStorageBackend = Literal["pgvector", "lancedb"]


# Chronicle temporal-decay parameters - single source of truth for defaults.
# Bump in lockstep with config/schema.py (QuerySettings.temporal_half_life_hours,
# chronicle_decay_weight) and docs/engines/chronicle-engine.md.
DEFAULT_CHRONICLE_HALF_LIFE_HOURS: float = 168.0
DEFAULT_CHRONICLE_DECAY_WEIGHT: float = 0.30


# --- Abstention metrics (Phase 4) ---
# Module-level instruments so every recall() shares one OTel handle.
# No namespace label — Phase 0 audit identified 438 distinct namespaces,
# emitting per-namespace would blow cardinality. Aggregate-only by design.
_ABSTENTION_SIGNAL_COUNTER = metric_counter(
    "khora.chronicle.abstention_signal",
    description="Chronicle abstention signal firings, by signal name.",
)
_ABSTENTION_COMBINED_SCORE_HISTOGRAM = metric_histogram(
    "khora.chronicle.abstention_combined_score",
    unit="1",
    description="Chronicle abstention combined-score (0.0=confident, 1.0=should-abstain).",
)

# --- Reinforcement-on-recall metrics (#855) ---
# Each successful UPDATE batch increments ``updates_total``; each UPDATE that
# raised increments ``failures_total``; ``duration`` measures the wall-clock
# time from task schedule to UPDATE completion (or failure). Aggregate-only -
# no namespace label per the cardinality rule.
_REINFORCE_UPDATES_COUNTER = metric_counter(
    "khora.chronicle.reinforce.updates_total",
    description="Chronicle reinforcement UPDATE batches that completed successfully.",
)
_REINFORCE_FAILURES_COUNTER = metric_counter(
    "khora.chronicle.reinforce.failures_total",
    description="Chronicle reinforcement UPDATE batches that raised an exception.",
)
_REINFORCE_DURATION_HISTOGRAM = metric_histogram(
    "khora.chronicle.reinforce.duration",
    unit="s",
    description="Chronicle reinforcement task wall-clock duration, schedule to UPDATE completion.",
)

# --- Channel-degradation metric (PR #901, #906; ADR-001) ---
# Increments once per silent channel failure / fallback path the
# convention covers - see docs/architecture/failure-observability-contract.md.
# Reason is a low-cardinality label; channel is the BM25/semantic/temporal/entity
# bucket. NO namespace label - cardinality rule.
_CHANNEL_DEGRADED_COUNTER = metric_counter(
    "khora.chronicle.channel.degraded_total",
    description=(
        "Chronicle retrieval channel degradations (silent fallbacks, swallowed exceptions). "
        "Labels: channel, reason. No namespace label (cardinality rule)."
    ),
)


def _record_channel_degradation(
    degradations: list[Degradation],
    *,
    component: str,
    reason: str,
    detail: str | None = None,
    exc: BaseException | None = None,
) -> None:
    """Append a ``Degradation`` entry and bump the channel-degraded counter.

    Helper for the ADR-001 convention - the chronicle engine has several
    silent-fallback sites and we want one place that does both the
    metadata mutation and the metric emission.
    """
    entry: Degradation = {
        "component": component,
        "reason": reason,
        "detail": detail,
        "exception": type(exc).__name__ if exc is not None else None,
    }
    degradations.append(entry)
    # Label channel as the trailing token of component ("chronicle.bm25" -> "bm25").
    channel = component.rsplit(".", 1)[-1]
    _CHANNEL_DEGRADED_COUNTER.add(1, attributes={"channel": channel, "reason": reason})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_session_id_from_metadata(metadata: dict[str, Any] | None) -> UUID | None:
    """Pull ``session_id`` out of a metadata dict and coerce to UUID (#620).

    Mirrors the vectorcypher helper. Returns ``None`` on missing /
    malformed values so adapters can pass arbitrary metadata payloads
    without breaking ingest.
    """
    if not metadata:
        return None
    value = metadata.get("session_id")
    if value is None or value == "":
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _checksum_dedup_applies(existing: Document, *, external_id: str | None, session_id: UUID | None) -> bool:
    """Whether a checksum hit on *existing* counts as a duplicate (#1139).

    Mirrors the vectorcypher helper. Content checksum alone conflates
    distinct logical documents: a caller that supplies a new ``external_id``
    or ``session_id`` (e.g. the same message repeated in a different
    conversation) must get a new document, otherwise the write is silently
    dropped and ``forget_session`` of the original session deletes the only
    copy. Dedup applies only when every caller-supplied identity matches the
    existing row; callers that supply neither keep the checksum-only behavior.
    """
    if external_id is not None and existing.external_id != external_id:
        return False
    if session_id is not None and existing.session_id != session_id:
        return False
    return True


def _resolve_expertise(expertise: ExpertiseConfig | str | None) -> ExpertiseConfig | None:
    """Resolve a string expertise (registered name or YAML path) to a config.

    Chronicle reads ``expertise.events`` / ``expertise.facts`` /
    ``expertise.expansion`` attributes directly, so a string must be
    resolved up-front (the shared ingest pipeline resolves its own copy).
    Mirrors the pipeline's soft-fail: an unresolvable string logs a warning
    and behaves like no expertise.
    """
    if expertise is None or not isinstance(expertise, str):
        return expertise
    from khora.extraction.skills import load_expertise

    try:
        return load_expertise(expertise)
    except Exception as e:
        logger.warning(f"Failed to load expertise '{expertise}': {e}")
        return None


# ---------------------------------------------------------------------------
# Temporal decay helpers
# ---------------------------------------------------------------------------


def _ebbinghaus_decay(age_hours: float, *, half_life_hours: float = DEFAULT_CHRONICLE_HALF_LIFE_HOURS) -> float:
    """Compute a retention factor using an Ebbinghaus-inspired forgetting curve.

    R(t) = exp(-t / tau) where tau = half_life / ln(2).

    With the default half-life of 168 hours (7 days), a memory retains ~50 %
    strength after one week, ~25 % after two weeks, etc.

    Returns a value in (0, 1].
    """
    if age_hours <= 0:
        return 1.0
    tau = half_life_hours / math.log(2)
    return math.exp(-age_hours / tau)


def _apply_temporal_decay(
    chunks_with_scores: list[tuple[Chunk, float]],
    *,
    decay_weight: float = DEFAULT_CHRONICLE_DECAY_WEIGHT,
    half_life_hours: float = DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
    reference_time: datetime | None = None,
    enable_reinforcement: bool = False,
) -> list[tuple[Chunk, float]]:
    """Re-score chunks by blending relevance score with temporal decay.

    Multiplicative blend (matches Elasticsearch / Mem0 industry convention):

        final_score = relevance * ((1 - decay_weight) + decay_weight * retention)

    The max age penalty is ``decay_weight`` (when retention -> 0): a fully-faded
    memory keeps ``(1 - decay_weight)`` of its relevance score, while a fresh
    memory (retention -> 1) keeps 100%.

    Age is computed against ``chunk.source_timestamp`` (event time, supplied by
    the user via ``metadata['occurred_at']`` etc.), falling back to
    ``chunk.created_at`` (ingest time) only when no event time was supplied
    (#848). This prevents a 6-month-old conversation from being treated as
    "fresh" because it was just ingested.

    When ``enable_reinforcement`` is True (#855), the effective event time
    is ``max(source_timestamp, last_accessed_at)`` and falls back to
    ``created_at`` only when both are NULL. This keeps frequently-recalled
    chunks fresh even as their ``source_timestamp`` ages, matching the
    Stanford generative-agents reinforcement pattern.

    Uses Rust-accelerated ``batch_recency_scores`` from ``khora._accel``
    when available (~10x faster than per-item Python loop for large batches).
    Falls back to per-item computation otherwise.
    """
    if not chunks_with_scores or decay_weight <= 0:
        return chunks_with_scores

    now = reference_time or datetime.now(UTC)
    now_secs = now.timestamp()
    decay_days = half_life_hours / 24.0

    # Collect timestamps: prefer source_timestamp (event time) over created_at
    # (ingest time). This matters for backfilled / batched ingest where every
    # chunk's created_at is "now" but the events happened months ago (#848).
    # With reinforcement (#855), the effective time is the most recent of
    # source_timestamp and last_accessed_at.
    timestamps: list[float] = []
    for chunk, _score in chunks_with_scores:
        if enable_reinforcement:
            # Normalize each candidate to tz-aware UTC BEFORE max(): a naive
            # source_timestamp (sqlite_lance round-trips coerce_source_timestamp
            # output verbatim) must not crash against the always-aware
            # last_accessed_at stamped by _reinforce_last_accessed (#1145).
            candidates = [
                ts
                for ts in (_to_utc(chunk.source_timestamp), _to_utc(getattr(chunk, "last_accessed_at", None)))
                if ts is not None
            ]
            ts = max(candidates) if candidates else chunk.created_at
        else:
            ts = chunk.source_timestamp or chunk.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        timestamps.append(ts.timestamp())

    # Batch compute recency scores via Rust/NumPy/Python acceleration
    from khora._accel import batch_recency_scores

    recency_multipliers = batch_recency_scores(
        timestamps,
        now_secs,
        decay_days,
        decay_weight,
    )

    # Blend: relevance * recency_multiplier
    rescored: list[tuple[Chunk, float]] = [
        (chunk, relevance * mult) for (chunk, relevance), mult in zip(chunks_with_scores, recency_multipliers)
    ]
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored


def _compute_recency_multipliers(
    chunks: list[Chunk],
    *,
    decay_weight: float = DEFAULT_CHRONICLE_DECAY_WEIGHT,
    half_life_hours: float = DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
    reference_time: datetime | None = None,
    enable_reinforcement: bool = False,
) -> dict[UUID, float]:
    """Compute per-chunk recency multiplier without applying it to scores (#866).

    The cross-encoder reranker is timestamp-blind by construction; without
    a way to reapply decay AFTER the rerank, ``chronicle_decay_weight`` has
    no user-visible effect when ``enable_reranking`` is True. This helper
    returns the same multipliers ``_apply_temporal_decay`` uses internally
    so the recall pipeline can stash them pre-rerank and multiply them back
    onto rerank-output scores. Pattern matches Qdrant's decay re-scorer,
    Vespa's global-phase, and Elasticsearch's function_score with decay.

    Returns an empty dict when ``chunks`` is empty or ``decay_weight <= 0``;
    callers should treat a missing key as multiplier 1.0.
    """
    if not chunks or decay_weight <= 0:
        return {}

    now = reference_time or datetime.now(UTC)
    now_secs = now.timestamp()
    decay_days = half_life_hours / 24.0

    timestamps: list[float] = []
    for chunk in chunks:
        if enable_reinforcement:
            # Normalize to tz-aware UTC BEFORE max() - see _apply_temporal_decay (#1145).
            candidates = [
                ts
                for ts in (_to_utc(chunk.source_timestamp), _to_utc(getattr(chunk, "last_accessed_at", None)))
                if ts is not None
            ]
            ts = max(candidates) if candidates else chunk.created_at
        else:
            ts = chunk.source_timestamp or chunk.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        timestamps.append(ts.timestamp())

    from khora._accel import batch_recency_scores

    multipliers = batch_recency_scores(timestamps, now_secs, decay_days, decay_weight)
    return {chunk.id: float(mult) for chunk, mult in zip(chunks, multipliers)}


# ---------------------------------------------------------------------------
# Temporal-channel helpers (Chronicle #4)
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime | None) -> datetime | None:
    """Normalize a datetime to a tz-aware UTC datetime (or pass through None)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fact_event_time(fact: Any) -> datetime | None:
    """Effective event time of a fact for supersession ordering (#1144).

    Prefers the explicit ``event_time`` anchor (set at extraction from the
    source chunk's occurred_at / source_timestamp); falls back to
    ``created_at`` (ingestion time) for facts read back from storage, which
    carry no event-time column. Normalized to UTC so a tz-naive value never
    trips a naive-vs-aware comparison (#1145 class).
    """
    return _to_utc(getattr(fact, "event_time", None) or getattr(fact, "created_at", None))


def _target_is_newer(target: Any, new_fact: Any) -> bool:
    """True when ``target``'s event time is strictly newer than ``new_fact``'s.

    When both event times are known and the target is newer, superseding the
    target with the new fact would let an older real-world claim overwrite
    current state. Ties and missing event-times return ``False`` so the prior
    "new supersedes old" ingestion-order behavior is preserved.
    """
    target_t = _fact_event_time(target)
    new_t = _fact_event_time(new_fact)
    if target_t is None or new_t is None:
        return False
    return target_t > new_t


def _extract_temporal_bounds(temporal_filter: Any | None) -> tuple[datetime | None, datetime | None]:
    """Pull (start, end) UTC bounds out of a TemporalFilter-shaped object.

    Accepts ``occurred_after``/``occurred_before`` and ``start_time``/``end_time``
    attribute spellings to stay compatible with both Skeleton and Chronicle
    filters.
    """
    if temporal_filter is None:
        return None, None
    start = getattr(temporal_filter, "occurred_after", None) or getattr(temporal_filter, "start_time", None)
    end = getattr(temporal_filter, "occurred_before", None) or getattr(temporal_filter, "end_time", None)
    return _to_utc(start), _to_utc(end)


def _intersect_lower(window: datetime | None, filter_bound: datetime | None) -> datetime | None:
    """Intersect a lower bound with a filter's lower bound — narrow only.

    Returns the LATER of the two (the tighter lower bound); ``None`` on either
    side means unbounded below, so the other wins. The result is always >= the
    incoming ``window`` lower bound — the filter can shrink the recency window
    but never widen it. Both operands are coerced to tz-aware UTC so a zoneless
    bound never raises against an aware one.
    """
    fb = _to_utc(filter_bound)
    if fb is None:
        return window
    if window is None:
        return fb
    return max(window, fb)


def _intersect_upper(window: datetime | None, filter_bound: datetime | None) -> datetime | None:
    """Intersect an upper bound with a filter's upper bound — narrow only.

    Returns the EARLIER of the two (the tighter upper bound); ``None`` on either
    side means unbounded above, so the other wins. The result is always <= the
    incoming ``window`` upper bound — narrow only, never widening.
    """
    fb = _to_utc(filter_bound)
    if fb is None:
        return window
    if window is None:
        return fb
    return min(window, fb)


# System keys the recall-filter post-filter reads off a chunk record. All seven
# denormalized document keys live on the per-document ``DocumentProjection`` the
# recall path hydrates for filtered queries (``get_document_projections_batch``);
# ``DocumentSource`` carries only a subset (title / source / source_type), kept as
# a tertiary fallback so the projection-less paths don't regress. When a doc-key
# filter is present these resolve via the projection; on the short-circuited path
# (no doc-key leaf) the projection isn't fetched and the keys stay absent.
_DOC_PROJECTION_KEYS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)


def _chunk_to_record(chunk: Chunk, doc: DocumentProjection | None = None) -> dict[str, Any]:
    """Map a :class:`Chunk` to the record dict the recall-filter post-filter reads.

    The post-filter (``compile_python``) evaluates the full filter against this
    record; ``compile_chronicle`` pushes the ``source_timestamp`` bound into the
    recency window (its primary axis):

    * ``occurred_at`` is the effective EVENT time ``COALESCE(occurred_at,
      source_timestamp)`` — the chunk's real ``occurred_at`` column when carried
      (sqlite_lance), recovered from ``source_timestamp`` when the literal field is
      ``None`` (the legacy pgvector ``chunks`` DTO has no ``occurred_at`` column, so
      the faithful event time there IS ``source_timestamp``). This recovery is why
      an ``occurred_at`` filter does NOT false-empty on PG. Mirrors the engine's
      ``RecallChunk.occurred_at`` surface derivation. ``occurred_at`` is enforced by
      this post-filter, not pushed (it is the event-time axis, not the window axis).
    * ``created_at`` and ``source_timestamp`` are the LITERAL column values (a
      filter on those names is post-filtered against the real column).

    The seven denormalized document keys resolve from ``doc`` (the per-document
    :class:`DocumentProjection` the recall path hydrates when a doc-key filter is
    present), falling back to ``chunk.source_document`` (a subset). When ``doc`` is
    ``None`` — the short-circuited path (no doc-key leaf) or a degraded hydration —
    a key found nowhere stays absent, so a positive predicate on it returns empty.
    ``metadata`` is the chunk's own dict.
    """
    record: dict[str, Any] = {
        # Effective event time = COALESCE(occurred_at, source_timestamp). occurred_at
        # is conceptually PRESENT (it's the event time, == source_timestamp on the
        # legacy path where the DTO leaves the literal field None); the adapter
        # recovers it so an occurred_at filter does not false-empty. No created_at
        # fallback — ingest time is not event time.
        "occurred_at": chunk.occurred_at if chunk.occurred_at is not None else chunk.source_timestamp,
        "created_at": chunk.created_at,
        "source_timestamp": chunk.source_timestamp,
        "metadata": chunk.metadata or {},
    }
    # Resolve each denorm doc key: chunk attr → hydrated projection (``doc``) →
    # chunk.source_document → absent. The Chunk dataclass does not carry these
    # today (they live on the document), but trying the chunk first is robust if a
    # future DTO denormalizes them onto the chunk. ``doc`` carries all seven keys;
    # source_document is the tertiary fallback for projection-less paths. A key
    # found nowhere stays absent → compile_python's §4 missing-semantics →
    # positive predicate empty.
    source_doc = chunk.source_document
    for key in _DOC_PROJECTION_KEYS:
        value = getattr(chunk, key, None)
        if value is None and doc is not None:
            value = getattr(doc, key, None)
        if value is None and source_doc is not None:
            value = getattr(source_doc, key, None)
        if value is not None:
            record[key] = value
    return record


def _temporal_proximity(
    referenced_date: datetime | None,
    start: datetime | None,
    end: datetime | None,
) -> float | None:
    """Score how close ``referenced_date`` is to the query's temporal window.

    * ``None`` is returned when the event has no temporal anchor — callers
      should fall back to cosine-only scoring.
    * In-range events score 1.0.
    * Out-of-range events decay exponentially with distance from the nearest
      bound (1-week half-life ≈ ``exp(-days_outside / 7)``).
    * When only a single instant is supplied (start == end, or only one of
      them set), proximity is ``exp(-|days_diff| / 7)`` from that focal date.
    """
    if referenced_date is None:
        return None

    ref = _to_utc(referenced_date)
    assert ref is not None  # _to_utc only returns None for None input
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)

    if start_utc is None and end_utc is None:
        # Caller shouldn't reach here (channel checks signal first), but be
        # safe and return a neutral score.
        return 0.5

    # Single focal date: |days_diff| with 1-week half-life.
    focal: datetime | None = None
    if start_utc is not None and end_utc is not None and start_utc == end_utc:
        focal = start_utc
    elif start_utc is None:
        focal = end_utc
    elif end_utc is None:
        focal = start_utc

    if focal is not None:
        days_diff = abs((ref - focal).total_seconds()) / 86400.0
        return math.exp(-days_diff / 7.0)

    # Range case: in-range = 1.0; outside = decay from the nearest bound.
    assert start_utc is not None and end_utc is not None
    if start_utc <= ref <= end_utc:
        return 1.0
    if ref < start_utc:
        days_outside = (start_utc - ref).total_seconds() / 86400.0
    else:
        days_outside = (ref - end_utc).total_seconds() / 86400.0
    return math.exp(-days_outside / 7.0)


# ---------------------------------------------------------------------------
# Version-aware scoring helpers
# ---------------------------------------------------------------------------

# Patterns that signal the user wants the latest/current state
_VERSION_INTENT_PATTERNS = re.compile(
    r"\b(current|currently|latest|newest|most\s+recent|up[- ]?to[- ]?date|now|today|present"
    r"|status|state|active|existing)\b",
    re.IGNORECASE,
)


def _has_version_intent(query: str) -> bool:
    """Return True when the query signals interest in the latest version/state."""
    return _VERSION_INTENT_PATTERNS.search(query) is not None


def _apply_version_scoring(
    chunks_with_scores: list[tuple[Chunk, float]],
    query: str,
) -> list[tuple[Chunk, float]]:
    """Penalize superseded document versions so the latest state floats up.

    Only activates when:
      - The query contains temporal/state intent keywords
      - At least one chunk carries ``version`` metadata

    Chunks are grouped by entity (first ``entity_refs`` entry in
    ``chunk.metadata``, falling back to ``chunk.document_id``). Within
    each group the maximum version is identified, and older versions
    receive a soft penalty::

        score *= (version / max_version) ** 0.5

    The square-root exponent ensures old versions are demoted but not
    aggressively filtered out.
    """
    if not chunks_with_scores:
        return chunks_with_scores

    # Gate: only apply when the query signals latest/current intent
    if not _has_version_intent(query):
        return chunks_with_scores

    # Collect version info — bail quickly if no chunk has a version
    has_any_version = False
    chunk_meta: list[tuple[str, int | None]] = []  # (group_key, version)
    for chunk, _score in chunks_with_scores:
        custom = chunk.metadata or {}
        version = custom.get("version")
        if version is not None:
            has_any_version = True
            try:
                version = int(version)
            except (TypeError, ValueError):
                version = None

        # Group key: first entity_ref, or document_id
        entity_refs = custom.get("entity_refs")
        if entity_refs and isinstance(entity_refs, list) and len(entity_refs) > 0:
            group_key = str(entity_refs[0])
        else:
            group_key = str(chunk.document_id)

        chunk_meta.append((group_key, version))

    if not has_any_version:
        return chunks_with_scores

    # Build max-version lookup per entity group
    max_version: dict[str, int] = {}
    for group_key, version in chunk_meta:
        if version is not None:
            if group_key not in max_version or version > max_version[group_key]:
                max_version[group_key] = version

    # Re-score
    rescored: list[tuple[Chunk, float]] = []
    for (chunk, score), (group_key, version) in zip(chunks_with_scores, chunk_meta):
        if version is not None and group_key in max_version and max_version[group_key] > 0:
            penalty = (version / max_version[group_key]) ** 0.5
            score = score * penalty
        rescored.append((chunk, score))

    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored


_CROSS_SESSION_INTENT = re.compile(
    r"\b(chang(ed?|es|ing)|switch(ed)?|over\s+time|history|before\s+and\s+after"
    r"|different|evolv(ed?|ing)|transition|progress(ed|ion)?|moved?\s+(to|from)"
    r"|used\s+to|previous(ly)?|started|stopped|quit|left|joined)\b",
    re.IGNORECASE,
)


def _weighted_normalized_rrf_multi(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    weights: dict[str, float],
    *,
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """N-channel weighted RRF with per-channel min-max score normalization.

    Replaces the rank-only ``khora.query.fusion.reciprocal_rank_fusion`` call
    with a fusion that also blends in normalized raw scores. Each channel's
    raw scores are min-max normalized to [0, 1] independently — this neutralises
    the BM25-vs-cosine score-scale mismatch that ``reciprocal_rank_fusion``
    cannot detect (rank-only fusion treats rank-1 BM25 == rank-1 cosine).

    Score per chunk:

        sum_over_channels( w_c * (1 / (k + rank_c) + 0.01 * normalized_score_c) )

    The 0.01 factor keeps the normalized score subordinate to RRF (it acts as
    a tiebreaker / signal-preservation nudge, not the dominant signal).
    Mirrors :func:`khora.engines.vectorcypher.fusion.weighted_rrf_normalized`
    but extends it to N channels in a single pass — pairwise folding on top
    of that helper would re-normalise accumulator scores at every fold step
    and break the score-scale invariant.

    Args:
        ranked_lists: Channel name -> list of (Chunk, raw_score) tuples.
        weights: Channel name -> weight. Channels missing from ``weights``
            default to 1.0; channels missing from ``ranked_lists`` are skipped.
        k: RRF constant (default 60).

    Returns:
        List of (Chunk, fused_score) sorted by fused score descending.
    """
    if not ranked_lists:
        return []

    fused_scores: dict[Any, float] = {}
    chunk_by_id: dict[Any, Chunk] = {}

    for channel, results in ranked_lists.items():
        if not results:
            continue
        weight = weights.get(channel, 1.0)
        raw_scores = [score for _chunk, score in results]
        s_min = min(raw_scores)
        s_max = max(raw_scores)
        s_range = s_max - s_min
        for rank, (chunk, raw_score) in enumerate(results, start=1):
            cid = chunk.id
            chunk_by_id[cid] = chunk
            if s_range > 0:
                norm = (raw_score - s_min) / s_range
            else:
                norm = 1.0
            contribution = weight * (1.0 / (k + rank) + 0.01 * norm)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + contribution

    fused = [(chunk_by_id[cid], score) for cid, score in fused_scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


class ChronicleEngine:
    """Chronicle engine — temporal-semantic memory for benchmark-optimized recall.

    Key features:
    - Full entity extraction via shared ingest pipeline
    - 4-channel parallel retrieval (Phase 1: semantic + BM25; temporal + entity stubbed)
    - Reciprocal Rank Fusion for multi-channel result merging
    - Ebbinghaus temporal decay scoring
    - No graph database required — PostgreSQL + pgvector only
    - Event extraction (Chronicle #2): every persisted chunk is decomposed
      into SVO ``ChronicleEvent`` rows by ``EventExtractor`` and written to
      ``chronicle_events`` after the ingest pipeline finishes.

    Event-extraction toggle resolution order (highest priority first):

    1. ``namespace.config_overrides["events"]["enabled"]`` — runtime override
       on the namespace, useful for opt-out without changing the global expertise.
    2. ``expertise.events.enabled`` — the ``ExpertiseConfig`` default
       (``True`` unless overridden).

    When neither is set the extractor runs (default-on). Per-chunk extraction
    failures are swallowed with a warning so a single bad LLM call cannot
    take down the whole ``remember()``.

    Usage:
        engine = ChronicleEngine(config)
        await engine.connect()

        # Or via Khora facade:
        async with Khora(db_url, engine="chronicle") as kb:
            await kb.remember(content, namespace=ns_id,
                entity_types=["PERSON"], relationship_types=["KNOWS"])
            result = await kb.recall("query", namespace=ns_id)
    """

    # #833: Chronicle's 4-channel design supports VECTOR (semantic only),
    # HYBRID (semantic + BM25 + temporal + entity), and ALL (same as HYBRID
    # here - no extra channels to add). KEYWORD and GRAPH are NOT supported:
    # KEYWORD would require disabling the temporal + entity channels and
    # returning only BM25 results, which contradicts chronicle's design
    # intent (temporal scoring is its differentiator); GRAPH is impossible
    # because chronicle has no graph backend. Both raise
    # ``EngineCapabilityError``.
    supported_modes: ClassVar[frozenset[SearchMode]] = frozenset({SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL})

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
        storage_backend: ChronicleStorageBackend | None = None,
        lancedb_path: str | None = None,
        temporal_use_events: bool = True,
        temporal_event_cosine_weight: float = 0.5,
        entity_limit: int = 20,
        router_enabled: bool = True,
        abstention_min_chunks: int | None = None,
        abstention_min_top_score: float | None = None,
        abstention_combined_threshold: float | None = None,
        abstention_mode: Literal["cosine_floor", "weighted"] | None = None,
    ) -> None:
        """Initialize the Chronicle engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (derived from config if None) - deprecated
            storage_backend: Selects the chunk vector backend.

                - ``"pgvector"``: PostgreSQL + pgvector (the original chronicle path).
                - ``"lancedb"``: SQLite (relational + FTS5) + LanceDB (vectors). Embedded,
                  zero-infrastructure. Reuses the ``sqlite_lance`` storage backend.
                - ``None`` (default): inherit from ``config.storage.backend`` —
                  ``"sqlite_lance"`` selects LanceDB, ``"surrealdb"`` selects SurrealDB,
                  anything else falls back to pgvector.

            lancedb_path: When ``storage_backend="lancedb"``, points at the SQLite db
                file (the LanceDB directory is the sibling ``.lance`` path). Defaults
                to ``./chronicle.db`` if neither this nor ``config.storage.sqlite_lance``
                is provided.
            temporal_use_events: When True (default), the temporal channel queries
                ``chronicle_events`` and ranks by ``referenced_date`` (the date the
                source text refers to) plus event-summary cosine. When False (or when
                the events table is empty / the query has no temporal signal), the
                channel falls back to chunk semantic search with Ebbinghaus decay on
                ``chunks.created_at``.
            temporal_event_cosine_weight: Blend factor between event-summary cosine
                and temporal-proximity scores in the events path. ``0.5`` weights
                them equally; higher values bias toward semantic match, lower toward
                temporal anchoring.
            entity_limit: Cap on the number of entities surfaced in
                ``RecallResult.entities``. The fused list combines direct entity-channel
                hits (full score) with event-derived entities (score attenuated by
                ``0.5``) and is sorted by score before truncation. Defaults to 20.
            router_enabled: When True (default), classify queries via
                ``QueryComplexityRouter`` and skip the BM25 + entity channels for
                SIMPLE queries. The temporal channel is always preserved (chronicle's
                differentiator) — temporal queries that the router would otherwise
                classify SIMPLE keep the temporal channel via ``temporal_signal``.
                When False, all four channels run on every query (legacy behaviour).
            abstention_min_chunks: Minimum chunk count below which the
                ``chunks_below_min`` abstention flag fires. ``None`` (default)
                inherits ``config.query.abstention_min_chunks``; an explicit
                kwarg wins (#1331).
            abstention_min_top_score: Top-chunk score below which the
                ``top_score_low`` abstention flag fires. ``None`` inherits
                ``config.query.abstention_min_top_score``.
            abstention_combined_threshold: Combined-score threshold at or above
                which ``should_abstain`` becomes True in ``abstention_mode=
                "weighted"``. ``None`` inherits the config value. See
                ``_compute_abstention_signals`` for the weighting scheme.
            abstention_mode: ``should_abstain`` derivation - ``"cosine_floor"``
                (default) or the legacy ``"weighted"`` escape hatch. ``None``
                inherits ``config.query.abstention_mode``.
        """
        self._config = config
        self._storage_backend: ChronicleStorageBackend | None = storage_backend
        self._lancedb_path = lancedb_path
        self._temporal_use_events = temporal_use_events
        self._temporal_event_cosine_weight = temporal_event_cosine_weight
        self._entity_limit = entity_limit
        self._router_enabled = router_enabled
        # Router is cheap (regex + dataclass); construct eagerly. Disabling
        # via ``router_enabled=False`` short-circuits at the ``recall()`` site.
        self._router: QueryComplexityRouter = QueryComplexityRouter(RouterConfig(enabled=router_enabled))

        if storage_config is not None:
            self._storage_config = storage_config
        elif storage_backend == "lancedb":
            self._storage_config = self._build_lancedb_storage_config(config, lancedb_path)
        else:
            # pgvector / inherit from config — chronicle skips the graph backend
            # for pgvector mode (entity SQL columns live on pgvector itself).
            self._storage_config = build_storage_config(config, skip_graph=True)

        # Explicit kwarg wins; otherwise inherit from config.query (#1331).
        _q = config.query
        self._abstention_min_chunks = (
            abstention_min_chunks if abstention_min_chunks is not None else _q.abstention_min_chunks
        )
        self._abstention_min_top_score = (
            abstention_min_top_score if abstention_min_top_score is not None else _q.abstention_min_top_score
        )
        self._abstention_combined_threshold = (
            abstention_combined_threshold
            if abstention_combined_threshold is not None
            else _q.abstention_combined_threshold
        )
        self._abstention_mode = abstention_mode if abstention_mode is not None else _q.abstention_mode
        self._abstention_weight_entities_empty = _q.abstention_weight_entities_empty
        self._abstention_weight_chunks_below_min = _q.abstention_weight_chunks_below_min
        self._abstention_weight_top_score_low = _q.abstention_weight_top_score_low
        self._abstention_confidence_target_cosine = _q.abstention_confidence_target_cosine
        self._abstention_confidence_target_gap = _q.abstention_confidence_target_gap

        self._storage: StorageCoordinator | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._event_extractor: EventExtractor | None = None
        self._fact_extractor: FactExtractor | None = None
        self._compressor: MemoryCompressor | None = None
        # Soft cap on concurrent LLM extractions per remember/remember_batch.
        # Both event and fact extraction are one LLM call per chunk; the
        # same semaphore caps the combined fan-out — they share the LLM
        # provider as a single resource pool.
        self._max_concurrent_extractions: int = 10
        self._connected = False
        # Live references to in-flight reinforcement-on-recall tasks (#855).
        # Python's GC may collect a task with no live reference mid-flight;
        # holding the set keeps the task alive and gives ``disconnect()`` a
        # handle to cancel + drain on shutdown. The ``add_done_callback``
        # registered at the spawn site discards entries when each task
        # finishes so the set never grows unbounded.
        self._reinforce_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _build_lancedb_storage_config(
        config: KhoraConfig,
        lancedb_path: str | None,
    ) -> StorageConfig:
        """Construct a StorageConfig pointing at the sqlite_lance unified backend.

        Reuses the user's existing ``config.storage.sqlite_lance`` entry when set;
        otherwise synthesizes one from ``lancedb_path`` (or the default
        ``./chronicle.db``) and the configured embedding dimension.
        """
        from khora.config.schema import SQLiteLanceConfig

        sl_cfg = getattr(config.storage, "sqlite_lance", None)
        if sl_cfg is None:
            db_path = lancedb_path or "./chronicle.db"
            sl_cfg = SQLiteLanceConfig(
                db_path=db_path,
                embedding_dimension=config.llm.embedding_dimension,
            )
        elif lancedb_path is not None:
            # Caller-supplied path overrides the config entry's db_path so a
            # single config can support multiple chronicle deployments.
            sl_cfg = sl_cfg.model_copy(update={"db_path": lancedb_path})

        return StorageConfig(
            backend="sqlite_lance",
            sqlite_lance_config=sl_cfg,
            postgresql_url=None,
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Chronicle engine...")

        # When using the embedded sqlite_lance backend, the SQLite schema must
        # exist before the coordinator opens the file. Postgres deployments
        # are expected to migrate via ``alembic upgrade head`` out-of-band.
        if self._storage_config.backend == "sqlite_lance":
            await self._ensure_sqlite_schema()

        # Create and connect storage coordinator
        self._storage = create_storage_coordinator(self._storage_config)
        await self._storage.connect()

        # Reinforcement-on-recall capability check (#855).
        # Vector backends that don't implement ``update_last_accessed`` would
        # silently no-op the UPDATE - users on those backends would see a
        # config flag with no effect. Fail loudly at connect time instead so
        # the operator knows reinforcement is unavailable on this stack.
        # Currently supported: pgvector, sqlite_lance, surrealdb.
        qs = self._config.query
        reinforcement_on = getattr(qs, "chronicle_enable_recall_reinforcement", False) if qs else False
        if reinforcement_on:
            vector = self._storage._vector
            if vector is not None and not hasattr(vector, "update_last_accessed"):
                backend_name = type(vector).__name__
                raise ConfigurationError(
                    "Chronicle reinforcement-on-recall "
                    "(KHORA_QUERY_CHRONICLE_ENABLE_RECALL_REINFORCEMENT=true) "
                    f"requires the vector backend to implement update_last_accessed; "
                    f"{backend_name} does not. Supported backends: pgvector, sqlite_lance, surrealdb."
                )

        # Create embedder
        llm_config = LiteLLMConfig(
            model=self._config.llm.model,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            timeout=self._config.llm.timeout,
            max_retries=self._config.llm.max_retries,
        )
        self._embedder = LiteLLMEmbedder.from_config(llm_config)

        # Initialize telemetry (no-op if KHORA_TELEMETRY_DATABASE_URL not set)
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("Chronicle engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Chronicle engine...")

        # Cancel and drain any in-flight reinforcement-on-recall tasks (#855)
        # BEFORE tearing down the storage coordinator they write to. Otherwise
        # a task can race the coordinator disconnect and try to UPDATE against
        # a closed pool.
        if self._reinforce_tasks:
            for task in list(self._reinforce_tasks):
                task.cancel()
            await asyncio.gather(*self._reinforce_tasks, return_exceptions=True)
            self._reinforce_tasks.clear()

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._connected = False

        logger.info("Chronicle engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        """Get storage coordinator, raising if not connected."""
        if self._storage is None:
            raise RuntimeError("Chronicle engine not connected. Call connect() first.")
        return self._storage

    async def _ensure_sqlite_schema(self) -> None:
        """Run Alembic migrations against the sqlite_lance database file.

        The unified backend factory (``StorageFactory._create_sqlite_lance_coordinator``)
        opens the SQLite file directly via aiosqlite — it relies on the schema
        already being in place. This mirrors the integration-test fixture
        (``tests/integration/_sqlite_lance_fixtures.py``) so chronicle users
        get a one-call setup instead of having to run ``alembic upgrade head``
        manually.
        """
        from khora.db.session import run_migrations

        sl_cfg = self._storage_config.sqlite_lance_config
        if sl_cfg is None:
            return  # nothing to migrate
        db_path = getattr(sl_cfg, "db_path", None)
        if not db_path:
            return
        url = f"sqlite+aiosqlite:///{db_path}"
        result = await run_migrations(url)
        if not result.success:
            raise RuntimeError(f"Chronicle sqlite_lance migration failed: {result.error}")

    def _get_embedder(self) -> LiteLLMEmbedder:
        """Get embedder, raising if not connected."""
        if self._embedder is None:
            raise RuntimeError("Chronicle engine not connected. Call connect() first.")
        return self._embedder

    # =========================================================================
    # Event extraction wiring (Chronicle #2)
    # =========================================================================

    @staticmethod
    def _events_enabled(namespace: MemoryNamespace | None, expertise: ExpertiseConfig | None) -> bool:
        """Resolve the events.enabled flag.

        Per-namespace ``config_overrides["events"]["enabled"]`` beats
        ``expertise.events.enabled``; both default to True (default-on) when
        absent.
        """
        if namespace is not None and namespace.config_overrides:
            ns_events = namespace.config_overrides.get("events")
            if isinstance(ns_events, dict) and "enabled" in ns_events:
                return bool(ns_events["enabled"])
        if expertise is not None:
            return bool(expertise.events.enabled)
        return True

    def _get_event_extractor(self, expertise: ExpertiseConfig | None) -> EventExtractor:
        """Lazily construct the EventExtractor.

        The model comes from ``expertise.events.model`` when available, falling
        back to ``config.llm.extraction_model`` and finally ``config.llm.model``.
        Constructed once per engine instance — one extractor handles all
        remember/remember_batch calls.
        """
        if self._event_extractor is None:
            model = (
                (expertise.events.model if expertise is not None else None)
                or self._config.llm.extraction_model
                or self._config.llm.model
            )
            self._event_extractor = EventExtractor(model=model)
        return self._event_extractor

    async def _extract_events_for_chunk(
        self,
        chunk: Chunk,
        namespace_id: UUID,
        extractor: EventExtractor,
        sem: asyncio.Semaphore,
        errors_out: list[ErrorRecord] | None = None,
    ) -> list[ChronicleEvent]:
        """Run the extractor on a single chunk, swallowing per-chunk failures.

        Returns an empty list if extraction raises so a single bad LLM call
        cannot fail the whole ``remember()``. The chunk_id and namespace_id
        are stamped onto every returned event so callers can pass the list
        straight to ``coordinator.write_events``. Issue #903: transient
        extractor failures are forwarded to ``errors_out`` so the caller
        can surface them on ``RememberResult.metadata['errors']``.
        """
        async with sem:
            try:
                events = await extractor.extract_events(
                    chunk.content,
                    chunk_id=chunk.id,
                    namespace_id=namespace_id,
                    errors_out=errors_out,
                )
            except Exception as exc:
                logger.warning(
                    "Event extraction failed for chunk {}: {}",
                    chunk.id,
                    exc,
                )
                return []
        # The extractor already sets chunk_id/namespace_id, but be defensive
        # in case a downstream replacement breaks that contract.
        for ev in events:
            ev.chunk_id = chunk.id
            ev.namespace_id = namespace_id
        return events

    async def _embed_events(self, events: list[ChronicleEvent]) -> None:
        """Populate ``event.embedding`` using the chronicle embedder.

        Embedded in a single batch call to amortize the API round-trip.
        Skips events whose summary is empty. Embedding failures are
        swallowed (events still persist with ``embedding=None``).
        """
        if not events:
            return
        embedder = self._get_embedder()
        targets = [ev for ev in events if ev.summary.strip()]
        if not targets:
            return
        try:
            vectors = await embedder.embed_batch([ev.summary for ev in targets])
        except Exception as exc:
            logger.warning("Event summary embedding failed: {}", exc)
            return
        for ev, vec in zip(targets, vectors):
            ev.embedding = vec

    async def _extract_and_persist_events(
        self,
        chunks: list[Chunk],
        namespace_id: UUID,
        expertise: ExpertiseConfig | None,
        errors_out: list[ErrorRecord] | None = None,
        degradations_out: list[Degradation] | None = None,
    ) -> int:
        """Extract events from chunks, embed summaries, persist, and return count.

        Resolution of the enable flag is the caller's responsibility — this
        helper assumes events are already enabled and unconditionally runs.
        Issue #903: transient extractor failures are forwarded to ``errors_out``.
        """
        if not chunks:
            return 0
        extractor = self._get_event_extractor(expertise)
        sem = asyncio.Semaphore(self._max_concurrent_extractions)
        per_chunk = await asyncio.gather(
            *(self._extract_events_for_chunk(c, namespace_id, extractor, sem, errors_out=errors_out) for c in chunks),
            return_exceptions=False,
        )
        events: list[ChronicleEvent] = [ev for sub in per_chunk for ev in sub]
        if not events:
            return 0
        await self._embed_events(events)
        try:
            await self._get_storage().write_events(events, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("write_events failed (skipping event persistence): {}", exc, exc_info=True)
            if degradations_out is not None:
                _record_channel_degradation(
                    degradations_out,
                    component="chronicle.events",
                    reason="write_events_failed",
                    detail=f"{len(events)} event(s) dropped; persistence raised",
                    exc=exc,
                )
            return 0
        logger.debug("Persisted {} chronicle events across {} chunks", len(events), len(chunks))
        return len(events)

    # =========================================================================
    # Fact extraction wiring (Chronicle #3)
    # =========================================================================

    @staticmethod
    def _facts_enabled(namespace: MemoryNamespace | None, expertise: ExpertiseConfig | None) -> bool:
        """Resolve the ``facts.enabled`` flag.

        Per-namespace ``config_overrides["facts"]["enabled"]`` beats
        ``expertise.facts.enabled``; both default to True (default-on) when
        absent. Mirrors ``_events_enabled``.
        """
        if namespace is not None and namespace.config_overrides:
            ns_facts = namespace.config_overrides.get("facts")
            if isinstance(ns_facts, dict) and "enabled" in ns_facts:
                return bool(ns_facts["enabled"])
        if expertise is not None:
            return bool(expertise.facts.enabled)
        return True

    def _get_fact_extractor(self, expertise: ExpertiseConfig | None) -> FactExtractor:
        """Lazily construct the FactExtractor.

        The model comes from ``expertise.facts.model`` when available, falling
        back to ``config.llm.extraction_model`` and finally ``config.llm.model``.
        """
        if self._fact_extractor is None:
            model = (
                (expertise.facts.model if expertise is not None else None)
                or self._config.llm.extraction_model
                or self._config.llm.model
            )
            self._fact_extractor = FactExtractor(model=model)
        return self._fact_extractor

    def _get_compressor(self, expertise: ExpertiseConfig | None) -> MemoryCompressor:
        """Lazily construct the MemoryCompressor (used for reconciliation)."""
        if self._compressor is None:
            model = (
                (expertise.facts.model if expertise is not None else None)
                or self._config.llm.extraction_model
                or self._config.llm.model
            )
            self._compressor = MemoryCompressor(model=model)
        return self._compressor

    async def _extract_facts_for_chunk(
        self,
        chunk: Chunk,
        namespace_id: UUID,
        extractor: FactExtractor,
        sem: asyncio.Semaphore,
        errors_out: list[ErrorRecord] | None = None,
    ) -> list[MemoryFact]:
        """Run the fact extractor on a single chunk, swallowing per-chunk failures.

        Issue #903: transient extractor failures are forwarded to ``errors_out``
        so the caller can surface them on ``RememberResult.metadata['errors']``.
        """
        async with sem:
            try:
                facts = await extractor.extract_facts(
                    chunk.content,
                    chunk_id=chunk.id,
                    namespace_id=namespace_id,
                    errors_out=errors_out,
                )
            except Exception as exc:
                logger.warning(
                    "Fact extraction failed for chunk {}: {}",
                    chunk.id,
                    exc,
                )
                return []
        # Defensive: stamp namespace_id and chunk linkage in case the
        # extractor was replaced with a stub that doesn't set them.
        # #1144: anchor the fact's event time on the source chunk's event time
        # (occurred_at preferred, source_timestamp fallback — the same COALESCE
        # convention the temporal channel uses) so reconciliation supersedes by
        # real-world time, not ingestion order.
        chunk_event_time = chunk.occurred_at if chunk.occurred_at is not None else chunk.source_timestamp
        for f in facts:
            f.namespace_id = namespace_id
            if chunk.id not in f.source_chunk_ids:
                f.source_chunk_ids.append(chunk.id)
            if f.event_time is None:
                f.event_time = chunk_event_time
        return facts

    async def _reconcile_facts(
        self,
        new_facts: list[MemoryFact],
        namespace_id: UUID,
        expertise: ExpertiseConfig | None,
        degradations_out: list[Degradation] | None = None,
    ) -> tuple[int, int]:
        """Apply ADD/UPDATE/DELETE/NOOP/SKIP reconciliation and persist results.

        Strategy:
          1. Group new facts by subject.
          2. Cache active facts per subject (one query per subject).
          3. For each new fact, ask the compressor what to do:
             - ADD     → queue for write
             - UPDATE  → queue for write + supersede the target
             - DELETE  → supersede only (no write)
             - NOOP    → skip (duplicate)
             - SKIP    → drop fact, bump reconcile-error counter (issue #892)
          4. After processing a subject, append accepted facts to its in-memory
             cache so subsequent new facts about the same subject reconcile
             against everything we just decided to keep — prevents within-batch
             thrashing when two sentences contain the same fact.

        Returns ``(facts_persisted, reconcile_errors)`` where ``reconcile_errors``
        is the number of facts dropped because the contradiction-check LLM call
        failed transiently. Issue #892 - previously these errors were swallowed
        and the fact was silently ADDed, accumulating contradictions.
        """
        if not new_facts:
            return 0, 0

        compressor = self._get_compressor(expertise)
        storage = self._get_storage()

        # Cache of active facts per subject — populated lazily.
        active_by_subject: dict[str, list[MemoryFact]] = {}

        facts_to_write: list[MemoryFact] = []
        # Pairs of (old_fact_id, new_fact_id) for supersede calls. We resolve
        # the new ID *after* write_facts so the new fact has a real DB id.
        pending_supersedes: list[tuple[UUID, MemoryFact]] = []
        # DELETE only — supersede pointing at *no* new fact (NULL) — done now
        # because no write is needed.
        deletes: list[UUID] = []
        # Issue #892: count facts dropped due to transient LLM failures in
        # the contradiction check.
        reconcile_errors = 0

        for new_fact in new_facts:
            subject = new_fact.subject
            if subject not in active_by_subject:
                try:
                    existing = await storage.query_active_facts_for_subject(namespace_id, subject)
                except Exception as exc:
                    logger.warning(
                        "query_active_facts_for_subject failed for subject {!r}: {} — falling back to ADD",
                        subject,
                        exc,
                    )
                    existing = []
                active_by_subject[subject] = list(existing)

            existing_for_subject = active_by_subject[subject]
            action = await compressor.reconcile_fact(existing_for_subject, new_fact)

            if action.op is FactOperation.ADD:
                facts_to_write.append(new_fact)
                active_by_subject[subject].append(new_fact)
            elif action.op is FactOperation.UPDATE:
                # #1144: supersede by EVENT time, not ingestion order. If the
                # target's event time is strictly newer than the new fact's,
                # the new fact is older real-world info (a backfill); inverting
                # keeps the newer fact active and records the older one as
                # superseded-by it instead of the other way round. Ties /
                # missing event-times keep the prior "new supersedes old".
                if action.target is not None and _target_is_newer(action.target, new_fact):
                    new_fact.is_active = False
                    new_fact.superseded_by = action.target.id
                    facts_to_write.append(new_fact)
                    pending_supersedes.append((new_fact.id, action.target))
                    # Target stays active in the in-memory cache; do not append
                    # the now-inactive new fact.
                else:
                    facts_to_write.append(new_fact)
                    if action.target is not None:
                        pending_supersedes.append((action.target.id, new_fact))
                        # Drop the old fact from the in-memory cache so subsequent
                        # reconciliations within this batch don't see it as active.
                        active_by_subject[subject] = [f for f in active_by_subject[subject] if f.id != action.target.id]
                    active_by_subject[subject].append(new_fact)
            elif action.op is FactOperation.DELETE:
                # #1144: never delete a target whose event time is newer than
                # the new fact's — that would drop current state in favour of a
                # backfilled older claim. Record the older new fact as inactive
                # instead (it never enters the active set).
                if action.target is not None and _target_is_newer(action.target, new_fact):
                    new_fact.is_active = False
                    new_fact.superseded_by = action.target.id
                    facts_to_write.append(new_fact)
                elif action.target is not None:
                    deletes.append(action.target.id)
                    active_by_subject[subject] = [f for f in active_by_subject[subject] if f.id != action.target.id]
            elif action.op is FactOperation.SKIP:
                reconcile_errors += 1
            # NOOP: skip — nothing to do.

        # Persist new ADDs/UPDATEs first so we have stable IDs to point at.
        if facts_to_write:
            try:
                await storage.write_facts(facts_to_write, namespace_id=namespace_id)
            except Exception as exc:
                logger.warning("write_facts failed during reconciliation: {}", exc, exc_info=True)
                if degradations_out is not None:
                    _record_channel_degradation(
                        degradations_out,
                        component="chronicle.facts",
                        reason="write_facts_failed",
                        detail=f"{len(facts_to_write)} fact(s) dropped during reconciliation; persistence raised",
                        exc=exc,
                    )
                return 0, reconcile_errors

        # Supersede old → new for UPDATEs.
        for old_id, new_fact in pending_supersedes:
            try:
                await storage.supersede_fact(old_id, new_fact.id, namespace_id=namespace_id)
            except Exception as exc:
                logger.warning("supersede_fact failed for {} -> {}: {}", old_id, new_fact.id, exc)

        # DELETE: mark the old fact inactive without a replacement. The
        # storage contract takes a UUID for ``superseded_by``; passing the
        # fact's own id keeps the row marked inactive while leaving a
        # self-reference as the "tombstone" pointer.
        for old_id in deletes:
            try:
                await storage.supersede_fact(old_id, old_id, namespace_id=namespace_id)
            except Exception as exc:
                logger.warning("supersede_fact (delete) failed for {}: {}", old_id, exc)

        return len(facts_to_write), reconcile_errors

    async def _extract_and_persist_facts(
        self,
        chunks: list[Chunk],
        namespace_id: UUID,
        expertise: ExpertiseConfig | None,
        errors_out: list[ErrorRecord] | None = None,
        degradations_out: list[Degradation] | None = None,
    ) -> tuple[int, int]:
        """Extract facts from chunks, reconcile, persist, and return count.

        Returns ``(facts_persisted, reconcile_errors)`` where ``reconcile_errors``
        is the number of facts dropped because the contradiction-check LLM call
        failed transiently (issue #892). The non-reconcile fast path always
        reports zero reconcile errors. Resolution of the ``facts.enabled``
        flag is the caller's responsibility. Issue #903: transient extractor
        failures are forwarded to ``errors_out``.
        """
        if not chunks:
            return 0, 0
        extractor = self._get_fact_extractor(expertise)
        sem = asyncio.Semaphore(self._max_concurrent_extractions)
        per_chunk = await asyncio.gather(
            *(self._extract_facts_for_chunk(c, namespace_id, extractor, sem, errors_out=errors_out) for c in chunks),
            return_exceptions=False,
        )
        new_facts: list[MemoryFact] = [f for sub in per_chunk for f in sub]
        if not new_facts:
            return 0, 0

        reconcile = expertise.facts.reconcile if expertise is not None else True
        if reconcile:
            return await self._reconcile_facts(new_facts, namespace_id, expertise, degradations_out=degradations_out)

        # Fast path: no reconciliation, write everything as ADD.
        try:
            await self._get_storage().write_facts(new_facts, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("write_facts failed (skipping fact persistence): {}", exc, exc_info=True)
            if degradations_out is not None:
                _record_channel_degradation(
                    degradations_out,
                    component="chronicle.facts",
                    reason="write_facts_failed",
                    detail=f"{len(new_facts)} fact(s) dropped; persistence raised",
                    exc=exc,
                )
            return 0, 0
        logger.debug("Persisted {} memory facts across {} chunks (no reconcile)", len(new_facts), len(chunks))
        return len(new_facts), 0

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    @trace("khora.chronicle.remember")
    async def remember(
        self,
        content: str,
        namespace_id: UUID,
        *,
        title: str = "",
        source: str = "",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | str | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        external_id: str | None = None,
    ) -> RememberResult:
        """Store content in the memory engine.

        Uses the shared ingest pipeline for chunking, embedding, and entity
        extraction — the same pipeline VectorCypher uses, for maximum
        extraction quality.

        Args:
            content: Content to remember
            namespace_id: Target namespace UUID
            title: Optional title
            source: Optional source identifier
            metadata: Optional metadata dict
            skill_name: Extraction skill to use
            entity_types: Entity types to extract
            relationship_types: Relationship types to extract
            expertise: Optional expertise — an ``ExpertiseConfig``, a
                registered expertise name, or a YAML file path
            extraction_config_hash: Optional hash for change detection
            chunk_strategy: Override chunking strategy for this call
            chunk_size: Override target chunk size (in tokens) for this call.
                When None (default), uses the configured
                ``config.pipeline.chunk_size``.

        Returns:
            RememberResult with document_id and counts
        """
        expertise = _resolve_expertise(expertise)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Compute content checksum
        start = time.perf_counter()
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        timings["checksum_ms"] = (time.perf_counter() - start) * 1000

        storage = self._get_storage()

        # Dedup check. Scoped by caller-supplied identity (#1139): a checksum
        # hit on a document stored under a different external_id or
        # session_id is a distinct logical document, not a duplicate.
        start = time.perf_counter()
        session_id = _coerce_session_id_from_metadata(metadata)
        existing = await storage.get_document_by_checksum(namespace_id, checksum)
        timings["dedup_check_ms"] = (time.perf_counter() - start) * 1000

        if existing and _checksum_dedup_applies(existing, external_id=external_id, session_id=session_id):
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            logger.debug(f"Document already exists (checksum={checksum[:8]}..., status={existing.status})")
            return RememberResult(
                document_id=existing.id,
                namespace_id=namespace_id,
                chunks_created=existing.chunk_count,
                entities_extracted=existing.entity_count,
                relationships_created=existing.relationship_count,
                metadata={"duplicate": True, "status": str(existing.status), "timings": timings},
            )

        # Create document record
        start = time.perf_counter()
        document = Document(
            namespace_id=namespace_id,
            content=content,
            title=title or None,
            source=source or None,
            source_type=source_type,
            source_name=source_name or None,
            source_url=source_url or None,
            source_timestamp=source_timestamp,
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            metadata=dict(metadata or {}),
            extraction_config_hash=extraction_config_hash,
            external_id=external_id,
            session_id=session_id,
        )
        document = await storage.create_document(document)
        timings["document_create_ms"] = (time.perf_counter() - start) * 1000

        # Process through shared ingest pipeline (chunking, embedding, extraction)
        from khora.pipelines.flows.ingest import process_document

        start = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            extraction_wave_size=self._config.llm.extraction_wave_size,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
            ketrag_skeleton_channel=self._config.pipeline.ketrag_skeleton_channel,
            extraction_second_pass=self._config.pipeline.extraction_second_pass,
        )
        if chunk_strategy is not None:
            kwargs["chunk_strategy"] = chunk_strategy
        # Issue #1426: fall back to the configured pipeline default (like
        # VectorCypher/Skeleton) instead of the pipeline's hardcoded 512.
        kwargs["chunk_size"] = chunk_size if chunk_size is not None else self._config.pipeline.chunk_size
        result = await process_document(document, storage, **kwargs)
        timings["pipeline_ms"] = (time.perf_counter() - start) * 1000

        # Chronicle #2/#3: extract SVO events + atomic facts from every
        # persisted chunk. Default-on per ExpertiseConfig with per-namespace
        # overrides via ``namespace.config_overrides[{"events"|"facts"}]``.
        # Both share the same chunk fetch and per-chunk semaphore.
        events_extracted = 0
        facts_extracted = 0
        reconcile_errors = 0
        # Issue #903 (ADR-001): transient extractor failures are appended here
        # and surfaced on the response under ``metadata["errors"]``.
        extraction_errors: list[ErrorRecord] = []
        # Issue #907 / #893 (ADR-001): partial-failure degradations surfaced on
        # ``metadata["degradations"]``. Initialized before the namespace fetch
        # so a failed ``get_namespace`` can record a non-silent degradation.
        ingest_degradations: list[Degradation] = []
        # Issue #893: a transient ``get_namespace`` failure must NOT silently
        # revert a namespace that set ``config_overrides["events"|"facts"]
        # ["enabled"] = False`` back to default-on. We still soft-fall to the
        # expertise/default path (default-on), but the dropped override is now
        # observable: WARNING + a Degradation on the result.
        try:
            namespace = await storage.get_namespace(namespace_id)
        except Exception as exc:
            namespace = None
            logger.warning(
                "get_namespace failed for namespace_id={}: {} - per-namespace "
                "event/fact overrides may be dropped, falling back to default-on",
                namespace_id,
                exc,
                exc_info=True,
            )
            _record_channel_degradation(
                ingest_degradations,
                component="chronicle.namespace_overrides",
                reason="get_namespace_failed",
                detail="per-namespace event/fact overrides could not be resolved; fell back to expertise/default (default-on)",
                exc=exc,
            )

        run_events = self._events_enabled(namespace, expertise)
        run_facts = self._facts_enabled(namespace, expertise)

        if run_events or run_facts:
            chunk_ids = result.get("chunk_ids", []) or []
            if chunk_ids:
                chunks_map = await storage.get_chunks_batch(list(chunk_ids), namespace_id=namespace_id)
                chunks = [chunks_map[cid] for cid in chunk_ids if cid in chunks_map]
                if run_events:
                    start = time.perf_counter()
                    events_extracted = await self._extract_and_persist_events(
                        chunks,
                        namespace_id,
                        expertise,
                        errors_out=extraction_errors,
                        degradations_out=ingest_degradations,
                    )
                    timings["event_extraction_ms"] = (time.perf_counter() - start) * 1000
                if run_facts:
                    start = time.perf_counter()
                    facts_extracted, reconcile_errors = await self._extract_and_persist_facts(
                        chunks,
                        namespace_id,
                        expertise,
                        errors_out=extraction_errors,
                        degradations_out=ingest_degradations,
                    )
                    timings["fact_extraction_ms"] = (time.perf_counter() - start) * 1000

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"remember() completed: {result['chunks']} chunks, {result['entities']} entities, "
            f"{result['relationships']} relationships, {events_extracted} events, "
            f"{facts_extracted} facts in {timings['total_ms']:.1f}ms"
        )

        # Issue #907: surface unremappable-relationship drops from the
        # shared ingest pipeline. ``process_document`` returns
        # ``relationships_skipped`` as a top-level key; we propagate it to
        # RememberResult and append a Degradation entry under ADR-001 so
        # callers can detect partial success without changing the signature.
        rels_skipped = int(result.get("relationships_skipped", 0) or 0)
        if rels_skipped > 0:
            ingest_degradations.append(
                Degradation(
                    component="ingest.relationships",
                    reason="unremappable",
                    detail=f"{rels_skipped} relationship(s) dropped due to missing entity mappings",
                    exception=None,
                )
            )

        metadata: dict[str, Any] = {
            "timings": timings,
            "events_extracted": events_extracted,
            "facts_extracted": facts_extracted,
            # Issue #892: per-call signal that fact reconciliation dropped
            # facts due to transient LLM failures. 0 on the happy path.
            "reconcile_errors": reconcile_errors,
        }
        if extraction_errors:
            metadata["errors"] = list(extraction_errors)
        if ingest_degradations:
            metadata["degradations"] = ingest_degradations

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
            relationships_skipped=rels_skipped,
            metadata=metadata,
        )

    @trace("khora.chronicle.recall")
    async def recall(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        temporal_filter: Any | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> RecallResult:
        """Recall memories using 4-channel parallel retrieval with RRF fusion.

        Phase 1 channels:
          1. Semantic (vector similarity via pgvector)
          2. BM25 (PostgreSQL full-text search)
          3. Temporal - stubbed, returns empty (Phase 2)
          4. Entity - stubbed, returns empty (Phase 2)

        Results are fused via Reciprocal Rank Fusion and then re-scored
        with Ebbinghaus temporal decay.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum results
            mode: Search mode (VECTOR, KEYWORD, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            temporal_filter: Reserved for Phase 2 temporal filtering
            recency_bias: Override temporal decay weight (0.0-1.0)
            filter_ast: Canonical recall-filter AST. When supplied,
                Chronicle pushes the conjunctive ``occurred_at`` / ``created_at``
                date bounds into the recency window (narrow only) and post-filters
                every retrieved chunk against the full filter — the eight
                denormalized document keys and all metadata — before top-k. When
                ``None``, the recency behavior is unchanged.

        Returns:
            RecallResult with fused and decay-scored chunks
        """
        # #833: validate the mode contract before doing any storage work.
        if mode not in self.supported_modes:
            raise EngineCapabilityError("chronicle", mode, self.supported_modes)

        storage = self._get_storage()
        embedder = self._get_embedder()
        timings: dict[str, float] = {}
        # ADR-001: silent channel failures / fallbacks are appended here and
        # surfaced on the response under ``engine_info["degradations"]``.
        degradations: list[Degradation] = []
        total_start = time.perf_counter()

        # Read Chronicle-specific config from QuerySettings (with safe defaults)
        qs = getattr(self._config, "query", None)
        _overfetch = getattr(qs, "chronicle_overfetch_multiplier", 4) if qs else 4
        _rrf_w_semantic = getattr(qs, "chronicle_rrf_semantic_weight", 1.0) if qs else 1.0
        _rrf_w_bm25 = getattr(qs, "chronicle_rrf_bm25_weight", 0.8) if qs else 0.8
        _rrf_w_temporal = getattr(qs, "chronicle_rrf_temporal_weight", 0.9) if qs else 0.9
        _rrf_w_entity = getattr(qs, "chronicle_rrf_entity_weight", 0.85) if qs else 0.85
        _cfg_decay = (
            getattr(qs, "chronicle_decay_weight", DEFAULT_CHRONICLE_DECAY_WEIGHT)
            if qs
            else DEFAULT_CHRONICLE_DECAY_WEIGHT
        )
        _cfg_half_life = (
            getattr(qs, "temporal_half_life_hours", DEFAULT_CHRONICLE_HALF_LIFE_HOURS)
            if qs
            else DEFAULT_CHRONICLE_HALF_LIFE_HOURS
        )
        _enable_reinforcement = getattr(qs, "chronicle_enable_recall_reinforcement", False) if qs else False
        overfetch_limit = limit * _overfetch

        # Resolve temporal references from query (fast dateparser, ~0.25ms)
        # when the caller didn't supply an explicit temporal_filter.
        _enable_resolver = getattr(qs, "enable_temporal_resolver", True) if qs else True
        if _enable_resolver and temporal_filter is None:
            from khora.query.temporal import TemporalFilter
            from khora.query.temporal_detection import TemporalDetector
            from khora.query.temporal_resolver import TemporalResolver

            # Gate the date parser on temporal intent first (matches
            # VectorCypher). Without this, resolve_fast's bare-year regex
            # (``20\d{2}``) treats an incidental year-like token in a
            # non-temporal query (a version, a room/model number) as a date and
            # silently narrows every channel, dropping older results (#1222).
            temporal_signal = TemporalDetector().detect(query)
            resolved = TemporalResolver().resolve_fast(query) if temporal_signal.is_temporal else None
            if resolved and resolved.confidence > 0.5 and (resolved.start or resolved.end):
                temporal_filter = TemporalFilter(
                    start_time=resolved.start,
                    end_time=resolved.end,
                )
                logger.debug(
                    "Temporal resolver: {!r} -> {} to {} (confidence={:.2f})",
                    resolved.expression,
                    resolved.start,
                    resolved.end,
                    resolved.confidence,
                )

        # ── Query routing (Chronicle #6) ──────────────────────────────
        # Classify SIMPLE / MODERATE / COMPLEX. SIMPLE chronicle queries skip
        # BM25 + entity channels (~50ms each) but ALWAYS keep the temporal
        # channel — temporal scoring is chronicle's differentiator. Temporal
        # signal is wired into the routing decision so dateparser-resolved
        # temporal queries are forced to MODERATE (matches vectorcypher).
        # Failures fall back to running all 4 channels (no behavioural skip).
        run_bm25 = True
        run_entity = True
        force_temporal = temporal_filter is not None
        routing_complexity: str = "disabled"
        if self._router_enabled:
            try:
                from khora.query.temporal_detection import TemporalCategory, TemporalSignal

                temporal_signal = TemporalSignal(
                    is_temporal=force_temporal,
                    category=TemporalCategory.EXPLICIT if force_temporal else TemporalCategory.NONE,
                    confidence=0.9 if force_temporal else 1.0,
                    source="resolver" if force_temporal else "none",
                    temporal_filter=temporal_filter,
                )
                routing = await self._router.route(query, temporal_signal=temporal_signal)
                routing_complexity = routing.complexity.value
                if routing.complexity == QueryComplexity.SIMPLE:
                    run_bm25 = False
                    run_entity = False
                elif routing.complexity == QueryComplexity.ENTITY_ANCHORED:
                    # Boost the entity-channel RRF weight 2× for queries pivoting
                    # on a single named entity. Other channels keep their weights;
                    # the entity surface just gets a stronger pull during fusion.
                    _rrf_w_entity *= 2.0
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Chronicle router failed, running all channels: {}", exc)
                routing_complexity = "fallback"

        # Extract temporal bounds once and forward them to every channel.
        # WHY: previously only the temporal channel honored these bounds,
        # so a 20-day-old chunk could leak through a 7-day window via the
        # semantic / BM25 / entity channels. Pgvector's search_similar and
        # search_fulltext already pushdown COALESCE(source_timestamp,
        # created_at) when these are set; we just need to pass them in.
        created_after, created_before = _extract_temporal_bounds(temporal_filter)

        # ── Deterministic recall filter ──────────────────────────────
        # When a filter AST is supplied, compile it two ways:
        #   1. compile_chronicle distills the conjunctive source_timestamp clauses
        #      into a narrowing date-bound we intersect with the recency window
        #      above (narrow only — max of lowers, min of uppers — never widening
        #      it). Only source_timestamp pushes down: it is the window's own
        #      primary axis COALESCE(source_timestamp, created_at), so the pushdown
        #      is same-axis and superset-safe. occurred_at (the event-time axis)
        #      and created_at (the window's fallback) are post-filter-only.
        #   2. compile_python compiles the FULL AST to an in-memory predicate that
        #      post-filters the retrieved chunk candidates (each mapped via
        #      _chunk_to_record) — the safety net that enforces every predicate
        #      (occurred_at as COALESCE(occurred_at, source_timestamp), created_at,
        #      source_timestamp, the eight denormalized document keys, metadata)
        #      against the field each record carries. Pushdown is only a
        #      candidate-narrowing optimization on top.
        # When filter_ast is None, both stay None and the recency behavior is
        # unchanged.
        post_filter: Callable[[Any], bool] | None = None
        # Date keys the date-bound actually pushed into the recency window, for the
        # honest engine_info["filter"] report below. Chronicle is partial-pushdown
        # by design (only source_timestamp pushes down; the full filter is always
        # enforced by the post-filter), so this is the source_timestamp subset, not the
        # whole filter.
        filter_pushed_keys: frozenset[str] = frozenset()
        if filter_ast is not None:
            from khora.filter.execute import plan_chronicle_filter

            plan = plan_chronicle_filter(filter_ast)
            date_bound = plan.date_bound
            filter_pushed_keys = plan.pushed_keys
            created_after = _intersect_lower(created_after, date_bound.created_after)
            created_before = _intersect_upper(created_before, date_bound.created_before)
            post_filter = plan.post_filter

        # ── Phase 1: Embed query + BM25 in parallel ───────────────────
        # BM25 needs only the query text (no embedding), so start it
        # concurrently with embedding to save one round-trip of latency.
        # Skipped entirely when the router classifies SIMPLE.
        query_embedding: list[float] | None = None
        bm25_results: list[tuple[Chunk, float]] = []

        bm25_task: asyncio.Task[list[tuple[Chunk, float]]] | None = None
        if run_bm25 and mode in (SearchMode.HYBRID, SearchMode.ALL):
            bm25_task = asyncio.create_task(
                storage.search_fulltext_chunks(
                    namespace_id,
                    query,
                    limit=overfetch_limit,
                    created_after=created_after,
                    created_before=created_before,
                )
            )

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            start = time.perf_counter()
            query_embedding = await embedder.embed(query)
            timings["embed_ms"] = (time.perf_counter() - start) * 1000

        # Collect BM25 result (already running in background)
        if bm25_task is not None:
            start = time.perf_counter()
            try:
                bm25_results = await bm25_task
            except RuntimeError as exc:
                # Fulltext backend not configured - the channel is opted out,
                # not failing. Still surface as a degradation so callers can
                # see why BM25 contributed nothing.
                logger.warning(
                    "Chronicle BM25 channel unavailable: fulltext backend not configured ({})",
                    exc,
                )
                _record_channel_degradation(
                    degradations,
                    component="chronicle.bm25",
                    reason="fulltext_backend_unavailable",
                    detail=str(exc),
                    exc=exc,
                )
            except Exception as exc:
                logger.warning("Chronicle BM25 channel failed: {}", exc, exc_info=True)
                _record_channel_degradation(
                    degradations,
                    component="chronicle.bm25",
                    reason="channel_exception",
                    detail=str(exc),
                    exc=exc,
                )
            timings["bm25_ms"] = (time.perf_counter() - start) * 1000

        # ── Phase 2: Semantic + Temporal + Entity in parallel ───────────
        # All three need the embedding, so they run after Phase 1 completes.
        semantic_results: list[tuple[Chunk, float]] = []
        temporal_results: list[tuple[Chunk, float]] = []
        entity_results: list[tuple[Chunk, float]] = []
        # Side-channel accumulators populated by _temporal_channel and
        # _entity_channel — read after gather to build RecallResult.entities.
        temporal_subject_scores: dict[str, float] = {}
        entity_channel_hits: dict[UUID, tuple[Entity, float]] = {}

        # chronicle_temporal_window_days semantics:
        #   -1  = disable temporal channel entirely
        #    0  = unlimited window (search ALL data with recency-primary scoring)
        #   >0  = N-day window filter
        # NOTE: temporal channel always runs when enabled (window >= 0).
        # Conditional skipping based on detect_temporal_category was reverted
        # because it caused paraphrase instability — different code paths for
        # semantically equivalent queries.
        _temporal_window_days = getattr(qs, "chronicle_temporal_window_days", 0.0) if qs else 0.0
        run_temporal = mode in (SearchMode.HYBRID, SearchMode.ALL) and (
            _temporal_window_days >= 0 or temporal_filter is not None
        )

        channel_coros: list[tuple[str, Any]] = []

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
            channel_coros.append(
                (
                    "semantic",
                    storage.search_similar_chunks(
                        namespace_id,
                        query_embedding,
                        limit=overfetch_limit,
                        min_similarity=min_similarity,
                        created_after=created_after,
                        created_before=created_before,
                    ),
                )
            )

        if run_temporal and query_embedding is not None:
            channel_coros.append(
                (
                    "temporal",
                    self._temporal_channel(
                        namespace_id,
                        query,
                        query_embedding,
                        overfetch_limit,
                        temporal_filter,
                        subject_scores=temporal_subject_scores,
                        degradations=degradations,
                    ),
                )
            )

        if run_entity and mode in (SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
            channel_coros.append(
                (
                    "entity",
                    self._entity_channel(
                        namespace_id,
                        query,
                        query_embedding,
                        overfetch_limit,
                        entity_hits=entity_channel_hits,
                        created_after=created_after,
                        created_before=created_before,
                        degradations=degradations,
                    ),
                )
            )

        if channel_coros:
            start = time.perf_counter()
            gathered = await asyncio.gather(
                *[coro for _name, coro in channel_coros],
                return_exceptions=True,
            )
            elapsed = (time.perf_counter() - start) * 1000

            for (name, _coro), result in zip(channel_coros, gathered):
                if isinstance(result, BaseException):
                    logger.warning(
                        "Chronicle {} channel failed: {}",
                        name,
                        result,
                        exc_info=(type(result), result, result.__traceback__),
                    )
                    _record_channel_degradation(
                        degradations,
                        component=f"chronicle.{name}",
                        reason="channel_exception",
                        detail=str(result),
                        exc=result,
                    )
                    timings[f"{name}_ms"] = elapsed
                    continue
                timings[f"{name}_ms"] = elapsed
                if name == "semantic":
                    semantic_results = result
                elif name == "temporal":
                    temporal_results = result
                elif name == "entity":
                    entity_results = result

        # Capture max raw cosine similarity for abstention signals
        max_raw_cosine = max((score for _, score in semantic_results), default=0.0) if semantic_results else 0.0

        # ── Fusion via weighted RRF with per-channel score normalization ─
        # Channels skipped by the router contribute empty lists; the helper
        # ignores them. Per-channel min-max normalisation neutralises the
        # BM25-vs-cosine score-scale mismatch (Chronicle #6).
        start = time.perf_counter()

        ranked_lists: dict[str, list[tuple[Chunk, float]]] = {}
        weights: dict[str, float] = {}

        if semantic_results:
            ranked_lists["semantic"] = semantic_results
            weights["semantic"] = _rrf_w_semantic
        if bm25_results:
            ranked_lists["bm25"] = bm25_results
            weights["bm25"] = _rrf_w_bm25
        if temporal_results:
            ranked_lists["temporal"] = temporal_results
            weights["temporal"] = _rrf_w_temporal
        if entity_results:
            ranked_lists["entity"] = entity_results
            weights["entity"] = _rrf_w_entity

        chunks_with_scores: list[tuple[Chunk, float]]
        if ranked_lists:
            chunks_with_scores = _weighted_normalized_rrf_multi(ranked_lists, weights)[:overfetch_limit]
        elif semantic_results:
            chunks_with_scores = semantic_results[:overfetch_limit]
        elif bm25_results:
            chunks_with_scores = bm25_results[:overfetch_limit]
        else:
            chunks_with_scores = []

        timings["fusion_ms"] = (time.perf_counter() - start) * 1000

        # ── Temporal decay scoring ───────────────────────────────────────
        start = time.perf_counter()
        decay_weight = recency_bias if recency_bias is not None else _cfg_decay
        chunks_with_scores = _apply_temporal_decay(
            chunks_with_scores,
            decay_weight=decay_weight,
            half_life_hours=_cfg_half_life,
            enable_reinforcement=_enable_reinforcement,
        )
        timings["decay_ms"] = (time.perf_counter() - start) * 1000

        # ── Version-aware scoring ───────────────────────────────────────
        # Penalize superseded document versions so the latest state
        # surfaces first.  Only fires when the query has temporal/state
        # intent and chunks carry version metadata.
        start = time.perf_counter()
        chunks_with_scores = _apply_version_scoring(chunks_with_scores, query)
        timings["version_scoring_ms"] = (time.perf_counter() - start) * 1000

        # ── Cross-encoder reranking (post-fusion) ───────────────────────
        _enable_reranking = getattr(qs, "enable_reranking", False) if qs else False
        if _enable_reranking and chunks_with_scores:
            start = time.perf_counter()
            _reranking_model = getattr(qs, "reranking_model", None) if qs else None
            _reranking_top_n = getattr(qs, "reranking_top_n", 30) if qs else 30

            # #866: capture per-chunk recency multipliers so we can reapply
            # decay AFTER rerank. The cross-encoder is timestamp-blind by
            # construction; without this reapplication the user-visible
            # ``chronicle_decay_weight`` knob has no effect on rerank-driven
            # ordering. Pattern matches Qdrant / Vespa / Elasticsearch.
            _recency_multipliers = _compute_recency_multipliers(
                [c for c, _ in chunks_with_scores],
                decay_weight=decay_weight,
                half_life_hours=_cfg_half_life,
                enable_reinforcement=_enable_reinforcement,
            )

            try:
                from khora.query.reranking import rerank_chunks

                reranked = await rerank_chunks(
                    query,
                    chunks_with_scores[:_reranking_top_n],
                    method="cross_encoder",
                    top_k=limit,
                    model=_reranking_model,
                )
                # Reapply decay multiplier on top of rerank output. The
                # reranker decides which chunks are on-topic; the decay
                # multiplier discounts older ones. Missing-key default of
                # 1.0 covers the ``decay_weight <= 0`` case (multipliers
                # dict is empty by design then).
                chunks_with_scores = [(c, score * _recency_multipliers.get(c.id, 1.0)) for c, score in reranked]
                chunks_with_scores.sort(key=lambda pair: pair[1], reverse=True)
            except Exception as e:
                logger.warning("Chronicle cross-encoder reranking failed: {}", e)
            timings["reranking_ms"] = (time.perf_counter() - start) * 1000

        # ── Cross-session expansion ─────────────────────────────────────
        start = time.perf_counter()
        chunks_with_scores = await self._cross_session_expand(
            chunks_with_scores, query, namespace_id, query_embedding, limit, degradations=degradations
        )
        timings["cross_session_ms"] = (time.perf_counter() - start) * 1000

        # ── Deterministic recall-filter post-filter ─────────────────────
        # Applied ONCE here — AFTER cross-session expansion (which fetches
        # entity-source chunks via get_chunks_batch that bypass the channel-level
        # recency window) and BEFORE the final top-k trim — so the filter is the
        # exact final narrowing force over EVERY candidate path. The date-bound
        # was already pushed into the recency window for the channel reads above;
        # this predicate covers the denormalized document keys + metadata
        # Chronicle cannot push down, and re-checks the date bounds on
        # window-bypassing expansion chunks.
        #
        # Doc-key hydration: when the filter constrains one of the seven
        # denormalized document keys (and only then), batch-fetch the per-document
        # DocumentProjection so the post-filter resolves those keys instead of
        # treating them as absent. The common case (no filter, recency-only,
        # occurred_at, or metadata-only filter) short-circuits and pays ZERO extra
        # query. One fetch per recall, not N+1.
        if post_filter is not None:
            from khora.filter.execute import filter_leaf_keys

            projections: dict[UUID, DocumentProjection] = {}
            needs_doc_keys = filter_ast is not None and bool(filter_leaf_keys(filter_ast) & set(_DOC_PROJECTION_KEYS))
            if needs_doc_keys and chunks_with_scores:
                doc_ids = list({chunk.document_id for chunk, _ in chunks_with_scores})
                try:
                    projections = await storage.get_document_projections_batch(doc_ids, namespace_id=namespace_id)
                except Exception as exc:
                    # ADR-001: hydration is best-effort. On failure the post-filter
                    # still runs with doc keys absent for every chunk (positive
                    # doc-key predicates return empty) — recall never crashes.
                    logger.warning(
                        "Chronicle doc-key hydration failed; doc-key predicates will treat keys as absent: {}",
                        exc,
                        exc_info=True,
                    )
                    _record_channel_degradation(
                        degradations,
                        component="chronicle.doc_hydration",
                        reason="projection_fetch_failed",
                        detail=str(exc),
                        exc=exc,
                    )
                    projections = {}

            start = time.perf_counter()
            chunks_with_scores = [
                pair
                for pair in chunks_with_scores
                if post_filter(_chunk_to_record(pair[0], projections.get(pair[0].document_id)))
            ]
            timings["post_filter_ms"] = (time.perf_counter() - start) * 1000

        # Trim to requested limit
        chunks_with_scores = chunks_with_scores[:limit]

        # ── Surface entity hits (Chronicle #5) ──────────────────────────
        start = time.perf_counter()
        entity_hits = await self._collect_entities(
            namespace_id=namespace_id,
            entity_channel_hits=entity_channel_hits,
            temporal_event_subjects=temporal_subject_scores,
            limit=self._entity_limit,
        )
        timings["entity_collect_ms"] = (time.perf_counter() - start) * 1000

        # ── Abstention signals ───────────────────────────────
        # Passive metadata for downstream consumers (LLM answer-generation)
        # to decide whether to refuse vs answer.  Does NOT alter retrieval.
        # ``top_vector_score`` is the pre-rerank raw cosine captured above
        # (NOT ``chunks_with_scores[0][1]``, which is the post-fusion /
        # post-rerank display score) - cross-encoder reranking compresses
        # scores into a narrow high-side band even for off-topic queries,
        # so feeding the post-rerank value would make ``top_score_low``
        # a steady-state false negative on rerank-enabled queries (#809).
        abstention_signals = self._compute_abstention_signals(
            chunks_with_scores, entity_hits, top_vector_score=max_raw_cosine
        )
        # Calibrated retrieval confidence (#1331). Inputs are absolute cosines
        # (#1319): the top raw semantic cosine and the gap to the runner-up.
        # Use the raw semantic channel, not the post-rerank fused scores, for
        # the same reason top_score_low does (#809).
        _raw_cosines = sorted((score for _, score in semantic_results), reverse=True)
        _top_gap = (_raw_cosines[0] - _raw_cosines[1]) if len(_raw_cosines) >= 2 else 0.0
        confidence = compute_confidence(
            top_cosine=max_raw_cosine,
            top_score_gap=_top_gap,
            target_cosine=self._abstention_confidence_target_cosine,
            target_gap=self._abstention_confidence_target_gap,
        )

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"recall() completed: {len(chunks_with_scores)} chunks, {len(entity_hits)} entities "
            f"(semantic={len(semantic_results)}, bm25={len(bm25_results)}, "
            f"temporal={len(temporal_results)}, entity={len(entity_results)}) "
            f"in {timings['total_ms']:.1f}ms"
        )

        # #834: ``RecallChunk.score`` is a min-max normalized rank in [0, 1]
        # across all engines. The raw post-rerank fused score (cross-encoder +
        # temporal decay + version + RRF) lives on an arbitrary scale; min-max
        # collapses it to the documented top=1.0 / bottom=0.0 shape. Normalize
        # AFTER cross-session expansion and the final ``[:limit]`` trim so the
        # top of the returned set gets 1.0, not some pre-expansion intermediate.
        normalized_chunk_scores = min_max_normalize([s for _, s in chunks_with_scores])
        recall_chunks = [
            RecallChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score,
                created_at=chunk.created_at,
                occurred_at=(chunk.occurred_at if chunk.occurred_at is not None else chunk.source_timestamp),
                chunker_info=chunk.chunker_info or {},
            )
            for (chunk, _), score in zip(chunks_with_scores, normalized_chunk_scores, strict=False)
        ]

        recall_entities = [
            RecallEntity(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description or "",
                score=score,
                attributes=dict(entity.attributes or {}),
                mention_count=entity.mention_count or 0,
                source_document_ids=list(entity.source_document_ids),
                source_chunk_ids=list(entity.source_chunk_ids),
            )
            for entity, score in entity_hits
        ]

        # Document stubs — fuller projections land with the recall-method rewrite.
        seen_doc_ids: set[UUID] = set()
        documents: list[DocumentProjection] = []
        for chunk, _ in chunks_with_scores:
            if chunk.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(chunk.document_id)
            src = chunk.source_document
            documents.append(
                DocumentProjection(
                    id=chunk.document_id,
                    created_at=chunk.created_at,
                    source_type=(src.source_type if src and src.source_type else "library"),
                    title=(src.title if src and src.title else None),
                    source=(src.source if src and src.source else None),
                    source_timestamp=(src.source_timestamp if src else None),
                    metadata=dict(chunk.metadata or {}),
                )
            )
        # Producer invariant: every id in entities[i].source_document_ids must
        # appear in documents[]. Add stubs for entity-referenced docs not
        # already covered by the chunk loop above.
        for re_ in recall_entities:
            for did in re_.source_document_ids:
                if did in seen_doc_ids:
                    continue
                seen_doc_ids.add(did)
                documents.append(
                    DocumentProjection(
                        id=did,
                        created_at=datetime.now(UTC),
                        source_type="library",
                    )
                )

        # ── Reinforcement on recall (#855) ──────────────────────────────
        # Fire-and-forget: stamp ``last_accessed_at = now`` on the chunks
        # we're about to return so the next decay pass treats them as
        # fresh. Wrapped in a task so the recall response returns
        # immediately; failures log a warning but never fail recall.
        if _enable_reinforcement and chunks_with_scores:
            reinforce_ids = [c.id for c, _ in chunks_with_scores]
            reinforce_ts = datetime.now(UTC)
            task = asyncio.create_task(
                self._reinforce_last_accessed(storage, namespace_id, reinforce_ids, reinforce_ts)
            )
            # Hold a live reference so the GC can't drop the task mid-flight
            # (CPython collects asyncio tasks with no strong reference -
            # see https://docs.python.org/3/library/asyncio-task.html#creating-tasks).
            self._reinforce_tasks.add(task)
            task.add_done_callback(self._reinforce_tasks.discard)

        # ── Honest recall-filter pushdown report ────────────────────────
        # Chronicle is partial-pushdown by design: only the source_timestamp
        # date bound pushes into the recency window; the full filter
        # (occurred_at / created_at / denorm doc keys / metadata) is always
        # enforced by the in-memory post-filter. We feed both halves of that
        # truth into the canonical builder as a single "chunks" channel so the
        # report exposes which leaves pushed (pushed_keys) versus which were
        # re-checked in memory (post_filtered_keys), rather than a single
        # all-or-nothing flag.
        from khora.filter.execute import filter_leaf_keys
        from khora.filter.report import ChannelPlan, build_filter_report

        filter_report = build_filter_report(
            filter_ast,
            {
                "chunks": ChannelPlan(
                    pushed_keys=filter_pushed_keys,
                    post_filtered_keys=(
                        filter_leaf_keys(filter_ast) - filter_pushed_keys if filter_ast is not None else frozenset()
                    ),
                    defensive_recheck=(filter_ast is not None and bool(filter_ast.children)),
                )
            },
        )

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            documents=documents,
            chunks=recall_chunks,
            entities=recall_entities,
            relationships=[],
            engine_info={
                "engine": "chronicle",
                "channels": {
                    "semantic": len(semantic_results),
                    "bm25": len(bm25_results),
                    "temporal": len(temporal_results),
                    "entity": len(entity_results),
                },
                "routing": routing_complexity,
                "decay_weight": decay_weight,
                "max_raw_vector_score": max_raw_cosine,
                "abstention_signals": abstention_signals,
                "confidence": round(confidence, 4),
                "timings": timings,
                # Honest recall-filter pushdown report. Chronicle is
                # partial-pushdown by design: only the source_timestamp date bound
                # pushes into the recency window, while the full filter is always
                # enforced by the in-memory post-filter — so the canonical
                # FilterPushdownReport (built above) carries the pushed_keys /
                # post_filtered_keys split rather than a single all-or-nothing flag.
                "filter": filter_report.model_dump(mode="json"),
                # ADR-001: callers can detect silent channel failures /
                # fallbacks via this list. Empty when nothing degraded.
                "degradations": degradations,
            },
        )

    async def _reinforce_last_accessed(
        self,
        storage: StorageCoordinator,
        namespace_id: UUID,
        chunk_ids: list[UUID],
        ts: datetime,
    ) -> None:
        """Best-effort UPDATE of ``last_accessed_at`` for the given chunks (#855).

        Runs in a fire-and-forget task spawned by ``recall``. Logs a
        warning on failure (DB down, network blip) but never raises -
        reinforcement loss is acceptable; breaking recall is not.
        Emits ``updates_total`` / ``failures_total`` / ``duration`` metrics
        so operators can see whether reinforcement is actually landing.
        """
        started = time.monotonic()
        try:
            await storage.update_last_accessed(namespace_id, chunk_ids, ts)
            _REINFORCE_UPDATES_COUNTER.add(1)
        except asyncio.CancelledError:
            # disconnect() cancelled us - don't count as a failure, just exit.
            raise
        except Exception as exc:
            _REINFORCE_FAILURES_COUNTER.add(1)
            logger.warning(
                "Chronicle reinforcement update failed for {} chunks: {}",
                len(chunk_ids),
                exc,
            )
        finally:
            _REINFORCE_DURATION_HISTOGRAM.record(time.monotonic() - started)

    def _compute_abstention_signals(
        self,
        chunks: list[tuple[Chunk, float]],
        entities: list[tuple[Entity, float]],
        *,
        top_vector_score: float,
    ) -> dict[str, Any]:
        """Compute passive abstention signals for downstream answer-generation.

        ``top_vector_score`` is the pre-rerank, pre-fusion raw cosine of
        the top semantic-channel chunk. The post-fusion ``chunks[0][1]``
        score is unfit for the ``top_score_low`` check because cross-
        encoder reranking compresses scores into a narrow high-side band
        regardless of query relevance (#809).

        See ``ChronicleEngine.__init__`` for the tunable thresholds.
        """
        signals = compute_abstention_signals(
            chunk_count=len(chunks),
            top_vector_score=top_vector_score,
            entity_count=len(entities),
            min_chunks=self._abstention_min_chunks,
            min_top_score=self._abstention_min_top_score,
            combined_threshold=self._abstention_combined_threshold,
            weight_entities_empty=self._abstention_weight_entities_empty,
            weight_chunks_below_min=self._abstention_weight_chunks_below_min,
            weight_top_score_low=self._abstention_weight_top_score_low,
            mode=self._abstention_mode,
        )

        if signals["entities_empty"]:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "entities_empty"})
        if signals["chunks_empty"]:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_empty"})
        if signals["chunks_below_min"]:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_below_min"})
        if signals["top_score_low"]:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "top_score_low"})
        _ABSTENTION_COMBINED_SCORE_HISTOGRAM.record(signals["combined_score"])

        return signals

    # ------------------------------------------------------------------
    # Retrieval channels (Phase 2)
    # ------------------------------------------------------------------

    async def _temporal_channel(
        self,
        namespace_id: UUID,
        query: str,
        query_embedding: list[float] | None,
        limit: int,
        temporal_filter: Any | None,
        subject_scores: dict[str, float] | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Channel 3: Time-scoped chunk retrieval.

        Two paths share this channel:

        1. **Events path (default)**: queries ``chronicle_events`` filtered by
           ``referenced_date`` (the date the source text refers to, NOT ingest
           time). Each candidate event is scored by a blend of cosine similarity
           between its SVO summary embedding and the query embedding, and a
           temporal-proximity score against the query's temporal window.
           Events are deduped by ``chunk_id`` (max score per chunk) and the
           output unit of the channel remains a chunk.
        2. **Legacy chunk-fallback path**: kicks in when (a) the events path is
           disabled via ``temporal_use_events=False``, (b) there is no temporal
           signal at all (no filter, no configured window), (c) the events
           table returns zero candidates for the namespace, or (d) the events
           query raises. Runs ``search_similar_chunks`` with ``created_at``
           bounds and applies an Ebbinghaus 72h decay — preserves the original
           behaviour pre Chronicle #4.

        When ``subject_scores`` is provided, the events path also records the
        subject of each scored event there (max combined score wins per
        subject). The recall fusion site uses these to surface entities
        mentioned in temporally-relevant chunks via
        ``RecallResult.entities``. Only the events path populates this — the
        chunk-fallback path operates without per-event subjects.
        """
        # Extract bounds once — both paths use them.
        start, end = _extract_temporal_bounds(temporal_filter)
        has_signal = start is not None or end is not None

        if not self._temporal_use_events or not has_signal:
            logger.debug(
                "temporal channel: chunk_fallback (use_events={}, has_signal={})",
                self._temporal_use_events,
                has_signal,
            )
            # No degradation: this is the declared default routing (no temporal
            # signal in query, or events path disabled by config) - not a failure.
            return await self._temporal_channel_chunks_fallback(
                namespace_id, query_embedding, limit, temporal_filter, degradations=degradations
            )

        storage = self._get_storage()

        # Over-fetch events so re-ranking has more headroom.
        try:
            events = await storage.query_events(
                namespace_id,
                since=start,
                until=end,
                limit=max(limit * 4, 1),
            )
        except Exception as exc:
            # The events path is the chronicle differentiator; falling back
            # to chunk-similarity here materially changes the result set
            # ordering (no SVO scoring, no referenced_date proximity). Surface
            # the demotion via the ADR-001 convention.
            logger.warning(
                "Chronicle temporal channel: query_events failed ({}); falling back to chunks",
                exc,
                exc_info=True,
            )
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.temporal_channel",
                    reason="events_query_failed",
                    detail=str(exc),
                    exc=exc,
                )
            return await self._temporal_channel_chunks_fallback(
                namespace_id, query_embedding, limit, temporal_filter, degradations=degradations
            )

        if not events:
            # No events ever ingested for this namespace - expected on
            # cold-start; do NOT record a degradation (false positive churn).
            logger.debug("temporal channel: no events for namespace; falling back to chunks")
            return await self._temporal_channel_chunks_fallback(
                namespace_id, query_embedding, limit, temporal_filter, degradations=degradations
            )

        logger.debug("temporal channel: events ({} candidates in scope)", len(events))

        # Pre-compute event-summary cosines in one batched call.
        cosine_by_index: dict[int, float] = {}
        if query_embedding is not None:
            indexed_embeddings: list[tuple[int, list[float]]] = [
                (i, list(ev.embedding)) for i, ev in enumerate(events) if getattr(ev, "embedding", None) is not None
            ]
            if indexed_embeddings:
                try:
                    from khora._accel import batch_cosine_similarity

                    sims = batch_cosine_similarity(
                        query_embedding,
                        [emb for _, emb in indexed_embeddings],
                    )
                    # batch_cosine_similarity returns (local_idx, score) pairs.
                    for local_idx, score in sims:
                        original_idx = indexed_embeddings[local_idx][0]
                        cosine_by_index[original_idx] = float(score)
                except Exception as exc:
                    # No-cosine path materially changes scoring: events fall
                    # back to proximity-only ranking. Surface via ADR-001.
                    logger.warning(
                        "Chronicle temporal channel: cosine batch failed ({}); falling back to proximity-only scoring",
                        exc,
                        exc_info=True,
                    )
                    if degradations is not None:
                        _record_channel_degradation(
                            degradations,
                            component="chronicle.temporal_channel",
                            reason="cosine_batch_failed",
                            detail=str(exc),
                            exc=exc,
                        )

        cw = self._temporal_event_cosine_weight
        tw = 1.0 - cw

        # Pick max combined score per chunk_id.
        per_chunk_score: dict[UUID, float] = {}
        for idx, ev in enumerate(events):
            chunk_id = getattr(ev, "chunk_id", None)
            if chunk_id is None:
                continue

            cosine = cosine_by_index.get(idx)
            ref_date = getattr(ev, "referenced_date", None)
            proximity = _temporal_proximity(ref_date, start, end)

            if cosine is None and proximity is None:
                # No usable signal on this event — skip entirely.
                if getattr(ev, "embedding", None) is None and ref_date is None:
                    logger.debug("temporal channel: event {} has neither embedding nor referenced_date", idx)
                continue

            if cosine is None:
                if getattr(ev, "embedding", None) is None:
                    logger.debug("temporal channel: event {} missing embedding; using proximity only", idx)
                combined = proximity if proximity is not None else 0.0
            elif proximity is None:
                logger.debug("temporal channel: event {} missing referenced_date; using cosine only", idx)
                combined = cosine
            else:
                combined = cosine * cw + proximity * tw

            prev = per_chunk_score.get(chunk_id)
            if prev is None or combined > prev:
                per_chunk_score[chunk_id] = combined

            # Stash the event subject (entity name) so recall() can resolve
            # it back to an Entity record. Max combined score wins per subject.
            if subject_scores is not None:
                subject = (getattr(ev, "subject", "") or "").strip()
                if subject:
                    prev_subj = subject_scores.get(subject)
                    if prev_subj is None or combined > prev_subj:
                        subject_scores[subject] = combined

        if not per_chunk_score:
            # Events exist but none had usable signals (no embedding, no
            # referenced_date). Fallback materially changes the result set:
            # surface as a degradation.
            logger.warning(
                "Chronicle temporal channel: {} events produced no usable scores; falling back to chunks",
                len(events),
            )
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.temporal_channel",
                    reason="events_no_usable_signal",
                    detail=f"event_count={len(events)}",
                )
            return await self._temporal_channel_chunks_fallback(
                namespace_id, query_embedding, limit, temporal_filter, degradations=degradations
            )

        # Hydrate the top-scoring chunks. We only need ``limit`` of them.
        top_chunk_ids = sorted(per_chunk_score, key=lambda cid: per_chunk_score[cid], reverse=True)[:limit]
        try:
            chunks_map = await storage.get_chunks_batch(top_chunk_ids, namespace_id=namespace_id)
        except Exception as exc:
            # Hard failure: events scored, but we cannot hydrate the chunks.
            # Channel returns [] - the fused result loses the temporal signal
            # entirely. Surface via ADR-001.
            logger.warning(
                "Chronicle temporal channel: get_chunks_batch failed ({}); channel returns empty",
                exc,
                exc_info=True,
            )
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.temporal_channel",
                    reason="chunk_fetch_failed",
                    detail=str(exc),
                    exc=exc,
                )
            return []

        scored: list[tuple[Chunk, float]] = [
            (chunks_map[cid], per_chunk_score[cid]) for cid in top_chunk_ids if cid in chunks_map
        ]
        # Already in score order (top_chunk_ids is sorted), but be defensive.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    async def _temporal_channel_chunks_fallback(
        self,
        namespace_id: UUID,
        query_embedding: list[float] | None,
        limit: int,
        temporal_filter: Any | None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Legacy temporal channel: chunks scoped by ``created_at`` + Ebbinghaus decay.

        Preserved as the fallback path for Chronicle #4 — see ``_temporal_channel``
        for the routing rules. Body unchanged from the pre-#4 implementation.
        """
        storage = self._get_storage()

        # Determine time bounds from temporal_filter or default to recent
        created_after = None
        created_before = None
        if temporal_filter is not None:
            created_after = getattr(temporal_filter, "occurred_after", None) or getattr(
                temporal_filter, "start_time", None
            )
            created_before = getattr(temporal_filter, "occurred_before", None) or getattr(
                temporal_filter, "end_time", None
            )
            if created_after is None and created_before is None:
                logger.debug(
                    "temporal_filter provided but no time bounds extracted "
                    "(expected occurred_after/occurred_before or start_time/end_time); "
                    "using default temporal window"
                )

        if created_after is None and created_before is None:
            # Use configurable temporal window (0 = unlimited — let decay handle scoring)
            qs = getattr(self._config, "query", None)
            window_days = getattr(qs, "chronicle_temporal_window_days", 0.0) if qs else 0.0
            if window_days > 0:
                from datetime import timedelta

                created_after = datetime.now(UTC) - timedelta(days=window_days)

        # Use semantic search with temporal bounds
        if query_embedding is None:
            return []

        try:
            results = await storage.search_similar_chunks(
                namespace_id,
                query_embedding,
                limit=limit,
                created_after=created_after,
                created_before=created_before,
            )
        except Exception as exc:
            # Chunk-fallback path failed too. The temporal channel returns
            # nothing - fused result loses the entire temporal signal.
            logger.warning(
                "Chronicle temporal channel (chunk fallback): search_similar_chunks failed ({}); returning empty",
                exc,
                exc_info=True,
            )
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.temporal_channel",
                    reason="chunk_fallback_failed",
                    detail=str(exc),
                    exc=exc,
                )
            return []

        # Detect timestamp collapse: if all chunks were created within ~1 hour
        # (e.g., benchmark batch ingestion), recency scoring is pure noise.
        # Fall back to semantic-only scores so the channel doesn't pollute RRF.
        if results:
            times = []
            for chunk, _ in results:
                ct = getattr(chunk, "source_timestamp", None) or chunk.created_at
                if ct:
                    if ct.tzinfo is None:
                        ct = ct.replace(tzinfo=UTC)
                    times.append(ct.timestamp())
            if len(times) > 1:
                import statistics

                if statistics.stdev(times) < 3600:  # < 1 hour spread
                    return results  # Pure semantic scores

        # Balanced scoring: 60% semantic, 40% recency — gives RRF
        # a meaningfully different ranking without drowning out relevance.
        now = datetime.now(UTC)
        scored = []
        for chunk, sim in results:
            chunk_time = getattr(chunk, "source_timestamp", None) or chunk.created_at
            if chunk_time:
                if chunk_time.tzinfo is None:
                    chunk_time = chunk_time.replace(tzinfo=UTC)
                hours_old = max(0, (now - chunk_time).total_seconds() / 3600)
                recency_factor = _ebbinghaus_decay(hours_old, half_life_hours=72)  # 3-day half-life
                blended = sim * 0.6 + recency_factor * 0.4
            else:
                blended = sim
            scored.append((chunk, blended))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _entity_channel(
        self,
        namespace_id: UUID,
        query: str,
        query_embedding: list[float],
        limit: int,
        entity_hits: dict[UUID, tuple[Entity, float]] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Channel 4: Entity co-occurrence retrieval.

        Finds entities similar to the query, then retrieves chunks that
        mention those entities. Provides a "follow the entities" signal
        complementary to pure semantic search.

        When ``entity_hits`` is provided, the resolved Entity records are also
        recorded there (keyed by entity id) along with their similarity score
        so the recall fusion site can surface them in ``RecallResult.entities``.
        """
        storage = self._get_storage()

        # Step 1: Find entities similar to the query
        try:
            entity_results = await storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=10,
            )
        except Exception as e:
            logger.warning("Entity channel: search_similar_entities failed: {}", e, exc_info=True)
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.entity",
                    reason="channel_exception",
                    detail=str(e),
                    exc=e,
                )
            return []

        if not entity_results:
            logger.debug("Entity channel: no similar entities found")
            return []

        logger.debug("Entity channel: found {} similar entities", len(entity_results))

        # Step 2: Get the source chunk IDs from matching entities
        entity_ids = [eid for eid, _score in entity_results]
        entity_scores = {eid: score for eid, score in entity_results}

        try:
            entities = await storage.get_entities_batch(entity_ids, namespace_id=namespace_id)
        except Exception as e:
            logger.warning(
                "Entity channel: get_entities_batch failed for {} IDs: {}", len(entity_ids), e, exc_info=True
            )
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.entity",
                    reason="channel_exception",
                    detail=str(e),
                    exc=e,
                )
            return []

        logger.debug("Entity channel: resolved {}/{} entities", len(entities), len(entity_ids))

        # Surface resolved entities for the recall fusion site so consumers
        # can see *which* entities the channel matched, not just the chunks
        # that mention them. Highest score wins on dedup.
        if entity_hits is not None:
            for eid, ent in entities.items():
                escore = entity_scores.get(eid, 0.0)
                prev = entity_hits.get(eid)
                if prev is None or escore > prev[1]:
                    entity_hits[eid] = (ent, escore)

        # Collect chunk IDs from entity sources, weighted by entity similarity
        chunk_scores: dict[UUID, float] = {}
        for eid, entity in entities.items():
            escore = entity_scores.get(eid, 0.0)
            for cid in entity.source_chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), escore)

        if not chunk_scores:
            logger.debug("Entity channel: no source chunks from matched entities")
            return []

        # Step 3: Fetch the actual chunks
        chunk_ids = sorted(chunk_scores, key=lambda k: chunk_scores.get(k, 0.0), reverse=True)[:limit]
        try:
            chunks_map = await storage.get_chunks_batch(chunk_ids, namespace_id=namespace_id)
        except Exception as e:
            logger.warning("Entity channel: get_chunks_batch failed for {} IDs: {}", len(chunk_ids), e, exc_info=True)
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.entity",
                    reason="channel_exception",
                    detail=str(e),
                    exc=e,
                )
            return []

        # Apply temporal filter post-hydration. get_chunks_batch has no
        # WHERE-clause hook, so we filter by COALESCE(source_timestamp,
        # created_at) here to match the pgvector pushdown rule used by the
        # other channels.
        if created_after is not None or created_before is not None:
            filtered_map: dict[UUID, Chunk] = {}
            for cid, chunk in chunks_map.items():
                ct = getattr(chunk, "source_timestamp", None) or chunk.created_at
                if ct is None:
                    continue
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=UTC)
                if created_after is not None and ct < created_after:
                    continue
                if created_before is not None and ct > created_before:
                    continue
                filtered_map[cid] = chunk
            chunks_map = filtered_map

        # Semantic relevance gate: filter out entity-adjacent chunks that
        # share entity mentions but are semantically irrelevant to the query.
        # Uses Rust batch_cosine_similarity for ~10x speedup over Python loop.
        ordered_chunks = [(chunks_map[cid], cid) for cid in chunk_ids if cid in chunks_map]
        if query_embedding is not None and ordered_chunks:
            embeddings_with_idx = []
            chunks_without_embedding = []
            for i, (chunk, cid) in enumerate(ordered_chunks):
                if chunk.embedding is not None:
                    embeddings_with_idx.append((i, chunk, cid))
                else:
                    chunks_without_embedding.append((i, chunk, cid))

            # Batch cosine similarity via Rust (GIL-released, SIMD-accelerated)
            sims = {}
            if embeddings_with_idx:
                # Seed every embedded chunk below the relevance threshold.
                # batch_cosine_similarity drops pairs below its threshold (0.0),
                # so a chunk with a negative cosine is absent from the result;
                # seeding 0.0 means it stays gated out instead of leaking through.
                for _, _chunk, cid in embeddings_with_idx:
                    sims[cid] = 0.0
                try:
                    from khora._accel import batch_cosine_similarity

                    chunk_embeddings = [chunk.embedding for _, chunk, _ in embeddings_with_idx]
                    # Returns (local_idx, score) pairs sorted descending, NOT
                    # input-ordered - map each back to its chunk by index, like
                    # the temporal channel does. (A positional zip would both
                    # misalign and call float() on a tuple; see #1143.)
                    sim_scores = batch_cosine_similarity(query_embedding, chunk_embeddings)
                    for local_idx, score in sim_scores:
                        _, _chunk, cid = embeddings_with_idx[local_idx]
                        sims[cid] = float(score)
                except ImportError as exc:
                    # Accel module genuinely missing: skip the gate rather than
                    # drop every entity-adjacent chunk. Surface as a degradation
                    # so the relaxed filtering is visible (ADR-001). Narrowed to
                    # ImportError so logic bugs (e.g. the #1143 float(tuple)
                    # TypeError) propagate instead of silently disabling the gate.
                    sims = {}
                    if degradations is not None:
                        _record_channel_degradation(
                            degradations,
                            component="chronicle.entity",
                            reason="cosine_batch_failed",
                            detail=str(exc),
                            exc=exc,
                        )

            results = []
            for chunk, cid in ordered_chunks:
                sim = sims.get(cid)
                if sim is not None:
                    if sim < 0.3:
                        continue  # Below relevance threshold
                    results.append((chunk, chunk_scores[cid] * sim))
                else:
                    # No embedding -> can't be cosine-checked. Penalize at the
                    # 0.3 relevance floor instead of passing through at full
                    # base score, so an un-verifiable chunk never outranks one
                    # the gate verified as relevant (cosine >= 0.3) at the same
                    # base score (#1226).
                    results.append((chunk, chunk_scores[cid] * 0.3))
        else:
            results = [(chunk, chunk_scores[cid]) for chunk, cid in ordered_chunks]

        logger.debug("Entity channel: returning {} chunks (after relevance gate)", len(results))
        return results

    async def _collect_entities(
        self,
        *,
        namespace_id: UUID,
        entity_channel_hits: dict[UUID, tuple[Entity, float]],
        temporal_event_subjects: dict[str, float],
        limit: int,
    ) -> list[tuple[Entity, float]]:
        """Merge entity-channel hits with event-subject lookups for RecallResult.

        Two sources contribute to ``RecallResult.entities``:

        1. **Direct entity-channel hits** — already resolved Entity records
           with similarity scores in ``[0, 1]``. Highest signal; kept at full
           score.
        2. **Temporal-channel event subjects** — string subjects that we look
           up by name. Resolved entities receive their event score attenuated
           by ``0.5`` so direct hits outrank derived ones on ties.

        Dedupe by ``entity.id`` (max-score wins). Sort by score desc and
        truncate at ``limit``. Returns ``[]`` when both sources are empty
        — preserves the pre-#5 behaviour of ``RecallResult.entities=[]``.
        """
        # Start from a fresh dedup map; entity_channel_hits is the priority
        # source (full scores).
        merged: dict[UUID, tuple[Entity, float]] = dict(entity_channel_hits)

        # Resolve event subjects we don't already have. Skip names that match
        # an entity already in `merged` *by name* — saves a DB roundtrip and
        # avoids attenuating an entity that the entity-channel already scored
        # at full strength.
        already_named = {ent.name for ent, _ in merged.values()}
        unresolved_subjects = [s for s in temporal_event_subjects if s not in already_named]

        if unresolved_subjects:
            try:
                resolved = await self._get_storage().get_entities_by_names_batch(namespace_id, unresolved_subjects)
            except Exception as exc:
                logger.debug("collect_entities: get_entities_by_names_batch failed ({})", exc)
                resolved = {}
            for name, entity in resolved.items():
                event_score = temporal_event_subjects.get(name, 0.0)
                attenuated = event_score * 0.5
                prev = merged.get(entity.id)
                if prev is None or attenuated > prev[1]:
                    merged[entity.id] = (entity, attenuated)

        if not merged:
            return []

        ordered = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)
        return ordered[:limit]

    async def _cross_session_expand(
        self,
        chunks_with_scores: list[tuple[Chunk, float]],
        query: str,
        namespace_id: UUID,
        query_embedding: list[float] | None,
        limit: int,
        *,
        degradations: list[Degradation] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Expand results with chunks from other sessions mentioning same entities.

        After initial retrieval finds content from session A, this method
        identifies key entities and fetches additional chunks from sessions
        B, C, ... that mention those entities. This bridges the session gap
        for temporal/change queries.

        Only triggers when the query has cross-session intent (change,
        switch, history, etc.) and results span fewer sessions than expected.
        """
        if not chunks_with_scores or query_embedding is None:
            return chunks_with_scores

        # Use BOTH Rust temporal category detection AND regex for maximum
        # coverage.  The Rust detector catches explicit temporal patterns
        # (STATE_QUERY, ORDINAL, CHANGE, RECENCY) while the regex catches
        # implicit change vocabulary ("switched", "moved", "used to").
        # Either match triggers expansion.
        should_expand = _CROSS_SESSION_INTENT.search(query) is not None
        if not should_expand:
            try:
                from khora._accel import detect_temporal_category

                cat = detect_temporal_category(query)
                should_expand = cat in (1, 2, 3, 6)  # RECENCY, STATE_QUERY, ORDINAL, CHANGE
            except Exception:  # noqa: S110 — _accel import failure is non-fatal
                pass

        if not should_expand:
            return chunks_with_scores

        storage = self._get_storage()

        # Collect session IDs from current results
        seen_sessions: set[str] = set()
        seen_chunk_ids: set[UUID] = set()
        for chunk, _ in chunks_with_scores:
            seen_chunk_ids.add(chunk.id)
            custom = chunk.metadata or {}
            sid = custom.get("session_id") or custom.get("thread_id")
            if sid:
                seen_sessions.add(str(sid))

        # Always attempt expansion — even with multiple sessions, cross-session
        # entity links can surface important state changes across sessions.

        # Find entities similar to query, then fetch their source chunks
        # from OTHER sessions
        try:
            entity_results = await storage.search_similar_entities(namespace_id, query_embedding, limit=10)
        except Exception as exc:
            logger.warning("Chronicle cross-session expansion: search_similar_entities failed: {}", exc, exc_info=True)
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.cross_session",
                    reason="channel_exception",
                    detail="search_similar_entities failed",
                    exc=exc,
                )
            return chunks_with_scores

        if not entity_results:
            return chunks_with_scores

        entity_ids = [eid for eid, _ in entity_results]
        try:
            entities = await storage.get_entities_batch(entity_ids, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("Chronicle cross-session expansion: get_entities_batch failed: {}", exc, exc_info=True)
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.cross_session",
                    reason="channel_exception",
                    detail="get_entities_batch failed",
                    exc=exc,
                )
            return chunks_with_scores

        # Collect chunk IDs from entity sources that aren't already in results
        expansion_chunk_ids: list[UUID] = []
        for entity in entities.values():
            for cid in entity.source_chunk_ids:
                if cid not in seen_chunk_ids:
                    expansion_chunk_ids.append(cid)
                    seen_chunk_ids.add(cid)

        if not expansion_chunk_ids:
            return chunks_with_scores

        # Fetch and score expansion chunks
        expansion_chunk_ids = expansion_chunk_ids[:limit]  # Cap expansion size
        try:
            chunks_map = await storage.get_chunks_batch(expansion_chunk_ids, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("Chronicle cross-session expansion: get_chunks_batch failed: {}", exc, exc_info=True)
            if degradations is not None:
                _record_channel_degradation(
                    degradations,
                    component="chronicle.cross_session",
                    reason="channel_exception",
                    detail="get_chunks_batch failed",
                    exc=exc,
                )
            return chunks_with_scores

        # Discount factor: expansion results score close to reranked results
        # to ensure cross-session content is competitive in the final ranking
        avg_score = sum(s for _, s in chunks_with_scores[:5]) / min(5, len(chunks_with_scores))
        discount = avg_score * 0.8

        expanded = list(chunks_with_scores)
        added = 0
        for cid in expansion_chunk_ids:
            chunk = chunks_map.get(cid)
            if chunk:
                # Include chunks from ANY session (not just different ones)
                # to also surface same-session entity co-occurrences
                expanded.append((chunk, discount * (0.95**added)))
                added += 1

        if added > 0:
            logger.debug("Cross-session expansion: added {} chunks from other sessions", added)

        # #1118: the discounted expansion scores are designed to be competitive
        # in the final ranking, but appending them leaves the merged list out of
        # score order. Re-sort descending so the caller's positional ``[:limit]``
        # trim keeps the genuinely top-scored chunks and the subsequent
        # min_max_normalize assigns 1.0 to the real top chunk (the #834 contract:
        # top chunk = 1.0, scores as normalized rank).
        expanded.sort(key=lambda pair: pair[1], reverse=True)

        return expanded

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()

        # namespace_id is required for IDOR-safe lookup (IDOR family). Callers
        # going through Khora.forget always resolve it before calling here;
        # bail loudly rather than allow a cross-tenant id probe.
        if namespace_id is None:
            logger.warning(f"Cannot forget document {document_id}: namespace_id is required")
            return False

        document = await storage.get_document(document_id, namespace_id=namespace_id)
        if document is None:
            # Either the document doesn't exist or it lives in another
            # namespace — either way, nothing to forget for this caller.
            return False

        # Collect chunk ids and delete derived memory_facts BEFORE
        # delete_document - the chunks FK cascade destroys the rows the
        # provenance match needs (#1140).
        await self._forget_memory_facts(document_id, namespace_id)
        await self._cascade_forget_extraction(document_id, namespace_id)

        return await storage.delete_document(document_id, namespace_id=namespace_id)

    async def _forget_memory_facts(self, document_id: UUID, namespace_id: UUID) -> None:
        """Hard-delete memory_facts derived from a document's chunks (#1140).

        ``memory_facts`` has no FK to chunks or documents - provenance is the
        non-FK ``source_chunk_ids`` array - so the chunks cascade from
        ``delete_document`` never reaches it. Facts are extracted per chunk
        (single-chunk provenance), and the forget contract favours removing
        derived content over retaining forgotten text, so any fact
        referencing one of the document's chunks is deleted. Failures
        degrade per ADR-001 (WARNING + counter) rather than blocking the
        document delete.
        """
        storage = self._get_storage()
        try:
            chunks = await storage.get_chunks_by_document(document_id, namespace_id=namespace_id)
            chunk_ids = [chunk.id for chunk in chunks]
            if not chunk_ids:
                return
            deleted = await storage.delete_facts_for_chunks(chunk_ids, namespace_id=namespace_id)
            if deleted:
                logger.debug(
                    "Forget cascade removed {} memory fact(s) derived from document {}",
                    deleted,
                    document_id,
                )
        except Exception as exc:
            _FORGET_DEGRADED_COUNTER.add(1, {"reason": "memory_facts_cleanup_failed"})
            logger.warning(
                "Forget cascade could not delete memory facts for document {}: {}",
                document_id,
                exc,
                exc_info=True,
            )

    async def _cascade_forget_extraction(self, document_id: UUID, namespace_id: UUID) -> list[Degradation]:
        """Drop / decrement entities and relationships extracted from a document.

        Vector-anchored refcounting (#923): hard-deletes orphans (entities /
        relationships whose only ``source_document_ids`` entry is
        ``document_id``) and strips ``document_id`` from survivors' source
        arrays. Cleanup is anchored on whichever store actually holds the
        entities (the pgvector ``entities`` table on chronicle's PG stack,
        the graph adapter tables otherwise) and mirrored opportunistically to
        the other store. Runs on every backend, not just Neo4j.
        """
        storage = self._get_storage()
        return await cascade_forget_extraction(
            graph=storage.graph,
            vector=storage.vector,
            document_id=document_id,
            namespace_id=namespace_id,
            engine="khora.chronicle",
        )

    @trace("khora.chronicle.remember_batch")
    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        max_concurrent: int = 10,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | str | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        extraction_batch_size: int | None = None,
        extraction_max_tokens: int | None = None,
    ) -> BatchResult:
        """Store multiple documents via the shared ingest pipeline.

        Delegates to the same ``ingest_documents`` pipeline used by VectorCypher
        for full entity extraction, deduplication, and optional expansion.

        Args:
            documents: List of document dicts with content, title, source, metadata
            namespace_id: Target namespace UUID
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents
            infer_relationships: Infer relationships after ingestion
            on_progress: Callback(processed_count, total_count)
            entity_types: Entity types to extract
            relationship_types: Relationship types to extract
            expertise: Optional expertise — an ``ExpertiseConfig``, a
                registered expertise name, or a YAML file path
            extraction_config_hash: Hash for change detection
            chunk_strategy: Override chunking strategy
            chunk_size: Override target chunk size in tokens
                (None = configured ``config.pipeline.chunk_size``)
            extraction_batch_size: Max texts per LLM extraction call (None = pipeline default)
            extraction_max_tokens: Max tokens for extraction LLM calls (None = pipeline default)

        Returns:
            BatchResult with aggregated statistics
        """
        expertise = _resolve_expertise(expertise)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        if not documents:
            return BatchResult(
                total=0,
                processed=0,
                skipped=0,
                failed=0,
                chunks=0,
                entities=0,
                relationships=0,
            )

        # Build doc inputs for ingest_documents
        start = time.perf_counter()
        doc_inputs: list[dict[str, Any]] = []
        for doc_data in documents:
            entry: dict[str, Any] = {
                "content": doc_data.get("content", ""),
                "title": doc_data.get("title", ""),
                "source": doc_data.get("source", ""),
                "source_type": doc_data.get("source_type", source_type),
                "source_name": doc_data.get("source_name", source_name),
                "source_url": doc_data.get("source_url", source_url),
                "source_timestamp": doc_data.get("source_timestamp", source_timestamp),
                "metadata": doc_data.get("metadata", {}),
            }
            if extraction_config_hash is not None:
                entry["extraction_config_hash"] = extraction_config_hash
            if "external_id" in doc_data:
                entry["external_id"] = doc_data["external_id"]
            doc_inputs.append(entry)
        timings["prepare_inputs_ms"] = (time.perf_counter() - start) * 1000

        from khora.pipelines.flows.ingest import ingest_documents

        # Create shared embedder
        shared_embedder = LiteLLMEmbedder(
            model=self._config.llm.embedding_model,
            dimension=self._config.llm.embedding_dimension,
        )

        # Optional cross-document entity deduplication
        shared_entity_index = None
        if deduplicate:
            start = time.perf_counter()
            from khora.extraction.expansion.entity_index import EntityIndex

            shared_entity_index = EntityIndex()

            existing_entities = await self._get_storage().list_entities(namespace_id, limit=50000)
            for entity in existing_entities:
                shared_entity_index.add(entity)

            timings["entity_preload_ms"] = (time.perf_counter() - start) * 1000
            if existing_entities:
                logger.debug(f"Preloaded {len(existing_entities)} existing entities into EntityIndex")

        # Determine expansion
        effective_expansion = infer_relationships
        if expertise is not None and expertise.expansion.enabled:
            effective_expansion = True

        start = time.perf_counter()
        ingest_kwargs: dict[str, Any] = dict(
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            extraction_timeout=self._config.llm.timeout,
            extraction_wave_size=self._config.llm.extraction_wave_size,
            max_concurrent_documents=max_concurrent,
            shared_embedder=shared_embedder,
            shared_entity_index=shared_entity_index,
            enable_expansion=effective_expansion,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
            ketrag_skeleton_channel=self._config.pipeline.ketrag_skeleton_channel,
            extraction_second_pass=self._config.pipeline.extraction_second_pass,
        )
        if chunk_strategy is not None:
            ingest_kwargs["chunk_strategy"] = chunk_strategy
        # Issue #1426: fall back to the configured pipeline default (like
        # VectorCypher/Skeleton) instead of the pipeline's hardcoded 512.
        ingest_kwargs["chunk_size"] = chunk_size if chunk_size is not None else self._config.pipeline.chunk_size
        if extraction_batch_size is not None:
            ingest_kwargs["extraction_batch_size"] = extraction_batch_size
        if extraction_max_tokens is not None:
            ingest_kwargs["extraction_max_tokens"] = extraction_max_tokens
        result = await ingest_documents(namespace_id, doc_inputs, self._get_storage(), **ingest_kwargs)
        timings["ingest_pipeline_ms"] = (time.perf_counter() - start) * 1000

        # Chronicle #2/#3: extract events + facts for every chunk created
        # across the batch.  Both helpers iterate the full chunk set
        # produced by the ingest pipeline; they share the per-chunk
        # extraction semaphore.
        events_extracted = 0
        facts_extracted = 0
        reconcile_errors = 0
        # Issue #893 (ADR-001): a transient ``get_namespace`` failure must not
        # silently revert a namespace that disabled events/facts back to
        # default-on. Surface the dropped override on the result.
        batch_degradations: list[Degradation] = []
        try:
            namespace = await self._get_storage().get_namespace(namespace_id)
        except Exception as exc:
            namespace = None
            logger.warning(
                "get_namespace failed for namespace_id={}: {} - per-namespace "
                "event/fact overrides may be dropped, falling back to default-on",
                namespace_id,
                exc,
                exc_info=True,
            )
            _record_channel_degradation(
                batch_degradations,
                component="chronicle.namespace_overrides",
                reason="get_namespace_failed",
                detail="per-namespace event/fact overrides could not be resolved; fell back to expertise/default (default-on)",
                exc=exc,
            )

        run_events = self._events_enabled(namespace, expertise)
        run_facts = self._facts_enabled(namespace, expertise)

        if run_events or run_facts:
            all_chunk_ids: list[UUID] = []
            for per_doc in result.get("per_document_results", []):
                all_chunk_ids.extend(per_doc.get("chunk_ids", []) or [])
            if all_chunk_ids:
                chunks_map = await self._get_storage().get_chunks_batch(all_chunk_ids, namespace_id=namespace_id)
                chunks = [chunks_map[cid] for cid in all_chunk_ids if cid in chunks_map]
                if run_events:
                    start = time.perf_counter()
                    events_extracted = await self._extract_and_persist_events(
                        chunks, namespace_id, expertise, degradations_out=batch_degradations
                    )
                    timings["event_extraction_ms"] = (time.perf_counter() - start) * 1000
                if run_facts:
                    start = time.perf_counter()
                    facts_extracted, reconcile_errors = await self._extract_and_persist_facts(
                        chunks, namespace_id, expertise, degradations_out=batch_degradations
                    )
                    timings["fact_extraction_ms"] = (time.perf_counter() - start) * 1000

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        processed = result.get("processed_documents", 0)
        if processed > 0 and timings["total_ms"] > 0:
            timings["docs_per_second"] = processed / (timings["total_ms"] / 1000)
            timings["avg_doc_ms"] = timings["ingest_pipeline_ms"] / processed

        logger.info(
            f"remember_batch() completed: {processed}/{len(documents)} docs, "
            f"{result.get('total_chunks', 0)} chunks, {result.get('total_entities', 0)} entities, "
            f"{events_extracted} events, {facts_extracted} facts in {timings['total_ms']:.1f}ms "
            f"({timings.get('docs_per_second', 0):.1f} docs/sec)"
        )

        # Emit per-document progress. Chronicle delegates the batch to the
        # shared ``ingest_documents`` pipeline, which is batch-shaped (one
        # embed_batch, one extraction call across all docs), so the callbacks
        # all arrive here after the pipeline returns rather than strictly
        # one-at-a-time during processing. Still, fire once per accounted
        # document with an incrementing count instead of a single
        # ``(total, total)`` call (#898). See the ``Khora.remember_batch``
        # docstring for the batched-progress caveat.
        if on_progress:
            total_documents = result.get("total_documents", len(documents))
            accounted = (
                result.get("processed_documents", 0)
                + result.get("skipped_documents", 0)
                + result.get("failed_documents", 0)
            )
            for completed in range(1, min(accounted, total_documents) + 1):
                on_progress(completed, total_documents)

        return BatchResult(
            total=result.get("total_documents", len(documents)),
            processed=result.get("processed_documents", 0),
            skipped=result.get("skipped_documents", 0),
            failed=result.get("failed_documents", 0),
            chunks=result.get("total_chunks", 0),
            entities=result.get("total_entities", 0),
            relationships=result.get("total_relationships", 0) + result.get("total_inferred_relationships", 0),
            metadata={
                "timings": timings,
                "events_extracted": events_extracted,
                "facts_extracted": facts_extracted,
                # Issue #892: per-batch count of facts dropped because the
                # contradiction-check LLM call failed transiently. 0 on
                # the happy path.
                "reconcile_errors": reconcile_errors,
                **({"degradations": batch_degradations} if batch_degradations else {}),
            },
        )

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
            config_overrides=config_overrides or {},
            metadata=metadata or {},
        )
        return await self._get_storage().create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_storage().get_namespace(namespace_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``."""
        return await self._get_storage().get_entity(entity_id, namespace_id=namespace_id)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        return await self._get_storage().list_entities(namespace_id, entity_type=entity_type, limit=limit)

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity.

        Chronicle is designed for PostgreSQL-only operation without a graph
        backend. Returns an empty list. Use VectorCypher for graph-based
        entity traversal.
        """
        return []

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id`` (IDOR family)."""
        return await self._get_storage().get_document(document_id, namespace_id=namespace_id)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace."""
        return await self._get_storage().list_documents(namespace_id, limit=limit)

    @trace("khora.chronicle.search_entities", exclude={"query"}, result=lambda r: {"result_count": len(r)})
    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Uses batch entity fetching to avoid N+1 queries.
        """
        embedder = self._get_embedder()
        storage = self._get_storage()

        query_embedding = await embedder.embed(query)

        entity_ids_scores = await storage.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        if not entity_ids_scores:
            return []

        entity_ids = [entity_id for entity_id, _ in entity_ids_scores]
        entities_map = await storage.get_entities_batch(entity_ids, namespace_id=namespace_id)

        return [entities_map[eid] for eid, _score in entity_ids_scores if eid in entities_map]

    # =========================================================================
    # Stats
    # =========================================================================

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()

        doc_count = 0
        last_activity_at = None

        try:
            doc_count, last_activity_at = await storage.get_document_stats(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        chunk_count, entity_count, relationship_count, metadata = await gather_counts(
            storage, namespace_id, engine="chronicle"
        )

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            relationships=relationship_count,
            last_activity_at=last_activity_at,
            metadata=metadata,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components.

        Returns a dict with:
        - status: 'healthy', 'degraded', or 'disconnected'
        - engine: Engine name
        - checks: Individual component results
        """
        if not self._connected:
            return {"status": "disconnected", "engine": "chronicle"}

        health: dict[str, Any] = {
            "engine": "chronicle",
            "status": "healthy",
            "checks": {},
        }

        # Check storage (PostgreSQL + pgvector)
        try:
            storage_health = await self._get_storage().health_check()
            health["checks"]["storage"] = storage_health.summary
            if not storage_health.is_healthy:
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["storage"] = f"error: {e}"
            health["status"] = "degraded"

        # Check embedder availability
        try:
            if self._embedder is not None:
                health["checks"]["embedder"] = "ok"
            else:
                health["checks"]["embedder"] = "not configured"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["embedder"] = f"error: {e}"
            health["status"] = "degraded"

        return health


# Register the deterministic recall-filter compiler for this engine/target at
# import time (idempotent — same function object). The storage target is "chunks":
# the filter attaches to the chunk read path, so "chunks" is the honest label.
# Chronicle pushes down only the conjunctive source_timestamp date bound (the
# recency window's primary stored axis); the engine post-filters the remainder via
# compile_python. Mirrors the skeleton.pgvector registration site.
from khora.filter import CompilerRegistry  # noqa: E402
from khora.filter.compilers.chronicle import compile_chronicle  # noqa: E402

CompilerRegistry.register("chronicle", "chunks", compile_chronicle)
