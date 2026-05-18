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
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import Chunk, Document, DocumentMetadata, Entity, MemoryNamespace
from khora.engines._storage_config import build_storage_config
from khora.engines.chronicle.compression import (
    FactExtractor,
    FactOperation,
    MemoryCompressor,
    MemoryFact,
)
from khora.engines.chronicle.events import ChronicleEvent, EventExtractor
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

ChronicleStorageBackend = Literal["pgvector", "lancedb"]


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


# ---------------------------------------------------------------------------
# Temporal decay helpers
# ---------------------------------------------------------------------------


def _ebbinghaus_decay(age_hours: float, *, half_life_hours: float = 168.0) -> float:
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
    decay_weight: float = 0.15,
    half_life_hours: float = 168.0,
    reference_time: datetime | None = None,
) -> list[tuple[Chunk, float]]:
    """Re-score chunks by blending relevance score with temporal decay.

    final_score = (1 - decay_weight) * relevance + decay_weight * retention

    Uses Rust-accelerated ``batch_recency_scores`` from ``khora._accel``
    when available (~10x faster than per-item Python loop for large batches).
    Falls back to per-item computation otherwise.
    """
    if not chunks_with_scores or decay_weight <= 0:
        return chunks_with_scores

    now = reference_time or datetime.now(UTC)
    now_secs = now.timestamp()
    decay_days = half_life_hours / 24.0

    # Collect timestamps
    timestamps: list[float] = []
    for chunk, _score in chunks_with_scores:
        created = chunk.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        timestamps.append(created.timestamp())

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

    Chunks are grouped by entity (first ``entity_refs`` entry, falling back
    to document title stored in ``chunk.metadata.custom``).  Within each
    group the maximum version is identified, and older versions receive a
    soft penalty::

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
        custom = chunk.metadata.custom if chunk.metadata else {}
        version = custom.get("version")
        if version is not None:
            has_any_version = True
            try:
                version = int(version)
            except (TypeError, ValueError):
                version = None

        # Group key: first entity_ref, or title, or document_id
        entity_refs = custom.get("entity_refs")
        if entity_refs and isinstance(entity_refs, list) and len(entity_refs) > 0:
            group_key = str(entity_refs[0])
        else:
            group_key = custom.get("title", str(chunk.document_id))

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
        abstention_min_chunks: int = 1,
        abstention_min_top_score: float = 0.3,
        abstention_combined_threshold: float = 0.5,
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
                ``chunks_below_min`` abstention flag fires.
            abstention_min_top_score: Top-chunk score below which the
                ``top_score_low`` abstention flag fires.
            abstention_combined_threshold: Combined-score threshold at or above
                which ``should_abstain`` becomes True. See
                ``_compute_abstention_signals`` for the weighting scheme.
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

        self._abstention_min_chunks = abstention_min_chunks
        self._abstention_min_top_score = abstention_min_top_score
        self._abstention_combined_threshold = abstention_combined_threshold

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
    ) -> list[ChronicleEvent]:
        """Run the extractor on a single chunk, swallowing per-chunk failures.

        Returns an empty list if extraction raises so a single bad LLM call
        cannot fail the whole ``remember()``. The chunk_id and namespace_id
        are stamped onto every returned event so callers can pass the list
        straight to ``coordinator.write_events``.
        """
        async with sem:
            try:
                events = await extractor.extract_events(
                    chunk.content,
                    chunk_id=chunk.id,
                    namespace_id=namespace_id,
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
    ) -> int:
        """Extract events from chunks, embed summaries, persist, and return count.

        Resolution of the enable flag is the caller's responsibility — this
        helper assumes events are already enabled and unconditionally runs.
        """
        if not chunks:
            return 0
        extractor = self._get_event_extractor(expertise)
        sem = asyncio.Semaphore(self._max_concurrent_extractions)
        per_chunk = await asyncio.gather(
            *(self._extract_events_for_chunk(c, namespace_id, extractor, sem) for c in chunks),
            return_exceptions=False,
        )
        events: list[ChronicleEvent] = [ev for sub in per_chunk for ev in sub]
        if not events:
            return 0
        await self._embed_events(events)
        try:
            await self._get_storage().write_events(events, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("write_events failed (skipping event persistence): {}", exc)
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
    ) -> list[MemoryFact]:
        """Run the fact extractor on a single chunk, swallowing per-chunk failures."""
        async with sem:
            try:
                facts = await extractor.extract_facts(
                    chunk.content,
                    chunk_id=chunk.id,
                    namespace_id=namespace_id,
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
        for f in facts:
            f.namespace_id = namespace_id
            if chunk.id not in f.source_chunk_ids:
                f.source_chunk_ids.append(chunk.id)
        return facts

    async def _reconcile_facts(
        self,
        new_facts: list[MemoryFact],
        namespace_id: UUID,
        expertise: ExpertiseConfig | None,
    ) -> int:
        """Apply ADD/UPDATE/DELETE/NOOP reconciliation and persist results.

        Strategy:
          1. Group new facts by subject.
          2. Cache active facts per subject (one query per subject).
          3. For each new fact, ask the compressor what to do:
             - ADD     → queue for write
             - UPDATE  → queue for write + supersede the target
             - DELETE  → supersede only (no write)
             - NOOP    → skip
          4. After processing a subject, append accepted facts to its in-memory
             cache so subsequent new facts about the same subject reconcile
             against everything we just decided to keep — prevents within-batch
             thrashing when two sentences contain the same fact.

        Returns the number of new facts effectively persisted (ADD + UPDATE).
        """
        if not new_facts:
            return 0

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
                facts_to_write.append(new_fact)
                if action.target is not None:
                    pending_supersedes.append((action.target.id, new_fact))
                    # Drop the old fact from the in-memory cache so subsequent
                    # reconciliations within this batch don't see it as active.
                    active_by_subject[subject] = [f for f in active_by_subject[subject] if f.id != action.target.id]
                active_by_subject[subject].append(new_fact)
            elif action.op is FactOperation.DELETE:
                if action.target is not None:
                    deletes.append(action.target.id)
                    active_by_subject[subject] = [f for f in active_by_subject[subject] if f.id != action.target.id]
            # NOOP: skip — nothing to do.

        # Persist new ADDs/UPDATEs first so we have stable IDs to point at.
        if facts_to_write:
            try:
                await storage.write_facts(facts_to_write, namespace_id=namespace_id)
            except Exception as exc:
                logger.warning("write_facts failed during reconciliation: {}", exc)
                return 0

        # Supersede old → new for UPDATEs.
        for old_id, new_fact in pending_supersedes:
            try:
                await storage.supersede_fact(old_id, new_fact.id)
            except Exception as exc:
                logger.warning("supersede_fact failed for {} -> {}: {}", old_id, new_fact.id, exc)

        # DELETE: mark the old fact inactive without a replacement. The
        # storage contract takes a UUID for ``superseded_by``; passing the
        # fact's own id keeps the row marked inactive while leaving a
        # self-reference as the "tombstone" pointer.
        for old_id in deletes:
            try:
                await storage.supersede_fact(old_id, old_id)
            except Exception as exc:
                logger.warning("supersede_fact (delete) failed for {}: {}", old_id, exc)

        return len(facts_to_write)

    async def _extract_and_persist_facts(
        self,
        chunks: list[Chunk],
        namespace_id: UUID,
        expertise: ExpertiseConfig | None,
    ) -> int:
        """Extract facts from chunks, reconcile, persist, and return count.

        Returns the number of facts effectively persisted (ADD + UPDATE).
        Resolution of the ``facts.enabled`` flag is the caller's responsibility.
        """
        if not chunks:
            return 0
        extractor = self._get_fact_extractor(expertise)
        sem = asyncio.Semaphore(self._max_concurrent_extractions)
        per_chunk = await asyncio.gather(
            *(self._extract_facts_for_chunk(c, namespace_id, extractor, sem) for c in chunks),
            return_exceptions=False,
        )
        new_facts: list[MemoryFact] = [f for sub in per_chunk for f in sub]
        if not new_facts:
            return 0

        reconcile = expertise.facts.reconcile if expertise is not None else True
        if reconcile:
            return await self._reconcile_facts(new_facts, namespace_id, expertise)

        # Fast path: no reconciliation, write everything as ADD.
        try:
            await self._get_storage().write_facts(new_facts, namespace_id=namespace_id)
        except Exception as exc:
            logger.warning("write_facts failed (skipping fact persistence): {}", exc)
            return 0
        logger.debug("Persisted {} memory facts across {} chunks (no reconcile)", len(new_facts), len(chunks))
        return len(new_facts)

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
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
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
            expertise: Optional expertise config
            extraction_config_hash: Optional hash for change detection
            chunk_strategy: Override chunking strategy for this call

        Returns:
            RememberResult with document_id and counts
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Compute content checksum
        start = time.perf_counter()
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        timings["checksum_ms"] = (time.perf_counter() - start) * 1000

        storage = self._get_storage()

        # Dedup check
        start = time.perf_counter()
        existing = await storage.get_document_by_checksum(namespace_id, checksum)
        timings["dedup_check_ms"] = (time.perf_counter() - start) * 1000

        if existing:
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
        doc_metadata = DocumentMetadata(
            title=title,
            source=source,
            source_type="api",
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            custom=metadata or {},
        )
        document = Document(
            namespace_id=namespace_id,
            content=content,
            metadata=doc_metadata,
            extraction_config_hash=extraction_config_hash,
            external_id=external_id,
            session_id=_coerce_session_id_from_metadata(metadata),
        )
        document = await storage.create_document(document)
        timings["document_create_ms"] = (time.perf_counter() - start) * 1000

        # Process through shared ingest pipeline (chunking, embedding, extraction)
        from khora.pipelines.flows.ingest import process_document

        start = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
        )
        if chunk_strategy is not None:
            kwargs["chunk_strategy"] = chunk_strategy
        result = await process_document(document, storage, **kwargs)
        timings["pipeline_ms"] = (time.perf_counter() - start) * 1000

        # Chronicle #2/#3: extract SVO events + atomic facts from every
        # persisted chunk. Default-on per ExpertiseConfig with per-namespace
        # overrides via ``namespace.config_overrides[{"events"|"facts"}]``.
        # Both share the same chunk fetch and per-chunk semaphore.
        events_extracted = 0
        facts_extracted = 0
        try:
            namespace = await storage.get_namespace(namespace_id)
        except Exception:
            namespace = None

        run_events = self._events_enabled(namespace, expertise)
        run_facts = self._facts_enabled(namespace, expertise)

        if run_events or run_facts:
            chunk_ids = result.get("chunk_ids", []) or []
            if chunk_ids:
                chunks_map = await storage.get_chunks_batch(list(chunk_ids), namespace_id=namespace_id)
                chunks = [chunks_map[cid] for cid in chunk_ids if cid in chunks_map]
                if run_events:
                    start = time.perf_counter()
                    events_extracted = await self._extract_and_persist_events(chunks, namespace_id, expertise)
                    timings["event_extraction_ms"] = (time.perf_counter() - start) * 1000
                if run_facts:
                    start = time.perf_counter()
                    facts_extracted = await self._extract_and_persist_facts(chunks, namespace_id, expertise)
                    timings["fact_extraction_ms"] = (time.perf_counter() - start) * 1000

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"remember() completed: {result['chunks']} chunks, {result['entities']} entities, "
            f"{result['relationships']} relationships, {events_extracted} events, "
            f"{facts_extracted} facts in {timings['total_ms']:.1f}ms"
        )

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
            metadata={
                "timings": timings,
                "events_extracted": events_extracted,
                "facts_extracted": facts_extracted,
            },
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
        agentic: bool = False,
        raw: bool = False,
        temporal_filter: Any | None = None,
        recency_bias: float | None = None,
    ) -> RecallResult:
        """Recall memories using 4-channel parallel retrieval with RRF fusion.

        Phase 1 channels:
          1. Semantic (vector similarity via pgvector)
          2. BM25 (PostgreSQL full-text search)
          3. Temporal — stubbed, returns empty (Phase 2)
          4. Entity — stubbed, returns empty (Phase 2)

        Results are fused via Reciprocal Rank Fusion and then re-scored
        with Ebbinghaus temporal decay.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum results
            mode: Search mode (VECTOR, KEYWORD, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: Reserved for future multi-step search
            raw: If True, skip LLM features (temporal decay still applies)
            temporal_filter: Reserved for Phase 2 temporal filtering
            recency_bias: Override temporal decay weight (0.0-1.0)

        Returns:
            RecallResult with fused and decay-scored chunks
        """
        storage = self._get_storage()
        embedder = self._get_embedder()
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Read Chronicle-specific config from QuerySettings (with safe defaults)
        qs = getattr(self._config, "query", None)
        _overfetch = getattr(qs, "chronicle_overfetch_multiplier", 4) if qs else 4
        _rrf_w_semantic = getattr(qs, "chronicle_rrf_semantic_weight", 1.0) if qs else 1.0
        _rrf_w_bm25 = getattr(qs, "chronicle_rrf_bm25_weight", 0.8) if qs else 0.8
        _rrf_w_temporal = getattr(qs, "chronicle_rrf_temporal_weight", 0.9) if qs else 0.9
        _rrf_w_entity = getattr(qs, "chronicle_rrf_entity_weight", 0.85) if qs else 0.85
        _cfg_decay = getattr(qs, "chronicle_decay_weight", 0.25) if qs else 0.25
        _cfg_half_life = getattr(qs, "temporal_half_life_hours", 168.0) if qs else 168.0
        overfetch_limit = limit * _overfetch

        # Resolve temporal references from query (fast dateparser, ~0.25ms)
        # when the caller didn't supply an explicit temporal_filter.
        _enable_resolver = getattr(qs, "enable_temporal_resolver", True) if qs else True
        if _enable_resolver and temporal_filter is None:
            from khora.query.temporal import TemporalFilter
            from khora.query.temporal_resolver import TemporalResolver

            resolver = TemporalResolver()
            resolved = resolver.resolve_fast(query)
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
            except RuntimeError:
                logger.debug("Fulltext backend not available for BM25 search")
            except Exception:
                logger.debug("BM25 channel failed")
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
                    logger.debug(f"{name.capitalize()} channel failed: {result}")
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
            try:
                from khora.query.reranking import rerank_chunks

                reranked = await rerank_chunks(
                    query,
                    chunks_with_scores[:_reranking_top_n],
                    method="cross_encoder",
                    top_k=limit,
                    model=_reranking_model,
                )
                chunks_with_scores = reranked
            except Exception as e:
                logger.warning("Chronicle cross-encoder reranking failed: {}", e)
            timings["reranking_ms"] = (time.perf_counter() - start) * 1000

        # ── Cross-session expansion ─────────────────────────────────────
        start = time.perf_counter()
        chunks_with_scores = await self._cross_session_expand(
            chunks_with_scores, query, namespace_id, query_embedding, limit
        )
        timings["cross_session_ms"] = (time.perf_counter() - start) * 1000

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

        # ── Build context text ───────────────────────────────────────────
        context_parts = [chunk.content for chunk, _score in chunks_with_scores]
        context_text = "\n\n---\n\n".join(context_parts[:limit])

        # ── Abstention signals ───────────────────────────────
        # Passive metadata for downstream consumers (LLM answer-generation)
        # to decide whether to refuse vs answer.  Does NOT alter retrieval.
        entity_hits: list[tuple[Entity, float]] = []  # Phase 2: entity-level results
        abstention_signals = self._compute_abstention_signals(chunks_with_scores, entity_hits)

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"recall() completed: {len(chunks_with_scores)} chunks, {len(entity_hits)} entities "
            f"(semantic={len(semantic_results)}, bm25={len(bm25_results)}, "
            f"temporal={len(temporal_results)}, entity={len(entity_results)}) "
            f"in {timings['total_ms']:.1f}ms"
        )

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=chunks_with_scores,
            entities=entity_hits,
            context_text=context_text,
            metadata={
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
                "timings": timings,
            },
        )

    def _compute_abstention_signals(
        self,
        chunks: list[tuple[Chunk, float]],
        entities: list[tuple[Entity, float]],
    ) -> dict[str, Any]:
        """Compute passive abstention signals for downstream answer-generation.

        Returns a dict with four boolean flags, a combined float score, and
        a convenience ``should_abstain`` flag derived from the configured
        threshold.  Pure function over ``chunks`` and ``entities`` — does
        not touch storage.  See ``ChronicleEngine.__init__`` for the
        tunable thresholds.
        """
        entities_empty = len(entities) == 0
        chunks_empty = len(chunks) == 0
        chunks_below_min = len(chunks) < self._abstention_min_chunks
        top_score = chunks[0][1] if chunks else 0.0
        top_score_low = top_score < self._abstention_min_top_score

        # Weighted boolean signals: any one fires → 0.3-0.4; all three → 1.0.
        # chunks_below_min is weighted highest because zero/few chunks is the
        # strongest signal that retrieval failed.
        combined = 0.3 * float(entities_empty) + 0.4 * float(chunks_below_min) + 0.3 * float(top_score_low)

        # Aggregate metrics (Phase 4). Per-signal counter increments
        # 0-4 times per recall; histogram observed once per recall. Both are
        # aggregate-only — no namespace_id label (cardinality safety).
        if entities_empty:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "entities_empty"})
        if chunks_empty:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_empty"})
        if chunks_below_min:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "chunks_below_min"})
        if top_score_low:
            _ABSTENTION_SIGNAL_COUNTER.add(1, attributes={"signal": "top_score_low"})
        _ABSTENTION_COMBINED_SCORE_HISTOGRAM.record(combined)

        return {
            "entities_empty": entities_empty,
            "chunks_empty": chunks_empty,
            "chunks_below_min": chunks_below_min,
            "top_score_low": top_score_low,
            "combined_score": combined,
            "should_abstain": combined >= self._abstention_combined_threshold,
        }

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
            return await self._temporal_channel_chunks_fallback(namespace_id, query_embedding, limit, temporal_filter)

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
            logger.debug("temporal channel: query_events failed ({}); falling back to chunks", exc)
            return await self._temporal_channel_chunks_fallback(namespace_id, query_embedding, limit, temporal_filter)

        if not events:
            logger.debug("temporal channel: no events for namespace; falling back to chunks")
            return await self._temporal_channel_chunks_fallback(namespace_id, query_embedding, limit, temporal_filter)

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
                    logger.debug("temporal channel: cosine batch failed ({}); falling back to no-cosine", exc)

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
            logger.debug("temporal channel: events produced no usable scores; falling back to chunks")
            return await self._temporal_channel_chunks_fallback(namespace_id, query_embedding, limit, temporal_filter)

        # Hydrate the top-scoring chunks. We only need ``limit`` of them.
        top_chunk_ids = sorted(per_chunk_score, key=lambda cid: per_chunk_score[cid], reverse=True)[:limit]
        try:
            chunks_map = await storage.get_chunks_batch(top_chunk_ids, namespace_id=namespace_id)
        except Exception as exc:
            logger.debug("temporal channel: get_chunks_batch failed ({})", exc)
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
        except Exception:
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
            logger.warning("Entity channel: search_similar_entities failed: {}", e)
            return []

        if not entity_results:
            logger.debug("Entity channel: no similar entities found")
            return []

        logger.debug("Entity channel: found {} similar entities", len(entity_results))

        # Step 2: Get the source chunk IDs from matching entities
        entity_ids = [eid for eid, _score in entity_results]
        entity_scores = {eid: score for eid, score in entity_results}

        try:
            entities = await storage.get_entities_batch(entity_ids)
        except Exception as e:
            logger.warning("Entity channel: get_entities_batch failed for {} IDs: {}", len(entity_ids), e)
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
            logger.warning("Entity channel: get_chunks_batch failed for {} IDs: {}", len(chunk_ids), e)
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
                try:
                    from khora._accel import batch_cosine_similarity

                    chunk_embeddings = [chunk.embedding for _, chunk, _ in embeddings_with_idx]
                    sim_scores = batch_cosine_similarity(query_embedding, chunk_embeddings)
                    for (idx, chunk, cid), sim in zip(embeddings_with_idx, sim_scores):
                        sims[cid] = float(sim)
                except Exception:
                    # Fallback: no filtering
                    for idx, chunk, cid in embeddings_with_idx:
                        sims[cid] = 1.0

            results = []
            for chunk, cid in ordered_chunks:
                sim = sims.get(cid)
                if sim is not None:
                    if sim < 0.3:
                        continue  # Below relevance threshold
                    results.append((chunk, chunk_scores[cid] * sim))
                else:
                    results.append((chunk, chunk_scores[cid]))
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
            custom = chunk.metadata.custom if chunk.metadata else {}
            sid = custom.get("session_id") or custom.get("thread_id")
            if sid:
                seen_sessions.add(str(sid))

        # Always attempt expansion — even with multiple sessions, cross-session
        # entity links can surface important state changes across sessions.

        # Find entities similar to query, then fetch their source chunks
        # from OTHER sessions
        try:
            entity_results = await storage.search_similar_entities(namespace_id, query_embedding, limit=10)
        except Exception:
            return chunks_with_scores

        if not entity_results:
            return chunks_with_scores

        entity_ids = [eid for eid, _ in entity_results]
        try:
            entities = await storage.get_entities_batch(entity_ids)
        except Exception:
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
        except Exception:
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

        return expanded

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()

        # Verify namespace if provided
        ns_id = namespace_id
        if ns_id:
            document = await storage.get_document(document_id)
            if document and document.namespace_id != ns_id:
                logger.warning(f"Document {document_id} not in namespace {ns_id}")
                return False
        else:
            document = await storage.get_document(document_id)
            if document is None:
                return False
            ns_id = document.namespace_id

        await self._cascade_forget_extraction(document_id, ns_id)

        return await storage.delete_document(document_id)

    async def _cascade_forget_extraction(self, document_id: UUID, namespace_id: UUID) -> None:
        """Drop / decrement entities and relationships extracted from a document.

        Hard-deletes orphans (single-source entities/relationships whose only
        ``source_document_ids`` entry is ``document_id``) and strips
        ``document_id`` from survivors' ``source_document_ids`` arrays in
        both the graph and the vector backends. No-op when the graph backend
        does not expose ``fetch_document_extraction_state`` (e.g. chronicle
        pgvector-only deployments where the graph backend is absent).
        """
        storage = self._get_storage()
        graph = storage.graph
        vector = storage.vector
        fetch = getattr(graph, "fetch_document_extraction_state", None)
        if fetch is None:
            return

        entities, relationships = await fetch(document_id, namespace_id=namespace_id)

        orphan_ent_ids = [UUID(e["id"]) for e in entities if e["source_document_count"] == 1]
        survive_ent_ids = [UUID(e["id"]) for e in entities if e["source_document_count"] > 1]
        orphan_rel_ids = [UUID(r["id"]) for r in relationships if r["source_document_count"] == 1]
        survive_rel_ids = [UUID(r["id"]) for r in relationships if r["source_document_count"] > 1]

        if orphan_ent_ids:
            await graph.delete_entities_batch(orphan_ent_ids, namespace_id)  # type: ignore[unresolved-attribute]
            if vector is not None and hasattr(vector, "delete_entities_batch"):
                await vector.delete_entities_batch(orphan_ent_ids)
        if orphan_rel_ids:
            await graph.delete_relationships_batch(orphan_rel_ids)  # type: ignore[unresolved-attribute]
            if vector is not None and hasattr(vector, "delete_relationships_batch"):
                await vector.delete_relationships_batch(orphan_rel_ids)

        if survive_ent_ids:
            await graph.remove_document_from_entity_sources_batch(  # type: ignore[unresolved-attribute]
                survive_ent_ids, document_id, namespace_id
            )
            if vector is not None and hasattr(vector, "remove_document_from_entity_sources"):
                await vector.remove_document_from_entity_sources(survive_ent_ids, document_id)
        if survive_rel_ids:
            await graph.remove_document_from_relationship_sources_batch(  # type: ignore[unresolved-attribute]
                survive_rel_ids, document_id
            )
            if vector is not None and hasattr(vector, "remove_document_from_relationship_sources"):
                await vector.remove_document_from_relationship_sources(survive_rel_ids, document_id)

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
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
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
            expertise: Optional expertise config
            extraction_config_hash: Hash for change detection
            chunk_strategy: Override chunking strategy
            extraction_batch_size: Max texts per LLM extraction call (None = pipeline default)
            extraction_max_tokens: Max tokens for extraction LLM calls (None = pipeline default)

        Returns:
            BatchResult with aggregated statistics
        """
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
                "source_type": "api",
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
        shared_embedder = LiteLLMEmbedder(model=self._config.llm.embedding_model)

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
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            extraction_timeout=self._config.llm.timeout,
            max_concurrent_documents=max_concurrent,
            shared_embedder=shared_embedder,
            shared_entity_index=shared_entity_index,
            enable_expansion=effective_expansion,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
        )
        if chunk_strategy is not None:
            ingest_kwargs["chunk_strategy"] = chunk_strategy
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
        try:
            namespace = await self._get_storage().get_namespace(namespace_id)
        except Exception:
            namespace = None

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
                    events_extracted = await self._extract_and_persist_events(chunks, namespace_id, expertise)
                    timings["event_extraction_ms"] = (time.perf_counter() - start) * 1000
                if run_facts:
                    start = time.perf_counter()
                    facts_extracted = await self._extract_and_persist_facts(chunks, namespace_id, expertise)
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

        if on_progress:
            on_progress(
                result.get("processed_documents", 0),
                result.get("total_documents", len(documents)),
            )

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
            },
        )

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
            config_overrides=config_overrides or {},
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

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        return await self._get_storage().get_document(document_id)

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
        entities_map = await storage.get_entities_batch(entity_ids)

        return [entities_map[eid] for eid, _score in entity_ids_scores if eid in entities_map]

    # =========================================================================
    # Stats
    # =========================================================================

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()

        doc_count = 0
        chunk_count = 0
        entity_count = 0
        relationship_count = 0
        last_activity_at = None

        try:
            doc_count, last_activity_at = await storage.get_document_stats(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            chunk_count = await storage.count_chunks(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            entity_count = await storage.count_entities(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            relationship_count = await storage.count_relationships(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            relationships=relationship_count,
            last_activity_at=last_activity_at,
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
