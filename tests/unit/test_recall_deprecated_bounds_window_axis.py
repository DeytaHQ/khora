"""Complementary edge cases for the deprecated ``start_time``/``end_time`` axis.

Sibling to ``test_recall_deprecated_bounds_shim.py`` (which pins the core
facade translation: the deprecated bounds forward a ``temporal_filter`` window and
NOT an ``occurred_at`` ``filter_ast``). This file widens that guard along the
boundary shapes the window-axis API has to get right, and adds the BEHAVIORAL
contrast the shim ultimately protects:

1. **Window-forwarding edge shapes** — start-only, end-only, both, and an inverted
   pair all forward a recency-window ``temporal_filter`` and keep ``filter_ast``
   ``None``. A naive (tz-less) bound is coerced to UTC on the window. These pin the
   facade translation across every one/both/none combination of the two bounds.

2. **The event-time vs window-axis divergence, end to end through the engine** — an
   anchor-less chunk (``occurred_at`` AND ``source_timestamp`` both ``None`` — the
   shape a plain ``remember(content=...)`` with no timestamp produces) SURVIVES a
   ``start_time`` recency window but is EXCLUDED by an equivalent ``occurred_at``
   filter. The window path runs no event-time post-filter (``filter_ast`` is
   ``None``), so the chunk passes through; the ``occurred_at`` filter compiles a
   post-filter whose record ``occurred_at`` is ``COALESCE(occurred_at,
   source_timestamp)`` — ``None`` for an unanchored chunk — so the positive
   predicate drops it. Folding the deprecated bounds into an ``occurred_at`` filter
   (the regression) would collapse the first path into the second and false-empty
   the chunk.

Hermetic: the facade tests use a capture-stub engine (same pattern as the sibling
file); the behavioral test drives the real ``ChronicleEngine`` over a mocked
storage + embedder (same pattern as ``test_chronicle_filter_composition.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk
from khora.core.models.recall import RecallResult
from khora.engines.chronicle.engine import ChronicleEngine
from khora.khora import Khora
from khora.query import SearchMode

pytestmark = pytest.mark.unit


# ===========================================================================
# (1) Facade window-forwarding edge shapes — capture-stub engine.
# ===========================================================================
#
# Same minimal capture harness as the sibling file: the engine records the
# kwargs the facade hands it and returns empty, so the assertion is purely on the
# facade's start_time/end_time → (temporal_filter, filter_ast) translation.


def _capture_engine(captured: dict[str, Any]) -> Any:
    async def _recall(query: str, namespace_id: Any, **kwargs: Any) -> RecallResult:
        captured.update(kwargs)
        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            documents=[],
            chunks=[],
            entities=[],
            relationships=[],
            engine_info={"engine": "chronicle"},
        )

    engine = AsyncMock()
    engine.recall = _recall
    return engine


def _facade(captured: dict[str, Any]) -> Khora:
    kb = Khora.__new__(Khora)
    kb._engine_name = "chronicle"
    engine = _capture_engine(captured)
    kb._get_engine = lambda: engine  # type: ignore[method-assign]
    kb._resolve_namespace = AsyncMock(return_value=uuid4())  # type: ignore[method-assign]
    kb._dispatch_hook = AsyncMock(return_value=None)  # type: ignore[method-assign]

    async def _passthrough(result: RecallResult, namespace_id: Any) -> RecallResult:
        return result

    kb._upgrade_recall_documents = _passthrough  # type: ignore[method-assign]
    return kb


_START = datetime(2026, 4, 1, tzinfo=UTC)
_END = datetime(2026, 6, 1, tzinfo=UTC)


async def _recall_bounds(*, start: datetime | None = None, end: datetime | None = None) -> dict[str, Any]:
    """Drive the facade with the given deprecated bounds, return captured kwargs."""
    captured: dict[str, Any] = {}
    kb = _facade(captured)
    with pytest.warns(DeprecationWarning):
        await kb.recall("alpha", namespace=uuid4(), mode=SearchMode.VECTOR, start_time=start, end_time=end)
    return captured


@pytest.mark.asyncio
async def test_both_bounds_forward_window_no_filter_ast() -> None:
    captured = await _recall_bounds(start=_START, end=_END)
    tf = captured["temporal_filter"]
    assert tf is not None
    assert tf.occurred_after == _START
    assert tf.occurred_before == _END
    assert captured["filter_ast"] is None, "a both-bounds window must not fold an occurred_at filter_ast"


@pytest.mark.asyncio
async def test_start_only_leaves_upper_bound_open() -> None:
    captured = await _recall_bounds(start=_START)
    tf = captured["temporal_filter"]
    assert tf.occurred_after == _START
    assert tf.occurred_before is None, "start_time only must leave the window's upper bound open"
    assert captured["filter_ast"] is None


@pytest.mark.asyncio
async def test_end_only_leaves_lower_bound_open() -> None:
    captured = await _recall_bounds(end=_END)
    tf = captured["temporal_filter"]
    assert tf.occurred_before == _END
    assert tf.occurred_after is None, "end_time only must leave the window's lower bound open"
    assert captured["filter_ast"] is None


@pytest.mark.asyncio
async def test_inverted_range_forwards_verbatim_no_special_casing() -> None:
    # An inverted pair (start later than end) is forwarded verbatim to the window —
    # the facade does not special-case it into an empty filter_ast. The engine's
    # recency window decides the (empty) outcome; the facade contract is just "no
    # occurred_at fold".
    captured = await _recall_bounds(start=_END, end=_START)
    tf = captured["temporal_filter"]
    assert tf.occurred_after == _END
    assert tf.occurred_before == _START
    assert captured["filter_ast"] is None


@pytest.mark.asyncio
async def test_naive_bound_is_coerced_to_utc_on_window() -> None:
    # A tz-naive bound is normalized to tz-aware UTC before it reaches the window,
    # so a downstream tz-aware compare never raises.
    naive = datetime(2026, 4, 1)  # noqa: DTZ001 — intentionally naive for the coercion test
    captured = await _recall_bounds(start=naive)
    tf = captured["temporal_filter"]
    assert tf.occurred_after.tzinfo is not None, "naive start_time must be coerced to tz-aware"
    assert tf.occurred_after == naive.replace(tzinfo=UTC)
    assert tf.occurred_after.utcoffset() == UTC.utcoffset(None), "the coerced bound must carry a zero (UTC) offset"
    assert captured["filter_ast"] is None


# ===========================================================================
# (2) Event-time vs window-axis divergence — through the FACADE, real engine.
# ===========================================================================
#
# The behavioral heart of the contract, driven end to end through
# ``Khora.recall`` so it exercises the facade translation under test (NOT the
# engine kwargs directly): the same anchor-less chunk passes a deprecated
# ``start_time`` window but is dropped by the equivalent public ``occurred_at``
# filter. On pre-fix code the ``start_time`` path folded an ``occurred_at``
# ``filter_ast`` and the chunk was false-emptied, so this guard bites the
# regression while also documenting the user-visible behavior the two APIs must
# keep distinct.


def _anchorless_chunk() -> Chunk:
    """A chunk with NO event-time anchor — occurred_at AND source_timestamp None.

    This is the shape a plain ``remember(content=...)`` with no timestamp produces.
    ``created_at`` is recent so it would sit inside any sane recency window.
    """
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="alpha anchorless",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        metadata={},
        occurred_at=None,
        source_timestamp=None,
    )


def _real_engine(results: list[tuple[Chunk, float]]) -> ChronicleEngine:
    """A connected ChronicleEngine whose semantic channel returns ``results``.

    Reranking is OFF (it only reorders, never adds/drops a row) to keep the test
    hermetic and fast. The mocked storage ignores the recency-window
    ``created_after`` / ``created_before`` kwargs, so the WINDOW path's narrowing
    is NOT what this test measures — it measures that the window path runs no
    event-time post-filter at all, while the occurred_at path does.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    storage.search_similar_chunks = AsyncMock(return_value=results)
    storage.search_similar_entities = AsyncMock(return_value=[])
    engine._storage = storage

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    engine._embedder = embedder
    engine._connected = True
    return engine


def _facade_with_engine(engine: ChronicleEngine) -> Khora:
    """A Khora wired to drive ``recall`` end to end into ``engine``.

    Same ``__new__`` construction as the capture harness, but the engine is the
    REAL ChronicleEngine (over mocked storage) so the facade's start_time/filter
    translation flows all the way through the post-filter.
    """
    kb = Khora.__new__(Khora)
    kb._engine_name = "chronicle"
    kb._get_engine = lambda: engine  # type: ignore[method-assign]
    kb._resolve_namespace = AsyncMock(return_value=uuid4())  # type: ignore[method-assign]
    kb._dispatch_hook = AsyncMock(return_value=None)  # type: ignore[method-assign]

    async def _passthrough(result: RecallResult, namespace_id: Any) -> RecallResult:
        return result

    kb._upgrade_recall_documents = _passthrough  # type: ignore[method-assign]
    return kb


@pytest.mark.asyncio
async def test_anchorless_chunk_survives_window_but_excluded_by_occurred_at_filter() -> None:
    # WINDOW axis (deprecated start_time): the facade forwards a temporal_filter
    # window and NO filter_ast, so no event-time post-filter runs → the anchor-less
    # chunk survives.
    chunk_window = _anchorless_chunk()
    kb_window = _facade_with_engine(_real_engine([(chunk_window, 0.9)]))
    with pytest.warns(DeprecationWarning):
        window_result = await kb_window.recall(
            "alpha",
            namespace=uuid4(),
            limit=10,
            mode=SearchMode.VECTOR,
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
    assert chunk_window.id in {c.id for c in window_result.chunks}, (
        "an anchor-less chunk recent by ingest time must SURVIVE the deprecated "
        "start_time recency window (the facade folds NO event-time post-filter)"
    )

    # EVENT-TIME axis (public filter={'occurred_at': ...}): the facade forwards a
    # filter_ast → the post-filter record's occurred_at = COALESCE(occurred_at,
    # source_timestamp) is None for an unanchored chunk → the positive $gte drops it.
    chunk_filter = _anchorless_chunk()
    kb_filter = _facade_with_engine(_real_engine([(chunk_filter, 0.9)]))
    filter_result = await kb_filter.recall(
        "alpha",
        namespace=uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}},
    )
    assert filter_result.chunks == [], (
        "the equivalent occurred_at filter EXCLUDES the same anchor-less chunk "
        "(no created_at fallback) — proving the two axes diverge. Folding the "
        "deprecated bounds into this filter (the regression) would false-empty the "
        "window path too."
    )


# ===========================================================================
# (3) Boundary inclusivity on the PUBLIC occurred_at filter path.
# ===========================================================================
#
# The deprecated bounds no longer emit any range AST after the fix, so the
# ``$gte``-inclusive / ``$lt``-exclusive boundary semantics now live ONLY on the
# public ``filter={"occurred_at": {...}}`` path. These pin the exact-instant
# behavior a row sitting EXACTLY on the bound must get: the lower bound includes
# it (``$gte``), the upper bound EXCLUDES it (``$lt``) while the inclusive upper
# (``$lte``) keeps it — the ``$lte`` vs ``$lt`` contrast is what makes the
# exclusivity load-bearing rather than a one-second rounding accident. Driven
# end to end through ``Khora.recall(filter=...)`` like the section above.

# The exact instant a boundary row sits on; the filter operand is the SAME instant.
_BOUND_INSTANT = datetime(2026, 6, 1, tzinfo=UTC)
_BOUND_ISO = "2026-06-01T00:00:00Z"


def _chunk_at(instant: datetime) -> Chunk:
    """A chunk whose effective event time is exactly ``instant``.

    ``occurred_at`` is left ``None`` (the legacy pgvector DTO shape) so the record's
    effective event time resolves via ``COALESCE(occurred_at, source_timestamp)`` =
    ``source_timestamp`` — exercising the same recovery path the shim protects.
    """
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="alpha boundary",
        created_at=instant,
        metadata={},
        occurred_at=None,
        source_timestamp=instant,
    )


@pytest.mark.asyncio
async def test_occurred_at_gte_is_inclusive_at_the_exact_bound() -> None:
    # A row whose event time equals the bound SURVIVES a $gte (inclusive lower).
    at_bound = _chunk_at(_BOUND_INSTANT)
    kb = _facade_with_engine(_real_engine([(at_bound, 0.9)]))
    result = await kb.recall(
        "alpha",
        namespace=uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$gte": _BOUND_ISO}},
    )
    assert at_bound.id in {c.id for c in result.chunks}, (
        "occurred_at $gte must be INCLUSIVE at the exact bound — a row on the lower edge survives"
    )


@pytest.mark.asyncio
async def test_occurred_at_lt_is_exclusive_at_the_exact_bound_unlike_lte() -> None:
    # A row whose event time equals the bound is DROPPED by $lt (exclusive upper)
    # but KEPT by $lte (inclusive upper). The contrast pins the exclusivity rather
    # than an off-by-a-second artifact.
    dropped = _chunk_at(_BOUND_INSTANT)
    kb_lt = _facade_with_engine(_real_engine([(dropped, 0.9)]))
    lt_result = await kb_lt.recall(
        "alpha",
        namespace=uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$lt": _BOUND_ISO}},
    )
    assert lt_result.chunks == [], "occurred_at $lt must EXCLUDE a row sitting exactly on the upper bound"

    kept = _chunk_at(_BOUND_INSTANT)
    kb_lte = _facade_with_engine(_real_engine([(kept, 0.9)]))
    lte_result = await kb_lte.recall(
        "alpha",
        namespace=uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$lte": _BOUND_ISO}},
    )
    assert kept.id in {c.id for c in lte_result.chunks}, (
        "occurred_at $lte must KEEP the same boundary row — the $lt/$lte contrast is "
        "what makes the $lt exclusivity meaningful"
    )
