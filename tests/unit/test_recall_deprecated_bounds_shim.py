"""The deprecated ``start_time``/``end_time`` recall bounds are a WINDOW axis.

``Khora.recall(start_time=..., end_time=...)`` is the legacy recency-window API:
each engine narrows its channel reads on ``COALESCE(source_timestamp, created_at)``,
so a chunk that is recent by ingest time survives the window even when it carries
no event-time anchor (``source_timestamp`` / ``occurred_at`` both ``None`` — the
shape a plain ``remember(content=...)`` produces).

These tests pin the facade-level translation in ``Khora.recall``: the deprecated
bounds forward ONLY a ``temporal_filter`` recency window and DO NOT fold an
``occurred_at`` ``filter_ast``. Folding one would AND an event-time post-filter
(``record.occurred_at = COALESCE(occurred_at, source_timestamp)``, no created_at
fallback) on top of the window and false-empty every anchor-less chunk — the
regression these guards prevent. The public ``filter={"occurred_at": {...}}``
kwarg, by contrast, intentionally DOES produce a ``filter_ast`` and keeps its
documented event-time semantics; the final test pins that the two paths stay
distinct.

Hermetic: the engine is a capture stub, so no database, embedder, or LLM is
touched — the assertion is purely on what the facade hands to ``engine.recall``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.core.models.recall import RecallResult
from khora.khora import Khora
from khora.query import SearchMode

pytestmark = pytest.mark.unit


def _capture_engine(captured: dict[str, Any]) -> Any:
    """A minimal engine stub whose ``recall`` records its kwargs and returns empty."""
    ns = uuid4()

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
    engine._ns = ns
    return engine


def _facade(captured: dict[str, Any]) -> Khora:
    """A Khora wired only enough to drive ``recall``'s filter-resolution shim.

    Built via ``__new__`` so no config / connection is needed; the engine is the
    capture stub and the recall method's collaborators (namespace resolve, hook
    dispatch, document upgrade) are no-op stubs.
    """
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


@pytest.mark.asyncio
async def test_deprecated_start_time_forwards_window_not_occurred_at_filter() -> None:
    captured: dict[str, Any] = {}
    kb = _facade(captured)

    start = datetime(2026, 6, 1, tzinfo=UTC)
    with pytest.warns(DeprecationWarning):
        await kb.recall("alpha", namespace=uuid4(), mode=SearchMode.VECTOR, start_time=start)

    # The window is forwarded on the temporal axis.
    tf = captured["temporal_filter"]
    assert tf is not None, "start_time must forward a recency-window temporal_filter"
    assert tf.occurred_after == start

    # No event-time post-filter is folded on top — that is the false-empty guard.
    assert captured["filter_ast"] is None, (
        "the deprecated window bounds must NOT fold an occurred_at filter_ast; "
        "doing so AND-s an event-time post-filter with no created_at fallback and "
        "false-empties anchor-less chunks"
    )


@pytest.mark.asyncio
async def test_deprecated_end_time_forwards_window_not_occurred_at_filter() -> None:
    captured: dict[str, Any] = {}
    kb = _facade(captured)

    end = datetime(2026, 6, 1, tzinfo=UTC)
    with pytest.warns(DeprecationWarning):
        await kb.recall("alpha", namespace=uuid4(), mode=SearchMode.VECTOR, end_time=end)

    tf = captured["temporal_filter"]
    assert tf is not None
    assert tf.occurred_before == end
    assert captured["filter_ast"] is None


@pytest.mark.asyncio
async def test_public_occurred_at_filter_still_forwards_filter_ast() -> None:
    # Contrast: the NEW public API keeps its documented event-time semantics —
    # an occurred_at filter DOES produce a filter_ast (and no temporal_filter).
    captured: dict[str, Any] = {}
    kb = _facade(captured)

    await kb.recall(
        "alpha",
        namespace=uuid4(),
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}},
    )

    assert captured["filter_ast"] is not None, "filter={'occurred_at': ...} must still forward a filter_ast"
    assert captured["temporal_filter"] is None
