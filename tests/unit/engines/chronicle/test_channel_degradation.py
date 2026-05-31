"""ADR-001 reference impl: chronicle channel-failure observability.

When a chronicle retrieval channel silently fails or takes a fallback
path, the engine MUST:

1. Append a ``Degradation`` entry to
   ``RecallResult.engine_info["degradations"]``
2. Bump ``khora.chronicle.channel.degraded_total{channel, reason}``
3. Log at WARNING (not DEBUG)

These tests cover PR #901 (BM25 / channel-gather degradation) and
PR #906 (temporal-channel silent fallback).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from loguru import logger as loguru_logger

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk
from khora.engines.chronicle.engine import ChronicleEngine
from khora.query import SearchMode


# Bridge loguru into pytest caplog. Chronicle warnings use loguru's logger,
# but pytest's ``caplog`` only captures stdlib ``logging`` records by default.
# Adding a propagating sink reroutes loguru output to a stdlib logger that
# caplog can observe.
@pytest.fixture
def loguru_caplog() -> Iterator[str]:
    bridge_name = "khora.test.channel_degradation"

    def _sink(message: Any) -> None:
        record = message.record
        level = record["level"].no
        logging.getLogger(bridge_name).log(level, record["message"])

    sink_id = loguru_logger.add(_sink, level="WARNING", format="{message}")
    try:
        yield bridge_name
    finally:
        loguru_logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "x", *, namespace_id: UUID | None = None) -> Chunk:
    ns_id = namespace_id or uuid4()
    return Chunk(
        namespace_id=ns_id,
        document_id=uuid4(),
        content=content,
        chunk_index=0,
    )


class _Coord:
    """Minimal coordinator double, configurable per-method to raise."""

    def __init__(
        self,
        *,
        semantic_results: list[tuple[Chunk, float]] | None = None,
        bm25_raises: Exception | None = None,
        semantic_raises: Exception | None = None,
        entity_search_raises: Exception | None = None,
        query_events_raises: Exception | None = None,
        get_chunks_batch_raises: Exception | None = None,
    ) -> None:
        self._semantic_results = semantic_results or []
        self._bm25_raises = bm25_raises
        self._semantic_raises = semantic_raises
        self._entity_search_raises = entity_search_raises
        self._query_events_raises = query_events_raises
        self._get_chunks_batch_raises = get_chunks_batch_raises

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        if self._semantic_raises is not None:
            raise self._semantic_raises
        return list(self._semantic_results)

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query: str,
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        if self._bm25_raises is not None:
            raise self._bm25_raises
        return []

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        if self._entity_search_raises is not None:
            raise self._entity_search_raises
        return []

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._query_events_raises is not None:
            raise self._query_events_raises
        return []

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        if self._get_chunks_batch_raises is not None:
            raise self._get_chunks_batch_raises
        return {}

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Any]:
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    # Disable reranking - it loads a 90MB cross-encoder model on first use
    # and is irrelevant to channel-degradation behaviour.
    cfg = KhoraConfig(
        database_url="postgresql://localhost/test",
        query=QuerySettings(enable_reranking=False),
    )
    return ChronicleEngine(cfg, **kwargs)


def _wire(engine: ChronicleEngine, coord: _Coord) -> None:
    engine._storage = coord  # type: ignore[assignment]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# BM25 channel (#901)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_runtime_error_records_degradation(caplog: pytest.LogCaptureFixture, loguru_caplog: str) -> None:
    """RuntimeError from BM25 (fulltext backend missing) records degradation + WARNING."""
    ns_id = uuid4()
    coord = _Coord(
        semantic_results=[(_make_chunk("a"), 0.8)],
        bm25_raises=RuntimeError("fulltext backend not configured"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    with caplog.at_level(logging.WARNING, logger=loguru_caplog):
        result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    degradations = result.engine_info["degradations"]
    bm25_entries = [d for d in degradations if d["component"] == "chronicle.bm25"]
    assert len(bm25_entries) == 1
    entry = bm25_entries[0]
    assert entry["reason"] == "fulltext_backend_unavailable"
    assert entry["exception"] == "RuntimeError"

    # WARNING was emitted (not DEBUG).
    warn_logs = [r for r in caplog.records if r.levelno >= logging.WARNING and "BM25" in r.getMessage()]
    assert warn_logs, "Expected a WARNING-level log for BM25 degradation"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_arbitrary_exception_records_channel_exception() -> None:
    """Non-RuntimeError exceptions from BM25 record reason='channel_exception'."""
    ns_id = uuid4()
    coord = _Coord(
        semantic_results=[(_make_chunk("a"), 0.8)],
        bm25_raises=ValueError("malformed tsquery"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    degradations = result.engine_info["degradations"]
    bm25_entries = [d for d in degradations if d["component"] == "chronicle.bm25"]
    assert len(bm25_entries) == 1
    assert bm25_entries[0]["reason"] == "channel_exception"
    assert bm25_entries[0]["exception"] == "ValueError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_bumps_metric_counter() -> None:
    """``khora.chronicle.channel.degraded_total`` increments on BM25 failure."""
    ns_id = uuid4()
    coord = _Coord(
        semantic_results=[(_make_chunk("a"), 0.8)],
        bm25_raises=RuntimeError("backend missing"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    with patch("khora.engines.chronicle.engine._CHANNEL_DEGRADED_COUNTER") as mock_counter:
        await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    assert mock_counter.add.called, "Expected channel-degraded counter to be bumped"
    # Inspect first call: increment of 1, channel="bm25".
    args, kwargs = mock_counter.add.call_args_list[0]
    assert args[0] == 1
    attrs = kwargs.get("attributes", {})
    assert attrs.get("channel") == "bm25"
    assert attrs.get("reason") == "fulltext_backend_unavailable"


# ---------------------------------------------------------------------------
# Channel-gather failures (semantic / temporal / entity raise inside gather)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_channel_gather_exception_records_degradation(
    caplog: pytest.LogCaptureFixture, loguru_caplog: str
) -> None:
    """Exception from the semantic channel inside the gather() block is recorded.

    ``search_similar_chunks`` raises -> ``asyncio.gather(return_exceptions=True)``
    captures it -> the recall site records a ``chronicle.semantic`` degradation.
    """
    ns_id = uuid4()
    coord = _Coord(
        semantic_raises=RuntimeError("vector backend offline"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    with caplog.at_level(logging.WARNING, logger=loguru_caplog):
        result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    degradations = result.engine_info["degradations"]
    semantic_entries = [d for d in degradations if d["component"] == "chronicle.semantic"]
    assert len(semantic_entries) >= 1
    assert semantic_entries[0]["reason"] == "channel_exception"
    assert semantic_entries[0]["exception"] == "RuntimeError"

    warn_logs = [r for r in caplog.records if r.levelno >= logging.WARNING and "semantic" in r.getMessage().lower()]
    assert warn_logs, "Expected a WARNING-level log when semantic channel raises"


# ---------------------------------------------------------------------------
# Temporal channel (#906)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_temporal_query_events_failure_records_degradation(
    caplog: pytest.LogCaptureFixture, loguru_caplog: str
) -> None:
    """``query_events`` raising forces fallback - records temporal degradation."""
    from datetime import UTC, datetime

    from khora.query.temporal import TemporalFilter

    ns_id = uuid4()
    coord = _Coord(
        semantic_results=[(_make_chunk("a"), 0.8)],
        query_events_raises=RuntimeError("events table missing"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    temporal_filter = TemporalFilter(
        start_time=datetime(2026, 4, 1, tzinfo=UTC),
        end_time=datetime(2026, 4, 30, tzinfo=UTC),
    )

    with caplog.at_level(logging.WARNING, logger=loguru_caplog):
        result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID, temporal_filter=temporal_filter)

    degradations = result.engine_info["degradations"]
    temporal_entries = [d for d in degradations if d["component"] == "chronicle.temporal_channel"]
    assert len(temporal_entries) >= 1
    assert temporal_entries[0]["reason"] == "events_query_failed"
    assert temporal_entries[0]["exception"] == "RuntimeError"

    warn_logs = [r for r in caplog.records if r.levelno >= logging.WARNING and "temporal" in r.getMessage().lower()]
    assert warn_logs, "Expected a WARNING-level log for temporal degradation"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_temporal_bumps_metric_counter() -> None:
    """``khora.chronicle.channel.degraded_total{channel=temporal_channel}`` bumped."""
    from datetime import UTC, datetime

    from khora.query.temporal import TemporalFilter

    ns_id = uuid4()
    coord = _Coord(
        semantic_results=[(_make_chunk("a"), 0.8)],
        query_events_raises=RuntimeError("events offline"),
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    temporal_filter = TemporalFilter(
        start_time=datetime(2026, 4, 1, tzinfo=UTC),
        end_time=datetime(2026, 4, 30, tzinfo=UTC),
    )

    with patch("khora.engines.chronicle.engine._CHANNEL_DEGRADED_COUNTER") as mock_counter:
        await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID, temporal_filter=temporal_filter)

    # Find the temporal_channel bump.
    temporal_calls = [
        call
        for call in mock_counter.add.call_args_list
        if call.kwargs.get("attributes", {}).get("channel") == "temporal_channel"
    ]
    assert temporal_calls, "Expected a temporal_channel degraded_total bump"
    assert temporal_calls[0].args[0] == 1
    assert temporal_calls[0].kwargs["attributes"]["reason"] == "events_query_failed"


# ---------------------------------------------------------------------------
# Happy path - no degradations, list is empty
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_records_empty_degradations() -> None:
    """When nothing fails, ``degradations`` is present and empty."""
    ns_id = uuid4()
    coord = _Coord(semantic_results=[(_make_chunk("a"), 0.8)])
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    assert "degradations" in result.engine_info
    assert result.engine_info["degradations"] == []
