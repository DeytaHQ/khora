"""Cross-session expansion: re-sort-before-trim (#1118) and ADR-001 degradations (#1119).

``_cross_session_expand`` appends entity-linked chunks from other sessions with
discounted scores that are *designed to be competitive in the final ranking*.
Two contracts must hold:

* #1118 - The merged list MUST be score-sorted before ``recall`` trims it to
  ``[:limit]`` and min-max-normalizes it. Otherwise a high-scoring expansion
  chunk gets positionally cut while a lower-scoring primary survives, and the
  returned ``chunks`` order diverges from score order (violating #834: top
  chunk = 1.0, scores as normalized rank).

* #1119 - The three storage calls inside the helper
  (``search_similar_entities`` / ``get_entities_batch`` / ``get_chunks_batch``)
  MUST record an ADR-001 ``Degradation`` (component ``chronicle.cross_session``)
  and log at WARNING when they raise - not silently swallow and return the
  un-expanded list.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from loguru import logger as loguru_logger

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk, Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.query import SearchMode


@pytest.fixture
def loguru_caplog() -> Iterator[str]:
    """Bridge loguru WARNING output into pytest's stdlib ``caplog``."""
    bridge_name = "khora.test.cross_session_expand"

    def _sink(message: Any) -> None:
        record = message.record
        logging.getLogger(bridge_name).log(record["level"].no, record["message"])

    sink_id = loguru_logger.add(_sink, level="WARNING", format="{message}")
    try:
        yield bridge_name
    finally:
        loguru_logger.remove(sink_id)


def _make_chunk(content: str, *, namespace_id: UUID, session_id: str | None = None) -> Chunk:
    metadata = {"session_id": session_id} if session_id else {}
    return Chunk(
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        metadata=metadata,
    )


class _ExpandCoord:
    """Coordinator double that drives ``_cross_session_expand`` to completion.

    The semantic channel returns ``primary_chunks``; the entity-expansion path
    returns one entity whose ``source_chunk_ids`` point at ``expansion_chunks``.
    Any of the three expansion storage calls can be made to raise.
    """

    def __init__(
        self,
        *,
        namespace_id: UUID,
        primary_chunks: list[tuple[Chunk, float]],
        expansion_chunks: list[Chunk],
        entity_search_raises: Exception | None = None,
        entities_batch_raises: Exception | None = None,
        chunks_batch_raises: Exception | None = None,
    ) -> None:
        self._ns = namespace_id
        self._primary = primary_chunks
        self._expansion = expansion_chunks
        self._entity_search_raises = entity_search_raises
        self._entities_batch_raises = entities_batch_raises
        self._chunks_batch_raises = chunks_batch_raises
        self._entity_id = uuid4()

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        return list(self._primary)

    async def search_fulltext_chunks(
        self, namespace_id: UUID, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[tuple[Chunk, float]]:
        return []

    async def search_similar_entities(
        self, namespace_id: UUID, query_embedding: list[float], *, limit: int = 10, **kwargs: Any
    ) -> list[tuple[UUID, float]]:
        if self._entity_search_raises is not None:
            raise self._entity_search_raises
        return [(self._entity_id, 0.9)]

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        if self._entities_batch_raises is not None:
            raise self._entities_batch_raises
        entity = Entity(
            namespace_id=self._ns,
            name="ProjectX",
            entity_type="CONCEPT",
            source_chunk_ids=[c.id for c in self._expansion],
        )
        return {self._entity_id: entity}

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        if self._chunks_batch_raises is not None:
            raise self._chunks_batch_raises
        wanted = set(chunk_ids)
        return {c.id: c for c in self._expansion if c.id in wanted}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}


def _bare_engine() -> ChronicleEngine:
    cfg = KhoraConfig(
        database_url="postgresql://localhost/test",
        query=QuerySettings(enable_reranking=False),
    )
    return ChronicleEngine(cfg, router_enabled=False)


def _wire(engine: ChronicleEngine, coord: _ExpandCoord) -> None:
    from unittest.mock import AsyncMock, MagicMock

    engine._storage = coord  # type: ignore[assignment]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# #1118 - re-sort before the [:limit] trim
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expansion_chunk_outscores_primary_survives_trim() -> None:
    """A high-scoring expansion chunk must outrank and survive over a low primary.

    Primary scores: 0.9, 0.1. avg(top-5) = 0.5 -> discount = 0.4, so the first
    expansion chunk scores 0.4 - above the 0.1 primary. With limit=2 the trim
    must keep [0.9 primary, 0.4 expansion] in score order and DROP the 0.1
    primary. Pre-fix the un-sorted order is [0.9, 0.1, 0.4] and the trim keeps
    [0.9, 0.1], dropping the more-relevant expansion chunk.
    """
    ns_id = uuid4()
    high = _make_chunk("strong primary", namespace_id=ns_id, session_id="s1")
    low = _make_chunk("weak primary", namespace_id=ns_id, session_id="s1")
    expansion = _make_chunk("cross-session content", namespace_id=ns_id, session_id="s2")

    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(high, 0.9), (low, 0.1)],
        expansion_chunks=[expansion],
    )
    engine = _bare_engine()
    _wire(engine, coord)

    # "changed" triggers _CROSS_SESSION_INTENT.
    result = await engine.recall("how has this changed", ns_id, limit=2, mode=SearchMode.HYBRID)

    returned_ids = [c.id for c in result.chunks]
    assert len(returned_ids) == 2
    # The high primary and the expansion chunk survive; the low primary is cut.
    assert high.id in returned_ids
    assert expansion.id in returned_ids
    assert low.id not in returned_ids

    # #834: returned chunks are in score order, top chunk normalized to 1.0.
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merged_chunks_returned_in_score_order() -> None:
    """Without a trim cut, the full merged list is still score-descending (#834)."""
    ns_id = uuid4()
    high = _make_chunk("strong primary", namespace_id=ns_id, session_id="s1")
    low = _make_chunk("weak primary", namespace_id=ns_id, session_id="s1")
    expansion = _make_chunk("cross-session content", namespace_id=ns_id, session_id="s2")

    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(high, 0.9), (low, 0.1)],
        expansion_chunks=[expansion],
    )
    engine = _bare_engine()
    _wire(engine, coord)

    result = await engine.recall("how has this changed", ns_id, limit=10, mode=SearchMode.HYBRID)

    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    # Expansion chunk (score 0.4) ranks above the 0.1 primary.
    order = [c.id for c in result.chunks]
    assert order.index(expansion.id) < order.index(low.id)


# ---------------------------------------------------------------------------
# #1119 - ADR-001 degradation on swallowed expansion-storage exceptions
# ---------------------------------------------------------------------------


def _expansion_degradations(result: Any) -> list[dict[str, Any]]:
    return [d for d in result.engine_info["degradations"] if d["component"] == "chronicle.cross_session"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_search_failure_records_cross_session_degradation(
    caplog: pytest.LogCaptureFixture, loguru_caplog: str
) -> None:
    """``search_similar_entities`` raising records a cross_session degradation + WARNING."""
    ns_id = uuid4()
    primary = _make_chunk("primary", namespace_id=ns_id, session_id="s1")
    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(primary, 0.8)],
        expansion_chunks=[],
        entity_search_raises=RuntimeError("entity store offline"),
    )
    engine = _bare_engine()
    _wire(engine, coord)

    with caplog.at_level(logging.WARNING, logger=loguru_caplog):
        result = await engine.recall("how has this changed", ns_id, limit=5, mode=SearchMode.HYBRID)

    entries = _expansion_degradations(result)
    assert len(entries) == 1
    assert entries[0]["reason"] == "channel_exception"
    assert entries[0]["exception"] == "RuntimeError"

    # recall still returns the un-expanded primary chunk.
    assert [c.id for c in result.chunks] == [primary.id]

    warn_logs = [
        r for r in caplog.records if r.levelno >= logging.WARNING and "cross-session" in r.getMessage().lower()
    ]
    assert warn_logs, "Expected a WARNING-level log for cross-session degradation"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entities_batch_failure_records_cross_session_degradation() -> None:
    """``get_entities_batch`` raising records a cross_session degradation."""
    ns_id = uuid4()
    primary = _make_chunk("primary", namespace_id=ns_id, session_id="s1")
    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(primary, 0.8)],
        expansion_chunks=[],
        entities_batch_raises=RuntimeError("entities batch failed"),
    )
    engine = _bare_engine()
    _wire(engine, coord)

    result = await engine.recall("how has this changed", ns_id, limit=5, mode=SearchMode.HYBRID)

    entries = _expansion_degradations(result)
    assert len(entries) == 1
    assert entries[0]["reason"] == "channel_exception"
    assert entries[0]["exception"] == "RuntimeError"
    assert [c.id for c in result.chunks] == [primary.id]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chunks_batch_failure_records_cross_session_degradation() -> None:
    """``get_chunks_batch`` raising inside expansion records a cross_session degradation."""
    ns_id = uuid4()
    primary = _make_chunk("primary", namespace_id=ns_id, session_id="s1")
    expansion = _make_chunk("cross-session", namespace_id=ns_id, session_id="s2")
    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(primary, 0.8)],
        expansion_chunks=[expansion],
        chunks_batch_raises=RuntimeError("chunk fetch failed"),
    )
    engine = _bare_engine()
    _wire(engine, coord)

    result = await engine.recall("how has this changed", ns_id, limit=5, mode=SearchMode.HYBRID)

    entries = _expansion_degradations(result)
    assert len(entries) == 1
    assert entries[0]["reason"] == "channel_exception"
    assert entries[0]["exception"] == "RuntimeError"
    assert [c.id for c in result.chunks] == [primary.id]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expansion_bumps_degraded_counter() -> None:
    """``khora.chronicle.channel.degraded_total{channel=cross_session}`` is bumped."""
    from unittest.mock import patch

    ns_id = uuid4()
    primary = _make_chunk("primary", namespace_id=ns_id, session_id="s1")
    coord = _ExpandCoord(
        namespace_id=ns_id,
        primary_chunks=[(primary, 0.8)],
        expansion_chunks=[],
        entity_search_raises=RuntimeError("entity store offline"),
    )
    engine = _bare_engine()
    _wire(engine, coord)

    with patch("khora.engines.chronicle.engine._CHANNEL_DEGRADED_COUNTER") as mock_counter:
        await engine.recall("how has this changed", ns_id, limit=5, mode=SearchMode.HYBRID)

    cross_calls = [
        call
        for call in mock_counter.add.call_args_list
        if call.kwargs.get("attributes", {}).get("channel") == "cross_session"
    ]
    assert cross_calls, "Expected a cross_session degraded_total bump"
    assert cross_calls[0].args[0] == 1
    assert cross_calls[0].kwargs["attributes"]["reason"] == "channel_exception"
