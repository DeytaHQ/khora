"""Regression (#1222): a bare year-like number must not auto-apply a date filter.

Chronicle's recall ran ``TemporalResolver.resolve_fast(query)`` unconditionally
whenever the caller passed no explicit ``temporal_filter`` and
``enable_temporal_resolver`` was on. ``resolve_fast`` falls back to a ``20\\d{2}``
regex, so any query mentioning a four-digit ``20xx`` token (a version, a room /
model number) was treated as a date and turned into a recency filter at
confidence 0.85 - silently narrowing every channel on
``COALESCE(source_timestamp, created_at)`` and dropping older results.

VectorCypher does not have this bug: it only resolves a temporal filter once a
temporal-intent detector flags the query as actually about time. These tests
pin that Chronicle gates the resolver on the same intent check, so a
non-temporal year-bearing query pushes ``created_after=None`` into the channels.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.query import SearchMode
from khora.query.router import QueryComplexity, RoutingDecision


class _RecordingCoordinator:
    """Coordinator double recording the temporal bounds each channel receives."""

    def __init__(self) -> None:
        self.search_similar_chunks_calls: list[dict[str, Any]] = []
        self.search_fulltext_chunks_calls: list[dict[str, Any]] = []

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        created_after: Any | None = None,
        created_before: Any | None = None,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_similar_chunks_calls.append({"created_after": created_after, "created_before": created_before})
        return []

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query: str,
        *,
        limit: int = 10,
        created_after: Any | None = None,
        created_before: Any | None = None,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_fulltext_chunks_calls.append({"created_after": created_after, "created_before": created_before})
        return []

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        return []

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        return {}

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


def _wire(engine: ChronicleEngine, coord: _RecordingCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


def _routing(complexity: QueryComplexity) -> RoutingDecision:
    return RoutingDecision(
        complexity=complexity,
        use_graph=complexity is not QueryComplexity.SIMPLE,
        graph_depth=0 if complexity is QueryComplexity.SIMPLE else 1,
        confidence=0.9,
        reasoning="test",
    )


def _bounded(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if c["created_after"] is not None or c["created_before"] is not None]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["release notes 2023", "config for room 2099"])
async def test_year_token_does_not_apply_date_filter(query: str) -> None:
    """A non-temporal query mentioning a 20xx token must not push a date filter."""
    ns_id = uuid4()
    coord = _RecordingCoordinator()
    engine = _bare_engine()
    _wire(engine, coord)

    with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
        await engine.recall(query, ns_id, limit=5, mode=SearchMode.HYBRID)

    bad = _bounded(coord.search_similar_chunks_calls) + _bounded(coord.search_fulltext_chunks_calls)
    assert not bad, (
        f"Query {query!r} has no temporal intent but a date filter was pushed into the channels: {bad!r} (#1222)"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_temporal_query_still_resolves() -> None:
    """A genuinely temporal query must still resolve and push the date filter."""
    ns_id = uuid4()
    coord = _RecordingCoordinator()
    engine = _bare_engine()
    _wire(engine, coord)

    with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
        await engine.recall("meetings last week", ns_id, limit=5, mode=SearchMode.HYBRID)

    bounded = _bounded(coord.search_similar_chunks_calls) + _bounded(coord.search_fulltext_chunks_calls)
    assert bounded, "A 'last week' query should still resolve and push a recency window into the channels"
