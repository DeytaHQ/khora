"""Regression: temporal filter must reach all 4 Chronicle channels.

Pre-fix, only the temporal channel forwarded ``created_after``/``created_before``
to the SQL layer; semantic + BM25 + entity channels scanned the full namespace,
so a 20-day-old chunk leaked through a 7-day window. These tests pin the wiring
contract: when a ``temporal_filter`` is provided, every channel that hits the
storage layer carries the bounds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, ChunkMetadata, Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.query import SearchMode
from khora.query.router import QueryComplexity, RoutingDecision
from khora.query.temporal import TemporalFilter


def _make_chunk(
    *,
    namespace_id: UUID,
    created_at: datetime | None = None,
    source_timestamp: datetime | None = None,
) -> Chunk:
    document_id = uuid4()
    return Chunk(
        namespace_id=namespace_id,
        document_id=document_id,
        content="x",
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
        created_at=created_at or datetime.now(UTC),
        source_timestamp=source_timestamp,
    )


class _PushdownCoordinator:
    """Coordinator double that records the kwargs every channel receives."""

    def __init__(
        self,
        *,
        chunks: dict[UUID, Chunk] | None = None,
        entity_search_results: list[tuple[UUID, float]] | None = None,
        entities: dict[UUID, Entity] | None = None,
    ) -> None:
        self._chunks = chunks or {}
        self._entity_search_results = entity_search_results or []
        self._entities = entities or {}

        self.search_similar_chunks_calls: list[dict[str, Any]] = []
        self.search_fulltext_chunks_calls: list[dict[str, Any]] = []
        self.search_similar_entities_calls: list[dict[str, Any]] = []
        self.query_events_calls: list[dict[str, Any]] = []

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_similar_chunks_calls.append(
            {"created_after": created_after, "created_before": created_before, "limit": limit}
        )
        return [(c, 0.9) for c in self._chunks.values()]

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query: str,
        *,
        limit: int = 10,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_fulltext_chunks_calls.append(
            {"created_after": created_after, "created_before": created_before, "limit": limit}
        )
        return [(c, 0.5) for c in self._chunks.values()]

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        self.search_similar_entities_calls.append({"limit": limit})
        return list(self._entity_search_results)

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        return {eid: self._entities[eid] for eid in entity_ids if eid in self._entities}

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.query_events_calls.append({"kwargs": kwargs})
        return []


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


def _wire(engine: ChronicleEngine, coord: _PushdownCoordinator) -> None:
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


@pytest.mark.unit
class TestChronicleTemporalPushdown:
    """Regression: every channel must forward the temporal bounds."""

    @pytest.mark.asyncio
    async def test_semantic_channel_forwards_temporal_bounds(self) -> None:
        ns_id = uuid4()
        coord = _PushdownCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        start = datetime.now(UTC) - timedelta(days=7)
        end = datetime.now(UTC)
        tf = TemporalFilter(start_time=start, end_time=end)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
            await engine.recall("falcon launch", ns_id, limit=5, mode=SearchMode.HYBRID, temporal_filter=tf)

        # The semantic channel calls search_similar_chunks WITH the bounds.
        # The temporal channel's chunk-fallback path also calls
        # search_similar_chunks, so we just need to assert that AT LEAST one
        # call carries the bounds (the semantic-channel one).
        bounded_calls = [c for c in coord.search_similar_chunks_calls if c["created_after"] is not None]
        assert bounded_calls, (
            f"semantic channel did not forward created_after; calls={coord.search_similar_chunks_calls!r}"
        )
        assert all(c["created_before"] is not None for c in bounded_calls)
        assert bounded_calls[0]["created_after"] == start
        assert bounded_calls[0]["created_before"] == end

    @pytest.mark.asyncio
    async def test_bm25_channel_forwards_temporal_bounds(self) -> None:
        ns_id = uuid4()
        coord = _PushdownCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        start = datetime.now(UTC) - timedelta(days=7)
        end = datetime.now(UTC)
        tf = TemporalFilter(start_time=start, end_time=end)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
            await engine.recall("falcon launch", ns_id, limit=5, mode=SearchMode.HYBRID, temporal_filter=tf)

        assert coord.search_fulltext_chunks_calls, "BM25 channel should have run"
        bm25_call = coord.search_fulltext_chunks_calls[0]
        assert bm25_call["created_after"] == start
        assert bm25_call["created_before"] == end

    @pytest.mark.asyncio
    async def test_entity_channel_filters_old_chunks(self) -> None:
        """Entity channel filters post-hydration since get_chunks_batch has no WHERE hook."""
        ns_id = uuid4()

        # One chunk inside the 7-day window, one 20 days old.
        recent = _make_chunk(namespace_id=ns_id, created_at=datetime.now(UTC) - timedelta(days=5))
        old = _make_chunk(namespace_id=ns_id, created_at=datetime.now(UTC) - timedelta(days=20))

        eid_recent, eid_old = uuid4(), uuid4()
        ent_recent = Entity(
            id=eid_recent,
            namespace_id=ns_id,
            name="Falcon",
            entity_type="VEHICLE",
            source_chunk_ids=[recent.id],
        )
        ent_old = Entity(
            id=eid_old,
            namespace_id=ns_id,
            name="FalconOld",
            entity_type="VEHICLE",
            source_chunk_ids=[old.id],
        )

        coord = _PushdownCoordinator(
            chunks={recent.id: recent, old.id: old},
            entity_search_results=[(eid_recent, 0.9), (eid_old, 0.85)],
            entities={eid_recent: ent_recent, eid_old: ent_old},
        )
        engine = _bare_engine()
        _wire(engine, coord)

        start = datetime.now(UTC) - timedelta(days=7)
        end = datetime.now(UTC) + timedelta(days=1)

        results = await engine._entity_channel(
            ns_id,
            "q",
            [0.1] * 8,
            limit=10,
            created_after=start,
            created_before=end,
        )

        returned_ids = {chunk.id for chunk, _ in results}
        assert recent.id in returned_ids
        assert old.id not in returned_ids, "20-day-old chunk leaked through 7-day window"

    @pytest.mark.asyncio
    async def test_no_filter_means_no_pushdown(self) -> None:
        """Without temporal_filter, channels still run with created_after=None."""
        ns_id = uuid4()
        coord = _PushdownCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
            await engine.recall("falcon", ns_id, limit=5, mode=SearchMode.HYBRID)

        # BM25 should run with created_after=None (no filter).
        if coord.search_fulltext_chunks_calls:
            assert coord.search_fulltext_chunks_calls[0]["created_after"] is None
            assert coord.search_fulltext_chunks_calls[0]["created_before"] is None
