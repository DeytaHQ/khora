"""Unit tests for community projection onto RecallResult (#1308).

After the VectorCypher engine assembles the result entities it fetches the
materialized dream communities (#1276) those entities belong to via the
backend-capability-gated ``get_entity_communities`` reader and surfaces them on
``RecallResult.communities``. These tests cover the projection helper in
isolation: de-dupe, cap, capability-gate -> empty, zero-cost when no entities,
and the ADR-001 degrade-on-failure path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import CommunityNode
from khora.core.models.recall import RecallEntity
from khora.engines.vectorcypher.engine import _VC_MAX_COMMUNITIES, VectorCypherEngine


def _engine_with_storage(storage: object) -> VectorCypherEngine:
    """Build a VectorCypherEngine bypassing __init__, with a mocked storage.

    ``_project_communities`` only depends on ``self._get_storage()``, which
    reads ``self._storage``; constructing a full engine would need a heavy
    KhoraConfig + live backends.
    """
    engine = object.__new__(VectorCypherEngine)
    engine._storage = storage  # type: ignore[attr-defined]
    return engine


def _entity(entity_id: UUID) -> RecallEntity:
    return RecallEntity(
        id=entity_id,
        name="Alice",
        entity_type="PERSON",
        description="",
        score=0.9,
        attributes={},
        mention_count=1,
        source_document_ids=[],
        source_chunk_ids=[],
    )


def _community(community_id: UUID, *, summary: str = "summary", depth: int = 1) -> CommunityNode:
    return CommunityNode(
        id=community_id,
        namespace_id=uuid4(),
        summary=summary,
        member_ids=[uuid4()],
        summary_depth=depth,
    )


@pytest.mark.unit
class TestCommunityProjection:
    """Projection wiring: de-dupe, cap, capability-gate, zero-cost, degrade."""

    @pytest.mark.asyncio
    async def test_projects_communities_for_member_entities(self) -> None:
        ns_id = uuid4()
        entity_id = uuid4()
        community = _community(uuid4(), summary="dream community summary")
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(return_value=[community])
        engine = _engine_with_storage(storage)

        degradations: list = []
        result = await engine._project_communities([_entity(entity_id)], namespace_id=ns_id, degradations=degradations)

        assert [c.id for c in result] == [community.id]
        assert result[0].summary == "dream community summary"
        assert degradations == []
        storage.get_entity_communities.assert_awaited_once_with([entity_id], namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_dedupes_communities_across_matched_entities(self) -> None:
        ns_id = uuid4()
        shared_id = uuid4()
        # Two entities both members of the same community: the reader may return
        # the community twice; the projection must de-dupe by community id.
        dup = _community(shared_id)
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(return_value=[dup, _community(shared_id)])
        engine = _engine_with_storage(storage)

        result = await engine._project_communities(
            [_entity(uuid4()), _entity(uuid4())], namespace_id=ns_id, degradations=[]
        )

        assert [c.id for c in result] == [shared_id]

    @pytest.mark.asyncio
    async def test_caps_community_count(self) -> None:
        ns_id = uuid4()
        many = [_community(uuid4()) for _ in range(_VC_MAX_COMMUNITIES + 5)]
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(return_value=many)
        engine = _engine_with_storage(storage)

        result = await engine._project_communities([_entity(uuid4())], namespace_id=ns_id, degradations=[])

        assert len(result) == _VC_MAX_COMMUNITIES

    @pytest.mark.asyncio
    async def test_caps_keeps_shallowest_depth_first(self) -> None:
        ns_id = uuid4()
        deep = [_community(uuid4(), depth=9) for _ in range(_VC_MAX_COMMUNITIES)]
        shallow = _community(uuid4(), depth=0, summary="shallow")
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(return_value=[*deep, shallow])
        engine = _engine_with_storage(storage)

        result = await engine._project_communities([_entity(uuid4())], namespace_id=ns_id, degradations=[])

        assert len(result) == _VC_MAX_COMMUNITIES
        # The shallowest (most specific) community survives the cap.
        assert shallow.id in {c.id for c in result}
        assert result[0].summary_depth == 0

    @pytest.mark.asyncio
    async def test_empty_entities_skips_reader_zero_cost(self) -> None:
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock()
        engine = _engine_with_storage(storage)

        result = await engine._project_communities([], namespace_id=uuid4(), degradations=[])

        assert result == []
        # Zero-cost: the reader is never called when no entities matched.
        storage.get_entity_communities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_materialized_communities_returns_empty(self) -> None:
        # A backend without materialized communities (or lacking the reader) is
        # surfaced as the coordinator returning []: projection is empty, no
        # degradation recorded.
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(return_value=[])
        engine = _engine_with_storage(storage)

        degradations: list = []
        result = await engine._project_communities([_entity(uuid4())], namespace_id=uuid4(), degradations=degradations)

        assert result == []
        assert degradations == []

    @pytest.mark.asyncio
    async def test_reader_failure_degrades_to_empty(self) -> None:
        storage = AsyncMock()
        storage.get_entity_communities = AsyncMock(side_effect=RuntimeError("neo4j down"))
        engine = _engine_with_storage(storage)

        degradations: list = []
        result = await engine._project_communities([_entity(uuid4())], namespace_id=uuid4(), degradations=degradations)

        # Degrades to empty, never raises (ADR-001).
        assert result == []
        assert len(degradations) == 1
        deg = degradations[0]
        assert deg["component"] == "vectorcypher.community_projection"
        assert deg["reason"] == "fetch_failed"
        assert "neo4j down" in deg["exception"]
