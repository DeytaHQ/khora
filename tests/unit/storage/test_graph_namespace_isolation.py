"""Namespace-isolation tests for StorageCoordinator graph getters.

`StorageCoordinator.get_entity / get_relationship / get_episode` historically
took only an ID and returned whatever the graph backend held under that ID. The
backend protocol does the same, so a caller scoped to namespace B that knew an
entity ID belonging to namespace A would receive the namespace-A entity
verbatim — a cross-namespace IDOR primitive on the public graph-adapter API in
multi-tenant deployments.

These tests assert the facade now requires a ``namespace_id`` kwarg and returns
``None`` whenever the persisted row's namespace_id does not match the caller's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Entity, Episode, Relationship
from khora.storage.coordinator import StorageCoordinator


def _relational_mock() -> MagicMock:
    """Relational backend whose ``resolve_namespace`` echoes its input.

    The coordinator now resolves the stable namespace_id to the row id at
    the top of its public read methods (idempotent on row ids), so the mock
    must return whatever id the test passes through unchanged.
    """
    rel = MagicMock()
    rel.resolve_namespace = AsyncMock(side_effect=lambda ns: ns)
    return rel


@pytest.mark.asyncio
async def test_get_entity_cross_namespace_returns_none() -> None:
    """Entity belongs to ns A; caller scoped to ns B receives None."""
    ns_a = uuid4()
    ns_b = uuid4()
    entity = Entity(namespace_id=ns_a, name="alice", entity_type="PERSON")

    graph = MagicMock()
    graph.get_entity = AsyncMock(return_value=entity)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_entity(entity.id, namespace_id=ns_b)

    assert result is None


@pytest.mark.asyncio
async def test_get_entity_same_namespace_returns_entity() -> None:
    """Caller scoped to the entity's namespace receives the entity."""
    ns_a = uuid4()
    entity = Entity(namespace_id=ns_a, name="alice", entity_type="PERSON")

    graph = MagicMock()
    graph.get_entity = AsyncMock(return_value=entity)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_entity(entity.id, namespace_id=ns_a)

    assert result is entity


@pytest.mark.asyncio
async def test_get_entity_missing_returns_none() -> None:
    """Unknown entity ID returns None regardless of namespace."""
    graph = MagicMock()
    graph.get_entity = AsyncMock(return_value=None)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_entity(uuid4(), namespace_id=uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_get_entity_requires_namespace_id_kwarg() -> None:
    """Calling without namespace_id raises TypeError (breaking API change)."""
    graph = MagicMock()
    graph.get_entity = AsyncMock(return_value=None)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    with pytest.raises(TypeError):
        await coord.get_entity(uuid4())  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_get_relationship_cross_namespace_returns_none() -> None:
    """Relationship belongs to ns A; caller scoped to ns B receives None."""
    ns_a = uuid4()
    ns_b = uuid4()
    rel = Relationship(
        namespace_id=ns_a,
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type="KNOWS",
    )

    graph = MagicMock()
    graph.get_relationship = AsyncMock(return_value=rel)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_relationship(rel.id, namespace_id=ns_b)

    assert result is None


@pytest.mark.asyncio
async def test_get_relationship_same_namespace_returns_relationship() -> None:
    """Caller scoped to the relationship's namespace receives the relationship."""
    ns_a = uuid4()
    rel = Relationship(
        namespace_id=ns_a,
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type="KNOWS",
    )

    graph = MagicMock()
    graph.get_relationship = AsyncMock(return_value=rel)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_relationship(rel.id, namespace_id=ns_a)

    assert result is rel


@pytest.mark.asyncio
async def test_get_relationship_requires_namespace_id_kwarg() -> None:
    """Calling without namespace_id raises TypeError."""
    graph = MagicMock()
    graph.get_relationship = AsyncMock(return_value=None)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    with pytest.raises(TypeError):
        await coord.get_relationship(uuid4())  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_get_episode_cross_namespace_returns_none() -> None:
    """Episode belongs to ns A; caller scoped to ns B receives None."""
    ns_a = uuid4()
    ns_b = uuid4()
    ep = Episode(
        namespace_id=ns_a,
        name="meeting",
        occurred_at=datetime.now(UTC),
    )

    graph = MagicMock()
    graph.get_episode = AsyncMock(return_value=ep)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_episode(ep.id, namespace_id=ns_b)

    assert result is None


@pytest.mark.asyncio
async def test_get_episode_same_namespace_returns_episode() -> None:
    """Caller scoped to the episode's namespace receives the episode."""
    ns_a = uuid4()
    ep = Episode(
        namespace_id=ns_a,
        name="meeting",
        occurred_at=datetime.now(UTC),
    )

    graph = MagicMock()
    graph.get_episode = AsyncMock(return_value=ep)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    result = await coord.get_episode(ep.id, namespace_id=ns_a)

    assert result is ep


@pytest.mark.asyncio
async def test_get_episode_requires_namespace_id_kwarg() -> None:
    """Calling without namespace_id raises TypeError."""
    graph = MagicMock()
    graph.get_episode = AsyncMock(return_value=None)
    coord = StorageCoordinator(relational=_relational_mock(), graph=graph)

    with pytest.raises(TypeError):
        await coord.get_episode(uuid4())  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_get_entity_no_graph_returns_none() -> None:
    """When no graph backend is configured, returns None."""
    coord = StorageCoordinator(relational=_relational_mock())
    result = await coord.get_entity(uuid4(), namespace_id=uuid4())
    assert result is None
