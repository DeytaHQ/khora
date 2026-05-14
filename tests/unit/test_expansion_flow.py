"""Regression tests for the expansion pipeline (Issue #587).

The bug: ``load_entities`` / ``load_relationships`` in
``khora.pipelines.flows.expansion`` called ``get_entities_by_namespace``
on the graph or relational backends — a method that exists on no
backend. The correct method is ``list_entities`` (graph) and the
operation should also work on graph-less stacks where entities live on
the vector backend.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.pipelines.flows.expansion import load_entities, load_relationships


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_entities_routes_through_coordinator_list_entities() -> None:
    """load_entities must call coordinator.list_entities, not a non-existent backend method."""
    ns_id = uuid4()
    sample_entity = MagicMock(spec_set=["id", "name", "namespace_id"])

    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[sample_entity])

    result = await load_entities(ns_id, storage, limit=500)

    assert result == [sample_entity]
    storage.list_entities.assert_awaited_once_with(ns_id, limit=500)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_relationships_routes_through_coordinator_list_relationships() -> None:
    """load_relationships must call coordinator.list_relationships."""
    ns_id = uuid4()
    sample_rel = MagicMock(spec_set=["id", "source_entity_id", "target_entity_id"])

    storage = MagicMock()
    storage.list_relationships = AsyncMock(return_value=[sample_rel])

    result = await load_relationships(ns_id, storage, limit=42)

    assert result == [sample_rel]
    storage.list_relationships.assert_awaited_once_with(ns_id, limit=42)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_entities_graph_less_stack_returns_empty_list() -> None:
    """Graph-less stacks fall back to the vector backend, which returns empty
    when no entities exist — must not raise AttributeError as in #587.
    """
    ns_id = uuid4()

    # Simulate a real coordinator with no graph backend but a pgvector
    # backend that exposes list_entities (the fix).
    from khora.storage.coordinator import StorageCoordinator

    vector = MagicMock()
    vector.list_entities = AsyncMock(return_value=[])
    storage = StorageCoordinator.__new__(StorageCoordinator)
    storage.graph = None
    storage.vector = vector
    storage.relational = None
    storage.event_store = None

    result = await load_entities(ns_id, storage, limit=100)

    assert result == []
    vector.list_entities.assert_awaited_once_with(ns_id, entity_type=None, limit=100, offset=0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_relationships_graph_less_stack_returns_empty_list() -> None:
    """Same as above for relationships — no graph backend, but coordinator
    falls back to vector backend without raising.
    """
    ns_id = uuid4()
    from khora.storage.coordinator import StorageCoordinator

    vector = MagicMock()
    vector.list_relationships = AsyncMock(return_value=[])
    storage = StorageCoordinator.__new__(StorageCoordinator)
    storage.graph = None
    storage.vector = vector
    storage.relational = None
    storage.event_store = None

    result = await load_relationships(ns_id, storage, limit=200)

    assert result == []
    vector.list_relationships.assert_awaited_once_with(ns_id, relationship_type=None, limit=200, offset=0)
