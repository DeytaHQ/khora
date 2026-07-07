"""Regression tests for the expansion pipeline (Issue #587).

The bug: ``load_entities`` / ``load_relationships`` in
``khora.pipelines.flows.expansion`` called ``get_entities_by_namespace``
on the graph or relational backends — a method that exists on no
backend. The correct method is ``list_entities`` (graph) and the
operation should also work on graph-less stacks where entities live on
the vector backend.

Also covers the ``unify_entities`` pipeline diagnostic surface and
threshold passthrough added for Issue #865.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Entity
from khora.pipelines.flows.expansion import (
    load_entities,
    load_relationships,
    unify_entities,
)


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
    vector.list_entities.assert_awaited_once_with(ns_id, entity_type=None, source_chunk_ids=None, limit=100, offset=0)


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


# ---------------------------------------------------------------------------
# Issue #865: ``unify_entities`` threshold passthrough + diagnostic counters
# ---------------------------------------------------------------------------


def _build_oakhurst_storage() -> tuple[MagicMock, list[Entity]]:
    """Build a storage mock seeded with the Oakhurst repro pair.

    levenshtein_similarity("oakhurst", "john oakhurst") is ~0.615, which
    is below the default 0.85 fuzzy threshold but above 0.5.
    """
    ns_id = uuid4()
    e1 = Entity(id=uuid4(), name="oakhurst", entity_type="PERSON", namespace_id=ns_id)
    e2 = Entity(id=uuid4(), name="john oakhurst", entity_type="PERSON", namespace_id=ns_id)

    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[e1, e2])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.update_entity = AsyncMock(return_value=None)
    storage.dispatch_hook = AsyncMock(return_value=None)
    return storage, [e1, e2]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unify_entities_threshold_passthrough() -> None:
    """Lowering ``fuzzy_threshold`` via the pipeline kwarg flips merged_count."""
    storage, entities = _build_oakhurst_storage()
    namespace_id = entities[0].namespace_id

    # Default behavior: no merge for the Oakhurst variant pair.
    default_result = await unify_entities(
        namespace_id,
        storage=storage,
        store_results=False,
    )
    assert default_result["merged_count"] == 0

    # Reseed (the previous call may have mutated entity state in place).
    storage, _ = _build_oakhurst_storage()
    relaxed_result = await unify_entities(
        entities[0].namespace_id,
        storage=storage,
        store_results=False,
        fuzzy_threshold=0.5,
    )
    assert relaxed_result["merged_count"] == 1
    assert relaxed_result["unified_entities"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unify_entities_diagnostic_counters() -> None:
    """``unify_entities`` exposes per-strategy counters in its result dict."""
    storage, entities = _build_oakhurst_storage()
    namespace_id = entities[0].namespace_id

    default_result = await unify_entities(
        namespace_id,
        storage=storage,
        store_results=False,
    )

    # Counters present with all four strategy keys.
    assert set(default_result["candidates_evaluated"].keys()) == {
        "correlation",
        "exact_name",
        "embedding",
        "fuzzy",
    }
    assert set(default_result["pairs_above_threshold"].keys()) == {
        "correlation",
        "exact_name",
        "embedding",
        "fuzzy",
    }
    # Fuzzy strategy considered the Oakhurst pair but rejected it.
    assert default_result["candidates_evaluated"]["fuzzy"] >= 1
    assert default_result["pairs_above_threshold"]["fuzzy"] == 0

    # Flip the threshold and rerun: pair now clears the gate.
    storage, _ = _build_oakhurst_storage()
    relaxed_result = await unify_entities(
        namespace_id,
        storage=storage,
        store_results=False,
        fuzzy_threshold=0.5,
    )
    assert relaxed_result["candidates_evaluated"]["fuzzy"] >= 1
    assert relaxed_result["pairs_above_threshold"]["fuzzy"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unify_entities_empty_namespace_includes_counter_keys() -> None:
    """Empty-namespace early-return still includes the counter keys."""
    ns_id = uuid4()
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    result = await unify_entities(ns_id, storage=storage, store_results=False)

    assert result["merged_count"] == 0
    assert result["candidates_evaluated"] == {
        "correlation": 0,
        "exact_name": 0,
        "embedding": 0,
        "fuzzy": 0,
    }
    assert result["pairs_above_threshold"] == {
        "correlation": 0,
        "exact_name": 0,
        "embedding": 0,
        "fuzzy": 0,
    }
