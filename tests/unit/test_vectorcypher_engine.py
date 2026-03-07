"""Unit tests for VectorCypher engine — entity search (DYT-180).

Verifies that VectorCypherEngine.search_entities() uses
``search_similar_entities()`` + ``get_entities_batch()`` on
StorageCoordinator (the same pattern as GraphRAG and Skeleton engines).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Entity
from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine


def _make_engine_with_mocks(
    *,
    storage: MagicMock | None = None,
    embedder: MagicMock | None = None,
) -> VectorCypherEngine:
    """Build a VectorCypherEngine with pre-injected mock internals.

    Bypasses ``connect()`` by setting private attributes directly so tests
    don't need real database connections.
    """
    config = MagicMock()
    engine = VectorCypherEngine.__new__(VectorCypherEngine)

    # Minimal internal state — only what search_entities touches
    engine._config = config
    engine._vc_config = VectorCypherConfig()
    engine._storage = storage
    engine._embedder = embedder
    engine._temporal_store = None
    engine._neo4j_driver = None
    engine._retriever = None
    engine._dual_nodes = None
    engine._router = None
    engine._connected = True
    engine._default_namespace_id = None

    return engine


def _mock_storage_coordinator() -> MagicMock:
    """Create a strict mock StorageCoordinator via spec.

    Using spec= ensures any access to non-existent attributes (such as the
    old ``search_entities_by_embedding``) raises AttributeError, preventing
    regressions.
    """
    from khora.storage.coordinator import StorageCoordinator

    storage = MagicMock(spec=StorageCoordinator)
    storage.search_similar_entities = AsyncMock(return_value=[])
    storage.get_entities_batch = AsyncMock(return_value={})
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorCypherSearchEntities:
    """Tests for VectorCypherEngine.search_entities (DYT-180)."""

    @pytest.mark.asyncio
    async def test_search_entities_uses_correct_storage_methods(self) -> None:
        """search_entities calls search_similar_entities + get_entities_batch."""
        namespace_id = uuid4()
        entity_id = uuid4()
        query_embedding = [0.1] * 128

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=query_embedding)

        entity = Entity(
            id=entity_id,
            namespace_id=namespace_id,
            name="Test Entity",
            entity_type="CONCEPT",
            description="A test entity for search.",
        )

        storage = _mock_storage_coordinator()
        storage.search_similar_entities = AsyncMock(
            return_value=[(entity_id, 0.95)],
        )
        storage.get_entities_batch = AsyncMock(
            return_value={entity_id: entity},
        )

        engine = _make_engine_with_mocks(storage=storage, embedder=embedder)

        results = await engine.search_entities("test query", namespace_id, limit=5)

        # Verify correct methods were called
        embedder.embed.assert_awaited_once_with("test query")
        storage.search_similar_entities.assert_awaited_once_with(
            namespace_id,
            query_embedding,
            limit=5,
            min_similarity=0.0,
        )
        storage.get_entities_batch.assert_awaited_once_with([entity_id])

        # Verify results
        assert len(results) == 1
        assert results[0].id == entity_id
        assert results[0].name == "Test Entity"

    @pytest.mark.asyncio
    async def test_search_entities_returns_empty_on_no_matches(self) -> None:
        """search_entities returns [] when no similar entities found."""
        namespace_id = uuid4()
        query_embedding = [0.1] * 128

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=query_embedding)

        storage = _mock_storage_coordinator()
        storage.search_similar_entities = AsyncMock(return_value=[])

        engine = _make_engine_with_mocks(storage=storage, embedder=embedder)

        results = await engine.search_entities("unknown query", namespace_id)

        assert results == []
        storage.get_entities_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_entities_does_not_call_nonexistent_method(self) -> None:
        """Regression guard: search_entities must not call search_entities_by_embedding.

        The spec-based mock will raise AttributeError if the engine tries to
        access any method that doesn't exist on StorageCoordinator, catching
        any regression to the old broken code.
        """
        namespace_id = uuid4()
        query_embedding = [0.1] * 128

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=query_embedding)

        storage = _mock_storage_coordinator()
        engine = _make_engine_with_mocks(storage=storage, embedder=embedder)

        # Should succeed without AttributeError
        results = await engine.search_entities("test query", namespace_id, limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_entities_preserves_score_ordering(self) -> None:
        """Entities are returned in descending score order from search_similar_entities."""
        namespace_id = uuid4()
        id_a, id_b, id_c = uuid4(), uuid4(), uuid4()
        query_embedding = [0.1] * 128

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=query_embedding)

        entities = {
            id_a: Entity(id=id_a, namespace_id=namespace_id, name="A"),
            id_b: Entity(id=id_b, namespace_id=namespace_id, name="B"),
            id_c: Entity(id=id_c, namespace_id=namespace_id, name="C"),
        }

        storage = _mock_storage_coordinator()
        # Return in descending score order: B > A > C
        storage.search_similar_entities = AsyncMock(
            return_value=[(id_b, 0.95), (id_a, 0.80), (id_c, 0.60)],
        )
        storage.get_entities_batch = AsyncMock(return_value=entities)

        engine = _make_engine_with_mocks(storage=storage, embedder=embedder)

        results = await engine.search_entities("test query", namespace_id, limit=3)

        assert [e.name for e in results] == ["B", "A", "C"]

    @pytest.mark.asyncio
    async def test_search_entities_filters_missing_entities(self) -> None:
        """Entities not found by get_entities_batch are silently skipped."""
        namespace_id = uuid4()
        id_found, id_missing = uuid4(), uuid4()
        query_embedding = [0.1] * 128

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=query_embedding)

        entity = Entity(id=id_found, namespace_id=namespace_id, name="Found")

        storage = _mock_storage_coordinator()
        storage.search_similar_entities = AsyncMock(
            return_value=[(id_found, 0.90), (id_missing, 0.70)],
        )
        storage.get_entities_batch = AsyncMock(
            return_value={id_found: entity},  # id_missing not in results
        )

        engine = _make_engine_with_mocks(storage=storage, embedder=embedder)

        results = await engine.search_entities("test query", namespace_id)

        assert len(results) == 1
        assert results[0].name == "Found"
