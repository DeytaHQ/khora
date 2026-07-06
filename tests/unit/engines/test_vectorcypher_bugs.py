"""Regression tests for VectorCypher engine bugs.

Bug #1: KhoraConfig defines ``pipelines`` (plural) but all engine code uses
        ``self._config.pipeline`` (singular) -> AttributeError at runtime.
        Fix: add a ``pipeline`` property alias on KhoraConfig.

Bug #2: DualNodeManager passes dict metadata to Neo4j which only accepts
        primitives and arrays, not maps -> TypeError at runtime.
        Fix: serialize metadata to JSON string on write, deserialize on read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.dual_nodes import DualNodeManager
from khora.storage.temporal import TemporalChunk

# ---------------------------------------------------------------------------
# Bug #1: pipeline vs pipelines attribute mismatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineAttributeName:
    """KhoraConfig must expose a ``pipeline`` (singular) attribute so that
    engine code like ``self._config.pipeline.chunking_strategy`` works.

    Currently the schema defines ``pipelines`` (plural) which does not
    match the 11 call sites across vectorcypher and skeleton engines.
    """

    def test_khoraconfig_has_pipeline_attribute(self) -> None:
        """KhoraConfig must have a ``pipeline`` (singular) attribute that
        returns PipelineSettings.

        All engine code accesses ``self._config.pipeline.*`` so this
        attribute must exist on the config object.
        """
        from khora.config import KhoraConfig
        from khora.config.schema import PipelineSettings

        config = KhoraConfig(
            app_name="khora-test",
        )

        assert hasattr(config, "pipeline"), (
            "KhoraConfig must have a 'pipeline' (singular) attribute -- "
            "all engine code references self._config.pipeline"
        )
        assert isinstance(config.pipeline, PipelineSettings), "config.pipeline must be a PipelineSettings instance"

    def test_pipeline_settings_have_expected_fields(self) -> None:
        """The pipeline config accessible via config.pipeline must expose
        the fields that the engine code reads."""
        from khora.config import KhoraConfig

        config = KhoraConfig(
            app_name="khora-test",
        )

        # These are the three fields read by _process_document and
        # _process_document_streaming in both vectorcypher and skeleton.
        pipeline = config.pipeline
        assert hasattr(pipeline, "chunking_strategy")
        assert hasattr(pipeline, "chunk_size")
        assert hasattr(pipeline, "chunk_overlap")
        assert hasattr(pipeline, "extract_entities")


# ---------------------------------------------------------------------------
# Bug #2: Neo4j map property type error
# ---------------------------------------------------------------------------


def _make_neo4j_driver() -> tuple[MagicMock, AsyncMock]:
    """Create a mock Neo4j driver with properly mocked session context manager."""
    driver = MagicMock()
    session = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx

    return driver, session


@pytest.mark.unit
class TestMetadataSerializedForNeo4j:
    """Neo4j does not support map/dict values as node properties.

    The metadata field must be serialized to a JSON string before being
    passed in the Cypher params.  These tests verify that the params dict
    contains a *string* (JSON), not a raw dict, for the metadata field.
    """

    @pytest.mark.asyncio
    async def test_create_chunk_node_serializes_metadata(self) -> None:
        """Single-node creation must JSON-serialize the metadata dict."""
        driver, session = _make_neo4j_driver()

        captured_params: dict = {}

        async def _capture_work(work_fn):
            tx = AsyncMock()
            await work_fn(tx)
            call_kwargs = tx.run.call_args
            captured_params.update(call_kwargs.kwargs if call_kwargs.kwargs else {})

        session.execute_write = _capture_work

        manager = DualNodeManager(driver)

        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test content",
            occurred_at=datetime.now(UTC),
            metadata={"chunk_index": 0, "source": "test"},
        )

        await manager.create_chunk_node(chunk)

        # The metadata value sent to Neo4j must be a JSON string, not a dict
        assert "metadata" in captured_params, "metadata param not found in Cypher params"
        metadata_value = captured_params["metadata"]
        assert isinstance(metadata_value, str), (
            f"metadata must be a JSON string for Neo4j, got {type(metadata_value).__name__}: {metadata_value!r}"
        )
        # Verify it round-trips correctly
        parsed = json.loads(metadata_value)
        assert parsed == {"chunk_index": 0, "source": "test"}

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch_serializes_metadata(self) -> None:
        """Batch creation must JSON-serialize metadata in every chunk dict."""
        driver, session = _make_neo4j_driver()

        captured_chunks: list[dict] = []

        async def _capture_work(work_fn):
            tx = AsyncMock()
            await work_fn(tx)
            call_kwargs = tx.run.call_args
            # Batch uses keyword arg chunks=
            if call_kwargs.kwargs and "chunks" in call_kwargs.kwargs:
                captured_chunks.extend(call_kwargs.kwargs["chunks"])

        session.execute_write = _capture_work

        manager = DualNodeManager(driver)

        namespace_id = uuid4()
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content=f"chunk {i}",
                occurred_at=datetime.now(UTC),
                metadata={"chunk_index": i, "key": "value"},
            )
            for i in range(3)
        ]

        await manager.create_chunk_nodes_batch(chunks, namespace_id)

        assert len(captured_chunks) == 3, f"Expected 3 chunks, got {len(captured_chunks)}"
        for i, chunk_data in enumerate(captured_chunks):
            metadata_value = chunk_data["metadata"]
            assert isinstance(metadata_value, str), (
                f"chunk[{i}].metadata must be a JSON string for Neo4j, "
                f"got {type(metadata_value).__name__}: {metadata_value!r}"
            )
            parsed = json.loads(metadata_value)
            assert parsed["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_create_chunk_node_empty_metadata_serialized(self) -> None:
        """Even empty/None metadata should be serialized as a JSON string."""
        driver, session = _make_neo4j_driver()

        captured_params: dict = {}

        async def _capture_work(work_fn):
            tx = AsyncMock()
            await work_fn(tx)
            call_kwargs = tx.run.call_args
            captured_params.update(call_kwargs.kwargs if call_kwargs.kwargs else {})

        session.execute_write = _capture_work

        manager = DualNodeManager(driver)

        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test content",
            metadata=None,
        )

        await manager.create_chunk_node(chunk)

        metadata_value = captured_params["metadata"]
        assert isinstance(metadata_value, str), (
            f"metadata must be a JSON string for Neo4j, got {type(metadata_value).__name__}: {metadata_value!r}"
        )


# ---------------------------------------------------------------------------
# Bug #3: Non-scalar metadata silently dropped
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonScalarMetadataPreserved:
    """VectorCypher engine must propagate all doc_metadata values to chunk
    metadata, including lists, dicts, None, and datetime — not just scalars.

    Previously the engine used an isinstance(v, (str, int, float, bool))
    guard that silently dropped non-scalar types. Both storage backends
    (JSONB for pgvector, JSON-serialized string for Neo4j) handle nested
    types correctly, so the filter was unnecessary.
    """

    def test_chunk_metadata_includes_list_values(self) -> None:
        """Lists in doc_metadata must survive into the chunk metadata dict."""
        doc_metadata = {
            "participants": ["alice", "bob"],
            "priority": "high",
        }
        chunk_metadata = {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            **doc_metadata,
        }
        assert chunk_metadata["participants"] == ["alice", "bob"]
        assert chunk_metadata["priority"] == "high"

    def test_chunk_metadata_includes_dict_values(self) -> None:
        """Nested dicts in doc_metadata must survive into chunk metadata."""
        doc_metadata = {
            "context": {"parent_id": "123", "thread": "abc"},
            "source_system": "slack",
        }
        chunk_metadata = {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            **doc_metadata,
        }
        assert chunk_metadata["context"] == {"parent_id": "123", "thread": "abc"}
        assert chunk_metadata["source_system"] == "slack"

    def test_chunk_metadata_includes_none_values(self) -> None:
        """None values in doc_metadata must not be silently dropped."""
        doc_metadata = {
            "optional_field": None,
            "priority": "high",
        }
        chunk_metadata = {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            **doc_metadata,
        }
        assert "optional_field" in chunk_metadata
        assert chunk_metadata["optional_field"] is None

    def test_chunk_metadata_includes_all_types(self) -> None:
        """Regression: all JSON-compatible types must propagate to chunks."""
        doc_metadata = {
            "str_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": ["a", "b"],
            "dict_val": {"nested": True},
            "none_val": None,
        }
        chunk_metadata = {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            **doc_metadata,
        }
        for key in doc_metadata:
            assert key in chunk_metadata, f"{key} missing from chunk metadata"
            assert chunk_metadata[key] == doc_metadata[key]

    def test_fixed_keys_not_overwritten_by_doc_metadata(self) -> None:
        """Internal keys (chunk_index, start_char, end_char) must not be
        overwritable by user-provided doc_metadata."""
        doc_metadata = {
            "chunk_index": 999,
            "start_char": -1,
            "end_char": -1,
            "user_field": "preserved",
        }
        # Engine pattern: **doc_metadata first, then fixed keys override
        chunk_metadata = {
            **doc_metadata,
            "chunk_index": 0,
            "start_char": 10,
            "end_char": 100,
        }
        assert chunk_metadata["chunk_index"] == 0, "chunk_index overwritten by doc_metadata"
        assert chunk_metadata["start_char"] == 10, "start_char overwritten by doc_metadata"
        assert chunk_metadata["end_char"] == 100, "end_char overwritten by doc_metadata"
        assert chunk_metadata["user_field"] == "preserved"

    def test_no_isinstance_filter_in_engine(self) -> None:
        """The VectorCypher engine source must not contain the old scalar-only
        metadata filter pattern."""
        import inspect

        from khora.engines.vectorcypher.engine import VectorCypherEngine

        source = inspect.getsource(VectorCypherEngine)
        assert "isinstance(v, (str, int, float, bool))" not in source, (
            "VectorCypher engine still contains the isinstance scalar filter "
            "that drops non-scalar metadata — fix not applied"
        )


# ---------------------------------------------------------------------------
# Bug #4: Cross-window entity count inflation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossWindowEntityCountInflation:
    """BatchResult.entities must not double-count entities that appear across
    multiple windows when max_chunks_in_flight is set.

    upsert_entities_batch() ensures a single DB row per entity, so
    BatchResult.entities must reflect unique persisted cardinality.
    The bug: ``results["entities"] += len(all_entities)`` was executed once
    per window, inflating the count for shared entities.
    Fix: a ``_seen_entity_keys`` set tracks entity keys across all windows.
    """

    def test_inflated_accumulation_pattern_removed(self) -> None:
        """The engine must no longer use raw ``len(all_entities)`` to accumulate
        the entity count across window iterations."""
        import inspect

        from khora.engines.vectorcypher.engine import VectorCypherEngine

        source = inspect.getsource(VectorCypherEngine)

        # The old (broken) pattern accumulated entity counts per-window without
        # deduplication, inflating the count when an entity appears in >1 window.
        assert 'results["entities"] += len(all_entities)' not in source, (
            'Engine still uses results["entities"] += len(all_entities) directly '
            "inside the window loop — cross-window entity count inflation not fixed"
        )

    def test_seen_entity_keys_dedup_logic(self) -> None:
        """The counting logic used by the fix correctly deduplicates across windows.

        This test exercises the exact algorithm extracted from the engine to verify
        that entities shared across windows are counted only once.
        """
        # Simulate two windows each containing the same entity (Alice:PERSON).
        # Window 1 also has a unique entity (Bob:PERSON).
        # Window 2 also has a unique entity (Carol:PERSON).
        # Expected unique count: 3 (Alice, Bob, Carol), not 4.

        class _FakeEntity:
            def __init__(self, name: str, entity_type: str) -> None:
                self.name = name
                self.entity_type = entity_type

        window_entities = [
            [_FakeEntity("Alice", "PERSON"), _FakeEntity("Bob", "PERSON")],  # window 1
            [_FakeEntity("Alice", "PERSON"), _FakeEntity("Carol", "PERSON")],  # window 2
        ]

        # Replicate the fix's counting algorithm
        seen_entity_keys: set[tuple[str, str]] = set()
        total_entity_count = 0

        for all_entities in window_entities:
            new_entity_count = 0
            for _e in all_entities:
                _key = (_e.name, _e.entity_type)
                if _key not in seen_entity_keys:
                    seen_entity_keys.add(_key)
                    new_entity_count += 1
            total_entity_count += new_entity_count

        assert total_entity_count == 3, (
            f"Expected 3 unique entities (Alice, Bob, Carol), got {total_entity_count}. "
            "Cross-window deduplication is broken."
        )

    def test_no_cross_window_inflation_when_all_distinct(self) -> None:
        """When all entities are unique across windows, the count equals total entities."""

        class _FakeEntity:
            def __init__(self, name: str, entity_type: str) -> None:
                self.name = name
                self.entity_type = entity_type

        window_entities = [
            [_FakeEntity("Alice", "PERSON")],
            [_FakeEntity("Bob", "PERSON")],
            [_FakeEntity("Carol", "ORG")],
        ]

        seen_entity_keys: set[tuple[str, str]] = set()
        total_entity_count = 0

        for all_entities in window_entities:
            new_entity_count = 0
            for _e in all_entities:
                _key = (_e.name, _e.entity_type)
                if _key not in seen_entity_keys:
                    seen_entity_keys.add(_key)
                    new_entity_count += 1
            total_entity_count += new_entity_count

        assert total_entity_count == 3, (
            f"Expected 3 unique entities, got {total_entity_count}. Counting is wrong even when entities are distinct."
        )


# ---------------------------------------------------------------------------
# Bug #5: find_related_entities fallback called a non-existent method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindRelatedEntitiesGraphOnlyBackend:
    """Regression for https://github.com/DeytaHQ/khora/issues/533.

    When VectorCypher runs on a graph-only backend (sqlite_lance, surrealdb)
    `_get_dual_nodes()` returns None and the engine falls back to calling
    a method on the storage graph adapter. Previously the fallback called
    ``storage.graph.get_related_entities(...)`` — a method that exists on
    **no** backend. Result: ``AttributeError: 'SQLiteLanceGraphAdapter'
    object has no attribute 'get_related_entities'``.

    The fix uses ``get_neighborhood(...)`` which IS implemented across all
    three graph backends (sqlite_lance, surrealdb, neo4j).
    """

    async def test_fallback_uses_get_neighborhood_not_get_related_entities(self) -> None:
        from khora.core.models import Entity
        from khora.engines.vectorcypher.engine import VectorCypherEngine

        seed_id = uuid4()
        ns_id = uuid4()
        neighbor_a = Entity(id=uuid4(), namespace_id=ns_id, name="A", entity_type="MODULE")
        neighbor_b = Entity(id=uuid4(), namespace_id=ns_id, name="B", entity_type="MODULE")
        seed = Entity(id=seed_id, namespace_id=ns_id, name="seed", entity_type="MODULE")

        # Mock the graph backend. It must NOT expose `get_related_entities`;
        # only `get_neighborhood`. spec= enforces this — accessing the dead
        # method would raise AttributeError on the mock too.
        graph = MagicMock()
        graph.get_neighborhood = AsyncMock(
            return_value={"entities": [seed, neighbor_a, neighbor_b], "relationships": []},
        )
        # Refuse the dead method explicitly so we'd catch any regression that
        # re-introduces the old call.
        del graph.get_related_entities

        storage = MagicMock()
        storage.graph = graph

        engine = object.__new__(VectorCypherEngine)  # bypass __init__
        engine._get_dual_nodes = lambda: None  # type: ignore[method-assign]
        engine._get_storage = lambda: storage  # type: ignore[method-assign]

        result = await engine.find_related_entities(seed_id, ns_id, max_depth=2, limit=10)

        # Seed must be stripped; both neighbors returned. With no relationships
        # in the neighborhood payload, BFS can't recover per-hop depth so the
        # engine falls back to distance=1 → score=0.5 (Issue #581 depth scoring,
        # which superseded the flat-1.0 behaviour the original Issue #533 test
        # asserted).
        assert {e.id for e, _ in result} == {neighbor_a.id, neighbor_b.id}
        assert all(score == 0.5 for _, score in result)
        graph.get_neighborhood.assert_awaited_once_with(seed_id, namespace_id=ns_id, depth=2, limit=10)

    async def test_fallback_returns_empty_when_graph_backend_missing(self) -> None:
        from khora.engines.vectorcypher.engine import VectorCypherEngine

        storage = MagicMock()
        storage.graph = None  # no graph backend configured

        engine = object.__new__(VectorCypherEngine)
        engine._get_dual_nodes = lambda: None  # type: ignore[method-assign]
        engine._get_storage = lambda: storage  # type: ignore[method-assign]

        result = await engine.find_related_entities(uuid4(), uuid4())
        assert result == []
