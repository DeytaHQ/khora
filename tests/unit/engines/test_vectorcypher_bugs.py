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

from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.vectorcypher.dual_nodes import DualNodeManager

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
            environment="test",
            debug=True,
            auth_enabled=False,
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
            environment="test",
            debug=True,
            auth_enabled=False,
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
# Bug #3: Non-scalar metadata silently dropped (DYT-1114)
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
            "that drops non-scalar metadata — DYT-1114 fix not applied"
        )
