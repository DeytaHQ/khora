"""Regression tests for ``_record_storage_op`` namespace_id propagation.

Background: a Feb-2026 refactor (``c948760``) replaced per-method telemetry
recording in ``storage/coordinator.py`` with a generic decorator that dropped
``namespace_id`` from every ``record_storage_op`` call. The Phase-0 telemetry
audit observed ``storage_events.namespace_id`` was 100% NULL in the recent
14-day window. These tests pin the extraction logic so the column stays
populated regardless of method shape (positional UUID, model arg, list of
models, kwarg).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, Relationship
from khora.storage.coordinator import StorageCoordinator, _extract_namespace_id

# ---------------------------------------------------------------------------
# _extract_namespace_id pure unit
# ---------------------------------------------------------------------------


class TestExtractNamespaceId:
    """Pure tests for the extraction helper — no decorator/coordinator state."""

    def test_kwarg(self) -> None:
        ns = uuid4()
        assert _extract_namespace_id((object(),), {"namespace_id": ns}) == ns

    def test_positional_uuid(self) -> None:
        # args[0] is `self`, args[1] is the namespace UUID
        ns = uuid4()
        assert _extract_namespace_id((object(), ns, [1, 2]), {}) == ns

    def test_model_attribute(self) -> None:
        ns = uuid4()
        doc = Document(namespace_id=ns, content="x")
        assert _extract_namespace_id((object(), doc), {}) == ns

    def test_list_of_models(self) -> None:
        ns = uuid4()
        chunks = [Chunk(namespace_id=ns, document_id=uuid4(), content="c")]
        assert _extract_namespace_id((object(), chunks), {}) == ns

    def test_empty_list_returns_none(self) -> None:
        assert _extract_namespace_id((object(), []), {}) is None

    def test_no_namespace_returns_none(self) -> None:
        assert _extract_namespace_id((object(), "string", 42), {}) is None

    def test_kwarg_takes_priority_over_positional(self) -> None:
        kwarg_ns = uuid4()
        positional_ns = uuid4()
        result = _extract_namespace_id((object(), positional_ns), {"namespace_id": kwarg_ns})
        assert result == kwarg_ns

    def test_non_uuid_kwarg_falls_through(self) -> None:
        # kwargs.namespace_id is not a UUID — fall through to positional
        ns = uuid4()
        result = _extract_namespace_id((object(), ns), {"namespace_id": "not-a-uuid"})
        assert result == ns


# ---------------------------------------------------------------------------
# Decorator integration: namespace_id reaches record_storage_op
# ---------------------------------------------------------------------------


class _RecordingCollector:
    """Drop-in replacement for the global telemetry collector."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_storage_op(self, **kwargs) -> None:
        self.calls.append(kwargs)

    # Methods unused in these tests but required by the collector surface.
    def record_llm_call(self, **kwargs) -> None:  # pragma: no cover
        pass

    def record_pipeline_stage(self, **kwargs) -> None:  # pragma: no cover
        pass


@pytest.fixture
def recording_collector():
    collector = _RecordingCollector()
    with patch("khora.telemetry._collector", collector):
        yield collector


class TestStorageOpRecordsNamespaceId:
    """Each decorated coordinator method must populate storage_events.namespace_id."""

    @pytest.mark.asyncio
    async def test_create_document_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        rel = MagicMock()
        rel.create_document = AsyncMock(side_effect=lambda d: d)
        coord = StorageCoordinator(relational=rel)

        doc = Document(namespace_id=ns, content="hello")
        await coord.create_document(doc)

        assert len(recording_collector.calls) == 1
        call = recording_collector.calls[0]
        assert call["operation"] == "create_document"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_create_chunks_batch_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        vec = MagicMock()
        vec.create_chunks_batch = AsyncMock(side_effect=lambda chunks: chunks)
        coord = StorageCoordinator(vector=vec)

        chunks = [Chunk(namespace_id=ns, document_id=uuid4(), content="c")]
        await coord.create_chunks_batch(chunks)

        call = recording_collector.calls[0]
        assert call["operation"] == "create_chunks_batch"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_search_similar_chunks_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        vec = MagicMock()
        vec.search_similar = AsyncMock(return_value=[])
        coord = StorageCoordinator(vector=vec)

        await coord.search_similar_chunks(ns, [0.1, 0.2, 0.3], limit=5)

        call = recording_collector.calls[0]
        assert call["operation"] == "search_similar_chunks"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_create_entity_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        graph = MagicMock()
        graph.create_entity = AsyncMock(side_effect=lambda e: e)
        coord = StorageCoordinator(graph=graph)

        entity = Entity(namespace_id=ns, name="Alice", entity_type="PERSON")
        await coord.create_entity(entity)

        call = recording_collector.calls[0]
        assert call["operation"] == "create_entity"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_upsert_entities_batch_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        graph = MagicMock()
        graph.upsert_entities_batch = AsyncMock(return_value=[])
        coord = StorageCoordinator(graph=graph)

        entities = [Entity(namespace_id=ns, name="Bob", entity_type="PERSON")]
        await coord.upsert_entities_batch(ns, entities)

        call = recording_collector.calls[0]
        assert call["operation"] == "upsert_entities_batch"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_create_relationship_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        graph = MagicMock()
        graph.create_relationship = AsyncMock(side_effect=lambda r: r)
        coord = StorageCoordinator(graph=graph)

        rel = Relationship(
            namespace_id=ns,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
        )
        await coord.create_relationship(rel)

        call = recording_collector.calls[0]
        assert call["operation"] == "create_relationship"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_create_relationships_batch_records_namespace_id(self, recording_collector) -> None:
        ns = uuid4()
        graph = MagicMock()
        graph.create_relationships_batch = AsyncMock(return_value=1)
        coord = StorageCoordinator(graph=graph)

        rels = [
            Relationship(
                namespace_id=ns,
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="KNOWS",
            )
        ]
        await coord.create_relationships_batch(rels)

        call = recording_collector.calls[0]
        assert call["operation"] == "create_relationships_batch"
        assert call["namespace_id"] == ns

    @pytest.mark.asyncio
    async def test_error_path_still_records_namespace_id(self, recording_collector) -> None:
        """Even when the inner call raises, namespace_id is recorded for the error event."""
        ns = uuid4()
        rel = MagicMock()
        rel.create_document = AsyncMock(side_effect=RuntimeError("boom"))
        coord = StorageCoordinator(relational=rel)

        doc = Document(namespace_id=ns, content="hello")
        with pytest.raises(RuntimeError, match="boom"):
            await coord.create_document(doc)

        call = recording_collector.calls[0]
        assert call["status"] == "error"
        assert call["namespace_id"] == ns
