"""Tests for the CHUNK_ENTITIES_RESOLVED hook event (Issue #579 Phase 2 Item B).

The chunk-level event fires after every entity event for that chunk has
been dispatched, and carries the per-chunk entity set so subscribers can
express co-occurrence filters that single-entity events cannot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models.event import EventType, MemoryEvent


def _make_storage_mock(upsert_results) -> MagicMock:
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    storage.upsert_entities_batch = AsyncMock(return_value=upsert_results)
    storage.update_entity_embeddings_batch = AsyncMock()
    storage.create_relationships_batch = AsyncMock(return_value=0)
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.dispatch_hook = AsyncMock()
    storage.get_entity_by_name = AsyncMock(return_value=None)
    return storage


def _make_chunk(ns_id: UUID, doc_id: UUID, *, content: str = "x", occurred_at=None):
    from khora.core.models import Chunk

    chunk = Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.now(UTC),
    )
    if occurred_at is not None:
        # The core Chunk dataclass has slots, so we wrap it in a thin
        # proxy that adds ``occurred_at`` and remains read/write for the
        # rest of the chunk fields ingest manipulates (created_at,
        # content, embedding, etc.). Downstream code reads occurred_at
        # via ``getattr(chunk, "occurred_at", None)``.
        return _ChunkProxy(chunk, occurred_at)
    return chunk


class _ChunkProxy:
    """Read/write proxy that adds an ``occurred_at`` attribute to a slotted Chunk."""

    def __init__(self, inner, occurred_at) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "occurred_at", occurred_at)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value) -> None:
        if name in ("_inner", "occurred_at"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


def _make_entity(ns_id: UUID, name: str, entity_type: str, chunk_ids):
    from khora.core.models import Entity

    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        description=f"{name} desc",
        source_chunk_ids=list(chunk_ids),
        confidence=0.9,
    )


def _make_document_mock(doc_id, ns_id):
    doc = MagicMock()
    doc.id = doc_id
    doc.namespace_id = ns_id
    doc.content = "irrelevant"
    doc.metadata = MagicMock(custom={}, title="")
    doc.created_at = datetime.now(UTC)
    doc.mark_processing = MagicMock()
    doc.mark_completed = MagicMock()
    doc.mark_failed = MagicMock()
    doc.status = "pending"
    return doc


def _collect_chunk_events(storage_mock) -> list[MemoryEvent]:
    """Pull the CHUNK_ENTITIES_RESOLVED events out of dispatch_hook calls."""
    events = []
    for call in storage_mock.dispatch_hook.await_args_list:
        evt = call.args[0]
        if evt.event_type == EventType.CHUNK_ENTITIES_RESOLVED:
            events.append(evt)
    return events


def _fake_embedder() -> MagicMock:
    """Build a fake LiteLLMEmbedder that returns deterministic embeddings."""
    fake = MagicMock()
    fake.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    return fake


@pytest.mark.unit
class TestChunkEntitiesResolved:
    @pytest.mark.asyncio
    async def test_single_chunk_three_entities_fires_one_event(self) -> None:
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        document = _make_document_mock(doc_id, ns_id)
        chunk = _make_chunk(ns_id, doc_id)

        entities = [
            _make_entity(ns_id, "Acme", "ORGANIZATION", [chunk.id]),
            _make_entity(ns_id, "Alice", "PERSON", [chunk.id]),
            _make_entity(ns_id, "Globex", "ORGANIZATION", [chunk.id]),
        ]
        upsert_results = [(e, True) for e in entities]
        storage = _make_storage_mock(upsert_results)

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document,
                storage,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        chunk_events = _collect_chunk_events(storage)
        assert len(chunk_events) == 1
        evt = chunk_events[0]
        assert evt.resource_type == "chunk"
        assert evt.resource_id == chunk.id
        assert evt.data["entity_count"] == 3
        assert evt.data["chunk_id"] == str(chunk.id)
        assert evt.data["document_id"] == str(doc_id)
        assert set(evt.data["entity_ids"]) == {str(e.id) for e in entities}

    @pytest.mark.asyncio
    async def test_entity_names_grouped_by_type(self) -> None:
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        document = _make_document_mock(doc_id, ns_id)
        chunk = _make_chunk(ns_id, doc_id)

        entities = [
            _make_entity(ns_id, "Acme", "ORGANIZATION", [chunk.id]),
            _make_entity(ns_id, "Globex", "ORGANIZATION", [chunk.id]),
            _make_entity(ns_id, "Alice", "PERSON", [chunk.id]),
        ]
        upsert_results = [(e, True) for e in entities]
        storage = _make_storage_mock(upsert_results)

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document,
                storage,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=[],
            )

        chunk_events = _collect_chunk_events(storage)
        assert len(chunk_events) == 1
        by_type = chunk_events[0].data["entity_names_by_type"]
        assert set(by_type.keys()) == {"ORGANIZATION", "PERSON"}
        assert set(by_type["ORGANIZATION"]) == {"Acme", "Globex"}
        assert by_type["PERSON"] == ["Alice"]

    @pytest.mark.asyncio
    async def test_occurred_at_passthrough_when_set_and_none_otherwise(self) -> None:
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()

        # Case A: chunk has occurred_at
        document_a = _make_document_mock(doc_id, ns_id)
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        chunk_a = _make_chunk(ns_id, doc_id, occurred_at=ts)
        entities_a = [_make_entity(ns_id, "Acme", "ORGANIZATION", [chunk_a.id])]
        storage_a = _make_storage_mock([(entities_a[0], True)])

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk_a])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk_a])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities_a, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document_a,
                storage_a,
                entity_types=["ORGANIZATION"],
                relationship_types=[],
            )

        events_a = _collect_chunk_events(storage_a)
        assert len(events_a) == 1
        assert events_a[0].data["occurred_at"] == ts.isoformat()

        # Case B: chunk has no occurred_at
        document_b = _make_document_mock(uuid4(), ns_id)
        chunk_b = _make_chunk(ns_id, document_b.id)
        entities_b = [_make_entity(ns_id, "Globex", "ORGANIZATION", [chunk_b.id])]
        storage_b = _make_storage_mock([(entities_b[0], True)])

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk_b])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk_b])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities_b, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document_b,
                storage_b,
                entity_types=["ORGANIZATION"],
                relationship_types=[],
            )

        events_b = _collect_chunk_events(storage_b)
        assert len(events_b) == 1
        assert events_b[0].data["occurred_at"] is None

    @pytest.mark.asyncio
    async def test_sixty_entities_truncates_at_fifty(self) -> None:
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        document = _make_document_mock(doc_id, ns_id)
        chunk = _make_chunk(ns_id, doc_id)

        entities = [_make_entity(ns_id, f"Entity{i:02d}", "CONCEPT", [chunk.id]) for i in range(60)]
        upsert_results = [(e, True) for e in entities]
        storage = _make_storage_mock(upsert_results)

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document,
                storage,
                entity_types=["CONCEPT"],
                relationship_types=[],
            )

        chunk_events = _collect_chunk_events(storage)
        assert len(chunk_events) == 1
        evt = chunk_events[0]
        assert evt.data["entity_count"] == 60
        assert len(evt.data["entity_ids"]) == 50
        assert evt.data.get("truncated") is True
        # Names list should also be capped (50 names spread across types).
        total_names = sum(len(v) for v in evt.data["entity_names_by_type"].values())
        assert total_names == 50

    @pytest.mark.asyncio
    async def test_two_chunks_two_separate_events_each_with_own_set(self) -> None:
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        document = _make_document_mock(doc_id, ns_id)

        chunk_a = _make_chunk(ns_id, doc_id, content="chunk a")
        chunk_b = _make_chunk(ns_id, doc_id, content="chunk b")

        # Entity "Acme" mentioned in BOTH chunks: should appear in both events.
        # Entity "Alice" only in chunk_a, "Bob" only in chunk_b.
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", [chunk_a.id, chunk_b.id])
        alice = _make_entity(ns_id, "Alice", "PERSON", [chunk_a.id])
        bob = _make_entity(ns_id, "Bob", "PERSON", [chunk_b.id])
        entities = [acme, alice, bob]
        upsert_results = [(e, True) for e in entities]
        storage = _make_storage_mock(upsert_results)

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk_a, chunk_b])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk_a, chunk_b])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document,
                storage,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=[],
            )

        chunk_events = _collect_chunk_events(storage)
        assert len(chunk_events) == 2
        by_chunk = {evt.resource_id: evt for evt in chunk_events}
        assert chunk_a.id in by_chunk
        assert chunk_b.id in by_chunk

        evt_a = by_chunk[chunk_a.id]
        assert evt_a.data["entity_count"] == 2
        assert set(evt_a.data["entity_ids"]) == {str(acme.id), str(alice.id)}

        evt_b = by_chunk[chunk_b.id]
        assert evt_b.data["entity_count"] == 2
        assert set(evt_b.data["entity_ids"]) == {str(acme.id), str(bob.id)}

    @pytest.mark.asyncio
    async def test_chunk_event_fires_after_all_entity_events_for_that_chunk(self) -> None:
        """Ordering invariant: every entity.created event for chunk X must be
        dispatched BEFORE the chunk.entities_resolved event for chunk X.
        """
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        document = _make_document_mock(doc_id, ns_id)
        chunk = _make_chunk(ns_id, doc_id)

        entities = [
            _make_entity(ns_id, "Acme", "ORGANIZATION", [chunk.id]),
            _make_entity(ns_id, "Alice", "PERSON", [chunk.id]),
        ]
        upsert_results = [(e, True) for e in entities]
        storage = _make_storage_mock(upsert_results)

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=(entities, []))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                document,
                storage,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=[],
            )

        # Collect dispatched events in order
        types_in_order = [call.args[0].event_type for call in storage.dispatch_hook.await_args_list]
        # The chunk event must come AFTER the last entity event for that chunk.
        chunk_evt_idx = types_in_order.index(EventType.CHUNK_ENTITIES_RESOLVED)
        last_entity_idx = max(i for i, t in enumerate(types_in_order) if t == EventType.ENTITY_CREATED)
        assert chunk_evt_idx > last_entity_idx
