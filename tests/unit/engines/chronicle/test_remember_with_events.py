"""Chronicle #2: tests for EventExtractor wiring in remember + remember_batch.

These tests exercise the new event-extraction path that runs after the
ingest pipeline finishes:

* The toggle resolution rule (per-namespace beats expertise default).
* Event persistence on every chunk.
* Per-chunk failure resilience.
* Per-document fan-out in remember_batch.

The ingest pipeline itself is patched out — these tests stay purely
behavioural at the chronicle layer. The LLM is replaced by a deterministic
stub returning a fixed list of ``ChronicleEvent`` objects per chunk.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, MemoryNamespace
from khora.engines.chronicle.engine import ChronicleEngine
from khora.engines.chronicle.events import ChronicleEvent
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, namespace_id: UUID) -> Chunk:
    """Build a Chunk that matches the shape produced by the ingest pipeline."""
    document_id = uuid4()
    return Chunk(
        namespace_id=namespace_id,
        document_id=document_id,
        content=content,
        chunk_index=0,
    )


def _make_event(text: str) -> ChronicleEvent:
    """Build a minimal ChronicleEvent for the stub extractor to return."""
    return ChronicleEvent(
        subject="alice",
        verb="met",
        object="bob",
        source_text=text,
        confidence=0.9,
    )


class _StubEmbedder:
    """Returns deterministic, length-3 fake embeddings."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class _RecordingCoordinator:
    """Minimal storage coordinator double for the event-extraction path."""

    def __init__(self, namespace: MemoryNamespace, chunks: list[Chunk]) -> None:
        self._namespace = namespace
        self._chunks_by_id = {c.id: c for c in chunks}
        self.write_events_calls: list[tuple[list[ChronicleEvent], UUID]] = []

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        if self._namespace and self._namespace.namespace_id == namespace_id:
            return self._namespace
        return None

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def write_events(
        self,
        events: list[ChronicleEvent],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        # Snapshot to avoid mutation surprises in assertions.
        self.write_events_calls.append((list(events), namespace_id))
        return [ev.id for ev in events]


def _bare_engine() -> ChronicleEngine:
    """Return a ChronicleEngine without going through ``connect()``."""
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


def _wire_engine(
    engine: ChronicleEngine,
    coord: _RecordingCoordinator,
    extractor: Any,
) -> None:
    """Inject the stubs the engine would normally pick up in connect()."""
    engine._storage = coord  # type: ignore[assignment]
    engine._embedder = _StubEmbedder()  # type: ignore[assignment]
    engine._event_extractor = extractor


# ---------------------------------------------------------------------------
# _events_enabled — resolution priority
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventsEnabledResolution:
    def test_namespace_override_disables_when_expertise_enables(self) -> None:
        ns = MemoryNamespace(config_overrides={"events": {"enabled": False}})
        expertise = ExpertiseConfig(name="x", events=EventExtractionConfig(enabled=True))
        assert ChronicleEngine._events_enabled(ns, expertise) is False

    def test_namespace_override_enables_when_expertise_disables(self) -> None:
        ns = MemoryNamespace(config_overrides={"events": {"enabled": True}})
        expertise = ExpertiseConfig(name="x", events=EventExtractionConfig(enabled=False))
        assert ChronicleEngine._events_enabled(ns, expertise) is True

    def test_falls_back_to_expertise_when_no_namespace_override(self) -> None:
        ns = MemoryNamespace()  # empty config_overrides
        expertise = ExpertiseConfig(name="x", events=EventExtractionConfig(enabled=False))
        assert ChronicleEngine._events_enabled(ns, expertise) is False

    def test_default_on_when_neither_provided(self) -> None:
        assert ChronicleEngine._events_enabled(None, None) is True

    def test_irrelevant_namespace_keys_do_not_override(self) -> None:
        ns = MemoryNamespace(config_overrides={"unrelated": {"enabled": False}})
        expertise = ExpertiseConfig(name="x", events=EventExtractionConfig(enabled=True))
        assert ChronicleEngine._events_enabled(ns, expertise) is True


# ---------------------------------------------------------------------------
# remember() event wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberWithEvents:
    @pytest.mark.asyncio
    async def test_remember_writes_events_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk_a = _make_chunk("alice met bob", ns_id)
        chunk_b = _make_chunk("carol greeted dan", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk_a, chunk_b])

        async def fake_extract(
            text: str,
            *,
            chunk_id: UUID,
            namespace_id: UUID,
            errors_out: list | None = None,
        ) -> list[ChronicleEvent]:
            del chunk_id, namespace_id, errors_out  # signature-shape only
            return [_make_event(text)]

        extractor = AsyncMock()
        extractor.extract_events.side_effect = fake_extract

        engine = _bare_engine()
        _wire_engine(engine, coord, extractor)

        # Stub the upstream pipeline + dedup so remember() reaches our wiring
        # without doing real I/O.
        async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
            return None

        async def fake_create_document(doc: Any) -> Any:
            return doc

        coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
        coord.create_document = fake_create_document  # type: ignore[attr-defined]

        async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {
                "document_id": str(uuid4()),
                "chunks": 2,
                "entities": 0,
                "relationships": 0,
                "extracted_relationships": 0,
                "inferred_relationships": 0,
                "entity_ids": [],
                "chunk_ids": [chunk_a.id, chunk_b.id],
                "phase_times": {},
            }

        monkeypatch.setattr(
            "khora.pipelines.flows.ingest.process_document",
            fake_process_document,
        )

        result = await engine.remember(
            "alice met bob. carol greeted dan.",
            ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            expertise=ExpertiseConfig(
                name="x",
                events=EventExtractionConfig(enabled=True),
            ),
        )

        # One write_events batch, two events (one per chunk), each linked to its chunk.
        assert len(coord.write_events_calls) == 1
        events, written_ns = coord.write_events_calls[0]
        assert written_ns == ns_id
        assert len(events) == 2
        assert {ev.chunk_id for ev in events} == {chunk_a.id, chunk_b.id}
        # Embeddings were populated by the stub embedder before persistence.
        assert all(ev.embedding is not None and len(ev.embedding) == 3 for ev in events)
        assert result.metadata["events_extracted"] == 2

    @pytest.mark.asyncio
    async def test_remember_skips_events_when_expertise_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice met bob", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])

        extractor = AsyncMock()
        extractor.extract_events.side_effect = AssertionError("must not run")

        engine = _bare_engine()
        _wire_engine(engine, coord, extractor)

        async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
            return None

        async def fake_create_document(doc: Any) -> Any:
            return doc

        coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
        coord.create_document = fake_create_document  # type: ignore[attr-defined]

        async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {
                "document_id": str(uuid4()),
                "chunks": 1,
                "entities": 0,
                "relationships": 0,
                "extracted_relationships": 0,
                "inferred_relationships": 0,
                "entity_ids": [],
                "chunk_ids": [chunk.id],
                "phase_times": {},
            }

        monkeypatch.setattr(
            "khora.pipelines.flows.ingest.process_document",
            fake_process_document,
        )

        result = await engine.remember(
            "alice met bob",
            ns_id,
            entity_types=["PERSON"],
            relationship_types=[],
            expertise=ExpertiseConfig(
                name="x",
                events=EventExtractionConfig(enabled=False),
            ),
        )

        assert coord.write_events_calls == []
        extractor.extract_events.assert_not_called()
        assert result.metadata["events_extracted"] == 0

    @pytest.mark.asyncio
    async def test_namespace_override_beats_expertise_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-namespace override (False) wins over enabled expertise default (True)."""
        ns_id = uuid4()
        namespace = MemoryNamespace(
            namespace_id=ns_id,
            config_overrides={"events": {"enabled": False}},
        )
        chunk = _make_chunk("alice met bob", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])

        extractor = AsyncMock()
        extractor.extract_events.side_effect = AssertionError("must not run")

        engine = _bare_engine()
        _wire_engine(engine, coord, extractor)

        async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
            return None

        async def fake_create_document(doc: Any) -> Any:
            return doc

        coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
        coord.create_document = fake_create_document  # type: ignore[attr-defined]

        async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {
                "document_id": str(uuid4()),
                "chunks": 1,
                "entities": 0,
                "relationships": 0,
                "extracted_relationships": 0,
                "inferred_relationships": 0,
                "entity_ids": [],
                "chunk_ids": [chunk.id],
                "phase_times": {},
            }

        monkeypatch.setattr(
            "khora.pipelines.flows.ingest.process_document",
            fake_process_document,
        )

        await engine.remember(
            "alice met bob",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=ExpertiseConfig(
                name="x",
                events=EventExtractionConfig(enabled=True),  # would be ON but namespace says OFF
            ),
        )

        assert coord.write_events_calls == []
        extractor.extract_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_chunk_failure_does_not_block_other_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One bad LLM call must not nuke the whole remember()."""
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        good_chunk = _make_chunk("alice met bob", ns_id)
        bad_chunk = _make_chunk("trigger boom", ns_id)
        coord = _RecordingCoordinator(namespace, [good_chunk, bad_chunk])

        async def flaky_extract(
            text: str,
            *,
            chunk_id: UUID,
            namespace_id: UUID,
            errors_out: list | None = None,
        ) -> list[ChronicleEvent]:
            del namespace_id, errors_out
            if chunk_id == bad_chunk.id:
                raise RuntimeError("LLM is on fire")
            return [_make_event(text)]

        extractor = AsyncMock()
        extractor.extract_events.side_effect = flaky_extract

        engine = _bare_engine()
        _wire_engine(engine, coord, extractor)

        async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
            return None

        async def fake_create_document(doc: Any) -> Any:
            return doc

        coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
        coord.create_document = fake_create_document  # type: ignore[attr-defined]

        async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {
                "document_id": str(uuid4()),
                "chunks": 2,
                "entities": 0,
                "relationships": 0,
                "extracted_relationships": 0,
                "inferred_relationships": 0,
                "entity_ids": [],
                "chunk_ids": [good_chunk.id, bad_chunk.id],
                "phase_times": {},
            }

        monkeypatch.setattr(
            "khora.pipelines.flows.ingest.process_document",
            fake_process_document,
        )

        result = await engine.remember(
            "alice met bob. trigger boom.",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=ExpertiseConfig(
                name="x",
                events=EventExtractionConfig(enabled=True),
            ),
        )

        assert len(coord.write_events_calls) == 1
        events, _ = coord.write_events_calls[0]
        assert len(events) == 1
        assert events[0].chunk_id == good_chunk.id
        assert result.metadata["events_extracted"] == 1


# ---------------------------------------------------------------------------
# remember_batch() event wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchWithEvents:
    @pytest.mark.asyncio
    async def test_remember_batch_links_events_to_correct_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunks = [_make_chunk(f"doc {i} content", ns_id) for i in range(5)]
        coord = _RecordingCoordinator(namespace, chunks)

        async def fake_extract(
            text: str,
            *,
            chunk_id: UUID,
            namespace_id: UUID,
            errors_out: list | None = None,
        ) -> list[ChronicleEvent]:
            del errors_out
            ev = _make_event(text)
            ev.chunk_id = chunk_id
            ev.namespace_id = namespace_id
            return [ev]

        extractor = AsyncMock()
        extractor.extract_events.side_effect = fake_extract

        engine = _bare_engine()
        _wire_engine(engine, coord, extractor)

        # Bypass entity preload (deduplicate path) by stubbing list_entities.
        async def fake_list_entities(*_a: Any, **_kw: Any) -> list[Any]:
            return []

        coord.list_entities = fake_list_entities  # type: ignore[attr-defined]

        async def fake_ingest_documents(
            namespace_id: UUID,
            doc_inputs: list[dict[str, Any]],
            storage: Any,
            **_kw: Any,
        ) -> dict[str, Any]:
            # Pretend the pipeline produced one chunk per doc, in order.
            per_doc = [
                {
                    "chunk_ids": [chunks[i].id],
                    "phase_times": {},
                }
                for i in range(len(doc_inputs))
            ]
            return {
                "total_documents": len(doc_inputs),
                "processed_documents": len(doc_inputs),
                "skipped_documents": 0,
                "failed_documents": 0,
                "total_chunks": len(doc_inputs),
                "total_entities": 0,
                "total_relationships": 0,
                "total_inferred_relationships": 0,
                "episodes_created": 0,
                "per_document_results": per_doc,
                "timing": {"staging_s": 0.0, "processing_s": 0.0, "phase_totals": {}},
            }

        monkeypatch.setattr(
            "khora.pipelines.flows.ingest.ingest_documents",
            fake_ingest_documents,
        )

        result = await engine.remember_batch(
            [{"content": f"doc {i}"} for i in range(5)],
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=ExpertiseConfig(
                name="x",
                events=EventExtractionConfig(enabled=True),
            ),
            deduplicate=False,  # skip preload to avoid extra coord calls
        )

        assert len(coord.write_events_calls) == 1
        events, written_ns = coord.write_events_calls[0]
        assert written_ns == ns_id
        # 5 events, one per chunk, each correctly linked.
        assert len(events) == 5
        assert {ev.chunk_id for ev in events} == {c.id for c in chunks}
        assert result.metadata["events_extracted"] == 5
