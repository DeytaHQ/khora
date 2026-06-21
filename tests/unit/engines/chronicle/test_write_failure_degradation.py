"""Issue #1228: a chronicle write_events / write_facts failure must surface a
structured ADR-001 Degradation on ``RememberResult.metadata['degradations']``,
not just log a WARNING and silently report ``events: 0`` / ``facts: 0``.

Before the fix, ``_extract_and_persist_events`` / ``_extract_and_persist_facts``
caught the persistence exception, logged at WARNING, and returned 0 -- so a
caller saw a zero count with no way to tell a clean "nothing extracted" run
from a swallowed write failure (and, on surrealdb, a partial write).

The storage double here raises on the write call to model that failure; the
tests assert the engine threads a Degradation through to the result.
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
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig

pytestmark = pytest.mark.unit


def _make_chunk(content: str, namespace_id: UUID) -> Chunk:
    return Chunk(
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        chunk_index=0,
    )


def _make_event(text: str) -> ChronicleEvent:
    return ChronicleEvent(subject="alice", verb="met", object="bob", source_text=text, confidence=0.9)


class _StubEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class _FailingWriteCoordinator:
    """Storage double whose write_events / write_facts raise, like a backend
    error mid-write would."""

    def __init__(self, namespace: MemoryNamespace, chunks: list[Chunk]) -> None:
        self._namespace = namespace
        self._chunks_by_id = {c.id: c for c in chunks}

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        if self._namespace and self._namespace.namespace_id == namespace_id:
            return self._namespace
        return None

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def query_active_facts_for_subject(self, namespace_id: UUID, subject: str) -> list[Any]:
        return []

    async def write_events(self, events: list[Any], *, namespace_id: UUID) -> list[UUID]:
        raise RuntimeError("INJECTED: write_events backend failure (#1228)")

    async def write_facts(self, facts: list[Any], *, namespace_id: UUID) -> list[UUID]:
        raise RuntimeError("INJECTED: write_facts backend failure (#1228)")


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


def _patch_ingest(monkeypatch: pytest.MonkeyPatch, chunk_ids: list[UUID]) -> None:
    async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "document_id": str(uuid4()),
            "chunks": len(chunk_ids),
            "entities": 0,
            "relationships": 0,
            "extracted_relationships": 0,
            "inferred_relationships": 0,
            "entity_ids": [],
            "chunk_ids": chunk_ids,
            "phase_times": {},
        }

    monkeypatch.setattr("khora.pipelines.flows.ingest.process_document", fake_process_document)


def _wire(engine: ChronicleEngine, coord: _FailingWriteCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]
    engine._embedder = _StubEmbedder()  # type: ignore[assignment]

    async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
        return None

    async def fake_create_document(doc: Any) -> Any:
        return doc

    coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
    coord.create_document = fake_create_document  # type: ignore[attr-defined]


async def test_write_events_failure_surfaces_degradation(monkeypatch: pytest.MonkeyPatch) -> None:
    ns_id = uuid4()
    namespace = MemoryNamespace(namespace_id=ns_id)
    chunk = _make_chunk("alice met bob", ns_id)
    coord = _FailingWriteCoordinator(namespace, [chunk])

    extractor = AsyncMock()
    extractor.extract_events.side_effect = lambda text, **_kw: [_make_event(text)]

    engine = _bare_engine()
    _wire(engine, coord)
    engine._event_extractor = extractor
    _patch_ingest(monkeypatch, [chunk.id])

    result = await engine.remember(
        "alice met bob",
        ns_id,
        entity_types=[],
        relationship_types=[],
        expertise=ExpertiseConfig(
            name="x",
            events=EventExtractionConfig(enabled=True),
            facts=FactExtractionConfig(enabled=False),
        ),
    )

    # Count is still 0 (nothing persisted), but the failure is now observable.
    assert result.metadata["events_extracted"] == 0
    degradations = result.metadata.get("degradations", [])
    assert any(d["component"] == "chronicle.events" and d["reason"] == "write_events_failed" for d in degradations), (
        f"expected a write_events degradation, got {degradations!r}"
    )


async def test_write_facts_failure_surfaces_degradation(monkeypatch: pytest.MonkeyPatch) -> None:
    ns_id = uuid4()
    namespace = MemoryNamespace(namespace_id=ns_id)
    chunk = _make_chunk("alice works at acme", ns_id)
    coord = _FailingWriteCoordinator(namespace, [chunk])

    from khora.engines.chronicle.compression import MemoryFact

    fact = MemoryFact(
        id=uuid4(),
        subject="alice",
        predicate="works_at",
        object_="acme",
        fact_text="alice works at acme",
        confidence=0.9,
    )

    extractor = AsyncMock()

    async def fake_extract_facts(text: str, *, chunk_id: UUID, namespace_id: UUID, errors_out: Any = None) -> list:
        del text, chunk_id, namespace_id, errors_out
        return [fact]

    extractor.extract_facts.side_effect = fake_extract_facts

    engine = _bare_engine()
    _wire(engine, coord)
    engine._fact_extractor = extractor
    _patch_ingest(monkeypatch, [chunk.id])

    result = await engine.remember(
        "alice works at acme",
        ns_id,
        entity_types=[],
        relationship_types=[],
        expertise=ExpertiseConfig(
            name="x",
            events=EventExtractionConfig(enabled=False),
            # reconcile=False exercises the fast-path write; reconcile=True
            # exercises the _reconcile_facts write -- both must degrade.
            facts=FactExtractionConfig(enabled=True, reconcile=False),
        ),
    )

    assert result.metadata["facts_extracted"] == 0
    degradations = result.metadata.get("degradations", [])
    assert any(d["component"] == "chronicle.facts" and d["reason"] == "write_facts_failed" for d in degradations), (
        f"expected a write_facts degradation, got {degradations!r}"
    )


async def test_write_facts_failure_under_reconcile_surfaces_degradation(monkeypatch: pytest.MonkeyPatch) -> None:
    ns_id = uuid4()
    namespace = MemoryNamespace(namespace_id=ns_id)
    chunk = _make_chunk("alice works at acme", ns_id)
    coord = _FailingWriteCoordinator(namespace, [chunk])

    from khora.engines.chronicle.compression import FactOperation, MemoryFact, ReconcileAction

    fact = MemoryFact(
        id=uuid4(),
        subject="alice",
        predicate="works_at",
        object_="acme",
        fact_text="alice works at acme",
        confidence=0.9,
    )

    extractor = AsyncMock()

    async def fake_extract_facts(text: str, *, chunk_id: UUID, namespace_id: UUID, errors_out: Any = None) -> list:
        del text, chunk_id, namespace_id, errors_out
        return [fact]

    extractor.extract_facts.side_effect = fake_extract_facts

    compressor = AsyncMock()
    compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.ADD)

    engine = _bare_engine()
    _wire(engine, coord)
    engine._fact_extractor = extractor
    engine._compressor = compressor
    _patch_ingest(monkeypatch, [chunk.id])

    result = await engine.remember(
        "alice works at acme",
        ns_id,
        entity_types=[],
        relationship_types=[],
        expertise=ExpertiseConfig(
            name="x",
            events=EventExtractionConfig(enabled=False),
            facts=FactExtractionConfig(enabled=True, reconcile=True),
        ),
    )

    assert result.metadata["facts_extracted"] == 0
    degradations = result.metadata.get("degradations", [])
    assert any(d["component"] == "chronicle.facts" and d["reason"] == "write_facts_failed" for d in degradations), (
        f"expected a write_facts degradation under reconcile, got {degradations!r}"
    )
