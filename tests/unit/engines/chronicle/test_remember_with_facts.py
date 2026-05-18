"""Chronicle #3: tests for FactExtractor wiring with reconciliation.

These tests exercise the fact-extraction path that runs after the
ingest pipeline finishes:

* The toggle resolution rule (per-namespace beats expertise default).
* ADD / UPDATE / DELETE / NOOP reconciliation.
* Per-chunk failure resilience.
* ``facts.reconcile=False`` fast path.
* ``metadata["facts_extracted"]`` accounting.

The ingest pipeline and the LLM are patched out — these tests stay
purely behavioural at the chronicle layer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, ChunkMetadata, MemoryNamespace
from khora.engines.chronicle.compression import (
    FactOperation,
    MemoryFact,
    ReconcileAction,
)
from khora.engines.chronicle.engine import ChronicleEngine
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import (
    EventExtractionConfig,
    FactExtractionConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, namespace_id: UUID) -> Chunk:
    document_id = uuid4()
    return Chunk(
        namespace_id=namespace_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
    )


def _make_fact(
    subject: str = "alice",
    predicate: str = "works_at",
    obj: str = "acme",
    *,
    fact_text: str | None = None,
    fact_id: UUID | None = None,
) -> MemoryFact:
    return MemoryFact(
        id=fact_id or uuid4(),
        subject=subject,
        predicate=predicate,
        object_=obj,
        fact_text=fact_text or f"{subject} {predicate} {obj}",
        confidence=0.9,
    )


def _expertise(*, facts_enabled: bool = True, reconcile: bool = True) -> ExpertiseConfig:
    return ExpertiseConfig(
        name="x",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=facts_enabled, reconcile=reconcile),
    )


class _StubEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class _RecordingCoordinator:
    """Storage coordinator double exposing the methods the engine calls."""

    def __init__(
        self,
        namespace: MemoryNamespace,
        chunks: list[Chunk],
        *,
        active_facts_by_subject: dict[str, list[MemoryFact]] | None = None,
    ) -> None:
        self._namespace = namespace
        self._chunks_by_id = {c.id: c for c in chunks}
        self._active_facts = active_facts_by_subject or {}
        self.write_facts_calls: list[tuple[list[MemoryFact], UUID]] = []
        self.supersede_calls: list[tuple[UUID, UUID]] = []
        self.write_events_calls: list[tuple[list[Any], UUID]] = []
        self.query_active_subjects: list[str] = []

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        if self._namespace and self._namespace.namespace_id == namespace_id:
            return self._namespace
        return None

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def query_active_facts_for_subject(self, namespace_id: UUID, subject: str) -> list[MemoryFact]:
        self.query_active_subjects.append(subject)
        return list(self._active_facts.get(subject, []))

    async def write_facts(self, facts: list[MemoryFact], *, namespace_id: UUID) -> list[UUID]:
        self.write_facts_calls.append((list(facts), namespace_id))
        return [f.id for f in facts]

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID) -> None:
        self.supersede_calls.append((fact_id, superseded_by))

    async def write_events(self, events: list[Any], *, namespace_id: UUID) -> list[UUID]:
        self.write_events_calls.append((list(events), namespace_id))
        return [getattr(ev, "id", uuid4()) for ev in events]


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


def _wire_engine(
    engine: ChronicleEngine,
    coord: _RecordingCoordinator,
    *,
    fact_extractor: Any | None = None,
    compressor: Any | None = None,
) -> None:
    engine._storage = coord  # type: ignore[assignment]
    engine._embedder = _StubEmbedder()  # type: ignore[assignment]
    if fact_extractor is not None:
        engine._fact_extractor = fact_extractor
    if compressor is not None:
        engine._compressor = compressor


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, chunk_ids: list[UUID]) -> None:
    """Bypass ``process_document`` so remember() reaches the fact wiring."""

    async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "document_id": str(uuid4()),
            "chunks": len(chunk_ids),
            "entities": 0,
            "relationships": 0,
            "extracted_relationships": 0,
            "inferred_relationships": 0,
            "entity_ids": [],
            "chunk_ids": list(chunk_ids),
            "phase_times": {},
        }

    monkeypatch.setattr("khora.pipelines.flows.ingest.process_document", fake_process_document)


def _stub_remember_doc_helpers(coord: _RecordingCoordinator) -> None:
    """Bypass the document-create / dedup hits remember() does."""

    async def fake_get_doc_by_checksum(*_a: Any, **_kw: Any) -> None:
        return None

    async def fake_create_document(doc: Any) -> Any:
        return doc

    coord.get_document_by_checksum = fake_get_doc_by_checksum  # type: ignore[attr-defined]
    coord.create_document = fake_create_document  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryFactDataclass:
    def test_roundtrip_to_dict(self) -> None:
        cid = uuid4()
        fact = MemoryFact(
            subject="alice",
            predicate="works_at",
            object_="acme",
            fact_text="alice works at acme",
            confidence=0.85,
            source_chunk_ids=[cid],
        )
        d = fact.to_dict()
        assert d["subject"] == "alice"
        assert d["predicate"] == "works_at"
        assert d["object"] == "acme"
        assert d["fact_text"] == "alice works at acme"
        assert d["confidence"] == 0.85
        assert d["is_active"] is True
        assert d["superseded_by"] is None
        assert d["source_chunk_ids"] == [str(cid)]

    def test_default_id_is_uuid(self) -> None:
        f = MemoryFact(subject="s", predicate="p", object_="o", fact_text="s p o")
        assert isinstance(f.id, UUID)


# ---------------------------------------------------------------------------
# _facts_enabled — toggle resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactsEnabledResolution:
    def test_namespace_override_disables_when_expertise_enables(self) -> None:
        ns = MemoryNamespace(config_overrides={"facts": {"enabled": False}})
        expertise = _expertise(facts_enabled=True)
        assert ChronicleEngine._facts_enabled(ns, expertise) is False

    def test_namespace_override_enables_when_expertise_disables(self) -> None:
        ns = MemoryNamespace(config_overrides={"facts": {"enabled": True}})
        expertise = _expertise(facts_enabled=False)
        assert ChronicleEngine._facts_enabled(ns, expertise) is True

    def test_falls_back_to_expertise_default(self) -> None:
        ns = MemoryNamespace()
        expertise = _expertise(facts_enabled=False)
        assert ChronicleEngine._facts_enabled(ns, expertise) is False

    def test_default_on_when_neither_provided(self) -> None:
        assert ChronicleEngine._facts_enabled(None, None) is True

    def test_irrelevant_namespace_keys_do_not_override(self) -> None:
        ns = MemoryNamespace(config_overrides={"events": {"enabled": False}})
        expertise = _expertise(facts_enabled=True)
        assert ChronicleEngine._facts_enabled(ns, expertise) is True


# ---------------------------------------------------------------------------
# remember() with facts enabled / disabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberWithFacts:
    @pytest.mark.asyncio
    async def test_remember_writes_facts_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice works at acme", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        async def fake_extract(text: str, *, chunk_id: UUID, namespace_id: UUID) -> list[MemoryFact]:
            return [_make_fact()]

        extractor = AsyncMock()
        extractor.extract_facts.side_effect = fake_extract

        # Compressor with no LLM call: empty existing → ADD by definition.
        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.ADD)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(facts_enabled=True, reconcile=True),
        )

        assert len(coord.write_facts_calls) == 1
        facts, written_ns = coord.write_facts_calls[0]
        assert written_ns == ns_id
        assert len(facts) == 1
        assert facts[0].subject == "alice"
        assert facts[0].predicate == "works_at"
        assert result.metadata["facts_extracted"] == 1

    @pytest.mark.asyncio
    async def test_remember_skips_facts_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice works at acme", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        extractor = AsyncMock()
        extractor.extract_facts.side_effect = AssertionError("must not run")

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor)

        result = await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(facts_enabled=False),
        )

        assert coord.write_facts_calls == []
        extractor.extract_facts.assert_not_called()
        assert result.metadata["facts_extracted"] == 0

    @pytest.mark.asyncio
    async def test_namespace_override_disables_facts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(
            namespace_id=ns_id,
            config_overrides={"facts": {"enabled": False}},
        )
        chunk = _make_chunk("alice works at acme", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        extractor = AsyncMock()
        extractor.extract_facts.side_effect = AssertionError("must not run")

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor)

        await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            # Expertise enables facts, namespace overrides it off.
            expertise=_expertise(facts_enabled=True),
        )

        assert coord.write_facts_calls == []
        extractor.extract_facts.assert_not_called()


# ---------------------------------------------------------------------------
# Reconciliation: ADD / UPDATE / DELETE / NOOP
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReconciliation:
    @pytest.mark.asyncio
    async def test_add_when_no_existing_fact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice works at acme", ns_id)
        coord = _RecordingCoordinator(namespace, [chunk])
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        new_fact = _make_fact()

        extractor = AsyncMock()
        extractor.extract_facts.return_value = [new_fact]

        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.ADD)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(),
        )

        assert len(coord.write_facts_calls) == 1
        assert coord.write_facts_calls[0][0] == [new_fact]
        assert coord.supersede_calls == []
        assert result.metadata["facts_extracted"] == 1

    @pytest.mark.asyncio
    async def test_update_writes_new_and_supersedes_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing fact about same subject + UPDATE → new fact written + old superseded."""
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice now lives in berlin", ns_id)
        old_fact = _make_fact(predicate="lives_in", obj="paris", fact_text="alice lives in paris")
        coord = _RecordingCoordinator(
            namespace,
            [chunk],
            active_facts_by_subject={"alice": [old_fact]},
        )
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        new_fact = _make_fact(predicate="lives_in", obj="berlin", fact_text="alice lives in berlin")

        extractor = AsyncMock()
        extractor.extract_facts.return_value = [new_fact]

        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.UPDATE, target=old_fact)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice now lives in berlin",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(),
        )

        # New fact written, old fact superseded by new fact's id.
        assert len(coord.write_facts_calls) == 1
        assert coord.write_facts_calls[0][0] == [new_fact]
        assert coord.supersede_calls == [(old_fact.id, new_fact.id)]
        assert result.metadata["facts_extracted"] == 1

    @pytest.mark.asyncio
    async def test_noop_skips_when_identical_fact_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice works at acme", ns_id)
        existing = _make_fact()
        coord = _RecordingCoordinator(
            namespace,
            [chunk],
            active_facts_by_subject={"alice": [existing]},
        )
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        # Same SVO triple → reconcile_fact will see triple-match and short-circuit
        # to NOOP without an LLM call (handled inside MemoryCompressor). Use the
        # real compressor here to exercise that fast path.
        new_fact = _make_fact()

        extractor = AsyncMock()
        extractor.extract_facts.return_value = [new_fact]

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor)

        result = await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(),
        )

        assert coord.write_facts_calls == []
        assert coord.supersede_calls == []
        assert result.metadata["facts_extracted"] == 0

    @pytest.mark.asyncio
    async def test_delete_supersedes_without_writing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice no longer works at acme", ns_id)
        old_fact = _make_fact()
        coord = _RecordingCoordinator(
            namespace,
            [chunk],
            active_facts_by_subject={"alice": [old_fact]},
        )
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        new_fact = _make_fact(predicate="left", obj="acme", fact_text="alice left acme")

        extractor = AsyncMock()
        extractor.extract_facts.return_value = [new_fact]

        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.DELETE, target=old_fact)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice no longer works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(),
        )

        # No write, but supersede was called (self-pointer = tombstone).
        assert coord.write_facts_calls == []
        assert len(coord.supersede_calls) == 1
        assert coord.supersede_calls[0][0] == old_fact.id
        assert result.metadata["facts_extracted"] == 0


# ---------------------------------------------------------------------------
# Per-chunk failure resilience + reconcile=False fast path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResilienceAndFastPath:
    @pytest.mark.asyncio
    async def test_per_chunk_failure_does_not_block_other_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        good = _make_chunk("alice works at acme", ns_id)
        bad = _make_chunk("trigger boom", ns_id)
        coord = _RecordingCoordinator(namespace, [good, bad])
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [good.id, bad.id])

        good_fact = _make_fact()

        async def flaky_extract(text: str, *, chunk_id: UUID, namespace_id: UUID) -> list[MemoryFact]:
            if chunk_id == bad.id:
                raise RuntimeError("LLM is on fire")
            return [good_fact]

        extractor = AsyncMock()
        extractor.extract_facts.side_effect = flaky_extract

        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.ADD)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice works at acme. trigger boom.",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(),
        )

        assert len(coord.write_facts_calls) == 1
        facts, _ = coord.write_facts_calls[0]
        assert len(facts) == 1
        assert facts[0] is good_fact
        assert result.metadata["facts_extracted"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_false_writes_all_as_add(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunk = _make_chunk("alice works at acme", ns_id)
        # Active fact exists, but reconcile=False means we ignore it.
        coord = _RecordingCoordinator(
            namespace,
            [chunk],
            active_facts_by_subject={"alice": [_make_fact()]},
        )
        _stub_remember_doc_helpers(coord)
        _stub_pipeline(monkeypatch, [chunk.id])

        f1 = _make_fact()
        f2 = _make_fact(predicate="lives_in", obj="paris")

        extractor = AsyncMock()
        extractor.extract_facts.return_value = [f1, f2]

        # Compressor must NOT be consulted when reconcile=False.
        compressor = AsyncMock()
        compressor.reconcile_fact.side_effect = AssertionError("reconcile must not run")

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        result = await engine.remember(
            "alice works at acme",
            ns_id,
            entity_types=[],
            relationship_types=[],
            expertise=_expertise(reconcile=False),
        )

        assert len(coord.write_facts_calls) == 1
        facts, _ = coord.write_facts_calls[0]
        assert facts == [f1, f2]
        assert coord.query_active_subjects == []  # never queried
        assert coord.supersede_calls == []
        assert result.metadata["facts_extracted"] == 2


# ---------------------------------------------------------------------------
# remember_batch()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchWithFacts:
    @pytest.mark.asyncio
    async def test_remember_batch_persists_all_facts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns_id = uuid4()
        namespace = MemoryNamespace(namespace_id=ns_id)
        chunks = [_make_chunk(f"doc {i} content", ns_id) for i in range(5)]
        coord = _RecordingCoordinator(namespace, chunks)

        async def fake_extract(text: str, *, chunk_id: UUID, namespace_id: UUID) -> list[MemoryFact]:
            # Each chunk yields one unique fact to keep ADDs distinct.
            return [
                _make_fact(
                    subject=f"subj-{chunk_id}",
                    predicate="says",
                    obj=text,
                    fact_text=text,
                )
            ]

        extractor = AsyncMock()
        extractor.extract_facts.side_effect = fake_extract

        compressor = AsyncMock()
        compressor.reconcile_fact.return_value = ReconcileAction(op=FactOperation.ADD)

        engine = _bare_engine()
        _wire_engine(engine, coord, fact_extractor=extractor, compressor=compressor)

        async def fake_list_entities(*_a: Any, **_kw: Any) -> list[Any]:
            return []

        coord.list_entities = fake_list_entities  # type: ignore[attr-defined]

        async def fake_ingest_documents(
            namespace_id: UUID,
            doc_inputs: list[dict[str, Any]],
            storage: Any,
            **_kw: Any,
        ) -> dict[str, Any]:
            per_doc = [{"chunk_ids": [chunks[i].id], "phase_times": {}} for i in range(len(doc_inputs))]
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
            expertise=_expertise(),
            deduplicate=False,
        )

        assert len(coord.write_facts_calls) == 1
        facts, written_ns = coord.write_facts_calls[0]
        assert written_ns == ns_id
        assert len(facts) == 5
        assert result.metadata["facts_extracted"] == 5
