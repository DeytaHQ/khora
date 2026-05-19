"""Unit tests for ``Khora.recall()`` document-upgrade pass.

Covers the contract:

- ``recall()`` no longer accepts ``include_sources``; the entity-read
  methods still do.
- ``RecallResult.documents`` is unconditionally populated by unioning
  every doc id referenced by chunks / entities / relationships.
- Unresolvable doc ids materialise minimal ``DocumentProjection`` stubs
  (referential-integrity invariant) and bump the
  ``khora.recall.dangling_ref`` counter, tagged by ``referrer``.
- Storage failures during the upgrade pass fail open: the result is
  returned with engine stubs intact and ``engine_info["document_upgrade_failed"]``
  set.
- ``connected_entity_ids`` is derived by inverting
  ``RecallEntity.source_chunk_ids``, ordered by entity score.
- ``DocumentProjection`` ids round-trip as ``UUID`` even when an engine
  hands in str-typed ids.
- The ``khora.recall.dangling_ref`` counter never carries a
  ``namespace_id`` attribute.
- Every storage backend's ``get_document_projections_batch`` returns
  the wider field set.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from opentelemetry import metrics as _otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from khora.core.models import (
    DocumentProjection,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
)

from .helpers import make_kb as _make_kb

# ---------------------------------------------------------------------------
# OTel meter fixture — installs an in-memory MeterProvider and rebinds
# the cached ``_RECALL_DANGLING_REF_COUNTER`` in khora.khora so the
# counter routes to the test reader.
# ---------------------------------------------------------------------------


def _reset_otel_globals() -> None:
    import opentelemetry.metrics._internal as _m
    import opentelemetry.trace as _t
    from opentelemetry.metrics._internal import _ProxyMeterProvider
    from opentelemetry.trace import ProxyTracerProvider

    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()

    _m._METER_PROVIDER_SET_ONCE = _m.Once()
    _m._METER_PROVIDER = None
    _m._PROXY_METER_PROVIDER = _ProxyMeterProvider()


@pytest.fixture
def metric_reader(monkeypatch: pytest.MonkeyPatch):
    """Install an in-memory MeterProvider and rebind khora's counter cache."""
    _reset_otel_globals()
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    _otel_metrics.set_meter_provider(mp)

    from khora import khora as khora_mod
    from khora.telemetry import _otel as _otel_module
    from khora.telemetry.metrics import metric_counter

    _otel_module._METER = _otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)
    new_counter = metric_counter(
        "khora.recall.dangling_ref",
        description="Dangling document references in recall results, by referrer kind.",
    )
    monkeypatch.setattr(khora_mod, "_RECALL_DANGLING_REF_COUNTER", new_counter)

    yield reader

    _reset_otel_globals()
    _otel_module._METER = _otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)


def _dangling_ref_points(reader: InMemoryMetricReader) -> list:
    """Return all data points emitted for ``khora.recall.dangling_ref``."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    points = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "khora.recall.dangling_ref":
                    points.extend(metric.data.data_points)
    return points


# ---------------------------------------------------------------------------
# 1. kwarg removal — recall() rejects include_sources, entity-read keeps it
# ---------------------------------------------------------------------------


class TestIncludeSourcesKwargRemoval:
    @pytest.mark.asyncio
    async def test_recall_rejects_include_sources_kwarg(self) -> None:
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        with pytest.raises(TypeError):
            await kb.recall(query="x", namespace=ns_id, include_sources=True)  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_entity_read_methods_still_accept_include_sources(self) -> None:
        """``get_entity`` / ``list_entities`` / ``find_related_entities``
        / ``search_entities`` must still accept ``include_sources=True``."""
        kb = _make_kb(connected=True)
        ns_id = uuid4()

        # All four should not raise TypeError on the kwarg. They may
        # return None / [] given the mocked engine.
        await kb.get_entity(uuid4(), namespace=ns_id, include_sources=True)
        await kb.list_entities(namespace=ns_id, include_sources=True)
        await kb.find_related_entities(uuid4(), namespace=ns_id, include_sources=True)
        await kb.search_entities("q", namespace=ns_id, include_sources=True)


# ---------------------------------------------------------------------------
# 2. documents[] populated unconditionally — union over all referrers
# ---------------------------------------------------------------------------


class TestDocumentsPopulatedUnconditionally:
    @pytest.mark.asyncio
    async def test_documents_union_chunks_entities_relationships(self) -> None:
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)

        doc_a = uuid4()
        doc_b = uuid4()
        doc_c = uuid4()
        doc_d = uuid4()

        chunk_a = RecallChunk(id=uuid4(), document_id=doc_a, content="a", score=0.9, created_at=now)
        chunk_b = RecallChunk(id=uuid4(), document_id=doc_b, content="b", score=0.8, created_at=now)
        entity = RecallEntity(
            id=uuid4(),
            name="E",
            entity_type="X",
            description="",
            score=0.7,
            attributes={},
            mention_count=0,
            source_document_ids=[doc_c],
            source_chunk_ids=[],
        )
        rel = RecallRelationship(
            id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="R",
            description="",
            score=0.6,
            valid_from=None,
            valid_until=None,
            source_document_ids=[doc_d],
        )

        # Engine emits stubs only for the chunk-referenced docs;
        # entity / rel docs must still appear after the upgrade pass.
        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[
                DocumentProjection(id=doc_a, created_at=now),
                DocumentProjection(id=doc_b, created_at=now),
            ],
            chunks=[chunk_a, chunk_b],
            entities=[entity],
            relationships=[rel],
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(
            return_value={
                doc_a: DocumentProjection(id=doc_a, created_at=now, title="A"),
                doc_b: DocumentProjection(id=doc_b, created_at=now, title="B"),
                doc_c: DocumentProjection(id=doc_c, created_at=now, title="C"),
                doc_d: DocumentProjection(id=doc_d, created_at=now, title="D"),
            }
        )

        result = await kb.recall(query="q", namespace=ns_id)

        assert {d.id for d in result.documents} == {doc_a, doc_b, doc_c, doc_d}
        by_id = {d.id: d for d in result.documents}
        assert by_id[doc_c].title == "C"
        assert by_id[doc_d].title == "D"


# ---------------------------------------------------------------------------
# 3. Referential integrity stub — dangling chunk doc id → minimal stub
#    and the dangling_ref counter ticks with referrer="chunk", value=1.
# ---------------------------------------------------------------------------


class TestReferentialIntegrityStub:
    @pytest.mark.asyncio
    async def test_dangling_chunk_doc_id_yields_stub_and_counter(self, metric_reader) -> None:
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)
        missing_doc = uuid4()

        chunk = RecallChunk(id=uuid4(), document_id=missing_doc, content="x", score=0.5, created_at=now)
        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[],
            chunks=[chunk],
            entities=[],
            relationships=[],
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(return_value={})

        # Does not raise.
        result = await kb.recall(query="q", namespace=ns_id)

        by_id = {d.id: d for d in result.documents}
        assert missing_doc in by_id
        stub = by_id[missing_doc]
        assert isinstance(stub, DocumentProjection)
        assert stub.id == missing_doc
        assert stub.source_type == "library"
        # Stub carries no upstream metadata.
        assert stub.title is None
        assert stub.external_id is None

        points = _dangling_ref_points(metric_reader)
        chunk_points = [p for p in points if dict(p.attributes).get("referrer") == "chunk"]
        assert len(chunk_points) == 1
        assert chunk_points[0].value == 1


# ---------------------------------------------------------------------------
# 4. Fail-open path — storage raises, recall still returns, warning logged.
# ---------------------------------------------------------------------------


class TestFailOpenOnStorageFailure:
    @pytest.mark.asyncio
    async def test_storage_exception_records_reason_and_does_not_crash(self) -> None:
        from loguru import logger

        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)

        doc_id = uuid4()
        chunk = RecallChunk(id=uuid4(), document_id=doc_id, content="x", score=0.5, created_at=now)
        engine_stub = DocumentProjection(id=doc_id, created_at=now, title="engine-stub")

        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[engine_stub],
            chunks=[chunk],
            entities=[],
            relationships=[],
            engine_info={"engine": "skeleton"},
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(side_effect=RuntimeError("boom"))

        captured: list[str] = []
        handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            result = await kb.recall(query="q", namespace=ns_id)
        finally:
            logger.remove(handler_id)

        # engine_info reports the failure reason.
        assert result.engine_info["document_upgrade_failed"].startswith("RuntimeError:")
        assert "boom" in result.engine_info["document_upgrade_failed"]
        # Engine stub survived.
        by_id = {d.id: d for d in result.documents}
        assert by_id[doc_id].title == "engine-stub"
        # Existing engine_info keys are preserved.
        assert result.engine_info["engine"] == "skeleton"
        # A warning landed in loguru.
        assert any("document upgrade failed" in line for line in captured), captured


# ---------------------------------------------------------------------------
# 5. connected_entity_ids inversion — invert RecallEntity.source_chunk_ids,
#    ordered by entity score.
# ---------------------------------------------------------------------------


class TestConnectedEntityIdsInversion:
    @pytest.mark.asyncio
    async def test_chunk_to_entity_inversion_with_score_ordering(self) -> None:
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)
        doc_id = uuid4()

        c1 = RecallChunk(id=uuid4(), document_id=doc_id, content="c1", score=0.9, created_at=now)
        c2 = RecallChunk(id=uuid4(), document_id=doc_id, content="c2", score=0.8, created_at=now)

        # e1 has the higher score, so on c2 it must appear before e2.
        e1 = RecallEntity(
            id=uuid4(),
            name="E1",
            entity_type="X",
            description="",
            score=0.95,
            attributes={},
            mention_count=0,
            source_document_ids=[doc_id],
            source_chunk_ids=[c1.id, c2.id],
        )
        e2 = RecallEntity(
            id=uuid4(),
            name="E2",
            entity_type="X",
            description="",
            score=0.50,
            attributes={},
            mention_count=0,
            source_document_ids=[doc_id],
            source_chunk_ids=[c2.id],
        )

        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[DocumentProjection(id=doc_id, created_at=now)],
            chunks=[c1, c2],
            entities=[e1, e2],  # already score-sorted by the engine
            relationships=[],
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(
            return_value={doc_id: DocumentProjection(id=doc_id, created_at=now, title="D")}
        )

        result = await kb.recall(query="q", namespace=ns_id)

        out_by_id = {c.id: c for c in result.chunks}
        assert out_by_id[c1.id].connected_entity_ids == [e1.id]
        assert out_by_id[c2.id].connected_entity_ids == [e1.id, e2.id]


# ---------------------------------------------------------------------------
# 6. UUID coercion at the projection boundary — a fake engine that hands
#    in str-typed UUIDs should still produce ``UUID`` instances on the
#    typed projections (the dataclass is the contract surface).
# ---------------------------------------------------------------------------


class TestProjectionUUIDCoercion:
    @pytest.mark.asyncio
    async def test_projection_emits_uuid_for_document_id(self) -> None:
        """The dataclass annotation is ``UUID``. Any engine that constructs
        a ``RecallChunk`` with ``document_id`` already typed as ``UUID``
        round-trips through ``recall()`` as ``UUID`` — even when the engine
        previously stored ids as strings, the construction site must
        coerce. This is a contract test on the projection boundary.
        """
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)

        # Simulate an engine that has str-typed ids at the raw row level
        # but coerces at the projection boundary.
        raw_doc_id_str = str(uuid4())
        doc_id = UUID(raw_doc_id_str)
        raw_chunk_id_str = str(uuid4())
        chunk = RecallChunk(
            id=UUID(raw_chunk_id_str),
            document_id=UUID(raw_doc_id_str),
            content="hello",
            score=0.9,
            created_at=now,
        )

        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[DocumentProjection(id=doc_id, created_at=now)],
            chunks=[chunk],
            entities=[],
            relationships=[],
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(
            return_value={doc_id: DocumentProjection(id=doc_id, created_at=now, title="D")}
        )

        result = await kb.recall(query="q", namespace=ns_id)

        assert isinstance(result.chunks[0].document_id, UUID)
        assert isinstance(result.chunks[0].id, UUID)
        assert isinstance(result.documents[0].id, UUID)


# ---------------------------------------------------------------------------
# 7. Cardinality regression — dangling_ref MUST NOT carry namespace_id.
# ---------------------------------------------------------------------------


class TestCounterCardinality:
    @pytest.mark.asyncio
    async def test_no_namespace_id_on_dangling_ref_counter(self, metric_reader) -> None:
        kb = _make_kb(connected=True)
        ns_id = uuid4()
        now = datetime.now(UTC)

        # Force every referrer kind to register a dangling-ref hit.
        missing_chunk_doc = uuid4()
        missing_entity_doc = uuid4()
        missing_rel_doc = uuid4()

        chunk = RecallChunk(
            id=uuid4(),
            document_id=missing_chunk_doc,
            content="x",
            score=0.5,
            created_at=now,
        )
        entity = RecallEntity(
            id=uuid4(),
            name="E",
            entity_type="X",
            description="",
            score=0.5,
            attributes={},
            mention_count=0,
            source_document_ids=[missing_entity_doc],
            source_chunk_ids=[],
        )
        rel = RecallRelationship(
            id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="R",
            description="",
            score=0.5,
            valid_from=None,
            valid_until=None,
            source_document_ids=[missing_rel_doc],
        )

        mock_result = RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[],
            chunks=[chunk],
            entities=[entity],
            relationships=[rel],
        )
        kb._engine.recall = AsyncMock(return_value=mock_result)
        kb._engine._storage.get_document_projections_batch = AsyncMock(return_value={})

        await kb.recall(query="q", namespace=ns_id)

        points = _dangling_ref_points(metric_reader)
        assert points, "expected dangling_ref points to be emitted"
        seen_referrers = set()
        for p in points:
            attrs = dict(p.attributes)
            # The cardinality contract: only "referrer" is permitted.
            assert "namespace_id" not in attrs, f"khora.recall.dangling_ref must not carry namespace_id: {attrs}"
            assert set(attrs.keys()) <= {"referrer"}, f"unexpected labels on dangling_ref: {set(attrs.keys())}"
            seen_referrers.add(attrs.get("referrer"))
        # All three referrer kinds must have been recorded.
        assert seen_referrers == {"chunk", "entity", "relationship"}


# ---------------------------------------------------------------------------
# 8. Backend smoke — ``get_document_projections_batch`` on a real backend
#    returns the wider field set. Uses SQLite in-memory.
# ---------------------------------------------------------------------------


class TestBackendProjectionsBatch:
    @pytest.mark.asyncio
    async def test_sqlite_backend_returns_wider_field_set(self) -> None:
        from khora.core.models import Document, MemoryNamespace
        from khora.core.models.document import DocumentStatus
        from khora.core.models.tenancy import TenancyMode
        from khora.storage.backends.sqlite import SQLiteRelationalBackend

        backend = SQLiteRelationalBackend(":memory:")
        await backend.connect()
        try:
            now = datetime.now(UTC)
            ns = MemoryNamespace(
                id=uuid4(),
                namespace_id=uuid4(),
                tenancy_mode=TenancyMode.SHARED,
                version=1,
                is_active=True,
                config_overrides={},
                sync_checkpoints={},
                metadata={},
                created_at=now,
                updated_at=now,
            )
            await backend.create_namespace(ns)

            doc = Document(
                id=uuid4(),
                namespace_id=ns.id,
                content="hello world",
                status=DocumentStatus.PENDING,
                title="Test Title",
                external_id="ext-123",
                source="src.txt",
                source_name="Source Name",
                source_url="https://example.com/x",
                source_type="file",
                content_type="text/plain",
                checksum="abc",
                metadata={"k": "v"},
                created_at=now,
                updated_at=now,
            )
            await backend.create_document(doc)

            projections = await backend.get_document_projections_batch([doc.id])

            assert doc.id in projections
            proj = projections[doc.id]
            assert isinstance(proj, DocumentProjection)
            assert isinstance(proj.id, UUID)
            # Wider field set — every public projection column survives.
            assert proj.title == "Test Title"
            assert proj.external_id == "ext-123"
            assert proj.source == "src.txt"
            assert proj.source_name == "Source Name"
            assert proj.source_url == "https://example.com/x"
            assert proj.source_type == "file"
            assert proj.content_type == "text/plain"
            assert proj.metadata == {"k": "v"}
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_sqlite_backend_empty_input_returns_empty(self) -> None:
        from khora.storage.backends.sqlite import SQLiteRelationalBackend

        backend = SQLiteRelationalBackend(":memory:")
        await backend.connect()
        try:
            assert await backend.get_document_projections_batch([]) == {}
        finally:
            await backend.disconnect()
