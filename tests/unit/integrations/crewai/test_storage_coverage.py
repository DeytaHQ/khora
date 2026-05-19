"""Coverage tests for ``khora.integrations.crewai.storage``.

Existing tests in ``test_adapter.py`` cover the basic surface. This
module fills the gaps:
- ``delete()`` filter paths (scope_prefix, categories, older_than, metadata_filter)
- ``async`` siblings (asave / asearch / adelete)
- ``list_scopes`` walking documents
- ``list_categories`` aggregation
- ``count`` with and without scope_prefix
- ``reset`` delegates to delete
- ``_matches_filters`` cases
- ``_first_chunk_for`` when document fetch returns None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models.document import Chunk, ChunkMetadata, Document, DocumentMetadata
from khora.integrations.crewai.storage import (
    KhoraStorageBackend,
    _matches_filters,
    _peek_query_text,
    _stash_query_text,
)
from khora.khora import Khora


@dataclass
class _FakeMemoryRecord:
    id: str
    content: str
    scope: str = "/"
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    embedding: list[float] | None = None
    source: str | None = None
    private: bool = False


@dataclass
class _RememberResultStub:
    document_id: UUID
    namespace_id: UUID = field(default_factory=uuid4)
    chunks_created: int = 1
    entities_extracted: int = 0
    relationships_created: int = 0


@dataclass
class _RecallResultStub:
    chunks: list[tuple[Any, float]]
    query: str = ""
    namespace_id: UUID = field(default_factory=uuid4)
    entities: list[Any] = field(default_factory=list)
    context_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StatsStub:
    documents: int = 0


def _make_kb() -> Any:
    kb = AsyncMock(spec=Khora)
    kb.storage = MagicMock()
    return kb


def _make_backend(kb: Any, *, namespace_id: UUID | None = None) -> KhoraStorageBackend:
    return KhoraStorageBackend(
        kb=kb,
        namespace_id=namespace_id or uuid4(),
        user_id="user-1234567890",
        app_id="crewai",
        memory_record_cls=_FakeMemoryRecord,
    )


def _make_chunk(*, content: str = "x", document_id: UUID | None = None, custom: dict[str, Any] | None = None) -> Chunk:
    md = ChunkMetadata(document_id=document_id or uuid4(), custom=dict(custom or {}))
    return Chunk(content=content, document_id=md.document_id, metadata=md)


def _make_doc(
    *,
    namespace_id: UUID,
    custom: dict[str, Any] | None = None,
    external_id: str | None = None,
    created_at: datetime | None = None,
) -> Document:
    doc = Document(
        id=uuid4(),
        namespace_id=namespace_id,
        external_id=external_id,
        metadata=DocumentMetadata(custom=dict(custom or {})),
    )
    if created_at is not None:
        doc.created_at = created_at
    return doc


# ---------------------------------------------------------------------------
# _matches_filters helper (lines 618-628)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatchesFilters:
    def test_scope_match(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", scope="/team/eng/x")
        assert _matches_filters(rec, scope_prefix="/team/eng", categories=None, metadata_filter=None)

    def test_scope_no_match(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", scope="/team/sales")
        assert not _matches_filters(rec, scope_prefix="/team/eng", categories=None, metadata_filter=None)

    def test_scope_none_record_scope(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", scope="")
        assert not _matches_filters(rec, scope_prefix="/team", categories=None, metadata_filter=None)

    def test_category_disjoint_excluded(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", categories=["onboarding"])
        assert not _matches_filters(rec, scope_prefix=None, categories=["billing"], metadata_filter=None)

    def test_category_intersect_included(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", categories=["onboarding", "tips"])
        assert _matches_filters(rec, scope_prefix=None, categories=["tips"], metadata_filter=None)

    def test_metadata_match(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", metadata={"k": "v"})
        assert _matches_filters(rec, scope_prefix=None, categories=None, metadata_filter={"k": "v"})

    def test_metadata_mismatch(self) -> None:
        rec = _FakeMemoryRecord(id="r", content="c", metadata={"k": "v"})
        assert not _matches_filters(rec, scope_prefix=None, categories=None, metadata_filter={"k": "other"})

    def test_private_record_still_visible(self) -> None:
        # Adapter does not filter private (intentional)
        rec = _FakeMemoryRecord(id="r", content="c", private=True)
        assert _matches_filters(rec, scope_prefix=None, categories=None, metadata_filter=None)


# ---------------------------------------------------------------------------
# Query text stash
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryStash:
    def test_stash_and_peek(self) -> None:
        _stash_query_text("hello")
        assert _peek_query_text() == "hello"

    def test_peek_empty_returns_none(self) -> None:
        # Use a fresh attribute on a different thread; just verify that
        # peek can return None when nothing is stashed.
        import threading

        result_box: list[Any] = []

        def _run() -> None:
            result_box.append(_peek_query_text())

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        assert result_box[0] is None


# ---------------------------------------------------------------------------
# delete() filter paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteFilters:
    def test_delete_by_scope_prefix(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        keep = _make_doc(namespace_id=ns, custom={"crewai_scope": "/keep"})
        drop = _make_doc(namespace_id=ns, custom={"crewai_scope": "/drop/x"})
        kb.storage.list_documents = AsyncMock(side_effect=[[keep, drop], []])
        kb.forget = AsyncMock(return_value=True)

        n = backend.delete(scope_prefix="/drop")
        assert n == 1
        assert kb.forget.await_count == 1

    def test_delete_by_categories(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        with_cat = _make_doc(namespace_id=ns, custom={"crewai_categories": ["onboarding"]})
        without_cat = _make_doc(namespace_id=ns, custom={"crewai_categories": ["billing"]})
        kb.storage.list_documents = AsyncMock(side_effect=[[with_cat, without_cat], []])
        kb.forget = AsyncMock(return_value=True)

        n = backend.delete(categories=["onboarding"])
        assert n == 1

    def test_delete_by_older_than(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        now = datetime.now(UTC)
        old_doc = _make_doc(namespace_id=ns, created_at=now - timedelta(days=30))
        new_doc = _make_doc(namespace_id=ns, created_at=now)
        kb.storage.list_documents = AsyncMock(side_effect=[[old_doc, new_doc], []])
        kb.forget = AsyncMock(return_value=True)

        n = backend.delete(older_than=now - timedelta(days=1))
        assert n == 1

    def test_delete_by_metadata_filter(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        match = _make_doc(namespace_id=ns, custom={"author": "alice"})
        no_match = _make_doc(namespace_id=ns, custom={"author": "bob"})
        kb.storage.list_documents = AsyncMock(side_effect=[[match, no_match], []])
        kb.forget = AsyncMock(return_value=True)

        n = backend.delete(metadata_filter={"author": "alice"})
        assert n == 1

    def test_delete_clears_record_id_mapping(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        doc = _make_doc(namespace_id=ns, external_id="rid-1", custom={"crewai_scope": "/x"})
        backend._record_to_document["rid-1"] = doc.id
        kb.storage.list_documents = AsyncMock(side_effect=[[doc], []])
        kb.forget = AsyncMock(return_value=True)

        backend.delete(scope_prefix="/x")
        assert "rid-1" not in backend._record_to_document


# ---------------------------------------------------------------------------
# Async siblings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncSiblings:
    async def test_asave_forwards(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        doc_id = uuid4()
        kb.remember.return_value = _RememberResultStub(document_id=doc_id)
        backend = _make_backend(kb, namespace_id=ns)

        await backend.asave([_FakeMemoryRecord(id="r-1", content="hello")])
        kb.remember.assert_awaited_once()
        assert backend._record_to_document["r-1"] == doc_id

    async def test_asearch_returns_filtered(self) -> None:
        kb = _make_kb()
        chunk = _make_chunk(content="hit", custom={"crewai_scope": "/"})
        kb.recall.return_value = _RecallResultStub(chunks=[(chunk, 0.85)])
        backend = _make_backend(kb)
        _stash_query_text("q")
        out = await backend.asearch([0.0], limit=5)
        assert len(out) == 1
        assert out[0][0].content == "hit"

    async def test_asearch_post_filters_scope(self) -> None:
        kb = _make_kb()
        in_scope = _make_chunk(content="in", custom={"crewai_scope": "/team/eng"})
        out_scope = _make_chunk(content="out", custom={"crewai_scope": "/team/sales"})
        kb.recall.return_value = _RecallResultStub(chunks=[(in_scope, 0.9), (out_scope, 0.8)])
        backend = _make_backend(kb)
        _stash_query_text("q")
        out = await backend.asearch([0.0], scope_prefix="/team/eng", limit=10)
        assert [r.content for r, _ in out] == ["in"]

    async def test_adelete_by_record_ids(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1, d2 = uuid4(), uuid4()
        backend._record_to_document["a"] = d1
        backend._record_to_document["b"] = d2
        kb.forget.return_value = True

        n = await backend.adelete(record_ids=["a", "b"])
        assert n == 2

    async def test_adelete_record_ids_unknown(self) -> None:
        kb = _make_kb()
        backend = _make_backend(kb)
        n = await backend.adelete(record_ids=["unknown"])
        assert n == 0

    async def test_adelete_by_filter(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        doc = _make_doc(namespace_id=ns, custom={"crewai_scope": "/x"})
        kb.storage.list_documents = AsyncMock(side_effect=[[doc], []])
        kb.forget = AsyncMock(return_value=True)
        n = await backend.adelete(scope_prefix="/x")
        assert n == 1


# ---------------------------------------------------------------------------
# list_scopes / list_categories / count / reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScopeAdminSurface:
    def test_list_scopes_filters_by_parent(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/team/eng"})
        d2 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/team/sales"})
        d3 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/other"})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2, d3], []])

        scopes = backend.list_scopes(parent="/team")
        assert sorted(scopes) == ["/team/eng", "/team/sales"]

    def test_list_categories(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_categories": ["a", "b"]})
        d2 = _make_doc(namespace_id=ns, custom={"crewai_categories": ["a"]})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2], []])

        counts = backend.list_categories()
        assert counts == {"a": 2, "b": 1}

    def test_list_categories_with_scope_prefix(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/t/e", "crewai_categories": ["x"]})
        d2 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/t/s", "crewai_categories": ["y"]})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2], []])

        counts = backend.list_categories(scope_prefix="/t/e")
        assert counts == {"x": 1}

    def test_list_categories_handles_non_list(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_categories": "not-a-list"})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1], []])
        counts = backend.list_categories()
        assert counts == {}

    def test_count_without_scope_uses_stats(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        kb.stats.return_value = _StatsStub(documents=42)

        assert backend.count() == 42

    def test_count_with_scope_walks(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/x"})
        d2 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/y"})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2], []])

        assert backend.count(scope_prefix="/x") == 1

    def test_get_scope_info(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns, custom={"crewai_scope": "/x/y"})
        kb.storage.list_documents = AsyncMock(side_effect=[[d1], []])

        info = backend.get_scope_info("/x")
        assert info == {"scope": "/x", "count": 1}

    def test_reset_delegates_to_delete(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        kb.storage.list_documents = AsyncMock(return_value=[])

        backend.reset()
        kb.storage.list_documents.assert_awaited()


# ---------------------------------------------------------------------------
# list_records edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListRecordsEdges:
    def test_list_records_zero_limit_returns_empty(self) -> None:
        kb = _make_kb()
        backend = _make_backend(kb)
        out = backend.list_records(limit=0)
        assert out == []

    def test_list_records_offset_skips(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)

        d1 = _make_doc(namespace_id=ns)
        d2 = _make_doc(namespace_id=ns)
        d3 = _make_doc(namespace_id=ns)
        c2 = _make_chunk(content="two", document_id=d2.id)
        c3 = _make_chunk(content="three", document_id=d3.id)

        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2, d3], []])

        async def _chunks_for(doc_id: UUID, *, namespace_id: UUID) -> list[Any]:
            if doc_id == d2.id:
                return [c2]
            if doc_id == d3.id:
                return [c3]
            return [_make_chunk(content="one", document_id=d1.id)]

        kb.storage.get_chunks_by_document = AsyncMock(side_effect=_chunks_for)
        out = backend.list_records(limit=10, offset=1)
        # First doc skipped — output starts at doc2
        assert [r.content for r in out] == ["two", "three"]

    def test_list_records_skips_chunkless_docs(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        d1 = _make_doc(namespace_id=ns)
        d2 = _make_doc(namespace_id=ns)
        c1 = _make_chunk(content="one", document_id=d1.id)

        kb.storage.list_documents = AsyncMock(side_effect=[[d1, d2], []])

        async def _chunks(doc_id: UUID, *, namespace_id: UUID) -> list[Any]:
            return [c1] if doc_id == d1.id else []

        kb.storage.get_chunks_by_document = AsyncMock(side_effect=_chunks)
        out = backend.list_records(limit=10)
        assert [r.content for r in out] == ["one"]


# ---------------------------------------------------------------------------
# get_record with cached mapping path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetRecordWithMapping:
    def test_get_record_uses_cached_mapping(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        doc_id = uuid4()
        backend._record_to_document["r-1"] = doc_id
        chunk = _make_chunk(content="hello", document_id=doc_id)
        kb.storage.get_chunks_by_document = AsyncMock(return_value=[chunk])

        record = backend.get_record("r-1")
        assert record is not None
        assert record.content == "hello"
        # external_id lookup was NOT used
        if hasattr(kb.storage, "get_document_by_external_id"):
            kb.storage.get_document_by_external_id.assert_not_called()

    def test_get_record_returns_none_when_no_chunks(self) -> None:
        kb = _make_kb()
        ns = uuid4()
        backend = _make_backend(kb, namespace_id=ns)
        doc_id = uuid4()
        backend._record_to_document["r-1"] = doc_id
        kb.storage.get_chunks_by_document = AsyncMock(return_value=[])

        assert backend.get_record("r-1") is None
