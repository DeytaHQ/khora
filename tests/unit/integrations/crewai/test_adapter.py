"""Unit tests for ``KhoraStorageBackend`` (the 6-method Protocol surface).

Each ``StorageBackend`` method exercised against an ``AsyncMock(spec=Khora)``
to keep the test isolated from any storage backend, async runtime, or
the real ``crewai`` package. CrewAI's ``MemoryRecord`` is duck-typed
with a tiny stand-in here — the adapter never does isinstance checks
on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.exceptions import KhoraIntegrationError
from khora.integrations.crewai._mapping import (
    record_to_remember_kwargs,
    session_id_from_scope,
)
from khora.integrations.crewai.storage import (
    KhoraStorageBackend,
    _raise_invalid_user_id,
    _stash_query_text,
)
from khora.khora import Khora

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeMemoryRecord:
    """Duck-typed stand-in for ``crewai.memory.types.MemoryRecord``."""

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


def _make_kb() -> Any:
    """Return an ``AsyncMock(spec=Khora)`` with a stub storage attr."""
    kb = AsyncMock(spec=Khora)
    # The ``storage`` attribute is a synchronous property on Khora.
    # AsyncMock(spec=...) only mocks coroutine methods; we attach a
    # MagicMock manually so attribute access returns a stub coordinator.
    from unittest.mock import MagicMock

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


def _make_chunk(
    *,
    content: str,
    document_id: UUID | None = None,
    custom: dict[str, Any] | None = None,
) -> Any:
    from khora.core.models.document import Chunk, ChunkMetadata

    md = ChunkMetadata(document_id=document_id or uuid4(), custom=dict(custom or {}))
    return Chunk(content=content, document_id=md.document_id, metadata=md)


# ---------------------------------------------------------------------------
# user_id validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "default", "u", "short"])
def test_user_id_validation_rejects_bad_values(bad: str) -> None:
    with pytest.raises(KhoraIntegrationError):
        _raise_invalid_user_id(bad)


def test_user_id_validation_accepts_long_opaque_id() -> None:
    _raise_invalid_user_id("user-1234567890")


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_session_id_from_scope_pulls_trailing_uuid() -> None:
    sid = uuid4()
    assert session_id_from_scope(f"/crew/research/{sid}") == sid


def test_session_id_from_scope_returns_none_for_semantic_tail() -> None:
    assert session_id_from_scope("/crew/research/ai") is None
    assert session_id_from_scope("") is None
    assert session_id_from_scope("/") is None


def test_record_to_remember_kwargs_carries_crewai_metadata() -> None:
    record = _FakeMemoryRecord(
        id="r-1",
        content="hello",
        scope="/crew/eng",
        categories=["onboarding"],
        metadata={"author": "alice"},
        importance=0.9,
        source="alice@example.com",
        private=True,
    )
    out = record_to_remember_kwargs(record, user_id="user-12345678", app_id="crewai")

    assert out["content"] == "hello"
    assert out["external_id"] == "r-1"
    assert out["entity_types"] == []
    assert out["relationship_types"] == []
    md = out["metadata"]
    assert md["author"] == "alice"  # user metadata preserved
    assert md["crewai_scope"] == "/crew/eng"
    assert md["crewai_categories"] == ["onboarding"]
    assert md["crewai_importance"] == 0.9
    assert md["crewai_private"] is True
    assert md["crewai_user_id"] == "user-12345678"


def test_record_to_remember_kwargs_extracts_session_id_from_scope() -> None:
    sid = uuid4()
    record = _FakeMemoryRecord(id="r-2", content="x", scope=f"/crew/{sid}")
    out = record_to_remember_kwargs(record, user_id="user-12345678", app_id="crewai")
    assert out["session_id"] == sid


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_forwards_to_kb_remember_with_correct_namespace() -> None:
    kb = _make_kb()
    ns = uuid4()
    doc_id = uuid4()
    kb.remember.return_value = _RememberResultStub(document_id=doc_id)

    backend = _make_backend(kb, namespace_id=ns)
    record = _FakeMemoryRecord(id="r-1", content="hello")
    backend.save([record])

    kb.remember.assert_awaited_once()
    _, kwargs = kb.remember.call_args
    assert kwargs["namespace"] == ns
    assert kwargs["content"] == "hello"
    assert kwargs["external_id"] == "r-1"
    # Adapter never asks khora to extract entities — that would be a
    # second LLM call on top of CrewAI's own analysis.
    assert kwargs["entity_types"] == []
    assert kwargs["relationship_types"] == []


def test_save_records_record_id_to_document_id_mapping() -> None:
    kb = _make_kb()
    doc_id = uuid4()
    kb.remember.return_value = _RememberResultStub(document_id=doc_id)
    backend = _make_backend(kb)
    backend.save([_FakeMemoryRecord(id="r-7", content="x")])
    assert backend._record_to_document["r-7"] == doc_id


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_discards_pre_computed_embedding_and_uses_stashed_text() -> None:
    kb = _make_kb()
    ns = uuid4()
    chunk = _make_chunk(content="found it", custom={"crewai_scope": "/"})
    kb.recall.return_value = _RecallResultStub(chunks=[(chunk, 0.9)])

    backend = _make_backend(kb, namespace_id=ns)
    _stash_query_text("what did we decide?")

    out = backend.search([0.1, 0.2, 0.3], limit=5)

    kb.recall.assert_awaited_once()
    _, kwargs = kb.recall.call_args
    # The text query — not the embedding — drives the recall.
    assert kwargs["namespace"] == ns
    args, _ = kb.recall.call_args
    assert args[0] == "what did we decide?"
    assert kwargs["limit"] == 5

    assert len(out) == 1
    record, score = out[0]
    assert record.content == "found it"
    assert score == pytest.approx(0.9)


def test_search_post_filters_by_scope_prefix() -> None:
    kb = _make_kb()
    in_scope = _make_chunk(content="in", custom={"crewai_scope": "/team/eng"})
    out_scope = _make_chunk(content="out", custom={"crewai_scope": "/team/sales"})
    kb.recall.return_value = _RecallResultStub(chunks=[(in_scope, 0.9), (out_scope, 0.8)])
    backend = _make_backend(kb)
    _stash_query_text("eng")

    out = backend.search([0.0], scope_prefix="/team/eng", limit=10)

    assert len(out) == 1
    assert out[0][0].content == "in"


def test_search_post_filters_by_categories() -> None:
    kb = _make_kb()
    a = _make_chunk(content="a", custom={"crewai_categories": ["onboarding"]})
    b = _make_chunk(content="b", custom={"crewai_categories": ["billing"]})
    kb.recall.return_value = _RecallResultStub(chunks=[(a, 0.9), (b, 0.7)])
    backend = _make_backend(kb)
    _stash_query_text("q")

    out = backend.search([0.0], categories=["onboarding"], limit=10)

    assert [r.content for r, _ in out] == ["a"]


def test_search_returns_tuples_of_record_and_score_in_crewai_shape() -> None:
    kb = _make_kb()
    chunk = _make_chunk(content="x")
    kb.recall.return_value = _RecallResultStub(chunks=[(chunk, 0.42)])
    backend = _make_backend(kb)
    _stash_query_text("q")

    out = backend.search([0.0])

    assert isinstance(out, list)
    assert isinstance(out[0], tuple)
    assert isinstance(out[0][0], _FakeMemoryRecord)
    assert isinstance(out[0][1], float)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_by_record_ids_forgets_each_mapped_document() -> None:
    kb = _make_kb()
    ns = uuid4()
    backend = _make_backend(kb, namespace_id=ns)

    d1, d2 = uuid4(), uuid4()
    backend._record_to_document["r-a"] = d1
    backend._record_to_document["r-b"] = d2
    kb.forget.return_value = True

    n = backend.delete(record_ids=["r-a", "r-b"])

    assert n == 2
    assert kb.forget.await_count == 2
    # All forgets target our namespace.
    for call in kb.forget.await_args_list:
        _, kwargs = call
        assert kwargs["namespace"] == ns


def test_delete_skips_unknown_record_ids() -> None:
    kb = _make_kb()
    backend = _make_backend(kb)
    n = backend.delete(record_ids=["nope"])
    assert n == 0
    kb.forget.assert_not_awaited()


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_forgets_then_resaves_under_same_record_id() -> None:
    kb = _make_kb()
    ns = uuid4()
    backend = _make_backend(kb, namespace_id=ns)
    old_doc = uuid4()
    new_doc = uuid4()
    backend._record_to_document["r-1"] = old_doc
    kb.forget.return_value = True
    kb.remember.return_value = _RememberResultStub(document_id=new_doc)

    record = _FakeMemoryRecord(id="r-1", content="new")
    backend.update(record)

    # forget on the old document_id, then a fresh remember for the new content.
    kb.forget.assert_awaited_once()
    _, fkw = kb.forget.call_args
    assert fkw["namespace"] == ns
    kb.remember.assert_awaited_once()
    assert backend._record_to_document["r-1"] == new_doc


# ---------------------------------------------------------------------------
# get_record()
# ---------------------------------------------------------------------------


def test_get_record_returns_none_when_external_id_missing() -> None:
    kb = _make_kb()
    backend = _make_backend(kb)
    kb.storage.get_document_by_external_id = AsyncMock(return_value=None)

    assert backend.get_record("missing") is None


def test_get_record_returns_memory_record_for_known_id() -> None:
    kb = _make_kb()
    ns = uuid4()
    backend = _make_backend(kb, namespace_id=ns)

    doc_id = uuid4()
    chunk = _make_chunk(
        content="hello",
        document_id=doc_id,
        custom={"crewai_scope": "/team", "crewai_categories": ["x"]},
    )
    # First path: backend has no cached mapping — falls back to external_id lookup.
    from khora.core.models.document import Document

    kb.storage.get_document_by_external_id = AsyncMock(
        return_value=Document(id=doc_id, namespace_id=ns, external_id="r-1")
    )
    kb.storage.get_chunks_by_document = AsyncMock(return_value=[chunk])

    record = backend.get_record("r-1")

    assert record is not None
    assert record.content == "hello"
    assert record.scope == "/team"
    assert record.categories == ["x"]
    # Cache populated for next call.
    assert backend._record_to_document["r-1"] == doc_id


# ---------------------------------------------------------------------------
# list_records()
# ---------------------------------------------------------------------------


def test_list_records_walks_documents_and_returns_first_chunk_each() -> None:
    kb = _make_kb()
    ns = uuid4()
    backend = _make_backend(kb, namespace_id=ns)

    from khora.core.models.document import Document

    d1_id, d2_id = uuid4(), uuid4()
    docs = [
        Document(id=d1_id, namespace_id=ns, external_id="r-1"),
        Document(id=d2_id, namespace_id=ns, external_id="r-2"),
    ]
    c1 = _make_chunk(content="one", document_id=d1_id)
    c2 = _make_chunk(content="two", document_id=d2_id)
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    async def _chunks_for(doc_id: UUID, *, namespace_id: UUID) -> list[Any]:
        return [c1] if doc_id == d1_id else [c2]

    kb.storage.get_chunks_by_document = AsyncMock(side_effect=_chunks_for)

    out = backend.list_records(limit=10)
    assert [r.content for r in out] == ["one", "two"]


def test_list_records_honours_scope_prefix_filter() -> None:
    kb = _make_kb()
    ns = uuid4()
    backend = _make_backend(kb, namespace_id=ns)

    from khora.core.models.document import Document, DocumentMetadata

    d1_id, d2_id = uuid4(), uuid4()
    docs = [
        Document(
            id=d1_id,
            namespace_id=ns,
            metadata=DocumentMetadata(custom={"crewai_scope": "/team/eng"}),
        ),
        Document(
            id=d2_id,
            namespace_id=ns,
            metadata=DocumentMetadata(custom={"crewai_scope": "/team/sales"}),
        ),
    ]
    c1 = _make_chunk(content="eng-doc", document_id=d1_id, custom={"crewai_scope": "/team/eng"})
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])
    kb.storage.get_chunks_by_document = AsyncMock(return_value=[c1])

    out = backend.list_records(scope_prefix="/team/eng", limit=10)
    assert [r.content for r in out] == ["eng-doc"]


# ---------------------------------------------------------------------------
# Result stubs
# ---------------------------------------------------------------------------


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
