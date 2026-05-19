"""LangGraph ``BaseStore`` conformance suite for ``KhoraStore``.

LangGraph 1.x does NOT ship a public ``test_base_store`` mixin we can
plug in directly (we checked — ``langgraph.store.base`` only ships
``BaseStore`` and ``InMemoryStore``, no test harness). So this file
hand-rolls the equivalent: exercise every contract the ``InMemoryStore``
honours and assert ``KhoraStore`` matches.

The conformance tests run against ``AsyncMock(spec=Khora)`` plus an in-
memory shadow store so they don't require infrastructure. End-to-end
behaviour against a real khora is covered by
``tests/integration/integrations/langgraph/test_e2e.py``.

When/if LangGraph publishes a ``test_base_store`` mixin we should swap
this file's manual asserts for inheritance. Until then this file is the
contract we're explicitly conforming to.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora import Khora
from khora.core.models.document import Document, DocumentStatus


class _ShadowStore:
    """Tiny in-memory store the ``Khora`` mock delegates to.

    The conformance suite needs round-trip semantics (put then get
    returns the value, delete then get returns None, ...). A flat
    ``AsyncMock`` return-value setup doesn't model state, so we keep a
    real dict and have the mock's side_effect read from it.
    """

    def __init__(self, namespace_id: UUID) -> None:
        self.namespace_id = namespace_id
        self.by_external_id: dict[str, Document] = {}

    async def get_document_by_external_id(self, namespace_id: UUID, external_id: str | None) -> Document | None:
        if external_id is None:
            return None
        return self.by_external_id.get(external_id)

    async def remember(self, content: str, **kwargs: Any) -> Any:
        external_id = kwargs.get("external_id")
        metadata = kwargs.get("metadata") or {}
        custom = dict(metadata)
        doc = Document(
            id=uuid4(),
            namespace_id=self.namespace_id,
            content=content,
            external_id=external_id,
            title=kwargs.get("title", ""),
            source=kwargs.get("source", ""),
            metadata=custom,
            status=DocumentStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        if external_id is not None:
            self.by_external_id[external_id] = doc
        return MagicMock(document_id=doc.id)

    async def forget(self, document_id: UUID, *, namespace: UUID) -> bool:
        for ext_id, doc in list(self.by_external_id.items()):
            if doc.id == document_id:
                del self.by_external_id[ext_id]
                return True
        return False

    async def list_documents(self, *, namespace: UUID, limit: int = 100) -> list[Document]:
        return list(self.by_external_id.values())[:limit]


@pytest.fixture
def store():
    """Build a ``KhoraStore`` whose Khora is backed by ``_ShadowStore``."""
    from khora.integrations.langgraph import KhoraStore

    kb = AsyncMock(spec=Khora)
    kb._config = MagicMock()
    kb._config.llm = MagicMock()
    kb._config.llm.embedding_dimension = 1536

    ns_uuid = uuid4()  # placeholder; overwritten by KhoraStore's derivation
    shadow = _ShadowStore(ns_uuid)

    kb.storage = MagicMock()
    kb.storage.get_document_by_external_id = AsyncMock(side_effect=shadow.get_document_by_external_id)
    kb.storage.create_namespace = AsyncMock()
    kb._resolve_namespace = AsyncMock(side_effect=lambda nid: nid)
    kb.list_documents = AsyncMock(side_effect=shadow.list_documents)
    kb.remember = AsyncMock(side_effect=shadow.remember)
    kb.forget = AsyncMock(side_effect=shadow.forget)
    kb.recall = AsyncMock(return_value=MagicMock(chunks=[]))

    s = KhoraStore(kb, user_id="conform-1234")
    shadow.namespace_id = s.namespace_id  # align after the fact
    return s


# ----------------------------------------------------------------------
# BaseStore contract (mirrors langgraph.store.memory.InMemoryStore behaviour)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_then_get_returns_same_value(store):
    """Round-trip: put then get returns the same value dict."""
    await store.aput(("ns",), "k1", {"text": "hello"})
    item = await store.aget(("ns",), "k1")
    assert item is not None
    assert item.value == {"text": "hello"}
    assert item.namespace == ("ns",)
    assert item.key == "k1"


@pytest.mark.asyncio
async def test_get_missing_returns_none(store):
    assert await store.aget(("ns",), "does-not-exist") is None


@pytest.mark.asyncio
async def test_delete_then_get_returns_none(store):
    await store.aput(("ns",), "k1", {"text": "hi"})
    await store.adelete(("ns",), "k1")
    assert await store.aget(("ns",), "k1") is None


@pytest.mark.asyncio
async def test_delete_missing_is_noop(store):
    """LangGraph spec: delete on a missing key must not raise."""
    await store.adelete(("ns",), "missing")  # must not raise


@pytest.mark.asyncio
async def test_put_overwrites_existing(store):
    await store.aput(("ns",), "k1", {"text": "v1"})
    await store.aput(("ns",), "k1", {"text": "v2"})
    item = await store.aget(("ns",), "k1")
    assert item is not None
    assert item.value == {"text": "v2"}


@pytest.mark.asyncio
async def test_namespaces_are_isolated(store):
    """Same key under different namespaces must not collide."""
    await store.aput(("a",), "k", {"text": "in-a"})
    await store.aput(("b",), "k", {"text": "in-b"})
    item_a = await store.aget(("a",), "k")
    item_b = await store.aget(("b",), "k")
    assert item_a is not None and item_b is not None
    assert item_a.value == {"text": "in-a"}
    assert item_b.value == {"text": "in-b"}


@pytest.mark.asyncio
async def test_search_no_query_returns_items_under_prefix(store):
    await store.aput(("a", "x"), "k1", {"text": "alpha"})
    await store.aput(("a", "y"), "k2", {"text": "beta"})
    await store.aput(("b",), "k3", {"text": "gamma"})
    results = await store.asearch(("a",))
    keys = sorted(r.key for r in results)
    assert keys == ["k1", "k2"]


@pytest.mark.asyncio
async def test_list_namespaces_returns_distinct(store):
    await store.aput(("a", "x"), "k1", {"text": "."})
    await store.aput(("a", "x"), "k2", {"text": "."})
    await store.aput(("a", "y"), "k3", {"text": "."})
    namespaces = await store.alist_namespaces()
    assert set(namespaces) == {("a", "x"), ("a", "y")}


@pytest.mark.asyncio
async def test_empty_namespace_rejected(store):
    """LangGraph's _validate_namespace rejects an empty tuple."""
    with pytest.raises(Exception):  # InvalidNamespaceError (subclass of ValueError)
        await store.aput((), "k", {"text": "."})


@pytest.mark.asyncio
async def test_namespace_with_period_rejected(store):
    """LangGraph forbids '.' in any namespace segment."""
    with pytest.raises(Exception):
        await store.aput(("a.b",), "k", {"text": "."})


@pytest.mark.asyncio
async def test_langgraph_reserved_root_rejected(store):
    """First segment of "langgraph" is reserved."""
    with pytest.raises(Exception):
        await store.aput(("langgraph", "x"), "k", {"text": "."})
