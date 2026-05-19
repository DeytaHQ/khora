"""Unit tests for ``KhoraStore`` (#624).

Tests against ``AsyncMock(spec=Khora)`` so they run without infrastructure.
Exercises the 6 async core methods (``aput`` / ``aget`` / ``asearch`` /
``adelete`` / ``alist_namespaces`` / ``abatch``), namespace tuple
flattening, the user_id disaster-mode guards, and the ``IndexConfig``
dim mismatch fail-fast.

Integration coverage (real khora + sqlite_lance + 1-node LangGraph
graph) lives in ``tests/integration/integrations/langgraph/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora import Khora
from khora.core.models.document import (
    Document,
    DocumentStatus,
)


def _mk_kb(embedding_dimension: int = 1536) -> Khora:
    """Build an ``AsyncMock(spec=Khora)`` with the bits the store reaches for."""
    kb = AsyncMock(spec=Khora)
    kb._config = MagicMock()
    kb._config.llm = MagicMock()
    kb._config.llm.embedding_dimension = embedding_dimension
    kb.storage = MagicMock()
    kb.storage.get_document_by_external_id = AsyncMock(return_value=None)
    kb.storage.create_namespace = AsyncMock()
    kb._resolve_namespace = AsyncMock(side_effect=lambda nid: nid)
    kb.list_documents = AsyncMock(return_value=[])
    kb.remember = AsyncMock()
    kb.forget = AsyncMock()
    kb.recall = AsyncMock()
    return kb


def _mk_document(
    namespace_id: UUID,
    *,
    lg_namespace: tuple[str, ...],
    lg_key: str,
    lg_value: dict[str, Any],
    external_id: str | None = None,
    doc_id: UUID | None = None,
) -> Document:
    """Build a minimal ``Document`` projecting through ``item_from_metadata``."""
    return Document(
        id=doc_id or uuid4(),
        namespace_id=namespace_id,
        content=str(lg_value.get("text", "stub")),
        external_id=external_id,
        title=lg_key,
        source="langgraph:test",
        metadata={
            "lg_namespace": list(lg_namespace),
            "lg_namespace_flat": "/".join(lg_namespace),
            "lg_key": lg_key,
            "lg_value": lg_value,
        },
        status=DocumentStatus.COMPLETED,
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
        updated_at=datetime(2026, 5, 15, tzinfo=UTC),
    )


# ----------------------------------------------------------------------
# Construction + validation
# ----------------------------------------------------------------------


def test_construct_rejects_empty_user_id():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    with pytest.raises(ValueError, match="disallowed list"):
        KhoraStore(kb, user_id="")


def test_construct_rejects_default_user_id():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    with pytest.raises(ValueError, match="disallowed list"):
        KhoraStore(kb, user_id="default")


def test_construct_rejects_short_user_id():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    with pytest.raises(ValueError, match="shorter than"):
        KhoraStore(kb, user_id="abc")


def test_construct_rejects_whitespace_user_id():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    with pytest.raises(ValueError, match="whitespace"):
        KhoraStore(kb, user_id=" alice-1234 ")


def test_construct_accepts_valid_user_id_and_derives_namespace_id():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    assert isinstance(store.namespace_id, UUID)
    # Deterministic: same args produce same UUID.
    store2 = KhoraStore(kb, user_id="alice-1234")
    assert store.namespace_id == store2.namespace_id
    # Different user → different UUID.
    store3 = KhoraStore(kb, user_id="bob-9999-2222")
    assert store.namespace_id != store3.namespace_id


def test_construct_rejects_index_config_dim_mismatch():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb(embedding_dimension=1536)
    with pytest.raises(ValueError, match="does not match khora"):
        KhoraStore(kb, user_id="alice-1234", index_config={"dims": 768})


def test_construct_accepts_matching_index_config_dim():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb(embedding_dimension=1536)
    store = KhoraStore(kb, user_id="alice-1234", index_config={"dims": 1536})
    assert store.namespace_id is not None


def test_construct_rejects_multichar_separator():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    with pytest.raises(ValueError, match="single character"):
        KhoraStore(kb, user_id="alice-1234", namespace_sep="::")


def test_construct_implements_base_store():
    """KhoraStore satisfies ``isinstance(_, BaseStore)`` after construction."""
    from langgraph.store.base import BaseStore

    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    assert isinstance(store, BaseStore)


# ----------------------------------------------------------------------
# aput
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aput_flattens_namespace_and_calls_remember():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    await store.aput(("memories", "facts"), "k1", {"text": "the sky is blue"})

    kb.remember.assert_awaited_once()
    _, kwargs = kb.remember.await_args
    assert kwargs["external_id"] == "memories/facts::k1"
    assert kwargs["namespace"] == store.namespace_id
    assert kwargs["metadata"]["lg_namespace"] == ["memories", "facts"]
    assert kwargs["metadata"]["lg_key"] == "k1"
    assert kwargs["metadata"]["lg_value"] == {"text": "the sky is blue"}


@pytest.mark.asyncio
async def test_aput_creates_namespace_when_missing():
    """When ``_resolve_namespace`` raises, the store creates the row."""
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    kb._resolve_namespace = AsyncMock(side_effect=ValueError("no namespace"))
    store = KhoraStore(kb, user_id="alice-1234")
    await store.aput(("ns",), "k1", {"text": "hi"})
    kb.storage.create_namespace.assert_awaited_once()


@pytest.mark.asyncio
async def test_aput_rejects_separator_in_segment():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    with pytest.raises(ValueError, match="contains the configured separator"):
        await store.aput(("memories/private",), "k1", {"text": "hi"})


@pytest.mark.asyncio
async def test_aput_overwrite_deletes_existing_first():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    existing = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k1",
        lg_value={"text": "old"},
        external_id="ns::k1",
    )
    kb.storage.get_document_by_external_id.return_value = existing
    await store.aput(("ns",), "k1", {"text": "new"})
    kb.forget.assert_awaited_once_with(existing.id, namespace=store.namespace_id)
    kb.remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_aput_ttl_warning_emitted_once():
    import warnings

    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    with pytest.warns(RuntimeWarning, match="TTL"):
        await store.aput(("ns",), "k1", {"text": "x"}, ttl=60.0)
    # Second call: no warning (already emitted).
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        await store.aput(("ns",), "k2", {"text": "y"}, ttl=60.0)
    ttl_warnings = [w for w in record if "TTL" in str(w.message)]
    assert ttl_warnings == []


@pytest.mark.asyncio
async def test_aput_index_false_warning_emitted_once():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    with pytest.warns(RuntimeWarning, match="index=False"):
        await store.aput(("ns",), "k1", {"text": "x"}, index=False)


@pytest.mark.asyncio
async def test_aput_rejects_non_dict_value():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    with pytest.raises(TypeError, match="value must be a dict"):
        await store.aput(("ns",), "k1", "not a dict")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# aget
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_returns_item_when_document_exists():
    from langgraph.store.base import Item

    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    doc = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k1",
        lg_value={"text": "hi"},
    )
    kb.storage.get_document_by_external_id.return_value = doc

    result = await store.aget(("ns",), "k1")
    assert isinstance(result, Item)
    assert result.key == "k1"
    assert result.namespace == ("ns",)
    assert result.value == {"text": "hi"}


@pytest.mark.asyncio
async def test_aget_returns_none_when_missing():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    kb.storage.get_document_by_external_id.return_value = None
    assert await store.aget(("ns",), "k1") is None


@pytest.mark.asyncio
async def test_aget_returns_none_on_foreign_document():
    """A document with no lg_ metadata is treated as foreign (not ours)."""
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    foreign = Document(
        namespace_id=store.namespace_id,
        content="hi",
        metadata={"other": "system"},
    )
    kb.storage.get_document_by_external_id.return_value = foreign
    assert await store.aget(("ns",), "k1") is None


# ----------------------------------------------------------------------
# asearch
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_with_query_maps_chunks_to_search_items():
    from langgraph.store.base import SearchItem

    from khora.core.models.document import Chunk
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")

    # Construct two chunks belonging to two different LangGraph items.
    c1 = Chunk(
        namespace_id=store.namespace_id,
        content="alpha",
        metadata={
            "lg_namespace": ["memories", "facts"],
            "lg_key": "k1",
            "lg_value": {"text": "alpha"},
        },
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
    )
    c2 = Chunk(
        namespace_id=store.namespace_id,
        content="beta",
        metadata={
            "lg_namespace": ["memories", "other"],
            "lg_key": "k2",
            "lg_value": {"text": "beta"},
        },
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
    )
    recall_result = MagicMock()
    recall_result.chunks = [(c1, 0.9), (c2, 0.5)]
    kb.recall.return_value = recall_result

    results = await store.asearch(("memories", "facts"), query="alpha", limit=10)
    assert len(results) == 1
    assert isinstance(results[0], SearchItem)
    assert results[0].namespace == ("memories", "facts")
    assert results[0].key == "k1"
    assert results[0].score == 0.9


@pytest.mark.asyncio
async def test_asearch_no_query_lists_documents():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    doc = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k1",
        lg_value={"text": "hi", "tag": "a"},
    )
    kb.list_documents.return_value = [doc]
    results = await store.asearch(("ns",))
    assert len(results) == 1
    assert results[0].key == "k1"
    assert results[0].score is None


@pytest.mark.asyncio
async def test_asearch_applies_client_side_filter():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    doc_a = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k1",
        lg_value={"text": "hi", "tag": "a"},
    )
    doc_b = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k2",
        lg_value={"text": "hi", "tag": "b"},
    )
    kb.list_documents.return_value = [doc_a, doc_b]
    results = await store.asearch(("ns",), filter={"tag": "a"})
    assert [r.key for r in results] == ["k1"]


@pytest.mark.asyncio
async def test_asearch_honours_limit_and_offset():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    docs = [
        _mk_document(
            store.namespace_id,
            lg_namespace=("ns",),
            lg_key=f"k{i}",
            lg_value={"text": f"v{i}"},
        )
        for i in range(5)
    ]
    kb.list_documents.return_value = docs
    page = await store.asearch(("ns",), limit=2, offset=1)
    assert [r.key for r in page] == ["k1", "k2"]


# ----------------------------------------------------------------------
# adelete
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adelete_calls_forget_when_document_exists():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    doc = _mk_document(
        store.namespace_id,
        lg_namespace=("ns",),
        lg_key="k1",
        lg_value={"text": "hi"},
    )
    kb.storage.get_document_by_external_id.return_value = doc
    await store.adelete(("ns",), "k1")
    kb.forget.assert_awaited_once_with(doc.id, namespace=store.namespace_id)


@pytest.mark.asyncio
async def test_adelete_missing_is_noop():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    kb.storage.get_document_by_external_id.return_value = None
    await store.adelete(("ns",), "missing")
    kb.forget.assert_not_called()


# ----------------------------------------------------------------------
# alist_namespaces
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alist_namespaces_aggregates_distinct_tuples():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    kb.list_documents.return_value = [
        _mk_document(store.namespace_id, lg_namespace=("a", "b"), lg_key="k1", lg_value={}),
        _mk_document(store.namespace_id, lg_namespace=("a", "b"), lg_key="k2", lg_value={}),
        _mk_document(store.namespace_id, lg_namespace=("a", "c"), lg_key="k3", lg_value={}),
    ]
    result = await store.alist_namespaces()
    assert result == [("a", "b"), ("a", "c")]


@pytest.mark.asyncio
async def test_alist_namespaces_prefix_filter():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    kb.list_documents.return_value = [
        _mk_document(store.namespace_id, lg_namespace=("a", "b"), lg_key="k", lg_value={}),
        _mk_document(store.namespace_id, lg_namespace=("x", "y"), lg_key="k", lg_value={}),
    ]
    result = await store.alist_namespaces(prefix=("a",))
    assert result == [("a", "b")]


@pytest.mark.asyncio
async def test_alist_namespaces_max_depth_truncates():
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    kb.list_documents.return_value = [
        _mk_document(store.namespace_id, lg_namespace=("a", "b", "c"), lg_key="k1", lg_value={}),
        _mk_document(store.namespace_id, lg_namespace=("a", "b", "d"), lg_key="k2", lg_value={}),
    ]
    result = await store.alist_namespaces(max_depth=2)
    assert result == [("a", "b")]


@pytest.mark.asyncio
async def test_alist_namespaces_in_memory_set_is_included():
    """Just-written tuples should surface even without a DB round-trip."""
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    await store.aput(("freshly", "written"), "k1", {"text": "x"})
    # No documents in list_documents — only the in-memory set.
    result = await store.alist_namespaces()
    assert ("freshly", "written") in result


# ----------------------------------------------------------------------
# abatch
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abatch_dispatches_each_op_type():
    from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")

    # Pre-populate get and search returns.
    doc = _mk_document(store.namespace_id, lg_namespace=("ns",), lg_key="k1", lg_value={"text": "hi"})
    kb.storage.get_document_by_external_id.return_value = doc
    kb.list_documents.return_value = [doc]

    ops = [
        PutOp(("ns",), "k1", {"text": "hi"}),
        GetOp(("ns",), "k1"),
        PutOp(("ns",), "k1", None),  # delete
        SearchOp(("ns",), None, 10, 0, None),
        ListNamespacesOp(None, None, 100, 0),
    ]
    results = await store.abatch(ops)
    assert results[0] is None  # put
    assert results[1] is not None and results[1].key == "k1"  # get
    assert results[2] is None  # delete
    assert isinstance(results[3], list)  # search
    assert isinstance(results[4], list)  # list_namespaces


# ----------------------------------------------------------------------
# Sync surface
# ----------------------------------------------------------------------


def test_sync_put_get_delete_route_through_run_sync():
    """Sync methods bridge to the async ones via ``run_sync``."""
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")

    store.put(("ns",), "k1", {"text": "hi"})
    kb.remember.assert_called()

    doc = _mk_document(store.namespace_id, lg_namespace=("ns",), lg_key="k1", lg_value={"text": "hi"})
    kb.storage.get_document_by_external_id.return_value = doc
    item = store.get(("ns",), "k1")
    assert item is not None
    assert item.key == "k1"

    store.delete(("ns",), "k1")
    kb.forget.assert_called()


@pytest.mark.asyncio
async def test_sync_method_works_inside_running_loop():
    """``run_sync`` dispatches to a separate daemon-thread loop, so it is
    safe to call from inside an async context. This pins that contract —
    LangGraph's sync ``BaseStore`` abstracts are exercised by callers
    that may themselves run inside an asyncio loop (test runners, graph
    compile-time hooks).
    """
    from khora.integrations.langgraph import KhoraStore

    kb = _mk_kb()
    store = KhoraStore(kb, user_id="alice-1234")
    # Should not raise — dispatches to the bridge's daemon loop.
    store.put(("ns",), "k1", {"text": "hi"})
    kb.remember.assert_awaited()
