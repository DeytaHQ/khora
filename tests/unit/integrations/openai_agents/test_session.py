"""Unit tests for ``KhoraSession`` against an ``AsyncMock(spec=Khora)``.

These tests exercise the four ``SessionABC`` methods (``get_items``,
``add_items``, ``pop_item``, ``clear_session``) plus the Protocol-
conformance check. The SDK Session ABC is imported lazily inside
``KhoraSession.__init__``; the test file imports it too — skip the
whole module when the extra isn't installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("agents")  # openai-agents SDK

from agents.memory.session import SessionABC  # noqa: E402

from khora.integrations.openai_agents._mapping import (  # noqa: E402
    KEY_ITEM_JSON,
    KEY_SEQ,
    KEY_SESSION_ID,
)
from khora.integrations.openai_agents.session import KhoraSession  # noqa: E402
from khora.khora import Khora  # noqa: E402

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _RememberResultStub:
    document_id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    chunks_created: int = 1
    entities_extracted: int = 0
    relationships_created: int = 0


def _make_kb() -> Any:
    kb = AsyncMock(spec=Khora)
    kb.storage = MagicMock()
    # remember() is awaited; default return = a fresh RememberResult stub.
    kb.remember.side_effect = lambda *_a, **_kw: _RememberResultStub()
    # KhoraSession resolves the public namespace UUID to a row-level UUID
    # via the public kb.storage.resolve_namespace. Default it to identity
    # so tests don't need to wire two distinct UUIDs.
    kb.storage.resolve_namespace = AsyncMock(side_effect=lambda ns: ns)
    return kb


def _make_session(kb: Any, *, session_id: str = "conv-1") -> KhoraSession:
    return KhoraSession(kb=kb, namespace=uuid4(), session_id=session_id)


def _stored_doc(
    namespace_id: UUID,
    *,
    session_id: str = "conv-1",
    seq: int,
    item: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Any:
    """Build a khora Document stamped as if KhoraSession.add_items wrote it."""
    from khora.core.models.document import Document

    custom: dict[str, Any] = {
        KEY_SESSION_ID: session_id,
        KEY_SEQ: seq,
        KEY_ITEM_JSON: json.dumps(item),
    }
    if extra:
        custom.update(extra)
    return Document(
        id=uuid4(),
        namespace_id=namespace_id,
        external_id=f"oai:{session_id}:{seq}",
        metadata=custom,
    )


# ---------------------------------------------------------------------------
# Construction + Protocol conformance
# ---------------------------------------------------------------------------


def test_session_is_runtime_subclass_of_session_abc() -> None:
    """Acceptance: ``isinstance(session, SessionABC)`` passes.

    Catches SDK rename drift — if upstream renames ``SessionABC``, this
    test breaks at the import site rather than at the next ``Runner.run``.
    """
    session = _make_session(_make_kb())
    assert isinstance(session, SessionABC)


def test_session_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        KhoraSession(kb=_make_kb(), namespace=uuid4(), session_id="")


def test_session_rejects_non_uuid_namespace() -> None:
    with pytest.raises(TypeError, match="namespace"):
        KhoraSession(kb=_make_kb(), namespace="not-a-uuid", session_id="x")  # type: ignore[arg-type]


def test_session_rejects_empty_app_id() -> None:
    with pytest.raises(ValueError, match="app_id"):
        KhoraSession(kb=_make_kb(), namespace=uuid4(), session_id="x", app_id="")


def test_session_id_attr_is_str_per_protocol() -> None:
    """SessionABC requires ``session_id: str``."""
    session = _make_session(_make_kb(), session_id="conv-42")
    assert session.session_id == "conv-42"
    assert isinstance(session.session_id, str)


# ---------------------------------------------------------------------------
# add_items / get_items round-trip
# ---------------------------------------------------------------------------


async def test_add_items_forwards_each_item_to_kb_remember() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    # No existing session — discover_max_seq returns -1, first seq = 0.
    kb.storage.list_documents = AsyncMock(return_value=[])
    kb.storage.get_document_by_external_id = AsyncMock(return_value=None)

    items = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    await session.add_items(items)

    # Two remember() calls, one per item, scoped to the session's namespace.
    assert kb.remember.await_count == 2
    seen_external_ids = []
    for call in kb.remember.await_args_list:
        _, kwargs = call
        assert kwargs["namespace"] == session.namespace_id
        assert kwargs["entity_types"] == []  # no extraction on chat turns
        seen_external_ids.append(kwargs["external_id"])
    # Sequential external ids.
    assert seen_external_ids == ["oai:conv-1:0", "oai:conv-1:1"]


async def test_add_items_no_op_on_empty_list() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    await session.add_items([])
    kb.remember.assert_not_awaited()


async def test_add_items_drops_pre_existing_document_at_same_external_id() -> None:
    """A retry under the same seq must not accumulate duplicate chunks."""
    kb = _make_kb()
    session = _make_session(kb)
    existing_doc = MagicMock()
    existing_doc.id = uuid4()
    kb.storage.list_documents = AsyncMock(return_value=[])
    kb.storage.get_document_by_external_id = AsyncMock(return_value=existing_doc)
    kb.forget = AsyncMock(return_value=True)

    await session.add_items([{"role": "user", "content": "x"}])

    kb.forget.assert_awaited_once()
    fkwargs = kb.forget.await_args.kwargs
    assert fkwargs["namespace"] == session.namespace_id


async def test_get_items_returns_documents_in_seq_order() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    # Write order was 0, 1, 2; list_documents returns them shuffled to
    # prove ordering is by seq, not insertion order.
    items = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    docs = [
        _stored_doc(ns, seq=2, item=items[2]),
        _stored_doc(ns, seq=0, item=items[0]),
        _stored_doc(ns, seq=1, item=items[1]),
    ]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    out = await session.get_items()
    assert out == items


async def test_get_items_honours_limit() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    items = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]
    docs = [_stored_doc(ns, seq=i, item=items[i]) for i in range(5)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    out = await session.get_items(limit=2)
    # Latest two items, chronological order.
    assert out == items[-2:]


async def test_get_items_skips_foreign_documents_in_same_namespace() -> None:
    """Documents missing the session_id stamp must be skipped silently."""
    from khora.core.models.document import Document

    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    own_item = {"role": "user", "content": "mine"}
    own = _stored_doc(ns, seq=0, item=own_item)
    foreign = Document(id=uuid4(), namespace_id=ns, metadata={"foo": "bar"})
    other_session = _stored_doc(ns, seq=0, item={"role": "user", "content": "other"}, session_id="conv-2")
    kb.storage.list_documents = AsyncMock(side_effect=[[foreign, own, other_session], []])

    out = await session.get_items()
    assert out == [own_item]


async def test_get_items_empty_session_returns_empty_list() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    kb.storage.list_documents = AsyncMock(return_value=[])
    out = await session.get_items()
    assert out == []


# ---------------------------------------------------------------------------
# pop_item
# ---------------------------------------------------------------------------


async def test_pop_item_returns_and_deletes_last_item() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    items = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    docs = [_stored_doc(ns, seq=i, item=items[i]) for i in range(2)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])
    kb.forget = AsyncMock(return_value=True)

    popped = await session.pop_item()
    assert popped == items[-1]
    # forget targeted the most-recent document.
    kb.forget.assert_awaited_once()
    fargs, fkwargs = kb.forget.call_args
    assert fargs[0] == docs[1].id
    assert fkwargs["namespace"] == ns


async def test_pop_item_returns_none_for_empty_session() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    kb.storage.list_documents = AsyncMock(return_value=[])
    assert await session.pop_item() is None


async def test_pop_item_forces_seq_rescan_on_next_add() -> None:
    """After a successful pop, the next add must scan storage to find the new max."""
    kb = _make_kb()
    session = _make_session(kb)
    session._next_seq = 5  # pretend we already discovered a max.

    doc = _stored_doc(session.namespace_id, seq=4, item={"role": "user", "content": "x"})
    kb.storage.list_documents = AsyncMock(side_effect=[[doc], []])
    kb.forget = AsyncMock(return_value=True)

    await session.pop_item()
    assert session._next_seq is None  # forced re-scan on the next add_items


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------


async def test_clear_session_forgets_every_document_once() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    items = [{"role": "user", "content": f"m-{i}"} for i in range(3)]
    docs = [_stored_doc(ns, seq=i, item=items[i]) for i in range(3)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])
    kb.forget = AsyncMock(return_value=True)

    await session.clear_session()
    assert kb.forget.await_count == 3
    assert session._next_seq == 0  # reset for next add
