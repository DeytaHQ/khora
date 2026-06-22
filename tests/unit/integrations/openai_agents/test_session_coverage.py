"""Extra coverage tests for ``khora.integrations.openai_agents.session``.

The mainline test_session.py covers the happy paths. This file targets
the remaining branches: ``_decode_item_from_doc`` corruption paths,
``_load_session_documents`` paging / missing seq, ``clear_session``
error swallowing, ``get_items`` with a corrupt doc, and pagination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("agents")

from khora.integrations.openai_agents._mapping import (  # noqa: E402
    KEY_ITEM_JSON,
    KEY_SEQ,
    KEY_SESSION_ID,
)
from khora.integrations.openai_agents.session import (  # noqa: E402
    KhoraSession,
    _decode_item_from_doc,
)
from khora.khora import Khora  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
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
    kb.remember.side_effect = lambda *_a, **_kw: _RememberResultStub()
    kb.storage.resolve_namespace = AsyncMock(side_effect=lambda ns: ns)
    return kb


def _make_session(kb: Any, *, session_id: str = "conv-1") -> KhoraSession:
    return KhoraSession(kb=kb, namespace=uuid4(), session_id=session_id)


def _doc(
    ns: UUID,
    *,
    sid: str = "conv-1",
    seq: Any = 0,
    item: dict[str, Any] | None = None,
    item_raw: Any = "AUTO",
    extra: dict[str, Any] | None = None,
) -> Any:
    """Build a Document-shaped row. ``item_raw`` overrides the JSON column."""
    from khora.core.models.document import Document

    if item is None and item_raw == "AUTO":
        item = {"role": "user", "content": "x"}

    if item_raw == "AUTO":
        custom: dict[str, Any] = {
            KEY_SESSION_ID: sid,
            KEY_SEQ: seq,
            KEY_ITEM_JSON: json.dumps(item),
        }
    else:
        custom = {
            KEY_SESSION_ID: sid,
            KEY_SEQ: seq,
            KEY_ITEM_JSON: item_raw,
        }
    if extra:
        custom.update(extra)
    return Document(id=uuid4(), namespace_id=ns, metadata=custom)


# ---------------------------------------------------------------------------
# _decode_item_from_doc
# ---------------------------------------------------------------------------


def test_decode_returns_none_when_metadata_has_no_oai_item() -> None:
    doc = MagicMock()
    doc.metadata = {"some_other_key": "value"}
    assert _decode_item_from_doc(doc) is None


def test_decode_returns_none_when_metadata_is_none() -> None:
    doc = MagicMock()
    doc.metadata = None
    assert _decode_item_from_doc(doc) is None


def test_decode_returns_none_for_corrupt_json_string() -> None:
    doc = MagicMock()
    doc.metadata = {KEY_ITEM_JSON: "not json {"}
    assert _decode_item_from_doc(doc) is None


def test_decode_passes_through_already_decoded_value() -> None:
    """Some backends surface JSONB as a dict directly — return as-is."""
    payload = {"role": "user", "content": "hi"}
    doc = MagicMock()
    doc.metadata = {KEY_ITEM_JSON: payload}
    assert _decode_item_from_doc(doc) == payload


# ---------------------------------------------------------------------------
# get_items — corrupt doc is skipped, not raised
# ---------------------------------------------------------------------------


async def test_get_items_skips_documents_with_corrupt_json() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    good = _doc(ns, seq=0, item={"role": "user", "content": "yes"})
    bad = _doc(ns, seq=1, item_raw="not json {")
    kb.storage.list_documents = AsyncMock(side_effect=[[good, bad], []])

    out = await session.get_items()
    assert out == [{"role": "user", "content": "yes"}]


async def test_get_items_limit_zero_returns_all_items_due_to_negative_slice() -> None:
    """``limit=0`` falls into ``items[-0:]`` which is the full list — documented quirk."""
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id
    items_payload = [{"role": "user", "content": f"m-{i}"} for i in range(3)]
    docs = [_doc(ns, seq=i, item=items_payload[i]) for i in range(3)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    out = await session.get_items(limit=0)
    # Python: items[-0:] == items[0:] == full list. This documents the current behaviour.
    assert out == items_payload


async def test_get_items_negative_limit_skipped_passthrough() -> None:
    """``limit<0`` falls through the if-guard — entire list returned."""
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id
    items = [{"role": "user", "content": f"m-{i}"} for i in range(3)]
    docs = [_doc(ns, seq=i, item=items[i]) for i in range(3)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    out = await session.get_items(limit=-1)
    assert out == items


# ---------------------------------------------------------------------------
# _load_session_documents — paging + seq parsing branches
# ---------------------------------------------------------------------------


async def test_load_session_documents_paginates_through_full_pages() -> None:
    """A full page must trigger another fetch with a higher offset."""
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    page_size = 200
    # First page = exactly page_size full → must request another page.
    full_page = [_doc(ns, seq=i, item={"role": "user", "content": f"m-{i}"}) for i in range(page_size)]
    short_page = [_doc(ns, seq=page_size, item={"role": "user", "content": "tail"})]
    empty_page: list[Any] = []
    kb.storage.list_documents = AsyncMock(side_effect=[full_page, short_page, empty_page])

    out = await session.get_items()
    assert len(out) == page_size + 1
    # ``list_documents`` was called with cursor=0 then cursor=200.
    calls = kb.storage.list_documents.call_args_list
    assert len(calls) >= 2
    _, kw0 = calls[0]
    assert kw0.get("offset", 0) == 0
    _, kw1 = calls[1]
    assert kw1.get("offset") == page_size


async def test_load_session_documents_handles_string_seq_value() -> None:
    """A digit-string seq must parse as int and sort numerically."""
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    # Three docs with seq values: int, digit-string, non-digit-string.
    d0 = _doc(ns, seq=0, item={"role": "user", "content": "first"})
    d1 = _doc(ns, seq="5", item={"role": "user", "content": "second"})  # digit-string
    d_bogus = _doc(ns, seq="nope", item={"role": "user", "content": "tail"})  # non-digit-string
    kb.storage.list_documents = AsyncMock(side_effect=[[d_bogus, d0, d1], []])

    out = await session.get_items()
    # int-seq 0, then digit-string-seq "5" → 5, then non-digit → 1<<31 → last.
    assert [item["content"] for item in out] == ["first", "second", "tail"]


async def test_load_session_documents_skips_documents_with_wrong_session_id() -> None:
    kb = _make_kb()
    session = _make_session(kb, session_id="conv-A")
    ns = session.namespace_id
    own = _doc(ns, sid="conv-A", seq=0, item={"role": "user", "content": "mine"})
    other = _doc(ns, sid="conv-B", seq=0, item={"role": "user", "content": "other"})
    kb.storage.list_documents = AsyncMock(side_effect=[[other, own], []])
    out = await session.get_items()
    assert out == [{"role": "user", "content": "mine"}]


# ---------------------------------------------------------------------------
# _discover_max_seq — drops the 1<<31 sentinel
# ---------------------------------------------------------------------------


async def test_discover_max_seq_ignores_sentinel_seq() -> None:
    """Docs with un-parseable seq (1<<31 sentinel) must not poison max-seq scan."""
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id

    d_real = _doc(ns, seq=7, item={"role": "user", "content": "real"})
    d_bogus = _doc(ns, seq="bogus", item={"role": "user", "content": "weird"})
    # First call inside _discover_max_seq; second is for the real add_items path.
    kb.storage.list_documents = AsyncMock(side_effect=[[d_bogus, d_real], [], [d_bogus, d_real], []])
    kb.storage.get_document_by_external_id = AsyncMock(return_value=None)

    await session.add_items([{"role": "user", "content": "new"}])

    # The next seq must be max(7) + 1 = 8 (not 1<<31 + 1).
    seen = [call.kwargs["external_id"] for call in kb.remember.await_args_list]
    assert seen == ["oai:conv-1:8"]


# ---------------------------------------------------------------------------
# clear_session — best-effort delete swallows forget() errors
# ---------------------------------------------------------------------------


async def test_clear_session_continues_when_forget_raises() -> None:
    kb = _make_kb()
    session = _make_session(kb)
    ns = session.namespace_id
    items = [{"role": "user", "content": f"m-{i}"} for i in range(3)]
    docs = [_doc(ns, seq=i, item=items[i]) for i in range(3)]
    kb.storage.list_documents = AsyncMock(side_effect=[docs, []])

    # First forget raises, the rest succeed — we still call forget 3 times.
    kb.forget = AsyncMock(side_effect=[RuntimeError("race"), None, None])
    await session.clear_session()
    assert kb.forget.await_count == 3
    # Counter still reset for the next add_items call.
    assert session._next_seq == 0


# ---------------------------------------------------------------------------
# Cached row_namespace_id round-trip
# ---------------------------------------------------------------------------


async def test_resolved_namespace_is_cached_across_calls() -> None:
    """``_resolved_namespace`` calls ``kb.storage.resolve_namespace`` exactly once."""
    kb = _make_kb()
    row_id = uuid4()
    kb.storage.resolve_namespace = AsyncMock(return_value=row_id)
    session = _make_session(kb)

    # Two consecutive empty list_documents calls; both go through _resolved_namespace.
    kb.storage.list_documents = AsyncMock(return_value=[])
    await session.get_items()
    await session.get_items()

    # Should be exactly one call — the row id is cached on the session.
    assert kb.storage.resolve_namespace.await_count == 1
    assert session._row_namespace_id == row_id
