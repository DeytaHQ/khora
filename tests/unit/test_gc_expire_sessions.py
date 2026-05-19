"""Tests for ``khora.gc.expire_sessions`` (#620).

The GC helper is a thin coroutine that calls :meth:`Khora.forget_session`
for every session whose newest document predates ``before``. The tests
mock the ``Khora`` facade — we don't need a live engine here, just the
storage adapter providing ``list_namespaces`` and ``list_documents``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from khora.core.models import Document
from khora.gc import expire_sessions


def _make_doc(ns_id, sid, *, ts: datetime) -> Document:
    return Document(
        namespace_id=ns_id,
        content="x",
        created_at=ts,
        source_timestamp=ts,
        session_id=sid,
    )


def _fake_kb(documents: list[Document], namespace_ids: list) -> SimpleNamespace:
    """Build the minimal ``Khora``-shaped object the GC needs."""
    storage = SimpleNamespace()
    storage.list_documents = AsyncMock(return_value=documents)
    # Build a fake PaginatedResult.items
    nss = SimpleNamespace(
        items=[SimpleNamespace(id=nid) for nid in namespace_ids],
        total=len(namespace_ids),
        limit=1000,
        offset=0,
    )
    storage.list_namespaces = AsyncMock(return_value=nss)

    kb = SimpleNamespace()
    kb.storage = storage
    kb.forget_session = AsyncMock(return_value=0)
    kb._resolve_namespace = AsyncMock(side_effect=lambda x: x)
    return kb


async def test_no_sessions_returns_zero() -> None:
    """Empty namespace = nothing to expire."""
    ns_id = uuid4()
    kb = _fake_kb([], [ns_id])
    count = await expire_sessions(kb=kb, before=datetime.now(UTC))
    assert count == 0
    kb.forget_session.assert_not_called()


async def test_session_after_cutoff_kept() -> None:
    """A session whose newest doc is after ``before`` survives."""
    ns_id = uuid4()
    sid = uuid4()
    now = datetime.now(UTC)
    docs = [_make_doc(ns_id, sid, ts=now)]
    kb = _fake_kb(docs, [ns_id])

    count = await expire_sessions(kb=kb, before=now - timedelta(hours=1))
    assert count == 0
    kb.forget_session.assert_not_called()


async def test_session_before_cutoff_expired() -> None:
    """A session whose newest doc predates ``before`` is forgotten."""
    ns_id = uuid4()
    sid = uuid4()
    long_ago = datetime.now(UTC) - timedelta(days=30)
    docs = [_make_doc(ns_id, sid, ts=long_ago)]
    kb = _fake_kb(docs, [ns_id])

    count = await expire_sessions(kb=kb, before=datetime.now(UTC))
    assert count == 1
    kb.forget_session.assert_awaited_once_with(ns_id, sid)


async def test_mixed_sessions_partition_correctly() -> None:
    """Old sessions go; fresh sessions stay."""
    ns_id = uuid4()
    sid_old = uuid4()
    sid_fresh = uuid4()
    sid_partial = uuid4()  # has both old and fresh docs — fresh wins
    now = datetime.now(UTC)
    old_ts = now - timedelta(days=30)
    docs = [
        _make_doc(ns_id, sid_old, ts=old_ts),
        _make_doc(ns_id, sid_fresh, ts=now),
        _make_doc(ns_id, sid_partial, ts=old_ts),
        _make_doc(ns_id, sid_partial, ts=now),
    ]
    kb = _fake_kb(docs, [ns_id])

    count = await expire_sessions(kb=kb, before=now - timedelta(hours=1))
    assert count == 1
    kb.forget_session.assert_awaited_once_with(ns_id, sid_old)


async def test_documents_without_session_id_ignored() -> None:
    """Docs with NULL session_id can't be expired (no session to identify them)."""
    ns_id = uuid4()
    long_ago = datetime.now(UTC) - timedelta(days=30)
    docs = [_make_doc(ns_id, None, ts=long_ago)]  # NULL session_id
    kb = _fake_kb(docs, [ns_id])

    count = await expire_sessions(kb=kb, before=datetime.now(UTC))
    assert count == 0
    kb.forget_session.assert_not_called()


async def test_namespace_filter_limits_scope() -> None:
    """Passing ``namespace_id=…`` skips the active-namespace scan."""
    ns_id = uuid4()
    sid = uuid4()
    long_ago = datetime.now(UTC) - timedelta(days=30)
    docs = [_make_doc(ns_id, sid, ts=long_ago)]
    kb = _fake_kb(docs, [ns_id])

    count = await expire_sessions(kb=kb, before=datetime.now(UTC), namespace_id=ns_id)
    assert count == 1
    # ``list_namespaces`` should NOT have been called when a specific
    # namespace was requested — the helper short-circuits to a single resolve.
    kb.storage.list_namespaces.assert_not_called()
    kb._resolve_namespace.assert_awaited_once_with(ns_id)


async def test_forget_session_failure_does_not_abort_run() -> None:
    """One bad session shouldn't poison the rest."""
    ns_id = uuid4()
    sid_a = uuid4()
    sid_b = uuid4()
    long_ago = datetime.now(UTC) - timedelta(days=30)
    docs = [
        _make_doc(ns_id, sid_a, ts=long_ago),
        _make_doc(ns_id, sid_b, ts=long_ago),
    ]
    kb = _fake_kb(docs, [ns_id])

    # First forget_session raises, second succeeds.
    kb.forget_session = AsyncMock(side_effect=[RuntimeError("graph down"), 0])

    count = await expire_sessions(kb=kb, before=datetime.now(UTC))
    # Only the second one counts (first raised) — but the helper continued
    # past the failure.
    assert count == 1
    assert kb.forget_session.await_count == 2


async def test_source_timestamp_preferred_over_created_at() -> None:
    """source_timestamp wins so back-fills don't reset the TTL clock."""
    ns_id = uuid4()
    sid = uuid4()
    old_event = datetime.now(UTC) - timedelta(days=30)
    recent_ingest = datetime.now(UTC)
    # Recent ingest but old event time → should still be expired.
    doc = Document(
        namespace_id=ns_id,
        content="x",
        created_at=recent_ingest,
        source_timestamp=old_event,
        session_id=sid,
    )
    kb = _fake_kb([doc], [ns_id])

    count = await expire_sessions(kb=kb, before=datetime.now(UTC) - timedelta(hours=1))
    assert count == 1
    kb.forget_session.assert_awaited_once_with(ns_id, sid)
