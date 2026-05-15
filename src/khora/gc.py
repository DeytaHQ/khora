"""Background-coroutine helpers for retention / garbage collection (#620).

These are opt-in helpers — Khora does not run a scheduler on its own.
Adapters and downstream services that want session-scoped retention call
:func:`expire_sessions` from their own background loop / cron / task queue.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.khora import Khora


async def expire_sessions(
    *,
    kb: Khora,
    before: datetime,
    namespace_id: UUID | None = None,
) -> int:
    """Delete every session whose newest document predates ``before``.

    Calls :meth:`Khora.forget_session` for each matching session. Returns the
    number of *sessions* expired (not documents). Sessions whose newest
    document is at-or-after ``before`` are left alone — a single in-flight
    turn keeps the whole session alive.

    Args:
        kb: Connected :class:`Khora` instance.
        before: Cutoff timestamp. Sessions whose newest document was created
            (using ``COALESCE(source_timestamp, created_at)``) before this
            instant are eligible for deletion.
        namespace_id: Optional namespace filter. If omitted, every active
            namespace is scanned — useful for a per-deployment GC loop, but
            potentially slow on large fleets. Pass an explicit namespace
            for tenant-scoped sweeps.

    Returns:
        Count of sessions expired (each may have spanned multiple documents).
    """
    storage = kb.storage

    with trace_span(
        "khora.gc.expire_sessions",
        namespace_id=str(namespace_id) if namespace_id else "*",
    ):
        # Resolve namespace(s) to scan.
        if namespace_id is not None:
            resolved_namespaces = [await kb._resolve_namespace(namespace_id)]
        else:
            ns_page = await storage.list_namespaces(active_only=True, limit=1000, offset=0)
            resolved_namespaces = [ns.id for ns in ns_page.items]

        expired = 0
        for ns_row_id in resolved_namespaces:
            # Materialize all documents in the namespace. For very large
            # namespaces this is the bottleneck — callers should partition
            # by namespace if it gets unwieldy. The list paginates so we
            # don't blow out memory on a single fetch.
            # ``datetime`` comparisons mix tz-aware (DB rows) with
            # tz-naive (callers passing ``datetime.utcnow()``) badly, so we
            # use ``None`` as the "no-value" sentinel instead of
            # ``datetime.min`` and gate the comparison on it.
            session_latest: dict[UUID, datetime] = {}
            offset = 0
            page_size = 500
            while True:
                page = await storage.list_documents(ns_row_id, limit=page_size, offset=offset)
                if not page:
                    break
                for doc in page:
                    if doc.session_id is None:
                        continue
                    # Event time wins over ingest time: an old turn imported
                    # late shouldn't reset the session's TTL clock.
                    ts = doc.source_timestamp or doc.created_at
                    if ts is None:
                        continue
                    prev = session_latest.get(doc.session_id)
                    if prev is None or ts > prev:
                        session_latest[doc.session_id] = ts
                if len(page) < page_size:
                    break
                offset += page_size

            for sid, latest in session_latest.items():
                if latest < before:
                    try:
                        await kb.forget_session(ns_row_id, sid)
                        expired += 1
                    except Exception as exc:
                        # One session failing shouldn't abort the rest —
                        # operators may have partial outages on the graph
                        # backend that we don't want to cascade.
                        logger.warning(
                            "gc.expire_sessions: failed to forget session {session} in namespace {ns}: {exc}",
                            session=sid,
                            ns=ns_row_id,
                            exc=exc,
                        )

        return expired


__all__ = ["expire_sessions"]
