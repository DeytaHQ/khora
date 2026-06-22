"""``KhoraSession`` — ``agents.memory.session.SessionABC`` backed by khora.

Drop-in replacement for the SDK's reference ``SQLiteSession``::

    from agents import Agent, Runner
    from khora.integrations.openai_agents import KhoraSession

    session = KhoraSession(kb=kb, namespace=ns_id, session_id="conv-1")
    result = await Runner.run(agent, "Hello", session=session)

Module-load discipline: nothing from ``agents`` is imported at module
top level. The ``SessionABC`` base is resolved lazily on first
``KhoraSession`` instantiation, mirroring the trick the LangGraph and
Google ADK adapters use. The AST lint
(``tools/check_optional_imports.py``) does not catch ``import agents``
directly (the directory name is ``openai_agents``), so the
``test_no_eager_imports.py`` subprocess probe is the gate of record.

Stability: experimental. The ``openai-agents`` SDK is pre-1.0 (17
releases in 7 months as of v0.17). Pin tight in ``pyproject.toml`` and
expect monthly rework as the upstream surface shifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.integrations.openai_agents._mapping import (
    KEY_ITEM_JSON,
    KEY_SEQ,
    KEY_SESSION_ID,
    event_external_id,
    item_to_remember_kwargs,
    session_uuid,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agents.items import TResponseInputItem

    from khora.khora import Khora


# Cached pointer to the SDK SessionABC class. Resolved on first
# ``KhoraSession`` instantiation; reused thereafter.
_SessionABC: type | None = None


def _resolve_session_abc() -> type:
    """Lazy-resolve ``agents.memory.session.SessionABC``.

    Centralised so all method-body imports route through one helper —
    one place to maintain the error message when the extra isn't
    installed, and one place to pay the import cost.
    """
    global _SessionABC
    if _SessionABC is not None:
        return _SessionABC
    try:
        from agents.memory.session import SessionABC as _Abc  # noqa: PLC0415 — lazy
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraSession requires the optional `openai-agents` extra. Install with: pip install 'khora[openai-agents]'"
        ) from exc
    _SessionABC = _Abc
    return _Abc


class KhoraSession:
    """OpenAI Agents SDK ``SessionABC`` implementation backed by khora.

    Args:
        kb: A connected :class:`khora.Khora` instance. The session does
            NOT own the connection lifecycle — the caller does.
        namespace: Stable khora namespace UUID. Every read and write is
            scoped to this namespace.
        session_id: The SDK session id. Maps to a deterministic khora
            ``session_id`` via UUID5 (or passes through verbatim if the
            string parses as a UUID). Required and non-empty.
        app_id: Free-form app identifier stamped into stored metadata.
            Default ``"openai_agents"``.

    Concurrency: every async method touches khora through async
    primitives, so concurrent calls are safe relative to the shared khora
    pool. ``add_items`` reserves sequence numbers under an in-instance
    asyncio lock so two concurrent adds don't collide on ``external_id``.

    Stability: experimental until the upstream SDK reaches 1.0.
    """

    name: str = "openai_agents"
    """Identifier for ``khora.integrations`` registry / telemetry."""

    # The SDK ``Session`` Protocol requires both ``session_id: str`` and
    # an optional ``session_settings`` attribute. We declare them at the
    # class level so static checkers see them without us having to
    # ``__init_subclass__`` from the resolved base.
    session_settings: Any = None

    def __init__(
        self,
        *,
        kb: Khora,
        namespace: UUID,
        session_id: str,
        app_id: str = "openai_agents",
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
        if not isinstance(app_id, str) or not app_id.strip():
            raise ValueError(f"app_id must be a non-empty string, got {app_id!r}")
        if not isinstance(namespace, UUID):
            raise TypeError(f"namespace must be a UUID, got {type(namespace).__name__}")

        # Resolve and inject the SDK base class on first construction so
        # ``isinstance(s, SessionABC)`` passes at runtime. Same trick the
        # LangGraph + ADK adapters use to defer the framework import to
        # first instantiation. ``SessionABC`` is an ABC — we MUST be a
        # subclass to instantiate without an ABC TypeError.
        base_cls = _resolve_session_abc()
        if base_cls not in type(self).__mro__:
            new_cls = type("KhoraSession", (KhoraSession, base_cls), {})
            self.__class__ = new_cls

        self.kb = kb
        self.namespace_id = namespace
        self.session_id = session_id
        self.app_id = app_id
        self._khora_session_id: UUID = session_uuid(session_id)

        # Resolved (row-level) namespace id used by ``kb.storage.list_documents``.
        # Public ``namespace_id`` is the stable identifier callers pass in;
        # the row id is what every ``kb.storage.*`` call expects. We resolve
        # lazily on first read so construction stays sync.
        self._row_namespace_id: UUID | None = None
        # Monotonic write counter. Initialised on first ``add_items`` by
        # scanning existing chunks for the highest stamped seq. Held
        # under a lock so two concurrent ``add_items`` calls don't
        # collide on ``external_id`` collisions.
        self._next_seq: int | None = None
        # Lazy lock to avoid binding to a particular event loop at
        # construction time. KhoraSession may be built on one loop and
        # used from another (Runner.run vs. Runner.run_sync).
        self._seq_lock: Any = None

    # ------------------------------------------------------------------
    # SessionABC contract
    # ------------------------------------------------------------------

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return the items in this session, in insertion order.

        Args:
            limit: When set, returns the latest ``limit`` items (still in
                chronological order). When ``None``, returns every item.

        Returns:
            A fresh list — callers may mutate it without affecting state.
        """
        ordered = await self._load_session_documents()
        items: list[Any] = []
        for _seq, doc in ordered:
            decoded = _decode_item_from_doc(doc)
            if decoded is None:
                continue
            items.append(decoded)
        if limit is not None and limit >= 0 and len(items) > limit:
            items = items[-limit:]
        return items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Persist ``items`` to khora, preserving order.

        Each item becomes one khora document, stamped with this session's
        ``session_id`` and a monotonic ``seq`` so ``get_items`` can
        reconstruct chronological order on read-back.
        """
        if not items:
            return
        import asyncio  # noqa: PLC0415 — only needed to bind the lock lazily

        if self._seq_lock is None:
            self._seq_lock = asyncio.Lock()

        async with self._seq_lock:
            if self._next_seq is None:
                self._next_seq = await self._discover_max_seq() + 1
            seq_start = self._next_seq
            self._next_seq = seq_start + len(items)

        row_ns = await self._resolved_namespace()
        for offset, item in enumerate(items):
            seq = seq_start + offset
            kwargs = item_to_remember_kwargs(
                item,
                session_id=self.session_id,
                app_id=self.app_id,
                seq=seq,
            )
            # Idempotency: if a document at this (session, seq) external
            # id already exists, drop it first so chunk lists don't grow
            # on re-ingest (a Runner retry, say).
            existing = await self.kb.storage.get_document_by_external_id(kwargs["external_id"], namespace_id=row_ns)
            if existing is not None:
                await self.kb.forget(existing.id, namespace=self.namespace_id)
            await self.kb.remember(namespace=self.namespace_id, **kwargs)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item, or ``None`` if empty.

        Implementation: find the document with the highest ``seq`` stamped
        for this session, decode it, delete the document, and return the
        decoded item. The next ``add_items`` call will re-discover the
        max seq automatically.
        """
        ordered = await self._load_session_documents()
        if not ordered:
            return None
        _last_seq, last_doc = ordered[-1]
        item = _decode_item_from_doc(last_doc)
        await self.kb.forget(last_doc.id, namespace=self.namespace_id)
        # Force a seq re-scan on the next add so we don't reuse the seq
        # we just dropped.
        self._next_seq = None
        return item

    async def clear_session(self) -> None:
        """Delete every item belonging to this session."""
        ordered = await self._load_session_documents()
        for _seq, doc in ordered:
            try:
                await self.kb.forget(doc.id, namespace=self.namespace_id)
            except Exception as exc:  # noqa: BLE001 — best-effort cascade delete
                # khora.forget raises if the document was already deleted
                # by a racing pop_item. Log and continue — we still want
                # to drop the rest of the session.
                logger.debug("KhoraSession.clear_session skipping {}: {}", doc.id, exc)
        self._next_seq = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _resolved_namespace(self) -> UUID:
        """Return the row-level namespace id (resolved once, cached).

        Document-level ``kb.storage.*`` reads (``list_documents`` /
        ``get_document_by_external_id``) expect the row id, not the public
        ``namespace_id``. We resolve through the public
        ``kb.storage.resolve_namespace`` (idempotent on row ids) on first
        access and cache for the lifetime of the session. ``namespace_id``
        is validated as a ``UUID`` in ``__init__``.
        """
        if self._row_namespace_id is None:
            self._row_namespace_id = await self.kb.storage.resolve_namespace(self.namespace_id)
        return self._row_namespace_id

    async def _load_session_documents(self) -> list[tuple[int, Any]]:
        """Return every document written by this session as ``(seq, doc)`` pairs.

        Iterates ``storage.list_documents`` and keeps documents whose
        metadata is stamped with the matching ``oai_session_id``. We work
        at the document level (not chunk level) because khora's document
        store is the only universally reliable iteration surface — chunk
        storage layout varies per backend (sqlite_lance stores chunks in
        LanceDB + a Skeleton temporal table that ``storage.list_chunks``
        doesn't see). Each document carries the verbatim item JSON in
        ``metadata.custom["oai_item"]`` so we don't need to read any
        chunks at all on the recall path.
        """
        storage = self.kb.storage
        row_ns = await self._resolved_namespace()
        gathered: list[tuple[int, Any]] = []
        cursor = 0
        page_size = 200
        while True:
            page = await storage.list_documents(row_ns, limit=page_size, offset=cursor)
            if not page:
                break
            for doc in page:
                custom = doc.metadata or {}
                if custom.get(KEY_SESSION_ID) != self.session_id:
                    continue
                seq_val = custom.get(KEY_SEQ)
                if isinstance(seq_val, int):
                    seq = seq_val
                elif isinstance(seq_val, str) and seq_val.isdigit():
                    seq = int(seq_val)
                else:
                    # Documents missing a seq fall to the end (legacy /
                    # foreign writes that landed in the same session
                    # bucket).
                    seq = 1 << 31
                gathered.append((seq, doc))
            if len(page) < page_size:
                break
            cursor += page_size
        gathered.sort(key=lambda pair: pair[0])
        return gathered

    async def _discover_max_seq(self) -> int:
        """Return the highest ``oai_seq`` stamped in this session.

        Returns ``-1`` for an empty session so the first issued seq is
        ``0``. Single scan of the namespace's documents; cheap for the
        bounded conversation sizes the SDK targets (hundreds to low
        thousands of turns).
        """
        ordered = await self._load_session_documents()
        highest = -1
        for seq, _doc in ordered:
            if seq != (1 << 31) and seq > highest:
                highest = seq
        return highest


def _decode_item_from_doc(doc: Any) -> Any | None:
    """Recover the original ``TResponseInputItem`` from a stored document.

    Returns ``None`` if the document wasn't written by this adapter (no
    ``oai_item`` key in ``metadata``) or if the stored JSON is
    corrupt. The SDK silently skips invalid items in its reference
    ``SQLiteSession``; we mirror that behaviour.
    """
    import json  # noqa: PLC0415 — only used here, keeps the import surface tight

    custom = doc.metadata or {}
    raw = custom.get(KEY_ITEM_JSON)
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None
    return raw


__all__ = ["KhoraSession", "event_external_id", "session_uuid"]
