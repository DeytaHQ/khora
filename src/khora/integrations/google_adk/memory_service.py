"""``KhoraMemoryService`` — ``google.adk.memory.BaseMemoryService`` backed by khora.

Drop-in replacement for ADK's ``InMemoryMemoryService`` /
``VertexAiMemoryBankService``. Wires khora's vector + entity-graph
recall behind ``BaseMemoryService`` so an ADK ``Runner`` ingests every
``Session.events[i]`` into khora and retrieves them later via
``search_memory(*, app_name, user_id, query)``.

Scope (per issue #626): this module ships ``KhoraMemoryService`` only.
``KhoraSessionService`` is intentionally **not** part of v1 — ADK's
``DatabaseSessionService`` already covers short-term turn state.

Module-load discipline: nothing from ``google.adk`` or ``google.genai``
is imported at module top level. Framework classes are resolved lazily
on first instantiation via :func:`_resolve_adk_classes` so the AST lint
(``tools/check_optional_imports.py``) and the subprocess no-import test
both pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.integrations.google_adk._mapping import (
    KEY_AUTHOR,
    KEY_EVENT_ID,
    KEY_PARTS,
    KEY_SESSION_ID,
    KEY_TIMESTAMP,
    chunk_to_memory_entry,
    event_to_remember_kwargs,
    namespace_uuid,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from google.adk.events.event import Event
    from google.adk.memory.base_memory_service import (
        SearchMemoryResponse,
    )
    from google.adk.sessions.session import Session

    from khora.khora import Khora


# Cached pointers to the ADK / genai classes. Resolved on first
# ``KhoraMemoryService`` instantiation; reused thereafter.
_AdkClasses: dict[str, type] | None = None


def _resolve_adk_classes() -> dict[str, type]:
    """Lazy-resolve every ADK / genai class the service needs.

    Centralised so all method-body imports route through one helper —
    one place to maintain the error message when the extra isn't
    installed, and one place to pay the import cost.
    """
    global _AdkClasses
    if _AdkClasses is not None:
        return _AdkClasses
    try:
        from google.adk.memory.base_memory_service import (  # noqa: PLC0415 — lazy
            BaseMemoryService,
            SearchMemoryResponse,
        )
        from google.adk.memory.memory_entry import MemoryEntry  # noqa: PLC0415
        from google.genai import types as genai_types  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraMemoryService requires the optional `google-adk` extra. Install with: pip install 'khora[google-adk]'"
        ) from exc

    _AdkClasses = {
        "BaseMemoryService": BaseMemoryService,
        "SearchMemoryResponse": SearchMemoryResponse,
        "MemoryEntry": MemoryEntry,
        "Content": genai_types.Content,
        "Part": genai_types.Part,
    }
    return _AdkClasses


# We construct the ``BaseMemoryService`` subclass on first
# ``KhoraMemoryService`` instantiation so this module imports cleanly
# without the extra installed. The runtime class is substituted into
# the MRO via ``__class__`` rebinding (same trick as the LangGraph
# adapter — see #624 for the rationale).
_KhoraMemoryBase: type | None = None


def _get_base_class() -> type:
    """Return (and cache) the dynamically-resolved ``BaseMemoryService`` class."""
    global _KhoraMemoryBase
    if _KhoraMemoryBase is None:
        _KhoraMemoryBase = _resolve_adk_classes()["BaseMemoryService"]
    return _KhoraMemoryBase


class KhoraMemoryService:
    """ADK ``BaseMemoryService`` backed by a khora knowledge base.

    Wire it into a ``Runner`` so the agent's long-term memory is khora's
    vector + entity graph::

        runner = Runner(
            app_name="my_app",
            agent=my_agent,
            session_service=InMemorySessionService(),
            memory_service=KhoraMemoryService(kb=kb),
        )

    Args:
        kb: A connected :class:`khora.Khora` instance. The service does
            NOT own the connection lifecycle — the caller does.
        app_id: Free-form identifier stamped into stored metadata.
            Default ``"google_adk"``. Distinct from the ``app_name``
            ADK passes per call — the latter is part of the namespace
            key, this is just a metadata tag for debugging / audits.
        recall_limit: Default ``limit`` forwarded to ``Khora.recall``.
            Default 10 (matches ADK convention).
        min_similarity: Default similarity floor forwarded to
            ``Khora.recall``. Default 0.0 (no floor).

    Namespace scheme:
        Per #618 canonical mapping, each ADK (app_name, user_id) pair
        maps to a deterministic khora namespace UUID5
        (``adk:{app_name}:{user_id}``). Two services pointing at the
        same khora deployment with the same (app_name, user_id) thus
        see the same memory. The namespace row is lazily created on
        first ingest so callers don't need to pre-allocate.

    Session scheme (#620):
        ``Session.id`` (an arbitrary string) maps to a UUID5-derived
        ``session_id`` per the convention in :func:`_mapping.session_uuid`.
        Use ``Khora.forget_session(namespace, session_id)`` to drop a
        whole session's events.

    Concurrency:
        Every public method touches khora via async primitives, so
        concurrent calls are safe relative to the shared khora pool.
        The adapter has no internal state past the ``_seen_namespaces``
        cache (best-effort, race-tolerant).

    Stability: experimental until v0.14 ships one full minor without a
    breaking change to this surface.
    """

    name: str = "google_adk"
    """Identifier for ``khora.integrations`` registry / telemetry."""

    def __init__(
        self,
        *,
        kb: Khora,
        app_id: str = "google_adk",
        recall_limit: int = 10,
        min_similarity: float = 0.0,
    ) -> None:
        # Resolve the BaseMemoryService class once and rebind ``self.__class__``
        # to a subclass with it in the MRO. Same trick the LangGraph adapter
        # uses to defer the framework import to instantiation while still
        # letting ADK's internals see ``isinstance(svc, BaseMemoryService)``.
        base_cls = _get_base_class()
        if base_cls not in type(self).__mro__:
            new_cls = type("KhoraMemoryService", (KhoraMemoryService, base_cls), {})
            self.__class__ = new_cls

        if not isinstance(app_id, str) or not app_id.strip():
            raise ValueError(f"app_id must be a non-empty string, got {app_id!r}")
        if recall_limit < 1:
            raise ValueError(f"recall_limit must be >= 1, got {recall_limit}")

        self.kb = kb
        self.app_id = app_id
        self.recall_limit = recall_limit
        self.min_similarity = min_similarity

        # Per-process cache of namespace UUIDs we've already ensured exist.
        # Avoids a SELECT per add_session_to_memory call once the row is up.
        self._ensured_namespaces: set[UUID] = set()

    # ------------------------------------------------------------------
    # Public KhoraIntegration marker Protocol attrs
    # ------------------------------------------------------------------

    @property
    def namespace_id(self) -> UUID:
        """Deprecated: namespace is per-call in ADK. Returns the zero UUID.

        ADK's ``BaseMemoryService`` contract is per-call namespaced via
        ``(app_name, user_id)`` rather than fixed at construction. The
        ``KhoraIntegration`` marker Protocol requires a ``namespace_id``
        attribute, so we expose the zero UUID as a sentinel — adapters
        that share a single namespace use a real UUID here.
        """
        return UUID(int=0)

    # ------------------------------------------------------------------
    # BaseMemoryService surface
    # ------------------------------------------------------------------

    async def add_session_to_memory(self, session: Session) -> None:
        """Ingest every event in ``session`` as a separate khora document.

        Matches ``InMemoryMemoryService.add_session_to_memory``: events
        whose ``content`` has no parts are skipped. Each retained event
        becomes one document with ``session_id`` stamped per #620, so a
        later ``Khora.forget_session`` can drop the whole conversation
        atomically.

        Re-ingesting the same ``Session`` is safe: deduplication keys off
        ``event.id`` (encoded into ``external_id``). Existing documents
        for an event are deleted before re-writing so chunk lists do not
        accumulate.
        """
        namespace_id = namespace_uuid(app_name=session.app_name, user_id=session.user_id)
        await self._ensure_namespace(namespace_id, session.app_name, session.user_id)

        events = getattr(session, "events", None) or []
        for event in events:
            await self._ingest_event(event, session=session, namespace_id=namespace_id)

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Ingest an explicit list of events as an incremental delta.

        Treats ``events`` as a delta on top of whatever is already in
        memory — only new events (by ``event.id``) are written. The
        deduplication probe uses
        ``storage.get_document_by_external_id(adk_event:<id>, namespace_id=namespace)``
        which is indexed in every khora backend.

        ``custom_metadata`` is merged into each event's
        ``Document.metadata`` so callers can stamp portable
        fields (e.g. ``ttl_hint``) without subclassing the service.
        """
        namespace_id = namespace_uuid(app_name=app_name, user_id=user_id)
        await self._ensure_namespace(namespace_id, app_name, user_id)

        # Build a lightweight session-shaped duck so the mapping helper
        # can pull ``app_name`` / ``user_id`` / ``id`` from one place.
        synthetic_session = _SyntheticSession(
            app_name=app_name,
            user_id=user_id,
            id=session_id or "",
        )

        extra_meta = dict(custom_metadata or {})
        for event in events:
            await self._ingest_event(
                event,
                session=synthetic_session,
                namespace_id=namespace_id,
                extra_metadata=extra_meta,
            )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Vector-search the (app_name, user_id) khora namespace.

        Wraps ``Khora.recall`` (HYBRID mode by default) and maps the
        returned chunks back to ``MemoryEntry`` instances. Returns an
        empty response (not an error) when the namespace has no
        ingested events yet — matches ADK's in-memory implementation.
        """
        classes = _resolve_adk_classes()
        SearchMemoryResponse = classes["SearchMemoryResponse"]  # noqa: N806
        MemoryEntry = classes["MemoryEntry"]  # noqa: N806
        Content = classes["Content"]  # noqa: N806
        Part = classes["Part"]  # noqa: N806

        namespace_id = namespace_uuid(app_name=app_name, user_id=user_id)
        # Don't ensure-create on read — keeps recall a strict read path.
        if not await self._namespace_exists(namespace_id):
            return SearchMemoryResponse()

        recall = await self.kb.recall(
            query,
            namespace=namespace_id,
            limit=self.recall_limit,
            min_similarity=self.min_similarity,
        )

        # Document-level metadata (where KEY_EVENT_ID lives) is on the
        # top-level ``RecallResult.documents`` list; join via document_id.
        doc_metadata: dict[UUID, dict[str, Any]] = {doc.id: dict(doc.metadata or {}) for doc in recall.documents}

        response = SearchMemoryResponse()
        seen_events: set[str] = set()
        for chunk in recall.chunks:
            custom = doc_metadata.get(chunk.document_id, {})
            event_id = str(custom.get(KEY_EVENT_ID) or "")
            # Same event may produce multiple chunks; surface each event once.
            if event_id and event_id in seen_events:
                continue
            seen_events.add(event_id)
            entry = chunk_to_memory_entry(
                chunk,
                custom_metadata=custom,
                memory_entry_cls=MemoryEntry,
                content_cls=Content,
                part_cls=Part,
            )
            response.memories.append(entry)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ingest_event(
        self,
        event: Any,
        *,
        session: Any,
        namespace_id: UUID,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Translate one Event and persist it via ``Khora.remember``."""
        kwargs = event_to_remember_kwargs(event, session=session, app_id=self.app_id)
        if kwargs is None:
            return  # event had no usable content (control-flow only)
        if extra_metadata:
            # Caller-supplied keys take precedence over our defaults — they're
            # explicit per-call overrides. Adapter-owned ``adk_*`` keys are
            # never overwritten silently.
            merged = dict(extra_metadata)
            merged.update(kwargs["metadata"])
            kwargs["metadata"] = merged

        # ``source_timestamp_iso`` is a helper field for the service, not a
        # ``Khora.remember`` parameter — strip before forwarding.
        kwargs.pop("source_timestamp_iso", None)

        external_id = kwargs["external_id"]
        # Idempotency: if a document for this event already exists, forget
        # it first so we don't accumulate duplicate chunks on re-ingest.
        existing = await self.kb.storage.get_document_by_external_id(external_id, namespace_id=namespace_id)
        if existing is not None:
            await self.kb.forget(existing.id, namespace=namespace_id)

        await self.kb.remember(namespace=namespace_id, **kwargs)

    async def _ensure_namespace(self, namespace_id: UUID, app_name: str, user_id: str) -> None:
        """Create the khora namespace row if it doesn't exist yet.

        Idempotent. Falls back gracefully on a race — if another caller
        created the same UUID5 namespace concurrently, we swallow the
        error after re-verifying the row exists.
        """
        if namespace_id in self._ensured_namespaces:
            return
        try:
            await self.kb._resolve_namespace(namespace_id)
            self._ensured_namespaces.add(namespace_id)
            return
        except ValueError:
            pass

        from khora.core.models.tenancy import MemoryNamespace  # noqa: PLC0415

        ns = MemoryNamespace(
            id=namespace_id,
            namespace_id=namespace_id,
            metadata={
                "source": "khora.integrations.google_adk",
                "app_name": app_name,
                "user_id": user_id,
                "app_id": self.app_id,
            },
        )
        try:
            await self.kb.storage.create_namespace(ns)
        except Exception as exc:  # pragma: no cover - race-safe creation
            try:
                await self.kb._resolve_namespace(namespace_id)
            except ValueError:
                raise exc from None
            logger.debug("KhoraMemoryService namespace creation race resolved cleanly: {}", exc)
        self._ensured_namespaces.add(namespace_id)

    async def _namespace_exists(self, namespace_id: UUID) -> bool:
        """Return True if the khora namespace has been created."""
        if namespace_id in self._ensured_namespaces:
            return True
        try:
            await self.kb._resolve_namespace(namespace_id)
        except ValueError:
            return False
        self._ensured_namespaces.add(namespace_id)
        return True


class _SyntheticSession:
    """Minimal duck-typed session for ``add_events_to_memory``.

    ADK's ``add_events_to_memory`` takes a plain (app_name, user_id,
    session_id) triple rather than a full ``Session`` object. The
    mapping helper expects an object with ``app_name`` / ``user_id`` /
    ``id`` attributes, so we wrap the triple in this small struct
    rather than fishing for ``Session`` to construct properly (it has
    required fields like ``state`` we'd just be filling with defaults).
    """

    __slots__ = ("app_name", "id", "user_id")

    def __init__(self, *, app_name: str, user_id: str, id: str) -> None:  # noqa: A002 — match Session.id
        self.app_name = app_name
        self.user_id = user_id
        self.id = id


# Re-export the metadata keys from _mapping so test files have one canonical
# import path. Kept here so a caller can do
# ``from khora.integrations.google_adk.memory_service import KEY_EVENT_ID``.
__all__ = [
    "KEY_AUTHOR",
    "KEY_EVENT_ID",
    "KEY_PARTS",
    "KEY_SESSION_ID",
    "KEY_TIMESTAMP",
    "KhoraMemoryService",
]
