"""Unit tests for ``KhoraMemoryService`` (#626).

Tests against ``AsyncMock(spec=Khora)`` so they run without infrastructure.
Exercises the three abstract methods (``add_session_to_memory`` /
``add_events_to_memory`` / ``search_memory``), the per-call namespace
resolution, idempotent re-ingest, and the ``BaseMemoryService`` Protocol
conformance.

Integration coverage (real khora + sqlite_lance + ADK Runner) lives in
``tests/integration/integrations/google_adk/``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

try:
    from google.adk.events.event import Event
    from google.adk.memory.base_memory_service import BaseMemoryService
    from google.adk.sessions.session import Session
    from google.genai import types as genai_types

    _HAS_ADK = True
except ImportError:
    _HAS_ADK = False


pytestmark = pytest.mark.skipif(not _HAS_ADK, reason="google-adk not installed")


from khora import Khora  # noqa: E402
from khora.core.models.document import (  # noqa: E402
    Chunk,
    Document,
    DocumentStatus,
)
from khora.integrations.google_adk._mapping import (  # noqa: E402
    KEY_AUTHOR,
    KEY_EVENT_ID,
    KEY_SESSION_ID,
    KEY_TIMESTAMP,
    namespace_uuid,
    session_uuid,
)


def _mk_kb() -> Khora:
    """Build an ``AsyncMock(spec=Khora)`` with the bits the service reaches for."""
    kb = AsyncMock(spec=Khora)
    kb.storage = MagicMock()
    kb.storage.get_document_by_external_id = AsyncMock(return_value=None)
    kb.storage.create_namespace = AsyncMock()
    kb._resolve_namespace = AsyncMock(side_effect=ValueError("not yet"))
    kb.remember = AsyncMock()
    kb.forget = AsyncMock(return_value=True)
    kb.recall = AsyncMock()
    return kb


def _mk_session(*, app_name: str = "app", user_id: str = "user-xyz", id: str = "s1") -> Session:
    return Session(id=id, app_name=app_name, user_id=user_id)


def _user_event(text: str, *, author: str = "user", ts: float | None = None) -> Event:
    return Event(
        author=author,
        content=genai_types.Content(role="user", parts=[genai_types.Part(text=text)]),
        timestamp=ts if ts is not None else time.time(),
    )


def _mk_chunk(
    namespace_id: UUID,
    *,
    content: str,
    event_id: str,
    session_id: str,
    author: str = "user",
) -> Chunk:
    custom = {
        KEY_AUTHOR: author,
        KEY_EVENT_ID: event_id,
        KEY_SESSION_ID: session_id,
        KEY_TIMESTAMP: "2026-05-15T00:00:00+00:00",
    }
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        metadata=custom,
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
    )


# ----------------------------------------------------------------------
# Construction + protocol conformance
# ----------------------------------------------------------------------


def test_service_is_a_base_memory_service():
    from khora.integrations.google_adk import KhoraMemoryService

    svc = KhoraMemoryService(kb=_mk_kb())
    assert isinstance(svc, BaseMemoryService)


def test_service_rejects_empty_app_id():
    from khora.integrations.google_adk import KhoraMemoryService

    with pytest.raises(ValueError, match="app_id"):
        KhoraMemoryService(kb=_mk_kb(), app_id="")


def test_service_rejects_zero_recall_limit():
    from khora.integrations.google_adk import KhoraMemoryService

    with pytest.raises(ValueError, match="recall_limit"):
        KhoraMemoryService(kb=_mk_kb(), recall_limit=0)


# ----------------------------------------------------------------------
# add_session_to_memory
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_session_remembers_each_event():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)
    session = _mk_session()
    session.events = [
        _user_event("hello", ts=1.0),
        _user_event("how are you", author="model", ts=2.0),
    ]

    await svc.add_session_to_memory(session)

    assert kb.remember.await_count == 2
    namespace = namespace_uuid(app_name="app", user_id="user-xyz")
    for call in kb.remember.await_args_list:
        assert call.kwargs["namespace"] == namespace
        assert call.kwargs["session_id"] == session_uuid("s1")
        assert call.kwargs["external_id"].startswith("adk_event:")


@pytest.mark.asyncio
async def test_add_session_skips_events_with_no_content():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)
    session = _mk_session()
    session.events = [
        Event(author="user", content=genai_types.Content(role="user", parts=[]), timestamp=1.0),
        _user_event("after the empty one", ts=2.0),
    ]

    await svc.add_session_to_memory(session)

    assert kb.remember.await_count == 1


@pytest.mark.asyncio
async def test_add_session_creates_namespace_on_first_call():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)
    session = _mk_session()
    session.events = [_user_event("hello", ts=1.0)]

    await svc.add_session_to_memory(session)

    kb.storage.create_namespace.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_session_is_idempotent_on_reingest():
    """A second ingest of the same event forgets the prior doc first."""
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    existing_doc = Document(
        id=uuid4(),
        namespace_id=uuid4(),
        content="hello",
        external_id="adk_event:abc",
        status=DocumentStatus.COMPLETED,
    )
    kb.storage.get_document_by_external_id = AsyncMock(return_value=existing_doc)
    svc = KhoraMemoryService(kb=kb)

    session = _mk_session()
    session.events = [_user_event("hello", ts=1.0)]

    await svc.add_session_to_memory(session)

    kb.forget.assert_awaited()
    kb.remember.assert_awaited_once()


# ----------------------------------------------------------------------
# add_events_to_memory
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_events_to_memory_uses_explicit_app_and_user():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)
    events = [_user_event("delta event", ts=1.0)]

    await svc.add_events_to_memory(app_name="appA", user_id="userB-12345", events=events, session_id="sess")

    assert kb.remember.await_count == 1
    call = kb.remember.await_args_list[0]
    assert call.kwargs["namespace"] == namespace_uuid(app_name="appA", user_id="userB-12345")
    assert call.kwargs["session_id"] == session_uuid("sess")


@pytest.mark.asyncio
async def test_add_events_to_memory_merges_custom_metadata():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)
    events = [_user_event("text", ts=1.0)]

    await svc.add_events_to_memory(
        app_name="app",
        user_id="user-xyz",
        events=events,
        session_id="sess",
        custom_metadata={"ttl_hint": 600},
    )

    metadata = kb.remember.await_args_list[0].kwargs["metadata"]
    assert metadata["ttl_hint"] == 600
    # Adapter-owned keys still present:
    assert metadata[KEY_AUTHOR] == "user"


# ----------------------------------------------------------------------
# search_memory
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memory_returns_empty_for_unknown_namespace():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    svc = KhoraMemoryService(kb=kb)

    response = await svc.search_memory(app_name="app", user_id="user-xyz", query="hello")
    assert response.memories == []
    kb.recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_memory_maps_chunks_to_memory_entries():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    namespace = namespace_uuid(app_name="app", user_id="user-xyz")
    kb._resolve_namespace = AsyncMock(return_value=namespace)
    chunk1 = _mk_chunk(namespace, content="alpha", event_id="ev1", session_id="s1")
    chunk2 = _mk_chunk(namespace, content="beta", event_id="ev2", session_id="s1", author="model")

    recall_result = MagicMock()
    recall_result.chunks = [(chunk1, 0.9), (chunk2, 0.8)]
    kb.recall = AsyncMock(return_value=recall_result)

    svc = KhoraMemoryService(kb=kb)
    response = await svc.search_memory(app_name="app", user_id="user-xyz", query="alpha")

    assert len(response.memories) == 2
    assert response.memories[0].author == "user"
    assert response.memories[1].author == "model"
    assert response.memories[0].timestamp == "2026-05-15T00:00:00+00:00"


@pytest.mark.asyncio
async def test_search_memory_dedupes_chunks_by_event_id():
    """Two chunks from the same event collapse to one MemoryEntry."""
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    namespace = namespace_uuid(app_name="app", user_id="user-xyz")
    kb._resolve_namespace = AsyncMock(return_value=namespace)
    chunk_a = _mk_chunk(namespace, content="first half", event_id="same-event", session_id="s1")
    chunk_b = _mk_chunk(namespace, content="second half", event_id="same-event", session_id="s1")

    recall_result = MagicMock()
    recall_result.chunks = [(chunk_a, 0.9), (chunk_b, 0.85)]
    kb.recall = AsyncMock(return_value=recall_result)

    svc = KhoraMemoryService(kb=kb)
    response = await svc.search_memory(app_name="app", user_id="user-xyz", query="anything")

    assert len(response.memories) == 1


@pytest.mark.asyncio
async def test_search_memory_forwards_recall_limit_and_min_similarity():
    from khora.integrations.google_adk import KhoraMemoryService

    kb = _mk_kb()
    namespace = namespace_uuid(app_name="app", user_id="user-xyz")
    kb._resolve_namespace = AsyncMock(return_value=namespace)
    empty = MagicMock()
    empty.chunks = []
    kb.recall = AsyncMock(return_value=empty)

    svc = KhoraMemoryService(kb=kb, recall_limit=25, min_similarity=0.4)
    await svc.search_memory(app_name="app", user_id="user-xyz", query="q")

    kb.recall.assert_awaited_once()
    kwargs = kb.recall.await_args.kwargs
    assert kwargs["limit"] == 25
    assert kwargs["min_similarity"] == 0.4
    assert kwargs["namespace"] == namespace


# ----------------------------------------------------------------------
# KhoraIntegration marker Protocol
# ----------------------------------------------------------------------


def test_service_satisfies_khora_integration_marker():
    from khora.integrations.google_adk import KhoraMemoryService

    svc = KhoraMemoryService(kb=_mk_kb())
    assert svc.name == "google_adk"
    # ``namespace_id`` is a per-call concept in ADK; the marker exposes
    # the zero UUID sentinel for compatibility.
    assert isinstance(svc.namespace_id, UUID)
    assert svc.kb is not None


# Suppress "F401-via-import" pollution: importing ``Any`` keeps the type-
# annotations in the helper section honest under future expansion.
_ = Any
