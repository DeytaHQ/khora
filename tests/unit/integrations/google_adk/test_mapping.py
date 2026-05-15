"""Unit tests for ``khora.integrations.google_adk._mapping``.

Covers the round-trip between ADK's ``Event`` / ``Content`` / ``Part``
shapes and khora's document / chunk metadata. Uses the real
``google.adk`` types where available — they're cheap to construct,
pydantic-validated, and that gives the round-trip tests genuine
confidence rather than mock-on-mock.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

try:
    from google.adk.events.event import Event
    from google.adk.memory.memory_entry import MemoryEntry
    from google.adk.sessions.session import Session
    from google.genai import types as genai_types

    _HAS_ADK = True
except ImportError:
    _HAS_ADK = False


pytestmark = pytest.mark.skipif(not _HAS_ADK, reason="google-adk not installed")


from khora.core.models.document import Chunk, ChunkMetadata  # noqa: E402
from khora.integrations.google_adk._mapping import (  # noqa: E402
    KEY_AUTHOR,
    KEY_EVENT_ID,
    KEY_PARTS,
    KEY_SESSION_ID,
    KEY_TIMESTAMP,
    chunk_to_memory_entry,
    content_to_text,
    event_external_id,
    event_to_remember_kwargs,
    namespace_uuid,
    serialise_non_text_parts,
    session_uuid,
)


def _mk_session(*, app_name: str = "app", user_id: str = "user-1234", id: str = "s1") -> Session:
    return Session(id=id, app_name=app_name, user_id=user_id)


def _text_event(text: str, *, author: str = "user", ts: float | None = None) -> Event:
    return Event(
        author=author,
        content=genai_types.Content(
            role="user" if author == "user" else "model",
            parts=[genai_types.Part(text=text)],
        ),
        timestamp=ts if ts is not None else time.time(),
    )


# ----------------------------------------------------------------------
# namespace_uuid / session_uuid — deterministic UUID5 derivation
# ----------------------------------------------------------------------


def test_namespace_uuid_is_deterministic():
    a = namespace_uuid(app_name="my_app", user_id="user-1234")
    b = namespace_uuid(app_name="my_app", user_id="user-1234")
    assert a == b
    assert isinstance(a, UUID)


def test_namespace_uuid_distinguishes_apps_and_users():
    a = namespace_uuid(app_name="app_a", user_id="user-1234")
    b = namespace_uuid(app_name="app_b", user_id="user-1234")
    c = namespace_uuid(app_name="app_a", user_id="other-user")
    assert len({a, b, c}) == 3


def test_session_uuid_passes_uuid_strings_through():
    raw = uuid4()
    derived = session_uuid(str(raw))
    assert derived == raw


def test_session_uuid_derives_from_arbitrary_strings():
    a = session_uuid("session-1")
    b = session_uuid("session-1")
    c = session_uuid("session-2")
    assert a == b
    assert a != c


# ----------------------------------------------------------------------
# event_external_id
# ----------------------------------------------------------------------


def test_event_external_id_short_form():
    assert event_external_id("abc-123") == "adk_event:abc-123"


def test_event_external_id_long_form_hashes():
    long_id = "x" * 600
    out = event_external_id(long_id)
    assert out.startswith("adk_event:h")
    assert len(out) <= 512


# ----------------------------------------------------------------------
# content_to_text / serialise_non_text_parts
# ----------------------------------------------------------------------


def test_content_to_text_concatenates_text_parts():
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text="hello"), genai_types.Part(text="world")],
    )
    assert content_to_text(content) == "hello\nworld"


def test_content_to_text_returns_empty_for_no_text_parts():
    content = genai_types.Content(role="user", parts=[])
    assert content_to_text(content) == ""


def test_content_to_text_handles_none():
    assert content_to_text(None) == ""


def test_serialise_function_call_part():
    content = genai_types.Content(
        role="model",
        parts=[genai_types.Part(function_call=genai_types.FunctionCall(name="lookup", args={"query": "weather"}))],
    )
    out = serialise_non_text_parts(content)
    assert len(out) == 1
    assert out[0]["function_call"]["name"] == "lookup"
    assert out[0]["function_call"]["args"] == {"query": "weather"}


def test_serialise_function_response_part():
    content = genai_types.Content(
        role="user",
        parts=[
            genai_types.Part(
                function_response=genai_types.FunctionResponse(name="lookup", response={"result": "sunny"})
            )
        ],
    )
    out = serialise_non_text_parts(content)
    assert out[0]["function_response"]["name"] == "lookup"
    assert out[0]["function_response"]["response"] == {"result": "sunny"}


def test_serialise_inline_data_drops_bytes_keeps_mime_and_hash():
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(inline_data={"mime_type": "image/png", "data": b"\x89PNG\r\n"})],
    )
    out = serialise_non_text_parts(content)
    assert out[0]["inline_data"]["mime_type"] == "image/png"
    assert out[0]["inline_data"]["data_sha1"] is not None
    # No raw bytes round-tripped.
    assert "data" not in out[0]["inline_data"]


# ----------------------------------------------------------------------
# event_to_remember_kwargs
# ----------------------------------------------------------------------


def test_event_to_remember_kwargs_text_event():
    session = _mk_session()
    event = _text_event("Hello world", ts=1_700_000_000.0)
    kwargs = event_to_remember_kwargs(event, session=session, app_id="google_adk")
    assert kwargs is not None
    assert kwargs["content"] == "Hello world"
    assert kwargs["metadata"][KEY_AUTHOR] == "user"
    assert kwargs["metadata"][KEY_EVENT_ID] == event.id
    assert kwargs["metadata"][KEY_SESSION_ID] == "s1"
    assert kwargs["metadata"][KEY_TIMESTAMP] is not None
    assert kwargs["entity_types"] == []
    assert kwargs["relationship_types"] == []
    assert kwargs["session_id"] == session_uuid("s1")
    assert kwargs["external_id"] == event_external_id(event.id)


def test_event_to_remember_kwargs_skips_empty_events():
    session = _mk_session()
    # Pydantic forbids extra; build an event with empty parts list.
    event = Event(
        author="user",
        content=genai_types.Content(role="user", parts=[]),
        timestamp=time.time(),
    )
    assert event_to_remember_kwargs(event, session=session, app_id="google_adk") is None


def test_event_to_remember_kwargs_function_call_event_uses_placeholder_content():
    session = _mk_session()
    event = Event(
        author="model",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(function_call=genai_types.FunctionCall(name="lookup", args={"q": "x"}))],
        ),
        timestamp=time.time(),
    )
    kwargs = event_to_remember_kwargs(event, session=session, app_id="google_adk")
    assert kwargs is not None
    assert "lookup" in kwargs["content"]
    decoded = json.loads(kwargs["metadata"][KEY_PARTS])
    assert decoded[0]["function_call"]["name"] == "lookup"


# ----------------------------------------------------------------------
# chunk_to_memory_entry — round trip back to ADK shape
# ----------------------------------------------------------------------


def _mk_chunk(*, content: str, custom: dict[str, Any]) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=ChunkMetadata(document_id=uuid4(), custom=custom),
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
    )


def test_chunk_to_memory_entry_text_only():
    chunk = _mk_chunk(
        content="hello world",
        custom={
            KEY_AUTHOR: "user",
            KEY_EVENT_ID: "ev-1",
            KEY_TIMESTAMP: "2026-05-15T00:00:00+00:00",
        },
    )
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=MemoryEntry,
        content_cls=genai_types.Content,
        part_cls=genai_types.Part,
    )
    assert entry.author == "user"
    assert entry.timestamp == "2026-05-15T00:00:00+00:00"
    assert entry.content.parts and entry.content.parts[0].text == "hello world"


def test_chunk_to_memory_entry_with_function_call_part():
    custom = {
        KEY_AUTHOR: "model",
        KEY_EVENT_ID: "ev-2",
        KEY_TIMESTAMP: "2026-05-15T00:00:00+00:00",
        KEY_PARTS: json.dumps([{"function_call": {"name": "lookup", "args": {"q": "weather"}}}]),
    }
    chunk = _mk_chunk(content="tool call: lookup", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=MemoryEntry,
        content_cls=genai_types.Content,
        part_cls=genai_types.Part,
    )
    parts = entry.content.parts or []
    # First part is the text body; second is the function call.
    fc_parts = [p for p in parts if p.function_call is not None]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call.name == "lookup"
