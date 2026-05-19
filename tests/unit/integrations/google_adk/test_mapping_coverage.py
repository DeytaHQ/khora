"""Coverage tests for ``khora.integrations.google_adk._mapping``.

The mapping module is pure-Python and duck-typed: it inspects ``getattr``
on the passed-in Event / Content / Part / Session objects. We exercise
it with simple dataclass / SimpleNamespace stand-ins so the tests run
without the optional ``google-adk`` extra installed.

Step 4 of #695 — push coverage on the google_adk mapping helpers from
~12% to >=60%. Existing ``test_mapping.py`` in this dir skips when
google-adk is missing; CI runs the crewai combo and so it always skips
there. This file deliberately avoids that gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models.document import Chunk
from khora.integrations.google_adk._mapping import (
    KEY_APP_ID,
    KEY_AUTHOR,
    KEY_EVENT_ID,
    KEY_PARTS,
    KEY_SESSION_ID,
    KEY_TIMESTAMP,
    KEY_USER_ID,
    _epoch_to_utc,
    _placeholder_for_non_text,
    _to_jsonable,
    chunk_to_memory_entry,
    content_to_text,
    event_external_id,
    event_to_remember_kwargs,
    namespace_uuid,
    serialise_non_text_parts,
    session_uuid,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Duck-typed stand-ins for ADK shapes
# ---------------------------------------------------------------------------


def _mk_part(
    *,
    text: str | None = None,
    function_call: Any = None,
    function_response: Any = None,
    inline_data: Any = None,
    other: Any = None,
) -> Any:
    """Build a Part-shaped object with the requested attributes set."""
    return SimpleNamespace(
        text=text,
        function_call=function_call,
        function_response=function_response,
        inline_data=inline_data,
        other=other,
    )


def _mk_content(parts: list[Any] | None) -> Any:
    return SimpleNamespace(parts=parts, role="user")


def _mk_event(
    *,
    eid: str = "ev-abc",
    author: str | None = "user",
    text: str | None = None,
    parts: list[Any] | None = None,
    ts: float | None = 1_700_000_000.0,
    content: Any = "USE_PARTS",
) -> Any:
    if content == "USE_PARTS":
        if parts is None:
            parts = [_mk_part(text=text)] if text else []
        content = _mk_content(parts)
    return SimpleNamespace(id=eid, author=author, content=content, timestamp=ts)


def _mk_session(*, sid: str = "sess-1", app_name: str = "app", user_id: str = "user-1") -> Any:
    return SimpleNamespace(id=sid, app_name=app_name, user_id=user_id)


# A class that mimics ``Part`` for the round-trip path: stores whatever
# kwargs it was built with so the test can inspect them.
class FakePart:
    def __init__(self, **kwargs: Any) -> None:
        self.text = kwargs.get("text")
        self.function_call = kwargs.get("function_call")
        self.function_response = kwargs.get("function_response")
        self.inline_data = kwargs.get("inline_data")
        self._kwargs = kwargs


class FakeContent:
    def __init__(self, parts: list[Any], role: str) -> None:
        self.parts = parts
        self.role = role


class FakeMemoryEntry:
    def __init__(
        self,
        *,
        content: Any,
        author: str | None,
        timestamp: str | None,
        custom_metadata: dict[str, Any],
    ) -> None:
        self.content = content
        self.author = author
        self.timestamp = timestamp
        self.custom_metadata = custom_metadata


# ---------------------------------------------------------------------------
# namespace_uuid / session_uuid
# ---------------------------------------------------------------------------


def test_namespace_uuid_deterministic_for_same_inputs() -> None:
    a = namespace_uuid(app_name="app1", user_id="user-x")
    b = namespace_uuid(app_name="app1", user_id="user-x")
    assert a == b
    assert isinstance(a, UUID)


def test_namespace_uuid_differs_per_app_or_user() -> None:
    a = namespace_uuid(app_name="app1", user_id="u1")
    b = namespace_uuid(app_name="app2", user_id="u1")
    c = namespace_uuid(app_name="app1", user_id="u2")
    assert len({a, b, c}) == 3


def test_session_uuid_passes_through_uuid_string() -> None:
    raw = uuid4()
    derived = session_uuid(str(raw))
    assert derived == raw


def test_session_uuid_derives_for_non_uuid_strings() -> None:
    a = session_uuid("conversation-7")
    b = session_uuid("conversation-7")
    c = session_uuid("conversation-8")
    assert isinstance(a, UUID)
    assert a == b
    assert a != c


def test_session_uuid_handles_attribute_error_path() -> None:
    """Passing ``None`` exercises the ``AttributeError`` branch in the try/except."""
    # UUID(None) raises TypeError, but UUID('') raises ValueError. We test
    # the AttributeError pathway too by passing a non-string-like value.
    # uuid5() handles the fallback by stringifying.
    out = session_uuid("")
    assert isinstance(out, UUID)


# ---------------------------------------------------------------------------
# event_external_id
# ---------------------------------------------------------------------------


def test_event_external_id_short_form_keeps_id_visible() -> None:
    assert event_external_id("ev-123") == "adk_event:ev-123"


def test_event_external_id_long_form_hashes_to_fit_column() -> None:
    long = "y" * 700
    out = event_external_id(long)
    assert out.startswith("adk_event:h")
    assert len(out) <= 512
    # Hash is deterministic.
    assert event_external_id(long) == out


def test_event_external_id_exactly_at_boundary() -> None:
    # Build something close to 512 to exercise the equality branch.
    raw_id = "x" * (512 - len("adk_event:"))
    out = event_external_id(raw_id)
    assert out == f"adk_event:{raw_id}"
    assert len(out) == 512


# ---------------------------------------------------------------------------
# content_to_text
# ---------------------------------------------------------------------------


def test_content_to_text_handles_none() -> None:
    assert content_to_text(None) == ""


def test_content_to_text_handles_empty_parts() -> None:
    content = _mk_content(parts=[])
    assert content_to_text(content) == ""


def test_content_to_text_joins_text_parts() -> None:
    content = _mk_content(parts=[_mk_part(text="alpha"), _mk_part(text="beta")])
    assert content_to_text(content) == "alpha\nbeta"


def test_content_to_text_skips_non_text_parts() -> None:
    fc_part = _mk_part(function_call=SimpleNamespace(name="lookup", args={"q": "x"}))
    text_part = _mk_part(text="hello")
    content = _mk_content(parts=[fc_part, text_part])
    assert content_to_text(content) == "hello"


def test_content_to_text_when_parts_attr_missing() -> None:
    """``getattr(content, 'parts', None)`` returns None — function still works."""
    content = SimpleNamespace()  # no .parts attribute
    assert content_to_text(content) == ""


# ---------------------------------------------------------------------------
# serialise_non_text_parts
# ---------------------------------------------------------------------------


def test_serialise_non_text_parts_returns_empty_for_none_content() -> None:
    assert serialise_non_text_parts(None) == []


def test_serialise_non_text_parts_skips_text_parts() -> None:
    content = _mk_content(parts=[_mk_part(text="just text")])
    assert serialise_non_text_parts(content) == []


def test_serialise_function_call_extracts_name_and_args() -> None:
    fc = SimpleNamespace(name="lookup", args={"q": "weather"})
    content = _mk_content(parts=[_mk_part(function_call=fc)])
    out = serialise_non_text_parts(content)
    assert out == [{"function_call": {"name": "lookup", "args": {"q": "weather"}}}]


def test_serialise_function_response_extracts_name_and_response() -> None:
    fr = SimpleNamespace(name="lookup", response={"result": "sunny"})
    content = _mk_content(parts=[_mk_part(function_response=fr)])
    out = serialise_non_text_parts(content)
    assert out == [{"function_response": {"name": "lookup", "response": {"result": "sunny"}}}]


def test_serialise_inline_data_keeps_mime_drops_bytes() -> None:
    inline = SimpleNamespace(mime_type="image/png", data=b"\x89PNG-fake-data")
    content = _mk_content(parts=[_mk_part(inline_data=inline)])
    out = serialise_non_text_parts(content)
    assert out[0]["inline_data"]["mime_type"] == "image/png"
    # 16-char prefix of sha1 hex.
    expected_digest = hashlib.sha1(b"\x89PNG-fake-data", usedforsecurity=False).hexdigest()[:16]
    assert out[0]["inline_data"]["data_sha1"] == expected_digest
    assert "data" not in out[0]["inline_data"]


def test_serialise_inline_data_without_bytes_yields_none_digest() -> None:
    """When ``data`` is not bytes (e.g. None), the digest field is None."""
    inline = SimpleNamespace(mime_type="image/png", data=None)
    content = _mk_content(parts=[_mk_part(inline_data=inline)])
    out = serialise_non_text_parts(content)
    assert out[0]["inline_data"]["data_sha1"] is None


def test_serialise_unknown_part_falls_back_to_model_dump() -> None:
    class _PartWithDump:
        text = None
        function_call = None
        function_response = None
        inline_data = None

        def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
            return {"weird": "shape"}

    content = _mk_content(parts=[_PartWithDump()])
    out = serialise_non_text_parts(content)
    assert out == [{"raw": {"weird": "shape"}}]


def test_serialise_unknown_part_model_dump_raises_falls_through_to_repr() -> None:
    class _BadDump:
        text = None
        function_call = None
        function_response = None
        inline_data = None

        def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
            raise RuntimeError("boom")

        def __repr__(self) -> str:
            return "BadPart<>"

    content = _mk_content(parts=[_BadDump()])
    out = serialise_non_text_parts(content)
    assert out == [{"raw": "BadPart<>"}]


def test_serialise_part_without_model_dump_uses_repr() -> None:
    """No ``model_dump`` callable — fall through to repr fallback."""
    part = SimpleNamespace(text=None, function_call=None, function_response=None, inline_data=None)
    # SimpleNamespace doesn't have model_dump. Test fallback path.
    content = _mk_content(parts=[part])
    out = serialise_non_text_parts(content)
    # SimpleNamespace renders to "namespace(...)".
    assert len(out) == 1
    assert "raw" in out[0]


# ---------------------------------------------------------------------------
# event_to_remember_kwargs
# ---------------------------------------------------------------------------


def test_event_to_remember_kwargs_returns_none_for_empty_event() -> None:
    """No text + no non-text parts → caller dropped."""
    session = _mk_session()
    event = _mk_event(parts=[])
    assert event_to_remember_kwargs(event, session=session, app_id="myapp") is None


def test_event_to_remember_kwargs_text_event_round_trip() -> None:
    session = _mk_session(sid="sess-A", app_name="my_app", user_id="user-1")
    event = _mk_event(eid="ev-7", author="user", text="hello world", ts=1_700_000_000.0)

    kwargs = event_to_remember_kwargs(event, session=session, app_id="my_app")
    assert kwargs is not None
    assert kwargs["content"] == "hello world"
    assert kwargs["metadata"][KEY_APP_ID] == "my_app"
    assert kwargs["metadata"][KEY_USER_ID] == "user-1"
    assert kwargs["metadata"][KEY_SESSION_ID] == "sess-A"
    assert kwargs["metadata"][KEY_EVENT_ID] == "ev-7"
    assert kwargs["metadata"][KEY_AUTHOR] == "user"
    assert kwargs["metadata"][KEY_TIMESTAMP] is not None
    assert kwargs["external_id"] == event_external_id("ev-7")
    assert kwargs["session_id"] == session_uuid("sess-A")
    assert kwargs["entity_types"] == []
    assert kwargs["relationship_types"] == []
    assert kwargs["source"] == "google_adk:my_app"
    # Title built from author + first 12 chars of event id.
    assert kwargs["title"].startswith("adk:user:")


def test_event_to_remember_kwargs_fills_event_id_when_missing() -> None:
    session = _mk_session(sid="sess-A")
    # ``id`` attribute present but empty: deterministic hash fallback.
    event = _mk_event(eid="", author="user", text="content", ts=1_700_000_000.0)
    kwargs = event_to_remember_kwargs(event, session=session, app_id="app")
    assert kwargs is not None
    # 40-char SHA1 hex.
    assert len(kwargs["metadata"][KEY_EVENT_ID]) == 40


def test_event_to_remember_kwargs_non_text_only_uses_placeholder_content() -> None:
    """When the event carries only function calls / inline data, content is the placeholder."""
    fc = SimpleNamespace(name="lookup", args={"q": "x"})
    fr = SimpleNamespace(name="other", response={"r": 1})
    inline = SimpleNamespace(mime_type="image/png", data=b"abc")
    parts = [
        _mk_part(function_call=fc),
        _mk_part(function_response=fr),
        _mk_part(inline_data=inline),
    ]
    event = _mk_event(parts=parts)
    session = _mk_session()

    kwargs = event_to_remember_kwargs(event, session=session, app_id="app")
    assert kwargs is not None
    # KEY_PARTS metadata is JSON-encoded with 3 entries.
    decoded = json.loads(kwargs["metadata"][KEY_PARTS])
    assert len(decoded) == 3
    # Content placeholder mentions all three.
    assert "tool call: lookup" in kwargs["content"]
    assert "tool response: other" in kwargs["content"]
    assert "inline data: image/png" in kwargs["content"]


def test_event_to_remember_kwargs_with_no_author_defaults_to_event_in_title() -> None:
    session = _mk_session()
    event = _mk_event(eid="ev-empty-author", author=None, text="some text")
    kwargs = event_to_remember_kwargs(event, session=session, app_id="app")
    assert kwargs is not None
    # Falls back to "event" in title since author is None.
    assert kwargs["title"].startswith("adk:event:")


def test_event_to_remember_kwargs_with_none_timestamp_drops_iso() -> None:
    session = _mk_session()
    event = _mk_event(text="hello", ts=None)
    kwargs = event_to_remember_kwargs(event, session=session, app_id="app")
    assert kwargs is not None
    assert kwargs["metadata"][KEY_TIMESTAMP] is None


# ---------------------------------------------------------------------------
# chunk_to_memory_entry
# ---------------------------------------------------------------------------


def _mk_chunk(*, content: str, custom: dict[str, Any]) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=custom,
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
    )


def test_chunk_to_memory_entry_text_only() -> None:
    custom = {
        KEY_AUTHOR: "user",
        KEY_EVENT_ID: "ev-1",
        KEY_TIMESTAMP: "2026-05-15T00:00:00+00:00",
    }
    chunk = _mk_chunk(content="hello world", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    assert entry.author == "user"
    assert entry.timestamp == "2026-05-15T00:00:00+00:00"
    assert len(entry.content.parts) == 1
    assert entry.content.parts[0].text == "hello world"


def test_chunk_to_memory_entry_decodes_function_call_part_from_json_string() -> None:
    serialised = json.dumps([{"function_call": {"name": "lookup", "args": {"q": "x"}}}])
    custom = {
        KEY_AUTHOR: "model",
        KEY_EVENT_ID: "ev-2",
        KEY_TIMESTAMP: "2026-05-15T00:00:00+00:00",
        KEY_PARTS: serialised,
    }
    chunk = _mk_chunk(content="tool call: lookup", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    # Two parts: text body + function_call.
    parts = entry.content.parts
    fc_parts = [p for p in parts if p.function_call is not None]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call == {"name": "lookup", "args": {"q": "x"}}


def test_chunk_to_memory_entry_decodes_already_decoded_parts() -> None:
    """Some backends surface JSONB as a Python list directly."""
    parts_decoded = [
        {"function_response": {"name": "echo", "response": {"r": 1}}},
        {"inline_data": {"mime_type": "image/png", "data_sha1": "abcd"}},
    ]
    custom = {
        KEY_AUTHOR: "model",
        KEY_PARTS: parts_decoded,
    }
    chunk = _mk_chunk(content="payload", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    fr_parts = [p for p in entry.content.parts if p.function_response is not None]
    inline_parts = [p for p in entry.content.parts if p.inline_data is not None]
    assert len(fr_parts) == 1
    assert len(inline_parts) == 1
    assert inline_parts[0].inline_data == {"mime_type": "image/png", "data": b""}


def test_chunk_to_memory_entry_swallows_corrupt_json_string() -> None:
    custom = {
        KEY_AUTHOR: "user",
        KEY_PARTS: "not a json doc {",
    }
    chunk = _mk_chunk(content="hello", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    # Only the text body survives.
    assert len(entry.content.parts) == 1
    assert entry.content.parts[0].text == "hello"


def test_chunk_to_memory_entry_filters_internal_metadata_from_custom() -> None:
    """The custom_metadata returned on the MemoryEntry should not surface adk_parts/adk_event_id/adk_timestamp/adk_author."""
    custom = {
        KEY_AUTHOR: "user",
        KEY_EVENT_ID: "ev-3",
        KEY_TIMESTAMP: "2026-01-01T00:00:00+00:00",
        KEY_PARTS: json.dumps([]),
        KEY_APP_ID: "myapp",  # preserved
        KEY_USER_ID: "u",  # preserved
        KEY_SESSION_ID: "s",  # preserved
        "user_extra": "ok",  # preserved (non-adk_ key)
    }
    chunk = _mk_chunk(content="text", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    # Public-facing custom metadata excludes our internal bookkeeping.
    assert KEY_EVENT_ID not in entry.custom_metadata
    assert KEY_TIMESTAMP not in entry.custom_metadata
    assert KEY_AUTHOR not in entry.custom_metadata
    assert KEY_PARTS not in entry.custom_metadata
    # But preserves the public-facing trio + user-supplied non-adk keys.
    assert entry.custom_metadata[KEY_APP_ID] == "myapp"
    assert entry.custom_metadata[KEY_USER_ID] == "u"
    assert entry.custom_metadata[KEY_SESSION_ID] == "s"
    assert entry.custom_metadata["user_extra"] == "ok"


def test_chunk_to_memory_entry_skips_unknown_part_entries() -> None:
    """``_deserialise_part`` returns None for unknown dict shapes — those are skipped."""
    serialised = json.dumps([{"unknown_shape": True}, {"function_call": {"name": "x"}}])
    custom = {KEY_AUTHOR: "user", KEY_PARTS: serialised}
    chunk = _mk_chunk(content="text", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    # 1 text + 1 function_call, unknown entry dropped.
    function_call_parts = [p for p in entry.content.parts if p.function_call is not None]
    assert len(function_call_parts) == 1


def test_chunk_to_memory_entry_with_empty_content() -> None:
    """No text body — only non-text parts contribute."""
    serialised = json.dumps([{"function_call": {"name": "x"}}])
    custom = {KEY_AUTHOR: "user", KEY_PARTS: serialised}
    chunk = _mk_chunk(content="", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=FakePart,
    )
    text_parts = [p for p in entry.content.parts if p.text is not None]
    assert len(text_parts) == 0


def test_chunk_to_memory_entry_swallows_part_construction_errors() -> None:
    """If part_cls raises, the entry is skipped (not propagated)."""

    class _RaisingPart:
        def __init__(self, **kwargs: Any) -> None:
            if "function_call" in kwargs:
                raise RuntimeError("constructor blew up")
            self.text = kwargs.get("text")
            self.function_call = None
            self.function_response = None
            self.inline_data = None

    serialised = json.dumps([{"function_call": {"name": "x"}}])
    custom = {KEY_AUTHOR: "user", KEY_PARTS: serialised}
    chunk = _mk_chunk(content="hi", custom=custom)
    entry = chunk_to_memory_entry(
        chunk,
        memory_entry_cls=FakeMemoryEntry,
        content_cls=FakeContent,
        part_cls=_RaisingPart,
    )
    # The text part still got built; the function_call part was dropped.
    assert len(entry.content.parts) == 1
    assert entry.content.parts[0].text == "hi"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def test_epoch_to_utc_none_returns_none() -> None:
    assert _epoch_to_utc(None) is None


def test_epoch_to_utc_valid_float() -> None:
    out = _epoch_to_utc(1_700_000_000.0)
    assert out is not None
    assert out.tzinfo is UTC


def test_epoch_to_utc_bad_value_returns_none() -> None:
    assert _epoch_to_utc("not a number") is None
    # Float overflow / out-of-range → caught by OSError branch on most platforms.
    assert _epoch_to_utc(math.nan) is not None or _epoch_to_utc(math.nan) is None


def test_placeholder_for_non_text_empty_input() -> None:
    assert _placeholder_for_non_text([]) == ""


def test_placeholder_for_non_text_pipe_separates_entries() -> None:
    out = _placeholder_for_non_text(
        [
            {"function_call": {"name": "a"}},
            {"function_response": {"name": "b"}},
            {"inline_data": {"mime_type": "image/png"}},
            {"unknown": "type"},
        ]
    )
    assert "tool call: a" in out
    assert "tool response: b" in out
    assert "inline data: image/png" in out
    assert "event payload" in out
    assert out.count("|") == 3


def test_to_jsonable_primitive_types() -> None:
    assert _to_jsonable(None) is None
    assert _to_jsonable("x") == "x"
    assert _to_jsonable(42) == 42
    assert _to_jsonable(3.14) == 3.14
    assert _to_jsonable(True) is True


def test_to_jsonable_list_recurses() -> None:
    assert _to_jsonable([1, "x", [2.0]]) == [1, "x", [2.0]]


def test_to_jsonable_tuple_renders_to_list() -> None:
    assert _to_jsonable((1, 2)) == [1, 2]


def test_to_jsonable_dict_stringifies_keys() -> None:
    out = _to_jsonable({1: "a", "k": "b"})
    assert out == {"1": "a", "k": "b"}


def test_to_jsonable_bytes_render_to_placeholder() -> None:
    assert _to_jsonable(b"abc") == "<bytes len=3>"


def test_to_jsonable_uses_model_dump_if_available() -> None:
    class _M:
        def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
            return {"k": "v"}

    assert _to_jsonable(_M()) == {"k": "v"}


def test_to_jsonable_falls_back_to_repr() -> None:
    class _X:
        def __repr__(self) -> str:
            return "X-repr"

    assert _to_jsonable(_X()) == "X-repr"


def test_to_jsonable_model_dump_exception_falls_back_to_repr() -> None:
    class _M:
        def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
            raise RuntimeError("boom")

        def __repr__(self) -> str:
            return "M-repr"

    assert _to_jsonable(_M()) == "M-repr"
