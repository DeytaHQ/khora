"""Unit tests for the OpenAI Agents SDK mapping helpers.

The mapping module is duck-typed — it never imports anything from the
SDK. These tests exercise the round-trip from a dict-shaped
``TResponseInputItem`` to khora kwargs and back without requiring the
``openai-agents`` extra to be installed.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.integrations.openai_agents._mapping import (
    KEY_ITEM_JSON,
    KEY_SEQ,
    KEY_SESSION_ID,
    chunk_seq,
    chunk_to_item,
    event_external_id,
    item_text,
    item_to_remember_kwargs,
    session_uuid,
)


def _make_chunk(custom: dict[str, Any]) -> dict[str, Any]:
    """Return the document-level metadata dict the mapping helpers consume.

    ``chunk_to_item`` / ``chunk_seq`` now take a plain dict (joined via
    ``DocumentProjection.metadata``) rather than the chunk object itself.
    """
    return dict(custom)


# ---------------------------------------------------------------------------
# session_uuid
# ---------------------------------------------------------------------------


def test_session_uuid_round_trips_uuid_string_verbatim() -> None:
    """A session_id that already parses as a UUID is returned verbatim."""
    sid = uuid4()
    assert session_uuid(str(sid)) == sid


def test_session_uuid_is_deterministic_for_non_uuid_strings() -> None:
    """Non-UUID session ids hash through UUID5 deterministically."""
    a = session_uuid("conversation-42")
    b = session_uuid("conversation-42")
    assert a == b
    assert isinstance(a, UUID)
    # Distinct inputs produce distinct outputs.
    assert session_uuid("conversation-42") != session_uuid("conversation-43")


# ---------------------------------------------------------------------------
# event_external_id
# ---------------------------------------------------------------------------


def test_event_external_id_packs_session_and_seq() -> None:
    eid = event_external_id("conv-1", 7)
    assert eid == "oai:conv-1:7"


def test_event_external_id_hashes_long_session_ids() -> None:
    """Pathological session ids must still fit under the 512-char DB cap."""
    long_session = "x" * 1024
    eid = event_external_id(long_session, 3)
    assert eid.startswith("oai:h")
    assert eid.endswith(":3")
    assert len(eid) <= 512


# ---------------------------------------------------------------------------
# item_text
# ---------------------------------------------------------------------------


def test_item_text_renders_role_plus_string_content() -> None:
    assert item_text({"role": "user", "content": "hello"}) == "user: hello"


def test_item_text_concatenates_list_content_parts() -> None:
    item = {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "alpha"},
            {"type": "output_text", "text": "beta"},
        ],
    }
    assert item_text(item) == "assistant: alpha\nbeta"


def test_item_text_renders_function_call_summary() -> None:
    item = {"type": "function_call", "name": "lookup", "arguments": '{"q":1}'}
    assert item_text(item) == 'tool call: lookup({"q":1})'


def test_item_text_truncates_long_function_outputs() -> None:
    payload = "x" * 3000
    item = {"type": "function_call_output", "output": payload}
    out = item_text(item)
    assert out.startswith("tool result: ")
    # 2000 chars of payload + ellipsis cap.
    assert len(out) <= len("tool result: ") + 2001


def test_item_text_falls_back_to_json_for_unknown_shape() -> None:
    out = item_text({"weird": "thing"})
    # Falls back to a JSON-encoded dump so the chunk isn't empty.
    decoded = json.loads(out)
    assert decoded == {"weird": "thing"}


# ---------------------------------------------------------------------------
# item_to_remember_kwargs
# ---------------------------------------------------------------------------


def test_item_to_remember_kwargs_round_trips_item_json() -> None:
    item = {"role": "user", "content": "hello"}
    kwargs = item_to_remember_kwargs(item, session_id="conv-1", app_id="myapp", seq=0)

    assert kwargs["content"] == "user: hello"
    assert kwargs["external_id"] == "oai:conv-1:0"
    assert kwargs["session_id"] == session_uuid("conv-1")
    assert kwargs["entity_types"] == []  # no extraction on conversation turns
    assert kwargs["relationship_types"] == []

    meta = kwargs["metadata"]
    assert meta[KEY_SESSION_ID] == "conv-1"
    assert meta[KEY_SEQ] == 0
    # The verbatim JSON is preserved so chunk_to_item can recover it.
    assert json.loads(meta[KEY_ITEM_JSON]) == item


def test_item_to_remember_kwargs_handles_function_call_payload() -> None:
    item = {"type": "function_call", "name": "lookup", "arguments": "{}"}
    kwargs = item_to_remember_kwargs(item, session_id="conv-1", app_id="myapp", seq=3)
    assert kwargs["external_id"] == "oai:conv-1:3"
    assert kwargs["content"] == "tool call: lookup({})"
    # Round-trip-friendly metadata.
    assert json.loads(kwargs["metadata"][KEY_ITEM_JSON]) == item


# ---------------------------------------------------------------------------
# chunk_to_item / chunk_seq
# ---------------------------------------------------------------------------


def test_chunk_to_item_recovers_original_payload() -> None:
    original = {"role": "user", "content": "hello"}
    chunk = _make_chunk(
        {
            KEY_ITEM_JSON: json.dumps(original),
            KEY_SEQ: 0,
            KEY_SESSION_ID: "conv-1",
        }
    )
    assert chunk_to_item(chunk) == original


def test_chunk_to_item_returns_none_for_foreign_chunk() -> None:
    """A chunk that wasn't written by this adapter must round-trip to None."""
    chunk = _make_chunk({})
    assert chunk_to_item(chunk) is None


def test_chunk_to_item_returns_none_for_corrupt_json() -> None:
    chunk = _make_chunk({KEY_ITEM_JSON: "{not json"})
    assert chunk_to_item(chunk) is None


def test_chunk_to_item_accepts_already_decoded_payload() -> None:
    """Some backends may surface JSONB columns as dicts already."""
    payload = {"role": "user", "content": "hi"}
    chunk = _make_chunk({KEY_ITEM_JSON: payload})
    assert chunk_to_item(chunk) == payload


@pytest.mark.parametrize(("stored", "expected"), [(0, 0), (12, 12), ("42", 42), (None, None), ("nope", None)])
def test_chunk_seq_parses_stamped_value(stored: Any, expected: int | None) -> None:
    chunk = _make_chunk({KEY_SEQ: stored} if stored is not None else {})
    assert chunk_seq(chunk) == expected
