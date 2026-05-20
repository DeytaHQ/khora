"""Unit tests for the Hermes mapping helpers.

The mapping module is pure khora — it never imports anything from
``hermes_agent``. These tests run without the ``hermes`` extra installed
and without any Khora, database, or asyncio plumbing.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from khora.integrations.hermes._mapping import (
    KEY_ASSISTANT_CONTENT,
    KEY_EXTERNAL_ID,
    KEY_OAI_SEQ,
    KEY_OCCURRED_AT,
    KEY_SESSION_ID,
    KEY_SOURCE,
    KEY_TURN_SEQ,
    KEY_USER_CONTENT,
    derive_namespace_uuid,
    format_memory_context,
    message_pair_iter,
    turn_external_id,
    turn_to_document,
)

# ---------------------------------------------------------------------------
# derive_namespace_uuid
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_derive_namespace_uuid_is_stable_across_calls() -> None:
    """Same (agent_identity, session_id) → same UUID, every call."""
    ns1 = derive_namespace_uuid("agent-alpha", "session-123")
    ns2 = derive_namespace_uuid("agent-alpha", "session-123")
    assert ns1 == ns2
    assert isinstance(ns1, UUID)


@pytest.mark.unit
def test_derive_namespace_uuid_differs_by_agent_identity() -> None:
    """Different agent_identity → different UUID for the same session."""
    ns_a = derive_namespace_uuid("agent-alpha", "session-123")
    ns_b = derive_namespace_uuid("agent-beta", "session-123")
    assert ns_a != ns_b


@pytest.mark.unit
def test_derive_namespace_uuid_differs_by_session_id() -> None:
    """Different session_id → different UUID for the same agent."""
    ns_1 = derive_namespace_uuid("agent-alpha", "session-1")
    ns_2 = derive_namespace_uuid("agent-alpha", "session-2")
    assert ns_1 != ns_2


# ---------------------------------------------------------------------------
# turn_external_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_turn_external_id_shape() -> None:
    """Format is ``hermes:{session_id}:{turn_seq}`` — regression-locked."""
    assert turn_external_id("session-abc", 0) == "hermes:session-abc:0"
    assert turn_external_id("session-abc", 42) == "hermes:session-abc:42"


# ---------------------------------------------------------------------------
# turn_to_document
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_turn_to_document_populates_metadata_and_namespace() -> None:
    """User+assistant content, external_id, namespace, and custom keys round-trip correctly."""
    ns = uuid4()
    doc = turn_to_document(
        "hello there",
        "general kenobi",
        session_id="session-xyz",
        turn_seq=3,
        namespace_id=ns,
    )

    assert doc.namespace_id == ns
    assert doc.content == "USER: hello there\n\nASSISTANT: general kenobi"
    assert doc.source_type == "conversation"

    custom = doc.metadata["custom"]
    assert custom[KEY_EXTERNAL_ID] == "hermes:session-xyz:3"
    assert custom[KEY_SOURCE] == "hermes"
    assert custom[KEY_SESSION_ID] == "session-xyz"
    assert custom[KEY_TURN_SEQ] == 3
    assert custom[KEY_USER_CONTENT] == "hello there"
    assert custom[KEY_ASSISTANT_CONTENT] == "general kenobi"
    assert custom[KEY_OAI_SEQ] == 3
    assert isinstance(custom[KEY_OCCURRED_AT], str)
    # ISO 8601 with timezone offset — sanity check it parses back.
    assert "T" in custom[KEY_OCCURRED_AT]


# ---------------------------------------------------------------------------
# format_memory_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_memory_context_returns_wrapper_tags() -> None:
    """Output is wrapped in ``<memory-context>...</memory-context>`` even when empty."""
    out = format_memory_context([])
    assert out.startswith("<memory-context>")
    assert out.endswith("</memory-context>")


@pytest.mark.unit
def test_format_memory_context_includes_first_chunk_content() -> None:
    """Placeholder body surfaces the first chunk's content until the AI Engineer's body lands."""

    class _StubChunk:
        content = "remembered fact"

    out = format_memory_context([_StubChunk()])  # type: ignore[list-item]
    assert "remembered fact" in out
    assert out.startswith("<memory-context>")
    assert out.endswith("</memory-context>")


# ---------------------------------------------------------------------------
# message_pair_iter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_message_pair_iter_handles_normal_alternating_messages() -> None:
    """U/A/U/A pairs cleanly into two tuples."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert list(message_pair_iter(messages)) == [("u1", "a1"), ("u2", "a2")]


@pytest.mark.unit
def test_message_pair_iter_handles_dangling_user() -> None:
    """Trailing user message with no assistant reply yields ``(user, "")``."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    assert list(message_pair_iter(messages)) == [("u1", "a1"), ("u2", "")]


@pytest.mark.unit
def test_message_pair_iter_handles_dangling_assistant() -> None:
    """Opening assistant message with no prior user yields ``("", assistant)``."""
    messages = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert list(message_pair_iter(messages)) == [("", "a0"), ("u1", "a1")]


@pytest.mark.unit
def test_message_pair_iter_skips_system_and_tool_messages() -> None:
    """System / tool messages are filtered out, U/A pairing continues across them."""
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "u1"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "a1"},
    ]
    assert list(message_pair_iter(messages)) == [("u1", "a1")]


@pytest.mark.unit
def test_message_pair_iter_empty_list() -> None:
    """Empty input yields no pairs."""
    assert list(message_pair_iter([])) == []
