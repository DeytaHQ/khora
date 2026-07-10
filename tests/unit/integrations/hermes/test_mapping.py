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
    derive_session_uuid,
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
    """Same (agent_identity, user_id) → same UUID, every call."""
    ns1 = derive_namespace_uuid("agent-alpha", "user-1")
    ns2 = derive_namespace_uuid("agent-alpha", "user-1")
    assert ns1 == ns2
    assert isinstance(ns1, UUID)


@pytest.mark.unit
def test_derive_namespace_uuid_differs_by_agent_identity() -> None:
    """Different agent_identity → different UUID for the same user."""
    ns_a = derive_namespace_uuid("agent-alpha", "user-1")
    ns_b = derive_namespace_uuid("agent-beta", "user-1")
    assert ns_a != ns_b


@pytest.mark.unit
def test_derive_namespace_uuid_differs_by_user_id() -> None:
    """Different user_id → different UUID for the same agent (per-user isolation)."""
    ns_1 = derive_namespace_uuid("agent-alpha", "user-1")
    ns_2 = derive_namespace_uuid("agent-alpha", "user-2")
    assert ns_1 != ns_2


@pytest.mark.unit
def test_derive_namespace_uuid_ignores_session_id() -> None:
    """Regression (#1466): the namespace is session-independent.

    Two sessions for the same agent MUST resolve to the same namespace,
    otherwise cross-session entity dedup and long-term recall are voided.
    ``derive_namespace_uuid`` no longer takes a session_id at all — this
    pins that the identity-only derivation is stable regardless of which
    session drives it.
    """
    ns_default = derive_namespace_uuid("agent-alpha")
    ns_user_none = derive_namespace_uuid("agent-alpha", None)
    assert ns_default == ns_user_none
    assert isinstance(ns_default, UUID)


@pytest.mark.unit
def test_derive_namespace_uuid_does_not_accept_session_positionally() -> None:
    """The old ``derive_namespace_uuid(agent, session)`` signature is gone.

    A caller that still passes a *session* string as the second positional
    arg now silently binds it to ``user_id`` — this test documents that the
    session value no longer participates in a distinct third dimension, so
    two sessions with an unset user land in one namespace.
    """
    # A value that used to be a distinct `session_id` dimension now binds
    # positionally to `user_id` — passing it explicitly must match.
    ns_positional = derive_namespace_uuid("agent-alpha", "old-session-value")
    ns_as_user_id = derive_namespace_uuid("agent-alpha", user_id="old-session-value")
    assert ns_positional == ns_as_user_id


# ---------------------------------------------------------------------------
# derive_session_uuid
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_derive_session_uuid_is_stable_and_deterministic() -> None:
    """Same free-form session string → same UUID, every call."""
    s1 = derive_session_uuid("demo-session")
    s2 = derive_session_uuid("demo-session")
    assert s1 == s2
    assert isinstance(s1, UUID)


@pytest.mark.unit
def test_derive_session_uuid_differs_by_session_string() -> None:
    """Different session strings → different session UUIDs."""
    assert derive_session_uuid("session-1") != derive_session_uuid("session-2")


@pytest.mark.unit
def test_derive_session_uuid_passes_through_uuid_strings() -> None:
    """A caller-supplied UUID string is honoured verbatim (mirrors google_adk)."""
    raw = uuid4()
    assert derive_session_uuid(str(raw)) == raw


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


@pytest.mark.unit
def test_turn_to_document_threads_session_id_to_first_class_column() -> None:
    """#1466: the Hermes session string maps to khora's first-class session_id.

    Both ingest paths must find it: ``remember`` reads ``Document.session_id``
    (a UUID), ``remember_batch`` coerces top-level ``metadata['session_id']``
    (a UUID string). The raw string stays under ``custom`` for round-trip.
    """
    ns = uuid4()
    doc = turn_to_document(
        "hi",
        "hello",
        session_id="conversation-42",
        turn_seq=1,
        namespace_id=ns,
    )

    expected_session = derive_session_uuid("conversation-42")
    # First-class column set for the single-remember path.
    assert doc.session_id == expected_session
    assert isinstance(doc.session_id, UUID)
    # Top-level metadata UUID string for the batch path's coerce.
    assert doc.metadata["session_id"] == str(expected_session)
    # Raw Hermes string preserved under custom for lossless round-trip.
    assert doc.metadata["custom"][KEY_SESSION_ID] == "conversation-42"


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
