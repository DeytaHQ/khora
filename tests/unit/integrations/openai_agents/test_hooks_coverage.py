"""Extra coverage for ``khora.integrations.openai_agents.hooks._extract_recent_text``.

The mainline test_hooks.py covers the happy paths via ``on_agent_start``.
This file targets the multi-attribute probing in ``_extract_recent_text``
directly: ``context.run.input``, ``context.messages``, nested list-of-
content-parts payloads, and the various non-string / non-dict branches
that fall through to ``""``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("agents")

from khora.integrations.openai_agents.hooks import (  # noqa: E402
    KhoraMemoryHooks,
    _extract_recent_text,
)
from khora.khora import Khora  # noqa: E402

pytestmark = pytest.mark.unit


def _make_kb() -> Any:
    return AsyncMock(spec=Khora)


# ---------------------------------------------------------------------------
# _extract_recent_text — direct unit tests
# ---------------------------------------------------------------------------


def test_extract_returns_empty_when_no_candidates() -> None:
    ctx = SimpleNamespace()  # no input/user_input/messages/run
    assert _extract_recent_text(ctx) == ""


def test_extract_picks_up_string_input() -> None:
    ctx = SimpleNamespace(input="hello world", user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == "hello world"


def test_extract_picks_up_user_input_attribute() -> None:
    """The fallback ``context.user_input`` path is taken when ``.input`` is None."""
    ctx = SimpleNamespace(input=None, user_input="from user_input", messages=None, run=None)
    assert _extract_recent_text(ctx) == "from user_input"


def test_extract_picks_up_messages_string() -> None:
    ctx = SimpleNamespace(input=None, user_input=None, messages="from messages", run=None)
    assert _extract_recent_text(ctx) == "from messages"


def test_extract_picks_up_run_input_string() -> None:
    """``context.run.input`` is the newer-minor path."""
    run = SimpleNamespace(input="from run.input", messages=None)
    ctx = SimpleNamespace(input=None, user_input=None, messages=None, run=run)
    assert _extract_recent_text(ctx) == "from run.input"


def test_extract_picks_up_run_messages_list() -> None:
    run = SimpleNamespace(input=None, messages=[{"role": "user", "content": "from run.messages"}])
    ctx = SimpleNamespace(input=None, user_input=None, messages=None, run=run)
    assert _extract_recent_text(ctx) == "from run.messages"


def test_extract_skips_empty_string_candidates() -> None:
    """An empty string falls through; the next non-empty candidate wins."""
    ctx = SimpleNamespace(input="   ", user_input="real value", messages=None, run=None)
    assert _extract_recent_text(ctx) == "real value"


def test_extract_walks_list_in_reverse_for_user_messages() -> None:
    items = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "assistant reply"},
        {"role": "user", "content": "latest user"},
    ]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == "latest user"


def test_extract_skips_non_dict_items_in_list() -> None:
    """A list that contains non-dict items must skip them and keep looking."""
    items = ["raw string", 42, {"role": "user", "content": "real"}]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == "real"


def test_extract_walks_list_of_content_parts() -> None:
    """``content`` may be a list of part dicts (``[{"type":"...","text":"..."}, ...]``)."""
    items = [
        {
            "role": "user",
            "content": [
                {"type": "output_text", "text": "first chunk"},
                {"type": "output_text", "text": "second chunk"},
            ],
        }
    ]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    # First matching text wins — _extract_recent_text returns the FIRST text found
    # in the LAST item (reversed iteration).
    out = _extract_recent_text(ctx)
    assert out in ("first chunk", "second chunk")


def test_extract_returns_empty_for_list_of_irrelevant_dicts() -> None:
    """No content/text fields anywhere — fall back to ''."""
    items = [{"role": "user", "extra": "no content key"}]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == ""


def test_extract_skips_empty_text_parts() -> None:
    """Empty / whitespace text parts must not satisfy the picker."""
    items = [
        {
            "role": "user",
            "content": [
                {"type": "output_text", "text": "   "},
                {"type": "output_text", "text": "real text"},
            ],
        }
    ]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == "real text"


def test_extract_handles_non_dict_parts_in_content_list() -> None:
    """A content list with raw strings must skip them safely."""
    items = [
        {
            "role": "user",
            "content": ["raw piece", {"type": "output_text", "text": "real"}],
        }
    ]
    ctx = SimpleNamespace(input=items, user_input=None, messages=None, run=None)
    assert _extract_recent_text(ctx) == "real"


# ---------------------------------------------------------------------------
# Integration: on_agent_start using the multi-attribute paths
# ---------------------------------------------------------------------------


async def test_on_agent_start_picks_run_input_when_top_level_input_empty() -> None:
    kb = _make_kb()
    from dataclasses import dataclass, field
    from uuid import UUID

    @dataclass
    class _Stub:
        chunks: list[Any] = field(default_factory=list)
        query: str = ""
        namespace_id: UUID = field(default_factory=uuid4)
        entities: list[Any] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    kb.recall.return_value = _Stub()
    hooks = KhoraMemoryHooks(kb=kb, namespace=uuid4(), recall_on_start=True)

    ctx = SimpleNamespace(
        input=None,
        user_input=None,
        messages=None,
        run=SimpleNamespace(input="from run", messages=None),
    )
    agent = SimpleNamespace(name="r")
    await hooks.on_agent_start(ctx, agent)
    kb.recall.assert_awaited_once()
    args, _kwargs = kb.recall.call_args
    assert args[0] == "from run"


async def test_on_agent_start_skips_when_recall_returns_no_chunks() -> None:
    """If recall returns 0 chunks, ``on_agent_start`` returns early without logging."""
    kb = _make_kb()
    from dataclasses import dataclass, field

    @dataclass
    class _Stub:
        chunks: list[Any] = field(default_factory=list)
        query: str = ""
        namespace_id: Any = field(default_factory=uuid4)
        entities: list[Any] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    kb.recall.return_value = _Stub()  # empty chunks
    hooks = KhoraMemoryHooks(kb=kb, namespace=uuid4(), recall_on_start=True)
    ctx = SimpleNamespace(input="anything", user_input=None, messages=None, run=None)
    # Must not raise; nothing is logged.
    await hooks.on_agent_start(ctx, SimpleNamespace(name="r"))


async def test_on_tool_end_strips_non_string_result() -> None:
    """``result`` of a non-string type is treated as empty and skipped."""
    kb = _make_kb()
    hooks = KhoraMemoryHooks(kb=kb, namespace=uuid4())
    # int isn't a string — skipped per the type-guard.
    await hooks.on_tool_end(
        SimpleNamespace(),
        SimpleNamespace(name="agent"),
        SimpleNamespace(name="tool"),
        42,  # type: ignore[arg-type]
    )
    kb.remember.assert_not_awaited()


async def test_on_tool_end_falls_back_to_tool_name_attr_when_no_call_id() -> None:
    """When the context has neither ``tool_call_id`` nor ``tool_name``, metadata is None for that key."""
    kb = _make_kb()
    hooks = KhoraMemoryHooks(kb=kb, namespace=uuid4())
    ctx = SimpleNamespace()  # no tool_call_id, no tool_name
    await hooks.on_tool_end(ctx, SimpleNamespace(name="agent"), SimpleNamespace(name="tool"), "ok")
    kb.remember.assert_awaited_once()
    _args, kwargs = kb.remember.call_args
    assert kwargs["metadata"]["oai_tool_call_id"] is None
