"""Unit tests for ``KhoraMemoryHooks``.

Exercises ``on_tool_end`` (writes the tool result to khora) and
``on_agent_start`` (recall + log when enabled). Both code paths run
against an ``AsyncMock(spec=Khora)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("agents")

from agents.lifecycle import RunHooksBase  # noqa: E402

from khora.integrations.openai_agents.hooks import KhoraMemoryHooks  # noqa: E402
from khora.khora import Khora  # noqa: E402


@dataclass
class _RecallResultStub:
    chunks: list[tuple[Any, float]]
    query: str = ""
    namespace_id: UUID = field(default_factory=uuid4)
    entities: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_kb() -> Any:
    return AsyncMock(spec=Khora)


def _make_hooks(kb: Any, **overrides: Any) -> KhoraMemoryHooks:
    return KhoraMemoryHooks(kb=kb, namespace=uuid4(), **overrides)


def _make_tool(name: str = "lookup") -> Any:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_agent(name: str = "researcher") -> Any:
    agent = MagicMock()
    agent.name = name
    return agent


def _make_ctx(**attrs: Any) -> Any:
    ctx = MagicMock()
    for k, v in attrs.items():
        setattr(ctx, k, v)
    return ctx


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_hooks_rejects_non_uuid_namespace() -> None:
    with pytest.raises(TypeError, match="namespace"):
        KhoraMemoryHooks(kb=_make_kb(), namespace="not-a-uuid")  # type: ignore[arg-type]


def test_hooks_rejects_empty_app_id() -> None:
    with pytest.raises(ValueError, match="app_id"):
        KhoraMemoryHooks(kb=_make_kb(), namespace=uuid4(), app_id=" ")


def test_hooks_rejects_zero_recall_top_k() -> None:
    with pytest.raises(ValueError, match="recall_top_k"):
        KhoraMemoryHooks(kb=_make_kb(), namespace=uuid4(), recall_top_k=0)


# ---------------------------------------------------------------------------
# on_tool_end
# ---------------------------------------------------------------------------


async def test_on_tool_end_persists_result_via_kb_remember() -> None:
    kb = _make_kb()
    hooks = _make_hooks(kb)
    tool = _make_tool("lookup")
    agent = _make_agent("researcher")
    ctx = _make_ctx(tool_call_id="call-7")

    await hooks.on_tool_end(ctx, agent, tool, "the answer is 42")

    kb.remember.assert_awaited_once()
    args, kwargs = kb.remember.call_args
    assert args[0] == "the answer is 42"
    assert kwargs["namespace"] == hooks.namespace_id
    assert kwargs["title"] == "oai_tool:lookup"
    assert kwargs["metadata"]["oai_tool_name"] == "lookup"
    assert kwargs["metadata"]["oai_agent_name"] == "researcher"
    assert kwargs["metadata"]["oai_tool_call_id"] == "call-7"
    assert kwargs["entity_types"] == []
    assert kwargs["relationship_types"] == []


async def test_on_tool_end_skips_empty_result() -> None:
    kb = _make_kb()
    hooks = _make_hooks(kb)
    await hooks.on_tool_end(_make_ctx(), _make_agent(), _make_tool(), "")
    kb.remember.assert_not_awaited()


async def test_on_tool_end_no_op_when_recording_disabled() -> None:
    kb = _make_kb()
    hooks = _make_hooks(kb, record_tool_results=False)
    await hooks.on_tool_end(_make_ctx(), _make_agent(), _make_tool(), "hi")
    kb.remember.assert_not_awaited()


async def test_on_tool_end_swallows_remember_errors() -> None:
    """Observability hooks must never bring down the run on a backend hiccup."""
    kb = _make_kb()
    kb.remember.side_effect = RuntimeError("backend down")
    hooks = _make_hooks(kb)
    # Must NOT raise.
    await hooks.on_tool_end(_make_ctx(), _make_agent(), _make_tool(), "result")


# ---------------------------------------------------------------------------
# on_agent_start
# ---------------------------------------------------------------------------


async def test_on_agent_start_does_nothing_by_default() -> None:
    """``recall_on_start=False`` (default) skips the recall path entirely."""
    kb = _make_kb()
    hooks = _make_hooks(kb)  # default: recall_on_start=False
    await hooks.on_agent_start(_make_ctx(input="what?"), _make_agent())
    kb.recall.assert_not_awaited()


async def test_on_agent_start_recalls_when_enabled() -> None:
    kb = _make_kb()
    kb.recall.return_value = _RecallResultStub(chunks=[(MagicMock(), 0.9)])
    hooks = _make_hooks(kb, recall_on_start=True, recall_top_k=2)
    ctx = _make_ctx(input="why does X happen?")

    await hooks.on_agent_start(ctx, _make_agent("researcher"))

    kb.recall.assert_awaited_once()
    args, kwargs = kb.recall.call_args
    assert args[0] == "why does X happen?"
    assert kwargs["namespace"] == hooks.namespace_id
    assert kwargs["limit"] == 2


async def test_on_agent_start_handles_list_of_message_dicts() -> None:
    """SDK passes input as ``list[TResponseInputItem]`` once a turn lands."""
    kb = _make_kb()
    kb.recall.return_value = _RecallResultStub(chunks=[])
    hooks = _make_hooks(kb, recall_on_start=True)

    ctx = _make_ctx(input=[{"role": "user", "content": "the latest question"}])
    await hooks.on_agent_start(ctx, _make_agent())

    args, _ = kb.recall.call_args
    assert args[0] == "the latest question"


async def test_on_agent_start_no_op_on_empty_input() -> None:
    kb = _make_kb()
    hooks = _make_hooks(kb, recall_on_start=True)
    await hooks.on_agent_start(_make_ctx(input=""), _make_agent())
    kb.recall.assert_not_awaited()


async def test_on_agent_start_swallows_recall_errors() -> None:
    kb = _make_kb()
    kb.recall.side_effect = RuntimeError("recall broken")
    hooks = _make_hooks(kb, recall_on_start=True)
    # Must NOT raise.
    await hooks.on_agent_start(_make_ctx(input="anything"), _make_agent())


# ---------------------------------------------------------------------------
# as_runhooks adapter
# ---------------------------------------------------------------------------


def test_as_runhooks_returns_real_runhooks_subclass_instance() -> None:
    hooks = _make_hooks(_make_kb())
    adapter = hooks.as_runhooks()
    # RunHooks itself is a subscripted generic (RunHooksBase[TContext, Agent])
    # so the isinstance check has to go through the unparameterised base.
    assert isinstance(adapter, RunHooksBase)


async def test_as_runhooks_forwards_on_tool_end_to_owner() -> None:
    kb = _make_kb()
    hooks = _make_hooks(kb)
    adapter = hooks.as_runhooks()
    await adapter.on_tool_end(_make_ctx(), _make_agent(), _make_tool("t"), "result")
    kb.remember.assert_awaited_once()
