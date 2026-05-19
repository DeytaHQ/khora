"""Unit tests for ``khora_recall_tool``.

The factory closes over the bound khora instance; the test exercises the
generated tool by invoking its underlying callable directly with mocked
``Khora.recall`` return values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("agents")

from agents.run import RunConfig  # noqa: E402
from agents.tool import FunctionTool  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402

from khora.integrations.openai_agents.tool import khora_recall_tool  # noqa: E402
from khora.khora import Khora  # noqa: E402


def _make_tool_context() -> ToolContext:
    """Build the minimum-shape ``ToolContext`` the tool callback needs."""
    return ToolContext(
        context=None,
        tool_name="recall_memory",
        tool_call_id="test-call-1",
        tool_arguments='{"query": "what?"}',
        run_config=RunConfig(),
    )


@dataclass
class _RecallResultStub:
    chunks: list[tuple[Any, float]]
    query: str = ""
    namespace_id: UUID = field(default_factory=uuid4)
    entities: list[Any] = field(default_factory=list)
    context_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_chunk(content: str) -> Any:
    from khora.core.models.document import Chunk

    return Chunk(content=content, document_id=uuid4())


def _make_kb() -> Any:
    return AsyncMock(spec=Khora)


def test_returns_a_function_tool_instance() -> None:
    kb = _make_kb()
    tool = khora_recall_tool(kb=kb, namespace=uuid4(), top_k=3)
    assert isinstance(tool, FunctionTool)


def test_tool_name_defaults_and_overrides() -> None:
    kb = _make_kb()
    default_tool = khora_recall_tool(kb=kb, namespace=uuid4())
    assert default_tool.name == "recall_memory"

    custom_tool = khora_recall_tool(kb=kb, namespace=uuid4(), name="lookup_facts")
    assert custom_tool.name == "lookup_facts"


def test_tool_validates_top_k_and_min_similarity() -> None:
    kb = _make_kb()
    with pytest.raises(ValueError, match="top_k"):
        khora_recall_tool(kb=kb, namespace=uuid4(), top_k=0)
    with pytest.raises(ValueError, match="min_similarity"):
        khora_recall_tool(kb=kb, namespace=uuid4(), min_similarity=2.0)


async def test_tool_invocation_calls_kb_recall_with_bound_namespace() -> None:
    """The factory closure must scope recall to the bound namespace + top_k."""
    kb = _make_kb()
    namespace = uuid4()
    kb.recall.return_value = _RecallResultStub(
        chunks=[(_make_chunk("hello"), 0.87), (_make_chunk("world"), 0.65)],
    )

    tool = khora_recall_tool(kb=kb, namespace=namespace, top_k=4, min_similarity=0.1)
    # FunctionTool exposes ``on_invoke_tool`` as the actual callback the
    # Runner uses. It takes a context wrapper + a JSON-string argument
    # payload; the closure inside us only cares about ``query``.
    result = await tool.on_invoke_tool(_make_tool_context(), '{"query": "what?"}')

    kb.recall.assert_awaited_once()
    args, kwargs = kb.recall.call_args
    assert args[0] == "what?"
    assert kwargs["namespace"] == namespace
    assert kwargs["limit"] == 4
    assert kwargs["min_similarity"] == pytest.approx(0.1)
    # Returned string includes both chunks with scores.
    assert "score=0.870" in result
    assert "hello" in result
    assert "world" in result


async def test_tool_returns_no_match_string_when_recall_empty() -> None:
    kb = _make_kb()
    kb.recall.return_value = _RecallResultStub(chunks=[])
    tool = khora_recall_tool(kb=kb, namespace=uuid4())
    out = await tool.on_invoke_tool(_make_tool_context(), '{"query": "anything"}')
    assert "no relevant memories found" in out
