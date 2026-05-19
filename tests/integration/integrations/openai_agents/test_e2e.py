"""End-to-end integration: ``KhoraSession`` round-trip on sqlite_lance.

Runs against an in-memory ``sqlite_lance`` khora (no Postgres, no
Neo4j). The mock LLM helper patches ``litellm.acompletion`` /
``litellm.aembedding`` so no API keys are needed.

This test proves the three OpenAI Agents SDK primitives are wired up
correctly end-to-end:

1. Build a real ``Khora`` on sqlite_lance.
2. Build a ``KhoraSession`` against a namespace.
3. ``add_items`` then ``get_items`` — items round-trip in order.
4. ``pop_item`` returns the latest, then drops it.
5. ``clear_session`` empties the session.
6. ``khora_recall_tool`` returns a real ``FunctionTool`` that calls
   ``Khora.recall`` under the hood.
7. ``KhoraMemoryHooks.on_tool_end`` writes tool output to khora.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    import agents  # noqa: F401

    _HAS_AGENTS = True
except ImportError:
    _HAS_AGENTS = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
    pytest.mark.skipif(not _HAS_AGENTS, reason="openai-agents not installed"),
]


@pytest.mark.asyncio
async def test_session_add_get_pop_clear_roundtrip(monkeypatch):
    """``KhoraSession`` honours the full SessionABC contract on real khora."""
    from agents.memory.session import SessionABC

    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.openai_agents import KhoraSession

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        session = KhoraSession(kb=kb, namespace=namespace.namespace_id, session_id="e2e-conv-1")

        # Acceptance criterion: KhoraSession is a runtime SessionABC.
        assert isinstance(session, SessionABC)

        items = [
            {"role": "user", "content": "What city hosts the 2026 Winter Olympics?"},
            {"role": "assistant", "content": "Milano-Cortina, Italy."},
            {"role": "user", "content": "And the summer games?"},
        ]
        await session.add_items(items)

        # Round-trip preserves order.
        got = await session.get_items()
        assert got == items

        # Limit returns the latest N in chronological order.
        latest_two = await session.get_items(limit=2)
        assert latest_two == items[-2:]

        # pop_item returns the most-recent item and drops it.
        popped = await session.pop_item()
        assert popped == items[-1]
        remaining = await session.get_items()
        assert remaining == items[:-1]

        # clear_session empties everything.
        await session.clear_session()
        assert await session.get_items() == []


@pytest.mark.asyncio
async def test_recall_tool_invocation_against_real_khora(monkeypatch):
    """The ``khora_recall_tool`` ``FunctionTool`` runs against a live khora."""
    from agents.run import RunConfig
    from agents.tool import FunctionTool
    from agents.tool_context import ToolContext

    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.openai_agents import khora_recall_tool

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # Seed a couple of memories so the tool has something to surface.
        memory_text = "We picked PostgreSQL for the primary user database."
        await kb.remember(
            memory_text,
            namespace=ns_id,
            entity_types=[],
            relationship_types=[],
        )

        tool = khora_recall_tool(kb=kb, namespace=ns_id, top_k=3)
        assert isinstance(tool, FunctionTool)
        assert tool.name == "recall_memory"

        ctx = ToolContext(
            context=None,
            tool_name="recall_memory",
            tool_call_id=str(uuid4()),
            tool_arguments=f'{{"query": "{memory_text}"}}',
            run_config=RunConfig(),
        )
        out = await tool.on_invoke_tool(ctx, f'{{"query": "{memory_text}"}}')
        # The mock embedder hashes deterministically, so an exact-text query
        # produces cosine=1.0 against its own chunk. The serialised tool
        # output must contain that chunk.
        assert "PostgreSQL" in out


@pytest.mark.asyncio
async def test_memory_hooks_persist_tool_results(monkeypatch):
    """``KhoraMemoryHooks.on_tool_end`` writes successful tool output to khora."""
    from unittest.mock import MagicMock

    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.openai_agents import KhoraMemoryHooks

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id
        hooks = KhoraMemoryHooks(kb=kb, namespace=ns_id, app_id="e2e_app")

        tool = MagicMock()
        tool.name = "lookup"
        agent = MagicMock()
        agent.name = "researcher"
        ctx = MagicMock()
        ctx.tool_call_id = "test-call"

        await hooks.on_tool_end(ctx, agent, tool, "The answer is forty-two.")

        # The hook should have written one document; recall it back.
        recall = await kb.recall("forty-two", namespace=ns_id, limit=5)
        contents = [chunk.content for chunk in recall.chunks]
        assert any("forty-two" in c for c in contents), f"hook write not recalled: {contents}"
