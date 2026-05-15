"""End-to-end integration: 1-node LangGraph graph wired to ``KhoraStore``.

Runs against an in-memory ``sqlite_lance`` khora (no Postgres, no
Neo4j). The mock LLM helper patches ``litellm.acompletion`` /
``litellm.aembedding`` so no API keys are needed.

This test proves the adapter is wired up correctly end-to-end:

1. Build a real ``Khora`` on sqlite_lance.
2. Build a ``KhoraStore`` over it.
3. Compile a tiny LangGraph state graph with ``store=KhoraStore(...)``.
4. Invoke the graph; the node's body writes a memory through the store.
5. Read the same memory back through ``KhoraStore.aget``.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
    pytest.mark.skipif(not _HAS_LANGGRAPH, reason="langgraph not installed"),
]


class _State(TypedDict):
    """Minimal graph state."""

    note: str


@pytest.mark.asyncio
async def test_one_node_graph_writes_and_reads_through_khorastore(monkeypatch):
    """Compile a 1-node graph, write a memory in the node, read it back."""
    from langgraph.graph import StateGraph

    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.langgraph import KhoraStore

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        store = KhoraStore(kb, user_id="e2e-user-1234")

        async def node(state: _State) -> _State:
            # Node writes a memory using its own store handle. The graph
            # API also injects ``store`` as a runtime arg; either works.
            await store.aput(("memories",), "first-note", {"text": state["note"]})
            return state

        builder = StateGraph(_State)
        builder.add_node("write", node)
        builder.set_entry_point("write")
        builder.set_finish_point("write")
        graph = builder.compile(store=store)

        await graph.ainvoke({"note": "the sky is blue"})

        # Verify round-trip through KhoraStore.
        item = await store.aget(("memories",), "first-note")
        assert item is not None
        assert item.value == {"text": "the sky is blue"}
        assert item.namespace == ("memories",)
        assert item.key == "first-note"

        # And the namespace is discoverable.
        namespaces = await store.alist_namespaces()
        assert ("memories",) in namespaces
