"""LangGraph + khora example — long-term memory via ``KhoraStore``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TypedDict

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.graph import StateGraph  # noqa: E402

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.langgraph import KhoraStore  # noqa: E402


class State(TypedDict):
    note: str


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        store = KhoraStore(kb, user_id="example-user-1234")

        async def write_note(state: State) -> State:
            await store.aput(("memories",), "note-1", {"text": state["note"]})
            return state

        builder = StateGraph(State)
        builder.add_node("write", write_note)
        builder.set_entry_point("write")
        builder.set_finish_point("write")
        graph = builder.compile(store=store)

        await graph.ainvoke({"note": "the sky is blue today"})

        item = await store.aget(("memories",), "note-1")
        assert item is not None
        print(f"Stored memory: {item.value['text']!r}")

        namespaces = await store.alist_namespaces()
        print(f"Namespaces in store: {namespaces}")


if __name__ == "__main__":
    asyncio.run(main())
