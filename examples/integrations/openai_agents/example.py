"""OpenAI Agents SDK + khora example - session memory via ``KhoraSession``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.

The example does NOT spin up a real ``agents.Runner`` - that would
require a live LLM. Instead it exercises the three khora primitives the
adapter exposes directly: ``KhoraSession`` (SessionABC contract),
``khora_recall_tool`` (FunctionTool factory), and ``KhoraMemoryHooks``
(RunHooks-shaped). Each is what an ``Agent`` would call into.

Kept deliberately small (single ``add_items`` write) so it finishes
well under the CI smoke budget - every khora write still runs the full
extraction pipeline against the mock LLM.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.openai_agents import (  # noqa: E402
    KhoraMemoryHooks,
    KhoraSession,
    khora_recall_tool,
)


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # 1) Session - one turn is enough to demonstrate the SessionABC
        #    contract. Every khora write runs full extraction, so we keep
        #    the example light to fit the CI smoke budget.
        session = KhoraSession(kb=kb, namespace=ns_id, session_id="example-conv-1")
        await session.add_items([{"role": "user", "content": "We picked PostgreSQL for the user DB."}])
        items = await session.get_items()
        assert len(items) == 1, "expected the session to round-trip exactly one item"
        assert items[-1]["content"] == "We picked PostgreSQL for the user DB."
        print(f"Session has {len(items)} item(s); latest: {items[-1]['content']!r}")

        # 2) Recall tool - closes over (kb, namespace, top_k). Construction
        #    is pure Python; no LLM I/O. An Agent would invoke it later.
        tool = khora_recall_tool(kb=kb, namespace=ns_id, top_k=3)
        print(f"Built recall tool: name={tool.name!r}")

        # 3) Memory hooks - construct only. ``on_tool_end`` would normally
        #    fire from inside ``Runner.run(...)`` and persist the tool
        #    output. We skip the live call here to keep the example fast.
        hooks = KhoraMemoryHooks(kb=kb, namespace=ns_id, app_id="example")
        print(f"Built memory hooks: app_id={hooks.app_id!r}")


if __name__ == "__main__":
    asyncio.run(main())
