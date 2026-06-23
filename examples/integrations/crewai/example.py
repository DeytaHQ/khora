"""Smoke example for the khora CrewAI adapter.

Runs without external services or API keys: the in-memory sqlite_lance
khora fixture plus the deterministic mock LLM cover everything the
adapter needs end-to-end.
"""

from __future__ import annotations

import asyncio

from examples._helpers import embedded_khora, install_mock_llm
from khora.integrations.crewai import KhoraMemory


async def _main() -> None:
    install_mock_llm()
    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        memory = KhoraMemory(
            kb=kb,
            namespace=namespace.namespace_id,
            user_id="user-example-12345678",
        )

        decision = "We decided to use PostgreSQL for the user database."
        memory.remember(
            decision,
            scope="/project/decisions",
            importance=0.9,
        )
        memory.remember(
            "The release window is the third week of every month.",
            scope="/project/process",
            importance=0.6,
        )

        # Query with the exact stored text: hash-derived embeddings give a
        # cosine-1.0 match, guaranteeing at least one result.
        matches = memory.recall(decision, limit=3)
        assert len(matches) > 0, "recall returned no results"
        for match in matches:
            print(f"[{match.score:.2f}] {match.record.content}")


if __name__ == "__main__":
    asyncio.run(_main())
