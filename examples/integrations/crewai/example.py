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

        memory.remember(
            "We decided to use PostgreSQL for the user database.",
            scope="/project/decisions",
            importance=0.9,
        )
        memory.remember(
            "The release window is the third week of every month.",
            scope="/project/process",
            importance=0.6,
        )

        # Verbatim recall: the mock LLM's hash-derived embeddings only
        # score an exact text match (cosine 1.0); a paraphrased query
        # like "which database did we pick?" lands near zero and falls
        # below the adapter's similarity floor. A real embedder handles
        # semantic queries.
        matches = memory.recall("We decided to use PostgreSQL for the user database.", limit=3)
        assert matches, "expected recall to return at least one match"
        assert any("PostgreSQL" in match.record.content for match in matches), (
            "expected the PostgreSQL decision to be recalled"
        )
        for match in matches:
            print(f"[{match.score:.2f}] {match.record.content}")


if __name__ == "__main__":
    asyncio.run(_main())
