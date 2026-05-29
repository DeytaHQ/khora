"""Quickstart 01 — remember & recall and comparison with text fact remembering

We are comparing pure text search with semantic similarity.

Engine choice: **skeleton** — the simplest, no entity extraction, no
graph writes. Just chunks + embeddings.

Run it
======
uv run python examples/00_quickstart/01_remember_recall.py
or
python examples/00_quickstart/01_remember_recall.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

# ── Logging: keep the terminal readable; full traces go to a file ─────
logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

_FACTS = [
    "Alice mentioned she can't eat peanuts — anaphylaxis.",
    "Alice lives in Seattle, in the Capitol Hill neighborhood.",
    "Alice works as a backend engineer focused on payments infrastructure.",
]


def naive_recall(facts: list[str], query: str) -> list[str]:
    """A reasonable straw-man: keyword scan over a Python list.

    This is what people reach for before installing a memory library —
    and it works fine until the user's query doesn't share any literal
    words with the stored fact.
    """
    needles = [w.lower() for w in query.split() if len(w) > 3]
    return [f for f in facts if any(n in f.lower() for n in needles)]


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)

    # ── Baseline: naive Python list ────────────────────────────────────
    print("Q: What food allergies should I know about?")
    print("\n[naive list, keyword scan]")
    for hit in naive_recall(_FACTS, "What food allergies should I know about?"):
        print(f"  - {hit}")
    print("  → 0 hits. The query and the stored fact don't share any words.")

    # ── khora ──────────────────────────────────────────────────────────
    print("\n[khora.recall, semantic]")
    async with Khora(config, engine="skeleton", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        for fact in _FACTS:
            await kb.remember(
                fact,
                namespace=ns_id,
                entity_types=["PERSON", "CONCEPT", "LOCATION"],
                relationship_types=["RELATES_TO"],
            )

        result = await kb.recall("What food allergies should I know about?", namespace=ns_id)
        for chunk in result.chunks[:3]:
            print(f"  [{chunk.score:.2f}] {chunk.content}")

        # Same library, different phrasing — works the same way.
        print('\nQ (rephrased): "any food-related medical issues?"')
        result = await kb.recall("any food-related medical issues?", namespace=ns_id)
        for chunk in result.chunks[:2]:
            print(f"  [{chunk.score:.2f}] {chunk.content}")


if __name__ == "__main__":
    asyncio.run(main())
