"""Quickstart 04 — per-user isolation with namespaces.

A namespace is khora's only tenancy boundary. Two users, two
namespaces — and the API doesn't let you forget to scope. Every
memory, every entity, every relationship is scoped at write time;
every recall is scoped at read time. If you forget to pass
``namespace=``, the call errors out.

The convincing test isn't "I queried both and got different answers"
— that's the happy path. The convincing test is the **needle**:
bury a unique secret string in Alice's namespace and verify it never
surfaces from Bob's, regardless of how you query for it (by exact
string, by semantic relative, by anchor noun). That's the same shape
as a real cross-tenant leak audit.

Engine choice: **skeleton** for simplicity, but namespace semantics
are uniform across engines.

Run it
======
uv run python examples/00_quickstart/04_namespaces_for_users.py
python examples/00_quickstart/04_namespaces_for_users.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

# The needle — a unique secret string that exists only in Alice's
# namespace. If any query against Bob's namespace returns a chunk
# containing this string, isolation is broken.
_NEEDLE = "OPERATION-ZEBRA-77"

_ALICE = [
    "I prefer dark roast coffee, never decaf.",
    "I'm a vegetarian — no fish, no chicken.",
    "Schedule my standups in the morning; afternoons are heads-down.",
    "I live in Seattle, in the Capitol Hill neighborhood.",
    "I'm allergic to penicillin.",
    f"My internal project codename is {_NEEDLE} — keep it confidential. It is about stabbing Bob in the back!",
    "I prefer dark mode in every app.",
    "I work out at 6am; book me no earlier than 8am for meetings.",
    "I drive a 2019 Subaru Outback.",
    "My favorite restaurant is Sabine, in Belltown.",
]

_BOB = [
    "I drink green tea, never coffee.",
    "I eat anything; no dietary restrictions to worry about.",
    "Block off mornings for deep work; afternoons are meeting-friendly.",
    "I live in Brooklyn, in Park Slope.",
    "No known drug allergies.",
    "My current side project is a rock-climbing route database.",
    "I prefer light mode — dark mode hurts my eyes.",
    "I rock-climb every Saturday morning.",
    "I drive a 2022 Tesla Model 3.",
    "My favorite restaurant is Roberta's pizza, in Bushwick.",
]


async def remember_each(kb, namespace, facts: list[str]) -> None:
    for fact in facts:
        await kb.remember(
            fact,
            namespace=namespace,
            entity_types=["PERSON", "CONCEPT"],
            relationship_types=["RELATES_TO"],
        )


async def needle_check(kb, bob_ns, query: str) -> bool:
    """Query Bob's namespace; return True if the needle is absent."""
    result = await kb.recall(query, namespace=bob_ns, limit=10)
    leaked = [c.content for c in result.chunks if _NEEDLE in c.content]
    if leaked:
        print(f"  ✗ LEAK on query {query!r}:")
        for content in leaked:
            print(f"      {content}")
        return False
    print(f"  ✓ needle absent on query {query!r}  ({len(result.chunks)} chunk(s) returned, none contained '{_NEEDLE}')")
    return True


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)
    async with Khora(config, engine="skeleton", run_migrations=True) as kb:
        alice_ns = (await kb.create_namespace()).namespace_id
        bob_ns = (await kb.create_namespace()).namespace_id

        await remember_each(kb, alice_ns, _ALICE)
        await remember_each(kb, bob_ns, _BOB)
        print(f"ingested {len(_ALICE)} facts into alice  (needle: {_NEEDLE})")
        print(f"ingested {len(_BOB)} facts into bob")

        # Happy-path sanity check — same question, different answers.
        question = "What should I order for them at lunch?"
        print(f"\nQ: {question}")
        for label, ns in [("alice", alice_ns), ("bob", bob_ns)]:
            result = await kb.recall(question, namespace=ns, limit=1)
            if result.chunks:
                print(f"  [{label}] {result.chunks[0].content}")

        # The needle test — three attempts to extract Alice's secret
        # from Bob's namespace, escalating in semantic distance from the
        # exact string:
        #
        #   1. exact-substring query  — brute-force lookup of the literal
        #   2. anchor-noun query      — "internal project codename"
        #   3. confidentiality cue    — "what's confidential about me?"
        #
        # All three must miss for isolation to hold.
        print("\n[needle test] Attempting to extract Alice's secret from Bob's namespace:")
        attempts = [
            _NEEDLE,
            "internal project codename",
            "what's confidential about me?",
        ]
        all_passed = all(
            [await needle_check(kb, bob_ns, q) for q in attempts]  # noqa: E501
        )
        print(
            "\n[verdict]  "
            + (
                "✓ isolation holds — Alice's needle never reached Bob's namespace."
                if all_passed
                else "✗ ISOLATION BROKEN — see leaks above."
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
