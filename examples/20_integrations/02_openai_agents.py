"""Resume a chat session days later — OpenAI Agents SDK + khora.

The defining demo of any memory library: an agent's user walks away,
comes back tomorrow, and the agent still knows what they talked about.
This example shows the *minimum* you need to make that work with khora.

Two ``KhoraSession`` instances share the same namespace but use two
different ``session_id`` values — simulating "Monday's session" and
"Tuesday's session" with the same user. Memory written in session A is
visible to a ``recall()`` from session B because the namespace is the
isolation boundary, not the session.

Why Chronicle (engine="chronicle"):
Chronicle's bi-temporal model + Ebbinghaus decay is purpose-built for
this pattern. Older facts fade gracefully but stay recallable; explicit
session boundaries let you scope retention with
``Khora.forget_session(namespace_id, session_id)`` when a conversation
ends. The KhoraSession adapter maps the SDK's session id onto khora's
``session_id`` via UUID5 so the IDs round-trip cleanly.

What this example does NOT do:
- Spin up a real ``agents.Runner`` (would need a live OpenAI API key).
  Instead it exercises the ``SessionABC`` contract directly — exactly
  what an Agent would call into. The mock LLM stubs khora's own
  extraction calls so the example stays hermetic.
- Use real wall-clock days. We just instantiate two sessions in
  sequence to demonstrate the cross-session contract.

Run it
======
uv run python examples/20_integrations/02_openai_agents.py
python examples/20_integrations/02_openai_agents.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402
from khora.integrations.openai_agents import KhoraSession  # noqa: E402
from khora.integrations.openai_agents.session import session_uuid  # noqa: E402

# ── Logging setup ───────────────────────────────────────────────────────
# Khora uses loguru (not stdlib `logging`) for its own output. The default
# loguru sink writes to stderr, which floods the terminal with extraction
# and recall traces. Route the noise to a file and keep the terminal
# showing only warnings/errors plus this script's `print()` output. Drop
# these lines if you'd rather see everything; tighten the file-level
# threshold (e.g. `level="INFO"`) if `khora.log` itself gets too big.
logger.remove()  # drop default stderr sink
logger.add("khora.log", level="TRACE", enqueue=True)  # every level → file (TRACE is the floor)
# logger.add(sys.stderr, level="WARNING")                      # only warn+ → terminal

# Tame third-party stdlib loggers that bypass loguru.
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


_DEFAULT_CONFIG = Path(__file__).parent / "khora.embedded.yaml"


def _load_config() -> KhoraConfig:
    """Parse ``--config`` and load the named Khora YAML.

    Kept inline (rather than in a shared helper) so each example is
    readable on its own — copy / paste a file into your project and it
    works without dragging an ``examples/_common.py`` along. Matches the
    convention used by the numbered tutorials (01_hello_memory.py through
    08_slack_archive_bulk.py).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Khora YAML config path (default: {_DEFAULT_CONFIG.name}).",
    )
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config)


async def main() -> None:
    # Chronicle is the right engine for chat-shaped data: temporal
    # decay built in, no Neo4j needed. The adapter doesn't care which
    # engine khora is configured with — KhoraSession only touches the
    # facade.
    config = _load_config()

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Session A — "Monday" ─────────────────────────────────────
        # The session_id can be any string the SDK gives you. The
        # adapter UUID5s non-UUID strings deterministically, so the
        # same string maps to the same khora session_id every time.
        session_monday = KhoraSession(kb=kb, namespace=ns_id, session_id="conv-2026-05-19")
        await session_monday.add_items(
            [
                {"role": "user", "content": "I'm allergic to peanuts."},
                {"role": "assistant", "content": "Noted — I'll keep that in mind."},
                {"role": "user", "content": "Also, I prefer dinner around 7pm."},
            ]
        )

        monday_items = await session_monday.get_items()
        print(f"Monday session: {len(monday_items)} item(s)")

        # ── Session B — "Tuesday" ────────────────────────────────────
        # A new session_id but the SAME namespace. Anything the user
        # said yesterday is still findable via the namespace-scoped
        # recall; the per-session transcript on the SDK side starts
        # fresh.
        session_tuesday = KhoraSession(kb=kb, namespace=ns_id, session_id="conv-2026-05-20")

        # Tuesday's transcript is empty (fresh session).
        tuesday_items_before = await session_tuesday.get_items()
        print(f"\nTuesday session before add: {len(tuesday_items_before)} item(s)")

        # But cross-session recall against the shared namespace pulls
        # up yesterday's facts. A real agent would query khora directly
        # (or via ``khora_recall_tool``) on each turn; here we just
        # demonstrate that the data is reachable.
        recall = await kb.recall(
            "What food restrictions does the user have?",
            namespace=ns_id,
            limit=3,
        )
        print(f"\nCross-session recall hits: {len(recall.chunks)}")
        for chunk in recall.chunks[:3]:
            preview = chunk.content[:80].replace("\n", " ")
            print(f"  [{chunk.score:.3f}] {preview}{'…' if len(chunk.content) > 80 else ''}")

        # Tuesday continues the conversation. The user repeats their
        # preference; the agent now has the prior context to fall back on.
        await session_tuesday.add_items(
            [
                {"role": "user", "content": "Book me dinner reservations."},
            ]
        )

        tuesday_items_after = await session_tuesday.get_items()
        print(f"\nTuesday session after add: {len(tuesday_items_after)} item(s)")

        # ── Session retention ────────────────────────────────────────
        # When a conversation truly ends, ``forget_session`` cascades
        # the delete: documents + chunks + Neo4j Chunk nodes for that
        # session. This is the GDPR-friendly cleanup hook.
        deleted = await kb.forget_session(ns_id, session_uuid("conv-2026-05-19"))
        print(f"\nDeleted {deleted} documents from Monday's session.")

        # After the delete, cross-session recall for Monday's content
        # returns nothing — Tuesday's data is untouched.
        recall_after = await kb.recall(
            "What food restrictions does the user have?",
            namespace=ns_id,
            limit=3,
        )
        print(f"Post-cleanup recall hits: {len(recall_after.chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
