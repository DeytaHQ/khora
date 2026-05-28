"""Quickstart 03 — explicit unlearning with ``forget()``.

Memories ingested under bad assumptions don't fix themselves. If an
agent learned "the standup is at 9am" and the time moved to 10am,
re-ingesting the new fact doesn't erase the old one — both end up
retrievable, and recall scores depend on which phrasing the user
happened to use.

This demo walks the lifecycle: remember a fact, see it come back from
recall, forget it by ``document_id``, see it gone. Then remember the
corrected version. The ``document_id`` is the only handle ``forget()``
takes — there's no fuzzy "find the bad memory by content match," which
is intentional: forgetting by content is racy and silently overshoots.

Engine choice: **skeleton** — forget semantics are the same across all
engines, but skeleton keeps the demo cheap.

Run it
======
uv run python examples/00_quickstart/03_forget_what_was_wrong.py
python examples/00_quickstart/03_forget_what_was_wrong.py
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


async def show_top(kb, namespace, query: str, label: str) -> None:
    result = await kb.recall(query, namespace=namespace, limit=2)
    if not result.chunks:
        print(f"  {label}: (nothing recalled)")
        return
    for chunk in result.chunks:
        print(f"  {label} [{chunk.score:.2f}] {chunk.content}")


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)
    async with Khora(config, engine="skeleton", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Step 1: ingest a fact that turns out to be wrong ───────────
        wrong = await kb.remember(
            "The team standup is at 9:00 AM every weekday.",
            namespace=ns_id,
            title="standup time (wrong)",
            entity_types=["EVENT", "CONCEPT"],
            relationship_types=["RELATES_TO"],
        )
        wrong_doc_id = wrong.document_id
        print(f"remembered wrong fact (document_id = {wrong_doc_id})")

        # Recall picks it up — exactly what we don't want post-correction.
        print("\nbefore forget:")
        await show_top(kb, ns_id, "when is the standup?", "  found")

        # ── Step 2: forget by document_id ──────────────────────────────
        # The id came from remember()'s return value. In a real app the
        # caller stores it next to whatever business object the memory
        # represents (a ticket, a chat message, a config row).
        ok = await kb.forget(wrong_doc_id, namespace=ns_id)
        print(f"\nforget({wrong_doc_id}) → {ok}")

        # ── Step 3: recall again — gone ────────────────────────────────
        print("\nafter forget:")
        await show_top(kb, ns_id, "when is the standup?", "  found")

        # ── Step 4: ingest the corrected fact ──────────────────────────
        # Note: this is a *separate* document. Forgetting + re-remembering
        # is the supported "I had it wrong" workflow; there's no in-place
        # mutation API by design (write-once memories preserve audit).
        await kb.remember(
            "The team standup is at 10:00 AM every weekday.",
            namespace=ns_id,
            title="standup time (corrected)",
            entity_types=["EVENT", "CONCEPT"],
            relationship_types=["RELATES_TO"],
        )
        print("\nafter re-ingest with corrected time:")
        await show_top(kb, ns_id, "when is the standup?", "  found")


if __name__ == "__main__":
    asyncio.run(main())
