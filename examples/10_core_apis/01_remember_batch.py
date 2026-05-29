"""Core API — bulk ingestion with ``remember_batch``.

Calling ``kb.remember()`` in a Python loop works for a handful of
records but burns through three avoidable taxes for anything larger:

  1. Each call re-opens / re-warms the embedder cache.
  2. Entity deduplication runs per-document with no cross-doc dedup
     in scope.
  3. Concurrency is whatever asyncio your caller wires up — usually
     none.

``remember_batch`` fixes all three. The embedder cache is shared
across the whole call, ``EntityIndex`` does cross-doc dedup, and the
``max_concurrent`` knob controls how many docs are in-flight against
the LLM at once. For the embedded backend on a laptop, 10 is a good
ceiling; on the standard stack (Postgres + Neo4j) you can push higher.

The ``on_progress=`` kwarg is wired to a counter callback so you can
see it fire — but in v0.17 it isn't useful for live monitoring on any
engine. Two compounding limitations make a tqdm-style bar decorative:

* **SF-Fr25** — Skeleton and Chronicle call ``on_progress`` exactly
  once at the end with ``(total, total)``. Only VectorCypher fires
  per-document.
* **SF-Fr26** — even VectorCypher's per-doc callbacks burst at the
  end of ``asyncio.gather()`` rather than streaming as each document
  actually lands. The elapsed-time spread between the first and last
  callback is typically milliseconds, after a multi-minute batch.

So we print the doc count + total elapsed time after the batch
returns. That conveys everything ``on_progress`` can honestly tell
you today. When SF-Fr25 / SF-Fr26 are fixed, this demo grows back a
real tqdm bar.

Engine choice: **vectorcypher** — extracts entities + relationships
per doc so the result counts at the end show non-zero values
(Skeleton would have shown chunks only).

Run it
======
uv run python examples/10_core_apis/01_remember_batch.py
python examples/10_core_apis/01_remember_batch.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_TICKETS = Path(__file__).parent.parent / "data" / "support_tickets.jsonl"


def _load_tickets(limit: int) -> list[dict]:
    """Read the first ``limit`` support tickets from the shared corpus.

    ``examples/data/support_tickets.jsonl`` is the canonical support-
    ticket dataset shared with `30_workloads/03_support_ticket_graph`.
    Reusing it keeps the demo's content realistic without inventing a
    parallel synthetic corpus.
    """
    docs: list[dict] = []
    for line in _TICKETS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(
            {
                "content": row["content"],
                "title": row.get("title", row.get("ticket_id", "ticket")),
                "source": row.get("source", "support"),
            }
        )
        if len(docs) >= limit:
            break
    return docs


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)
    # Limiting tickets to 5 just to keep load time minimal,
    # increase if you want to see how it works with more data
    docs = _load_tickets(limit=5)

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # Use this list to capture what is completed
        # illustrates how to capture the progress of a batch
        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_calls.append((done, total))

        t0 = time.perf_counter()
        result = await kb.remember_batch(
            docs,
            namespace=ns_id,
            max_concurrent=5,
            on_progress=on_progress,
            entity_types=["CONCEPT", "EVENT"],
            relationship_types=["RELATES_TO"],
        )
        elapsed = time.perf_counter() - t0

        print(
            f"\nbatch done in {elapsed:.1f}s: {result.processed} processed, "
            f"{result.skipped} skipped, {result.failed} failed"
        )
        print(f"  chunks={result.chunks}, entities={result.entities}, relationships={result.relationships}")
        print(f"  on_progress fired {len(progress_calls)} time(s) — {progress_calls}")

        # ── Recall sanity check ────────────────────────────────────────
        # VectorCypher extracts entities + relationships per doc, so the
        # batch result reports non-zero counts for both — different from
        # what Skeleton would have shown (chunks only).
        query = "WarehouseIQ login failures"
        recall = await kb.recall(query, namespace=ns_id, limit=3)
        print(f"\nrecall: {len(recall.chunks)} chunks for {query!r}")
        for c in recall.chunks:
            print(f"  [{c.score:.2f}] {c.content[:80]}…")


if __name__ == "__main__":
    asyncio.run(main())
