"""Core API — recall with filters.

``kb.recall()`` exposes three knobs that change *what comes back* (not
just the ranking). Knowing them removes the need for most application-
side post-filtering.

  * ``limit``         — cap the response size at the engine level. Cheaper
                        than asking for 100 chunks and trimming to 5.
  * ``min_similarity`` — drop chunks whose raw cosine is below the
                        threshold. Operates BEFORE score normalization,
                        so it's a real semantic-quality cutoff (unlike
                        thresholding the post-normalize ``chunk.score``).
  * ``mode``          — SearchMode.{VECTOR, GRAPH, HYBRID, ALL, KEYWORD}.
                        HYBRID is the default — vector + graph + BM25
                        fused. Pure VECTOR skips the graph channel;
                        KEYWORD is BM25 only.

Engine choice: **vectorcypher** — the only engine with all three
channels populated, so ``mode`` actually has something to differentiate.
Skeleton collapses GRAPH/HYBRID to VECTOR transparently.

What's missing: the ``start_time`` / ``end_time`` kwargs on ``recall()``
also exist, but in v0.17 they aren't useful on VectorCypher or
Skeleton because ``source_timestamp=`` at ingest doesn't propagate
to chunk ``occurred_at`` on either backend — tracked as
https://github.com/DeytaHQ/khora/issues/859. When that's fixed, this
demo grows back a fourth section. The Chronicle engine *does*
propagate ``source_timestamp`` end-to-end; for time-windowed recall
today, that's the engine to reach for.

Run it
======
uv run python examples/10_core_apis/02_recall_with_filters.py
python examples/10_core_apis/02_recall_with_filters.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig
from khora.query.engine import SearchMode

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)

    facts = [
        "Released v1.0 of the data ingest pipeline.",
        "Hot-fixed an OOM regression in the ingest worker.",
        "Wrote a design doc for cross-region replication.",
        "Shipped the new admin UI for namespace management.",
        "Closed all P0 tickets for the Q2 release.",
    ]

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        for content in facts:
            await kb.remember(
                content,
                namespace=ns_id,
                entity_types=["EVENT", "CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

        # ── 1. limit ───────────────────────────────────────────────────
        print("limit=2 — only the top 2 chunks come back:")
        result = await kb.recall("ingest pipeline work", namespace=ns_id, limit=2)
        for c in result.chunks:
            print(f"  [{c.score:.2f}] {c.content}")

        # ── 2. min_similarity ──────────────────────────────────────────
        # ``min_similarity`` is the raw vector cosine cutoff on the
        # semantic channel; the engine may still surface lower-ranked
        # chunks from other channels (BM25, entity), which is why the
        # bottom of the result list shows normalized scores near zero.
        print("\nmin_similarity=0.4 — semantic-channel cutoff:")
        result = await kb.recall(
            "memory leak in ingestion service",
            namespace=ns_id,
            limit=10,
            min_similarity=0.4,
        )
        for c in result.chunks[:3]:
            print(f"  [{c.score:.2f}] {c.content}")

        # ── 3. mode ────────────────────────────────────────────────────
        # Same query, two modes. HYBRID (default) blends vector + graph
        # + BM25. KEYWORD is BM25-only — pure lexical match. The
        # difference shows up when a query has rare keywords that hit
        # exactly (KEYWORD shines) vs. abstract wording where vector
        # semantics carry (HYBRID wins).
        query = "OOM regression worker"
        print(f"\nmode=HYBRID (default), query={query!r}:")
        result = await kb.recall(query, namespace=ns_id, limit=3)
        for c in result.chunks:
            print(f"  [{c.score:.2f}] {c.content}")

        print(f"\nmode=KEYWORD (BM25 only), query={query!r}:")
        result = await kb.recall(query, namespace=ns_id, limit=3, mode=SearchMode.KEYWORD)
        for c in result.chunks:
            print(f"  [{c.score:.2f}] {c.content}")


if __name__ == "__main__":
    asyncio.run(main())
