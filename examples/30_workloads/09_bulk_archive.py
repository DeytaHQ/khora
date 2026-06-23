"""Demo 08 — Bulk Slack archive ingest (the cost-story demo).

Ingest a "Slack archive" of timestamped messages with bi-temporal
metadata (occurred_at + ingested_at), then run a time-filtered query
against it. The point of the demo is the **cost shape**, not the
absolute size — a real archive is 50k–500k messages; the demo uses 50
so it runs in under a minute on a laptop.

WHY SKELETON CONSTRUCTION
=========================
The Skeleton engine extracts entities from only ~10% of chunks (the
high-importance ones picked by the skeleton-indexer's PageRank-style
scorer). The other 90% get embeddings + keyword-derived pseudo-entities.

Napkin math on the full archive shape:

  • 50k messages → ~5k LLM extraction calls (Skeleton, 10%)
  • vs ~35k LLM extraction calls (VectorCypher, default 70%)
  • vs ~50k LLM extraction calls (full extraction)

For a chat log corpus where most messages are "ok", "thanks", or
single emoji, the cost saving is real and the recall loss is small.

It also runs on PostgreSQL alone — no graph backend. For an archive
sidecar that exists to make old messages searchable, that's the right
deployment shape.

DUAL-BACKEND SUPPORT
====================
Skeleton is **Available** on PG+pgvector and **Experimental** on
sqlite_lance — both run. The hierarchical time navigation works on
both.

Run it
======
uv run python examples/30_workloads/09_bulk_archive.py
python examples/30_workloads/09_bulk_archive.py
uv run python examples/30_workloads/09_bulk_archive.py --config examples/khora.standard.yaml
python examples/30_workloads/09_bulk_archive.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

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


_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY"]
_RELATIONSHIP_TYPES = ["MENTIONS", "RELATES_TO"]

# Vocabulary for the synthetic archive. Six authors over six channels,
# with a small canned vocabulary so the corpus has SOME extractable
# signal without being prose. The fact that ~75% of messages are
# colourless "ack" turns is the realistic shape — that's where Skeleton's
# cost savings come from (those chunks land below the importance
# threshold and don't go to the LLM).
_AUTHORS = ["alice", "bob", "chika", "dieter", "ed", "fatima"]
_CHANNELS = ["#engineering", "#design", "#deploys", "#oncall", "#random", "#announcements"]

_NOISE = [
    "ok",
    "thanks!",
    "got it",
    "+1",
    "lgtm",
    "ack",
    "will do",
    "looking now",
    ":eyes:",
]

_SIGNAL = [
    "We're deploying the new payment service to staging at 14:00 UTC.",
    "PagerDuty triggered for the search cluster — Bob is looking.",
    "Sales team wants a demo of the recommendations model next week.",
    "Migrated the user table to the new schema; rollback plan is in #deploys.",
    "Filed Linear issue PRO-441 about the broken OAuth flow on staging.",
    "The vendor finally fixed the CDN bug we hit on Tuesday.",
]


def _load_config() -> tuple[KhoraConfig, int]:
    """Parse CLI args; returns ``(config, n_messages)``.

    Unlike the other demos, demo 08 also accepts ``--n-messages`` so
    the corpus size can be cranked up for "feel the batch shape" runs.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--n-messages",
        type=int,
        default=50,
        help="Synthetic messages to ingest (default 50; bump to 500+ to see batch perf).",
    )
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config), args.n_messages


def _generate_archive(n: int, anchor: datetime) -> list[dict]:
    """Build a synthetic archive of ``n`` messages anchored at ``anchor``.

    Messages are spread uniformly across the 14 days **before** anchor.
    About 75% of messages come from the noise vocabulary so the corpus
    has the realistic "mostly low-signal" shape that motivates the
    Skeleton engine.

    Deterministic seed so the demo output is reproducible run-to-run.
    """
    rng = random.Random(0)  # noqa: S311 — non-crypto demo data
    window = timedelta(days=14)
    docs = []
    for i in range(n):
        # Uniform random offset across the window. A real archive
        # would be cyclical (more chatter during work hours) but
        # uniformity keeps the demo simple.
        offset = timedelta(seconds=rng.uniform(0, window.total_seconds()))
        when = anchor - window + offset
        author = rng.choice(_AUTHORS)
        channel = rng.choice(_CHANNELS)
        body = rng.choices([rng.choice(_SIGNAL), rng.choice(_NOISE)], weights=[1, 3])[0]
        docs.append(
            {
                "title": f"{channel} #{i}",
                "source": f"slack/{channel.strip('#')}/{i:04d}",
                "content": f"[{channel}] @{author}: {body}",
                "metadata": {
                    "occurred_at": when.isoformat(),
                    "author": author,
                    "channel": channel,
                },
            }
        )
    return docs


def _on_progress(processed: int, total: int) -> None:
    """remember_batch's progress hook — useful when ingesting bulk."""
    # Carriage-return so the line overwrites itself; final newline added
    # by the caller after batch.processed == batch.total.
    print(f"\r  …{processed}/{total} processed", end="", flush=True)


async def main() -> None:
    config, n_messages = _load_config()
    anchor = datetime.now(UTC)

    print(f"generating synthetic archive: {n_messages} messages across {len(_CHANNELS)} channels over the last 14 days")
    archive = _generate_archive(n_messages, anchor)

    async with Khora(config, engine="skeleton", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Bulk ingest ───────────────────────────────────────────
        # remember_batch shares the embedder LRU cache and deduplicates
        # entities across documents — the right shape for 10k+ docs.
        # max_concurrent controls how many docs run through the
        # pipeline simultaneously.
        print("ingesting (engine=skeleton, ~10% chunks → LLM)…")
        batch = await kb.remember_batch(
            archive,
            namespace=ns_id,
            max_concurrent=10,
            on_progress=_on_progress,
            entity_types=_ENTITY_TYPES,
            relationship_types=_RELATIONSHIP_TYPES,
        )
        # Newline after the progress line.
        print()
        print(
            f"  done: {batch.processed}/{batch.total} processed "
            f"({batch.failed} failed), "
            f"{batch.chunks} chunks, "
            f"{batch.entities} entities, "
            f"{batch.relationships} relationships"
        )

        # Cost-story note: with the Skeleton engine, the LLM was only
        # called for ~10% of chunks (the high-importance ones). For
        # 50 messages that's barely visible, but at 50k messages the
        # difference is the difference between $50 and $5 of OpenAI
        # spend on a single ingest.

        # ── Time-filtered recall ──────────────────────────────────
        # Ask about events in the last 3 days. Rather than rely on NLP
        # temporal detection, pass an explicit recall filter on the
        # event-time axis — filter={"occurred_at": {"$gte": ..., "$lt": ...}}
        # (lower bound inclusive, upper bound exclusive). It's deterministic
        # and pushes down into the SQL WHERE clause instead of fetching
        # everything then filtering in Python — that's what makes this scale.
        window_start = anchor - timedelta(days=3)
        print("\nQ (time-filtered): What happened in the last 3 days?")
        print(f"   window = [{window_start.isoformat()}, {anchor.isoformat()}]")
        result = await kb.recall(
            "What happened in the last 3 days?",
            namespace=ns_id,
            filter={"occurred_at": {"$gte": window_start, "$lt": anchor}},
            limit=5,
        )
        for chunk in result.chunks[:5]:
            age_days = (anchor - chunk.occurred_at).days if chunk.occurred_at else "?"
            preview = chunk.content[:90].replace("\n", " ")
            print(f"  [{chunk.score:.3f} | {age_days}d ago] {preview}")


if __name__ == "__main__":
    asyncio.run(main())
