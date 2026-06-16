"""Core API — recall with deterministic metadata filters.

``remember()`` carries structured provenance alongside the text: the
denormalized **system fields** (``source_name``, ``source_type``,
``source_timestamp``, ...) and a free-form **metadata** dict. ``recall()``
can then gate on those with ``filter=`` — a deterministic, operator-based
predicate that runs as a hard AND on top of the semantic ranking. A chunk
either satisfies the filter or it doesn't; relevance still orders what's left.

This is the precise, repeatable complement to vector search: "relevant AND
from Linear", "AND owned by the platform team", "AND priority ≥ 2".

Two filter forms, same result:
  * a plain ``dict`` — ``filter={"source_name": "linear"}`` (validated for you)
  * a typed ``RecallFilter`` — IDE autocomplete on the system keys

Filterable system keys: ``occurred_at`` / ``created_at`` / ``source_timestamp``
(dates), and ``source_type`` / ``source_name`` / ``source_url`` /
``external_id`` / ``content_type`` / ``source`` / ``title`` (strings). Plus the
``metadata`` blob and any ``metadata.<path>`` predicate.

Operators: ``$eq $ne $gt $gte $lt $lte $in $nin $exists`` and the logical
``$and $or $nor $not``. Bare-value sugar: a scalar is ``$eq``; a *list* is
exact-array equality (use ``$in`` for membership).

Engine choice: **vectorcypher** on the embedded backend, where the recall
filter compiles to a SQLite ``WHERE`` (system keys + JSON metadata pushdown)
and re-checks the rest in memory. ``result.engine_info["filter"]`` reports
exactly which keys were pushed down versus re-checked.

Run it
======
uv run python examples/10_core_apis/08_recall_with_metadata_filters.py
python examples/10_core_apis/08_recall_with_metadata_filters.py

# Production stack (postgres + neo4j):
python examples/10_core_apis/08_recall_with_metadata_filters.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from khora import Khora, RecallFilter
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"


# Sample corpus: engineering updates with provenance and metadata. The text is
# deliberately similar so the filters — not the query — decide what comes back.
# Edit freely; the filters below key on these fields.
_DOCS: list[dict[str, Any]] = [
    {
        "content": "Released v2.0 of the data ingest pipeline.",
        "source_name": "github",
        "source_type": "release",
        "source_timestamp": datetime(2026, 1, 15, tzinfo=UTC),
        "metadata": {"team": "ingest", "priority": 1, "tier": "gold"},
    },
    {
        "content": "Hot-fixed an out-of-memory regression in the ingest worker.",
        "source_name": "linear",
        "source_type": "ticket",
        "source_timestamp": datetime(2026, 2, 3, tzinfo=UTC),
        "metadata": {"team": "ingest", "priority": 1, "tier": "gold"},
    },
    {
        "content": "Wrote a design doc for cross-region replication.",
        "source_name": "notion",
        "source_type": "doc",
        "source_timestamp": datetime(2026, 2, 20, tzinfo=UTC),
        "metadata": {"team": "platform", "priority": 2, "tier": "silver"},
    },
    {
        "content": "Shipped the new admin UI for namespace management.",
        "source_name": "github",
        "source_type": "release",
        "source_timestamp": datetime(2026, 3, 10, tzinfo=UTC),
        "metadata": {"team": "frontend", "priority": 3, "tier": "bronze"},
    },
    {
        "content": "Triaged intermittent 500s on the recall endpoint.",
        "source_name": "linear",
        "source_type": "ticket",
        "source_timestamp": datetime(2026, 3, 22, tzinfo=UTC),
        "metadata": {"team": "platform", "priority": 2, "tier": "silver"},
    },
    {
        "content": "Closed all P0 tickets for the Q1 release.",
        "source_name": "linear",
        "source_type": "ticket",
        "source_timestamp": datetime(2026, 4, 1, tzinfo=UTC),
        "metadata": {"team": "platform", "priority": 1, "tier": "gold"},
    },
]

# Purely semantic, no temporal words: time is demonstrated only via filter= below,
# not smuggled into the query (which would trip VectorCypher's temporal detection).
_QUERY = "engineering work"


def _print(label: str, result: Any) -> None:
    """Print the matched chunks plus the honest pushdown report."""
    print(f"\n{label}")
    for chunk in result.chunks:
        print(f"  [{chunk.score:.2f}] {chunk.content}")
    if not result.chunks:
        print("  (no matches)")
    report = result.engine_info.get("filter")
    if report is not None:
        print(
            f"  pushed_down={report['pushed_down']}  pushed_keys={report['pushed_keys']}  post_filtered_keys={report['post_filtered_keys']}"
        )


async def main(config_path: Path) -> None:
    config = KhoraConfig.from_yaml(config_path)

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        for doc in _DOCS:
            await kb.remember(
                doc["content"],
                namespace=ns_id,
                source_name=doc["source_name"],
                source_type=doc["source_type"],
                source_timestamp=doc["source_timestamp"],
                metadata=doc["metadata"],
                entity_types=["EVENT", "CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

        # Baseline: no filter — every doc is a candidate, ranked by relevance.
        _print("no filter — all candidates:", await kb.recall(_QUERY, namespace=ns_id, limit=10))

        # 1. System key, scalar sugar: source_name == "linear" (the three tickets).
        _print(
            'filter={"source_name": "linear"}:',
            await kb.recall(_QUERY, namespace=ns_id, limit=10, filter={"source_name": "linear"}),
        )

        # 2. Metadata membership: team in {ingest, platform}.
        _print(
            'filter={"metadata.team": {"$in": ["ingest", "platform"]}}:',
            await kb.recall(
                _QUERY,
                namespace=ns_id,
                limit=10,
                filter={"metadata.team": {"$in": ["ingest", "platform"]}},
            ),
        )

        # 3. Metadata range on a numeric field: priority >= 2.
        _print(
            'filter={"metadata.priority": {"$gte": 2}}:',
            await kb.recall(_QUERY, namespace=ns_id, limit=10, filter={"metadata.priority": {"$gte": 2}}),
        )

        # 4. Logical AND across a system key and a metadata field — typed form.
        gold_tickets = RecallFilter.model_validate({"$and": [{"source_type": "ticket"}, {"metadata.tier": "gold"}]})
        _print(
            "filter=$and(source_type=ticket, metadata.tier=gold):",
            await kb.recall(_QUERY, namespace=ns_id, limit=10, filter=gold_tickets),
        )

        # 5. Time bound on the event axis. Prefer this over the deprecated
        #    start_time/end_time kwargs (they cannot be combined with filter=).
        _print(
            'filter={"source_timestamp": {"$gte": 2026-03-01}}:',
            await kb.recall(
                _QUERY,
                namespace=ns_id,
                limit=10,
                filter={"source_timestamp": {"$gte": datetime(2026, 3, 1, tzinfo=UTC)}},
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, help="Path to a Khora YAML config.")
    args = parser.parse_args()
    asyncio.run(main(args.config))
