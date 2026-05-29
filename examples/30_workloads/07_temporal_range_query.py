"""Temporal range queries — Chronicle's killer demo.

Khora's Chronicle engine is **event-shaped**: every chunk carries an
``occurred_at`` timestamp distinct from when it was ingested. That
distinction lets ``Khora.recall(..., start_time=..., end_time=...)``
answer questions like "what happened between February and April?" by
filtering at the SQL/Cypher layer before semantic ranking — much
cheaper than post-filtering a vector hit-list.

This example seeds ~30 events spanning six simulated months, then
runs increasingly specific time-bounded queries:

1. Unconstrained recall — semantic match only, no time window.
2. Single-month window — recall scoped to one month.
3. Quarter window — three months around a topic.
4. Anchored to a specific reference date — "what was Alice working on
   around January 15?"
5. Negative-test — query a month with no relevant events; show
   Chronicle's abstention signal firing.

Pattern from Cognee's ``temporal_awareness_example.py`` ported to
khora's primitives. Cognee uses ``SearchType.TEMPORAL``; khora's
equivalent is the ``start_time`` / ``end_time`` kwargs on
``Khora.recall()`` — applied via SQL pushdown so the time filter
doesn't pay for embeddings outside the window.

Why Chronicle (engine="chronicle"):
The bi-temporal model is the whole point. VectorCypher honors temporal
filters too but spends extraction cost on every chunk; Chronicle's
selective-extraction + temporal indexing pair makes time-bounded
queries cheap. For an event-stream workload (chat logs, audit trails,
release-cycle data) Chronicle is the right choice.

Loads YAML config via ``--config`` (default ``khora.embedded.yaml``); switch to PG+Neo4j with ``--config examples/khora.standard.yaml``. Requires ``OPENAI_API_KEY``.
With the mock LLM, embeddings are deterministic-by-text-hash so
semantic ranking is stable across runs; only the time-window filter
varies.

Run it
======
uv run python examples/30_workloads/07_temporal_range_query.py
python examples/30_workloads/07_temporal_range_query.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402

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


_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "PRODUCT", "EVENT"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "PART_OF", "MENTIONS"]


# Synthetic six-month corpus of release-engineering chatter. Each
# entry is (year, month, day, text). Topics threaded across months:
# - "deploy pipeline" (Jan, Feb, Mar)
# - "Alice / postgres migration" (Jan-Feb)
# - "security audit" (Mar-Apr)
# - "rebrand launch" (May-Jun)
# - "Bob / hiring" (Apr-May)
_CORPUS = [
    # January
    (2026, 1, 5, "Alice kicked off the postgres migration planning today."),
    (2026, 1, 12, "Deploy pipeline now uses GitHub Actions with manual promote."),
    (2026, 1, 18, "Alice mapped the schema diff for the postgres migration."),
    (2026, 1, 25, "Deploy pipeline failed once — env-var rotation bug, fixed."),
    # February
    (2026, 2, 3, "Postgres migration tested against staging; rollback drilled."),
    (2026, 2, 10, "Alice cut over to postgres in prod; 4-hour maintenance window."),
    (2026, 2, 17, "Deploy pipeline gained smoke-test gate before promote."),
    (2026, 2, 24, "Deploy pipeline P95 dropped from 22min to 8min."),
    # March
    (2026, 3, 4, "Security audit kicked off; SOC 2 Type II prep."),
    (2026, 3, 11, "Deploy pipeline added secret-scanning step (security audit ask)."),
    (2026, 3, 18, "Auditor reviewed deploy pipeline access controls."),
    (2026, 3, 25, "Security audit found one finding: stale IAM role; remediated."),
    # April
    (2026, 4, 2, "Security audit closed with no open findings."),
    (2026, 4, 9, "Bob started — first task: deploy pipeline observability."),
    (2026, 4, 16, "Bob proposed splitting the deploy pipeline into stages."),
    (2026, 4, 23, "Hiring plan: two more SRE roles, target Q3 start."),
    # May
    (2026, 5, 1, "Rebrand launch prep started — new domain procurement."),
    (2026, 5, 8, "Bob owns deploy pipeline observability now; on-call rotation."),
    (2026, 5, 15, "Rebrand launch date set: end of June."),
    (2026, 5, 22, "Hiring: SRE #1 signed, starts in June."),
    # June
    (2026, 6, 5, "Rebrand launch in two weeks; freeze non-critical merges."),
    (2026, 6, 12, "Rebrand launch postponed by one week — DNS issue."),
    (2026, 6, 19, "Rebrand launched. Marketing site live, blog migrated."),
    (2026, 6, 26, "Rebrand post-mortem: DNS issue root-caused to TTL config."),
    # July (out-of-window noise to test boundary handling)
    (2026, 7, 5, "Post-rebrand traffic up 18%."),
    (2026, 7, 12, "Hiring: SRE #1 onboarded, ramping."),
]


async def _seed(kb: Khora, namespace_id: UUID) -> None:
    """Ingest the corpus with explicit per-event ``occurred_at``.

    ``source_timestamp=`` is the load-bearing kwarg — Chronicle reads it,
    writes it to ``chunk.occurred_at``, and indexes it for the
    ``start_time`` / ``end_time`` filter pushdown. The ``metadata`` dict
    is opaque storage that the engine never consults; stuffing the
    timestamp there silently leaves every chunk with ``occurred_at=None``
    and breaks every windowed query that doesn't happen to span the
    ingest date (Chronicle then falls back to ``created_at``).
    """
    for year, month, day, text in _CORPUS:
        when = datetime(year, month, day, 10, 0, tzinfo=UTC)
        await kb.remember(
            text,
            namespace=namespace_id,
            title=text[:60],
            source_timestamp=when,
            entity_types=_ENTITY_TYPES,
            relationship_types=_RELATIONSHIP_TYPES,
        )


async def _query(
    kb: Khora,
    namespace_id: UUID,
    *,
    label: str,
    question: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> None:
    """Run a recall, print the top hits with their occurred_at."""
    result = await kb.recall(
        question,
        namespace=namespace_id,
        start_time=start_time,
        end_time=end_time,
        limit=4,
    )
    window = ""
    if start_time and end_time:
        window = f" [{start_time.date()} … {end_time.date()}]"

    print(f"\n=== {label}{window} ===")
    print(f"Q: {question}")
    if not result.chunks:
        print("  (no hits)")
    for chunk in result.chunks:
        ts = chunk.occurred_at.date() if chunk.occurred_at else "?"
        preview = chunk.content[:90].replace("\n", " ")
        print(f"  [{chunk.score:.3f} | {ts}] {preview}{'…' if len(chunk.content) > 90 else ''}")

    signals = result.engine_info.get("abstention_signals", {})
    if signals.get("should_abstain"):
        print(
            f"  ⚠ abstain — combined={signals.get('combined_score', 0):.2f}, "
            f"chunks_below_min={signals.get('chunks_below_min')}"
        )


async def main() -> None:
    config = _load_config()

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        await _seed(kb, ns_id)
        stats = await kb.stats(namespace=ns_id)
        print(f"Seeded {stats.documents} events across 7 months.")

        # 1) Unconstrained — pulls top semantic matches across all
        # months. Useful as a baseline against the windowed queries.
        await _query(
            kb,
            ns_id,
            label="Unconstrained",
            question="What was the deploy pipeline status?",
        )

        # 2) Single-month window — February only. Should surface the
        # cutover + smoke-test additions; January's planning and
        # March's audit-driven changes are out of window.
        await _query(
            kb,
            ns_id,
            label="February only",
            question="What happened with the deploy pipeline?",
            start_time=datetime(2026, 2, 1, tzinfo=UTC),
            end_time=datetime(2026, 2, 28, 23, 59, tzinfo=UTC),
        )

        # 3) Quarter window — Mar–May. Pulls the security-audit story
        # and the start of the rebrand prep.
        await _query(
            kb,
            ns_id,
            label="Q2-ish (Mar–May)",
            question="What security work was done?",
            start_time=datetime(2026, 3, 1, tzinfo=UTC),
            end_time=datetime(2026, 5, 31, 23, 59, tzinfo=UTC),
        )

        # 4) Anchored to a specific date — "around January 15".
        # Two-week window centered on the anchor.
        anchor = datetime(2026, 1, 15, tzinfo=UTC)
        await _query(
            kb,
            ns_id,
            label="Anchored: around 2026-01-15",
            question="What was Alice working on?",
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            end_time=datetime(2026, 1, 31, 23, 59, tzinfo=UTC),
        )

        # 5) Negative — query a month with no relevant events. The
        # corpus has nothing about "rebrand" in January; Chronicle
        # should abstain via the combined-score threshold.
        await _query(
            kb,
            ns_id,
            label="Empty window (Jan rebrand)",
            question="What rebrand work happened?",
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            end_time=datetime(2026, 1, 31, 23, 59, tzinfo=UTC),
        )

        _ = anchor  # silence linter; anchor documents intent in label above


if __name__ == "__main__":
    asyncio.run(main())
