"""Dream-phase rule extraction — Chronicle + event clustering.

The canonical demo for khora's **dream phase**: a background pass that
walks accumulated memory, finds near-duplicate observations, and
consolidates them into representative "rule" events.

Pattern inspired by Cognee's ``memify_coding_agent_rule_extraction_example``
ported to khora's primitives. The cluster-then-promote shape is the
closest thing khora ships to Letta's "sleeptime" consolidation.

Why Chronicle (engine="chronicle"):
Chronicle stores events with SVO (subject-verb-object) extraction and a
bi-temporal model. The dream-phase op ``CHRONICLE_EVENT_CLUSTERING``
groups events whose SVO-summary cosine ≥ threshold within a sliding
``referenced_date`` window — exactly the right shape for "the user
expressed this preference N times across sessions, treat it as a rule."

What the example demonstrates:

1. Ingest ~10 coding-assistant chat turns showing recurring patterns
   (tab vs space preference, test-first habits, language version).
2. Run a **dry-run** dream pass with ``OpKind.CLUSTER_EVENTS`` —
   returns a ``DreamResult`` with planned clusters, no side effects.
3. Run **apply** mode — clusters are committed via bi-temporal
   soft-delete; the namespace now has one representative event per
   cluster plus tombstones for the merged source events.
4. Recall against the consolidated namespace — see the "rule" view.

Note on output quality:
Without a real LLM, SVO extraction returns deterministic stubs that
don't necessarily produce meaningful clustering output. The example
exercises the dream-phase **API contract** (the plan/apply flow,
``DreamScope``, ``DreamConfig``); cluster *quality* requires
``OPENAI_API_KEY`` to be set (or another LLM provider via litellm).

Run it
======
# Default: embedded sqlite_lance, requires OPENAI_API_KEY
uv run python examples/30_workloads/05_dream_phase_consolidation.py
python examples/30_workloads/05_dream_phase_consolidation.py

# Switch to PostgreSQL + pgvector + Neo4j:
uv run python examples/30_workloads/05_dream_phase_consolidation.py --config examples/khora.standard.yaml
python examples/30_workloads/05_dream_phase_consolidation.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402
from khora.dream import DreamConfig, DreamOpsConfig, DreamScope  # noqa: E402
from khora.dream.plan import OpKind  # noqa: E402

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


# Recurring coding-preference signals scattered across simulated
# sessions. The first three each repeat 2-3 times with minor wording
# variation — that's the cluster-fodder. The last few are unique.
_OBSERVATIONS = [
    # "user prefers tabs" — cluster fodder x3
    ("session-001", "User said: I always use tabs, not spaces.", -7),
    ("session-002", "User confirmed: tab indentation is the only correct choice.", -5),
    ("session-003", "Reminder: this codebase uses tabs everywhere.", -2),
    # "user wants tests first" — cluster fodder x3
    ("session-001", "User said: write the test before the implementation.", -7),
    ("session-002", "Direction: tests first, no exceptions, even for prototypes.", -4),
    ("session-004", "TDD reminder — tests come first.", -1),
    # "uses Python 3.13" — cluster fodder x2
    ("session-003", "Target Python version: 3.13 only.", -2),
    ("session-005", "Codebase requires Python 3.13+.", 0),
    # One-offs (should not cluster)
    ("session-002", "User mentioned: deploys go out on Friday mornings.", -5),
    ("session-004", "User said: prefers Polars over Pandas for new code.", -1),
]

_ENTITY_TYPES = ["PERSON", "CONCEPT", "PRODUCT", "TECHNOLOGY", "PREFERENCE"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "PART_OF", "MENTIONS"]


async def _seed(kb: Khora, namespace_id: UUID) -> int:
    """Ingest the observation corpus with explicit occurred_at timestamps."""
    now = datetime.now(UTC)
    for session_id, text, day_offset in _OBSERVATIONS:
        when = now + timedelta(days=day_offset)
        await kb.remember(
            text,
            namespace=namespace_id,
            title=text[:60],
            metadata={"occurred_at": when.isoformat(), "session": session_id},
            entity_types=_ENTITY_TYPES,
            relationship_types=_RELATIONSHIP_TYPES,
        )
    return len(_OBSERVATIONS)


def _summarise_result(label: str, result) -> None:
    """Print the planned / applied op count and a short per-op trace."""
    print(f"\n=== {label} ===")
    print(f"  run_id:        {result.run.run_id}")
    print(f"  mode:          {result.run.mode}")
    print(f"  total_ops:     {len(result.ops)}")
    print(f"  applied:       {sum(1 for op in result.ops if getattr(op, 'applied', False))}")
    for op in result.ops[:5]:
        print(f"  • {op.kind} → decision={getattr(op, 'decision', '?')!s}")
    if len(result.ops) > 5:
        print(f"  … ({len(result.ops) - 5} more ops)")


async def main() -> None:

    # Chronicle is the right engine for SVO-shaped event clustering.
    # The CLUSTER_EVENTS op walks chronicle_events grouped by
    # (namespace_id, subject) within a sliding referenced_date window.
    config = _load_config()

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Seed ─────────────────────────────────────────────────────
        count = await _seed(kb, ns_id)
        stats = await kb.stats(namespace=ns_id)
        # NOTE: ``kb.stats(...).entities`` under-reports on the
        # sqlite_lance backend (SF-Fr22 — see `khorabugs/ISSUES.md`).
        # We also call list_entities() so the seed summary reflects
        # what's actually in the graph rather than what stats() thinks.
        listed_entities = await kb.list_entities(namespace=ns_id, limit=200)
        print(f"Seeded {count} observations.")
        print(
            f"Namespace docs={stats.documents}, chunks={stats.chunks}, "
            f"entities={len(listed_entities)} (via list_entities; stats.entities={stats.entities})"
        )

        # ── Dry-run dream pass ───────────────────────────────────────
        # Master switch defaults to disabled; the example flips it ON
        # for this run only via an explicit DreamConfig. The per-op
        # toggle is also off by default — opt in for cluster_events.
        dream_config = DreamConfig(
            enabled=True,
            ops=DreamOpsConfig(cluster_events=True),
        )
        scope = DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS,))

        dry = await kb.dream(ns_id, mode="dry-run", scope=scope, config=dream_config)
        _summarise_result("Dry-run plan", dry)

        # ── Apply ────────────────────────────────────────────────────
        # The apply pass commits the plan: cluster events get merged
        # via bi-temporal soft-delete, an undo.json snapshot is
        # written before any mutation, and the per-op transactions
        # roll back on failure. Five guardrails are in force (see
        # dream-phase.md): retention floor, KHORA_DREAM_DISABLE_APPLY
        # kill-switch, advisory lock, chunk_id assertion,
        # snapshot-before-mutate.
        applied = await kb.dream(ns_id, mode="apply", scope=scope, config=dream_config)
        _summarise_result("Apply result", applied)

        # ── Recall after the dream pass ──────────────────────────────
        # We label the output by what actually happened. If
        # ``applied.ops`` is non-empty, near-duplicate observations have
        # been clustered and the representative event/fact survives at
        # the top of the result with lower-rank duplicates suppressed.
        # If ``applied.ops`` is zero (the engine found nothing to
        # consolidate, e.g. when Chronicle didn't extract enough events
        # off the seeded chunks — see SF-Fr20 in `khorabugs/ISSUES.md`),
        # the recall returns the raw chunks unchanged. The honest label
        # below tells you which case you're in.
        recall = await kb.recall(
            "What are the user's coding preferences?",
            namespace=ns_id,
            limit=5,
        )
        if len(applied.ops) == 0:
            print(
                f"\n=== Recall after dream pass — {len(recall.chunks)} chunks (no consolidation ran; total_ops=0) ==="
            )
        else:
            print(
                f"\n=== Recall after dream pass — {len(recall.chunks)} chunks "
                f"(consolidated by {len(applied.ops)} ops) ==="
            )
        for chunk in recall.chunks:
            preview = chunk.content[:100].replace("\n", " ")
            print(f"  [{chunk.score:.3f}] {preview}{'…' if len(chunk.content) > 100 else ''}")

        # ── Inspect history ──────────────────────────────────────────
        # dream_history returns recent runs newest-first. Useful for
        # cron-driven consolidation jobs that want to verify "the
        # nightly run actually happened."
        #
        # SF-Fr23 (see `khorabugs/ISSUES.md`): this returns `[]` on
        # sqlite_lance even after successful dream() calls. The runs
        # do happen — both DreamResult.run.run_id values printed above
        # confirm it — they just don't land in the embedded backend's
        # history surface. Works correctly on postgres+neo4j.
        history = await kb.dream_history(ns_id, limit=5)
        if not history:
            print(
                f"\n=== Dream-run history (0 run(s)) — but {2 if applied else 1} ran "
                f"this session (SF-Fr23 on sqlite_lance) ==="
            )
        else:
            print(f"\n=== Dream-run history ({len(history)} run(s)) ===")
            # DreamRunInfo fields: run_id, namespace_id, mode, started_at,
            # finished_at, duration_ms, resume_of. No total_ops, no status —
            # those live on DreamResult, not on the run-info projection.
            for run in history:
                duration = f"{run.duration_ms:.0f}ms" if run.duration_ms is not None else "?"
                print(f"  {str(run.run_id)[:8]}  mode={run.mode}  duration={duration}")


if __name__ == "__main__":
    asyncio.run(main())
