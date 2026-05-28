"""Workload 01 — Per-user preferences with temporal drift.

Two Chronicle properties carry this demo:

  1. **Abstention on a cold namespace.** First recall against a new
     user's namespace fires the four-flag abstention block
     (``chunks_empty``, ``chunks_below_min``, ``top_score_low``,
     ``entities_empty``) plus the convenience ``should_abstain``
     boolean. The downstream LLM agent reads ``should_abstain`` and
     refuses to answer instead of hallucinating into a vacuum.

  2. **Recency-aware ranking without explicit forget().** Users
     change their minds. Alice was a tea drinker six months ago, a
     coffee person three months ago, and switched to matcha last
     week. All three statements stay in the namespace; **temporal
     decay ranks them by event time** so a recall for "what does
     Alice prefer to drink?" returns matcha at the top.

     This works because Chronicle's Ebbinghaus decay multiplies the
     relevance score by a retention curve keyed off the chunk's
     ``occurred_at`` (sourced from ``source_timestamp=`` at ingest).
     Fresh statements keep ~100% of their semantic score; statements
     N half-lives old keep ``0.5^N``. The decay weight knob blends
     decay with raw relevance — at ``chronicle_decay_weight=0.7``
     and ``temporal_half_life_hours=720`` (30 days), six months of
     drift drops a chunk to ~30% of its raw score, more than enough
     to flip the ranking against a same-topic competitor that's only
     a week old.

     If you need *deterministic* ordering — "the old preference is
     gone, full stop, no chance of resurrection by a future tuning
     change" — use the ``forget()`` + ``remember()`` pattern from
     ``00_quickstart/03_forget_what_was_wrong.py``. Recency-based
     drift is the right tool when the historical statements still
     have *some* value (audit, "what was their position back then?")
     even if they shouldn't dominate the current answer.

WHY CHRONICLE
=============
Two reasons no other engine fits as cleanly:

  * Abstention signals are **Chronicle-only**. VectorCypher's recall
    will happily return the closest neighbour even when there's
    nothing on-topic; Chronicle exposes a passive "should I refuse?"
    signal on every recall result.
  * Decay is **Chronicle-only**. VectorCypher has soft temporal
    scoring as one of its fusion channels, but Chronicle's
    Ebbinghaus retention curve is the one that maps cleanly onto
    "memory grows weaker with age" — which is exactly the user-
    preferences workload.

No graph backend needed; identical code runs on the embedded SQLite
path and on PostgreSQL+pgvector.

Run it
======
uv run python examples/30_workloads/01_per_user_preferences.py
python examples/30_workloads/01_per_user_preferences.py
uv run python examples/30_workloads/01_per_user_preferences.py --config examples/khora.standard.yaml
python examples/30_workloads/01_per_user_preferences.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)

_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

_ENTITY_TYPES = ["PERSON", "CONCEPT", "PRODUCT"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "MENTIONS"]


def _load_config() -> KhoraConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    config = KhoraConfig.from_yaml(args.config)

    # Aggressive temporal decay for this demo.
    #
    # Defaults in v0.17.3 are conservative (weight=0.30, half_life=168h)
    # so production deployments don't accidentally bury still-relevant
    # historical memories. For a "preferences drift over months" story
    # we want decay to dominate same-topic recall:
    #   30-day half-life  → 6mo-old prefs decay to retention ≈ 0.016
    #   weight 0.7        → multiplier = 0.3 + 0.7 · retention ≈ 0.31
    #                       (vs. ~0.95 for a 7-day-old chunk)
    # Net effect: a 7-day-old statement outranks a 6mo-old statement
    # of equal raw relevance by ~3×.
    config.query.chronicle_decay_weight = 0.7
    config.query.temporal_half_life_hours = 720.0  # 30 days

    # Disable cross-encoder reranking for this demo. Chronicle's recall
    # pipeline applies decay then runs a cross-encoder rerank
    # (``enable_reranking`` defaults True), and the cross-encoder
    # re-scores chunks on raw query↔chunk semantic similarity — which
    # discards the decay-influenced ordering. For a recency-drift
    # demo we want the reader to see decay dominate, so we turn the
    # reranker off. In production, leaving reranking on is the right
    # call when accuracy matters more than recency; the two pulls in
    # opposite directions is a real Chronicle tradeoff worth knowing.
    config.query.enable_reranking = False
    return config


def _print_abstention(engine_info: dict) -> None:
    signals = engine_info.get("abstention_signals", {})
    if not signals:
        print("  (no abstention metadata — engine is not Chronicle)")
        return
    print(
        f"  abstention: should_abstain={signals['should_abstain']} "
        f"combined={signals['combined_score']:.2f} "
        f"(chunks_empty={signals['chunks_empty']}, "
        f"chunks_below_min={signals['chunks_below_min']}, "
        f"top_score_low={signals['top_score_low']}, "
        f"entities_empty={signals['entities_empty']})"
    )


def _age_label(occurred_at, now: datetime) -> str:
    if occurred_at is None:
        return "?"
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    delta_days = (now - occurred_at).days
    if delta_days < 14:
        return f"{delta_days}d ago"
    if delta_days < 90:
        return f"{delta_days // 7}w ago"
    return f"{delta_days // 30}mo ago"


async def main() -> None:
    config = _load_config()
    now = datetime.now(UTC)

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        # ──────────────────────────────────────────────────────────────
        # Part 1 — Abstention on a cold namespace
        # ──────────────────────────────────────────────────────────────
        bob_id = (await kb.create_namespace()).namespace_id

        print("=== Bob (cold namespace): 'what does Bob prefer to drink?' ===")
        bob_result = await kb.recall("What does Bob prefer to drink?", namespace=bob_id)
        print(f"  chunks returned   = {len(bob_result.chunks)}")
        print(f"  entities returned = {len(bob_result.entities)}")
        _print_abstention(bob_result.engine_info)

        if bob_result.engine_info.get("abstention_signals", {}).get("should_abstain"):
            print("  → downstream LLM should refuse to answer (cold namespace)")

        # ──────────────────────────────────────────────────────────────
        # Part 2 — Alice's preferences drift over 6 months
        # ──────────────────────────────────────────────────────────────
        # Three statements about the same topic (what to drink) at
        # progressively recent timestamps. The ingest order doesn't
        # matter — what matters is each chunk's source_timestamp,
        # which Chronicle reads as occurred_at and feeds into the
        # decay curve.
        alice_id = (await kb.create_namespace()).namespace_id

        timeline = [
            ("I prefer English breakfast tea with milk in the morning.", now - timedelta(days=180)),
            ("I'm into pour-over coffee now — single-origin Ethiopian, no milk.", now - timedelta(days=90)),
            ("Matcha latte. Started last week and I'm not going back.", now - timedelta(days=7)),
        ]

        print("\n=== Alice: ingesting 6 months of drink-preference statements ===")
        for content, when in timeline:
            await kb.remember(
                content,
                namespace=alice_id,
                source_timestamp=when,
                entity_types=_ENTITY_TYPES,
                relationship_types=_RELATIONSHIP_TYPES,
            )
            print(f"  [{_age_label(when, now):>7}]  {content}")

        # ── The recall ────────────────────────────────────────────────
        # All three chunks are semantically equally relevant to the
        # query (each is a drink preference). With decay configured
        # aggressively above, the ranking is dominated by recency:
        # matcha (1w) > coffee (3mo) > tea (6mo).
        print("\n=== Alice: 'What does Alice prefer to drink?' ===")
        result = await kb.recall("What does Alice prefer to drink?", namespace=alice_id, limit=5)
        for chunk in result.chunks:
            age = _age_label(chunk.occurred_at, now)
            preview = chunk.content[:80].replace("\n", " ")
            print(f"  [{chunk.score:.3f} | {age:>7}]  {preview}{'…' if len(chunk.content) > 80 else ''}")
        _print_abstention(result.engine_info)

        # Same data, recall framed as "what did Alice prefer last
        # winter?" — point-in-time intent. Today's `recall()` doesn't
        # accept a target_date kwarg on this engine, so the framing
        # has to live in the user-facing prompt: the LLM reads the
        # full ranked list and the ``occurred_at`` on each chunk, and
        # picks the one that matches the question's timeframe.
        print("\n=== Alice: 'What was Alice drinking around January?' ===")
        result = await kb.recall("What was Alice drinking around January?", namespace=alice_id, limit=5)
        for chunk in result.chunks:
            age = _age_label(chunk.occurred_at, now)
            preview = chunk.content[:80].replace("\n", " ")
            print(f"  [{chunk.score:.3f} | {age:>7}]  {preview}{'…' if len(chunk.content) > 80 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
