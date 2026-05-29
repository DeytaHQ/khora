"""Workload 02 — Document Q&A with multi-signal abstention.

The classic "ingest a corpus, ask questions about it" demo, with one
question the original abstention scheme can't answer:

  **How do you distinguish "in-domain but unanchored" from "out-of-domain"?**

Both look identical to a single-threshold abstention scheme (the boolean
flags fire identically). But the underlying signal lives in the raw
vector cosine and the per-channel hit counts, which Chronicle exposes
on every recall via ``engine_info``.

This demo prints **three complementary signals per query** and lets the
caller see how to combine them:

  1. **Raw cosine band** of the top semantic hit
     (``engine_info["max_raw_vector_score"]``).
     HIGH ≥ 0.50 → confident
     MID 0.15–0.50 → in-domain but weak; needs corroboration
     LOW < 0.15 → out-of-domain, no signal

  2. **Retrieval breadth** — how many chunks each retrieval channel
     surfaced (``engine_info["channels"]``). A semantically dense corpus
     query returns multiple semantic-channel hits even when the top one
     has a modest cosine; a truly off-topic query returns one or zero.

  3. **Entity anchor strength** — count of entities returned. An
     entity-anchored in-domain query (Q1, Q2) gets entity-channel
     rescue even when the chunk-channel cosine is weak. An unanchored
     in-domain query (Q3) has no entity match. An off-topic query (Q4)
     has neither.

Combined, these three signals separate the four query types cleanly:

  - **Q1, Q2 entity-anchored in-domain**: MID cosine + entity_count > 0
    → answer (entity channel carries the rescue)
  - **Q3 unanchored in-domain**: MID cosine + entity_count = 0 +
    semantic_count ≥ 2 → answer with hedging (corpus does cover this,
    query just doesn't anchor)
  - **Q4 out-of-domain**: LOW cosine → abstain (no rescue possible)

This is the "two-band cosine + channel-count" approach from the dense
retrieval literature, simplified to fit on one screen. The deeper
production techniques — LLM-as-judge groundedness (Ragas, ARES),
Self-RAG critique tokens (Asai et al. 2023), Corrective RAG (Yan et al.
2024) — all build on the same foundation: a single threshold throws
away signal that the raw cosine + channel breadth still carry.

WHY CHRONICLE
=============
The retrieval shape (corpus-wide Q&A) doesn't need graph traversal, but
``engine_info["channels"]`` and ``max_raw_vector_score`` are
Chronicle-only telemetry. VectorCypher with full extraction
(``skeleton_core_ratio=1.0``) gives richer entity-aware answers but
doesn't surface the same diagnostic shape — see demo 08 for that lane.

DUAL-BACKEND SUPPORT
====================
Chronicle is production-ready on the standard PG+pgvector stack and
available on the embedded SQLite+LanceDB stack — both work.

REUSING INGESTED DATA
=====================
Ingest takes 30–90 seconds (8 docs × LLM extraction). Pass
``--reuse-data`` on subsequent runs to skip ingest and re-recall against
the namespace from the previous ingest. The namespace stable id is
persisted to ``.khora_demo_02_namespace`` in the current working
directory alongside the embedded DB. Re-running ingest (no flag)
creates a new namespace and overwrites the sidecar.

Run it
======
uv run python examples/30_workloads/02_document_qa_with_abstention.py
python examples/30_workloads/02_document_qa_with_abstention.py
uv run python examples/30_workloads/02_document_qa_with_abstention.py --reuse-data
python examples/30_workloads/02_document_qa_with_abstention.py --reuse-data
uv run python examples/30_workloads/02_document_qa_with_abstention.py --config examples/khora.standard.yaml
python examples/30_workloads/02_document_qa_with_abstention.py --config examples/khora.standard.yaml
uv run python examples/30_workloads/02_document_qa_with_abstention.py --data path/to/your_corpus.jsonl
python examples/30_workloads/02_document_qa_with_abstention.py --data path/to/your_corpus.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

# ── Logging setup ───────────────────────────────────────────────────────
logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_DEFAULT_DATA = Path(__file__).parent.parent / "data" / "hr_policies.jsonl"
_NAMESPACE_SIDECAR = Path.cwd() / ".khora_demo_02_namespace"
_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "PART_OF", "MENTIONS"]

# Cosine bands — the load-bearing thresholds. Calibrated for
# text-embedding-3-small on the HR policy corpus; recalibrate per
# corpus + embedding model.
_BAND_HIGH = 0.50
_BAND_LOW = 0.15


def _load_policies(path: Path) -> list[dict]:
    """Load HR policy documents from a JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus not found at {path!s}. Expected a JSONL file with one "
            f"document per line ({{'title': ..., 'source': ..., 'content': ...}}). "
            f"The repo ships {_DEFAULT_DATA.name} alongside this script — pass "
            f"`--data <path>` to point at a different corpus."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--data", type=Path, default=_DEFAULT_DATA, help=f"JSONL corpus path (default: {_DEFAULT_DATA.name})."
    )
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help=(
            "Skip ingest and reuse the namespace from a prior run. The stable "
            "namespace id is read from ``.khora_demo_02_namespace`` in the "
            "current directory."
        ),
    )
    return parser.parse_args()


def _band_for(raw_top: float) -> str:
    if raw_top >= _BAND_HIGH:
        return "HIGH"
    if raw_top >= _BAND_LOW:
        return "MID"
    return "LOW"


def _classify(engine_info: dict, entity_count: int) -> tuple[str, str, str]:
    """Combine raw cosine band + channel breadth + entity anchor into a verdict.

    Returns ``(band, channels_str, verdict)``. The verdict is the demo's
    pedagogical payload — it's the multi-signal answer to "should I
    answer this?" that a single boolean threshold cannot produce.
    """
    channels = engine_info.get("channels", {})
    raw_top = engine_info.get("max_raw_vector_score", 0.0)
    band = _band_for(raw_top)
    sem = channels.get("semantic", 0)
    bm25 = channels.get("bm25", 0)
    ent = channels.get("entity", 0)
    channels_str = f"sem={sem} bm25={bm25} ent={ent}"

    if band == "HIGH":
        verdict = "CONFIDENT → answer"
    elif band == "LOW":
        verdict = "OUT-OF-DOMAIN → abstain"
    elif entity_count > 0:
        verdict = "ENTITY-ANCHORED in-domain → answer"
    elif sem >= 2:
        verdict = "UNANCHORED in-domain → answer with hedging"
    else:
        verdict = "WEAK signal → abstain"
    return band, channels_str, verdict


async def _ask(kb: Khora, namespace, question: str) -> dict:
    """Run one recall, print the multi-signal classification, return a row dict for the summary table."""
    print(f"\nQ: {question}")
    result = await kb.recall(question, namespace=namespace, limit=4)
    if result.chunks:
        top = result.chunks[0]
        snippet = " ".join(top.content.split())[:140]
        print(f"   top chunk: {snippet}{'…' if len(top.content) > 140 else ''}")
    else:
        print("   (no chunks returned)")

    raw_top = result.engine_info.get("max_raw_vector_score", 0.0)
    signals = result.engine_info.get("abstention_signals", {})
    entity_count = len(result.entities)
    band, channels_str, verdict = _classify(result.engine_info, entity_count)

    flags = [k for k in ("entities_empty", "chunks_empty", "chunks_below_min", "top_score_low") if signals.get(k)]
    flags_str = ", ".join(flags) if flags else "∅"

    # The three-signal block. Each line corresponds to one of the
    # techniques the docstring outlines:
    #   raw cosine band   → cosine-band approach
    #   channel counts    → retrieval-breadth approach
    #   entity rescue     → entity-anchor approach
    # The "engine flags" line is the original boolean abstention summary,
    # kept so the reader can see how the multi-signal verdict differs
    # from the single-threshold scheme.
    print(f"   raw cosine     = {raw_top:.3f}    band={band}")
    print(f"   channel hits   = {channels_str}")
    print(f"   entities recalled = {entity_count}")
    print(
        f"   engine flags   = should_abstain={signals.get('should_abstain', False)}  combined={signals.get('combined_score', 0):.2f}  flags=[{flags_str}]"
    )
    print(f"   verdict        → {verdict}")

    return {
        "question": question,
        "raw_top": raw_top,
        "band": band,
        "channels_str": channels_str,
        "entity_count": entity_count,
        "engine_abstain": signals.get("should_abstain", False),
        "verdict": verdict,
    }


def _print_summary(rows: list[dict]) -> None:
    """Side-by-side comparison of the four queries.

    Reading this table is the demo's payoff: the single-threshold scheme
    (``engine_abstain``) groups Q3 and Q4 as identical refusals. The
    multi-signal verdict separates them — Q3 answers (with hedging), Q4
    abstains.
    """
    print("\n┌─ summary ─────────────────────────────────────────────────────────────────────┐")
    header = f"{'#':>2}  {'raw':>5}  {'band':>4}  {'ents':>4}  {'flags-abstain':>13}  verdict"
    print(header)
    for i, r in enumerate(rows, 1):
        flag = "abstain" if r["engine_abstain"] else "answer"
        print(f"{i:>2}  {r['raw_top']:>5.3f}  {r['band']:>4}  {r['entity_count']:>4}  {flag:>13}  {r['verdict']}")
    print("└───────────────────────────────────────────────────────────────────────────────┘")


async def _ingest(kb: Khora, ns_id: UUID, policies: list[dict]) -> None:
    """Ingest the corpus into ``ns_id``."""
    total_chunks = total_entities = 0
    for doc in policies:
        result = await kb.remember(
            doc["content"],
            namespace=ns_id,
            title=doc["title"],
            source=doc["source"],
            entity_types=_ENTITY_TYPES,
            relationship_types=_RELATIONSHIP_TYPES,
        )
        total_chunks += result.chunks_created
        total_entities += result.entities_extracted
    print(f"ingested: {len(policies)} docs, {total_chunks} chunks, {total_entities} entities")


async def main() -> None:
    args = _parse_args()
    config = KhoraConfig.from_yaml(args.config)

    # Cosine threshold dropped from 0.55 → 0.30 because text-embedding-3-small
    # produces lower raw cosines than the original calibration target. The
    # demo's pedagogy now lives in the multi-signal classification rather
    # than in the boolean flags themselves.
    engine_kwargs = {
        "abstention_min_top_score": 0.30,
        "abstention_min_chunks": 1,
        "abstention_combined_threshold": 0.5,
    }

    async with Khora(
        config,
        engine="chronicle",
        engine_kwargs=engine_kwargs,
        run_migrations=True,
    ) as kb:
        # ── Resolve namespace ──────────────────────────────────────
        # Two paths:
        #   • --reuse-data: read the prior namespace id from the
        #     sidecar, verify it still resolves (db not wiped),
        #     skip ingest.
        #   • default: create a fresh namespace, ingest, persist the
        #     id to the sidecar so the next --reuse-data run finds it.
        if args.reuse_data:
            if not _NAMESPACE_SIDECAR.is_file():
                raise SystemExit(
                    f"--reuse-data requires a prior ingest run. No namespace "
                    f"sidecar at {_NAMESPACE_SIDECAR!s}. Run without --reuse-data first."
                )
            ns_id_str = _NAMESPACE_SIDECAR.read_text().strip()
            ns_id = UUID(ns_id_str)
            existing = await kb.get_namespace_by_stable_id(ns_id)
            if existing is None:
                raise SystemExit(
                    f"namespace {ns_id!s} from {_NAMESPACE_SIDECAR.name} no longer "
                    f"resolves — the DB may have been wiped. Re-run without --reuse-data."
                )
            print(f"reusing namespace {ns_id!s} (sidecar={_NAMESPACE_SIDECAR.name})")
        else:
            namespace = await kb.create_namespace()
            ns_id = namespace.namespace_id
            policies = _load_policies(args.data)
            await _ingest(kb, ns_id, policies)
            _NAMESPACE_SIDECAR.write_text(str(ns_id))
            print(f"namespace {ns_id!s} persisted to {_NAMESPACE_SIDECAR.name}")

        # ── Q1: entity-anchored, named person ──────────────────────
        # The Travel policy names "Sarah Chen". Question anchors on an
        # extracted PERSON entity → entity_count > 0 carries the rescue
        # even though the raw chunk cosine is modest.
        rows = []
        rows.append(await _ask(kb, ns_id, "What does Sarah Chen approve?"))

        # ── Q2: entity-anchored, named vendor ──────────────────────
        # "MacBook Pro" appears as a PRODUCT entity. Lexical overlap is
        # strong here so the raw cosine is the highest of the four
        # queries, and the entity panel also fires. Both signals agree.
        rows.append(await _ask(kb, ns_id, "Can I buy a MacBook Pro from Apple?"))

        # ── Q3: in-domain but UNANCHORED ───────────────────────────
        # The Travel policy answers this directly, but the question's
        # phrasing ("per-night lodging limit") doesn't name any extracted
        # entity. Single-threshold abstention conflates this with Q4;
        # the multi-signal verdict separates them on channel breadth.
        rows.append(await _ask(kb, ns_id, "What is the per-night lodging limit?"))

        # ── Q4: out-of-domain ──────────────────────────────────────
        # Corpus has nothing about photosynthesis. Raw cosine collapses
        # to the LOW band; verdict = abstain on the cosine alone.
        rows.append(await _ask(kb, ns_id, "How does photosynthesis work in plants?"))

        _print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
