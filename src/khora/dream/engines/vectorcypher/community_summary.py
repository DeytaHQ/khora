"""Vectorcypher community detection + per-community LLM summary (#670).

Phase 5.1 of the dream-phase rollout (umbrella #649). This is the **first
LLM-using dream op** — it pioneers the LLM-budget integration that the
remaining Phase 5 ops will reuse.

Pipeline:

1. ``plan_vectorcypher_community_summary`` walks the entity-relationship
   graph in a namespace, builds a weighted edge list with
   ``ASSOCIATED_WITH`` co-occurrence edges down-weighted to 0.2, and runs
   community detection via :func:`khora._accel.detect_communities`.

   .. note::

      The kernel is **single-pass Louvain** today. The umbrella ticket
      (#670 sub-bullet) calls for a Rust-kernel Leiden upgrade replacing
      the current implementation at ``_accel.py:1497``. That replacement
      is tracked as a follow-up; the planner consumes whatever the kernel
      returns and is not coupled to the specific algorithm.

   One :class:`DreamOp` is emitted per community whose size is at least
   ``min_size``. Each op carries the community member ids, names, and
   the top relationship modes (relationship_type → count) so the apply
   handler can ground its prompt without re-querying the graph.

2. ``apply_vectorcypher_community_summary`` issues one LLM call per
   planned op via :func:`khora.config.llm.acompletion` with
   ``_telemetry_op="dream_community_summary"`` — that wrapper records
   tokens against the dream-phase budget bookkeeping. The model is
   **configurable** (caller-supplied, defaults to ``gpt-4o-mini``).

   The LLM is asked to return a JSON object matching :class:`GroundedSummary`
   (text + a list of :class:`SummaryClaim`). Every claim must cite the
   member ``entity_id``s that support it. A post-hoc grounding
   validator (:func:`validate_grounding`) drops:

   * claims with no citations, and
   * claims citing an ``entity_id`` not in the community's member set
     (defends against fabrication).

   The kept claims are persisted to the ``khora_dream_communities`` table
   (Postgres-only in v0.15; SQLite raises :class:`DreamForbiddenOpError`).
   Each row carries bi-temporal ``valid_from`` / ``valid_to`` columns.
   Replaying the same op_id against an already-live row returns a noop
   :class:`UndoRecord`.

   Always re-summarises from raw members + chunks, never from prior
   summaries — the planner reads ``coordinator.list_entities`` /
   ``list_relationships`` directly. Summary embeddings carry
   ``summary_depth=1`` metadata; retrieval-side enforcement that
   summary-depth pools must not mix depths lives in the retrieval layer
   (out of scope for this ticket).

Cost framing
------------

At ``gpt-4o-mini`` rates, a single dream cycle over a 100k-entity
namespace runs ~$15-25 in API spend. The dream orchestrator enforces two
token budgets read off :class:`khora.dream.DreamConfig`:
``llm_max_tokens_per_run`` (default 200k) and
``llm_max_tokens_per_namespace_per_day`` (default 1M). Each
``acompletion`` call here records its spend through the context-local
usage path; the orchestrator reads that spend back and checks both
budgets *before* dispatching the next community-summary op, skipping the
remaining ops (and emitting ``khora.dream.llm.throttled_total``) once a
budget is exhausted (#1270). Operators control the cost surface via the
configurable ``community_summary_*`` knobs on
:class:`khora.dream.DreamConfig`.

Stability
---------

Internal — Phase 5.1 of an active rollout. The DreamOp inputs/outputs
shape may evolve while Phase 5 settles.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora import _accel
from khora.config.llm import LiteLLMConfig, acompletion
from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import bounded_text_hash, trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.core.models.entity import Entity
    from khora.storage.coordinator import StorageCoordinator


# Canonical co-occurrence relationship label emitted by selective
# extraction. Matched case-insensitively so adapter-specific casing
# (Neo4j upper-case vs SurrealDB free-form) does not slip through.
_COOCCURRENCE_REL_TYPE = "ASSOCIATED_WITH"

# Default weight for ``ASSOCIATED_WITH`` co-occurrence edges in the
# community-detection graph. Below 1.0 so co-occurrence cannot glue
# otherwise-disconnected clusters together.
_COOCCURRENCE_EDGE_WEIGHT = 0.2

# Default LLM model — kept configurable on the apply handler so callers
# can swap to ``gpt-4o`` for higher quality at higher cost.
_DEFAULT_MODEL = "gpt-4o-mini"

# Default ``max_members_per_prompt`` — bounded so a single community of
# 1k entities cannot run an unbounded LLM context.
_DEFAULT_MAX_MEMBERS = 20

# Default min cluster size — communities of fewer members are not worth
# summarising and the LLM cost is wasted.
_DEFAULT_MIN_SIZE = 5

_PHASE_PLAN = "audit"
_PHASE_APPLY = "mutation"


# ---------------------------------------------------------------------------
# Grounding contract
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SummaryClaim:
    """A single grounded claim inside an :class:`GroundedSummary`.

    ``cited_entity_ids`` is the list of community member ``entity_id``s
    (as strings) that the LLM cites as evidence. The grounding validator
    drops a claim if this list is empty or if any cited id is unknown.
    """

    text: str
    cited_entity_ids: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class GroundedSummary:
    """Structured LLM output for one community summary.

    ``text`` is the 1-2 sentence natural-language summary. ``claims`` is
    the per-claim breakdown with citations — the validator scrubs this
    list before persistence.
    """

    text: str
    claims: list[SummaryClaim] = field(default_factory=list)


def validate_grounding(
    summary: GroundedSummary,
    member_ids: set[str],
) -> tuple[list[SummaryClaim], list[SummaryClaim]]:
    """Split ``summary.claims`` into ``(kept, dropped)`` by citation rule.

    A claim is kept iff:

      * ``cited_entity_ids`` is non-empty, AND
      * every cited id is in ``member_ids`` (no fabrication).

    Returns ``(kept, dropped)`` so the caller can record a dropped-count
    for audit. The validator is intentionally strict — partial-grounding
    weakens the abstention story downstream.
    """
    kept: list[SummaryClaim] = []
    dropped: list[SummaryClaim] = []
    for claim in summary.claims:
        if not claim.cited_entity_ids:
            dropped.append(claim)
            continue
        if any(cid not in member_ids for cid in claim.cited_entity_ids):
            dropped.append(claim)
            continue
        kept.append(claim)
    return kept, dropped


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


async def plan_vectorcypher_community_summary(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    min_size: int = _DEFAULT_MIN_SIZE,
    cooccurrence_edge_weight: float = _COOCCURRENCE_EDGE_WEIGHT,
    max_members_per_prompt: int = _DEFAULT_MAX_MEMBERS,
) -> tuple[DreamOp, ...]:
    """Detect communities + emit one planned :class:`DreamOp` per cluster.

    Args:
        namespace_id: Namespace to scan.
        coordinator: Storage coordinator (DI for tests).
        min_size: Minimum cluster size to emit. Default 5.
        cooccurrence_edge_weight: Weight for ``ASSOCIATED_WITH`` edges.
            Default 0.2 (down-weighted from 1.0).
        max_members_per_prompt: Cap on member ids carried in the op
            payload — bounds the apply-side prompt size. Default 20.

    Returns:
        One :class:`DreamOp` per community above ``min_size``. Empty
        tuple when the namespace has no entities or no community clears
        the threshold.
    """
    op_id_audit = uuid4()
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.community_summary",
        op_id=str(op_id_audit),
        namespace_id=str(namespace_id),
        phase=_PHASE_PLAN,
        weighting_used=cooccurrence_edge_weight,
    ) as span:
        entities = await coordinator.list_entities(namespace_id, limit=100_000)
        total_entities = len(entities)
        span.set_attribute("total_entities", total_entities)

        if total_entities == 0:
            span.set_attribute("community_count", 0)
            return ()

        relationships = await coordinator.list_relationships(namespace_id, limit=1_000_000)

        index_by_id: dict[UUID, int] = {e.id: idx for idx, e in enumerate(entities)}
        edges: list[tuple[int, int, float]] = []
        rel_modes: dict[int, Counter[str]] = {}  # community-local later

        for rel in relationships:
            src_idx = index_by_id.get(rel.source_entity_id)
            tgt_idx = index_by_id.get(rel.target_entity_id)
            if src_idx is None or tgt_idx is None:
                continue
            weight = cooccurrence_edge_weight if rel.relationship_type.upper() == _COOCCURRENCE_REL_TYPE else 1.0
            # Detect_communities expects undirected — push both directions.
            edges.append((src_idx, tgt_idx, weight))
            edges.append((tgt_idx, src_idx, weight))

        labels = _accel.detect_communities(total_entities, edges)

        # Group entity indices by community label (skip -1 isolated nodes).
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(idx)

        # Per-community relationship-mode counts via a second pass.
        for rel in relationships:
            src_idx = index_by_id.get(rel.source_entity_id)
            tgt_idx = index_by_id.get(rel.target_entity_id)
            if src_idx is None or tgt_idx is None:
                continue
            label_src = labels[src_idx]
            label_tgt = labels[tgt_idx]
            if label_src != -1 and label_src == label_tgt:
                bucket = rel_modes.setdefault(label_src, Counter())
                bucket[rel.relationship_type] += 1

        ops: list[DreamOp] = []
        for label, member_idxs in clusters.items():
            if len(member_idxs) < min_size:
                continue
            ops.append(
                _build_planned_op(
                    namespace_id=namespace_id,
                    entities=entities,
                    member_idxs=member_idxs,
                    rel_modes=dict(rel_modes.get(label, Counter())),
                    max_members_per_prompt=max_members_per_prompt,
                    cooccurrence_edge_weight=cooccurrence_edge_weight,
                    started_at=started_at,
                )
            )

        span.set_attribute("community_count", len(ops))
        span.set_attribute("duration_ms", (time.perf_counter() - t0) * 1000.0)

    return tuple(ops)


def _build_planned_op(
    *,
    namespace_id: UUID,
    entities: list[Entity],
    member_idxs: list[int],
    rel_modes: dict[str, int],
    max_members_per_prompt: int,
    cooccurrence_edge_weight: float,
    started_at: datetime,
) -> DreamOp:
    """Construct one ``decision="planned"`` :class:`DreamOp` per community.

    Top-k members (by ``mention_count``) are carried in the payload so
    the apply handler can render a prompt without re-loading entities.
    The community_id is deterministically derived so a replay produces
    the same id for the same member set.
    """
    members = [entities[i] for i in member_idxs]
    # Top-k by mention_count, falling back to name for stable ordering.
    members.sort(key=lambda e: (-e.mention_count, e.name))
    top_k = members[:max_members_per_prompt]

    community_id = _derive_community_id(namespace_id, [e.id for e in members])

    payload = {
        "community_id": str(community_id),
        "cluster_size": len(members),
        "member_ids": [str(e.id) for e in top_k],
        "member_names": [e.name for e in top_k],
        "member_types": [e.entity_type for e in top_k],
        "total_members": len(members),
        "relationship_modes": rel_modes,
        "cooccurrence_edge_weight": cooccurrence_edge_weight,
    }

    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE_PLAN,
        op_type=OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
        inputs=(payload,),
        outputs=(),
        decision="planned",
        rationale=(
            f"Community of {len(members)} entities flagged for summarisation; "
            f"top relationship modes: {', '.join(f'{k}={v}' for k, v in rel_modes.items()) or 'n/a'}."
        ),
        started_at=started_at,
        namespace_id=namespace_id,
    )


def _derive_community_id(namespace_id: UUID, member_ids: list[UUID]) -> UUID:
    """Stable UUID5 of (namespace_id, sorted member ids).

    Determinism matters for idempotent replay — the same membership set
    must produce the same community_id across runs so the apply handler
    can short-circuit when the row already exists.
    """
    import hashlib

    sorted_ids = sorted(str(mid) for mid in member_ids)
    digest = hashlib.sha1(  # noqa: S324
        ("|".join([str(namespace_id), *sorted_ids])).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return UUID(digest[:32])


# ---------------------------------------------------------------------------
# Apply handler
# ---------------------------------------------------------------------------


_SUMMARY_SYSTEM_PROMPT = (
    "You write 1-2 sentence summaries of small communities of related "
    "entities in a knowledge graph. Output strict JSON matching this "
    "schema:\n\n"
    "  {\n"
    '    "text": "<1-2 sentence summary>",\n'
    '    "claims": [\n'
    '      {"text": "<short factual claim>", "cited_entity_ids": ["<member id>", ...]}\n'
    "    ]\n"
    "  }\n\n"
    "Every claim MUST cite at least one community member entity_id from "
    "the provided list. Do NOT invent entity_ids. Do NOT cite ids outside "
    "the provided list. Claims without citations will be dropped."
)


async def apply_vectorcypher_community_summary(
    op: DreamOp,
    *,
    coordinator: StorageCoordinator | None,
    session: AsyncSession,
    model: str = _DEFAULT_MODEL,
) -> UndoRecord:
    """Apply one planned community-summary op — caller owns the transaction.

    Single LLM call per op through :func:`khora.config.llm.acompletion`
    with ``_telemetry_op="dream_community_summary"`` so the dream-phase
    rolling-hour token budget records the spend. The returned summary is
    parsed into a :class:`GroundedSummary`, the grounding validator
    drops uncited or fabricated claims, and the remaining payload is
    written to the ``khora_dream_communities`` table.

    Postgres-only in v0.15. On SQLite, the embedded path lands later
    once the orchestrator's session-aware SQLite wiring matures.

    Replay safety: the ``community_id`` is a deterministic UUID5 of
    ``(namespace_id, sorted member ids)``. A second run on the same
    membership finds an existing live row, returns a noop UndoRecord,
    and never re-issues the LLM call.

    Cost: roughly ~$15-25 per dream cycle on a 100k-entity namespace at
    ``gpt-4o-mini`` rates. The orchestrator caps the upper bound by
    checking :attr:`khora.dream.DreamConfig.llm_max_tokens_per_run` and
    :attr:`khora.dream.DreamConfig.llm_max_tokens_per_namespace_per_day`
    before each LLM-using op and skipping the remainder once a budget is
    exhausted (#1270); this handler does not enforce the budget itself.
    """
    del coordinator  # apply writes via session

    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect and dialect != "postgresql":
        raise DreamForbiddenOpError("community_summary apply is Postgres-only in v0.15; embedded path lands later.")

    inputs = op.inputs[0] if op.inputs else {}
    community_id = UUID(str(inputs["community_id"]))
    member_ids = list(inputs.get("member_ids") or [])
    member_names = list(inputs.get("member_names") or [])
    member_types = list(inputs.get("member_types") or [])
    rel_modes = dict(inputs.get("relationship_modes") or {})
    namespace_id = op.namespace_id

    with trace_span(
        "khora.dream.vectorcypher.community_summary",
        op_id=str(op.op_id),
        namespace_id=str(namespace_id) if namespace_id else "",
        phase=_PHASE_APPLY,
        community_size=len(member_ids),
        community_id_hash=bounded_text_hash(str(community_id)),
    ) as span:
        # Idempotent-replay short-circuit: if a live row already exists
        # for this community_id, return a noop without calling the LLM.
        existing = await _select_live_community(session, community_id)
        if existing is not None:
            span.set_attribute("decision", "noop_replay")
            return UndoRecord(
                op_id=op.op_id,
                op_type=str(op.op_type),
                before={"noop": True, "reason": "already_live"},
                applied_at=datetime.now(UTC),
            )

        # Build prompt — always uses the planned member ids as the
        # grounding set. We never read from prior summary nodes.
        prompt = _build_prompt(
            member_ids=member_ids,
            member_names=member_names,
            member_types=member_types,
            rel_modes=rel_modes,
        )
        span.set_attribute("prompt_hash", bounded_text_hash(prompt))

        config = LiteLLMConfig(model=model)
        raw = await acompletion(
            prompt,
            config,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            _telemetry_op="dream_community_summary",
        )

        summary = _parse_summary(raw)
        member_id_set = set(member_ids)
        kept, dropped = validate_grounding(summary, member_id_set)

        span.set_attribute("kept_claims", len(kept))
        span.set_attribute("dropped_claims", len(dropped))

        if not kept:
            span.set_attribute("decision", "no_grounded_claims")
            return UndoRecord(
                op_id=op.op_id,
                op_type=str(op.op_type),
                before={
                    "noop": True,
                    "reason": "no_grounded_claims",
                    "dropped_claims": len(dropped),
                },
                applied_at=datetime.now(UTC),
            )

        now = datetime.now(UTC)
        payload = {
            "text": summary.text,
            "claims": [{"text": c.text, "cited_entity_ids": list(c.cited_entity_ids)} for c in kept],
            "summary_depth": 1,
            "model": model,
        }

        await session.execute(
            text(
                "INSERT INTO khora_dream_communities "
                "(id, namespace_id, op_id, member_ids, payload, summary_depth, "
                " valid_from, valid_to, created_at) "
                "VALUES (:cid, :ns, :oid, :mids, :payload, :depth, :vf, NULL, :now)"
            ),
            {
                "cid": community_id,
                "ns": namespace_id,
                "oid": op.op_id,
                "mids": list(member_ids),
                "payload": json.dumps(payload),
                "depth": 1,
                "vf": now,
                "now": now,
            },
        )
        span.set_attribute("decision", "persisted")

        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={
                "community_id": str(community_id),
                "kept_claims": len(kept),
                "dropped_claims": len(dropped),
            },
            applied_at=now,
        )


async def _select_live_community(session: AsyncSession, community_id: UUID) -> Any:
    """Return the live row (``valid_to IS NULL``) for ``community_id`` or ``None``."""
    result = await session.execute(
        text("SELECT id, namespace_id, valid_to FROM khora_dream_communities WHERE id = :cid AND valid_to IS NULL"),
        {"cid": community_id},
    )
    return result.first()


def _build_prompt(
    *,
    member_ids: list[str],
    member_names: list[str],
    member_types: list[str],
    rel_modes: dict[str, int],
) -> str:
    """Render the per-community LLM prompt.

    Always re-summarises from raw members + relationship modes — never
    from prior summary text — so the apply handler stays consistent with
    the no-summary-of-summaries grounding rule.
    """
    member_lines = []
    for mid, name, etype in zip(member_ids, member_names, member_types, strict=False):
        member_lines.append(f"  - {mid}: {name} ({etype})")

    rel_line = ", ".join(f"{k}={v}" for k, v in sorted(rel_modes.items(), key=lambda kv: -kv[1])) or "(none)"

    return (
        "Summarise the following community of entities in 1-2 sentences. "
        "Every claim MUST cite at least one member entity_id. Do NOT "
        "invent entity_ids.\n\n"
        f"Members ({len(member_ids)} shown):\n"
        + "\n".join(member_lines)
        + f"\n\nRelationship modes within community: {rel_line}\n\n"
        "Return JSON only — no markdown, no commentary."
    )


def _parse_summary(raw: str) -> GroundedSummary:
    """Parse the LLM's JSON response into :class:`GroundedSummary`.

    Tolerant of leading/trailing whitespace and markdown code-fences.
    Malformed payloads degrade to an empty :class:`GroundedSummary` so
    the grounding validator drops everything (the apply handler will
    return a ``no_grounded_claims`` noop).
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fence.
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return GroundedSummary(text="", claims=[])

    if not isinstance(data, dict):
        return GroundedSummary(text="", claims=[])

    text_field = str(data.get("text") or "")
    raw_claims = data.get("claims") or []
    claims: list[SummaryClaim] = []
    if isinstance(raw_claims, list):
        for entry in raw_claims:
            if not isinstance(entry, dict):
                continue
            claim_text = str(entry.get("text") or "")
            cited = entry.get("cited_entity_ids") or []
            if not isinstance(cited, list):
                cited = []
            claims.append(SummaryClaim(text=claim_text, cited_entity_ids=[str(c) for c in cited]))
    return GroundedSummary(text=text_field, claims=claims)


__all__ = [
    "GroundedSummary",
    "SummaryClaim",
    "apply_vectorcypher_community_summary",
    "plan_vectorcypher_community_summary",
    "validate_grounding",
]
