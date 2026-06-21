"""Vectorcypher contradiction reconciliation — two-LLM-judged, opt-in (#1281).

Phase 5 (final) of the dream-on-graph umbrella (#1282). This op promotes the
report-only ``contradiction_detect`` op (#672) to a mutating reconcile op gated
by a two-LLM judge:

  * The planner reuses the detector's pairwise logic to find candidate
    contradiction pairs (same ``(source, target, type)`` bucket, low textual
    similarity OR contradicting property keys), enriching each finding with the
    two relationships' confidence + grounding source ids so the apply-side judge
    can decide and cite.
  * At APPLY time (where the dream LLM token budget lives, #1270) each finding is
    routed through ``run_contradiction_judge`` — two independent LLM calls. Both
    judges must agree on ``invalidate`` (same loser) with confidence above the
    floor AND cite at least one supporting chunk/doc id from the candidates'
    source set (grounded-citation validation). Disagreement, timeout, low
    confidence, or an ungrounded verdict degrades to ``defer`` — NO mutation.
  * Only judge-AGREED invalidations mutate: the **losing edge** (lower
    confidence; ties broken by canonical id ordering) is **soft-deleted**
    (``relationships.valid_to = NOW()``), never hard-deleted. The soft-delete is
    mirrored to the graph through the existing #1271 verbs / #1272 mirror path
    (the op kind is in ``MIRRORABLE_OP_KINDS`` and the ``UndoRecord.before``
    carries a ``relationships`` list shaped exactly like ``prune_edges``).
  * Every outcome (invalidate / defer / keep) writes a **triage row**: the apply
    handler UPSERTs into ``dream_conflicts`` (migration 048 added the reconcile
    columns) so a pair already detected report-only is upgraded in place to its
    resolution.

Budget: the op kind is in the orchestrator's ``_LLM_OP_KINDS`` set, so the
two-LLM judge is refused once a per-run / per-namespace-per-day token budget is
exhausted — the orchestrator records a structured ``llm_budget_exhausted``
skip_reason and the op does NOT advance its checkpoint (a later run can retry).

ADR-001: defer / timeout / budget-skip outcomes each record a structured
``SkipReason`` / ``Degradation`` so a non-mutating outcome is never silent.

Stability: **internal**.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text

from khora.dream.engines.vectorcypher.contradiction_detect import (
    _find_contradictions,
    _is_live,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import bounded_text_hash, trace_span
from khora.telemetry.metrics import metric_counter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.core.models.entity import Relationship
    from khora.dream.config import DreamConfig
    from khora.storage.coordinator import StorageCoordinator

_PHASE_PLAN = "audit"
_PHASE_APPLY = "mutation"


# Emitted once per flagged pair the two-LLM judge declined to invalidate
# (defer / keep / ungrounded). No labels - the namespace_id cardinality rule.
RECONCILE_DEFERRED_COUNTER = metric_counter(
    "khora.dream.contradiction.reconcile_deferred_total",
    description=(
        "Contradiction pairs the two-LLM reconcile judge did NOT invalidate "
        "(disagreement, timeout, below-confidence, ungrounded citation, or a "
        "judged keep). No mutation occurred; a triage row was written. NO "
        "namespace_id label - cardinality rule."
    ),
)


# ---------------------------------------------------------------------------
# Judge schema + dispatcher
# ---------------------------------------------------------------------------


class ReconcileVerdict(BaseModel):
    """Structured Pydantic output for a single LLM judge call.

    The verifier and the auditor each return one ``ReconcileVerdict``;
    :func:`run_contradiction_judge` combines both before returning the joint
    :class:`JudgeOutcome`.
    """

    decision: str = Field(
        description="One of invalidate / keep / defer.",
    )
    loser: str = Field(
        default="",
        description="Which side to invalidate when decision=invalidate: 'a' or 'b'.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in the decision.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Supporting chunk / document UUIDs the judge cites.",
    )
    rationale: str = Field(
        default="",
        description="Short free-text rationale (hashed before becoming a span attribute).",
    )


class JudgeOutcome:
    """Joint verdict from the two-LLM contradiction judge.

    ``decision`` is the dispatcher's combined ruling:
      * ``"invalidate"`` only when both judges return ``invalidate`` on the
        SAME loser, both above the confidence floor, AND at least one cited
        evidence id is grounded in the candidate pair's source set.
      * ``"keep"`` only when both judges return ``keep``.
      * ``"defer"`` for any disagreement, below-floor confidence, ungrounded
        citation, or any parse / network / timeout failure.
    """

    __slots__ = ("auditor_verdict", "decision", "loser", "rationale", "verifier_verdict")

    def __init__(
        self,
        *,
        decision: str,
        loser: str | None,
        verifier_verdict: ReconcileVerdict | None,
        auditor_verdict: ReconcileVerdict | None,
        rationale: str,
    ) -> None:
        self.decision = decision
        self.loser = loser
        self.verifier_verdict = verifier_verdict
        self.auditor_verdict = auditor_verdict
        self.rationale = rationale


_SYSTEM_PROMPT = (
    "You are a knowledge-graph contradiction judge. Two relationship records "
    "connect the SAME pair of entities with the SAME type but appear to "
    "disagree. Decide whether one record genuinely contradicts (and should be "
    "retired) the other (invalidate), whether they actually coexist and both "
    "should stay (keep), or whether you cannot tell (defer). When you choose "
    "invalidate you MUST name the losing side ('a' or 'b') and you MUST cite "
    "at least one supporting source id (a chunk or document UUID) drawn ONLY "
    "from the source ids listed for the records - never invent ids. Respond "
    "with a single JSON object matching the schema "
    '{"decision": "invalidate|keep|defer", "loser": "a|b", "confidence": '
    'float in [0,1], "evidence_ids": [string], "rationale": string}. '
    "Output JSON only - no prose, no markdown fences."
)


def _build_user_prompt(finding: dict[str, Any]) -> str:
    a_chunks = ", ".join(finding.get("a_source_chunk_ids") or []) or "(none)"
    a_docs = ", ".join(finding.get("a_source_document_ids") or []) or "(none)"
    b_chunks = ", ".join(finding.get("b_source_chunk_ids") or []) or "(none)"
    b_docs = ", ".join(finding.get("b_source_document_ids") or []) or "(none)"
    return (
        f"Relationship type: {finding['relationship_type']}\n"
        f"Pairwise textual similarity: {float(finding['similarity']):.4f}\n"
        f"Contradicting property keys: {', '.join(finding.get('contradicting_keys') or []) or '(none)'}\n"
        f"\n"
        f"Record A (id={finding['relationship_a_id']}, confidence={finding.get('a_confidence')}):\n"
        f"  description: {finding.get('a_description') or '(empty)'}\n"
        f"  source chunk ids: {a_chunks}\n"
        f"  source document ids: {a_docs}\n"
        f"\n"
        f"Record B (id={finding['relationship_b_id']}, confidence={finding.get('b_confidence')}):\n"
        f"  description: {finding.get('b_description') or '(empty)'}\n"
        f"  source chunk ids: {b_chunks}\n"
        f"  source document ids: {b_docs}\n"
        f"\n"
        f"Return the JSON verdict now."
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_verdict(raw: str) -> ReconcileVerdict | None:
    """Parse an LLM response into a :class:`ReconcileVerdict` or ``None``.

    Defensive: strips whitespace, peels a fenced ```json``` block, then runs
    Pydantic validation. Returns ``None`` on any failure; the dispatcher treats
    a ``None`` as ``decision="defer"``.
    """
    if not raw:
        return None
    payload_text = raw.strip()
    match = _FENCE_RE.match(payload_text)
    if match is not None:
        payload_text = match.group(1).strip()
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return ReconcileVerdict.model_validate(payload)
    except ValidationError:
        return None


async def _ask_one_judge(
    finding: dict[str, Any],
    *,
    model: str,
    direction: str,
    timeout_s: float,
) -> ReconcileVerdict | None:
    """Call one judge model and parse its verdict.

    ``direction`` is ``"verifier"`` or ``"auditor"`` - a bounded-cardinality
    span attribute (never a metric label). Any timeout / transport / parse
    error returns ``None`` so the dispatcher degrades to defer.
    """
    from khora.config.llm import LiteLLMConfig, acompletion

    config = LiteLLMConfig(
        model=model,
        temperature=0.0,
        max_tokens=400,
        timeout=int(max(1.0, timeout_s)),
    )
    user_prompt = _build_user_prompt(finding)

    with trace_span(
        "khora.dream.vectorcypher.contradiction.judge",
        direction=direction,
        model=model,
        similarity=str(finding.get("similarity")),
        prompt_hash=bounded_text_hash(user_prompt),
    ) as span:
        try:
            raw = await asyncio.wait_for(
                acompletion(
                    user_prompt,
                    config,
                    system_prompt=_SYSTEM_PROMPT,
                    _telemetry_op=f"dream.contradiction.{direction}",
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            span.set_attribute("outcome", "timeout")
            span.set_attribute("error_type", type(exc).__name__)
            logger.warning(
                "contradiction reconcile {direction} judge ({model}) timed out; degrading to defer",
                direction=direction,
                model=model,
                exc_info=True,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - never crash the dream run on a flaky API
            span.set_attribute("outcome", "error")
            span.set_attribute("error_type", type(exc).__name__)
            logger.warning(
                "contradiction reconcile {direction} judge ({model}) raised; degrading to defer",
                direction=direction,
                model=model,
                exc_info=True,
            )
            return None

        verdict = _parse_verdict(raw)
        if verdict is None:
            span.set_attribute("outcome", "parse_error")
            return None
        span.set_attribute("outcome", "parsed")
        span.set_attribute("decision", verdict.decision)
        span.set_attribute("confidence", str(verdict.confidence))
        return verdict


def _grounded(verdict: ReconcileVerdict, allowed_ids: set[str]) -> bool:
    """Return True when the verdict cites at least one allowed source id."""
    return any(eid in allowed_ids for eid in verdict.evidence_ids)


def _combine_verdicts(
    verifier: ReconcileVerdict | None,
    auditor: ReconcileVerdict | None,
    *,
    min_confidence: float,
    allowed_ids: set[str],
) -> tuple[str, str | None, str]:
    """Dispatcher rule: agree to invalidate (grounded) or defer."""
    if verifier is None or auditor is None:
        return "defer", None, "one or both judges returned no parseable verdict"
    if verifier.decision == "invalidate" and auditor.decision == "invalidate":
        if verifier.loser != auditor.loser or verifier.loser not in ("a", "b"):
            return "defer", None, f"judges disagree on loser (verifier={verifier.loser!r}, auditor={auditor.loser!r})"
        if not (verifier.confidence >= min_confidence and auditor.confidence >= min_confidence):
            return "defer", None, "both judges said invalidate but confidence below floor"
        if not (_grounded(verifier, allowed_ids) and _grounded(auditor, allowed_ids)):
            return "defer", None, "judge verdict not grounded in the candidate pair's source ids"
        return "invalidate", verifier.loser, "both judges agreed: invalidate (grounded)"
    if verifier.decision == "keep" and auditor.decision == "keep":
        return "keep", None, "both judges agreed: keep"
    return "defer", None, f"judges disagree (verifier={verifier.decision}, auditor={auditor.decision})"


async def run_contradiction_judge(
    finding: dict[str, Any],
    *,
    config: DreamConfig | None = None,
) -> JudgeOutcome:
    """Run the two-LLM judge for one flagged contradiction pair.

    Both judges are queried concurrently. The dispatcher combines their
    verdicts per :func:`_combine_verdicts` - invalidate only on a grounded,
    above-floor, same-loser agreement; defer on anything else.

    Args:
        finding: An enriched finding dict (carries both relationships'
            confidence, description, and source ids).
        config: :class:`DreamConfig` providing model names / timeout /
            confidence floor. ``None`` falls back to ``DreamConfig()``.

    Returns:
        :class:`JudgeOutcome` with the dispatcher decision + the chosen loser.
    """
    from khora.dream.config import DreamConfig as _DreamConfig

    cfg = config if config is not None else _DreamConfig()
    timeout_s = float(cfg.contradiction_reconcile_timeout_seconds)

    allowed_ids = {
        *(finding.get("a_source_chunk_ids") or []),
        *(finding.get("a_source_document_ids") or []),
        *(finding.get("b_source_chunk_ids") or []),
        *(finding.get("b_source_document_ids") or []),
    }

    verifier_task = _ask_one_judge(
        finding,
        model=cfg.contradiction_reconcile_model,
        direction="verifier",
        timeout_s=timeout_s,
    )
    auditor_task = _ask_one_judge(
        finding,
        model=cfg.contradiction_reconcile_auditor_model,
        direction="auditor",
        timeout_s=timeout_s,
    )
    verifier_verdict, auditor_verdict = await asyncio.gather(verifier_task, auditor_task)

    decision, loser, rationale = _combine_verdicts(
        verifier_verdict,
        auditor_verdict,
        min_confidence=float(cfg.contradiction_reconcile_min_confidence),
        allowed_ids=allowed_ids,
    )
    return JudgeOutcome(
        decision=decision,
        loser=loser,
        verifier_verdict=verifier_verdict,
        auditor_verdict=auditor_verdict,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Planner — pure detection, no LLM (the judge runs at apply time, budget-gated)
# ---------------------------------------------------------------------------


async def plan_vectorcypher_contradiction_reconcile(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    similarity_threshold: float = 0.5,
) -> DreamOp:
    """Detect candidate contradiction pairs for two-LLM reconciliation.

    Pure planner: zero writes, zero LLM calls (the budget-gated judge runs in
    the apply handler). Reuses the report-only detector's pairwise logic, then
    enriches each finding with the two relationships' ``confidence``,
    ``description`` and source ids so the apply-side judge can decide + cite and
    the handler can pick the loser deterministically.
    """
    op_id = uuid4()
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.contradiction_reconcile",
        op_id=str(op_id),
        namespace_id=str(namespace_id),
        phase=_PHASE_PLAN,
        similarity_threshold=float(similarity_threshold),
    ) as span:
        relationships = await _list_relationships_for_detection(coordinator, namespace_id)
        by_id: dict[str, Relationship] = {str(rel.id): rel for rel in relationships}
        live = [rel for rel in relationships if _is_live(rel)]
        span.set_attribute("total_relationships", len(relationships))
        span.set_attribute("live_relationships", len(live))

        buckets: dict[tuple[UUID, UUID, str], list[Relationship]] = defaultdict(list)
        for rel in live:
            buckets[(rel.source_entity_id, rel.target_entity_id, rel.relationship_type)].append(rel)

        findings: list[dict[str, Any]] = []
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for finding in _find_contradictions(bucket, similarity_threshold):
                findings.append(_enrich_finding(finding, by_id))

        span.set_attribute("finding_count", len(findings))
        duration_ms = (time.perf_counter() - t0) * 1000.0
        span.set_attribute("duration_ms", duration_ms)

    return DreamOp(
        op_id=op_id,
        phase=_PHASE_PLAN,
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE,
        inputs=(
            {
                "namespace_id": str(namespace_id),
                "similarity_threshold": float(similarity_threshold),
                "total_relationships": len(relationships),
                "live_relationships": len(live),
            },
        ),
        outputs=tuple(findings),
        decision="planned",
        rationale=(
            f"Flagged {len(findings)} candidate contradiction pair(s) for two-LLM "
            f"reconciliation across {len(live)} live relationships."
        ),
        started_at=started_at,
        duration_ms=duration_ms,
        namespace_id=namespace_id,
    )


async def _list_relationships_for_detection(coordinator: Any, namespace_id: UUID) -> list[Relationship]:
    """Read relationships from the row-level (PG) source of truth for detection.

    A contradiction is two *distinct* ``relationships`` rows in the same
    ``(source, target, type)`` bucket. The graph backend MERGEs edges on
    endpoints+type, so ``coordinator.list_relationships`` (graph-preferring)
    collapses a contradicting pair into one edge and the detector would never
    see both. The vector backend (``PgvectorBackend.list_relationships``) serves
    the table rows directly with per-row id / confidence / source ids, so prefer
    it. Falls back to the coordinator view when no row-level backend is wired
    (e.g. a graph-only embedded stub).
    """
    vector = getattr(coordinator, "_vector", None)
    if vector is not None and hasattr(vector, "list_relationships"):
        # PG ``relationships`` rows carry the row-level namespace id (ingest
        # resolves the stable id before any write), so resolve before reading.
        row_ns = namespace_id
        resolver = getattr(coordinator, "resolve_namespace", None)
        if resolver is not None:
            try:
                row_ns = await resolver(namespace_id)
            except Exception:  # noqa: BLE001 - resolve is a cheap read; fall back to input
                logger.warning(
                    "namespace resolve failed during contradiction reconcile planning; "
                    "falling back to the input namespace_id",
                    exc_info=True,
                )
                row_ns = namespace_id
        return await vector.list_relationships(row_ns, limit=1_000_000)
    return await coordinator.list_relationships(namespace_id, limit=1_000_000)


def _enrich_finding(finding: dict[str, Any], by_id: dict[str, Relationship]) -> dict[str, Any]:
    """Attach confidence / description / source ids for both sides of a pair."""
    rel_a = by_id.get(finding["relationship_a_id"])
    rel_b = by_id.get(finding["relationship_b_id"])
    enriched = dict(finding)
    enriched["a_confidence"] = float(getattr(rel_a, "confidence", 1.0)) if rel_a else None
    enriched["b_confidence"] = float(getattr(rel_b, "confidence", 1.0)) if rel_b else None
    enriched["a_description"] = (getattr(rel_a, "description", "") or "") if rel_a else ""
    enriched["b_description"] = (getattr(rel_b, "description", "") or "") if rel_b else ""
    enriched["a_source_chunk_ids"] = _id_strs(getattr(rel_a, "source_chunk_ids", None))
    enriched["b_source_chunk_ids"] = _id_strs(getattr(rel_b, "source_chunk_ids", None))
    enriched["a_source_document_ids"] = _id_strs(getattr(rel_a, "source_document_ids", None))
    enriched["b_source_document_ids"] = _id_strs(getattr(rel_b, "source_document_ids", None))
    return enriched


def _id_strs(value: Any) -> list[str]:
    if not value:
        return []
    return [str(v) for v in value]


# ---------------------------------------------------------------------------
# Apply handler — two-LLM judge → soft-delete loser (mirrored) + triage row
# ---------------------------------------------------------------------------


_SOFT_DELETE_SQL = text(
    "UPDATE relationships SET valid_to = :ts, updated_at = :ts WHERE id = :rid AND valid_to IS NULL"
)


_UPSERT_TRIAGE_SQL = text(
    """
    INSERT INTO dream_conflicts (
        id,
        namespace_id,
        relationship_a_id,
        relationship_b_id,
        source_entity_id,
        target_entity_id,
        relationship_type,
        similarity,
        contradicting_keys,
        reason,
        description_a_hash,
        description_b_hash,
        detected_by_op_id,
        valid_from,
        resolution,
        loser_relationship_id,
        winner_relationship_id,
        judge_rationale_hash,
        resolved_by_op_id,
        resolved_at
    ) VALUES (
        :id,
        :namespace_id,
        :relationship_a_id,
        :relationship_b_id,
        :source_entity_id,
        :target_entity_id,
        :relationship_type,
        :similarity,
        :contradicting_keys,
        :reason,
        :description_a_hash,
        :description_b_hash,
        :detected_by_op_id,
        :valid_from,
        :resolution,
        :loser_relationship_id,
        :winner_relationship_id,
        :judge_rationale_hash,
        :resolved_by_op_id,
        :resolved_at
    )
    ON CONFLICT (namespace_id, relationship_a_id, relationship_b_id) DO UPDATE SET
        resolution = EXCLUDED.resolution,
        loser_relationship_id = EXCLUDED.loser_relationship_id,
        winner_relationship_id = EXCLUDED.winner_relationship_id,
        judge_rationale_hash = EXCLUDED.judge_rationale_hash,
        resolved_by_op_id = EXCLUDED.resolved_by_op_id,
        resolved_at = EXCLUDED.resolved_at
    """
)


async def apply_vectorcypher_contradiction_reconcile(
    op: DreamOp,
    *,
    coordinator: Any = None,
    session: AsyncSession,
    dream_config: DreamConfig | None = None,
) -> UndoRecord:
    """Judge each flagged pair; soft-delete the loser only on agreement.

    For every finding the two-LLM judge runs (concurrently per side). On a
    grounded, above-floor, same-loser ``invalidate`` agreement the losing edge
    (lower confidence; ties broken by canonical id ordering) is soft-deleted
    (``relationships.valid_to = NOW()``) and recorded in
    ``UndoRecord.before["relationships"]`` so the orchestrator's #1272 mirror
    folds it onto the graph ``valid_until``. defer / keep outcomes mutate
    nothing. EVERY outcome UPSERTs a triage row into ``dream_conflicts``.

    ADR-001: defer / keep / ungrounded outcomes append a structured
    ``SkipReason`` to ``UndoRecord.before["skip_reasons"]``; budget exhaustion is
    enforced upstream by the orchestrator before this handler is dispatched.

    Args:
        op: The planned reconcile op; ``outputs`` holds the enriched findings.
        coordinator: Unused - the session is the only write surface.
        session: Orchestrator-owned async session.
        dream_config: Forwarded by the orchestrator's ``_invoke_handler`` (it
            declares this kwarg) so the judge reads the configured models /
            timeout / confidence floor.

    Returns:
        :class:`UndoRecord`. ``before["relationships"]`` lists soft-deleted
        edges (mirror + undo source of truth); ``before["triage"]`` lists the
        upserted triage row keys; ``before["skip_reasons"]`` lists non-mutating
        outcomes per ADR-001.
    """
    del coordinator  # session is the only write surface

    findings = list(op.outputs)
    applied_at = datetime.now(UTC)

    namespace_id = _coerce_uuid(op.namespace_id) or _coerce_uuid(
        (op.inputs[0] if op.inputs else {}).get("namespace_id")
    )

    invalidated: list[dict[str, Any]] = []
    triage_entries: list[dict[str, Any]] = []
    skip_reasons: list[dict[str, Any]] = []

    for finding in findings:
        outcome = await run_contradiction_judge(finding, config=dream_config)

        loser_id, winner_id = _resolve_loser_winner(finding, outcome)
        resolution = _resolution_label(outcome.decision)
        rationale_hash = bounded_text_hash(_rationale_material(outcome))

        if outcome.decision == "invalidate" and loser_id is not None:
            result = await session.execute(_SOFT_DELETE_SQL, {"ts": applied_at, "rid": loser_id})
            if getattr(result, "rowcount", 0):
                invalidated.append({"relationship_id": str(loser_id)})
        else:
            # defer / keep — no mutation. Record per ADR-001.
            RECONCILE_DEFERRED_COUNTER.add(1)
            skip_reasons.append(
                {
                    "op_kind": str(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE),
                    "reason": f"reconcile_{outcome.decision}",
                    "detail": outcome.rationale,
                    "relationship_a_id": finding["relationship_a_id"],
                    "relationship_b_id": finding["relationship_b_id"],
                }
            )

        await session.execute(
            _UPSERT_TRIAGE_SQL,
            {
                "id": uuid4(),
                "namespace_id": namespace_id,
                "relationship_a_id": _coerce_uuid(finding["relationship_a_id"]),
                "relationship_b_id": _coerce_uuid(finding["relationship_b_id"]),
                "source_entity_id": _coerce_uuid(finding["source_entity_id"]),
                "target_entity_id": _coerce_uuid(finding["target_entity_id"]),
                "relationship_type": finding["relationship_type"],
                "similarity": float(finding["similarity"]),
                "contradicting_keys": list(finding.get("contradicting_keys") or []),
                "reason": finding["reason"],
                "description_a_hash": finding["description_a_hash"],
                "description_b_hash": finding["description_b_hash"],
                "detected_by_op_id": op.op_id,
                "valid_from": applied_at,
                "resolution": resolution,
                "loser_relationship_id": loser_id if resolution == "invalidated" else None,
                "winner_relationship_id": winner_id if resolution == "invalidated" else None,
                "judge_rationale_hash": rationale_hash,
                "resolved_by_op_id": op.op_id,
                "resolved_at": applied_at,
            },
        )
        triage_entries.append(
            {
                "relationship_a_id": finding["relationship_a_id"],
                "relationship_b_id": finding["relationship_b_id"],
                "resolution": resolution,
            }
        )

    before: dict[str, Any] = {
        "relationships": invalidated,
        "triage": triage_entries,
    }
    if skip_reasons:
        before["skip_reasons"] = skip_reasons
    return UndoRecord(
        op_id=op.op_id,
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value,
        before=before,
        applied_at=applied_at,
    )


def _resolve_loser_winner(finding: dict[str, Any], outcome: JudgeOutcome) -> tuple[UUID | None, UUID | None]:
    """Map the judge's 'a'/'b' loser onto concrete relationship ids.

    Only meaningful when ``outcome.decision == "invalidate"``; returns
    ``(None, None)`` otherwise. The judge already agreed on the loser side; we
    trust that mapping (the planner ordered a/b canonically). When the judge
    omits a side (defensive), fall back to the lower-confidence edge.
    """
    if outcome.decision != "invalidate":
        return None, None
    a_id = _coerce_uuid(finding["relationship_a_id"])
    b_id = _coerce_uuid(finding["relationship_b_id"])
    if outcome.loser == "a":
        return a_id, b_id
    if outcome.loser == "b":
        return b_id, a_id
    # Defensive fallback: lower confidence loses (ties: 'b' by canonical order).
    a_conf = finding.get("a_confidence")
    b_conf = finding.get("b_confidence")
    if a_conf is not None and b_conf is not None and a_conf < b_conf:
        return a_id, b_id
    return b_id, a_id


def _resolution_label(decision: str) -> str:
    if decision == "invalidate":
        return "invalidated"
    if decision == "keep":
        return "kept"
    return "deferred"


def _rationale_material(outcome: JudgeOutcome) -> str:
    """Text hashed into ``judge_rationale_hash`` for triage audit value.

    Prefer the two judges' own rationales (distinct per pair) over the
    dispatcher's generic summary string, so the persisted hash distinguishes
    materially different verdicts. Falls back to the dispatcher summary when
    neither judge returned a parseable rationale (e.g. both timed out).
    """
    parts = [
        outcome.verifier_verdict.rationale if outcome.verifier_verdict else "",
        outcome.auditor_verdict.rationale if outcome.auditor_verdict else "",
    ]
    material = "\n".join(p for p in parts if p)
    return material or (outcome.rationale or "")


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "JudgeOutcome",
    "ReconcileVerdict",
    "apply_vectorcypher_contradiction_reconcile",
    "plan_vectorcypher_contradiction_reconcile",
    "run_contradiction_judge",
]
