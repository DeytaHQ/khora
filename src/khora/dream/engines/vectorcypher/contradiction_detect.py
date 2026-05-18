"""Vectorcypher contradiction detection — report-only (#672, Phase 5.3).

For each ``(source_entity_id, target_entity_id, relationship_type)``
bucket containing two or more **live** relationships (``valid_to IS NULL``
in DB terms, ``valid_until is None`` on the in-memory dataclass), compute
pairwise textual similarity over ``description`` and detect shared
property keys whose stringified values disagree. Pairs whose similarity
falls below a configurable threshold, OR whose properties contradict,
are flagged.

**No auto-resolution.** The apply path persists each finding to the
``dream_conflicts`` table (migration 035) so a human triage queue can
review them; the underlying ``relationships`` rows are never touched.
Phase 5.4 (#673) consumes the same findings as the natural source of
mapping recommendations.

Idempotency: the persist statement uses
``ON CONFLICT (namespace_id, relationship_a_id, relationship_b_id) DO
NOTHING`` against a canonical ordering of the two relationship ids, so
replays of the same op never duplicate findings.

Stability: **internal**. The OpKind constant is a stable string id; the
finding payload shape may evolve until Phase 5 closes.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import bounded_text_hash, trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.core.models.entity import Relationship
    from khora.storage.coordinator import StorageCoordinator

_PHASE_PLAN = "audit"


async def plan_vectorcypher_contradiction_detect(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    similarity_threshold: float = 0.5,
) -> DreamOp:
    """Detect contradictions among live relationships — report only.

    Reads ``coordinator.list_relationships``, buckets by
    ``(source_entity_id, target_entity_id, relationship_type)``, and for
    every bucket with more than one **live** relationship computes
    pairwise textual similarity on ``description`` and checks for
    contradicting property values. Pairs below ``similarity_threshold``
    OR with at least one contradicting shared property key are flagged.

    The planner is pure: zero writes, zero LLM calls. Apply-side
    persistence lives in :func:`apply_vectorcypher_contradiction_detect`.

    Args:
        namespace_id: Namespace to audit.
        coordinator: Storage coordinator (DI for tests).
        similarity_threshold: Pairs with textual similarity strictly
            below this threshold are flagged as low-similarity. Default
            ``0.5``. Property contradictions are flagged independently.

    Returns:
        :class:`DreamOp` with ``op_type=VECTORCYPHER_CONTRADICTION_DETECT``.
        ``outputs`` is a tuple of finding dicts, one per flagged pair;
        each finding carries ``relationship_a_id``, ``relationship_b_id``,
        ``source_entity_id``, ``target_entity_id``, ``relationship_type``,
        ``similarity``, ``contradicting_keys``, ``reason`` (one of
        ``"low_similarity"`` / ``"property_contradiction"`` / ``"both"``),
        and hashed ``description_a_hash`` / ``description_b_hash`` for
        triage cross-reference without leaking raw text.
    """
    op_id = uuid4()
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.contradiction_detect",
        run_id="",
        op_id=str(op_id),
        namespace_id=str(namespace_id),
        phase=_PHASE_PLAN,
        similarity_threshold=float(similarity_threshold),
    ) as span:
        relationships = await coordinator.list_relationships(namespace_id, limit=1_000_000)

        # Only live relationships participate. The core dataclass exposes
        # bi-temporal state via ``valid_until`` (the DB column the
        # planner aliases for; migration-033's ``valid_to`` is also a
        # soft-delete signal — when either is set the row is closed).
        live = [rel for rel in relationships if _is_live(rel)]
        span.set_attribute("total_relationships", len(relationships))
        span.set_attribute("live_relationships", len(live))

        # Bucket by (source, target, type). Only buckets with >= 2 live
        # rows can produce a contradiction pair.
        buckets: dict[tuple[UUID, UUID, str], list[Relationship]] = defaultdict(list)
        for rel in live:
            buckets[(rel.source_entity_id, rel.target_entity_id, rel.relationship_type)].append(rel)

        findings: list[dict[str, Any]] = []
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            findings.extend(_find_contradictions(bucket, similarity_threshold))

        # Per-finding hashed attributes — never raw text. The free-text
        # cardinality rule from the telemetry contract applies here.
        for idx, finding in enumerate(findings):
            span.set_attribute(f"finding_{idx}_desc_a_hash", finding["description_a_hash"])
            span.set_attribute(f"finding_{idx}_desc_b_hash", finding["description_b_hash"])
            span.set_attribute(f"finding_{idx}_reason", finding["reason"])
            span.set_attribute(f"finding_{idx}_similarity", float(finding["similarity"]))

        span.set_attribute("finding_count", len(findings))
        duration_ms = (time.perf_counter() - t0) * 1000.0
        span.set_attribute("duration_ms", duration_ms)

    return DreamOp(
        op_id=op_id,
        phase=_PHASE_PLAN,
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT,
        inputs=(
            {
                "namespace_id": str(namespace_id),
                "similarity_threshold": float(similarity_threshold),
                "total_relationships": len(relationships),
                "live_relationships": len(live),
            },
        ),
        outputs=tuple(findings),
        decision="audit_complete",
        rationale=(
            f"Flagged {len(findings)} contradiction pair(s) across "
            f"{len(live)} live relationships in {len(buckets)} (src,tgt,type) "
            f"bucket(s)."
        ),
        started_at=started_at,
        duration_ms=duration_ms,
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_live(rel: Relationship) -> bool:
    """A relationship is live when both bi-temporal end-markers are NULL.

    The core dataclass exposes ``valid_until`` (the legacy column) and
    Phase 0.3's bi-temporal soft-delete added ``valid_to`` on the SQL
    side; the core dataclass mirrors ``valid_to`` via the same
    ``valid_until`` slot for now. When either is non-NULL the row has
    been closed and is excluded from contradiction detection.
    """
    return getattr(rel, "valid_until", None) is None and getattr(rel, "valid_to", None) is None


def _find_contradictions(
    bucket: list[Relationship],
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Return one finding per flagged pair within a single bucket.

    A pair is flagged when:

    - Textual similarity of ``description`` (SequenceMatcher ratio over
      the concatenation of description + sorted property values) is
      strictly less than ``similarity_threshold``, OR
    - The two relationships share at least one ``properties`` key whose
      stringified values differ.

    A pair flagged by both rules carries ``reason="both"``.
    """
    findings: list[dict[str, Any]] = []
    for i in range(len(bucket)):
        for j in range(i + 1, len(bucket)):
            a, b = bucket[i], bucket[j]
            similarity = _text_similarity(a, b)
            contradicting_keys = _contradicting_property_keys(a, b)

            low_sim = similarity < similarity_threshold
            has_contra = bool(contradicting_keys)
            if not (low_sim or has_contra):
                continue

            if low_sim and has_contra:
                reason = "both"
            elif low_sim:
                reason = "low_similarity"
            else:
                reason = "property_contradiction"

            # Canonical ordering so the persisted finding is stable
            # regardless of in-memory enumeration order.
            rel_a, rel_b = (a, b) if str(a.id) < str(b.id) else (b, a)

            findings.append(
                {
                    "relationship_a_id": str(rel_a.id),
                    "relationship_b_id": str(rel_b.id),
                    "source_entity_id": str(rel_a.source_entity_id),
                    "target_entity_id": str(rel_a.target_entity_id),
                    "relationship_type": rel_a.relationship_type,
                    "similarity": similarity,
                    "contradicting_keys": list(contradicting_keys),
                    "reason": reason,
                    "description_a_hash": bounded_text_hash(rel_a.description or ""),
                    "description_b_hash": bounded_text_hash(rel_b.description or ""),
                }
            )
    return findings


def _text_similarity(a: Relationship, b: Relationship) -> float:
    """Textual similarity ratio over description + serialized properties.

    Uses :class:`difflib.SequenceMatcher` (stdlib, no extra deps) — a
    junk-aware Ratcliff/Obershelp variant that returns ``[0.0, 1.0]``.
    Property values are appended in sorted-key order so the comparison
    is order-independent.
    """
    text_a = _comparable_text(a)
    text_b = _comparable_text(b)
    if not text_a and not text_b:
        # Two empty descriptions with no properties — treat as similar
        # to avoid false-positive flagging on data we have no signal on.
        return 1.0
    return SequenceMatcher(a=text_a, b=text_b).ratio()


def _comparable_text(rel: Relationship) -> str:
    """Build the text blob used for pairwise similarity scoring."""
    parts: list[str] = [rel.description or ""]
    props = rel.properties or {}
    for key in sorted(props):
        parts.append(f"{key}={props[key]}")
    return " | ".join(parts)


def _contradicting_property_keys(a: Relationship, b: Relationship) -> list[str]:
    """Return the sorted list of shared property keys whose values differ."""
    pa = a.properties or {}
    pb = b.properties or {}
    shared = set(pa) & set(pb)
    return sorted(k for k in shared if str(pa[k]) != str(pb[k]))


# ---------------------------------------------------------------------------
# Apply handler — persists findings, NEVER mutates relationships
# ---------------------------------------------------------------------------


_INSERT_FINDING_SQL = text(
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
        valid_from
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
        :valid_from
    )
    ON CONFLICT (namespace_id, relationship_a_id, relationship_b_id) DO NOTHING
    """
)


async def apply_vectorcypher_contradiction_detect(
    op: DreamOp,
    *,
    coordinator: Any = None,
    session: AsyncSession,
) -> UndoRecord:
    """Persist contradiction findings — never touches ``relationships``.

    Inserts one row per finding into ``dream_conflicts`` with
    ``valid_from=NOW()``. Each insert is gated by
    ``ON CONFLICT (namespace_id, relationship_a_id, relationship_b_id)
    DO NOTHING`` so a replay of the same op is a no-op. The handler is a
    pure additive write to a dedicated finding table — the underlying
    relationships are never read or modified.

    Args:
        op: The planned op. ``outputs`` holds the finding dicts produced
            by :func:`plan_vectorcypher_contradiction_detect`.
        coordinator: Unused — kept to satisfy the apply-handler protocol.
        session: Orchestrator-owned async session.

    Returns:
        :class:`UndoRecord` whose ``before["findings"]`` lists the
        inserted finding ids so ``dream_undo`` can delete them.
    """
    del coordinator  # session is the only write surface

    findings = list(op.outputs)
    applied_at = datetime.now(UTC)

    if not findings:
        return UndoRecord(
            op_id=op.op_id,
            op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT.value,
            before={"findings": []},
            applied_at=applied_at,
        )

    namespace_id = _coerce_uuid(op.namespace_id) or _coerce_uuid(
        (op.inputs[0] if op.inputs else {}).get("namespace_id")
    )

    undo_entries: list[dict[str, Any]] = []

    for finding in findings:
        finding_id = uuid4()
        params: dict[str, Any] = {
            "id": finding_id,
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
        }
        await session.execute(_INSERT_FINDING_SQL, params)
        undo_entries.append(
            {
                "finding_id": str(finding_id),
                "relationship_a_id": finding["relationship_a_id"],
                "relationship_b_id": finding["relationship_b_id"],
            }
        )

    return UndoRecord(
        op_id=op.op_id,
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT.value,
        before={"findings": undo_entries},
        applied_at=applied_at,
    )


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None
