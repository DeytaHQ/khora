"""Schema-drift report for the vectorcypher engine (#655, Phase 1.3).

Diffs the multiset of ``entity_type`` and ``relationship_type`` strings
observed in a namespace's data against the active ``ExpertiseConfig``.
Surfaces drift in three buckets per axis; never auto-normalizes —
``ExpertiseConfig`` is declarative user intent and normalization is a
Phase 5.4 (#673) concern that requires an operator-supplied mapping.

Stability: **internal**. The op kind string
``vectorcypher_schema_drift_report`` is part of the
:class:`khora.dream.OpKind` enum and stable as a string identifier;
the planning helper signature and op-inputs/outputs shape may evolve
through Phase 1.

Read previous-run frequencies from ``khora_dream_runs`` (migration 032)
via the caller. ``previous_run_id`` is stamped into ``DreamOp.inputs``
for audit; ``previous_entity_frequencies`` /
``previous_relationship_frequencies`` carry the actual counts the
caller resolved from that run's persisted report (file sink) or from
an in-memory cache. When neither previous-frequency map is supplied
(typical first-run case), the ``frequency_delta`` buckets are empty
rather than crashing — we can't compute a delta without a baseline.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from khora.dream.plan import DreamOp, OpKind

if TYPE_CHECKING:
    from khora.extraction.skills.base import ExpertiseConfig
    from khora.storage.coordinator import StorageCoordinator

# Page size used when scanning the namespace's entities and
# relationships. Picked to match the existing
# ``coordinator.list_entities`` / ``list_relationships`` default upper
# bounds without forcing the backend into a single megaquery.
_PAGE_SIZE = 1000

# Threshold below which a frequency change is not considered a drift
# signal. The spec calls out ">=50%" — symmetric: a count that doubles
# trips, and a count that halves trips.
_FREQUENCY_DELTA_THRESHOLD = 0.50


async def plan_vectorcypher_schema_drift(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    expertise: ExpertiseConfig,
    previous_run_id: UUID | None = None,
    previous_entity_frequencies: dict[str, int] | None = None,
    previous_relationship_frequencies: dict[str, int] | None = None,
) -> DreamOp:
    """Plan (and immediately resolve, since this is a read-only audit op)
    a schema-drift report for ``namespace_id``.

    Returns a single :class:`DreamOp` carrying:

    - ``op_type`` = ``OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT``
    - ``decision`` = ``"audit_complete"`` (always — this op never
      proposes a mutation)
    - ``inputs`` — ``(previous_run_id,)`` for audit / replay
    - ``outputs`` — a single dict with six diff buckets plus the raw
      observed-frequency multisets and total counts

    Output dict shape::

        {
            "new_entity_types": ["EMPLOYEE", ...],          # in data, not in config
            "unused_entity_types": ["RECIPE", ...],         # in config, not in data
            "entity_frequency_delta": {"PERSON": (100, 200)},
            "new_relationship_types": [...],
            "unused_relationship_types": [...],
            "relationship_frequency_delta": {...},
            "entity_frequencies": {"PERSON": 200, ...},
            "relationship_frequencies": {...},
            "entity_total": int,
            "relationship_total": int,
        }

    All free-text values are bounded enum strings (entity / relationship
    type names) — they ride on the file-sink payload but never become
    metric labels.
    """
    # 1. Observe — paginate through entities and relationships,
    #    accumulating type-name multisets. Coordinator falls back to the
    #    vector backend on graph-less stacks; we don't care which one
    #    answers as long as the rows come back.
    entity_counts = await _count_entity_types(coordinator, namespace_id)
    relationship_counts = await _count_relationship_types(coordinator, namespace_id)

    # 2. Diff against ExpertiseConfig.
    declared_entity_types = set(expertise.get_entity_type_names())
    declared_relationship_types = set(expertise.get_relationship_type_names())

    new_entity_types = sorted(set(entity_counts) - declared_entity_types)
    unused_entity_types = sorted(declared_entity_types - set(entity_counts))
    new_relationship_types = sorted(set(relationship_counts) - declared_relationship_types)
    unused_relationship_types = sorted(declared_relationship_types - set(relationship_counts))

    # 3. Frequency delta vs previous run (when supplied).
    entity_frequency_delta = _frequency_delta(previous_entity_frequencies, entity_counts)
    relationship_frequency_delta = _frequency_delta(previous_relationship_frequencies, relationship_counts)

    outputs = {
        "new_entity_types": new_entity_types,
        "unused_entity_types": unused_entity_types,
        "entity_frequency_delta": entity_frequency_delta,
        "new_relationship_types": new_relationship_types,
        "unused_relationship_types": unused_relationship_types,
        "relationship_frequency_delta": relationship_frequency_delta,
        "entity_frequencies": dict(entity_counts),
        "relationship_frequencies": dict(relationship_counts),
        "entity_total": sum(entity_counts.values()),
        "relationship_total": sum(relationship_counts.values()),
    }

    return DreamOp(
        op_id=uuid4(),
        phase="audit",
        op_type=OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT,
        inputs=(previous_run_id,),
        outputs=(outputs,),
        decision="audit_complete",
        rationale=(
            "Schema-drift audit comparing observed entity/relationship type "
            "frequencies against the active ExpertiseConfig declaration."
        ),
        namespace_id=namespace_id,
    )


async def _count_entity_types(coordinator: StorageCoordinator, namespace_id: UUID) -> Counter[str]:
    """Paginate entities in ``namespace_id`` and tally type-name frequencies."""
    counts: Counter[str] = Counter()
    offset = 0
    while True:
        batch = await coordinator.list_entities(
            namespace_id,
            limit=_PAGE_SIZE,
            offset=offset,
        )
        if not batch:
            break
        for entity in batch:
            counts[entity.entity_type] += 1
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return counts


async def _count_relationship_types(coordinator: StorageCoordinator, namespace_id: UUID) -> Counter[str]:
    """Paginate relationships in ``namespace_id`` and tally type-name frequencies."""
    counts: Counter[str] = Counter()
    offset = 0
    while True:
        batch = await coordinator.list_relationships(
            namespace_id,
            limit=_PAGE_SIZE,
            offset=offset,
        )
        if not batch:
            break
        for rel in batch:
            counts[rel.relationship_type] += 1
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return counts


def _frequency_delta(
    previous: dict[str, int] | None,
    current: Counter[str],
) -> dict[str, tuple[int, int]]:
    """Return types whose frequency changed by >= 50% vs ``previous``.

    Empty result when ``previous`` is ``None`` (no baseline available —
    first-run case) or when no type meets the threshold.

    Map value is ``(previous_count, current_count)`` so the report sink
    can render both ends of the delta without a second lookup.
    """
    if previous is None:
        return {}
    delta: dict[str, tuple[int, int]] = {}
    keys = set(previous) | set(current)
    for key in keys:
        prev = previous.get(key, 0)
        curr = current.get(key, 0)
        if prev == 0 and curr == 0:
            continue
        # Use the larger of the two as the denominator so a fresh type
        # (prev=0) trips, and a deleted type (curr=0) also trips. This
        # matches the symmetric ">= 50% change" reading of the spec.
        denom = max(prev, curr)
        change = abs(curr - prev) / denom
        if change >= _FREQUENCY_DELTA_THRESHOLD:
            delta[key] = (prev, curr)
    return delta
