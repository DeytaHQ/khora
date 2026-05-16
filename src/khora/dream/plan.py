"""Dream plan and op dataclasses.

Stability:

- :class:`OpKind` — the enum type itself is part of the public API; the
  set of individual enum members may grow over Phase 0 without a major
  bump.
- :class:`DreamScope`, :class:`DreamOp`, :class:`DreamPlan` — internal
  stability (may evolve until the orchestrator lands in #661).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class OpKind(StrEnum):
    """The kinds of operation a dream plan can schedule.

    Values are stable strings so reports / logs can be diffed across
    versions; the *set* of values is internal until Phase 0 closes.
    """

    DEDUPE_ENTITIES = "dedupe_entities"
    PRUNE_EDGES = "prune_edges"
    COMPACT_FACTS = "compact_facts"
    CLUSTER_EVENTS = "cluster_events"
    RECOMPUTE_CENTROIDS = "recompute_centroids"
    VECTORCYPHER_SCHEMA_DRIFT_REPORT = "vectorcypher_schema_drift_report"
    VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT = "vectorcypher_source_chunk_ids_audit"
    CHRONICLE_ABSTENTION_DRIFT_REPORT = "chronicle_abstention_drift_report"
    VECTORCYPHER_ORPHAN_REPORT = "vectorcypher_orphan_report"


@dataclass(slots=True, frozen=True)
class DreamScope:
    """Bounded scope for a dream run.

    Lets callers limit what the orchestrator looks at — by op kind, by a
    time window over event ingestion, or by a hand-rolled set of entity
    or document ids. ``None`` fields mean "no restriction".
    """

    op_kinds: tuple[OpKind, ...] | None = None
    since: datetime | None = None
    until: datetime | None = None
    entity_ids: tuple[UUID, ...] | None = None
    document_ids: tuple[UUID, ...] | None = None


@dataclass(slots=True, frozen=True)
class DreamOp:
    """A single planned (or executed) dream operation.

    Records both the *intent* (op_type, inputs, decision, rationale) and
    the *outcome* (outputs, started_at, duration_ms). Optional ``undo``
    field is a backend-specific recipe to reverse the op when the
    orchestrator supports rollback.
    """

    op_id: UUID
    phase: str
    op_type: OpKind
    inputs: tuple[Any, ...] = field(default_factory=tuple)
    outputs: tuple[Any, ...] = field(default_factory=tuple)
    decision: str = ""
    rationale: str = ""
    source_llm_call_ids: tuple[str, ...] = field(default_factory=tuple)
    undo: dict[str, Any] | None = None
    started_at: datetime | None = None
    duration_ms: float | None = None
    namespace_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class DreamPlan:
    """An ordered list of :class:`DreamOp` instances for a single run."""

    plan_id: UUID
    namespace_id: UUID
    ops: tuple[DreamOp, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class Checkpoint:
    """Resume marker for a partially-applied dream run.

    Carries the ``plan_hash`` so a resume can validate the world hasn't
    changed under the orchestrator between attempts. ``plan_hash`` is
    ``sha1(canonical_json(plan))`` computed by the orchestrator when the
    plan is first persisted.
    """

    run_id: UUID
    last_committed_op_seq: int
    plan_hash: str
