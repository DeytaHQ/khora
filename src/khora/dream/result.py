"""Dream result, run-info, diff, progress, and op-summary dataclasses.

Stability:

- :class:`DreamResult`, :class:`DreamMode`, :class:`DreamRunInfo` — public.
- :class:`DreamDiff`, :class:`DreamProgress`, :class:`OpSummary` —
  internal stability (may evolve through Phase 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

if TYPE_CHECKING:
    from khora.khora import LLMUsage

DreamMode = Literal["dry-run", "apply"]
"""Run mode: ``dry-run`` produces a plan + diff only; ``apply`` executes ops."""


@dataclass(slots=True, frozen=True)
class OpSummary:
    """Aggregate counters for a single op kind within a run."""

    op_type: str
    planned: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass(slots=True, frozen=True)
class DreamDiff:
    """Counts of entities / edges / facts / clusters touched by a run.

    ``before`` / ``after`` snapshots are intentionally just int tallies in
    Phase 0 — full structural diffs land in #666.
    """

    entities_merged: int = 0
    entities_added: int = 0
    edges_pruned: int = 0
    edges_added: int = 0
    facts_compacted: int = 0
    clusters_created: int = 0
    centroids_recomputed: int = 0


@dataclass(slots=True, frozen=True)
class DreamProgress:
    """Progress event delivered to the ``on_progress`` callback.

    Carries enough context for a UI to render a progress bar without
    needing to fetch the full plan.
    """

    run_id: UUID
    phase: str
    op_index: int
    op_total: int
    op_type: str | None = None
    message: str = ""


@dataclass(slots=True, frozen=True)
class UndoRecord:
    """Per-op snapshot recorded by the apply-phase before each commit.

    The orchestrator collects an ``UndoRecord`` from every apply handler
    and persists the run's full list to ``{run_id}.undo.json`` (schema
    ``dream-undo/1``). The ``before`` payload is a JSON-serializable
    snapshot of whatever the apply handler needs to reverse the op (row
    contents, edge tuples, fact rows, etc.). A handler that performs no
    mutation returns an :class:`UndoRecord` with an empty ``before``.

    Stability: internal — the on-disk schema is versioned via
    ``dream-undo/<n>``; this in-memory dataclass may evolve freely.
    """

    op_id: UUID
    op_type: str
    before: dict[str, Any]
    applied_at: datetime


@dataclass(slots=True, frozen=True)
class DreamRunInfo:
    """Run-level metadata recorded alongside the plan / diff."""

    run_id: UUID
    namespace_id: UUID
    mode: DreamMode
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None
    resume_of: UUID | None = None


@dataclass(slots=True, frozen=True)
class DreamResult:
    """Top-level return of :meth:`khora.Khora.dream`.

    Phase 0 keeps the schema deliberately flat — richer relationship
    metadata lands in #666 (reports) and #661 (orchestrator).
    """

    run: DreamRunInfo
    diff: DreamDiff
    ops: tuple[OpSummary, ...] = field(default_factory=tuple)
    llm_usage: tuple[LLMUsage, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
