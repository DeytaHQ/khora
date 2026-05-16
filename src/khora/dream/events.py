"""Dream-phase reporting event payloads (#666).

Pydantic v2 models for the six dream report event types consumed by the
three sinks in :mod:`khora.dream.report`. The orchestrator emits these
events; sinks subscribe to the same stream.

Stability: **internal** (importable from ``khora.dream`` but not in the
top-level ``khora.__all__``). The public surface is the six
``EventType.DREAM_*`` values + the operator-facing top-level spans /
metrics in ``docs/telemetry-contract.json``. See module ``__init__``
docstring for the stability split.

Cardinality + redaction discipline:

- ``namespace_id`` is SAFE on the event payload (which lands on a file
  row or a span attribute). It is NEVER permitted on a metric label —
  see ``CLAUDE.md`` "Cardinality rule".
- Free text (rationale, summary, raw inputs) goes through
  :func:`khora.telemetry.bounded_text_hash` before being attached as a
  span attribute. ``rationale_hash`` and ``text_refs`` carry the hashes;
  the verbatim text rides on the file-sink payload only when
  ``DreamConfig.redact_text`` permits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Sub-payloads
# ---------------------------------------------------------------------------


class DreamRationale(BaseModel):
    """The "why" behind a dream operation decision.

    All free text goes through :func:`bounded_text_hash` before becoming a
    span attribute; the hash is what ``rationale_hash`` carries. The file
    sink may persist the verbatim rationale string separately, subject to
    ``DreamConfig.redact_text``.
    """

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(
        description="Short stable identifier for the decision strategy "
        "(e.g. 'cosine_above_threshold', 'llm_verifier').",
    )
    score: float | None = Field(
        default=None,
        description="Primary numeric score that drove the decision (cosine, weight, etc.).",
    )
    threshold: float | None = Field(
        default=None,
        description="The threshold that ``score`` was compared against.",
    )
    llm_confidence: float | None = Field(
        default=None,
        description="LLM confidence in [0,1] when ``strategy`` involved an LLM.",
    )
    rationale_hash: str | None = Field(
        default=None,
        description="bounded_text_hash of the free-text rationale, if any. "
        "Verbatim text never appears on this model — see CLAUDE.md "
        "free-text rule.",
    )


class UndoHandle(BaseModel):
    """Backend-agnostic recipe for reversing a single dream op."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "split",
        "restore_edge",
        "delete_summary",
        "rewrite_fact",
        "noop",
    ] = Field(description="Reversal strategy.")
    payload_ref: str = Field(
        description="Reference to the persisted undo payload, e.g. ``undo/<op_id>.json`` for the file sink.",
    )
    reversible_until: datetime | None = Field(
        default=None,
        description="UTC timestamp after which the undo is no longer guaranteed.",
    )


# ---------------------------------------------------------------------------
# Run-level events
# ---------------------------------------------------------------------------


class DreamRunStarted(BaseModel):
    """Emitted when the orchestrator opens a new run."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    namespace_id: UUID
    mode: Literal["dry-run", "apply"]
    trigger: str = Field(
        default="manual",
        description="What triggered this run (manual, scheduled, api, ...).",
    )
    started_at: datetime


class DreamPhaseStarted(BaseModel):
    """Emitted when a single dream phase begins (audit / mutation / ...)."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    namespace_id: UUID
    phase: str
    started_at: datetime


class DreamOperationEvent(BaseModel):
    """The per-op record shipped to all three sinks.

    File sink writes this verbatim to ``{run_id}.events.jsonl``. Event
    sink wraps it into ``MemoryEvent(event_type=DREAM_OP_DECIDED, ...)``
    via the existing :class:`HookDispatcher`. Collector sink reads only
    the low-cardinality scalars to populate metric labels and span
    attributes — every free-text field is hashed before it lands on a
    span.
    """

    model_config = ConfigDict(frozen=True)

    op_id: UUID
    run_id: UUID
    phase: str = Field(description="Phase the op belongs to (e.g. 'audit', 'mutation').")
    op_type: str = Field(
        description="Op kind — matches ``khora.dream.OpKind`` values plus "
        "future extensions; bounded to a small enum at runtime.",
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    decision: str = Field(description="Bounded-enum decision string.")
    rationale: DreamRationale
    source_llm_call_ids: list[UUID] = Field(default_factory=list)
    undo: UndoHandle | None = None
    started_at: datetime
    duration_ms: float = Field(ge=0.0)
    namespace_id: UUID
    text_refs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of input-name → bounded_text_hash for any free-text "
        "input that the op consulted. Hash-only; verbatim text "
        "rides on the file-sink payload subject to redact_text.",
    )


class DreamPhaseCompleted(BaseModel):
    """Emitted when a phase finishes (success or planned skip)."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    namespace_id: UUID
    phase: str
    outcome: Literal["success", "skipped", "failed"]
    ops_total: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)


class DreamRunCompleted(BaseModel):
    """Emitted on a clean run finish."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    namespace_id: UUID
    mode: Literal["dry-run", "apply"]
    duration_ms: float = Field(ge=0.0)
    ops_total: int = Field(ge=0)


class DreamRunFailed(BaseModel):
    """Emitted on a run abort. ``error_hash`` is a bounded_text_hash, not raw."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    namespace_id: UUID
    mode: Literal["dry-run", "apply"]
    duration_ms: float = Field(ge=0.0)
    error_hash: str
    error_type: str = Field(description="Exception class name, e.g. 'TimeoutError'.")


# Union of every payload a sink can receive. Order matches the orchestrator's
# emission order over a run.
DreamReportEvent = (
    DreamRunStarted | DreamPhaseStarted | DreamOperationEvent | DreamPhaseCompleted | DreamRunCompleted | DreamRunFailed
)


__all__ = [
    "DreamRationale",
    "UndoHandle",
    "DreamRunStarted",
    "DreamPhaseStarted",
    "DreamOperationEvent",
    "DreamPhaseCompleted",
    "DreamRunCompleted",
    "DreamRunFailed",
    "DreamReportEvent",
]
