"""Local fallback for :class:`UndoRecord` while #667 is in flight.

The parallel orchestrator-apply work (#667) adds the canonical
``UndoRecord`` to :mod:`khora.dream.result`. Until that PR lands, the
chronicle apply handlers in this module need a working dataclass to
return from. The shape is the contract the orchestrator will consume:

* ``op_id`` — the originating :class:`khora.dream.plan.DreamOp` id
* ``op_type`` — the canonical op-kind string
* ``before`` — JSON-serialisable snapshot of the row(s) the op touched
  *before* mutation. For ``fact_compaction`` this carries the full row
  content of every deleted ``memory_facts`` row.
* ``applied_at`` — when the handler ran (tz-aware UTC).

The handler MUST NOT log, emit telemetry, or commit — the orchestrator
owns the transaction. On hand-off (when #667 merges), this module
collapses into a re-export of :mod:`khora.dream.result.UndoRecord` and
existing imports continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


@dataclass(slots=True, frozen=True)
class UndoRecord:
    """Reversibility snapshot returned by an apply handler.

    ``before`` is the *pre*-state; the orchestrator persists this to the
    file sink (``undo.json``) so an operator can hand-roll a restore.
    Restore is best-effort and lossy under concurrent writes — that
    constraint is documented per-op in the handler docstrings.
    """

    op_id: UUID
    op_type: str
    before: dict[str, Any] = field(default_factory=dict)
    applied_at: datetime = field(default_factory=lambda: datetime.now(UTC))


__all__ = ["UndoRecord"]
