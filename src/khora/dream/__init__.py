"""Dream phase — periodic consolidation, dedupe, and pruning pass.

Phase 0.1 lands the typed surface only. The orchestrator that turns a
plan into observed changes ships in #661; the report sinks ship in #666.

Stability:

- **Public** (in ``__all__``): :class:`DreamConfig`, :class:`DreamMode`,
  :class:`DreamScope`, :class:`DreamResult`, :class:`DreamRunInfo`,
  :class:`OpKind`, :func:`acquire_namespace_dream_lock`,
  :class:`DreamLockUnavailable`, :class:`ReportSink`.
- **Internal** (importable but may evolve without a major-version bump):
  :class:`DreamOp`, :class:`DreamPlan`, :class:`Checkpoint`,
  :class:`DreamProgress`, :class:`DreamDiff`, :class:`OpSummary`,
  :class:`DreamCapable`, :class:`DreamOrchestrator`,
  :class:`DreamRationale`, :class:`UndoHandle`,
  :class:`DreamRunStarted`, :class:`DreamPhaseStarted`,
  :class:`DreamOperationEvent`, :class:`DreamPhaseCompleted`,
  :class:`DreamRunCompleted`, :class:`DreamRunFailed`,
  :class:`DreamFileSink`, :class:`DreamEventSink`,
  :class:`DreamCollectorSink`.
"""

from __future__ import annotations

from khora.dream.config import DreamConfig, DreamOpsConfig
from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamRationale,
    DreamReportEvent,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
    UndoHandle,
)
from khora.dream.locks import DreamLockUnavailable, acquire_namespace_dream_lock
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import Checkpoint, DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.protocol import DreamCapable
from khora.dream.report import (
    DreamCollectorSink,
    DreamEventSink,
    DreamFileSink,
    DreamReportSchemaMismatchError,
    ReportSink,
)
from khora.dream.result import (
    DreamDiff,
    DreamMode,
    DreamProgress,
    DreamResult,
    DreamRunInfo,
    OpSummary,
)

__all__ = [
    "DreamConfig",
    "DreamLockUnavailable",
    "DreamMode",
    "DreamResult",
    "DreamRunInfo",
    "DreamScope",
    "OpKind",
    "acquire_namespace_dream_lock",
    # Sink Protocol (public; sinks themselves remain internal).
    "ReportSink",
]

# Re-bind internal symbols at module level so static-analysis sees them as
# used; not in __all__ — see module docstring for the stability split.
_INTERNAL = (
    DreamOp,
    DreamPlan,
    Checkpoint,
    DreamDiff,
    DreamProgress,
    OpSummary,
    DreamOpsConfig,
    DreamCapable,
    DreamOrchestrator,
    DreamRationale,
    UndoHandle,
    DreamRunStarted,
    DreamPhaseStarted,
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamRunCompleted,
    DreamRunFailed,
    DreamReportEvent,
    DreamFileSink,
    DreamEventSink,
    DreamCollectorSink,
    DreamReportSchemaMismatchError,
)
del _INTERNAL
