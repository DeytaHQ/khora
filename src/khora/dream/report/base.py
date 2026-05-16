"""Common :class:`ReportSink` Protocol shared by the three dream sinks.

A sink consumes :class:`DreamReportEvent` payloads emitted by the
orchestrator. The three concrete implementations (file, event, collector)
all bind to this Protocol so the orchestrator can fan out to whichever
combination is enabled.

Stability: **internal** (Phase 0; the orchestrator wiring lands in #661).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from khora.dream.events import DreamReportEvent


class DreamReportSchemaMismatchError(Exception):
    """Raised when a persisted dream report carries an unknown schema version.

    The file sink writes ``"schema": "dream-report/1"`` on every artifact;
    readers that load older or newer schemas without a migration path
    surface the mismatch loudly rather than silently misinterpreting fields.
    """


class ReportSink(ABC):
    """Single-method interface every dream report sink implements.

    :meth:`emit` is allowed to block on I/O (file sink writes JSONL,
    collector sink calls OTel APIs). The orchestrator dispatches each
    sink in its own task so a slow sink can't stall the others.
    """

    @abstractmethod
    async def emit(self, event: DreamReportEvent) -> None:
        """Consume one event payload."""

    async def flush(self) -> None:
        """Flush any buffered state. Default no-op."""

    async def close(self) -> None:
        """Release any persistent resources. Default no-op."""


__all__ = [
    "ReportSink",
    "DreamReportSchemaMismatchError",
]
