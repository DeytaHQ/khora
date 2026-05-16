"""Dream-phase reporting sinks (#666).

Three independently-togglable sinks consume the same
:class:`DreamReportEvent` stream emitted by the orchestrator:

- :class:`DreamFileSink` — append-only JSONL + summary / manifest /
  undo files under ``DreamConfig.report_dir``. Schema-versioned via
  :data:`SCHEMA_VERSION`.
- :class:`DreamEventSink` — bridges into the existing
  :class:`HookDispatcher` with six new :class:`EventType.DREAM_*`
  values; reuses the full filter cascade.
- :class:`DreamCollectorSink` — OTel spans + metrics. Cardinality rule
  respected: ``namespace_id`` is a span attribute only, never a metric
  label.

All sinks implement :class:`ReportSink`. The orchestrator dispatches to
whichever set is enabled in :class:`khora.dream.config.DreamConfig`.

Stability: **internal** (Phase 0). The public surface is the
:class:`EventType.DREAM_*` values plus the operator-facing top-level
spans + metrics documented in ``docs/telemetry-contract.json``.
"""

from __future__ import annotations

from khora.dream.report.base import DreamReportSchemaMismatchError, ReportSink
from khora.dream.report.collector_sink import (
    DreamCollectorSink,
    record_llm_tokens,
    record_undo_invocation,
)
from khora.dream.report.event_sink import DreamEventSink
from khora.dream.report.file_sink import (
    SCHEMA_VERSION,
    DreamFileSink,
    expire_dream_reports,
    load_manifest,
)

__all__ = [
    "ReportSink",
    "DreamReportSchemaMismatchError",
    "DreamFileSink",
    "DreamEventSink",
    "DreamCollectorSink",
    "SCHEMA_VERSION",
    "load_manifest",
    "expire_dream_reports",
    "record_llm_tokens",
    "record_undo_invocation",
]
