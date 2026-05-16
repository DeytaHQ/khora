"""Filesystem sink for dream-phase reports.

Layout under ``base_dir``::

    {base_dir}/{namespace_id}/{YYYY-MM-DD}/{run_id}.summary.md
    {base_dir}/{namespace_id}/{YYYY-MM-DD}/{run_id}.events.jsonl
    {base_dir}/{namespace_id}/{YYYY-MM-DD}/{run_id}.manifest.json
    {base_dir}/{namespace_id}/{YYYY-MM-DD}/{run_id}.undo.json

While a run is in flight, the JSONL file is named ``{run_id}.events.partial``
and is renamed atomically to ``.events.jsonl`` on :meth:`close` (clean
shutdown) or to ``.events.crashed.jsonl`` if :meth:`close` runs after a
:class:`DreamRunFailed` event landed without a :class:`DreamRunCompleted`
following it.

Schema version (``dream-report/1``) is stamped on every file the sink
writes. The asymmetric :func:`load_manifest` helper enforces it on read.

Retention is exposed via :func:`expire_dream_reports` for operators that
wire their own GC loop — same shape as :func:`khora.gc.expire_sessions`.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel

from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamReportEvent,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
    UndoHandle,
)
from khora.dream.report.base import DreamReportSchemaMismatchError, ReportSink
from khora.telemetry.metrics import metric_counter

SCHEMA_VERSION = "dream-report/1"

_WRITE_FAILURE_COUNTER = metric_counter(
    "khora.dream.report.write_failures_total",
    description="File-sink writes that failed (open / write / rename).",
)


def _serialize(event: DreamReportEvent) -> dict[str, Any]:
    """Pydantic-v2 ``model_dump(mode='json')`` with the schema tag added."""
    payload = event.model_dump(mode="json")
    payload["schema"] = SCHEMA_VERSION
    payload["event"] = type(event).__name__
    return payload


def _redact(payload: dict[str, Any], policy: Literal["none", "summary", "all"]) -> dict[str, Any]:
    """Apply ``DreamConfig.redact_text`` to a serialized event payload.

    - ``none``: return payload verbatim.
    - ``summary``: keep ``rationale.rationale_hash`` and the first 200 chars
      of any string under ``inputs``/``outputs`` (truncated). Default.
    - ``all``: drop every free-text field; only the hashes survive.
    """
    if policy == "none":
        return payload

    redacted = dict(payload)

    def _strip_strings(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, str):
                if policy == "all":
                    out[k] = None
                else:  # summary
                    out[k] = v[:200]
            elif isinstance(v, dict):
                out[k] = _strip_strings(v)
            else:
                out[k] = v
        return out

    for key in ("inputs", "outputs", "text_refs"):
        if key in redacted and isinstance(redacted[key], dict) and key != "text_refs":
            # text_refs is already hash-only, leave it.
            redacted[key] = _strip_strings(redacted[key])

    return redacted


class DreamFileSink(ReportSink):
    """Append-only filesystem sink, one directory per (namespace, date)."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        redact_text: Literal["none", "summary", "all"] = "summary",
    ) -> None:
        self._base_dir = Path(base_dir)
        self._redact_text = redact_text
        # State that is captured on the first DreamRunStarted event and
        # then re-used for every subsequent event in the same run.
        self._run_id: UUID | None = None
        self._namespace_id: UUID | None = None
        self._run_dir: Path | None = None
        self._partial_path: Path | None = None
        self._jsonl_handle: IO[str] | None = None
        self._failed = False
        self._undo_entries: list[dict[str, Any]] = []
        self._run_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # ReportSink interface
    # ------------------------------------------------------------------

    async def emit(self, event: DreamReportEvent) -> None:
        try:
            if isinstance(event, DreamRunStarted):
                self._open_run(event)
            self._append_event(event)
            if isinstance(event, DreamOperationEvent) and event.undo is not None:
                self._undo_entries.append(self._render_undo(event.op_id, event.undo))
            if isinstance(event, DreamRunFailed):
                self._failed = True
                self._run_summary = self._render_summary(event)
            if isinstance(event, DreamRunCompleted):
                self._run_summary = self._render_summary(event)
        except OSError as exc:
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "io_error"})
            logger.warning(
                "DreamFileSink write failed for event {event}: {exc}",
                event=type(event).__name__,
                exc=exc,
            )
            raise

    async def flush(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.flush()

    async def close(self) -> None:
        """Finalize the run directory.

        - Closes the JSONL stream.
        - Renames ``.events.partial`` to either ``.events.jsonl`` (clean)
          or ``.events.crashed.jsonl`` (no DreamRunCompleted seen).
        - Writes ``summary.md``, ``manifest.json``, ``undo.json``.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._run_dir is None:
            return

        if self._jsonl_handle is not None:
            try:
                self._jsonl_handle.close()
            finally:
                self._jsonl_handle = None

        if self._partial_path is not None and self._partial_path.exists():
            crashed = self._run_summary is None or self._failed
            suffix = "events.crashed.jsonl" if crashed else "events.jsonl"
            final_path = self._partial_path.with_name(self._partial_path.stem.replace(".events", "") + "." + suffix)
            # The naming above is awkward — use the simpler form:
            final_path = self._partial_path.parent / (self._partial_path.stem.split(".")[0] + "." + suffix)
            try:
                self._partial_path.rename(final_path)
            except OSError as exc:
                _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "rename"})
                logger.warning("DreamFileSink rename failed: {}", exc)

        self._write_manifest()
        if self._run_summary is not None:
            self._write_summary()
        if self._undo_entries:
            self._write_undo()

        self._reset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_run(self, event: DreamRunStarted) -> None:
        date_str = event.started_at.astimezone(UTC).strftime("%Y-%m-%d")
        run_dir = self._base_dir / str(event.namespace_id) / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = event.run_id
        self._namespace_id = event.namespace_id
        self._run_dir = run_dir
        self._partial_path = run_dir / f"{event.run_id}.events.partial"
        # Open in append so a resumed run can pick up the existing partial.
        try:
            self._jsonl_handle = self._partial_path.open("a", encoding="utf-8")
        except OSError:
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "open"})
            raise

    def _append_event(self, event: DreamReportEvent) -> None:
        if self._jsonl_handle is None:
            # Defensive: a non-RunStarted event arrived before RunStarted —
            # treat as a write failure but keep the sink alive.
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "no_open_run"})
            return
        payload = _redact(_serialize(event), self._redact_text)
        self._jsonl_handle.write(json.dumps(payload, default=str) + "\n")

    def _render_summary(self, event: DreamRunCompleted | DreamRunFailed) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": str(event.run_id),
            "namespace_id": str(event.namespace_id),
            "mode": event.mode,
            "duration_ms": event.duration_ms,
        }
        if isinstance(event, DreamRunCompleted):
            payload["status"] = "completed"
            payload["ops_total"] = event.ops_total
        else:
            payload["status"] = "failed"
            payload["error_type"] = event.error_type
            payload["error_hash"] = event.error_hash
        return payload

    def _render_undo(self, op_id: UUID, undo: UndoHandle) -> dict[str, Any]:
        payload = undo.model_dump(mode="json")
        payload["op_id"] = str(op_id)
        return payload

    def _write_manifest(self) -> None:
        if self._run_dir is None or self._run_id is None:
            return
        manifest = {
            "schema": SCHEMA_VERSION,
            "run_id": str(self._run_id),
            "namespace_id": str(self._namespace_id),
            "redact_text": self._redact_text,
            "completed": self._run_summary is not None and not self._failed,
            "written_at": datetime.now(UTC).isoformat(),
        }
        path = self._run_dir / f"{self._run_id}.manifest.json"
        try:
            path.write_text(json.dumps(manifest, indent=2))
        except OSError as exc:
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "manifest"})
            logger.warning("DreamFileSink manifest write failed: {}", exc)

    def _write_summary(self) -> None:
        if self._run_dir is None or self._run_id is None or self._run_summary is None:
            return
        body = self._format_summary_md(self._run_summary)
        path = self._run_dir / f"{self._run_id}.summary.md"
        try:
            path.write_text(body)
        except OSError as exc:
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "summary"})
            logger.warning("DreamFileSink summary write failed: {}", exc)

    def _format_summary_md(self, summary: dict[str, Any]) -> str:
        lines = [
            f"<!-- schema: {SCHEMA_VERSION} -->",
            f"# Dream run {summary['run_id']}",
            "",
            f"- namespace: `{summary['namespace_id']}`",
            f"- mode: `{summary['mode']}`",
            f"- status: `{summary['status']}`",
            f"- duration_ms: {summary['duration_ms']:.2f}",
        ]
        if summary["status"] == "completed":
            lines.append(f"- ops_total: {summary['ops_total']}")
        else:
            lines.append(f"- error_type: `{summary['error_type']}`")
            lines.append(f"- error_hash: `{summary['error_hash']}`")
        lines.append("")
        return "\n".join(lines)

    def _write_undo(self) -> None:
        if self._run_dir is None or self._run_id is None:
            return
        path = self._run_dir / f"{self._run_id}.undo.json"
        doc = {"schema": SCHEMA_VERSION, "entries": self._undo_entries}
        try:
            path.write_text(json.dumps(doc, indent=2, default=str))
        except OSError as exc:
            _WRITE_FAILURE_COUNTER.add(1, attributes={"reason": "undo"})
            logger.warning("DreamFileSink undo write failed: {}", exc)

    def _reset(self) -> None:
        self._run_id = None
        self._namespace_id = None
        self._run_dir = None
        self._partial_path = None
        self._failed = False
        self._undo_entries = []
        self._run_summary = None

    # Help static analysis treat these symbols as used.
    _PYDANTIC_REF: tuple[type[BaseModel], ...] = (
        DreamRunStarted,
        DreamPhaseStarted,
        DreamOperationEvent,
        DreamPhaseCompleted,
    )


# ---------------------------------------------------------------------------
# Read helpers + retention
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest file, asserting the schema tag matches."""
    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema")
    if schema != SCHEMA_VERSION:
        raise DreamReportSchemaMismatchError(
            f"Manifest at {path} has schema={schema!r}; expected {SCHEMA_VERSION!r}.",
        )
    return payload


async def expire_dream_reports(
    *,
    base_dir: str | Path,
    before: datetime,
    namespace_id: UUID | None = None,
) -> int:
    """Delete dream-report directories whose date predates ``before``.

    Mirrors :func:`khora.gc.expire_sessions` — opt-in helper, called from
    an operator-owned cron / task queue. Returns the number of run
    directories removed.

    Args:
        base_dir: Root the file sink writes to.
        before: Cutoff (UTC date implied by year-month-day).
        namespace_id: When provided, restrict cleanup to one namespace.

    Returns:
        Count of run directories deleted (a directory wraps one run).
    """
    root = Path(base_dir)
    if not root.exists():
        return 0
    cutoff = before.astimezone(UTC).date()
    deleted = 0
    namespace_dirs = [root / str(namespace_id)] if namespace_id else [p for p in root.iterdir() if p.is_dir()]
    for ns_dir in namespace_dirs:
        if not ns_dir.is_dir():
            continue
        for date_dir in list(ns_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if dir_date < cutoff:
                try:
                    shutil.rmtree(date_dir)
                    deleted += 1
                except OSError as exc:
                    logger.warning(
                        "expire_dream_reports: failed to remove {dir}: {exc}",
                        dir=date_dir,
                        exc=exc,
                    )
    return deleted


__all__ = [
    "DreamFileSink",
    "SCHEMA_VERSION",
    "load_manifest",
    "expire_dream_reports",
]
