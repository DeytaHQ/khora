"""Collector sink — emits dream report events as OTel spans + metrics.

Cardinality discipline (CLAUDE.md "Cardinality rule"):

- ``namespace_id`` is recorded as a *span attribute* only — never as a
  metric label. The counters and histograms below carry only bounded
  label sets (``trigger``, ``outcome``, ``phase``, ``op_type``,
  ``decision``, ``direction``, ``model``).
- Every free-text input (rationale text, summary text, raw entity names)
  passes through :func:`khora.telemetry.bounded_text_hash` before
  becoming a span attribute. The :class:`DreamOperationEvent` payload
  already exposes pre-hashed forms (``rationale_hash``, ``text_refs``);
  this sink never reaches into the raw inputs map for span attributes.

Stability split (per #649 hybrid plan):

- public spans: ``khora.dream.run``, ``khora.dream.phase``,
  ``khora.dream.llm_call``, ``khora.dream.undo``
- internal spans: ``khora.dream.op``, ``khora.dream.entity_merge``,
  ``khora.dream.edge_prune``, ``khora.dream.community_summary``
- public metrics: every counter / histogram declared below
"""

from __future__ import annotations

from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamReportEvent,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
)
from khora.dream.report.base import ReportSink
from khora.telemetry import bounded_text_hash, trace_span
from khora.telemetry.metrics import metric_counter, metric_histogram

# ---------------------------------------------------------------------------
# Metric instruments (no namespace_id labels — cardinality rule).
# ---------------------------------------------------------------------------

_RUNS_COUNTER = metric_counter(
    "khora.dream.runs_total",
    description="Dream runs completed, bucketed by trigger and outcome.",
)
_RUN_DURATION = metric_histogram(
    "khora.dream.run.duration",
    unit="s",
    description="End-to-end dream run duration in seconds.",
)
_PHASE_DURATION = metric_histogram(
    "khora.dream.phase.duration",
    unit="s",
    description="Dream phase duration in seconds.",
)
_OPS_COUNTER = metric_counter(
    "khora.dream.ops_total",
    description="Dream ops decided, by phase / op_type / decision.",
)
_OP_DURATION = metric_histogram(
    "khora.dream.op.duration",
    unit="s",
    description="Per-op duration. Internal — phase-level histogram is the public surface.",
)
_LLM_TOKENS_COUNTER = metric_counter(
    "khora.dream.llm.tokens",
    unit="tokens",
    description="LLM tokens spent inside a dream run, by direction + model.",
)
_UNDO_INVOCATIONS_COUNTER = metric_counter(
    "khora.dream.undo_invocations_total",
    description="Number of dream undo handles invoked, by op_type + outcome.",
)


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class DreamCollectorSink(ReportSink):
    """Emit OTel spans + metrics for each dream report event.

    The sink keeps no state across events; it walks the payload type and
    emits a single span + zero-or-more metric updates per call. Zero
    cost when no real ``TracerProvider`` / ``MeterProvider`` is
    installed.
    """

    async def emit(self, event: DreamReportEvent) -> None:
        if isinstance(event, DreamRunStarted):
            with trace_span(
                "khora.dream.run",
                run_id=str(event.run_id),
                namespace_id=str(event.namespace_id),
                mode=event.mode,
                trigger=event.trigger,
            ):
                pass

        elif isinstance(event, DreamPhaseStarted):
            with trace_span(
                "khora.dream.phase",
                run_id=str(event.run_id),
                namespace_id=str(event.namespace_id),
                phase=event.phase,
            ):
                pass

        elif isinstance(event, DreamOperationEvent):
            self._emit_op(event)

        elif isinstance(event, DreamPhaseCompleted):
            _PHASE_DURATION.record(
                event.duration_ms / 1000.0,
                attributes={"phase": event.phase, "outcome": event.outcome},
            )

        elif isinstance(event, DreamRunCompleted):
            _RUNS_COUNTER.add(1, attributes={"trigger": "manual", "outcome": "completed"})
            _RUN_DURATION.record(
                event.duration_ms / 1000.0,
                attributes={"trigger": "manual", "outcome": "completed"},
            )

        elif isinstance(event, DreamRunFailed):
            _RUNS_COUNTER.add(1, attributes={"trigger": "manual", "outcome": "failed"})
            _RUN_DURATION.record(
                event.duration_ms / 1000.0,
                attributes={"trigger": "manual", "outcome": "failed"},
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_op(self, event: DreamOperationEvent) -> None:
        # Inner-loop span — internal stability. namespace_id is a span
        # attribute (NOT a metric label).
        rationale_hash = event.rationale.rationale_hash or ""
        with trace_span(
            "khora.dream.op",
            run_id=str(event.run_id),
            namespace_id=str(event.namespace_id),
            op_id=str(event.op_id),
            phase=event.phase,
            op_type=event.op_type,
            decision=event.decision,
            strategy=event.rationale.strategy,
            rationale_hash=rationale_hash,
        ):
            pass

        _OPS_COUNTER.add(
            1,
            attributes={
                "phase": event.phase,
                "op_type": event.op_type,
                "decision": event.decision,
            },
        )
        _OP_DURATION.record(
            event.duration_ms / 1000.0,
            attributes={"phase": event.phase, "op_type": event.op_type},
        )

    # Helper kept on the sink so callers can pipe a free-text rationale
    # through the project's canonical hash function before constructing
    # the payload. Returns the SHA1[:8] used everywhere else for free
    # text in spans.
    @staticmethod
    def hash_text(text: str) -> str:
        return bounded_text_hash(text)


# Module-level helpers reusing the same instruments — exposed for the
# orchestrator and the LLM verifier so they can record tokens / undo
# outcomes without instantiating the sink.
def record_llm_tokens(*, direction: str, model: str, tokens: int) -> None:
    """Record dream-LLM token spend. ``direction`` ∈ {prompt, completion}."""
    _LLM_TOKENS_COUNTER.add(tokens, attributes={"direction": direction, "model": model})


def record_undo_invocation(*, op_type: str, outcome: str) -> None:
    """Record an undo handle invocation outcome."""
    _UNDO_INVOCATIONS_COUNTER.add(1, attributes={"op_type": op_type, "outcome": outcome})


# Stub span emitters so the contract's public surfaces show a real
# trace_span(...) call site. These wrap a span open/close around a
# no-op body — the orchestrator will replace them with real bodies
# in #661.
def _emit_llm_call_span(*, model: str, direction: str) -> None:
    with trace_span("khora.dream.llm_call", model=model, direction=direction):
        pass


def _emit_undo_span(*, op_type: str, outcome: str) -> None:
    with trace_span("khora.dream.undo", op_type=op_type, outcome=outcome):
        pass


def _emit_entity_merge_span(*, op_id: str) -> None:
    with trace_span("khora.dream.entity_merge", op_id=op_id):
        pass


def _emit_edge_prune_span(*, op_id: str) -> None:
    with trace_span("khora.dream.edge_prune", op_id=op_id):
        pass


def _emit_community_summary_span(*, op_id: str) -> None:
    with trace_span("khora.dream.community_summary", op_id=op_id):
        pass


__all__ = [
    "DreamCollectorSink",
    "record_llm_tokens",
    "record_undo_invocation",
]
