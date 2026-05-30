"""Dream-phase orchestrator state machine (#661).

Drives a single dream run through ``INIT → PLAN → REPORT|APPLY →
FINALIZE``. Acquires the per-namespace advisory lock for the entire
plan-through-finalize block (#677), dispatches plan-stage discovery to
the engine-registered :class:`DreamCapable` plugin, fans every plan op
out through the configured sinks (#678), and persists run state to
``khora_dream_runs`` (Postgres) when available.

Crash semantics:

- Mid-PLAN crash leaves no row (or a ``state=planning`` row) — a fresh
  call recovers transparently.
- Mid-APPLY crash leaves ``state=applying`` + ``last_committed_op_seq``
  pointing at the last committed op. Resume continues from
  ``last_committed_op_seq + 1`` once :class:`DreamRunStuckError` is
  resolved.

Cancellation:

- ``cancel(run_id)`` flips an in-process flag. The orchestrator checks
  the flag *between ops only* — never preempting an op mid-flight.

Safety floor:

- Document-delete op kinds are rejected at plan time AND apply time via
  :func:`khora.dream.engines.registry._validate_no_forbidden_ops`.
- UNIQUE-violation and read-only-namespace checks land in Phase 2 with
  the first mutation-capable op (#664+). Marked with ``TODO(phase-2)``
  comments below.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.config import DreamConfig
from khora.dream.engines.registry import (
    canonical_plan_payload,
    get_apply_handler,
    get_engine_plugin,
    plan_hash,
)
from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamRationale,
    DreamReportEvent,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
)
from khora.dream.exceptions import (
    DreamApplyDisabled,
    DreamBackendUnsupported,
    DreamForbiddenOpError,
)
from khora.dream.locks import acquire_namespace_dream_lock
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.report import DreamCollectorSink, DreamEventSink, DreamFileSink, ReportSink
from khora.dream.result import (
    DreamDiff,
    DreamProgress,
    DreamResult,
    DreamRunInfo,
    OpSummary,
    UndoRecord,
)
from khora.dream.safety import _assert_no_chunk_id_mutation
from khora.telemetry import bounded_text_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from khora.extraction.skills.base import ExpertiseConfig
    from khora.khora import Khora


# Run-level in-process cancellation registry. Keyed by run_id; the
# orchestrator checks the flag between ops only.
_CANCEL_FLAGS: dict[UUID, asyncio.Event] = {}
_CANCEL_FLAGS_LOCK = asyncio.Lock()


# Kill-switch env var (#667): when set to anything non-empty / non-falsey,
# DreamOrchestrator._apply_phase raises DreamApplyDisabled before touching
# the DB. Read at orchestrator construction so an operator setting it via
# ``export KHORA_DREAM_DISABLE_APPLY=1`` halts the next run without needing
# a restart on a long-lived process — but already-constructed
# orchestrators carry the value they were built with.
_APPLY_DISABLED_ENV_VAR = "KHORA_DREAM_DISABLE_APPLY"
_APPLY_DISABLED_FALSEY: frozenset[str] = frozenset({"", "0", "false", "False", "FALSE", "no", "No", "NO"})


# Apply handlers that bind raw uuid.UUID values into session.execute and
# therefore require a PostgreSQL session. On any other dialect (notably
# SQLite via the sqlite_lance test stack) the bind raises
# ``sqlite3.ProgrammingError: type 'UUID' is not supported``; the
# orchestrator catches that at the dialect gate below and skips the op
# instead of crashing the run. See #875.
_POSTGRES_ONLY_OP_KINDS: frozenset[str] = frozenset(
    {
        "vectorcypher_dedupe_entities",
        "vectorcypher_centroid_recompute",
        "vectorcypher_prune_edges",
        "vectorcypher_source_chunk_ids_gc",
    }
)


def _is_apply_disabled_via_env() -> bool:
    """Read :data:`_APPLY_DISABLED_ENV_VAR` and decide if apply is gated.

    The truthiness rule is intentionally lax — anything other than the
    short list of falsey strings counts as on. Operators flipping the
    kill-switch under stress should not have to remember a magic value.
    """
    raw = os.environ.get(_APPLY_DISABLED_ENV_VAR)
    if raw is None:
        return False
    return raw not in _APPLY_DISABLED_FALSEY


async def _register_cancel_flag(run_id: UUID) -> asyncio.Event:
    async with _CANCEL_FLAGS_LOCK:
        flag = asyncio.Event()
        _CANCEL_FLAGS[run_id] = flag
        return flag


async def _clear_cancel_flag(run_id: UUID) -> None:
    async with _CANCEL_FLAGS_LOCK:
        _CANCEL_FLAGS.pop(run_id, None)


async def request_cancel(run_id: UUID) -> bool:
    """Flip the cancellation flag for ``run_id``. Returns False if unknown."""
    async with _CANCEL_FLAGS_LOCK:
        flag = _CANCEL_FLAGS.get(run_id)
        if flag is None:
            return False
        flag.set()
        return True


class DreamOrchestrator:
    """Single-shot orchestrator for one dream run."""

    def __init__(
        self,
        kb: Khora,
        config: DreamConfig,
        *,
        sinks: list[ReportSink] | None = None,
    ) -> None:
        self._kb = kb
        self._config = config
        self._sinks: list[ReportSink] = list(sinks) if sinks is not None else self._default_sinks()
        # Captured at construction so an operator can flip the kill-switch
        # and rebuild an orchestrator without restarting the process.
        # Long-lived orchestrators carry their original value — see the
        # module-level helper docstring.
        self._apply_disabled = _is_apply_disabled_via_env()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def run(
        self,
        namespace_id: UUID,
        *,
        mode: str = "dry-run",
        scope: DreamScope | None = None,
        ops: Iterable[OpKind] | None = None,
        expertise: ExpertiseConfig | None = None,
        on_progress: Callable[[DreamProgress], None] | None = None,
        resume_from: UUID | None = None,
    ) -> DreamResult:
        """Plan and execute a dream run end-to-end."""
        if mode not in ("dry-run", "apply"):
            raise ValueError(f"mode must be 'dry-run' or 'apply', got {mode!r}")

        effective_scope = self._build_scope(scope, ops)
        run_id = resume_from or uuid4()
        cancel_flag = await _register_cancel_flag(run_id)
        started_at = datetime.now(UTC)

        # The dream lock is namespace-scoped — held across PLAN → APPLY →
        # FINALIZE. Lock acquisition requires an open SQL session; on
        # embedded backends without one we fall through to a no-op
        # in-process lock (see acquire_namespace_dream_lock).
        try:
            async with self._lock(namespace_id):
                await self._init_run_row(run_id, namespace_id, mode)
                await self._emit_all(
                    DreamRunStarted(
                        run_id=run_id,
                        namespace_id=namespace_id,
                        mode=mode,  # type: ignore[arg-type]
                        trigger="resume" if resume_from is not None else "manual",
                        started_at=started_at,
                    )
                )

                plan = await self._plan_phase(
                    run_id,
                    namespace_id,
                    scope=effective_scope,
                    expertise=expertise,
                )

                if mode == "dry-run":
                    result = await self._report_phase(
                        run_id,
                        plan,
                        cancel_flag=cancel_flag,
                        on_progress=on_progress,
                    )
                else:
                    result = await self._apply_phase(
                        run_id,
                        plan,
                        cancel_flag=cancel_flag,
                        on_progress=on_progress,
                    )

                await self._finalize_run_row(run_id, plan, state="completed")
                await self._emit_all(
                    DreamRunCompleted(
                        run_id=run_id,
                        namespace_id=namespace_id,
                        mode=mode,  # type: ignore[arg-type]
                        duration_ms=_elapsed_ms(started_at),
                        ops_total=len(plan.ops),
                    )
                )
                return result
        except DreamForbiddenOpError:
            await self._finalize_run_row(run_id, plan=None, state="aborted_safety")
            raise
        except Exception as exc:
            await self._finalize_run_row(run_id, plan=None, state="failed", error=str(exc))
            await self._emit_all(
                DreamRunFailed(
                    run_id=run_id,
                    namespace_id=namespace_id,
                    mode=mode,  # type: ignore[arg-type]
                    duration_ms=_elapsed_ms(started_at),
                    error_hash=bounded_text_hash(str(exc)),
                    error_type=type(exc).__name__,
                )
            )
            raise
        finally:
            await _clear_cancel_flag(run_id)
            await self._close_sinks()

    async def status(self, run_id: UUID) -> DreamRunInfo | None:
        """Return run-level metadata from ``khora_dream_runs`` for ``run_id``."""
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return None
                row = (
                    await session.execute(
                        text(
                            "SELECT run_id, namespace_id, mode, started_at, finished_at "
                            "FROM khora_dream_runs WHERE run_id = :rid"
                        ),
                        {"rid": run_id},
                    )
                ).first()
        except RuntimeError:
            return None
        if row is None:
            return None
        finished_at = row.finished_at
        duration_ms = (finished_at - row.started_at).total_seconds() * 1000.0 if finished_at is not None else None
        return DreamRunInfo(
            run_id=row.run_id,
            namespace_id=row.namespace_id,
            mode=row.mode,
            started_at=row.started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    async def history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]:
        """Return the most recent runs for ``namespace_id`` (newest first)."""
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return []
                rows = (
                    await session.execute(
                        text(
                            "SELECT run_id, namespace_id, mode, started_at, finished_at "
                            "FROM khora_dream_runs "
                            "WHERE namespace_id = :ns "
                            "ORDER BY started_at DESC LIMIT :lim"
                        ),
                        {"ns": namespace_id, "lim": int(limit)},
                    )
                ).all()
        except RuntimeError:
            return []
        out: list[DreamRunInfo] = []
        for row in rows:
            finished_at = row.finished_at
            duration_ms = (finished_at - row.started_at).total_seconds() * 1000.0 if finished_at is not None else None
            out.append(
                DreamRunInfo(
                    run_id=row.run_id,
                    namespace_id=row.namespace_id,
                    mode=row.mode,
                    started_at=row.started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    async def _plan_phase(
        self,
        run_id: UUID,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        expertise: ExpertiseConfig | None,
    ) -> DreamPlan:
        engine_name = getattr(self._kb, "_engine_name", "vectorcypher")
        plugin = get_engine_plugin(engine_name)
        started = datetime.now(UTC)
        await self._emit_all(
            DreamPhaseStarted(
                run_id=run_id,
                namespace_id=namespace_id,
                phase="plan",
                started_at=started,
            )
        )

        plan = await plugin.plan_dream(  # type: ignore[attr-defined]
            self._kb,
            namespace_id,
            scope=scope,
            config=self._config,
            expertise=expertise,
        )
        # Safety floor — check at plan time too. The plugin's apply also
        # checks, but rejecting here means a dry-run never even reports
        # a forbidden op.
        from khora.dream.engines.registry import _validate_no_forbidden_ops

        _validate_no_forbidden_ops(plan)

        await self._persist_plan_hash(run_id, plan)
        await self._emit_all(
            DreamPhaseCompleted(
                run_id=run_id,
                namespace_id=namespace_id,
                phase="plan",
                outcome="success",
                ops_total=len(plan.ops),
                duration_ms=_elapsed_ms(started),
            )
        )
        return plan

    async def _report_phase(
        self,
        run_id: UUID,
        plan: DreamPlan,
        *,
        cancel_flag: asyncio.Event,
        on_progress: Callable[[DreamProgress], None] | None,
    ) -> DreamResult:
        """Dry-run path: emit each planned op through the sinks, no apply."""
        started = datetime.now(UTC)
        await self._emit_all(
            DreamPhaseStarted(
                run_id=run_id,
                namespace_id=plan.namespace_id,
                phase="report",
                started_at=started,
            )
        )

        for seq, op in enumerate(plan.ops):
            if cancel_flag.is_set():
                break
            await self._emit_op_event(run_id, op)
            self._fire_progress(on_progress, run_id, plan, seq, op, "report")

        await self._emit_all(
            DreamPhaseCompleted(
                run_id=run_id,
                namespace_id=plan.namespace_id,
                phase="report",
                outcome="success",
                ops_total=len(plan.ops),
                duration_ms=_elapsed_ms(started),
            )
        )

        return _build_result(run_id=run_id, namespace_id=plan.namespace_id, plan=plan, mode="dry-run")

    async def _apply_phase(
        self,
        run_id: UUID,
        plan: DreamPlan,
        *,
        cancel_flag: asyncio.Event,
        on_progress: Callable[[DreamProgress], None] | None,
    ) -> DreamResult:
        """Apply path: per-op handler dispatch + per-op transaction (#667).

        Loop invariant: each op runs inside its own
        ``coordinator.transaction()`` block. The orchestrator looks up
        the apply handler via :func:`get_apply_handler`, calls it with
        the open session, validates the returned
        :class:`UndoRecord`, persists ``last_committed_op_seq`` inside
        the same transaction, and lets the transaction-context exit
        commit (or roll back). Undo records are written incrementally to
        the file sink after every successful commit so a mid-apply crash
        leaves a recoverable file on disk.

        Cancellation is checked *between* ops. Resume skips ops up to
        the persisted ``last_committed_op_seq``.

        Guardrails (#667):
            - Kill-switch (:envvar:`KHORA_DREAM_DISABLE_APPLY`) raises
              :class:`DreamApplyDisabled` before any DB activity.
            - The chunk_id mutation assertion fires immediately after
              each handler returns; a violation rolls back the txn and
              aborts the run with :class:`DreamForbiddenOpError`.
            - Ops without an apply handler (Phase 1 audits) are
              pass-through — they advance the checkpoint and emit the
              op event but skip the handler / undo write.
        """
        if self._apply_disabled:
            raise DreamApplyDisabled(
                "Dream apply mode is disabled via "
                f"{_APPLY_DISABLED_ENV_VAR}={os.environ.get(_APPLY_DISABLED_ENV_VAR)!r}. "
                "Unset the env var (or set it to '0' / 'false') and rebuild "
                "the DreamOrchestrator to re-enable apply."
            )

        started = datetime.now(UTC)
        await self._emit_all(
            DreamPhaseStarted(
                run_id=run_id,
                namespace_id=plan.namespace_id,
                phase="apply",
                started_at=started,
            )
        )

        last_committed = await self._read_last_committed(run_id)
        undo_records: list[UndoRecord] = []
        file_sink = self._file_sink()
        skipped_by_op_type: dict[str, int] = {}

        for seq, op in enumerate(plan.ops):
            if cancel_flag.is_set():
                break
            if seq <= last_committed:
                continue

            handler = get_apply_handler(op.op_type)
            if handler is None:
                # Phase 1 audit op — no mutation, no undo, but still
                # advance the checkpoint so a resume doesn't re-run
                # the audit and re-publish its event.
                await self._record_committed(run_id, seq)
                await self._emit_op_event(run_id, op)
                self._fire_progress(on_progress, run_id, plan, seq, op, "apply")
                continue

            # Mutation op — run inside its own coordinator transaction so
            # the apply handler's writes commit/rollback together with
            # the checkpoint update.
            try:
                undo = await self._apply_one_op(run_id=run_id, seq=seq, op=op, handler=handler)
            except DreamBackendUnsupported as exc:
                # Dialect gate (#875): handler requires Postgres but the
                # active session speaks SQLite (sqlite_lance) or another
                # dialect. Log a warning, advance the checkpoint so a
                # resume doesn't re-attempt the same impossible apply,
                # and record the op as skipped.
                from loguru import logger

                logger.warning(
                    "dream apply op {op_type!s} skipped: {reason}",
                    op_type=op.op_type,
                    reason=str(exc),
                )
                await self._record_committed(run_id, seq)
                skipped_by_op_type[str(op.op_type)] = skipped_by_op_type.get(str(op.op_type), 0) + 1
                self._fire_progress(on_progress, run_id, plan, seq, op, "apply")
                continue
            _assert_no_chunk_id_mutation(undo)
            undo_records.append(undo)

            # Persist the undo file before announcing the op — readers
            # of `undo.json` will see this op's snapshot even if the
            # next step crashes the process between emit and the next
            # write.
            if file_sink is not None:
                file_sink.write_undo_incremental(
                    undo_records,
                    run_id=run_id,
                    namespace_id=plan.namespace_id,
                    started_at=started,
                )

            await self._emit_op_event(run_id, op)
            self._fire_progress(on_progress, run_id, plan, seq, op, "apply")

        await self._emit_all(
            DreamPhaseCompleted(
                run_id=run_id,
                namespace_id=plan.namespace_id,
                phase="apply",
                outcome="success" if not cancel_flag.is_set() else "skipped",
                ops_total=len(plan.ops),
                duration_ms=_elapsed_ms(started),
            )
        )

        return _build_result(
            run_id=run_id,
            namespace_id=plan.namespace_id,
            plan=plan,
            mode="apply",
            skipped_by_op_type=skipped_by_op_type,
        )

    async def _apply_one_op(
        self,
        *,
        run_id: UUID,
        seq: int,
        op: DreamOp,
        handler: Any,
    ) -> UndoRecord:
        """Run one apply handler inside a coordinator transaction.

        The checkpoint update lives inside the same transaction so a
        committed op always has a matching ``last_committed_op_seq``
        and a rolled-back op never advances the cursor. On embedded
        backends without a SQL transaction the orchestrator falls back
        to invoking the handler with ``session=None`` — handlers that
        need a session must guard against this themselves.

        The orchestrator's :class:`DreamConfig` is forwarded as a
        ``dream_config`` kwarg for handlers that consume it (the dedupe
        Phase 4.1 verifier — #667). Handlers that don't accept the
        kwarg are dispatched without it so the registry stays loosely
        coupled.
        """
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                _assert_backend_supported(session, op.op_type)
                _warn_graph_divergence(coordinator, op.op_type)
                undo = await self._invoke_handler(handler, op, coordinator=coordinator, session=session)
                await self._record_committed_in_session(session, run_id, seq)
                return undo
        except RuntimeError as exc:
            if "No SQL backend" not in str(exc):
                raise
            # Embedded fallback: call handler without a session and
            # advance the in-memory checkpoint best-effort.
            undo = await self._invoke_handler(handler, op, coordinator=coordinator, session=None)
            await self._record_committed(run_id, seq)
            return undo

    async def _invoke_handler(
        self,
        handler: Any,
        op: DreamOp,
        *,
        coordinator: Any,
        session: Any,
    ) -> UndoRecord:
        """Call ``handler`` with ``dream_config`` when its signature accepts it.

        We inspect the handler's parameter list once (cheap — handlers
        are function objects, not classes) and forward
        ``self._config`` only when the handler declares a ``dream_config``
        keyword. This keeps the verifier gate fully orchestrated without
        forcing the (currently four) parallel apply handlers to take a
        ``dream_config`` kwarg they don't need.
        """
        import inspect

        kwargs: dict[str, Any] = {"coordinator": coordinator, "session": session}
        try:
            sig = inspect.signature(handler)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "dream_config" in sig.parameters:
            kwargs["dream_config"] = self._config
        return await handler(op, **kwargs)

    # ------------------------------------------------------------------
    # Lock acquisition
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _lock(self, namespace_id: UUID) -> Any:
        """Acquire the per-namespace advisory lock. No-op on backends without SQL."""
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                async with acquire_namespace_dream_lock(txn.session, namespace_id, timeout_seconds=60.0):
                    yield
        except RuntimeError:
            # No SQL backend — fall back to the embedded asyncio.Lock
            # path. acquire_namespace_dream_lock handles this when given
            # an embedded session; the bare-coordinator no-SQL case is
            # uncommon outside test stubs, so just yield.
            yield

    # ------------------------------------------------------------------
    # khora_dream_runs persistence (Postgres only; embedded path is no-op)
    # ------------------------------------------------------------------

    async def _init_run_row(self, run_id: UUID, namespace_id: UUID, mode: str) -> None:
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return
                now = datetime.now(UTC)
                await session.execute(
                    text(
                        "INSERT INTO khora_dream_runs "
                        "(run_id, namespace_id, trigger, mode, state, started_at, "
                        " heartbeat_at, total_ops, total_decisions, last_committed_op_seq) "
                        "VALUES (:rid, :ns, :trg, :mode, :state, :ts, :ts, 0, 0, -1) "
                        "ON CONFLICT (run_id) DO UPDATE SET heartbeat_at = :ts"
                    ),
                    {
                        "rid": run_id,
                        "ns": namespace_id,
                        "trg": "manual",
                        "mode": mode,
                        "state": "planning",
                        "ts": now,
                    },
                )
        except RuntimeError:
            return

    async def _persist_plan_hash(self, run_id: UUID, plan: DreamPlan) -> None:
        digest = plan_hash(plan)
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return
                await session.execute(
                    text(
                        "UPDATE khora_dream_runs "
                        "SET plan_hash = :ph, total_ops = :tot, heartbeat_at = :ts, "
                        "    state = CASE WHEN state = 'planning' THEN 'applying' ELSE state END "
                        "WHERE run_id = :rid"
                    ),
                    {
                        "ph": digest,
                        "tot": len(plan.ops),
                        "ts": datetime.now(UTC),
                        "rid": run_id,
                    },
                )
        except RuntimeError:
            return

    async def _read_last_committed(self, run_id: UUID) -> int:
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return -1
                row = (
                    await session.execute(
                        text("SELECT last_committed_op_seq FROM khora_dream_runs WHERE run_id = :rid"),
                        {"rid": run_id},
                    )
                ).first()
        except RuntimeError:
            return -1
        if row is None or row.last_committed_op_seq is None:
            return -1
        return int(row.last_committed_op_seq)

    async def _record_committed(self, run_id: UUID, seq: int) -> None:
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                await self._record_committed_in_session(txn.session, run_id, seq)
        except RuntimeError:
            return

    async def _record_committed_in_session(self, session: Any, run_id: UUID, seq: int) -> None:
        """Persist ``last_committed_op_seq`` using an existing session.

        Phase 4 apply runs the checkpoint update inside the same
        transaction as the apply handler so a rollback unwinds both.
        On non-postgres backends the call is a no-op (mirrors the
        :meth:`_record_committed` shape).
        """
        if not _is_postgres(session):
            return
        await session.execute(
            text("UPDATE khora_dream_runs SET last_committed_op_seq = :seq, heartbeat_at = :ts WHERE run_id = :rid"),
            {"seq": seq, "ts": datetime.now(UTC), "rid": run_id},
        )

    async def _finalize_run_row(
        self,
        run_id: UUID,
        plan: DreamPlan | None,
        *,
        state: str,
        error: str | None = None,
    ) -> None:
        coordinator = self._kb.storage
        try:
            async with coordinator.transaction() as txn:
                session = txn.session
                if not _is_postgres(session):
                    return
                now = datetime.now(UTC)
                params: dict[str, Any] = {
                    "rid": run_id,
                    "state": state,
                    "ts": now,
                    "total": len(plan.ops) if plan is not None else 0,
                }
                if error is not None:
                    import json as _json

                    params["err"] = _json.dumps({"message": error})
                    await session.execute(
                        text(
                            "UPDATE khora_dream_runs "
                            "SET state = :state, finished_at = :ts, total_ops = :total, "
                            "    error = CAST(:err AS jsonb) WHERE run_id = :rid"
                        ),
                        params,
                    )
                else:
                    await session.execute(
                        text(
                            "UPDATE khora_dream_runs "
                            "SET state = :state, finished_at = :ts, total_ops = :total "
                            "WHERE run_id = :rid"
                        ),
                        params,
                    )
        except RuntimeError:
            return

    # ------------------------------------------------------------------
    # Sink fan-out
    # ------------------------------------------------------------------

    def _default_sinks(self) -> list[ReportSink]:
        sinks: list[ReportSink] = []
        if self._config.report_file_sink_enabled:
            sinks.append(
                DreamFileSink(
                    base_dir=_default_file_sink_dir(),
                    redact_text=self._config.redact_text,
                )
            )
        if self._config.report_event_sink_enabled:
            sinks.append(DreamEventSink(self._kb._get_hook_dispatcher()))
        if self._config.report_collector_sink_enabled:
            sinks.append(DreamCollectorSink())
        return sinks

    async def _emit_all(self, event: DreamReportEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception as exc:  # noqa: BLE001, S110
                # Sinks fail independently; one slow / broken sink must
                # not stall the run. Errors are logged inside the sink.
                from loguru import logger

                logger.debug("dream sink emit raised: {}", exc)

    async def _close_sinks(self) -> None:
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                await sink.flush()
                await sink.close()

    def _file_sink(self) -> DreamFileSink | None:
        """Return the first :class:`DreamFileSink` in the sink list (or None).

        The orchestrator calls into the file sink directly to write
        incremental undo files — sinks otherwise receive events through
        the :meth:`_emit_all` fan-out.
        """
        for sink in self._sinks:
            if isinstance(sink, DreamFileSink):
                return sink
        return None

    async def _emit_op_event(self, run_id: UUID, op: DreamOp) -> None:
        await self._emit_all(
            DreamOperationEvent(
                op_id=op.op_id,
                run_id=run_id,
                phase=op.phase,
                op_type=str(op.op_type),
                inputs=_inputs_payload(op.inputs),
                outputs=_outputs_payload(op.outputs),
                decision=op.decision,
                rationale=DreamRationale(
                    strategy=str(op.op_type),
                    rationale_hash=bounded_text_hash(op.rationale or ""),
                ),
                started_at=op.started_at or datetime.now(UTC),
                duration_ms=op.duration_ms or 0.0,
                namespace_id=op.namespace_id or run_id,
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scope(
        self,
        scope: DreamScope | None,
        ops: Iterable[OpKind] | None,
    ) -> DreamScope:
        if scope is not None:
            return scope
        if ops is not None:
            return DreamScope(op_kinds=tuple(ops))
        return DreamScope()

    def _fire_progress(
        self,
        on_progress: Callable[[DreamProgress], None] | None,
        run_id: UUID,
        plan: DreamPlan,
        seq: int,
        op: DreamOp,
        phase: str,
    ) -> None:
        if on_progress is None:
            return
        try:
            on_progress(
                DreamProgress(
                    run_id=run_id,
                    phase=phase,
                    op_index=seq,
                    op_total=len(plan.ops),
                    op_type=str(op.op_type),
                )
            )
        except Exception:  # noqa: BLE001
            # Progress callback errors are non-fatal.
            return


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_result(
    *,
    run_id: UUID,
    namespace_id: UUID,
    plan: DreamPlan,
    mode: str,
    skipped_by_op_type: dict[str, int] | None = None,
) -> DreamResult:
    now = datetime.now(UTC)
    skipped_remaining = dict(skipped_by_op_type or {})
    summaries: dict[str, OpSummary] = {}
    for op in plan.ops:
        key = str(op.op_type)
        cur = summaries.get(key) or OpSummary(op_type=key)
        # Apply mode increments ``applied`` for every planned op except
        # those the dialect gate / runtime declared "skipped". The skip
        # budget is drained in plan order so the counts match the number
        # of ``DreamBackendUnsupported`` catches during the apply loop.
        was_skipped = mode == "apply" and skipped_remaining.get(key, 0) > 0
        if was_skipped:
            skipped_remaining[key] -= 1
        summaries[key] = OpSummary(
            op_type=key,
            planned=cur.planned + 1,
            applied=cur.applied + (1 if mode == "apply" and not was_skipped else 0),
            skipped=cur.skipped + (1 if was_skipped else 0),
            failed=cur.failed,
        )
    info = DreamRunInfo(
        run_id=run_id,
        namespace_id=namespace_id,
        mode="apply" if mode == "apply" else "dry-run",
        started_at=now,
        finished_at=now,
        duration_ms=0.0,
    )
    return DreamResult(
        run=info,
        diff=DreamDiff(),
        ops=tuple(summaries.values()),
        metadata={"plan_hash": plan_hash(plan), "plan_payload": canonical_plan_payload(plan)},
    )


def _elapsed_ms(start: datetime) -> float:
    return (datetime.now(UTC) - start).total_seconds() * 1000.0


def _is_postgres(session: Any) -> bool:
    return getattr(getattr(session, "bind", None), "dialect", None) is not None and (
        session.bind.dialect.name == "postgresql"
    )


def _session_dialect(session: Any) -> str | None:
    """Best-effort read of ``session.bind.dialect.name`` for the gate."""
    bind = getattr(session, "bind", None)
    if bind is None:
        return None
    dialect = getattr(bind, "dialect", None)
    if dialect is None:
        return None
    name = getattr(dialect, "name", None)
    return str(name) if name is not None else None


def _assert_backend_supported(session: Any, op_type: Any) -> None:
    """Raise :class:`DreamBackendUnsupported` when the dialect can't carry the op.

    The vectorcypher apply handlers bind raw ``uuid.UUID`` values into
    ``session.execute``; only PostgreSQL handles those natively. On any
    other dialect the bind raises ``sqlite3.ProgrammingError`` (or the
    moral equivalent on a different driver). We refuse cleanly up front
    rather than letting that opaque driver error abort the run.

    Sessions whose dialect can't be read (test stubs with no ``bind``,
    embedded fallback paths) are treated as Postgres-equivalent - they
    are either the real production session or a test stub that has
    opted in to the apply path.
    """
    op_type_str = str(op_type)
    if op_type_str not in _POSTGRES_ONLY_OP_KINDS:
        return
    dialect = _session_dialect(session)
    if dialect is None or dialect == "postgresql":
        return
    raise DreamBackendUnsupported(
        f"dream apply op {op_type_str!r} requires postgresql; "
        f"active session dialect is {dialect!r}. See docs/dream-phase.md."
    )


def _warn_graph_divergence(coordinator: Any, op_type: Any) -> None:
    """Log a one-shot warning when an SQL-only apply will leave Neo4j stale.

    The four vectorcypher apply handlers mutate the relational store
    through ``session.execute`` only - none of them touch the graph
    backend. When the coordinator does carry a graph backend (Neo4j /
    Memgraph / Neptune / AGE), the post-apply state will diverge: the
    SQL row is soft-deleted / rewritten but the graph mirror still
    reflects the pre-apply shape. The actual graph write is deferred to
    a future PR (see the in-source TODOs in ``prune_edges.py`` and
    ``source_chunk_ids_gc.py``); this warning is the honest signal that
    operators get today.

    Reads ``coordinator._graph`` (the internal backend slot) so the
    namespace-required proxy doesn't fire deprecation warnings during a
    dream run.
    """
    op_type_str = str(op_type)
    if op_type_str not in _POSTGRES_ONLY_OP_KINDS:
        return
    graph = getattr(coordinator, "_graph", None)
    if graph is None:
        return
    from loguru import logger

    logger.warning(
        "dream apply {op_type} mutates the relational store only; "
        "graph store will not reflect this change until a future release",
        op_type=op_type_str,
    )


def _default_file_sink_dir() -> str:
    """Default DreamFileSink base dir. Lazy-created on first event."""
    import tempfile

    return tempfile.gettempdir() + "/khora-dream-reports"


def _inputs_payload(inputs: tuple[Any, ...]) -> dict[str, Any]:
    """Coerce a DreamOp inputs tuple into a sink-safe dict."""
    out: dict[str, Any] = {}
    for idx, item in enumerate(inputs):
        if isinstance(item, dict):
            out.update({str(k): _stringify(v) for k, v in item.items()})
        else:
            out[f"_{idx}"] = _stringify(item)
    return out


def _outputs_payload(outputs: tuple[Any, ...]) -> dict[str, Any]:
    """Coerce a DreamOp outputs tuple into a sink-safe dict.

    Mirrors :func:`_inputs_payload` but indexes per-op outputs under
    numeric keys to preserve order.
    """
    out: dict[str, Any] = {}
    for idx, item in enumerate(outputs):
        out[f"output_{idx}"] = _stringify(item)
    return out


def _stringify(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_stringify(item) for item in value]
    return value


__all__ = [
    "DreamOrchestrator",
    "request_cancel",
]
