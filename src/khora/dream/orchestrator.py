"""Dream-phase orchestrator state machine (#661).

Drives a single dream run through ``INIT → PLAN → REPORT|APPLY →
FINALIZE``. Acquires the per-namespace advisory lock for the entire
plan-through-finalize block (#677), dispatches plan-stage discovery to
the engine-registered :class:`DreamCapable` plugin, fans every plan op
out through the configured sinks (#678), and persists run state to
``khora_dream_runs`` via a backend-portable :class:`DreamRunStore` -
PostgreSQL, a SQLite sidecar, or a SurrealDB-relational table (#1274).

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
from khora.dream.graph_mirror import (
    GRAPH_MIRROR_PARTIAL_FAILURE_COUNTER,
    MIRRORABLE_OP_KINDS,
    apply_mirror_payload,
    apply_mirror_targets,
    extract_mirror_targets,
    mirror_payload,
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
from khora.dream.runstore import DreamRunStore, GraphMirrorPending, select_run_store
from khora.dream.safety import _assert_no_chunk_id_mutation
from khora.telemetry import bounded_text_hash
from khora.telemetry.metrics import metric_counter

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
#
# ``vectorcypher_normalize_schema`` (#1264) belongs here too: its apply
# handler rewrites ``entities.entity_type`` / ``relationships.relationship_type``
# via ``session.execute`` binding raw ``uuid.UUID`` row ids, so it needs
# Postgres for the same reason. Its graph-label relabel mirror is deferred
# (out of scope for #1272, which mirrors the prune / dedupe soft-deletes).
_POSTGRES_ONLY_OP_KINDS: frozenset[str] = frozenset(
    {
        "vectorcypher_dedupe_entities",
        "vectorcypher_centroid_recompute",
        "vectorcypher_prune_edges",
        "vectorcypher_source_chunk_ids_gc",
        "vectorcypher_normalize_schema",
    }
)


# Op kinds that issue LLM calls during apply. The dream LLM token budget
# (#1270) is checked before dispatching any of these; non-LLM mutation
# ops are never gated. ``community_summary`` is the only reachable
# LLM-using op today; future LLM ops add their kind here.
_LLM_OP_KINDS: frozenset[str] = frozenset(
    {
        "vectorcypher_community_summary",
    }
)


# Number of seconds in the rolling "per day" window for the
# per-namespace token budget (#1270).
_LLM_BUDGET_DAY_SECONDS = 86_400.0


# Emitted once per LLM-using op skipped by the token budget. No labels -
# the namespace_id cardinality rule forbids a per-tenant label.
_THROTTLE_COUNTER = metric_counter(
    "khora.dream.llm.throttled_total",
    description="Dream LLM-using ops skipped because a token budget was exhausted.",
)


class _NamespaceDayBudget:
    """Process-global rolling-day token bucket for one namespace (#1270).

    Mirrors the hooks Level-2 rolling-hour bucket shape: a monotonic
    window start plus a running token count. The window resets lazily on
    read once :data:`_LLM_BUDGET_DAY_SECONDS` have elapsed.
    """

    __slots__ = ("tokens_used", "window_started_at")

    def __init__(self) -> None:
        self.tokens_used = 0
        self.window_started_at = time_monotonic()


# Keyed by namespace_id. Persists across DreamOrchestrator instances /
# runs in the same process so the per-day budget spans runs.
_NAMESPACE_DAY_BUDGETS: dict[UUID, _NamespaceDayBudget] = {}


def time_monotonic() -> float:
    """Indirection over ``time.monotonic`` so tests can freeze the clock."""
    import time

    return time.monotonic()


def _reset_namespace_llm_budgets() -> None:
    """Clear the process-global per-namespace-per-day buckets.

    Test-only escape hatch - production code never resets the buckets
    (they self-expire on the rolling window).
    """
    _NAMESPACE_DAY_BUDGETS.clear()


def _get_namespace_day_budget(namespace_id: UUID) -> _NamespaceDayBudget:
    """Return the rolling-day bucket for ``namespace_id``, resetting it
    when the window has rolled over."""
    bucket = _NAMESPACE_DAY_BUDGETS.get(namespace_id)
    now = time_monotonic()
    if bucket is None:
        bucket = _NamespaceDayBudget()
        _NAMESPACE_DAY_BUDGETS[namespace_id] = bucket
        return bucket
    if now - bucket.window_started_at >= _LLM_BUDGET_DAY_SECONDS:
        bucket.tokens_used = 0
        bucket.window_started_at = now
    return bucket


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
        # Per-run LLM token spend, accumulated from the context-local usage
        # path each LLM-using op records into (#1270). Reset per run.
        self._llm_tokens_this_run = 0
        # Run-state store, resolved lazily from the active stack (#1274):
        # PG / SQLite-sidecar / SurrealDB-relational. ``None`` means no
        # run-state backend is reachable (graph-only embedded stub).
        self._run_store_cache: DreamRunStore | None = None
        self._run_store_resolved = False

    # ------------------------------------------------------------------
    # Run-state store (#1274)
    # ------------------------------------------------------------------

    def _run_store(self) -> DreamRunStore | None:
        """Return the run-state store for the active stack (cached)."""
        if not self._run_store_resolved:
            self._run_store_cache = select_run_store(self._kb.storage)
            self._run_store_resolved = True
        return self._run_store_cache

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
                await self._init_run_row(run_id, namespace_id, mode, is_resume=resume_from is not None)
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
        store = self._run_store()
        if store is None:
            return None
        return await store.status(run_id)

    async def history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]:
        """Return the most recent runs for ``namespace_id`` (newest first)."""
        store = self._run_store()
        if store is None:
            return []
        return await store.history(namespace_id, limit=limit)

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
        budget_skip_reasons: list[dict[str, Any]] = []
        mirror_degradations: list[dict[str, Any]] = []

        # Reconciler drain (#1272): re-attempt any committed-but-unmirrored ops
        # left by a crash between the PG commit and the graph mirror. The
        # checkpoint advances INSIDE the PG commit, so resume alone skips those
        # ops; this drain is the only path that heals them.
        mirror_degradations.extend(await self._drain_graph_mirror_pending(run_id, plan.namespace_id))

        # Reset the per-run token counter and open a context-local usage
        # accumulator so each LLM-using op's spend (recorded through
        # ``record_usage`` inside ``acompletion``) can be read back and
        # checked against the budgets (#1270).
        self._llm_tokens_this_run = 0
        from khora.telemetry.context import start_usage_collection

        start_usage_collection()

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

            # LLM token budget (#1270): for LLM-using ops, refuse to
            # dispatch once a per-run or per-namespace-per-day budget is
            # exhausted. Already-applied ops stay committed; the skipped
            # op does NOT advance the checkpoint so a later run (or a
            # fresh rolling-day window) can retry it.
            breach = self._llm_budget_breach(op, plan.namespace_id)
            if breach is not None:
                _THROTTLE_COUNTER.add(1)
                skipped_by_op_type[str(op.op_type)] = skipped_by_op_type.get(str(op.op_type), 0) + 1
                budget_skip_reasons.append(
                    {
                        "op_kind": str(op.op_type),
                        "reason": "llm_budget_exhausted",
                        "detail": breach,
                    }
                )
                from loguru import logger

                logger.warning(
                    "dream apply op {op_type!s} skipped: LLM token budget exhausted ({detail})",
                    op_type=op.op_type,
                    detail=breach,
                )
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

            # Post-commit Neo4j tombstone-mirror (#1272). The PG apply +
            # checkpoint are durable now; mirror the soft-deletes to the graph
            # OUTSIDE the tx. A failure here leaves PG ahead of the graph - it
            # is queued in graph_mirror_pending and re-attempted by the
            # reconciler on the next run, never rolling back the PG commit.
            degradation = await self._mirror_dream_op(run_id, seq, plan.namespace_id, op, undo)
            if degradation is not None:
                mirror_degradations.append(degradation)

            # Roll the LLM token spend this op just incurred into both
            # budget buckets (#1270). Non-LLM ops record nothing, so the
            # drain returns 0 and the buckets are untouched.
            self._drain_llm_usage(plan.namespace_id)

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

        # Drain and discard any usage left in the context accumulator so
        # we don't leak the contextvar into a later call on this loop.
        from khora.telemetry.context import collect_usage

        collect_usage()

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
            extra_skip_reasons=budget_skip_reasons,
            extra_degradations=mirror_degradations,
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
                undo = await self._invoke_handler(handler, op, coordinator=coordinator, session=session)
                await self._record_committed_in_session(session, run_id, seq)
                # Queue the graph mirror durably WITH the checkpoint (#1292): a
                # hard crash between this commit and the post-commit mirror then
                # leaves a pending row the reconciler can drain. Cleared by
                # ``_mirror_dream_op`` on a successful mirror.
                await self._pre_mark_graph_mirror_pending(session, run_id, seq, op, undo)
                return undo
        except RuntimeError as exc:
            if "No SQL backend" not in str(exc):
                raise
            # Embedded fallback: call handler without a session and
            # advance the in-memory checkpoint best-effort.
            undo = await self._invoke_handler(handler, op, coordinator=coordinator, session=None)
            await self._record_committed(run_id, seq)
            await self._pre_mark_graph_mirror_pending(None, run_id, seq, op, undo)
            return undo

    async def _mirror_dream_op(
        self,
        run_id: UUID,
        seq: int,
        namespace_id: UUID,
        op: DreamOp,
        undo: UndoRecord,
    ) -> dict[str, Any] | None:
        """Mirror one just-committed apply op's soft-deletes to the graph (#1272).

        Runs AFTER the op's PG transaction committed (eventual consistency).
        Reads the just-committed ``UndoRecord`` (source of truth for what PG
        accepted) and translates the soft-deletes onto the graph ``valid_until``
        via the #1271 capability-gated verbs. Idempotent by-id.

        Gating (ADR-001):

          - No graph backend -> nothing to mirror, returns ``None``.
          - The op kind is not in the backend's ``supports_dream_mirror()``
            probe, or not a soft-delete shape this PR mirrors -> returns a
            structured ``SkipReason`` dict so the divergence surfaces on the
            result (#1292), not just a log.
          - The mirror raises after the PG commit (or the namespace resolve
            fails) -> increments the partial-failure counter, queues the op in
            ``graph_mirror_pending`` for the reconciler, and returns a
            ``Degradation`` dict.
        """
        graph = _graph_backend(self._kb.storage)
        if graph is None:
            return None

        op_type = str(op.op_type)
        supported = _supported_mirror_kinds(graph)
        if op_type not in supported or op_type not in MIRRORABLE_OP_KINDS:
            # The backend can't (or this PR doesn't) mirror this op kind. Surface
            # the skip on the result so the divergence is accounted for (ADR-001).
            return self._record_mirror_skip(op, reason="unsupported_op_kind")

        targets = extract_mirror_targets(op_type, undo)
        if not targets["retire_entity_ids"] and not targets["invalidate_relationship_ids"]:
            # No-op apply (already pruned / verifier-rejected merge): nothing to
            # mirror, and nothing to queue. Clean convergence.
            return None

        stamp_at = undo.applied_at or datetime.now(UTC)
        try:
            # The graph nodes/edges carry the row-level namespace id (ingest
            # resolves the stable id before any write); resolve here, INSIDE the
            # failure-handling path, so the #1271 verbs' ``WHERE namespace_id =
            # $namespace_id`` matches. A resolver error must NOT fall back to the
            # stable id (it would match zero graph rows yet report success) -
            # let it queue + degrade like any other mirror failure (#1292).
            row_namespace_id = await self._resolve_namespace_for_mirror(namespace_id)
            await apply_mirror_targets(graph, targets, namespace_id=row_namespace_id, stamp_at=stamp_at)
        except Exception as exc:
            GRAPH_MIRROR_PARTIAL_FAILURE_COUNTER.add(1)
            # Queue for the reconciler so a later run re-mirrors this exact
            # committed-but-unmirrored op. The apply loop already wrote a durable
            # pending row inside the PG commit (#1292); this re-mark is idempotent
            # (keyed on op_seq) and covers direct callers that skip the pre-mark.
            await self._queue_graph_mirror_pending(run_id, seq, op, undo)
            from loguru import logger

            logger.warning(
                "dream graph mirror failed for op {op_type} (PG committed, graph queued for reconcile): {exc}",
                op_type=op_type,
                exc=exc,
                exc_info=True,
            )
            return {
                "component": "dream.graph_mirror",
                "reason": "graph_mirror_failed_after_pg_commit",
                "detail": f"op_type={op_type} op_id={op.op_id}",
                "exception": type(exc).__name__,
            }
        # Mirror succeeded: clear the durable pending row written before the
        # mirror attempt (#1292) so the reconciler does not replay it.
        await self._clear_graph_mirror_pending(run_id, seq)
        return None

    async def _resolve_namespace_for_mirror(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace id to the row id the graph rows carry.

        ``resolve_namespace`` is idempotent (returns a row id unchanged), so a
        coordinator stub that already passes a row id is unaffected. A coordinator
        that does not expose the method returns the input unchanged.

        A resolver *error* propagates (#1292): silently falling back to the stable
        id makes the mirror match zero graph rows (they carry the row id) yet
        report success - silent cross-store divergence. The caller runs this
        inside the mirror failure-handling path so a resolver error queues +
        degrades like any other mirror failure.
        """
        resolver = getattr(self._kb.storage, "resolve_namespace", None)
        if resolver is None:
            return namespace_id
        return await resolver(namespace_id)

    async def _pre_mark_graph_mirror_pending(
        self, session: Any, run_id: UUID, seq: int, op: DreamOp, undo: UndoRecord
    ) -> None:
        """Durably queue a mirrorable op BEFORE the mirror runs (#1292).

        Called inside the apply op's PG transaction (same ``session`` as the
        checkpoint advance) so a hard crash between the PG commit and the graph
        mirror still leaves a pending row the reconciler can drain. Only marks
        ops the active graph backend can mirror AND that carry targets - a
        no-op apply or an unmirrorable kind queues nothing. ``_mirror_dream_op``
        clears the row on a successful mirror.
        """
        store = self._run_store()
        if store is None:
            return
        graph = _graph_backend(self._kb.storage)
        if graph is None:
            return
        op_type = str(op.op_type)
        if op_type not in _supported_mirror_kinds(graph) or op_type not in MIRRORABLE_OP_KINDS:
            return
        targets = extract_mirror_targets(op_type, undo)
        if not targets["retire_entity_ids"] and not targets["invalidate_relationship_ids"]:
            return
        await store.mark_graph_mirror_pending(
            run_id,
            GraphMirrorPending(
                op_seq=seq,
                op_id=op.op_id,
                op_type=op_type,
                payload=mirror_payload(op, undo),
            ),
            session=session,
        )

    async def _queue_graph_mirror_pending(self, run_id: UUID, seq: int, op: DreamOp, undo: UndoRecord) -> None:
        """Persist a committed-but-unmirrored op for the reconciler to retry."""
        store = self._run_store()
        if store is None:
            return
        await store.mark_graph_mirror_pending(
            run_id,
            GraphMirrorPending(
                op_seq=seq,
                op_id=op.op_id,
                op_type=str(op.op_type),
                payload=mirror_payload(op, undo),
            ),
        )

    async def _clear_graph_mirror_pending(self, run_id: UUID, seq: int) -> None:
        """Drop the durable pending row once its mirror succeeded (#1292)."""
        store = self._run_store()
        if store is None:
            return
        await store.clear_graph_mirror_pending(run_id, seq)

    def _record_mirror_skip(self, op: DreamOp, *, reason: str) -> dict[str, Any]:
        """Return a structured skip when an op kind cannot be mirrored (ADR-001).

        Since #1272 made the PG read filter unconditional, a backend that does
        not advertise an op kind has PG hiding rows while graph recall still
        shows them - silent divergence. Surfacing the skip on the result lets
        operators see it (#1292), not just a log line.
        """
        from loguru import logger

        logger.info(
            "dream graph mirror skipped op {op_type}: {reason} (graph backend does not advertise this op kind)",
            op_type=str(op.op_type),
            reason=reason,
        )
        return {
            "component": "dream.graph_mirror",
            "reason": f"graph_mirror_{reason}",
            "detail": f"op_type={op.op_type!s} op_id={op.op_id}",
        }

    async def _drain_graph_mirror_pending(self, run_id: UUID, namespace_id: UUID) -> list[dict[str, Any]]:
        """Reconcile committed-but-unmirrored ops left by a prior crash (#1272, #1292).

        The checkpoint advances inside the PG commit, BEFORE the mirror runs, so
        a crash in that window leaves a committed op the resume loop skips. This
        drain re-attempts each queued op (idempotent by-id) and clears it on
        success. It reads ALL open pending ops for the NAMESPACE (not just the
        current ``run_id``, #1292) so a later run with a fresh ``run_id`` heals a
        prior run's failures. A still-failing op stays queued and surfaces a
        fresh degradation. Returns the degradations for ops that could not be
        drained (including a failed pending read, per ADR-001).

        ``run_id`` is unused for the read now but kept in the signature so the
        apply loop's call site is stable and the drain can log the draining run.
        """
        del run_id  # the drain is namespace-scoped now (#1292)
        store = self._run_store()
        graph = _graph_backend(self._kb.storage)
        if store is None or graph is None:
            return []

        from loguru import logger

        try:
            pending = await store.get_open_graph_mirror_pending(namespace_id)
        except Exception as exc:
            # A failed pending read hides committed mirror lag - record it rather
            # than silently returning empty (#1292, ADR-001).
            GRAPH_MIRROR_PARTIAL_FAILURE_COUNTER.add(1)
            logger.warning(
                "dream graph mirror reconcile: pending read failed (mirror lag may be hidden): {exc}",
                exc=exc,
                exc_info=True,
            )
            return [
                {
                    "component": "dream.graph_mirror.reconcile",
                    "reason": "graph_mirror_pending_read_failed",
                    "detail": "get_open_graph_mirror_pending raised",
                    "exception": type(exc).__name__,
                }
            ]
        if not pending:
            return []

        degradations: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for pending_run_id, entry in pending:
            try:
                # Resolve per-entry inside the try so a resolver error degrades
                # the entry (stays queued) rather than aborting the apply run or
                # silently matching zero graph rows (#1292).
                row_namespace_id = await self._resolve_namespace_for_mirror(namespace_id)
                await apply_mirror_payload(graph, entry.payload, namespace_id=row_namespace_id, fallback_stamp=now)
            except Exception as exc:
                GRAPH_MIRROR_PARTIAL_FAILURE_COUNTER.add(1)
                logger.warning(
                    "dream graph mirror reconcile failed for op {op_type} (still queued): {exc}",
                    op_type=entry.op_type,
                    exc=exc,
                    exc_info=True,
                )
                degradations.append(
                    {
                        "component": "dream.graph_mirror.reconcile",
                        "reason": "graph_mirror_reconcile_failed",
                        "detail": f"op_type={entry.op_type} op_id={entry.op_id}",
                        "exception": type(exc).__name__,
                    }
                )
                continue
            await store.clear_graph_mirror_pending(pending_run_id, entry.op_seq)
        return degradations

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
    # LLM token budget (#1270)
    # ------------------------------------------------------------------

    def _llm_budget_breach(self, op: DreamOp, namespace_id: UUID) -> str | None:
        """Return a breach detail string if ``op`` would exceed a budget.

        Returns ``None`` when ``op`` is not an LLM-using op or when both
        budgets still have headroom. A budget value of ``0`` disables
        that budget (mirrors the hooks per-subscription convention).

        The check is pre-dispatch and conservative: it compares the
        spend already accumulated this run / this rolling-day window
        against the cap. An op is refused once the cap is met (a fresh
        run with the cap already consumed by prior ops fans out zero new
        LLM calls).
        """
        if str(op.op_type) not in _LLM_OP_KINDS:
            return None

        per_run = self._config.llm_max_tokens_per_run
        if per_run and self._llm_tokens_this_run >= per_run:
            return f"per_run cap reached: {self._llm_tokens_this_run} >= {per_run} (DreamConfig.llm_max_tokens_per_run)"

        per_day = self._config.llm_max_tokens_per_namespace_per_day
        if per_day:
            bucket = _get_namespace_day_budget(namespace_id)
            if bucket.tokens_used >= per_day:
                return (
                    f"per_namespace_per_day cap reached: {bucket.tokens_used} >= {per_day} "
                    "(DreamConfig.llm_max_tokens_per_namespace_per_day)"
                )
        return None

    def _drain_llm_usage(self, namespace_id: UUID) -> int:
        """Drain context-local LLM usage into both budget buckets.

        Reads back whatever the just-applied op recorded through
        ``record_usage`` (the same path ``record_llm_call`` feeds), adds
        the total tokens to the per-run counter and the per-namespace
        rolling-day bucket, then re-opens a fresh accumulator for the
        next op. Returns the number of tokens drained (0 for non-LLM
        ops, which record nothing).
        """
        from khora.telemetry.context import collect_usage, start_usage_collection

        entries = collect_usage()
        start_usage_collection()
        total = sum(int(getattr(u, "total_tokens", 0) or 0) for u in entries)
        if total:
            self._llm_tokens_this_run += total
            _get_namespace_day_budget(namespace_id).tokens_used += total
        return total

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
    # Run-state persistence - delegated to the stack's DreamRunStore
    # (PG / SQLite-sidecar / SurrealDB-relational, #1274). A ``None`` store
    # (graph-only stub with no SQL / SurrealDB) makes every call a no-op.
    # ------------------------------------------------------------------

    async def _init_run_row(self, run_id: UUID, namespace_id: UUID, mode: str, *, is_resume: bool = False) -> None:
        # On resume the run row already exists; re-running record_run would
        # reset state/last_committed_op_seq on backends whose write is not
        # conflict-preserving (the SurrealDB UPSERT), replaying committed ops.
        # The SQL stores guard this with ON CONFLICT DO UPDATE heartbeat_at, but
        # skipping on resume is correct and uniform across backends.
        if is_resume:
            return
        store = self._run_store()
        if store is None:
            return
        await store.record_run(run_id, namespace_id, mode=mode, trigger="manual")

    async def _persist_plan_hash(self, run_id: UUID, plan: DreamPlan) -> None:
        store = self._run_store()
        if store is None:
            return
        await store.persist_plan(run_id, plan_hash=plan_hash(plan), total_ops=len(plan.ops))

    async def _read_last_committed(self, run_id: UUID) -> int:
        store = self._run_store()
        if store is None:
            return -1
        return await store.read_last_committed(run_id)

    async def _record_committed(self, run_id: UUID, seq: int) -> None:
        store = self._run_store()
        if store is None:
            return
        await store.advance_checkpoint(run_id, seq)

    async def _record_committed_in_session(self, session: Any, run_id: UUID, seq: int) -> None:
        """Persist ``last_committed_op_seq`` using an existing session.

        Phase 4 apply runs the checkpoint update inside the same
        transaction as the apply handler so a rollback unwinds both.
        The PG / SQLite stores honor the passed ``session``; the
        SurrealDB store ignores it (no shared SQL session to enroll).
        """
        store = self._run_store()
        if store is None:
            return
        await store.advance_checkpoint(run_id, seq, session=session)

    async def _finalize_run_row(
        self,
        run_id: UUID,
        plan: DreamPlan | None,
        *,
        state: str,
        error: str | None = None,
    ) -> None:
        store = self._run_store()
        if store is None:
            return
        await store.finalize_run(
            run_id,
            state=state,
            total_ops=len(plan.ops) if plan is not None else 0,
            error=error,
        )

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
    extra_skip_reasons: list[dict[str, Any]] | None = None,
    extra_degradations: list[dict[str, Any]] | None = None,
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
    # ``skip_reasons`` is the #876 observability fix: planners attach
    # entries to ``plan.metadata["skip_reasons"]`` when an op was
    # requested but produced no work (op not supported by the active
    # engine, no candidate rows, runtime flag off, guardrail tripped).
    # The orchestrator appends apply-time skips here too - e.g. the
    # ``llm_budget_exhausted`` entries from the #1270 token budget. An
    # empty list signals "every requested op did work".
    skip_reasons = list(plan.metadata.get("skip_reasons", ()))
    if extra_skip_reasons:
        skip_reasons.extend(extra_skip_reasons)
    metadata: dict[str, Any] = {
        "plan_hash": plan_hash(plan),
        "plan_payload": canonical_plan_payload(plan),
        "skip_reasons": skip_reasons,
    }
    # Graph-mirror degradations (#1272): a post-commit Neo4j mirror that raised
    # after the PG apply committed. PG state is durable; the op is queued for
    # the reconciler. Recorded per ADR-001 so operators see the divergence.
    if extra_degradations:
        metadata["degradations"] = list(extra_degradations)
    return DreamResult(
        run=info,
        diff=DreamDiff(),
        ops=tuple(summaries.values()),
        metadata=metadata,
    )


def _elapsed_ms(start: datetime) -> float:
    return (datetime.now(UTC) - start).total_seconds() * 1000.0


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


def _graph_backend(coordinator: Any) -> Any | None:
    """Read ``coordinator._graph`` (the internal backend slot).

    Uses the internal slot so the namespace-required proxy doesn't fire
    deprecation warnings during a dream run.
    """
    return getattr(coordinator, "_graph", None)


def _supported_mirror_kinds(graph: Any) -> frozenset[str]:
    """The op-kind *string* values the graph backend can mirror (#1271 probe)."""
    probe = getattr(graph, "supports_dream_mirror", None)
    if probe is None:
        return frozenset()
    try:
        return frozenset(str(k) for k in probe())
    except Exception:  # pragma: no cover - defensive; probe is pure
        return frozenset()


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
