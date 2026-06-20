"""DreamRunStore: backend-portable dream run-state (#1274).

Run-state (record run / advance checkpoint / status / history / resume)
and the #1272 reconciler's per-op ``graph_mirror_pending`` list used to
be PostgreSQL-only on the write side. This module factors that state into
a small :class:`DreamRunStore` protocol with three impls so the dream
orchestrator works on non-PG stacks:

  * :class:`PostgresDreamRunStore` - the existing behavior, byte-identical
    SQL through ``coordinator.transaction()`` (PG present);
  * :class:`SqliteDreamRunStore` - the default for any non-PG SQL stack
    (the sqlite_lance fixture), backed by the same SQLite file that ships
    the ``khora_dream_runs`` table (migration 032/047); zero new
    dependency, transactional;
  * :class:`SurrealDreamRunStore` - a ``DEFINE``-d relational table on the
    unified SurrealDB stack (no Alembic).

The orchestrator selects an impl via :func:`select_run_store`.

``graph_mirror_pending`` is a per-op list the #1272 reconciler reads to
re-attempt committed-but-unmirrored ops. The checkpoint advances inside
the PG apply commit *before* the graph mirror runs, so a failed mirror
leaves a committed op with an entry in this list; the reconciler drains
it. Each entry is a :class:`GraphMirrorPending`.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import text

from khora.dream.result import DreamRunInfo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.storage.backends.surrealdb.connection import SurrealDBConnection
    from khora.storage.coordinator import StorageCoordinator


@dataclass(slots=True, frozen=True)
class GraphMirrorPending:
    """One committed-but-unmirrored op awaiting a graph-side re-attempt.

    Recorded by the #1272 post-commit mirror when a graph write fails
    after the PG checkpoint already advanced. The reconciler reads the
    list via :meth:`DreamRunStore.get_graph_mirror_pending`, replays the
    graph write, then drops the entry with
    :meth:`DreamRunStore.clear_graph_mirror_pending`.

    Stability: internal - the #1272 reconciler is the only consumer and
    lands in the same workstream.
    """

    op_seq: int
    op_id: UUID
    op_type: str
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "op_seq": self.op_seq,
            "op_id": str(self.op_id),
            "op_type": self.op_type,
            "payload": self.payload,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> GraphMirrorPending:
        return cls(
            op_seq=int(raw["op_seq"]),
            op_id=UUID(str(raw["op_id"])),
            op_type=str(raw["op_type"]),
            payload=dict(raw.get("payload") or {}),
        )


@runtime_checkable
class DreamRunStore(Protocol):
    """Backend-portable persistence for dream run-state (#1274).

    Minimal surface - only what the orchestrator / api / #1272
    reconciler need. Implementations persist to PostgreSQL, a SQLite
    sidecar, or a SurrealDB-relational table.
    """

    async def record_run(self, run_id: UUID, namespace_id: UUID, *, mode: str, trigger: str = "manual") -> None:
        """Insert (or heartbeat) the run row in the ``planning`` state."""
        ...

    async def persist_plan(self, run_id: UUID, *, plan_hash: str, total_ops: int) -> None:
        """Record the plan hash / op count and move ``planning`` -> ``applying``."""
        ...

    async def read_last_committed(self, run_id: UUID) -> int:
        """Return the resume cursor (``-1`` when nothing has committed)."""
        ...

    async def advance_checkpoint(self, run_id: UUID, op_seq: int, *, session: Any | None = None) -> None:
        """Advance ``last_committed_op_seq`` to ``op_seq``.

        ``session`` lets the PG impl write the checkpoint inside the same
        transaction as the apply handler so a rollback unwinds both.
        Other impls ignore it.
        """
        ...

    async def finalize_run(self, run_id: UUID, *, state: str, total_ops: int, error: str | None = None) -> None:
        """Stamp the terminal ``state`` + ``finished_at`` (and optional error)."""
        ...

    async def status(self, run_id: UUID) -> DreamRunInfo | None:
        """Return run-level metadata for ``run_id`` (or ``None``)."""
        ...

    async def history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]:
        """Return recent runs for ``namespace_id`` (newest first)."""
        ...

    async def mark_graph_mirror_pending(
        self, run_id: UUID, entry: GraphMirrorPending, *, session: Any | None = None
    ) -> None:
        """Record (or replace, keyed on ``op_seq``) a pending mirror op.

        ``session`` lets the PG impl write the pending row inside the same
        transaction as the checkpoint advance so a hard crash between the PG
        commit and the graph mirror still leaves a durable pending row the
        reconciler can drain (#1292). Other impls ignore it.
        """
        ...

    async def get_graph_mirror_pending(self, run_id: UUID) -> list[GraphMirrorPending]:
        """Return the pending mirror ops for ``run_id`` (empty when none)."""
        ...

    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        """Return every open ``(run_id, entry)`` pair for ``namespace_id`` (#1292).

        Spans ALL runs in the namespace, not just the current one, so a later
        run with a fresh ``run_id`` drains a prior run's committed-but-unmirrored
        ops left by a crash.
        """
        ...

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        """Drop the pending entry for ``op_seq`` once it has been mirrored."""
        ...


# ---------------------------------------------------------------------------
# Dialect-aware bind / coerce helpers (mirror orchestrator's #896 helpers)
# ---------------------------------------------------------------------------


def _is_postgres_session(session: Any) -> bool:
    bind = getattr(session, "bind", None)
    dialect = getattr(bind, "dialect", None)
    return dialect is not None and dialect.name == "postgresql"


def _uuid_param(session: Any, value: UUID) -> Any:
    return value if _is_postgres_session(session) else str(value)


def _ts_param(session: Any, value: datetime) -> Any:
    return value if _is_postgres_session(session) else value.isoformat()


def _coerce_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _coerce_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _run_info_from_row(row: Any) -> DreamRunInfo:
    started_at = _coerce_dt(row.started_at)
    finished_at = _coerce_dt(row.finished_at)
    duration_ms = (finished_at - started_at).total_seconds() * 1000.0 if finished_at is not None else None
    return DreamRunInfo(
        run_id=_coerce_uuid(row.run_id),
        namespace_id=_coerce_uuid(row.namespace_id),
        mode=row.mode,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
    )


def _load_pending(raw: Any) -> list[GraphMirrorPending]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else []
    return [GraphMirrorPending.from_json(item) for item in (raw or [])]


def _merge_pending(existing: list[GraphMirrorPending], entry: GraphMirrorPending) -> list[GraphMirrorPending]:
    merged = [p for p in existing if p.op_seq != entry.op_seq]
    merged.append(entry)
    merged.sort(key=lambda p: p.op_seq)
    return merged


# ---------------------------------------------------------------------------
# SQL impls (PostgreSQL + SQLite share the SQL; differ only in binding /
# the JSON column read-back and how the session is obtained)
# ---------------------------------------------------------------------------


class _SqlDreamRunStore:
    """Shared SQL implementation for the PostgreSQL and SQLite stores.

    Subclasses provide a session via :meth:`_open` (an async context
    manager yielding an :class:`AsyncSession`). The SQL is identical
    across dialects; UUID / timestamp binding is dialect-aware via the
    ``_uuid_param`` / ``_ts_param`` helpers so SQLite stores text and
    Postgres stores native types.
    """

    def _open(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    async def record_run(self, run_id: UUID, namespace_id: UUID, *, mode: str, trigger: str = "manual") -> None:
        now = datetime.now(UTC)
        async with self._open() as session:
            await session.execute(
                text(
                    "INSERT INTO khora_dream_runs "
                    "(run_id, namespace_id, trigger, mode, state, started_at, "
                    " heartbeat_at, total_ops, total_decisions, last_committed_op_seq) "
                    "VALUES (:rid, :ns, :trg, :mode, :state, :ts, :ts, 0, 0, -1) "
                    "ON CONFLICT (run_id) DO UPDATE SET heartbeat_at = :ts"
                ),
                {
                    "rid": _uuid_param(session, run_id),
                    "ns": _uuid_param(session, namespace_id),
                    "trg": trigger,
                    "mode": mode,
                    "state": "planning",
                    "ts": _ts_param(session, now),
                },
            )

    async def persist_plan(self, run_id: UUID, *, plan_hash: str, total_ops: int) -> None:
        async with self._open() as session:
            await session.execute(
                text(
                    "UPDATE khora_dream_runs "
                    "SET plan_hash = :ph, total_ops = :tot, heartbeat_at = :ts, "
                    "    state = CASE WHEN state = 'planning' THEN 'applying' ELSE state END "
                    "WHERE run_id = :rid"
                ),
                {
                    "ph": plan_hash,
                    "tot": total_ops,
                    "ts": _ts_param(session, datetime.now(UTC)),
                    "rid": _uuid_param(session, run_id),
                },
            )

    async def read_last_committed(self, run_id: UUID) -> int:
        async with self._open() as session:
            row = (
                await session.execute(
                    text("SELECT last_committed_op_seq FROM khora_dream_runs WHERE run_id = :rid"),
                    {"rid": _uuid_param(session, run_id)},
                )
            ).first()
        if row is None or row.last_committed_op_seq is None:
            return -1
        return int(row.last_committed_op_seq)

    async def advance_checkpoint(self, run_id: UUID, op_seq: int, *, session: Any | None = None) -> None:
        if session is not None:
            await self._advance_in_session(session, run_id, op_seq)
            return
        async with self._open() as own_session:
            await self._advance_in_session(own_session, run_id, op_seq)

    @staticmethod
    async def _advance_in_session(session: Any, run_id: UUID, op_seq: int) -> None:
        await session.execute(
            text("UPDATE khora_dream_runs SET last_committed_op_seq = :seq, heartbeat_at = :ts WHERE run_id = :rid"),
            {
                "seq": op_seq,
                "ts": _ts_param(session, datetime.now(UTC)),
                "rid": _uuid_param(session, run_id),
            },
        )

    async def finalize_run(self, run_id: UUID, *, state: str, total_ops: int, error: str | None = None) -> None:
        now = datetime.now(UTC)
        async with self._open() as session:
            params: dict[str, Any] = {
                "rid": _uuid_param(session, run_id),
                "state": state,
                "ts": _ts_param(session, now),
                "total": total_ops,
            }
            if error is not None:
                params["err"] = json.dumps({"message": error})
                cast = "CAST(:err AS jsonb)" if _is_postgres_session(session) else ":err"
                await session.execute(
                    text(
                        "UPDATE khora_dream_runs "  # noqa: S608 - cast is a hardcoded literal
                        "SET state = :state, finished_at = :ts, total_ops = :total, "
                        f"    error = {cast} WHERE run_id = :rid"
                    ),
                    params,
                )
            else:
                await session.execute(
                    text(
                        "UPDATE khora_dream_runs "
                        "SET state = :state, finished_at = :ts, total_ops = :total WHERE run_id = :rid"
                    ),
                    params,
                )

    async def status(self, run_id: UUID) -> DreamRunInfo | None:
        async with self._open() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT run_id, namespace_id, mode, started_at, finished_at "
                        "FROM khora_dream_runs WHERE run_id = :rid"
                    ),
                    {"rid": _uuid_param(session, run_id)},
                )
            ).first()
        if row is None:
            return None
        return _run_info_from_row(row)

    async def history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]:
        async with self._open() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT run_id, namespace_id, mode, started_at, finished_at "
                        "FROM khora_dream_runs "
                        "WHERE namespace_id = :ns "
                        "ORDER BY started_at DESC LIMIT :lim"
                    ),
                    {"ns": _uuid_param(session, namespace_id), "lim": int(limit)},
                )
            ).all()
        return [_run_info_from_row(row) for row in rows]

    async def mark_graph_mirror_pending(
        self, run_id: UUID, entry: GraphMirrorPending, *, session: Any | None = None
    ) -> None:
        # An external session (the apply loop's checkpoint transaction) makes
        # the pending row durable atomically with the PG commit (#1292); the
        # caller commits it. Without one, own the session lifecycle.
        if session is not None:
            await self._mark_in_session(session, run_id, entry)
            return
        async with self._open() as own_session:
            await self._mark_in_session(own_session, run_id, entry)

    @classmethod
    async def _mark_in_session(cls, session: Any, run_id: UUID, entry: GraphMirrorPending) -> None:
        existing = await cls._read_pending(session, run_id)
        merged = _merge_pending(existing, entry)
        await cls._write_pending(session, run_id, merged)

    async def get_graph_mirror_pending(self, run_id: UUID) -> list[GraphMirrorPending]:
        async with self._open() as session:
            return await self._read_pending(session, run_id)

    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        async with self._open() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT run_id, graph_mirror_pending FROM khora_dream_runs "
                        "WHERE namespace_id = :ns AND graph_mirror_pending IS NOT NULL"
                    ),
                    {"ns": _uuid_param(session, namespace_id)},
                )
            ).all()
        out: list[tuple[UUID, GraphMirrorPending]] = []
        for row in rows:
            run_id = _coerce_uuid(row.run_id)
            for entry in _load_pending(row.graph_mirror_pending):
                out.append((run_id, entry))
        return out

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        async with self._open() as session:
            existing = await self._read_pending(session, run_id)
            remaining = [p for p in existing if p.op_seq != op_seq]
            await self._write_pending(session, run_id, remaining)

    @staticmethod
    async def _read_pending(session: Any, run_id: UUID) -> list[GraphMirrorPending]:
        row = (
            await session.execute(
                text("SELECT graph_mirror_pending FROM khora_dream_runs WHERE run_id = :rid"),
                {"rid": _uuid_param(session, run_id)},
            )
        ).first()
        if row is None:
            return []
        return _load_pending(row.graph_mirror_pending)

    @staticmethod
    async def _write_pending(session: Any, run_id: UUID, pending: list[GraphMirrorPending]) -> None:
        payload = json.dumps([p.to_json() for p in pending])
        cast = "CAST(:pending AS jsonb)" if _is_postgres_session(session) else ":pending"
        await session.execute(
            text(f"UPDATE khora_dream_runs SET graph_mirror_pending = {cast} WHERE run_id = :rid"),  # noqa: S608 - cast is a hardcoded literal
            {"pending": payload, "rid": _uuid_param(session, run_id)},
        )


class PostgresDreamRunStore(_SqlDreamRunStore):
    """Run-state on PostgreSQL via the coordinator's shared transaction.

    Byte-identical to the pre-#1274 inline orchestrator SQL. The shared
    ``coordinator.transaction()`` session lets the apply loop pass its
    open session into :meth:`advance_checkpoint` so the checkpoint commits
    atomically with the apply handler.
    """

    def __init__(self, coordinator: StorageCoordinator) -> None:
        self._coordinator = coordinator

    @asynccontextmanager
    async def _open(self) -> Any:
        # coordinator.transaction() yields a TransactionContext, not the
        # AsyncSession the shared _SqlDreamRunStore SQL runs against - unwrap
        # it. The coordinator commits on clean exit and rolls back on
        # exception, mirroring the SQLite store's _CommittingSession.
        async with self._coordinator.transaction() as txn:
            yield txn.session


class SqliteDreamRunStore(_SqlDreamRunStore):
    """Run-state on a SQLite file (default for any non-PG SQL stack).

    Constructed either with a ``db_path`` (opens its own aiosqlite engine
    - the sidecar shape) or with an existing ``session_factory`` so the
    sqlite_lance stack reuses the relational adapter's engine and shares
    the ``khora_dream_runs`` table (migration 032/047). :meth:`ensure_schema`
    creates the table if it does not already exist so the sidecar path
    works without Alembic.
    """

    def __init__(self, db_path: str | None = None, *, session_factory: Any | None = None) -> None:
        if (db_path is None) == (session_factory is None):
            raise ValueError("SqliteDreamRunStore needs exactly one of db_path / session_factory")
        self._db_path = db_path
        self._engine: Any = None
        self._session_factory = session_factory

    def _ensure_factory(self) -> Any:
        if self._session_factory is not None:
            return self._session_factory
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}", future=True)
        self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        return self._session_factory

    def _open(self) -> Any:
        return _CommittingSession(self._ensure_factory())

    async def ensure_schema(self) -> None:
        """Create ``khora_dream_runs`` if absent (idempotent).

        A no-op on the sqlite_lance stack where Alembic already created
        the table; required for the bare-``db_path`` sidecar shape.
        """
        async with self._open() as session:
            await session.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS khora_dream_runs ("
                    " run_id TEXT PRIMARY KEY,"
                    " namespace_id TEXT NOT NULL,"
                    " trigger VARCHAR(32) NOT NULL,"
                    " mode VARCHAR(16) NOT NULL,"
                    " state VARCHAR(32) NOT NULL,"
                    " plan_hash VARCHAR(64),"
                    " started_at DATETIME NOT NULL,"
                    " finished_at DATETIME,"
                    " last_committed_op_seq INTEGER DEFAULT -1,"
                    " heartbeat_at DATETIME NOT NULL,"
                    " total_ops INTEGER DEFAULT 0,"
                    " total_decisions INTEGER DEFAULT 0,"
                    " report_path TEXT,"
                    " manifest_sha256 VARCHAR(64),"
                    " config_fingerprint VARCHAR(64),"
                    " error JSON,"
                    " graph_mirror_pending JSON"
                    ")"
                )
            )

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


class _CommittingSession:
    """Open a session, yield it, commit on success / roll back on error.

    The standalone SQLite store owns its session lifecycle (the PG store
    leans on ``coordinator.transaction()`` for the same shape).
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = self._factory()
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._session is not None
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()


# ---------------------------------------------------------------------------
# SurrealDB-relational impl (unified stack)
# ---------------------------------------------------------------------------


class SurrealDreamRunStore:
    """Run-state on a SurrealDB-relational ``khora_dream_runs`` table.

    The unified stack has no Alembic - the table is ``DEFINE``-d in
    ``storage/backends/surrealdb/schema.py`` and auto-initialized on
    ``connect()``. The RecordID is derived from ``run_id`` for O(1)
    upsert / lookup; UUIDs round-trip as strings and ``graph_mirror_pending``
    is a SurrealDB ``array`` of objects.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection

    @staticmethod
    def _record(run_id: UUID) -> str:
        # SurrealDB RecordID id-part: angle brackets quote arbitrary text.
        return f"khora_dream_runs:⟨{run_id}⟩"

    async def ensure_schema(self) -> None:
        from khora.storage.backends.surrealdb.schema import initialize_schema

        await initialize_schema(self._conn)

    async def record_run(self, run_id: UUID, namespace_id: UUID, *, mode: str, trigger: str = "manual") -> None:
        # UPSERT creates the record (plain UPDATE on a missing SurrealDB
        # record is a silent no-op); subsequent writes use UPDATE.
        await self._conn.execute(
            f"UPSERT {self._record(run_id)} SET "  # noqa: S608 - record id is a UUID, not user input
            "run_id = $rid, namespace_id = $ns, trigger = $trg, mode = $mode, "
            "state = 'planning', started_at = time::now(), heartbeat_at = time::now(), "
            "last_committed_op_seq = -1, total_ops = 0, graph_mirror_pending = []",
            {"rid": str(run_id), "ns": str(namespace_id), "trg": trigger, "mode": mode},
        )

    async def persist_plan(self, run_id: UUID, *, plan_hash: str, total_ops: int) -> None:
        await self._conn.execute(
            f"UPDATE {self._record(run_id)} SET plan_hash = $ph, total_ops = $tot, heartbeat_at = time::now(), "  # noqa: S608 - record id is a UUID, not user input
            "state = IF state = 'planning' THEN 'applying' ELSE state END",
            {"ph": plan_hash, "tot": total_ops},
        )

    async def read_last_committed(self, run_id: UUID) -> int:
        row = await self._conn.query_one(
            f"SELECT last_committed_op_seq FROM {self._record(run_id)}",  # noqa: S608 - record id is a UUID, not user input
        )
        if row is None or row.get("last_committed_op_seq") is None:
            return -1
        return int(row["last_committed_op_seq"])

    async def advance_checkpoint(self, run_id: UUID, op_seq: int, *, session: Any | None = None) -> None:
        del session  # SurrealDB has no shared SQL session to enroll.
        await self._conn.execute(
            f"UPDATE {self._record(run_id)} SET last_committed_op_seq = $seq, heartbeat_at = time::now()",  # noqa: S608 - record id is a UUID, not user input
            {"seq": op_seq},
        )

    async def finalize_run(self, run_id: UUID, *, state: str, total_ops: int, error: str | None = None) -> None:
        if error is not None:
            await self._conn.execute(
                f"UPDATE {self._record(run_id)} SET state = $state, finished_at = time::now(), "  # noqa: S608 - record id is a UUID, not user input
                "total_ops = $tot, error = $err",
                {"state": state, "tot": total_ops, "err": {"message": error}},
            )
        else:
            await self._conn.execute(
                f"UPDATE {self._record(run_id)} SET state = $state, finished_at = time::now(), total_ops = $tot",  # noqa: S608 - record id is a UUID, not user input
                {"state": state, "tot": total_ops},
            )

    async def status(self, run_id: UUID) -> DreamRunInfo | None:
        row = await self._conn.query_one(
            f"SELECT run_id, namespace_id, mode, started_at, finished_at FROM {self._record(run_id)}",  # noqa: S608 - record id is a UUID, not user input
        )
        if not row or not row.get("run_id"):
            return None
        return self._row_to_info(row)

    async def history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]:
        rows = await self._conn.query(
            "SELECT run_id, namespace_id, mode, started_at, finished_at FROM khora_dream_runs "
            "WHERE namespace_id = $ns ORDER BY started_at DESC LIMIT $lim",
            {"ns": str(namespace_id), "lim": int(limit)},
        )
        return [self._row_to_info(row) for row in rows if row.get("run_id")]

    async def mark_graph_mirror_pending(
        self, run_id: UUID, entry: GraphMirrorPending, *, session: Any | None = None
    ) -> None:
        del session  # SurrealDB has no shared SQL session to enroll.
        existing = await self.get_graph_mirror_pending(run_id)
        merged = _merge_pending(existing, entry)
        await self._write_pending(run_id, merged)

    async def get_graph_mirror_pending(self, run_id: UUID) -> list[GraphMirrorPending]:
        row = await self._conn.query_one(
            f"SELECT graph_mirror_pending FROM {self._record(run_id)}",  # noqa: S608 - record id is a UUID, not user input
        )
        if not row:
            return []
        return _load_pending(row.get("graph_mirror_pending"))

    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        rows = await self._conn.query(
            "SELECT run_id, graph_mirror_pending FROM khora_dream_runs "
            "WHERE namespace_id = $ns AND graph_mirror_pending != []",
            {"ns": str(namespace_id)},
        )
        out: list[tuple[UUID, GraphMirrorPending]] = []
        for row in rows:
            run_id = row.get("run_id")
            if not run_id:
                continue
            for entry in _load_pending(row.get("graph_mirror_pending")):
                out.append((_coerce_uuid(run_id), entry))
        return out

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        existing = await self.get_graph_mirror_pending(run_id)
        remaining = [p for p in existing if p.op_seq != op_seq]
        await self._write_pending(run_id, remaining)

    async def _write_pending(self, run_id: UUID, pending: list[GraphMirrorPending]) -> None:
        await self._conn.execute(
            f"UPDATE {self._record(run_id)} SET graph_mirror_pending = $pending",  # noqa: S608 - record id is a UUID, not user input
            {"pending": [p.to_json() for p in pending]},
        )

    @staticmethod
    def _row_to_info(row: dict[str, Any]) -> DreamRunInfo:
        started_at = _coerce_dt(row.get("started_at"))
        finished_at = _coerce_dt(row.get("finished_at"))
        duration_ms = (finished_at - started_at).total_seconds() * 1000.0 if finished_at is not None else None
        return DreamRunInfo(
            run_id=_coerce_uuid(row["run_id"]),
            namespace_id=_coerce_uuid(row["namespace_id"]),
            mode=row["mode"],
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_run_store(coordinator: StorageCoordinator) -> DreamRunStore | None:
    """Pick the run-state store for ``coordinator``'s stack.

    PG present -> :class:`PostgresDreamRunStore` (byte-identical SQL).
    SurrealDB-unified -> :class:`SurrealDreamRunStore`.
    Any other SQL stack (sqlite_lance) -> :class:`SqliteDreamRunStore`
    reusing the relational adapter's session factory.
    Returns ``None`` when no run-state backend is reachable (a graph-only
    embedded stub with no SQL and no SurrealDB) - the orchestrator then
    no-ops run-state, matching the pre-#1274 ``RuntimeError`` fallback.
    """
    surreal_conn = _surreal_connection(coordinator)
    if surreal_conn is not None:
        return SurrealDreamRunStore(surreal_conn)

    factory = _sql_session_factory(coordinator)
    if factory is None:
        return None
    if _factory_is_postgres(factory):
        return PostgresDreamRunStore(coordinator)
    return SqliteDreamRunStore(session_factory=factory)


def _surreal_connection(coordinator: StorageCoordinator) -> SurrealDBConnection | None:
    if not getattr(coordinator, "_is_unified_backend", False):
        return None
    for attr in ("_graph", "_vector", "_relational"):
        backend = getattr(coordinator, attr, None)
        conn = getattr(backend, "_conn", None)
        if conn is not None:
            return conn
    return None


def _sql_session_factory(coordinator: StorageCoordinator) -> Any | None:
    for attr in ("_relational", "_vector", "_event_store"):
        backend = getattr(coordinator, attr, None)
        factory = getattr(backend, "_session_factory", None)
        if factory is not None:
            return factory
    return None


def _factory_is_postgres(factory: Any) -> bool:
    # async_sessionmaker exposes the bound engine via .kw["bind"] or the
    # first positional bind; the engine's dialect name is the source of truth.
    bind = getattr(factory, "kw", {}).get("bind") if hasattr(factory, "kw") else None
    if bind is None:
        bind = getattr(factory, "bind", None)
    dialect = getattr(bind, "dialect", None)
    return dialect is not None and dialect.name == "postgresql"


__all__ = [
    "DreamRunStore",
    "GraphMirrorPending",
    "PostgresDreamRunStore",
    "SqliteDreamRunStore",
    "SurrealDreamRunStore",
    "select_run_store",
]
