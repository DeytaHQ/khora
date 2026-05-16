"""End-to-end orchestrator integration tests (#661).

Runs ``Khora.dream()`` against a real Postgres instance (started via
``make dev``). Verifies:

- dry-run + apply mode against the registered chronicle plugin
- ``khora_dream_runs`` row is persisted with ``state='completed'``
- ``Khora.dream_history`` returns the run
- crash + resume round-trip uses ``resume_from``

Skipped cleanly when Postgres isn't reachable.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.db.session import run_migrations
from khora.dream.api import dream, dream_history, dream_status
from khora.dream.config import DreamConfig
from khora.dream.plan import DreamScope, OpKind

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pg_reachable(),
        reason="PostgreSQL not reachable (run `make dev` first)",
    ),
]


@pytest.fixture(scope="module")
async def _migrated() -> None:
    result = await run_migrations(DATABASE_URL)
    if not result.success and not result.skipped:
        raise RuntimeError(f"migration failed: {result.error}")


@pytest.fixture
async def session(_migrated: None) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(DATABASE_URL)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def _create_namespace(session: AsyncSession, ns_id: UUID) -> None:
    await session.execute(
        text(
            "INSERT INTO memory_namespaces (id, namespace_id, tenancy_mode, version, is_active) "
            "VALUES (:id, :ns, 'shared', 1, true) ON CONFLICT DO NOTHING"
        ),
        {"id": ns_id, "ns": ns_id},
    )
    await session.commit()


# ---------------------------------------------------------------------------
# Test KB — minimal Khora-shaped wrapper around the real DB
# ---------------------------------------------------------------------------


class _IntegrationKB:
    """Real-DB Khora stub for orchestrator e2e — wires the storage path."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._config = type("_Cfg", (), {"dream": DreamConfig(enabled=True)})()
        self._engine_name = "chronicle"
        self._engine = None
        self.storage = _RealStorage(factory)

    def _get_engine(self) -> object:
        # Chronicle abstention drift is the only op that reads the
        # engine. Other Phase 1 chronicle ops (tombstone audit) only
        # need a session. Provide the threshold attrs so abstention
        # drift can plan; it will short-circuit on insufficient samples.
        class _EngineAttrs:
            _abstention_min_top_score = 0.3
            _abstention_combined_threshold = 0.5
            _abstention_min_chunks = 1

        return _EngineAttrs()


class _RealStorage:
    """Coordinator-shaped adapter exposing a ``transaction()`` over an asyncpg session factory."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    def transaction(self) -> _TxnCtx:
        return _TxnCtx(self._factory)

    async def resolve_namespace(self, ns: UUID) -> UUID:
        return ns


class _TxnCtx:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> _TxnCtx:
        self._session = self._factory()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._session is not None
        if exc is None:
            await self._session.commit()
        else:
            await self._session.rollback()
        await self._session.close()

    @property
    def session(self) -> AsyncSession:
        assert self._session is not None
        return self._session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb(_migrated: None) -> AsyncIterator[_IntegrationKB]:
    engine = create_async_engine(DATABASE_URL)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        yield _IntegrationKB(factory)
    finally:
        await engine.dispose()


async def test_dream_dry_run_writes_run_row(kb: _IntegrationKB, session: AsyncSession) -> None:
    """A dry-run records a row in ``khora_dream_runs`` with ``state='completed'``."""
    ns = uuid4()
    await _create_namespace(session, ns)

    # Run only the tombstone audit (cheap; no abstention samples needed).
    result = await dream(
        kb,
        ns,
        mode="dry-run",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
    )

    assert result.run.mode == "dry-run"
    assert result.run.namespace_id == ns

    row = (
        await session.execute(
            text("SELECT state, mode, total_ops FROM khora_dream_runs WHERE run_id = :rid"),
            {"rid": result.run.run_id},
        )
    ).first()
    assert row is not None
    assert row.state == "completed"
    assert row.mode == "dry-run"
    assert row.total_ops == 1


async def test_dream_apply_mode_pass_through(kb: _IntegrationKB, session: AsyncSession) -> None:
    """Apply mode on Phase-1 ops records ``state='completed'`` + advances seq."""
    ns = uuid4()
    await _create_namespace(session, ns)

    result = await dream(
        kb,
        ns,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
    )
    assert result.run.mode == "apply"

    row = (
        await session.execute(
            text("SELECT state, last_committed_op_seq FROM khora_dream_runs WHERE run_id = :rid"),
            {"rid": result.run.run_id},
        )
    ).first()
    assert row is not None
    assert row.state == "completed"
    assert row.last_committed_op_seq == 0


async def test_dream_history_returns_completed_runs(kb: _IntegrationKB, session: AsyncSession) -> None:
    """``Khora.dream_history`` returns the run records for a namespace."""
    ns = uuid4()
    await _create_namespace(session, ns)

    await dream(
        kb,
        ns,
        mode="dry-run",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
    )
    history = await dream_history(kb, ns, limit=10)
    assert len(history) >= 1
    assert history[0].namespace_id == ns


async def test_dream_status_returns_finished_at(kb: _IntegrationKB, session: AsyncSession) -> None:
    """``Khora.dream_status`` resolves a finished run."""
    ns = uuid4()
    await _create_namespace(session, ns)
    result = await dream(
        kb,
        ns,
        mode="dry-run",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
    )
    status = await dream_status(kb, result.run.run_id)
    assert status["run_id"] == str(result.run.run_id)
    assert status["finished_at"] is not None


async def test_dream_resume_continues_from_checkpoint(kb: _IntegrationKB, session: AsyncSession) -> None:
    """Resuming a run with ``resume_from`` advances ``last_committed_op_seq``."""
    ns = uuid4()
    await _create_namespace(session, ns)

    first = await dream(
        kb,
        ns,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
    )
    run_id = first.run.run_id

    # Mark the run as crashed mid-apply with last_committed_op_seq=-1
    # so the resume picks up from op 0.
    await session.execute(
        text(
            "UPDATE khora_dream_runs SET state='applying', last_committed_op_seq=-1, finished_at=NULL "
            "WHERE run_id = :rid"
        ),
        {"rid": run_id},
    )
    await session.commit()

    resumed = await dream(
        kb,
        ns,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_TOMBSTONE_AUDIT,)),
        resume_from=run_id,
    )
    assert resumed.run.run_id == run_id

    row = (
        await session.execute(
            text("SELECT state, last_committed_op_seq FROM khora_dream_runs WHERE run_id = :rid"),
            {"rid": run_id},
        )
    ).first()
    assert row is not None
    assert row.state == "completed"
    assert row.last_committed_op_seq == 0
