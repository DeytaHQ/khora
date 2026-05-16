"""PG-backed integration test for ``plan_chronicle_tombstone_audit`` (#654).

Validates the Postgres ``percentile_disc`` path of the audit on a real
PG instance. Skips cleanly when PostgreSQL isn't reachable on the
dev-compose port.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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
from khora.dream.config import DreamConfig
from khora.dream.engines.chronicle import plan_chronicle_tombstone_audit

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
            await s.rollback()
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


async def _insert_inactive_fact(
    session: AsyncSession,
    *,
    namespace_id: UUID,
    updated_at: datetime,
) -> None:
    await session.execute(
        text(
            "INSERT INTO memory_facts "
            "(id, namespace_id, subject, predicate, object, fact_text, "
            "confidence, is_active, source_chunk_ids, created_at, updated_at) "
            "VALUES (:id, :ns, 'Alice', 'lives_in', 'city', 'Alice lives in X', "
            "0.9, false, ARRAY[]::uuid[], :ts, :ts)"
        ),
        {"id": uuid4(), "ns": namespace_id, "ts": updated_at},
    )


async def test_pg_audit_uses_percentile_disc(session: AsyncSession) -> None:
    """The Postgres path computes percentile_disc over inactive facts."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    for d in (1, 5, 10, 30, 100):
        await _insert_inactive_fact(session, namespace_id=ns, updated_at=now - timedelta(days=d))
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    assert op.decision == "audit_complete"
    assert outputs["active_count"] == 0
    assert outputs["inactive_count"] == 5
    # Tolerances absorb sub-second wall-clock drift between INSERT and NOW().
    assert outputs["oldest_tombstone_age_days"] == pytest.approx(100.0, abs=0.1)
    assert outputs["p50_age_days"] == pytest.approx(10.0, abs=0.1)
    assert outputs["p90_age_days"] == pytest.approx(100.0, abs=0.1)


async def test_pg_audit_empty_namespace(session: AsyncSession) -> None:
    """Empty namespace on Postgres → decision='empty_namespace', no aggregates."""
    ns = uuid4()
    await _create_namespace(session, ns)

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    assert op.decision == "empty_namespace"
    assert outputs["total_count"] == 0
    assert outputs["p50_age_days"] is None
