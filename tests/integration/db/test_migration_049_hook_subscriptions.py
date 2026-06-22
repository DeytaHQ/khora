"""Live-PostgreSQL integration test for migration 049 + the persistent
hook-subscription store (#599).

Verifies on real Postgres:
1. ``alembic upgrade head`` builds ``khora_hook_subscriptions`` with the
   promised columns + index, version table points at 049.
2. ``HookSubscriptionStore`` round-trips a subscription through pgvector's
   native UUID / JSONB columns (persist -> load_all -> delete).
3. Restart simulation: persist via dispatcher #1, build a fresh dispatcher
   #2 against the same DB, load_persistent, replay an event, observe the
   delivery sink fire (the worker path).
4. Migration is reversible (downgrade drops the table).

Run with an explicit DB URL (the shell leaks a different one)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/db/test_migration_049_hook_subscriptions.py \
        -o addopts="" -q
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.core.models.event import EventType, MemoryEvent
from khora.db.session import run_migrations
from khora.hooks.dispatcher import HookDispatcher
from khora.hooks.models import SemanticFilter
from khora.hooks.subscription_store import HookSubscriptionStore, PersistentSubscription

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


pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

_HEAD = "049_hook_subscriptions"
_PREV = "048_dream_conflicts_reconcile"


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


async def _reset_public_schema(eng: AsyncEngine) -> None:
    async with eng.begin() as conn:
        r = await conn.execute(
            sa.text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
        )
        for (typname,) in r.fetchall():
            await conn.execute(sa.text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
        await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        await conn.execute(sa.text("CREATE SCHEMA public"))
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            sa.text(
                "CREATE TABLE khora_alembic_version ("
                "  version_num VARCHAR(64) NOT NULL,"
                "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                ")"
            )
        )


@pytest.fixture
def pg_url() -> Iterator[str]:
    if not _pg_reachable():
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    async def _setup() -> None:
        eng = create_async_engine(DATABASE_URL)
        try:
            await _reset_public_schema(eng)
        finally:
            await eng.dispose()
        result = await run_migrations(DATABASE_URL)
        assert result.success, f"migrations failed: {result.error}"

    asyncio.run(_setup())
    yield DATABASE_URL


def _factory(url: str) -> tuple[AsyncEngine, async_sessionmaker]:
    engine = create_async_engine(url)
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _event(namespace_id):
    return MemoryEvent.entity_created(
        namespace_id=namespace_id,
        entity_id=uuid4(),
        data={"name": "Acme Corp", "entity_type": "ORGANIZATION"},
    )


class TestMigration049Schema:
    def test_creates_table_and_index(self, pg_url: str) -> None:
        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    exists = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                            "WHERE table_name = 'khora_hook_subscriptions')"
                        )
                    )
                    assert exists.scalar() is True

                    cols = await conn.execute(
                        sa.text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'khora_hook_subscriptions'"
                        )
                    )
                    names = {row[0] for row in cols.fetchall()}
                    assert names == {
                        "id",
                        "namespace_id",
                        "event_type",
                        "filter",
                        "delivery",
                        "created_at",
                        "last_delivered_at",
                        "delivery_failure_count",
                        "paused_at",
                    }

                    idx = await conn.execute(
                        sa.text("SELECT indexname FROM pg_indexes WHERE tablename = 'khora_hook_subscriptions'")
                    )
                    idx_names = {row[0] for row in idx.fetchall()}
                    assert "ix_khora_hook_subscriptions_ns_event" in idx_names

                    ver = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert ver.scalar() == _HEAD
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_downgrade_drops_table(self, pg_url: str) -> None:
        cfg = _make_config(pg_url)
        command.downgrade(cfg, _PREV)

        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    exists = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                            "WHERE table_name = 'khora_hook_subscriptions')"
                        )
                    )
                    assert exists.scalar() is False
            finally:
                await engine.dispose()

        asyncio.run(check())


class TestHookSubscriptionStorePG:
    def test_round_trip(self, pg_url: str) -> None:
        async def run() -> None:
            engine, factory = _factory(pg_url)
            try:
                store = HookSubscriptionStore(factory)
                ns = uuid4()
                sub = PersistentSubscription(
                    event_type="entity.created",
                    delivery={"type": "webhook", "url": "https://example.test/hook"},
                    namespace_id=ns,
                    filter=SemanticFilter(name="orgs", entity_types=["ORGANIZATION"], namespace_id=ns),
                )
                await store.persist(sub)

                loaded = await store.load_all()
                assert len(loaded) == 1
                got = loaded[0]
                assert got.id == sub.id
                assert got.namespace_id == ns
                assert got.event_type == "entity.created"
                assert got.delivery["url"] == "https://example.test/hook"
                assert got.filter is not None
                assert got.filter.entity_types == ["ORGANIZATION"]

                assert await store.delete(sub.id) is True
                assert await store.load_all() == []
            finally:
                await engine.dispose()

        asyncio.run(run())

    def test_restart_replays_to_worker(self, pg_url: str) -> None:
        async def run() -> None:
            engine1, factory1 = _factory(pg_url)
            engine2, factory2 = _factory(pg_url)
            try:
                store = HookSubscriptionStore(factory1)
                ns = uuid4()

                # Process #1 registers a persistent subscription.
                d1 = HookDispatcher(subscription_store=store)
                sub_id = await d1.register_persistent(
                    EventType.ENTITY_CREATED,
                    {"type": "webhook", "url": "https://example.test/hook"},
                    namespace_id=ns,
                )

                # --- restart: fresh dispatcher, fresh store, same DB ---
                delivered: list = []

                async def sink(s: PersistentSubscription, e: MemoryEvent) -> None:
                    delivered.append((s.id, e.resource_id))

                d2 = HookDispatcher(subscription_store=HookSubscriptionStore(factory2), delivery_sink=sink)
                assert await d2.load_persistent() == 1

                event = _event(ns)
                await d2.dispatch(event)
                assert delivered == [(sub_id, event.resource_id)]
            finally:
                await engine1.dispose()
                await engine2.dispose()

        asyncio.run(run())
