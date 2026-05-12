"""Factory wiring test for the sqlite_lance backend.

Exercises the full ``StorageFactory`` -> ``StorageCoordinator`` path
with ``backend='sqlite_lance'``: one ``EmbeddedStorageHandle`` shared
across four adapters (relational, graph, vector, event_store), Alembic
migrations against the SQLite file, and backend health checks.

Per-adapter write/read contracts are covered by the per-adapter test
modules (DYT-2728..2731); a ``Khora.remember()`` round-trip will
be added in DYT-2734 (integration tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config.schema import SQLiteLanceConfig
from khora.db.session import run_migrations
from khora.storage.coordinator import StorageCoordinator
from khora.storage.factory import StorageConfig, StorageFactory

pytestmark = pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed")


@pytest.fixture
async def coordinator(tmp_path: Path):
    """Factory-built coordinator against a freshly-migrated SQLite DB."""
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    # DYT-2727 made these migrations dialect-aware; they work on SQLite.
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    assert result.success, f"migration failed: {result.error}"

    storage_config = StorageConfig(
        backend="sqlite_lance",
        sqlite_lance_config=SQLiteLanceConfig(
            db_path=db_path,
            lance_path=lance_path,
            embedding_dimension=8,
        ),
    )
    factory = StorageFactory(config=storage_config)
    coord = factory.create_coordinator()
    await coord.connect()
    try:
        yield coord
    finally:
        await coord.disconnect()


async def test_factory_builds_all_four_adapters(tmp_path: Path) -> None:
    """All four adapters are wired and share one EmbeddedStorageHandle."""
    from khora.storage.backends.sqlite_lance import (
        SQLiteLanceEventStoreAdapter,
        SQLiteLanceGraphAdapter,
        SQLiteLanceRelationalAdapter,
        SQLiteLanceVectorAdapter,
    )

    cfg = StorageConfig(
        backend="sqlite_lance",
        sqlite_lance_config=SQLiteLanceConfig(
            db_path=str(tmp_path / "k.db"),
            lance_path=str(tmp_path / "k.lance"),
            embedding_dimension=8,
        ),
    )
    coord = StorageFactory(config=cfg).create_coordinator()

    assert isinstance(coord.relational, SQLiteLanceRelationalAdapter)
    assert isinstance(coord.graph, SQLiteLanceGraphAdapter)
    assert isinstance(coord.vector, SQLiteLanceVectorAdapter)
    assert isinstance(coord.event_store, SQLiteLanceEventStoreAdapter)

    # Two engines (SQLite + LanceDB), so unified-backend flag must stay False;
    # the coordinator still dual-writes entities to graph and vector.
    assert coord._is_unified_backend is False

    # All four adapters must share the same EmbeddedStorageHandle instance.
    handle = coord.graph._handle  # type: ignore[attr-defined]
    assert coord.vector._handle is handle  # type: ignore[attr-defined]
    assert coord.event_store._handle is handle  # type: ignore[attr-defined]
    assert coord.relational._handle is handle  # type: ignore[attr-defined]


async def test_factory_raises_when_config_missing(tmp_path: Path) -> None:
    cfg = StorageConfig(backend="sqlite_lance", sqlite_lance_config=None)
    with pytest.raises(ValueError, match="sqlite_lance_config is not set"):
        StorageFactory(config=cfg).create_coordinator()


async def test_coordinator_connect_and_health(coordinator: StorageCoordinator) -> None:
    """All four backends report healthy after factory-driven connect."""
    health = await coordinator.health_check()
    assert health.relational is True
    assert health.vector is True
    assert health.graph is True
    assert health.event_store is True


async def test_coordinator_relational_session_runs_migrated_schema(
    coordinator: StorageCoordinator,
) -> None:
    """The relational adapter's SQLAlchemy session sees the migrated tables.

    Proves the factory wires the relational adapter with its own
    SQLAlchemy engine pointing at the same SQLite file that Alembic
    migrated, so ``StorageCoordinator.transaction()`` (which pulls
    ``_session_factory`` off the relational backend) has a working
    session to hand out.
    """
    assert coordinator.relational is not None
    session_factory = coordinator.relational._session_factory  # type: ignore[attr-defined]
    assert session_factory is not None

    async with session_factory() as session:
        # chunks is one of the tables the Alembic migrations create;
        # selecting the empty schema confirms it exists on this SQLite file.
        await session.execute(text("SELECT 1 FROM chunks LIMIT 0"))
        await session.execute(text("SELECT 1 FROM documents LIMIT 0"))
        await session.execute(text("SELECT 1 FROM memory_namespaces LIMIT 0"))
