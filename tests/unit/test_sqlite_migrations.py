"""Tests that Alembic migrations run cleanly against a fresh SQLite database.

Verify the dialect gate keeps Postgres-only DDL from breaking
SQLite (sqlite_lance backend) while leaving Postgres behaviour intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine


def _make_config(url: str) -> Config:
    """Build a programmatic Alembic Config pointing at the bundled migrations."""
    cfg = Config()
    migrations_dir = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations"
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


@pytest.mark.unit
class TestSqliteMigrations:
    def test_upgrade_head_fresh_sqlite(self, sqlite_url: str) -> None:
        """Fresh SQLite database: alembic upgrade head must succeed."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        # Verify key tables exist via async introspection.
        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    )
                    tables = {r[0] for r in result}

                    # Core khora tables must exist.
                    expected = {
                        "memory_namespaces",
                        "documents",
                        "chunks",
                        "entities",
                        "relationships",
                        "episodes",
                        "memory_events",
                        "permissions",
                        "sync_checkpoints",
                        "expertise_definitions",
                        "time_nodes",
                        "temporal_edges",
                        "time_edge_links",
                        # Chronicle engine tables (024)
                        "chronicle_events",
                        "memory_facts",
                        # Dream-run checkpoint table, now created on SQLite too (032, #896)
                        "khora_dream_runs",
                        # FTS5 virtual table from migration 002
                        "chunks_fts",
                        # Alembic version table
                        "khora_alembic_version",
                    }
                    missing = expected - tables
                    assert not missing, f"Missing tables after migrate: {missing}"

                    # Dropped in 010 — must NOT be present.
                    assert "workspaces" not in tables
                    assert "organizations" not in tables

                    # Version table must point at head.
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    version = result.scalar()
                    assert version == "046_chunks_occurred_at"
            finally:
                await engine.dispose()

        import asyncio

        asyncio.run(check())

    def test_key_columns_present(self, sqlite_url: str) -> None:
        """After migration, columns added by later migrations must exist."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    # documents: source_timestamp (009), extraction_config_hash (015/016),
                    # external_id (021)
                    result = await conn.execute(sa.text("PRAGMA table_info(documents)"))
                    doc_cols = {r[1] for r in result}
                    assert {"source_timestamp", "extraction_config_hash", "external_id"} <= doc_cols

                    # memory_namespaces flattened: no workspace_id, no slug, no name,
                    # no description, no previous_version_id — but namespace_id present.
                    result = await conn.execute(sa.text("PRAGMA table_info(memory_namespaces)"))
                    ns_cols = {r[1] for r in result}
                    assert "namespace_id" in ns_cols
                    assert "tenancy_mode" in ns_cols
                    for dropped in ("workspace_id", "slug", "name", "description", "previous_version_id"):
                        assert dropped not in ns_cols, f"{dropped} was not dropped"

                    # chunks/entities must NOT have embedding column on SQLite — LanceDB owns it.
                    result = await conn.execute(sa.text("PRAGMA table_info(chunks)"))
                    chunk_cols = {r[1] for r in result}
                    assert "embedding" not in chunk_cols
                    # occurred_at (046) is added on both dialects.
                    assert "occurred_at" in chunk_cols
                    result = await conn.execute(sa.text("PRAGMA table_info(entities)"))
                    entity_cols = {r[1] for r in result}
                    assert "embedding" not in entity_cols
            finally:
                await engine.dispose()

        import asyncio

        asyncio.run(check())

    def test_fts5_triggers_wired(self, sqlite_url: str) -> None:
        """chunks_fts triggers must keep FTS index in sync with chunks table."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.begin() as conn:
                    result = await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name")
                    )
                    triggers = {r[0] for r in result}
                    assert {"chunks_ai", "chunks_ad", "chunks_au"} <= triggers
            finally:
                await engine.dispose()

        import asyncio

        asyncio.run(check())

    def test_downgrade_to_base(self, sqlite_url: str) -> None:
        """Upgrade to head, then downgrade to base — must leave no migration tables."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    )
                    tables = {r[0] for r in result}
                    # Only alembic version + sqlite internal tables may remain.
                    core = {
                        "memory_namespaces",
                        "documents",
                        "chunks",
                        "entities",
                        "relationships",
                        "episodes",
                        "chronicle_events",
                        "memory_facts",
                    }
                    leftover = core & tables
                    assert not leftover, f"Downgrade left core tables behind: {leftover}"
            finally:
                await engine.dispose()

        import asyncio

        asyncio.run(check())
