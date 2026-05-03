"""Shared fixtures for chronicle engine unit tests.

The lancedb-backed tests in ``test_lancedb_backend.py`` build a SQLite
DB on disk and let ``ChronicleEngine.connect()`` migrate it. The fresh
migration costs ~150-200ms; running it 5+ times per file (and again
across other chronicle tests) adds up. We pre-seed the DB file with a
worker-scoped migrated template so ``run_migrations`` runs as a fast
no-op (~20ms) instead.

Under ``pytest-xdist`` each worker gets its own session, so the
template lives in the worker's own ``tmp_path_factory`` directory.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False


if _HAS_EMBEDDED:
    from khora.db.session import run_migrations


@pytest.fixture(scope="session")
def _chronicle_sqlite_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fully-migrated SQLite DB once per worker session."""
    if not _HAS_EMBEDDED:
        pytest.skip("aiosqlite/lancedb not installed")
    template_dir = tmp_path_factory.mktemp("chronicle_sqlite_template")
    template_path = template_dir / "template.db"

    async def _migrate() -> None:
        result = await run_migrations(f"sqlite+aiosqlite:///{template_path}")
        if not result.success:
            raise RuntimeError(f"chronicle template migration failed: {result.error}")

    asyncio.run(_migrate())
    return template_path


@pytest.fixture
def chronicle_sqlite_db(_chronicle_sqlite_template: Path, tmp_path: Path) -> Path:
    """Copy the worker's migrated template into ``tmp_path/chronicle.db``.

    The chronicle engine still calls ``run_migrations`` during connect(),
    but on a pre-migrated DB that's a no-op (~20ms vs 200ms fresh).
    """
    target = tmp_path / "chronicle.db"
    shutil.copy(_chronicle_sqlite_template, target)
    return target
