"""Shared fixtures for sqlite_lance unit tests.

Each test in this directory needs a SQLite DB with the full Alembic
migration chain applied. Running ``run_migrations`` per test costs
~150-300ms — for ~110 tests that's 30+s of pure setup. We migrate
once per pytest worker and then ``shutil.copy`` the resulting file
into each test's ``tmp_path``. A migrated empty SQLite file is small
(<200 KB) so the copy is microseconds.

Under ``pytest-xdist`` each worker gets its own session, so the
template lives in the worker's own ``tmp_path_factory`` directory —
no cross-worker sharing, no locking.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from khora.db.session import run_migrations


@pytest.fixture(scope="session")
def _migrated_sqlite_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fully-migrated SQLite DB once per worker session.

    Returns the path to the template file. Tests must copy it — never
    open it directly, since aiosqlite would mutate the shared template.
    """
    template_dir = tmp_path_factory.mktemp("sqlite_lance_template")
    template_path = template_dir / "template.db"

    async def _migrate() -> None:
        result = await run_migrations(f"sqlite+aiosqlite:///{template_path}")
        if not result.success:
            raise RuntimeError(f"template migration failed: {result.error}")

    asyncio.run(_migrate())
    return template_path


@pytest.fixture
def migrated_sqlite_db(_migrated_sqlite_template: Path, tmp_path: Path) -> Path:
    """Copy the worker's migrated SQLite template into this test's tmp_path.

    Returns the path to a fresh, isolated DB ready for use. Each test
    gets its own copy so writes never cross test boundaries.
    """
    target = tmp_path / "khora.db"
    shutil.copy(_migrated_sqlite_template, target)
    return target
