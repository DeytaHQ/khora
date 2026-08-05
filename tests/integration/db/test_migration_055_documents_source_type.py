"""``055_documents_source_type_alignment`` — PostgreSQL lane.

The SQLite lane is ``tests/unit/db/test_migration_055_documents_source_type.py``
and carries the coverage rationale; it drives ``op.batch_alter_table``'s table
copy and needs no server. This module runs the same three archetypes against
the ``ALTER TABLE ... SET NOT NULL`` branch, which is a genuinely different
code path — not the same test with a different DSN.

Every test owns a throwaway database (``tests/test_helpers/pg_scratch_db.py``).
These tests seed rows, upgrade, and in one case downgrade, so running them
against the shared dev database would destroy a concurrent test's schema — and
under CI's ``--timeout-method=thread`` a kill skips ``finally``, which could
strand the shared database mid-rewind and fail every later test in the serial
job with the real cause several tests back. A leaked, uniquely named database
is the worst case here instead.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_055_documents_source_type.py \
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_helpers.documents_source_type import (
    BACKFILL_VALUE,
    DECLARED_INDEX_COLUMNS,
    ID_EMPTY_STRING,
    ID_NULL,
    ID_REAL_VALUE,
    INDEX_NAME,
    NS,
    PREV_REVISION,
    REAL_VALUE,
    TARGET_REVISION,
    make_config,
    read_source_types,
    seed_rows,
)
from tests.test_helpers.pg_scratch_db import pg_reachable, scratch_database

pytestmark = pytest.mark.integration


async def _prepare_scratch(url: str) -> None:
    """Make a brand-new database ready for the chain.

    No ``DROP SCHEMA`` — the database was created seconds ago by
    ``scratch_database`` and is dropped afterwards, so there is nothing to
    wipe.

    ``khora_alembic_version`` is pre-created at VARCHAR(64) because alembic
    would otherwise create it at VARCHAR(32), and several revision ids in this
    chain are wider.
    """
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                sa.text(
                    "CREATE TABLE IF NOT EXISTS khora_alembic_version ("
                    "  version_num VARCHAR(64) NOT NULL,"
                    "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                    ")"
                )
            )
    finally:
        await engine.dispose()


def _seed_pg(url: str, *, pre_create_index: bool) -> None:
    """Bring a fresh scratch database to the previous revision and seed it.

    Synchronous on purpose, mirroring the SQLite lane's seeder:
    ``command.upgrade`` must run OUTSIDE a running event loop, so the two
    async halves are driven by separate ``asyncio.run`` calls with the alembic
    step between them. An earlier version of this function was ``async def``
    and was called without ``await``, so the coroutine was created and
    discarded — no preparation, no upgrade to the previous revision, no seeded
    rows — and the tests silently ran against whatever schema the CI job had
    already migrated to head. CI is where that surfaced; it passes locally
    either way, because locally the whole class skips.
    """
    asyncio.run(_prepare_scratch(url))
    command.upgrade(make_config(url), PREV_REVISION)
    asyncio.run(seed_rows(url, NS, pre_create_index=pre_create_index))


async def _pg_source_type_is_nullable(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    sa.text(
                        "SELECT is_nullable FROM information_schema.columns "
                        "WHERE table_name = 'documents' AND column_name = 'source_type'"
                    )
                )
            ).scalar() == "YES"
    finally:
        await engine.dispose()


async def _pg_index_columns(url: str) -> list[str] | None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda sync_conn: next(
                    (
                        list(idx["column_names"] or [])
                        for idx in sa.inspect(sync_conn).get_indexes("documents")
                        if idx["name"] == INDEX_NAME
                    ),
                    None,
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.skipif(not pg_reachable(), reason="PostgreSQL not reachable (run `make dev` first)")
class TestMigration055OnPostgres:
    """Same archetypes against ``ALTER TABLE ... SET NOT NULL``.

    Two conditions skip and no others: the server is unreachable, or the role
    lacks CREATEDB. Anything else re-raises — a skip reads as "fine here", so
    suppressing a real migration failure into one would turn this lane green
    for the wrong reason.
    """

    @pytest.fixture
    def pg_url(self) -> Iterator[str]:
        with scratch_database("mig055") as url:
            _seed_pg(url, pre_create_index=False)
            yield url

    @pytest.fixture
    def pg_url_preindexed(self) -> Iterator[str]:
        with scratch_database("mig055_preindexed") as url:
            _seed_pg(url, pre_create_index=True)
            yield url

    def test_backfill_rewrites_only_the_null_row(self, pg_url: str) -> None:
        before = asyncio.run(read_source_types(pg_url, [ID_NULL]))
        assert before[ID_NULL] is None

        command.upgrade(make_config(pg_url), TARGET_REVISION)

        after = asyncio.run(read_source_types(pg_url, [ID_NULL, ID_REAL_VALUE, ID_EMPTY_STRING]))
        assert after[ID_NULL] == BACKFILL_VALUE, "the NULL row was not backfilled"
        assert after[ID_REAL_VALUE] == REAL_VALUE, "the backfill flattened a real value"
        assert after[ID_EMPTY_STRING] == "", "the empty string was rewritten — 055 does not do that"

    def test_not_null_and_index_are_installed(self, pg_url: str) -> None:
        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is True, (
            f"precondition: source_type must be NULLABLE at revision {PREV_REVISION}"
        )

        command.upgrade(make_config(pg_url), TARGET_REVISION)

        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is False
        assert asyncio.run(_pg_index_columns(pg_url)) == DECLARED_INDEX_COLUMNS

    def test_pre_existing_index_does_not_wedge_the_backfill(self, pg_url_preindexed: str) -> None:
        command.upgrade(make_config(pg_url_preindexed), TARGET_REVISION)

        after = asyncio.run(read_source_types(pg_url_preindexed, [ID_NULL]))
        assert after[ID_NULL] == BACKFILL_VALUE
        assert asyncio.run(_pg_source_type_is_nullable(pg_url_preindexed)) is False
        assert asyncio.run(_pg_index_columns(pg_url_preindexed)) == DECLARED_INDEX_COLUMNS

    def test_downgrade_drops_the_constraint_and_index_but_not_the_backfill(self, pg_url: str) -> None:
        cfg = make_config(pg_url)
        command.upgrade(cfg, TARGET_REVISION)

        command.downgrade(cfg, PREV_REVISION)

        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is True, "NOT NULL was not dropped"
        assert asyncio.run(_pg_index_columns(pg_url)) is None, "the index was not dropped"
        after = asyncio.run(read_source_types(pg_url, [ID_NULL]))
        assert after[ID_NULL] == BACKFILL_VALUE
