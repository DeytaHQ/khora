"""``055_documents_source_type_alignment`` — SQLite lane (primary coverage).

055 does three things to ``documents``, and until this module existed only
the first was observable anywhere in the suite:

1. Creates ``ix_documents_namespace_source_type`` (declared in
   ``DocumentModel.__table_args__``, never built by the chain) with
   ``if_not_exists=True``.
2. Backfills ``UPDATE documents SET source_type = 'library' WHERE
   source_type IS NULL``.
3. Flips ``source_type`` to NOT NULL.

Step 2 is the one that needed a dedicated module. It is an irreversible data
transformation, and every other test that touches 055 builds an empty
database — so the UPDATE matches zero rows everywhere and replacing its body
with a no-op leaves the whole suite green. The sibling migration tests
(041 / 044 / 049 / 052 / 053) each seed rows for exactly this reason.

This lane drives the real migration through ``op.batch_alter_table``'s table
copy and needs no server, so it lives in the unit job and runs on every PR.
The Postgres lane — the ``ALTER TABLE ... SET NOT NULL`` branch — is
``tests/integration/db/test_migration_055_documents_source_type.py``. The two
were one module until the split; see
``tests/test_helpers/documents_source_type.py`` for why that meant neither CI
job ran this class, and for the three archetypes both lanes seed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError
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
    insert_document,
    make_config,
    read_source_types,
    seed_rows,
)

pytestmark = pytest.mark.unit


def _seed_sqlite(url: str, *, pre_create_index: bool = False) -> None:
    """Bring a fresh SQLite file to the previous revision and seed the archetypes.

    ``command.upgrade`` must be called from OUTSIDE a running event loop —
    the bundled ``env.py`` drives the async migration with ``asyncio.run``,
    which raises if a loop is already running. Hence the sync wrapper around
    an ``asyncio.run`` for the inserts only.
    """
    command.upgrade(make_config(url), PREV_REVISION)
    asyncio.run(seed_rows(url, str(NS), pre_create_index=pre_create_index))


async def _sqlite_source_type_is_nullable(url: str) -> bool:
    """True when a NULL ``source_type`` insert is accepted."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await insert_document(conn, str(uuid4()), str(NS), None)
        return True
    except IntegrityError:
        return False
    finally:
        await engine.dispose()


def _documents_indexes(url: str) -> dict[str, list[str]]:
    engine = sa.create_engine(url.replace("sqlite+aiosqlite", "sqlite"))
    try:
        return {idx["name"]: list(idx["column_names"] or []) for idx in sa.inspect(engine).get_indexes("documents")}
    finally:
        engine.dispose()


class TestMigration055OnSqlite:
    @pytest.fixture
    def sqlite_url(self, tmp_path: Path) -> str:
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_backfill_rewrites_only_the_null_row(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)

        # Precondition: the NULL really is NULL before the upgrade. Without
        # this the whole test could pass against a database where the
        # server_default already filled the column in.
        before = asyncio.run(read_source_types(sqlite_url, [ID_NULL]))
        assert before[ID_NULL] is None

        command.upgrade(make_config(sqlite_url), TARGET_REVISION)

        after = asyncio.run(read_source_types(sqlite_url, [ID_NULL, ID_REAL_VALUE, ID_EMPTY_STRING]))
        assert after[ID_NULL] == BACKFILL_VALUE, "the NULL row was not backfilled"
        assert after[ID_REAL_VALUE] == REAL_VALUE, "the backfill flattened a real value"
        # NOT NULL does not rule out '' and 055 deliberately adds no CHECK.
        assert after[ID_EMPTY_STRING] == "", "the empty string was rewritten — 055 does not do that"

    def test_not_null_is_enforced_after_upgrade(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is True, (
            f"precondition: source_type must be NULLABLE at revision {PREV_REVISION}"
        )

        command.upgrade(make_config(sqlite_url), TARGET_REVISION)

        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is False

    def test_index_is_created_over_the_declared_columns(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        command.upgrade(make_config(sqlite_url), TARGET_REVISION)

        assert _documents_indexes(sqlite_url).get(INDEX_NAME) == DECLARED_INDEX_COLUMNS

    def test_pre_existing_index_does_not_wedge_the_backfill(self, sqlite_url: str) -> None:
        """The realistic operator state: optimize_storage() already ran.

        ``tests/unit/test_migration_drift.py`` covers ``if_not_exists`` on an
        empty database. This is the same collision with rows present, so the
        backfill and the NOT NULL flip still have to complete after it.
        """
        _seed_sqlite(sqlite_url, pre_create_index=True)

        command.upgrade(make_config(sqlite_url), TARGET_REVISION)

        after = asyncio.run(read_source_types(sqlite_url, [ID_NULL]))
        assert after[ID_NULL] == BACKFILL_VALUE
        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is False

        names = list(_documents_indexes(sqlite_url))
        assert names.count(INDEX_NAME) == 1

    def test_downgrade_drops_the_constraint_and_index_but_not_the_backfill(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        cfg = make_config(sqlite_url)
        command.upgrade(cfg, TARGET_REVISION)

        command.downgrade(cfg, PREV_REVISION)

        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is True, "NOT NULL was not dropped"
        assert INDEX_NAME not in _documents_indexes(sqlite_url), "the index was not dropped"

        # The backfill is one-way by design — the rewritten row stays
        # 'library'. Asserting it pins the documented irreversibility rather
        # than leaving it as a docstring claim.
        after = asyncio.run(read_source_types(sqlite_url, [ID_NULL]))
        assert after[ID_NULL] == BACKFILL_VALUE
