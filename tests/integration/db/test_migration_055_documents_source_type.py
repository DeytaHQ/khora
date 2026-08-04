"""Coverage for migration ``055_documents_source_type_alignment``.

055 does three things to ``documents``, and until this module existed only
the first was observable anywhere in the suite:

1. Creates ``ix_documents_namespace_source_type`` (declared in
   ``DocumentModel.__table_args__``, never built by the chain) with
   ``if_not_exists=True``.
2. Backfills ``UPDATE documents SET source_type = 'library' WHERE
   source_type IS NULL``.
3. Flips ``source_type`` to NOT NULL.

Step 2 is the one that needed a dedicated module. It is an irreversible
data transformation, and every other test that touches 055 builds an empty
database — so the UPDATE matches zero rows everywhere and replacing its
body with a no-op leaves the whole suite green. The sibling migration
tests (041 / 044 / 049 / 052 / 053) each seed rows for exactly this
reason; this module follows their shape.

Archetypes seeded at the previous revision and asserted after the upgrade:

* ``NULL`` → rewritten to ``'library'`` (the backfill; the only row it
  touches).
* ``'slack'`` → untouched. The backfill must not flatten real values.
* ``''`` → **stays** ``''``. NOT NULL does not rule out the degenerate
  empty string, and 055 deliberately does not add a ``CHECK (source_type
  <> '')`` — see the migration docstring. This assertion is what keeps
  that documented limitation honest.

Plus the three structural properties: the NOT NULL rejects a NULL insert
afterwards, the index lands over the declared columns, and a database that
already carries the index (``optimize_storage()`` ships the same
``CREATE INDEX IF NOT EXISTS``) migrates without raising.

Downgrade is covered too — nothing previously asserted that it drops the
constraint or the index, or that it leaves the backfilled rows rewritten
(it cannot restore them; a row that was NULL is now indistinguishable from
one that always said ``'library'``).

The SQLite lane runs the real migration through ``op.batch_alter_table``'s
table copy, needs no Docker, and is marked ``unit`` — it is the primary
coverage. The Postgres lane runs the same archetypes against the
``ALTER TABLE ... SET NOT NULL`` branch and self-skips when Postgres is
unreachable; it is CI-only, and did not run when this module was written.

Run the Postgres lane locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_055_documents_source_type.py \
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from tests.test_helpers.pg_scratch_db import pg_reachable, scratch_database

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

_HEAD_REVISION = "055_documents_source_type_alignment"
_PREV_REVISION = "054_documents_namespace_created_at_id"

_INDEX_NAME = "ix_documents_namespace_source_type"
_DECLARED_INDEX_COLUMNS = ["namespace_id", "source_type"]

# The same CREATE the optimize_storage() catch-up list ships.
_OPTIMIZE_CREATE_INDEX = f"CREATE INDEX IF NOT EXISTS {_INDEX_NAME} ON documents (namespace_id, source_type)"  # noqa: S608

_NS = UUID("00000000-0000-0000-0000-0000000000aa")

# One row per archetype, so each assertion names the case it covers.
_ID_NULL = UUID("00000000-0000-0000-0000-000000000001")
_ID_REAL_VALUE = UUID("00000000-0000-0000-0000-000000000002")
_ID_EMPTY_STRING = UUID("00000000-0000-0000-0000-000000000003")

_REAL_VALUE = "slack"
_BACKFILL_VALUE = "library"


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


async def _insert_namespace(conn: AsyncConnection, ns_id: UUID | str) -> None:
    await conn.execute(
        sa.text(
            "INSERT INTO memory_namespaces "
            "(id, namespace_id, version, is_active, created_at, updated_at) "
            "VALUES (:id, :id, 1, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": ns_id},
    )


async def _insert_document(
    conn: AsyncConnection, doc_id: UUID | str, ns_id: UUID | str, source_type: str | None
) -> None:
    """Insert a documents row with an explicit ``source_type`` (NULL included).

    The column carries ``server_default='library'``, so the value has to be
    named explicitly — omitting it would silently produce 'library' and the
    NULL archetype would never exist.
    """
    await conn.execute(
        sa.text(
            "INSERT INTO documents (id, namespace_id, content, status, source_type, created_at, updated_at) "
            "VALUES (:id, :ns, 'body', 'completed', :st, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": doc_id, "ns": ns_id, "st": source_type},
    )


async def _read_source_types(url: str, ids: list[UUID]) -> dict[UUID, str | None]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            out: dict[UUID, str | None] = {}
            for doc_id in ids:
                out[doc_id] = (
                    await conn.execute(
                        sa.text("SELECT source_type FROM documents WHERE id = :id"),
                        {"id": _bind_id(url, doc_id)},
                    )
                ).scalar()
            return out
    finally:
        await engine.dispose()


def _bind_id(url: str, value: UUID) -> UUID | str:
    """SQLite stores the UUID columns as TEXT; Postgres wants the UUID."""
    return value if url.startswith("postgresql") else str(value)


# ---------------------------------------------------------------------------
# SQLite lane — the real migration through batch_alter_table (primary coverage)
# ---------------------------------------------------------------------------


def _seed_sqlite(url: str, *, pre_create_index: bool = False) -> None:
    """Bring a fresh SQLite file to the previous revision and seed the three archetype rows.

    ``command.upgrade`` must be called from OUTSIDE a running event loop —
    the bundled ``env.py`` drives the async migration with ``asyncio.run``,
    which raises if a loop is already running. Hence the sync wrapper around
    an ``asyncio.run`` for the inserts only.
    """
    command.upgrade(_make_config(url), _PREV_REVISION)
    asyncio.run(_seed_rows(url, str(_NS), pre_create_index=pre_create_index))


async def _seed_rows(url: str, ns_id: UUID | str, *, pre_create_index: bool) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await _insert_namespace(conn, ns_id)
            await _insert_document(conn, _bind_id(url, _ID_NULL), ns_id, None)
            await _insert_document(conn, _bind_id(url, _ID_REAL_VALUE), ns_id, _REAL_VALUE)
            await _insert_document(conn, _bind_id(url, _ID_EMPTY_STRING), ns_id, "")
            if pre_create_index:
                await conn.execute(sa.text(_OPTIMIZE_CREATE_INDEX))
    finally:
        await engine.dispose()


async def _sqlite_source_type_is_nullable(url: str) -> bool:
    """True when a NULL ``source_type`` insert is accepted."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await _insert_document(conn, str(uuid4()), str(_NS), None)
        return True
    except IntegrityError:
        return False
    finally:
        await engine.dispose()


@pytest.mark.unit
class TestMigration055OnSqlite:
    @pytest.fixture
    def sqlite_url(self, tmp_path: Path) -> str:
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_backfill_rewrites_only_the_null_row(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)

        # Precondition: the NULL really is NULL before the upgrade. Without
        # this the whole test could pass against a database where the
        # server_default already filled the column in.
        before = asyncio.run(_read_source_types(sqlite_url, [_ID_NULL]))
        assert before[_ID_NULL] is None

        command.upgrade(_make_config(sqlite_url), _HEAD_REVISION)

        after = asyncio.run(_read_source_types(sqlite_url, [_ID_NULL, _ID_REAL_VALUE, _ID_EMPTY_STRING]))
        assert after[_ID_NULL] == _BACKFILL_VALUE, "the NULL row was not backfilled"
        assert after[_ID_REAL_VALUE] == _REAL_VALUE, "the backfill flattened a real value"
        # NOT NULL does not rule out '' and 054 deliberately adds no CHECK.
        assert after[_ID_EMPTY_STRING] == "", "the empty string was rewritten — 054 does not do that"

    def test_not_null_is_enforced_after_upgrade(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is True, (
            "precondition: source_type must be NULLABLE at revision 053"
        )

        command.upgrade(_make_config(sqlite_url), _HEAD_REVISION)

        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is False

    def test_index_is_created_over_the_declared_columns(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        command.upgrade(_make_config(sqlite_url), _HEAD_REVISION)

        engine = sa.create_engine(sqlite_url.replace("sqlite+aiosqlite", "sqlite"))
        try:
            indexes = {
                idx["name"]: list(idx["column_names"] or []) for idx in sa.inspect(engine).get_indexes("documents")
            }
        finally:
            engine.dispose()

        assert indexes.get(_INDEX_NAME) == _DECLARED_INDEX_COLUMNS

    def test_pre_existing_index_does_not_wedge_the_backfill(self, sqlite_url: str) -> None:
        """The realistic operator state: optimize_storage() already ran.

        ``tests/unit/test_migration_drift.py`` covers ``if_not_exists`` on an
        empty database. This is the same collision with rows present, so the
        backfill and the NOT NULL flip still have to complete after it.
        """
        _seed_sqlite(sqlite_url, pre_create_index=True)

        command.upgrade(_make_config(sqlite_url), _HEAD_REVISION)

        after = asyncio.run(_read_source_types(sqlite_url, [_ID_NULL]))
        assert after[_ID_NULL] == _BACKFILL_VALUE
        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is False

        engine = sa.create_engine(sqlite_url.replace("sqlite+aiosqlite", "sqlite"))
        try:
            names = [idx["name"] for idx in sa.inspect(engine).get_indexes("documents")]
        finally:
            engine.dispose()
        assert names.count(_INDEX_NAME) == 1

    def test_downgrade_drops_the_constraint_and_index_but_not_the_backfill(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, _HEAD_REVISION)

        command.downgrade(cfg, _PREV_REVISION)

        assert asyncio.run(_sqlite_source_type_is_nullable(sqlite_url)) is True, "NOT NULL was not dropped"

        engine = sa.create_engine(sqlite_url.replace("sqlite+aiosqlite", "sqlite"))
        try:
            names = [idx["name"] for idx in sa.inspect(engine).get_indexes("documents")]
        finally:
            engine.dispose()
        assert _INDEX_NAME not in names, "the index was not dropped"

        # The backfill is one-way by design — the rewritten row stays
        # 'library'. Asserting it pins the documented irreversibility rather
        # than leaving it as a docstring claim.
        after = asyncio.run(_read_source_types(sqlite_url, [_ID_NULL]))
        assert after[_ID_NULL] == _BACKFILL_VALUE


# ---------------------------------------------------------------------------
# Postgres lane — the ALTER TABLE ... SET NOT NULL branch (skips without a bind)
# ---------------------------------------------------------------------------


async def _prepare_scratch(url: str) -> None:
    """Make a brand-new database ready for the chain.

    No ``DROP SCHEMA`` here, unlike the sibling migration tests: the database
    is created fresh by ``scratch_database`` and dropped afterwards, so there
    is nothing to wipe. That is the whole point — the previous version of this
    module reset ``public`` on the *shared* dev database, which destroys any
    concurrent test's schema and, under CI's ``--timeout-method=thread``, can
    leave the shared database stranded mid-reset because ``finally`` never
    runs.

    ``khora_alembic_version`` is pre-created at VARCHAR(64) because alembic
    would otherwise create it at VARCHAR(32) and several revision ids in this
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


async def _seed_pg_rows(url: str, *, pre_create_index: bool) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await _insert_namespace(conn, _NS)
            await _insert_document(conn, _ID_NULL, _NS, None)
            await _insert_document(conn, _ID_REAL_VALUE, _NS, _REAL_VALUE)
            await _insert_document(conn, _ID_EMPTY_STRING, _NS, "")
            if pre_create_index:
                await conn.execute(sa.text(_OPTIMIZE_CREATE_INDEX))
    finally:
        await engine.dispose()


def _seed_pg(url: str, *, pre_create_index: bool) -> None:
    """Bring a fresh scratch database to the previous revision and seed it.

    Synchronous on purpose, mirroring ``_seed_sqlite``: ``command.upgrade``
    must run OUTSIDE a running event loop, so the two async halves are driven
    by separate ``asyncio.run`` calls with the alembic step between them. An
    earlier version of this function was ``async def`` and was called without
    ``await``, so the coroutine was created and discarded — no preparation, no
    upgrade to the previous revision, no seeded rows — and the tests silently
    ran against whatever schema the CI job had already migrated to head. CI is
    where that surfaced; it passes locally either way, because locally the
    whole class skips.
    """
    asyncio.run(_prepare_scratch(url))
    command.upgrade(_make_config(url), _PREV_REVISION)
    asyncio.run(_seed_pg_rows(url, pre_create_index=pre_create_index))


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
                        if idx["name"] == _INDEX_NAME
                    ),
                    None,
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.skipif(not pg_reachable(), reason="PostgreSQL not reachable (run `make dev` first)")
class TestMigration055OnPostgres:
    """Same archetypes against ``ALTER TABLE ... SET NOT NULL``.

    Every test owns a throwaway database. These tests seed rows, upgrade, and
    in one case downgrade, so running them against the shared dev database
    would destroy a concurrent test's schema — and under CI's
    ``--timeout-method=thread`` a kill skips ``finally``, which could strand
    the shared database mid-rewind and fail every later test in the serial job
    with the real cause several tests back. A leaked, uniquely named database
    is the worst case here instead.

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
        before = asyncio.run(_read_source_types(pg_url, [_ID_NULL]))
        assert before[_ID_NULL] is None

        command.upgrade(_make_config(pg_url), _HEAD_REVISION)

        after = asyncio.run(_read_source_types(pg_url, [_ID_NULL, _ID_REAL_VALUE, _ID_EMPTY_STRING]))
        assert after[_ID_NULL] == _BACKFILL_VALUE, "the NULL row was not backfilled"
        assert after[_ID_REAL_VALUE] == _REAL_VALUE, "the backfill flattened a real value"
        assert after[_ID_EMPTY_STRING] == "", "the empty string was rewritten — 054 does not do that"

    def test_not_null_and_index_are_installed(self, pg_url: str) -> None:
        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is True, (
            "precondition: source_type must be NULLABLE at revision 053"
        )

        command.upgrade(_make_config(pg_url), _HEAD_REVISION)

        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is False
        assert asyncio.run(_pg_index_columns(pg_url)) == _DECLARED_INDEX_COLUMNS

    def test_pre_existing_index_does_not_wedge_the_backfill(self, pg_url_preindexed: str) -> None:
        command.upgrade(_make_config(pg_url_preindexed), _HEAD_REVISION)

        after = asyncio.run(_read_source_types(pg_url_preindexed, [_ID_NULL]))
        assert after[_ID_NULL] == _BACKFILL_VALUE
        assert asyncio.run(_pg_source_type_is_nullable(pg_url_preindexed)) is False
        assert asyncio.run(_pg_index_columns(pg_url_preindexed)) == _DECLARED_INDEX_COLUMNS

    def test_downgrade_drops_the_constraint_and_index_but_not_the_backfill(self, pg_url: str) -> None:
        cfg = _make_config(pg_url)
        command.upgrade(cfg, _HEAD_REVISION)

        command.downgrade(cfg, _PREV_REVISION)

        assert asyncio.run(_pg_source_type_is_nullable(pg_url)) is True, "NOT NULL was not dropped"
        assert asyncio.run(_pg_index_columns(pg_url)) is None, "the index was not dropped"
        after = asyncio.run(_read_source_types(pg_url, [_ID_NULL]))
        assert after[_ID_NULL] == _BACKFILL_VALUE
