"""``056_documents_created_at_not_null`` — SQLite lane (primary coverage).

056 does two things to ``documents``, and both need rows present to be
observable at all:

1. Backfills ``created_at`` — from ``updated_at`` where that is available, and
   from the Unix epoch where it is not.
2. Flips ``created_at`` to NOT NULL, which on SQLite means
   ``op.batch_alter_table``'s table rebuild rather than an ``ALTER COLUMN``.

Step 1 is the one that needs a dedicated module: it is an irreversible data
transformation, and every other test that touches the chain builds an *empty*
database, so both ``UPDATE``s match zero rows everywhere and replacing their
bodies with a no-op would leave the whole suite green. The sibling migration
modules (041 / 044 / 049 / 052 / 053 / 055) each seed rows for exactly this
reason.

This lane drives the real rebuild and needs no server, so it lives in the unit
job and runs on every PR. The Postgres lane — the ``ALTER TABLE ... SET NOT
NULL`` branch — is
``tests/integration/db/test_migration_056_documents_created_at.py``. See
``tests/test_helpers/documents_created_at.py`` for the three archetypes both
lanes seed and for why the two lanes are separate modules.

Two things here are not about 056 in particular and are worth naming:

* ``TestEpochIsBoundNotInlined`` compiles against the asyncpg dialect and needs
  no server, so it sits in this lane rather than the Postgres one. It pins the
  property the epoch's *bound parameter* carries, which is a correctness
  requirement rather than a style preference — see the revision docstring.
* The chain-wide populated-database regression test lives in
  ``tests/unit/db/test_migration_chain_populated_sqlite.py``. The batch rebuild
  056 performs is only non-destructive because FK enforcement is off for the
  migration connection, and that guarantee spans the whole chain rather than
  this revision, so it is gated in its own module.
"""

from __future__ import annotations

import ast
import asyncio
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.script import ScriptDirectory
from loguru import logger
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_helpers.documents_created_at import (
    EPOCH,
    EXISTING_CREATED_AT,
    HEAD_REVISION,
    ID_INFERRED,
    ID_INVENTED,
    ID_UNTOUCHED,
    NS,
    PREV_REVISION,
    UPDATED_AT,
    bind_id,
    insert_document,
    make_config,
    read_created_at,
    seed_rows,
)

pytestmark = pytest.mark.unit

_REVISION_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "khora"
    / "db"
    / "migrations"
    / "versions"
    / "056_documents_created_at_not_null.py"
)

#: What every other writer of this column produces on SQLite: a space
#: separator, no timezone suffix. The column is compared as *text*, so a row
#: written in any other shape sorts wrong against its neighbours.
_SQLITE_DATETIME_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$")


def _code_string_constants(source: str) -> list[str]:
    """Every string literal in *source* except the docstrings.

    ``ast`` rather than a substring scan, because the revision's docstring
    legitimately quotes the literal spelling it forbids.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _seed_sqlite(url: str) -> None:
    """Bring a fresh SQLite file to the previous revision and seed the archetypes.

    ``command.upgrade`` must be called from OUTSIDE a running event loop — the
    bundled ``env.py`` drives the async migration with ``asyncio.run``, which
    raises if a loop is already running. Hence the sync wrapper around an
    ``asyncio.run`` for the inserts only.
    """
    command.upgrade(make_config(url), PREV_REVISION)
    asyncio.run(seed_rows(url, str(NS)))


@contextmanager
def _capture_migration_events() -> Iterator[list[dict[str, Any]]]:
    """Capture loguru records (not formatted strings) emitted at INFO+.

    The migration reports its two backfill counts via
    ``logger.bind(...).info("khora.migration.applied")``; the full record dict
    is what makes ``record["extra"]`` assertable.
    """
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(dict(message.record))

    handler_id = logger.add(_sink, level="INFO", format="{message}")
    try:
        yield records
    finally:
        logger.remove(handler_id)


async def _created_at_is_nullable(url: str) -> bool:
    """True when a NULL ``created_at`` insert is accepted."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await insert_document(conn, str(uuid4()), str(NS), None, UPDATED_AT)
        return True
    except IntegrityError:
        return False
    finally:
        await engine.dispose()


async def _read_created_at_raw(url: str, ids: list[Any]) -> dict[Any, Any]:
    """Read ``created_at`` as the driver returns it — a string on SQLite.

    Deliberately *not* routed through a ``DateTime`` result processor: the
    stored text is the thing under test, and the processor would normalize
    away exactly the divergence being looked for.
    """
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return {
                doc_id: (
                    await conn.execute(
                        sa.text("SELECT created_at FROM documents WHERE id = :id"),
                        {"id": bind_id(url, doc_id)},
                    )
                ).scalar()
                for doc_id in ids
            }
    finally:
        await engine.dispose()


def _documents_index_sql(url: str) -> dict[str, str]:
    """Every index on ``documents``, name → the SQL that defines it.

    Read from ``sqlite_master`` rather than ``PRAGMA index_list`` on purpose.
    ``index_list`` enumerates in reverse-creation order, so a table rebuild
    renumbers its ``seq`` column and a naive before/after comparison of its
    output is flaky by construction while the index *set* is unchanged. The
    name → SQL mapping has no such ordering component, and comparing the SQL
    text also covers the two partial indexes and the unique one, whose
    predicates a column-name comparison would drop.
    """
    engine = sa.create_engine(url.replace("sqlite+aiosqlite", "sqlite"))
    try:
        with engine.connect() as conn:
            return {
                name: sql
                for name, sql in conn.execute(
                    sa.text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'")
                )
                if sql is not None  # implicit indexes (UNIQUE constraints) carry no SQL
            }
    finally:
        engine.dispose()


def _foreign_keys(url: str, table: str) -> set[tuple[str, str, str, str]]:
    """``(from_column, referenced_table, referenced_column, on_delete)`` for *table*."""
    engine = sa.create_engine(url.replace("sqlite+aiosqlite", "sqlite"))
    try:
        with engine.connect() as conn:
            return {
                # ``from`` is a keyword, so the mapping view is the only way
                # to reach that column of the PRAGMA's result.
                (row._mapping["from"], row._mapping["table"], row._mapping["to"], row._mapping["on_delete"])
                for row in conn.execute(sa.text(f"PRAGMA foreign_key_list({table})"))  # noqa: S608
            }
    finally:
        engine.dispose()


class TestMigration056OnSqlite:
    @pytest.fixture
    def sqlite_url(self, tmp_path: Path) -> str:
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_backfill_outcome_per_archetype(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)

        # Precondition: the two NULLs really are NULL before the upgrade.
        # Without this the whole test could pass against a database where the
        # server_default had already filled the column in.
        before = asyncio.run(read_created_at(sqlite_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        assert before[ID_INFERRED] is None
        assert before[ID_INVENTED] is None

        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        after = asyncio.run(read_created_at(sqlite_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        # SQLite stores DATETIME without an offset, so the round trip is naive
        # on this leg; compare against the naive form of each expected instant.
        assert after[ID_INFERRED] == UPDATED_AT.replace(tzinfo=None), "the NULL row was not inferred from updated_at"
        assert after[ID_INVENTED] == EPOCH.replace(tzinfo=None), "the doubly-NULL row was not epoch-stamped"
        assert after[ID_UNTOUCHED] == EXISTING_CREATED_AT.replace(tzinfo=None), "the backfill flattened a real value"

    def test_reported_counts_match_the_seeded_archetypes(self, sqlite_url: str) -> None:
        """The two counts are the operator's audit trail, so they are asserted.

        ``rows_epoch_stamped`` is the only place the migration *invents* data.
        A single ``COALESCE`` would produce identical rows and lose exactly
        this distinction, which is the whole reason the backfill is two
        statements.
        """
        _seed_sqlite(sqlite_url)

        with _capture_migration_events() as records:
            command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1, f"expected exactly one applied event, got {[r['message'] for r in records]}"
        extra = applied[0]["extra"]
        assert extra["migration_id"] == HEAD_REVISION
        assert extra["rows_backfilled_from_updated_at"] == 1
        assert extra["rows_epoch_stamped"] == 1
        assert extra["lock_timeout_tripped"] is False

    def test_epoch_row_is_stored_in_the_same_format_as_a_normal_row(self, sqlite_url: str) -> None:
        """The highest-value assertion in this module.

        SQLite has no datetime type — ``created_at`` is TEXT under the hood and
        is compared lexicographically. An epoch written as the raw literal
        ``'1970-01-01T00:00:00+00:00'`` would satisfy every value-equality
        assertion above and still be wrong twice over: ``'T'`` (0x54) sorts
        above ``' '`` (0x20), so two rows at the same instant would order
        differently in the very column this revision exists to make orderable;
        and the trailing offset makes the row round-trip tz-**aware** while
        every other row round-trips naive, so a Python-side comparison mixing
        them raises ``TypeError``.

        Both halves are checked against the untouched row, which was written
        through the same ``DateTime`` bind processor ``DocumentModel`` uses —
        so this compares the migration against the ORM rather than against the
        test's own formatting.
        """
        _seed_sqlite(sqlite_url)
        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        raw = asyncio.run(_read_created_at_raw(sqlite_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        reference = raw[ID_UNTOUCHED]
        assert _SQLITE_DATETIME_TEXT.match(reference), f"reference row is not in the expected shape: {reference!r}"
        for doc_id in (ID_INFERRED, ID_INVENTED):
            assert _SQLITE_DATETIME_TEXT.match(raw[doc_id]), (
                f"{doc_id} was stored as {raw[doc_id]!r}, which is not the format every other row uses — "
                f"the epoch must bind through a typed parameter, never a string literal"
            )

        # Aware/naive parity: the epoch row must be directly comparable with an
        # ORM-written one. Under the literal spelling this line raises.
        after = asyncio.run(read_created_at(sqlite_url, [ID_INVENTED, ID_UNTOUCHED]))
        assert (after[ID_INVENTED].tzinfo is None) == (after[ID_UNTOUCHED].tzinfo is None)
        assert after[ID_INVENTED] < after[ID_UNTOUCHED]

    def test_not_null_is_enforced_after_upgrade(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        assert asyncio.run(_created_at_is_nullable(sqlite_url)) is True, (
            f"precondition: created_at must be NULLABLE at revision {PREV_REVISION}"
        )

        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        assert asyncio.run(_created_at_is_nullable(sqlite_url)) is False

    def test_rebuild_preserves_the_index_set(self, sqlite_url: str) -> None:
        """056 creates and drops no index, so the set must be byte-identical.

        This is the assertion that would catch a rebuild losing an index the
        chain built earlier — including the two partial indexes and the unique
        one, whose predicates live only in the SQL text.
        """
        _seed_sqlite(sqlite_url)
        before = _documents_index_sql(sqlite_url)
        assert before, "precondition: documents carries indexes at the previous revision"

        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        assert _documents_index_sql(sqlite_url) == before

    def test_rebuild_preserves_foreign_keys_in_both_directions(self, sqlite_url: str) -> None:
        """The outbound FK and the inbound cascade both have to survive.

        The inbound one is the reason the rebuild is dangerous at all:
        ``chunks.document_id`` cascades on delete, which is what empties
        ``chunks`` if FK enforcement is left on for the migration connection.
        Losing the constraint entirely would be the opposite failure and is
        equally worth catching.
        """
        _seed_sqlite(sqlite_url)
        documents_before = _foreign_keys(sqlite_url, "documents")
        chunks_before = _foreign_keys(sqlite_url, "chunks")
        assert ("document_id", "documents", "id", "CASCADE") in chunks_before, (
            "precondition: chunks.document_id cascades from documents"
        )

        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        assert _foreign_keys(sqlite_url, "documents") == documents_before
        assert _foreign_keys(sqlite_url, "chunks") == chunks_before

    def test_rebuild_preserves_the_server_default(self, sqlite_url: str) -> None:
        """An omitted ``created_at`` must still be filled by the column default.

        056 deliberately omits ``existing_server_default`` and relies on batch
        mode reflecting the live table. That is a claim about behaviour, so it
        is asserted by inserting rather than by reading the DDL: the default
        has to actually fire.
        """
        _seed_sqlite(sqlite_url)
        command.upgrade(make_config(sqlite_url), HEAD_REVISION)

        doc_id = str(uuid4())

        async def _insert_without_created_at() -> Any:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, status, source_type) "
                            "VALUES (:id, :ns, 'body', 'completed', 'library')"
                        ),
                        {"id": doc_id, "ns": str(NS)},
                    )
                async with engine.connect() as conn:
                    return (
                        await conn.execute(sa.text("SELECT created_at FROM documents WHERE id = :id"), {"id": doc_id})
                    ).scalar()
            finally:
                await engine.dispose()

        assert asyncio.run(_insert_without_created_at()) is not None

    def test_downgrade_restores_nullability_but_not_the_backfill(self, sqlite_url: str) -> None:
        _seed_sqlite(sqlite_url)
        cfg = make_config(sqlite_url)
        command.upgrade(cfg, HEAD_REVISION)

        command.downgrade(cfg, PREV_REVISION)

        assert asyncio.run(_created_at_is_nullable(sqlite_url)) is True, "NOT NULL was not dropped"

        # The backfill is one-way by design. Asserting it pins the documented
        # irreversibility rather than leaving it as a docstring claim.
        after = asyncio.run(read_created_at(sqlite_url, [ID_INFERRED, ID_INVENTED]))
        assert after[ID_INFERRED] == UPDATED_AT.replace(tzinfo=None)
        assert after[ID_INVENTED] == EPOCH.replace(tzinfo=None)

    def test_upgrade_downgrade_upgrade_round_trips(self, sqlite_url: str) -> None:
        """Two rebuilds in a row must leave the table where one did.

        The second upgrade's backfill matches zero rows — there is nothing left
        to repair — so this also covers the no-op path through ``_upgrade_impl``
        that a fresh database always takes.
        """
        _seed_sqlite(sqlite_url)
        cfg = make_config(sqlite_url)
        command.upgrade(cfg, HEAD_REVISION)
        indexes = _documents_index_sql(sqlite_url)
        chunks_fks = _foreign_keys(sqlite_url, "chunks")

        command.downgrade(cfg, PREV_REVISION)
        command.upgrade(cfg, HEAD_REVISION)

        assert asyncio.run(_created_at_is_nullable(sqlite_url)) is False
        assert _documents_index_sql(sqlite_url) == indexes
        assert _foreign_keys(sqlite_url, "chunks") == chunks_fks
        after = asyncio.run(read_created_at(sqlite_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        assert after[ID_INFERRED] == UPDATED_AT.replace(tzinfo=None)
        assert after[ID_INVENTED] == EPOCH.replace(tzinfo=None)
        assert after[ID_UNTOUCHED] == EXISTING_CREATED_AT.replace(tzinfo=None)


def _revision_module() -> Any:
    """The loaded 056 revision module, via alembic's own script directory."""
    return ScriptDirectory.from_config(make_config("sqlite://")).get_revision(HEAD_REVISION).module


class TestLockTimeoutClassificationAndErrorPath:
    """The failure branch of ``upgrade()``, which no lifecycle test reaches.

    Both halves are pure logic and need no server, but neither is exercised by
    a successful migration — so without this class the ``except DBAPIError``
    handler and every branch of ``_is_lock_timeout`` ship untested.

    The asyncpg case is the one that matters. ``_is_lock_timeout`` reads
    ``sqlstate`` *and* ``pgcode`` because the two drivers disagree: asyncpg
    carries the code on ``sqlstate`` and leaves ``pgcode`` as ``None``, while
    psycopg uses ``pgcode``. ``env.py`` normalizes Postgres URLs to
    ``postgresql+asyncpg``, so a ``pgcode``-only check would be dead on the
    only driver khora actually runs — the exact regression the asyncpg case
    below pins.
    """

    @staticmethod
    def _dbapi_error(*, sqlstate: Any = None, pgcode: Any = None) -> DBAPIError:
        return DBAPIError("ALTER TABLE documents ...", {}, SimpleNamespace(sqlstate=sqlstate, pgcode=pgcode))

    def test_asyncpg_shape_is_recognised(self) -> None:
        """asyncpg: code on ``sqlstate``, ``pgcode`` left None."""
        module = _revision_module()
        assert module._is_lock_timeout(self._dbapi_error(sqlstate="55P03", pgcode=None)) is True

    def test_psycopg_shape_is_recognised(self) -> None:
        module = _revision_module()
        assert module._is_lock_timeout(self._dbapi_error(pgcode="55P03")) is True

    def test_a_different_sqlstate_is_not_a_lock_timeout(self) -> None:
        """A unique violation must not be misreported as a lock timeout."""
        module = _revision_module()
        assert module._is_lock_timeout(self._dbapi_error(sqlstate="23505")) is False

    def test_missing_orig_is_not_a_lock_timeout(self) -> None:
        module = _revision_module()
        error = DBAPIError("stmt", {}, None)
        assert module._is_lock_timeout(error) is False

    def test_upgrade_logs_at_error_and_reraises(self) -> None:
        """Nothing is swallowed: the handler logs and bare-``raise``s.

        Also pins that the error event carries the same field set as the
        success path — the counts are initialized before the try block
        precisely so a failure does not emit a differently-shaped event.
        """
        module = _revision_module()
        failure = self._dbapi_error(sqlstate="55P03", pgcode=None)

        def _boom() -> tuple[int, int]:
            raise failure

        original = module._upgrade_impl
        module._upgrade_impl = _boom
        try:
            with _capture_migration_events() as records:
                with pytest.raises(DBAPIError) as caught:
                    module.upgrade()
        finally:
            module._upgrade_impl = original

        assert caught.value is failure, "the original exception must propagate, not a wrapped one"

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        assert applied[0]["level"].name == "ERROR"
        extra = applied[0]["extra"]
        assert extra["lock_timeout_tripped"] is True
        assert extra["migration_id"] == HEAD_REVISION
        assert extra["rows_backfilled_from_updated_at"] == 0
        assert extra["rows_epoch_stamped"] == 0


class TestEpochIsBoundNotInlined:
    """Pin the epoch's binding without needing a Postgres server.

    Compiling against ``postgresql.asyncpg.dialect()`` is a pure rendering
    question, so these belong in the lane that runs on every PR rather than in
    the integration job.
    """

    @staticmethod
    def _revision_module() -> Any:
        return _revision_module()

    @staticmethod
    def _compiled_epoch_update(epoch: Any, *, typed: bool = True) -> Any:
        bind = (
            sa.bindparam("epoch", value=epoch, type_=sa.DateTime(timezone=True))
            if typed
            else sa.bindparam("epoch", value=epoch)
        )
        stmt = sa.text("UPDATE documents SET created_at = :epoch WHERE created_at IS NULL").bindparams(bind)
        return stmt.compile(dialect=postgresql.asyncpg.dialect())

    def test_the_revision_binds_the_epoch_rather_than_interpolating_it(self) -> None:
        """Source-level guard: no date literal may appear in the revision's code.

        The compile assertions below can only test a statement this test file
        reconstructs. This one reads the revision itself, so it is what
        actually fails if someone replaces the bindparam with a literal.

        Docstrings are excluded, and not merely for convenience: the revision's
        own docstring quotes ``'1970-01-01T00:00:00+00:00'`` as the spelling to
        avoid, so a plain substring scan would fail on the documentation of the
        very rule it is enforcing.
        """
        source = _REVISION_SOURCE.read_text()
        assert 'bindparam("epoch"' in source, "the epoch is no longer a bound parameter"
        assert ":epoch" in source, "the UPDATE no longer references the bound parameter"

        offenders = [value for value in _code_string_constants(source) if "1970" in value]
        assert not offenders, (
            f"a date literal appeared in the revision's code: {offenders}. The epoch must be a typed "
            f"bindparam, never a string literal — see the revision docstring for what a literal breaks."
        )

    def test_the_epoch_is_never_inlined_into_the_asyncpg_sql(self) -> None:
        compiled = self._compiled_epoch_update(self._revision_module()._EPOCH)

        assert "1970" not in str(compiled), f"the epoch was inlined into the SQL text: {compiled}"
        assert compiled.params["epoch"] == EPOCH

    def test_the_asyncpg_render_carries_an_explicit_timestamptz_cast(self) -> None:
        """The guard ``type_=sa.DateTime(timezone=True)`` exists to hold.

        With ``_EPOCH`` tz-aware as shipped, SQLAlchemy infers the same type
        and the typed and untyped renders are identical — so this test does
        **not** fail if only the ``type_=`` is dropped, and that is correct.
        It fails when ``_EPOCH`` loses its ``tzinfo``, which is the state in
        which an untyped bind would silently render ``WITHOUT TIME ZONE``
        against a ``timestamptz`` column. The final assertion below records
        that mechanism directly so the guard's purpose is not lost.
        """
        epoch = self._revision_module()._EPOCH
        assert epoch.tzinfo is not None, "_EPOCH must stay tz-aware"
        assert "TIMESTAMP WITH TIME ZONE" in str(self._compiled_epoch_update(epoch))

        naive = epoch.replace(tzinfo=None)
        assert "TIMESTAMP WITHOUT TIME ZONE" in str(self._compiled_epoch_update(naive, typed=False))
        assert "TIMESTAMP WITH TIME ZONE" in str(self._compiled_epoch_update(naive, typed=True))
