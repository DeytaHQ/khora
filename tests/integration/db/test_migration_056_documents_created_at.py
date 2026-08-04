"""``056_documents_created_at_not_null`` — PostgreSQL lane.

The SQLite lane is
``tests/unit/db/test_migration_056_documents_created_at.py`` and carries the
coverage rationale; it drives ``op.batch_alter_table``'s table rebuild and
needs no server. This module runs the same three archetypes against the
``ALTER TABLE ... SET NOT NULL`` branch, which is a genuinely different code
path — not the same test with a different DSN.

Scope is deliberately narrow: only what needs a real server. The epoch's
*rendering* is a pure compile question and is pinned in the SQLite lane against
``postgresql.asyncpg.dialect()`` at zero cost, so it is not repeated here. What
this lane adds is the round trip through asyncpg's binary encoder and Postgres
``timestamptz`` storage — the last unverified step — plus the ``SET NOT NULL``
branch itself, which the SQLite lane never reaches.

Every test owns a throwaway database (``tests/test_helpers/pg_scratch_db.py``).
These tests seed rows, upgrade, and in two cases downgrade, so running them
against the shared dev database would destroy a concurrent test's schema — and
under CI's ``--timeout-method=thread`` a kill skips ``finally``, which could
strand the shared database mid-rewind and fail every later test in the serial
job with the real cause several tests back. A leaked, uniquely named database
is the worst case here instead.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_056_documents_created_at.py \
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from loguru import logger
from sqlalchemy.exc import IntegrityError
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
    insert_document,
    make_config,
    read_created_at,
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


def _seed_pg(url: str) -> None:
    """Bring a fresh scratch database to the previous revision and seed it.

    Synchronous on purpose, mirroring the SQLite lane's seeder:
    ``command.upgrade`` must run OUTSIDE a running event loop, so the two async
    halves are driven by separate ``asyncio.run`` calls with the alembic step
    between them. The 055 lane records what happens when this is written as an
    ``async def`` and called without ``await`` — the coroutine is discarded, no
    seeding happens, and the tests silently assert against whatever schema the
    CI job had already migrated to.
    """
    asyncio.run(_prepare_scratch(url))
    command.upgrade(make_config(url), PREV_REVISION)
    asyncio.run(seed_rows(url, NS))


@contextmanager
def _capture_migration_events() -> Iterator[list[dict[str, Any]]]:
    """Capture loguru records emitted at INFO+, for the two backfill counts."""
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(dict(message.record))

    handler_id = logger.add(_sink, level="INFO", format="{message}")
    try:
        yield records
    finally:
        logger.remove(handler_id)


async def _created_at_is_nullable(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    sa.text(
                        "SELECT is_nullable FROM information_schema.columns "
                        "WHERE table_name = 'documents' AND column_name = 'created_at'"
                    )
                )
            ).scalar() == "YES"
    finally:
        await engine.dispose()


async def _explicit_null_insert_is_rejected(url: str) -> bool:
    """True when an explicit NULL ``created_at`` INSERT raises.

    Asserted alongside ``information_schema`` rather than instead of it: the
    catalog says the constraint is declared, this says it is enforced.
    """
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await insert_document(conn, uuid4(), NS, None, UPDATED_AT)
        return False
    except IntegrityError:
        return True
    finally:
        await engine.dispose()


@pytest.mark.skipif(not pg_reachable(), reason="PostgreSQL not reachable (run `make dev` first)")
class TestMigration056OnPostgres:
    """Same archetypes against ``ALTER TABLE ... SET NOT NULL``.

    Two conditions skip and no others: the server is unreachable, or the role
    lacks CREATEDB. Anything else re-raises — a skip reads as "fine here", so
    suppressing a real migration failure into one would turn this lane green
    for the wrong reason.
    """

    @pytest.fixture
    def pg_url(self) -> Iterator[str]:
        with scratch_database("mig056") as url:
            _seed_pg(url)
            yield url

    def test_backfill_outcome_per_archetype(self, pg_url: str) -> None:
        """Includes the epoch's full round trip through asyncpg and timestamptz.

        The compile-level guarantees (never inlined, explicit
        ``::TIMESTAMP WITH TIME ZONE`` cast) are pinned in the SQLite lane.
        This is the leg that could not be decided offline: that asyncpg's
        binary encoder and Postgres' storage return the same instant, tz-aware.
        """
        before = asyncio.run(read_created_at(pg_url, [ID_INFERRED, ID_INVENTED]))
        assert before[ID_INFERRED] is None
        assert before[ID_INVENTED] is None

        command.upgrade(make_config(pg_url), HEAD_REVISION)

        after = asyncio.run(read_created_at(pg_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        assert after[ID_INFERRED] == UPDATED_AT, "the NULL row was not inferred from updated_at"
        assert after[ID_INVENTED] == EPOCH, "the doubly-NULL row was not epoch-stamped"
        assert after[ID_UNTOUCHED] == EXISTING_CREATED_AT, "the backfill flattened a real value"
        # timestamptz round-trips aware. A naive value here would mean the
        # column or the bind lost its timezone somewhere in the encoder.
        assert after[ID_INVENTED].tzinfo is not None

    def test_reported_counts_match_the_seeded_archetypes(self, pg_url: str) -> None:
        with _capture_migration_events() as records:
            command.upgrade(make_config(pg_url), HEAD_REVISION)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1, f"expected exactly one applied event, got {[r['message'] for r in records]}"
        extra = applied[0]["extra"]
        assert extra["migration_id"] == HEAD_REVISION
        assert extra["rows_backfilled_from_updated_at"] == 1
        assert extra["rows_epoch_stamped"] == 1
        assert extra["lock_timeout_tripped"] is False

    def test_not_null_is_installed_and_enforced(self, pg_url: str) -> None:
        assert asyncio.run(_created_at_is_nullable(pg_url)) is True, (
            f"precondition: created_at must be NULLABLE at revision {PREV_REVISION}"
        )

        command.upgrade(make_config(pg_url), HEAD_REVISION)

        assert asyncio.run(_created_at_is_nullable(pg_url)) is False
        assert asyncio.run(_explicit_null_insert_is_rejected(pg_url)) is True

    def test_downgrade_restores_nullability_but_not_the_backfill(self, pg_url: str) -> None:
        cfg = make_config(pg_url)
        command.upgrade(cfg, HEAD_REVISION)

        command.downgrade(cfg, PREV_REVISION)

        assert asyncio.run(_created_at_is_nullable(pg_url)) is True, "NOT NULL was not dropped"
        # The backfill is one-way by design; asserting it pins the documented
        # irreversibility rather than leaving it as a docstring claim.
        after = asyncio.run(read_created_at(pg_url, [ID_INFERRED, ID_INVENTED]))
        assert after[ID_INFERRED] == UPDATED_AT
        assert after[ID_INVENTED] == EPOCH

    def test_upgrade_downgrade_upgrade_round_trips(self, pg_url: str) -> None:
        """The second upgrade's backfill matches zero rows.

        Nothing is left to repair, so this also covers the no-op path through
        ``_upgrade_impl`` that a fresh database always takes — on the branch
        where ``SET LOCAL lock_timeout`` runs.
        """
        cfg = make_config(pg_url)
        command.upgrade(cfg, HEAD_REVISION)
        command.downgrade(cfg, PREV_REVISION)

        with _capture_migration_events() as records:
            command.upgrade(cfg, HEAD_REVISION)

        assert asyncio.run(_created_at_is_nullable(pg_url)) is False
        assert asyncio.run(_explicit_null_insert_is_rejected(pg_url)) is True

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        assert applied[0]["extra"]["rows_backfilled_from_updated_at"] == 0
        assert applied[0]["extra"]["rows_epoch_stamped"] == 0

        after = asyncio.run(read_created_at(pg_url, [ID_INFERRED, ID_INVENTED, ID_UNTOUCHED]))
        assert after[ID_INFERRED] == UPDATED_AT
        assert after[ID_INVENTED] == EPOCH
        assert after[ID_UNTOUCHED] == EXISTING_CREATED_AT
