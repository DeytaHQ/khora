"""Lifecycle coverage for migration ``054_documents_namespace_created_at_id``.

The migration widens the documents sort index from
``ix_documents_namespace_created_at (namespace_id, created_at)`` to
``ix_documents_namespace_created_at_id (namespace_id, created_at, id)`` so that
``list_documents``' pinned ``ORDER BY created_at DESC, id DESC`` is served by a
backward index scan instead of an index scan plus a sort.

The migration has two branches and both are exercised here. Postgres builds the
indexes with ``CREATE INDEX CONCURRENTLY`` inside an autocommit block; those
classes need a live server and skip when one is unreachable. Every other
dialect gets plain DDL, and :class:`TestMigration054SqliteLifecycle` drives
that branch on a throwaway SQLite file with no server at all - so the non-
Postgres path, which a Postgres-only reviewer never exercises, is covered on
every run.

What is verified here:

1. **Upgrade** builds the 3-column index with exactly those columns in exactly
   that order, and removes the 2-column index it supersedes (two indexes
   sharing a prefix would both be maintained on every insert - pure write
   amplification on the ingest path).
2. **Downgrade** is an exact mirror: the 2-column index comes back and the
   3-column one goes away.
3. **Downgrade stays compatible with the migration that first added the
   2-column index.** That earlier migration's ``downgrade()`` issues an
   UNCONDITIONAL ``op.drop_index("ix_documents_namespace_created_at", ...)`` -
   no ``if_exists=True``. If migration 054's downgrade failed to restore that index,
   any downgrade walking further back would abort on a missing index. This is
   the highest-risk failure mode in the change, so it is covered twice: once
   directly (the unconditional drop is replayed against the post-downgrade state)
   and once end-to-end (an actual downgrade walk past it, on a throwaway
   database).

No Postgres class touches the shared dev database. Both own a throwaway one,
created and dropped per module or per test. That is deliberate rather than
tidy: CI runs the integration job with ``--timeout-method=thread``, which kills
the process outright, so ``finally`` blocks do not run. A rewind-then-restore
against the shared database would strand it at the previous revision on
timeout, and every later test in the serial job would then fail against a stale
schema with the real cause several tests back. Owning the database bounds the
worst case to one leaked, uniquely named database.

Run explicitly (the shell may leak a different URL)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/db/test_migration_054_documents_namespace_created_at_id.py \
        -o addopts="" --no-cov -q
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_helpers.pg_scratch_db import (
    pg_reachable,
    scratch_database,
)

pytestmark = [pytest.mark.integration]

# Applied per class rather than module-wide: the SQLite lifecycle class below
# runs the same migration through its non-Postgres branch and needs no server.
skip_no_pg = pytest.mark.skipif(
    not pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

_HEAD = "054_documents_namespace_created_at_id"
_PREV = "053_khora_chunks_bookkeeping_to_chunker_info"

# The revision that first added the 2-column index, and the one immediately
# below it. Walking to ``_BELOW_ORIGIN`` forces that migration's unconditional
# ``drop_index`` to actually execute.
_ORIGIN = "019_document_last_activity_index"
_BELOW_ORIGIN = "018_halfvec_hnsw_indexes"

NEW_INDEX = "ix_documents_namespace_created_at_id"
OLD_INDEX = "ix_documents_namespace_created_at"


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


async def _documents_indexes(url: str) -> dict[str, str]:
    """Map index name -> index definition for the two indexes under test."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE tablename = 'documents' AND indexname = ANY(:names)"
                ),
                {"names": [NEW_INDEX, OLD_INDEX]},
            )
            return {row[0]: row[1] for row in result.fetchall()}
    finally:
        await engine.dispose()


def _indexed_columns(indexdef: str) -> list[str]:
    """Ordered column list out of a ``pg_indexes.indexdef`` string.

    Order is the whole point of this migration, so the columns are compared as
    a sequence; a set comparison would accept ``(namespace_id, id, created_at)``,
    which does not cover the query at all.
    """
    inner = indexdef[indexdef.index("(") + 1 : indexdef.rindex(")")]
    return [c.strip() for c in inner.split(",")]


@pytest.fixture(scope="module")
def scratch_db_url() -> Iterator[str]:
    """A throwaway Postgres database at head, for the rewind tests below.

    These tests downgrade and re-upgrade, so they must NOT run against the
    shared dev database. CI runs the integration job with
    ``--timeout-method=thread``, which kills the process outright - ``finally``
    blocks do not run. A rewind-then-restore against the shared database would
    therefore leave it stranded at the previous revision on timeout, and every
    later test in the serial job would fail against a stale schema with the real
    cause several tests back. Owning the database removes the hazard rather than
    narrowing the window: the worst a timeout can do here is leak one uniquely
    named database.
    """
    with scratch_database("mig054_lifecycle") as url:
        command.upgrade(_make_config(url), "head")
        yield url


@skip_no_pg
class TestMigration054IndexLifecycle:
    def test_head_has_widened_index_and_dropped_the_narrow_one(self, scratch_db_url: str) -> None:
        """At head: 3-column index present with the covering column order, 2-column gone."""
        indexes = asyncio.run(_documents_indexes(scratch_db_url))

        assert NEW_INDEX in indexes, f"{NEW_INDEX} missing at head; found {sorted(indexes)}"
        assert _indexed_columns(indexes[NEW_INDEX]) == ["namespace_id", "created_at", "id"], (
            f"wrong column order: {indexes[NEW_INDEX]}"
        )
        # All keys ASC. With namespace_id equality-constrained the residual
        # order is (created_at ASC, id ASC), which a backward scan reads as
        # (created_at DESC, id DESC). A stray DESC would break that symmetry.
        assert "DESC" not in indexes[NEW_INDEX].upper(), f"expected an all-ASC index: {indexes[NEW_INDEX]}"

        assert OLD_INDEX not in indexes, (
            f"{OLD_INDEX} still present at head - it shares a prefix with {NEW_INDEX}, "
            "so keeping both costs an index maintenance write per document insert"
        )

    def test_downgrade_restores_the_narrow_index_and_removes_the_wide_one(self, scratch_db_url: str) -> None:
        """Downgrading to the previous revision mirrors the upgrade exactly."""
        cfg = _make_config(scratch_db_url)

        try:
            command.downgrade(cfg, _PREV)
            indexes = asyncio.run(_documents_indexes(scratch_db_url))

            assert OLD_INDEX in indexes, f"downgrade did not restore {OLD_INDEX}; found {sorted(indexes)}"
            assert _indexed_columns(indexes[OLD_INDEX]) == ["namespace_id", "created_at"]
            assert NEW_INDEX not in indexes, f"downgrade left {NEW_INDEX} behind"
        finally:
            # Leave the module's database back at head for the sibling tests.
            # This is intra-module hygiene only - the database is this module's
            # own, so a killed process cannot strand anything shared.
            command.upgrade(cfg, "head")

        # Re-running the upgrade rebuilds the wide index: IF NOT EXISTS re-run safety.
        indexes = asyncio.run(_documents_indexes(scratch_db_url))
        assert NEW_INDEX in indexes
        assert OLD_INDEX not in indexes

    def test_post_downgrade_state_satisfies_the_unconditional_drop_below(self, scratch_db_url: str) -> None:
        """After downgrading past 054, the earlier migration's drop still works.

        That migration removes the 2-column index with a bare
        ``op.drop_index(...)`` - no ``IF EXISTS`` - so it raises if the index is
        absent. Rather than infer that from the index listing, this replays the
        exact statement against the post-downgrade database inside a transaction
        that is rolled back, so the check is the real thing and leaves no trace.
        """
        cfg = _make_config(scratch_db_url)

        try:
            command.downgrade(cfg, _PREV)
            asyncio.run(_assert_unconditional_drop_succeeds(scratch_db_url))
        finally:
            command.upgrade(cfg, "head")


async def _assert_unconditional_drop_succeeds(url: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                # Exactly what the earlier migration's downgrade emits.
                await conn.exec_driver_sql(f"DROP INDEX {OLD_INDEX}")
            except Exception as exc:  # pragma: no cover - the failure this test exists to catch
                raise AssertionError(
                    f"the unconditional DROP INDEX {OLD_INDEX} that the earlier migration's "
                    f"downgrade() issues would fail after downgrading past 054: {exc}. "
                    "Migration 054's downgrade() must recreate that index."
                ) from exc
            finally:
                await trans.rollback()
    finally:
        await engine.dispose()


@skip_no_pg
@pytest.mark.slow
class TestMigration054DowngradeWalk:
    """End-to-end downgrade walk, on a throwaway database.

    Walking dozens of migrations backwards drops real tables, so this never
    touches the shared dev database - it creates its own, walks it, and drops
    it again.

    Marked ``slow``: one test does CREATE DATABASE, the full migration chain up,
    a ~36-step walk back down, and DROP DATABASE. That is far and away the most
    expensive test in this module, and it is excluded from the default local run
    by the ``-m "not slow"`` default in ``pyproject.toml``. Note the CI
    integration job selects on ``-m "integration and not filter_conformance"``,
    which does NOT exclude ``slow`` - so this still runs there, which is where
    the coverage is wanted. The SQLite lifecycle class below walks the same path
    cheaply on every run, so nothing is lost when this one is skipped locally.
    """

    def test_downgrade_walks_past_the_index_origin_without_error(self) -> None:
        with scratch_database("mig054") as scratch_url:
            cfg = _make_config(scratch_url)
            command.upgrade(cfg, "head")

            indexes = asyncio.run(_documents_indexes(scratch_url))
            assert NEW_INDEX in indexes, "fresh database did not reach the widened index"

            try:
                command.downgrade(cfg, _BELOW_ORIGIN)
            except Exception as exc:
                # Distinguish the failure this test is about from an unrelated,
                # pre-existing broken downgrade somewhere else in the chain. Only
                # the former is evidence against this change.
                #
                # Both index names are checked, not just the old one. 054's own
                # downgrade is the FIRST step of this walk, so a failure there -
                # rebuilding the 2-column index, or dropping the 3-column one -
                # can name either. Skipping on anything that fails to mention
                # OLD_INDEX would silently retire the coverage this test exists
                # for. (OLD_INDEX is a prefix of NEW_INDEX, so the first check
                # subsumes the second; both are spelled out because the
                # subsumption is an accident of naming, not a property to rely
                # on.)
                message = str(exc)
                named = [index for index in (OLD_INDEX, NEW_INDEX) if index in message]
                if named:
                    raise AssertionError(
                        f"downgrading past {_ORIGIN} failed, naming {named}: {exc}. "
                        f"Migration 054's downgrade() must restore {OLD_INDEX}, because "
                        f"{_ORIGIN}'s downgrade() drops it unconditionally."
                    ) from exc
                pytest.skip(f"downgrade chain broke for an unrelated reason before reaching {_ORIGIN}: {exc}")

            # Below the origin revision neither index should exist.
            indexes = asyncio.run(_documents_indexes(scratch_url))
            assert indexes == {}, f"expected no documents sort index below {_ORIGIN}, found {sorted(indexes)}"


def _sqlite_documents_indexes(db_path: Path) -> dict[str, str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'"
        ).fetchall()
    finally:
        con.close()
    return {name: sql or "" for name, sql in rows if name in {NEW_INDEX, OLD_INDEX}}


class TestMigration054SqliteLifecycle:
    """The same lifecycle through the migration's non-Postgres branch.

    The embedded stack takes its schema from this chain and nothing else, so
    the plain-DDL branch is as load-bearing there as the concurrent one is on
    Postgres - and it is the branch most likely to rot, because a reviewer
    working only on Postgres never exercises it. Runs on a throwaway file with
    no server, so unlike the Postgres classes above it executes everywhere.
    """

    def _fresh_db(self, tmp_path: Path) -> tuple[Path, Config]:
        db_path = tmp_path / "lifecycle.db"
        url = f"sqlite:///{db_path}"
        cfg = Config()
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", url)
        cfg.attributes["database_url"] = url
        return db_path, cfg

    def test_round_trip_restores_each_index_in_turn(self, tmp_path: Path) -> None:
        """head -> 053 -> head, asserting the index set at every stop."""
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, "head")
        indexes = _sqlite_documents_indexes(db_path)
        assert set(indexes) == {NEW_INDEX}, f"at head expected only the 3-column index, found {sorted(indexes)}"
        assert "(namespace_id, created_at, id)" in indexes[NEW_INDEX], indexes[NEW_INDEX]

        command.downgrade(cfg, _PREV)
        indexes = _sqlite_documents_indexes(db_path)
        assert set(indexes) == {OLD_INDEX}, f"after downgrade expected only the 2-column index, found {sorted(indexes)}"
        assert "(namespace_id, created_at)" in indexes[OLD_INDEX], indexes[OLD_INDEX]

        # Back up again - the plain branch must be re-runnable, not one-shot.
        command.upgrade(cfg, "head")
        assert set(_sqlite_documents_indexes(db_path)) == {NEW_INDEX}

    def test_downgrade_walks_past_the_index_origin(self, tmp_path: Path) -> None:
        """The full walk past the revision that drops the 2-column index unconditionally.

        This is the failure mode the whole downgrade branch exists for: that
        revision issues a bare ``op.drop_index`` with no ``IF EXISTS``, so it
        raises outright if migration 054's downgrade did not put the index
        back. Verified by mutation - deleting the restore from 054's non-
        Postgres downgrade branch makes this walk fail with
        ``no such index: ix_documents_namespace_created_at``.
        """
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, "head")
        command.downgrade(cfg, _BELOW_ORIGIN)

        assert _sqlite_documents_indexes(db_path) == {}, (
            f"expected neither documents sort index below {_ORIGIN}, found {sorted(_sqlite_documents_indexes(db_path))}"
        )
