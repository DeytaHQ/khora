"""``057_drop_documents_created_at_index`` — SQLite lane (runs on every PR).

Migration 009 created ``ix_documents_created_at ON documents (created_at)``
unconditionally on *both* dialects, so 057 removes it on both. This lane
drives the non-Postgres branch on a throwaway file with no server, which
makes it the branch most likely to rot — a reviewer working only against
Postgres never exercises it, and the embedded stack takes its schema from
this chain and nothing else.

The Postgres lane is
``tests/integration/db/test_migration_057_drop_documents_created_at_index.py``;
``tests/test_helpers/documents_created_at_index.py`` carries the shared
revision pins and the reason the two lanes are separate modules.

What is verified here:

1. **Upgrade** removes the index; **downgrade** to 055 puts it back over
   ``created_at`` alone; **re-upgrade** removes it again, so the revision is
   re-runnable rather than one-shot.
2. **The restore actually satisfies the migration that drops it below.**
   Migration 009's ``downgrade()`` issues a bare
   ``op.drop_index("ix_documents_created_at", "documents")`` with no
   ``if_exists``, so it raises outright if 057's downgrade did not restore
   the index. That is covered twice — once by replaying 009's literal
   statement against the post-downgrade database, and once end-to-end by
   walking the chain down past 009.
3. **Idempotency in both directions**, which is what ``if_exists`` /
   ``if_not_exists`` are there to buy: the drop succeeds against a database
   where the index is already gone (an operator who took migration 054's
   "measure rather than assume" advice, or a schema built by the deprecated
   ``create_all`` path), and the restore succeeds where it is already
   present (an operator who pre-built it with ``CREATE INDEX CONCURRENTLY``
   ahead of a downgrade).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from tests.test_helpers.documents_created_at_index import (
    BELOW_ORIGIN_REVISION,
    HEAD_REVISION,
    INDEX_NAME,
    ORIGIN_REVISION,
    PREV_REVISION,
    UNCONDITIONAL_DROP_SQL,
    indexed_columns,
    make_config,
)

pytestmark = pytest.mark.unit


def _documents_indexes(db_path: Path) -> dict[str, str]:
    """Map index name -> CREATE statement for every ``documents`` index.

    Read straight out of ``sqlite_master`` rather than through SQLAlchemy
    reflection: the exact DDL text is what the column assertion below checks,
    and reflection discards it.
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'"
        ).fetchall()
    finally:
        con.close()
    return {name: sql or "" for name, sql in rows}


def _execute(db_path: Path, statement: str) -> None:
    con = sqlite3.connect(db_path, isolation_level=None)
    try:
        con.execute(statement)
    finally:
        con.close()


def _replay_unconditional_drop(db_path: Path) -> None:
    """Run 009's literal ``DROP INDEX`` and roll it back.

    ``isolation_level=None`` puts the driver in autocommit so the ``BEGIN``
    below is the real transaction boundary; SQLite's DDL is transactional, so
    the rollback leaves the schema exactly as found.
    """
    con = sqlite3.connect(db_path, isolation_level=None)
    try:
        con.execute("BEGIN")
        try:
            con.execute(UNCONDITIONAL_DROP_SQL)
        except sqlite3.Error as exc:
            raise AssertionError(
                f"the unconditional `{UNCONDITIONAL_DROP_SQL}` that {ORIGIN_REVISION}'s "
                f"downgrade() issues would fail after downgrading past 057: {exc}. "
                f"Migration 057's downgrade() must recreate {INDEX_NAME}."
            ) from exc
        finally:
            con.execute("ROLLBACK")
    finally:
        con.close()


class TestMigration057OnSqlite:
    def _fresh_db(self, tmp_path: Path) -> tuple[Path, Config]:
        db_path = tmp_path / "lifecycle.db"
        return db_path, make_config(f"sqlite:///{db_path}")

    def test_round_trip_drops_then_restores_the_index(self, tmp_path: Path) -> None:
        """head -> 055 -> head, asserting the index at every stop."""
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, HEAD_REVISION)
        assert INDEX_NAME not in _documents_indexes(db_path), f"{INDEX_NAME} survived the upgrade to {HEAD_REVISION}"

        command.downgrade(cfg, PREV_REVISION)
        indexes = _documents_indexes(db_path)
        assert INDEX_NAME in indexes, f"downgrade did not restore {INDEX_NAME}; found {sorted(indexes)}"
        # Single column, and that column is created_at — a restore over the
        # wrong column would satisfy a name-only check while covering nothing.
        assert indexed_columns(indexes[INDEX_NAME]) == ["created_at"], indexes[INDEX_NAME]

        # Back up again — the revision must be re-runnable, not one-shot.
        command.upgrade(cfg, HEAD_REVISION)
        assert INDEX_NAME not in _documents_indexes(db_path)

    def test_post_downgrade_state_satisfies_the_unconditional_drop_at_the_origin(self, tmp_path: Path) -> None:
        """After downgrading past 057, 009's bare ``DROP INDEX`` still works.

        Replays 009's exact statement rather than inspecting the index
        listing. An index-listing assertion would only restate 057's own
        source; executing the statement is the actual contract.
        """
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, HEAD_REVISION)
        command.downgrade(cfg, PREV_REVISION)

        _replay_unconditional_drop(db_path)

    def test_downgrade_walks_past_the_index_origin(self, tmp_path: Path) -> None:
        """The full walk past the revision that drops the index unconditionally.

        This is the failure mode 057's downgrade branch exists for: 009
        issues a bare ``op.drop_index`` with no ``if_exists``, so it raises
        outright if 057's downgrade did not put the index back. Verified by
        mutation — deleting the ``op.create_index`` from 057's ``downgrade()``
        makes this walk fail with ``sqlalchemy.exc.OperationalError:
        (sqlite3.OperationalError) no such index: ix_documents_created_at``
        (SQLAlchemy wraps the driver error; the inner text is the driver's).

        The Postgres lane marks its equivalent ``slow``; this one is cheap
        enough to run unmarked on every job, which is what keeps the coverage
        alive for contributors without Docker.
        """
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, HEAD_REVISION)
        command.downgrade(cfg, BELOW_ORIGIN_REVISION)

        assert INDEX_NAME not in _documents_indexes(db_path), (
            f"{INDEX_NAME} should not exist below {ORIGIN_REVISION}, which is the revision that creates it"
        )

    def test_upgrade_succeeds_when_the_index_is_already_absent(self, tmp_path: Path) -> None:
        """``if_exists=True`` on the drop, exercised rather than asserted.

        Two real database shapes reach 057 without this index: one whose
        operator dropped it on migration 054's explicit advice to measure,
        and one built by the deprecated ``create_all`` path, which never
        creates it because the ORM does not declare it.

        **The assertion is the upgrade not raising.** Mutation-checked:
        removing ``if_exists=True`` makes ``command.upgrade`` below raise
        ``no such index``, and this is the only test in the module that
        catches it. The closing state check is a tripwire, not the point —
        it restates the precondition (the index was already dropped by hand),
        so it also passes against an emptied ``upgrade()``. Deleting the drop
        entirely is caught by ``test_round_trip...`` and the parity gate.
        """
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, PREV_REVISION)
        assert INDEX_NAME in _documents_indexes(db_path), f"precondition: {INDEX_NAME} exists at {PREV_REVISION}"
        _execute(db_path, f"DROP INDEX {INDEX_NAME}")

        command.upgrade(cfg, HEAD_REVISION)

        assert INDEX_NAME not in _documents_indexes(db_path)

    def test_downgrade_succeeds_when_the_index_is_already_present(self, tmp_path: Path) -> None:
        """``if_not_exists=True`` on the restore, exercised rather than asserted.

        The documented runbook for downgrading a large table is to pre-build
        the index with ``CREATE INDEX CONCURRENTLY`` and let the migration's
        create fall through as a no-op. That is only a safe instruction if
        the create really tolerates the index being there.
        """
        db_path, cfg = self._fresh_db(tmp_path)

        command.upgrade(cfg, HEAD_REVISION)
        _execute(db_path, f"CREATE INDEX {INDEX_NAME} ON documents (created_at)")

        command.downgrade(cfg, PREV_REVISION)

        indexes = _documents_indexes(db_path)
        assert INDEX_NAME in indexes, f"the restore removed the pre-built index; found {sorted(indexes)}"
        assert indexed_columns(indexes[INDEX_NAME]) == ["created_at"], indexes[INDEX_NAME]

        # And the migration did not build a SECOND index over the same column
        # under a different name, which is the real duplication hazard when a
        # create falls through on a pre-built index. Compared by columns, not
        # by name: a name comparison cannot see a differently-named duplicate.
        over_created_at = [name for name, sql in indexes.items() if indexed_columns(sql) == ["created_at"]]
        assert over_created_at == [INDEX_NAME], (
            f"expected exactly one index over documents(created_at), found {over_created_at}"
        )
