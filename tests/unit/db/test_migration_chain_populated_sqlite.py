"""The Alembic chain must not destroy rows on a populated SQLite database.

Why this module exists
======================
Every other migration/drift gate in the suite builds the chain on an **empty**
database. A cascade that deletes every child row is invisible there — zero rows
before, zero rows after — which is how three separate revisions shipped with a
data-loss bug nobody noticed.

The bug: ``db/migrations/env.py`` issued ``PRAGMA foreign_keys = ON`` on the
SQLite migration connection. Alembic's batch mode implements "alter column" as
SQLite's documented table-rebuild procedure — create temp table, copy rows,
``DROP TABLE``, rename — and SQLite's ``DROP TABLE`` performs an implicit
``DELETE FROM`` when enforcement is on, firing every inbound ``ON DELETE
CASCADE`` before the rename puts the table back. Rebuilding ``documents``
therefore emptied ``chunks``, ``keyword_chunks`` and ``chronicle_events``;
rebuilding ``memory_namespaces`` emptied every data table in the schema. On the
``sqlite_lance`` stack LanceDB still holds the embeddings keyed by chunk id, so
the residue is orphaned vectors rather than merely missing rows.

The gate is deliberately chain-wide rather than tied to one revision. The
rebuilds span ``memory_namespaces`` (001 / 010 / 011 / 012 / 013),
``entities`` (008) and ``documents`` (016 / 037 / 055 / 056), so a test pinned
to the newest revision would cover the smallest leg of the hazard.

What is asserted, and why each assertion is the shape it is
===========================================================
* **The upgrade completes.** Not a formality: from early revisions the unfixed
  chain does not merely lose rows, it *raises* — the cascade trips a
  constraint, the per-migration transaction rolls back, and the database is
  stranded at its starting revision, permanently unable to upgrade. The exact
  exception is seed-data dependent (``IntegrityError`` and ``SQLITE_CORRUPT``
  have both been observed for the same condition), so this asserts that the
  upgrade *succeeds* and lands on head rather than matching a message.
* **A full-schema row-count snapshot, not a hand-listed subset.** The cascade
  is transitive, so naming tables in advance would under-cover it. Exact counts
  rather than non-emptiness, because a partial cascade passes a non-emptiness
  check.
* **``PRAGMA foreign_key_check``** is clean. This deliberately does **not**
  detect the cascade — a cascade deletes children precisely so that no
  violation remains, leaving a consistent-but-empty database, and the check
  reports clean on the damaged file. It is here for the *inverse* risk that
  turning enforcement off introduces: a future revision that deletes parent
  rows would now orphan children silently. The row counts catch the cascade;
  this catches its opposite.

``organizations`` and ``workspaces`` are the only tables the chain legitimately
drops (migration 010 flattens the namespace hierarchy), so they are the only
permitted deletions and are named explicitly rather than tolerated by a
weaker assertion.

A row count that *grows* fails this test too. Nothing in the chain backfills a
derived table from a seeded one today; if a future revision does, that is a
deliberate change and should be recorded here rather than absorbed silently.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.test_helpers.schema_drift import upgrade

pytestmark = pytest.mark.unit

#: Starting revisions chosen to cross each distinct family of batch rebuild:
#: ``000`` and ``009`` cross the ``memory_namespaces`` rebuilds in
#: 001 / 010 / 011 / 012 / 013 (the largest blast radius, and the two starts
#: from which the unfixed chain raises rather than silently losing rows);
#: ``013`` and ``016`` cross the ``documents`` rebuilds in 016 / 037; ``054``
#: is the shortest walk, crossing 055 and 056 only.
_START_REVISIONS = [
    "000_initial_schema",
    "009_temporal_search_indexes",
    "013_drop_previous_version_id",
    "016_widen_extraction_config_hash",
    "054_documents_namespace_created_at_id",
]

_HEAD_REVISION = "056_documents_created_at_not_null"

#: Dropped by ``010_flatten_namespace_hierarchy``. The only tables the chain
#: removes on the walk from any revision above.
_TABLES_DROPPED_BY_THE_CHAIN = {"organizations", "workspaces"}

#: Bound as text rather than as a ``datetime``. SQLite stores DATETIME as text
#: regardless, and Python 3.12 deprecated sqlite3's implicit datetime adapter —
#: passing the object emits a DeprecationWarning on every seeded timestamp.
#: This is the format SQLAlchemy's own SQLite DateTime processor produces.
_TIMESTAMP_TEXT = "2026-01-01 00:00:00.000000"


def _user_tables(conn: sa.Connection) -> list[str]:
    """Every table SQLite reports, minus its own internal ones.

    FTS shadow tables (``chunks_fts``, ``chunks_fts_docsize``, …) are kept on
    purpose — they are where the cascade surfaced most visibly, because the
    ``chunks`` triggers propagate a delete into them.
    """
    return sorted(
        row[0]
        for row in conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        )
    )


def _row_counts(url: str) -> dict[str, int]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            counts: dict[str, int] = {}
            for table in _user_tables(conn):
                # Interpolated because a table name cannot be a bind parameter.
                # The names come from sqlite_master, never from user input.
                count_sql = sa.text(f'SELECT COUNT(*) FROM "{table}"')  # noqa: S608
                counts[table] = conn.execute(count_sql).scalar()
            return counts
    finally:
        engine.dispose()


def _insert_row(engine: sa.Engine, table: str, **explicit: Any) -> dict[str, Any]:
    """Insert one row into *table*, inventing a value for every required column.

    Reflection-driven rather than a hand-written INSERT per revision: the
    schema differs at every starting revision above (columns are added, widened
    and dropped across the walk), so a literal statement would have to be
    written five times and would break whenever an early revision changed.
    Only columns that are NOT NULL and carry no default need a value; anything
    else is left to the schema.
    """
    values = dict(explicit)
    for column in sa.inspect(engine).get_columns(table):
        name = column["name"]
        if name in values or column["nullable"] or column["default"] is not None:
            continue
        type_name = str(column["type"]).upper()
        if "INT" in type_name:
            values[name] = 1
        elif "FLOAT" in type_name or "REAL" in type_name or "NUMERIC" in type_name:
            values[name] = 1.0
        elif "BOOL" in type_name:
            values[name] = 1
        elif "DATE" in type_name or "TIME" in type_name:
            values[name] = _TIMESTAMP_TEXT
        elif "JSON" in type_name:
            values[name] = "{}"
        elif name == "id" or name.endswith("_id"):
            values[name] = str(uuid.uuid4())
        else:
            values[name] = "x"

    columns = ", ".join(values)
    binds = ", ".join(f":{key}" for key in values)
    with engine.begin() as conn:
        conn.execute(sa.text(f"INSERT INTO {table} ({columns}) VALUES ({binds})"), values)  # noqa: S608
    return values


def _seed(url: str) -> None:
    """Populate every table the rebuilds cascade through — and do not trim this.

    Which rows are seeded *is* this test's teeth, and the choice is not
    obvious, so it is recorded here. Measured against a tree with the pragma
    forced back ON, the tables that survive the damaged upgrade unchanged are
    ``documents`` (1 → 1), ``memory_namespaces`` (1 → 1) and ``entities``
    (1 → 1). **The rebuilt table itself always survives** — only its cascade
    children die. So the natural test for a migration about ``documents``
    ("seed documents, upgrade, assert the rows survived") is green on the
    broken code, and every row below that is not ``documents`` is what makes
    this test discriminate at all:

    * ``chunks`` — the direct cascade child of ``documents``. The one table
      that goes 1 → 0 on the plain ``documents`` rebuild.
    * ``keyword_chunks`` and ``chronicle_events`` — cascade off ``chunks``,
      not off ``documents``. They are the *transitive* leg, and they are what
      makes the test assert transitivity rather than assume it.
    * ``entities`` — cascades off ``memory_namespaces``, covering the 008
      rebuild family rather than the ``documents`` one.

    Seeding is defensive because the schema differs by starting revision:
    ``keyword_chunks`` arrives in 050 and ``chronicle_events`` in 024, so from
    an early start they simply do not exist yet and the walk covers them from
    the revision that creates them onward.
    """
    engine = sa.create_engine(url)
    try:
        tables = set(sa.inspect(engine).get_table_names())
        namespace_parent: dict[str, Any] = {}
        if "organizations" in tables:
            # Pre-010 the hierarchy is organization -> workspace -> namespace.
            organization = _insert_row(engine, "organizations")
            workspace = _insert_row(engine, "workspaces", organization_id=organization["id"])
            namespace_parent["workspace_id"] = workspace["id"]

        namespace = _insert_row(engine, "memory_namespaces", **namespace_parent)
        document = _insert_row(engine, "documents", namespace_id=namespace["id"])
        chunk = _insert_row(engine, "chunks", namespace_id=namespace["id"], document_id=document["id"])
        _insert_row(engine, "entities", namespace_id=namespace["id"])

        for table in ("keyword_chunks", "chronicle_events"):
            if table in tables:
                _insert_row(engine, table, namespace_id=namespace["id"], chunk_id=chunk["id"])
    finally:
        engine.dispose()


def _foreign_key_violations(url: str) -> list[Any]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return list(conn.execute(sa.text("PRAGMA foreign_key_check")))
    finally:
        engine.dispose()


def _documents_index_sql(url: str) -> dict[str, str]:
    """Name → defining SQL for every index on ``documents``.

    ``sqlite_master`` rather than ``PRAGMA index_list``: the latter enumerates
    in reverse-creation order, so a rebuild renumbers its ``seq`` column and a
    before/after comparison of its output is flaky by construction even when
    the index set is identical.
    """
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return {
                name: sql
                for name, sql in conn.execute(
                    sa.text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'")
                )
                if sql is not None
            }
    finally:
        engine.dispose()


def _head_revision(url: str) -> str | None:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(sa.text("SELECT version_num FROM khora_alembic_version")).scalar()
    finally:
        engine.dispose()


@pytest.mark.parametrize("start_revision", _START_REVISIONS)
def test_populated_database_survives_the_upgrade_to_head(tmp_path: Path, start_revision: str) -> None:
    url = f"sqlite:///{tmp_path / 'chain.db'}"
    upgrade(url, start_revision)
    _seed(url)

    before = _row_counts(url)
    documents_indexes_before = _documents_index_sql(url)
    assert sum(before.values()) > 0, "precondition: the database must actually carry rows"

    # No pytest.raises wrapper: from an early revision the unfixed chain raises
    # here, and the failure should read as "the upgrade could not complete",
    # which is what it is.
    upgrade(url, "head")

    assert _head_revision(url) == _HEAD_REVISION, "the chain did not reach head"

    after = _row_counts(url)
    disappeared = set(before) - set(after)
    assert disappeared == disappeared & _TABLES_DROPPED_BY_THE_CHAIN, (
        f"the chain dropped tables it is not expected to drop: {sorted(disappeared - _TABLES_DROPPED_BY_THE_CHAIN)}"
    )

    surviving = {table: count for table, count in before.items() if table not in _TABLES_DROPPED_BY_THE_CHAIN}
    assert {table: after.get(table) for table in surviving} == surviving, (
        f"row counts changed across the upgrade from {start_revision}. A drop to zero in a child table is the "
        f"batch-rebuild cascade: SQLite's DROP TABLE fires ON DELETE CASCADE when foreign_keys is ON, so the "
        f"migration connection must keep it OFF (db/migrations/env.py)."
    )

    assert _foreign_key_violations(url) == [], (
        "PRAGMA foreign_key_check found orphans. FK enforcement is off during migrations, so a revision that "
        "deletes parent rows now orphans children silently instead of cascading — this is the control for that."
    )

    # The chain adds indexes on the way to head (054 and 055 each add one), so
    # this is a subset check: nothing that existed may be lost, and anything
    # still present must be defined identically.
    documents_indexes_after = _documents_index_sql(url)
    assert documents_indexes_before.items() <= documents_indexes_after.items(), (
        f"the documents rebuild lost or altered an index: "
        f"{sorted(set(documents_indexes_before.items()) - set(documents_indexes_after.items()))}"
    )
