"""Shared fixtures for the ``documents.source_type`` alignment migration.

Migration ``055_documents_source_type_alignment`` has two lanes that must run
in two different CI jobs, so the archetype rows they both seed live here
rather than in either module:

* ``tests/unit/db/test_migration_055_documents_source_type.py`` — the SQLite
  lane, which drives ``op.batch_alter_table``'s table copy, needs no server,
  and is the primary coverage. It belongs in the unit job so it runs on every
  PR and for contributors without Docker.
* ``tests/integration/db/test_migration_055_documents_source_type.py`` — the
  Postgres lane, which drives ``ALTER TABLE ... SET NOT NULL`` against a real
  server and is selected by the integration job's ``-m integration``.

Splitting them is what makes both demonstrably execute. While the two classes
shared one module under ``tests/integration/`` with only a ``unit`` marker on
the SQLite class, **neither** CI job selected it: the unit job selects by path
(``tests/unit/``) and the integration job selects by marker
(``-m "integration and not filter_conformance"``), so a ``unit``-marked class
under ``tests/integration/`` falls through both.

The three archetypes, seeded at the revision below 055 and asserted after the
upgrade:

* ``NULL`` → rewritten to ``'library'``. The backfill; the only row it touches.
* ``'slack'`` → untouched. The backfill must not flatten real values.
* ``''`` → **stays** ``''``. NOT NULL does not rule out the degenerate empty
  string, and 055 deliberately adds no ``CHECK (source_type <> '')``. That
  assertion is what keeps the documented limitation honest.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

__all__ = [
    "BACKFILL_VALUE",
    "DECLARED_INDEX_COLUMNS",
    "ID_EMPTY_STRING",
    "ID_NULL",
    "ID_REAL_VALUE",
    "INDEX_NAME",
    "NS",
    "OPTIMIZE_CREATE_INDEX",
    "PREV_REVISION",
    "REAL_VALUE",
    "TARGET_REVISION",
    "bind_id",
    "insert_document",
    "insert_namespace",
    "make_config",
    "read_source_types",
    "seed_rows",
]

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations"

TARGET_REVISION = "055_documents_source_type_alignment"
PREV_REVISION = "054_documents_namespace_created_at_id"

INDEX_NAME = "ix_documents_namespace_source_type"
DECLARED_INDEX_COLUMNS = ["namespace_id", "source_type"]

#: The same CREATE the ``optimize_storage()`` catch-up list ships.
OPTIMIZE_CREATE_INDEX = f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON documents (namespace_id, source_type)"  # noqa: S608

NS = UUID("00000000-0000-0000-0000-0000000000aa")

# One row per archetype, so each assertion names the case it covers.
ID_NULL = UUID("00000000-0000-0000-0000-000000000001")
ID_REAL_VALUE = UUID("00000000-0000-0000-0000-000000000002")
ID_EMPTY_STRING = UUID("00000000-0000-0000-0000-000000000003")

REAL_VALUE = "slack"
BACKFILL_VALUE = "library"


def make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


def bind_id(url: str, value: UUID) -> UUID | str:
    """SQLite stores the UUID columns as TEXT; Postgres wants the UUID."""
    return value if url.startswith("postgresql") else str(value)


async def insert_namespace(conn: AsyncConnection, ns_id: UUID | str) -> None:
    await conn.execute(
        sa.text(
            "INSERT INTO memory_namespaces "
            "(id, namespace_id, version, is_active, created_at, updated_at) "
            "VALUES (:id, :id, 1, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": ns_id},
    )


async def insert_document(
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


async def seed_rows(url: str, ns_id: UUID | str, *, pre_create_index: bool) -> None:
    """Insert the three archetypes, optionally pre-creating the index."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await insert_namespace(conn, ns_id)
            await insert_document(conn, bind_id(url, ID_NULL), ns_id, None)
            await insert_document(conn, bind_id(url, ID_REAL_VALUE), ns_id, REAL_VALUE)
            await insert_document(conn, bind_id(url, ID_EMPTY_STRING), ns_id, "")
            if pre_create_index:
                await conn.execute(sa.text(OPTIMIZE_CREATE_INDEX))
    finally:
        await engine.dispose()


async def read_source_types(url: str, ids: list[UUID]) -> dict[UUID, str | None]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            out: dict[UUID, str | None] = {}
            for doc_id in ids:
                out[doc_id] = (
                    await conn.execute(
                        sa.text("SELECT source_type FROM documents WHERE id = :id"),
                        {"id": bind_id(url, doc_id)},
                    )
                ).scalar()
            return out
    finally:
        await engine.dispose()
