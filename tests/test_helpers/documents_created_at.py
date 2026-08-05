"""Shared fixtures for the ``documents.created_at`` NOT NULL migration.

Migration ``056_documents_created_at_not_null`` has two lanes that must run in
two different CI jobs, so the archetype rows they both seed live here rather
than in either module:

* ``tests/unit/db/test_migration_056_documents_created_at.py`` — the SQLite
  lane, which drives ``op.batch_alter_table``'s table rebuild, needs no server,
  and is the primary coverage. It belongs in the unit job so it runs on every
  PR and for contributors without Docker.
* ``tests/integration/db/test_migration_056_documents_created_at.py`` — the
  Postgres lane, which drives ``ALTER TABLE ... SET NOT NULL`` against a real
  server and is selected by the integration job's ``-m integration``.

Splitting them is what makes both demonstrably execute; the sibling helper
``documents_source_type.py`` records the CI-selection trap that motivated the
split for 055 (a ``unit``-marked class living under ``tests/integration/`` is
selected by neither job — the unit job selects by path, the integration job by
marker).

The three archetypes, seeded at the revision below 056 and asserted after the
upgrade:

* ``created_at`` NULL, ``updated_at`` present → **inferred**: rewritten to the
  ``updated_at`` value. A real timestamp, not a guess.
* ``created_at`` NULL, ``updated_at`` NULL → **invented**: stamped with the
  Unix epoch. This branch is reachable precisely because ``updated_at`` is
  nullable too — migration 000 declared both columns the same way — so it is
  not dead code, and its row count is the migration's real blast radius.
* ``created_at`` present → **untouched**. The backfill must not flatten a real
  value.

The untouched row doubles as the *format reference*. Its ``created_at`` is
bound through ``sa.DateTime(timezone=True)``, which is the same bind processor
``DocumentModel`` writes through, so comparing the epoch row's stored text
against it is what catches an epoch written as a raw string literal — a defect
that is invisible to a value-equality assertion but reorders the column on
SQLite, which compares ``DATETIME`` as text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

__all__ = [
    "EPOCH",
    "EXISTING_CREATED_AT",
    "HEAD_REVISION",
    "ID_INFERRED",
    "ID_INVENTED",
    "ID_UNTOUCHED",
    "NS",
    "PREV_REVISION",
    "UPDATED_AT",
    "bind_id",
    "insert_document",
    "insert_namespace",
    "make_config",
    "read_created_at",
    "seed_rows",
]

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations"

HEAD_REVISION = "056_documents_created_at_not_null"
PREV_REVISION = "055_documents_source_type_alignment"

#: The value the migration stamps when ``updated_at`` cannot supply one. Kept
#: here rather than imported from the revision module so a change to the
#: revision has to be made deliberately in two places.
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

NS = UUID("00000000-0000-0000-0000-0000000000bb")

# One row per archetype, so each assertion names the case it covers.
ID_INFERRED = UUID("00000000-0000-0000-0000-000000000011")
ID_INVENTED = UUID("00000000-0000-0000-0000-000000000012")
ID_UNTOUCHED = UUID("00000000-0000-0000-0000-000000000013")

#: Sub-second precision on both, deliberately: a backfill that round-tripped
#: through a second-resolution format would show up as a mismatch rather than
#: passing by coincidence.
UPDATED_AT = datetime(2025, 6, 1, 12, 30, 45, 123456, tzinfo=UTC)
EXISTING_CREATED_AT = datetime(2024, 1, 2, 3, 4, 5, 654321, tzinfo=UTC)


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


#: Both timestamps are bound through ``DateTime(timezone=True)`` — the same
#: type ``DocumentModel`` declares — so the seeded rows land in whatever
#: storage format the dialect's own processor produces. Writing them as string
#: literals would make the format-parity assertions compare the test's
#: formatting choices against the migration's, which is not the question.
_INSERT_DOCUMENT = sa.text(
    "INSERT INTO documents (id, namespace_id, content, status, source_type, created_at, updated_at) "
    "VALUES (:id, :ns, 'body', 'completed', 'library', :created_at, :updated_at)"
).bindparams(
    sa.bindparam("created_at", type_=sa.DateTime(timezone=True)),
    sa.bindparam("updated_at", type_=sa.DateTime(timezone=True)),
)


async def insert_document(
    conn: AsyncConnection,
    doc_id: UUID | str,
    ns_id: UUID | str,
    created_at: datetime | None,
    updated_at: datetime | None,
) -> None:
    """Insert a documents row with explicit timestamps (NULL included).

    Both columns carry ``server_default=CURRENT_TIMESTAMP``, so they have to be
    named explicitly — omitting either would silently fill it in and the NULL
    archetypes would never exist.
    """
    await conn.execute(
        _INSERT_DOCUMENT,
        {"id": doc_id, "ns": ns_id, "created_at": created_at, "updated_at": updated_at},
    )


async def seed_rows(url: str, ns_id: UUID | str) -> None:
    """Insert the three archetypes at the revision below 056."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await insert_namespace(conn, ns_id)
            await insert_document(conn, bind_id(url, ID_INFERRED), ns_id, None, UPDATED_AT)
            await insert_document(conn, bind_id(url, ID_INVENTED), ns_id, None, None)
            await insert_document(conn, bind_id(url, ID_UNTOUCHED), ns_id, EXISTING_CREATED_AT, UPDATED_AT)
    finally:
        await engine.dispose()


async def read_created_at(url: str, ids: list[UUID]) -> dict[UUID, datetime | None]:
    """Read ``created_at`` back through SQLAlchemy's ``DateTime`` processor.

    ``.columns(...)`` is what applies the result processor — a bare
    ``sa.text()`` hands back the raw driver value, which on SQLite is a string.
    The SQLite lane reads it both ways on purpose; see that module.
    """
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            out: dict[UUID, datetime | None] = {}
            for doc_id in ids:
                out[doc_id] = (
                    await conn.execute(
                        sa.text("SELECT created_at FROM documents WHERE id = :id").columns(
                            created_at=sa.DateTime(timezone=True)
                        ),
                        {"id": bind_id(url, doc_id)},
                    )
                ).scalar()
            return out
    finally:
        await engine.dispose()
