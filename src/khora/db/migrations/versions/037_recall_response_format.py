"""Recall response format schema changes.

Revision ID: 037_recall_response_format
Revises: 036_dream_conflicts
Create Date: 2026-05-18

Schema changes:

* ``documents.source_name`` ``VARCHAR(64)`` NULL — provider-level
  identifier (e.g. ``notion``, ``slack``). Backfilled from
  ``source`` matching ``nango://<provider>/...`` — the captured
  provider is lower-cased to match the case-insensitive convention
  in ``core.models.source.register_source_alias``.
* ``documents.source_url`` ``VARCHAR(2048)`` NULL — original-source URL
  for the document. No backfill: callers populate going forward.
* ``chunks.chunker_info`` ``JSONB`` NOT NULL DEFAULT ``'{}'::jsonb`` —
  per-chunk metadata describing which chunker produced the chunk and
  with what params.

Nullability flip on ``documents`` for six columns: ``source``,
``content_type``, ``title``, ``author``, ``language``, ``checksum``.
After this migration they are nullable with no default. Their starting
state depends on how the database was created:

* Alembic-chain-created databases: already nullable with DEFAULT ``''``.
  The flip is effectively a no-op on nullability and just drops the
  default.
* Databases created via the legacy ``create_tables()`` /
  ``Base.metadata.create_all`` path (from the old ``Mapped[str]`` ORM,
  which implied ``nullable=False``): ``NOT NULL DEFAULT ''``. The flip
  drops both the NOT NULL constraint and the default.

Because of the second case, the ``DROP NOT NULL`` MUST precede the
empty-string → NULL normalization UPDATE. If the UPDATE ran first while
the columns were still NOT NULL, writing NULL would violate the
constraint and roll back the entire revision, leaving the database
stuck at the prior revision. Existing rows whose values equal the empty
string are normalized to NULL — see "Idempotency model" below.

``source_type`` stays ``NOT NULL`` but its DB default switches from
``''`` to ``'library'`` (matching the new ORM default). Existing
empty-string rows are rewritten to ``'library'``.

PostgreSQL version requirement: ``chunker_info JSONB NOT NULL
DEFAULT '{}'::jsonb`` relies on the PG ≥11 ``ADD COLUMN ... NOT NULL
DEFAULT`` fast-path to avoid a full ``chunks`` table rewrite. The
upgrade asserts ``server_version_num >= 110000`` and raises on
older Postgres so we fail fast instead of silently triggering a
multi-hour rewrite. (The two new ``documents`` columns are nullable
so they never trigger a rewrite regardless of PG version.)

Lock-timeout safety: a Postgres-only ``SET lock_timeout = '5s'`` is
issued **before any DDL or DML** so EVERY lock acquisition in the
migration — including the ``ADD COLUMN`` AccessExclusiveLock — is
bounded. The setting is scoped to the migration's enclosing
transaction (env.py wraps the whole upgrade in a single transaction;
the SET expires at COMMIT). On lock-timeout the upgrade logs
``khora.migration.applied`` at ERROR with ``lock_timeout_tripped=True``
(detected by the Postgres SQLSTATE ``55P03`` / ``lock_not_available``
on the ``OperationalError.orig.pgcode`` — any other ``OperationalError``
class logs ``lock_timeout_tripped=False`` so dashboards don't conflate
deadlocks / connection drops / syntax errors with the lock-timeout
signal). Either path re-raises so Alembic rolls back the transaction.

Backfill: nango source_name extraction runs as a single set-based
UPDATE on Postgres using ``substring(source from 'nango://([^/]+)/')``
and ``lower(...)`` to match ``core.models.source.register_source_alias``.
On SQLite (sqlite_lance fixture stack — no native regex substring) the
same logic runs as a per-row Python loop over the much smaller
``LIKE 'nango://%'`` candidate set. The unmatched count is queried
separately and explicitly excludes NULL sources (which were normalized
to NULL by the earlier empty-string pass), so the dashboard signal
reflects only populated, non-nango rows.

Idempotency model:

* The whole upgrade runs inside a single transaction (see
  ``env.py:do_run_migrations`` — ``with context.begin_transaction()``
  wraps ``context.run_migrations()``). On any failure Postgres rolls
  back the entire revision including ADD COLUMN / DROP NOT NULL DDL.
  Alembic's version table is updated atomically with the DDL, so
  there is no partial-state retry — either the revision applied
  completely or not at all.
* The ``WHERE col = ''`` predicates on the backfill UPDATEs are
  re-run-safe within that transaction (the second pass updates zero
  rows). They are NOT what makes ADD COLUMN / DROP NOT NULL
  idempotent — that comes from Alembic's revision tracking.

Cross-dialect: the sqlite_lance fixture stack runs the full Alembic
chain. SQLite cannot ``ALTER COLUMN DROP NOT NULL`` directly, so the
nullability flip uses ``op.batch_alter_table("documents")`` — which on
Postgres issues direct ALTERs and on SQLite performs a table-copy.
The Postgres ``JSONB`` type is replaced with portable ``sa.JSON``
(TEXT-backed) on SQLite, mirroring migration-000's helper pattern.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError

revision: str = "037_recall_response_format"
down_revision: str | Sequence[str] | None = "036_dream_conflicts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_NULLABLE_FLIP_COLUMNS: tuple[tuple[str, sa.types.TypeEngine], ...] = (
    ("source", sa.String(1024)),
    ("content_type", sa.String(128)),
    ("title", sa.String(512)),
    ("author", sa.String(255)),
    ("language", sa.String(10)),
    ("checksum", sa.String(64)),
)

_MIN_PG_VERSION = 110000  # PG 11 — chunker_info NOT NULL DEFAULT fast-path

# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

# Compiled regex for the SQLite-path nango backfill. The Postgres path uses
# ``substring(source from 'nango://([^/]+)/')`` directly in SQL; SQLite has
# no native regex substring so we fall back to a Python loop over
# ``WHERE source LIKE 'nango://%'`` rows.
_NANGO_SOURCE_RE = re.compile(r"^nango://([^/]+)/")


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: OperationalError) -> bool:
    """Distinguish a real lock_timeout trip from any other OperationalError.

    OperationalError is a broad SQLAlchemy class — it wraps deadlocks,
    connection drops, syntax errors, server shutdowns, AND lock_timeout
    failures. The structured log field ``lock_timeout_tripped`` must
    only be True for the latter so monitoring dashboards aren't misled.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    return getattr(orig, "pgcode", None) == _PG_LOCK_NOT_AVAILABLE


def _assert_pg_version_ok() -> None:
    """Fail fast if Postgres is older than 11.

    ``chunks.chunker_info NOT NULL DEFAULT '{}'::jsonb`` triggers a full
    table rewrite on PG ≤ 10 — a multi-hour outage on a large chunks
    table. The hosted version meets this; we assert here so a misconfigured
    deploy fails in seconds rather than silently rewriting.
    """
    bind = op.get_bind()
    version_num = bind.execute(sa.text("SELECT current_setting('server_version_num')::int")).scalar()
    if version_num is None or int(version_num) < _MIN_PG_VERSION:
        raise RuntimeError(
            f"Migration {revision} requires PostgreSQL >= 11 "
            f"(server_version_num >= {_MIN_PG_VERSION}); got {version_num}. "
            "The chunks.chunker_info NOT NULL DEFAULT fast-path requires PG 11+."
        )


def _backfill_source_name() -> tuple[int, int]:
    """Backfill ``documents.source_name`` from ``nango://<provider>/...`` sources.

    Returns ``(backfilled, unmatched)`` where:

    * ``backfilled`` is the number of rows whose ``source`` matched the
      ``nango://<provider>/...`` shape; the lower-cased provider is
      written to ``source_name`` (case folded so downstream
      ``is_known_source`` lookups match the alias registry, which is
      keyed lowercase — see ``core.models.source.register_source_alias``).
    * ``unmatched`` is the number of populated-but-non-nango rows
      whose ``source_name`` remained NULL after the backfill. Rows with
      ``source IS NULL`` are EXCLUDED from this count — the migration's
      earlier empty-string-to-NULL pass would otherwise make
      ``unmatched`` ≈ row-count and erase the dashboard signal's
      meaning.

    Postgres: single set-based UPDATE using
    ``substring(source from 'nango://([^/]+)/')`` — O(1) round-trips on
    the data path. Critical for the production ``documents`` table where
    a per-row loop would hold an AccessExclusiveLock for every round-trip.
    SQLite: per-row Python regex over the much smaller
    ``WHERE source LIKE 'nango://%'`` candidate set. SQLite lacks
    ``substring(... from <regex>)`` so a SQL-native form is not portable;
    the sqlite_lance fixture stack is the only consumer of this branch
    and it carries hundreds-not-millions of rows, making the loop
    acceptable.
    """
    bind = op.get_bind()
    if _is_postgres():
        result = bind.execute(
            sa.text(
                "UPDATE documents "
                "SET source_name = lower(substring(source from 'nango://([^/]+)/')) "
                "WHERE source LIKE 'nango://%/%' "
                "AND source_name IS NULL"
            )
        )
        backfilled = int(result.rowcount or 0)
        unmatched_row = bind.execute(
            sa.text("SELECT COUNT(*) FROM documents WHERE source_name IS NULL AND source IS NOT NULL")
        ).scalar()
        unmatched = int(unmatched_row or 0)
        return backfilled, unmatched

    # SQLite path: per-row regex match in Python. The ``LIKE 'nango://%'``
    # filter narrows the candidate set so non-nango rows never enter the
    # Python loop and don't inflate ``backfilled``; ``unmatched`` is then
    # a separate COUNT over populated, non-nango rows.
    backfilled = 0
    rows = bind.execute(
        sa.text("SELECT id, source FROM documents WHERE source_name IS NULL AND source LIKE 'nango://%'")
    ).fetchall()
    for row in rows:
        source = row.source
        if source is None:
            continue
        match = _NANGO_SOURCE_RE.match(source)
        if match is None:
            continue
        provider = match.group(1).lower()
        bind.execute(
            sa.text("UPDATE documents SET source_name = :p WHERE id = :id"),
            {"p": provider, "id": row.id},
        )
        backfilled += 1
    unmatched_row = bind.execute(
        sa.text("SELECT COUNT(*) FROM documents WHERE source_name IS NULL AND source IS NOT NULL")
    ).scalar()
    unmatched = int(unmatched_row or 0)
    return backfilled, unmatched


def _upgrade_impl() -> tuple[int, int]:
    """Perform the schema changes. Returns ``(backfilled, unmatched)``."""
    is_pg = _is_postgres()

    if is_pg:
        _assert_pg_version_ok()
        # Bound EVERY lock acquisition — ADD COLUMN's AccessExclusiveLock
        # included — so a stuck pg_stat_activity entry on documents/chunks
        # cannot stall the deploy past 5 seconds. Issued before any DDL.
        op.execute("SET lock_timeout = '5s'")

    # New columns. On Postgres, ``ADD COLUMN NOT NULL DEFAULT '{}'::jsonb`` is
    # the PG ≥11 fast-path that avoids a full table rewrite.
    op.add_column("documents", sa.Column("source_name", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("source_url", sa.String(2048), nullable=True))
    if is_pg:
        op.add_column(
            "chunks",
            sa.Column(
                "chunker_info",
                JSONB,
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )
    else:
        # SQLite: Postgres JSONB is unrenderable. Use sa.JSON (TEXT-backed) with
        # a JSON literal default — mirrors the migration-000 dialect-helper pattern.
        op.add_column(
            "chunks",
            sa.Column(
                "chunker_info",
                sa.JSON,
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )

    # source_type DB default flip + empty-string normalization. ``source_type``
    # stays NOT NULL, but its default becomes 'library' to match the new ORM.
    op.execute(sa.text("UPDATE documents SET source_type = 'library' WHERE source_type = ''"))
    if is_pg:
        op.execute("ALTER TABLE documents ALTER COLUMN source_type SET DEFAULT 'library'")
    else:
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "source_type",
                existing_type=sa.String(64),
                existing_nullable=False,
                server_default="library",
            )

    # Nullability flip for the six columns. Drop NOT NULL + DEFAULT FIRST, then
    # run the empty-string → NULL normalization. The flip must precede the
    # UPDATE: on databases where these columns are still NOT NULL (legacy
    # create_tables() shape), setting them to NULL before dropping the
    # constraint would violate it and roll back the whole revision. The UPDATE
    # is re-run-safe within this txn via the WHERE predicate. ADD COLUMN /
    # DROP NOT NULL idempotency comes from Alembic's revision tracking + the
    # surrounding transaction, NOT from the WHERE predicate.
    if is_pg:
        for col, _ in _NULLABLE_FLIP_COLUMNS:
            op.execute(f"ALTER TABLE documents ALTER COLUMN {col} DROP NOT NULL")
            op.execute(f"ALTER TABLE documents ALTER COLUMN {col} DROP DEFAULT")
    else:
        with op.batch_alter_table("documents") as batch:
            for col, col_type in _NULLABLE_FLIP_COLUMNS:
                batch.alter_column(
                    col,
                    existing_type=col_type,
                    existing_nullable=False,
                    nullable=True,
                    server_default=None,
                )

    for col, _ in _NULLABLE_FLIP_COLUMNS:
        op.execute(sa.text(f"UPDATE documents SET {col} = NULL WHERE {col} = ''"))  # noqa: S608

    # Backfill source_name from nango sources. Counts go into the structured log.
    return _backfill_source_name()


def upgrade() -> None:
    start = time.monotonic()
    # Initialize log fields up-front so the error path always emits a uniform
    # 5-field event for dashboards / alerts.
    source_name_backfilled = 0
    source_name_unmatched = 0
    try:
        source_name_backfilled, source_name_unmatched = _upgrade_impl()
    except OperationalError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            source_name_backfilled=source_name_backfilled,
            source_name_unmatched=source_name_unmatched,
        ).error("khora.migration.applied")
        # Bare ``raise`` re-raises the active exception with the original
        # traceback preserved. env.py's wrapping context.begin_transaction()
        # rolls back all DDL from this revision.
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        source_name_backfilled=source_name_backfilled,
        source_name_unmatched=source_name_unmatched,
    ).info("khora.migration.applied")


def downgrade() -> None:
    is_pg = _is_postgres()

    # Restore NOT NULL DEFAULT '' on the six flipped columns. UPDATE NULL → ''
    # first so the NOT NULL flip does not fail on rows that the upgrade emptied.
    for col, _ in _NULLABLE_FLIP_COLUMNS:
        op.execute(sa.text(f"UPDATE documents SET {col} = '' WHERE {col} IS NULL"))  # noqa: S608

    if is_pg:
        for col, _ in _NULLABLE_FLIP_COLUMNS:
            op.execute(f"ALTER TABLE documents ALTER COLUMN {col} SET DEFAULT ''")
            op.execute(f"ALTER TABLE documents ALTER COLUMN {col} SET NOT NULL")
        # source_type default back to ''.
        op.execute("ALTER TABLE documents ALTER COLUMN source_type SET DEFAULT ''")
    else:
        with op.batch_alter_table("documents") as batch:
            for col, col_type in _NULLABLE_FLIP_COLUMNS:
                batch.alter_column(
                    col,
                    existing_type=col_type,
                    existing_nullable=True,
                    nullable=False,
                    server_default="",
                )
            batch.alter_column(
                "source_type",
                existing_type=sa.String(64),
                existing_nullable=False,
                server_default="",
            )

    # Drop the new columns.
    op.drop_column("chunks", "chunker_info")
    op.drop_column("documents", "source_url")
    op.drop_column("documents", "source_name")
