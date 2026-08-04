"""Align the live ``documents`` schema with the ``DocumentModel`` declaration.

Revision ID: 055_documents_source_type_alignment
Revises: 054_documents_namespace_created_at_id
Create Date: 2026-08-04

Two pre-existing drifts between ``db/models.py::DocumentModel`` and the
schema the Alembic chain actually builds. Neither is a behaviour change
request — both are the schema catching up to a declaration that has been
in the ORM all along.

1. ``Index("ix_documents_namespace_source_type", "namespace_id",
   "source_type")`` is declared in ``DocumentModel.__table_args__`` but no
   migration ever created it. On an Alembic-built database the index does
   not exist; on a database where an operator ran ``optimize_storage()``
   it does (see below).

2. ``documents.source_type`` is ``Mapped[str] ... nullable=False`` in the
   ORM but NULLABLE in the live schema on both Alembic-managed dialects —
   migration 000 declared the column with ``server_default=""`` and no
   ``nullable`` argument, so it was created NULL-able. Migration 037
   states that "``source_type`` stays ``NOT NULL``" and passes
   ``existing_nullable=False``; that is a *hint* to Alembic's batch
   renderer, not a constraint application, so the intended NOT NULL was
   never actually installed. This revision installs it, passing
   ``nullable=False`` explicitly.

Scope: this covers the Alembic-managed schema only — Postgres and the
SQLite the ``sqlite_lance`` stack migrates. A **third** relational tier
exists and is NOT covered: ``SQLiteRelationalBackend`` /
``SQLiteVectorBackend`` (``backend: sqlite``, the zero-infrastructure
mode, registered in ``storage/factory.py``) build ``documents`` from a
hand-written DDL string literal in ``storage/backends/sqlite.py`` at
``connect()`` time, entirely outside Alembic. There ``source_type`` stays
nullable with the pre-037 ``''`` default and no
``ix_documents_namespace_source_type`` exists under any name. That
divergence is stated, not closed: the backend has no migration chain to
hang a fix on, and because its schema is a Python string rather than
migration output, an Alembic-vs-ORM drift gate cannot see it either.

Why the index, honestly
-----------------------
This is **not** a performance change. No production query path filters
``documents`` by ``(namespace_id, source_type)`` today:
``PostgreSQLBackend.list_documents`` filters on ``status`` /
``updated_before`` only, and the documents filter-compile context has no
production caller yet. The index is created because the ORM declaration
and the live schema must agree — a declared-but-absent index is a trap for
the next person who reads ``__table_args__`` and assumes the predicate is
served. If a document-enumeration path is ever added, the shape it would
actually want is ``(namespace_id, source_type, created_at DESC)`` to also
serve the total order pinned on document listing (#1576). That is a note
for a future revision, not a change made here.

``if_not_exists=True`` is mandatory
-----------------------------------
``storage/optimize.py`` already ships ``CREATE INDEX IF NOT EXISTS
ix_documents_namespace_source_type ON documents (namespace_id,
source_type)`` in its catch-up list, and ``optimize_storage()`` is public
API. Any database on which an operator ran it already carries the index,
so a bare ``op.create_index`` would raise ``DuplicateTable`` and wedge the
chain there permanently. The two definitions are byte-compatible (same
name, same table, same column order), so ``IF NOT EXISTS`` converges on
one index either way.

The backfill is load-bearing and NOT reversible
-----------------------------------------------
``UPDATE documents SET source_type = 'library' WHERE source_type IS NULL``
runs before the flip. Without it a single pre-existing NULL row makes
``SET NOT NULL`` raise ``IntegrityError`` and rolls the revision back.
``'library'`` is the ORM default and the default migration 037 installed
on the column. Like the empty-string normalization in 037, the rewrite is
one-way: after it runs, a row that was NULL is indistinguishable from a
row that always said ``'library'``, so ``downgrade()`` drops the
constraint but cannot restore the NULLs.

One wrinkle worth naming: ``'library'`` is the *ORM* default, but the
ingest pipeline's own default for a document that supplies no
``source_type`` is ``'manual'``. A historic NULL row that originated in
the ingest pipeline is therefore labelled ``'library'`` by this backfill
rather than the ``'manual'`` it would carry if written today. The
alternative — guessing provenance per row — is not something the
migration can do correctly, and ``'library'`` matches both the ORM
default and the column default, so it is the defensible choice. The count
is emitted as ``rows_backfilled`` on the ``khora.migration.applied`` log
event if anyone needs to audit the blast radius.

Downstream note: migration 044's idempotency argument (its
``AND kc.source_type IS NULL`` restart sentinel) is justified in its
docstring by ``documents.source_type`` being ``NOT NULL``. That premise
was false when 044 was written; this revision is what finally makes it
true. 044's own docstring is left alone.

What NOT NULL does and does not buy
-----------------------------------
It rules out NULL. It does **not** rule out the degenerate empty string,
which is what this codebase historically produced — migration 000 created
the column with ``server_default=""``, and 037 had to rewrite those rows
to ``'library'``. A ``CHECK (source_type <> '')`` would close that gap;
it is deliberately not added here, because PostgreSQL has no ``ADD
CONSTRAINT IF NOT EXISTS`` and a failed retry would wedge the chain.

What the companion call-site normalization covers
--------------------------------------------------
It collapses ``None`` **and** ``''`` to the applicable default on every
``documents`` write path reached through ``Khora.remember`` /
``remember_batch`` / ``submit_batch`` or the ingest pipeline — the
internal paths, where the value actually gets decided. Both the
top-level kwarg and the per-doc dict value are normalized, so
``remember(..., source_type="")`` now persists ``'library'`` where it
used to persist ``''``. That is a deliberate behaviour change on a public
entry point, consistent with what migration 037 already did to existing
``''`` rows, and it is recorded in the CHANGELOG.

Three surfaces are deliberately NOT normalized. What the constraint does
to each differs, and the difference is not intuitive:

* ``Document(source_type=None)`` → ``create_document``: **does not
  raise.** ``DocumentModel.source_type`` carries a Python-side
  ``default="library"`` and SQLAlchemy applies a column default on INSERT
  when the value is ``None``, without distinguishing "unset" from
  "explicitly ``None``". The row silently stores ``'library'``.
  ``Document(source_type="")`` stores ``''`` — the default does not fire
  for a non-``None`` value and the constraint permits it.
* ``Document(source_type=None)`` → ``update_document``, and
  ``partial_update_document(..., source_type=None)``: **these raise
  ``IntegrityError``.** Both compile to Core ``update().values(...)``,
  where no column default applies and the ``None`` reaches the database
  as SQL NULL. ``partial_update_document`` additionally carries
  ``source_type`` in its public field whitelist and bypasses ``Document``
  entirely.
* The engines' own ``remember`` / ``remember_batch`` entry points, called
  directly rather than through ``Khora``. They declare
  ``source_type: str`` and pass the value through unnormalized, so they
  inherit whichever of the two behaviours above the write takes.

Raising is the intended outcome wherever it happens — ``source_type`` is
annotated ``str`` and ``None`` was always a type violation. No code
papers over any of it.

Deploy ordering
---------------
**Ship the call-site normalization before running this migration.** A
live instance on the old code that writes a NULL ``source_type`` while
the revision is in flight aborts it (the ``SET NOT NULL`` fails against
the new row) and forces a retry. Old code against the new schema is
otherwise fine — it only ever reads the column.

Atomicity
---------
No ``autocommit_block``, no ``CREATE INDEX CONCURRENTLY``. Under
``transaction_per_migration=True`` (``env.py:do_run_migrations``) this
revision runs in one transaction that commits together with its version
stamp, so there is no partial-apply state to retry into: either every
statement landed or none did. ``CONCURRENTLY`` would have to run in an
autocommit block, which is precisely what makes a mid-revision failure —
and the INVALID-index left behind by a failed concurrent build —
reachable.

Locking — this migration blocks ``documents``
---------------------------------------------
Read this before deploying against a large table. The atomic shape is
deliberate, but it is not free:

* ``ALTER TABLE ... SET NOT NULL`` takes an ``ACCESS EXCLUSIVE`` lock and
  performs a **full heap scan** to verify no NULL remains. ``ACCESS
  EXCLUSIVE`` blocks readers as well as writers. (PG 12+ can skip the
  scan when a validated ``CHECK (col IS NOT NULL)`` already proves the
  invariant; no such constraint exists here, so the scan always runs.)
* ``CREATE INDEX`` (non-concurrent) takes a ``SHARE`` lock for the whole
  build, blocking writes to ``documents`` — reads still proceed.
* Because the revision is one transaction, **every lock is held until
  COMMIT**, not released when its statement finishes. The two windows
  therefore add rather than overlap.

``SET lock_timeout = '5s'`` bounds how long each statement will *wait to
acquire* a lock. It does **not** bound how long a lock is *held* once
acquired, and it does not bound the heap scan or the index build. Order
of magnitude: tens of seconds of blocking at ~10M rows, minutes at
~100M.

Operator runbook for a large table: run

    CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_namespace_source_type
        ON documents (namespace_id, source_type);

**before** the deploy. Because the ``op.create_index`` below passes
``if_not_exists=True``, the migration then finds the index present and
its create becomes a no-op, shrinking the blocking window to the
``SET NOT NULL`` scan alone.

Cross-dialect: SQLite cannot ``ALTER COLUMN ... SET NOT NULL``, so the
flip goes through ``op.batch_alter_table("documents")`` (table copy),
mirroring 037. The sqlite_lance fixture stack runs the full chain, so
the SQLite branch is load-bearing, not a green-keeper.

Not touched: ``khora_chunks.source_type`` (migrations 041/044). That
denormalized copy is deliberately nullable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import DBAPIError

revision: str = "055_documents_source_type_alignment"
down_revision: str | Sequence[str] | None = "054_documents_namespace_created_at_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

_INDEX_NAME = "ix_documents_namespace_source_type"

# The ORM default (``DocumentModel.source_type``) and the DB default migration
# 037 installed on the column.
_DEFAULT_SOURCE_TYPE = "library"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: DBAPIError) -> bool:
    """Distinguish a real lock_timeout trip from any other database error.

    The caught class is ``DBAPIError``, not ``OperationalError``, and that
    is load-bearing rather than defensive breadth. On asyncpg — the driver
    khora runs on Postgres — a lock-timeout arrives as
    ``asyncpg.exceptions.LockNotAvailableError``, which subclasses
    ``PostgresError``. The asyncpg dialect's translation table maps
    ``PostgresError`` to the *base* DBAPI ``Error`` class, so SQLAlchemy
    wraps it in a plain ``sqlalchemy.exc.DBAPIError``:
    ``isinstance(wrapped, OperationalError)`` is **False**. An
    ``except OperationalError`` here would never fire on the one event the
    5s timeout exists to produce. (Migrations 037 / 038 / 042 / 053 carry
    that narrower clause; fixing them is out of scope for this revision.)

    ``DBAPIError`` also covers the ``IntegrityError`` raised by the
    deploy-ordering race documented above — a live old instance writing a
    NULL while ``SET NOT NULL`` runs — so that failure is logged rather
    than silently re-raised.

    Since ``DBAPIError`` is broad (deadlocks, connection drops, syntax
    errors, constraint violations), the ``lock_timeout_tripped`` field
    must stay keyed on the SQLSTATE so dashboards aren't misled.

    Both attribute spellings are read, and that is the whole point rather
    than belt-and-braces: **asyncpg carries the code on ``sqlstate`` and
    leaves ``pgcode`` as ``None``**, while psycopg2/psycopg use ``pgcode``.
    ``env.py`` normalizes Postgres URLs to ``postgresql+asyncpg``, so a
    ``pgcode``-only check — which is what migrations 037 / 038 / 042 / 053
    do — is dead on this driver: it can never report a lock timeout that
    actually happened. Verified against the pinned asyncpg:
    ``LockNotAvailableError('probe').sqlstate == '55P03'`` and
    ``.pgcode is None``.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    for attr in ("sqlstate", "pgcode"):
        if getattr(orig, attr, None) == _PG_LOCK_NOT_AVAILABLE:
            return True
    return False


def _upgrade_impl() -> int:
    """Apply both alignments. Returns the number of NULL rows backfilled."""
    is_pg = _is_postgres()

    if is_pg:
        # Bound EVERY lock ACQUISITION — the SET NOT NULL AccessExclusiveLock
        # and the index build's ShareLock included — so a stuck
        # pg_stat_activity entry on documents cannot stall the deploy past 5s
        # waiting to start. This does NOT bound how long those locks are held
        # once taken; see the Locking section of the module docstring.
        # Issued before any DDL or DML.
        #
        # SET LOCAL, not a bare SET: env.py runs with
        # transaction_per_migration=True, so this revision has its own
        # transaction and LOCAL scopes the setting to it. A bare SET is
        # session-scoped and would silently leak the 5s timeout onto every
        # later revision in the same `alembic upgrade head` run — revision 056
        # would inherit a timeout it never asked for. (Migrations 037 and 053
        # use the bare form and their docstrings claim it "expires at COMMIT",
        # which is not true of a session-level SET; they are not corrected here
        # because they are not this change's to touch.)
        op.execute("SET LOCAL lock_timeout = '5s'")

    # Statement order is backfill -> SET NOT NULL -> CREATE INDEX, and it is
    # deliberate. Both DDL statements are unconditional and order-independent
    # for correctness, so the ordering is chosen for retry economics: SET NOT
    # NULL takes the AccessExclusiveLock and is by far the likeliest statement
    # to trip the 5s lock_timeout, and the whole revision rolls back as one
    # transaction. Building the index first would mean every timed-out retry
    # discarded and repaid the most expensive work in the migration. Doing the
    # backfill before the index also avoids churning index entries that the
    # UPDATE would immediately dead-tuple.

    # Backfill before the flip — a single NULL row would otherwise make
    # SET NOT NULL raise and roll the whole revision back. Not reversible.
    result = op.get_bind().execute(
        sa.text("UPDATE documents SET source_type = :default WHERE source_type IS NULL"),
        {"default": _DEFAULT_SOURCE_TYPE},
    )
    # max(..., 0): the DBAPI contract lets rowcount be -1 when a driver cannot
    # report a count, and a negative row count in a log field is noise.
    backfilled = max(int(result.rowcount or 0), 0)

    if is_pg:
        op.execute("ALTER TABLE documents ALTER COLUMN source_type SET NOT NULL")
    else:
        # SQLite has no ALTER COLUMN; batch mode performs a table copy.
        # ``nullable=False`` is passed EXPLICITLY — ``existing_nullable`` alone
        # is only a hint to the renderer and applies nothing (migration 037 is
        # the proof: it passed ``existing_nullable=False`` and the column
        # stayed nullable).
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "source_type",
                existing_type=sa.String(64),
                existing_nullable=True,
                nullable=False,
                existing_server_default=_DEFAULT_SOURCE_TYPE,
            )

    # ``if_not_exists`` is mandatory: optimize_storage() ships the same
    # CREATE INDEX IF NOT EXISTS, so this index may already exist. It is also
    # what lets an operator pre-build the index CONCURRENTLY before the deploy
    # and reduce this statement to a no-op (see the module docstring).
    op.create_index(
        _INDEX_NAME,
        "documents",
        ["namespace_id", "source_type"],
        if_not_exists=True,
    )

    return backfilled


def upgrade() -> None:
    start = time.monotonic()
    # Initialize log fields up-front so the error path emits the same field set
    # as the success path. "Same shape", not "always": only a DBAPIError is
    # caught, so a failure that is not database-originated (a bug in this
    # module, say) propagates without an event.
    rows_backfilled = 0
    try:
        rows_backfilled = _upgrade_impl()
    # DBAPIError, not OperationalError: on asyncpg a lock-timeout is wrapped as
    # a plain DBAPIError and would slip past the narrower clause entirely. See
    # _is_lock_timeout's docstring. This also covers the IntegrityError from
    # the deploy-ordering race.
    except DBAPIError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            rows_backfilled=rows_backfilled,
        ).error("khora.migration.applied")
        # Bare ``raise`` re-raises the active exception with the original
        # traceback preserved. The per-migration transaction rolls back both
        # the flip and the index.
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        rows_backfilled=rows_backfilled,
    ).info("khora.migration.applied")


def downgrade() -> None:
    """Drop the NOT NULL constraint and the index.

    Two asymmetries with ``upgrade()``, both intentional:

    * The ``NULL -> 'library'`` backfill is NOT undone. Once rewritten, a
      row that was NULL is indistinguishable from one that always carried
      the default, so there is nothing to restore (migration 037 documents
      its analogous empty-string normalization the same way).
    * The index drop is unconditional, which means it also drops an index
      that ``optimize_storage()`` may have created independently of this
      migration. Re-running ``optimize_storage()`` restores it.
    """
    is_pg = _is_postgres()

    if is_pg:
        # SET LOCAL — transaction-scoped, so the timeout does not leak into
        # whatever the downgrade walk runs next. Same reasoning as upgrade().
        op.execute("SET LOCAL lock_timeout = '5s'")
        op.execute("ALTER TABLE documents ALTER COLUMN source_type DROP NOT NULL")
    else:
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "source_type",
                existing_type=sa.String(64),
                existing_nullable=False,
                nullable=True,
                existing_server_default=_DEFAULT_SOURCE_TYPE,
            )

    op.drop_index(_INDEX_NAME, table_name="documents", if_exists=True)
