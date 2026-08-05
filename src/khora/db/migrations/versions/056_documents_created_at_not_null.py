"""Make ``documents.created_at`` NOT NULL, backfilling any NULL first.

Revision ID: 056_documents_created_at_not_null
Revises: 055_documents_source_type_alignment
Create Date: 2026-08-04

``DocumentModel.created_at`` is ``Mapped[datetime]`` — non-optional, with a
Python-side default (``db/models.py``) — but the live column is NULLABLE on
both Alembic-managed dialects. Migration 000 created it as
``sa.Column("created_at", DateTime(timezone=True), server_default=sa.func.now())``
with no ``nullable`` argument, so it was created NULL-able, and nothing in the
chain has altered it since (009 / 019 / 054 index the column; 016 / 037 / 055
are the only revisions that ``alter_column`` on ``documents`` and none of them
names ``created_at``). Like 055, this is not a behaviour-change request — it is
the schema catching up to a declaration that has been in the ORM all along.

Why a NULL here is not a cosmetic drift
---------------------------------------
``created_at`` is the leading sort key of the total order pinned on document
listing (``ORDER BY created_at DESC, id DESC``), and a NULL breaks that order
in two distinct ways:

1. **The order diverges across backends.** SQLite sorts NULLs *last* under
   ``DESC``; PostgreSQL sorts them *first* (``NULLS FIRST`` is the ``DESC``
   default). The same namespace therefore paginates differently depending on
   which relational backend is underneath it — the exact class of divergence
   the pinned total order exists to eliminate.
2. **Keyset pagination skips the row, forever and silently.** A cursor
   predicate ``(created_at, id) < (:ts, :id)`` evaluates to NULL — not
   false — for a NULL-``created_at`` row, and NULL is not true, so the row is
   filtered out of *every* page. It is never returned and never raises. A
   document that exists is simply invisible to enumeration.

``updated_at`` is nullable too
------------------------------
Worth stating because it makes the second backfill branch below reachable
rather than dead code: migration 000 declared ``updated_at`` the same way, so a
row can carry NULL in both columns and the ``COALESCE``-style fallback to
``updated_at`` is not guaranteed to find a value.

The backfill is irreversible, and the epoch is deliberate
---------------------------------------------------------
Two ``UPDATE``s run before the flip; without them a single NULL row makes
``SET NOT NULL`` raise and rolls the revision back.

* ``created_at = updated_at`` where ``updated_at`` is present. An inferred but
  real timestamp — the document was last written then, so it existed then.
* ``created_at = <Unix epoch>`` for whatever remains. This value is
  **invented**, and its count is reported separately on the
  ``khora.migration.applied`` log event (``rows_epoch_stamped``, alongside
  ``rows_backfilled_from_updated_at``) precisely so an operator auditing the
  blast radius can see how many rows were guessed rather than inferred.

Neither rewrite is undone by ``downgrade()``. Once written, a repaired row is
indistinguishable from one that always carried a timestamp, so there is nothing
to restore (055 and 037 document their analogous normalizations the same way).

Why the epoch rather than ``now()`` or a hard failure:

* **Failing loudly has no operator remedy that differs from what this does.**
  A raise wedges the deploy; the manual fix is "pick a timestamp for the
  affected rows", and the epoch is that timestamp. Raising converts a
  repairable condition into an outage.
* **The epoch corrupts the tail; ``now()`` would corrupt the head.** Under
  ``created_at DESC`` an epoch row sorts dead last — deterministic, identical
  on both dialects, and *reachable by pagination*, which is the entire point.
  ``now()`` would float a genuinely old document to the top of every "most
  recent documents" page.
* **Either way it beats the status quo.** Today such a row is skipped on every
  page forever. After this revision it is returned, at a visibly wrong
  position. Visible-and-wrong beats invisible.
* **The population is near-empty by construction.** ``created_at`` is excluded
  from the partial-update whitelist in ``storage/backends/postgresql.py``
  ("id, namespace_id, created_at must never be patched"), is absent from the
  full-update column set, and ``DocumentModel.created_at`` carries a
  Python-side ``default=`` so even an explicit ``None`` on INSERT fires the
  default rather than reaching SQL. This is a defensive backfill and must not
  be able to block a deploy.

The epoch is passed as a **bound parameter**, never inlined into the SQL text,
and that is not stylistic. A literal ``'1970-01-01T00:00:00+00:00'`` ships two
defects on SQLite, both measured: it stores an ISO ``T``-separated string into a
column every other writer fills with ``'YYYY-MM-DD HH:MM:SS.ffffff'`` (and
SQLite compares that column as *text*, where ``'T'`` (0x54) sorts above ``' '``
(0x20) — so two rows at the same instant order differently), and it round-trips
through SQLAlchemy's ``DateTime`` as tz-**aware** while every other row
round-trips naive, so any Python-side comparison mixing them raises
``TypeError: can't compare offset-naive and offset-aware datetimes``.
Introducing format divergence into the ordering column is precisely the class
of bug this revision exists to remove. The bound parameter renders through the
dialect's own ``DateTime`` processor and lands in the format every other row
uses.

``type_=sa.DateTime(timezone=True)`` on that bindparam is a **separate** guard
and does *not* fix the above — attribute the two correctly. SQLAlchemy infers
``DateTime(timezone=True)`` from an aware datetime, so with ``_EPOCH`` as
shipped the typed and untyped forms compile byte-identically on both dialects
(verified against the asyncpg and pysqlite dialects). The annotation earns its
place only if ``_EPOCH`` ever loses its ``tzinfo``: the untyped form would then
silently bind ``WITHOUT TIME ZONE`` against a ``timestamptz`` column. It is a
forward-looking guard on a future edit, not what makes today's statement
correct.

On Postgres the binding is settled rather than assumed. The compiled statement
carries an explicit ``$1::TIMESTAMP WITH TIME ZONE`` cast and the value never
enters the SQL text, so a session ``TimeZone`` hazard is structurally
unreachable; the ORM column type and this bindparam resolve to the *same*
asyncpg dialect impl class, so the migration binds through the exact type every
document INSERT already uses. (The end-to-end round-trip was verified on the CI
Postgres legs; there was no PostgreSQL in the development container where this
was written, so the reasoning above was derived by compiling the statement
against the asyncpg dialect offline.)

Locking — this migration blocks ``documents`` on Postgres
----------------------------------------------------------
Read this before deploying against a large table.

* ``ALTER TABLE ... SET NOT NULL`` takes an ``ACCESS EXCLUSIVE`` lock — which
  blocks readers as well as writers — and performs a **full heap scan** to
  verify no NULL remains.
* Unlike 055 there is no index build, so the blocking window is the scan alone.
  Order of magnitude: tens of seconds at ~10M rows.
* ``SET LOCAL lock_timeout = '5s'`` bounds how long the statement waits to
  *acquire* the lock. It does **not** bound how long the lock is *held* once
  acquired, and it does not bound the scan.
* Compounding this: ``run_migrations()`` holds a session-scoped
  ``pg_advisory_lock`` for the whole run, and a concurrent caller waits only
  60s before raising ``TimeoutError`` — surfacing as
  ``MigrationResult(success=False)`` and then ``RuntimeError: Database
  migration failed`` at startup. On a deployment where the heap scan approaches
  that budget, other services booting with ``run_migrations=True`` fail to
  start for the duration. Migration 054's docstring warns about the same trap
  for its index build; run this out-of-band on such a deployment rather than
  during a rolling deploy.

Operator runbook for a large table, and why it is not in the migration
-----------------------------------------------------------------------
PG 12+ skips the verification scan when a **validated** ``CHECK (col IS NOT
NULL)`` already proves the invariant. An operator can therefore run, ahead of
the deploy::

    ALTER TABLE documents ADD CONSTRAINT tmp_documents_created_at_nn
        CHECK (created_at IS NOT NULL) NOT VALID;   -- brief ACCESS EXCLUSIVE
    ALTER TABLE documents VALIDATE CONSTRAINT tmp_documents_created_at_nn;
                                               -- SHARE UPDATE EXCLUSIVE, online

and drop the constraint afterwards. The ``SET NOT NULL`` below then applies
without the scan.

This is deliberately **not** done inside the migration. Under
``transaction_per_migration=True`` the ``ADD CONSTRAINT``'s ``ACCESS
EXCLUSIVE`` is held until COMMIT regardless, so in-transaction the split buys
nothing; getting the benefit requires separate transactions via an autocommit
block, which reintroduces exactly the partial-apply state that 055 deliberately
rejected. Note also that migrations 037 / 038 pin a PG 11 floor
(``_MIN_PG_VERSION = 110000``) and on PG 11 the skip does not exist at all —
the scan always runs, runbook or not.

Atomicity
---------
No ``autocommit_block``, no ``CONCURRENTLY``. One transaction that commits
together with its version stamp, so there is no partial-apply state to retry
into: either the backfill and the flip both landed, or neither did.

Cross-dialect
-------------
SQLite has no ``ALTER COLUMN``, so the flip goes through
``op.batch_alter_table("documents")`` — the documented table-rebuild procedure
(create temp, copy, ``DROP TABLE``, rename). That rebuild is **only** safe
because ``env.py`` leaves ``PRAGMA foreign_keys`` OFF on the migration
connection: with enforcement on, the ``DROP TABLE`` fires
``chunks.document_id ON DELETE CASCADE`` and deletes every chunk row before the
rename restores the table. See the comment in ``db/migrations/env.py``. The
``sqlite_lance`` fixture stack runs the full chain, so the SQLite branch is
load-bearing, not a green-keeper.

Deploy ordering
---------------
Unlike 055, **no call-site change has to ship first.** No reachable write path
emits a NULL ``created_at`` (see the backfill rationale above), so a live
instance on the old code cannot abort the ``SET NOT NULL`` mid-flight. Stated
explicitly because a reader arriving from 055 will expect the same constraint.

What NOT NULL does and does not buy
------------------------------------
It rules out NULL, which is what makes the total order well-defined and keyset
pagination complete **on the two Alembic-managed tiers** — Postgres and the
SQLite the ``sqlite_lance`` stack migrates. It does **not** make an
epoch-stamped row's timestamp *correct* — only orderable and reachable.

Scope the ordering claim there deliberately, because the other tiers are not
covered by it. ``SQLiteRelationalBackend`` (``backend: sqlite``, the
zero-infrastructure mode) builds ``documents`` from a string literal in
``storage/backends/sqlite.py`` at ``connect()`` time with no Alembic chain, and
the SurrealDB backend declares the field ``TYPE datetime`` rather than
``option<datetime>``. Both already reject a NULL ``created_at`` independently,
so the NOT NULL invariant holds on all four; see the comments at those two
sites. But the raw-sqlite tier writes ``datetime.isoformat()`` into a ``TEXT``
column and orders it lexicographically, which stores an offset-carrying,
``T``-separated string (e.g. ``'2026-08-04T22:24:25.150607+00:00'``) — the same
storage-format divergence the epoch bindparam above exists to avoid, already
present there for every row. That is a pre-existing property of a separate
backend, not something this revision introduces or repairs; it is named so the
"ordering is now well-defined" claim is not read as global.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import DBAPIError

revision: str = "056_documents_created_at_not_null"
down_revision: str | Sequence[str] | None = "055_documents_source_type_alignment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

# Last-resort backfill value for a row where BOTH created_at and updated_at are
# NULL. Sorts dead last under ``created_at DESC`` on every backend. Bound as a
# typed parameter, never interpolated — see the module docstring.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: DBAPIError) -> bool:
    """Distinguish a real lock_timeout trip from any other database error.

    The caught class is ``DBAPIError``, not ``OperationalError``, and that is
    load-bearing. On asyncpg — the driver khora runs on Postgres — a
    lock-timeout arrives as ``asyncpg.exceptions.LockNotAvailableError``, which
    subclasses ``PostgresError``; the asyncpg dialect maps ``PostgresError`` to
    the *base* DBAPI ``Error`` class, so SQLAlchemy wraps it in a plain
    ``DBAPIError`` and ``isinstance(wrapped, OperationalError)`` is **False**.
    An ``except OperationalError`` here would never fire on the one event the 5s
    timeout exists to produce.

    Both attribute spellings are read because they are genuinely different
    drivers, not belt-and-braces: **asyncpg carries the code on ``sqlstate`` and
    leaves ``pgcode`` as ``None``**, while psycopg2/psycopg use ``pgcode``.
    ``env.py`` normalizes Postgres URLs to ``postgresql+asyncpg``, so a
    ``pgcode``-only check is dead on this driver. Same reasoning as 055, whose
    docstring records the verification.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    for attr in ("sqlstate", "pgcode"):
        if getattr(orig, attr, None) == _PG_LOCK_NOT_AVAILABLE:
            return True
    return False


def _upgrade_impl() -> tuple[int, int]:
    """Backfill NULLs, then install the constraint.

    Returns ``(rows_backfilled_from_updated_at, rows_epoch_stamped)``.
    """
    is_pg = _is_postgres()

    if is_pg:
        # Issued before any DML, so it bounds lock *acquisition* for every
        # statement in this revision — both backfill UPDATEs (which take a
        # ROW EXCLUSIVE lock and can queue behind a conflicting transaction) and
        # the SET NOT NULL's ACCESS EXCLUSIVE. A stuck pg_stat_activity entry on
        # documents therefore cannot stall the deploy past 5s waiting to start,
        # at any step. It does NOT bound how long a lock is held once acquired,
        # nor the heap scan — see the Locking section of the module docstring.
        #
        # SET LOCAL, not a bare SET: with transaction_per_migration=True this
        # revision owns its transaction and LOCAL scopes the setting to it. A
        # bare SET is session-scoped and would leak the 5s timeout onto every
        # later revision in the same `alembic upgrade head` run — which is
        # exactly why 055 uses SET LOCAL and names this revision as the reason.
        op.execute("SET LOCAL lock_timeout = '5s'")

    bind = op.get_bind()

    # Two statements rather than one COALESCE, purely for observability: the
    # split separates rows whose timestamp was *inferred* from a real value
    # from rows whose timestamp was *invented*. Same final data, same scan
    # count. Statement order matters — the epoch sweep must run second so it
    # only sees rows the inference could not repair.
    result_inferred = bind.execute(
        sa.text("UPDATE documents SET created_at = updated_at WHERE created_at IS NULL AND updated_at IS NOT NULL")
    )
    result_epoch = bind.execute(
        sa.text("UPDATE documents SET created_at = :epoch WHERE created_at IS NULL").bindparams(
            # BOUND, not inlined: that is what makes the value render through
            # the dialect's DateTime processor into the same storage format
            # every other row uses. type_ is a SEPARATE, forward-looking guard —
            # redundant today (SQLAlchemy infers timezone=True from an aware
            # _EPOCH) but it stops a future edit that drops tzinfo from silently
            # binding WITHOUT TIME ZONE. See the module docstring; do not
            # conflate the two.
            sa.bindparam("epoch", value=_EPOCH, type_=sa.DateTime(timezone=True))
        )
    )
    # max(..., 0): the DBAPI contract lets rowcount be -1 when a driver cannot
    # report a count, and a negative row count in a log field is noise (055
    # carries the same guard).
    rows_from_updated_at = max(int(result_inferred.rowcount or 0), 0)
    rows_epoch_stamped = max(int(result_epoch.rowcount or 0), 0)

    if is_pg:
        op.execute("ALTER TABLE documents ALTER COLUMN created_at SET NOT NULL")
    else:
        # SQLite has no ALTER COLUMN; batch mode performs the table rebuild.
        # Safe only because env.py leaves PRAGMA foreign_keys OFF — see the
        # Cross-dialect section of the module docstring.
        #
        # ``nullable=False`` is passed EXPLICITLY: ``existing_nullable`` alone is
        # only a hint to the renderer and applies nothing (migration 037 is the
        # proof — it passed ``existing_nullable=False`` and the column stayed
        # nullable).
        #
        # ``existing_server_default`` is deliberately omitted. Batch mode
        # reflects the live table, so the CURRENT_TIMESTAMP default survives
        # either way; passing it would hardcode SQLite's rendering of a
        # ``sa.func.now()`` default — a claim that can go stale silently, where
        # reflection cannot. (055 passes one, but ``'library'`` is a
        # dialect-neutral literal; ``CURRENT_TIMESTAMP`` is not.)
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "created_at",
                existing_type=sa.DateTime(timezone=True),
                existing_nullable=True,
                nullable=False,
            )

    return rows_from_updated_at, rows_epoch_stamped


def upgrade() -> None:
    start = time.monotonic()
    # Initialize log fields up-front so the error path emits the same field set
    # as the success path. "Same shape", not "always": only a DBAPIError is
    # caught, so a failure that is not database-originated propagates without an
    # event. Nothing is swallowed and no default is returned — the handler logs
    # and re-raises — so no structured degradation record applies here.
    rows_from_updated_at = 0
    rows_epoch_stamped = 0
    try:
        rows_from_updated_at, rows_epoch_stamped = _upgrade_impl()
    except DBAPIError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            rows_backfilled_from_updated_at=rows_from_updated_at,
            rows_epoch_stamped=rows_epoch_stamped,
        ).error("khora.migration.applied")
        # Bare ``raise`` preserves the original traceback. The per-migration
        # transaction rolls back the backfill along with the flip.
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        rows_backfilled_from_updated_at=rows_from_updated_at,
        rows_epoch_stamped=rows_epoch_stamped,
    ).info("khora.migration.applied")


def downgrade() -> None:
    """Drop the NOT NULL constraint.

    The constraint is the only thing reversed. **The backfill is not undone**:
    once ``created_at`` has been rewritten, a repaired row is indistinguishable
    from one that always carried a timestamp, so there is nothing to restore.
    Downgrading therefore returns the column to nullable but leaves every
    inferred and epoch-stamped value in place. 055 and 037 document their
    analogous normalizations the same way.

    No index is created by this revision, so none is dropped.
    """
    if _is_postgres():
        # SET LOCAL — transaction-scoped, so the timeout does not leak into
        # whatever the downgrade walk runs next. Same reasoning as upgrade().
        op.execute("SET LOCAL lock_timeout = '5s'")
        op.execute("ALTER TABLE documents ALTER COLUMN created_at DROP NOT NULL")
    else:
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "created_at",
                existing_type=sa.DateTime(timezone=True),
                existing_nullable=False,
                nullable=True,
            )
