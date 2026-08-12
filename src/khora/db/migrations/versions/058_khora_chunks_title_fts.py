"""Fold ``khora_chunks.title`` into ``content_tsv`` and recompute every row.

Revision ID: 058_khora_chunks_title_fts
Revises: 057_drop_documents_created_at_index
Create Date: 2026-08-12

Migration 041 added the denormalized ``khora_chunks.title`` column and 044
backfilled it from the parent ``documents`` row, but the lexical index never
saw it: the ``khora_chunks_content_tsv_trigger()`` function has only ever
computed ``to_tsvector('english', NEW.content)``. A chunk whose *title* is the
only place a term appears is therefore invisible to the BM25 / ts_rank channel
— the gap #1574 closes. This revision swaps the function to a weighted
concatenation and recomputes the stored vectors::

    setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
 || setweight(to_tsvector('english', NEW.content), 'B')

``coalesce`` because ``title`` is nullable and ``to_tsvector(NULL)`` is NULL,
which would annihilate the whole concatenation and blank the vector.

Lockstep with the runtime store — this is a convergence contract
-----------------------------------------------------------------
``khora_chunks`` is **not** part of the Alembic-managed schema: it is created
at runtime by ``PgVectorTemporalStore.connect()``, which issues its own
``CREATE OR REPLACE FUNCTION`` / ``DROP TRIGGER`` / ``CREATE TRIGGER`` on every
boot. So does this revision. Whichever runs last wins, and both must therefore
install the *same* function — ``_TSV_FUNCTION_SQL`` below is byte-identical to
the constant of the same name in ``khora/storage/temporal/pgvector.py``. Change
both in the same commit. This mirrors the arrangement 044 already has with that
store's filter-index DDL, and 055 has with ``storage/optimize.py``.

The copy is duplicated rather than imported on purpose: a migration is a frozen
snapshot of an intent at a point in the chain, and importing runtime code would
let a future refactor retroactively change what this revision does.

Mixed-version deployments have a write window that loses title tokens
----------------------------------------------------------------------
Because the runtime reinstalls the function at ``connect()``, an instance still
on the **old** khora version that (re)connects after this migration runs will
``CREATE OR REPLACE`` the content-only formula back over it. Rows it writes
until a new-version instance next connects carry a vector with no title tokens.
Nothing errors and nothing is corrupted — those rows are simply not
title-searchable, and are repaired by any subsequent write to them (or by
re-running this migration). Single-app-version deployments — the normal case —
never enter this window. During a rolling deploy, keep it short by rolling
forward promptly; it closes the moment the last old-version pod is gone and any
new-version pod has connected.

Step order is load-bearing
--------------------------
1. ``CREATE OR REPLACE FUNCTION`` with the new formula.
2. ``DROP TRIGGER IF EXISTS`` + ``CREATE TRIGGER``.
3. ``UPDATE khora_chunks SET content_tsv = NULL``.

Step 3 does **not** inline the formula. ``khora_chunks_content_tsv_update`` is a
``BEFORE INSERT OR UPDATE`` trigger, so it overwrites ``NEW.content_tsv`` on
every row this UPDATE touches — an inlined expression would be computed and
then immediately discarded, i.e. dead code that reads like the load-bearing
part. Assigning NULL makes the trigger the single definition of the formula.

That is also exactly why step 3 must run *after* steps 1-2 and never before.
Before the swap it would recompute the old formula (pointless); and without
step 2 it would blank ``content_tsv`` outright on any database where the
trigger is missing or disabled, silently turning off full-text search for the
whole table. Step 2 makes the revision self-contained rather than dependent on
the runtime having booted first, and it repairs one real state: 044's documented
crash caveat can leave the trigger ``DISABLE``d, and a disabled trigger would
not fire on step 3's UPDATE. ``DROP``+``CREATE`` restores it enabled.

Cost — this is a full-table rewrite; read before deploying at scale
--------------------------------------------------------------------
Step 3 updates every row in ``khora_chunks``:

* Postgres writes a new heap tuple per row (an UPDATE is never a no-op even
  when the recomputed value is unchanged), so expect table bloat on the order
  of the table's own size until autovacuum catches up. 044 produced the same
  shape of churn and documents it the same way.
* ``content_tsv`` is GIN-indexed (``ix_khora_chunks_content_tsv``), so no update
  can be HOT — every row costs index maintenance too. This dominates the
  runtime.
* The lock taken is ``ROW EXCLUSIVE``. **Concurrent reads are unaffected**;
  concurrent writers to the *same rows* block until commit.
* ``SET LOCAL lock_timeout = '5s'`` bounds how long the UPDATE waits to
  *acquire* its lock. It does not bound how long the statement runs, nor how
  long its locks are held once acquired.

The compounding hazard is the advisory lock, not the table lock:
``run_migrations()`` holds a session-scoped ``pg_advisory_lock`` for the entire
run and a concurrent caller waits only 60s before raising ``TimeoutError``,
which surfaces as ``RuntimeError: Database migration failed`` at startup. On a
deployment with a large ``khora_chunks``, other services booting with
``run_migrations=True`` will fail to start for as long as the rewrite takes.
Run it out-of-band there rather than during a rolling deploy — the same
guidance migrations 054 and 056 give for their own long statements.

Re-running is safe but not cheap: the outcome is idempotent, the cost is not.
There is no sentinel that distinguishes an already-recomputed row (both
formulas leave ``content_tsv`` non-NULL), so a re-run rewrites the whole table
again. This is deliberate — the alternatives (probing for weight labels in the
stored vector) are fragile and no cheaper.

Atomicity
---------
One transaction, no ``autocommit_block``, no ``CONCURRENTLY``. The function
swap, the trigger recreate and the recompute commit together with the version
stamp, so there is no partial state where the new function is installed but the
existing rows were never recomputed. The trade is that the UPDATE's snapshot is
open for the whole rewrite; that is the accepted cost of not shipping a
half-applied formula.

Ranking compatibility
---------------------
Labelling content ``'B'`` changes nothing about ranking on its own *because*
``_bm25_search`` passes ``ts_rank_cd`` an explicit ``{D, C, B, A}`` weights
vector rather than relying on the implicit default ``{0.1, 0.2, 0.4, 1.0}``.
At the default ``title_weight=1.0`` it passes ``{0.1, 0.2, 0.1, 0.1}``, so a
content hit scores exactly what it scored as an unlabelled (D) token before
this revision. The match predicate ``content_tsv @@ tsquery`` is
label-independent and is unchanged.

Dialect and embedded posture
----------------------------
Postgres-only. Early-returns on other dialects (SQLite), so it is a clean no-op
on the ``sqlite_lance`` test fixture.

On Postgres it is additionally guarded by a two-part precondition — the
``khora_chunks`` table must exist *and* must carry the denormalized ``title``
column — and no-ops cleanly when either half is missing. Both halves are
reachable, and neither is an error:

* **No table.** ``khora_chunks`` is runtime-managed, so on a fresh deploy it
  does not exist when migrations run. The store's ``connect()`` installs the
  new function itself and there are no rows to recompute. Same gate 041 / 044
  use.
* **Table without ``title``.** A legacy or partially-migrated shape (anything
  predating 041, or a minimal hand-built table) can exist without the
  denormalized column. There is no title to fold in, so there is nothing to do.

The second half is load-bearing, not defensive tidiness. plpgsql resolves
``NEW.title`` **when the trigger fires**, not when the function is created, so
against a title-less table the ``CREATE OR REPLACE FUNCTION`` and
``CREATE TRIGGER`` both succeed and the recompute ``UPDATE`` is what raises
``UndefinedColumnError: record "new" has no field "title"``. The blast radius
would exceed this revision: the swapped function stays installed, so every
later INSERT or UPDATE to ``khora_chunks`` would fail too — it is only the
per-migration transaction rolling the whole revision back that contains it.
The gate applies to ``_run``, so upgrade and downgrade skip the same databases.

The embedded (``sqlite_lance``) stack gets its title FTS on the DDL side
instead, in its own store. **Pre-existing embedded databases keep content-only
FTS**: there is no migration chain over that store's hand-written schema, so
enabling title search there means recreating the store and re-ingesting.

Downgrade
---------
Fully symmetric and complete: it reinstalls the pre-#1574 content-only
function, recreates the trigger, and recomputes every row back to a
content-only vector. Unlike 055 / 056 nothing here is irreversible — the vector
is derived data, reconstructible in either direction from ``title`` and
``content``, which are untouched. The downgrade pays the same full-table
rewrite cost as the upgrade.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import DBAPIError

revision: str = "058_khora_chunks_title_fts"
down_revision: str | Sequence[str] | None = "057_drop_documents_created_at_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

_TSV_TRIGGER = "khora_chunks_content_tsv_update"

# LOCKSTEP: byte-identical to ``_TSV_FUNCTION_SQL`` in
# ``khora/storage/temporal/pgvector.py``, which issues the same statement at
# every ``PgVectorTemporalStore.connect()``. Whichever runs last wins, so a
# drift between the two makes the installed formula depend on boot order.
# Change both in the same commit. See the module docstring.
_TSV_FUNCTION_SQL = """CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
                    || setweight(to_tsvector('english', NEW.content), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql"""

# The pre-#1574 formula, restored by ``downgrade()``. Whitespace-normalized
# relative to the string this revision replaces (that copy was indented inside
# the store's ``connect()``); the statement itself is unchanged.
_TSV_FUNCTION_SQL_CONTENT_ONLY = """CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql"""

_DROP_TRIGGER_SQL = f"DROP TRIGGER IF EXISTS {_TSV_TRIGGER} ON khora_chunks"

_CREATE_TRIGGER_SQL = f"""CREATE TRIGGER {_TSV_TRIGGER}
BEFORE INSERT OR UPDATE ON khora_chunks
FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()"""

# Assigning NULL, not the formula: the BEFORE trigger overwrites
# ``NEW.content_tsv`` on every touched row, so an inlined expression would be
# dead code. See the module docstring.
_RECOMPUTE_SQL = sa.text("UPDATE khora_chunks SET content_tsv = NULL")


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _has_khora_chunks() -> bool:
    return sa.inspect(op.get_bind()).has_table("khora_chunks")


def _has_title_column() -> bool:
    """Is there a ``khora_chunks`` table that actually carries ``title``?

    Both halves of the precondition, because both are reachable and neither is
    an error. ``khora_chunks`` is runtime-managed, so on a fresh deploy it does
    not exist yet; and a legacy or partially-migrated shape (anything predating
    041, or a minimal hand-built table) can exist *without* the denormalized
    ``title`` column. Either way there is no title to fold into the vector, so
    the revision is a clean no-op rather than a failure.

    The column half is not defensive tidiness — without it this revision breaks
    such a database. plpgsql resolves ``NEW.title`` **when the trigger fires**,
    not when the function is created, so ``CREATE OR REPLACE FUNCTION`` and
    ``CREATE TRIGGER`` both succeed against a title-less table and the recompute
    ``UPDATE`` is what raises ``UndefinedColumnError: record "new" has no field
    "title"``. Worse, the failure is not confined to this revision: the swapped
    function would remain installed, so every subsequent INSERT or UPDATE to
    ``khora_chunks`` would fail too, had the per-migration transaction not
    rolled it back.
    """
    if not _has_khora_chunks():
        return False
    columns = sa.inspect(op.get_bind()).get_columns("khora_chunks")
    return any(c["name"] == "title" for c in columns)


def _is_lock_timeout(exc: DBAPIError) -> bool:
    """Distinguish a real lock_timeout trip from any other database error.

    The caught class is ``DBAPIError``, not ``OperationalError``, and that is
    load-bearing: on asyncpg a lock-timeout arrives as
    ``LockNotAvailableError``, which the dialect maps to the *base* DBAPI
    ``Error`` class, so SQLAlchemy wraps it in a plain ``DBAPIError`` and an
    ``except OperationalError`` would never fire. Both attribute spellings are
    read because asyncpg carries the code on ``sqlstate`` (leaving ``pgcode``
    ``None``) while psycopg2/psycopg use ``pgcode``. Same reasoning as 056,
    whose docstring records the verification.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    for attr in ("sqlstate", "pgcode"):
        if getattr(orig, attr, None) == _PG_LOCK_NOT_AVAILABLE:
            return True
    return False


def _swap_formula(function_sql: str) -> int:
    """Install ``function_sql``, recreate the trigger, recompute every row.

    Returns the number of rows recomputed. Shared by both directions — they
    differ only in which function body is installed.
    """
    # Issued before any DML so it bounds lock *acquisition* for every statement
    # in this revision: the DDL's ACCESS EXCLUSIVE on the trigger and the
    # UPDATE's ROW EXCLUSIVE. LOCAL, so the 5s setting dies with this
    # revision's transaction rather than leaking onto the next revision in the
    # same ``alembic upgrade head`` run (055 / 056 / 057 do the same).
    op.execute("SET LOCAL lock_timeout = '5s'")

    op.execute(function_sql)
    op.execute(_DROP_TRIGGER_SQL)
    op.execute(_CREATE_TRIGGER_SQL)

    result = op.get_bind().execute(_RECOMPUTE_SQL)
    # max(..., 0): the DBAPI contract lets rowcount be -1 when a driver cannot
    # report a count, and a negative row count in a log field is noise (055 /
    # 056 carry the same guard).
    return max(int(result.rowcount or 0), 0)


def _run(function_sql: str) -> None:
    """Apply ``function_sql`` in either direction, with the structured log.

    The precondition is checked here rather than in ``upgrade`` / ``downgrade``
    so both directions no-op on exactly the same set of databases — a downgrade
    that tried to recompute a table the upgrade had skipped would fail the same
    way, for the same reason.
    """
    if not _is_postgres() or not _has_title_column():
        return

    start = time.monotonic()
    # Initialized up-front so the error path emits the same field set as the
    # success path. Only DBAPIError is caught; a non-database failure
    # propagates without an event. Nothing is swallowed — the handler logs and
    # re-raises — so no ADR-001 degradation record applies here.
    rows_recomputed = 0
    try:
        rows_recomputed = _swap_formula(function_sql)
    except DBAPIError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            rows_recomputed=rows_recomputed,
        ).error("khora.migration.applied")
        # Bare ``raise`` preserves the traceback. The per-migration transaction
        # rolls the function swap back along with the recompute.
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        rows_recomputed=rows_recomputed,
    ).info("khora.migration.applied")


def upgrade() -> None:
    _run(_TSV_FUNCTION_SQL)


def downgrade() -> None:
    """Restore the content-only formula and recompute every row.

    Fully reversible: ``content_tsv`` is derived data and both ``title`` and
    ``content`` are untouched, so the vector is reconstructible in either
    direction. Pays the same full-table rewrite cost as the upgrade.
    """
    _run(_TSV_FUNCTION_SQL_CONTENT_ONLY)
