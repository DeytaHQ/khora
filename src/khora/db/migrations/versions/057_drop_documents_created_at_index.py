"""Drop the single-column ``ix_documents_created_at`` index.

Revision ID: 057_drop_documents_created_at_index
Revises: 056_documents_created_at_not_null
Create Date: 2026-08-04

Migration 009 created ``ix_documents_created_at ON documents (created_at)``
unconditionally on both dialects, to serve temporal-filter pushdown. This
revision removes it.

Operational profile — read this first
-------------------------------------
* **The upgrade is cheap.** It takes ``ACCESS EXCLUSIVE`` on
  ``documents``, but the work behind that lock is a catalog delete and an
  unlink — milliseconds, independent of table size — and ``lock_timeout``
  bounds the wait to acquire it at 5s. Do **not** carry 055's "tens of
  seconds at ~10M rows, minutes at ~100M" language over to this revision;
  that describes a full heap scan and an index build, and neither happens
  here.
* **The downgrade is the expensive direction, and on a busy table the
  concurrent pre-build is what makes it likely to succeed at all.** A
  plain ``CREATE INDEX`` takes ``SHARE`` for the whole build and blocks
  writes to ``documents`` until it finishes. ``SHARE`` also conflicts with
  ``ROW EXCLUSIVE``, which every concurrent INSERT/UPDATE/DELETE holds, so
  with ``lock_timeout`` at 5s the acquisition will typically **fail**
  rather than block on a table taking sustained writes — the downgrade
  errors out instead of stalling. That is the safer failure, but it means
  the runbook here is not merely a speed-up. Same shape as 055's: run

      CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_created_at
          ON documents (created_at);

  **before** the downgrade, so ``if_not_exists=True`` reduces the
  migration's create to a no-op.

  Be precise about what that buys, because the obvious reading is wrong:
  it removes the *hold*, not the *acquisition*. Postgres opens the table
  with ``SHARE`` inside ``DefineIndex()`` **before** it evaluates
  ``IF NOT EXISTS``, and the skip path then closes the relation with
  ``NoLock`` — i.e. the lock is still taken and still held to COMMIT. So a
  pre-built index turns a minutes-long ``SHARE`` hold into a brief one,
  but the acquisition can still trip the 5s ``lock_timeout`` under
  sustained writes. Pre-building makes the downgrade far more likely to
  succeed; it does not guarantee it.
* **No window without a usable index.** ``ix_documents_namespace_created_at_id``
  is untouched in both directions, so every namespace-scoped
  ``created_at`` read stays served throughout.

Scope — who was using this index
--------------------------------
Nobody, through the public API. Every public ``documents`` accessor takes
``namespace_id`` as a required parameter, so a ``kb.storage`` consumer
cannot issue a namespace-free ``created_at`` scan without bypassing the
API into raw SQL. Namespace-scoped reads are served by the 054 index,
which leads on ``namespace_id``.

Two shapes genuinely regress, and both share one mitigation — recreate
the index with the ``CREATE INDEX CONCURRENTLY`` one-liner above, which
needs no migration:

1. **Ad-hoc operator SQL** doing a namespace-free ``created_at`` scan.
   This cannot be enumerated from inside the repository, so it is named
   rather than dismissed.
2. **A heavily single-tenant ``documents`` table** — the shape migration
   054 explicitly warned about. Where one namespace is most of the table,
   the dropped index was plausibly the better plan for full-drain offset
   pagination: the same heap fetches, roughly a third of the index pages,
   and the Incremental Sort it forces runs over tie-groups that the plan
   test's own fixture comment expects to be size 1 in production. The
   decision does not change — multi-tenant is khora's stated regime, and
   the 054 index is the right default — but a single-tenant deployment
   measuring a regression here is not imagining it.

Two surfaces look like counterexamples when grepping for ``created_at``
and are not:

* ``_documents_compile_context`` (``storage/backends/postgresql.py`` plus
  the backend mirrors) carries ``created_at`` in ``_BACKED_SYSTEM_KEYS``,
  so the documents filter tier does compile ``created_at`` predicates. It
  has no production caller — the only construction site is the documents
  compile-context unit test, as 055's docstring also records — and when
  one arrives it will be namespace-scoped and served by the 054 index.
* The ``backend: sqlite`` stack (``SQLiteRelationalBackend`` /
  ``SQLiteVectorBackend``) hand-writes its ``documents`` DDL as a string
  literal in ``storage/backends/sqlite.py`` at ``connect()`` time,
  entirely outside Alembic, and never created this index under any name.

Why it goes — one reason, not four
----------------------------------
The reason is **planner competition**, and it is the only independent
ground. Migration 054 widened ``ix_documents_namespace_created_at`` to
``(namespace_id, created_at, id)`` so that ``list_documents``' pinned
``ORDER BY created_at DESC, id DESC`` can be answered by an index-order
scan instead of an Incremental Sort. This index competes with that one.

Be exact about the evidence, because the short version overstates it.
What was **observed**: on 054's branch, at a deep offset, the planner
chose this single-column index over the 3-column one and the sort-free
plan was lost. That happened **once**, in a fixture state that no longer
exists — before ``DECOY_NAMESPACES`` was introduced. The decoy commit
made ``namespace_id`` discriminating and displaced this index from the
losing plan; it has not appeared in one since, and the later failures
recorded in
``tests/integration/storage/test_list_documents_index_plan_pg.py`` are a
bitmap heap scan and ``ix_documents_namespace_id`` + sort, neither of
which this revision touches.

So: **the fixture change already removed this index as a CI competitor.
057 removes it from production**, where the hazard is inferred rather
than measured. The inference is that production resembles the *pre*-decoy
fixture more than the current one — ``documents`` is appended in
``created_at`` order, so that column's correlation sits near +1 on a real
table, against the interleaved fixture's ~-0.2. Do not read this docstring
as describing a hazard live in CI at head; it is not, and a reader in six
months should not have to reconstruct that from the commit log.

The **mechanism** is inference too — nobody instrumented
``btcostestimate``. Both indexes can *supply* the required ``created_at``
order: the dropped one leads on ``created_at`` directly, the 054 one
yields it as residual order once ``namespace_id`` is equality-constrained.
Index entry width is the obvious candidate for the tiebreak — one
timestamptz (~8 bytes) against uuid + timestamptz + uuid (~40), a 3-4x
page difference at a depth of thousands of entries, which also fits the
failure appearing at depth rather than on the first page.

Note what that argument may **not** lean on. The two are not priced off a
common heap-access discount: ``btcostestimate`` derives that discount from
the correlation of the index's *leading* column, and the leading columns
differ — ``created_at`` for the dropped index, ``namespace_id`` for the
054 one. Neither correlation figure above prices the 054 index; both
describe ``created_at``, which prices only the index this revision drops.
So there is no "same discount, therefore width decides it" derivation
available, which makes the width story a guess about the outcome rather
than a reconstruction of the arithmetic.

The limit of the claim
----------------------
057 removes the cheapest-to-choose competitor to the 054 index. It does
**not** restore an unaided sort-free plan at depth: two other losing
plans are on the record and survive this revision untouched. The
deep-offset assertion therefore stays on its ``enable_sort = off``
capability form, and nothing here changes that.

That the index is *also* undeclared in the ORM, and *also* costs a write
on every document insert, is what makes the drop cheap and clean — no ORM
edit, no drift-ledger line, and, on the index set the chain builds for
SQLite, exact ORM/schema agreement on ``documents`` afterwards. That last
clause is dialect-qualified on purpose: Postgres keeps two further
ORM-undeclared ``documents`` indexes that this revision does not touch and
SQLite never builds — ``ix_documents_ns_session`` (migration 031, behind a
Postgres gate) and ``ix_documents_graph_mirror_pending`` (migration 051,
likewise). Parity on ``documents`` is therefore exact on the SQLite leg
and merely improved on Postgres.

Neither property is a reason the drop is correct. ``chunks`` carries five
similarly ORM-undeclared indexes on that same SQLite leg — measured at
head: ``ix_chunks_created_at``, ``ix_chunks_ns_created``,
``ix_chunks_source_ts`` (all three from migration 009),
``ix_chunks_ns_temporal`` (017), and
``ix_chunks_last_accessed_at``. ``ix_chunks_ns_temporal`` hides from two
different audits and is worth naming: 017 issues it as raw ``op.execute``
rather than ``op.create_index``, so grepping the migrations for ``create_index``
misses it, **and** it is expression-based
(``(namespace_id, COALESCE(source_timestamp, created_at))``), so
SQLAlchemy's SQLite reflection skips it with a warning and it is absent
from ``inspect(engine).get_indexes()`` too. Count this set against
``sqlite_master`` rather than against reflection.
Postgres carries more, not fewer. ``chunks`` is also the high-write table
of the two. They all stay, because nothing has measured them competing
with anything. Any argument from "undeclared" or "write amplification"
would apply to those five with equal or greater force, so neither can be
the argument here.

Read "nothing has measured them" as absence of evidence, not evidence of
absence — nobody wrote a plan test for ``chunks``, so that is a property
of the test suite rather than of the indexes. The same competition this
revision reasons about plausibly exists there right now:
``PgvectorBackend.list_chunks`` is the same namespace-scoped
``ORDER BY created_at DESC`` shape, and ``chunks`` carries
``ix_chunks_created_at`` and ``ix_chunks_ns_created`` from 009 plus
``idx_chunks_namespace_created`` shipped by the public
``optimize_storage()`` — candidate indexes on overlapping leading columns,
on the bigger and higher-write table. Whoever measures that is the one who
gets to decide it; this revision deliberately does not.

No ``CONCURRENTLY``, therefore no invalid-index probe
-----------------------------------------------------
A reviewer who knows migration 054 will look for its
``_drop_invalid_index`` / ``pg_index.indisvalid`` helper and find it
absent. That is deliberate. ``CONCURRENTLY`` is what *creates* the
INVALID-index failure mode — 054 needed the probe only because it also
needed a concurrent *build*. Copying the helper here would be dead code
justifying a hazard that cannot occur.

054 needed ``CONCURRENTLY`` because a ``CREATE INDEX`` on a large table
holds ``SHARE`` for the whole build, which is minutes. A ``DROP INDEX``
is a catalog delete plus a file unlink: the ``ACCESS EXCLUSIVE`` lock is
*held* for milliseconds. What ``DROP INDEX CONCURRENTLY`` would buy in
exchange is an unbounded wait for every conflicting transaction to drain
— while holding the 60s migration advisory lock that ``run_migrations()``
takes for the whole run — plus a revision that is no longer atomic with
its version stamp (``CONCURRENTLY`` cannot run inside a transaction), plus
a dead-index recovery path to maintain. All cost, no benefit. See 055's
Atomicity section, which makes this argument and applies here verbatim:
under ``transaction_per_migration=True`` this revision commits together
with its version stamp, so there is no partial-apply state to retry into.

``SET LOCAL lock_timeout``, Postgres-gated
------------------------------------------
The drop still has to *acquire* ``ACCESS EXCLUSIVE``, and while it waits
it queues ahead of every subsequent ``documents`` query. A single
long-running reader would otherwise stall both the deploy and the table
behind it indefinitely. ``lock_timeout`` bounds the acquisition at 5s;
the revision then fails and is retried rather than blocking the table.

``SET LOCAL``, not a bare ``SET``: ``env.py`` runs with
``transaction_per_migration=True``, so this revision has its own
transaction and ``LOCAL`` scopes the setting to it. A bare ``SET`` is
session-scoped and would leak the 5s timeout onto every later revision in
the same ``alembic upgrade head`` run — 055's docstring warns about this
by name, naming *this* revision as the one that would inherit it. Note
that the absence of an ``autocommit_block`` is what puts ``SET LOCAL`` in
scope for the DDL at all: dropping ``CONCURRENTLY`` is what makes the
timeout work.

The gate is on the ``SET``, not on the DDL. ``op.drop_index`` /
``op.create_index`` render ``IF EXISTS`` / ``IF NOT EXISTS`` natively on
SQLite as well (055 already relies on both), so the statements themselves
need no dialect branch.

``if_exists=True`` on the drop
------------------------------
"Wherever 009 ran, the index exists" holds for the migration chain but
not for real databases, on exactly two grounds:

* Migration 054's docstring explicitly tells operators that on a heavily
  single-tenant ``documents`` table this index may out-compete the new
  one, and to "measure rather than assume". An operator who took that
  advice, measured, and dropped the index would have their upgrade
  aborted by a bare drop.
* ``db/session.py`` still ships the deprecated ``init_db()``
  (``Base.metadata.create_all``), and the ORM declares no such index. A
  database built that way has the table and not the index.

There is no SQLite ``batch_alter_table`` copy hazard here — 055's batch
copy reflects and recreates existing indexes correctly — so that is
deliberately not claimed.

This does not weaken the downgrade-symmetry argument 054 makes. 054's
symmetry is load-bearing because its drop targets an index its own
upgrade created moments earlier: absence there means something went
wrong, and swallowing it would hide a real fault. This revision's target
was created 47 revisions earlier, under arbitrary operational history.
``if_exists`` plus ``if_not_exists`` buys idempotency in both directions,
which is the property that matters at that distance.

The downgrade restore is load-bearing
-------------------------------------
Migration 009's ``downgrade()`` calls
``op.drop_index("ix_documents_created_at", "documents")`` — table name
passed, ``if_exists`` **absent**. This is the shape 054 warns about: do
not read a qualified call as proof the restore is unnecessary. If this
revision's ``downgrade()`` failed to put the index back, any downgrade
walk past 009 would abort on both dialects. Ordering was checked and
there is no hazard: the walk is 057 -> 056 -> 055 -> 054 -> ... -> 009.
056's downgrade touches only ``documents.created_at``'s nullability, 055's
only ``source_type`` and ``ix_documents_namespace_source_type``, and 054's
only the two ``namespace_created_at*`` indexes. None of them drops or
rebuilds ``ix_documents_created_at``, so the index this revision restores
survives to 009, which is what makes the restore load-bearing rather than
merely tidy. Note 056 and 055 both rewrite ``documents`` through a SQLite
``batch_alter_table`` copy on the embedded path; Alembic's batch mode
reflects and recreates the table's indexes, so the restored index survives
that copy too. The downgrade-walk lanes are what actually prove this.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "057_drop_documents_created_at_index"
down_revision: str | Sequence[str] | None = "056_documents_created_at_not_null"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_INDEX_NAME = "ix_documents_created_at"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if _is_postgres():
        # Bound the ACCESS EXCLUSIVE *acquisition* at 5s so a long-running
        # reader on ``documents`` cannot stall the deploy — nor block every
        # other ``documents`` query behind the queued lock request. LOCAL, so
        # the setting dies with this revision's transaction instead of leaking
        # onto the next one in the same upgrade run.
        op.execute("SET LOCAL lock_timeout = '5s'")

    # Plain DDL, no CONCURRENTLY: the lock is held for a catalog delete, not a
    # table scan. ``if_exists`` because an operator may already have dropped
    # this index on 054's advice, and because the deprecated create_all path
    # never creates it. See the module docstring.
    op.drop_index(_INDEX_NAME, table_name="documents", if_exists=True)


def downgrade() -> None:
    """Restore the index.

    Load-bearing, not decorative: migration 009's ``downgrade()`` drops
    ``ix_documents_created_at`` with no ``if_exists``, so a walk past 009
    aborts on both dialects if this does not put it back.

    Unlike the upgrade, this direction is expensive — a plain
    ``CREATE INDEX`` holds ``SHARE`` for the whole build and blocks writes
    to ``documents``. ``SHARE`` conflicts with the ``ROW EXCLUSIVE`` every
    concurrent write holds, so under the 5s ``lock_timeout`` this will
    typically fail to acquire rather than block on a busy table. The
    ``CREATE INDEX CONCURRENTLY`` pre-build in the module docstring is
    what makes it likely to succeed there: ``if_not_exists=True`` then
    reduces this statement to a no-op. Note that a no-op create is not
    lock-free — Postgres takes ``SHARE`` before evaluating
    ``IF NOT EXISTS`` — so the pre-build removes the build-length hold,
    not the acquisition, and the acquisition can still time out.
    """
    if _is_postgres():
        op.execute("SET LOCAL lock_timeout = '5s'")

    op.create_index(_INDEX_NAME, "documents", ["created_at"], if_not_exists=True)
