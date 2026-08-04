"""Widen the documents namespace/created_at index to include ``id``.

Revision ID: 054_documents_namespace_created_at_id
Revises: 053_khora_chunks_bookkeeping_to_chunker_info

``list_documents`` now pins a total order — ``ORDER BY created_at DESC, id
DESC`` — across the relational backends so that offset pagination cannot drop
or repeat a row when two documents share a ``created_at``. The pre-existing
2-column index ``ix_documents_namespace_created_at (namespace_id, created_at)``
only supplies a *prefix* of that order, so the planner has to add a sort step
(an incremental sort on PG 13+). First-page reads barely notice; the cost lands
on full-drain offset pagination, which real callers perform (GC / session
expiry, ``forget_session``, and the agent-framework adapters that walk every
document in a namespace). The regression does not require any ties to exist —
the planner cannot know the tail key is unique, so it sorts regardless.

Widening the index to ``(namespace_id, created_at, id)`` restores an
index-order scan. All three keys are declared **ASC**: with ``namespace_id``
equality-constrained, the residual index order is ``(created_at ASC, id ASC)``
and a backward scan yields exactly ``(created_at DESC, id DESC)``.
DESC-declared keys only buy anything for *mixed* sort directions, which the
uniform-DESC ordering rules out.

The 2-column index is dropped: two indexes sharing a prefix are both
maintained on every insert, i.e. pure write amplification on the ingest path.
The 3-column index is a strict superset for lookup purposes — migration 019
added the 2-column index for ``get_last_activity_at()``'s
``MAX(created_at) WHERE namespace_id = ?``, and the widened index serves that
query on its ``(namespace_id, created_at)`` prefix. The new index is created
**before** the old one is dropped so that query never runs unindexed.

Both dialects are handled. Postgres builds concurrently (see the caveats
below); every other dialect takes the plain-DDL branch, which is what migration
019 already did when it created the 2-column index unconditionally. That keeps
the invariant simple: wherever 019 ran, 054 runs, so the ORM declaration and the
physical schema agree on every backend. The embedded ``sqlite_lance`` stack runs
this chain too, and its own full-drain pagination callers are the ones that
motivated the change, so skipping SQLite would have left the regression in place
on the stack it was measured on.

Operational caveat — this migration can be slow, and it holds the migration
advisory lock while it runs. ``run_migrations()`` takes a session-scoped
``pg_advisory_lock`` *before* Alembic's transaction demarcation and holds it for
the whole run, including the autocommit block below. A concurrent caller waits
only 60s before raising ``TimeoutError``, which surfaces as
``MigrationResult(success=False)`` and then ``RuntimeError: Database migration
failed`` at startup. A btree ``CREATE INDEX CONCURRENTLY`` over a large
``documents`` table can easily exceed that, so on a large deployment every other
service booting with ``run_migrations=True`` would fail to start for the
duration of the build. Run this migration out-of-band on such deployments rather
than during a rolling deploy. (Migration 029's BRIN build does not carry this
risk — BRIN indexes are KB-sized and build in seconds.)

Operational caveat — ``IF NOT EXISTS`` does NOT self-heal a failed concurrent
build. If ``CREATE INDEX CONCURRENTLY`` fails (deadlock, cancellation, unique
violation, connection loss), Postgres leaves behind an **INVALID** index: it is
ignored by the planner for reads but still maintained on every write. Re-running
this migration will match that leftover *by name*, skip the build, and report
success while leaving the database with an index that costs writes and serves no
reads. An operator recovering from a failed run must drop the invalid index
manually (check ``pg_index.indisvalid``, then ``DROP INDEX CONCURRENTLY``)
before re-running.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "054_documents_namespace_created_at_id"
down_revision: str | Sequence[str] | None = "053_khora_chunks_bookkeeping_to_chunker_info"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    # Create first, drop second on both branches: dropping first would leave
    # ``get_last_activity_at()``'s MAX(created_at) with no index for the
    # duration of the (potentially long) build.
    if _is_postgres():
        # ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction, so we
        # open an autocommit block.
        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_namespace_created_at_id "
                "ON documents (namespace_id, created_at, id)"
            )
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_documents_namespace_created_at")
    else:
        # SQLite (and any other dialect running this chain) has no concurrent
        # build and does not need one — plain DDL, exactly as 019 issued it.
        op.create_index(
            "ix_documents_namespace_created_at_id",
            "documents",
            ["namespace_id", "created_at", "id"],
        )
        op.drop_index("ix_documents_namespace_created_at", "documents")


def downgrade() -> None:
    # Exact mirror on both branches. The 2-column index MUST come back:
    # migration 019's downgrade issues an unqualified drop of it, so leaving it
    # absent would break any downgrade that walks past 019 — on either dialect.
    if _is_postgres():
        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_namespace_created_at "
                "ON documents (namespace_id, created_at)"
            )
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_documents_namespace_created_at_id")
    else:
        op.create_index(
            "ix_documents_namespace_created_at",
            "documents",
            ["namespace_id", "created_at"],
        )
        op.drop_index("ix_documents_namespace_created_at_id", "documents")
