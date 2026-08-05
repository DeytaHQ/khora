"""Widen the documents namespace/created_at index to include ``id``.

Revision ID: 054_documents_namespace_created_at_id
Revises: 053_khora_chunks_bookkeeping_to_chunker_info
Create Date: 2026-08-04

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

Invalid-index recovery — a failed ``CREATE INDEX CONCURRENTLY`` (deadlock,
cancellation, connection loss) leaves an **INVALID** index behind: ignored by
the planner for reads, but still maintained on every write. ``IF NOT EXISTS``
matches such a leftover *by name* and skips the build, so a naive re-run would
skip the create and then still run the drop — leaving the table with an invalid
3-column index and **no valid index at all**, precisely on the deployment large
enough for the first build to have failed. Both directions therefore probe
``pg_index.indisvalid`` and drop an invalid leftover before building. This
follows migration 022, which solved the same problem on this same table; its
``_drop_invalid_index`` helper is reproduced here.

Planner precondition — **superseded by ``057_drop_documents_created_at_index``,
which drops the single-column index this paragraph was about.** As written when
054 landed: the sort-free plan assumes ``namespace_id`` is selective against the
table, and where a single namespace held most of ``documents``, Postgres could
instead prefer the single-column ``ix_documents_created_at`` (migration 009) and
reintroduce the Incremental Sort, because walking that index directly can beat a
lookup that matches most of the table anyway. Operators sizing 054 on a heavily
single-tenant ``documents`` table were told to measure rather than assume the
win. That competitor no longer exists at head; the advice is kept on the record
because operators who acted on it are precisely why 057's drop passes
``if_exists=True``.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "054_documents_namespace_created_at_id"
down_revision: str | Sequence[str] | None = "053_khora_chunks_bookkeeping_to_chunker_info"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _drop_invalid_index(index_name: str) -> None:
    """Drop an index if it exists and is marked invalid (indisvalid = false).

    An interrupted ``CREATE INDEX CONCURRENTLY`` leaves an INVALID index behind.
    ``IF NOT EXISTS`` matches it by name and skips the rebuild, so without this
    probe a re-run would skip the create yet still run the drop, leaving the
    table with no valid index at all. Mirrors migration 022's helper.
    """
    conn = op.get_bind()
    is_invalid = conn.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_class c"
            "  JOIN pg_index i ON i.indexrelid = c.oid"
            "  WHERE c.relname = :name AND NOT i.indisvalid"
            ")"
        ),
        {"name": index_name},
    ).scalar()
    if is_invalid:
        with op.get_context().autocommit_block():
            op.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}"))


def upgrade() -> None:
    # Create first, drop second on both branches: dropping first would leave
    # ``get_last_activity_at()``'s MAX(created_at) with no index for the
    # duration of the (potentially long) build.
    if _is_postgres():
        # Clear an INVALID leftover from an interrupted earlier build first.
        # Without this, IF NOT EXISTS would match it by name, skip the build,
        # and the drop below would then remove the only valid index.
        _drop_invalid_index("ix_documents_namespace_created_at_id")
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
    # migration 019's downgrade() drops ix_documents_namespace_created_at
    # unconditionally — a plain ``op.drop_index`` with no ``if_exists=True`` —
    # so 054's downgrade must restore it or any walk past 019 aborts, on either
    # dialect. (The table name IS passed there; it is the missing if_exists that
    # makes the restore load-bearing, so do not read a qualified call as proof
    # this is unnecessary.)
    if _is_postgres():
        # Same invalid-leftover trap as upgrade(), mirrored onto the 2-column
        # index this direction rebuilds.
        _drop_invalid_index("ix_documents_namespace_created_at")
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
