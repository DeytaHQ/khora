"""Add khora_try_timestamptz() — a safe text→timestamptz cast for $date filters.

Revision ID: 045_khora_try_timestamptz
Revises: 044_khora_chunks_backfill_denormalized
Create Date: 2026-06-05

Part of the deterministic recall-filter contract (§4/§7). The Postgres recall-filter compiler needs to
compare metadata strings against a timestamp for the ``$date`` operator. A bare
``txt::timestamptz`` cast aborts the whole statement the moment it meets a
non-parseable string, which is fatal when a JSONB metadata value is malformed or
of an unexpected shape. This function wraps the cast in an exception handler so a
bad value yields ``NULL`` (and therefore a non-matching row) instead of erroring
out the query.

It is ``IMMUTABLE`` (a pure cast of its input — the same text always maps to the
same timestamptz, given a fixed session ``TimeZone``, which the connection pins
to UTC) and ``PARALLEL SAFE`` (no side effects, no shared state). ``IMMUTABLE``
also leaves the door open to backing a functional index on the cast later.

Postgres-only: the function is plpgsql. The sqlite_lance test fixtures run the
full Alembic chain against SQLite, so we skip silently on non-Postgres dialects
rather than emit SQL SQLite cannot parse — following migrations 029 and 044.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "045_khora_try_timestamptz"
down_revision: str | Sequence[str] | None = "044_khora_chunks_backfill_denormalized"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_CREATE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION khora_try_timestamptz(txt text)
RETURNS timestamptz
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    RETURN txt::timestamptz;
EXCEPTION WHEN others THEN
    RETURN NULL;
END;
$$;
"""

_DROP_FUNCTION_SQL = "DROP FUNCTION IF EXISTS khora_try_timestamptz(text);"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    op.execute(_CREATE_FUNCTION_SQL)


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute(_DROP_FUNCTION_SQL)
