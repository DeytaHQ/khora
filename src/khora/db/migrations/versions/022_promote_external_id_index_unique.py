"""Promote external_id partial index to UNIQUE.

Revision ID: 022_promote_external_id_index_unique
Revises: 021_add_document_external_id
Create Date: 2026-04-21

Promote the partial composite index (namespace_id, external_id)
WHERE external_id IS NOT NULL to a UNIQUE constraint, enabling idempotent
upsert-by-external_id.

Uses CREATE/DROP INDEX CONCURRENTLY on PostgreSQL (cannot run inside a
transaction), so each index operation uses an autocommit block.  Invalid
indexes left behind by interrupted builds are detected via
pg_index.indisvalid, dropped, and recreated.  SQLite uses the standard
transactional approach (no CONCURRENTLY support).
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "022_promote_external_id_index_unique"
down_revision: str | Sequence[str] | None = "021_add_document_external_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_invalid_index(index_name: str) -> None:
    """Drop an index if it exists and is marked invalid (indisvalid = false).

    An invalid index is left behind when CREATE INDEX CONCURRENTLY is
    interrupted.  We must remove it before re-creating, because
    IF NOT EXISTS will skip creation even for invalid indexes.
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
    """Replace non-unique partial index with a UNIQUE partial index."""
    if op.get_bind().dialect.name == "postgresql":
        # Drop old non-unique index concurrently
        with op.get_context().autocommit_block():
            op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_documents_namespace_external_id"))

        # Clean up any invalid index from interrupted previous build
        _drop_invalid_index("ix_documents_namespace_external_id_unique")

        # Create new unique partial index concurrently
        with op.get_context().autocommit_block():
            op.execute(
                text(
                    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS"
                    " ix_documents_namespace_external_id_unique"
                    " ON documents (namespace_id, external_id)"
                    " WHERE external_id IS NOT NULL"
                )
            )
    else:
        # SQLite: no CONCURRENTLY, use transactional approach
        op.drop_index("ix_documents_namespace_external_id", table_name="documents")
        op.create_index(
            "ix_documents_namespace_external_id_unique",
            "documents",
            ["namespace_id", "external_id"],
            unique=True,
            sqlite_where=text("external_id IS NOT NULL"),
        )


def downgrade() -> None:
    """Restore original non-unique partial index."""
    if op.get_bind().dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_documents_namespace_external_id_unique"))

        _drop_invalid_index("ix_documents_namespace_external_id")

        with op.get_context().autocommit_block():
            op.execute(
                text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS"
                    " ix_documents_namespace_external_id"
                    " ON documents (namespace_id, external_id)"
                    " WHERE external_id IS NOT NULL"
                )
            )
    else:
        op.drop_index("ix_documents_namespace_external_id_unique", table_name="documents")
        op.create_index(
            "ix_documents_namespace_external_id",
            "documents",
            ["namespace_id", "external_id"],
            sqlite_where=text("external_id IS NOT NULL"),
        )
