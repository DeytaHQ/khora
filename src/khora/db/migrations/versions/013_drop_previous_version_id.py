"""Drop previous_version_id column from memory_namespaces.

Revision ID: 013_drop_previous_version_id
Revises: 012_add_stable_namespace_id
Create Date: 2026-03-11

The previous_version_id FK is no longer needed — namespace version chains
are now tracked via the stable namespace_id column (added in migration 012).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "013_drop_previous_version_id"
down_revision: str | Sequence[str] | None = "012_add_stable_namespace_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    if is_postgres:
        # Find the actual FK constraint name — it varies depending on migration path:
        #   - "memory_namespaces_previous_version_id_fkey" (auto-generated in 000)
        #   - "fk_namespace_previous_version" (explicit name in 001)
        result = conn.execute(
            text(
                "SELECT constraint_name FROM information_schema.table_constraints "
                "WHERE table_name = 'memory_namespaces' "
                "AND constraint_type = 'FOREIGN KEY' "
                "AND constraint_name IN ("
                "  'memory_namespaces_previous_version_id_fkey',"
                "  'fk_namespace_previous_version'"
                ")"
            )
        )
        for row in result:
            op.drop_constraint(row[0], "memory_namespaces", type_="foreignkey")
        op.drop_column("memory_namespaces", "previous_version_id")
    else:
        # SQLite: batch mode drops the column and its FK in one table rewrite.
        with op.batch_alter_table("memory_namespaces") as batch:
            batch.drop_column("previous_version_id")


def downgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"

    # Re-add nullable UUID column
    op.add_column(
        "memory_namespaces",
        sa.Column("previous_version_id", UUID(as_uuid=True) if is_postgres else sa.String(36), nullable=True),
    )

    if is_postgres:
        # Recreate FK constraint
        op.create_foreign_key(
            "fk_namespace_previous_version",
            "memory_namespaces",
            "memory_namespaces",
            ["previous_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

        # Backfill using namespace_id + version (Postgres UPDATE ... FROM syntax)
        op.execute(
            text(
                "UPDATE memory_namespaces mn SET previous_version_id = prev.id "
                "FROM memory_namespaces prev "
                "WHERE prev.namespace_id = mn.namespace_id AND prev.version = mn.version - 1"
            )
        )
