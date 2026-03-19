"""Drop previous_version_id column from memory_namespaces.

Revision ID: 013_drop_previous_version_id
Revises: 012_add_stable_namespace_id
Create Date: 2026-03-11

The previous_version_id FK is no longer needed — namespace version chains
are now tracked via the stable namespace_id column (added in migration 012).
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID

revision: str = "013_drop_previous_version_id"
down_revision: str | Sequence[str] | None = "012_add_stable_namespace_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Find the actual FK constraint name — it varies depending on migration path:
    #   - "memory_namespaces_previous_version_id_fkey" (auto-generated in 000)
    #   - "fk_namespace_previous_version" (explicit name in 001)
    conn = op.get_bind()
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


def downgrade() -> None:
    # Re-add nullable UUID column
    op.add_column(
        "memory_namespaces",
        sa.Column("previous_version_id", UUID(as_uuid=True), nullable=True),
    )

    # Recreate FK constraint
    op.create_foreign_key(
        "fk_namespace_previous_version",
        "memory_namespaces",
        "memory_namespaces",
        ["previous_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Backfill using namespace_id + version
    op.execute(
        text(
            "UPDATE memory_namespaces mn SET previous_version_id = prev.id "
            "FROM memory_namespaces prev "
            "WHERE prev.namespace_id = mn.namespace_id AND prev.version = mn.version - 1"
        )
    )
