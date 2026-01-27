"""Add namespace versioning columns.

Revision ID: 001_namespace_versioning
Revises:
Create Date: 2026-01-27

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_namespace_versioning"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add versioning columns to memory_namespaces table."""
    # Add version column with default 1
    op.add_column(
        "memory_namespaces",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )

    # Add is_active column with default True
    op.add_column(
        "memory_namespaces",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    )

    # Add previous_version_id column (nullable, self-referencing FK)
    op.add_column(
        "memory_namespaces",
        sa.Column("previous_version_id", sa.UUID(), nullable=True),
    )

    # Add foreign key constraint for previous_version_id
    op.create_foreign_key(
        "fk_namespace_previous_version",
        "memory_namespaces",
        "memory_namespaces",
        ["previous_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create partial index for efficient active namespace queries
    op.create_index(
        "idx_namespace_active",
        "memory_namespaces",
        ["workspace_id", "slug"],
        postgresql_where=sa.text("is_active = true"),
    )

    # Drop old unique constraint if exists
    op.drop_constraint("uq_namespace_workspace_slug", "memory_namespaces", type_="unique")

    # Add new unique constraint with version
    op.create_unique_constraint(
        "uq_namespace_workspace_slug_version",
        "memory_namespaces",
        ["workspace_id", "slug", "version"],
    )


def downgrade() -> None:
    """Remove versioning columns from memory_namespaces table."""
    # Drop new unique constraint
    op.drop_constraint("uq_namespace_workspace_slug_version", "memory_namespaces", type_="unique")

    # Restore old unique constraint
    op.create_unique_constraint(
        "uq_namespace_workspace_slug",
        "memory_namespaces",
        ["workspace_id", "slug"],
    )

    # Drop partial index
    op.drop_index("idx_namespace_active", table_name="memory_namespaces")

    # Drop foreign key constraint
    op.drop_constraint("fk_namespace_previous_version", "memory_namespaces", type_="foreignkey")

    # Drop columns
    op.drop_column("memory_namespaces", "previous_version_id")
    op.drop_column("memory_namespaces", "is_active")
    op.drop_column("memory_namespaces", "version")
