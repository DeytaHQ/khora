"""Add namespace versioning columns.

Revision ID: 001_namespace_versioning
Revises: 000_initial_schema
Create Date: 2026-01-27

This migration adds versioning columns to memory_namespaces for existing databases.
For fresh installs, these columns are already included in 000_initial_schema.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_namespace_versioning"
down_revision: str | Sequence[str] | None = "000_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table_name AND column_name = :column_name
            )
            """),
        {"table_name": table_name, "column_name": column_name},
    )
    return result.scalar()


def constraint_exists(constraint_name: str) -> bool:
    """Check if a constraint exists."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = :constraint_name
            )
            """),
        {"constraint_name": constraint_name},
    )
    return result.scalar()


def index_exists(index_name: str) -> bool:
    """Check if an index exists."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = :index_name
            )
            """),
        {"index_name": index_name},
    )
    return result.scalar()


def upgrade() -> None:
    """Add versioning columns to memory_namespaces table (idempotent)."""
    # Add version column if it doesn't exist
    if not column_exists("memory_namespaces", "version"):
        op.add_column(
            "memory_namespaces",
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        )

    # Add is_active column if it doesn't exist
    if not column_exists("memory_namespaces", "is_active"):
        op.add_column(
            "memory_namespaces",
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        )

    # Add previous_version_id column if it doesn't exist
    if not column_exists("memory_namespaces", "previous_version_id"):
        op.add_column(
            "memory_namespaces",
            sa.Column("previous_version_id", sa.UUID(), nullable=True),
        )

        # Add foreign key constraint
        op.create_foreign_key(
            "fk_namespace_previous_version",
            "memory_namespaces",
            "memory_namespaces",
            ["previous_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # Create partial index if it doesn't exist
    if not index_exists("idx_namespace_active"):
        op.create_index(
            "idx_namespace_active",
            "memory_namespaces",
            ["workspace_id", "slug"],
            postgresql_where=sa.text("is_active = true"),
        )

    # Handle unique constraint migration
    if constraint_exists("uq_namespace_workspace_slug"):
        op.drop_constraint("uq_namespace_workspace_slug", "memory_namespaces", type_="unique")

    if not constraint_exists("uq_namespace_workspace_slug_version"):
        op.create_unique_constraint(
            "uq_namespace_workspace_slug_version",
            "memory_namespaces",
            ["workspace_id", "slug", "version"],
        )


def downgrade() -> None:
    """Remove versioning columns from memory_namespaces table."""
    # Drop new unique constraint if exists
    if constraint_exists("uq_namespace_workspace_slug_version"):
        op.drop_constraint("uq_namespace_workspace_slug_version", "memory_namespaces", type_="unique")

    # Restore old unique constraint
    if not constraint_exists("uq_namespace_workspace_slug"):
        op.create_unique_constraint(
            "uq_namespace_workspace_slug",
            "memory_namespaces",
            ["workspace_id", "slug"],
        )

    # Drop partial index if exists
    if index_exists("idx_namespace_active"):
        op.drop_index("idx_namespace_active", table_name="memory_namespaces")

    # Drop foreign key constraint if exists
    if constraint_exists("fk_namespace_previous_version"):
        op.drop_constraint("fk_namespace_previous_version", "memory_namespaces", type_="foreignkey")

    # Drop columns if they exist
    if column_exists("memory_namespaces", "previous_version_id"):
        op.drop_column("memory_namespaces", "previous_version_id")
    if column_exists("memory_namespaces", "is_active"):
        op.drop_column("memory_namespaces", "is_active")
    if column_exists("memory_namespaces", "version"):
        op.drop_column("memory_namespaces", "version")
