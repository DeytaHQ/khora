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


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    if _is_postgres():
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
    # SQLite: use PRAGMA table_info
    result = bind.execute(sa.text(f"PRAGMA table_info({table_name})"))
    return any(row[1] == column_name for row in result)


def constraint_exists(constraint_name: str) -> bool:
    """Check if a constraint exists."""
    bind = op.get_bind()
    if _is_postgres():
        result = bind.execute(
            sa.text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = :constraint_name
                )
                """),
            {"constraint_name": constraint_name},
        )
        return result.scalar()
    # SQLite: match CONSTRAINT keyword + name + whitespace delimiter to avoid
    # false positives from substrings (e.g. 'foo' matching inside 'foo_version').
    result = bind.execute(
        sa.text(
            "SELECT sql FROM sqlite_master WHERE type='table' AND ("
            " sql LIKE :p1 OR sql LIKE :p2 OR sql LIKE :p3 OR sql LIKE :p4)"
        ),
        {
            "p1": f"%CONSTRAINT {constraint_name} %",
            "p2": f"%CONSTRAINT {constraint_name}\t%",
            "p3": f"%CONSTRAINT {constraint_name}\n%",
            "p4": f"%CONSTRAINT {constraint_name}(%",
        },
    )
    return result.first() is not None


def index_exists(index_name: str) -> bool:
    """Check if an index exists."""
    bind = op.get_bind()
    if _is_postgres():
        result = bind.execute(
            sa.text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes WHERE indexname = :index_name
                )
                """),
            {"index_name": index_name},
        )
        return result.scalar()
    result = bind.execute(
        sa.text("SELECT 1 FROM sqlite_master WHERE type='index' AND name = :name"),
        {"name": index_name},
    )
    return result.first() is not None


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

        # Add foreign key constraint (Postgres only — SQLite cannot ADD FK
        # without rewriting the table, and on fresh SQLite the column already
        # has the FK from migration 000).
        if _is_postgres():
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
        if _is_postgres():
            op.create_index(
                "idx_namespace_active",
                "memory_namespaces",
                ["workspace_id", "slug"],
                postgresql_where=sa.text("is_active = true"),
            )
        else:
            op.create_index(
                "idx_namespace_active",
                "memory_namespaces",
                ["workspace_id", "slug"],
                sqlite_where=sa.text("is_active = 1"),
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
    is_postgres = _is_postgres()

    # Drop new unique constraint if exists / restore old — SQLite needs batch.
    if is_postgres:
        if constraint_exists("uq_namespace_workspace_slug_version"):
            op.drop_constraint("uq_namespace_workspace_slug_version", "memory_namespaces", type_="unique")
        if not constraint_exists("uq_namespace_workspace_slug"):
            op.create_unique_constraint(
                "uq_namespace_workspace_slug",
                "memory_namespaces",
                ["workspace_id", "slug"],
            )
    else:
        with op.batch_alter_table("memory_namespaces") as batch:
            if constraint_exists("uq_namespace_workspace_slug_version"):
                batch.drop_constraint("uq_namespace_workspace_slug_version", type_="unique")
            if not constraint_exists("uq_namespace_workspace_slug"):
                batch.create_unique_constraint(
                    "uq_namespace_workspace_slug",
                    ["workspace_id", "slug"],
                )

    # Drop partial index if exists
    if index_exists("idx_namespace_active"):
        op.drop_index("idx_namespace_active", table_name="memory_namespaces")

    if is_postgres:
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
    else:
        with op.batch_alter_table("memory_namespaces") as batch:
            if column_exists("memory_namespaces", "previous_version_id"):
                batch.drop_column("previous_version_id")
            if column_exists("memory_namespaces", "is_active"):
                batch.drop_column("is_active")
            if column_exists("memory_namespaces", "version"):
                batch.drop_column("version")
