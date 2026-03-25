"""Drop slug, name, description columns from memory_namespaces (DYT-315).

Revision ID: 011_drop_namespace_slug
Revises: 010_flatten_namespace_hierarchy
Create Date: 2026-03-09

The ORM model no longer has slug, name, or description fields — namespaces
are identified by UUID only.  These physical columns must be removed so
INSERTs don't fail on NOT NULL constraints for removed fields.

Steps:
1. Drop uq_namespace_slug_version unique constraint (from migration 010)
2. Drop idx_namespace_slug_active partial index (from migration 010)
3. Drop slug column
4. Drop name and description columns (no longer in domain model)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision: str = "011_drop_namespace_slug"
down_revision: str | Sequence[str] | None = "010_flatten_namespace_hierarchy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # =========================================================================
    # Step 1: Drop slug unique constraint (created in migration 010)
    # =========================================================================
    op.drop_constraint("uq_namespace_slug_version", "memory_namespaces", type_="unique")

    # =========================================================================
    # Step 2: Drop slug partial index (created in migration 010)
    # =========================================================================
    op.drop_index("idx_namespace_slug_active", table_name="memory_namespaces")

    # =========================================================================
    # Step 3: Drop the slug column
    # =========================================================================
    # Drop the auto-created index on slug first, then the column.
    op.drop_index("ix_memory_namespaces_slug", table_name="memory_namespaces")
    op.drop_column("memory_namespaces", "slug")

    # =========================================================================
    # Step 4: Drop name and description columns
    # =========================================================================
    op.drop_column("memory_namespaces", "name")
    op.drop_column("memory_namespaces", "description")


def downgrade() -> None:
    # Reverse step 4: Re-add name and description columns
    op.add_column(
        "memory_namespaces",
        sa.Column("name", sa.String(255), nullable=True),
    )
    op.add_column(
        "memory_namespaces",
        sa.Column("description", sa.Text(), server_default="", nullable=True),
    )
    # Backfill name from id (best effort — cannot reconstruct original values)
    op.execute(text("UPDATE memory_namespaces SET name = id::text WHERE name IS NULL"))

    # Reverse step 3: Re-add slug column as NULLABLE (cannot reconstruct values)
    op.add_column(
        "memory_namespaces",
        sa.Column("slug", sa.String(255), nullable=True),
    )
    # Backfill slug from name (best effort — lowercase with hyphens)
    op.execute(text("UPDATE memory_namespaces SET slug = lower(replace(name, ' ', '-'))"))
    # Recreate the auto-generated index on slug
    op.create_index("ix_memory_namespaces_slug", "memory_namespaces", ["slug"])

    # Reverse step 2: Re-add slug partial index
    op.create_index(
        "idx_namespace_slug_active",
        "memory_namespaces",
        ["slug"],
        postgresql_where=sa.text("is_active = true"),
    )

    # Reverse step 1: Re-add slug unique constraint
    # Note: slug is nullable after downgrade, so this constraint allows NULLs
    op.create_unique_constraint("uq_namespace_slug_version", "memory_namespaces", ["slug", "version"])
