"""Drop slug column from memory_namespaces (DYT-315).

Revision ID: 011_drop_namespace_slug
Revises: 010_flatten_namespace_hierarchy
Create Date: 2026-03-09

The ORM model no longer has a slug field — namespaces are identified by
UUID and looked up by name.  The physical slug column is NOT NULL, so
INSERTs will fail unless the column is removed.

Steps:
1. Drop uq_namespace_slug_version unique constraint (from migration 010)
2. Drop idx_namespace_slug_active partial index (from migration 010)
3. Drop slug column
4. Add uq_namespace_name_version unique constraint on (name, version)
5. Add idx_namespace_name_active partial index on name WHERE is_active
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
    # The name column already exists (created in migration 000).
    # Drop the auto-created index on slug first, then the column.
    op.drop_index("ix_memory_namespaces_slug", table_name="memory_namespaces")
    op.drop_column("memory_namespaces", "slug")

    # =========================================================================
    # Step 4: Add unique constraint on (name, version)
    # =========================================================================
    op.create_unique_constraint("uq_namespace_name_version", "memory_namespaces", ["name", "version"])

    # =========================================================================
    # Step 5: Add partial index on name for active namespaces
    # =========================================================================
    op.create_index(
        "idx_namespace_name_active",
        "memory_namespaces",
        ["name"],
        postgresql_where=sa.text("is_active = true"),
    )


def downgrade() -> None:
    # Reverse step 5: Drop name partial index
    op.drop_index("idx_namespace_name_active", table_name="memory_namespaces")

    # Reverse step 4: Drop name unique constraint
    op.drop_constraint("uq_namespace_name_version", "memory_namespaces", type_="unique")

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
