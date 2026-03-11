"""Add stable namespace_id column to memory_namespaces (DYT-396).

Revision ID: 012_add_stable_namespace_id
Revises: 011_drop_namespace_slug
Create Date: 2026-03-11

Adds a stable `namespace_id` UUID that persists across namespace versions.
For existing rows, backfills namespace_id = id (all existing namespaces are
single-version, so this is safe). Adds a unique constraint on
(namespace_id, version) and a partial index for fast active-version resolution.

Steps:
1. Add namespace_id column (nullable initially)
2. Backfill: namespace_id = id for all existing rows
3. ALTER COLUMN to NOT NULL
4. Add unique constraint uq_namespace_stable_id_version on (namespace_id, version)
5. Add partial index idx_namespace_stable_active on namespace_id WHERE is_active = true
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision: str = "012_add_stable_namespace_id"
down_revision: str | Sequence[str] | None = "011_drop_namespace_slug"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # =========================================================================
    # Step 1: Add namespace_id column (nullable initially for backfill)
    # =========================================================================
    op.add_column(
        "memory_namespaces",
        sa.Column("namespace_id", sa.UUID(), nullable=True),
    )

    # =========================================================================
    # Step 2: Backfill — set namespace_id = id for all existing rows
    # =========================================================================
    op.execute(text("UPDATE memory_namespaces SET namespace_id = id WHERE namespace_id IS NULL"))

    # =========================================================================
    # Step 3: ALTER COLUMN to NOT NULL
    # =========================================================================
    op.alter_column("memory_namespaces", "namespace_id", nullable=False)

    # =========================================================================
    # Step 4: Add unique constraint (namespace_id, version)
    # =========================================================================
    op.create_unique_constraint(
        "uq_namespace_stable_id_version",
        "memory_namespaces",
        ["namespace_id", "version"],
    )

    # =========================================================================
    # Step 5: Add partial index for fast active-version resolution
    # =========================================================================
    op.create_index(
        "idx_namespace_stable_active",
        "memory_namespaces",
        ["namespace_id"],
        unique=False,
        postgresql_where=sa.text("is_active = true"),
    )


def downgrade() -> None:
    # Reverse step 5: Drop partial index
    op.drop_index("idx_namespace_stable_active", table_name="memory_namespaces")

    # Reverse step 4: Drop unique constraint
    op.drop_constraint("uq_namespace_stable_id_version", "memory_namespaces", type_="unique")

    # Reverse steps 1-3: Drop the column
    op.drop_column("memory_namespaces", "namespace_id")
