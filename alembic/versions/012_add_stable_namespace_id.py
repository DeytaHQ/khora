"""Add stable namespace_id column to memory_namespaces (DYT-396).

Revision ID: 012_add_stable_namespace_id
Revises: 011_drop_namespace_slug
Create Date: 2026-03-11

Adds a stable `namespace_id` UUID that persists across namespace versions.
For existing rows, traverses the previous_version_id chain to group all
versions of the same namespace, then assigns namespace_id = id of the
highest-version row in each chain. Adds a unique constraint on
(namespace_id, version) and a partial index for fast active-version resolution.

Steps:
1. Add namespace_id column (nullable initially)
2. Backfill: traverse version chains, set namespace_id = max-version row's id
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
    # Step 2: Backfill — traverse version chains, assign namespace_id from
    # the highest-version row in each chain.
    #
    # Uses a recursive CTE to walk previous_version_id links, grouping all
    # versions of the same logical namespace under one chain_root. Then picks
    # the id of the max-version row as the stable namespace_id for the group.
    # =========================================================================
    op.execute(text("""
            WITH RECURSIVE chain AS (
                -- Roots: rows with no parent (first version of each namespace)
                SELECT id, id AS chain_root, version
                FROM memory_namespaces
                WHERE previous_version_id IS NULL
                  AND namespace_id IS NULL

                UNION ALL

                -- Walk forward: find rows whose parent is in the chain
                SELECT mn.id, c.chain_root, mn.version
                FROM memory_namespaces mn
                JOIN chain c ON mn.previous_version_id = c.id
                WHERE mn.namespace_id IS NULL
            ),
            -- For each chain, find the id of the row with the highest version
            max_version_ids AS (
                SELECT DISTINCT ON (chain_root)
                    chain_root,
                    id AS stable_id
                FROM chain
                ORDER BY chain_root, version DESC
            )
            UPDATE memory_namespaces mn
            SET namespace_id = mv.stable_id
            FROM chain c
            JOIN max_version_ids mv ON c.chain_root = mv.chain_root
            WHERE mn.id = c.id
              AND mn.namespace_id IS NULL
        """))

    # Verify backfill completeness before enforcing NOT NULL
    conn = op.get_bind()
    orphans = conn.execute(
        text("SELECT id, version FROM memory_namespaces WHERE namespace_id IS NULL LIMIT 5")
    ).fetchall()
    if orphans:
        ids = ", ".join(str(row[0]) for row in orphans)
        raise RuntimeError(
            f"Backfill incomplete: {len(orphans)}+ rows still have NULL namespace_id "
            f"(likely due to broken previous_version_id chain). Sample IDs: {ids}"
        )

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
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )


def downgrade() -> None:
    # Reverse step 5: Drop partial index
    op.drop_index("idx_namespace_stable_active", table_name="memory_namespaces")

    # Reverse step 4: Drop unique constraint
    op.drop_constraint("uq_namespace_stable_id_version", "memory_namespaces", type_="unique")

    # Reverse steps 1-3: Drop the column
    op.drop_column("memory_namespaces", "namespace_id")
