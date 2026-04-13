"""Flatten multi-tenancy to namespace-only (DYT-220).

Revision ID: 010_flatten_namespace_hierarchy
Revises: 009_temporal_search_indexes
Create Date: 2026-03-05

Removes the Organization -> Workspace -> Namespace hierarchy.
Namespace becomes the sole data isolation boundary with globally unique slugs.

Steps:
1. Pre-check for duplicate slugs across workspaces
2. Delete orphaned permission rows referencing org/workspace
3. Drop inherited_from columns from permissions
4. Drop old indexes on memory_namespaces that reference workspace_id
5. Remove workspace_id FK and column from memory_namespaces
6. Add tenancy_mode column to memory_namespaces
7. Add new unique constraint UNIQUE(slug, version)
8. Add new partial index idx_namespace_slug_active
9. Drop workspaces table
10. Drop organizations table
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

revision: str = "010_flatten_namespace_hierarchy"
down_revision: str | Sequence[str] | None = "009_temporal_search_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # =========================================================================
    # Step 1: Pre-check for duplicate slugs across workspaces
    # =========================================================================
    conn = op.get_bind()
    result = conn.execute(
        text("""
        SELECT slug, version, count(*) AS cnt
        FROM memory_namespaces
        GROUP BY slug, version
        HAVING count(*) > 1
    """)
    )
    duplicates = result.fetchall()
    if duplicates:
        lines = [f"  slug={row[0]!r}, version={row[1]}, count={row[2]}" for row in duplicates]
        raise RuntimeError(
            "Cannot flatten namespace hierarchy: duplicate (slug, version) pairs found.\n"
            "Resolve these manually before re-running the migration:\n" + "\n".join(lines)
        )

    # =========================================================================
    # Step 2: Delete orphaned permission rows
    # =========================================================================
    op.execute(text("DELETE FROM permissions WHERE resource_type IN ('organization', 'workspace')"))
    op.execute(text("DELETE FROM permissions WHERE inherited_from_type IN ('organization', 'workspace')"))

    # =========================================================================
    # Step 3: Drop inherited_from columns from permissions
    # =========================================================================
    op.drop_column("permissions", "inherited_from_type")
    op.drop_column("permissions", "inherited_from_id")

    # =========================================================================
    # Step 4: Drop old indexes on memory_namespaces that reference workspace_id
    # =========================================================================
    # Partial index: idx_namespace_active ON (workspace_id, slug) WHERE is_active
    op.drop_index("idx_namespace_active", table_name="memory_namespaces")
    # Unique constraint: uq_namespace_workspace_slug_version ON (workspace_id, slug, version)
    op.drop_constraint("uq_namespace_workspace_slug_version", "memory_namespaces", type_="unique")
    # Auto-created index on workspace_id column
    op.drop_index("ix_memory_namespaces_workspace_id", table_name="memory_namespaces")

    # =========================================================================
    # Step 5: Remove workspace_id FK and column from memory_namespaces
    # =========================================================================
    op.drop_constraint("memory_namespaces_workspace_id_fkey", "memory_namespaces", type_="foreignkey")
    op.drop_column("memory_namespaces", "workspace_id")

    # =========================================================================
    # Step 6: Add tenancy_mode column
    # =========================================================================
    op.add_column(
        "memory_namespaces",
        sa.Column(
            "tenancy_mode",
            postgresql.ENUM("shared", "isolated", name="tenancy_mode", create_type=False),
            server_default="shared",
            nullable=False,
        ),
    )

    # =========================================================================
    # Step 7: Add new unique constraint UNIQUE(slug, version)
    # =========================================================================
    op.create_unique_constraint("uq_namespace_slug_version", "memory_namespaces", ["slug", "version"])

    # =========================================================================
    # Step 8: Add new partial index
    # =========================================================================
    op.create_index(
        "idx_namespace_slug_active",
        "memory_namespaces",
        ["slug"],
        postgresql_where=sa.text("is_active = true"),
    )

    # =========================================================================
    # Step 9: Drop workspaces table
    # =========================================================================
    op.drop_table("workspaces")

    # =========================================================================
    # Step 10: Drop organizations table
    # =========================================================================
    op.drop_table("organizations")


def downgrade() -> None:
    # WARNING: This downgrade is intended for one-time emergency recovery only.
    # Repeated downgrade->upgrade cycles may produce inconsistent state because
    # the upgrade deletes org/workspace permission rows and drops the tables,
    # and the downgrade creates synthetic defaults that won't match original data.

    # Reverse step 10: Recreate organizations table
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column(
            "tenancy_mode",
            postgresql.ENUM("shared", "isolated", name="tenancy_mode", create_type=False),
            server_default="shared",
        ),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Reverse step 9: Recreate workspaces table
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),
    )

    # Reverse step 8: Drop new partial index
    op.drop_index("idx_namespace_slug_active", table_name="memory_namespaces")

    # Reverse step 7: Drop new unique constraint
    op.drop_constraint("uq_namespace_slug_version", "memory_namespaces", type_="unique")

    # Reverse step 6: Drop tenancy_mode column
    op.drop_column("memory_namespaces", "tenancy_mode")

    # Reverse step 5: Re-add workspace_id column with FK
    # Create a default org and workspace to satisfy FK constraints
    op.execute(
        text("""
        INSERT INTO organizations (id, name, slug)
        VALUES ('00000000-0000-0000-0000-000000000001', 'Default Organization', 'default')
        ON CONFLICT DO NOTHING
    """)
    )
    op.execute(
        text("""
        INSERT INTO workspaces (id, organization_id, name, slug)
        VALUES (
            '00000000-0000-0000-0000-000000000002',
            '00000000-0000-0000-0000-000000000001',
            'Default Workspace',
            'default'
        )
        ON CONFLICT DO NOTHING
    """)
    )

    op.add_column(
        "memory_namespaces",
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
    )
    # Point all existing namespaces to the default workspace
    op.execute(text("UPDATE memory_namespaces SET workspace_id = '00000000-0000-0000-0000-000000000002'"))
    op.alter_column("memory_namespaces", "workspace_id", nullable=False)
    op.create_foreign_key(
        "memory_namespaces_workspace_id_fkey",
        "memory_namespaces",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Reverse step 4: Restore old indexes
    op.create_index(
        "ix_memory_namespaces_workspace_id",
        "memory_namespaces",
        ["workspace_id"],
    )
    op.create_unique_constraint(
        "uq_namespace_workspace_slug_version",
        "memory_namespaces",
        ["workspace_id", "slug", "version"],
    )
    op.create_index(
        "idx_namespace_active",
        "memory_namespaces",
        ["workspace_id", "slug"],
        postgresql_where=sa.text("is_active = true"),
    )

    # Reverse step 3: Re-add inherited_from columns to permissions
    op.add_column(
        "permissions",
        sa.Column("inherited_from_type", sa.String(64), nullable=True),
    )
    op.add_column(
        "permissions",
        sa.Column("inherited_from_id", postgresql.UUID(as_uuid=False), nullable=True),
    )

    # Reverse steps 2 & 1: No action needed — deleted rows cannot be restored
