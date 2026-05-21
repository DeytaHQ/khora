"""Add composite (namespace_id, entity_type, created_at DESC) index on entities.

Revision ID: 028_typed_entity_recency_index
Revises: 027_migrate_uppercase_document_status
Create Date: 2026-05-13

Issue #569 — typed-entity recency fast path. Queries like "latest action
items from recent meetings" pivot directly on
``(entity_type, created_at DESC)`` filtered by ``namespace_id`` in a
single Cypher query. The PostgreSQL side needs a matching composite
index so any equivalent SQL lookup of typed entities ordered by
recency stays index-only.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "028_typed_entity_recency_index"
down_revision: str | Sequence[str] | None = "027_migrate_uppercase_document_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_entities_ns_type_created",
        "entities",
        ["namespace_id", "entity_type", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_entities_ns_type_created", table_name="entities")
