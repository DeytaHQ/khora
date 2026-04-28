"""Add extraction_params column to documents table.

Revision ID: 025_add_document_extraction_params
Revises: 024_chronicle_events_and_facts
Create Date: 2026-04-28

DYT-3305: Store extraction parameters (skill_name, entity_types,
relationship_types, expertise, chunk_strategy) on each PENDING document
so the unified pending processor can reconstruct the original extraction
intent without hardcoding defaults.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "025_add_document_extraction_params"
down_revision: str | Sequence[str] | None = "024_chronicle_events_and_facts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Add extraction_params column (JSONB on Postgres, JSON on SQLite)."""
    col_type = JSONB if _is_postgres() else sa.JSON
    op.add_column("documents", sa.Column("extraction_params", col_type, nullable=True))


def downgrade() -> None:
    """Remove extraction_params column."""
    op.drop_column("documents", "extraction_params")
