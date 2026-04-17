"""Widen extraction_config_hash from VARCHAR(64) to VARCHAR(255).

Revision ID: 016_widen_extraction_config_hash
Revises: 015_add_extraction_config_hash
Create Date: 2026-03-22

DYT-761: The original DYT-697 widening was reverted by the DYT-752 merge.
VARCHAR(255) accommodates compound keys and longer hash algorithms.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "016_widen_extraction_config_hash"
down_revision: str = "015_add_extraction_config_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.alter_column(
            "documents",
            "extraction_config_hash",
            existing_type=sa.String(64),
            type_=sa.String(255),
            existing_nullable=True,
        )
    else:
        # SQLite ignores VARCHAR length anyway (stored as TEXT), but batch mode
        # keeps the migration consistent with Alembic's expectations.
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "extraction_config_hash",
                existing_type=sa.String(64),
                type_=sa.String(255),
                existing_nullable=True,
            )


def downgrade() -> None:
    # SQLite lacks LEFT(); use SUBSTR. Run the truncate on either dialect.
    if op.get_bind().dialect.name == "postgresql":
        truncate_sql = (
            "UPDATE documents SET extraction_config_hash = LEFT(extraction_config_hash, 64) "
            "WHERE LENGTH(extraction_config_hash) > 64"
        )
    else:
        truncate_sql = (
            "UPDATE documents SET extraction_config_hash = SUBSTR(extraction_config_hash, 1, 64) "
            "WHERE LENGTH(extraction_config_hash) > 64"
        )
    op.execute(sa.text(truncate_sql))

    if op.get_bind().dialect.name == "postgresql":
        op.alter_column(
            "documents",
            "extraction_config_hash",
            existing_type=sa.String(255),
            type_=sa.String(64),
            existing_nullable=True,
        )
    else:
        with op.batch_alter_table("documents") as batch:
            batch.alter_column(
                "extraction_config_hash",
                existing_type=sa.String(255),
                type_=sa.String(64),
                existing_nullable=True,
            )
