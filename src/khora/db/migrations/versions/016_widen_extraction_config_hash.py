"""Widen extraction_config_hash from VARCHAR(64) to VARCHAR(255).

Revision ID: 016_widen_extraction_config_hash
Revises: 015_add_extraction_config_hash
Create Date: 2026-03-22

DYT-761: The original DYT-697 widening was reverted by the DYT-752 merge.
VARCHAR(255) accommodates SHA-512, compound keys, and other hash algorithms.
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
    op.alter_column(
        "documents",
        "extraction_config_hash",
        existing_type=sa.String(64),
        type_=sa.String(255),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Truncate any hashes longer than 64 chars before shrinking the column,
    # so the ALTER doesn't fail or silently truncate on strict databases.
    op.execute(
        sa.text(
            "UPDATE documents SET extraction_config_hash = LEFT(extraction_config_hash, 64) "
            "WHERE LENGTH(extraction_config_hash) > 64"
        )
    )
    op.alter_column(
        "documents",
        "extraction_config_hash",
        existing_type=sa.String(255),
        type_=sa.String(64),
        existing_nullable=True,
    )
