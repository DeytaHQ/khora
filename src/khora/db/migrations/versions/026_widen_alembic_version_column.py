"""Widen khora_alembic_version.version_num to VARCHAR(64).

Revision ID: 026_widen_alembic_version_column
Revises: 025_add_document_extraction_params
Create Date: 2026-05-01

DYT-3546: Alembic's default version_num column is VARCHAR(32). Khora
revision IDs (e.g. "022_promote_external_id_index_unique" = 38 chars)
exceed that, causing INSERT failures on fresh databases. env.py now
creates the table at width 64 from the start; this migration widens
existing deployments. Idempotent: skips if the column is already wide
enough.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "026_widen_alembic_version_column"
down_revision: str | Sequence[str] | None = "025_add_document_extraction_params"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VERSION_TABLE = "khora_alembic_version"
TARGET_WIDTH = 64


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Widen version_num column. Postgres-only — SQLite stores TEXT regardless."""
    if not _is_postgres():
        # SQLite VARCHAR(n) is just TEXT with no length enforcement; nothing to do.
        return

    bind = op.get_bind()
    current_width = bind.execute(
        text(
            "SELECT character_maximum_length FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = 'version_num'"
        ),
        {"table": VERSION_TABLE},
    ).scalar()

    # Idempotent: skip if already at or above target width, or if table is missing
    # (fresh DB — env.py created it at the new width directly).
    if current_width is None or current_width >= TARGET_WIDTH:
        return

    op.execute(
        text(f"ALTER TABLE {VERSION_TABLE} ALTER COLUMN version_num TYPE VARCHAR({TARGET_WIDTH})")  # noqa: S608
    )


def downgrade() -> None:
    """Narrow version_num back to VARCHAR(32). Fails if any row exceeds 32 chars."""
    if not _is_postgres():
        return
    op.execute(text(f"ALTER TABLE {VERSION_TABLE} ALTER COLUMN version_num TYPE VARCHAR(32)"))  # noqa: S608
