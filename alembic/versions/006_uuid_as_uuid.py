"""UUID as_uuid migration - Python-side mapping change.

Revision ID: 006_uuid_as_uuid
Revises: 005_index_improvements
Create Date: 2026-02-15

Changes as_uuid=False to as_uuid=True for all UUID columns.
This is a no-op on the database side as PostgreSQL columns are already UUID type.
The change only affects Python-side mapping (str → uuid.UUID objects).
"""

from collections.abc import Sequence

from alembic import op  # noqa: F401

revision: str = "006_uuid_as_uuid"
down_revision: str | Sequence[str] | None = "005_index_improvements"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No-op: as_uuid only affects Python-side mapping.
    # PostgreSQL columns are already UUID type.
    pass


def downgrade() -> None:
    # No-op: as_uuid only affects Python-side mapping.
    pass
