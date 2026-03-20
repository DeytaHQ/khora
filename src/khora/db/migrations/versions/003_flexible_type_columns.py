"""Convert entity_type and relationship_type from Enum to String.

Revision ID: 003_flexible_type_columns
Revises: 002_search_improvements
Create Date: 2026-02-02

The EntityType and RelationshipType enums only supported a fixed set of
standard types (PERSON, CONCEPT, RELATES_TO, etc.).  Domain-specific types
extracted by the LLM (EMPLOYEE, CHANNEL, MEMBER_OF, AUTHORED, etc.) were
silently mapped to generic fallbacks (CONCEPT / RELATES_TO), which broke
inference rules that expected the original types.

This migration converts both columns to VARCHAR(64) so any type string
can be stored and retrieved faithfully.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_flexible_type_columns"
down_revision: str | Sequence[str] | None = "002_search_improvements"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop column defaults that reference the enum types before converting,
    # otherwise PostgreSQL refuses to DROP TYPE due to dependent defaults.
    op.execute("ALTER TABLE entities ALTER COLUMN entity_type DROP DEFAULT")
    op.execute("ALTER TABLE relationships ALTER COLUMN relationship_type DROP DEFAULT")

    # Convert entity_type from enum to varchar
    op.execute("""
        ALTER TABLE entities
        ALTER COLUMN entity_type TYPE VARCHAR(64)
        USING entity_type::text
        """)

    # Convert relationship_type from enum to varchar
    op.execute("""
        ALTER TABLE relationships
        ALTER COLUMN relationship_type TYPE VARCHAR(64)
        USING relationship_type::text
        """)

    # Drop the old enum types (they are no longer referenced)
    op.execute("DROP TYPE IF EXISTS entity_type")
    op.execute("DROP TYPE IF EXISTS relationship_type")


def downgrade() -> None:
    # Recreate the enum types
    op.execute("""
        CREATE TYPE entity_type AS ENUM (
            'PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT',
            'EVENT', 'PRODUCT', 'TECHNOLOGY', 'DATE', 'CUSTOM'
        )
        """)
    op.execute("""
        CREATE TYPE relationship_type AS ENUM (
            'WORKS_FOR', 'KNOWS', 'MANAGES', 'REPORTS_TO',
            'COLLABORATES_WITH', 'OWNS', 'PART_OF', 'COMPETES_WITH',
            'PARTNERS_WITH', 'LOCATED_IN', 'HEADQUARTERED_IN',
            'RELATES_TO', 'DEPENDS_ON', 'IMPLEMENTS', 'DERIVED_FROM',
            'PRECEDES', 'FOLLOWS', 'CONCURRENT_WITH',
            'ASSOCIATED_WITH', 'CUSTOM'
        )
        """)

    # Convert back — any non-standard values become CUSTOM/CONCEPT
    op.execute("""
        ALTER TABLE entities
        ALTER COLUMN entity_type TYPE entity_type
        USING CASE
            WHEN entity_type IN (
                'PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT',
                'EVENT', 'PRODUCT', 'TECHNOLOGY', 'DATE', 'CUSTOM'
            ) THEN entity_type::entity_type
            ELSE 'CONCEPT'::entity_type
        END
        """)
    op.execute("""
        ALTER TABLE relationships
        ALTER COLUMN relationship_type TYPE relationship_type
        USING CASE
            WHEN relationship_type IN (
                'WORKS_FOR', 'KNOWS', 'MANAGES', 'REPORTS_TO',
                'COLLABORATES_WITH', 'OWNS', 'PART_OF', 'COMPETES_WITH',
                'PARTNERS_WITH', 'LOCATED_IN', 'HEADQUARTERED_IN',
                'RELATES_TO', 'DEPENDS_ON', 'IMPLEMENTS', 'DERIVED_FROM',
                'PRECEDES', 'FOLLOWS', 'CONCURRENT_WITH',
                'ASSOCIATED_WITH', 'CUSTOM'
            ) THEN relationship_type::relationship_type
            ELSE 'RELATES_TO'::relationship_type
        END
        """)
