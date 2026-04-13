"""Entity dedup, unique constraint, and index improvements.

Revision ID: 008_entity_dedup_and_indexes
Revises: 007_hnsw_parameter_tuning
Create Date: 2026-02-23

D-4: Entity deduplication and unique constraint
- Dedup entities: for each (namespace_id, name, entity_type) group with duplicates,
  keep the row with highest mention_count (lowest id as tiebreaker),
  merge source_document_ids and source_chunk_ids from all duplicates into the survivor,
  sum mention_counts across the group
- Re-point relationships and temporal_edges from duplicate entities to survivors
- Delete non-survivor duplicates
- Drop old non-unique index ix_entities_namespace_name_type
- Add UNIQUE CONSTRAINT uq_entities_namespace_name_type on (namespace_id, name, entity_type)

5.3: khora_chunks composite index
- Add ix_khora_chunks_ns_doc on khora_chunks(namespace_id, document_id)
  Supports Skeleton/VectorCypher engine queries that filter by namespace + document

5.5: Entity temporal partial indexes
- ix_entities_valid_from on entities(valid_from) WHERE valid_from IS NOT NULL
- ix_entities_valid_until on entities(valid_until) WHERE valid_until IS NOT NULL
  Accelerates temporal filtering without bloating the index with NULL rows

Note on downgrade: The dedup step is irreversible — downgrade drops the constraint
and indexes but cannot restore deleted duplicate entities. This is documented and
expected.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "008_entity_dedup_and_indexes"
down_revision: str | Sequence[str] | None = "007_hnsw_parameter_tuning"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # =========================================================================
    # D-4: Entity deduplication
    # =========================================================================

    # Step 1: Merge source_document_ids, source_chunk_ids, and mention_counts
    # from all duplicates into the survivor (highest mention_count, lowest id).
    op.execute(
        text("""
            WITH duplicate_groups AS (
                SELECT namespace_id, name, entity_type
                FROM entities
                GROUP BY namespace_id, name, entity_type
                HAVING count(*) > 1
            ),
            survivors AS (
                SELECT DISTINCT ON (e.namespace_id, e.name, e.entity_type)
                    e.id, e.namespace_id, e.name, e.entity_type
                FROM entities e
                INNER JOIN duplicate_groups dg
                    ON e.namespace_id = dg.namespace_id
                    AND e.name = dg.name
                    AND e.entity_type = dg.entity_type
                ORDER BY e.namespace_id, e.name, e.entity_type,
                         e.mention_count DESC, e.id ASC
            ),
            merged_data AS (
                SELECT
                    s.id AS survivor_id,
                    COALESCE(
                        (SELECT array_agg(DISTINCT doc_id)
                         FROM entities e2
                         CROSS JOIN LATERAL unnest(
                             COALESCE(e2.source_document_ids, '{}'::uuid[])
                         ) AS doc_id
                         WHERE e2.namespace_id = s.namespace_id
                           AND e2.name = s.name
                           AND e2.entity_type = s.entity_type
                        ),
                        '{}'::uuid[]
                    ) AS merged_doc_ids,
                    COALESCE(
                        (SELECT array_agg(DISTINCT chunk_id)
                         FROM entities e3
                         CROSS JOIN LATERAL unnest(
                             COALESCE(e3.source_chunk_ids, '{}'::uuid[])
                         ) AS chunk_id
                         WHERE e3.namespace_id = s.namespace_id
                           AND e3.name = s.name
                           AND e3.entity_type = s.entity_type
                        ),
                        '{}'::uuid[]
                    ) AS merged_chunk_ids,
                    (SELECT COALESCE(sum(e4.mention_count), 0)
                     FROM entities e4
                     WHERE e4.namespace_id = s.namespace_id
                       AND e4.name = s.name
                       AND e4.entity_type = s.entity_type
                    ) AS total_mentions
                FROM survivors s
            )
            UPDATE entities e
            SET source_document_ids = md.merged_doc_ids,
                source_chunk_ids = md.merged_chunk_ids,
                mention_count = md.total_mentions
            FROM merged_data md
            WHERE e.id = md.survivor_id
        """)
    )

    # Step 2: Re-point relationships from duplicate entities to survivors.
    # This prevents cascade-delete from dropping valid relationship data.
    _remap_sql = """
        WITH survivor_map AS (
            SELECT
                id,
                first_value(id) OVER w AS survivor_id
            FROM entities
            WINDOW w AS (
                PARTITION BY namespace_id, name, entity_type
                ORDER BY mention_count DESC, id ASC
            )
        )
        UPDATE {table} t
        SET {column} = sm.survivor_id
        FROM survivor_map sm
        WHERE t.{column} = sm.id
          AND sm.id != sm.survivor_id
    """
    for table in ("relationships", "temporal_edges"):
        for column in ("source_entity_id", "target_entity_id"):
            op.execute(text(_remap_sql.format(table=table, column=column)))

    # Step 3: Delete non-survivor duplicates.
    op.execute(
        text("""
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY namespace_id, name, entity_type
                           ORDER BY mention_count DESC, id ASC
                       ) AS rn
                FROM entities
            )
            DELETE FROM entities
            WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
        """)
    )

    # Step 4: Drop the old non-unique index and create a unique constraint.
    op.execute(text("DROP INDEX IF EXISTS ix_entities_namespace_name_type"))
    op.create_unique_constraint(
        "uq_entities_namespace_name_type",
        "entities",
        ["namespace_id", "name", "entity_type"],
    )

    # =========================================================================
    # 5.3: khora_chunks composite index
    # =========================================================================
    conn = op.get_bind()
    has_khora_chunks = conn.execute(
        text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'khora_chunks')")
    ).scalar()
    if has_khora_chunks:
        op.execute(
            text("CREATE INDEX IF NOT EXISTS ix_khora_chunks_ns_doc ON khora_chunks (namespace_id, document_id)")
        )

    # =========================================================================
    # 5.5: Entity temporal partial indexes
    # =========================================================================
    op.execute(
        text("CREATE INDEX IF NOT EXISTS ix_entities_valid_from ON entities (valid_from) WHERE valid_from IS NOT NULL")
    )
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_entities_valid_until ON entities (valid_until) WHERE valid_until IS NOT NULL"
        )
    )


def downgrade() -> None:
    # Drop temporal partial indexes
    op.execute(text("DROP INDEX IF EXISTS ix_entities_valid_until"))
    op.execute(text("DROP INDEX IF EXISTS ix_entities_valid_from"))

    # Drop khora_chunks composite index
    op.execute(text("DROP INDEX IF EXISTS ix_khora_chunks_ns_doc"))

    # Drop unique constraint and restore the non-unique index
    op.drop_constraint("uq_entities_namespace_name_type", "entities", type_="unique")
    op.create_index(
        "ix_entities_namespace_name_type",
        "entities",
        ["namespace_id", "name", "entity_type"],
    )

    # NOTE: The dedup step is irreversible — deleted duplicate entities
    # cannot be restored. Re-ingestion of source documents will recreate them.
