"""Storage optimization for Khora.

Creates additional indexes on PostgreSQL and Neo4j that improve query
and search performance beyond the base indexes created at schema init time.
Designed to run after bulk data ingestion when tables have enough data
for PostgreSQL's planner statistics to be meaningful.

All index creation statements use IF NOT EXISTS for idempotency —
safe to run multiple times.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from loguru import logger

# ---------------------------------------------------------------------------
# PostgreSQL indexes (beyond what SQLAlchemy models already define)
# ---------------------------------------------------------------------------

PG_INDEXES = [
    {
        "name": "idx_chunks_namespace_created",
        "sql": ("CREATE INDEX IF NOT EXISTS idx_chunks_namespace_created ON chunks (namespace_id, created_at DESC)"),
        "purpose": "Temporal filtering within namespace",
    },
    {
        "name": "idx_documents_namespace_created",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_documents_namespace_created ON documents (namespace_id, created_at DESC)"
        ),
        "purpose": "Document temporal queries",
    },
    {
        "name": "idx_entities_namespace_type_name",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_namespace_type_name ON entities (namespace_id, entity_type, name)"
        ),
        "purpose": "Entity type filtering + name lookup",
    },
    {
        "name": "idx_entities_description_fts",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_description_fts "
            "ON entities USING gin (to_tsvector('english', description))"
        ),
        "purpose": "Full-text search on entity descriptions",
    },
    {
        "name": "idx_entities_namespace_confidence",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_namespace_confidence ON entities (namespace_id, confidence DESC)"
        ),
        "purpose": "Confidence-based filtering",
    },
    {
        "name": "idx_chunks_document_namespace",
        "sql": ("CREATE INDEX IF NOT EXISTS idx_chunks_document_namespace ON chunks (document_id, namespace_id)"),
        "purpose": "Document-to-chunk lookups",
    },
    # --- Synced from models.py __table_args__ for catch-up on existing databases ---
    {
        "name": "ix_documents_namespace_source_type",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_documents_namespace_source_type ON documents (namespace_id, source_type)"
        ),
        "purpose": "Document queries filtered by source_type within namespace",
    },
    {
        "name": "ix_entities_namespace_mentions",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_entities_namespace_mentions ON entities (namespace_id, mention_count DESC)"
        ),
        "purpose": "Entity importance ranking (top N entities in namespace)",
    },
    {
        "name": "ix_relationships_namespace_type",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_relationships_namespace_type "
            "ON relationships (namespace_id, relationship_type)"
        ),
        "purpose": "Graph-emulated queries filtering by relationship type within namespace",
    },
    {
        "name": "ix_relationships_target_source",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_relationships_target_source "
            "ON relationships (target_entity_id, source_entity_id)"
        ),
        "purpose": "Reverse traversal (inbound relationships)",
    },
    # --- Temporal indexes on entities ---
    {
        "name": "idx_entities_ns_valid_from",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_ns_valid_from "
            "ON entities (namespace_id, valid_from) WHERE valid_from IS NOT NULL"
        ),
        "purpose": "Temporal queries on entity validity start (partial index)",
    },
    {
        "name": "idx_entities_ns_valid_until",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_ns_valid_until "
            "ON entities (namespace_id, valid_until) WHERE valid_until IS NOT NULL"
        ),
        "purpose": "Temporal queries on entity validity end (partial index)",
    },
]

#: Tables to run ANALYZE on after index creation.
PG_ANALYZE_TABLES = ["chunks", "documents", "entities", "relationships", "temporal_edges"]

#: HNSW indexes to reindex after bulk inserts for optimal recall.
HNSW_INDEXES = [
    "ix_chunks_embedding_hnsw",
    "ix_entities_embedding_hnsw",
]

#: Halfvec expression indexes (created alongside standard HNSW indexes when enabled).
#: These use float16 precision for ~50% smaller index size with minimal recall loss.
#: Requires pgvector extension >= 0.7.0.
HALFVEC_INDEXES = [
    {
        "name": "ix_chunks_embedding_halfvec_hnsw",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_halfvec_hnsw "
            "ON chunks USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops) "
            "WITH (m = {m}, ef_construction = {ef_construction})"
        ),
        "purpose": "Half-precision HNSW index for chunk embeddings (~50% smaller)",
    },
    {
        "name": "ix_entities_embedding_halfvec_hnsw",
        "sql": (
            "CREATE INDEX IF NOT EXISTS ix_entities_embedding_halfvec_hnsw "
            "ON entities USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops) "
            "WITH (m = {m}, ef_construction = {ef_construction})"
        ),
        "purpose": "Half-precision HNSW index for entity embeddings (~50% smaller)",
    },
]

# ---------------------------------------------------------------------------
# Neo4j indexes and constraints
# ---------------------------------------------------------------------------

NEO4J_INDEXES = [
    {
        "name": "entity_ns_name_type_unique",
        "cypher": (
            "CREATE CONSTRAINT entity_ns_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.namespace_id, e.name, e.entity_type) IS UNIQUE"
        ),
        "purpose": "Uniqueness on MERGE key — prevents duplicate entities",
    },
    {
        "name": "entity_namespace_id_unique",
        "cypher": (
            "CREATE CONSTRAINT entity_namespace_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.namespace_id, e.id) IS UNIQUE"
        ),
        "purpose": "Prevent duplicate entities",
    },
    {
        "name": "entity_fulltext",
        "cypher": (
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]"
        ),
        "purpose": "Fuzzy name/description search",
    },
    {
        "name": "episode_namespace_occurred",
        "cypher": (
            "CREATE INDEX episode_namespace_occurred IF NOT EXISTS FOR (e:Episode) ON (e.namespace_id, e.occurred_at)"
        ),
        "purpose": "Temporal episode queries",
    },
    {
        "name": "relates_to_weight",
        "cypher": ("CREATE INDEX relates_to_weight IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.weight)"),
        "purpose": "Weighted traversal",
    },
    {
        "name": "mentioned_in_confidence",
        "cypher": ("CREATE INDEX mentioned_in_confidence IF NOT EXISTS FOR ()-[r:MENTIONED_IN]-() ON (r.confidence)"),
        "purpose": "Confidence filtering",
    },
    # --- Synced from neo4j.py _create_indexes for catch-up on existing databases ---
    {
        "name": "entity_ns_type",
        "cypher": ("CREATE INDEX entity_ns_type IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.entity_type)"),
        "purpose": "Entity namespace + type composite (list queries filtering by type)",
    },
    {
        "name": "entity_source_tool",
        "cypher": "CREATE INDEX entity_source_tool IF NOT EXISTS FOR (e:Entity) ON (e.source_tool)",
        "purpose": "Source-aware queries (e.g. what does Slack say about X?)",
    },
    {
        "name": "entity_confidence",
        "cypher": "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
        "purpose": "Entity confidence threshold filtering",
    },
    {
        "name": "rel_collaborates_ns",
        "cypher": "CREATE INDEX rel_collaborates_ns IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on COLLABORATES_WITH",
    },
    {
        "name": "rel_associated_ns",
        "cypher": "CREATE INDEX rel_associated_ns IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on ASSOCIATED_WITH",
    },
    {
        "name": "rel_depends_ns",
        "cypher": "CREATE INDEX rel_depends_ns IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on DEPENDS_ON",
    },
    {
        "name": "rel_owns_ns",
        "cypher": "CREATE INDEX rel_owns_ns IF NOT EXISTS FOR ()-[r:OWNS]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on OWNS",
    },
    {
        "name": "rel_works_for_ns",
        "cypher": "CREATE INDEX rel_works_for_ns IF NOT EXISTS FOR ()-[r:WORKS_FOR]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on WORKS_FOR",
    },
    {
        "name": "rel_implements_ns",
        "cypher": "CREATE INDEX rel_implements_ns IF NOT EXISTS FOR ()-[r:IMPLEMENTS]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on IMPLEMENTS",
    },
    {
        "name": "rel_part_of_ns",
        "cypher": "CREATE INDEX rel_part_of_ns IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.namespace_id)",
        "purpose": "Namespace filtering on PART_OF",
    },
    {
        "name": "rel_collaborates_conf",
        "cypher": "CREATE INDEX rel_collaborates_conf IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.confidence)",
        "purpose": "Confidence filtering on COLLABORATES_WITH",
    },
    {
        "name": "rel_associated_conf",
        "cypher": "CREATE INDEX rel_associated_conf IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.confidence)",
        "purpose": "Confidence filtering on ASSOCIATED_WITH",
    },
    {
        "name": "rel_depends_conf",
        "cypher": "CREATE INDEX rel_depends_conf IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.confidence)",
        "purpose": "Confidence filtering on DEPENDS_ON",
    },
    # --- Temporal indexes on entities and relationships ---
    {
        "name": "entity_ns_valid_until",
        "cypher": "CREATE INDEX entity_ns_valid_until IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.valid_until)",
        "purpose": "Temporal queries on entity validity end",
    },
    {
        "name": "rel_relates_to_valid_from",
        "cypher": "CREATE INDEX rel_relates_to_valid_from IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_from)",
        "purpose": "Temporal filtering on RELATES_TO relationships",
    },
    {
        "name": "rel_collaborates_valid_from",
        "cypher": (
            "CREATE INDEX rel_collaborates_valid_from IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.valid_from)"
        ),
        "purpose": "Temporal filtering on COLLABORATES_WITH relationships",
    },
    {
        "name": "rel_associated_valid_from",
        "cypher": (
            "CREATE INDEX rel_associated_valid_from IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.valid_from)"
        ),
        "purpose": "Temporal filtering on ASSOCIATED_WITH relationships",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def drop_hnsw_indexes(engine) -> dict:
    """Drop HNSW indexes before bulk ingestion to speed up inserts.

    HNSW indexes add significant overhead to every INSERT because each new
    vector must be connected into the graph.  For bulk loads (--rewrite),
    dropping them beforehand and recreating via optimize_storage() after
    ingestion is much faster.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.

    Returns:
        Dict with ``indexes_dropped`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_dropped": 0,
        "errors": [],
    }

    all_hnsw = list(HNSW_INDEXES) + [idx["name"] for idx in HALFVEC_INDEXES]
    for idx_name in all_hnsw:
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                row = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                if row.scalar() is None:
                    continue
                logger.info(f"Dropping HNSW index {idx_name} for bulk load")
                await conn.execute(text(f"DROP INDEX IF EXISTS {idx_name}"))
            result["indexes_dropped"] += 1
        except Exception as e:
            msg = f"DROP INDEX {idx_name}: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    return result


async def ensure_hnsw_indexes(engine) -> dict:
    """Recreate HNSW indexes if they don't exist.

    Used after bulk load to restore indexes that were dropped by
    ``drop_hnsw_indexes``.  Uses CREATE INDEX IF NOT EXISTS for idempotency.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.

    Returns:
        Dict with ``indexes_created`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_created": 0,
        "errors": [],
    }

    hnsw_ddl = [
        (
            "ix_chunks_embedding_hnsw",
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 24, ef_construction = 128)",
        ),
        (
            "ix_entities_embedding_hnsw",
            "CREATE INDEX IF NOT EXISTS ix_entities_embedding_hnsw "
            "ON entities USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 24, ef_construction = 128)",
        ),
    ]

    import asyncio

    async def _create_one(idx_name: str, ddl: str) -> tuple[bool, str | None]:
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                # Check if index already exists to know if CREATE is a no-op
                row = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                already_exists = row.scalar() is not None
                if already_exists:
                    logger.debug(f"HNSW index {idx_name} already exists, skipping CREATE")
                    return False, None
                logger.info(f"Creating HNSW index {idx_name}")
                await conn.execute(text(ddl))
                return True, None
        except Exception as e:
            return False, f"CREATE INDEX {idx_name}: {e}"

    # Build both indexes in parallel (independent tables, no lock contention)
    tasks = [_create_one(name, ddl) for name, ddl in hnsw_ddl]
    outcomes = await asyncio.gather(*tasks)

    freshly_created = 0
    for created, error in outcomes:
        if error:
            result["errors"].append(error)
            logger.warning(error)
        elif created:
            freshly_created += 1

    result["indexes_created"] = freshly_created
    result["freshly_created"] = freshly_created
    return result


#: m / ef_construction for halfvec HNSW recreation. Must match migration 018
#: (018_halfvec_hnsw_indexes) so a recreate produces the same index the
#: migration would have, and the default embedding dimension used there.
_HALFVEC_HNSW_M = 24
_HALFVEC_HNSW_EF_CONSTRUCTION = 128
_HALFVEC_DEFAULT_DIMENSION = 1536


async def ensure_halfvec_indexes(engine, *, embedding_dimension: int = _HALFVEC_DEFAULT_DIMENSION) -> dict:
    """Recreate the halfvec HNSW expression indexes if they don't exist.

    Pairs with ``drop_hnsw_indexes``, which drops the halfvec indexes along
    with the float32 ones before a bulk load.  Without this recreate step the
    ``HALFVEC_INDEXES`` DDL was never executed, so after one drop the halfvec
    indexes were gone permanently (migration 018 will not re-run on a stamped
    database) and every halfvec-cast search sequential-scanned (#1137).

    Formats the ``HALFVEC_INDEXES`` SQL templates with the configured
    ``embedding_dimension`` and the migration-018 ``m`` / ``ef_construction``
    values.  Postgres + pgvector >= 0.7.0 only; on older pgvector the
    ``CREATE`` fails (no ``halfvec`` type) and is recorded as a per-index error
    without affecting the float32 indexes, matching the fail-soft pattern used
    throughout this module.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        embedding_dimension: Vector dimension to cast to ``halfvec``.

    Returns:
        Dict with ``indexes_created`` / ``freshly_created`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_created": 0,
        "errors": [],
    }

    halfvec_ddl = [
        (
            idx["name"],
            idx["sql"].format(
                dim=embedding_dimension,
                m=_HALFVEC_HNSW_M,
                ef_construction=_HALFVEC_HNSW_EF_CONSTRUCTION,
            ),
        )
        for idx in HALFVEC_INDEXES
    ]

    import asyncio

    async def _create_one(idx_name: str, ddl: str) -> tuple[bool, str | None]:
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                row = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                already_exists = row.scalar() is not None
                if already_exists:
                    logger.debug(f"halfvec index {idx_name} already exists, skipping CREATE")
                    return False, None
                logger.info(f"Creating halfvec HNSW index {idx_name}")
                await conn.execute(text(ddl))
                return True, None
        except Exception as e:
            return False, f"CREATE INDEX {idx_name}: {e}"

    tasks = [_create_one(name, ddl) for name, ddl in halfvec_ddl]
    outcomes = await asyncio.gather(*tasks)

    freshly_created = 0
    for created, error in outcomes:
        if error:
            result["errors"].append(error)
            logger.warning(error)
        elif created:
            freshly_created += 1

    result["indexes_created"] = freshly_created
    result["freshly_created"] = freshly_created
    return result


async def prepare_for_bulk_load(coordinator) -> dict:
    """Prepare storage for bulk data ingestion by dropping HNSW indexes.

    Call before bulk ingestion (e.g., --rewrite mode) to eliminate HNSW
    overhead on every INSERT.  Call ``optimize_storage()`` after ingestion
    to recreate and reindex.

    Args:
        coordinator: A connected ``StorageCoordinator`` instance.

    Returns:
        Dict with results from index dropping.
    """
    backend = coordinator.vector or coordinator.relational
    if backend is not None:
        engine = getattr(backend, "_engine", None)
        if engine is not None:
            return await drop_hnsw_indexes(engine)
    return {"indexes_dropped": 0, "errors": []}


async def reindex_hnsw_concurrently(engine) -> dict:
    """Reindex HNSW indexes concurrently after bulk inserts.

    HNSW indexes can become suboptimal after large batch inserts because
    new vectors are appended without full graph connectivity.  Running
    ``REINDEX INDEX CONCURRENTLY`` rebuilds the index in the background
    without blocking reads or writes.

    Requires PostgreSQL 12+.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.

    Returns:
        Dict with ``indexes_reindexed`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_reindexed": 0,
        "errors": [],
    }

    # REINDEX CONCURRENTLY cannot run inside a transaction block,
    # so we use the raw connection with autocommit.
    for idx_name in HNSW_INDEXES:
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                row = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                if row.scalar() is None:
                    logger.debug(f"Index {idx_name} does not exist, skipping reindex")
                    continue
                logger.info(f"Reindexing {idx_name} concurrently...")
                await conn.execute(text(f"REINDEX INDEX CONCURRENTLY {idx_name}"))
            result["indexes_reindexed"] += 1
            logger.info(f"Reindexed {idx_name}")
        except Exception as e:
            msg = f"REINDEX {idx_name}: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    return result


# ---------------------------------------------------------------------------
# Per-namespace partial HNSW indexes (policy-gated, operator-driven)
# ---------------------------------------------------------------------------
#
# A partial HNSW index ``... WHERE namespace_id = <ns>`` lets a
# namespace-scoped vector query use a dedicated index instead of
# post-filtering the shared index. Measured 50x faster with better recall on
# a tight-filter recall at 500k rows (#1470). The cost is catalog pressure:
# N promoted namespaces = N extra indexes on one table (planner overhead,
# autovacuum load, catalog bloat). Because of that trade-off this is a
# POLICY-GATED mechanism, NEVER automatic per-namespace — the promotion
# functions no-op unless an operator opts in via config, and even then only
# promote namespaces that cross a row threshold, capped by a hard ceiling.
#
# m / ef_construction match the tuned float32 indexes so a promoted namespace
# gets the same graph quality as the shared index.
_PARTIAL_HNSW_M = 24
_PARTIAL_HNSW_EF_CONSTRUCTION = 128

#: Tables that carry a per-namespace HNSW index. Each has an ``embedding
#: vector`` column and a ``namespace_id uuid`` column.
_PARTIAL_HNSW_TABLES = ("chunks", "entities")


def _partial_hnsw_index_name(table: str, namespace_id: UUID) -> str:
    """Deterministic, identifier-safe index name for a namespace partial index.

    Uses the UUID hex (no hyphens) so the name is a valid unquoted Postgres
    identifier and round-trips: the same namespace always maps to the same
    index name, which makes CREATE / DROP idempotent.
    """
    return f"ix_{table}_embedding_hnsw_ns_{namespace_id.hex}"


async def list_partial_hnsw_indexes(engine, *, table: str | None = None) -> list[str]:
    """List existing per-namespace partial HNSW index names.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        table: If given, restrict to one table (``chunks`` / ``entities``).

    Returns:
        Sorted list of matching index names.
    """
    from sqlalchemy import text

    tables = (table,) if table else _PARTIAL_HNSW_TABLES
    found: list[str] = []
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        for tbl in tables:
            # ``ix_<table>_embedding_hnsw_ns_<32 hex chars>`` — the trailing
            # ``\_ns\_`` LIKE anchor avoids matching the shared
            # ``ix_chunks_embedding_hnsw`` index.
            rows = await conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :tbl AND indexname LIKE :pat ESCAPE '\\'"),
                {"tbl": tbl, "pat": f"ix\\_{tbl}\\_embedding\\_hnsw\\_ns\\_%"},
            )
            found.extend(r[0] for r in rows)
    return sorted(found)


async def _namespace_row_count(engine, namespace_id: UUID) -> int:
    """Chunk count for a namespace — the promotion-eligibility metric."""
    from sqlalchemy import text

    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        row = await conn.execute(
            text("SELECT count(*) FROM chunks WHERE namespace_id = :ns"),
            {"ns": namespace_id},
        )
        return int(row.scalar() or 0)


async def promote_namespace_hnsw(
    engine,
    namespace_id: UUID,
    *,
    m: int = _PARTIAL_HNSW_M,
    ef_construction: int = _PARTIAL_HNSW_EF_CONSTRUCTION,
) -> dict:
    """Create per-namespace partial HNSW indexes for ``namespace_id``.

    This is the raw MECHANISM — it always builds the indexes, bypassing the
    promotion policy. Prefer ``maybe_promote_namespace`` for operator/task
    use, which enforces the enabled flag, row threshold, and index ceiling.

    Builds one partial index per table in ``_PARTIAL_HNSW_TABLES`` with a
    ``WHERE namespace_id = <ns>`` predicate, using
    ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` so it never blocks writes and
    is idempotent. Postgres-only.

    The namespace UUID is bound via a parameter for the row-count read and
    formatted into the DDL as a literal for the partial predicate — DDL cannot
    take bind parameters, but a ``UUID`` renders to a fixed 36-char hex form
    with no injection surface.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        namespace_id: The namespace (row-level and stable IDs are the same on
            these tables) to promote.
        m: HNSW M parameter for the partial indexes.
        ef_construction: HNSW ef_construction for the partial indexes.

    Returns:
        Dict with ``indexes_created`` count, ``indexes`` (names built), and
        ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {"indexes_created": 0, "indexes": [], "errors": []}
    ns_literal = str(namespace_id)

    for table in _PARTIAL_HNSW_TABLES:
        idx_name = _partial_hnsw_index_name(table, namespace_id)
        ddl = (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx_name} "
            f"ON {table} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)}) "
            f"WHERE namespace_id = '{ns_literal}'::uuid"
        )
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                existing = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                if existing.scalar() is not None:
                    logger.debug(f"Partial HNSW index {idx_name} already exists, skipping CREATE")
                    result["indexes"].append(idx_name)
                    continue
                logger.info(f"Creating partial HNSW index {idx_name} for namespace {ns_literal}")
                await conn.execute(text(ddl))
            result["indexes_created"] += 1
            result["indexes"].append(idx_name)
        except Exception as e:  # noqa: BLE001
            msg = f"CREATE INDEX {idx_name}: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    return result


async def demote_namespace_hnsw(engine, namespace_id: UUID) -> dict:
    """Drop the per-namespace partial HNSW indexes for ``namespace_id``.

    Idempotent (``DROP INDEX CONCURRENTLY IF EXISTS``). Call this on namespace
    deletion so a deleted tenant does not leave orphan indexes behind, and to
    reclaim a slot under the ``hnsw_partial_max_indexes`` ceiling.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        namespace_id: The namespace whose partial indexes to drop.

    Returns:
        Dict with ``indexes_dropped`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {"indexes_dropped": 0, "errors": []}

    for table in _PARTIAL_HNSW_TABLES:
        idx_name = _partial_hnsw_index_name(table, namespace_id)
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                existing = await conn.execute(
                    text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
                    {"name": idx_name},
                )
                if existing.scalar() is None:
                    continue
                logger.info(f"Dropping partial HNSW index {idx_name}")
                await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name}"))
            result["indexes_dropped"] += 1
        except Exception as e:  # noqa: BLE001
            msg = f"DROP INDEX {idx_name}: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    return result


async def maybe_promote_namespace(
    engine,
    namespace_id: UUID,
    *,
    enabled: bool = False,
    min_rows: int = 50000,
    max_indexes: int = 64,
) -> dict:
    """Promote a namespace to partial HNSW indexes IF the policy allows it.

    This is the POLICY gate around ``promote_namespace_hnsw``. It is
    default-OFF and never runs automatically on the write path — an operator
    or a maintenance task must call it with ``enabled=True`` (wire the
    arguments from ``KhoraConfig.storage.hnsw_partial_*``). It promotes only
    when all of the following hold:

    * ``enabled`` is True (the master switch);
    * the namespace has at least ``min_rows`` chunks (hot enough to justify a
      dedicated index);
    * fewer than ``max_indexes`` per-namespace partial indexes already exist
      on the ``chunks`` table (the catalog-bloat ceiling).

    A refused promotion is reported in the result (``promoted=False`` with a
    ``reason``), never raised — callers treat this as advisory.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        namespace_id: The namespace to consider for promotion.
        enabled: Master switch (``KHORA_STORAGE_HNSW_PARTIAL_ENABLED``).
        min_rows: Row-count threshold (``KHORA_STORAGE_HNSW_PARTIAL_MIN_ROWS``).
        max_indexes: Per-table ceiling
            (``KHORA_STORAGE_HNSW_PARTIAL_MAX_INDEXES``).

    Returns:
        Dict with ``promoted`` (bool), ``reason`` (str), ``row_count`` (int),
        and, when promoted, the ``promote_namespace_hnsw`` result under
        ``created``.
    """
    if not enabled:
        return {"promoted": False, "reason": "disabled", "row_count": 0}

    # Already promoted? Idempotent short-circuit — do not count it against the
    # ceiling twice or rebuild it.
    existing = await list_partial_hnsw_indexes(engine, table="chunks")
    if _partial_hnsw_index_name("chunks", namespace_id) in existing:
        return {"promoted": False, "reason": "already_promoted", "row_count": 0}

    row_count = await _namespace_row_count(engine, namespace_id)
    if row_count < min_rows:
        return {"promoted": False, "reason": "below_min_rows", "row_count": row_count}

    if len(existing) >= max_indexes:
        logger.warning(
            f"Refusing to promote namespace {namespace_id}: partial HNSW index ceiling "
            f"({max_indexes}) reached ({len(existing)} exist). Demote a colder namespace first."
        )
        return {"promoted": False, "reason": "ceiling_reached", "row_count": row_count}

    created = await promote_namespace_hnsw(engine, namespace_id)
    return {"promoted": True, "reason": "promoted", "row_count": row_count, "created": created}


async def optimize_postgresql(
    engine,
    *,
    reindex_hnsw: bool = True,
    embedding_dimension: int = _HALFVEC_DEFAULT_DIMENSION,
) -> dict:
    """Create optimal PostgreSQL indexes, run ANALYZE, and optionally reindex HNSW.

    Executes raw DDL against the provided SQLAlchemy async engine,
    so that callers don't need to provide a raw connection URL.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        reindex_hnsw: If True, run ``REINDEX INDEX CONCURRENTLY`` on HNSW
            indexes after creating other indexes.  Set to False to skip
            (e.g. for small datasets where reindexing adds no benefit).
        embedding_dimension: Vector dimension used when recreating the halfvec
            HNSW indexes (must match the deployment's embedding dimension).

    Returns:
        Dict with ``indexes_created``, ``tables_analyzed``,
        ``hnsw_reindexed``, and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_created": 0,
        "tables_analyzed": 0,
        "hnsw_reindexed": 0,
        "errors": [],
    }

    async with engine.begin() as conn:
        for idx in PG_INDEXES:
            try:
                logger.debug(f"Creating index {idx['name']} ({idx['purpose']})")
                await conn.execute(text(idx["sql"]))
                result["indexes_created"] += 1
            except Exception as e:
                msg = f"Index {idx['name']}: {e}"
                result["errors"].append(msg)
                logger.warning(msg)

        for table in PG_ANALYZE_TABLES:
            try:
                logger.debug(f"Analyzing table {table}")
                await conn.execute(text(f"ANALYZE {table}"))
                result["tables_analyzed"] += 1
            except Exception as e:
                msg = f"ANALYZE {table}: {e}"
                result["errors"].append(msg)
                logger.warning(msg)

    # Ensure HNSW indexes exist (may have been dropped by prepare_for_bulk_load).
    # Only reindex when indexes already existed before ensure — a freshly created
    # index is already optimal, so REINDEX would duplicate the build work.
    if reindex_hnsw:
        ensure_result = await ensure_hnsw_indexes(engine)
        result["indexes_created"] += ensure_result["indexes_created"]
        result["errors"].extend(ensure_result["errors"])

        # Recreate the halfvec HNSW indexes too — drop_hnsw_indexes drops them
        # alongside the float32 indexes, so the recreate must restore both for
        # the drop/recreate pair to round-trip (#1137).
        halfvec_result = await ensure_halfvec_indexes(engine, embedding_dimension=embedding_dimension)
        result["indexes_created"] += halfvec_result["indexes_created"]
        result["errors"].extend(halfvec_result["errors"])

        freshly_created = ensure_result.get("freshly_created", 0)
        if freshly_created == 0:
            # Indexes pre-existed — they may be suboptimal after bulk inserts
            hnsw_result = await reindex_hnsw_concurrently(engine)
            result["hnsw_reindexed"] = hnsw_result["indexes_reindexed"]
            result["errors"].extend(hnsw_result["errors"])
        else:
            logger.info(f"Skipping REINDEX — {freshly_created} HNSW index(es) freshly created")

    return result


async def optimize_neo4j(driver, *, database: str = "neo4j") -> dict:
    """Create optimal Neo4j indexes and constraints.

    Args:
        driver: An ``neo4j.AsyncDriver`` instance.
        database: Neo4j database name.

    Returns:
        Dict with ``indexes_created``, ``duplicates_removed``, and ``errors``.
    """
    result: dict[str, Any] = {
        "indexes_created": 0,
        "duplicates_removed": 0,
        "errors": [],
    }

    # De-duplicate existing Entity nodes before creating unique constraint.
    # Merges metadata from duplicates into the kept node, then deletes dupes.
    dedup_cypher = """
    MATCH (e:Entity)
    WITH e.namespace_id AS ns, e.name AS name, e.entity_type AS type,
         collect(e) AS nodes
    WHERE size(nodes) > 1
    WITH nodes[0] AS keep, tail(nodes) AS dupes
    UNWIND dupes AS dup
    SET keep.source_document_ids = keep.source_document_ids +
            [x IN dup.source_document_ids WHERE NOT x IN keep.source_document_ids],
        keep.source_chunk_ids = keep.source_chunk_ids +
            [x IN dup.source_chunk_ids WHERE NOT x IN keep.source_chunk_ids],
        keep.mention_count = keep.mention_count + dup.mention_count
    DETACH DELETE dup
    RETURN count(dup) AS duplicates_removed
    """
    async with driver.session(database=database) as session:
        try:
            dedup_result = await session.run(dedup_cypher)
            record = await dedup_result.single()
            removed = record["duplicates_removed"] if record else 0
            result["duplicates_removed"] = removed
            if removed:
                logger.info(f"De-duplicated {removed} Entity nodes in Neo4j")
        except Exception as e:
            msg = f"Neo4j entity de-duplication: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    # Migration: drop legacy plain indexes that conflict with uniqueness constraints.
    # Neo4j refuses to create a CONSTRAINT when a plain INDEX on the same properties
    # already exists.  This handles databases created before the code switched to
    # constraints.  Safe to run repeatedly — DROP IF EXISTS is a no-op once gone.
    legacy_indexes_to_drop = [
        "entity_ns_name_type_unique",  # was a plain index, now a constraint
    ]
    async with driver.session(database=database) as session:
        for idx_name in legacy_indexes_to_drop:
            try:
                await session.run(f"DROP INDEX {idx_name} IF EXISTS")
            except Exception as e:
                logger.debug(f"Failed to drop legacy index {idx_name}: {e}")

    async with driver.session(database=database) as session:
        for idx in NEO4J_INDEXES:
            try:
                logger.debug(f"Creating Neo4j index {idx['name']} ({idx['purpose']})")
                await session.run(idx["cypher"])
                result["indexes_created"] += 1
            except Exception as e:
                msg = f"Neo4j index {idx['name']}: {e}"
                result["errors"].append(msg)
                logger.warning(msg)

    return result


async def optimize_storage(coordinator) -> dict:
    """Run all optimizations against a connected StorageCoordinator.

    This is the main entry point for callers that already have a
    ``StorageCoordinator`` (e.g. via ``Khora.storage``).

    Args:
        coordinator: A connected ``StorageCoordinator`` instance.

    Returns:
        Combined results: ``{"postgresql": {...}, "neo4j": {...}}``.
    """
    results: dict[str, dict | None] = {"postgresql": None, "neo4j": None, "surrealdb": None}

    # Optimize PostgreSQL / pgvector (they share the same engine)
    backend = coordinator.vector or coordinator.relational
    if backend is not None:
        engine = getattr(backend, "_engine", None)
        if engine is not None:
            embedding_dimension = getattr(backend, "_embedding_dimension", _HALFVEC_DEFAULT_DIMENSION)
            results["postgresql"] = await optimize_postgresql(engine, embedding_dimension=embedding_dimension)
        else:
            logger.warning("No SQLAlchemy engine found on backend; skipping PostgreSQL optimization")

    # Optimize Neo4j
    graph = coordinator.graph
    if graph is not None:
        driver = getattr(graph, "_driver", None)
        database = getattr(graph, "_database", "neo4j")
        if driver is not None:
            results["neo4j"] = await optimize_neo4j(driver, database=database)
        else:
            logger.warning("No Neo4j driver found on backend; skipping Neo4j optimization")

    # Create deferred SurrealDB search indexes (HNSW + BM25)
    surreal_conn = getattr(coordinator.relational, "_conn", None) if coordinator.relational else None
    if surreal_conn is not None and hasattr(surreal_conn, "_client"):
        from khora.storage.backends.surrealdb.schema import ensure_search_indexes

        await ensure_search_indexes(surreal_conn)
        results["surrealdb"] = {"search_indexes_created": True}

    return results
