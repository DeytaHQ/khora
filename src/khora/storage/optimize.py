"""Storage optimization for Khora Memory Lake.

Creates additional indexes on PostgreSQL and Neo4j that improve query
and search performance beyond the base indexes created at schema init time.
Designed to run after bulk data ingestion when tables have enough data
for PostgreSQL's planner statistics to be meaningful.

All index creation statements use IF NOT EXISTS for idempotency —
safe to run multiple times.
"""

from __future__ import annotations

from typing import Any

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
            "CREATE INDEX rel_collaborates_valid_from IF NOT EXISTS "
            "FOR ()-[r:COLLABORATES_WITH]-() ON (r.valid_from)"
        ),
        "purpose": "Temporal filtering on COLLABORATES_WITH relationships",
    },
    {
        "name": "rel_associated_valid_from",
        "cypher": (
            "CREATE INDEX rel_associated_valid_from IF NOT EXISTS " "FOR ()-[r:ASSOCIATED_WITH]-() ON (r.valid_from)"
        ),
        "purpose": "Temporal filtering on ASSOCIATED_WITH relationships",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
            logger.info(f"Reindexing {idx_name} concurrently...")
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                await conn.execute(text(f"REINDEX INDEX CONCURRENTLY IF EXISTS {idx_name}"))
            result["indexes_reindexed"] += 1
            logger.info(f"Reindexed {idx_name}")
        except Exception as e:
            msg = f"REINDEX {idx_name}: {e}"
            result["errors"].append(msg)
            logger.warning(msg)

    return result


async def create_halfvec_indexes(
    engine,
    *,
    embedding_dimension: int = 1536,
    hnsw_m: int = 24,
    hnsw_ef_construction: int = 128,
) -> dict:
    """Create halfvec expression HNSW indexes for reduced index size.

    These indexes cast the full-precision ``vector`` column to ``halfvec``
    (float16) at index time, yielding ~50% smaller HNSW indexes with
    minimal recall loss.

    Requires pgvector extension >= 0.7.0.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        embedding_dimension: Dimension of embedding vectors.
        hnsw_m: HNSW M parameter.
        hnsw_ef_construction: HNSW ef_construction parameter.

    Returns:
        Dict with ``indexes_created`` count and ``errors``.
    """
    from sqlalchemy import text

    result: dict[str, Any] = {
        "indexes_created": 0,
        "errors": [],
    }

    async with engine.begin() as conn:
        for idx in HALFVEC_INDEXES:
            try:
                sql = idx["sql"].format(
                    dim=embedding_dimension,
                    m=hnsw_m,
                    ef_construction=hnsw_ef_construction,
                )
                logger.debug(f"Creating halfvec index {idx['name']} ({idx['purpose']})")
                await conn.execute(text(sql))
                result["indexes_created"] += 1
            except Exception as e:
                msg = f"Halfvec index {idx['name']}: {e}"
                result["errors"].append(msg)
                logger.warning(msg)

    return result


async def optimize_postgresql(engine, *, reindex_hnsw: bool = True) -> dict:
    """Create optimal PostgreSQL indexes, run ANALYZE, and optionally reindex HNSW.

    Executes raw DDL against the provided SQLAlchemy async engine,
    so that callers don't need to provide a raw connection URL.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.
        reindex_hnsw: If True, run ``REINDEX INDEX CONCURRENTLY`` on HNSW
            indexes after creating other indexes.  Set to False to skip
            (e.g. for small datasets where reindexing adds no benefit).

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

    # Reindex HNSW indexes after bulk operations for optimal recall
    if reindex_hnsw:
        hnsw_result = await reindex_hnsw_concurrently(engine)
        result["hnsw_reindexed"] = hnsw_result["indexes_reindexed"]
        result["errors"].extend(hnsw_result["errors"])

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
            except Exception:
                pass  # index doesn't exist or already dropped

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
    ``StorageCoordinator`` (e.g. via ``MemoryLake.storage``).

    Args:
        coordinator: A connected ``StorageCoordinator`` instance.

    Returns:
        Combined results: ``{"postgresql": {...}, "neo4j": {...}}``.
    """
    results: dict[str, dict | None] = {"postgresql": None, "neo4j": None}

    # Optimize PostgreSQL / pgvector (they share the same engine)
    backend = coordinator.vector or coordinator.relational
    if backend is not None:
        engine = getattr(backend, "_engine", None)
        if engine is not None:
            results["postgresql"] = await optimize_postgresql(engine)
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

    return results
