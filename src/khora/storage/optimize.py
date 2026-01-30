"""Storage optimization for Khora Memory Lake.

Creates additional indexes on PostgreSQL and Neo4j that improve query
and search performance beyond the base indexes created at schema init time.
Designed to run after bulk data ingestion when tables have enough data
for PostgreSQL's planner statistics to be meaningful.

All index creation statements use IF NOT EXISTS for idempotency —
safe to run multiple times.
"""

from __future__ import annotations

from loguru import logger

# ---------------------------------------------------------------------------
# PostgreSQL indexes (beyond what SQLAlchemy models already define)
# ---------------------------------------------------------------------------

PG_INDEXES = [
    {
        "name": "idx_chunks_namespace_created",
        "sql": ("CREATE INDEX IF NOT EXISTS idx_chunks_namespace_created " "ON chunks (namespace_id, created_at DESC)"),
        "purpose": "Temporal filtering within namespace",
    },
    {
        "name": "idx_documents_namespace_created",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_documents_namespace_created " "ON documents (namespace_id, created_at DESC)"
        ),
        "purpose": "Document temporal queries",
    },
    {
        "name": "idx_entities_namespace_type_name",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_entities_namespace_type_name "
            "ON entities (namespace_id, entity_type, name)"
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
            "CREATE INDEX IF NOT EXISTS idx_entities_namespace_confidence "
            "ON entities (namespace_id, confidence DESC)"
        ),
        "purpose": "Confidence-based filtering",
    },
    {
        "name": "idx_relationships_source_target",
        "sql": (
            "CREATE INDEX IF NOT EXISTS idx_relationships_source_target "
            "ON relationships (source_entity_id, target_entity_id)"
        ),
        "purpose": "Relationship traversal",
    },
    {
        "name": "idx_chunks_document_namespace",
        "sql": ("CREATE INDEX IF NOT EXISTS idx_chunks_document_namespace " "ON chunks (document_id, namespace_id)"),
        "purpose": "Document-to-chunk lookups",
    },
]

#: Tables to run ANALYZE on after index creation.
PG_ANALYZE_TABLES = ["chunks", "documents", "entities", "relationships"]

# ---------------------------------------------------------------------------
# Neo4j indexes and constraints
# ---------------------------------------------------------------------------

NEO4J_INDEXES = [
    {
        "name": "entity_namespace_name_type",
        "cypher": (
            "CREATE INDEX entity_namespace_name_type IF NOT EXISTS "
            "FOR (e:Entity) ON (e.namespace_id, e.name, e.entity_type)"
        ),
        "purpose": "Primary entity lookup (composite)",
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
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS " "FOR (e:Entity) ON EACH [e.name, e.description]"
        ),
        "purpose": "Fuzzy name/description search",
    },
    {
        "name": "episode_namespace_occurred",
        "cypher": (
            "CREATE INDEX episode_namespace_occurred IF NOT EXISTS "
            "FOR (e:Episode) ON (e.namespace_id, e.occurred_at)"
        ),
        "purpose": "Temporal episode queries",
    },
    {
        "name": "relates_to_weight",
        "cypher": ("CREATE INDEX relates_to_weight IF NOT EXISTS " "FOR ()-[r:RELATES_TO]-() ON (r.weight)"),
        "purpose": "Weighted traversal",
    },
    {
        "name": "mentioned_in_confidence",
        "cypher": (
            "CREATE INDEX mentioned_in_confidence IF NOT EXISTS " "FOR ()-[r:MENTIONED_IN]-() ON (r.confidence)"
        ),
        "purpose": "Confidence filtering",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def optimize_postgresql(engine) -> dict:
    """Create optimal PostgreSQL indexes and run ANALYZE.

    Executes raw DDL against the provided SQLAlchemy async engine,
    so that callers don't need to provide a raw connection URL.

    Args:
        engine: An ``sqlalchemy.ext.asyncio.AsyncEngine`` instance.

    Returns:
        Dict with ``indexes_created``, ``tables_analyzed``, and ``errors``.
    """
    from sqlalchemy import text

    result = {
        "indexes_created": 0,
        "tables_analyzed": 0,
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

    return result


async def optimize_neo4j(driver, *, database: str = "neo4j") -> dict:
    """Create optimal Neo4j indexes and constraints.

    Args:
        driver: An ``neo4j.AsyncDriver`` instance.
        database: Neo4j database name.

    Returns:
        Dict with ``indexes_created`` and ``errors``.
    """
    result = {
        "indexes_created": 0,
        "errors": [],
    }

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
