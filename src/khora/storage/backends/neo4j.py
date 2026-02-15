"""Neo4j backend for knowledge graph storage.

Handles storage and traversal of entities, relationships, and episodes
in Neo4j graph database.
"""

from __future__ import annotations

import asyncio
import re as _re
from datetime import datetime
from typing import Any
from uuid import UUID

from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction

from khora.core.models import Entity, Episode, Relationship
from khora.core.models.entity import EntityType, RelationshipType
from khora.storage.backends.mixins import (
    GraphBackendBase,
)
from khora.storage.backends.mixins import deserialize_dict as _deserialize_dict
from khora.storage.backends.mixins import element_to_dict as _element_to_dict
from khora.storage.backends.mixins import serialize_dict as _serialize_dict

# Neo4j relationship labels must be valid identifiers: letters, digits, underscores.
# LLM-generated types like "at-risk" or "works for" need sanitizing.
_NEO4J_LABEL_RE = _re.compile(r"[^A-Za-z0-9_]")


def _sanitize_neo4j_label(label: str) -> str:
    """Sanitize a string for use as a Neo4j relationship type label."""
    sanitized = _NEO4J_LABEL_RE.sub("_", label.strip())
    return sanitized.upper() if sanitized else "RELATES_TO"


def _entity_to_cypher_params(entity: Entity) -> dict[str, Any]:
    """Convert Entity to Cypher-compatible parameter dict."""
    return {
        "id": str(entity.id),
        "namespace_id": str(entity.namespace_id),
        "name": entity.name,
        "entity_type": (entity.entity_type.value if isinstance(entity.entity_type, EntityType) else entity.entity_type),
        "description": entity.description,
        "attributes": _serialize_dict(entity.attributes),
        "source_document_ids": [str(d) for d in entity.source_document_ids],
        "source_chunk_ids": [str(c) for c in entity.source_chunk_ids],
        "mention_count": entity.mention_count,
        "valid_from": entity.valid_from.isoformat() if entity.valid_from else None,
        "valid_until": entity.valid_until.isoformat() if entity.valid_until else None,
        "confidence": entity.confidence,
        "metadata": _serialize_dict(entity.metadata),
        "created_at": entity.created_at.isoformat(),
        "updated_at": entity.updated_at.isoformat(),
    }


def _relationship_to_cypher_params(rel: Relationship) -> dict[str, Any]:
    """Convert Relationship to Cypher-compatible parameter dict."""
    return {
        "id": str(rel.id),
        "namespace_id": str(rel.namespace_id),
        "source_id": str(rel.source_entity_id),
        "target_id": str(rel.target_entity_id),
        "description": rel.description,
        "properties": _serialize_dict(rel.properties),
        "source_document_ids": [str(d) for d in rel.source_document_ids],
        "source_chunk_ids": [str(c) for c in rel.source_chunk_ids],
        "valid_from": rel.valid_from.isoformat() if rel.valid_from else None,
        "valid_until": rel.valid_until.isoformat() if rel.valid_until else None,
        "confidence": rel.confidence,
        "weight": rel.weight,
        "metadata": _serialize_dict(rel.metadata),
        "created_at": rel.created_at.isoformat(),
        "updated_at": rel.updated_at.isoformat(),
    }


class Neo4jBackend(GraphBackendBase):
    """Neo4j backend for knowledge graph operations.

    Stores entities as nodes and relationships as edges in Neo4j,
    enabling efficient graph traversal and pattern matching.
    """

    def __init__(
        self,
        url: str,
        *,
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ) -> None:
        """Initialize the Neo4j backend.

        Args:
            url: Neo4j connection URL (bolt:// or neo4j://)
            user: Database user
            password: Database password
            database: Database name
            max_connection_pool_size: Maximum connection pool size
        """
        self._url = url
        self._user = user
        self._password = password
        self._database = database
        self._max_connection_pool_size = max_connection_pool_size
        self._driver: AsyncDriver | None = None
        self._owns_driver: bool = True

    @classmethod
    def from_config(cls, config: Any) -> Neo4jBackend:
        """Create a Neo4jBackend from a Neo4jConfig object."""
        return cls(
            url=config.url or "",
            user=config.user,
            password=config.password,
            database=config.database,
        )

    @classmethod
    def from_driver(cls, driver: AsyncDriver, *, database: str = "neo4j") -> Neo4jBackend:
        """Create a Neo4jBackend from an existing AsyncDriver.

        The backend will NOT close the driver on disconnect, since
        it does not own it.

        Args:
            driver: An existing Neo4j async driver
            database: Database name

        Returns:
            Neo4jBackend wrapping the shared driver
        """
        instance = cls.__new__(cls)
        instance._url = ""
        instance._user = ""
        instance._password = ""
        instance._database = database
        instance._max_connection_pool_size = 0
        instance._driver = driver
        instance._owns_driver = False
        return instance

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is not None:
            # Already connected (either by connect() or from_driver())
            await self._create_indexes()
            return

        logger.info(f"Connecting to Neo4j at {self._url}...")
        self._driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._user, self._password),
            max_connection_pool_size=self._max_connection_pool_size,
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

        # Create indexes for performance
        await self._create_indexes()
        logger.info("Connected to Neo4j")

    async def disconnect(self) -> None:
        """Close Neo4j connections."""
        if self._driver is not None:
            if self._owns_driver:
                logger.info("Disconnecting from Neo4j...")
                await self._driver.close()
                logger.info("Disconnected from Neo4j")
            self._driver = None

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    async def _create_indexes(self) -> None:
        """Create indexes for common queries."""
        if self._driver is None:
            return

        indexes = [
            # Entity indexes
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_namespace IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            # Unique constraint on MERGE key (namespace_id, name, entity_type) —
            # prevents duplicate entities and implicitly creates the composite index.
            "CREATE CONSTRAINT entity_ns_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.namespace_id, e.name, e.entity_type) IS UNIQUE",
            # Composite: namespace + type (for list queries filtering by type without name)
            "CREATE INDEX entity_ns_type IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.entity_type)",
            # Entity source_tool (for source-aware queries)
            "CREATE INDEX entity_source_tool IF NOT EXISTS FOR (e:Entity) ON (e.source_tool)",
            # Entity confidence (for threshold filtering: min_entity_confidence)
            "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            # Episode indexes
            "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)",
            "CREATE INDEX episode_namespace IF NOT EXISTS FOR (ep:Episode) ON (ep.namespace_id)",
            "CREATE INDEX episode_occurred_at IF NOT EXISTS FOR (ep:Episode) ON (ep.occurred_at)",
        ]

        # Relationship property indexes require Neo4j ≥5.7 or Enterprise Edition
        rel_indexes = [
            "CREATE INDEX rel_namespace IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.namespace_id)",
            # namespace_id on high-volume relationship types
            "CREATE INDEX rel_collaborates_ns IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_associated_ns IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_depends_ns IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.namespace_id)",
            "CREATE INDEX rel_owns_ns IF NOT EXISTS FOR ()-[r:OWNS]-() ON (r.namespace_id)",
            "CREATE INDEX rel_works_for_ns IF NOT EXISTS FOR ()-[r:WORKS_FOR]-() ON (r.namespace_id)",
            "CREATE INDEX rel_implements_ns IF NOT EXISTS FOR ()-[r:IMPLEMENTS]-() ON (r.namespace_id)",
            "CREATE INDEX rel_part_of_ns IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.namespace_id)",
            # confidence on highest-volume relationship types
            "CREATE INDEX rel_collaborates_conf IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.confidence)",
            "CREATE INDEX rel_associated_conf IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.confidence)",
            "CREATE INDEX rel_depends_conf IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.confidence)",
        ]

        async with self._driver.session(database=self._database) as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")

            for index in rel_indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning(f"Relationship index creation skipped (may require Neo4j ≥5.7 or Enterprise): {e}")

    def _get_driver(self) -> AsyncDriver:
        """Get the Neo4j driver."""
        if self._driver is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._driver

    # =========================================================================
    # Entity operations
    # =========================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node in the graph."""
        driver = self._get_driver()
        params = _entity_to_cypher_params(entity)

        async def _create(tx: AsyncManagedTransaction) -> None:
            query = """
            CREATE (e:Entity {
                id: $id,
                namespace_id: $namespace_id,
                name: $name,
                entity_type: $entity_type,
                description: $description,
                attributes: $attributes,
                source_document_ids: $source_document_ids,
                source_chunk_ids: $source_chunk_ids,
                mention_count: $mention_count,
                valid_from: $valid_from,
                valid_until: $valid_until,
                confidence: $confidence,
                metadata: $metadata,
                created_at: $created_at,
                updated_at: $updated_at
            })
            """
            await tx.run(query, **params)

        async with driver.session(database=self._database) as session:
            await session.execute_write(_create)

        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        driver = self._get_driver()

        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=str(entity_id),
            )
            record = await result.single()
            if record:
                return self._record_to_entity(record["e"])
            return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type (for deduplication)."""
        driver = self._get_driver()

        async with driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (e:Entity {namespace_id: $namespace_id, name: $name, entity_type: $entity_type})
                RETURN e
                LIMIT 1
                """,
                namespace_id=str(namespace_id),
                name=name,
                entity_type=entity_type,
            )
            record = await result.single()
            if record:
                return self._record_to_entity(record["e"])
            return None

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query.

        Args:
            entity_ids: List of entity IDs to fetch

        Returns:
            Dictionary mapping entity ID to Entity object
        """
        if not entity_ids:
            return {}

        driver = self._get_driver()
        id_strings = [str(eid) for eid in entity_ids]

        async with driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids
                RETURN e
                """,
                ids=id_strings,
            )
            records = await result.data()
            return {UUID(r["e"]["id"]): self._record_to_entity(r["e"]) for r in records}

    async def update_entity(self, entity: Entity) -> Entity:
        """Update an entity."""
        driver = self._get_driver()
        params = _entity_to_cypher_params(entity)

        async def _update(tx: AsyncManagedTransaction) -> None:
            query = """
            MATCH (e:Entity {id: $id})
            SET e.name = $name,
                e.description = $description,
                e.attributes = $attributes,
                e.source_document_ids = $source_document_ids,
                e.source_chunk_ids = $source_chunk_ids,
                e.mention_count = $mention_count,
                e.valid_from = $valid_from,
                e.valid_until = $valid_until,
                e.confidence = $confidence,
                e.metadata = $metadata,
                e.updated_at = $updated_at
            """
            await tx.run(query, **params)

        async with driver.session(database=self._database) as session:
            await session.execute_write(_update)

        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity and its relationships."""
        driver = self._get_driver()

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                id=str(entity_id),
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with driver.session(database=self._database) as session:
            deleted = await session.execute_write(_delete)
            return deleted > 0

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace."""
        driver = self._get_driver()

        query = "MATCH (e:Entity {namespace_id: $namespace_id})"
        params: dict[str, Any] = {"namespace_id": str(namespace_id)}

        if entity_type:
            query += " WHERE e.entity_type = $entity_type"
            params["entity_type"] = entity_type

        query += " RETURN e ORDER BY e.name SKIP $offset LIMIT $limit"
        params["offset"] = offset
        params["limit"] = limit

        async with driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 50,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using UNWIND + MERGE.

        Matches on (namespace_id, name, entity_type).  Creates if new,
        updates if existing.  Returns (entity, is_new) tuples.
        """
        if not entities:
            return []

        driver = self._get_driver()

        _UPSERT_CYPHER = """
            UNWIND $rows AS row
            MERGE (e:Entity {namespace_id: row.namespace_id, name: row.name, entity_type: row.entity_type})
            ON CREATE SET
                e.id = row.id,
                e.description = row.description,
                e.attributes = row.attributes,
                e.source_document_ids = row.source_document_ids,
                e.source_chunk_ids = row.source_chunk_ids,
                e.mention_count = row.mention_count,
                e.valid_from = row.valid_from,
                e.valid_until = row.valid_until,
                e.confidence = row.confidence,
                e.metadata = row.metadata,
                e.created_at = row.created_at,
                e.updated_at = row.updated_at
            ON MATCH SET
                e.description = CASE WHEN size(row.description) > size(coalesce(e.description, ''))
                    THEN row.description ELSE e.description END,
                e.attributes = row.attributes,
                e.source_document_ids = e.source_document_ids + [x IN row.source_document_ids WHERE NOT x IN e.source_document_ids],
                e.source_chunk_ids = e.source_chunk_ids + [x IN row.source_chunk_ids WHERE NOT x IN e.source_chunk_ids],
                e.mention_count = e.mention_count + row.mention_count,
                e.confidence = CASE WHEN row.confidence > e.confidence THEN row.confidence ELSE e.confidence END,
                e.updated_at = row.updated_at
            RETURN e.id AS id, e.name AS name, row.id AS input_id,
                   CASE WHEN e.id = row.id THEN true ELSE false END AS is_new
        """

        results: list[tuple[Entity, bool]] = []

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]
            rows = [_entity_to_cypher_params(e) for e in batch]

            async def _upsert_tx(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
                result = await tx.run(_UPSERT_CYPHER, rows=rows)
                return await result.data()

            async with driver.session(database=self._database) as session:
                records = await session.execute_write(_upsert_tx)

            # Build result mapping - each input entity should get exactly one result
            input_id_to_entity = {str(e.id): e for e in batch}
            logger.debug(f"Neo4j batch: {len(batch)} entities, {len(records)} records returned")

            # De-duplicate: MERGE can match multiple nodes if duplicates exist
            # in the DB (no unique constraint on the MERGE key). Keep only the
            # first result per input entity.
            seen_input_ids: set[str] = set()
            for record in records:
                input_id = record["input_id"]
                if input_id in seen_input_ids:
                    continue
                seen_input_ids.add(input_id)
                entity = input_id_to_entity.get(input_id)
                if entity is not None:
                    neo4j_id = record["id"]
                    if neo4j_id != input_id:
                        entity.id = UUID(neo4j_id)
                        logger.debug(f"Entity '{entity.name}' ID synced: {input_id} -> {neo4j_id}")
                    results.append((entity, record["is_new"]))
                else:
                    logger.warning(
                        f"Neo4j returned input_id '{input_id}' not in batch. "
                        f"Batch IDs: {list(input_id_to_entity.keys())[:5]}..."
                    )

        logger.debug(f"Batch upserted {len(results)} entities ({sum(1 for _, n in results if n)} new)")
        return results

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 200,
    ) -> int:
        """Batch create relationships using UNWIND with parallel type processing.

        Relationships are grouped by type and each type group is processed
        in parallel using separate Neo4j sessions for better throughput.
        """
        if not relationships:
            return 0

        driver = self._get_driver()

        # Group by relationship type (required for dynamic rel type in Cypher)
        type_groups: dict[str, list[Relationship]] = {}
        for rel in relationships:
            rel_type = _sanitize_neo4j_label(
                rel.relationship_type.value
                if isinstance(rel.relationship_type, RelationshipType)
                else rel.relationship_type
            )
            type_groups.setdefault(rel_type, []).append(rel)

        async def _create_type_group(rel_type: str, rels: list[Relationship]) -> int:
            """Create all batches for a single relationship type sequentially."""
            type_total = 0
            for start in range(0, len(rels), batch_size):
                batch = rels[start : start + batch_size]
                rows = [_relationship_to_cypher_params(r) for r in batch]
                query = f"""
                UNWIND $rows AS row
                MATCH (source:Entity {{id: row.source_id}})
                MATCH (target:Entity {{id: row.target_id}})
                CREATE (source)-[r:{rel_type} {{
                    id: row.id,
                    namespace_id: row.namespace_id,
                    description: row.description,
                    properties: row.properties,
                    source_document_ids: row.source_document_ids,
                    source_chunk_ids: row.source_chunk_ids,
                    valid_from: row.valid_from,
                    valid_until: row.valid_until,
                    confidence: row.confidence,
                    weight: row.weight,
                    metadata: row.metadata,
                    created_at: row.created_at,
                    updated_at: row.updated_at
                }}]->(target)
                RETURN count(r) AS created
                """

                async def _tx(tx: AsyncManagedTransaction) -> int:
                    result = await tx.run(query, rows=rows)
                    record = await result.single()
                    return record["created"] if record else 0

                async with driver.session(database=self._database) as session:
                    type_total += await session.execute_write(_tx)
            return type_total

        # Type groups in parallel (different Cypher queries), batches sequential within each
        results = await asyncio.gather(*[_create_type_group(rel_type, rels) for rel_type, rels in type_groups.items()])
        total_created = sum(results)

        logger.debug(f"Batch created {total_created} relationships ({len(type_groups)} types in parallel)")
        return total_created

    def _record_to_entity(self, node: dict[str, Any]) -> Entity:
        """Convert a Neo4j node to a domain Entity."""
        return Entity(
            id=UUID(node["id"]),
            namespace_id=UUID(node["namespace_id"]),
            name=node["name"],
            entity_type=(
                EntityType(node["entity_type"])
                if node["entity_type"] in EntityType.__members__
                else node["entity_type"]
            ),
            description=node.get("description", ""),
            attributes=_deserialize_dict(node.get("attributes")),
            source_document_ids=[UUID(d) for d in node.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in node.get("source_chunk_ids", [])],
            mention_count=node.get("mention_count", 1),
            valid_from=datetime.fromisoformat(node["valid_from"]) if node.get("valid_from") else None,
            valid_until=datetime.fromisoformat(node["valid_until"]) if node.get("valid_until") else None,
            confidence=node.get("confidence", 1.0),
            metadata=_deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    # =========================================================================
    # Relationship operations
    # =========================================================================

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        driver = self._get_driver()
        params = _relationship_to_cypher_params(relationship)

        rel_type = _sanitize_neo4j_label(
            relationship.relationship_type.value
            if isinstance(relationship.relationship_type, RelationshipType)
            else relationship.relationship_type
        )

        async def _create(tx: AsyncManagedTransaction) -> None:
            # Use dynamic relationship type
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CREATE (source)-[r:{rel_type} {{
                id: $id,
                namespace_id: $namespace_id,
                description: $description,
                properties: $properties,
                source_document_ids: $source_document_ids,
                source_chunk_ids: $source_chunk_ids,
                valid_from: $valid_from,
                valid_until: $valid_until,
                confidence: $confidence,
                weight: $weight,
                metadata: $metadata,
                created_at: $created_at,
                updated_at: $updated_at
            }}]->(target)
            """
            await tx.run(query, **params)

        async with driver.session(database=self._database) as session:
            await session.execute_write(_create)

        return relationship

    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        """Get a relationship by ID."""
        driver = self._get_driver()

        async with driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (source:Entity)-[r {id: $id}]->(target:Entity)
                RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
                """,
                id=str(relationship_id),
            )
            record = await result.single()
            if record:
                return self._record_to_relationship(
                    record["r"],
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                )
            return None

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        """Delete a relationship."""
        driver = self._get_driver()

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                MATCH ()-[r {id: $id}]->()
                DELETE r
                RETURN count(r) as deleted
                """,
                id=str(relationship_id),
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with driver.session(database=self._database) as session:
            deleted = await session.execute_write(_delete)
            return deleted > 0

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        driver = self._get_driver()

        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # Build direction query
        if direction == "outgoing":
            pattern = f"(e)-[r{rel_filter}]->(other)"
        elif direction == "incoming":
            pattern = f"(other)-[r{rel_filter}]->(e)"
        else:  # both
            pattern = f"(e)-[r{rel_filter}]-(other)"

        query = f"""
        MATCH {pattern}
        WHERE e.id = $entity_id
        RETURN r, e.id as source_id, other.id as target_id, type(r) as rel_type
        LIMIT $limit
        """

        async with driver.session(database=self._database) as session:
            result = await session.run(query, entity_id=str(entity_id), limit=limit)
            records = await result.data()
            return [
                self._record_to_relationship(r["r"], r["source_id"], r["target_id"], r["rel_type"]) for r in records
            ]

    def _record_to_relationship(
        self, rel: dict[str, Any], source_id: str, target_id: str, rel_type: str
    ) -> Relationship:
        """Convert a Neo4j relationship to a domain Relationship."""
        return Relationship(
            id=UUID(rel["id"]),
            namespace_id=UUID(rel["namespace_id"]),
            source_entity_id=UUID(source_id),
            target_entity_id=UUID(target_id),
            relationship_type=(RelationshipType(rel_type) if rel_type in RelationshipType.__members__ else rel_type),
            description=rel.get("description", ""),
            properties=_deserialize_dict(rel.get("properties")),
            source_document_ids=[UUID(d) for d in rel.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in rel.get("source_chunk_ids", [])],
            valid_from=datetime.fromisoformat(rel["valid_from"]) if rel.get("valid_from") else None,
            valid_until=datetime.fromisoformat(rel["valid_until"]) if rel.get("valid_until") else None,
            confidence=rel.get("confidence", 1.0),
            weight=rel.get("weight", 1.0),
            metadata=_deserialize_dict(rel.get("metadata")),
            created_at=datetime.fromisoformat(rel["created_at"]) if rel.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(rel["updated_at"]) if rel.get("updated_at") else datetime.now(),
        )

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace."""
        driver = self._get_driver()

        # Build relationship type filter
        rel_filter = f":{_sanitize_neo4j_label(relationship_type)}" if relationship_type else ""

        query = f"""
        MATCH (source)-[r{rel_filter}]->(target)
        WHERE r.namespace_id = $namespace_id
        RETURN properties(r) as rel_props, source.id as source_id, target.id as target_id, type(r) as rel_type
        ORDER BY r.created_at DESC
        SKIP $offset
        LIMIT $limit
        """

        async with driver.session(database=self._database) as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                offset=offset,
                limit=limit,
            )
            records = await result.data()
            return [
                self._record_to_relationship(r["rel_props"], r["source_id"], r["target_id"], r["rel_type"])
                for r in records
            ]

    # =========================================================================
    # Episode operations
    # =========================================================================

    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode node."""
        driver = self._get_driver()

        async def _create(tx: AsyncManagedTransaction) -> None:
            query = """
            CREATE (ep:Episode {
                id: $id,
                namespace_id: $namespace_id,
                name: $name,
                description: $description,
                occurred_at: $occurred_at,
                duration_seconds: $duration_seconds,
                entity_ids: $entity_ids,
                source_document_ids: $source_document_ids,
                source_chunk_ids: $source_chunk_ids,
                metadata: $metadata,
                created_at: $created_at,
                updated_at: $updated_at
            })
            """
            await tx.run(
                query,
                id=str(episode.id),
                namespace_id=str(episode.namespace_id),
                name=episode.name,
                description=episode.description,
                occurred_at=episode.occurred_at.isoformat(),
                duration_seconds=episode.duration_seconds,
                entity_ids=[str(e) for e in episode.entity_ids],
                source_document_ids=[str(d) for d in episode.source_document_ids],
                source_chunk_ids=[str(c) for c in episode.source_chunk_ids],
                metadata=_serialize_dict(episode.metadata),
                created_at=episode.created_at.isoformat(),
                updated_at=episode.updated_at.isoformat(),
            )

            # Create links to entities
            if episode.entity_ids:
                link_query = """
                MATCH (ep:Episode {id: $episode_id})
                MATCH (e:Entity) WHERE e.id IN $entity_ids
                CREATE (ep)-[:INVOLVES]->(e)
                """
                await tx.run(
                    link_query,
                    episode_id=str(episode.id),
                    entity_ids=[str(e) for e in episode.entity_ids],
                )

        async with driver.session(database=self._database) as session:
            await session.execute_write(_create)

        return episode

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Get an episode by ID."""
        driver = self._get_driver()

        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (ep:Episode {id: $id}) RETURN ep",
                id=str(episode_id),
            )
            record = await result.single()
            if record:
                return self._record_to_episode(record["ep"])
            return None

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes in a time range."""
        driver = self._get_driver()

        query = "MATCH (ep:Episode {namespace_id: $namespace_id})"
        params: dict[str, Any] = {"namespace_id": str(namespace_id)}
        conditions = []

        if start_time:
            conditions.append("ep.occurred_at >= $start_time")
            params["start_time"] = start_time.isoformat()
        if end_time:
            conditions.append("ep.occurred_at <= $end_time")
            params["end_time"] = end_time.isoformat()

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " RETURN ep ORDER BY ep.occurred_at DESC LIMIT $limit"
        params["limit"] = limit

        async with driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_episode(r["ep"]) for r in records]

    def _record_to_episode(self, node: dict[str, Any]) -> Episode:
        """Convert a Neo4j node to a domain Episode."""
        return Episode(
            id=UUID(node["id"]),
            namespace_id=UUID(node["namespace_id"]),
            name=node["name"],
            description=node.get("description", ""),
            occurred_at=datetime.fromisoformat(node["occurred_at"]),
            duration_seconds=node.get("duration_seconds"),
            entity_ids=[UUID(e) for e in node.get("entity_ids", [])],
            source_document_ids=[UUID(d) for d in node.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in node.get("source_chunk_ids", [])],
            metadata=_deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    # =========================================================================
    # Graph traversal
    # =========================================================================

    async def find_paths(
        self,
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities."""
        driver = self._get_driver()

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id}})-[r{rel_filter}*1..{max_depth}]-(target:Entity {{id: $target_id}})
        )
        WHERE source.namespace_id = $namespace_id AND target.namespace_id = $namespace_id
        RETURN path
        LIMIT 10
        """

        async with driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=str(source_entity_id),
                target_id=str(target_entity_id),
                namespace_id=str(namespace_id),
            )
            records = await result.data()

            paths = []
            for record in records:
                path = record["path"]
                path_elements = []
                for element in path:
                    if hasattr(element, "items"):  # Node
                        path_elements.append({"type": "node", "data": _element_to_dict(element)})
                    else:  # Relationship
                        path_elements.append({"type": "relationship", "data": _element_to_dict(element)})
                paths.append(path_elements)

            return paths

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth."""
        driver = self._get_driver()

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        query = f"""
        MATCH (center:Entity {{id: $entity_id}})
        CALL apoc.path.subgraphAll(center, {{
            maxLevel: {depth},
            relationshipFilter: '{rel_filter.lstrip(":")}',
            limit: $limit
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        # Fallback query if APOC is not available
        fallback_query = f"""
        MATCH (center:Entity {{id: $entity_id}})-[r{rel_filter}*1..{depth}]-(other:Entity)
        RETURN collect(DISTINCT other) as nodes, collect(DISTINCT r) as relationships
        LIMIT $limit
        """

        async with driver.session(database=self._database) as session:
            try:
                result = await session.run(query, entity_id=str(entity_id), limit=limit)
                record = await result.single()
            except Exception:
                # Fallback if APOC not available
                result = await session.run(fallback_query, entity_id=str(entity_id), limit=limit)
                record = await result.single()

            if record:
                nodes = [_element_to_dict(n) for n in record.get("nodes", [])]
                relationships = [_element_to_dict(r) for r in record.get("relationships", [])]
                return {"entities": nodes, "relationships": relationships}

            return {"entities": [], "relationships": []}

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities in parallel.

        Args:
            entity_ids: List of entity IDs
            depth: Max traversal depth
            relationship_types: Optional relationship type filter
            limit_per_entity: Max nodes per entity neighborhood

        Returns:
            Dictionary mapping entity ID to neighborhood data
        """
        if not entity_ids:
            return {}

        driver = self._get_driver()
        id_strings = [str(eid) for eid in entity_ids]

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # Use UNWIND to process all entities in a single query
        query = f"""
        UNWIND $entity_ids AS eid
        MATCH (center:Entity {{id: eid}})
        OPTIONAL MATCH (center)-[r{rel_filter}*1..{depth}]-(other:Entity)
        WITH eid, center, collect(DISTINCT other)[0..$limit] as neighbors, collect(DISTINCT r)[0..$limit] as rels
        RETURN eid, neighbors, rels
        """

        async with driver.session(database=self._database) as session:
            result = await session.run(query, entity_ids=id_strings, limit=limit_per_entity)
            records = await result.data()

            neighborhoods = {}
            for record in records:
                eid = UUID(record["eid"])
                nodes = [_element_to_dict(n) for n in (record.get("neighbors") or []) if n]
                relationships = []
                for rel_list in record.get("rels") or []:
                    if rel_list:
                        for r in rel_list if isinstance(rel_list, list) else [rel_list]:
                            if r:
                                relationships.append(_element_to_dict(r))
                neighborhoods[eid] = {"entities": nodes, "relationships": relationships}

            return neighborhoods

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attribute value."""
        driver = self._get_driver()

        query = """
        MATCH (e:Entity {namespace_id: $namespace_id})
        WHERE e.attributes[$attribute_name] = $attribute_value
        RETURN e
        LIMIT $limit
        """

        async with driver.session(database=self._database) as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                attribute_name=attribute_name,
                attribute_value=attribute_value,
                limit=limit,
            )
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]
