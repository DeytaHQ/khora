"""Dual-node manager for HippoRAG 2 architecture in Neo4j.

Implements the dual-node structure where:
- (:Chunk) nodes represent text chunks with content and embeddings
- (:Entity) nodes represent extracted entities
- [:MENTIONED_IN] edges link entities to chunks where they appear
- [:AT_TIME] edges link chunks/entities to time hierarchy nodes

This structure enables efficient retrieval by:
1. Finding entry entities via vector similarity
2. Expanding to related entities via graph traversal
3. Retrieving chunks via MENTIONED_IN relationships
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.storage.backends.mixins import deserialize_dict, serialize_dict
from khora.telemetry import trace

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter


@dataclass
class ChunkNode:
    """Chunk node representation for Neo4j."""

    id: UUID
    namespace_id: UUID
    document_id: UUID
    content: str
    embedding: list[float] | None = None
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EntityChunkLink:
    """Link between entity and chunk."""

    entity_id: UUID
    chunk_id: UUID
    mention_count: int = 1
    context: str = ""


class DualNodeManager:
    """Manages HippoRAG 2 dual-node structure in Neo4j.

    Creates and maintains:
    - (:Chunk) nodes with content and embeddings
    - [:MENTIONED_IN] relationships from Entity to Chunk
    - [:AT_TIME] relationships to time hierarchy

    The dual-node structure enables efficient retrieval:
    - Vector search on Entity nodes for entry points
    - Graph expansion to find related entities
    - Chunk retrieval via MENTIONED_IN for context
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j"):
        """Initialize the manager.

        Args:
            driver: Neo4j async driver
            database: Database name
        """
        self._driver = driver
        self._database = database

    async def ensure_indexes(self) -> None:
        """Create indexes for Chunk and TimeNode nodes."""
        indexes = [
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX chunk_namespace IF NOT EXISTS FOR (c:Chunk) ON (c.namespace_id)",
            "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX chunk_occurred_at IF NOT EXISTS FOR (c:Chunk) ON (c.occurred_at)",
            # Composite for efficient namespace + time queries
            "CREATE INDEX chunk_ns_time IF NOT EXISTS FOR (c:Chunk) ON (c.namespace_id, c.occurred_at)",
            # Chunk filter indexes for structured queries
            "CREATE INDEX chunk_source_system IF NOT EXISTS FOR (c:Chunk) ON (c.source_system)",
            "CREATE INDEX chunk_author IF NOT EXISTS FOR (c:Chunk) ON (c.author)",
            "CREATE INDEX chunk_channel IF NOT EXISTS FOR (c:Chunk) ON (c.channel)",
            # TimeNode indexes for time hierarchy traversal
            "CREATE INDEX timenode_id IF NOT EXISTS FOR (t:TimeNode) ON (t.id)",
            "CREATE INDEX timenode_namespace IF NOT EXISTS FOR (t:TimeNode) ON (t.namespace_id)",
        ]

        async with self._driver.session(database=self._database) as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")

    async def create_chunk_node(self, chunk: TemporalChunk) -> UUID:
        """Create a single Chunk node in Neo4j.

        Args:
            chunk: Temporal chunk to create node for

        Returns:
            Chunk node ID
        """
        chunk_id = chunk.id or uuid4()

        query = """
        CREATE (c:Chunk {
            id: $id,
            namespace_id: $namespace_id,
            document_id: $document_id,
            content: $content,
            occurred_at: $occurred_at,
            created_at: $created_at,
            source_system: $source_system,
            author: $author,
            channel: $channel,
            confidence: $confidence,
            metadata: $metadata
        })
        RETURN c.id AS id
        """

        params = dict(
            id=str(chunk_id),
            namespace_id=str(chunk.namespace_id),
            document_id=str(chunk.document_id),
            content=chunk.content,
            occurred_at=chunk.occurred_at.isoformat() if chunk.occurred_at else None,
            created_at=chunk.created_at.isoformat() if chunk.created_at else datetime.now(UTC).isoformat(),
            source_system=chunk.source_system,
            author=chunk.author,
            channel=chunk.channel,
            confidence=chunk.confidence,
            metadata=serialize_dict(chunk.metadata or {}),
        )

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(query, **params)

            await session.execute_write(_work)

        logger.debug(f"Created Chunk node: {chunk_id}")
        return chunk_id

    @trace("khora.neo4j.create_chunk_nodes_batch")
    async def create_chunk_nodes_batch(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
    ) -> list[UUID]:
        """Create Chunk nodes in batch.

        Args:
            chunks: List of temporal chunks
            namespace_id: Namespace ID

        Returns:
            List of created chunk IDs
        """
        if not chunks:
            return []

        # Prepare batch data
        chunk_data = []
        chunk_ids = []

        for chunk in chunks:
            chunk_id = chunk.id or uuid4()
            chunk_ids.append(chunk_id)

            chunk_data.append(
                {
                    "id": str(chunk_id),
                    "namespace_id": str(namespace_id),
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "occurred_at": chunk.occurred_at.isoformat() if chunk.occurred_at else None,
                    "created_at": (chunk.created_at.isoformat() if chunk.created_at else datetime.now(UTC).isoformat()),
                    "source_system": chunk.source_system,
                    "author": chunk.author,
                    "channel": chunk.channel,
                    "confidence": chunk.confidence,
                    "metadata": serialize_dict(chunk.metadata or {}),
                }
            )

        query = """
        UNWIND $chunks AS chunk
        CREATE (c:Chunk {
            id: chunk.id,
            namespace_id: chunk.namespace_id,
            document_id: chunk.document_id,
            content: chunk.content,
            occurred_at: chunk.occurred_at,
            created_at: chunk.created_at,
            source_system: chunk.source_system,
            author: chunk.author,
            channel: chunk.channel,
            confidence: chunk.confidence,
            metadata: chunk.metadata
        })
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(query, chunks=chunk_data)

            await session.execute_write(_work)

        logger.debug(f"Created {len(chunk_ids)} Chunk nodes in batch")
        return chunk_ids

    async def link_entity_to_chunk(
        self,
        entity_id: UUID,
        chunk_id: UUID,
        *,
        mention_count: int = 1,
        context: str = "",
    ) -> None:
        """Create MENTIONED_IN relationship from Entity to Chunk.

        Args:
            entity_id: Entity node ID
            chunk_id: Chunk node ID
            mention_count: Number of times entity is mentioned in chunk
            context: Surrounding context of the mention
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        ON CREATE SET r.mention_count = $mention_count, r.context = $context
        ON MATCH SET r.mention_count = r.mention_count + $mention_count
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(
                    query,
                    entity_id=str(entity_id),
                    chunk_id=str(chunk_id),
                    mention_count=mention_count,
                    context=context,
                )

            await session.execute_write(_work)

    @trace("khora.neo4j.link_entities_to_chunks_batch")
    async def link_entities_to_chunks_batch(
        self,
        links: list[EntityChunkLink],
    ) -> None:
        """Create MENTIONED_IN relationships in batch.

        Args:
            links: List of EntityChunkLink objects
        """
        if not links:
            return

        link_data = [
            {
                "entity_id": str(link.entity_id),
                "chunk_id": str(link.chunk_id),
                "mention_count": link.mention_count,
                "context": link.context,
            }
            for link in links
        ]

        query = """
        UNWIND $links AS link
        MATCH (e:Entity {id: link.entity_id})
        MATCH (c:Chunk {id: link.chunk_id})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        ON CREATE SET r.mention_count = link.mention_count, r.context = link.context
        ON MATCH SET r.mention_count = r.mention_count + link.mention_count
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(query, links=link_data)

            await session.execute_write(_work)

        logger.debug(f"Created {len(links)} MENTIONED_IN relationships")

    async def link_chunk_to_time(
        self,
        chunk_id: UUID,
        time_node_id: UUID,
    ) -> None:
        """Create AT_TIME relationship from Chunk to TimeNode.

        Args:
            chunk_id: Chunk node ID
            time_node_id: TimeNode ID (usually a day node)
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (t:TimeNode {id: $time_node_id})
        MERGE (c)-[:AT_TIME]->(t)
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(
                    query,
                    chunk_id=str(chunk_id),
                    time_node_id=str(time_node_id),
                )

            await session.execute_write(_work)

    async def link_chunks_to_time_batch(
        self,
        chunk_time_links: list[tuple[UUID, UUID]],
    ) -> None:
        """Create AT_TIME relationships in batch.

        Args:
            chunk_time_links: List of (chunk_id, time_node_id) tuples
        """
        if not chunk_time_links:
            return

        link_data = [
            {"chunk_id": str(chunk_id), "time_node_id": str(time_id)} for chunk_id, time_id in chunk_time_links
        ]

        query = """
        UNWIND $links AS link
        MATCH (c:Chunk {id: link.chunk_id})
        MATCH (t:TimeNode {id: link.time_node_id})
        MERGE (c)-[:AT_TIME]->(t)
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                await tx.run(query, links=link_data)

            await session.execute_write(_work)

    @trace(
        "khora.neo4j.get_chunks_by_entities",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"chunk_count": len(r)},
    )
    async def get_chunks_by_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        *,
        temporal_filter: TemporalFilter | None = None,
        temporal_sort: bool = False,
        prefer_current: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get chunks connected to the given entities via MENTIONED_IN.

        Args:
            entity_ids: List of entity IDs to find chunks for
            namespace_id: Namespace to search within
            temporal_filter: Optional temporal constraints
            prefer_current: When True, prefer entities whose valid_until
                has not passed (for temporal queries). Entities without
                valid_until are kept (NULL = still valid).
            limit: Maximum chunks to return

        Returns:
            List of chunk dicts with entity connection info
        """
        if not entity_ids:
            return []

        # Build temporal filter conditions
        temporal_conditions = []
        params: dict[str, Any] = {
            "entity_ids": [str(eid) for eid in entity_ids],
            "namespace_id": str(namespace_id),
            "limit": limit,
        }

        if temporal_filter:
            if temporal_filter.occurred_after:
                temporal_conditions.append("c.occurred_at >= $occurred_after")
                params["occurred_after"] = temporal_filter.occurred_after.isoformat()
            if temporal_filter.occurred_before:
                temporal_conditions.append("c.occurred_at < $occurred_before")
                params["occurred_before"] = temporal_filter.occurred_before.isoformat()
            if temporal_filter.source_system:
                temporal_conditions.append("c.source_system = $source_system")
                params["source_system"] = temporal_filter.source_system
            if temporal_filter.author:
                temporal_conditions.append("c.author = $author")
                params["author"] = temporal_filter.author
            if temporal_filter.channel:
                temporal_conditions.append("c.channel = $channel")
                params["channel"] = temporal_filter.channel

        # For temporal queries, prefer entities whose validity hasn't expired
        if prefer_current:
            temporal_conditions.append("(e.valid_until IS NULL OR e.valid_until > datetime())")

        where_clause = ""
        if temporal_conditions:
            where_clause = "AND " + " AND ".join(temporal_conditions)

        order_clause = (
            "ORDER BY c.occurred_at DESC, total_mentions DESC" if temporal_sort else "ORDER BY total_mentions DESC"
        )

        query = f"""
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        WHERE e.id IN $entity_ids
        AND c.namespace_id = $namespace_id
        {where_clause}
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.occurred_at AS occurred_at,
               c.metadata AS metadata,
               collect(DISTINCT e.id) AS entity_ids,
               sum(r.mention_count) AS total_mentions
        {order_clause}
        LIMIT $limit
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(query, **params)
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        # Deserialize metadata from JSON string back to dict
        for record in records:
            if "metadata" in record:
                record["metadata"] = deserialize_dict(record["metadata"])

        return records

    @trace(
        "khora.neo4j.get_entity_neighborhoods",
        include={"entity_ids", "depth"},
        result=lambda r: {"result_count": len(r)},
    )
    async def get_entity_neighborhoods(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        *,
        depth: int = 2,
        limit_per_entity: int = 20,
        prefer_current: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get neighborhood of entities via relationship traversal.

        OPTIMIZATION: Uses a single Cypher query with UNWIND pattern to fetch
        all entity neighborhoods in one database round-trip. The query:
        1. Uses IN clause for batch entity matching (index-backed)
        2. Expands all neighborhoods in parallel within Neo4j
        3. Groups results by source entity
        4. Limits per-entity results to avoid explosion

        Args:
            entity_ids: Starting entity IDs
            namespace_id: Namespace constraint
            depth: Maximum traversal depth (1-4)
            limit_per_entity: Max related entities per starting entity
            prefer_current: When True, filter out entities whose valid_until
                has passed (for STATE_QUERY/RECENCY/CHANGE temporal categories).
                Entities without valid_until are kept (NULL = still valid).

        Returns:
            Dict mapping entity_id -> list of related entity info
        """
        if not entity_ids:
            return {}

        depth = min(max(1, depth), 4)  # Clamp to 1-4

        # When prefer_current is set, exclude entities whose validity has expired.
        # Entities with NULL valid_until are kept (NULL = no known end = still valid).
        temporal_clause = ""
        if prefer_current:
            temporal_clause = "AND (related.valid_until IS NULL OR related.valid_until > datetime())"

        # OPTIMIZATION: Single query fetches all neighborhoods in batch
        # Uses UNWIND internally via IN clause + collect() aggregation
        # This avoids N separate queries for N entity IDs
        query = f"""
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {{id: eid, namespace_id: $namespace_id}})
        OPTIONAL MATCH path = (e)-[*1..{depth}]-(related:Entity)
        WHERE related.namespace_id = $namespace_id
          AND related.id <> e.id
          {temporal_clause}
        WITH e, related,
             CASE WHEN related IS NOT NULL THEN length(path) ELSE null END AS distance
        ORDER BY e.id, distance
        With e, collect(DISTINCT CASE
            WHEN related IS NOT NULL THEN {{
                id: related.id,
                name: related.name,
                entity_type: related.entity_type,
                description: related.description,
                source_tool: related.source_tool,
                distance: distance
            }}
            ELSE null
        END)[0..$limit] AS related_raw
        RETURN e.id AS source_id,
               e.name AS source_name,
               e.entity_type AS source_entity_type,
               e.description AS source_description,
               e.source_tool AS source_source_tool,
               [x IN related_raw WHERE x IS NOT NULL] AS related_entities
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=[str(eid) for eid in entity_ids],
                    namespace_id=str(namespace_id),
                    limit=limit_per_entity,
                )
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        return {record["source_id"]: record["related_entities"] for record in records}

    @trace(
        "khora.neo4j.get_relationships_between",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"relationship_count": len(r)},
    )
    async def get_relationships_between(
        self,
        entity_ids: list[str],
        namespace_id: str,
        *,
        limit: int = 90,
    ) -> list[dict[str, Any]]:
        """Get relationships between a set of entities.

        Finds all directed relationships where both source and target
        are in the given entity_ids set, within the same namespace.

        Args:
            entity_ids: Entity IDs (strings) to find relationships between
            namespace_id: Namespace constraint (string)
            limit: Maximum number of relationships to return

        Returns:
            List of relationship dicts with id, source_entity_id,
            target_entity_id, relationship_type, description, confidence, weight,
            source_document_ids, source_chunk_ids
        """
        if len(entity_ids) < 2:
            return []

        query = """
        UNWIND $entity_ids AS sid
        MATCH (source:Entity {id: sid, namespace_id: $namespace_id})-[r]->(target:Entity)
        WHERE target.namespace_id = $namespace_id
          AND target.id IN $entity_ids
          AND source.id <> target.id
        RETURN DISTINCT r.id AS id, source.id AS source_entity_id, target.id AS target_entity_id,
               type(r) AS relationship_type, r.description AS description,
               r.confidence AS confidence, r.weight AS weight,
               r.source_document_ids AS source_document_ids,
               r.source_chunk_ids AS source_chunk_ids
        LIMIT $limit
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=entity_ids,
                    namespace_id=namespace_id,
                    limit=limit,
                )
                return [record.data() async for record in result]

            return await session.execute_read(_work)

    async def get_temporal_chunks(
        self,
        namespace_id: UUID,
        entity_ids: list[UUID],
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get chunks connected to entities via MENTIONED_IN within a time range.

        Matches (:Entity)-[:MENTIONED_IN]->(:Chunk) and filters chunks by
        occurred_at or created_at falling within the given time window.

        Args:
            namespace_id: Namespace to restrict results to
            entity_ids: Entity IDs whose connected chunks to retrieve
            after: Include only chunks with occurred_at/created_at >= this value
            before: Include only chunks with occurred_at/created_at <= this value
            limit: Maximum chunks to return

        Returns:
            List of chunk property dicts with entity connection info
        """
        if not entity_ids:
            return []

        params: dict[str, Any] = {
            "entity_ids": [str(eid) for eid in entity_ids],
            "namespace_id": str(namespace_id),
            "limit": limit,
        }

        conditions: list[str] = []
        if after is not None:
            conditions.append(
                "(coalesce(c.occurred_at, c.created_at) IS NULL OR " "coalesce(c.occurred_at, c.created_at) >= $after)"
            )
            params["after"] = after.isoformat()
        if before is not None:
            conditions.append(
                "(coalesce(c.occurred_at, c.created_at) IS NULL OR " "coalesce(c.occurred_at, c.created_at) <= $before)"
            )
            params["before"] = before.isoformat()

        where_extra = ""
        if conditions:
            where_extra = "\nAND " + "\nAND ".join(conditions)

        query = f"""
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        WHERE e.id IN $entity_ids
        AND c.namespace_id = $namespace_id{where_extra}
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.occurred_at AS occurred_at,
               c.created_at AS created_at,
               c.metadata AS metadata,
               collect(DISTINCT e.id) AS entity_ids,
               sum(r.mention_count) AS total_mentions
        ORDER BY coalesce(c.occurred_at, c.created_at) DESC
        LIMIT $limit
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(query, **params)
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        for record in records:
            if "metadata" in record:
                record["metadata"] = deserialize_dict(record["metadata"])

        return records

    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        namespace_id: UUID,
    ) -> int:
        """Delete all Chunk nodes for a document.

        Also removes MENTIONED_IN and AT_TIME relationships.

        Args:
            document_id: Document ID
            namespace_id: Namespace ID

        Returns:
            Number of chunks deleted
        """
        query = """
        MATCH (c:Chunk {document_id: $document_id, namespace_id: $namespace_id})
        DETACH DELETE c
        RETURN count(c) AS deleted
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    document_id=str(document_id),
                    namespace_id=str(namespace_id),
                )
                record = await result.single()
                return record["deleted"] if record else 0

            deleted = await session.execute_write(_work)
        logger.debug(f"Deleted {deleted} Chunk nodes for document {document_id}")
        return deleted

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count Chunk nodes in a namespace.

        Args:
            namespace_id: Namespace ID

        Returns:
            Number of chunks
        """
        query = """
        MATCH (c:Chunk {namespace_id: $namespace_id})
        RETURN count(c) AS count
        """

        async with self._driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(query, namespace_id=str(namespace_id))
                record = await result.single()
                return record["count"] if record else 0

            return await session.execute_read(_work)


__all__ = [
    "ChunkNode",
    "DualNodeManager",
    "EntityChunkLink",
]
