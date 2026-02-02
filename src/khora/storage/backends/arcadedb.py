"""ArcadeDB multi-model backend for graph and vector storage.

ArcadeDB is a multi-model database accessed via HTTP/REST API.
This backend implements both GraphBackendProtocol and VectorBackendProtocol,
allowing it to serve as a single storage engine for both roles.

Graph operations use Cypher via ArcadeDB's command endpoint.
Vector operations use ArcadeDB's native vector index.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.models import Chunk, ChunkMetadata, Entity, Episode, Relationship
from khora.core.models.entity import EntityType, RelationshipType
from khora.storage.backends.mixins import (
    GraphBackendBase,
    VectorBackendBase,
    deserialize_dict,
    parse_datetime,
    parse_uuid,
    parse_uuid_list,
    serialize_dict,
)


class ArcadeDBBackend(GraphBackendBase, VectorBackendBase):
    """ArcadeDB multi-model backend implementing both graph and vector protocols.

    Uses HTTP/REST API via httpx.AsyncClient. Graph queries use Cypher
    endpoint; vector operations use SQL with vector functions.
    """

    def __init__(
        self,
        url: str = "http://localhost:2480",
        *,
        database: str = "khora",
        user: str = "root",
        password: str = "",
        query_language: str = "cypher",
        embedding_dimension: int = 1536,
    ) -> None:
        self._url = url.rstrip("/")
        self._database = database
        self._user = user
        self._password = password
        self._query_language = query_language
        self._embedding_dimension = embedding_dimension
        self._client: Any = None  # httpx.AsyncClient

    @classmethod
    def from_config(cls, config: Any) -> ArcadeDBBackend:
        """Create an ArcadeDBBackend from an ArcadeDBGraphConfig or ArcadeDBVectorConfig."""
        return cls(
            url=config.url or "http://localhost:2480",
            database=getattr(config, "database", "khora"),
            user=getattr(config, "user", "root"),
            password=getattr(config, "password", ""),
            query_language=getattr(config, "query_language", "cypher"),
            embedding_dimension=getattr(config, "embedding_dimension", 1536),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._client is not None:
            return

        import httpx

        logger.info(f"Connecting to ArcadeDB at {self._url}...")
        self._client = httpx.AsyncClient(
            base_url=self._url,
            auth=(self._user, self._password) if self._user else None,
            timeout=30.0,
        )

        # Ensure database exists
        await self._ensure_database()
        # Create document types (schema)
        await self._create_schema()
        logger.info("Connected to ArcadeDB")

    async def disconnect(self) -> None:
        if self._client is not None:
            logger.info("Disconnecting from ArcadeDB...")
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from ArcadeDB")

    async def is_healthy(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = await self._client.get("/api/v1/server")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"ArcadeDB health check failed: {e}")
            return False

    def _get_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._client

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _command(self, language: str, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a command via ArcadeDB REST API."""
        client = self._get_client()
        payload: dict[str, Any] = {
            "language": language,
            "command": command,
        }
        if params:
            payload["params"] = params

        resp = await client.post(
            f"/api/v1/command/{self._database}",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def _cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return result rows."""
        result = await self._command("cypher", query, params)
        return result.get("result", [])

    async def _sql(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a SQL query and return result rows."""
        result = await self._command("sql", query, params)
        return result.get("result", [])

    async def _ensure_database(self) -> None:
        """Create the database if it doesn't exist."""
        client = self._get_client()
        try:
            resp = await client.get(f"/api/v1/exists/{self._database}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("result", False):
                    return
        except Exception:
            pass

        try:
            await client.post("/api/v1/server", json={"command": f"create database {self._database}"})
        except Exception as e:
            logger.debug(f"Database creation: {e}")

    async def _create_schema(self) -> None:
        """Create document types for entities, relationships, episodes, and chunks."""
        type_commands = [
            # Vertex types
            """
            CREATE VERTEX TYPE IF NOT EXISTS Entity (
                id STRING,
                namespace_id STRING,
                name STRING,
                entity_type STRING,
                description STRING,
                attributes STRING,
                source_document_ids LIST,
                source_chunk_ids LIST,
                mention_count INTEGER,
                valid_from STRING,
                valid_until STRING,
                confidence DOUBLE,
                metadata STRING,
                created_at STRING,
                updated_at STRING
            )
            """,
            """
            CREATE VERTEX TYPE IF NOT EXISTS Episode (
                id STRING,
                namespace_id STRING,
                name STRING,
                description STRING,
                occurred_at STRING,
                duration_seconds DOUBLE,
                entity_ids LIST,
                source_document_ids LIST,
                source_chunk_ids LIST,
                metadata STRING,
                created_at STRING,
                updated_at STRING
            )
            """,
            """
            CREATE VERTEX TYPE IF NOT EXISTS Chunk (
                id STRING,
                document_id STRING,
                namespace_id STRING,
                content STRING,
                chunk_index INTEGER,
                token_count INTEGER,
                embedding LIST,
                embedding_model STRING,
                metadata STRING,
                created_at STRING,
                updated_at STRING
            )
            """,
            # Edge types
            """
            CREATE EDGE TYPE IF NOT EXISTS RELATES_TO (
                id STRING,
                namespace_id STRING,
                relationship_type STRING,
                description STRING,
                properties STRING,
                source_document_ids LIST,
                source_chunk_ids LIST,
                valid_from STRING,
                valid_until STRING,
                confidence DOUBLE,
                weight DOUBLE,
                metadata STRING,
                created_at STRING,
                updated_at STRING
            )
            """,
            "CREATE EDGE TYPE IF NOT EXISTS INVOLVES",
        ]

        for cmd in type_commands:
            try:
                await self._sql(cmd.strip())
            except Exception as e:
                logger.debug(f"Schema creation: {e}")

        # Create indexes
        index_commands = [
            "CREATE INDEX IF NOT EXISTS ON Entity (id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Entity (namespace_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Episode (id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Episode (namespace_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Chunk (id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Chunk (document_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Chunk (namespace_id) NOTUNIQUE",
        ]
        for cmd in index_commands:
            try:
                await self._sql(cmd)
            except Exception as e:
                logger.debug(f"Index creation: {e}")

    # ------------------------------------------------------------------
    # Record conversion helpers
    # ------------------------------------------------------------------

    def _row_to_entity(self, row: dict[str, Any]) -> Entity:
        return Entity(
            id=parse_uuid(row["id"]),
            namespace_id=parse_uuid(row["namespace_id"]),
            name=row["name"],
            entity_type=(
                EntityType(row["entity_type"]) if row["entity_type"] in EntityType.__members__ else row["entity_type"]
            ),
            description=row.get("description", ""),
            attributes=deserialize_dict(row.get("attributes")),
            source_document_ids=parse_uuid_list(row.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(row.get("source_chunk_ids")),
            mention_count=row.get("mention_count", 1),
            valid_from=parse_datetime(row.get("valid_from")),
            valid_until=parse_datetime(row.get("valid_until")),
            confidence=row.get("confidence", 1.0),
            metadata=deserialize_dict(row.get("metadata")),
            created_at=parse_datetime(row.get("created_at"), default=datetime.now()) or datetime.now(),
            updated_at=parse_datetime(row.get("updated_at"), default=datetime.now()) or datetime.now(),
        )

    def _row_to_relationship(self, row: dict[str, Any], source_id: str, target_id: str) -> Relationship:
        rel_type = row.get("relationship_type", "CUSTOM")
        return Relationship(
            id=parse_uuid(row["id"]),
            namespace_id=parse_uuid(row["namespace_id"]),
            source_entity_id=parse_uuid(source_id),
            target_entity_id=parse_uuid(target_id),
            relationship_type=(RelationshipType(rel_type) if rel_type in RelationshipType.__members__ else rel_type),
            description=row.get("description", ""),
            properties=deserialize_dict(row.get("properties")),
            source_document_ids=parse_uuid_list(row.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(row.get("source_chunk_ids")),
            valid_from=parse_datetime(row.get("valid_from")),
            valid_until=parse_datetime(row.get("valid_until")),
            confidence=row.get("confidence", 1.0),
            weight=row.get("weight", 1.0),
            metadata=deserialize_dict(row.get("metadata")),
            created_at=parse_datetime(row.get("created_at"), default=datetime.now()) or datetime.now(),
            updated_at=parse_datetime(row.get("updated_at"), default=datetime.now()) or datetime.now(),
        )

    def _row_to_episode(self, row: dict[str, Any]) -> Episode:
        return Episode(
            id=parse_uuid(row["id"]),
            namespace_id=parse_uuid(row["namespace_id"]),
            name=row["name"],
            description=row.get("description", ""),
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            duration_seconds=row.get("duration_seconds"),
            entity_ids=parse_uuid_list(row.get("entity_ids")),
            source_document_ids=parse_uuid_list(row.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(row.get("source_chunk_ids")),
            metadata=deserialize_dict(row.get("metadata")),
            created_at=parse_datetime(row.get("created_at"), default=datetime.now()) or datetime.now(),
            updated_at=parse_datetime(row.get("updated_at"), default=datetime.now()) or datetime.now(),
        )

    def _row_to_chunk(self, row: dict[str, Any]) -> Chunk:
        return Chunk(
            id=parse_uuid(row["id"]),
            document_id=parse_uuid(row["document_id"]),
            namespace_id=parse_uuid(row["namespace_id"]),
            content=row.get("content", ""),
            chunk_index=row.get("chunk_index", 0),
            token_count=row.get("token_count", 0),
            embedding=row.get("embedding"),
            embedding_model=row.get("embedding_model", ""),
            metadata=ChunkMetadata(**deserialize_dict(row.get("metadata"))) if row.get("metadata") else ChunkMetadata(),
            created_at=parse_datetime(row.get("created_at"), default=datetime.now()) or datetime.now(),
            updated_at=parse_datetime(row.get("updated_at"), default=datetime.now()) or datetime.now(),
        )

    # ==================================================================
    # GraphBackendProtocol implementation
    # ==================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        entity_type_val = entity.entity_type.value if isinstance(entity.entity_type, EntityType) else entity.entity_type
        await self._cypher(
            """
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
            """,
            params={
                "id": str(entity.id),
                "namespace_id": str(entity.namespace_id),
                "name": entity.name,
                "entity_type": entity_type_val,
                "description": entity.description,
                "attributes": serialize_dict(entity.attributes),
                "source_document_ids": [str(d) for d in entity.source_document_ids],
                "source_chunk_ids": [str(c) for c in entity.source_chunk_ids],
                "mention_count": entity.mention_count,
                "valid_from": entity.valid_from.isoformat() if entity.valid_from else None,
                "valid_until": entity.valid_until.isoformat() if entity.valid_until else None,
                "confidence": entity.confidence,
                "metadata": serialize_dict(entity.metadata),
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
            },
        )
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        rows = await self._cypher(
            "MATCH (e:Entity {id: $id}) RETURN e",
            params={"id": str(entity_id)},
        )
        if not rows:
            return None
        node_data = rows[0].get("e", rows[0])
        return self._row_to_entity(node_data)

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        rows = await self._cypher(
            """
            MATCH (e:Entity {namespace_id: $ns, name: $name, entity_type: $et})
            RETURN e
            LIMIT 1
            """,
            params={"ns": str(namespace_id), "name": name, "et": entity_type},
        )
        if not rows:
            return None
        node_data = rows[0].get("e", rows[0])
        return self._row_to_entity(node_data)

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        if not entity_ids:
            return {}
        rows = await self._cypher(
            "MATCH (e:Entity) WHERE e.id IN $ids RETURN e",
            params={"ids": [str(eid) for eid in entity_ids]},
        )
        result = {}
        for row in rows:
            node_data = row.get("e", row)
            entity = self._row_to_entity(node_data)
            result[entity.id] = entity
        return result

    async def update_entity(self, entity: Entity) -> Entity:
        await self._cypher(
            """
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
            """,
            params={
                "id": str(entity.id),
                "name": entity.name,
                "description": entity.description,
                "attributes": serialize_dict(entity.attributes),
                "source_document_ids": [str(d) for d in entity.source_document_ids],
                "source_chunk_ids": [str(c) for c in entity.source_chunk_ids],
                "mention_count": entity.mention_count,
                "valid_from": entity.valid_from.isoformat() if entity.valid_from else None,
                "valid_until": entity.valid_until.isoformat() if entity.valid_until else None,
                "confidence": entity.confidence,
                "metadata": serialize_dict(entity.metadata),
                "updated_at": entity.updated_at.isoformat(),
            },
        )
        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        rows = await self._cypher(
            """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            RETURN count(e) as deleted
            """,
            params={"id": str(entity_id)},
        )
        if rows:
            return rows[0].get("deleted", 0) > 0
        return False

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        query = "MATCH (e:Entity {namespace_id: $ns})"
        params: dict[str, Any] = {"ns": str(namespace_id)}

        if entity_type:
            query += " WHERE e.entity_type = $et"
            params["et"] = entity_type

        query += " RETURN e ORDER BY e.name SKIP $offset LIMIT $limit"
        params["offset"] = offset
        params["limit"] = limit

        rows = await self._cypher(query, params=params)
        return [self._row_to_entity(row.get("e", row)) for row in rows]

    async def count_entities(self, namespace_id: UUID) -> int:
        rows = await self._cypher(
            "MATCH (e:Entity {namespace_id: $ns}) RETURN count(e) AS cnt",
            params={"ns": str(namespace_id)},
        )
        if rows:
            return rows[0].get("cnt", 0)
        return 0

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        rel_type = (
            relationship.relationship_type.value
            if isinstance(relationship.relationship_type, RelationshipType)
            else relationship.relationship_type
        )
        # ArcadeDB: use RELATES_TO edge type, store the logical type as a property
        await self._cypher(
            """
            MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATES_TO {
                id: $id,
                namespace_id: $namespace_id,
                relationship_type: $rel_type,
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
            }]->(target)
            """,
            params={
                "source_id": str(relationship.source_entity_id),
                "target_id": str(relationship.target_entity_id),
                "id": str(relationship.id),
                "namespace_id": str(relationship.namespace_id),
                "rel_type": rel_type,
                "description": relationship.description,
                "properties": serialize_dict(relationship.properties),
                "source_document_ids": [str(d) for d in relationship.source_document_ids],
                "source_chunk_ids": [str(c) for c in relationship.source_chunk_ids],
                "valid_from": relationship.valid_from.isoformat() if relationship.valid_from else None,
                "valid_until": relationship.valid_until.isoformat() if relationship.valid_until else None,
                "confidence": relationship.confidence,
                "weight": relationship.weight,
                "metadata": serialize_dict(relationship.metadata),
                "created_at": relationship.created_at.isoformat(),
                "updated_at": relationship.updated_at.isoformat(),
            },
        )
        return relationship

    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        rows = await self._cypher(
            """
            MATCH (source:Entity)-[r:RELATES_TO {id: $id}]->(target:Entity)
            RETURN r, source.id AS source_id, target.id AS target_id
            """,
            params={"id": str(relationship_id)},
        )
        if not rows:
            return None
        row = rows[0]
        rel_data = row.get("r", {})
        return self._row_to_relationship(rel_data, row["source_id"], row["target_id"])

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        rows = await self._cypher(
            """
            MATCH ()-[r:RELATES_TO {id: $id}]->()
            DELETE r
            RETURN count(r) AS deleted
            """,
            params={"id": str(relationship_id)},
        )
        if rows:
            return rows[0].get("deleted", 0) > 0
        return False

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        if direction == "outgoing":
            pattern = "(e:Entity {id: $eid})-[r:RELATES_TO]->(other:Entity)"
        elif direction == "incoming":
            pattern = "(other:Entity)-[r:RELATES_TO]->(e:Entity {id: $eid})"
        else:
            pattern = "(e:Entity {id: $eid})-[r:RELATES_TO]-(other:Entity)"

        query = f"""
        MATCH {pattern}
        RETURN r, e.id AS source_id, other.id AS target_id
        LIMIT $limit
        """

        rows = await self._cypher(query, params={"eid": str(entity_id), "limit": limit})
        rels = []
        for row in rows:
            rel_data = row.get("r", {})
            if relationship_types and rel_data.get("relationship_type") not in relationship_types:
                continue
            rels.append(self._row_to_relationship(rel_data, row["source_id"], row["target_id"]))
        return rels

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        query = "MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity) WHERE r.namespace_id = $ns"
        params: dict[str, Any] = {"ns": str(namespace_id), "offset": offset, "limit": limit}

        if relationship_type:
            query += " AND r.relationship_type = $rt"
            params["rt"] = relationship_type

        query += """
        RETURN r, source.id AS source_id, target.id AS target_id
        ORDER BY r.created_at DESC
        SKIP $offset LIMIT $limit
        """

        rows = await self._cypher(query, params=params)
        return [self._row_to_relationship(row.get("r", {}), row["source_id"], row["target_id"]) for row in rows]

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    async def create_episode(self, episode: Episode) -> Episode:
        await self._cypher(
            """
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
            """,
            params={
                "id": str(episode.id),
                "namespace_id": str(episode.namespace_id),
                "name": episode.name,
                "description": episode.description,
                "occurred_at": episode.occurred_at.isoformat(),
                "duration_seconds": episode.duration_seconds,
                "entity_ids": [str(e) for e in episode.entity_ids],
                "source_document_ids": [str(d) for d in episode.source_document_ids],
                "source_chunk_ids": [str(c) for c in episode.source_chunk_ids],
                "metadata": serialize_dict(episode.metadata),
                "created_at": episode.created_at.isoformat(),
                "updated_at": episode.updated_at.isoformat(),
            },
        )

        if episode.entity_ids:
            await self._cypher(
                """
                MATCH (ep:Episode {id: $ep_id}), (e:Entity)
                WHERE e.id IN $entity_ids
                CREATE (ep)-[:INVOLVES]->(e)
                """,
                params={
                    "ep_id": str(episode.id),
                    "entity_ids": [str(e) for e in episode.entity_ids],
                },
            )

        return episode

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        rows = await self._cypher(
            "MATCH (ep:Episode {id: $id}) RETURN ep",
            params={"id": str(episode_id)},
        )
        if not rows:
            return None
        return self._row_to_episode(rows[0].get("ep", rows[0]))

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        query = "MATCH (ep:Episode {namespace_id: $ns})"
        params: dict[str, Any] = {"ns": str(namespace_id), "limit": limit}
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

        rows = await self._cypher(query, params=params)
        return [self._row_to_episode(row.get("ep", row)) for row in rows]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def find_paths(
        self,
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        query = f"""
        MATCH path = (source:Entity {{id: $source_id}})-[r:RELATES_TO*1..{max_depth}]-(target:Entity {{id: $target_id}})
        WHERE source.namespace_id = $ns AND target.namespace_id = $ns
        RETURN path
        LIMIT 10
        """

        rows = await self._cypher(
            query,
            params={
                "source_id": str(source_entity_id),
                "target_id": str(target_entity_id),
                "ns": str(namespace_id),
            },
        )

        paths = []
        for row in rows:
            path = row.get("path", [])
            path_elements: list[dict[str, Any]] = []
            if isinstance(path, list):
                for element in path:
                    if isinstance(element, dict):
                        if "name" in element and "entity_type" in element:
                            path_elements.append({"type": "node", "data": element})
                        else:
                            path_elements.append({"type": "relationship", "data": element})
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
        query = f"""
        MATCH (center:Entity {{id: $eid}})-[r:RELATES_TO*1..{depth}]-(other:Entity)
        RETURN collect(DISTINCT other) AS nodes, collect(DISTINCT r) AS relationships
        LIMIT $limit
        """

        rows = await self._cypher(query, params={"eid": str(entity_id), "limit": limit})

        if not rows:
            return {"entities": [], "relationships": []}

        row = rows[0]
        nodes = row.get("nodes", [])
        raw_rels = row.get("relationships", [])
        relationships = []
        for rel in raw_rels:
            if isinstance(rel, list):
                relationships.extend(r for r in rel if r)
            elif rel:
                relationships.append(rel)

        if relationship_types:
            relationships = [r for r in relationships if r.get("relationship_type") in relationship_types]

        return {"entities": nodes, "relationships": relationships}

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        # Attributes stored as JSON string — can't do native key lookup
        # Use Cypher string contains for pre-filtering, then post-filter
        rows = await self._cypher(
            """
            MATCH (e:Entity {namespace_id: $ns})
            RETURN e
            LIMIT $limit
            """,
            params={"ns": str(namespace_id), "limit": limit * 5},
        )
        entities = []
        for row in rows:
            entity = self._row_to_entity(row.get("e", row))
            if entity.attributes.get(attribute_name) == attribute_value:
                entities.append(entity)
                if len(entities) >= limit:
                    break
        return entities

    # ==================================================================
    # VectorBackendProtocol implementation
    # ==================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        await self._sql(
            """
            INSERT INTO Chunk SET
                id = ?,
                document_id = ?,
                namespace_id = ?,
                content = ?,
                chunk_index = ?,
                token_count = ?,
                embedding = ?,
                embedding_model = ?,
                metadata = ?,
                created_at = ?,
                updated_at = ?
            """,
            params={
                "positionalParams": [
                    str(chunk.id),
                    str(chunk.document_id),
                    str(chunk.namespace_id),
                    chunk.content,
                    chunk.chunk_index,
                    chunk.token_count,
                    chunk.embedding,
                    chunk.embedding_model or "",
                    serialize_dict(chunk.metadata.model_dump() if hasattr(chunk.metadata, "model_dump") else {}),
                    chunk.created_at.isoformat(),
                    chunk.updated_at.isoformat(),
                ]
            },
        )
        return chunk

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            await self.create_chunk(chunk)
        return chunks

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        rows = await self._sql(
            "SELECT * FROM Chunk WHERE id = ?",
            params={"positionalParams": [str(chunk_id)]},
        )
        if not rows:
            return None
        return self._row_to_chunk(rows[0])

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        rows = await self._sql(
            "SELECT * FROM Chunk WHERE document_id = ? ORDER BY chunk_index",
            params={"positionalParams": [str(document_id)]},
        )
        return [self._row_to_chunk(row) for row in rows]

    async def delete_chunks_by_document(self, document_id: UUID) -> int:
        rows = await self._sql(
            "DELETE FROM Chunk WHERE document_id = ? RETURN count(*) AS cnt",
            params={"positionalParams": [str(document_id)]},
        )
        if rows:
            return rows[0].get("cnt", 0)
        return 0

    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        # ArcadeDB vector search via SQL
        # Note: actual syntax depends on ArcadeDB version; this is a reasonable approximation
        rows = await self._sql(
            """
            SELECT *, vectorDistance('cosine', embedding, ?) AS distance
            FROM Chunk
            WHERE namespace_id = ?
            ORDER BY distance ASC
            LIMIT ?
            """,
            params={
                "positionalParams": [query_embedding, str(namespace_id), limit * 2],
            },
        )

        results = []
        for row in rows:
            distance = row.get("distance", 1.0)
            similarity = 1.0 - distance  # cosine distance to similarity
            if similarity < min_similarity:
                continue
            if filter_document_ids:
                doc_id = row.get("document_id", "")
                try:
                    if UUID(doc_id) not in filter_document_ids:
                        continue
                except (ValueError, TypeError):
                    continue
            chunk = self._row_to_chunk(row)
            results.append((chunk, similarity))
            if len(results) >= limit:
                break

        return results

    # create_entity and update_entity are already implemented above
    # as part of the GraphBackendProtocol — they also satisfy the
    # VectorBackendProtocol's entity operations.

    async def entity_exists(self, entity_id: UUID) -> bool:
        entity = await self.get_entity(entity_id)
        return entity is not None

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        # Store entity embedding as a property
        await self._cypher(
            """
            MATCH (e:Entity {id: $id})
            SET e.embedding = $embedding, e.embedding_model = $model
            """,
            params={"id": str(entity_id), "embedding": embedding, "model": model},
        )

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        # Use SQL to search entity embeddings
        rows = await self._sql(
            """
            SELECT id, vectorDistance('cosine', embedding, ?) AS distance
            FROM Entity
            WHERE namespace_id = ? AND embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
            """,
            params={"positionalParams": [query_embedding, str(namespace_id), limit * 2]},
        )
        results = []
        for row in rows:
            distance = row.get("distance", 1.0)
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            results.append((parse_uuid(row["id"]), similarity))
            if len(results) >= limit:
                break
        return results

    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
    ) -> list[tuple[Chunk, float]]:
        # ArcadeDB fulltext search via SQL
        rows = await self._sql(
            """
            SELECT * FROM Chunk
            WHERE namespace_id = ? AND content CONTAINSTEXT ?
            LIMIT ?
            """,
            params={"positionalParams": [str(namespace_id), query_text, limit]},
        )
        # Return with uniform score since ArcadeDB CONTAINSTEXT doesn't return relevance
        return [(self._row_to_chunk(row), 1.0) for row in rows]

    async def count_chunks(self, namespace_id: UUID) -> int:
        rows = await self._sql(
            "SELECT count(*) AS cnt FROM Chunk WHERE namespace_id = ?",
            params={"positionalParams": [str(namespace_id)]},
        )
        if rows:
            return rows[0].get("cnt", 0)
        return 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        rows = await self._sql(
            "SELECT * FROM Chunk WHERE namespace_id = ? ORDER BY created_at SKIP ? LIMIT ?",
            params={"positionalParams": [str(namespace_id), offset, limit]},
        )
        return [self._row_to_chunk(row) for row in rows]
