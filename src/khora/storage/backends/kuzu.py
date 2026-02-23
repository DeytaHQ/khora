"""Kùzu embedded graph backend for knowledge graph storage.

Kùzu is an embedded graph database that supports Cypher queries.
All operations are synchronous and wrapped in asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.core.models.entity import EntityType, RelationshipType
from khora.storage.backends.mixins import (
    GraphBackendBase,
    deserialize_dict,
    parse_datetime,
    parse_uuid,
    parse_uuid_list,
    serialize_dict,
)


class KuzuBackend(GraphBackendBase):
    """Kùzu embedded graph backend.

    Uses an on-disk embedded database — no network needed.
    Ideal for single-process deployments, CI/CD testing, and edge devices.
    """

    def __init__(
        self,
        database_path: str = "./kuzu_db",
        *,
        read_only: bool = False,
    ) -> None:
        self._database_path = database_path
        self._read_only = read_only
        self._db: Any = None
        self._conn: Any = None

    @classmethod
    def from_config(cls, config: Any) -> KuzuBackend:
        """Create a KuzuBackend from a KuzuConfig object."""
        return cls(
            database_path=config.database_path,
            read_only=config.read_only,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._db is not None:
            return
        logger.info(f"Opening Kùzu database at {self._database_path}...")
        await asyncio.to_thread(self._open_database)
        await asyncio.to_thread(self._create_schema)
        logger.info("Kùzu database opened")

    def _open_database(self) -> None:
        import kuzu  # type: ignore[unresolved-import]

        self._db = kuzu.Database(self._database_path)
        self._conn = kuzu.Connection(self._db)

    def _create_schema(self) -> None:
        """Create node/relationship tables if they don't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                namespace_id STRING,
                name STRING,
                entity_type STRING,
                description STRING,
                attributes STRING,
                source_document_ids STRING[],
                source_chunk_ids STRING[],
                mention_count INT64,
                valid_from STRING,
                valid_until STRING,
                confidence DOUBLE,
                metadata STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY (id)
            )
            """)
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Episode(
                id STRING,
                namespace_id STRING,
                name STRING,
                description STRING,
                occurred_at STRING,
                duration_seconds DOUBLE,
                entity_ids STRING[],
                source_document_ids STRING[],
                source_chunk_ids STRING[],
                metadata STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY (id)
            )
            """)
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATES_TO(
                FROM Entity TO Entity,
                id STRING,
                namespace_id STRING,
                relationship_type STRING,
                description STRING,
                properties STRING,
                source_document_ids STRING[],
                source_chunk_ids STRING[],
                valid_from STRING,
                valid_until STRING,
                confidence DOUBLE,
                weight DOUBLE,
                metadata STRING,
                created_at STRING,
                updated_at STRING
            )
            """)
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS INVOLVES(
                FROM Episode TO Entity
            )
            """)

    async def disconnect(self) -> None:
        if self._conn is not None:
            logger.info("Closing Kùzu database...")
            self._conn = None
            self._db = None
            logger.info("Kùzu database closed")

    async def is_healthy(self) -> bool:
        if self._conn is None:
            return False
        try:
            await asyncio.to_thread(self._conn.execute, "RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Kùzu health check failed: {e}")
            return False

    def _get_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[list[Any]]:
        """Execute a Cypher query synchronously and return all rows."""
        conn = self._get_conn()
        result = conn.execute(query, parameters=parameters or {})
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    async def _arun_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[list[Any]]:
        """Execute a Cypher query asynchronously."""
        return await asyncio.to_thread(self._run_query, query, parameters)

    def _row_to_entity(self, row: dict[str, Any]) -> Entity:
        """Convert a result row dict to an Entity."""
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
        """Convert a result row dict to a Relationship."""
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
        """Convert a result row dict to an Episode."""
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

    def _entity_params(self, entity: Entity) -> dict[str, Any]:
        """Build parameter dict for entity creation/update."""
        return {
            "id": str(entity.id),
            "namespace_id": str(entity.namespace_id),
            "name": entity.name,
            "entity_type": (
                entity.entity_type.value if isinstance(entity.entity_type, EntityType) else entity.entity_type
            ),
            "description": entity.description,
            "attributes": serialize_dict(entity.attributes) or "{}",
            "source_document_ids": [str(d) for d in entity.source_document_ids],
            "source_chunk_ids": [str(c) for c in entity.source_chunk_ids],
            "mention_count": entity.mention_count,
            "valid_from": entity.valid_from.isoformat() if entity.valid_from else "",
            "valid_until": entity.valid_until.isoformat() if entity.valid_until else "",
            "confidence": entity.confidence,
            "metadata": serialize_dict(entity.metadata) or "{}",
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> Entity:
        params = self._entity_params(entity)
        await self._arun_query(
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
            parameters=params,
        )
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        rows = await self._arun_query(
            "MATCH (e:Entity {id: $id}) RETURN e.*",
            parameters={"id": str(entity_id)},
        )
        if not rows:
            return None
        # Kùzu returns columns as e.id, e.name, etc. — we need column names
        return await asyncio.to_thread(self._get_entity_by_id_sync, entity_id)

    def _get_entity_by_id_sync(self, entity_id: UUID) -> Entity | None:
        conn = self._get_conn()
        result = conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e.*",
            parameters={"id": str(entity_id)},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        columns = result.get_column_names()
        row_dict = {col.replace("e.", ""): val for col, val in zip(columns, row)}
        return self._row_to_entity(row_dict)

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        def _query() -> Entity | None:
            conn = self._get_conn()
            result = conn.execute(
                """
                MATCH (e:Entity {namespace_id: $ns, name: $name, entity_type: $et})
                RETURN e.*
                LIMIT 1
                """,
                parameters={"ns": str(namespace_id), "name": name, "et": entity_type},
            )
            if not result.has_next():
                return None
            row = result.get_next()
            columns = result.get_column_names()
            row_dict = {col.replace("e.", ""): val for col, val in zip(columns, row)}
            return self._row_to_entity(row_dict)

        return await asyncio.to_thread(_query)

    async def update_entity(self, entity: Entity) -> Entity:
        params = self._entity_params(entity)
        await self._arun_query(
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
            parameters=params,
        )
        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        def _delete() -> bool:
            conn = self._get_conn()
            # Delete relationships first, then entity
            conn.execute(
                "MATCH (e:Entity {id: $id})-[r]->() DELETE r",
                parameters={"id": str(entity_id)},
            )
            conn.execute(
                "MATCH ()-[r]->(e:Entity {id: $id}) DELETE r",
                parameters={"id": str(entity_id)},
            )
            conn.execute(
                "MATCH (e:Entity {id: $id}) DELETE e",
                parameters={"id": str(entity_id)},
            )
            return True

        try:
            return await asyncio.to_thread(_delete)
        except Exception:
            return False

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        def _query() -> list[Entity]:
            conn = self._get_conn()
            if entity_type:
                result = conn.execute(
                    """
                    MATCH (e:Entity)
                    WHERE e.namespace_id = $ns AND e.entity_type = $et
                    RETURN e.*
                    ORDER BY e.name
                    SKIP $offset LIMIT $limit
                    """,
                    parameters={
                        "ns": str(namespace_id),
                        "et": entity_type,
                        "offset": offset,
                        "limit": limit,
                    },
                )
            else:
                result = conn.execute(
                    """
                    MATCH (e:Entity)
                    WHERE e.namespace_id = $ns
                    RETURN e.*
                    ORDER BY e.name
                    SKIP $offset LIMIT $limit
                    """,
                    parameters={"ns": str(namespace_id), "offset": offset, "limit": limit},
                )
            entities = []
            columns = result.get_column_names()
            while result.has_next():
                row = result.get_next()
                row_dict = {col.replace("e.", ""): val for col, val in zip(columns, row)}
                entities.append(self._row_to_entity(row_dict))
            return entities

        return await asyncio.to_thread(_query)

    async def count_entities(self, namespace_id: UUID) -> int:
        def _query() -> int:
            conn = self._get_conn()
            result = conn.execute(
                "MATCH (e:Entity) WHERE e.namespace_id = $ns RETURN count(e)",
                parameters={"ns": str(namespace_id)},
            )
            if result.has_next():
                return result.get_next()[0]
            return 0

        return await asyncio.to_thread(_query)

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        rel_type = (
            relationship.relationship_type.value
            if isinstance(relationship.relationship_type, RelationshipType)
            else relationship.relationship_type
        )
        params = {
            "source_id": str(relationship.source_entity_id),
            "target_id": str(relationship.target_entity_id),
            "id": str(relationship.id),
            "namespace_id": str(relationship.namespace_id),
            "relationship_type": rel_type,
            "description": relationship.description,
            "properties": serialize_dict(relationship.properties) or "{}",
            "source_document_ids": [str(d) for d in relationship.source_document_ids],
            "source_chunk_ids": [str(c) for c in relationship.source_chunk_ids],
            "valid_from": relationship.valid_from.isoformat() if relationship.valid_from else "",
            "valid_until": relationship.valid_until.isoformat() if relationship.valid_until else "",
            "confidence": relationship.confidence,
            "weight": relationship.weight,
            "metadata": serialize_dict(relationship.metadata) or "{}",
            "created_at": relationship.created_at.isoformat(),
            "updated_at": relationship.updated_at.isoformat(),
        }
        # Kùzu uses a single relationship table RELATES_TO for all relationship types
        await self._arun_query(
            """
            MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATES_TO {
                id: $id,
                namespace_id: $namespace_id,
                relationship_type: $relationship_type,
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
            parameters=params,
        )
        return relationship

    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        def _query() -> Relationship | None:
            conn = self._get_conn()
            result = conn.execute(
                """
                MATCH (source:Entity)-[r:RELATES_TO {id: $id}]->(target:Entity)
                RETURN r.*, source.id, target.id
                """,
                parameters={"id": str(relationship_id)},
            )
            if not result.has_next():
                return None
            row = result.get_next()
            columns = result.get_column_names()
            row_dict = {}
            source_id = ""
            target_id = ""
            for col, val in zip(columns, row):
                if col == "source.id":
                    source_id = val
                elif col == "target.id":
                    target_id = val
                else:
                    row_dict[col.replace("r.", "")] = val
            return self._row_to_relationship(row_dict, source_id, target_id)

        return await asyncio.to_thread(_query)

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        try:
            await self._arun_query(
                """
                MATCH ()-[r:RELATES_TO {id: $id}]->()
                DELETE r
                """,
                parameters={"id": str(relationship_id)},
            )
            return True
        except Exception:
            return False

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        def _query() -> list[Relationship]:
            conn = self._get_conn()
            eid = str(entity_id)

            if direction == "outgoing":
                q = """
                MATCH (e:Entity {id: $eid})-[r:RELATES_TO]->(other:Entity)
                RETURN r.*, e.id AS source_id, other.id AS target_id
                LIMIT $limit
                """
            elif direction == "incoming":
                q = """
                MATCH (other:Entity)-[r:RELATES_TO]->(e:Entity {id: $eid})
                RETURN r.*, other.id AS source_id, e.id AS target_id
                LIMIT $limit
                """
            else:
                q = """
                MATCH (e:Entity {id: $eid})-[r:RELATES_TO]-(other:Entity)
                RETURN r.*, e.id AS source_id, other.id AS target_id
                LIMIT $limit
                """

            result = conn.execute(q, parameters={"eid": eid, "limit": limit})
            columns = result.get_column_names()
            rels = []
            while result.has_next():
                row = result.get_next()
                row_dict = {}
                source_id = ""
                target_id = ""
                for col, val in zip(columns, row):
                    if col == "source_id":
                        source_id = val
                    elif col == "target_id":
                        target_id = val
                    else:
                        row_dict[col.replace("r.", "")] = val
                if relationship_types and row_dict.get("relationship_type") not in relationship_types:
                    continue
                rels.append(self._row_to_relationship(row_dict, source_id, target_id))
            return rels

        return await asyncio.to_thread(_query)

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        def _query() -> list[Relationship]:
            conn = self._get_conn()
            params: dict[str, Any] = {"ns": str(namespace_id), "offset": offset, "limit": limit}

            if relationship_type:
                q = """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                WHERE r.namespace_id = $ns AND r.relationship_type = $rt
                RETURN r.*, source.id AS source_id, target.id AS target_id
                ORDER BY r.created_at DESC
                SKIP $offset LIMIT $limit
                """
                params["rt"] = relationship_type
            else:
                q = """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                WHERE r.namespace_id = $ns
                RETURN r.*, source.id AS source_id, target.id AS target_id
                ORDER BY r.created_at DESC
                SKIP $offset LIMIT $limit
                """

            result = conn.execute(q, parameters=params)
            columns = result.get_column_names()
            rels = []
            while result.has_next():
                row = result.get_next()
                row_dict = {}
                source_id = ""
                target_id = ""
                for col, val in zip(columns, row):
                    if col == "source_id":
                        source_id = val
                    elif col == "target_id":
                        target_id = val
                    else:
                        row_dict[col.replace("r.", "")] = val
                rels.append(self._row_to_relationship(row_dict, source_id, target_id))
            return rels

        return await asyncio.to_thread(_query)

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    async def create_episode(self, episode: Episode) -> Episode:
        params = {
            "id": str(episode.id),
            "namespace_id": str(episode.namespace_id),
            "name": episode.name,
            "description": episode.description,
            "occurred_at": episode.occurred_at.isoformat(),
            "duration_seconds": episode.duration_seconds or 0.0,
            "entity_ids": [str(e) for e in episode.entity_ids],
            "source_document_ids": [str(d) for d in episode.source_document_ids],
            "source_chunk_ids": [str(c) for c in episode.source_chunk_ids],
            "metadata": serialize_dict(episode.metadata) or "{}",
            "created_at": episode.created_at.isoformat(),
            "updated_at": episode.updated_at.isoformat(),
        }

        def _create() -> None:
            conn = self._get_conn()
            conn.execute(
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
                parameters=params,
            )
            # Link to entities
            if episode.entity_ids:
                conn.execute(
                    """
                    MATCH (ep:Episode {id: $ep_id}), (e:Entity)
                    WHERE e.id IN $entity_ids
                    CREATE (ep)-[:INVOLVES]->(e)
                    """,
                    parameters={
                        "ep_id": str(episode.id),
                        "entity_ids": [str(e) for e in episode.entity_ids],
                    },
                )

        await asyncio.to_thread(_create)
        return episode

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        def _query() -> Episode | None:
            conn = self._get_conn()
            result = conn.execute(
                "MATCH (ep:Episode {id: $id}) RETURN ep.*",
                parameters={"id": str(episode_id)},
            )
            if not result.has_next():
                return None
            row = result.get_next()
            columns = result.get_column_names()
            row_dict = {col.replace("ep.", ""): val for col, val in zip(columns, row)}
            return self._row_to_episode(row_dict)

        return await asyncio.to_thread(_query)

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        def _query() -> list[Episode]:
            conn = self._get_conn()
            conditions = ["ep.namespace_id = $ns"]
            params: dict[str, Any] = {"ns": str(namespace_id), "limit": limit}

            if start_time:
                conditions.append("ep.occurred_at >= $start_time")
                params["start_time"] = start_time.isoformat()
            if end_time:
                conditions.append("ep.occurred_at <= $end_time")
                params["end_time"] = end_time.isoformat()

            where = " AND ".join(conditions)
            q = f"""
            MATCH (ep:Episode)
            WHERE {where}
            RETURN ep.*
            ORDER BY ep.occurred_at DESC
            LIMIT $limit
            """

            result = conn.execute(q, parameters=params)
            columns = result.get_column_names()
            episodes = []
            while result.has_next():
                row = result.get_next()
                row_dict = {col.replace("ep.", ""): val for col, val in zip(columns, row)}
                episodes.append(self._row_to_episode(row_dict))
            return episodes

        return await asyncio.to_thread(_query)

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
        def _query() -> list[list[dict[str, Any]]]:
            conn = self._get_conn()
            # Kùzu supports variable-length paths
            q = f"""
            MATCH path = (source:Entity {{id: $source_id}})-[r:RELATES_TO*1..{max_depth}]-(target:Entity {{id: $target_id}})
            WHERE source.namespace_id = $ns AND target.namespace_id = $ns
            RETURN nodes(path), rels(path)
            LIMIT 10
            """
            result = conn.execute(
                q,
                parameters={
                    "source_id": str(source_entity_id),
                    "target_id": str(target_entity_id),
                    "ns": str(namespace_id),
                },
            )

            paths = []
            while result.has_next():
                row = result.get_next()
                nodes_data = row[0] if row[0] else []
                rels_data = row[1] if len(row) > 1 and row[1] else []

                path_elements: list[dict[str, Any]] = []
                for node in nodes_data:
                    data = dict(node) if hasattr(node, "items") else {"_raw": str(node)}
                    path_elements.append({"type": "node", "data": data})
                for rel in rels_data:
                    data = dict(rel) if hasattr(rel, "items") else {"_raw": str(rel)}
                    if relationship_types and data.get("relationship_type") not in relationship_types:
                        continue
                    path_elements.append({"type": "relationship", "data": data})
                paths.append(path_elements)

            return paths

        return await asyncio.to_thread(_query)

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        def _query() -> dict[str, Any]:
            conn = self._get_conn()
            q = f"""
            MATCH (center:Entity {{id: $eid}})-[r:RELATES_TO*1..{depth}]-(other:Entity)
            RETURN DISTINCT other.*, r
            LIMIT $limit
            """
            result = conn.execute(q, parameters={"eid": str(entity_id), "limit": limit})
            columns = result.get_column_names()

            nodes: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            while result.has_next():
                row = result.get_next()
                row_dict = {}
                for col, val in zip(columns, row):
                    if col == "r":
                        # Variable-length path returns list of relationships
                        if isinstance(val, list):
                            for rel in val:
                                rel_data = dict(rel) if hasattr(rel, "items") else {"_raw": str(rel)}
                                if relationship_types and rel_data.get("relationship_type") not in relationship_types:
                                    continue
                                relationships.append(rel_data)
                        elif val is not None:
                            rel_data = dict(val) if hasattr(val, "items") else {"_raw": str(val)}
                            relationships.append(rel_data)
                    else:
                        row_dict[col.replace("other.", "")] = val

                node_id = row_dict.get("id", "")
                if node_id and node_id not in seen_ids:
                    seen_ids.add(node_id)
                    nodes.append(row_dict)

            return {"entities": nodes, "relationships": relationships}

        return await asyncio.to_thread(_query)

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attribute value.

        Since Kùzu stores attributes as a JSON string, we search within
        the serialized string. For exact matching, deserialize and check.
        """

        def _query() -> list[Entity]:
            conn = self._get_conn()
            # Kùzu doesn't support JSON extraction natively — search in serialized string
            search_str = f'"{attribute_name}": ' if isinstance(attribute_value, str) else f'"{attribute_name}":'
            result = conn.execute(
                """
                MATCH (e:Entity)
                WHERE e.namespace_id = $ns AND contains(e.attributes, $search)
                RETURN e.*
                LIMIT $limit
                """,
                parameters={"ns": str(namespace_id), "search": search_str, "limit": limit},
            )
            columns = result.get_column_names()
            entities = []
            while result.has_next():
                row = result.get_next()
                row_dict = {col.replace("e.", ""): val for col, val in zip(columns, row)}
                entity = self._row_to_entity(row_dict)
                # Post-filter: check actual attribute value
                if entity.attributes.get(attribute_name) == attribute_value:
                    entities.append(entity)
            return entities

        return await asyncio.to_thread(_query)
