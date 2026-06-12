"""Memgraph graph backend for knowledge graph storage.

Memgraph speaks the Bolt protocol and supports Cypher queries.
By default it uses the neo4j Python driver (Memgraph is bolt-compatible).
Key differences from Neo4j:
- No APOC procedures
- Different index syntax: CREATE INDEX ON :Label(property)
- No multi-database support
- In-memory by default
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.mixins import (
    GraphBackendBase,
    deserialize_dict,
    element_to_dict,
    sanitize_cypher_label,
    serialize_dict,
)

from .._log_safe import _safe_url_for_log


class MemgraphBackend(GraphBackendBase):
    """Memgraph graph backend using the neo4j Python driver over Bolt.

    Memgraph is an in-memory graph database that speaks Bolt protocol.
    This backend uses pure Cypher (no APOC) for maximum compatibility.
    """

    def __init__(
        self,
        url: str,
        *,
        user: str = "memgraph",
        password: str = "",
        max_connection_pool_size: int = 50,
    ) -> None:
        self._url = url
        self._user = user
        self._password = password
        self._max_connection_pool_size = max_connection_pool_size
        self._driver: Any = None  # neo4j.AsyncDriver

    @classmethod
    def from_config(cls, config: Any) -> MemgraphBackend:
        """Create a MemgraphBackend from a MemgraphConfig object.

        ``config.password`` and ``config.url`` are unwrapped from
        ``SecretStr`` here so the driver receives plaintext.
        """
        from pydantic import SecretStr

        password = config.password
        if isinstance(password, SecretStr):
            password = password.get_secret_value()
        url = config.url
        if isinstance(url, SecretStr):
            url = url.get_secret_value()
        return cls(
            url=url or "bolt://localhost:7687",
            user=config.user,
            password=password,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._driver is not None:
            return

        from neo4j import AsyncGraphDatabase

        logger.info("Connecting to Memgraph at {url}...", url=_safe_url_for_log(self._url))
        self._driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._user, self._password),
            max_connection_pool_size=self._max_connection_pool_size,
        )
        await self._driver.verify_connectivity()
        await self._create_indexes()
        logger.info("Connected to Memgraph")

    async def disconnect(self) -> None:
        if self._driver is not None:
            logger.info("Disconnecting from Memgraph...")
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Memgraph")

    async def is_healthy(self) -> bool:
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Memgraph health check failed: {e}")
            return False

    async def _create_indexes(self) -> None:
        """Create indexes using Memgraph's syntax."""
        if self._driver is None:
            return

        indexes = [
            "CREATE INDEX ON :Entity(id)",
            "CREATE INDEX ON :Entity(namespace_id)",
            "CREATE INDEX ON :Entity(name)",
            "CREATE INDEX ON :Entity(entity_type)",
            "CREATE INDEX ON :Episode(id)",
            "CREATE INDEX ON :Episode(namespace_id)",
            "CREATE INDEX ON :Episode(occurred_at)",
        ]

        async with self._driver.session() as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    # Memgraph may raise if index already exists
                    logger.debug(f"Index creation: {e}")

    def _get_driver(self) -> Any:
        if self._driver is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._driver

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_to_entity(self, node: dict[str, Any]) -> Entity:
        return Entity(
            id=UUID(node["id"]),
            namespace_id=UUID(node["namespace_id"]),
            name=node["name"],
            entity_type=node["entity_type"],
            description=node.get("description", ""),
            attributes=deserialize_dict(node.get("attributes")),
            source_document_ids=[UUID(d) for d in node.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in node.get("source_chunk_ids", [])],
            mention_count=node.get("mention_count", 1),
            valid_from=datetime.fromisoformat(node["valid_from"]) if node.get("valid_from") else None,
            valid_until=datetime.fromisoformat(node["valid_until"]) if node.get("valid_until") else None,
            confidence=node.get("confidence", 1.0),
            metadata=deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    def _record_to_relationship(
        self, rel: dict[str, Any], source_id: str, target_id: str, rel_type: str
    ) -> Relationship:
        return Relationship(
            id=UUID(rel["id"]),
            namespace_id=UUID(rel["namespace_id"]),
            source_entity_id=UUID(source_id),
            target_entity_id=UUID(target_id),
            relationship_type=rel_type,
            description=rel.get("description", ""),
            properties=deserialize_dict(rel.get("properties")),
            source_document_ids=[UUID(d) for d in rel.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in rel.get("source_chunk_ids", [])],
            valid_from=datetime.fromisoformat(rel["valid_from"]) if rel.get("valid_from") else None,
            valid_until=datetime.fromisoformat(rel["valid_until"]) if rel.get("valid_until") else None,
            confidence=rel.get("confidence", 1.0),
            weight=rel.get("weight", 1.0),
            metadata=deserialize_dict(rel.get("metadata")),
            created_at=datetime.fromisoformat(rel["created_at"]) if rel.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(rel["updated_at"]) if rel.get("updated_at") else datetime.now(),
        )

    def _record_to_episode(self, node: dict[str, Any]) -> Episode:
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
            metadata=deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> Entity:
        driver = self._get_driver()

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
        params = {
            "id": str(entity.id),
            "namespace_id": str(entity.namespace_id),
            "name": entity.name,
            "entity_type": entity.entity_type,
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
        }

        async with driver.session() as session:
            await session.run(query, **params)

        return entity

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id`` (IDOR family)."""
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $id, namespace_id: $namespace_id}) RETURN e",
                id=str(entity_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_entity(element_to_dict(record["e"]))
            return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        driver = self._get_driver()

        async with driver.session() as session:
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
                return self._record_to_entity(element_to_dict(record["e"]))
            return None

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities scoped to ``namespace_id`` (IDOR family).

        Entities in any other namespace are silently dropped from the result.
        """
        if not entity_ids:
            return {}

        driver = self._get_driver()
        id_strings = [str(eid) for eid in entity_ids]

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids AND e.namespace_id = $namespace_id
                RETURN e
                """,
                ids=id_strings,
                namespace_id=str(namespace_id),
            )
            records = await result.data()
            return {
                UUID(element_to_dict(r["e"])["id"]): self._record_to_entity(element_to_dict(r["e"])) for r in records
            }

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id`` (IDOR family).

        The ``namespace_id`` kwarg is defense-in-depth — asserted equal to
        ``entity.namespace_id`` before the MATCH filter is applied.
        """
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        driver = self._get_driver()

        query = """
        MATCH (e:Entity {id: $id, namespace_id: $namespace_id})
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

        async with driver.session() as session:
            await session.run(
                query,
                id=str(entity.id),
                namespace_id=str(namespace_id),
                name=entity.name,
                description=entity.description,
                attributes=serialize_dict(entity.attributes),
                source_document_ids=[str(d) for d in entity.source_document_ids],
                source_chunk_ids=[str(c) for c in entity.source_chunk_ids],
                mention_count=entity.mention_count,
                valid_from=entity.valid_from.isoformat() if entity.valid_from else None,
                valid_until=entity.valid_until.isoformat() if entity.valid_until else None,
                confidence=entity.confidence,
                metadata=serialize_dict(entity.metadata),
                updated_at=entity.updated_at.isoformat(),
            )

        return entity

    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id`` (IDOR family)."""
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id, namespace_id: $namespace_id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                id=str(entity_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            return (record["deleted"] if record else 0) > 0

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        driver = self._get_driver()

        query = "MATCH (e:Entity {namespace_id: $namespace_id})"
        params: dict[str, Any] = {"namespace_id": str(namespace_id)}

        if entity_type:
            query += " WHERE e.entity_type = $entity_type"
            params["entity_type"] = entity_type

        query += " RETURN e ORDER BY e.name SKIP $offset LIMIT $limit"
        params["offset"] = offset
        params["limit"] = limit

        async with driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_entity(element_to_dict(r["e"])) for r in records]

    async def count_entities(self, namespace_id: UUID) -> int:
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {namespace_id: $ns}) RETURN count(e) AS cnt",
                ns=str(namespace_id),
            )
            record = await result.single()
            return record["cnt"] if record else 0

    async def count_relationships(self, namespace_id: UUID) -> int:
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                "MATCH ()-[r]->() WHERE r.namespace_id = $ns RETURN count(r) AS cnt",
                ns=str(namespace_id),
            )
            record = await result.single()
            return record["cnt"] if record else 0

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using UNWIND + MERGE (issue #919).

        Matches on ``(namespace_id, name, entity_type)``: creates the node if
        new, updates it in place if it exists. Returns ``(entity, is_new)``
        tuples. Pure Cypher (no APOC); unlike the Neo4j backend this path does
        not write bi-temporal version snapshots, matching this backend's
        singular ``create_entity``.

        ``bulk_mode`` is accepted for signature parity with the coordinator /
        Neo4j backend; this backend's MERGE is already safe for new namespaces
        so the flag is a no-op.
        """
        if not entities:
            return []

        driver = self._get_driver()

        # MERGE keeps the upsert idempotent on (namespace_id, name,
        # entity_type). On create the supplied id is stored; on match the
        # existing id is kept, so ``e.id = row.id`` distinguishes new rows.
        query = """
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
            e.source_document_ids = e.source_document_ids +
                [x IN row.source_document_ids WHERE NOT x IN e.source_document_ids],
            e.source_chunk_ids = e.source_chunk_ids +
                [x IN row.source_chunk_ids WHERE NOT x IN e.source_chunk_ids],
            e.mention_count = e.mention_count + row.mention_count,
            e.confidence = CASE WHEN row.confidence > e.confidence THEN row.confidence ELSE e.confidence END,
            e.attributes = row.attributes,
            e.updated_at = row.updated_at
        RETURN row.id AS input_id, e.id AS stored_id
        """

        results: list[tuple[Entity, bool]] = []
        async with driver.session() as session:
            for start in range(0, len(entities), batch_size):
                batch = entities[start : start + batch_size]
                rows = [
                    {
                        "id": str(e.id),
                        "namespace_id": str(e.namespace_id),
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "description": e.description,
                        "attributes": serialize_dict(e.attributes),
                        "source_document_ids": [str(d) for d in e.source_document_ids],
                        "source_chunk_ids": [str(c) for c in e.source_chunk_ids],
                        "mention_count": e.mention_count,
                        "valid_from": e.valid_from.isoformat() if e.valid_from else None,
                        "valid_until": e.valid_until.isoformat() if e.valid_until else None,
                        "confidence": e.confidence,
                        "metadata": serialize_dict(e.metadata),
                        "created_at": e.created_at.isoformat(),
                        "updated_at": e.updated_at.isoformat(),
                    }
                    for e in batch
                ]
                result = await session.run(query, rows=rows)
                records = await result.data()
                # is_new when the stored id equals the id we tried to insert.
                # On a MERGE match the graph keeps the pre-existing node id,
                # so sync the caller's Entity to the canonical stored id
                # (#806 contract, issue #1150) - the subsequent relationship
                # batch resolves endpoints by ``entity.id`` and would
                # silently drop edges pointing at the extraction-time id.
                stored_by_input = {r["input_id"]: r["stored_id"] for r in records}
                for e in batch:
                    input_id = str(e.id)
                    stored_id = stored_by_input.get(input_id, input_id)
                    if stored_id != input_id:
                        e.id = UUID(stored_id)
                        logger.debug(f"Entity '{e.name}' ID synced: {input_id} -> {stored_id}")
                    results.append((e, stored_id == input_id))

        return results

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        driver = self._get_driver()

        rel_type = sanitize_cypher_label(relationship.relationship_type)
        # Mirror the sanitised type back so the caller's object matches
        # the persisted edge label, the same way Neo4j / sqlite_lance now
        # do (issue #749).
        relationship.relationship_type = rel_type

        # Dynamic relationship type via f-string (parameterized labels not
        # supported in Cypher). MERGE on (source, target, type, namespace_id)
        # keeps the edge idempotent so re-asserting the same relationship
        # updates the single existing edge in place rather than appending a
        # duplicate, mirroring the Neo4j backend (issue #921).
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type} {{namespace_id: $namespace_id}}]->(target)
        ON CREATE SET
            r.id = $id,
            r.description = $description,
            r.properties = $properties,
            r.source_document_ids = $source_document_ids,
            r.source_chunk_ids = $source_chunk_ids,
            r.valid_from = $valid_from,
            r.valid_until = $valid_until,
            r.confidence = $confidence,
            r.weight = $weight,
            r.metadata = $metadata,
            r.created_at = $created_at,
            r.updated_at = $updated_at
        ON MATCH SET
            r.description = CASE WHEN size($description) > size(coalesce(r.description, ''))
                THEN $description ELSE r.description END,
            r.source_document_ids = r.source_document_ids +
                [x IN $source_document_ids WHERE NOT x IN r.source_document_ids],
            r.source_chunk_ids = r.source_chunk_ids +
                [x IN $source_chunk_ids WHERE NOT x IN r.source_chunk_ids],
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
            r.updated_at = $updated_at
        """

        async with driver.session() as session:
            await session.run(
                query,
                source_id=str(relationship.source_entity_id),
                target_id=str(relationship.target_entity_id),
                id=str(relationship.id),
                namespace_id=str(relationship.namespace_id),
                description=relationship.description,
                properties=serialize_dict(relationship.properties),
                source_document_ids=[str(d) for d in relationship.source_document_ids],
                source_chunk_ids=[str(c) for c in relationship.source_chunk_ids],
                valid_from=relationship.valid_from.isoformat() if relationship.valid_from else None,
                valid_until=relationship.valid_until.isoformat() if relationship.valid_until else None,
                confidence=relationship.confidence,
                weight=relationship.weight,
                metadata=serialize_dict(relationship.metadata),
                created_at=relationship.created_at.isoformat(),
                updated_at=relationship.updated_at.isoformat(),
            )

        return relationship

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,
    ) -> int:
        """Batch create relationships using UNWIND + MERGE (issue #919).

        Relationships are grouped by (sanitised) type because the edge label
        cannot be parameterised in Cypher. Each group is MERGEd on
        ``(source, target, type, namespace_id)`` so re-asserting an edge stays
        idempotent (issue #921). Returns the number of input relationships
        written. Pure Cypher (no APOC).
        """
        if not relationships:
            return 0

        driver = self._get_driver()

        # Normalise types in place so the caller's objects match what is
        # persisted (issue #749), then group by label for the dynamic MERGE.
        type_groups: dict[str, list[Relationship]] = {}
        for rel in relationships:
            rel.relationship_type = sanitize_cypher_label(rel.relationship_type)
            type_groups.setdefault(rel.relationship_type, []).append(rel)

        total = 0
        async with driver.session() as session:
            for rel_type, rels in type_groups.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (source:Entity {{id: row.source_id}})
                MATCH (target:Entity {{id: row.target_id}})
                MERGE (source)-[r:{rel_type} {{namespace_id: row.namespace_id}}]->(target)
                ON CREATE SET
                    r.id = row.id,
                    r.description = row.description,
                    r.properties = row.properties,
                    r.source_document_ids = row.source_document_ids,
                    r.source_chunk_ids = row.source_chunk_ids,
                    r.valid_from = row.valid_from,
                    r.valid_until = row.valid_until,
                    r.confidence = row.confidence,
                    r.weight = row.weight,
                    r.metadata = row.metadata,
                    r.created_at = row.created_at,
                    r.updated_at = row.updated_at
                ON MATCH SET
                    r.description = CASE WHEN size(row.description) > size(coalesce(r.description, ''))
                        THEN row.description ELSE r.description END,
                    r.source_document_ids = r.source_document_ids +
                        [x IN row.source_document_ids WHERE NOT x IN r.source_document_ids],
                    r.source_chunk_ids = r.source_chunk_ids +
                        [x IN row.source_chunk_ids WHERE NOT x IN r.source_chunk_ids],
                    r.confidence = CASE WHEN row.confidence > r.confidence THEN row.confidence ELSE r.confidence END,
                    r.weight = CASE WHEN row.weight > r.weight THEN row.weight ELSE r.weight END,
                    r.updated_at = row.updated_at
                RETURN count(r) AS written
                """
                for start in range(0, len(rels), batch_size):
                    batch = rels[start : start + batch_size]
                    rows = [
                        {
                            "source_id": str(r.source_entity_id),
                            "target_id": str(r.target_entity_id),
                            "id": str(r.id),
                            "namespace_id": str(r.namespace_id),
                            "description": r.description,
                            "properties": serialize_dict(r.properties),
                            "source_document_ids": [str(d) for d in r.source_document_ids],
                            "source_chunk_ids": [str(c) for c in r.source_chunk_ids],
                            "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                            "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                            "confidence": r.confidence,
                            "weight": r.weight,
                            "metadata": serialize_dict(r.metadata),
                            "created_at": r.created_at.isoformat(),
                            "updated_at": r.updated_at.isoformat(),
                        }
                        for r in batch
                    ]
                    result = await session.run(query, rows=rows)
                    record = await result.single()
                    total += (record["written"] if record else 0) or 0

        return total

    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id`` (IDOR family)."""
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (source:Entity {namespace_id: $namespace_id})-[r {id: $id, namespace_id: $namespace_id}]->(target:Entity {namespace_id: $namespace_id})
                RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
                """,
                id=str(relationship_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_relationship(
                    element_to_dict(record["r"]),
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                )
            return None

    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id`` (IDOR family)."""
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH ()-[r {id: $id, namespace_id: $namespace_id}]->()
                DELETE r
                RETURN count(r) as deleted
                """,
                id=str(relationship_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            return (record["deleted"] if record else 0) > 0

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id`` (IDOR family).

        Both endpoint nodes are constrained to ``namespace_id`` so cross-tenant
        edges don't surface.
        """
        driver = self._get_driver()

        rel_filter = ""
        if relationship_types:
            sanitized = [sanitize_cypher_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        if direction == "outgoing":
            pattern = f"(e:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(other:Entity {{namespace_id: $namespace_id}})"
        elif direction == "incoming":
            pattern = f"(other:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(e:Entity {{namespace_id: $namespace_id}})"
        else:
            pattern = f"(e:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]-(other:Entity {{namespace_id: $namespace_id}})"

        query = f"""
        MATCH {pattern}
        WHERE e.id = $entity_id
        RETURN r, e.id as source_id, other.id as target_id, type(r) as rel_type
        LIMIT $limit
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                entity_id=str(entity_id),
                namespace_id=str(namespace_id),
                limit=limit,
            )
            records = await result.data()
            return [
                self._record_to_relationship(element_to_dict(r["r"]), r["source_id"], r["target_id"], r["rel_type"])
                for r in records
            ]

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        driver = self._get_driver()

        rel_filter = f":{sanitize_cypher_label(relationship_type)}" if relationship_type else ""

        query = f"""
        MATCH (source)-[r{rel_filter}]->(target)
        WHERE r.namespace_id = $namespace_id
        RETURN properties(r) as rel_props, source.id as source_id, target.id as target_id, type(r) as rel_type
        ORDER BY r.created_at DESC
        SKIP $offset
        LIMIT $limit
        """

        async with driver.session() as session:
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

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    async def create_episode(self, episode: Episode) -> Episode:
        driver = self._get_driver()

        async with driver.session() as session:
            await session.run(
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
                id=str(episode.id),
                namespace_id=str(episode.namespace_id),
                name=episode.name,
                description=episode.description,
                occurred_at=episode.occurred_at.isoformat(),
                duration_seconds=episode.duration_seconds,
                entity_ids=[str(e) for e in episode.entity_ids],
                source_document_ids=[str(d) for d in episode.source_document_ids],
                source_chunk_ids=[str(c) for c in episode.source_chunk_ids],
                metadata=serialize_dict(episode.metadata),
                created_at=episode.created_at.isoformat(),
                updated_at=episode.updated_at.isoformat(),
            )

            if episode.entity_ids:
                await session.run(
                    """
                    MATCH (ep:Episode {id: $episode_id})
                    MATCH (e:Entity) WHERE e.id IN $entity_ids
                    CREATE (ep)-[:INVOLVES]->(e)
                    """,
                    episode_id=str(episode.id),
                    entity_ids=[str(e) for e in episode.entity_ids],
                )

        return episode

    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id`` (IDOR family)."""
        driver = self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                "MATCH (ep:Episode {id: $id, namespace_id: $namespace_id}) RETURN ep",
                id=str(episode_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_episode(element_to_dict(record["ep"]))
            return None

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
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

        async with driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_episode(element_to_dict(r["ep"])) for r in records]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def find_paths(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        driver = self._get_driver()

        rel_filter = ""
        if relationship_types:
            sanitized = [sanitize_cypher_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        # Memgraph supports BFS shortest path. All nodes — endpoints and
        # intermediates — must share ``namespace_id`` so the traversal never
        # crosses tenants (IDOR family).
        query = f"""
        MATCH path = (source:Entity {{id: $source_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{max_depth}]-(target:Entity {{id: $target_id, namespace_id: $namespace_id}})
        WHERE ALL(n IN nodes(path) WHERE n.namespace_id = $namespace_id)
        RETURN path
        LIMIT 10
        """

        async with driver.session() as session:
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
                    data = element_to_dict(element)
                    if hasattr(element, "labels") or (isinstance(data, dict) and "id" in data and "name" in data):
                        path_elements.append({"type": "node", "data": data})
                    else:
                        path_elements.append({"type": "relationship", "data": data})
                paths.append(path_elements)

            return paths

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity, scoped to ``namespace_id`` (IDOR family).

        Seed and every node reached during traversal are constrained to
        ``namespace_id`` so the result never crosses into another namespace.
        """
        driver = self._get_driver()

        rel_filter = ""
        if relationship_types:
            sanitized = [sanitize_cypher_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        # Pure Cypher — no APOC needed. Center and every expanded node must
        # share ``namespace_id``. Each edge is collected as
        # ``{props, type}`` so the relationship type (stored only as the edge
        # label, never as a property) survives onto the returned dict as
        # ``relationship_type``, mirroring Neo4j (issue #922).
        query = f"""
        MATCH (center:Entity {{id: $entity_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: $namespace_id}})
        RETURN collect(DISTINCT other) as nodes,
               collect(DISTINCT [rel IN r | {{props: properties(rel), type: type(rel)}}]) as relationships
        LIMIT $limit
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                entity_id=str(entity_id),
                namespace_id=str(namespace_id),
                limit=limit,
            )
            record = await result.single()

            if record:
                nodes = [element_to_dict(n) for n in record.get("nodes", [])]
                relationships = []
                for rel_list in record.get("relationships", []):
                    if not rel_list:
                        continue
                    for r in rel_list if isinstance(rel_list, list) else [rel_list]:
                        if r:
                            relationships.append({**r.get("props", {}), "relationship_type": r.get("type")})
                return {"entities": nodes, "relationships": relationships}

            return {"entities": [], "relationships": []}

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        driver = self._get_driver()

        # Attributes stored as JSON string — search within
        query = """
        MATCH (e:Entity {namespace_id: $namespace_id})
        WHERE e.attributes[$attribute_name] = $attribute_value
        RETURN e
        LIMIT $limit
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                attribute_name=attribute_name,
                attribute_value=attribute_value,
                limit=limit,
            )
            records = await result.data()
            return [self._record_to_entity(element_to_dict(r["e"])) for r in records]
