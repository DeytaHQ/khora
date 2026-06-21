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

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.dream.plan import OpKind
from khora.storage.backends.mixins import (
    GraphBackendBase,
    deserialize_dict,
    element_to_dict,
    parse_uuid_list,
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
            source_document_ids=parse_uuid_list(node.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(node.get("source_chunk_ids")),
            mention_count=node.get("mention_count", 1),
            valid_from=datetime.fromisoformat(node["valid_from"]) if node.get("valid_from") else None,
            valid_until=datetime.fromisoformat(node["valid_until"]) if node.get("valid_until") else None,
            confidence=node.get("confidence", 1.0),
            metadata=deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    def _record_to_relationship(
        self, rel: dict[str, Any], source_id: str | None, target_id: str | None, rel_type: str
    ) -> Relationship | None:
        """Convert a Memgraph relationship to a domain Relationship.

        Returns ``None`` when an endpoint id is null: a synthesized endpoint id
        would be a dangling FK, so the malformed row is skipped (#1238, porting
        the Neo4j #1237 guard). Missing ``id`` / ``namespace_id`` on the edge
        itself are synthesized (porting #767, which Memgraph never received).
        """
        if source_id is None or target_id is None:
            logger.warning(
                f"Dropping relationship with null endpoint id (type={rel_type}, "
                f"source_id={source_id}, target_id={target_id}); endpoint node is "
                "missing its id property."
            )
            return None
        rel_id = rel.get("id")
        rel_ns = rel.get("namespace_id")
        if rel_id is None or rel_ns is None:
            logger.warning(
                f"Relationship missing id/namespace_id (type={rel_type}, "
                f"{source_id}->{target_id}); using synthesized identity"
            )
        return Relationship(
            id=UUID(rel_id) if rel_id else uuid4(),
            namespace_id=UUID(rel_ns) if rel_ns else uuid4(),
            source_entity_id=UUID(source_id),
            target_entity_id=UUID(target_id),
            relationship_type=rel_type,
            description=rel.get("description", ""),
            properties=deserialize_dict(rel.get("properties")),
            source_document_ids=parse_uuid_list(rel.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(rel.get("source_chunk_ids")),
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
            entity_ids=parse_uuid_list(node.get("entity_ids")),
            source_document_ids=parse_uuid_list(node.get("source_document_ids")),
            source_chunk_ids=parse_uuid_list(node.get("source_chunk_ids")),
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
        """List entities in a namespace.

        Hides dream-retired nodes unconditionally (#1278): the flat soft-delete
        mirror stamps ``valid_until`` on the absorbed node, and recall filters it
        out in lockstep with the PG read filter so the two stores agree on the
        live set. Ported from the Neo4j read filter (#1272). ``valid_until`` is an
        ISO-8601 string, so compare lexicographically against an ISO ``$now``
        bind. A future ``valid_until`` is still a live temporal window;
        retirement stamps ``valid_until = retired_at`` (now), so it is hidden.
        """
        driver = self._get_driver()

        query = "MATCH (e:Entity {namespace_id: $namespace_id})"
        params: dict[str, Any] = {
            "namespace_id": str(namespace_id),
            "now": datetime.now(UTC).isoformat(),
        }

        conditions = ["(e.valid_until IS NULL OR e.valid_until > $now)"]
        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        query += " WHERE " + " AND ".join(conditions)

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
            rels = (
                self._record_to_relationship(element_to_dict(r["r"]), r["source_id"], r["target_id"], r["rel_type"])
                for r in records
            )
            return [rel for rel in rels if rel is not None]

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

        # Both endpoints constrained to in-namespace ``:Entity`` nodes, matching
        # get_relationship() / get_entity_relationships() (IDOR family): edges
        # with non-Entity or cross-namespace endpoints never surface (#1238).
        # Hides dream-pruned / merged-self-loop edges unconditionally (#1278): the
        # flat soft-delete mirror stamps ``valid_until`` on the edge (translating
        # PG's ``valid_to`` / ``invalidated_at``), and recall filters it out in
        # lockstep with the PG read filter. ``valid_until`` is an ISO-8601 string,
        # so compare lexicographically against an ISO ``$now`` bind.
        query = f"""
        MATCH (source:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(target:Entity {{namespace_id: $namespace_id}})
        WHERE r.namespace_id = $namespace_id
          AND (r.valid_until IS NULL OR r.valid_until > $now)
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
                now=datetime.now(UTC).isoformat(),
            )
            records = await result.data()
            rels = (
                self._record_to_relationship(r["rel_props"], r["source_id"], r["target_id"], r["rel_type"])
                for r in records
            )
            return [rel for rel in rels if rel is not None]

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
        # Slice inside the projection: a bare ``LIMIT $limit`` after
        # ``collect(...)`` aggregation is a no-op because aggregation already
        # produced a single row (#1154).
        query = f"""
        MATCH (center:Entity {{id: $entity_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: $namespace_id}})
        RETURN collect(DISTINCT other)[0..$limit] as nodes,
               collect(DISTINCT [rel IN r | {{props: properties(rel), type: type(rel)}}])[0..$limit] as relationships
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

        # ``attributes`` is persisted as a JSON *string* (``serialize_dict``),
        # so the old ``e.attributes[$attribute_name]`` map subscript never
        # matched (#1153). Prefilter server-side with a ``CONTAINS`` on the
        # serialized key, then deserialize each candidate and do the exact
        # key/value match in Python (correct for non-string values too).
        query = """
        MATCH (e:Entity {namespace_id: $namespace_id})
        WHERE e.attributes CONTAINS $key_pattern
        RETURN e
        """

        key_pattern = f'"{attribute_name}"'
        async with driver.session() as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                key_pattern=key_pattern,
            )
            records = await result.data()

        matches: list[Entity] = []
        for record in records:
            entity = self._record_to_entity(element_to_dict(record["e"]))
            if entity.attributes.get(attribute_name) == attribute_value:
                matches.append(entity)
                if len(matches) >= limit:
                    break
        return matches

    # ------------------------------------------------------------------
    # Dream flat soft-delete mirror verbs (#1278)
    # ------------------------------------------------------------------
    # Memgraph has no versioning primitives and no APOC: it stores ``valid_until``
    # as a plain string property (see ``create_entity`` / ``upsert_entities_batch``),
    # so the dream graph mirror is FLAT soft-delete only - SET ``valid_until`` by
    # id, endpoint rewrite = re-create + delete, relabel = re-create + delete.
    # There is NO :EntityVersion snapshot (unlike the Neo4j mirror, #1271/#1272);
    # the reverse verbs flat-restore by clearing ``valid_until``. All verbs are
    # namespace-scoped (IDOR family) and idempotent by id (replay is a no-op on
    # rows already in the target state). Empty batches return 0 (no Cypher).
    # ``valid_until`` is stamped as an ISO-8601 string so the lexicographic
    # ``list_*`` read filters hide the row in lockstep with the PG read filter.

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        """Memgraph flat-mirrors prune / dedupe / normalize-schema.

        - ``VECTORCYPHER_PRUNE_EDGES`` -> :meth:`soft_invalidate_relationships_batch`
        - ``VECTORCYPHER_DEDUPE_ENTITIES`` -> :meth:`soft_retire_entities_batch`
          (flat, no :EntityVersion) + :meth:`rewrite_relationship_endpoints_batch`
        - ``VECTORCYPHER_NORMALIZE_SCHEMA`` -> :meth:`rename_types_batch`

        ``VECTORCYPHER_COMMUNITY_SUMMARY`` is NOT advertised: community
        materialization (:Community nodes + read accessors) is the GraphRAG payoff
        and is out of scope for the flat soft-delete mirror.
        """
        return frozenset(
            {
                OpKind.VECTORCYPHER_PRUNE_EDGES,
                OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
                OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
            }
        )

    async def soft_invalidate_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
        invalidated_at: datetime,
    ) -> int:
        """Soft-delete edges by stamping ``valid_until`` (the column recall honors).

        Mirrors ``prune_edges``. Matched by relationship id within
        ``namespace_id`` (IDOR family); idempotent - only edges with a null
        ``valid_until`` are touched, so a replay is a no-op. Never deletes.

        Returns the number of edges actually invalidated.
        """
        if not relationship_ids:
            return 0
        driver = self._get_driver()
        ids = [str(r) for r in relationship_ids]
        ts = invalidated_at.isoformat()

        query = """
        UNWIND $relationship_ids AS rid
        MATCH ()-[rel {id: rid, namespace_id: $namespace_id}]-()
        WHERE rel.valid_until IS NULL
        SET rel.valid_until = $invalidated_at,
            rel.updated_at = $invalidated_at
        RETURN count(DISTINCT rel) AS invalidated
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                relationship_ids=ids,
                namespace_id=str(namespace_id),
                invalidated_at=ts,
            )
            record = await result.single()
            count = record["invalidated"] if record else 0
        logger.debug(f"Dream-invalidated {count} relationships in namespace {namespace_id}")
        return count

    async def soft_retire_entities_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        retired_at: datetime,
        reason: str = "dream_consolidated",
    ) -> int:
        """Flat soft-retire entities by id: SET ``valid_until`` (no :EntityVersion).

        Mirrors the absorbed-entity soft-delete in ``dedupe_entities``. Unlike the
        Neo4j mirror this backend has no versioning primitives, so it does NOT
        snapshot the live node into an :EntityVersion - it simply stamps
        ``valid_until`` so recall hides it. Scoped to ``namespace_id`` (IDOR
        family); idempotent - only entities with a null ``valid_until`` are
        retired, so a replay neither double-stamps nor re-touches. Never
        hard-deletes. ``reason`` is accepted for signature parity and recorded on
        the node so the soft-delete is auditable.

        Returns the number of entities actually retired.
        """
        if not entity_ids:
            return 0
        driver = self._get_driver()
        ids = [str(e) for e in entity_ids]
        ts = retired_at.isoformat()

        query = """
        UNWIND $entity_ids AS eid
        MATCH (current:Entity {id: eid, namespace_id: $namespace_id})
        WHERE current.valid_until IS NULL
        SET current.valid_until = $retired_at,
            current.updated_at = $retired_at,
            current.retirement_reason = $reason
        RETURN count(DISTINCT current) AS retired
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                entity_ids=ids,
                namespace_id=str(namespace_id),
                retired_at=ts,
                reason=reason,
            )
            record = await result.single()
            count = record["retired"] if record else 0
        logger.debug(f"Dream-retired {count} entities in namespace {namespace_id}")
        return count

    async def rewrite_relationship_endpoints_batch(
        self,
        rewrites: list[dict[str, Any]],
        *,
        namespace_id: UUID,
        rewritten_at: datetime,
    ) -> int:
        """Re-point relationship endpoints by id.

        Mirrors the absorbed-endpoint rewrite in ``dedupe_entities``. Memgraph
        cannot move an existing edge's endpoints in place, so each edge is
        re-created between the new endpoints with its properties preserved and the
        stale edge deleted - keyed by relationship id within ``namespace_id``
        (IDOR family). Idempotent: an edge already pointing at the new endpoints
        matches nothing to move.

        Each dict carries ``relationship_id``, ``source_entity_id``,
        ``target_entity_id`` (the post-rewrite endpoints), and
        ``relationship_type`` (the Cypher edge label). The type is a Cypher LABEL
        and CANNOT be ``$``-parameterized, so it is routed through
        ``sanitize_cypher_label`` and rewrites are grouped by sanitized label so
        each CREATE uses a static literal - no APOC dependency. Memgraph has no
        ``elementId()``, so the "skip when already on the new endpoints" guard
        compares the matched endpoints' ``id`` properties instead.

        Returns the number of edges actually re-pointed.
        """
        if not rewrites:
            return 0
        driver = self._get_driver()
        ts = rewritten_at.isoformat()

        # Group by sanitized relationship type so the new edge's label is a static
        # literal (labels cannot be $-parameterized).
        by_label: dict[str, list[dict[str, str]]] = {}
        for rw in rewrites:
            label = sanitize_cypher_label(str(rw["relationship_type"]))
            by_label.setdefault(label, []).append(
                {
                    "relationship_id": str(rw["relationship_id"]),
                    "source_entity_id": str(rw["source_entity_id"]),
                    "target_entity_id": str(rw["target_entity_id"]),
                }
            )

        total = 0
        async with driver.session() as session:
            # Sorted for deterministic lock ordering (deadlock avoidance).
            for label in sorted(by_label):
                rows = by_label[label]
                # Memgraph has no elementId(): guard on the endpoint id property so
                # an edge already on the target endpoints is left untouched.
                query = f"""
                UNWIND $rows AS r
                MATCH (oldSrc)-[rel:{label} {{id: r.relationship_id, namespace_id: $namespace_id}}]->(oldTgt)
                MATCH (newSrc:Entity {{id: r.source_entity_id, namespace_id: $namespace_id}})
                MATCH (newTgt:Entity {{id: r.target_entity_id, namespace_id: $namespace_id}})
                WHERE oldSrc.id <> newSrc.id OR oldTgt.id <> newTgt.id
                WITH rel, newSrc, newTgt, properties(rel) AS relProps
                CREATE (newSrc)-[newRel:{label}]->(newTgt)
                SET newRel = relProps,
                    newRel.updated_at = $rewritten_at
                DELETE rel
                RETURN count(newRel) AS rewritten
                """
                result = await session.run(
                    query,
                    rows=rows,
                    namespace_id=str(namespace_id),
                    rewritten_at=ts,
                )
                record = await result.single()
                total += (record["rewritten"] if record else 0) or 0
        logger.debug(f"Dream-rewrote endpoints on {total} relationships in namespace {namespace_id}")
        return total

    async def rename_types_batch(
        self,
        renames: list[dict[str, str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Relabel relationship types (Cypher edge labels) = re-create + delete.

        Mirrors ``normalize_schema``. The relationship type is a Cypher edge LABEL
        and CANNOT be ``$``-parameterized, so both ``old_type`` (the MATCH label)
        and ``new_type`` (the new label) are routed through
        ``sanitize_cypher_label`` (the Cypher-injection surface). One
        MATCH/CREATE/DELETE pass per distinct (old, new) pair so each label is a
        static, sanitized literal. Scoped to ``namespace_id``.

        Returns the total number of edges relabeled.
        """
        if not renames:
            return 0
        driver = self._get_driver()

        total = 0
        async with driver.session() as session:
            for rename in renames:
                old_label = sanitize_cypher_label(rename["old_type"])
                new_label = sanitize_cypher_label(rename["new_type"])
                if old_label == new_label:
                    continue
                ts = datetime.now(UTC).isoformat()
                # Labels are sanitized literals (NOT params). Endpoints / props
                # are preserved by re-creating the edge under the new label and
                # deleting the old one.
                query = f"""
                MATCH (s:Entity {{namespace_id: $namespace_id}})-[rel:{old_label} {{namespace_id: $namespace_id}}]->(t:Entity {{namespace_id: $namespace_id}})
                WITH s, t, rel, properties(rel) AS relProps
                CREATE (s)-[newRel:{new_label}]->(t)
                SET newRel = relProps,
                    newRel.updated_at = $updated_at
                DELETE rel
                RETURN count(newRel) AS renamed
                """
                result = await session.run(
                    query,
                    namespace_id=str(namespace_id),
                    updated_at=ts,
                )
                record = await result.single()
                total += (record["renamed"] if record else 0) or 0
        logger.debug(f"Dream-renamed {total} relationship-type edges in namespace {namespace_id}")
        return total

    # ------------------------------------------------------------------
    # Dream flat soft-delete mirror REVERSE verbs (#1278)
    # ------------------------------------------------------------------
    # ``dream_undo`` reverses the PG soft-deletes; these reverse the matching
    # forward flat mirror so undo restores PG and graph to identical pre-apply
    # live sets. Flat-restore = clear ``valid_until`` (no :EntityVersion snapshot
    # to delete, unlike the Neo4j reverse). Namespace-scoped and idempotent by id.

    async def restore_entities_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> int:
        """Un-retire entities by clearing ``valid_until`` (flat restore).

        Reverses :meth:`soft_retire_entities_batch`. There is no :EntityVersion
        snapshot to delete (this backend never created one). Matched by entity id
        within ``namespace_id`` (IDOR family); idempotent - only entities with a
        non-null ``valid_until`` transition. Returns the number restored.
        """
        if not entity_ids:
            return 0
        driver = self._get_driver()
        ids = [str(e) for e in entity_ids]
        ts = datetime.now(UTC).isoformat()

        query = """
        UNWIND $entity_ids AS eid
        MATCH (current:Entity {id: eid, namespace_id: $namespace_id})
        WHERE current.valid_until IS NOT NULL
        SET current.valid_until = NULL,
            current.updated_at = $updated_at,
            current.retirement_reason = NULL
        RETURN count(DISTINCT current) AS restored
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                entity_ids=ids,
                namespace_id=str(namespace_id),
                updated_at=ts,
            )
            record = await result.single()
            count = record["restored"] if record else 0
        logger.debug(f"Dream-restored {count} entities in namespace {namespace_id}")
        return count

    async def restore_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> int:
        """Un-invalidate relationships by clearing ``valid_until``.

        Reverses :meth:`soft_invalidate_relationships_batch`. Matched by
        relationship id within ``namespace_id`` (IDOR family); idempotent - only
        edges with a non-null ``valid_until`` are touched. Returns the number
        restored.
        """
        if not relationship_ids:
            return 0
        driver = self._get_driver()
        ids = [str(r) for r in relationship_ids]
        ts = datetime.now(UTC).isoformat()

        query = """
        UNWIND $relationship_ids AS rid
        MATCH ()-[rel {id: rid, namespace_id: $namespace_id}]-()
        WHERE rel.valid_until IS NOT NULL
        SET rel.valid_until = NULL,
            rel.updated_at = $updated_at
        RETURN count(DISTINCT rel) AS restored
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                relationship_ids=ids,
                namespace_id=str(namespace_id),
                updated_at=ts,
            )
            record = await result.single()
            count = record["restored"] if record else 0
        logger.debug(f"Dream-restored {count} relationships in namespace {namespace_id}")
        return count

    async def restore_relationship_endpoints_batch(
        self,
        rewrites: list[dict[str, Any]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Re-point relationship endpoints back to their pre-rewrite endpoints.

        Reverses :meth:`rewrite_relationship_endpoints_batch`. Same re-create +
        delete shape (Memgraph cannot move endpoints in place), keyed by
        relationship id within ``namespace_id`` (IDOR family). Each dict carries
        ``relationship_id``, ``source_entity_id``, ``target_entity_id`` (the
        PRE-rewrite endpoints to restore), and ``relationship_type`` (the Cypher
        edge label, routed through ``sanitize_cypher_label`` and grouped so each
        CREATE uses a static literal). Idempotent: an edge already on the old
        endpoints matches nothing to move. Returns the number re-pointed.
        """
        if not rewrites:
            return 0
        driver = self._get_driver()
        ts = datetime.now(UTC).isoformat()

        by_label: dict[str, list[dict[str, str]]] = {}
        for rw in rewrites:
            label = sanitize_cypher_label(str(rw["relationship_type"]))
            by_label.setdefault(label, []).append(
                {
                    "relationship_id": str(rw["relationship_id"]),
                    "source_entity_id": str(rw["source_entity_id"]),
                    "target_entity_id": str(rw["target_entity_id"]),
                }
            )

        total = 0
        async with driver.session() as session:
            for label in sorted(by_label):
                rows = by_label[label]
                query = f"""
                UNWIND $rows AS r
                MATCH (oldSrc)-[rel:{label} {{id: r.relationship_id, namespace_id: $namespace_id}}]->(oldTgt)
                MATCH (newSrc:Entity {{id: r.source_entity_id, namespace_id: $namespace_id}})
                MATCH (newTgt:Entity {{id: r.target_entity_id, namespace_id: $namespace_id}})
                WHERE oldSrc.id <> newSrc.id OR oldTgt.id <> newTgt.id
                WITH rel, newSrc, newTgt, properties(rel) AS relProps
                CREATE (newSrc)-[newRel:{label}]->(newTgt)
                SET newRel = relProps,
                    newRel.updated_at = $rewritten_at
                DELETE rel
                RETURN count(newRel) AS rewritten
                """
                result = await session.run(
                    query,
                    rows=rows,
                    namespace_id=str(namespace_id),
                    rewritten_at=ts,
                )
                record = await result.single()
                total += (record["rewritten"] if record else 0) or 0
        logger.debug(f"Dream-restored endpoints on {total} relationships in namespace {namespace_id}")
        return total
