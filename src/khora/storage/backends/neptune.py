"""AWS Neptune graph backend for knowledge graph storage.

Neptune speaks the Bolt protocol and supports openCypher queries.
It uses the neo4j Python driver (Neptune is bolt-compatible on port 8182).
Key differences from Neo4j:
- No APOC procedures
- No multi-database support (single graph per cluster)
- Auto-indexes all properties — no DDL needed
- Mandatory TLS (encrypted=True)
- Supports AWS IAM SigV4 authentication
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.mixins import (
    GraphBackendBase,
    deserialize_dict,
    element_to_dict,
    parse_uuid_list,
    sanitize_cypher_label,
    serialize_dict,
)

from .._log_safe import _safe_url_for_log

# SigV4 signatures are valid for at most 5 minutes. Refresh the cached auth
# token well before that so a connection opened near the expiry edge never
# presents a stale signature (#1152).
_IAM_TOKEN_TTL_SECONDS = 240.0


class NeptuneBackend(GraphBackendBase):
    """AWS Neptune graph backend using the neo4j Python driver over Bolt.

    Neptune is a managed graph database on AWS that speaks Bolt protocol
    with openCypher support. This backend uses pure Cypher (no APOC)
    for maximum compatibility.
    """

    def __init__(
        self,
        url: str,
        *,
        user: str = "",
        password: str = "",
        encrypted: bool = True,
        iam_auth: bool = False,
        aws_region: str = "us-east-1",
        max_connection_pool_size: int = 100,
    ) -> None:
        self._url = url
        self._user = user
        self._password = password
        self._encrypted = encrypted
        self._iam_auth = iam_auth
        self._aws_region = aws_region
        self._max_connection_pool_size = max_connection_pool_size
        self._driver: Any = None  # neo4j.AsyncDriver
        self._boto_session: Any = None  # boto3.Session, set on connect() when iam_auth

    @classmethod
    def from_config(cls, config: Any) -> NeptuneBackend:
        """Create a NeptuneBackend from a NeptuneConfig object.

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
            url=url or "bolt://localhost:8182",
            user=config.user,
            password=password,
            iam_auth=config.iam_auth,
            aws_region=config.aws_region,
            max_connection_pool_size=config.max_connection_pool_size,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._driver is not None:
            return

        import neo4j
        from neo4j import AsyncGraphDatabase

        if self._iam_auth:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for Neptune IAM auth. Install: pip install khora[neptune-iam]")

            from neo4j.auth_management import AsyncAuthManagers, ExpiringAuth

            # The provider chain resolves once here; the returned credentials
            # object refreshes STS/role credentials internally, so freezing
            # happens per signature in _sign_iam_token(), never at connect().
            self._boto_session = boto3.Session(region_name=self._aws_region)

            async def _fresh_iam_auth() -> ExpiringAuth:
                # Called by the driver whenever a new Bolt connection needs
                # auth and the cached token has expired - every connection
                # presents a signature at most _IAM_TOKEN_TTL_SECONDS old
                # instead of one frozen at connect() (#1152).
                token = self._sign_iam_token()
                return ExpiringAuth(neo4j.basic_auth("", token)).expires_in(_IAM_TOKEN_TTL_SECONDS)

            auth = AsyncAuthManagers.bearer(_fresh_iam_auth)
        else:
            auth = (self._user, self._password) if self._user else None

        logger.info("Connecting to Neptune at {url}...", url=_safe_url_for_log(self._url))
        self._driver = AsyncGraphDatabase.driver(
            self._url,
            auth=auth,
            encrypted=self._encrypted,
            max_connection_pool_size=self._max_connection_pool_size,
        )
        await self._driver.verify_connectivity()
        await self._create_indexes()
        logger.info("Connected to Neptune")

    def _sign_iam_token(self) -> str:
        """Build a freshly SigV4-signed Neptune Bolt-IAM auth token.

        SigV4 signatures expire after ~5 minutes, so this runs on every token
        refresh - the result must never be cached for the driver's lifetime
        (#1152). The JSON shape follows AWS's documented Bolt-IAM token
        format: the signed headers plus ``HttpMethod``.
        """
        import json
        from urllib.parse import urlparse

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials = self._boto_session.get_credentials().get_frozen_credentials()

        # Parse Neptune endpoint for signing
        parsed = urlparse(self._url.replace("bolt://", "https://").replace("bolt+s://", "https://"))
        request = AWSRequest(
            method="GET",
            url=f"https://{parsed.hostname}:{parsed.port or 8182}/opencypher",
            headers={"Host": parsed.hostname},
        )
        SigV4Auth(credentials, "neptune-db", self._aws_region).add_auth(request)

        token = dict(request.headers)
        token["HttpMethod"] = "GET"
        return json.dumps(token)

    async def disconnect(self) -> None:
        if self._driver is not None:
            logger.info("Disconnecting from Neptune...")
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neptune")

    async def is_healthy(self) -> bool:
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neptune health check failed: {e}")
            return False

    async def _create_indexes(self) -> None:
        """Neptune auto-indexes all properties — no DDL needed."""
        logger.debug("Neptune auto-indexes properties; skipping index creation")

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
        """Convert a Neptune relationship to a domain Relationship.

        Returns ``None`` when an endpoint id is null: a synthesized endpoint id
        would be a dangling FK, so the malformed row is skipped (#1238, porting
        the Neo4j #1237 guard). Missing ``id`` / ``namespace_id`` on the edge
        itself are synthesized (porting #767, which Neptune never received).
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
        raise NotImplementedError

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

        # Dynamic relationship type via f-string (parameterized labels not supported in Cypher)
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
                self._record_to_relationship(
                    element_to_dict(r["r"]),
                    r["source_id"],
                    r["target_id"],
                    r["rel_type"],
                )
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
        query = f"""
        MATCH (source:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(target:Entity {{namespace_id: $namespace_id}})
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
            rels = (
                self._record_to_relationship(
                    r["rel_props"],
                    r["source_id"],
                    r["target_id"],
                    r["rel_type"],
                )
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

        # All nodes — endpoints AND intermediates — must share ``namespace_id``
        # so the traversal never crosses tenants (IDOR family).
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
        # share ``namespace_id``.
        # Slice inside the projection: a bare ``LIMIT $limit`` after
        # ``collect(...)`` aggregation is a no-op because aggregation already
        # produced a single row (#1154).
        query = f"""
        MATCH (center:Entity {{id: $entity_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: $namespace_id}})
        RETURN collect(DISTINCT other)[0..$limit] as nodes, collect(DISTINCT r)[0..$limit] as relationships
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
                    if isinstance(rel_list, list):
                        for r in rel_list:
                            if r:
                                relationships.append(element_to_dict(r))
                    elif rel_list:
                        relationships.append(element_to_dict(rel_list))
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
