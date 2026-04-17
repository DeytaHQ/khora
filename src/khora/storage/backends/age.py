"""PostgreSQL AGE graph backend.

Uses Apache AGE extension to run openCypher queries inside PostgreSQL.
Can share the same connection pool as Khora's relational backend.

Install: ``pip install khora[age]`` (requires PostgreSQL with AGE extension).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models.entity import Entity, Episode, Relationship
from khora.telemetry import trace

from .mixins import GraphBackendBase, deserialize_dict, serialize_dict

# ---------------------------------------------------------------------------
# Regex to strip the ``::agtype`` suffix that AGE appends to result strings.
# ---------------------------------------------------------------------------
_AGTYPE_SUFFIX_RE = re.compile(r"::agtype$")


class AGEBackend(GraphBackendBase):
    """PostgreSQL AGE graph backend using openCypher wrapped in SQL.

    Apache AGE (A Graph Extension) adds openCypher support to PostgreSQL.
    Queries are Cypher strings wrapped in the ``cypher()`` SQL function::

        SELECT * FROM cypher('graph', $$ MATCH (n) RETURN n $$) AS (v agtype)

    This backend can share the same ``AsyncEngine`` / connection pool as
    Khora's relational PostgreSQL backend.
    """

    def __init__(
        self,
        database_url: str,
        *,
        graph_name: str = "khora_graph",
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._database_url = database_url
        self._graph_name = graph_name
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._engine = engine
        self._owns_engine = engine is None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @classmethod
    def from_config(cls, config: Any) -> AGEBackend:
        """Create an AGEBackend from an AGEConfig object."""
        return cls(
            database_url=config.url or "",
            graph_name=getattr(config, "graph_name", "khora_graph"),
            pool_size=getattr(config, "pool_size", 10),
            max_overflow=getattr(config, "max_overflow", 20),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the connection pool, AGE extension and graph."""
        if self._session_factory is not None:
            return

        url = self._database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)

        if self._engine is None:
            self._engine = create_async_engine(
                url,
                echo=self._echo,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
            )

        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

        # Bootstrap AGE extension and graph
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS age"))
                await session.execute(text("LOAD 'age'"))
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))

                # create_graph raises if the graph already exists; swallow.
                try:
                    await session.execute(text(f"SELECT create_graph('{self._graph_name}')"))
                except Exception as e:
                    logger.debug(f"Graph creation skipped (likely already exists): {e}")

        await self._create_indexes()
        logger.info(f"Connected to PostgreSQL AGE (graph: {self._graph_name})")

    async def disconnect(self) -> None:
        """Dispose the engine if we own it."""
        if self._engine is not None:
            logger.info("Disconnecting from PostgreSQL AGE...")
            if self._owns_engine:
                await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Disconnected from PostgreSQL AGE")

    async def is_healthy(self) -> bool:
        """Run a trivial Cypher query to verify connectivity."""
        if self._session_factory is None:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"AGE health check failed: {exc}")
            return False

    async def _create_indexes(self) -> None:
        """Create PostgreSQL indexes on AGE internal vertex/edge tables."""
        async with self._session_factory() as session:  # type: ignore[union-attr]
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                indexes = [
                    f"CREATE INDEX IF NOT EXISTS idx_age_entity_id "
                    f'ON {self._graph_name}."Entity" USING btree '
                    f"((properties::jsonb->>'id'))",
                    f"CREATE INDEX IF NOT EXISTS idx_age_entity_ns "
                    f'ON {self._graph_name}."Entity" USING btree '
                    f"((properties::jsonb->>'namespace_id'))",
                    f"CREATE INDEX IF NOT EXISTS idx_age_entity_name "
                    f'ON {self._graph_name}."Entity" USING btree '
                    f"((properties::jsonb->>'name'))",
                    f"CREATE INDEX IF NOT EXISTS idx_age_episode_id "
                    f'ON {self._graph_name}."Episode" USING btree '
                    f"((properties::jsonb->>'id'))",
                ]
                for idx_sql in indexes:
                    try:
                        await session.execute(text(idx_sql))
                    except Exception as exc:
                        logger.debug(f"Index creation skipped: {exc}")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    async def _cypher(
        self,
        session: AsyncSession,
        cypher_query: str,
        *,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query via AGE's ``cypher()`` SQL function.

        Wraps the Cypher in::

            SELECT * FROM cypher('graph', $$ <cypher> $$) AS (<columns>)

        Parses agtype results into Python dicts.
        """
        cols = columns or ["v agtype"]
        col_clause = ", ".join(cols)
        sql = f"SELECT * FROM cypher('{self._graph_name}', $$ {cypher_query} $$) AS ({col_clause})"  # noqa: S608

        result = await session.execute(text(sql))
        rows = result.fetchall()

        parsed: list[dict[str, Any]] = []
        for row in rows:
            row_dict: dict[str, Any] = {}
            for i, col_def in enumerate(cols):
                col_name = col_def.split()[0]
                raw_val = row[i]
                row_dict[col_name] = self._parse_agtype(raw_val)
            parsed.append(row_dict)
        return parsed

    @staticmethod
    def _parse_agtype(value: Any) -> Any:
        """Parse an agtype value into Python objects.

        agtype nodes look like::

            {"id": 123, "label": "Entity", "properties": {...}}

        agtype edges::

            {"id": 456, "label": "REL", "start_id": ..., "end_id": ..., "properties": {...}}

        Scalars are passed through as-is.
        """
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            # Strip the ``::agtype`` suffix if present
            cleaned = _AGTYPE_SUFFIX_RE.sub("", value).strip()
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                return cleaned
        if isinstance(value, dict):
            return value
        return str(value)

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a string value for safe inclusion in AGE Cypher queries.

        AGE parameters cannot be passed via ``$param`` -- values are
        interpolated into the Cypher string.  This escaper handles single
        quotes, backslashes, and control characters.
        """
        if not value:
            return ""
        # Replace backslashes first, then single quotes
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        # Strip null bytes and escape other control characters
        escaped = escaped.replace("\x00", "").replace("\r", "\\r").replace("\n", "\\n")
        return escaped

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Ensure a Cypher label/relationship-type is safe.

        Only alphanumeric characters and underscores are kept.
        """
        return re.sub(r"[^A-Za-z0-9_]", "_", label)

    # ------------------------------------------------------------------
    # agtype -> domain model converters
    # ------------------------------------------------------------------

    def _entity_from_agtype(self, data: dict[str, Any]) -> Entity | None:
        """Convert a parsed agtype node to an :class:`Entity`."""
        props = data.get("properties", data)
        if not props.get("id"):
            return None

        return Entity(
            id=UUID(str(props["id"])),
            namespace_id=UUID(str(props.get("namespace_id", ""))),
            name=props.get("name", ""),
            entity_type=props.get("entity_type", ""),
            description=props.get("description", ""),
            attributes=deserialize_dict(props.get("attributes")),
            source_document_ids=[UUID(d) for d in (props.get("source_document_ids") or [])],
            source_chunk_ids=[UUID(c) for c in (props.get("source_chunk_ids") or [])],
            mention_count=props.get("mention_count", 1),
            valid_from=(datetime.fromisoformat(props["valid_from"]) if props.get("valid_from") else None),
            valid_until=(datetime.fromisoformat(props["valid_until"]) if props.get("valid_until") else None),
            confidence=props.get("confidence", 1.0),
            metadata=deserialize_dict(props.get("metadata")),
            embedding=None,
            created_at=(datetime.fromisoformat(props["created_at"]) if props.get("created_at") else datetime.now(UTC)),
            updated_at=(datetime.fromisoformat(props["updated_at"]) if props.get("updated_at") else datetime.now(UTC)),
        )

    def _relationship_from_agtype(
        self,
        data: dict[str, Any],
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> Relationship:
        """Convert a parsed agtype edge to a :class:`Relationship`."""
        props = data.get("properties", data)
        return Relationship(
            id=UUID(str(props["id"])),
            namespace_id=UUID(str(props["namespace_id"])),
            source_entity_id=UUID(str(source_id)),
            target_entity_id=UUID(str(target_id)),
            relationship_type=rel_type,
            description=props.get("description", ""),
            properties=deserialize_dict(props.get("properties")),
            source_document_ids=[UUID(d) for d in (props.get("source_document_ids") or [])],
            source_chunk_ids=[UUID(c) for c in (props.get("source_chunk_ids") or [])],
            valid_from=(datetime.fromisoformat(props["valid_from"]) if props.get("valid_from") else None),
            valid_until=(datetime.fromisoformat(props["valid_until"]) if props.get("valid_until") else None),
            confidence=props.get("confidence", 1.0),
            weight=props.get("weight", 1.0),
            metadata=deserialize_dict(props.get("metadata")),
            created_at=(datetime.fromisoformat(props["created_at"]) if props.get("created_at") else datetime.now(UTC)),
            updated_at=(datetime.fromisoformat(props["updated_at"]) if props.get("updated_at") else datetime.now(UTC)),
        )

    def _episode_from_agtype(self, data: dict[str, Any]) -> Episode:
        """Convert a parsed agtype node to an :class:`Episode`."""
        props = data.get("properties", data)
        return Episode(
            id=UUID(str(props["id"])),
            namespace_id=UUID(str(props["namespace_id"])),
            name=props.get("name", ""),
            description=props.get("description", ""),
            occurred_at=datetime.fromisoformat(props["occurred_at"]),
            duration_seconds=props.get("duration_seconds"),
            entity_ids=[UUID(e) for e in (props.get("entity_ids") or [])],
            source_document_ids=[UUID(d) for d in (props.get("source_document_ids") or [])],
            source_chunk_ids=[UUID(c) for c in (props.get("source_chunk_ids") or [])],
            metadata=deserialize_dict(props.get("metadata")),
            created_at=(datetime.fromisoformat(props["created_at"]) if props.get("created_at") else datetime.now(UTC)),
            updated_at=(datetime.fromisoformat(props["updated_at"]) if props.get("updated_at") else datetime.now(UTC)),
        )

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._session_factory

    async def _age_session(self) -> AsyncSession:
        """Return a new session with AGE search_path already set."""
        session = self._get_session_factory()()
        await session.execute(text('SET search_path = ag_catalog, "$user", public'))
        return session

    # ------------------------------------------------------------------
    # Cypher list helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cypher_str_list(items: list[str]) -> str:
        """Format a Python list of strings as a Cypher list literal.

        AGE does not support parameter binding, so we inline lists.
        """
        inner = ", ".join(f"'{AGEBackend._escape(s)}'" for s in items)
        return f"[{inner}]"

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    @trace("khora.age.create_entity")
    async def create_entity(self, entity: Entity) -> Entity:
        now = datetime.now(UTC).isoformat()
        src_doc_ids = self._cypher_str_list([str(d) for d in entity.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in entity.source_chunk_ids])

        cypher = f"""
            CREATE (e:Entity {{
                id: '{entity.id}',
                namespace_id: '{entity.namespace_id}',
                name: '{self._escape(entity.name)}',
                entity_type: '{self._escape(entity.entity_type)}',
                description: '{self._escape(entity.description or "")}',
                attributes: '{serialize_dict(entity.attributes) or "{}"}',
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                mention_count: {entity.mention_count},
                valid_from: {f"'{entity.valid_from.isoformat()}'" if entity.valid_from else "null"},
                valid_until: {f"'{entity.valid_until.isoformat()}'" if entity.valid_until else "null"},
                confidence: {entity.confidence},
                metadata: '{serialize_dict(entity.metadata) or "{}"}',
                created_at: '{now}',
                updated_at: '{now}'
            }})
            RETURN e
        """

        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["e agtype"])
                if rows:
                    parsed = self._entity_from_agtype(rows[0]["e"])
                    if parsed is not None:
                        return parsed
        return entity

    @trace("khora.age.get_entity")
    async def get_entity(self, entity_id: UUID) -> Entity | None:
        cypher = f"""
            MATCH (e:Entity {{id: '{entity_id}'}})
            RETURN e
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["e agtype"])
                if rows:
                    return self._entity_from_agtype(rows[0]["e"])
        return None

    @trace("khora.age.get_entity_by_name")
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        cypher = f"""
            MATCH (e:Entity {{
                namespace_id: '{namespace_id}',
                name: '{self._escape(name)}',
                entity_type: '{self._escape(entity_type)}'
            }})
            RETURN e
            LIMIT 1
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["e agtype"])
                if rows:
                    return self._entity_from_agtype(rows[0]["e"])
        return None

    @trace("khora.age.update_entity")
    async def update_entity(self, entity: Entity) -> Entity:
        now = datetime.now(UTC).isoformat()
        src_doc_ids = self._cypher_str_list([str(d) for d in entity.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in entity.source_chunk_ids])

        cypher = f"""
            MATCH (e:Entity {{id: '{entity.id}'}})
            SET e.name = '{self._escape(entity.name)}',
                e.description = '{self._escape(entity.description or "")}',
                e.attributes = '{serialize_dict(entity.attributes) or "{}"}',
                e.source_document_ids = {src_doc_ids},
                e.source_chunk_ids = {src_chunk_ids},
                e.mention_count = {entity.mention_count},
                e.valid_from = {f"'{entity.valid_from.isoformat()}'" if entity.valid_from else "null"},
                e.valid_until = {f"'{entity.valid_until.isoformat()}'" if entity.valid_until else "null"},
                e.confidence = {entity.confidence},
                e.metadata = '{serialize_dict(entity.metadata) or "{}"}',
                e.updated_at = '{now}'
            RETURN e
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                await self._cypher(session, cypher, columns=["e agtype"])
        return entity

    @trace("khora.age.delete_entity")
    async def delete_entity(self, entity_id: UUID) -> bool:
        cypher = f"""
            MATCH (e:Entity {{id: '{entity_id}'}})
            DETACH DELETE e
            RETURN count(e) as deleted
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["deleted agtype"])
                if rows:
                    return (rows[0].get("deleted") or 0) > 0
        return False

    @trace("khora.age.list_entities")
    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        match_clause = f"MATCH (e:Entity {{namespace_id: '{namespace_id}'}})"
        where_clause = ""
        if entity_type:
            where_clause = f" WHERE e.entity_type = '{self._escape(entity_type)}'"

        cypher = f"""
            {match_clause}
            {where_clause}
            RETURN e
            ORDER BY e.name
            SKIP {offset}
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["e agtype"])
                entities = []
                for row in rows:
                    entity = self._entity_from_agtype(row["e"])
                    if entity is not None:
                        entities.append(entity)
                return entities

    @trace("khora.age.count_entities")
    async def count_entities(self, namespace_id: UUID) -> int:
        cypher = f"""
            MATCH (e:Entity {{namespace_id: '{namespace_id}'}})
            RETURN count(e) AS cnt
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["cnt agtype"])
                if rows:
                    return rows[0].get("cnt", 0)
        return 0

    async def count_relationships(self, namespace_id: UUID) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    @trace("khora.age.create_relationship")
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        now = datetime.now(UTC).isoformat()
        rel_type = self._sanitize_label(relationship.relationship_type)
        src_doc_ids = self._cypher_str_list([str(d) for d in relationship.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in relationship.source_chunk_ids])

        cypher = f"""
            MATCH (source:Entity {{id: '{relationship.source_entity_id}'}})
            MATCH (target:Entity {{id: '{relationship.target_entity_id}'}})
            CREATE (source)-[r:{rel_type} {{
                id: '{relationship.id}',
                namespace_id: '{relationship.namespace_id}',
                description: '{self._escape(relationship.description or "")}',
                properties: '{serialize_dict(relationship.properties) or "{}"}',
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                valid_from: {f"'{relationship.valid_from.isoformat()}'" if relationship.valid_from else "null"},
                valid_until: {f"'{relationship.valid_until.isoformat()}'" if relationship.valid_until else "null"},
                confidence: {relationship.confidence},
                weight: {relationship.weight},
                metadata: '{serialize_dict(relationship.metadata) or "{}"}',
                created_at: '{now}',
                updated_at: '{now}'
            }}]->(target)
            RETURN r, source.id as source_id, target.id as target_id
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                await self._cypher(
                    session,
                    cypher,
                    columns=["r agtype", "source_id agtype", "target_id agtype"],
                )
        return relationship

    @trace("khora.age.get_relationship")
    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        cypher = f"""
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE r.id = '{relationship_id}'
            RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(
                    session,
                    cypher,
                    columns=[
                        "r agtype",
                        "source_id agtype",
                        "target_id agtype",
                        "rel_type agtype",
                    ],
                )
                if rows:
                    row = rows[0]
                    return self._relationship_from_agtype(
                        row["r"],
                        str(row["source_id"]),
                        str(row["target_id"]),
                        str(row["rel_type"]),
                    )
        return None

    @trace("khora.age.delete_relationship")
    async def delete_relationship(self, relationship_id: UUID) -> bool:
        cypher = f"""
            MATCH ()-[r]->()
            WHERE r.id = '{relationship_id}'
            DELETE r
            RETURN count(r) as deleted
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["deleted agtype"])
                if rows:
                    return (rows[0].get("deleted") or 0) > 0
        return False

    @trace("khora.age.get_entity_relationships")
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        if direction == "outgoing":
            pattern = f"(e)-[r{rel_filter}]->(other)"
        elif direction == "incoming":
            pattern = f"(other)-[r{rel_filter}]->(e)"
        else:
            pattern = f"(e)-[r{rel_filter}]-(other)"

        cypher = f"""
            MATCH {pattern}
            WHERE e.id = '{entity_id}'
            RETURN r, e.id as source_id, other.id as target_id, type(r) as rel_type
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(
                    session,
                    cypher,
                    columns=[
                        "r agtype",
                        "source_id agtype",
                        "target_id agtype",
                        "rel_type agtype",
                    ],
                )
                return [
                    self._relationship_from_agtype(
                        row["r"],
                        str(row["source_id"]),
                        str(row["target_id"]),
                        str(row["rel_type"]),
                    )
                    for row in rows
                ]

    @trace("khora.age.list_relationships")
    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        rel_filter = ""
        if relationship_type:
            rel_filter = f":{self._sanitize_label(relationship_type)}"

        cypher = f"""
            MATCH (source:Entity)-[r{rel_filter}]->(target:Entity)
            WHERE r.namespace_id = '{namespace_id}'
            RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
            ORDER BY r.created_at DESC
            SKIP {offset}
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(
                    session,
                    cypher,
                    columns=[
                        "r agtype",
                        "source_id agtype",
                        "target_id agtype",
                        "rel_type agtype",
                    ],
                )
                return [
                    self._relationship_from_agtype(
                        row["r"],
                        str(row["source_id"]),
                        str(row["target_id"]),
                        str(row["rel_type"]),
                    )
                    for row in rows
                ]

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    @trace("khora.age.create_episode")
    async def create_episode(self, episode: Episode) -> Episode:
        now = datetime.now(UTC).isoformat()
        entity_id_list = self._cypher_str_list([str(e) for e in episode.entity_ids])
        src_doc_ids = self._cypher_str_list([str(d) for d in episode.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in episode.source_chunk_ids])

        cypher = f"""
            CREATE (ep:Episode {{
                id: '{episode.id}',
                namespace_id: '{episode.namespace_id}',
                name: '{self._escape(episode.name)}',
                description: '{self._escape(episode.description or "")}',
                occurred_at: '{episode.occurred_at.isoformat()}',
                duration_seconds: {episode.duration_seconds if episode.duration_seconds is not None else "null"},
                entity_ids: {entity_id_list},
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                metadata: '{serialize_dict(episode.metadata) or "{}"}',
                created_at: '{now}',
                updated_at: '{now}'
            }})
            RETURN ep
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                await self._cypher(session, cypher, columns=["ep agtype"])

                # Link episode to its entities
                if episode.entity_ids:
                    for eid in episode.entity_ids:
                        link_cypher = f"""
                            MATCH (ep:Episode {{id: '{episode.id}'}})
                            MATCH (e:Entity {{id: '{eid}'}})
                            CREATE (ep)-[:INVOLVES]->(e)
                        """
                        await self._cypher(session, link_cypher, columns=["v agtype"])

        return episode

    @trace("khora.age.get_episode")
    async def get_episode(self, episode_id: UUID) -> Episode | None:
        cypher = f"""
            MATCH (ep:Episode {{id: '{episode_id}'}})
            RETURN ep
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["ep agtype"])
                if rows:
                    return self._episode_from_agtype(rows[0]["ep"])
        return None

    @trace("khora.age.list_episodes")
    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        match_clause = f"MATCH (ep:Episode {{namespace_id: '{namespace_id}'}})"
        conditions: list[str] = []
        if start_time:
            conditions.append(f"ep.occurred_at >= '{start_time.isoformat()}'")
        if end_time:
            conditions.append(f"ep.occurred_at <= '{end_time.isoformat()}'")

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        cypher = f"""
            {match_clause}
            {where_clause}
            RETURN ep
            ORDER BY ep.occurred_at DESC
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["ep agtype"])
                return [self._episode_from_agtype(row["ep"]) for row in rows]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @trace("khora.age.find_paths")
    async def find_paths(
        self,
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        cypher = f"""
            MATCH path = (source:Entity {{id: '{source_entity_id}'}})-[r{rel_filter}*1..{max_depth}]-(target:Entity {{id: '{target_entity_id}'}})
            WHERE source.namespace_id = '{namespace_id}' AND target.namespace_id = '{namespace_id}'
            RETURN path
            LIMIT 10
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["path agtype"])

                paths: list[list[dict[str, Any]]] = []
                for row in rows:
                    path_data = row.get("path")
                    if isinstance(path_data, list):
                        path_elements: list[dict[str, Any]] = []
                        for element in path_data:
                            if isinstance(element, dict):
                                if "label" in element and "start_id" in element:
                                    path_elements.append({"type": "relationship", "data": element})
                                else:
                                    path_elements.append({"type": "node", "data": element})
                        paths.append(path_elements)
                    elif isinstance(path_data, dict):
                        # Single-element path
                        paths.append([{"type": "node", "data": path_data}])
                return paths

    @trace("khora.age.get_neighborhood")
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        cypher = f"""
            MATCH (center:Entity {{id: '{entity_id}'}})-[r{rel_filter}*1..{depth}]-(other:Entity)
            RETURN collect(DISTINCT other) as nodes, collect(DISTINCT r) as relationships
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(
                    session,
                    cypher,
                    columns=["nodes agtype", "relationships agtype"],
                )
                if rows:
                    row = rows[0]
                    nodes = row.get("nodes", [])
                    if not isinstance(nodes, list):
                        nodes = []
                    rels = row.get("relationships", [])
                    if not isinstance(rels, list):
                        rels = []
                    # Flatten nested relationship lists
                    flat_rels: list[Any] = []
                    for item in rels:
                        if isinstance(item, list):
                            flat_rels.extend(r for r in item if r)
                        elif item:
                            flat_rels.append(item)
                    return {"entities": nodes, "relationships": flat_rels}
        return {"entities": [], "relationships": []}

    @trace("khora.age.search_entities_by_attribute")
    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        # Attributes are stored as a JSON string, so we search with CONTAINS
        # pattern in the serialised JSON.  This is a best-effort approach.
        escaped_name = self._escape(str(attribute_name))
        escaped_value = self._escape(str(attribute_value))

        cypher = f"""
            MATCH (e:Entity {{namespace_id: '{namespace_id}'}})
            WHERE e.attributes CONTAINS '"{escaped_name}"'
              AND e.attributes CONTAINS '{escaped_value}'
            RETURN e
            LIMIT {limit}
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                rows = await self._cypher(session, cypher, columns=["e agtype"])
                entities = []
                for row in rows:
                    entity = self._entity_from_agtype(row["e"])
                    if entity is not None:
                        entities.append(entity)
                return entities
