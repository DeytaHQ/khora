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
from khora.dream.plan import OpKind
from khora.telemetry import trace

from .mixins import GraphBackendBase, deserialize_dict, sanitize_cypher_label, serialize_dict

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
        """Create an AGEBackend from an AGEConfig object.

        If ``config.url`` is a ``SecretStr`` it is unwrapped exactly here so
        the SQLAlchemy engine receives a plaintext DSN.
        """
        from pydantic import SecretStr

        url = config.url or ""
        if isinstance(url, SecretStr):
            url = url.get_secret_value()
        return cls(
            database_url=url,
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

        Wraps the Cypher in a *uniquely-tagged* PostgreSQL dollar-quoted
        string (``$khora_age$ ... $khora_age$``). The unique tag defangs a
        ``$$``-breakout escalation path: a Cypher payload containing the
        bare token ``$$`` no longer closes the surrounding SQL string. A
        payload containing the literal tag ``$khora_age$`` would still
        escape, so we reject that defensively before the round-trip.

        Parses agtype results into Python dicts.
        """
        if "$khora_age$" in cypher_query:
            # Defense in depth — the per-string-literal escape in
            # _serialize_dict_literal / _escape already handles single
            # quotes + backslashes, but the Cypher-level escape can't reach
            # the SQL-level dollar-quote. Refusing here is preferable to
            # silently mangling the tag.
            raise ValueError(
                "Cypher query contains the reserved dollar-quote tag '$khora_age$'. "
                "Run caller-derived input through AGEBackend._escape or "
                "AGEBackend._serialize_dict_literal before interpolation."
            )

        cols = columns or ["v agtype"]
        col_clause = ", ".join(cols)
        sql = f"SELECT * FROM cypher('{self._graph_name}', $khora_age$ {cypher_query} $khora_age$) AS ({col_clause})"  # noqa: S608

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
    def _serialize_dict_literal(value: dict[str, Any] | None) -> str:
        """Serialize a dict for safe interpolation inside a Cypher string literal.

        Caller-controlled keys / values may contain single quotes, backslashes,
        or other Cypher metacharacters that would otherwise close the
        surrounding ``'...'`` literal early and let the rest of the JSON be
        parsed as Cypher. This helper JSON-serialises the value and then
        runs the result through :meth:`_escape` so the embedded payload
        round-trips into the graph store verbatim.

        Covers every site that interpolates a dict into Cypher:
        ``Entity.attributes``, ``Entity.metadata``, ``Relationship.properties``,
        ``Relationship.metadata``, ``Episode.metadata`` — anything that
        flows through the LLM extractor and ends up inside a Cypher map.

        Returns ``"{}"`` for empty / None inputs so callers can drop the
        ``or "{}"`` fallback they previously used.
        """
        # serialize_dict({}) returns the literal string "{}"; serialize_dict(None)
        # is a TypeError. Treat None / falsy the same.
        if not value:
            return "{}"
        return AGEBackend._escape(serialize_dict(value))

    # Cypher relationship-type sanitisation is delegated to the shared
    # :func:`sanitize_cypher_label` helper (issue #749) so AGE produces the
    # same UPPER_SNAKE_CASE label as Neo4j / Memgraph / sqlite_lance.  Prior
    # to #749 AGE's bespoke regex preserved case (``reports_to`` instead of
    # ``REPORTS_TO``), causing identical inputs to read back differently from
    # different backends.
    _sanitize_label = staticmethod(sanitize_cypher_label)

    @staticmethod
    def _uuid_lit(value: UUID | str) -> str:
        """Coerce a UUID-shaped value to its canonical 36-char string form.

        AGE's ``cypher()`` SQL function rejects ``$param`` placeholders, so
        UUIDs are interpolated directly into the Cypher source.  Routing
        them through :class:`UUID` validates the input (type-safe today,
        injection-resistant if a caller is ever duck-typed to a ``str``) and
        normalises the form so any junk like surrounding quotes / spaces
        fails fast at the boundary rather than reaching the graph store
        (IDOR family).
        """
        if isinstance(value, UUID):
            return str(value)
        return str(UUID(str(value)))

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

        eid_lit = self._uuid_lit(entity.id)
        ns_lit_val = self._uuid_lit(entity.namespace_id)
        cypher = f"""
            CREATE (e:Entity {{
                id: '{eid_lit}',
                namespace_id: '{ns_lit_val}',
                name: '{self._escape(entity.name)}',
                entity_type: '{self._escape(entity.entity_type)}',
                description: '{self._escape(entity.description or "")}',
                attributes: '{self._serialize_dict_literal(entity.attributes)}',
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                mention_count: {entity.mention_count},
                valid_from: {f"'{entity.valid_from.isoformat()}'" if entity.valid_from else "null"},
                valid_until: {f"'{entity.valid_until.isoformat()}'" if entity.valid_until else "null"},
                confidence: {entity.confidence},
                metadata: '{self._serialize_dict_literal(entity.metadata)}',
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
    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id`` (the IDOR family / the IDOR family).

        AGE Cypher is f-string-interpolated (the ``cypher()`` SQL function
        rejects ``$param`` placeholders). UUID values are routed through
        :meth:`_uuid_lit` to validate type and normalise to the canonical
        36-char form, so a duck-typed string caller cannot inject Cypher.
        """
        eid_lit = self._uuid_lit(entity_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (e:Entity {{id: '{eid_lit}', namespace_id: '{ns_lit_val}'}})
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
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (e:Entity {{
                namespace_id: '{ns_lit_val}',
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
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id`` (IDOR family)."""
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        now = datetime.now(UTC).isoformat()
        src_doc_ids = self._cypher_str_list([str(d) for d in entity.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in entity.source_chunk_ids])
        eid_lit = self._uuid_lit(entity.id)
        ns_lit_val = self._uuid_lit(namespace_id)

        cypher = f"""
            MATCH (e:Entity {{id: '{eid_lit}', namespace_id: '{ns_lit_val}'}})
            SET e.name = '{self._escape(entity.name)}',
                e.description = '{self._escape(entity.description or "")}',
                e.attributes = '{self._serialize_dict_literal(entity.attributes)}',
                e.source_document_ids = {src_doc_ids},
                e.source_chunk_ids = {src_chunk_ids},
                e.mention_count = {entity.mention_count},
                e.valid_from = {f"'{entity.valid_from.isoformat()}'" if entity.valid_from else "null"},
                e.valid_until = {f"'{entity.valid_until.isoformat()}'" if entity.valid_until else "null"},
                e.confidence = {entity.confidence},
                e.metadata = '{self._serialize_dict_literal(entity.metadata)}',
                e.updated_at = '{now}'
            RETURN e
        """
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                await self._cypher(session, cypher, columns=["e agtype"])
        return entity

    @trace("khora.age.delete_entity")
    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id`` (IDOR family)."""
        eid_lit = self._uuid_lit(entity_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (e:Entity {{id: '{eid_lit}', namespace_id: '{ns_lit_val}'}})
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
        ns_lit_val = self._uuid_lit(namespace_id)
        match_clause = f"MATCH (e:Entity {{namespace_id: '{ns_lit_val}'}})"
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
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (e:Entity {{namespace_id: '{ns_lit_val}'}})
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
        # Mirror the sanitised label back to the caller so an in-memory
        # ``Relationship`` object stays consistent with the persisted edge
        # type across every backend (issue #749).
        relationship.relationship_type = rel_type
        src_doc_ids = self._cypher_str_list([str(d) for d in relationship.source_document_ids])
        src_chunk_ids = self._cypher_str_list([str(c) for c in relationship.source_chunk_ids])

        src_lit = self._uuid_lit(relationship.source_entity_id)
        tgt_lit = self._uuid_lit(relationship.target_entity_id)
        rid_lit = self._uuid_lit(relationship.id)
        rel_ns_lit = self._uuid_lit(relationship.namespace_id)
        cypher = f"""
            MATCH (source:Entity {{id: '{src_lit}'}})
            MATCH (target:Entity {{id: '{tgt_lit}'}})
            CREATE (source)-[r:{rel_type} {{
                id: '{rid_lit}',
                namespace_id: '{rel_ns_lit}',
                description: '{self._escape(relationship.description or "")}',
                properties: '{self._serialize_dict_literal(relationship.properties)}',
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                valid_from: {f"'{relationship.valid_from.isoformat()}'" if relationship.valid_from else "null"},
                valid_until: {f"'{relationship.valid_until.isoformat()}'" if relationship.valid_until else "null"},
                confidence: {relationship.confidence},
                weight: {relationship.weight},
                metadata: '{self._serialize_dict_literal(relationship.metadata)}',
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
    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id`` (IDOR family).

        Both endpoint nodes are constrained to ``namespace_id`` so cross-tenant
        edges never surface.
        """
        ns_lit_val = self._uuid_lit(namespace_id)
        rid_lit = self._uuid_lit(relationship_id)
        cypher = f"""
            MATCH (source:Entity {{namespace_id: '{ns_lit_val}'}})-[r]->(target:Entity {{namespace_id: '{ns_lit_val}'}})
            WHERE r.id = '{rid_lit}' AND r.namespace_id = '{ns_lit_val}'
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
    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id`` (IDOR family)."""
        rid_lit = self._uuid_lit(relationship_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH ()-[r]->()
            WHERE r.id = '{rid_lit}' AND r.namespace_id = '{ns_lit_val}'
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
        namespace_id: UUID,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id`` (IDOR family).

        Both endpoint nodes are constrained to ``namespace_id`` so edges
        crossing into other namespaces don't surface.
        """
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        ns_lit_val = self._uuid_lit(namespace_id)
        eid_lit = self._uuid_lit(entity_id)
        ns_lit = f"{{namespace_id: '{ns_lit_val}'}}"
        if direction == "outgoing":
            pattern = f"(e:Entity {ns_lit})-[r{rel_filter}]->(other:Entity {ns_lit})"
        elif direction == "incoming":
            pattern = f"(other:Entity {ns_lit})-[r{rel_filter}]->(e:Entity {ns_lit})"
        else:
            pattern = f"(e:Entity {ns_lit})-[r{rel_filter}]-(other:Entity {ns_lit})"

        cypher = f"""
            MATCH {pattern}
            WHERE e.id = '{eid_lit}'
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

        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (source:Entity)-[r{rel_filter}]->(target:Entity)
            WHERE r.namespace_id = '{ns_lit_val}'
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

        ep_id_lit = self._uuid_lit(episode.id)
        ep_ns_lit = self._uuid_lit(episode.namespace_id)
        cypher = f"""
            CREATE (ep:Episode {{
                id: '{ep_id_lit}',
                namespace_id: '{ep_ns_lit}',
                name: '{self._escape(episode.name)}',
                description: '{self._escape(episode.description or "")}',
                occurred_at: '{episode.occurred_at.isoformat()}',
                duration_seconds: {episode.duration_seconds if episode.duration_seconds is not None else "null"},
                entity_ids: {entity_id_list},
                source_document_ids: {src_doc_ids},
                source_chunk_ids: {src_chunk_ids},
                metadata: '{self._serialize_dict_literal(episode.metadata)}',
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
                        eid_lit = self._uuid_lit(eid)
                        link_cypher = f"""
                            MATCH (ep:Episode {{id: '{ep_id_lit}'}})
                            MATCH (e:Entity {{id: '{eid_lit}'}})
                            CREATE (ep)-[:INVOLVES]->(e)
                        """
                        await self._cypher(session, link_cypher, columns=["v agtype"])

        return episode

    @trace("khora.age.get_episode")
    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id`` (IDOR family)."""
        ep_id_lit = self._uuid_lit(episode_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH (ep:Episode {{id: '{ep_id_lit}', namespace_id: '{ns_lit_val}'}})
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
        ns_lit_val = self._uuid_lit(namespace_id)
        match_clause = f"MATCH (ep:Episode {{namespace_id: '{ns_lit_val}'}})"
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
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        # Every node on the path — endpoints AND intermediates — must share
        # ``namespace_id`` so the traversal never crosses tenants (IDOR family).
        src_lit = self._uuid_lit(source_entity_id)
        tgt_lit = self._uuid_lit(target_entity_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        cypher = f"""
            MATCH path = (source:Entity {{id: '{src_lit}', namespace_id: '{ns_lit_val}'}})-[r{rel_filter}*1..{max_depth}]-(target:Entity {{id: '{tgt_lit}', namespace_id: '{ns_lit_val}'}})
            WHERE ALL(n IN nodes(path) WHERE n.namespace_id = '{ns_lit_val}')
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
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity, scoped to ``namespace_id``.

        Seed and every node reached during traversal are constrained to
        ``namespace_id`` so the result never crosses into another namespace
        (IDOR family).
        """
        rel_filter = ""
        if relationship_types:
            sanitized = [self._sanitize_label(rt) for rt in relationship_types]
            rel_filter = ":" + "|".join(sanitized)

        eid_lit = self._uuid_lit(entity_id)
        ns_lit_val = self._uuid_lit(namespace_id)
        # Slice inside the projection: a bare ``LIMIT {limit}`` after
        # ``collect(...)`` aggregation is a no-op because aggregation already
        # produced a single row (#1154).
        cypher = f"""
            MATCH (center:Entity {{id: '{eid_lit}', namespace_id: '{ns_lit_val}'}})-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: '{ns_lit_val}'}})
            RETURN collect(DISTINCT other)[0..{limit}] as nodes, collect(DISTINCT r)[0..{limit}] as relationships
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
        ns_lit_val = self._uuid_lit(namespace_id)

        cypher = f"""
            MATCH (e:Entity {{namespace_id: '{ns_lit_val}'}})
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

    # ------------------------------------------------------------------
    # Dream bi-temporal mirror verbs (#1279) — soft-delete-ONLY subset
    # ------------------------------------------------------------------
    # AGE lacks entity-versioning primitives (no :EntityVersion snapshot, no
    # in-place endpoint rewrite), so it mirrors ONLY the flat ``valid_until``
    # SET-by-id used by ``prune_edges``. ``dedupe_entities`` (entity-version
    # snapshot + endpoint rewrite) and ``normalize_schema`` (relabel) are NOT
    # advertised: the orchestrator records a structured
    # ``graph_mirror_unsupported_op_kind`` skip for those op kinds BEFORE any
    # PG-committed verb runs, so the unsupported ops degrade to a clean
    # pre-commit skip rather than a post-commit partial failure (ADR-001). The
    # version/relabel verbs keep the ``GraphBackendBase`` default, which raises
    # ``DreamBackendUnsupported``.
    #
    # AGE commits each ``cypher()`` write out-of-band (it cannot join the dream
    # orchestrator transaction even though it lives in Postgres). The mirror
    # runs OUTSIDE the apply transaction by design (eventual consistency,
    # reconciler-backed), so this is the expected execution model. The whole
    # id batch is wrapped in a SINGLE ``session.begin()`` so it is one
    # auto-commit unit: a mid-batch failure rolls the whole batch back and the
    # reconciler retries it cleanly. An AGE-atomic SAME-transaction mirror
    # (folding the graph SET into the orchestrator's PG commit) is a possible
    # FUTURE follow-up since AGE lives in Postgres — out of scope for #1279.
    #
    # Convergence is verified by id-set / live-set (``valid_until IS NULL``),
    # NEVER by edge counts: ``count_relationships`` raises NotImplementedError
    # on this backend.

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        """AGE mirrors only the flat-soft-delete ``prune_edges`` op (#1279).

        - ``VECTORCYPHER_PRUNE_EDGES`` -> :meth:`soft_invalidate_relationships_batch`

        Entity-version (``dedupe_entities``) and relabel (``normalize_schema``)
        are deliberately absent: AGE has no versioning primitive, so those op
        kinds are recorded as a structured skip by the orchestrator.
        """
        return frozenset({OpKind.VECTORCYPHER_PRUNE_EDGES})

    @trace("khora.age.soft_invalidate_relationships_batch")
    async def soft_invalidate_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
        invalidated_at: datetime,
    ) -> int:
        """Soft-delete edges by stamping ``valid_until`` (flat SET by id, #1279).

        Mirrors ``prune_edges``. Matched by relationship id within
        ``namespace_id`` (IDOR family); idempotent — only edges with a null
        ``valid_until`` are touched, so a reconciler replay is a no-op. Never
        deletes. ``valid_until`` is stored as an ISO string literal to match the
        shape :meth:`create_relationship` writes. The whole batch is one
        ``session.begin()`` auto-commit unit so a partial batch retries cleanly.

        AGE rejects ``$param`` placeholders, so ids / namespace / timestamp are
        interpolated through :meth:`_uuid_lit` (UUID validation — the IDOR /
        injection boundary) and :meth:`_escape`.

        Returns the number of edges actually invalidated (an id-set / live-set
        count, NOT a total edge count — AGE cannot count edges).
        """
        if not relationship_ids:
            return 0
        ns_lit_val = self._uuid_lit(namespace_id)
        ts_lit = self._escape(invalidated_at.isoformat())

        count = 0
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                for rid in relationship_ids:
                    rid_lit = self._uuid_lit(rid)
                    cypher = f"""
                        MATCH ()-[r]-()
                        WHERE r.id = '{rid_lit}'
                          AND r.namespace_id = '{ns_lit_val}'
                          AND r.valid_until IS NULL
                        SET r.valid_until = '{ts_lit}',
                            r.updated_at = '{ts_lit}'
                        RETURN count(r) AS invalidated
                    """
                    rows = await self._cypher(session, cypher, columns=["invalidated agtype"])
                    if rows:
                        count += rows[0].get("invalidated") or 0
        logger.debug(f"Dream-invalidated {count} relationships in namespace {namespace_id}")
        return count

    @trace("khora.age.restore_relationships_batch")
    async def restore_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> int:
        """Un-invalidate edges by clearing ``valid_until`` (reverse of prune, #1279).

        Reverses :meth:`soft_invalidate_relationships_batch` so ``dream_undo``
        restores PG and the graph to identical pre-apply live sets. Matched by
        relationship id within ``namespace_id`` (IDOR family); idempotent — only
        edges with a non-null ``valid_until`` transition, so a replay reports
        zero. The dedupe-only reverse verbs keep the raising default. One
        ``session.begin()`` auto-commit unit for the whole batch.

        Returns the number of edges actually restored.
        """
        if not relationship_ids:
            return 0
        ns_lit_val = self._uuid_lit(namespace_id)
        ts_lit = self._escape(datetime.now(UTC).isoformat())

        count = 0
        async with self._get_session_factory()() as session:
            async with session.begin():
                await session.execute(text('SET search_path = ag_catalog, "$user", public'))
                for rid in relationship_ids:
                    rid_lit = self._uuid_lit(rid)
                    cypher = f"""
                        MATCH ()-[r]-()
                        WHERE r.id = '{rid_lit}'
                          AND r.namespace_id = '{ns_lit_val}'
                          AND r.valid_until IS NOT NULL
                        SET r.valid_until = null,
                            r.updated_at = '{ts_lit}'
                        RETURN count(r) AS restored
                    """
                    rows = await self._cypher(session, cypher, columns=["restored agtype"])
                    if rows:
                        count += rows[0].get("restored") or 0
        logger.debug(f"Dream-restored {count} relationships in namespace {namespace_id}")
        return count
