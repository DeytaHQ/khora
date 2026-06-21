"""SurrealDB graph adapter for Khora.

Implements GraphBackendProtocol using SurrealDB's native graph traversal
(RELATE, ``->`` / ``<-`` arrows) and record-link IDs.  All record IDs
follow the ``table:⟨uuid⟩`` convention expected by the unified SurrealDB
schema.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.mixins import GraphBackendBase, sanitize_cypher_label
from khora.storage.backends.surrealdb._helpers import (
    _entity_to_bindings,
    _parse_dt,
    _parse_uuid,
    _rid,
    _row_to_entity,
    _sanitize_field_name,
)
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection


# ---------------------------------------------------------------------------
# Entity key gate (prevents upsert race conditions)
# ---------------------------------------------------------------------------


class _SurrealDBEntityKeyGate:
    """Serialize access to entities by ``(namespace_id, name, entity_type)`` key.

    Prevents the prefetch-compare-update race condition in
    :meth:`SurrealDBGraphAdapter.upsert_entities_batch` by ensuring that
    two concurrent batches touching overlapping entity keys are executed
    sequentially.  Identical in design to Neo4j's ``_EntityKeyGate``.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._condition = asyncio.Condition()
        self._in_flight: set[tuple[str, str, str]] = set()
        self._active = 0
        self._max_concurrent = max_concurrent

    @asynccontextmanager
    async def acquire(self, entities: list[Entity]) -> AsyncIterator[None]:
        """Acquire exclusive access for a set of entity keys."""
        keys = {(str(e.namespace_id), e.name, str(e.entity_type)) for e in entities}

        async with self._condition:
            while (keys & self._in_flight) or self._active >= self._max_concurrent:
                await self._condition.wait()
            self._in_flight |= keys
            self._active += 1

        try:
            yield
        finally:
            async with self._condition:
                self._in_flight -= keys
                self._active -= 1
                self._condition.notify_all()


# ---------------------------------------------------------------------------
# Namespace-membership guard for traversal rows
# ---------------------------------------------------------------------------


def _row_in_namespace(row: dict[str, Any], ns_str: str) -> bool:
    """Return True if a SurrealDB row's namespace matches ``ns_str``.

    Traversal rows expose ``namespace`` (a record link to
    ``memory_namespace:⟨...⟩``) and ``namespace_id`` (a denormalized
    string copy).  Either form is accepted.  If neither field is
    present, the row is rejected — never silently passed through.
    """
    if "namespace_id" in row and row["namespace_id"] is not None:
        return str(row["namespace_id"]) == ns_str
    ns_field = row.get("namespace")
    if ns_field is None:
        return False
    if isinstance(ns_field, dict) and "namespace_id" in ns_field:
        return str(ns_field["namespace_id"]) == ns_str
    # Record link form like "memory_namespace:⟨<uuid>⟩" — parse out the uuid.
    try:
        return str(_parse_uuid(ns_field)) == ns_str
    except Exception:  # noqa: BLE001 — defensive: malformed shapes are not the caller's namespace
        return False


# ---------------------------------------------------------------------------
# Relationship row → domain model
# ---------------------------------------------------------------------------


def _row_to_relationship(row: dict[str, Any]) -> Relationship:
    """Map a SurrealDB ``relates_to`` result row to a domain :class:`Relationship`.

    The ``in`` / ``out`` fields are SurrealDB record links set automatically
    by ``RELATE``.  The custom ``rel_id`` field stores the Khora UUID.
    """
    rel_id = _parse_uuid(row.get("rel_id", row.get("id", "")))
    namespace_id = _parse_uuid(row.get("namespace_id", ""))

    # ``in`` and ``out`` are record links like ``entity:⟨uuid⟩``
    source_entity_id = _parse_uuid(row.get("in", ""))
    target_entity_id = _parse_uuid(row.get("out", ""))

    src_doc_ids = [UUID(s) for s in (row.get("source_document_ids") or [])]
    src_chunk_ids = [UUID(s) for s in (row.get("source_chunk_ids") or [])]

    return Relationship(
        id=rel_id,
        namespace_id=namespace_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=row.get("relationship_type", "RELATES_TO"),
        description=row.get("description", ""),
        properties=row.get("properties") or {},
        source_document_ids=src_doc_ids,
        source_chunk_ids=src_chunk_ids,
        valid_from=_parse_dt(row.get("valid_from")),
        valid_until=_parse_dt(row.get("valid_until")),
        confidence=float(row.get("confidence", 1.0)),
        weight=float(row.get("weight", 1.0)),
        metadata=row.get("metadata_") or {},
        created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
    )


def _row_to_episode(row: dict[str, Any]) -> Episode:
    """Map a SurrealDB result row to a domain :class:`Episode`."""
    episode_id = _parse_uuid(row.get("id", ""))
    namespace_id = _parse_uuid(row.get("namespace", ""))

    entity_ids = [UUID(s) for s in (row.get("entity_ids") or [])]
    src_doc_ids = [UUID(s) for s in (row.get("source_document_ids") or [])]
    src_chunk_ids = [UUID(s) for s in (row.get("source_chunk_ids") or [])]

    raw_embedding = row.get("embedding")
    embedding: list[float] | None = None
    if raw_embedding is not None:
        embedding = [float(v) for v in raw_embedding]

    return Episode(
        id=episode_id,
        namespace_id=namespace_id,
        name=row.get("name", ""),
        description=row.get("description", ""),
        occurred_at=_parse_dt(row.get("occurred_at")) or datetime.now(UTC),
        duration_seconds=row.get("duration_seconds"),
        entity_ids=entity_ids,
        source_document_ids=src_doc_ids,
        source_chunk_ids=src_chunk_ids,
        embedding=embedding,
        embedding_model=row.get("embedding_model", ""),
        metadata=row.get("metadata_") or {},
        created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
    )


def _relationship_to_bindings(rel: Relationship) -> dict[str, Any]:
    """Convert a :class:`Relationship` to SurrealQL parameter bindings."""
    return {
        "rel_id": str(rel.id),
        "namespace_id": str(rel.namespace_id),
        "source_id": str(rel.source_entity_id),
        "target_id": str(rel.target_entity_id),
        "relationship_type": rel.relationship_type,
        "description": rel.description,
        "properties": rel.properties or {},
        "source_document_ids": [str(d) for d in rel.source_document_ids],
        "source_chunk_ids": [str(c) for c in rel.source_chunk_ids],
        "valid_from": rel.valid_from,
        "valid_until": rel.valid_until,
        "confidence": rel.confidence,
        "weight": rel.weight,
        "metadata_": rel.metadata or {},
        "created_at": rel.created_at,
        "updated_at": rel.updated_at,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SurrealDBGraphAdapter(GraphBackendBase):
    """Graph backend backed by SurrealDB.

    Uses SurrealDB ``RELATE`` statements for edges, record-link IDs for
    traversal, and the ``->`` / ``<-`` arrow operators for path queries.

    The adapter delegates all I/O to a :class:`SurrealDBConnection`,
    which manages client lifecycle and authentication.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection
        self._entity_key_gate = _SurrealDBEntityKeyGate(max_concurrent=10)

    # ------------------------------------------------------------------
    # Factory / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBGraphAdapter:
        """Create an adapter from a configuration dictionary.

        ``password`` is unwrapped from ``pydantic.SecretStr`` if needed so
        the driver receives a plaintext credential.
        """
        from pydantic import SecretStr

        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn_kwargs: dict[str, Any] = {}
        for key in ("mode", "path", "url", "namespace", "database", "user", "password"):
            if key in config:
                value = config[key]
                if key == "password" and isinstance(value, SecretStr):
                    value = value.get_secret_value()
                conn_kwargs[key] = value

        connection = SurrealDBConnection(**conn_kwargs)
        return cls(connection)

    async def connect(self) -> None:
        await self._conn.connect()
        logger.info("SurrealDBGraphAdapter connected")

    async def disconnect(self) -> None:
        await self._conn.disconnect()
        logger.info("SurrealDBGraphAdapter disconnected")

    async def is_healthy(self) -> bool:
        return await self._conn.is_healthy()

    def _get_session(self) -> None:
        """Compatibility shim expected by some callers.  Returns *None*."""
        return None

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    @trace("khora.surrealdb.graph.create_entity", include={"entity"})
    async def create_entity(self, entity: Entity) -> Entity:
        sql = (
            "CREATE $rid SET "
            "namespace = $ns_rid, "
            "name = $name, "
            "entity_type = $entity_type, "
            "description = $description, "
            "attributes = $attributes, "
            "source_document_ids = $source_document_ids, "
            "source_chunk_ids = $source_chunk_ids, "
            "source_tool = $source_tool, "
            "mention_count = $mention_count, "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "valid_from = $valid_from, "
            "valid_until = $valid_until, "
            "confidence = $confidence, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "updated_at = $updated_at"
        )
        await self._conn.execute(sql, _entity_to_bindings(entity))
        return entity

    @trace("khora.surrealdb.graph.get_entity", include={"entity_id", "namespace_id"})
    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Fetch an entity by primary key, scoped to ``namespace_id``.

        Returns ``None`` if the entity does not exist OR belongs to a
        different namespace.  ``RecordID`` lookup is not namespace-scoped
        on its own, so we filter explicitly on the entity's ``namespace``
        record link to prevent cross-tenant IDOR (IDOR family).
        """
        sql = (
            "SELECT * FROM entity WHERE id = $rid AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) LIMIT 1"
        )
        row = await self._conn.query_one(
            sql,
            {
                "rid": _rid("entity", entity_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        if not row:
            return None
        return _row_to_entity(row)

    @trace("khora.surrealdb.graph.get_entity_by_name", include={"namespace_id", "name", "entity_type"})
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        ns_rid = _rid("memory_namespace", namespace_id)
        sql = "SELECT * FROM entity WHERE namespace = $ns_rid AND name = $name AND entity_type = $entity_type LIMIT 1"
        row = await self._conn.query_one(sql, {"ns_rid": ns_rid, "name": name, "entity_type": entity_type})
        if not row:
            return None
        return _row_to_entity(row)

    @trace("khora.surrealdb.graph.update_entity", include={"entity"})
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id`` (IDOR family).

        The ``namespace_id`` kwarg is defense-in-depth \u2014 asserted equal to
        ``entity.namespace_id`` before the UPDATE filter is applied.
        """
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        sql = (
            "UPDATE $rid SET "
            "name = $name, "
            "entity_type = $entity_type, "
            "description = $description, "
            "attributes = $attributes, "
            "source_document_ids = $source_document_ids, "
            "source_chunk_ids = $source_chunk_ids, "
            "source_tool = $source_tool, "
            "mention_count = $mention_count, "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "valid_from = $valid_from, "
            "valid_until = $valid_until, "
            "confidence = $confidence, "
            "metadata_ = $metadata_, "
            "updated_at = $updated_at "
            "WHERE namespace = $ns_rid OR namespace.namespace_id = $ns_str"
        )
        bindings = _entity_to_bindings(entity)
        bindings.pop("created_at", None)
        bindings["ns_rid"] = _rid("memory_namespace", namespace_id)
        bindings["ns_str"] = str(namespace_id)
        await self._conn.execute(sql, bindings)
        return entity

    @trace("khora.surrealdb.graph.delete_entity", include={"entity_id"})
    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id`` (IDOR family)."""
        eid = _rid("entity", entity_id)
        ns_rid = _rid("memory_namespace", namespace_id)
        ns_str = str(namespace_id)
        # Delete relationships first, then the entity. Scoped by namespace.
        await self._conn.execute(
            "DELETE FROM relates_to "
            "WHERE (in = $eid OR out = $eid) "
            "AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str)",
            {"eid": eid, "ns_rid": ns_rid, "ns_str": ns_str},
        )
        deleted = await self._conn.query(
            "DELETE FROM entity "
            "WHERE id = $eid AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) "
            "RETURN BEFORE",
            {"eid": eid, "ns_rid": ns_rid, "ns_str": ns_str},
        )
        return len(deleted) > 0

    @trace(
        "khora.surrealdb.graph.list_entities",
        include={"namespace_id", "entity_type", "limit", "offset"},
        result=lambda r: {"count": len(r)},
    )
    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        ns_rid = _rid("memory_namespace", namespace_id)
        # Hide dream-retired entities unconditionally (#1280): the SurrealQL
        # native dream-apply stamps ``valid_until = time::now()`` on a
        # soft-deleted entity, and recall must filter it out in lockstep with
        # the PG / Neo4j read filters so the stores agree on the live set
        # (the P1-4 cross-store invariant). A future ``valid_until`` is still a
        # live temporal window; retirement stamps ``valid_until = now``, so the
        # ``> time::now()`` comparison hides it.
        where = ["namespace = $ns_rid", "(valid_until IS NONE OR valid_until > time::now())"]
        bindings: dict[str, Any] = {"ns_rid": ns_rid, "limit": limit, "offset": offset}
        if entity_type is not None:
            where.append("entity_type = $entity_type")
            bindings["entity_type"] = entity_type

        sql = f"SELECT * FROM entity WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT $limit START $offset"  # noqa: S608
        rows = await self._conn.query(sql, bindings)
        return [_row_to_entity(r) for r in rows]

    @trace(
        "khora.surrealdb.graph.get_entities_batch",
        include={"entity_ids", "namespace_id"},
        result=lambda r: {"count": len(r)},
    )
    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query, scoped to ``namespace_id``.

        Entities belonging to a different namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).
        """
        if not entity_ids:
            return {}
        # Bind RecordIDs as a parameter — bare ``entity:<uuid>`` interpolation
        # breaks SurrealQL's parser on the UUID hyphens (issue #635).
        eids = [_rid("entity", uid) for uid in entity_ids]
        sql = "SELECT * FROM entity WHERE id IN $eids AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str)"
        rows = await self._conn.query(
            sql,
            {
                "eids": eids,
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        result: dict[UUID, Entity] = {}
        for row in rows:
            ent = _row_to_entity(row)
            result[ent.id] = ent
        return result

    @trace("khora.surrealdb.graph.count_entities", include={"namespace_id"})
    async def count_entities(self, namespace_id: UUID) -> int:
        ns_rid = _rid("memory_namespace", namespace_id)
        sql = "SELECT count() AS cnt FROM entity WHERE namespace = $ns_rid GROUP ALL"
        row = await self._conn.query_one(sql, {"ns_rid": ns_rid})
        return int(row.get("cnt", 0)) if row else 0

    async def count_relationships(self, namespace_id: UUID) -> int:
        raise NotImplementedError

    @trace(
        "khora.surrealdb.graph.upsert_entities_batch",
        include={"namespace_id"},
        result=lambda r: {"count": len(r)},
    )
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 200,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using match-by (namespace, name, entity_type).

        For existing entities: merge descriptions, sum mention_counts, union source_ids.
        For new entities: create.
        Returns list of (Entity, is_new) tuples.

        Uses :class:`_SurrealDBEntityKeyGate` to prevent race conditions
        between the prefetch and write phases.
        """
        if not entities:
            return []

        ns_rid = _rid("memory_namespace", namespace_id)

        # Acquire exclusive access for these entity keys to prevent
        # concurrent upsert races (prefetch-compare-update pattern).
        async with self._entity_key_gate.acquire(entities):
            return await self._upsert_entities_batch_locked(ns_rid, namespace_id, entities)

    async def _upsert_entities_batch_locked(
        self,
        ns_rid: Any,
        namespace_id: UUID,
        entities: list[Entity],
    ) -> list[tuple[Entity, bool]]:
        """Inner upsert logic, called while holding the entity key gate."""
        # 1. Batch-fetch all existing entities matching any (name, type) pair.
        #    Uses SurrealDB tuple IN syntax: [name, entity_type] IN $pairs
        #    which is more efficient than N OR clauses for large batches.
        unique_pairs = list({(e.name, e.entity_type) for e in entities})
        fetch_sql = "SELECT * FROM entity WHERE namespace = $ns_rid AND [name, entity_type] IN $pairs"
        existing_rows = await self._conn.query(
            fetch_sql,
            {"ns_rid": ns_rid, "pairs": [list(p) for p in unique_pairs]},
        )

        # Index existing entities by (name, entity_type)
        existing_map: dict[tuple[str, str], Entity] = {}
        for row in existing_rows:
            ent = _row_to_entity(row)
            existing_map[(ent.name, ent.entity_type)] = ent

        # 2. Separate into creates vs updates and process in batches
        results: list[tuple[Entity, bool]] = []
        to_create: list[Entity] = []
        to_update: list[Entity] = []  # merged entities

        for entity in entities:
            key = (entity.name, entity.entity_type)
            existing = existing_map.get(key)
            if existing:
                existing.merge_with(entity)
                # Sync the input entity's id to the persisted canonical id
                # (#806, #1151). Callers hold references to the input
                # ``entities`` list and build ``Relationship`` endpoints
                # from ``entity.id`` after this call - the vectorcypher
                # engine discards the return value entirely. Without the
                # in-place mutation the subsequent RELATE targets
                # ``entity:<extraction-uuid>``, a record that does not
                # exist, silently dropping relationships on repeat ingest.
                # Neo4j and sqlite_lance do the same remap.
                entity.id = existing.id
                to_update.append(existing)
                results.append((existing, False))
            else:
                entity.namespace_id = namespace_id
                to_create.append(entity)
                results.append((entity, True))

        # 3. Batch create new entities via INSERT INTO (faster than FOR loops).
        #    INSERT INTO expects records with schema field names as keys.
        if to_create:
            records = []
            for e in to_create:
                b = _entity_to_bindings(e)
                records.append(
                    {
                        "id": b["rid"],  # RecordID
                        "namespace": b["ns_rid"],  # record<memory_namespace>
                        "name": b["name"],
                        "entity_type": b["entity_type"],
                        "description": b["description"],
                        "attributes": b["attributes"],
                        "source_document_ids": b["source_document_ids"],
                        "source_chunk_ids": b["source_chunk_ids"],
                        "source_tool": b["source_tool"],
                        "mention_count": b["mention_count"],
                        "embedding": b["embedding"],
                        "embedding_model": b["embedding_model"],
                        "valid_from": b["valid_from"],
                        "valid_until": b["valid_until"],
                        "confidence": b["confidence"],
                        "metadata_": b["metadata_"],
                        "created_at": b["created_at"],
                        "updated_at": b["updated_at"],
                    }
                )
            await self._conn.execute("INSERT INTO entity $records", {"records": records})

        # 4. Batch update existing entities in a single FOR statement
        if to_update:
            update_data = []
            for ent in to_update:
                update_data.append(
                    {
                        "rid": _rid("entity", ent.id),
                        "description": ent.description,
                        "attributes": ent.attributes or {},
                        "source_document_ids": [str(uid) for uid in ent.source_document_ids],
                        "source_chunk_ids": [str(uid) for uid in ent.source_chunk_ids],
                        "mention_count": ent.mention_count,
                        "confidence": ent.confidence,
                        "metadata_": ent.metadata or {},
                        "updated_at": ent.updated_at,
                    }
                )
            update_sql = (
                "FOR $e IN $entities {"
                "  UPDATE (type::thing($e.rid)) SET "
                "    description = $e.description, "
                "    attributes = $e.attributes, "
                "    source_document_ids = $e.source_document_ids, "
                "    source_chunk_ids = $e.source_chunk_ids, "
                "    mention_count = $e.mention_count, "
                "    confidence = $e.confidence, "
                "    metadata_ = $e.metadata_, "
                "    updated_at = $e.updated_at;"
                "}"
            )
            await self._conn.execute(update_sql, {"entities": update_data})

        return results

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    @trace("khora.surrealdb.graph.create_relationship", include={"relationship"})
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        # Normalise relationship_type the same way Cypher-based backends
        # do (issue #749).  Before this, SurrealDB stored the raw user
        # string verbatim while Neo4j / sqlite_lance / AGE upper-snake-
        # cased it — the same input read back as two different values
        # depending on backend.
        relationship.relationship_type = sanitize_cypher_label(relationship.relationship_type)

        src = _rid("entity", relationship.source_entity_id)
        tgt = _rid("entity", relationship.target_entity_id)

        # Bind RecordIDs as parameters — bare ``entity:<uuid>`` interpolation
        # breaks SurrealQL's parser on the UUID hyphens (issue #635).
        sql = (
            "RELATE $src->relates_to->$tgt SET "
            "rel_id = $rel_id, "
            "namespace_id = $namespace_id, "
            "relationship_type = $relationship_type, "
            "description = $description, "
            "properties = $properties, "
            "source_document_ids = $source_document_ids, "
            "source_chunk_ids = $source_chunk_ids, "
            "valid_from = $valid_from, "
            "valid_until = $valid_until, "
            "confidence = $confidence, "
            "weight = $weight, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "updated_at = $updated_at"
        )
        bindings = _relationship_to_bindings(relationship)
        bindings["src"] = src
        bindings["tgt"] = tgt
        await self._conn.execute(sql, bindings)
        return relationship

    @trace("khora.surrealdb.graph.get_relationship", include={"relationship_id", "namespace_id"})
    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Fetch a relationship by id, scoped to ``namespace_id``.

        Returns ``None`` if the relationship does not exist OR belongs to
        a different namespace.  Prevents cross-tenant relationship access
        by id (IDOR).
        """
        sql = "SELECT * FROM relates_to WHERE rel_id = $rel_id AND namespace_id = $ns LIMIT 1"
        row = await self._conn.query_one(
            sql,
            {"rel_id": str(relationship_id), "ns": str(namespace_id)},
        )
        if not row:
            return None
        return _row_to_relationship(row)

    @trace("khora.surrealdb.graph.delete_relationship", include={"relationship_id"})
    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id`` (IDOR family)."""
        # DELETE ... RETURN BEFORE returns deleted rows (empty if nothing matched)
        deleted = await self._conn.query(
            "DELETE FROM relates_to WHERE rel_id = $rel_id AND namespace_id = $ns RETURN BEFORE",
            {"rel_id": str(relationship_id), "ns": str(namespace_id)},
        )
        return len(deleted) > 0

    @trace(
        "khora.surrealdb.graph.get_entity_relationships",
        include={"entity_id", "namespace_id", "direction", "limit"},
        result=lambda r: {"count": len(r)},
    )
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Return relationships for an entity, scoped to ``namespace_id``.

        Filters at the SurrealQL layer on the relationship's own
        ``namespace_id`` column — edges that cross into another namespace
        do not surface even if the seed entity is shared.  Prevents
        cross-tenant subgraph leakage (IDOR family).
        """
        eid = _rid("entity", entity_id)
        bindings: dict[str, Any] = {
            "eid": eid,
            "ns": str(namespace_id),
            "limit": limit,
        }

        if direction == "outgoing":
            where = "in = $eid"
        elif direction == "incoming":
            where = "out = $eid"
        else:
            where = "(in = $eid OR out = $eid)"

        conditions = [where, "namespace_id = $ns"]

        if relationship_types:
            conditions.append("relationship_type IN $rel_types")
            bindings["rel_types"] = list(relationship_types)

        sql = f"SELECT * FROM relates_to WHERE {' AND '.join(conditions)} LIMIT $limit"  # noqa: S608
        rows = await self._conn.query(sql, bindings)
        return [_row_to_relationship(r) for r in rows]

    @trace(
        "khora.surrealdb.graph.list_relationships",
        include={"namespace_id", "relationship_type", "limit", "offset"},
        result=lambda r: {"count": len(r)},
    )
    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        # Hide dream-pruned / merged-self-loop edges unconditionally (#1280):
        # the SurrealQL native dream-apply stamps ``valid_until = time::now()``
        # on a soft-deleted edge, and recall must filter it out in lockstep with
        # the PG / Neo4j read filters so the stores agree on the live set (the
        # P1-4 cross-store invariant).
        conditions = ["namespace_id = $ns", "(valid_until IS NONE OR valid_until > time::now())"]
        bindings: dict[str, Any] = {"ns": str(namespace_id), "limit": limit, "offset": offset}

        if relationship_type is not None:
            conditions.append("relationship_type = $rt")
            bindings["rt"] = relationship_type

        sql = (
            f"SELECT * FROM relates_to WHERE {' AND '.join(conditions)} "  # noqa: S608
            "ORDER BY created_at DESC LIMIT $limit START $offset"
        )
        rows = await self._conn.query(sql, bindings)
        return [_row_to_relationship(r) for r in rows]

    @trace(
        "khora.surrealdb.graph.create_relationships_batch",
        result=lambda r: {"created": r},
    )
    async def create_relationships_batch(self, relationships: list[Relationship], *, batch_size: int = 200) -> int:
        if not relationships:
            return 0

        # Normalise each relationship_type in place so SurrealDB matches the
        # Cypher backends (issue #749).
        for rel in relationships:
            rel.relationship_type = sanitize_cypher_label(rel.relationship_type)

        # Build a batch array and execute all RELATEs in a single round-trip
        rels_data: list[dict[str, Any]] = []
        for rel in relationships:
            b = _relationship_to_bindings(rel)
            b["source_rid"] = _rid("entity", rel.source_entity_id)
            b["target_rid"] = _rid("entity", rel.target_entity_id)
            rels_data.append(b)

        sql = (
            "FOR $rel IN $rels {"
            "  RELATE (type::thing($rel.source_rid))->relates_to->(type::thing($rel.target_rid)) SET "
            "    rel_id = $rel.rel_id, "
            "    namespace_id = $rel.namespace_id, "
            "    relationship_type = $rel.relationship_type, "
            "    description = $rel.description, "
            "    properties = $rel.properties, "
            "    source_document_ids = $rel.source_document_ids, "
            "    source_chunk_ids = $rel.source_chunk_ids, "
            "    valid_from = $rel.valid_from, "
            "    valid_until = $rel.valid_until, "
            "    confidence = $rel.confidence, "
            "    weight = $rel.weight, "
            "    metadata_ = $rel.metadata_, "
            "    created_at = $rel.created_at, "
            "    updated_at = $rel.updated_at;"
            "}"
        )
        try:
            await self._conn.execute(sql, {"rels": rels_data})
            return len(relationships)
        except Exception:
            logger.warning("Batch relationship creation failed, falling back to individual inserts")
            created = 0
            failed = 0
            for rel in relationships:
                try:
                    await self.create_relationship(rel)
                    created += 1
                except Exception:
                    failed += 1
                    logger.warning(f"Failed to create relationship {rel.id}, skipping")
            logger.info(f"Relationship batch fallback: {created}/{len(relationships)} succeeded, {failed} failed")
            return created

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    @trace("khora.surrealdb.graph.create_episode", include={"episode"})
    async def create_episode(self, episode: Episode) -> Episode:
        sql = (
            "CREATE $rid SET "
            "namespace = $ns_rid, "
            "name = $name, "
            "description = $description, "
            "occurred_at = $occurred_at, "
            "duration_seconds = $duration_seconds, "
            "entity_ids = $entity_ids, "
            "source_document_ids = $source_document_ids, "
            "source_chunk_ids = $source_chunk_ids, "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "updated_at = $updated_at"
        )
        bindings = {
            "rid": _rid("episode", episode.id),
            "ns_rid": _rid("memory_namespace", episode.namespace_id),
            "name": episode.name,
            "description": episode.description,
            "occurred_at": episode.occurred_at,
            "duration_seconds": episode.duration_seconds,
            "entity_ids": [str(eid) for eid in episode.entity_ids],
            "source_document_ids": [str(d) for d in episode.source_document_ids],
            "source_chunk_ids": [str(c) for c in episode.source_chunk_ids],
            "embedding": list(episode.embedding) if episode.embedding is not None else None,
            "embedding_model": episode.embedding_model,
            "metadata_": episode.metadata or {},
            "created_at": episode.created_at,
            "updated_at": episode.updated_at,
        }
        await self._conn.execute(sql, bindings)

        # Create involvement edges: episode -> entity (single batch round-trip)
        if episode.entity_ids:
            involve_records = [{"eid_rid": _rid("entity", eid)} for eid in episode.entity_ids]
            involve_sql = (
                "FOR $r IN $records {"
                "  RELATE $ep_rid->involves->(type::thing($r.eid_rid)) "
                "    SET namespace_id = $ns, created_at = $created_at;"
                "}"
            )
            try:
                await self._conn.execute(
                    involve_sql,
                    {
                        "records": involve_records,
                        "ep_rid": _rid("episode", episode.id),
                        "ns": str(episode.namespace_id),
                        "created_at": episode.created_at,
                    },
                )
            except Exception:
                logger.warning(
                    f"Could not create involves edges for episode {episode.id} ({len(episode.entity_ids)} entities)"
                )

        return episode

    @trace("khora.surrealdb.graph.get_episode", include={"episode_id", "namespace_id"})
    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Fetch an episode by id, scoped to ``namespace_id``.

        Returns ``None`` if the episode does not exist OR belongs to a
        different namespace.  Prevents cross-tenant episode access by id
        (IDOR \u2014 the IDOR family).
        """
        sql = (
            "SELECT * FROM episode "
            "WHERE id = $rid AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) "
            "LIMIT 1"
        )
        row = await self._conn.query_one(
            sql,
            {
                "rid": _rid("episode", episode_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        if not row:
            return None
        return _row_to_episode(row)

    @trace(
        "khora.surrealdb.graph.list_episodes",
        include={"namespace_id", "limit"},
        result=lambda r: {"count": len(r)},
    )
    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        ns_rid = _rid("memory_namespace", namespace_id)
        conditions = ["namespace = $ns_rid"]
        bindings: dict[str, Any] = {"ns_rid": ns_rid, "limit": limit}

        if start_time is not None:
            conditions.append("occurred_at >= $start_time")
            bindings["start_time"] = start_time

        if end_time is not None:
            conditions.append("occurred_at <= $end_time")
            bindings["end_time"] = end_time

        sql = (
            f"SELECT * FROM episode WHERE {' AND '.join(conditions)} "  # noqa: S608
            "ORDER BY occurred_at DESC LIMIT $limit"
        )
        rows = await self._conn.query(sql, bindings)
        return [_row_to_episode(r) for r in rows]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @trace(
        "khora.surrealdb.graph.find_paths",
        include={"namespace_id", "source_entity_id", "target_entity_id", "max_depth"},
        result=lambda r: {"path_count": len(r)},
    )
    async def find_paths(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities using SurrealDB graph arrows.

        Fetches all depths (1..max_depth) in a **single query** by selecting
        multiple arrow chains as separate columns.  This avoids the N
        round-trips of the previous depth-loop approach.
        """
        src = _rid("entity", source_entity_id)
        tgt_id_str = str(target_entity_id)
        paths: list[list[dict[str, Any]]] = []

        # Build an optional relationship-type filter applied to each hop.
        rel_filter = ""
        rel_bindings: dict[str, Any] = {}
        if relationship_types:
            rel_filter = "[WHERE relationship_type IN $rel_types]"
            rel_bindings["rel_types"] = list(relationship_types)

        # Cap at 6 to avoid excessive query complexity in arrow-chain syntax.
        # For deeper traversals, consider iterative BFS at the application level.
        effective_max = min(max_depth, 6)
        hop = "->relates_to" + rel_filter + "->entity"

        # Build a single SELECT with one column per depth level.
        columns = ", ".join(f"{hop * d} AS d{d}" for d in range(1, effective_max + 1))
        sql = f"SELECT {columns} FROM $src"  # noqa: S608
        rel_bindings["src"] = src

        rows = await self._conn.query(sql, rel_bindings)
        if not rows:
            return paths

        row = rows[0] if isinstance(rows[0], dict) else {}

        # Check depths shortest-first so we report the shortest path.
        for depth in range(1, effective_max + 1):
            targets_raw = row.get(f"d{depth}")
            if targets_raw is None:
                continue

            flat_targets = self._flatten(targets_raw)
            for target_row in flat_targets:
                if not isinstance(target_row, dict):
                    continue
                tid = str(_parse_uuid(target_row.get("id", "")))
                if tid != tgt_id_str:
                    continue
                path: list[dict[str, Any]] = [{"type": "node", "data": {"id": str(source_entity_id)}}]
                for h in range(depth):
                    path.append({"type": "relationship", "data": {"hop": h + 1}})
                    if h < depth - 1:
                        path.append({"type": "node", "data": {"intermediate": True}})
                path.append({"type": "node", "data": {"id": tgt_id_str}})
                paths.append(path)
                return paths  # Shortest path found — done

        return paths

    @trace(
        "khora.surrealdb.graph.get_neighborhood",
        include={"entity_id", "namespace_id", "depth", "limit"},
        result=lambda r: {"node_count": len(r.get("entities", [])), "rel_count": len(r.get("relationships", []))},
    )
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return the neighborhood of an entity, scoped to ``namespace_id``.

        The seed entity must belong to ``namespace_id`` — otherwise an
        empty ``{"entities": [], "relationships": []}`` is returned.  Every
        edge and node visited during traversal is filtered on
        ``namespace_id`` so the result never crosses tenants
        (IDOR family).
        """
        # Gate the traversal on the seed entity belonging to the caller's
        # namespace.  Without this, a leaked entity_id from another tenant
        # would still surface that tenant's first-hop neighbours.
        seed = await self.get_entity(entity_id, namespace_id=namespace_id)
        if seed is None:
            return {"entities": [], "relationships": []}

        eid = _rid("entity", entity_id)
        ns_str = str(namespace_id)
        ns_rid = _rid("memory_namespace", namespace_id)

        rel_bindings: dict[str, Any] = {
            "eid": eid,
            "ns_str": ns_str,
            "ns_rid": ns_rid,
        }

        # Filter edges to those whose denormalized namespace_id matches and
        # optionally to selected relationship types.
        rel_conds = ["namespace_id = $ns_str"]
        if relationship_types:
            rel_conds.append("relationship_type IN $rel_types")
            rel_bindings["rel_types"] = list(relationship_types)
        rel_filter = "[WHERE " + " AND ".join(rel_conds) + "]"

        # Filter neighbour nodes to the same namespace at each hop.
        node_filter = "[WHERE namespace = $ns_rid OR namespace.namespace_id = $ns_str]"
        hop_out = "->relates_to" + rel_filter + "->entity" + node_filter
        hop_in = "<-relates_to" + rel_filter + "<-entity" + node_filter

        # Combine outgoing + incoming neighbor traversal in a single query.
        # NOTE: the center record is bound as $eid (not interpolated) — a bare
        # ``entity:<uuid>`` string makes SurrealQL's parser split on the UUID
        # hyphens and crash with ``Invalid token, found unexpected character``.
        # See issue #635.
        out_arrow = hop_out * depth
        in_arrow = hop_in * depth
        combined_sql = (
            f"SELECT {out_arrow} AS out_neighbors, "  # noqa: S608
            f"{in_arrow} AS in_neighbors "
            "FROM $eid"
        )
        rows = await self._conn.query(combined_sql, rel_bindings)

        # Collect unique entities from both directions
        seen_ids: set[str] = set()
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        if rows:
            for key in ("out_neighbors", "in_neighbors"):
                # Support old-style single-key response shape (test mocks)
                raw = rows[0].get(key) if isinstance(rows[0], dict) else None
                if raw is None:
                    continue
                flat = self._flatten(raw)
                for item in flat:
                    if not isinstance(item, dict):
                        continue
                    # Defensive: even with WHERE filtering at the SurrealQL
                    # layer, drop any node whose namespace does not match.
                    if not _row_in_namespace(item, ns_str):
                        continue
                    item_id = str(_parse_uuid(item.get("id", "")))
                    if item_id not in seen_ids and len(entities) < limit:
                        seen_ids.add(item_id)
                        entities.append(item)

        # Also handle flat entity rows returned directly (e.g. from mocks
        # that don't use the out_neighbors/in_neighbors shape)
        if rows and not entities:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "id" in row and "name" in row:
                    if not _row_in_namespace(row, ns_str):
                        continue
                    item_id = str(_parse_uuid(row.get("id", "")))
                    if item_id not in seen_ids and len(entities) < limit:
                        seen_ids.add(item_id)
                        entities.append(row)

        # Fetch relationships connecting the center to discovered neighbors.
        # Bind both the center and neighbour RecordIDs — same reason as above
        # (issue #635): bare interpolation breaks on UUID hyphens.
        if seen_ids:
            neighbor_rids = [_rid("entity", UUID(nid)) for nid in seen_ids]
            rel_sql = (
                "SELECT * FROM relates_to WHERE "
                "((in = $eid AND out IN $neighbor_rids) OR "
                "(out = $eid AND in IN $neighbor_rids)) "
                "AND namespace_id = $ns_str "
                "LIMIT $limit"
            )
            rel_rows = await self._conn.query(
                rel_sql,
                {
                    "eid": eid,
                    "neighbor_rids": neighbor_rids,
                    "ns_str": ns_str,
                    "limit": limit,
                },
            )
            relationships = rel_rows

        return {"entities": entities, "relationships": relationships}

    @trace(
        "khora.surrealdb.graph.get_neighborhoods_batch",
        include={"entity_ids", "namespace_id", "depth"},
        result=lambda r: {"count": len(r)},
    )
    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Return neighborhoods for many entities, scoped to ``namespace_id``.

        Seed entities outside ``namespace_id`` are silently dropped (the
        seed-set is intersected with the namespace via the same WHERE
        predicate used by :meth:`get_entity`).  Every hop is filtered on
        ``namespace_id`` so the traversal cannot leak into another
        tenant's subgraph (IDOR family).
        """
        if not entity_ids:
            return {}

        ns_str = str(namespace_id)
        ns_rid = _rid("memory_namespace", namespace_id)

        rel_bindings: dict[str, Any] = {
            "ns_str": ns_str,
            "ns_rid": ns_rid,
        }
        rel_conds = ["namespace_id = $ns_str"]
        if relationship_types:
            rel_conds.append("relationship_type IN $rel_types")
            rel_bindings["rel_types"] = list(relationship_types)
        rel_filter = "[WHERE " + " AND ".join(rel_conds) + "]"
        node_filter = "[WHERE namespace = $ns_rid OR namespace.namespace_id = $ns_str]"

        out_arrow = ("->relates_to" + rel_filter + "->entity" + node_filter) * depth
        in_arrow = ("<-relates_to" + rel_filter + "<-entity" + node_filter) * depth

        # Fetch all neighborhoods in a single query using an IN filter, with
        # the seed-set also intersected with the namespace so leaked-id
        # callers get nothing back.
        # Bind the entity RecordIDs — bare interpolation of ``entity:<uuid>``
        # makes SurrealQL parse the hyphens as arithmetic (issue #635).
        rel_bindings["eids"] = [_rid("entity", uid) for uid in entity_ids]
        batch_sql = (
            f"SELECT id, {out_arrow} AS out_neighbors, "  # noqa: S608
            f"{in_arrow} AS in_neighbors "
            "FROM entity "
            "WHERE id IN $eids "
            "AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str)"
        )
        try:
            rows = await self._conn.query(batch_sql, rel_bindings)
        except Exception:
            logger.warning("Batch neighborhood query failed, falling back to individual queries")
            result: dict[UUID, dict[str, Any]] = {}
            for eid in entity_ids:
                try:
                    neighborhood = await self.get_neighborhood(
                        eid,
                        namespace_id=namespace_id,
                        depth=depth,
                        relationship_types=relationship_types,
                        limit=limit_per_entity,
                    )
                    result[eid] = neighborhood
                except Exception:
                    logger.warning(f"Failed to get neighborhood for entity {eid}")
                    result[eid] = {"entities": [], "relationships": []}
            return result

        # Index rows by entity ID
        row_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict) and "id" in row:
                rid_str = str(_parse_uuid(row["id"]))
                row_by_id[rid_str] = row

        pending_rels: dict[UUID, tuple[Any, list[str]]] = {}
        result = {}
        for eid in entity_ids:
            eid_str = str(eid)
            row = row_by_id.get(eid_str)
            # Seed missing from result-set => not in this namespace; drop.
            if not row:
                result[eid] = {"entities": [], "relationships": []}
                continue

            seen_ids: set[str] = set()
            entities: list[dict[str, Any]] = []
            for key in ("out_neighbors", "in_neighbors"):
                raw = row.get(key)
                if raw is None:
                    continue
                flat = self._flatten(raw)
                for item in flat:
                    if not isinstance(item, dict):
                        continue
                    # Defensive: drop any node whose namespace does not match.
                    if not _row_in_namespace(item, ns_str):
                        continue
                    item_id = str(_parse_uuid(item.get("id", "")))
                    if item_id not in seen_ids and len(entities) < limit_per_entity:
                        seen_ids.add(item_id)
                        entities.append(item)

            # Defer relationship fetch — collect pairs for batch query
            if seen_ids:
                center_rid = _rid("entity", eid)
                pending_rels[eid] = (center_rid, list(seen_ids))

            result[eid] = {"entities": entities, "relationships": []}

        # Batch-fetch relationships for all entities at once.
        # Bind each center + its neighbour list via parameters — bare
        # interpolation of ``entity:<uuid>`` breaks SurrealQL's parser on the
        # UUID hyphens (issue #635).
        if pending_rels:
            or_conditions: list[str] = []
            rel_query_bindings: dict[str, Any] = {"rel_ns": ns_str}
            for idx, (eid, (center_rid, neighbor_ids)) in enumerate(pending_rels.items()):
                center_key = f"c{idx}"
                neighbor_key = f"n{idx}"
                rel_query_bindings[center_key] = center_rid
                rel_query_bindings[neighbor_key] = [_rid("entity", UUID(nid)) for nid in neighbor_ids]
                or_conditions.append(
                    f"((in = ${center_key} AND out IN ${neighbor_key}) OR "
                    f"(out = ${center_key} AND in IN ${neighbor_key}))"
                )

            if or_conditions:
                rel_query_bindings["rel_limit"] = limit_per_entity * len(pending_rels)
                batch_sql = (
                    "SELECT * FROM relates_to WHERE ("  # noqa: S608
                    + " OR ".join(or_conditions)
                    + ") AND namespace_id = $rel_ns LIMIT $rel_limit"
                )
                try:
                    all_rels = await self._conn.query(batch_sql, rel_query_bindings)
                    # Distribute relationships back to each entity
                    for rel in all_rels:
                        rel_in = str(_parse_uuid(rel.get("in", "")))
                        rel_out = str(_parse_uuid(rel.get("out", "")))
                        for eid, (center_rid, _) in pending_rels.items():
                            center_str = str(eid)
                            if rel_in == center_str or rel_out == center_str:
                                if len(result[eid]["relationships"]) < limit_per_entity:
                                    result[eid]["relationships"].append(rel)
                except Exception as e:
                    logger.debug(f"Failed to parse relationship record: {e}")

        return result

    @trace(
        "khora.surrealdb.graph.search_entities_by_attribute",
        include={"namespace_id", "attribute_name", "limit"},
        result=lambda r: {"count": len(r)},
    )
    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        ns_rid = _rid("memory_namespace", namespace_id)
        safe_attr = _sanitize_field_name(attribute_name)
        sql = (
            "SELECT * FROM entity "  # noqa: S608
            "WHERE namespace = $ns_rid "
            f"AND attributes.{safe_attr} = $attr_value "
            "LIMIT $limit"
        )
        rows = await self._conn.query(sql, {"ns_rid": ns_rid, "attr_value": attribute_value, "limit": limit})
        return [_row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Temporal traversal
    # ------------------------------------------------------------------

    @trace(
        "khora.surrealdb.graph.get_temporal_neighbors",
        include={"entity_id", "namespace_id", "max_hops", "limit"},
        result=lambda r: {"count": len(r)},
    )
    async def get_temporal_neighbors(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        valid_after: datetime | None = None,
        valid_before: datetime | None = None,
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities connected via relationships within a time window.

        Traverses 1..max_hops outgoing relationships from the entity and
        filters by valid_from / valid_until temporal constraints on those
        relationships.  Returns property dicts for the connected neighbor
        entities.

        Args:
            entity_id: Starting entity ID
            namespace_id: Namespace to restrict traversal to
            valid_after: Only include relationships where valid_from >= this
            valid_before: Only include relationships where valid_until <= this
            max_hops: Maximum path length (1-3 recommended)
            limit: Maximum neighbor entities to return

        Returns:
            List of neighbor entity property dicts
        """
        eid = _rid("entity", entity_id)
        effective_max = min(max_hops, 6)

        all_neighbors: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Build temporal filter for the relationship hop
        rel_conditions: list[str] = []
        bindings: dict[str, Any] = {"limit": limit}

        if valid_after is not None:
            rel_conditions.append("(valid_from IS NULL OR valid_from >= $valid_after)")
            bindings["valid_after"] = valid_after
        if valid_before is not None:
            rel_conditions.append("(valid_until IS NULL OR valid_until <= $valid_before)")
            bindings["valid_before"] = valid_before

        rel_filter = ""
        if rel_conditions:
            rel_filter = "[WHERE " + " AND ".join(rel_conditions) + "]"

        # Build a single SELECT with one column per depth level (same pattern as find_paths).
        # Bind ``$eid`` rather than interpolating — bare ``entity:<uuid>`` breaks
        # SurrealQL's parser on the UUID hyphens (issue #635).
        hop = "->relates_to" + rel_filter + "->entity"
        columns = ", ".join(f"{hop * d} AS d{d}" for d in range(1, effective_max + 1))
        sql = f"SELECT {columns} FROM $eid"  # noqa: S608
        bindings["eid"] = eid

        rows = await self._conn.query(sql, bindings)
        if not rows:
            return all_neighbors

        row = rows[0] if isinstance(rows[0], dict) else {}

        # Process depths in order (closer neighbors first)
        for depth in range(1, effective_max + 1):
            targets_raw = row.get(f"d{depth}")
            if targets_raw is None:
                continue

            flat = self._flatten(targets_raw)
            for item in flat:
                if not isinstance(item, dict):
                    continue
                # Filter by namespace
                item_ns = str(_parse_uuid(item.get("namespace", "")))
                if item_ns != str(namespace_id):
                    continue
                item_id = str(_parse_uuid(item.get("id", "")))
                if item_id == str(entity_id):
                    continue
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    all_neighbors.append(item)
                    if len(all_neighbors) >= limit:
                        return all_neighbors

        return all_neighbors

    @trace(
        "khora.surrealdb.graph.create_session_links",
        include={"namespace_id"},
        result=lambda r: {"created": r},
    )
    async def create_session_links(self, namespace_id: UUID) -> int:
        """Create next_session edges between consecutive session chunks.

        Reads chunks from the namespace, groups them by ``session_id``
        stored in their metadata, orders sessions by earliest chunk
        timestamp, and RELATEs the last chunk of session A to the first
        chunk of session B via the ``next_session`` relation table.

        Args:
            namespace_id: Namespace to process

        Returns:
            Number of next_session edges created
        """
        ns_rid = _rid("memory_namespace", namespace_id)

        # 1. Fetch all chunks with their metadata.
        # Bind the namespace RecordID — bare ``memory_namespace:<uuid>``
        # interpolation breaks SurrealQL's parser on UUID hyphens (issue #635).
        rows = await self._conn.query(
            "SELECT id, metadata_, created_at, source_timestamp FROM chunk "
            "WHERE namespace = $ns_rid "
            "ORDER BY (source_timestamp ?? created_at) ASC",
            {"ns_rid": ns_rid},
        )
        if not rows:
            return 0

        # 2. Group by session_id from metadata
        sessions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            meta = row.get("metadata_") or {}
            if not isinstance(meta, dict):
                continue
            session_id = meta.get("session_id")
            if not session_id:
                continue
            sessions.setdefault(str(session_id), []).append(row)

        if len(sessions) < 2:
            return 0

        # 3. Sort sessions by earliest chunk timestamp
        def _earliest_ts(chunks: list[dict[str, Any]]) -> str:
            timestamps = []
            for c in chunks:
                ts = c.get("source_timestamp") or c.get("created_at")
                if ts is not None:
                    timestamps.append(str(ts))
            return min(timestamps) if timestamps else ""

        ordered_sessions = sorted(sessions.values(), key=_earliest_ts)

        # 4. Build link pairs: last chunk of session A -> first chunk of session B
        links: list[dict[str, Any]] = []
        for i in range(len(ordered_sessions) - 1):
            from_chunk = ordered_sessions[i][-1]
            to_chunk = ordered_sessions[i + 1][0]
            from_uuid = _parse_uuid(from_chunk["id"])
            to_uuid = _parse_uuid(to_chunk["id"])
            links.append({"from_rid": _rid("chunk", from_uuid), "to_rid": _rid("chunk", to_uuid)})

        if not links:
            return 0

        # 5. Create next_session edges in a single batch round-trip
        ns_str = str(namespace_id)
        now_iso = datetime.now(UTC)
        sql = (
            "FOR $link IN $links {"
            "  RELATE (type::thing($link.from_rid))->next_session->(type::thing($link.to_rid)) SET "
            "    namespace_id = $ns, "
            "    created_at = $created_at;"
            "}"
        )
        await self._conn.execute(
            sql,
            {
                "links": links,
                "ns": ns_str,
                "created_at": now_iso,
            },
        )

        logger.debug(f"Created {len(links)} next_session edges for namespace {namespace_id}")
        return len(links)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(data: Any) -> list[Any]:
        """Recursively flatten nested lists (SurrealDB graph traversal results)."""
        if isinstance(data, list):
            result: list[Any] = []
            for item in data:
                if isinstance(item, list):
                    result.extend(SurrealDBGraphAdapter._flatten(item))
                else:
                    result.append(item)
            return result
        return [data]
