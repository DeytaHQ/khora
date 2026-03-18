"""SurrealDB graph adapter for Khora.

Implements GraphBackendProtocol using SurrealDB's native graph traversal
(RELATE, ``->`` / ``<-`` arrows) and record-link IDs.  All record IDs
follow the ``table:⟨uuid⟩`` convention expected by the unified SurrealDB
schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.surrealdb._helpers import (
    _entity_to_bindings,
    _iso,
    _parse_dt,
    _parse_uuid,
    _rid,
    _row_to_entity,
)
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection


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
        "valid_from": _iso(rel.valid_from),
        "valid_until": _iso(rel.valid_until),
        "confidence": rel.confidence,
        "weight": rel.weight,
        "metadata_": rel.metadata or {},
        "created_at": _iso(rel.created_at),
        "updated_at": _iso(rel.updated_at),
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SurrealDBGraphAdapter:
    """Graph backend backed by SurrealDB.

    Uses SurrealDB ``RELATE`` statements for edges, record-link IDs for
    traversal, and the ``->`` / ``<-`` arrow operators for path queries.

    The adapter delegates all I/O to a :class:`SurrealDBConnection`,
    which manages client lifecycle and authentication.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Factory / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBGraphAdapter:
        """Create an adapter from a configuration dictionary."""
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn_kwargs: dict[str, Any] = {}
        for key in ("mode", "path", "url", "namespace", "database", "user", "password"):
            if key in config:
                conn_kwargs[key] = config[key]

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
            "CREATE entity:\u27e8$id\u27e9 SET "
            "namespace = memory_namespace:\u27e8$ns\u27e9, "
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

    @trace("khora.surrealdb.graph.get_entity", include={"entity_id"})
    async def get_entity(self, entity_id: UUID) -> Entity | None:
        sql = "SELECT * FROM entity:\u27e8$id\u27e9"
        row = await self._conn.query_one(sql, {"id": str(entity_id)})
        if not row:
            return None
        return _row_to_entity(row)

    @trace("khora.surrealdb.graph.get_entity_by_name", include={"namespace_id", "name", "entity_type"})
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        sql = (
            "SELECT * FROM entity "
            f"WHERE namespace = memory_namespace:\u27e8{namespace_id}\u27e9 "
            "AND name = $name AND entity_type = $entity_type LIMIT 1"
        )
        row = await self._conn.query_one(sql, {"name": name, "entity_type": entity_type})
        if not row:
            return None
        return _row_to_entity(row)

    @trace("khora.surrealdb.graph.update_entity", include={"entity"})
    async def update_entity(self, entity: Entity) -> Entity:
        sql = (
            "UPDATE entity:\u27e8$id\u27e9 SET "
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
            "updated_at = $updated_at"
        )
        bindings = _entity_to_bindings(entity)
        bindings.pop("created_at", None)
        await self._conn.execute(sql, bindings)
        return entity

    @trace("khora.surrealdb.graph.delete_entity", include={"entity_id"})
    async def delete_entity(self, entity_id: UUID) -> bool:
        # Check existence first
        check = await self._conn.query_one(
            "SELECT count() AS cnt FROM entity WHERE id = entity:\u27e8$id\u27e9 GROUP ALL",
            {"id": str(entity_id)},
        )
        if not check or int(check.get("cnt", 0)) == 0:
            return False

        eid = _rid("entity", entity_id)
        # Delete relationships referencing this entity
        await self._conn.execute(f"DELETE FROM relates_to WHERE in = {eid} OR out = {eid}")
        # Delete the entity itself
        await self._conn.execute(f"DELETE FROM entity WHERE id = {eid}")
        return True

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
        where = [f"namespace = memory_namespace:\u27e8{namespace_id}\u27e9"]
        bindings: dict[str, Any] = {"limit": limit, "offset": offset}
        if entity_type is not None:
            where.append("entity_type = $entity_type")
            bindings["entity_type"] = entity_type

        sql = f"SELECT * FROM entity WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT $limit START $offset"
        rows = await self._conn.query(sql, bindings)
        return [_row_to_entity(r) for r in rows]

    @trace(
        "khora.surrealdb.graph.get_entities_batch",
        include={"entity_ids"},
        result=lambda r: {"count": len(r)},
    )
    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        if not entity_ids:
            return {}
        ids_list = ", ".join(_rid("entity", uid) for uid in entity_ids)
        sql = f"SELECT * FROM entity WHERE id IN [{ids_list}]"
        rows = await self._conn.query(sql)
        result: dict[UUID, Entity] = {}
        for row in rows:
            ent = _row_to_entity(row)
            result[ent.id] = ent
        return result

    @trace("khora.surrealdb.graph.count_entities", include={"namespace_id"})
    async def count_entities(self, namespace_id: UUID) -> int:
        sql = (
            f"SELECT count() AS cnt FROM entity WHERE namespace = memory_namespace:\u27e8{namespace_id}\u27e9 GROUP ALL"
        )
        row = await self._conn.query_one(sql)
        return int(row.get("cnt", 0)) if row else 0

    @trace(
        "khora.surrealdb.graph.upsert_entities_batch",
        include={"namespace_id"},
        result=lambda r: {"count": len(r)},
    )
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using match-by (namespace, name, entity_type).

        For existing entities: merge descriptions, sum mention_counts, union source_ids.
        For new entities: create.
        Returns list of (Entity, is_new) tuples.
        """
        if not entities:
            return []

        results: list[tuple[Entity, bool]] = []
        ns_rid = _rid("memory_namespace", namespace_id)

        for entity in entities:
            # Check if entity already exists by (namespace, name, entity_type)
            existing_sql = (
                f"SELECT * FROM entity WHERE namespace = {ns_rid} "
                "AND name = $name AND entity_type = $entity_type LIMIT 1"
            )
            existing_row = await self._conn.query_one(
                existing_sql, {"name": entity.name, "entity_type": entity.entity_type}
            )

            if existing_row:
                # Merge into existing entity
                existing = _row_to_entity(existing_row)
                existing.merge_with(entity)

                update_sql = (
                    f"UPDATE {_rid('entity', existing.id)} SET "
                    "description = $description, "
                    "attributes = $attributes, "
                    "source_document_ids = $source_document_ids, "
                    "source_chunk_ids = $source_chunk_ids, "
                    "mention_count = $mention_count, "
                    "confidence = $confidence, "
                    "metadata_ = $metadata_, "
                    "updated_at = $updated_at"
                )
                await self._conn.execute(
                    update_sql,
                    {
                        "description": existing.description,
                        "attributes": existing.attributes or {},
                        "source_document_ids": [str(uid) for uid in existing.source_document_ids],
                        "source_chunk_ids": [str(uid) for uid in existing.source_chunk_ids],
                        "mention_count": existing.mention_count,
                        "confidence": existing.confidence,
                        "metadata_": existing.metadata or {},
                        "updated_at": _iso(existing.updated_at),
                    },
                )
                results.append((existing, False))
            else:
                # Ensure namespace_id matches
                entity.namespace_id = namespace_id
                await self.create_entity(entity)
                results.append((entity, True))

        return results

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    @trace("khora.surrealdb.graph.create_relationship", include={"relationship"})
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        src = _rid("entity", relationship.source_entity_id)
        tgt = _rid("entity", relationship.target_entity_id)

        sql = (
            f"RELATE {src}->relates_to->{tgt} SET "
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
        await self._conn.execute(sql, _relationship_to_bindings(relationship))
        return relationship

    @trace("khora.surrealdb.graph.get_relationship", include={"relationship_id"})
    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        sql = "SELECT * FROM relates_to WHERE rel_id = $rel_id LIMIT 1"
        row = await self._conn.query_one(sql, {"rel_id": str(relationship_id)})
        if not row:
            return None
        return _row_to_relationship(row)

    @trace("khora.surrealdb.graph.delete_relationship", include={"relationship_id"})
    async def delete_relationship(self, relationship_id: UUID) -> bool:
        # Check existence
        check = await self._conn.query_one(
            "SELECT count() AS cnt FROM relates_to WHERE rel_id = $rel_id GROUP ALL",
            {"rel_id": str(relationship_id)},
        )
        if not check or int(check.get("cnt", 0)) == 0:
            return False
        await self._conn.execute("DELETE FROM relates_to WHERE rel_id = $rel_id", {"rel_id": str(relationship_id)})
        return True

    @trace(
        "khora.surrealdb.graph.get_entity_relationships",
        include={"entity_id", "direction", "limit"},
        result=lambda r: {"count": len(r)},
    )
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        eid = _rid("entity", entity_id)

        if direction == "outgoing":
            where = f"in = {eid}"
        elif direction == "incoming":
            where = f"out = {eid}"
        else:
            where = f"(in = {eid} OR out = {eid})"

        conditions = [where]
        bindings: dict[str, Any] = {"limit": limit}

        if relationship_types:
            placeholders = ", ".join(f"$rt_{i}" for i in range(len(relationship_types)))
            conditions.append(f"relationship_type IN [{placeholders}]")
            for i, rt in enumerate(relationship_types):
                bindings[f"rt_{i}"] = rt

        sql = f"SELECT * FROM relates_to WHERE {' AND '.join(conditions)} LIMIT $limit"
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
        conditions = ["namespace_id = $ns"]
        bindings: dict[str, Any] = {"ns": str(namespace_id), "limit": limit, "offset": offset}

        if relationship_type is not None:
            conditions.append("relationship_type = $rt")
            bindings["rt"] = relationship_type

        sql = (
            f"SELECT * FROM relates_to WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at DESC LIMIT $limit START $offset"
        )
        rows = await self._conn.query(sql, bindings)
        return [_row_to_relationship(r) for r in rows]

    @trace(
        "khora.surrealdb.graph.create_relationships_batch",
        result=lambda r: {"created": r},
    )
    async def create_relationships_batch(self, relationships: list[Relationship]) -> int:
        if not relationships:
            return 0

        created = 0
        for rel in relationships:
            try:
                await self.create_relationship(rel)
                created += 1
            except Exception:
                logger.warning(f"Failed to create relationship {rel.id}, skipping")
        return created

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    @trace("khora.surrealdb.graph.create_episode", include={"episode"})
    async def create_episode(self, episode: Episode) -> Episode:
        sql = (
            "CREATE episode:\u27e8$id\u27e9 SET "
            "namespace = memory_namespace:\u27e8$ns\u27e9, "
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
            "id": str(episode.id),
            "ns": str(episode.namespace_id),
            "name": episode.name,
            "description": episode.description,
            "occurred_at": _iso(episode.occurred_at),
            "duration_seconds": episode.duration_seconds,
            "entity_ids": [str(eid) for eid in episode.entity_ids],
            "source_document_ids": [str(d) for d in episode.source_document_ids],
            "source_chunk_ids": [str(c) for c in episode.source_chunk_ids],
            "embedding": list(episode.embedding) if episode.embedding is not None else None,
            "embedding_model": episode.embedding_model,
            "metadata_": episode.metadata or {},
            "created_at": _iso(episode.created_at),
            "updated_at": _iso(episode.updated_at),
        }
        await self._conn.execute(sql, bindings)

        # Create involvement edges: episode -> entity
        for eid in episode.entity_ids:
            involve_sql = (
                f"RELATE episode:\u27e8{episode.id}\u27e9->involves->{_rid('entity', eid)} "
                f"SET namespace_id = $ns, created_at = $created_at"
            )
            try:
                await self._conn.execute(
                    involve_sql,
                    {
                        "ns": str(episode.namespace_id),
                        "created_at": _iso(episode.created_at),
                    },
                )
            except Exception:
                logger.debug(f"Could not create involves edge for episode {episode.id} -> entity {eid}")

        return episode

    @trace("khora.surrealdb.graph.get_episode", include={"episode_id"})
    async def get_episode(self, episode_id: UUID) -> Episode | None:
        sql = "SELECT * FROM episode:\u27e8$id\u27e9"
        row = await self._conn.query_one(sql, {"id": str(episode_id)})
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
        conditions = [f"namespace = memory_namespace:\u27e8{namespace_id}\u27e9"]
        bindings: dict[str, Any] = {"limit": limit}

        if start_time is not None:
            conditions.append("occurred_at >= $start_time")
            bindings["start_time"] = _iso(start_time)

        if end_time is not None:
            conditions.append("occurred_at <= $end_time")
            bindings["end_time"] = _iso(end_time)

        sql = f"SELECT * FROM episode WHERE {' AND '.join(conditions)} " "ORDER BY occurred_at DESC LIMIT $limit"
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
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities using chained SurrealDB graph arrows.

        For each depth from 1..max_depth, issues a query that chains
        ``->relates_to->entity`` arrows.  Results are de-duplicated and
        filtered by namespace and optional relationship types.
        """
        src = _rid("entity", source_entity_id)
        tgt_id_str = str(target_entity_id)
        paths: list[list[dict[str, Any]]] = []

        # Build an optional relationship-type filter applied to each hop.
        rel_filter = ""
        if relationship_types:
            rt_list = ", ".join(f"'{rt}'" for rt in relationship_types)
            rel_filter = f"[WHERE relationship_type IN [{rt_list}]]"

        for depth in range(1, min(max_depth, 3) + 1):
            # Build chained arrow expression for this depth.
            # e.g. depth=2: ->relates_to[filter]->entity->relates_to[filter]->entity
            arrow_chain = ("->relates_to" + rel_filter + "->entity") * depth

            sql = f"SELECT {arrow_chain} AS targets FROM {src}"
            rows = await self._conn.query(sql)
            if not rows:
                continue

            # The result contains nested arrays; we need to find the target
            targets_raw = rows[0].get("targets") if rows else None
            if targets_raw is None:
                continue

            # Flatten nested lists to find matching target entities
            flat_targets = self._flatten(targets_raw)
            for target_row in flat_targets:
                if not isinstance(target_row, dict):
                    continue
                tid = str(_parse_uuid(target_row.get("id", "")))
                if tid != tgt_id_str:
                    continue
                # Build a minimal path representation
                path: list[dict[str, Any]] = [{"type": "node", "data": {"id": str(source_entity_id)}}]
                for hop in range(depth):
                    path.append({"type": "relationship", "data": {"hop": hop + 1}})
                    if hop < depth - 1:
                        path.append({"type": "node", "data": {"intermediate": True}})
                path.append({"type": "node", "data": {"id": tgt_id_str}})
                paths.append(path)
                break  # One path per depth level is enough

        return paths

    @trace(
        "khora.surrealdb.graph.get_neighborhood",
        include={"entity_id", "depth", "limit"},
        result=lambda r: {"node_count": len(r.get("entities", [])), "rel_count": len(r.get("relationships", []))},
    )
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        eid = _rid("entity", entity_id)

        # Collect outgoing neighbors
        rel_filter = ""
        if relationship_types:
            rt_list = ", ".join(f"'{rt}'" for rt in relationship_types)
            rel_filter = f"[WHERE relationship_type IN [{rt_list}]]"

        # Outgoing: entity->relates_to->entity (chained for depth)
        out_arrow = ("->relates_to" + rel_filter + "->entity") * depth
        out_sql = f"SELECT {out_arrow} AS neighbors FROM {eid}"

        # Incoming: entity<-relates_to<-entity (chained for depth)
        in_arrow = ("<-relates_to" + rel_filter + "<-entity") * depth
        in_sql = f"SELECT {in_arrow} AS neighbors FROM {eid}"

        out_rows = await self._conn.query(out_sql)
        in_rows = await self._conn.query(in_sql)

        # Collect unique entities
        seen_ids: set[str] = set()
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for rows in (out_rows, in_rows):
            if not rows:
                continue
            raw = rows[0].get("neighbors")
            if raw is None:
                continue
            flat = self._flatten(raw)
            for item in flat:
                if not isinstance(item, dict):
                    continue
                item_id = str(_parse_uuid(item.get("id", "")))
                if item_id not in seen_ids and len(entities) < limit:
                    seen_ids.add(item_id)
                    entities.append(item)

        # Fetch relationships connecting the center to discovered neighbors
        if seen_ids:
            neighbor_rids = ", ".join(_rid("entity", UUID(nid)) for nid in seen_ids)
            rel_sql = (
                f"SELECT * FROM relates_to WHERE "
                f"(in = {eid} AND out IN [{neighbor_rids}]) OR "
                f"(out = {eid} AND in IN [{neighbor_rids}]) "
                f"LIMIT {limit}"
            )
            rel_rows = await self._conn.query(rel_sql)
            relationships = rel_rows

        return {"entities": entities, "relationships": relationships}

    @trace(
        "khora.surrealdb.graph.get_neighborhoods_batch",
        include={"entity_ids", "depth"},
        result=lambda r: {"count": len(r)},
    )
    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        if not entity_ids:
            return {}

        result: dict[UUID, dict[str, Any]] = {}
        for eid in entity_ids:
            try:
                neighborhood = await self.get_neighborhood(
                    eid,
                    depth=depth,
                    relationship_types=relationship_types,
                    limit=limit_per_entity,
                )
                result[eid] = neighborhood
            except Exception:
                logger.warning(f"Failed to get neighborhood for entity {eid}")
                result[eid] = {"entities": [], "relationships": []}
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
        sql = (
            f"SELECT * FROM entity "
            f"WHERE namespace = memory_namespace:\u27e8{namespace_id}\u27e9 "
            f"AND attributes.{attribute_name} = $attr_value "
            "LIMIT $limit"
        )
        rows = await self._conn.query(sql, {"attr_value": attribute_value, "limit": limit})
        return [_row_to_entity(r) for r in rows]

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
