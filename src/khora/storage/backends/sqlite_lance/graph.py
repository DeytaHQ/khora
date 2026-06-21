"""SQLite graph adapter for the embedded SQLite + LanceDB backend.

Implements :class:`khora.storage.backends.base.GraphBackendProtocol` on
raw SQL over the ``entities``, ``relationships`` and ``episodes`` tables
created by the Alembic migrations.  Traversal is expressed as
SQLite recursive CTEs — no Cypher, no graph engine required.

Vector storage is **not** handled here: entity embeddings live in
LanceDB and are managed by :class:`SQLiteLanceVectorAdapter`.  This
adapter never reads or writes the ``embedding`` column (it isn't even
present in the SQLite schema).

Concurrency safety for :meth:`upsert_entities_batch` is provided by
:class:`_SQLiteLanceEntityKeyGate`, which is a verbatim port of the
SurrealDB / Neo4j entity key gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.mixins import GraphBackendBase, sanitize_cypher_label

from ._entity_gate import _SQLiteLanceEntityKeyGate
from ._helpers import from_json_text, iso8601, to_json_text, uuid_to_text

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


# ---------------------------------------------------------------------------
# Row → domain helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime value from a SQLite TEXT column."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _parse_uuid_list(value: Any) -> list[UUID]:
    """Parse a JSON-encoded list of UUID strings from a TEXT column."""
    if value is None:
        return []
    if isinstance(value, list):
        return [UUID(str(v)) for v in value]
    try:
        import json

        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [UUID(str(v)) for v in parsed]


def _parse_json_dict(value: Any) -> dict[str, Any]:
    """Parse a JSON-encoded object from a TEXT column (``{}`` on failure)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return from_json_text(value)
    except (TypeError, ValueError):
        return {}


def _row_to_entity(row: Any) -> Entity:
    """Map an aiosqlite ``Row`` to a domain :class:`Entity`.

    Column layout follows the Alembic-generated ``entities`` table
    (JSON columns are stored as TEXT).  The SQLite schema has no
    ``embedding`` column — vectors live in LanceDB — so the returned
    Entity always has ``embedding=None`` here.
    """
    return Entity(
        id=UUID(row["id"]),
        namespace_id=UUID(row["namespace_id"]),
        name=row["name"],
        entity_type=row["entity_type"],
        description=row["description"] or "",
        attributes=_parse_json_dict(row["attributes"]),
        source_document_ids=_parse_uuid_list(row["source_document_ids"]),
        source_chunk_ids=_parse_uuid_list(row["source_chunk_ids"]),
        mention_count=int(row["mention_count"] or 1),
        embedding=None,
        embedding_model=row["embedding_model"] or "",
        valid_from=_parse_dt(row["valid_from"]),
        valid_until=_parse_dt(row["valid_until"]),
        confidence=float(row["confidence"] or 1.0),
        metadata=_parse_json_dict(row["metadata"]),
        created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
    )


def _row_to_relationship(row: Any) -> Relationship:
    """Map an aiosqlite ``Row`` to a domain :class:`Relationship`."""
    return Relationship(
        id=UUID(row["id"]),
        namespace_id=UUID(row["namespace_id"]),
        source_entity_id=UUID(row["source_entity_id"]),
        target_entity_id=UUID(row["target_entity_id"]),
        relationship_type=row["relationship_type"] or "RELATES_TO",
        description=row["description"] or "",
        properties=_parse_json_dict(row["properties"]),
        source_document_ids=_parse_uuid_list(row["source_document_ids"]),
        source_chunk_ids=_parse_uuid_list(row["source_chunk_ids"]),
        valid_from=_parse_dt(row["valid_from"]),
        valid_until=_parse_dt(row["valid_until"]),
        confidence=float(row["confidence"] or 1.0),
        weight=float(row["weight"] or 1.0),
        metadata=_parse_json_dict(row["metadata"]),
        created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
    )


def _row_to_episode(row: Any) -> Episode:
    """Map an aiosqlite ``Row`` to a domain :class:`Episode`."""
    return Episode(
        id=UUID(row["id"]),
        namespace_id=UUID(row["namespace_id"]),
        name=row["name"],
        description=row["description"] or "",
        occurred_at=_parse_dt(row["occurred_at"]) or datetime.now(UTC),
        duration_seconds=row["duration_seconds"],
        entity_ids=_parse_uuid_list(row["entity_ids"]),
        source_document_ids=_parse_uuid_list(row["source_document_ids"]),
        source_chunk_ids=_parse_uuid_list(row["source_chunk_ids"]),
        embedding=None,
        embedding_model=row["embedding_model"] or "",
        metadata=_parse_json_dict(row["metadata"]),
        created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
    )


def _entity_insert_params(entity: Entity) -> tuple:
    """Parameter tuple for an ``INSERT INTO entities`` statement."""
    return (
        uuid_to_text(entity.id),
        uuid_to_text(entity.namespace_id),
        entity.name,
        entity.entity_type,
        entity.description,
        to_json_text(entity.attributes or {}),
        to_json_text([uuid_to_text(d) for d in entity.source_document_ids]),
        to_json_text([uuid_to_text(c) for c in entity.source_chunk_ids]),
        entity.mention_count,
        entity.embedding_model,
        iso8601(entity.valid_from),
        iso8601(entity.valid_until),
        entity.confidence,
        to_json_text(entity.metadata or {}),
        iso8601(entity.created_at) or datetime.now(UTC).isoformat(),
        iso8601(entity.updated_at) or datetime.now(UTC).isoformat(),
    )


def _entity_update_params(entity: Entity) -> tuple:
    """Parameter tuple for an ``UPDATE entities`` statement (matched by id + namespace_id)."""
    now_iso = datetime.now(UTC).isoformat()
    return (
        entity.name,
        entity.entity_type,
        entity.description,
        to_json_text(entity.attributes or {}),
        to_json_text([uuid_to_text(d) for d in entity.source_document_ids]),
        to_json_text([uuid_to_text(c) for c in entity.source_chunk_ids]),
        entity.mention_count,
        entity.embedding_model,
        iso8601(entity.valid_from),
        iso8601(entity.valid_until),
        entity.confidence,
        to_json_text(entity.metadata or {}),
        iso8601(entity.updated_at) or now_iso,
        uuid_to_text(entity.id),
        uuid_to_text(entity.namespace_id),
    )


_ENTITY_INSERT_SQL = (
    "INSERT INTO entities ("
    "id, namespace_id, name, entity_type, description, attributes, "
    "source_document_ids, source_chunk_ids, mention_count, embedding_model, "
    "valid_from, valid_until, confidence, metadata, created_at, updated_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_ENTITY_UPDATE_SQL = (
    "UPDATE entities SET "
    "name = ?, entity_type = ?, description = ?, attributes = ?, "
    "source_document_ids = ?, source_chunk_ids = ?, mention_count = ?, "
    "embedding_model = ?, valid_from = ?, valid_until = ?, confidence = ?, "
    "metadata = ?, updated_at = ? WHERE id = ? AND namespace_id = ?"
)

_ENTITY_COLUMNS = (
    "id, namespace_id, name, entity_type, description, attributes, "
    "source_document_ids, source_chunk_ids, mention_count, embedding_model, "
    "valid_from, valid_until, confidence, metadata, created_at, updated_at"
)

_RELATIONSHIP_COLUMNS = (
    "id, namespace_id, source_entity_id, target_entity_id, relationship_type, "
    "description, properties, source_document_ids, source_chunk_ids, "
    "valid_from, valid_until, confidence, weight, metadata, created_at, updated_at"
)


# Live-set filters for the read paths (#1277). Soft-deleted rows stamped by the
# dream apply handlers must drop from the live set the recall path sees, in
# lockstep with the pgvector backend's ``_entity_live_filter`` /
# ``_relationship_live_filter`` and the PG ground-truth used by the cross-store
# invariant gate. ``datetime(...)`` normalizes the stored ISO string (space- or
# T-separated, with a ``+00:00`` offset) and ``datetime('now')`` is UTC, so the
# validity-window comparison is format- and timezone-robust on SQLite.
#
# Entities carry only ``valid_until`` (migration 033 added ``valid_to`` /
# ``invalidated_at`` to relationships + memory_facts, not entities). Dream-apply
# retires an absorbed entity by stamping ``valid_until = NOW()``.
_ENTITY_LIVE_FILTER = "(valid_until IS NULL OR datetime(valid_until) > datetime('now'))"
# Relationships additionally honor the bi-temporal soft-delete tombstones:
# prune_edges stamps ``valid_to`` and dedupe self-loop invalidation stamps
# ``invalidated_at``.
_RELATIONSHIP_LIVE_FILTER = (
    "valid_to IS NULL AND invalidated_at IS NULL AND (valid_until IS NULL OR datetime(valid_until) > datetime('now'))"
)

# Aliased, ``now``-parameterized variant of ``_RELATIONSHIP_LIVE_FILTER`` for the
# recursive-CTE traversal (#1302). The CTE references the ``relationships`` row
# through the ``r`` alias and binds ``now`` itself (so a caller may pass a fixed
# ``now`` for deterministic point-in-time traversal), so ``datetime('now')``
# becomes ``datetime(?)`` here. The predicate is otherwise byte-identical to
# ``list_relationships`` (same tombstone gates, same ``datetime(...)``
# normalization) - a dream-pruned / invalidated edge excluded from the
# ``list_relationships`` live set is now excluded from the traversal too.
_RELATIONSHIP_CTE_LIVE_FILTER = (
    "r.valid_to IS NULL AND r.invalidated_at IS NULL "
    "AND (r.valid_until IS NULL OR datetime(r.valid_until) > datetime(?))"
)
# ``now``-parameterized variant of ``_ENTITY_LIVE_FILTER`` for the traversal's
# final node fetch (#1302). Binds the caller-provided ``now`` (same value the
# edge filter uses) so a fixed point-in-time traversal evaluates nodes and edges
# at the same instant instead of mixing ``now`` against wall-clock.
_ENTITY_LIVE_FILTER_AT = "(valid_until IS NULL OR datetime(valid_until) > datetime(?))"


def _relationship_insert_params(rel: Relationship) -> tuple:
    rel_type = sanitize_cypher_label(rel.relationship_type or "RELATES_TO")
    return (
        uuid_to_text(rel.id),
        uuid_to_text(rel.namespace_id),
        uuid_to_text(rel.source_entity_id),
        uuid_to_text(rel.target_entity_id),
        rel_type,
        rel.description,
        to_json_text(rel.properties or {}),
        to_json_text([uuid_to_text(d) for d in rel.source_document_ids]),
        to_json_text([uuid_to_text(c) for c in rel.source_chunk_ids]),
        iso8601(rel.valid_from),
        iso8601(rel.valid_until),
        rel.confidence,
        rel.weight,
        to_json_text(rel.metadata or {}),
        iso8601(rel.created_at) or datetime.now(UTC).isoformat(),
        iso8601(rel.updated_at) or datetime.now(UTC).isoformat(),
    )


_RELATIONSHIP_INSERT_SQL = (
    f"INSERT INTO relationships ({_RELATIONSHIP_COLUMNS}) "  # noqa: S608
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# Re-saving an edge (same ``id``) updates its mutable columns instead of
# raising on the primary-key conflict. The temporal window (``valid_from`` /
# ``valid_until``) is the column that matters for ``prefer_current`` recall
# (#1087): a later ingest can close an edge's validity window, and the CTE
# traversal filters on ``valid_until``. ``id`` / ``namespace_id`` / endpoints
# / ``source_document_ids`` / ``source_chunk_ids`` / ``created_at`` are NOT in
# the SET list — re-saving must never rewrite provenance or the original
# creation stamp, and the bi-temporal soft-delete columns (``valid_to`` /
# ``invalidated_at`` / ``invalidated_by``) are likewise left untouched.
_RELATIONSHIP_UPSERT_SQL = _RELATIONSHIP_INSERT_SQL + (
    " ON CONFLICT(id) DO UPDATE SET "
    "relationship_type = excluded.relationship_type, "
    "description = excluded.description, "
    "properties = excluded.properties, "
    "valid_from = excluded.valid_from, "
    "valid_until = excluded.valid_until, "
    "confidence = excluded.confidence, "
    "weight = excluded.weight, "
    "metadata = excluded.metadata, "
    "updated_at = excluded.updated_at"
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SQLiteLanceGraphAdapter(GraphBackendBase):
    """Graph backend backed by SQLite (entities, relationships, episodes).

    The adapter speaks raw aiosqlite — it does not open a SQLAlchemy
    session.  Traversal is done with recursive CTEs.  Entity embeddings
    live in LanceDB; this adapter only touches the SQLite side.
    """

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
        self._entity_key_gate = _SQLiteLanceEntityKeyGate(max_concurrent=10)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._handle.connect()
        logger.debug("SQLiteLanceGraphAdapter connected")

    async def disconnect(self) -> None:
        # The handle is shared across adapters — disposal is owned by
        # the factory, not the individual adapter.
        logger.debug("SQLiteLanceGraphAdapter disconnect (handle not closed)")

    async def is_healthy(self) -> bool:
        return await self._handle.is_healthy()

    @property
    def _conn(self):
        return self._handle.sqlite

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> Entity:
        await self._conn.execute(_ENTITY_INSERT_SQL, _entity_insert_params(entity))
        await self._conn.commit()
        return entity

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id`` (IDOR family)."""
        sql = (
            f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
            "WHERE id = ? AND namespace_id = ? LIMIT 1"
        )
        async with self._conn.execute(sql, (uuid_to_text(entity_id), uuid_to_text(namespace_id))) as cur:
            row = await cur.fetchone()
        return _row_to_entity(row) if row else None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        sql = (
            f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
            "WHERE namespace_id = ? AND name = ? AND entity_type = ? LIMIT 1"
        )
        async with self._conn.execute(sql, (uuid_to_text(namespace_id), name, entity_type)) as cur:
            row = await cur.fetchone()
        return _row_to_entity(row) if row else None

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id`` (IDOR family).

        The ``namespace_id`` kwarg is defense-in-depth — asserted equal to
        ``entity.namespace_id`` before the WHERE clause filters by both id
        and namespace.
        """
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        entity.updated_at = datetime.now(UTC)
        await self._conn.execute(_ENTITY_UPDATE_SQL, _entity_update_params(entity))
        await self._conn.commit()
        return entity

    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id`` (IDOR family)."""
        eid = uuid_to_text(entity_id)
        ns_text = uuid_to_text(namespace_id)
        # Delete edges first so the operation succeeds regardless of whether
        # the Alembic-generated FK cascade is active. Scoped by namespace.
        await self._conn.execute(
            "DELETE FROM relationships WHERE namespace_id = ? AND (source_entity_id = ? OR target_entity_id = ?)",
            (ns_text, eid, eid),
        )
        async with self._conn.execute(
            "DELETE FROM entities WHERE id = ? AND namespace_id = ?",
            (eid, ns_text),
        ) as cur:
            deleted = cur.rowcount
        await self._conn.commit()
        return bool(deleted)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        conditions = ["namespace_id = ?", _ENTITY_LIVE_FILTER]
        params: list[Any] = [uuid_to_text(namespace_id)]
        if entity_type is not None:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        sql = (
            f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_entity(r) for r in rows]

    async def entity_exists(self, namespace_id: UUID, name: str, entity_type: str) -> bool:
        sql = "SELECT 1 FROM entities WHERE namespace_id = ? AND name = ? AND entity_type = ? LIMIT 1"
        async with self._conn.execute(sql, (uuid_to_text(namespace_id), name, entity_type)) as cur:
            row = await cur.fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Batch entity upsert — key-gated against concurrent callers
    # ------------------------------------------------------------------

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Upsert entities matched by ``(namespace_id, name, entity_type)``.

        - Existing rows: ``Entity.merge_with`` is called to combine
          attributes / sources / mention_count, then the row is updated.
        - New rows: inserted directly.

        Uses :class:`_SQLiteLanceEntityKeyGate` so two concurrent batches
        that share any key serialize around the prefetch→write window.
        """
        if not entities:
            return []

        async with self._entity_key_gate.acquire(entities):
            return await self._upsert_entities_locked(namespace_id, entities, batch_size=batch_size)

    async def _upsert_entities_locked(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int,
    ) -> list[tuple[Entity, bool]]:
        ns_text = uuid_to_text(namespace_id)

        # 1. Prefetch existing rows by (name, entity_type) pairs.
        unique_pairs = list({(e.name, e.entity_type) for e in entities})
        existing_map: dict[tuple[str, str], Entity] = {}
        if unique_pairs:
            placeholders = ", ".join("(?, ?)" for _ in unique_pairs)
            flat: list[Any] = []
            for name, etype in unique_pairs:
                flat.extend([name, etype])
            sql = (
                f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
                f"WHERE namespace_id = ? AND (name, entity_type) IN (VALUES {placeholders})"
            )
            async with self._conn.execute(sql, [ns_text, *flat]) as cur:
                rows = await cur.fetchall()
            for r in rows:
                ent = _row_to_entity(r)
                existing_map[(ent.name, ent.entity_type)] = ent

        # 2. Classify into creates vs updates (preserving input order).
        results: list[tuple[Entity, bool]] = []
        to_insert: list[Entity] = []
        to_update: list[Entity] = []

        for entity in entities:
            key = (entity.name, entity.entity_type)
            existing = existing_map.get(key)
            if existing is not None:
                existing.merge_with(entity)
                # Sync the input entity's id to the persisted canonical
                # id (#806). The Neo4j backend does this at
                # ``neo4j.py:1739`` for the same reason: callers hold
                # references to the input ``entities`` list and may
                # have already built ``Relationship`` objects whose
                # ``source_entity_id`` / ``target_entity_id`` were
                # captured from ``entity.id`` BEFORE upsert. Without
                # the in-place mutation, those relationships point at
                # the throwaway extraction-time UUID and the FK
                # constraint (sqlite_lance) / silent MATCH drop (Neo4j
                # smart-resolution path) bites on the second ingest
                # that shares an entity.
                entity.id = existing.id
                to_update.append(existing)
                results.append((existing, False))
                # Guard against two input entities that share the same key:
                # route subsequent hits to update the same merged row.
                existing_map[key] = existing
            else:
                entity.namespace_id = namespace_id
                to_insert.append(entity)
                results.append((entity, True))
                # Same-batch duplicates should merge into the first occurrence.
                existing_map[key] = entity

        # 3. Batched INSERT / UPDATE via executemany.
        for start in range(0, len(to_insert), batch_size):
            chunk = to_insert[start : start + batch_size]
            await self._conn.executemany(_ENTITY_INSERT_SQL, [_entity_insert_params(e) for e in chunk])

        for start in range(0, len(to_update), batch_size):
            chunk = to_update[start : start + batch_size]
            await self._conn.executemany(_ENTITY_UPDATE_SQL, [_entity_update_params(e) for e in chunk])

        if to_insert or to_update:
            await self._conn.commit()

        return results

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        await self._conn.execute(_RELATIONSHIP_UPSERT_SQL, _relationship_insert_params(relationship))
        await self._conn.commit()
        # Reflect the sanitized relationship type back to the caller.
        relationship.relationship_type = sanitize_cypher_label(relationship.relationship_type or "RELATES_TO")
        return relationship

    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id`` (IDOR family)."""
        sql = (
            f"SELECT {_RELATIONSHIP_COLUMNS} FROM relationships "  # noqa: S608
            "WHERE id = ? AND namespace_id = ? LIMIT 1"
        )
        async with self._conn.execute(sql, (uuid_to_text(relationship_id), uuid_to_text(namespace_id))) as cur:
            row = await cur.fetchone()
        return _row_to_relationship(row) if row else None

    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id`` (IDOR family)."""
        async with self._conn.execute(
            "DELETE FROM relationships WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(relationship_id), uuid_to_text(namespace_id)),
        ) as cur:
            deleted = cur.rowcount
        await self._conn.commit()
        return bool(deleted)

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

        ``relationships.namespace_id`` is filtered directly. Since edges in
        this schema are owned by a single namespace, this excludes any cross-
        tenant edges from the result.
        """
        eid = uuid_to_text(entity_id)
        ns = uuid_to_text(namespace_id)
        if direction == "outgoing":
            where = "source_entity_id = ? AND namespace_id = ?"
            params: list[Any] = [eid, ns]
        elif direction == "incoming":
            where = "target_entity_id = ? AND namespace_id = ?"
            params = [eid, ns]
        else:
            where = "(source_entity_id = ? OR target_entity_id = ?) AND namespace_id = ?"
            params = [eid, eid, ns]

        if relationship_types:
            placeholders = ", ".join("?" for _ in relationship_types)
            where += f" AND relationship_type IN ({placeholders})"
            params.extend(relationship_types)

        params.append(limit)
        sql = (
            f"SELECT {_RELATIONSHIP_COLUMNS} FROM relationships "  # noqa: S608
            f"WHERE {where} ORDER BY created_at DESC LIMIT ?"
        )
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_relationship(r) for r in rows]

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        conditions = ["namespace_id = ?", _RELATIONSHIP_LIVE_FILTER]
        params: list[Any] = [uuid_to_text(namespace_id)]
        if relationship_type is not None:
            conditions.append("relationship_type = ?")
            params.append(relationship_type)
        sql = (
            f"SELECT {_RELATIONSHIP_COLUMNS} FROM relationships "  # noqa: S608
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_relationship(r) for r in rows]

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,
    ) -> int:
        if not relationships:
            return 0
        total = 0
        for start in range(0, len(relationships), batch_size):
            chunk = relationships[start : start + batch_size]
            await self._conn.executemany(_RELATIONSHIP_UPSERT_SQL, [_relationship_insert_params(r) for r in chunk])
            total += len(chunk)
        await self._conn.commit()
        # Mirror the sanitized type back into the caller-provided objects
        # so consumers see what was actually persisted.
        for rel in relationships:
            rel.relationship_type = sanitize_cypher_label(rel.relationship_type or "RELATES_TO")
        return total

    # ------------------------------------------------------------------
    # Traversal — recursive CTEs
    # ------------------------------------------------------------------

    async def find_paths(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
        prefer_current: bool = False,
        now: datetime | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find directed paths from source→target using a recursive CTE.

        Returns a list of paths; each path is an ordered list of
        ``{"type": "relationship", "data": {...}}`` dicts representing
        the edges of the path.  Depth is bounded by ``max_depth``.

        When ``prefer_current`` is True, every edge of the path must
        satisfy ``valid_until IS NULL OR valid_until > now`` — mirroring
        the Neo4j ``all(r IN relationships(path) ...)`` predicate (see
        ``engines/vectorcypher/dual_nodes.py:599``).  ``now`` defaults to
        ``datetime.now(UTC)`` and is hoisted so it is bound once.
        """
        ns = uuid_to_text(namespace_id)
        src = uuid_to_text(source_entity_id)
        tgt = uuid_to_text(target_entity_id)
        effective_max = max(1, min(max_depth, 8))

        rel_filter = ""
        if relationship_types:
            placeholders = ", ".join("?" for _ in relationship_types)
            rel_filter = f" AND r.relationship_type IN ({placeholders})"

        valid_filter = ""
        now_iso: str | None = None
        if prefer_current:
            # Same live-set predicate ``list_relationships`` applies (#1302):
            # the bi-temporal tombstones (``valid_to`` / ``invalidated_at``)
            # plus the ``datetime(...)``-normalized ``valid_until`` window. A
            # dream-pruned / invalidated edge dropped from ``list_relationships``
            # is now also dropped from the traversal path.
            valid_filter = f" AND {_RELATIONSHIP_CTE_LIVE_FILTER}"
            now_iso = iso8601(now or datetime.now(UTC))

        # Anchor params: src, ns, [rel_types...], [now]
        # Recursive params: depth, ns, [rel_types...], [now]
        # Tail params: tgt
        params: list[Any] = [src, ns]
        if relationship_types:
            params.extend(relationship_types)
        if prefer_current:
            params.append(now_iso)
        params.append(effective_max)
        params.append(ns)
        if relationship_types:
            params.extend(relationship_types)
        if prefer_current:
            params.append(now_iso)
        params.append(tgt)

        # Path reconstruction trick: concatenate edge ids with a delimiter
        # and split after the fact.  SQLite supports recursive CTEs with
        # string concatenation but does not expose json_array aggregation
        # inside the recursive term, so we use a simple delimited string.
        #
        # ``visited`` tracks **edge ids**, not node ids — matches Neo4j's
        # ``MATCH [*1..N]`` semantics, which forbids reusing the same
        # relationship rather than the same node.
        sql = f"""
            WITH RECURSIVE walk(
                edge_id, src, cur, depth, edge_ids, visited
            ) AS (
                SELECT r.id, r.source_entity_id, r.target_entity_id, 1,
                       r.id,
                       '|' || r.id || '|'
                FROM relationships r
                WHERE r.source_entity_id = ?
                  AND r.namespace_id = ?
                  {rel_filter}
                  {valid_filter}
                UNION ALL
                SELECT r.id, walk.src, r.target_entity_id, walk.depth + 1,
                       walk.edge_ids || ',' || r.id,
                       walk.visited || r.id || '|'
                FROM walk
                JOIN relationships r ON r.source_entity_id = walk.cur
                WHERE walk.depth < ?
                  AND r.namespace_id = ?
                  AND instr(walk.visited, '|' || r.id || '|') = 0
                  {rel_filter}
                  {valid_filter}
            )
            SELECT edge_ids, depth
            FROM walk
            WHERE cur = ?
            ORDER BY depth ASC
        """  # noqa: S608

        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()

        if not rows:
            return []

        # Resolve edge ids back to full relationship rows in one query.
        all_edge_ids: set[str] = set()
        path_specs: list[list[str]] = []
        for row in rows:
            edge_ids = str(row["edge_ids"]).split(",")
            path_specs.append(edge_ids)
            all_edge_ids.update(edge_ids)

        rel_placeholders = ", ".join("?" for _ in all_edge_ids)
        rel_sql = (
            f"SELECT {_RELATIONSHIP_COLUMNS} FROM relationships "  # noqa: S608
            f"WHERE id IN ({rel_placeholders})"
        )
        async with self._conn.execute(rel_sql, list(all_edge_ids)) as rel_cur:
            rel_rows = await rel_cur.fetchall()
        rel_by_id = {str(r["id"]): _row_to_relationship(r) for r in rel_rows}

        paths: list[list[dict[str, Any]]] = []
        for edge_ids in path_specs:
            path: list[dict[str, Any]] = []
            for eid in edge_ids:
                rel = rel_by_id.get(eid)
                if rel is None:
                    break
                path.append(
                    {
                        "type": "relationship",
                        "data": {
                            "id": str(rel.id),
                            "source_entity_id": str(rel.source_entity_id),
                            "target_entity_id": str(rel.target_entity_id),
                            "relationship_type": rel.relationship_type,
                        },
                    }
                )
            else:
                paths.append(path)
        return paths

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
        prefer_current: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Get the outbound+inbound neighborhood of an entity up to ``depth``,
        scoped to ``namespace_id`` (IDOR family).

        Uses a single recursive CTE that expands both directions. Every hop
        is constrained to ``namespace_id`` so the traversal never crosses
        into another namespace. Returns ``{"entities": [...], "relationships": [...]}``
        with Entity / Relationship domain objects.

        See :meth:`get_neighborhoods_batch` for the semantics of
        ``prefer_current`` / ``now``.
        """
        result = await self.get_neighborhoods_batch(
            [entity_id],
            namespace_id=namespace_id,
            depth=depth,
            relationship_types=relationship_types,
            limit_per_entity=limit,
            prefer_current=prefer_current,
            now=now,
        )
        return result.get(entity_id, {"entities": [], "relationships": []})

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
        prefer_current: bool = False,
        now: datetime | None = None,
    ) -> dict[UUID, dict[str, Any]]:
        """Batch neighborhood expansion using a single recursive CTE,
        scoped to ``namespace_id`` (IDOR family).

        Overrides :meth:`GraphBackendBase.get_neighborhoods_batch` to
        avoid the default N+1 loop.  All entity seeds are expanded in
        one query; results are partitioned per seed on return.

        Every hop of the recursive CTE — anchor and recursive — adds
        ``r.namespace_id = ?`` so the traversal never visits an edge in
        another namespace. The seed entities are also filtered against
        ``namespace_id`` after the walk to drop any whose row was deleted
        or never belonged to the caller.

        When ``prefer_current`` is True, every edge of the expansion
        path must satisfy ``valid_until IS NULL OR valid_until > now`` —
        mirroring the Neo4j ``all(r IN relationships(path) ...)``
        predicate.  ``now`` defaults to ``datetime.now(UTC)``.
        """
        if not entity_ids:
            return {}

        effective_depth = max(1, min(depth, 6))
        ns_text = uuid_to_text(namespace_id)
        seed_texts = [uuid_to_text(eid) for eid in entity_ids]
        seed_placeholders = ", ".join("?" for _ in seed_texts)

        rel_filter = ""
        rel_params: list[Any] = []
        if relationship_types:
            placeholders = ", ".join("?" for _ in relationship_types)
            rel_filter = f" AND r.relationship_type IN ({placeholders})"
            rel_params = list(relationship_types)

        valid_filter = ""
        valid_params: list[Any] = []
        if prefer_current:
            # Same live-set predicate ``list_relationships`` applies (#1302).
            valid_filter = f" AND {_RELATIONSHIP_CTE_LIVE_FILTER}"
            valid_params = [iso8601(now or datetime.now(UTC))]

        # Every clause (2 anchor arms + 2 recursive arms) gets a
        # ``r.namespace_id = ?`` predicate. Param order is:
        #   anchor-out:    seed_texts..., ns, rel_params..., valid_params...
        #   anchor-in:     seed_texts..., ns, rel_params..., valid_params...
        #   recursive-out: depth, ns, rel_params..., valid_params...
        #   recursive-in:  depth, ns, rel_params..., valid_params...
        params: list[Any] = []
        params.extend(seed_texts)
        params.append(ns_text)
        params.extend(rel_params)
        params.extend(valid_params)
        params.extend(seed_texts)
        params.append(ns_text)
        params.extend(rel_params)
        params.extend(valid_params)
        params.append(effective_depth)
        params.append(ns_text)
        params.extend(rel_params)
        params.extend(valid_params)
        params.append(effective_depth)
        params.append(ns_text)
        params.extend(rel_params)
        params.extend(valid_params)

        # ``visited`` tracks **edge ids** (matching Neo4j's
        # ``MATCH [*1..N]`` semantics — forbid reusing the same edge,
        # not the same node).  Without this, a cycle like A→B→C→A
        # makes the recursion fan out exponentially with depth, even
        # though ``DISTINCT`` masks the row count after the fact.
        sql = f"""
            WITH RECURSIVE walk(seed, cur, depth, direction, edge_id, visited) AS (
                SELECT r.source_entity_id, r.target_entity_id, 1, 'out', r.id,
                       '|' || r.id || '|'
                FROM relationships r
                WHERE r.source_entity_id IN ({seed_placeholders})
                  AND r.namespace_id = ?
                  {rel_filter}
                  {valid_filter}
                UNION ALL
                SELECT r.target_entity_id, r.source_entity_id, 1, 'in', r.id,
                       '|' || r.id || '|'
                FROM relationships r
                WHERE r.target_entity_id IN ({seed_placeholders})
                  AND r.namespace_id = ?
                  {rel_filter}
                  {valid_filter}
                UNION ALL
                SELECT walk.seed, r.target_entity_id, walk.depth + 1, 'out', r.id,
                       walk.visited || r.id || '|'
                FROM walk
                JOIN relationships r ON r.source_entity_id = walk.cur
                WHERE walk.depth < ?
                  AND r.namespace_id = ?
                  AND instr(walk.visited, '|' || r.id || '|') = 0
                  {rel_filter}
                  {valid_filter}
                UNION ALL
                SELECT walk.seed, r.source_entity_id, walk.depth + 1, 'in', r.id,
                       walk.visited || r.id || '|'
                FROM walk
                JOIN relationships r ON r.target_entity_id = walk.cur
                WHERE walk.depth < ?
                  AND r.namespace_id = ?
                  AND instr(walk.visited, '|' || r.id || '|') = 0
                  {rel_filter}
                  {valid_filter}
            )
            SELECT DISTINCT seed, cur, edge_id FROM walk
        """  # noqa: S608

        async with self._conn.execute(sql, params) as cur:
            walk_rows = await cur.fetchall()

        if not walk_rows:
            return {eid: {"entities": [], "relationships": []} for eid in entity_ids}

        # Collect referenced entity + edge ids, load them in one round-trip each.
        needed_entity_ids: set[str] = set()
        needed_edge_ids: set[str] = set()
        by_seed: dict[str, dict[str, set[str]]] = {s: {"ents": set(), "edges": set()} for s in seed_texts}
        for row in walk_rows:
            seed = str(row["seed"])
            ent = str(row["cur"])
            edge = str(row["edge_id"])
            if seed in by_seed:
                by_seed[seed]["ents"].add(ent)
                by_seed[seed]["edges"].add(edge)
            needed_entity_ids.add(ent)
            needed_edge_ids.add(edge)

        # Belt-and-suspenders: entity/relationship row fetches are
        # additionally constrained by ``namespace_id``. The CTE already
        # filtered edges, but if a misrouted edge ever points to an
        # out-of-namespace entity, this final guard keeps it out.
        ent_by_id: dict[str, Entity] = {}
        if needed_entity_ids:
            placeholders = ", ".join("?" for _ in needed_entity_ids)
            ent_params: list[Any] = [*needed_entity_ids, ns_text]
            # When prefer_current, drop retired nodes from the path the same way
            # ``list_entities`` does (#1302) — entity ``valid_until`` window,
            # bound to the SAME ``now`` the edge filter uses (``valid_params``) so
            # a fixed point-in-time traversal gates nodes and edges consistently.
            entity_live = ""
            if prefer_current:
                entity_live = f" AND {_ENTITY_LIVE_FILTER_AT}"
                ent_params.extend(valid_params)
            sql_e = (
                f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
                f"WHERE id IN ({placeholders}) AND namespace_id = ?{entity_live}"
            )
            async with self._conn.execute(sql_e, ent_params) as cur_e:
                rows_e = await cur_e.fetchall()
            ent_by_id = {str(r["id"]): _row_to_entity(r) for r in rows_e}

        rel_by_id: dict[str, Relationship] = {}
        if needed_edge_ids:
            placeholders = ", ".join("?" for _ in needed_edge_ids)
            sql_r = (
                f"SELECT {_RELATIONSHIP_COLUMNS} FROM relationships "  # noqa: S608
                f"WHERE id IN ({placeholders}) AND namespace_id = ?"
            )
            async with self._conn.execute(sql_r, [*needed_edge_ids, ns_text]) as cur_r:
                rows_r = await cur_r.fetchall()
            rel_by_id = {str(r["id"]): _row_to_relationship(r) for r in rows_r}

        result: dict[UUID, dict[str, Any]] = {}
        for eid, seed_text in zip(entity_ids, seed_texts, strict=True):
            bucket = by_seed.get(seed_text, {"ents": set(), "edges": set()})
            entities = [ent_by_id[e] for e in bucket["ents"] if e in ent_by_id][:limit_per_entity]
            relationships = [rel_by_id[r] for r in bucket["edges"] if r in rel_by_id][:limit_per_entity]
            result[eid] = {"entities": entities, "relationships": relationships}
        return result

    # ------------------------------------------------------------------
    # Attribute search
    # ------------------------------------------------------------------

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities where ``attributes.<name> == value``.

        Uses SQLite ``json_extract`` — no need to sanitize ``attribute_name``
        because it's bound as a parameter to ``'$.'||?`` (still, names
        containing quoting characters won't match anything useful).
        """
        sql = (
            f"SELECT {_ENTITY_COLUMNS} FROM entities "  # noqa: S608
            "WHERE namespace_id = ? "
            "AND json_extract(attributes, '$.' || ?) = ? "
            "LIMIT ?"
        )
        # json_extract returns typed values — stringify the comparator if
        # the caller passed something non-scalar.
        compare_value = attribute_value
        if isinstance(compare_value, (dict, list)):
            compare_value = to_json_text(compare_value)
        async with self._conn.execute(sql, (uuid_to_text(namespace_id), attribute_name, compare_value, limit)) as cur:
            rows = await cur.fetchall()
        return [_row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    async def create_episode(self, episode: Episode) -> Episode:
        sql = (
            "INSERT INTO episodes ("
            "id, namespace_id, name, description, occurred_at, duration_seconds, "
            "entity_ids, source_document_ids, source_chunk_ids, embedding_model, "
            "metadata, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            uuid_to_text(episode.id),
            uuid_to_text(episode.namespace_id),
            episode.name,
            episode.description,
            iso8601(episode.occurred_at) or datetime.now(UTC).isoformat(),
            episode.duration_seconds,
            to_json_text([uuid_to_text(e) for e in episode.entity_ids]),
            to_json_text([uuid_to_text(d) for d in episode.source_document_ids]),
            to_json_text([uuid_to_text(c) for c in episode.source_chunk_ids]),
            episode.embedding_model,
            to_json_text(episode.metadata or {}),
            iso8601(episode.created_at) or datetime.now(UTC).isoformat(),
            iso8601(episode.updated_at) or datetime.now(UTC).isoformat(),
        )
        await self._conn.execute(sql, params)
        await self._conn.commit()
        return episode

    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id`` (IDOR family)."""
        sql = (
            "SELECT id, namespace_id, name, description, occurred_at, duration_seconds, "
            "entity_ids, source_document_ids, source_chunk_ids, embedding_model, "
            "metadata, created_at, updated_at "
            "FROM episodes WHERE id = ? AND namespace_id = ? LIMIT 1"
        )
        async with self._conn.execute(sql, (uuid_to_text(episode_id), uuid_to_text(namespace_id))) as cur:
            row = await cur.fetchone()
        return _row_to_episode(row) if row else None

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        conditions = ["namespace_id = ?"]
        params: list[Any] = [uuid_to_text(namespace_id)]
        if start_time is not None:
            conditions.append("occurred_at >= ?")
            params.append(iso8601(start_time))
        if end_time is not None:
            conditions.append("occurred_at <= ?")
            params.append(iso8601(end_time))
        where_sql = f"FROM episodes WHERE {' AND '.join(conditions)} "  # noqa: S608
        sql = (
            "SELECT id, namespace_id, name, description, occurred_at, duration_seconds, "
            "entity_ids, source_document_ids, source_chunk_ids, embedding_model, "
            "metadata, created_at, updated_at " + where_sql + "ORDER BY occurred_at DESC LIMIT ?"
        )
        params.append(limit)
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_episode(r) for r in rows]

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    async def count_entities(self, namespace_id: UUID) -> int:
        sql = "SELECT COUNT(*) AS c FROM entities WHERE namespace_id = ?"
        async with self._conn.execute(sql, (uuid_to_text(namespace_id),)) as cur:
            row = await cur.fetchone()
        return int(row["c"] if row else 0)

    async def count_relationships(self, namespace_id: UUID) -> int:
        sql = "SELECT COUNT(*) AS c FROM relationships WHERE namespace_id = ?"
        async with self._conn.execute(sql, (uuid_to_text(namespace_id),)) as cur:
            row = await cur.fetchone()
        return int(row["c"] if row else 0)
