"""SurrealDB vector adapter for Khora.

Implements VectorBackendProtocol using SurrealDB's native HNSW vector
indexing and BM25 full-text search.  All record IDs follow the
``table:⟨uuid⟩`` convention expected by the unified SurrealDB schema.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import Chunk, ChunkMetadata, Entity
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Regex to extract a UUID from a SurrealDB record ID.
# Handles both ``table:uuid`` and ``table:⟨uuid⟩`` forms.
_RECORD_ID_RE = re.compile(r"[^:]+:\u27e8?([0-9a-fA-F\-]{36})\u27e9?")


def _rid(table: str, uid: UUID) -> str:
    """Build a SurrealDB record-link literal ``table:⟨uuid⟩``."""
    return f"{table}:\u27e8{uid}\u27e9"


def _parse_uuid(record_id: str | dict | UUID | Any) -> UUID:
    """Extract a UUID from a SurrealDB record ID.

    Handles strings like ``chunk:018f...``, ``chunk:⟨018f...⟩``,
    bare UUID strings, and ``uuid.UUID`` objects.
    """
    if isinstance(record_id, UUID):
        return record_id
    raw = str(record_id)
    m = _RECORD_ID_RE.match(raw)
    if m:
        return UUID(m.group(1))
    # Fall back: try treating the whole string as a UUID
    return UUID(raw)


def _iso(dt: datetime | None) -> str | None:
    """Convert a datetime to an ISO-8601 string or *None*."""
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(val: Any) -> datetime | None:
    """Best-effort parse of a SurrealDB datetime value."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        raw = str(val)
        # SurrealDB may return ISO strings
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class SurrealDBVectorAdapter:
    """Vector backend backed by SurrealDB.

    Uses SurrealDB's HNSW vector index for approximate-nearest-neighbour
    search and its built-in BM25 analyser for full-text search.

    The adapter delegates all I/O to a :class:`SurrealDBConnection`,
    which manages client lifecycle and authentication.
    """

    def __init__(
        self,
        connection: SurrealDBConnection,
        *,
        hnsw_ef_search: int = 40,
    ) -> None:
        self._conn = connection
        self._hnsw_ef_search = hnsw_ef_search

    # ------------------------------------------------------------------
    # Factory / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBVectorAdapter:
        """Create an adapter from a configuration dictionary.

        Expected keys mirror :class:`SurrealDBConnection` init args, plus
        optional ``hnsw_ef_search``.
        """
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn_kwargs: dict[str, Any] = {}
        for key in ("mode", "path", "url", "namespace", "database", "user", "password"):
            if key in config:
                conn_kwargs[key] = config[key]

        connection = SurrealDBConnection(**conn_kwargs)
        return cls(connection, hnsw_ef_search=config.get("hnsw_ef_search", 40))

    async def connect(self) -> None:
        """Ensure the underlying connection is established."""
        await self._conn.connect()
        logger.info("SurrealDBVectorAdapter connected")

    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        await self._conn.disconnect()
        logger.info("SurrealDBVectorAdapter disconnected")

    async def is_healthy(self) -> bool:
        """Delegate health-check to the connection."""
        return await self._conn.is_healthy()

    def _get_session(self) -> None:
        """Compatibility shim expected by some callers.  Returns *None*."""
        return None

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: Chunk, *, session: Any = None) -> Chunk:
        """Create a single chunk record in SurrealDB."""
        sql = (
            "CREATE chunk:\u27e8$id\u27e9 SET "
            "namespace = memory_namespace:\u27e8$ns\u27e9, "
            "document = document:\u27e8$doc\u27e9, "
            "content = $content, "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "chunk_index = $chunk_index, "
            "start_char = $start_char, "
            "end_char = $end_char, "
            "token_count = $token_count, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "source_timestamp = $source_timestamp"
        )
        bindings = self._chunk_to_bindings(chunk)
        await self._conn.execute(sql, bindings)
        return chunk

    async def create_chunks_batch(self, chunks: list[Chunk], *, session: Any = None) -> list[Chunk]:
        """Batch-insert chunks using ``INSERT INTO chunk [...]``."""
        if not chunks:
            return []

        records: list[dict[str, Any]] = []
        for chunk in chunks:
            records.append(
                {
                    "id": _rid("chunk", chunk.id),
                    "namespace": _rid("memory_namespace", chunk.namespace_id),
                    "document": _rid("document", chunk.document_id),
                    "content": chunk.content,
                    "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
                    "embedding_model": chunk.embedding_model,
                    "chunk_index": chunk.metadata.chunk_index,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "token_count": chunk.metadata.token_count,
                    "metadata_": chunk.metadata.custom or {},
                    "created_at": _iso(chunk.created_at),
                    "source_timestamp": _iso(chunk.source_timestamp),
                }
            )

        sql = "INSERT INTO chunk $records"
        await self._conn.execute(sql, {"records": records})
        return chunks

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Fetch a single chunk by primary key."""
        sql = "SELECT * FROM chunk:\u27e8$id\u27e9"
        row = await self._conn.query_one(sql, {"id": str(chunk_id)})
        if not row:
            return None
        return self._row_to_chunk(row)

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        """Fetch multiple chunks in one round-trip."""
        if not chunk_ids:
            return {}

        ids_list = ", ".join(f"chunk:\u27e8{uid}\u27e9" for uid in chunk_ids)
        sql = f"SELECT * FROM chunk WHERE id IN [{ids_list}]"
        rows = await self._conn.query(sql)
        result: dict[UUID, Chunk] = {}
        for row in rows:
            chunk = self._row_to_chunk(row)
            result[chunk.id] = chunk
        return result

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Return all chunks belonging to a document, ordered by index."""
        sql = "SELECT * FROM chunk WHERE document = document:\u27e8$doc\u27e9 ORDER BY chunk_index ASC"
        rows = await self._conn.query(sql, {"doc": str(document_id)})
        return [self._row_to_chunk(r) for r in rows]

    async def delete_chunks_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document and return the count deleted."""
        # First count so we can report back
        count_sql = "SELECT count() AS cnt FROM chunk WHERE document = document:\u27e8$doc\u27e9 GROUP ALL"
        count_row = await self._conn.query_one(count_sql, {"doc": str(document_id)})
        count = int(count_row.get("cnt", 0)) if count_row else 0

        if count > 0:
            del_sql = "DELETE FROM chunk WHERE document = document:\u27e8$doc\u27e9"
            await self._conn.execute(del_sql, {"doc": str(document_id)})

        return count

    @trace(
        "khora.surrealdb.search_similar",
        include={"namespace_id", "limit"},
        result=lambda r: {"result_count": len(r)},
    )
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search using SurrealDB HNSW index.

        Uses the ``<|K,EF|>`` KNN operator for HNSW acceleration and
        computes an explicit cosine similarity score via
        ``vector::similarity::cosine``.
        """
        # Build WHERE predicates
        where_clauses = [f"namespace = memory_namespace:\u27e8{namespace_id}\u27e9", "embedding IS NOT NULL"]
        bindings: dict[str, Any] = {
            "query_embedding": list(query_embedding),
            "limit": limit,
            "ef": self._hnsw_ef_search,
        }

        if filter_document_ids:
            ids_list = ", ".join(f"document:\u27e8{uid}\u27e9" for uid in filter_document_ids)
            where_clauses.append(f"document IN [{ids_list}]")

        if created_after is not None:
            where_clauses.append("(source_timestamp ?? created_at) >= $created_after")
            bindings["created_after"] = _iso(created_after)

        if created_before is not None:
            where_clauses.append("(source_timestamp ?? created_at) <= $created_before")
            bindings["created_before"] = _iso(created_before)

        if metadata_filters:
            for i, (key, value) in enumerate(metadata_filters.items()):
                param = f"mf_{i}"
                where_clauses.append(f"metadata_.{key} = ${param}")
                bindings[param] = value

        where_sql = " AND ".join(where_clauses)
        sql = (
            "SELECT *, vector::similarity::cosine(embedding, $query_embedding) AS similarity "
            f"FROM chunk WHERE {where_sql} "
            "ORDER BY embedding <|$limit,$ef|> $query_embedding"
        )

        rows = await self._conn.query(sql, bindings)

        results: list[tuple[Chunk, float]] = []
        for row in rows:
            sim = float(row.get("similarity", 0.0))
            if sim < min_similarity:
                continue
            results.append((self._row_to_chunk(row), sim))
        return results

    @trace(
        "khora.surrealdb.search_fulltext",
        include={"namespace_id", "limit"},
        result=lambda r: {"result_count": len(r)},
    )
    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Full-text (BM25) search on chunk content.

        Uses SurrealDB's ``@1@`` match operator and ``search::score(1)``
        for BM25 ranking.
        """
        where_clauses = [
            f"namespace = memory_namespace:\u27e8{namespace_id}\u27e9",
            "content @1@ $query_text",
        ]
        bindings: dict[str, Any] = {"query_text": query_text, "limit": limit}

        if created_after is not None:
            where_clauses.append("(source_timestamp ?? created_at) >= $created_after")
            bindings["created_after"] = _iso(created_after)

        if created_before is not None:
            where_clauses.append("(source_timestamp ?? created_at) <= $created_before")
            bindings["created_before"] = _iso(created_before)

        where_sql = " AND ".join(where_clauses)
        sql = "SELECT *, search::score(1) AS rank " f"FROM chunk WHERE {where_sql} " "ORDER BY rank DESC LIMIT $limit"

        rows = await self._conn.query(sql, bindings)
        return [(self._row_to_chunk(row), float(row.get("rank", 0.0))) for row in rows]

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Return the total number of chunks in a namespace."""
        sql = "SELECT count() AS cnt FROM chunk WHERE namespace = memory_namespace:\u27e8$ns\u27e9 GROUP ALL"
        row = await self._conn.query_one(sql, {"ns": str(namespace_id)})
        return int(row.get("cnt", 0)) if row else 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """Paginated listing of chunks in a namespace."""
        sql = (
            "SELECT * FROM chunk "
            "WHERE namespace = memory_namespace:\u27e8$ns\u27e9 "
            "ORDER BY created_at DESC "
            "LIMIT $limit START $offset"
        )
        rows = await self._conn.query(sql, {"ns": str(namespace_id), "limit": limit, "offset": offset})
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity operations (vector storage side)
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> None:
        """Create an entity record for vector search."""
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
        await self._conn.execute(sql, self._entity_to_bindings(entity))

    async def update_entity(self, entity: Entity) -> None:
        """Update an existing entity record."""
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
        bindings = self._entity_to_bindings(entity)
        # No created_at on update
        bindings.pop("created_at", None)
        await self._conn.execute(sql, bindings)

    async def entity_exists(self, entity_id: UUID) -> bool:
        """Check whether an entity record exists."""
        sql = "SELECT count() AS cnt FROM entity WHERE id = entity:\u27e8$id\u27e9 GROUP ALL"
        row = await self._conn.query_one(sql, {"id": str(entity_id)})
        return int(row.get("cnt", 0)) > 0 if row else False

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Set the embedding vector on a single entity."""
        sql = (
            "UPDATE entity:\u27e8$id\u27e9 SET "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "updated_at = $updated_at"
        )
        await self._conn.execute(
            sql,
            {
                "id": str(entity_id),
                "embedding": list(embedding),
                "embedding_model": model,
                "updated_at": _iso(datetime.now(UTC)),
            },
        )

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        """Batch-update entity embeddings.

        Uses a SurrealQL ``FOR`` loop to apply all updates in a single
        round-trip.
        """
        if not updates:
            return 0

        now_iso = _iso(datetime.now(UTC))
        update_dicts: list[dict[str, Any]] = [
            {
                "rid": _rid("entity", eid),
                "embedding": list(emb),
                "embedding_model": mdl,
                "updated_at": now_iso,
            }
            for eid, emb, mdl in updates
        ]

        sql = (
            "FOR $upd IN $updates { "
            "UPDATE type::thing($upd.rid) SET "
            "embedding = $upd.embedding, "
            "embedding_model = $upd.embedding_model, "
            "updated_at = $upd.updated_at "
            "}"
        )
        await self._conn.execute(sql, {"updates": update_dicts})
        return len(updates)

    @trace(
        "khora.surrealdb.search_similar_entities",
        include={"namespace_id", "limit"},
        result=lambda r: {"result_count": len(r)},
    )
    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """HNSW KNN search over entity embeddings."""
        sql = (
            "SELECT id, vector::similarity::cosine(embedding, $query_embedding) AS similarity "
            f"FROM entity WHERE namespace = memory_namespace:\u27e8{namespace_id}\u27e9 AND embedding IS NOT NULL "
            "ORDER BY embedding <|$limit,$ef|> $query_embedding"
        )
        bindings: dict[str, Any] = {
            "query_embedding": list(query_embedding),
            "limit": limit,
            "ef": self._hnsw_ef_search,
        }

        rows = await self._conn.query(sql, bindings)
        results: list[tuple[UUID, float]] = []
        for row in rows:
            sim = float(row.get("similarity", 0.0))
            if sim < min_similarity:
                continue
            entity_id = _parse_uuid(row.get("id", ""))
            results.append((entity_id, sim))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_to_bindings(self, chunk: Chunk) -> dict[str, Any]:
        """Convert a :class:`Chunk` to SurrealQL parameter bindings."""
        return {
            "id": str(chunk.id),
            "ns": str(chunk.namespace_id),
            "doc": str(chunk.document_id),
            "content": chunk.content,
            "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
            "embedding_model": chunk.embedding_model,
            "chunk_index": chunk.metadata.chunk_index,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "token_count": chunk.metadata.token_count,
            "metadata_": chunk.metadata.custom or {},
            "created_at": _iso(chunk.created_at),
            "source_timestamp": _iso(chunk.source_timestamp),
        }

    def _entity_to_bindings(self, entity: Entity) -> dict[str, Any]:
        """Convert an :class:`Entity` to SurrealQL parameter bindings."""
        return {
            "id": str(entity.id),
            "ns": str(entity.namespace_id),
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "attributes": entity.attributes or {},
            "source_document_ids": [str(uid) for uid in entity.source_document_ids],
            "source_chunk_ids": [str(uid) for uid in entity.source_chunk_ids],
            "source_tool": entity.source_tool,
            "mention_count": entity.mention_count,
            "embedding": list(entity.embedding) if entity.embedding is not None else None,
            "embedding_model": entity.embedding_model,
            "valid_from": _iso(entity.valid_from),
            "valid_until": _iso(entity.valid_until),
            "confidence": entity.confidence,
            "metadata_": entity.metadata or {},
            "created_at": _iso(entity.created_at),
            "updated_at": _iso(entity.updated_at),
        }

    def _row_to_chunk(self, row: dict[str, Any]) -> Chunk:
        """Map a SurrealDB result row to a domain :class:`Chunk`."""
        chunk_id = _parse_uuid(row.get("id", ""))
        namespace_id = _parse_uuid(row.get("namespace", ""))
        document_id = _parse_uuid(row.get("document", ""))

        raw_embedding = row.get("embedding")
        if raw_embedding is not None:
            if _HAS_NUMPY:
                embedding: list[float] | Any = np.asarray(raw_embedding, dtype=np.float32)
            else:
                embedding = [float(v) for v in raw_embedding]
        else:
            embedding = None

        custom_meta = row.get("metadata_") or {}
        if not isinstance(custom_meta, dict):
            custom_meta = {}

        return Chunk(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content=row.get("content", ""),
            metadata=ChunkMetadata(
                document_id=document_id,
                chunk_index=int(row.get("chunk_index", 0)),
                start_char=int(row.get("start_char", 0)),
                end_char=int(row.get("end_char", 0)),
                token_count=int(row.get("token_count", 0)),
                custom=custom_meta,
            ),
            embedding=embedding,
            embedding_model=row.get("embedding_model", ""),
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            source_timestamp=_parse_dt(row.get("source_timestamp")),
        )

    def _row_to_entity(self, row: dict[str, Any]) -> Entity:
        """Map a SurrealDB result row to a domain :class:`Entity`."""
        entity_id = _parse_uuid(row.get("id", ""))
        namespace_id = _parse_uuid(row.get("namespace", ""))

        raw_embedding = row.get("embedding")
        if raw_embedding is not None:
            if _HAS_NUMPY:
                embedding: list[float] | Any = np.asarray(raw_embedding, dtype=np.float32)
            else:
                embedding = [float(v) for v in raw_embedding]
        else:
            embedding = None

        src_doc_ids = [UUID(s) for s in (row.get("source_document_ids") or [])]
        src_chunk_ids = [UUID(s) for s in (row.get("source_chunk_ids") or [])]

        return Entity(
            id=entity_id,
            namespace_id=namespace_id,
            name=row.get("name", ""),
            entity_type=row.get("entity_type", "CONCEPT"),
            description=row.get("description", ""),
            attributes=row.get("attributes") or {},
            source_tool=row.get("source_tool", ""),
            source_document_ids=src_doc_ids,
            source_chunk_ids=src_chunk_ids,
            mention_count=int(row.get("mention_count", 1)),
            embedding=embedding,
            embedding_model=row.get("embedding_model", ""),
            valid_from=_parse_dt(row.get("valid_from")),
            valid_until=_parse_dt(row.get("valid_until")),
            confidence=float(row.get("confidence", 1.0)),
            metadata=row.get("metadata_") or {},
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
        )
