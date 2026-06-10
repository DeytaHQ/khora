"""SurrealDB vector adapter for Khora.

Implements VectorBackendProtocol using SurrealDB's native HNSW vector
indexing and BM25 full-text search.  All record IDs follow the
``table:⟨uuid⟩`` convention expected by the unified SurrealDB schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import Chunk, Entity
from khora.storage.backends.surrealdb._helpers import (
    _HAS_NUMPY,
    _entity_to_bindings,
    _parse_dt,
    _parse_uuid,
    _rid,
    _row_to_entity,
    _sanitize_field_name,
)
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.filter.ast import FilterNode
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection

try:
    import numpy as np
except ImportError:
    pass


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
        hnsw_ef_search: int = 100,
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
        optional ``hnsw_ef_search``. ``password`` and ``url`` are unwrapped
        from ``pydantic.SecretStr`` if needed so the driver receives
        plaintext credentials / DSN.
        """
        from pydantic import SecretStr

        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn_kwargs: dict[str, Any] = {}
        for key in ("mode", "path", "url", "namespace", "database", "user", "password"):
            if key in config:
                value = config[key]
                if key in ("password", "url") and isinstance(value, SecretStr):
                    value = value.get_secret_value()
                conn_kwargs[key] = value

        connection = SurrealDBConnection(**conn_kwargs)
        return cls(connection, hnsw_ef_search=config.get("hnsw_ef_search", 100))

    async def create_tables(self) -> None:
        """Create SurrealDB tables and indexes (idempotent).

        Schema is also auto-initialized on connect(), so this is
        safe to call multiple times.
        """
        from .schema import initialize_schema

        await initialize_schema(self._conn)

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
            "CREATE $rid SET "
            "namespace = $ns_rid, "
            "document = $doc_rid, "
            "content = $content, "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "chunk_index = $chunk_index, "
            "start_char = $start_char, "
            "end_char = $end_char, "
            "token_count = $token_count, "
            "metadata_ = $metadata_, "
            "chunker_info = $chunker_info, "
            "created_at = $created_at, "
            "source_timestamp = $source_timestamp, "
            "occurred_at = $occurred_at, "
            "last_accessed_at = $last_accessed_at, "
            "session_id = $session_id"
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
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count,
                    "metadata_": chunk.metadata or {},
                    "chunker_info": chunk.chunker_info or {},
                    "created_at": chunk.created_at,
                    "source_timestamp": chunk.source_timestamp,
                    "occurred_at": chunk.occurred_at,
                    "last_accessed_at": chunk.last_accessed_at,
                    "session_id": str(chunk.session_id) if chunk.session_id else None,
                }
            )

        sql = "INSERT INTO chunk $records"
        await self._conn.execute(sql, {"records": records})
        return chunks

    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        """Fetch a single chunk by primary key, filtered to ``namespace_id``."""
        sql = (
            "SELECT * FROM chunk WHERE id = $rid AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) LIMIT 1"
        )
        row = await self._conn.query_one(
            sql,
            {
                "rid": _rid("chunk", chunk_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        if not row:
            return None
        return self._row_to_chunk(row)

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        """Fetch multiple chunks in one round-trip, filtered to ``namespace_id``."""
        if not chunk_ids:
            return {}

        chunk_rids = [_rid("chunk", uid) for uid in chunk_ids]
        sql = "SELECT * FROM chunk WHERE id IN $ids AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str)"
        rows = await self._conn.query(
            sql,
            {
                "ids": chunk_rids,
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        result: dict[UUID, Chunk] = {}
        for row in rows:
            chunk = self._row_to_chunk(row)
            result[chunk.id] = chunk
        return result

    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        """Return all chunks belonging to a document, filtered to ``namespace_id``."""
        sql = (
            "SELECT * FROM chunk "
            "WHERE document = $doc_rid "
            "AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) "
            "ORDER BY chunk_index ASC"
        )
        rows = await self._conn.query(
            sql,
            {
                "doc_rid": _rid("document", document_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        return [self._row_to_chunk(r) for r in rows]

    async def delete_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> int:
        """Delete all chunks for a document, scoped to ``namespace_id`` (IDOR family).

        Returns the count deleted. Chunks in other namespaces are silently
        skipped, preventing cross-tenant deletion by document id.
        """
        doc_rid = _rid("document", document_id)
        ns_str = str(namespace_id)
        # First count so we can report back
        count_sql = "SELECT count() AS cnt FROM chunk WHERE document = $doc_rid AND namespace_id = $ns GROUP ALL"
        count_row = await self._conn.query_one(
            count_sql,
            {"doc_rid": doc_rid, "ns": ns_str},
        )
        count = int(count_row.get("cnt", 0)) if count_row else 0

        if count > 0:
            del_sql = "DELETE FROM chunk WHERE document = $doc_rid AND namespace_id = $ns"
            await self._conn.execute(
                del_sql,
                {"doc_rid": doc_rid, "ns": ns_str},
            )

        return count

    async def update_last_accessed(
        self,
        namespace_id: UUID,
        chunk_ids: list[UUID],
        ts: datetime,
    ) -> int:
        """Stamp ``last_accessed_at = ts`` on the given chunks.

        Single UPDATE statement, scoped to ``namespace_id`` to prevent
        cross-tenant writes through forged ids. Returns the row count.
        Used by the Chronicle reinforcement-on-recall path.

        The namespace is resolved to its record id(s) up front and the
        chunk's ``namespace`` link is matched by *direct* RID equality
        (``namespace IN $ns_rids``) rather than the sibling read methods'
        ``OR namespace.namespace_id = $ns_str`` deref. The live record-link
        dereference is non-deterministic on the embedded engine, so it is
        avoided here to keep the cross-tenant guard deterministic — a flaky
        security predicate is worse than a missed reinforcement.
        """
        if not chunk_ids:
            return 0
        chunk_rids = [_rid("chunk", cid) for cid in chunk_ids]
        # Resolve the namespace to its record id(s) up front and match the chunk's
        # ``namespace`` link by direct RID equality. This avoids the non-deterministic
        # chunk->namespace ``.namespace_id`` dereference (flaky on the embedded engine)
        # while still honoring both the row-id and stable-id namespace forms.
        ns_rows = await self._conn.query(
            "SELECT id FROM memory_namespace WHERE namespace_id = $ns_str OR id = $ns_rid",
            {"ns_str": str(namespace_id), "ns_rid": _rid("memory_namespace", namespace_id)},
        )
        ns_rids = [row["id"] for row in ns_rows]
        if not ns_rids:
            return 0
        rows = await self._conn.query(
            "UPDATE chunk SET last_accessed_at = $ts WHERE id IN $ids AND namespace IN $ns_rids",
            {"ts": ts, "ids": chunk_rids, "ns_rids": ns_rids},
        )
        return len(rows)

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
        """Semantic search using cosine similarity.

        Computes cosine similarity via ``vector::similarity::cosine``
        and sorts by descending similarity.  HNSW index accelerates
        the distance computation when available.
        """
        # Build WHERE predicates
        ns_rid = _rid("memory_namespace", namespace_id)
        where_clauses = [
            "(namespace = $ns_rid OR namespace.namespace_id = $ns_str)",
            "embedding IS NOT NULL",
        ]
        bindings: dict[str, Any] = {
            "ns_rid": ns_rid,
            "ns_str": str(namespace_id),
            "query_embedding": list(query_embedding),
            "limit": limit,
            "ef": self._hnsw_ef_search,
        }

        if filter_document_ids:
            doc_rids = [_rid("document", uid) for uid in filter_document_ids]
            where_clauses.append("document IN $filter_doc_ids")
            bindings["filter_doc_ids"] = doc_rids

        if created_after is not None:
            where_clauses.append("(source_timestamp ?? created_at) >= $created_after")
            bindings["created_after"] = created_after

        if created_before is not None:
            where_clauses.append("(source_timestamp ?? created_at) <= $created_before")
            bindings["created_before"] = created_before

        if metadata_filters:
            for i, (key, value) in enumerate(metadata_filters.items()):
                safe_key = _sanitize_field_name(key)
                param = f"mf_{i}"
                where_clauses.append(f"metadata_.{safe_key} = ${param}")
                bindings[param] = value

        where_sql = " AND ".join(where_clauses)
        # Use brute-force similarity + ORDER BY instead of <|K|> KNN operator
        # (KNN is unreliable in embedded mode and rejects parameterised limits).
        # vector::dot() is ~3x faster than vector::similarity::cosine() and
        # produces identical results for L2-normalized embeddings (unit vectors).
        sql = (
            "SELECT *, vector::dot(embedding, $query_embedding) AS similarity "  # noqa: S608
            f"FROM chunk WHERE {where_sql} "
            f"ORDER BY similarity DESC LIMIT {int(limit)}"
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
        filter_ast: FilterNode | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Full-text (BM25) search on chunk content.

        Uses SurrealDB's ``@1@`` match operator and ``search::score(1)``
        for BM25 ranking.

        ``filter_ast`` is accepted for protocol parity (the coordinator
        forwards it uniformly); this backend does not compile the recall
        filter, so it is ignored.
        """
        ns_rid = _rid("memory_namespace", namespace_id)
        where_clauses = [
            "(namespace = $ns_rid OR namespace.namespace_id = $ns_str)",
            "content @1@ $query_text",
        ]
        bindings: dict[str, Any] = {
            "ns_rid": ns_rid,
            "ns_str": str(namespace_id),
            "query_text": query_text,
            "limit": limit,
        }

        if created_after is not None:
            where_clauses.append("(source_timestamp ?? created_at) >= $created_after")
            bindings["created_after"] = created_after

        if created_before is not None:
            where_clauses.append("(source_timestamp ?? created_at) <= $created_before")
            bindings["created_before"] = created_before

        where_sql = " AND ".join(where_clauses)
        sql = (
            "SELECT *, search::score(1) AS rank "  # noqa: S608
            f"FROM chunk WHERE {where_sql} "
            "ORDER BY rank DESC LIMIT $limit"
        )

        try:
            rows = await self._conn.query(sql, bindings)
        except Exception as e:
            if "no suitable index" in str(e).lower():
                logger.warning("BM25 index not available — run optimize_storage() to create search indexes")
                return []
            raise
        return [(self._row_to_chunk(row), float(row.get("rank", 0.0))) for row in rows]

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Return the total number of chunks in a namespace."""
        ns_rid = _rid("memory_namespace", namespace_id)
        sql = "SELECT count() AS cnt FROM chunk WHERE namespace = $ns_rid GROUP ALL"
        row = await self._conn.query_one(sql, {"ns_rid": ns_rid})
        return int(row.get("cnt", 0)) if row else 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """Paginated listing of chunks in a namespace."""
        ns_rid = _rid("memory_namespace", namespace_id)
        sql = "SELECT * FROM chunk WHERE namespace = $ns_rid ORDER BY created_at DESC LIMIT $limit START $offset"
        rows = await self._conn.query(sql, {"ns_rid": ns_rid, "limit": limit, "offset": offset})
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity operations (vector storage side)
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> None:
        """Create an entity record for vector search."""
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

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> None:
        """Update an existing entity record, scoped to ``namespace_id`` (IDOR family).

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
        # No created_at on update
        bindings.pop("created_at", None)
        bindings["ns_rid"] = _rid("memory_namespace", namespace_id)
        bindings["ns_str"] = str(namespace_id)
        await self._conn.execute(sql, bindings)

    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Check whether an entity record exists within ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace, preventing cross-tenant entity-existence
        enumeration (IDOR \u2014 the IDOR family / the IDOR family).
        """
        sql = (
            "SELECT count() AS cnt FROM entity "
            "WHERE id = $rid AND (namespace = $ns_rid OR namespace.namespace_id = $ns_str) "
            "GROUP ALL"
        )
        row = await self._conn.query_one(
            sql,
            {
                "rid": _rid("entity", entity_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        return int(row.get("cnt", 0)) > 0 if row else False

    async def update_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Set the embedding vector on a single entity, scoped to ``namespace_id`` (IDOR family)."""
        sql = (
            "UPDATE $rid SET "
            "embedding = $embedding, "
            "embedding_model = $embedding_model, "
            "updated_at = $updated_at "
            "WHERE namespace = $ns_rid OR namespace.namespace_id = $ns_str"
        )
        await self._conn.execute(
            sql,
            {
                "rid": _rid("entity", entity_id),
                "embedding": list(embedding),
                "embedding_model": model,
                "updated_at": datetime.now(UTC),
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Batch-update entity embeddings, scoped to ``namespace_id`` (IDOR family).

        Uses a SurrealQL ``FOR`` loop to apply all updates in a single
        round-trip. Ids outside the caller's namespace are silently filtered
        by the per-row WHERE clause.
        """
        if not updates:
            return 0

        now_iso = datetime.now(UTC)
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
            "WHERE namespace = $ns_rid OR namespace.namespace_id = $ns_str "
            "}"
        )
        await self._conn.execute(
            sql,
            {
                "updates": update_dicts,
                "ns_rid": _rid("memory_namespace", namespace_id),
                "ns_str": str(namespace_id),
            },
        )
        return len(updates)

    @trace(
        "khora.surrealdb.upsert_entities_batch",
        include={"namespace_id"},
        result=lambda r: {"count": len(r)},
    )
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 200,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using match-by (namespace, name, entity_type).

        For existing entities: merge descriptions, sum mention_counts, union source_ids.
        For new entities: create.
        Returns list of (Entity, is_new) tuples.
        """
        if not entities:
            return []

        ns_rid = _rid("memory_namespace", namespace_id)

        # 1. Batch-fetch existing entities using tuple IN syntax (faster than N OR clauses)
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

        # 2. Separate into creates vs updates
        results: list[tuple[Entity, bool]] = []
        to_create: list[Entity] = []
        to_update: list[Entity] = []

        for entity in entities:
            key = (entity.name, entity.entity_type)
            existing = existing_map.get(key)
            if existing:
                existing.merge_with(entity)
                to_update.append(existing)
                results.append((existing, False))
            else:
                entity.namespace_id = namespace_id
                to_create.append(entity)
                results.append((entity, True))

        # 3. Batch create new entities via INSERT INTO (faster than FOR loops)
        if to_create:
            records = []
            for e in to_create:
                b = _entity_to_bindings(e)
                records.append(
                    {
                        "id": b["rid"],
                        "namespace": b["ns_rid"],
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

        # 4. Batch update existing entities
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
        """Dot-product similarity search over entity embeddings.

        Uses ``vector::dot()`` which is equivalent to cosine similarity
        for L2-normalized embeddings (~3x faster).
        """
        ns_rid = _rid("memory_namespace", namespace_id)
        sql = (
            "SELECT id, vector::dot(embedding, $query_embedding) AS similarity "  # noqa: S608
            "FROM entity WHERE (namespace = $ns_rid OR namespace.namespace_id = $ns_str) AND embedding IS NOT NULL "
            f"ORDER BY similarity DESC LIMIT {int(limit)}"
        )
        bindings: dict[str, Any] = {
            "ns_rid": ns_rid,
            "ns_str": str(namespace_id),
            "query_embedding": list(query_embedding),
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
    # Entity retrieval (by ID + namespace)
    # ------------------------------------------------------------------

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Fetch an entity by primary key, scoped to ``namespace_id``.

        Returns ``None`` if the entity does not exist OR belongs to a
        different namespace.  Previously this method silently accepted a
        ``namespace_id`` kwarg and ignored it — that was an IDOR bug
        (the IDOR family / the IDOR family).  ``namespace_id`` is now required and used
        to filter at the SurrealQL layer.
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

    # ------------------------------------------------------------------
    # Embedding statistics
    # ------------------------------------------------------------------

    async def get_embedding_stats(self, namespace_id: UUID) -> dict[str, int]:
        """Get statistics about embeddings in a namespace.

        Returns:
            Dict with ``chunk_embeddings`` and ``entity_embeddings`` counts.
        """
        ns_rid = _rid("memory_namespace", namespace_id)

        chunk_row = await self._conn.query_one(
            "SELECT count() AS cnt FROM chunk WHERE namespace = $ns_rid AND embedding IS NOT NULL GROUP ALL",
            {"ns_rid": ns_rid},
        )
        entity_row = await self._conn.query_one(
            "SELECT count() AS cnt FROM entity WHERE namespace = $ns_rid AND embedding IS NOT NULL GROUP ALL",
            {"ns_rid": ns_rid},
        )

        return {
            "chunk_embeddings": int(chunk_row.get("cnt", 0)) if chunk_row else 0,
            "entity_embeddings": int(entity_row.get("cnt", 0)) if entity_row else 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_to_bindings(self, chunk: Chunk) -> dict[str, Any]:
        """Convert a :class:`Chunk` to SurrealQL parameter bindings."""
        return {
            "rid": _rid("chunk", chunk.id),
            "ns_rid": _rid("memory_namespace", chunk.namespace_id),
            "doc_rid": _rid("document", chunk.document_id),
            "content": chunk.content,
            "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
            "embedding_model": chunk.embedding_model,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "token_count": chunk.token_count,
            "metadata_": chunk.metadata or {},
            "chunker_info": chunk.chunker_info or {},
            "created_at": chunk.created_at,
            "source_timestamp": chunk.source_timestamp,
            "occurred_at": chunk.occurred_at,
            "last_accessed_at": chunk.last_accessed_at,
            "session_id": str(chunk.session_id) if chunk.session_id else None,
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

        chunker_info_raw = row.get("chunker_info") or {}
        if not isinstance(chunker_info_raw, dict):
            chunker_info_raw = {}

        return Chunk(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content=row.get("content", ""),
            chunk_index=int(row.get("chunk_index", 0)),
            start_char=int(row.get("start_char", 0)),
            end_char=int(row.get("end_char", 0)),
            token_count=int(row.get("token_count", 0)),
            metadata=dict(custom_meta),
            chunker_info=dict(chunker_info_raw),
            embedding=embedding,
            embedding_model=row.get("embedding_model", ""),
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            source_timestamp=_parse_dt(row.get("source_timestamp")),
            occurred_at=_parse_dt(row.get("occurred_at")),
            last_accessed_at=_parse_dt(row.get("last_accessed_at")),
            session_id=_parse_uuid(row.get("session_id")) if row.get("session_id") else None,
        )

    # _row_to_entity is now a module-level function in _helpers.py;
    # keep a thin instance-method wrapper for backward compatibility.
    @staticmethod
    def _row_to_entity(row: dict[str, Any]) -> Entity:  # noqa: D401
        """Map a SurrealDB result row to a domain :class:`Entity`."""
        return _row_to_entity(row)
