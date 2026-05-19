"""SQLite + LanceDB vector adapter.

Implements :class:`VectorBackendProtocol` using a shared
:class:`EmbeddedStorageHandle`:

* Chunk metadata lives in SQLite.  The ``chunks`` / ``entities`` /
  ``chunks_fts`` tables are created by the Alembic dialect-gated
  migrations.
* Embeddings live in LanceDB for ANN search — there is no ``embedding``
  column on the SQLite side.
* Full-text search uses SQLite FTS5 (BM25).  ``chunks_fts`` is an
  external-content virtual table linked to ``chunks`` by rowid, with
  ``AFTER INSERT / UPDATE / DELETE`` triggers keeping it in sync (see
  migration 002).  This adapter inserts into ``chunks`` only; the
  trigger handles FTS5.
* Entity SQLite rows are owned by :class:`SQLiteLanceGraphAdapter`;
  this adapter stores only the entity embedding vector in LanceDB.

Vector writes to LanceDB happen after SQLite commits so SQLite remains
consistent if the LanceDB write fails. Compensating deletes log warnings
rather than raise, matching the sibling SurrealDB adapter's policy.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pyarrow as pa
from loguru import logger

from khora.core.models import Chunk, Entity
from khora.exceptions import EmbeddingError
from khora.storage.backends._fts5 import escape_fts5_query

from ._helpers import from_json_text, to_json_text, uuid_to_text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from .connection import EmbeddedStorageHandle


# Lazy ANN-index kick-in threshold. Below this, LanceDB's brute-force
# scan is faster than paying training cost.
_ANN_INDEX_THRESHOLD = 5_000


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dt_to_str(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    return datetime.fromisoformat(val)


def _validate_embedding(embedding: list[float], expected_dim: int, *, context: str) -> None:
    """Reject embeddings that LanceDB would silently mangle or PyArrow would
    reject with a cryptic schema error.

    Catches three real-world mis-configurations: an embedder returning ``[]``
    (early-exit path), a wrong-dim vector (model swap without config update),
    and a NaN-containing vector (numerical-instability bug upstream of khora).
    All raise :class:`khora.exceptions.EmbeddingError` with a message that
    names the offending shape — much friendlier than a five-frame Arrow trace.
    """
    if not embedding:
        raise EmbeddingError(
            f"{context}: refusing to store an empty embedding (expected dim={expected_dim}). "
            f"Check the embedder for an early-exit path on whitespace/short input."
        )
    if len(embedding) != expected_dim:
        raise EmbeddingError(
            f"{context}: embedding dim={len(embedding)} but storage configured for dim={expected_dim}. "
            f"This usually means the embedder model was changed without updating "
            f"`config.storage.embedding_dimension`."
        )
    # math.isnan is the cheapest scalar check; NaN can corrupt LanceDB ANN
    # training silently and produces wildly wrong cosine scores.
    if any(math.isnan(x) for x in embedding):
        raise EmbeddingError(
            f"{context}: embedding contains NaN at index "
            f"{next(i for i, x in enumerate(embedding) if math.isnan(x))}. "
            f"Cosine similarity is undefined for NaN — check the embedder for "
            f"numerical instability on this input."
        )


class SQLiteLanceVectorAdapter:
    """Vector backend backed by SQLite (metadata) + LanceDB (embeddings)."""

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
        # Cache table handles after first open (LanceDB async open is cheap
        # but avoids the catalog round-trip on hot paths).
        self._chunks_vec: Any = None
        self._entities_vec: Any = None
        # Row count at last index training. ``None`` = never trained, so the
        # next opportunity will train if the corpus is large enough. We
        # retrain once the row count grows by ``retrain_factor`` so a
        # long-running process doesn't keep querying a stale index after
        # the corpus has 10x'd.
        self._chunks_at_last_index: int | None = None
        self._entities_at_last_index: int | None = None
        # In-flight retrain tasks — kept so we don't schedule a second
        # rebuild while the first is still running, and so callers (tests)
        # can await completion.
        self._chunks_retrain_task: asyncio.Task[None] | None = None
        self._entities_retrain_task: asyncio.Task[None] | None = None
        self._index_lock = asyncio.Lock()
        # LanceDB is a single-writer store — serialize concurrent writes
        # (add/delete/create_index) per-table to avoid lost rows seen under
        # asyncio.gather(create_chunk, ...).
        self._lance_write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._handle.connect()

    async def disconnect(self) -> None:
        await self._handle.disconnect()

    async def is_healthy(self) -> bool:
        return await self._handle.is_healthy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _sqlite(self) -> Any:
        return self._handle.sqlite

    @property
    def _lance(self) -> Any:
        return self._handle.lance

    async def _chunks_table(self) -> Any:
        if self._chunks_vec is None:
            async with self._lance_write_lock:
                if self._chunks_vec is None:
                    self._chunks_vec = await self._lance.open_table("chunks_vec")
        return self._chunks_vec

    async def _entities_table(self) -> Any:
        if self._entities_vec is None:
            async with self._lance_write_lock:
                if self._entities_vec is None:
                    self._entities_vec = await self._lance.open_table("entities_vec")
        return self._entities_vec

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        return (await self.create_chunks_batch([chunk]))[0]

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        now = _now_iso()
        sqlite_rows = []
        lance_rows: list[dict[str, Any]] = []
        for c in chunks:
            sqlite_rows.append(
                (
                    uuid_to_text(c.id),
                    uuid_to_text(c.namespace_id),
                    uuid_to_text(c.document_id),
                    c.content,
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    c.token_count,
                    to_json_text(c.metadata or {}),
                    to_json_text(c.chunker_info or {}),
                    c.embedding_model,
                    _dt_to_str(c.created_at) or now,
                    _dt_to_str(c.source_timestamp),
                    uuid_to_text(c.session_id) if c.session_id is not None else None,
                )
            )
            if c.embedding:
                _validate_embedding(
                    list(c.embedding),
                    self._handle.config.embedding_dimension,
                    context=f"chunk id={c.id}",
                )
                lance_rows.append(
                    {
                        "id": uuid_to_text(c.id),
                        "namespace_id": uuid_to_text(c.namespace_id),
                        "document_id": uuid_to_text(c.document_id),
                        "created_at": c.created_at or datetime.now(UTC),
                        "vector": list(c.embedding),
                    }
                )

        # 1) SQLite: chunk metadata.  FTS5 is kept in sync by the AFTER-INSERT
        # trigger created by migration 002 — we do NOT insert into chunks_fts
        # directly.  LanceDB owns the embedding vector (no SQLite column).
        await self._sqlite.executemany(
            "INSERT INTO chunks "
            "(id, namespace_id, document_id, content, chunk_index, start_char, "
            "end_char, token_count, metadata, chunker_info, embedding_model, "
            "created_at, source_timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sqlite_rows,
        )
        await self._sqlite.commit()

        # 2) LanceDB: embeddings (compensation on failure — SQLite is
        # already committed; log and re-raise so callers can clean up).
        if lance_rows:
            tbl = await self._chunks_table()
            schema = await tbl.schema()
            arrow_tbl = pa.Table.from_pylist(lance_rows, schema=schema)
            try:
                async with self._lance_write_lock:
                    await tbl.add(arrow_tbl)
            except Exception:
                logger.exception(
                    "LanceDB add failed after SQLite commit for {} chunks — "
                    "SQLite metadata retained; caller should reconcile",
                    len(lance_rows),
                )
                raise

        return chunks

    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        cur = await self._sqlite.execute(
            "SELECT * FROM chunks WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(chunk_id), uuid_to_text(namespace_id)),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        params = [uuid_to_text(c) for c in chunk_ids]
        params.append(uuid_to_text(namespace_id))
        cur = await self._sqlite.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND namespace_id = ?",  # noqa: S608
            params,
        )
        rows = await cur.fetchall()
        result: dict[UUID, Chunk] = {}
        for row in rows:
            chunk = self._row_to_chunk(row)
            result[chunk.id] = chunk
        return result

    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        cur = await self._sqlite.execute(
            "SELECT * FROM chunks WHERE document_id = ? AND namespace_id = ? ORDER BY chunk_index",
            (uuid_to_text(document_id), uuid_to_text(namespace_id)),
        )
        rows = await cur.fetchall()
        return [self._row_to_chunk(r) for r in rows]

    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        *,
        namespace_id: UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete chunks by document, scoped to ``namespace_id`` (IGR-226).

        The ``session`` parameter is part of the protocol contract but the
        SQLite+LanceDB backend never participates in a SQLAlchemy session —
        if one is passed we treat it as a caller-managed transaction
        signal: we do NOT commit the SQLite side (caller will) and we
        skip the LanceDB compensation until they do.  With no session,
        we commit immediately and run the LanceDB delete as compensation.
        """
        doc_text = uuid_to_text(document_id)
        ns_text = uuid_to_text(namespace_id)

        # Enumerate before delete so we know whether LanceDB compensation is
        # needed.  FTS5 is kept in sync by the AFTER-DELETE trigger on
        # ``chunks`` — no manual ``chunks_fts`` delete needed.
        cur = await self._sqlite.execute(
            "SELECT id FROM chunks WHERE document_id = ? AND namespace_id = ?",
            (doc_text, ns_text),
        )
        rows = await cur.fetchall()
        count = len(rows)

        cur = await self._sqlite.execute(
            "DELETE FROM chunks WHERE document_id = ? AND namespace_id = ?",
            (doc_text, ns_text),
        )
        rowcount = cur.rowcount

        if session is None:
            await self._sqlite.commit()
            # Compensation delete on LanceDB — SQLite is already committed,
            # so we log failures but don't re-raise to keep metadata/vector
            # consistency under eventual convergence (next compact/rewrite
            # will re-sync).
            if count > 0:
                try:
                    tbl = await self._chunks_table()
                    async with self._lance_write_lock:
                        await tbl.delete(f"document_id = '{doc_text}' AND namespace_id = '{ns_text}'")
                except Exception:
                    logger.warning(
                        "LanceDB delete for document {} failed — orphaned vectors remain until next compaction",
                        doc_text,
                    )

        return rowcount if rowcount is not None else count

    async def count_chunks(self, namespace_id: UUID) -> int:
        cur = await self._sqlite.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE namespace_id = ?",
            (uuid_to_text(namespace_id),),
        )
        row = await cur.fetchone()
        return row["cnt"] if row else 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        cur = await self._sqlite.execute(
            "SELECT * FROM chunks WHERE namespace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (uuid_to_text(namespace_id), limit, offset),
        )
        rows = await cur.fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Chunk search
    # ------------------------------------------------------------------

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
    ) -> list[tuple[Chunk, float]]:
        """ANN search over chunks.

        Cosine distance on LanceDB; scores are converted to similarity
        (``1 - distance``) and filtered by ``min_similarity``.
        """
        tbl = await self._chunks_table()

        # Build LanceDB where clause — SQL-ish subset
        where = [f"namespace_id = '{uuid_to_text(namespace_id)}'"]
        if filter_document_ids:
            ids = ", ".join(f"'{uuid_to_text(d)}'" for d in filter_document_ids)
            where.append(f"document_id IN ({ids})")
        if created_after is not None:
            # LanceDB stores timestamps as microseconds since epoch; use the
            # ISO form since LanceDB's SQL parser accepts timestamp literals.
            where.append(f"created_at >= timestamp '{created_after.isoformat()}'")
        if created_before is not None:
            where.append(f"created_at < timestamp '{created_before.isoformat()}'")
        where_sql = " AND ".join(where)

        await self._maybe_build_chunks_index()

        q = (await tbl.search(list(query_embedding))).distance_type("cosine").where(where_sql).limit(max(limit, 1) * 2)
        results = await q.to_list()
        if not results:
            return []

        # Map to SQLite metadata. Preserve order.
        id_order: list[str] = [r["id"] for r in results]
        sims: dict[str, float] = {r["id"]: 1.0 - float(r.get("_distance", 0.0)) for r in results}

        # Temporal refinement on the SQLite side. SQLite is the source of
        # truth for chunk metadata and tracks ``source_timestamp`` (LanceDB
        # only stores ``created_at``), so we re-apply the bounds here using
        # ``COALESCE(source_timestamp, created_at)`` — matches the pgvector
        # backend's column-precedence rule (PR #470). Half-open
        # interval ``>= start AND < end`` to match the Chronicle pushdown
        # contract.
        # Defense-in-depth (IGR-226): also enforce namespace_id on the SQLite
        # side rather than trusting LanceDB's filter alone. If LanceDB's
        # where-clause regressed, the SQLite filter would still keep cross-
        # namespace rows out of the result.
        sql_parts = [
            f"SELECT * FROM chunks WHERE id IN ({','.join('?' for _ in id_order)}) "  # noqa: S608
            "AND namespace_id = ?"
        ]
        params: list[Any] = [*id_order, uuid_to_text(namespace_id)]
        if created_after is not None:
            sql_parts.append("AND COALESCE(source_timestamp, created_at) >= ?")
            params.append(_dt_to_str(created_after))
        if created_before is not None:
            sql_parts.append("AND COALESCE(source_timestamp, created_at) < ?")
            params.append(_dt_to_str(created_before))
        cur = await self._sqlite.execute(" ".join(sql_parts), params)
        rows = await cur.fetchall()
        by_id = {row["id"]: row for row in rows}

        out: list[tuple[Chunk, float]] = []
        for cid in id_order:
            sim = sims[cid]
            if sim < min_similarity:
                continue
            row = by_id.get(cid)
            if row is None:
                # LanceDB row orphaned (SQLite metadata missing) — skip.
                continue
            out.append((self._row_to_chunk(row), sim))
            if len(out) >= limit:
                break
        return out

    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",  # noqa: ARG002 — accepted for protocol parity
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[tuple[Chunk, float]]:
        """FTS5 BM25 ranking.

        Migration 002 creates ``chunks_fts`` as an external-content FTS5
        table linked to ``chunks`` by rowid (``content='chunks'``,
        ``content_rowid='rowid'``) with triggers keeping them in sync.
        We join on ``chunks_fts.rowid = chunks.rowid`` and filter by
        ``namespace_id`` on the chunks side.

        SQLite's ``bm25()`` returns lower-is-better; we negate it so the
        return value matches the "higher is better" semantics used by
        the pgvector and SurrealDB siblings.
        """
        match_expr = escape_fts5_query(query_text)
        if not match_expr:
            return []
        sql_parts = [
            "SELECT c.*, bm25(chunks_fts) AS bm FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.namespace_id = ?"
        ]
        params: list[Any] = [match_expr, uuid_to_text(namespace_id)]
        if created_after is not None:
            sql_parts.append("AND COALESCE(c.source_timestamp, c.created_at) >= ?")
            params.append(_dt_to_str(created_after))
        if created_before is not None:
            sql_parts.append("AND COALESCE(c.source_timestamp, c.created_at) <= ?")
            params.append(_dt_to_str(created_before))
        sql_parts.append("ORDER BY bm ASC LIMIT ?")
        params.append(limit)

        cur = await self._sqlite.execute(" ".join(sql_parts), params)
        rows = await cur.fetchall()
        return [(self._row_to_chunk(r), -float(r["bm"])) for r in rows]

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> None:
        """Persist the entity's embedding to LanceDB.

        The SQLite ``entities`` row is owned by :class:`SQLiteLanceGraphAdapter`
        — the coordinator calls ``graph.create_entity`` and
        ``vector.create_entity`` in parallel, so this adapter must only
        touch LanceDB (writing to SQLite here would race the graph write
        and also conflict on the primary key).
        """
        if entity.embedding:
            await self._upsert_entity_vector(entity.id, entity.namespace_id, entity.embedding)

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> None:
        """Refresh the entity's embedding in LanceDB, scoped to ``namespace_id``.

        See :meth:`create_entity` — SQLite entity metadata belongs to
        the graph adapter. The ``namespace_id`` kwarg is defense-in-depth
        (IGR-226) — asserted equal to ``entity.namespace_id``.
        """
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        if entity.embedding:
            await self._upsert_entity_vector(entity.id, entity.namespace_id, entity.embedding)

    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Check if an entity exists in SQLite within ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant entity-existence
        enumeration (IDOR — IGR-221).
        """
        cur = await self._sqlite.execute(
            "SELECT 1 FROM entities WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(entity_id), uuid_to_text(namespace_id)),
        )
        row = await cur.fetchone()
        return row is not None

    async def update_entity_embedding(  # noqa: ARG002 — model is tracked on the graph row, not in LanceDB
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Update an entity's embedding in LanceDB, scoped to ``namespace_id``.

        No-op when the entity does not exist in the given namespace
        (cross-namespace IDOR — IGR-226).
        """
        # Verify the entity exists in this namespace before writing LanceDB.
        cur = await self._sqlite.execute(
            "SELECT 1 FROM entities WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(entity_id), uuid_to_text(namespace_id)),
        )
        row = await cur.fetchone()
        if row is None:
            return

        await self._upsert_entity_vector(entity_id, namespace_id, embedding)

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        if not updates:
            return 0

        # Fetch namespace_ids from SQLite — restricted to ``namespace_id``
        # so ids outside the caller's namespace are silently dropped
        # (cross-namespace IDOR — IGR-226). The ``embedding_model`` is not
        # persisted in LanceDB (it's a SQLite column on ``entities``,
        # managed by the graph adapter / ORM).
        ids_text = [uuid_to_text(u) for u, _, _ in updates]
        placeholders = ",".join("?" for _ in ids_text)
        cur = await self._sqlite.execute(
            f"SELECT id, namespace_id FROM entities WHERE id IN ({placeholders}) "  # noqa: S608
            "AND namespace_id = ?",
            [*ids_text, uuid_to_text(namespace_id)],
        )
        rows = await cur.fetchall()
        ns_by_id = {r["id"]: r["namespace_id"] for r in rows}

        # Upsert LanceDB side: batch delete + batch add.
        tbl = await self._entities_table()
        existing = [uuid_to_text(eid) for eid, _, _ in updates if uuid_to_text(eid) in ns_by_id]
        async with self._lance_write_lock:
            if existing:
                in_list = ", ".join(f"'{x}'" for x in existing)
                try:
                    await tbl.delete(f"id IN ({in_list})")
                except Exception:
                    logger.debug("LanceDB entity delete failed during batch upsert (may be first write)")

            schema = await tbl.schema()
            lance_rows: list[dict[str, Any]] = []
            for eid, emb, _ in updates:
                eid_text = uuid_to_text(eid)
                ns = ns_by_id.get(eid_text)
                if ns is None:
                    # Unknown entity — nothing to vector-store for it.
                    continue
                lance_rows.append(
                    {
                        "id": eid_text,
                        "namespace_id": ns,
                        "vector": list(emb),
                    }
                )
            if lance_rows:
                arrow_tbl = pa.Table.from_pylist(lance_rows, schema=schema)
                await tbl.add(arrow_tbl)

        return len(lance_rows)

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        tbl = await self._entities_table()
        await self._maybe_build_entities_index()
        q = (
            (await tbl.search(list(query_embedding)))
            .distance_type("cosine")
            .where(f"namespace_id = '{uuid_to_text(namespace_id)}'")
            .limit(max(limit, 1) * 2)
        )
        rows = await q.to_list()
        out: list[tuple[UUID, float]] = []
        for r in rows:
            sim = 1.0 - float(r.get("_distance", 0.0))
            if sim < min_similarity:
                continue
            out.append((UUID(r["id"]), sim))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # LanceDB helpers
    # ------------------------------------------------------------------

    async def _upsert_entity_vector(self, entity_id: UUID, namespace_id: UUID, embedding: list[float]) -> None:
        """Upsert a single row into ``entities_vec``.

        LanceDB has no in-place vector update, so we delete any existing
        row with the same id and then add.
        """
        _validate_embedding(
            list(embedding),
            self._handle.config.embedding_dimension,
            context=f"entity id={entity_id}",
        )
        tbl = await self._entities_table()
        eid_text = uuid_to_text(entity_id)
        async with self._lance_write_lock:
            try:
                await tbl.delete(f"id = '{eid_text}'")
            except Exception:
                # Table may be empty on first write — ignore.
                logger.debug("LanceDB entities_vec delete pre-upsert failed (may be empty)")
            schema = await tbl.schema()
            arrow_tbl = pa.Table.from_pylist(
                [
                    {
                        "id": eid_text,
                        "namespace_id": uuid_to_text(namespace_id),
                        "vector": list(embedding),
                    }
                ],
                schema=schema,
            )
            await tbl.add(arrow_tbl)

    async def _maybe_build_chunks_index(self) -> None:
        await self._maybe_build_or_retrain(
            self._chunks_table,
            "chunks_vec",
            kind="chunks",
        )

    async def _maybe_build_entities_index(self) -> None:
        await self._maybe_build_or_retrain(
            self._entities_table,
            "entities_vec",
            kind="entities",
        )

    async def _maybe_build_or_retrain(
        self,
        get_table: Any,
        label: str,
        *,
        kind: str,
    ) -> None:
        """Build the ANN index on first need, or retrain if the corpus has grown.

        Called from the search path. Cheap when no work is needed: just a
        ``count_rows()`` plus a ratio check. When a retrain is needed the
        actual rebuild runs in a background task so the search isn't blocked
        — readers keep using the previous index until the new one is swapped
        in atomically by ``create_index(replace=True)``.

        Trigger: ``current_rows >= retrain_factor * rows_at_last_train``.
        Retrain is disabled when ``retrain_factor <= 1.0``.
        """
        if self._handle.config.lance_index == "brute":
            return

        tbl = await get_table()
        try:
            rows = await tbl.count_rows()
        except Exception:
            rows = 0

        last = self._chunks_at_last_index if kind == "chunks" else self._entities_at_last_index
        retrain_factor = self._handle.config.retrain_factor

        if last is None:
            # Never trained — train inline so the first search after the
            # threshold is crossed sees the index. Below threshold, _build_index
            # no-ops and we leave _chunks_at_last_index unset so the next
            # search reconsiders.
            async with self._index_lock:
                # Re-check under the lock — another concurrent searcher may
                # have trained while we waited.
                last_locked = self._chunks_at_last_index if kind == "chunks" else self._entities_at_last_index
                if last_locked is not None:
                    return
                trained_at = await self._build_index(tbl, label, rows)
                if trained_at is not None:
                    if kind == "chunks":
                        self._chunks_at_last_index = trained_at
                    else:
                        self._entities_at_last_index = trained_at
            return

        if retrain_factor <= 1.0 or last <= 0:
            return
        if rows < int(last * retrain_factor):
            return

        # Retrain in the background — single-flight per table.
        existing = self._chunks_retrain_task if kind == "chunks" else self._entities_retrain_task
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(self._retrain_in_background(get_table, label, kind, last))
        if kind == "chunks":
            self._chunks_retrain_task = task
        else:
            self._entities_retrain_task = task

    async def _retrain_in_background(
        self,
        get_table: Any,
        label: str,
        kind: str,
        old_rows: int,
    ) -> None:
        async with self._index_lock:
            tbl = await get_table()
            try:
                rows_now = await tbl.count_rows()
            except Exception:
                rows_now = 0
            logger.info(
                "Retraining LanceDB IVF-PQ index on {}: {} -> {} rows",
                label,
                old_rows,
                rows_now,
            )
            trained_at = await self._build_index(tbl, label, rows_now)
            if trained_at is not None:
                if kind == "chunks":
                    self._chunks_at_last_index = trained_at
                else:
                    self._entities_at_last_index = trained_at

    async def _build_index(self, tbl: Any, label: str, rows: int) -> int | None:
        """Create an ANN index following the handle's ``lance_index`` policy.

        - ``"brute"``: skip index creation entirely (brute-force scans).
        - ``"hnsw"``: build an HNSW index unconditionally.
        - ``"auto"`` / ``"ivf_pq"``: build IVF_PQ only once the table has
          at least ``_ANN_INDEX_THRESHOLD`` rows; below that LanceDB's
          brute-force scan is faster than paying training cost.

        Returns the row count the index was trained on, or ``None`` if no
        index was built (so the caller leaves the "last trained" marker
        unset and the next search reconsiders).
        """
        mode = self._handle.config.lance_index
        if mode == "brute":
            return None

        async with self._lance_write_lock:
            try:
                if mode == "hnsw":
                    from lancedb.index import HnswSq

                    await tbl.create_index(
                        column="vector",
                        config=HnswSq(
                            distance_type="cosine",
                            m=self._handle.config.hnsw_m,
                        ),
                        replace=True,
                    )
                    logger.info("Created HNSW index on {} ({} rows)", label, rows)
                    return rows

                # auto / ivf_pq
                if rows < _ANN_INDEX_THRESHOLD:
                    return None

                from lancedb.index import IvfPq

                dim = self._handle.config.embedding_dimension
                partitions = self._handle.config.ivf_partitions or max(1, rows // _ANN_INDEX_THRESHOLD)
                sub_vectors = min(96, max(1, dim // 16))
                await tbl.create_index(
                    column="vector",
                    config=IvfPq(
                        distance_type="cosine",
                        num_partitions=partitions,
                        num_sub_vectors=sub_vectors,
                    ),
                    replace=True,
                )
                logger.info(
                    "Created IVF_PQ index on {} (rows={}, partitions={}, sub_vectors={})",
                    label,
                    rows,
                    partitions,
                    sub_vectors,
                )
                return rows
            except Exception as exc:
                logger.warning("Failed to build ANN index on {}: {}", label, exc)
                return None

    # ------------------------------------------------------------------
    # Row → domain helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: Any) -> Chunk:
        # Embeddings live in LanceDB — not in the SQLite ``chunks`` table.
        # Callers that need the vector should fetch it from LanceDB; the
        # search paths return it indirectly via similarity results.
        def _row_get(key: str) -> Any:
            try:
                return row[key]
            except (KeyError, IndexError):
                return None

        chunker_info_raw = _row_get("chunker_info")
        return Chunk(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            document_id=UUID(row["document_id"]),
            content=row["content"] or "",
            chunk_index=row["chunk_index"] or 0,
            start_char=row["start_char"] or 0,
            end_char=row["end_char"] or 0,
            token_count=row["token_count"] or 0,
            metadata=from_json_text(row["metadata"]) if row["metadata"] else {},
            chunker_info=from_json_text(chunker_info_raw) if chunker_info_raw else {},
            embedding=None,
            embedding_model=row["embedding_model"] or "",
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            source_timestamp=_parse_dt(row["source_timestamp"]),
            session_id=(UUID(row["session_id"]) if row["session_id"] else None),
        )


__all__ = ["SQLiteLanceVectorAdapter"]
