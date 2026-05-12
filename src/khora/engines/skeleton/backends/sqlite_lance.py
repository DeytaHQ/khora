"""SQLite + LanceDB backend for the Skeleton engine's temporal vector store.

Mirror of :class:`PgVectorTemporalStore` for the embedded unified backend.
Reuses the shared :class:`EmbeddedStorageHandle` opened by the unified
``StorageCoordinator`` so we don't fork a second SQLite or LanceDB
connection.

Schema layout
-------------
* ``khora_chunks`` (SQLite) — temporal-chunk metadata table managed
  directly by this store.  No Alembic migration: created by
  :meth:`connect` if absent (analogous to PgVectorTemporalStore which
  calls ``metadata.create_all`` in its connect()).
* ``khora_chunks_fts`` (SQLite FTS5) — external-content virtual table
  over ``khora_chunks.content``.  Triggers keep it in sync.
* ``khora_chunks_vec`` (LanceDB) — embeddings only.  Mirrors the
  ``chunks_vec`` Arrow schema used by the main vector adapter.

Search
------
Vector search uses LanceDB cosine distance; BM25 uses SQLite FTS5.
Hybrid mode fuses ranks via RRF (same constants as the PG backend).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pyarrow as pa
from loguru import logger

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
)
from khora.storage.backends._fts5 import escape_fts5_query
from khora.storage.backends.sqlite_lance._helpers import (
    from_json_text,
    to_json_text,
    uuid_to_text,
)
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.storage.backends.sqlite_lance.connection import EmbeddedStorageHandle


# DDL for the SQLite-side temporal chunk table + FTS5 mirror.  Mirrors the
# pgvector ``khora_chunks`` table column-for-column with two tweaks:
# - no ``embedding`` column (LanceDB owns the vector)
# - no ``content_tsv``; FTS5 lives in ``khora_chunks_fts`` and is kept in
#   sync via triggers (same pattern as the main ``chunks`` / ``chunks_fts``
#   pair set up by migration 002).
_KHORA_CHUNKS_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS khora_chunks (
        id TEXT PRIMARY KEY,
        namespace_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        content TEXT NOT NULL,
        occurred_at TEXT,
        created_at TEXT NOT NULL,
        source_system TEXT,
        author TEXT,
        channel TEXT,
        tags TEXT NOT NULL DEFAULT '[]',
        confidence REAL NOT NULL DEFAULT 1.0,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_namespace ON khora_chunks(namespace_id)",
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_document ON khora_chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_occurred ON khora_chunks(occurred_at)",
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_author ON khora_chunks(author)",
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_channel ON khora_chunks(channel)",
    "CREATE INDEX IF NOT EXISTS ix_khora_chunks_source ON khora_chunks(source_system)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS khora_chunks_fts USING fts5(
        content, content='khora_chunks', content_rowid='rowid', tokenize='porter'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS khora_chunks_ai AFTER INSERT ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS khora_chunks_ad AFTER DELETE ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(khora_chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS khora_chunks_au AFTER UPDATE ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(khora_chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
        INSERT INTO khora_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
)


def _vector_type(dim: int, use_halfvec: bool) -> pa.DataType:
    value_type = pa.float16() if use_halfvec else pa.float32()
    return pa.list_(value_type, list_size=dim)


def _khora_chunks_vec_schema(dim: int, use_halfvec: bool) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("namespace_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=True),
            pa.field("vector", _vector_type(dim, use_halfvec), nullable=False),
        ]
    )


def _dt_to_iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    return datetime.fromisoformat(val)


# RRF constant — matches PgVectorTemporalStore so test expectations carry over.
_RRF_K = 60


class SQLiteLanceTemporalStore(TemporalVectorStore):
    """SQLite + LanceDB implementation of :class:`TemporalVectorStore`.

    Reuses the unified backend's :class:`EmbeddedStorageHandle` so the
    temporal store and the main coordinator share one aiosqlite + LanceDB
    pair. The handle is opened by ``StorageCoordinator.connect()``; this
    store only provisions its own table footprint.
    """

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
        self._connected = False
        self._chunks_vec: Any = None
        # LanceDB is a single-writer store — serialize add/delete on the
        # khora_chunks_vec table per-instance to mirror the main vector
        # adapter's policy.
        self._lance_write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        # The handle is opened by StorageCoordinator before the temporal
        # store's connect() runs (see SkeletonConstructionEngine.connect).
        # Defensive call here keeps the contract symmetrical with the
        # other backends — handle.connect() is idempotent.
        await self._handle.connect()

        sqlite = self._handle.sqlite
        for stmt in _KHORA_CHUNKS_SCHEMA:
            await sqlite.execute(stmt)
        await sqlite.commit()

        # Create the LanceDB vector table for khora_chunks. exist_ok keeps
        # this idempotent across processes/test runs.
        lance = self._handle.lance
        cfg = self._handle.config
        await lance.create_table(
            "khora_chunks_vec",
            schema=_khora_chunks_vec_schema(cfg.embedding_dimension, cfg.use_halfvec),
            exist_ok=True,
        )

        self._connected = True
        logger.info("SQLiteLanceTemporalStore connected")

    async def disconnect(self) -> None:
        # The handle is owned by the unified coordinator; do NOT close it
        # here — StorageCoordinator.disconnect() handles that.
        self._chunks_vec = None
        self._connected = False
        logger.info("SQLiteLanceTemporalStore disconnected")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _sqlite(self) -> Any:
        return self._handle.sqlite

    async def _vec_table(self) -> Any:
        if self._chunks_vec is None:
            async with self._lance_write_lock:
                if self._chunks_vec is None:
                    self._chunks_vec = await self._handle.lance.open_table("khora_chunks_vec")
        return self._chunks_vec

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        return (await self.create_chunks_batch([chunk]))[0]

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        if not chunks:
            return []

        now_iso = datetime.now(UTC).isoformat()
        sqlite_rows: list[tuple[Any, ...]] = []
        lance_rows: list[dict[str, Any]] = []

        for c in chunks:
            cid = c.id or uuid4()
            c.id = cid
            sqlite_rows.append(
                (
                    uuid_to_text(cid),
                    uuid_to_text(c.namespace_id),
                    uuid_to_text(c.document_id),
                    c.content,
                    _dt_to_iso(c.occurred_at),
                    _dt_to_iso(c.created_at) or now_iso,
                    c.source_system,
                    c.author,
                    c.channel,
                    to_json_text(c.tags or []),
                    float(c.confidence) if c.confidence is not None else 1.0,
                    to_json_text(c.metadata or {}),
                )
            )
            if c.embedding:
                lance_rows.append(
                    {
                        "id": uuid_to_text(cid),
                        "namespace_id": uuid_to_text(c.namespace_id),
                        "document_id": uuid_to_text(c.document_id),
                        "vector": list(c.embedding),
                    }
                )

        # 1) SQLite first — FTS5 mirror is updated by triggers.
        await self._sqlite.executemany(
            "INSERT INTO khora_chunks "
            "(id, namespace_id, document_id, content, occurred_at, created_at, "
            "source_system, author, channel, tags, confidence, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sqlite_rows,
        )
        await self._sqlite.commit()

        # 2) LanceDB second.  Same compensation policy as
        # SQLiteLanceVectorAdapter — log-and-raise so the caller can
        # reconcile (SQLite is already committed).
        if lance_rows:
            tbl = await self._vec_table()
            schema = await tbl.schema()
            arrow_tbl = pa.Table.from_pylist(lance_rows, schema=schema)
            try:
                async with self._lance_write_lock:
                    await tbl.add(arrow_tbl)
            except Exception:
                logger.exception(
                    "LanceDB add failed after SQLite commit for {} khora_chunks rows",
                    len(lance_rows),
                )
                raise

        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        cur = await self._sqlite.execute(
            "SELECT * FROM khora_chunks WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(chunk_id), uuid_to_text(namespace_id)),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        cur = await self._sqlite.execute(
            "DELETE FROM khora_chunks WHERE id = ? AND namespace_id = ?",
            (uuid_to_text(chunk_id), uuid_to_text(namespace_id)),
        )
        rowcount = cur.rowcount
        await self._sqlite.commit()

        if rowcount and rowcount > 0:
            tbl = await self._vec_table()
            try:
                async with self._lance_write_lock:
                    await tbl.delete(f"id = '{uuid_to_text(chunk_id)}'")
            except Exception:
                logger.warning(
                    "LanceDB delete for khora_chunk {} failed — orphaned vector remains",
                    chunk_id,
                )
            return True
        return False

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        doc_text = uuid_to_text(document_id)
        ns_text = uuid_to_text(namespace_id)
        cur = await self._sqlite.execute(
            "DELETE FROM khora_chunks WHERE document_id = ? AND namespace_id = ?",
            (doc_text, ns_text),
        )
        rowcount = cur.rowcount or 0
        await self._sqlite.commit()

        if rowcount > 0:
            tbl = await self._vec_table()
            try:
                async with self._lance_write_lock:
                    await tbl.delete(f"document_id = '{doc_text}'")
            except Exception:
                logger.warning(
                    "LanceDB delete for khora_chunks document {} failed — orphaned vectors remain",
                    doc_text,
                )
        return rowcount

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        temporal_filter: TemporalFilter | None = None,
        hybrid_alpha: float | None = None,
        query_text: str | None = None,
    ) -> list[TemporalSearchResult]:
        with trace_span(
            "khora.temporal_store.search",
            namespace_id=str(namespace_id),
            limit=limit,
            hybrid=hybrid_alpha is not None,
        ) as _span:
            results = await self._search_inner(
                namespace_id,
                query_embedding,
                limit=limit,
                min_similarity=min_similarity,
                temporal_filter=temporal_filter,
                hybrid_alpha=hybrid_alpha,
                query_text=query_text,
            )
            _span.set_attribute("result_count", len(results))
            return results

    async def _search_inner(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int,
        min_similarity: float,
        temporal_filter: TemporalFilter | None,
        hybrid_alpha: float | None,
        query_text: str | None,
    ) -> list[TemporalSearchResult]:
        # Vector pass.  Fetch 2× when fusion is requested so RRF has a
        # broader rank corpus.
        vector_results = await self._vector_search(
            namespace_id,
            query_embedding,
            temporal_filter,
            limit * 2 if hybrid_alpha is not None else limit,
            min_similarity,
        )

        if hybrid_alpha is not None and query_text:
            bm25_results = await self._bm25_search(
                namespace_id,
                query_text,
                temporal_filter,
                limit * 2,
            )
            return self._rrf_fusion(vector_results, bm25_results, hybrid_alpha, limit)

        results = vector_results[:limit]

        # Quality fix mirrored from PgVectorTemporalStore: keyword fallback
        # when vector recall is thin.
        if len(results) < limit and query_text:
            needed = limit - len(results)
            existing_ids = {str(r.chunk.id) for r in results}
            bm25_results = await self._bm25_search(
                namespace_id,
                query_text,
                temporal_filter,
                needed + len(existing_ids),
            )
            for bm25_result in bm25_results:
                cid = str(bm25_result.chunk.id)
                if cid in existing_ids:
                    continue
                bm25_result.combined_score = (bm25_result.bm25_score or 0.0) * 0.8
                results.append(bm25_result)
                existing_ids.add(cid)
                if len(results) >= limit:
                    break

        return results

    async def _vector_search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        temporal_filter: TemporalFilter | None,
        limit: int,
        min_similarity: float,
    ) -> list[TemporalSearchResult]:
        tbl = await self._vec_table()
        ns_text = uuid_to_text(namespace_id)
        # LanceDB SQL-ish where: namespace + optional document_id IN list.
        where_parts = [f"namespace_id = '{ns_text}'"]
        # Per-temporal filter pushdown into LanceDB is expensive (no index);
        # we filter on the SQLite side after joining metadata.
        where_sql = " AND ".join(where_parts)

        # Over-fetch so the post-filter pass still has enough rows.
        fetch = max(limit * 4, 16)
        q = (await tbl.search(list(query_embedding))).distance_type("cosine").where(where_sql).limit(fetch)
        rows = await q.to_list()
        if not rows:
            return []

        sims: dict[str, float] = {r["id"]: 1.0 - float(r.get("_distance", 0.0)) for r in rows}
        ordered_ids = [r["id"] for r in rows]

        # Pull SQLite metadata for matched ids, applying temporal filters.
        placeholders = ",".join("?" for _ in ordered_ids)
        sql = f"SELECT * FROM khora_chunks WHERE id IN ({placeholders})"  # noqa: S608
        params: list[Any] = list(ordered_ids)
        filter_sql, filter_params = self._build_filter_clause(temporal_filter)
        if filter_sql:
            sql += f" AND {filter_sql}"
            params.extend(filter_params)

        cur = await self._sqlite.execute(sql, params)
        meta_rows = await cur.fetchall()
        by_id = {row["id"]: row for row in meta_rows}

        out: list[TemporalSearchResult] = []
        for cid in ordered_ids:
            sim = sims[cid]
            if sim < min_similarity:
                continue
            row = by_id.get(cid)
            if row is None:
                continue
            chunk = self._row_to_chunk(row)
            # Tag-filter is enforced post-decode because tags are JSON-text
            # in SQLite (not a native ARRAY).
            if temporal_filter and temporal_filter.tags:
                if not all(t in chunk.tags for t in temporal_filter.tags):
                    continue
            out.append(
                TemporalSearchResult(
                    chunk=chunk,
                    similarity=sim,
                    bm25_score=None,
                    combined_score=sim,
                )
            )
            if len(out) >= limit:
                break
        return out

    async def _bm25_search(
        self,
        namespace_id: UUID,
        query_text: str,
        temporal_filter: TemporalFilter | None,
        limit: int,
    ) -> list[TemporalSearchResult]:
        # FTS5 MATCH; SQLite's bm25() is lower-is-better — negate so the
        # semantics match the PG/SurrealDB siblings.
        match_expr = escape_fts5_query(query_text)
        if not match_expr:
            return []
        sql_parts = [
            "SELECT c.*, bm25(khora_chunks_fts) AS bm FROM khora_chunks_fts "
            "JOIN khora_chunks c ON c.rowid = khora_chunks_fts.rowid "
            "WHERE khora_chunks_fts MATCH ? AND c.namespace_id = ?"
        ]
        params: list[Any] = [match_expr, uuid_to_text(namespace_id)]

        filter_sql, filter_params = self._build_filter_clause(temporal_filter, alias="c")
        if filter_sql:
            sql_parts.append(f"AND {filter_sql}")
            params.extend(filter_params)
        # Over-fetch when tag filter is set so the post-decode pass still
        # has enough rows.
        fetch = limit * 4 if temporal_filter and temporal_filter.tags else limit
        sql_parts.append("ORDER BY bm ASC LIMIT ?")
        params.append(fetch)

        cur = await self._sqlite.execute(" ".join(sql_parts), params)
        rows = await cur.fetchall()

        out: list[TemporalSearchResult] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if temporal_filter and temporal_filter.tags:
                if not all(t in chunk.tags for t in temporal_filter.tags):
                    continue
            score = -float(row["bm"])
            out.append(
                TemporalSearchResult(
                    chunk=chunk,
                    similarity=0.0,
                    bm25_score=score,
                    combined_score=score,
                )
            )
            if len(out) >= limit:
                break
        return out

    def _rrf_fusion(
        self,
        vector_results: list[TemporalSearchResult],
        bm25_results: list[TemporalSearchResult],
        alpha: float,
        limit: int,
        k: int = _RRF_K,
    ) -> list[TemporalSearchResult]:
        vector_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(bm25_results)}
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            v_rank = vector_ranks.get(cid, len(vector_results) + 100)
            b_rank = bm25_ranks.get(cid, len(bm25_results) + 100)
            rrf_scores[cid] = alpha * (1 / (k + v_rank)) + (1 - alpha) * (1 / (k + b_rank))

        result_map: dict[str, TemporalSearchResult] = {str(r.chunk.id): r for r in vector_results}
        for r in bm25_results:
            cid = str(r.chunk.id)
            if cid in result_map:
                result_map[cid].bm25_score = r.bm25_score
            else:
                result_map[cid] = r

        out: list[TemporalSearchResult] = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            r = result_map[cid]
            r.combined_score = score
            out.append(r)
        return out

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _build_filter_clause(
        self,
        f: TemporalFilter | None,
        *,
        alias: str | None = None,
    ) -> tuple[str, list[Any]]:
        if f is None:
            return "", []

        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []

        if f.occurred_after:
            clauses.append(f"{prefix}occurred_at >= ?")
            params.append(_dt_to_iso(f.occurred_after))
        if f.occurred_before:
            clauses.append(f"{prefix}occurred_at < ?")
            params.append(_dt_to_iso(f.occurred_before))
        if f.created_after:
            clauses.append(f"{prefix}created_at >= ?")
            params.append(_dt_to_iso(f.created_after))
        if f.created_before:
            clauses.append(f"{prefix}created_at < ?")
            params.append(_dt_to_iso(f.created_before))
        if f.source_system:
            clauses.append(f"{prefix}source_system = ?")
            params.append(f.source_system)
        if f.author:
            clauses.append(f"{prefix}author = ?")
            params.append(f.author)
        if f.channel:
            clauses.append(f"{prefix}channel = ?")
            params.append(f.channel)

        # tags: applied post-decode in caller (JSON-text in SQLite).
        # additional metadata filters are also deferred — the test matrix
        # doesn't exercise them on the embedded path yet, and JSON-path
        # querying differs across SQLite versions.

        return " AND ".join(clauses), params

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: Any) -> TemporalChunk:
        # tags is JSON-text holding a list; `from_json_text` enforces dict
        # so parse raw json here instead.
        import json as _json

        raw_tags = row["tags"] or "[]"
        try:
            parsed = _json.loads(raw_tags)
            tags_list = parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            tags_list = []

        meta = from_json_text(row["metadata"]) if row["metadata"] else {}

        return TemporalChunk(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            document_id=UUID(row["document_id"]),
            content=row["content"] or "",
            embedding=None,  # vectors live in LanceDB
            occurred_at=_parse_dt(row["occurred_at"]),
            created_at=_parse_dt(row["created_at"]),
            source_system=row["source_system"],
            author=row["author"],
            channel=row["channel"],
            tags=tags_list,
            confidence=row["confidence"] if row["confidence"] is not None else 1.0,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        if not self._connected:
            return {"status": "disconnected", "backend": "sqlite_lance"}
        try:
            ok = await self._handle.is_healthy()
            return {
                "status": "healthy" if ok else "unhealthy",
                "backend": "sqlite_lance",
            }
        except Exception as e:
            return {"status": "unhealthy", "backend": "sqlite_lance", "error": str(e)}


__all__ = ["SQLiteLanceTemporalStore"]
