"""SurrealDB backend for the Skeleton engine's temporal vector store.

Implements TemporalVectorStore using SurrealDB's native HNSW vector
indexing and BM25 full-text search.  Reuses the shared
:class:`SurrealDBConnection` for client lifecycle management.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models.document import Chunk
from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
    temporal_chunk_to_chunk,
)
from khora.filter.model import Op
from khora.storage.backends.surrealdb._helpers import (
    _parse_dt,
    _parse_uuid,
    _rid,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection
from khora.telemetry import trace, trace_span

if TYPE_CHECKING:
    from khora.config import KhoraConfig
    from khora.config.schema import SurrealDBConfig
    from khora.filter.ast import FilterNode


def _ensure_list(value: Any) -> list:
    """Coerce a value to a list for SurrealDB array fields.

    Handles JSON strings (e.g., ``'["a","b"]'``) from PostgreSQL metadata
    that weren't deserialized to native Python lists.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [value] if value else []
    return list(value)


# ---------------------------------------------------------------------------
# Schema for the temporal_chunk table
# ---------------------------------------------------------------------------

_TEMPORAL_CHUNK_SCHEMA = """
DEFINE ANALYZER IF NOT EXISTS khora_fulltext TOKENIZERS blank, class FILTERS lowercase, snowball(english);

DEFINE TABLE IF NOT EXISTS temporal_chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS namespace ON temporal_chunk TYPE record<memory_namespace>;
DEFINE FIELD IF NOT EXISTS document ON temporal_chunk TYPE record<document>;
DEFINE FIELD IF NOT EXISTS content ON temporal_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON temporal_chunk TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS occurred_at ON temporal_chunk TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS created_at ON temporal_chunk TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS source_system ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS author ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS channel ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS tags ON temporal_chunk TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS confidence ON temporal_chunk TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS metadata_ ON temporal_chunk FLEXIBLE TYPE option<object>;
DEFINE FIELD IF NOT EXISTS chunker_info ON temporal_chunk FLEXIBLE TYPE object DEFAULT {};

DEFINE INDEX IF NOT EXISTS idx_tc_namespace ON temporal_chunk FIELDS namespace;
DEFINE INDEX IF NOT EXISTS idx_tc_document ON temporal_chunk FIELDS document;
DEFINE INDEX IF NOT EXISTS idx_tc_occurred_at ON temporal_chunk FIELDS occurred_at;
DEFINE INDEX IF NOT EXISTS idx_tc_author ON temporal_chunk FIELDS author;
DEFINE INDEX IF NOT EXISTS idx_tc_channel ON temporal_chunk FIELDS channel;
"""

# Expensive indexes deferred until after bulk ingestion via ensure_search_indexes().
# HNSW is maintained incrementally on every INSERT and never used for search
# (brute-force cosine is used instead due to KNN being unreliable in embedded mode).
# BM25 also adds per-INSERT tokenization overhead.
_TEMPORAL_CHUNK_SEARCH_INDEXES = """
DEFINE INDEX IF NOT EXISTS idx_tc_embedding ON temporal_chunk FIELDS embedding HNSW DIMENSION 1536 DIST COSINE TYPE F32 EFC 128 M 24;
DEFINE INDEX IF NOT EXISTS idx_tc_content_ft ON temporal_chunk FIELDS content SEARCH ANALYZER khora_fulltext BM25;
"""

# Legacy ``TemporalFilter.additional`` range-op names → the canonical filter Op,
# so the deterministic-filter compiler can build a type-gated metadata compare.
_LEGACY_RANGE_OPS: dict[str, Op] = {
    "gt": Op.GT,
    "gte": Op.GTE,
    "lt": Op.LT,
    "lte": Op.LTE,
}

# Every legacy ``additional`` op routed through the deterministic-filter compiler.
# ``eq`` is included so the bare-equality path is validated through the same
# injection guard as the range ops — its $eq metadata emission is a plain
# ``(metadata_.<path> = $b)`` (no type-gate), preserving the equality semantics
# callers depend on while no longer interpolating an unvalidated user key.
_LEGACY_OPS: dict[str, Op] = {"eq": Op.EQ, **_LEGACY_RANGE_OPS}


class SurrealDBTemporalStore(TemporalVectorStore):
    """SurrealDB implementation of :class:`TemporalVectorStore`.

    Uses SurrealDB's HNSW vector index for ANN search, built-in BM25
    analyser for full-text search, and RRF fusion for hybrid queries.
    """

    def __init__(
        self,
        config: KhoraConfig,
        *,
        surrealdb_config: SurrealDBConfig | None = None,
        hnsw_ef_search: int = 40,
        connection: SurrealDBConnection | None = None,
    ) -> None:
        self._config = config
        self._hnsw_ef_search = hnsw_ef_search

        if connection is not None:
            # Reuse an existing shared connection (avoids isolated embedded views)
            self._conn = connection
            self._owns_connection = False
        else:
            self._owns_connection = True
            # Resolve SurrealDB connection parameters
            surreal_cfg = surrealdb_config or getattr(config.storage, "surrealdb", None)
            if surreal_cfg is None:
                raise ValueError(
                    "SurrealDB configuration is required. Set config.storage.surrealdb "
                    "or pass surrealdb_config explicitly."
                )

            # SurrealDBConfig.url and .password are SecretStr; unwrap exactly
            # here so the driver receives plaintext.
            from pydantic import SecretStr as _SecretStr

            surreal_url = surreal_cfg.url
            if isinstance(surreal_url, _SecretStr):
                surreal_url = surreal_url.get_secret_value()
            surreal_password = surreal_cfg.password
            if isinstance(surreal_password, _SecretStr):
                surreal_password = surreal_password.get_secret_value()
            self._conn = SurrealDBConnection(
                mode=surreal_cfg.mode,
                path=surreal_cfg.path,
                url=surreal_url,
                namespace=surreal_cfg.namespace,
                database=surreal_cfg.database,
                user=surreal_cfg.user,
                password=surreal_password,
                sync_data=surreal_cfg.sync_data,
            )
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to SurrealDB and initialise temporal_chunk schema."""
        if self._connected:
            return

        await self._conn.connect()

        # Create temporal_chunk table + indexes (idempotent)
        await self._conn.execute(_TEMPORAL_CHUNK_SCHEMA)

        self._connected = True
        logger.info("SurrealDBTemporalStore connected")

    async def disconnect(self) -> None:
        """Close the SurrealDB connection (only if we own it)."""
        if self._conn and self._connected:
            if self._owns_connection:
                await self._conn.disconnect()
            self._connected = False
            logger.info("SurrealDBTemporalStore disconnected")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a single temporal chunk."""
        chunk_id = chunk.id or uuid4()

        # Bind the RecordIDs via $params so they carry UUID-typed inner ids
        # \u2014 matching what ``create_chunks_batch`` writes via ``_rid()``.  The
        # literal ``temporal_chunk:\u27e8{uuid}\u27e9`` syntax produces *string*-typed
        # ids that don't compare equal to SDK-written rows (see issue #716
        # and the ``_build_filter_clauses`` docstring).
        sql = (
            "CREATE $chunk_rid SET "
            "namespace = $ns_rid, "
            "document = $doc_rid, "
            "content = $content, "
            "embedding = $embedding, "
            "occurred_at = $occurred_at, "
            "created_at = $created_at, "
            "source_system = $source_system, "
            "author = $author, "
            "channel = $channel, "
            "tags = $tags, "
            "confidence = $confidence, "
            "metadata_ = $metadata_, "
            "chunker_info = $chunker_info"
        )
        bindings = self._chunk_to_bindings(chunk, chunk_id)
        bindings["chunk_rid"] = _rid("temporal_chunk", chunk_id)
        bindings["ns_rid"] = _rid("memory_namespace", chunk.namespace_id)
        bindings["doc_rid"] = _rid("document", chunk.document_id)
        await self._conn.execute(sql, bindings)

        chunk.id = chunk_id
        return chunk

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Batch-insert temporal chunks."""
        if not chunks:
            return []

        records: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = chunk.id or uuid4()
            chunk.id = chunk_id
            records.append(
                {
                    "id": _rid("temporal_chunk", chunk_id),
                    "namespace": _rid("memory_namespace", chunk.namespace_id),
                    "document": _rid("document", chunk.document_id),
                    "content": chunk.content,
                    "embedding": (list(chunk.embedding) if chunk.embedding is not None else None),
                    "occurred_at": chunk.occurred_at,
                    "created_at": (chunk.created_at or datetime.now(UTC)),
                    "source_system": chunk.source_system,
                    "author": chunk.author,
                    "channel": chunk.channel,
                    "tags": _ensure_list(chunk.tags),
                    "confidence": chunk.confidence,
                    "metadata_": chunk.metadata or {},
                    "chunker_info": chunk.chunker_info or {},
                }
            )

        sql = "INSERT INTO temporal_chunk $records"
        await self._conn.execute(sql, {"records": records})
        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Fetch a temporal chunk by ID, scoped to namespace."""
        # Bind RecordIDs via $params (see _build_filter_clauses docstring re #716
        # for why literal ``:\u27e8{uuid}\u27e9`` interpolation does not match SDK-written rows).
        sql = "SELECT * FROM $chunk_rid WHERE namespace = $ns_rid"
        row = await self._conn.query_one(
            sql,
            {
                "chunk_rid": _rid("temporal_chunk", chunk_id),
                "ns_rid": _rid("memory_namespace", namespace_id),
            },
        )
        if not row:
            return None
        return self._row_to_chunk(row)

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a temporal chunk by ID, scoped to namespace."""
        bindings = {
            "chunk_rid": _rid("temporal_chunk", chunk_id),
            "ns_rid": _rid("memory_namespace", namespace_id),
        }
        # Count first to determine if the record exists
        check_sql = "SELECT count() AS cnt FROM temporal_chunk WHERE id = $chunk_rid AND namespace = $ns_rid GROUP ALL"
        count_row = await self._conn.query_one(check_sql, bindings)
        exists = int(count_row.get("cnt", 0)) > 0 if count_row else False

        if exists:
            del_sql = "DELETE FROM temporal_chunk WHERE id = $chunk_rid AND namespace = $ns_rid"
            await self._conn.execute(del_sql, bindings)

        return exists

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all temporal chunks for a document within a namespace."""
        bindings = {
            "doc_rid": _rid("document", document_id),
            "ns_rid": _rid("memory_namespace", namespace_id),
        }
        count_sql = (
            "SELECT count() AS cnt FROM temporal_chunk WHERE document = $doc_rid AND namespace = $ns_rid GROUP ALL"
        )
        count_row = await self._conn.query_one(count_sql, bindings)
        count = int(count_row.get("cnt", 0)) if count_row else 0

        if count > 0:
            del_sql = "DELETE FROM temporal_chunk WHERE document = $doc_rid AND namespace = $ns_rid"
            await self._conn.execute(del_sql, bindings)

        return count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @trace(
        "khora.surrealdb_temporal.search",
        include={"namespace_id", "limit"},
        result=lambda r: {"result_count": len(r)},
    )
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
        filter_ast: FilterNode | None = None,
    ) -> list[TemporalSearchResult]:
        """Search temporal chunks with optional hybrid (vector + BM25) ranking.

        ``filter_ast`` is the deterministic recall-filter AST. When provided it
        is compiled to a SurrealQL ``WHERE`` predicate and AND-ed alongside the
        legacy ``temporal_filter`` clauses.
        """
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
                filter_ast=filter_ast,
            )
            _span.set_attribute("result_count", len(results))
            return results

    async def _search_inner(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        temporal_filter: TemporalFilter | None = None,
        hybrid_alpha: float | None = None,
        query_text: str | None = None,
        filter_ast: FilterNode | None = None,
    ) -> list[TemporalSearchResult]:
        # Build shared WHERE clauses from temporal_filter
        filter_clauses, filter_bindings = self._build_filter_clauses(namespace_id, temporal_filter)

        # Deterministic recall-filter AST: compile to a SurrealQL predicate and
        # AND it into the shared clauses. ``field_mapping`` maps the ``metadata``
        # root to the physical ``metadata_`` column so a metadata path descends
        # natively; system keys map identity (the table stores them as bare
        # columns). The compiler runs in on_unsupported="raise" mode — the
        # skeleton engine has no post-filter path, so partial pushdown ("split")
        # is never consumed here.
        if filter_ast is not None:
            from khora.filter import CompileContext
            from khora.filter.compilers.surrealdb import compile_surrealdb

            compiled = compile_surrealdb(
                filter_ast,
                CompileContext(
                    backend_target="temporal_chunk",
                    field_mapping={"metadata": "metadata_"},
                    on_unsupported="raise",
                ),
            )
            filter_clauses.append(compiled.predicate)
            filter_bindings.update(compiled.params)

        # Determine search strategy
        pure_bm25 = hybrid_alpha is not None and hybrid_alpha == 0.0
        pure_vector = hybrid_alpha is None or hybrid_alpha == 1.0
        hybrid = not pure_bm25 and not pure_vector

        if pure_bm25:
            if not query_text:
                return []
            return await self._bm25_search(filter_clauses, filter_bindings, query_text, limit)

        # Vector search (fetch extra for fusion if hybrid)
        vector_limit = limit * 2 if hybrid else limit
        vector_results = await self._vector_search(
            filter_clauses,
            filter_bindings,
            query_embedding,
            vector_limit,
            min_similarity,
        )

        if hybrid and query_text:
            bm25_results = await self._bm25_search(filter_clauses, filter_bindings, query_text, limit * 2)
            return self._rrf_fusion(vector_results, bm25_results, hybrid_alpha, limit)

        # Pure vector — possibly with keyword fallback
        results = vector_results[:limit]
        if len(results) < limit and query_text:
            needed = limit - len(results)
            existing_ids = {str(r.chunk.id) for r in results}
            bm25_results = await self._bm25_search(
                filter_clauses, filter_bindings, query_text, needed + len(existing_ids)
            )
            for bm25_result in bm25_results:
                if str(bm25_result.chunk.id) not in existing_ids:
                    bm25_result.combined_score = (bm25_result.bm25_score or 0.0) * 0.8
                    results.append(bm25_result)
                    existing_ids.add(str(bm25_result.chunk.id))
                    if len(results) >= limit:
                        break

        return results

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        where_clauses: list[str],
        bindings: dict[str, Any],
        query_embedding: list[float],
        limit: int,
        min_similarity: float,
    ) -> list[TemporalSearchResult]:
        clauses = [*where_clauses, "embedding IS NOT NULL"]
        where_sql = " AND ".join(clauses)

        # Use brute-force cosine similarity + ORDER BY + LIMIT.
        # The SurrealDB <|K|> KNN operator does not accept parameterised
        # values and is unreliable in embedded mode (returns 0 results).
        sql = (
            "SELECT *, vector::similarity::cosine(embedding, $query_embedding) AS similarity "  # noqa: S608
            f"FROM temporal_chunk WHERE {where_sql} "
            f"ORDER BY similarity DESC LIMIT {int(limit)}"
        )
        params = {
            **bindings,
            "query_embedding": list(query_embedding),
        }

        rows = await self._conn.query(sql, params)

        results: list[TemporalSearchResult] = []
        for row in rows:
            sim = float(row.get("similarity", 0.0))
            if sim < min_similarity:
                continue
            results.append(
                TemporalSearchResult(
                    chunk=self._row_to_chunk(row),
                    similarity=sim,
                    bm25_score=None,
                    combined_score=sim,
                )
            )
        return results

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        filter_ast: FilterNode | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Public BM25 lookup over the SurrealDB temporal-chunk table.

        See :func:`temporal_chunk_to_chunk` for the ``TemporalChunk`` →
        ``Chunk`` adaptation. Falls back to ``[]`` on backends without
        a BM25 index configured (``optimize_storage()`` not yet run).

        ``filter_ast`` is the deterministic recall-filter AST. When provided it
        is compiled to a SurrealQL ``WHERE`` predicate and AND-ed into the BM25
        query alongside the namespace + temporal-bound clauses.
        """
        if not query_text or not query_text.strip():
            return []
        temporal_filter: TemporalFilter | None = None
        if created_after is not None or created_before is not None:
            temporal_filter = TemporalFilter(
                created_after=created_after,
                created_before=created_before,
            )
        filter_clauses, filter_bindings = self._build_filter_clauses(namespace_id, temporal_filter)

        # Deterministic recall-filter AST: compile to a SurrealQL predicate and
        # AND it into the shared clauses (mirrors the vector path in
        # ``_search_inner``). ``field_mapping`` maps the ``metadata`` root to the
        # physical ``metadata_`` column; ``on_unsupported="raise"`` because the
        # skeleton engine has no post-filter path.
        if filter_ast is not None:
            from khora.filter import CompileContext
            from khora.filter.compilers.surrealdb import compile_surrealdb

            compiled = compile_surrealdb(
                filter_ast,
                CompileContext(
                    backend_target="temporal_chunk",
                    field_mapping={"metadata": "metadata_"},
                    on_unsupported="raise",
                ),
            )
            filter_clauses.append(compiled.predicate)
            filter_bindings.update(compiled.params)

        results = await self._bm25_search(filter_clauses, filter_bindings, query_text, limit)
        return [(temporal_chunk_to_chunk(r.chunk), float(r.bm25_score or 0.0)) for r in results]

    async def _bm25_search(
        self,
        where_clauses: list[str],
        bindings: dict[str, Any],
        query_text: str,
        limit: int,
    ) -> list[TemporalSearchResult]:
        clauses = [*where_clauses, "content @1@ $query_text"]
        where_sql = " AND ".join(clauses)

        sql = (
            "SELECT *, search::score(1) AS rank "  # noqa: S608
            f"FROM temporal_chunk WHERE {where_sql} "
            "ORDER BY rank DESC LIMIT $bm25_limit"
        )
        params = {**bindings, "query_text": query_text, "bm25_limit": limit}

        try:
            rows = await self._conn.query(sql, params)
        except Exception as e:
            if "no suitable index" in str(e).lower():
                logger.warning("BM25 index not available — run optimize_storage() to create search indexes")
                return []
            raise

        return [
            TemporalSearchResult(
                chunk=self._row_to_chunk(row),
                similarity=0.0,
                bm25_score=float(row.get("rank", 0.0)),
                combined_score=float(row.get("rank", 0.0)),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        vector_results: list[TemporalSearchResult],
        bm25_results: list[TemporalSearchResult],
        alpha: float,
        limit: int,
        k: int = 60,
    ) -> list[TemporalSearchResult]:
        """Reciprocal Rank Fusion.

        score = alpha / (k + vector_rank) + (1 - alpha) / (k + bm25_rank)
        """
        vector_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(bm25_results)}
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            vr = vector_ranks.get(cid, len(vector_results) + 100)
            br = bm25_ranks.get(cid, len(bm25_results) + 100)
            rrf_scores[cid] = alpha * (1 / (k + vr)) + (1 - alpha) * (1 / (k + br))

        result_map: dict[str, TemporalSearchResult] = {}
        for r in vector_results:
            result_map[str(r.chunk.id)] = r
        for r in bm25_results:
            cid = str(r.chunk.id)
            if cid in result_map:
                result_map[cid].bm25_score = r.bm25_score
            else:
                result_map[cid] = r

        results: list[TemporalSearchResult] = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result = result_map[cid]
            result.combined_score = score
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Health / stats
    # ------------------------------------------------------------------

    async def ensure_search_indexes(self) -> None:
        """Create HNSW and BM25 indexes on temporal_chunk.

        Call after bulk ingestion to avoid per-INSERT index maintenance
        overhead.  Idempotent (uses IF NOT EXISTS).
        """
        if self._conn:
            logger.info("Creating temporal_chunk search indexes (HNSW + BM25)...")
            await self._conn.execute(_TEMPORAL_CHUNK_SEARCH_INDEXES)
            logger.info("Temporal chunk search indexes created")

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        if not self._connected:
            return {"status": "disconnected", "backend": "surrealdb"}
        try:
            await self._conn.execute("RETURN 1")
            return {"status": "healthy", "backend": "surrealdb"}
        except Exception as e:
            return {"status": "unhealthy", "backend": "surrealdb", "error": str(e)}

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Return the total number of temporal chunks in a namespace."""
        sql = "SELECT count() AS cnt FROM temporal_chunk WHERE namespace = $ns_rid GROUP ALL"
        row = await self._conn.query_one(sql, {"ns_rid": _rid("memory_namespace", namespace_id)})
        return int(row.get("cnt", 0)) if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_bindings(chunk: TemporalChunk, chunk_id: UUID) -> dict[str, Any]:
        return {
            "id": str(chunk_id),
            "ns": str(chunk.namespace_id),
            "doc": str(chunk.document_id),
            "content": chunk.content,
            "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
            "occurred_at": chunk.occurred_at,
            "created_at": (chunk.created_at or datetime.now(UTC)),
            "source_system": chunk.source_system,
            "author": chunk.author,
            "channel": chunk.channel,
            "tags": _ensure_list(chunk.tags),
            "confidence": chunk.confidence,
            "metadata_": chunk.metadata or {},
            "chunker_info": chunk.chunker_info or {},
        }

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> TemporalChunk:
        """Map a SurrealDB result row to a :class:`TemporalChunk`."""
        chunk_id = _parse_uuid(row.get("id", ""))
        namespace_id = _parse_uuid(row.get("namespace", ""))
        document_id = _parse_uuid(row.get("document", ""))

        raw_embedding = row.get("embedding")
        if raw_embedding is not None:
            embedding: list[float] | None = [float(v) for v in raw_embedding]
        else:
            embedding = None

        tags_raw = row.get("tags")
        tags: list[str] = list(tags_raw) if tags_raw else []

        meta = row.get("metadata_") or {}
        if not isinstance(meta, dict):
            meta = {}

        chunker_info = row.get("chunker_info") or {}
        if not isinstance(chunker_info, dict):
            chunker_info = {}

        return TemporalChunk(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content=row.get("content", ""),
            embedding=embedding,
            occurred_at=_parse_dt(row.get("occurred_at")),
            created_at=_parse_dt(row.get("created_at")),
            source_system=row.get("source_system"),
            author=row.get("author"),
            channel=row.get("channel"),
            tags=tags,
            confidence=float(row.get("confidence", 1.0)),
            metadata=meta,
            chunker_info=chunker_info,
        )

    @staticmethod
    def _build_filter_clauses(
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Build WHERE clauses and bindings from a TemporalFilter.

        The namespace_id may be either the row-level ``id`` or the stable
        ``namespace_id`` — we match both so the filter works regardless of
        which ID the caller passes (recall resolves to row-level, but chunks
        store the stable namespace_id as record reference).

        The namespace RecordID is bound via parameter (``$ns_rid``)
        rather than interpolated as a SurrealQL literal.  The SDK's
        ``RecordID(table, uuid)`` constructor produces a *UUID-typed*
        record id (``memory_namespace:u'<uuid>'`` on the wire), whereas
        the literal ``memory_namespace:\u27e8{uuid-str}\u27e9`` is parsed
        as a *string-typed* record id.  SurrealDB's equality is
        type-strict, so the literal form never matches rows the writer
        inserted via the SDK \u2014 see issue #716.
        """
        clauses = [
            "(namespace = $ns_rid OR namespace.namespace_id = $ns_str)",
        ]
        bindings: dict[str, Any] = {
            "ns_rid": _rid("memory_namespace", namespace_id),
            "ns_str": str(namespace_id),
        }

        if not temporal_filter:
            return clauses, bindings

        f = temporal_filter

        if f.occurred_after is not None:
            clauses.append("occurred_at >= $occurred_after")
            bindings["occurred_after"] = f.occurred_after
        if f.occurred_before is not None:
            clauses.append("occurred_at < $occurred_before")
            bindings["occurred_before"] = f.occurred_before
        if f.created_after is not None:
            clauses.append("created_at >= $created_after")
            bindings["created_after"] = f.created_after
        if f.created_before is not None:
            clauses.append("created_at < $created_before")
            bindings["created_before"] = f.created_before

        if f.source_system:
            clauses.append("source_system = $source_system")
            bindings["source_system"] = f.source_system
        if f.author:
            clauses.append("author = $author")
            bindings["author"] = f.author
        if f.channel:
            clauses.append("channel = $channel")
            bindings["channel"] = f.channel

        if f.tags:
            # All tags must be present — use CONTAINSALL
            clauses.append("tags CONTAINSALL $filter_tags")
            bindings["filter_tags"] = f.tags

        # Additional structured filters. EVERY user key is routed through the
        # deterministic-filter SurrealDB compiler so its injection guard validates
        # each path segment (``TemporalFilter.additional`` keys are NOT
        # char-restricted upstream — interpolating one verbatim is an injection
        # surface). Range comparisons additionally get a ``type::is::*`` gate
        # (numeric ordering, not lexicographic; wrong-typed values gated out);
        # ``eq`` emits a plain ``(metadata_.<path> = $b)``, preserving the
        # equality semantics callers depend on. This replaces the old
        # ``_sanitize_field_name`` string mangling.
        for i, (key, value) in enumerate(f.additional.items()):
            if isinstance(value, dict):
                for op, val in value.items():
                    if op not in _LEGACY_OPS:
                        continue
                    predicate, binds = SurrealDBTemporalStore._legacy_metadata_predicate(key, op, val, f"af_{i}_{op}")
                    clauses.append(predicate)
                    bindings.update(binds)
            else:
                predicate, binds = SurrealDBTemporalStore._legacy_metadata_predicate(key, "eq", value, f"af_{i}_eq")
                clauses.append(predicate)
                bindings.update(binds)

        return clauses, bindings

    @staticmethod
    def _legacy_metadata_predicate(key: str, op: str, val: Any, param_namespace: str) -> tuple[str, dict[str, Any]]:
        """Compile a legacy ``additional`` metadata predicate via the recall-filter compiler.

        Reuses the deterministic-filter SurrealDB compiler so the (possibly
        dotted) ``key`` is validated through the compiler's injection guard before
        any interpolation — an unsafe segment raises
        :class:`~khora.filter.context.CompileError`. ``op`` is one of
        ``eq``/``gt``/``gte``/``lt``/``lte`` (an :data:`_LEGACY_OPS` key); a range
        op also emits a ``type::is::*`` gate so a numeric value orders numerically
        and a wrong-typed value is excluded, while ``eq`` emits a plain
        ``(metadata_.<path> = $b)`` equality. ``key`` is split on ``.`` into a
        nested path. Returns the SurrealQL fragment + its binds, named under
        ``param_namespace`` so concurrent legacy keys never collide.
        """
        from khora.filter.ast import FilterClause
        from khora.filter.compilers.surrealdb import _Builder
        from khora.filter.context import CompileContext

        builder = _Builder(
            ctx=CompileContext(
                backend_target="temporal_chunk",
                param_namespace=param_namespace,
                field_mapping={"metadata": "metadata_"},
            ),
            consumed=set(),
        )
        clause = FilterClause(path=("metadata", *key.split(".")), op=_LEGACY_OPS[op], operand=val)
        predicate = builder.compile_clause(clause)
        return predicate, builder.params


__all__ = ["SurrealDBTemporalStore"]


# Register the deterministic recall-filter compiler for this engine/target at
# import time (idempotent — same function object). ``surrealdb.py`` is imported
# lazily by ``create_temporal_store("surrealdb", ...)``, so registration happens
# exactly when the SurrealDB backend is first constructed — no eager cost, and no
# registration when the backend is unused. Mirrors ``pgvector.py``'s
# ``skeleton.pgvector`` registration.
from khora.filter import CompilerRegistry  # noqa: E402
from khora.filter.compilers.surrealdb import compile_surrealdb  # noqa: E402

CompilerRegistry.register("skeleton.surrealdb", "temporal_chunk", compile_surrealdb)
