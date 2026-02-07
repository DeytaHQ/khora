"""PostgreSQL+pgvector backend for the Skeleton engine.

This backend provides:
- BRIN-indexed temporal queries (99% space savings vs btree)
- Vector similarity search via pgvector HNSW index
- Full-text search via PostgreSQL tsvector/GIN
- Hybrid search via separate queries + RRF fusion
- Structured field filtering via SQL WHERE clauses
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
    and_,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
)

if TYPE_CHECKING:
    from khora.config import KhoraConfig

# Table definition for khora_chunks (separate from existing chunks table)
metadata = MetaData()

khora_chunks_table = Table(
    "khora_chunks",
    metadata,
    Column("id", PG_UUID(as_uuid=False), primary_key=True),
    Column("namespace_id", PG_UUID(as_uuid=False), nullable=False, index=True),
    Column("document_id", PG_UUID(as_uuid=False), nullable=False, index=True),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(1536), nullable=True),
    # Temporal fields
    Column("occurred_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), default=func.now()),
    # Metadata for filtering
    Column("source_system", String(64), nullable=True, index=True),
    Column("author", String(255), nullable=True, index=True),
    Column("channel", String(255), nullable=True, index=True),
    Column("tags", ARRAY(String), default=[]),
    Column("confidence", Float, default=1.0),
    Column("metadata", JSONB, default=dict),
    # Full-text search
    Column("content_tsv", TSVECTOR, nullable=True),
    # Indexes are defined below
)


class PgVectorTemporalStore(TemporalVectorStore):
    """PostgreSQL+pgvector implementation of TemporalVectorStore.

    Uses:
    - pgvector HNSW index for vector similarity search
    - BRIN index on occurred_at for time-series optimization
    - GIN index on content_tsv for full-text search
    - Standard B-tree indexes on filter fields
    """

    def __init__(self, config: KhoraConfig):
        """Initialize the backend.

        Args:
            config: Khora configuration
        """
        self._config = config
        self._engine = None
        self._connected = False
        self._embedding_dimension = config.llm.embedding_dimension or 1536

    async def connect(self) -> None:
        """Connect to PostgreSQL and ensure schema exists."""
        if self._connected:
            return

        database_url = self._config.get_postgresql_url()
        if not database_url:
            raise ValueError("PostgreSQL URL not configured")

        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

        self._engine = create_async_engine(database_url, pool_size=5, max_overflow=10)

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

            # Create BRIN index on occurred_at
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS ix_khora_chunks_occurred_brin
                ON khora_chunks USING BRIN (occurred_at)
                """
                )
            )

            # Create HNSW index on embedding
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS ix_khora_chunks_embedding_hnsw
                ON khora_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
                )
            )

            # Create GIN index on content_tsv
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS ix_khora_chunks_content_tsv
                ON khora_chunks USING GIN (content_tsv)
                """
                )
            )

            # Create trigger for auto-updating content_tsv
            # Note: Each statement must be executed separately (asyncpg limitation)
            await conn.execute(
                text(
                    """
                CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector('english', NEW.content);
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
                """
                )
            )
            await conn.execute(text("DROP TRIGGER IF EXISTS khora_chunks_content_tsv_update ON khora_chunks"))
            await conn.execute(
                text(
                    """
                CREATE TRIGGER khora_chunks_content_tsv_update
                BEFORE INSERT OR UPDATE ON khora_chunks
                FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()
                """
                )
            )

        self._connected = True
        logger.info("PgVectorTemporalStore connected")

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
        self._connected = False
        logger.info("PgVectorTemporalStore disconnected")

    def _get_session(self) -> AsyncSession:
        """Get a new async session."""
        if not self._engine:
            raise RuntimeError("Not connected")
        from sqlalchemy.ext.asyncio import AsyncSession

        return AsyncSession(self._engine, expire_on_commit=False)

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a chunk with temporal metadata."""
        chunk_id = chunk.id or uuid4()

        async with self._get_session() as session:
            stmt = khora_chunks_table.insert().values(
                id=str(chunk_id),
                namespace_id=str(chunk.namespace_id),
                document_id=str(chunk.document_id),
                content=chunk.content,
                embedding=chunk.embedding,
                occurred_at=chunk.occurred_at,
                created_at=chunk.created_at or datetime.now(UTC),
                source_system=chunk.source_system,
                author=chunk.author,
                channel=chunk.channel,
                tags=chunk.tags or [],
                confidence=chunk.confidence,
                metadata=chunk.metadata or {},
            )
            await session.execute(stmt)
            await session.commit()

        chunk.id = chunk_id
        return chunk

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Store multiple chunks in batch."""
        if not chunks:
            return []

        values = []
        for chunk in chunks:
            chunk_id = chunk.id or uuid4()
            chunk.id = chunk_id
            values.append(
                {
                    "id": str(chunk_id),
                    "namespace_id": str(chunk.namespace_id),
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "occurred_at": chunk.occurred_at,
                    "created_at": chunk.created_at or datetime.now(UTC),
                    "source_system": chunk.source_system,
                    "author": chunk.author,
                    "channel": chunk.channel,
                    "tags": chunk.tags or [],
                    "confidence": chunk.confidence,
                    "metadata": chunk.metadata or {},
                }
            )

        async with self._get_session() as session:
            await session.execute(khora_chunks_table.insert(), values)
            await session.commit()

        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Get a chunk by ID."""
        async with self._get_session() as session:
            stmt = select(khora_chunks_table).where(
                khora_chunks_table.c.id == str(chunk_id),
                khora_chunks_table.c.namespace_id == str(namespace_id),
            )
            result = await session.execute(stmt)
            row = result.fetchone()

        if not row:
            return None

        return self._row_to_chunk(row)

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a chunk by ID."""
        async with self._get_session() as session:
            stmt = khora_chunks_table.delete().where(
                khora_chunks_table.c.id == str(chunk_id),
                khora_chunks_table.c.namespace_id == str(namespace_id),
            )
            result = await session.execute(stmt)
            await session.commit()

        return result.rowcount > 0  # type: ignore[unresolved-attribute]

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all chunks for a document."""
        async with self._get_session() as session:
            stmt = khora_chunks_table.delete().where(
                khora_chunks_table.c.document_id == str(document_id),
                khora_chunks_table.c.namespace_id == str(namespace_id),
            )
            result = await session.execute(stmt)
            await session.commit()

        return result.rowcount  # type: ignore[unresolved-attribute]

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
        """Search for similar chunks with temporal filtering.

        Uses:
        - Vector similarity via pgvector cosine distance
        - BM25-style scoring via ts_rank for hybrid search
        - RRF fusion to combine scores

        QUALITY FIX: When vector search returns insufficient results, automatically
        falls back to keyword search to improve recall on non-core chunks.
        """
        async with self._get_session() as session:
            # Build base conditions
            conditions = [khora_chunks_table.c.namespace_id == str(namespace_id)]

            # Add temporal filters
            if temporal_filter:
                conditions.extend(self._build_filter_conditions(temporal_filter))

            # Vector search
            vector_results = await self._vector_search(
                session,
                query_embedding,
                conditions,
                limit * 2 if hybrid_alpha else limit,  # Fetch more for fusion
                min_similarity,
            )

            # If hybrid search is requested, also do BM25 search
            if hybrid_alpha is not None and query_text:
                bm25_results = await self._bm25_search(
                    session,
                    query_text,
                    conditions,
                    limit * 2,
                )

                # Fuse results using RRF
                results = self._rrf_fusion(vector_results, bm25_results, hybrid_alpha, limit)
            else:
                results = vector_results[:limit]

                # QUALITY FIX: Keyword fallback when vector search returns
                # insufficient results. This improves recall for non-core chunks
                # that may not have strong vector similarity but contain
                # relevant keywords.
                if len(results) < limit and query_text:
                    needed = limit - len(results)
                    existing_ids = {str(r.chunk.id) for r in results}

                    bm25_results = await self._bm25_search(
                        session,
                        query_text,
                        conditions,
                        needed + len(existing_ids),  # Fetch extra to account for overlap
                    )

                    # Add BM25 results that aren't already in vector results
                    for bm25_result in bm25_results:
                        if str(bm25_result.chunk.id) not in existing_ids:
                            # Discount BM25-only results slightly to prefer vector matches
                            bm25_result.combined_score = (bm25_result.bm25_score or 0.0) * 0.8
                            results.append(bm25_result)
                            existing_ids.add(str(bm25_result.chunk.id))
                            if len(results) >= limit:
                                break

        return results

    async def _vector_search(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        conditions: list,
        limit: int,
        min_similarity: float,
    ) -> list[TemporalSearchResult]:
        """Perform vector similarity search."""
        # Calculate cosine similarity: 1 - cosine_distance
        similarity = (1 - khora_chunks_table.c.embedding.cosine_distance(query_embedding)).label("similarity")

        stmt = (
            select(khora_chunks_table, similarity)
            .where(
                and_(
                    *conditions,
                    khora_chunks_table.c.embedding.isnot(None),
                )
            )
            .order_by(similarity.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.fetchall()

        results = []
        for row in rows:
            sim = row.similarity
            if sim >= min_similarity:
                chunk = self._row_to_chunk(row)
                results.append(
                    TemporalSearchResult(
                        chunk=chunk,
                        similarity=sim,
                        bm25_score=None,
                        combined_score=sim,
                    )
                )

        return results

    async def _bm25_search(
        self,
        session: AsyncSession,
        query_text: str,
        conditions: list,
        limit: int,
    ) -> list[TemporalSearchResult]:
        """Perform BM25-style full-text search using PostgreSQL ts_rank."""
        # Create tsquery from query text
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank(khora_chunks_table.c.content_tsv, tsquery).label("bm25_score")

        stmt = (
            select(khora_chunks_table, rank)
            .where(
                and_(
                    *conditions,
                    khora_chunks_table.c.content_tsv.isnot(None),
                    khora_chunks_table.c.content_tsv.op("@@")(tsquery),
                )
            )
            .order_by(rank.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.fetchall()

        results = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            results.append(
                TemporalSearchResult(
                    chunk=chunk,
                    similarity=0.0,  # Will be filled if in vector results
                    bm25_score=row.bm25_score,
                    combined_score=row.bm25_score,
                )
            )

        return results

    def _rrf_fusion(
        self,
        vector_results: list[TemporalSearchResult],
        bm25_results: list[TemporalSearchResult],
        alpha: float,
        limit: int,
        k: int = 60,  # RRF constant
    ) -> list[TemporalSearchResult]:
        """Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

        Score = alpha * (1 / (k + vector_rank)) + (1 - alpha) * (1 / (k + bm25_rank))
        """
        # Build maps of chunk_id -> rank
        vector_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {str(r.chunk.id): i + 1 for i, r in enumerate(bm25_results)}

        # Collect all chunk IDs
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        for chunk_id in all_ids:
            vector_rank = vector_ranks.get(chunk_id, len(vector_results) + 100)
            bm25_rank = bm25_ranks.get(chunk_id, len(bm25_results) + 100)

            rrf_score = alpha * (1 / (k + vector_rank)) + (1 - alpha) * (1 / (k + bm25_rank))
            rrf_scores[chunk_id] = rrf_score

        # Build result map
        result_map: dict[str, TemporalSearchResult] = {}
        for r in vector_results:
            chunk_id = str(r.chunk.id)
            result_map[chunk_id] = r

        for r in bm25_results:
            chunk_id = str(r.chunk.id)
            if chunk_id in result_map:
                # Merge BM25 score
                result_map[chunk_id].bm25_score = r.bm25_score
            else:
                result_map[chunk_id] = r

        # Update combined scores and sort
        results = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result = result_map[chunk_id]
            result.combined_score = rrf_score
            results.append(result)

        return results

    def _build_filter_conditions(self, f: TemporalFilter) -> list:
        """Build SQL conditions from TemporalFilter."""
        conditions = []

        if f.occurred_after:
            conditions.append(khora_chunks_table.c.occurred_at >= f.occurred_after)
        if f.occurred_before:
            conditions.append(khora_chunks_table.c.occurred_at < f.occurred_before)
        if f.created_after:
            conditions.append(khora_chunks_table.c.created_at >= f.created_after)
        if f.created_before:
            conditions.append(khora_chunks_table.c.created_at < f.created_before)

        if f.source_system:
            conditions.append(khora_chunks_table.c.source_system == f.source_system)
        if f.author:
            conditions.append(khora_chunks_table.c.author == f.author)
        if f.channel:
            conditions.append(khora_chunks_table.c.channel == f.channel)

        if f.tags:
            # All tags must be present
            conditions.append(khora_chunks_table.c.tags.contains(f.tags))

        # Handle additional filters
        for key, value in f.additional.items():
            if isinstance(value, dict):
                # Operator-style filter
                for op, val in value.items():
                    if op == "eq":
                        conditions.append(khora_chunks_table.c.metadata[key].astext == str(val))
                    elif op == "gte":
                        conditions.append(khora_chunks_table.c.metadata[key].astext >= str(val))
                    elif op == "lte":
                        conditions.append(khora_chunks_table.c.metadata[key].astext <= str(val))
                    elif op == "gt":
                        conditions.append(khora_chunks_table.c.metadata[key].astext > str(val))
                    elif op == "lt":
                        conditions.append(khora_chunks_table.c.metadata[key].astext < str(val))
            else:
                # Simple equality
                conditions.append(khora_chunks_table.c.metadata[key].astext == str(value))

        return conditions

    def _row_to_chunk(self, row) -> TemporalChunk:
        """Convert a database row to a TemporalChunk."""
        return TemporalChunk(
            id=UUID(row.id),
            namespace_id=UUID(row.namespace_id),
            document_id=UUID(row.document_id),
            content=row.content,
            embedding=list(row.embedding) if row.embedding else None,
            occurred_at=row.occurred_at,
            created_at=row.created_at,
            source_system=row.source_system,
            author=row.author,
            channel=row.channel,
            tags=list(row.tags) if row.tags else [],
            confidence=row.confidence or 1.0,
            metadata=row.metadata or {},
        )

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        if not self._connected or not self._engine:
            return {"status": "disconnected", "backend": "pgvector"}

        try:
            async with self._get_session() as session:
                await session.execute(text("SELECT 1"))
            return {"status": "healthy", "backend": "pgvector"}
        except Exception as e:
            return {"status": "unhealthy", "backend": "pgvector", "error": str(e)}


__all__ = ["PgVectorTemporalStore"]
