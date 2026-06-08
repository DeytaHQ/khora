"""pgvector backend for vector embeddings storage.

Handles storage and retrieval of embeddings for semantic search
using pgvector extension in PostgreSQL.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sa
from loguru import logger
from sqlalchemy import delete, func, literal_column, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from khora.core.diagnostics import Degradation
from khora.core.models import Chunk
from khora.db.models import (
    Base,
    ChronicleEventModel,
    ChunkModel,
    EntityModel,
    MemoryFactModel,
    RelationshipModel,
)
from khora.db.schema import sync_enum_values
from khora.storage.backends.mixins import AsyncSessionMixin
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.filter.ast import FilterNode

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


_DEADLOCK_MAX_RETRIES = 3

# Cap on retained source-id provenance per entity. Mirrors Neo4j's
# ``entity_source_document_ids_max`` default (see neo4j.py); the two backends
# must agree so forget()'s survivor-strip behaves identically across a
# PG+Neo4j stack.
_SOURCE_IDS_CAP = 100


def _accumulate_source_ids_sql(column: str) -> sa.TextClause:
    """Build the ON CONFLICT set-expression that accumulates source ids.

    Mirrors Neo4j's ``ON MATCH SET e.source_*_ids = (existing + incoming)[-N..]``
    semantics (neo4j.py:1714-1715) but adds a dedup pass so re-extracting the
    same document many times does NOT evict prior *distinct* provenance ids.

    The existing row is referenced by the table name ``entities``; the proposed
    row by ``excluded``. The two arrays are concatenated (existing first, so
    incoming ids carry the higher ordinals), grouped by element to dedup while
    keeping each id's most-recent position, capped to the
    ``_SOURCE_IDS_CAP`` most-recent DISTINCT ids, and re-aggregated newest-last
    so the tail holds the freshest provenance (matching Neo4j's tail-cap order).

    ``column`` is ``"source_document_ids"`` or ``"source_chunk_ids"``. The cap is
    an int constant inlined into the SQL (no bind param — a bind param could
    collide with the executemany VALUES binding); ``column`` is never
    caller-supplied so there is no injection surface.
    """
    # `column` is one of two hard-coded literals and the cap is an int constant,
    # so no user input reaches this SQL (bind params are deliberately avoided so
    # they can't collide with the executemany VALUES binding) — hence noqa: S608.
    return sa.text(
        f"(SELECT COALESCE(array_agg(elem ORDER BY last_ord), ARRAY[]::uuid[]) "  # noqa: S608
        f"FROM (SELECT elem, MAX(ord) AS last_ord "
        f"FROM unnest(COALESCE(entities.{column}, ARRAY[]::uuid[]) || excluded.{column}) "
        f"WITH ORDINALITY AS u(elem, ord) "
        f"GROUP BY elem ORDER BY MAX(ord) DESC LIMIT {_SOURCE_IDS_CAP}) d)"
    )


def _namespace_lock_keys(namespace_id: UUID) -> tuple[int, int]:
    """Derive a stable ``(int4, int4)`` advisory-lock pair from a namespace UUID.

    Postgres's two-int form ``pg_advisory_xact_lock(int4, int4)`` treats the
    pair as a single 64-bit lock id internally, so using two halves of the
    UUID gives ~2^64 lock-id entropy — birthday-safe at billions of
    namespaces. The previous implementation folded all 128 bits down to a
    single 32-bit key, which birthday-collided at ~65K namespaces (#738).
    """
    hi64, lo64 = struct.unpack(">QQ", namespace_id.bytes)
    # XOR each 64-bit half down to 32 bits so all 128 bits of the UUID feed
    # both slots. For random UUIDs (v4 / v5) every output bit depends on
    # multiple input bits.
    fold_hi = (hi64 ^ (hi64 >> 32)) & 0xFFFFFFFF
    fold_lo = (lo64 ^ (lo64 >> 32)) & 0xFFFFFFFF
    # pg int4 is signed [-2^31, 2^31 - 1]; convert via signed-overflow.
    return (
        struct.unpack(">i", struct.pack(">I", fold_hi))[0],
        struct.unpack(">i", struct.pack(">I", fold_lo))[0],
    )


async def _retry_on_deadlock(coro_fn, *args, **kwargs):
    """Retry an async operation on deadlock with exponential backoff."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(_DEADLOCK_MAX_RETRIES),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.4),
        retry=retry_if_exception(lambda e: "deadlock" in str(e).lower()),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying after deadlock (attempt {}): {!s}",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else "unknown",
        ),
        reraise=True,
    ):
        with attempt:
            return await coro_fn(*args, **kwargs)


class PgVectorBackend(AsyncSessionMixin):
    """pgvector backend for vector embeddings.

    Handles all vector operations including chunk storage,
    similarity search, and entity embeddings.
    """

    def __init__(
        self,
        database_url: str,
        *,
        embedding_dimension: int = 1536,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = False,
        hnsw_ef_search: int = 100,
        use_halfvec: bool = True,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the pgvector backend.

        Args:
            database_url: PostgreSQL connection URL (with pgvector extension)
            embedding_dimension: Dimension of embedding vectors
            echo: Enable SQL echo logging
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_pre_ping: Enable pool pre-ping to detect stale connections
            hnsw_ef_search: HNSW ef_search for query-time accuracy
            use_halfvec: Use halfvec (float16) for similarity search.
                Requires pgvector extension >= 0.7.0.
            engine: Optional shared engine (skip dispose on disconnect)
        """
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

        self._database_url = database_url
        self._embedding_dimension = embedding_dimension
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_pre_ping = pool_pre_ping
        self._hnsw_ef_search = hnsw_ef_search
        self._use_halfvec = use_halfvec
        self._halfvec_available: bool | None = None  # Detected at connect time
        self._engine: AsyncEngine | None = engine
        self._engine_shared: bool = engine is not None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Establish connection to the database."""
        if self._session_factory is not None:
            return

        logger.info("Connecting to pgvector...")
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=self._pool_pre_ping,
            )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Ensure pgvector extension is enabled
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Detect halfvec support (pgvector >= 0.7.0) and verify HNSW indexes
        if self._use_halfvec:
            halfvec_supported = await self._detect_halfvec_support()
            if not halfvec_supported:
                self._halfvec_available = False
                logger.warning("halfvec requested but pgvector < 0.7.0 — falling back to full-precision vectors")
            elif not await self._check_halfvec_indexes():
                self._halfvec_available = False
                # Distinguish "fresh DB, migrations not run yet" (benign — common path
                # for ephemeral per-run databases) from "tables exist but indexes are
                # missing" (real misconfiguration).
                if not await self._halfvec_target_tables_exist():
                    logger.info(
                        "halfvec HNSW indexes not yet created (fresh DB) — using full-precision "
                        "vectors until migrations run"
                    )
                else:
                    logger.warning(
                        "halfvec HNSW indexes not found — falling back to full-precision vectors. "
                        "Run migrations to create them."
                    )
            else:
                self._halfvec_available = True
                logger.info("halfvec (float16) support detected — enabled for similarity search")

        logger.info("Connected to pgvector")

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            logger.info("Disconnecting from pgvector...")
            if not self._engine_shared:
                await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Disconnected from pgvector")

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        if self._engine is None or self._session_factory is None:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"pgvector health check failed: {e}")
            return False

    @property
    def halfvec_enabled(self) -> bool:
        """Whether halfvec is both requested and available."""
        return self._use_halfvec and self._halfvec_available is True

    async def _detect_halfvec_support(self) -> bool:
        """Check if the pgvector extension supports halfvec (>= 0.7.0)."""
        try:
            async with self._get_session() as session:
                result = await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
                row = result.first()
                if row is None:
                    return False
                version_str = row[0]
                parts = version_str.split(".")
                major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                return (major, minor) >= (0, 7)
        except Exception as e:
            logger.debug(f"Failed to detect pgvector version: {e}")
            return False

    async def _halfvec_target_tables_exist(self) -> bool:
        """Return True iff both ``chunks`` and ``entities`` tables exist.

        Used to distinguish "fresh DB before migrations" (benign — vectors fall
        back to full precision until tables are created) from "tables exist but
        indexes were never built" (real misconfiguration worth a warning).
        """
        try:
            async with self._get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT count(*) FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE c.relkind = 'r' "
                        "AND n.nspname = 'public' "
                        "AND c.relname IN ('chunks', 'entities')"
                    )
                )
                count = result.scalar_one_or_none() or 0
                return count == 2
        except Exception as e:
            logger.debug(f"Failed to check halfvec target tables: {e}")
            return False

    async def _check_halfvec_indexes(self) -> bool:
        """Check if both halfvec HNSW indexes exist and are valid in the database."""
        required = {"ix_chunks_embedding_halfvec_hnsw", "ix_entities_embedding_halfvec_hnsw"}
        try:
            async with self._get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT c.relname, i.indisvalid FROM pg_class c "
                        "JOIN pg_index i ON i.indexrelid = c.oid "
                        "WHERE c.relname IN ('ix_chunks_embedding_halfvec_hnsw', "
                        "'ix_entities_embedding_halfvec_hnsw')"
                    )
                )
                rows = result.all()
                valid = set()
                for name, is_valid in rows:
                    if is_valid:
                        valid.add(name)
                    else:
                        logger.warning(
                            f"halfvec index {name} exists but is invalid "
                            "(interrupted build?) — re-run migrations to rebuild it"
                        )
                return required.issubset(valid)
        except Exception as e:
            logger.warning(f"Failed to check halfvec indexes: {e}")
            return False

    async def create_tables(self) -> None:
        """Create all database tables.

        .. deprecated::
            Use ``run_migrations()`` instead. ``create_tables()`` bypasses
            Alembic and masks missing migrations — tests pass locally but
            production breaks. Will be removed in a future release.
        """
        import warnings

        warnings.warn(
            "create_tables() is deprecated. Use khora.db.run_migrations() instead. "
            "create_tables() bypasses Alembic and masks missing migrations.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._engine is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await sync_enum_values(self._engine)

    # =========================================================================
    # Chunk operations
    # =========================================================================

    async def create_chunk(self, chunk: Chunk, *, session: AsyncSession | None = None) -> Chunk:
        """Create a new chunk with its embedding."""
        if session is not None:
            return await self._create_chunk_with(session, chunk)
        async with self._get_session() as own_session:
            return await self._create_chunk_with(own_session, chunk, commit=True)

    async def _create_chunk_with(self, session: AsyncSession, chunk: Chunk, *, commit: bool = False) -> Chunk:
        model = ChunkModel(
            id=chunk.id,
            namespace_id=chunk.namespace_id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            token_count=chunk.token_count,
            metadata_=chunk.metadata,
            chunker_info=chunk.chunker_info,
            embedding=chunk.embedding,
            embedding_model=chunk.embedding_model,
            created_at=chunk.created_at,
            source_timestamp=getattr(chunk, "source_timestamp", None),
            last_accessed_at=getattr(chunk, "last_accessed_at", None),
            session_id=getattr(chunk, "session_id", None),
        )
        session.add(model)
        if commit:
            await session.commit()
        else:
            await session.flush()
        await session.refresh(model)
        return self._chunk_model_to_domain(model)

    async def create_chunks_batch(self, chunks: list[Chunk], *, session: AsyncSession | None = None) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        if not chunks:
            return []

        if session is not None:
            return await self._create_chunks_batch_with(session, chunks)
        async with self._get_session() as own_session:
            return await self._create_chunks_batch_with(own_session, chunks, commit=True)

    async def _create_chunks_batch_with(
        self, session: AsyncSession, chunks: list[Chunk], *, commit: bool = False
    ) -> list[Chunk]:
        models = [
            ChunkModel(
                id=chunk.id,
                namespace_id=chunk.namespace_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=chunk.token_count,
                metadata_=chunk.metadata,
                chunker_info=chunk.chunker_info,
                embedding=chunk.embedding,
                embedding_model=chunk.embedding_model,
                created_at=chunk.created_at,
                source_timestamp=getattr(chunk, "source_timestamp", None),
                last_accessed_at=getattr(chunk, "last_accessed_at", None),
                session_id=getattr(chunk, "session_id", None),
            )
            for chunk in chunks
        ]
        session.add_all(models)
        if commit:
            await session.commit()
        return chunks

    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        """Get a chunk by ID, filtered to the caller's ``namespace_id``."""
        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel).where(
                    ChunkModel.id == chunk_id,
                    ChunkModel.namespace_id == namespace_id,
                )
            )
            model = result.scalar_one_or_none()
            return self._chunk_model_to_domain(model) if model else None

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        """Get multiple chunks by ID in a single query, scoped to ``namespace_id``.

        Args:
            chunk_ids: List of chunk IDs to fetch.
            namespace_id: Caller's namespace; cross-namespace ids are
                silently dropped from the result to prevent IDOR.

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing
            chunks within ``namespace_id``).
        """
        if not chunk_ids:
            return {}

        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel).where(
                    ChunkModel.id.in_(chunk_ids),
                    ChunkModel.namespace_id == namespace_id,
                )
            )
            models = result.scalars().all()
            return {m.id: self._chunk_model_to_domain(m) for m in models}

    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        """Get all chunks for a document, filtered to the caller's ``namespace_id``."""
        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel)
                .where(
                    ChunkModel.document_id == document_id,
                    ChunkModel.namespace_id == namespace_id,
                )
                .order_by(ChunkModel.chunk_index)
            )
            return [self._chunk_model_to_domain(m) for m in result.scalars().all()]

    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        *,
        namespace_id: UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete all chunks for a document, scoped to ``namespace_id``.

        When *session* is provided the caller owns the transaction —
        no commit is issued.  When ``None``, a private session is used
        and committed automatically. The ``namespace_id`` filter prevents
        cross-tenant deletion by document id (IDOR family).
        """
        if session is not None:
            result = await session.execute(
                delete(ChunkModel).where(
                    ChunkModel.document_id == document_id,
                    ChunkModel.namespace_id == namespace_id,
                )
            )
            return result.rowcount  # type: ignore[unresolved-attribute]
        async with self._get_session() as own_session:
            result = await own_session.execute(
                delete(ChunkModel).where(
                    ChunkModel.document_id == document_id,
                    ChunkModel.namespace_id == namespace_id,
                )
            )
            await own_session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    async def update_last_accessed(
        self,
        namespace_id: UUID,
        chunk_ids: list[UUID],
        ts: datetime,
    ) -> int:
        """Stamp ``last_accessed_at = ts`` on the given chunks (#855).

        Single UPDATE statement, scoped to ``namespace_id`` to prevent
        cross-tenant writes through forged ids. Returns the row count.
        Used by the Chronicle reinforcement-on-recall path.
        """
        if not chunk_ids:
            return 0
        async with self._get_session() as session:
            result = await session.execute(
                update(ChunkModel)
                .where(
                    ChunkModel.id.in_(chunk_ids),
                    ChunkModel.namespace_id == namespace_id,
                )
                .values(last_accessed_at=ts)
            )
            await session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    async def delete_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete entities by id, scoped to ``namespace_id``.

        Used by the forget cascade to remove orphan entities from pgvector
        after the graph backend has dropped them. The ``namespace_id``
        filter prevents cross-tenant deletion by id (IDOR family).
        """
        if not entity_ids:
            return 0
        async with self._get_session() as session:
            result = await session.execute(
                delete(EntityModel).where(
                    EntityModel.id.in_(entity_ids),
                    EntityModel.namespace_id == namespace_id,
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    async def delete_relationships_batch(self, relationship_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete relationships by id, scoped to ``namespace_id``.

        Sibling to :meth:`delete_entities_batch`. ``relationships`` is not
        actively written by pgvector today (edges live in the graph
        backend) but the table exists and is kept consistent for any
        downstream reader or future migration. The ``namespace_id``
        filter prevents cross-tenant deletion by id (IDOR family).
        """
        if not relationship_ids:
            return 0
        async with self._get_session() as session:
            result = await session.execute(
                delete(RelationshipModel).where(
                    RelationshipModel.id.in_(relationship_ids),
                    RelationshipModel.namespace_id == namespace_id,
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    async def remove_document_from_entity_sources(
        self,
        entity_ids: list[UUID],
        document_id: UUID,
    ) -> int:
        """Strip ``document_id`` from survivor entities' ``source_document_ids``."""
        if not entity_ids:
            return 0
        async with self._get_session() as session:
            result = await session.execute(
                update(EntityModel)
                .where(EntityModel.id.in_(entity_ids))
                .values(source_document_ids=func.array_remove(EntityModel.source_document_ids, document_id))
            )
            await session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    async def remove_document_from_relationship_sources(
        self,
        relationship_ids: list[UUID],
        document_id: UUID,
    ) -> int:
        """Strip ``document_id`` from survivor relationships' ``source_document_ids``."""
        if not relationship_ids:
            return 0
        async with self._get_session() as session:
            result = await session.execute(
                update(RelationshipModel)
                .where(RelationshipModel.id.in_(relationship_ids))
                .values(source_document_ids=func.array_remove(RelationshipModel.source_document_ids, document_id))
            )
            await session.commit()
            return result.rowcount  # type: ignore[unresolved-attribute]

    def _cosine_similarity(self, embedding_col, query_embedding: list[float]):
        """Build cosine similarity expression, using halfvec cast when enabled.

        When halfvec is enabled, both the column and query vector are cast to
        halfvec to ensure the planner uses the halfvec expression index and
        avoids upcasting back to float32.
        """
        if self.halfvec_enabled:
            from pgvector.sqlalchemy import HALFVEC

            dim = self._embedding_dimension
            casted_col = func.cast(embedding_col, HALFVEC(dim))
            casted_query = func.cast(query_embedding, HALFVEC(dim))
            return 1 - casted_col.cosine_distance(casted_query)
        return 1 - embedding_col.cosine_distance(query_embedding)

    async def _probe_iterative_scan_supported(self) -> bool:
        """One-time probe: does this Postgres + pgvector support hnsw.iterative_scan?

        pgvector >= 0.8 introduced ``hnsw.iterative_scan`` to fix the HNSW
        recall collapse that happens when a selective WHERE filter removes
        most of the candidates returned by the index scan. The probe runs
        once per backend instance and is cached.
        """
        if hasattr(self, "_iterative_scan_supported"):
            return self._iterative_scan_supported  # type: ignore[attr-defined]
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(text("SHOW hnsw.iterative_scan"))
                _ = result.scalar()
                self._iterative_scan_supported = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"hnsw.iterative_scan probe failed (pgvector < 0.8?): {e}")
            self._iterative_scan_supported = False
        return self._iterative_scan_supported

    @trace(
        "khora.pgvector.search_similar",
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
        """Search for similar chunks using vector similarity.

        Uses cosine similarity for matching. Returns list of (chunk, similarity_score) tuples.
        When halfvec is enabled, casts to float16 for faster index scans.
        """
        async with self._get_session() as session:
            # Increase HNSW search accuracy for this transaction
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {self._hnsw_ef_search}"))

            # When a temporal WHERE filter is present, the post-index filter can
            # collapse HNSW recall — enable iterative_scan when pgvector >= 0.8
            # supports it. ``strict_order`` preserves ORDER BY similarity DESC
            # invariants relied on by RRF fusion.
            if (
                created_after is not None or created_before is not None
            ) and await self._probe_iterative_scan_supported():
                await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                await session.execute(text("SET LOCAL hnsw.max_scan_tuples = 20000"))

            similarity = self._cosine_similarity(ChunkModel.embedding, query_embedding)

            query = (
                select(ChunkModel, similarity.label("similarity"))
                .where(
                    ChunkModel.namespace_id == namespace_id,
                    ChunkModel.embedding.is_not(None),
                )
                .order_by(similarity.desc())
                .limit(limit)
            )

            if filter_document_ids:
                query = query.where(ChunkModel.document_id.in_(filter_document_ids))

            if min_similarity > 0:
                query = query.where(similarity >= min_similarity)

            if created_after is not None:
                temporal_col = func.coalesce(ChunkModel.source_timestamp, ChunkModel.created_at)
                query = query.where(temporal_col >= created_after)

            if created_before is not None:
                temporal_col = func.coalesce(ChunkModel.source_timestamp, ChunkModel.created_at)
                query = query.where(temporal_col <= created_before)

            if metadata_filters:
                for key, value in metadata_filters.items():
                    query = query.where(ChunkModel.metadata_.op("->>")(key) == value)

            result = await session.execute(query)
            rows = result.all()

            return [(self._chunk_model_to_domain(row.ChunkModel), row.similarity) for row in rows]

    @trace(
        "khora.pgvector.search_recent_chunks",
        include={"namespace_id", "limit"},
        result=lambda r: {"result_count": len(r)},
    )
    async def search_recent_chunks(
        self,
        namespace_id: UUID,
        limit: int,
        *,
        created_after: datetime | None = None,
    ) -> list[tuple[Chunk, float | None]]:
        """Return the most-recent chunks in namespace, no semantic filter.

        Used by VectorCypher / Chronicle as a parallel channel for RECENCY
        queries, RRF-fused with cosine + BM25. The semantic-relevance gate
        is applied by the caller (not here) — this method is intentionally
        a pure recency sort.

        Returns ``(chunk, None)`` tuples to mirror :meth:`search_similar`'s
        shape; the ``None`` sentinel signals "no cosine score available" so
        downstream RRF code can branch on it.

        Uses the ``ix_chunks_ns_temporal`` expression index from migration
        017 (``namespace_id, COALESCE(source_timestamp, created_at)``) so
        the ORDER BY DESC + LIMIT is index-served.
        """
        async with self._get_session() as session:
            temporal_col = func.coalesce(ChunkModel.source_timestamp, ChunkModel.created_at)

            query = (
                select(ChunkModel)
                .where(ChunkModel.namespace_id == namespace_id)
                .order_by(temporal_col.desc())
                .limit(limit)
            )

            if created_after is not None:
                query = query.where(temporal_col >= created_after)

            result = await session.execute(query)
            rows = result.scalars().all()

            return [(self._chunk_model_to_domain(model), None) for model in rows]

    def _chunk_model_to_domain(self, model: ChunkModel) -> Chunk:
        """Convert ChunkModel to domain Chunk."""
        return Chunk(
            id=model.id,
            namespace_id=model.namespace_id,
            document_id=model.document_id,
            content=model.content,
            chunk_index=model.chunk_index,
            start_char=model.start_char,
            end_char=model.end_char,
            token_count=model.token_count,
            metadata=dict(model.metadata_) if model.metadata_ else {},
            chunker_info=dict(model.chunker_info) if model.chunker_info else {},
            embedding=(
                np.asarray(model.embedding, dtype=np.float32)
                if (_HAS_NUMPY and model.embedding is not None)
                else (list(model.embedding) if model.embedding is not None else None)
            ),
            embedding_model=model.embedding_model,
            created_at=model.created_at,
            source_timestamp=getattr(model, "source_timestamp", None),
            last_accessed_at=getattr(model, "last_accessed_at", None),
            session_id=getattr(model, "session_id", None),
        )

    # =========================================================================
    # Full-text search operations
    # =========================================================================

    @trace(
        "khora.pgvector.search_fulltext",
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
        """Search chunks using PostgreSQL full-text search with ts_rank.

        Uses the content_tsv generated column and GIN index for efficient
        full-text matching.

        ``filter_ast`` REFUSAL (ADR-001): this backend reads the relational
        ``chunks`` table, which lacks the denormalized filter columns the
        recall-filter compiler targets (``khora_chunks``). It therefore CANNOT
        honor a recall filter here. When ``filter_ast`` is present we return
        ``[]`` and log a WARNING rather than attempt to compile the
        ``khora_chunks`` predicate (would SQL-error) or return unfiltered rows
        (would smuggle filter-violating chunks into RRF — forbidden, no PG
        post-filter backstop). The filtered BM25 path is the ``khora_chunks``
        temporal store's ``search_fulltext``; callers route the filtered query
        there (the VectorCypher retriever skips this fallback under a filter).
        """
        if filter_ast is not None:
            # Deliberate refusal, not a caught exception: WARNING + Degradation
            # (ADR-001), no result carrier on this bare-list method so the log
            # is the surfaced signal. No new metric (telemetry-contract churn).
            degradation = Degradation(
                component="pgvector.search_fulltext",
                reason="recall_filter_unsupported_on_relational_chunks",
                detail=(
                    "filter_ast present but relational chunks table lacks denormalized "
                    "filter columns; use khora_chunks temporal store"
                ),
                exception=None,
            )
            logger.warning(
                "Relational chunks BM25 search refused under an active recall filter: {}",
                degradation,
            )
            return []
        async with self._get_session() as session:
            # OR the query terms instead of plainto_tsquery's implicit AND: a
            # full-sentence question rarely has every content word in one chunk,
            # so AND matched nothing (BM25 channel returned 0). ts_rank_cd ranks
            # by how many query terms co-occur and how closely.
            terms = "".join(c if c.isalnum() else " " for c in query_text).split()
            if not terms:
                return []
            tsquery = func.to_tsquery(language, " | ".join(terms))
            rank = func.ts_rank_cd(ChunkModel.content_tsv, tsquery)

            query = (
                select(ChunkModel, rank.label("rank"))
                .where(
                    ChunkModel.namespace_id == namespace_id,
                    ChunkModel.content_tsv.op("@@")(tsquery),
                )
                .order_by(rank.desc())
                .limit(limit)
            )

            if created_after is not None:
                temporal_col = func.coalesce(ChunkModel.source_timestamp, ChunkModel.created_at)
                query = query.where(temporal_col >= created_after)

            if created_before is not None:
                temporal_col = func.coalesce(ChunkModel.source_timestamp, ChunkModel.created_at)
                query = query.where(temporal_col <= created_before)

            result = await session.execute(query)
            rows = result.all()

            return [(self._chunk_model_to_domain(row.ChunkModel), float(row.rank)) for row in rows]

    # =========================================================================
    # Entity operations (for vector search via PostgreSQL)
    # =========================================================================

    async def create_entity(self, entity) -> None:
        """Create an entity record in PostgreSQL for vector search.

        This stores the entity metadata and embedding in PostgreSQL,
        complementing the Neo4j storage for graph traversal.

        Uses upsert pattern: if entity already exists, updates it instead.
        """
        await _retry_on_deadlock(self._upsert_entity, entity)

    async def update_entity(self, entity, *, namespace_id: UUID) -> None:
        """Update an entity record in PostgreSQL, scoped to ``namespace_id``.

        Uses upsert to handle race conditions and entities created before
        dual-storage was implemented. The ``namespace_id`` kwarg is
        defense-in-depth (IDOR family) — asserted equal to ``entity.namespace_id``.
        """
        if entity.namespace_id != namespace_id:
            raise ValueError(
                f"entity.namespace_id ({entity.namespace_id}) does not match namespace_id kwarg ({namespace_id})"
            )
        await _retry_on_deadlock(self._upsert_entity, entity)

    async def _upsert_entity(self, entity) -> None:
        """Internal upsert used by both create_entity and update_entity.

        Uses the unique constraint on (namespace_id, name, entity_type) to
        properly merge entities with the same identity, matching Neo4j's
        MERGE semantics.

        Acquires a namespace-scoped advisory lock to prevent deadlocks
        when concurrent coroutines upsert entities in the same namespace.
        """
        from sqlalchemy.dialects.postgresql import insert

        async with self._get_session() as session:
            # Advisory lock prevents deadlocks with concurrent upserts
            key1, key2 = _namespace_lock_keys(entity.namespace_id)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                {"key1": key1, "key2": key2},
            )

            stmt = insert(EntityModel).values(
                id=entity.id,
                namespace_id=entity.namespace_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                attributes=entity.attributes,
                source_document_ids=entity.source_document_ids,
                source_chunk_ids=entity.source_chunk_ids,
                mention_count=entity.mention_count,
                embedding=entity.embedding,
                embedding_model=entity.embedding_model,
                valid_from=entity.valid_from,
                valid_until=entity.valid_until,
                confidence=entity.confidence,
                metadata_=entity.metadata,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_entities_namespace_name_type",
                set_={
                    "description": stmt.excluded.description,
                    "attributes": stmt.excluded.attributes,
                    "source_document_ids": _accumulate_source_ids_sql("source_document_ids"),
                    "source_chunk_ids": _accumulate_source_ids_sql("source_chunk_ids"),
                    "mention_count": stmt.excluded.mention_count,
                    "embedding": stmt.excluded.embedding,
                    "embedding_model": stmt.excluded.embedding_model,
                    "valid_from": stmt.excluded.valid_from,
                    "valid_until": stmt.excluded.valid_until,
                    "confidence": stmt.excluded.confidence,
                    "metadata": stmt.excluded.metadata,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID):
        """Get an entity by ID from PostgreSQL, scoped to ``namespace_id``.

        Returns ``None`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant entity access by id
        (IDOR).
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(EntityModel).where(
                    EntityModel.id == entity_id,
                    EntityModel.namespace_id == namespace_id,
                )
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            return self._entity_model_to_domain(model)

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict:
        """Fetch multiple entities by ID from pgvector storage, scoped to ``namespace_id``.

        Provides a pgvector-backed fallback for Chronicle (which has no graph
        backend) so the entity co-occurrence channel can resolve entities.

        Entities belonging to any other namespace are silently dropped from
        the result to prevent cross-tenant IDOR (IDOR family).
        """
        if not entity_ids:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(EntityModel).where(
                    EntityModel.id.in_(entity_ids),
                    EntityModel.namespace_id == namespace_id,
                )
            )
            return {model.id: self._entity_model_to_domain(model) for model in result.scalars()}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict:
        """Fetch entities by name within a namespace (one-shot batch).

        Used by Chronicle to resolve event subjects to Entity records when
        no graph backend is available. Returns a dict keyed by name; if more
        than one entity shares the same name (e.g. different entity_types),
        the entry with the highest mention_count wins (stable, repeatable).
        Names not found are simply absent from the result.
        """
        if not names:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(EntityModel)
                .where(
                    EntityModel.namespace_id == namespace_id,
                    EntityModel.name.in_(names),
                )
                .order_by(EntityModel.mention_count.desc())
            )
            out: dict[str, Any] = {}
            for model in result.scalars():
                # First row per name wins (already sorted by mention_count DESC).
                if model.name not in out:
                    out[model.name] = self._entity_model_to_domain(model)
            return out

    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL within ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant entity-existence
        enumeration (IDOR).
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(func.count(EntityModel.id)).where(
                    EntityModel.id == entity_id,
                    EntityModel.namespace_id == namespace_id,
                )
            )
            return result.scalar_one() > 0

    def _entity_model_to_domain(self, model: EntityModel):
        """Convert EntityModel to domain Entity."""
        from khora.core.models import Entity

        return Entity(
            id=model.id,
            namespace_id=model.namespace_id,
            name=model.name,
            entity_type=model.entity_type,
            description=model.description,
            attributes=model.attributes or {},
            source_document_ids=model.source_document_ids or [],
            source_chunk_ids=model.source_chunk_ids or [],
            mention_count=model.mention_count,
            embedding=(
                np.asarray(model.embedding, dtype=np.float32)
                if (_HAS_NUMPY and model.embedding is not None)
                else (list(model.embedding) if model.embedding is not None else None)
            ),
            embedding_model=model.embedding_model or "",
            valid_from=model.valid_from,
            valid_until=model.valid_until,
            confidence=model.confidence,
            metadata=model.metadata_ or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def upsert_entities_batch(self, namespace_id: UUID, entities: list, *, batch_size: int = 200) -> list[tuple]:
        """Batch upsert entity records in PostgreSQL.

        Uses multi-row INSERT ... ON CONFLICT DO UPDATE statements, chunked
        into sub-batches to stay within asyncpg's parameter limit.

        Acquires a namespace-scoped PostgreSQL advisory lock to serialise
        concurrent entity upserts within the same namespace, preventing
        deadlocks when multiple documents share entities.  Upserts for
        different namespaces proceed in parallel without contention.

        Returns list of ``(entity, is_new)`` tuples. ``is_new`` is derived
        from Postgres's ``xmax`` system column on the RETURNING row: an
        ``INSERT ... ON CONFLICT DO UPDATE`` leaves ``xmax = 0`` on freshly
        inserted rows and a non-zero locking-tx id on rows that took the
        UPDATE branch, so ``(xmax = 0)`` reliably distinguishes "new" from
        "matched + updated" without a second round-trip. Matches the Neo4j
        adapter's MERGE semantics (#719).
        """
        if not entities:
            return []

        async def _do_upsert():
            from sqlalchemy.dialects.postgresql import insert

            # Sort by (namespace_id, name, entity_type) to ensure consistent lock ordering
            sorted_entities = sorted(entities, key=lambda e: (str(e.namespace_id), e.name, str(e.entity_type)))
            key1, key2 = _namespace_lock_keys(namespace_id)

            # (name, entity_type) → is_new. Within a single namespace batch
            # the uq_entities_namespace_name_type constraint reduces to this
            # pair, so it uniquely identifies each affected row.
            is_new_by_key: dict[tuple[str, str], bool] = {}

            async with self._get_session() as session:
                # Acquire namespace-scoped advisory lock for the duration of this
                # transaction.  pg_advisory_xact_lock auto-releases on commit/rollback.
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                    {"key1": key1, "key2": key2},
                )

                for start in range(0, len(sorted_entities), batch_size):
                    batch = sorted_entities[start : start + batch_size]
                    values = [
                        {
                            "id": entity.id,
                            "namespace_id": entity.namespace_id,
                            "name": entity.name,
                            "entity_type": (entity.entity_type),
                            "description": entity.description,
                            "attributes": entity.attributes,
                            "source_document_ids": entity.source_document_ids,
                            "source_chunk_ids": entity.source_chunk_ids,
                            "mention_count": entity.mention_count,
                            "embedding": entity.embedding,
                            "embedding_model": entity.embedding_model,
                            "valid_from": entity.valid_from,
                            "valid_until": entity.valid_until,
                            "confidence": entity.confidence,
                            "metadata_": entity.metadata,
                            "created_at": entity.created_at,
                            "updated_at": entity.updated_at,
                        }
                        for entity in batch
                    ]
                    stmt = insert(EntityModel).values(values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_entities_namespace_name_type",
                        set_={
                            "description": stmt.excluded.description,
                            "attributes": stmt.excluded.attributes,
                            "source_document_ids": _accumulate_source_ids_sql("source_document_ids"),
                            "source_chunk_ids": _accumulate_source_ids_sql("source_chunk_ids"),
                            "mention_count": stmt.excluded.mention_count,
                            "embedding": stmt.excluded.embedding,
                            "embedding_model": stmt.excluded.embedding_model,
                            "valid_from": stmt.excluded.valid_from,
                            "valid_until": stmt.excluded.valid_until,
                            "confidence": stmt.excluded.confidence,
                            "metadata": stmt.excluded.metadata,
                            "updated_at": stmt.excluded.updated_at,
                        },
                    )
                    # xmax = 0 on freshly inserted rows; non-zero on rows
                    # touched by the ON CONFLICT DO UPDATE branch. Canonical
                    # PG idiom for distinguishing INSERT vs UPDATE outcomes
                    # within a single upsert statement.
                    stmt = stmt.returning(
                        EntityModel.name,
                        EntityModel.entity_type,
                        literal_column("(xmax = 0)").label("is_new"),
                    )
                    result = await session.execute(stmt)
                    for row in result:
                        is_new_by_key[(row.name, row.entity_type)] = bool(row.is_new)

                # Single commit for all sub-batches under the advisory lock
                await session.commit()

            # Default to True for any input whose key didn't come back in
            # RETURNING (defensive — shouldn't happen in practice).
            return [(entity, is_new_by_key.get((entity.name, entity.entity_type), True)) for entity in sorted_entities]

        return await _retry_on_deadlock(_do_upsert)

    # =========================================================================
    # Entity embedding operations
    # =========================================================================

    async def update_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Update the embedding for an entity, scoped to ``namespace_id``.

        Updates are skipped silently when the entity belongs to a different
        namespace (cross-namespace IDOR).
        """
        async with self._get_session() as session:
            await session.execute(
                update(EntityModel)
                .where(
                    EntityModel.id == entity_id,
                    EntityModel.namespace_id == namespace_id,
                )
                .values(
                    embedding=embedding,
                    embedding_model=model,
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Update embeddings for multiple entities in a single transaction.

        Uses executemany semantics to send all updates in a single round-trip
        instead of N individual UPDATE statements. Updates are restricted to
        ``namespace_id``; ids in other namespaces are silently skipped
        (IDOR family).

        Args:
            updates: List of (entity_id, embedding, model) tuples

        Returns:
            Number of entities updated
        """
        if not updates:
            return 0

        async def _do_batch():
            from sqlalchemy import bindparam

            # Sort by entity_id for consistent lock ordering across concurrent batches
            sorted_updates = sorted(updates, key=lambda u: str(u[0]))
            now = datetime.now(UTC)

            # Use Core table to avoid ORM bulk-update PK requirements
            tbl = EntityModel.__table__
            stmt = (
                tbl.update()  # type: ignore[unresolved-attribute]
                .where(
                    tbl.c.id == bindparam("eid"),
                    tbl.c.namespace_id == namespace_id,
                )
                .values(
                    embedding=bindparam("emb"),
                    embedding_model=bindparam("mdl"),
                    updated_at=bindparam("ts"),
                )
            )
            params = [{"eid": eid, "emb": emb, "mdl": mdl, "ts": now} for eid, emb, mdl in sorted_updates]

            async with self._get_session() as session:
                await session.connection()
                await session.execute(stmt, params)
                await session.commit()
            # Return the input count — executemany rowcount is unreliable.
            # Ids outside the namespace are silently filtered by WHERE.
            return len(sorted_updates)

        return await _retry_on_deadlock(_do_batch)

    @trace(
        "khora.pgvector.search_similar_entities",
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
        """Search for similar entities by embedding.

        Returns list of (entity_id, similarity_score) tuples.
        When halfvec is enabled, casts to float16 for faster index scans.
        """
        async with self._get_session() as session:
            # Increase HNSW search accuracy for this transaction
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {self._hnsw_ef_search}"))

            similarity = self._cosine_similarity(EntityModel.embedding, query_embedding)

            query = (
                select(EntityModel.id, similarity.label("similarity"))
                .where(
                    EntityModel.namespace_id == namespace_id,
                    EntityModel.embedding.is_not(None),
                )
                .order_by(similarity.desc())
                .limit(limit)
            )

            if min_similarity > 0:
                query = query.where(similarity >= min_similarity)

            result = await session.execute(query)
            return [(row.id, row.similarity) for row in result.all()]

    # =========================================================================
    # Utility operations
    # =========================================================================

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count total chunks in a namespace."""
        async with self._get_session() as session:
            result = await session.execute(
                select(func.count(ChunkModel.id)).where(ChunkModel.namespace_id == namespace_id)
            )
            return result.scalar_one()

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count total entities in a namespace."""
        async with self._get_session() as session:
            result = await session.execute(
                select(func.count(EntityModel.id)).where(EntityModel.namespace_id == namespace_id)
            )
            return result.scalar_one()

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks in a namespace.

        Args:
            namespace_id: Namespace ID
            limit: Maximum chunks to return
            offset: Offset for pagination

        Returns:
            List of chunks
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel)
                .where(ChunkModel.namespace_id == namespace_id)
                .order_by(ChunkModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = result.scalars().all()
            return [self._chunk_model_to_domain(row) for row in rows]

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """List entities in a namespace.

        Coordinator fallback for graph-less stacks (e.g. chronicle on
        PostgreSQL-only).  Entities live in the ``entities`` table that
        pgvector owns, so we can serve listings directly even when no
        graph backend is wired up.
        """
        async with self._get_session() as session:
            stmt = select(EntityModel).where(EntityModel.namespace_id == namespace_id)
            if entity_type:
                stmt = stmt.where(EntityModel.entity_type == entity_type)
            stmt = stmt.order_by(EntityModel.name).limit(limit).offset(offset)
            result = await session.execute(stmt)
            return [self._entity_model_to_domain(model) for model in result.scalars()]

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list:
        """List relationships in a namespace.

        Coordinator fallback for graph-less stacks.  In chronicle+PG-only
        deployments the ``relationships`` table is not actively written to
        (relationships live in Neo4j when configured), so this will return
        an empty list — but it no longer crashes the caller.
        """
        from khora.core.models import Relationship

        async with self._get_session() as session:
            stmt = select(RelationshipModel).where(RelationshipModel.namespace_id == namespace_id)
            if relationship_type:
                stmt = stmt.where(RelationshipModel.relationship_type == relationship_type)
            stmt = stmt.order_by(RelationshipModel.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(stmt)
            return [
                Relationship(
                    id=m.id,
                    namespace_id=m.namespace_id,
                    source_entity_id=m.source_entity_id,
                    target_entity_id=m.target_entity_id,
                    relationship_type=m.relationship_type,
                    description=m.description,
                    properties=m.properties or {},
                    source_document_ids=m.source_document_ids or [],
                    source_chunk_ids=m.source_chunk_ids or [],
                    valid_from=m.valid_from,
                    valid_until=m.valid_until,
                    confidence=m.confidence,
                    weight=m.weight,
                    metadata=m.metadata_ or {},
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in result.scalars()
            ]

    # =========================================================================
    # Chronicle engine: events + facts (migration 024)
    #
    # Inputs use the duck-typed ChronicleEvent / MemoryFact dataclasses from
    # ``khora.engines.chronicle`` to avoid pulling LLM dependencies at import
    # time. Each helper reads/writes only the column set defined in the
    # ``chronicle_events`` / ``memory_facts`` tables.
    # =========================================================================

    async def write_events(
        self,
        events: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert chronicle_events rows; returns inserted IDs in input order."""
        if not events:
            return []
        models = [
            ChronicleEventModel(
                id=getattr(ev, "id", None),
                namespace_id=namespace_id,
                chunk_id=ev.chunk_id,
                subject=ev.subject,
                verb=ev.verb,
                object_=ev.object or None,
                observation_date=ev.observation_date or datetime.now(UTC),
                referenced_date=ev.referenced_date,
                relative_offset=ev.relative_offset or None,
                confidence=float(ev.confidence),
                source_text=ev.source_text or "",
                embedding=getattr(ev, "embedding", None),
            )
            for ev in events
        ]
        async with self._get_session() as session:
            session.add_all(models)
            await session.commit()
        return [m.id for m in models]

    async def write_facts(
        self,
        facts: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert memory_facts rows; returns inserted IDs in input order.

        Maps ``MemoryFact`` (subject/predicate/object_/fact_text) directly to
        the ``memory_facts`` table — Chronicle #3 reshaped the dataclass so
        the placeholder adapter from Chronicle #1 is no longer needed.
        """
        if not facts:
            return []
        models = [
            MemoryFactModel(
                id=getattr(f, "id", None),
                namespace_id=namespace_id,
                subject=f.subject or "",
                predicate=f.predicate or "",
                object_=f.object_ or "",
                fact_text=f.fact_text or "",
                confidence=float(f.confidence),
                is_active=bool(getattr(f, "is_active", True)),
                superseded_by=getattr(f, "superseded_by", None),
                source_chunk_ids=list(getattr(f, "source_chunk_ids", []) or []),
            )
            for f in facts
        ]
        async with self._get_session() as session:
            session.add_all(models)
            await session.commit()
        return [m.id for m in models]

    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[ChronicleEventModel]:
        """Query chronicle_events filtered by subject and referenced_date range."""
        async with self._get_session() as session:
            stmt = select(ChronicleEventModel).where(ChronicleEventModel.namespace_id == namespace_id)
            if subject is not None:
                stmt = stmt.where(ChronicleEventModel.subject == subject)
            if since is not None:
                stmt = stmt.where(ChronicleEventModel.referenced_date >= since)
            if until is not None:
                stmt = stmt.where(ChronicleEventModel.referenced_date <= until)
            stmt = stmt.order_by(ChronicleEventModel.referenced_date.desc().nullslast()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def query_active_facts_for_subject(
        self,
        namespace_id: UUID,
        subject: str,
    ) -> list[MemoryFactModel]:
        """Return all active (not superseded) memory facts for a subject."""
        async with self._get_session() as session:
            result = await session.execute(
                select(MemoryFactModel)
                .where(
                    MemoryFactModel.namespace_id == namespace_id,
                    MemoryFactModel.subject == subject,
                    MemoryFactModel.is_active.is_(True),
                )
                .order_by(MemoryFactModel.created_at.desc())
            )
            return list(result.scalars().all())

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID, *, namespace_id: UUID) -> None:
        """Mark a fact inactive and link it to its replacement.

        Scoped to ``namespace_id`` — no-op when the fact belongs to a
        different namespace (cross-namespace IDOR).
        """
        async with self._get_session() as session:
            await session.execute(
                update(MemoryFactModel)
                .where(
                    MemoryFactModel.id == fact_id,
                    MemoryFactModel.namespace_id == namespace_id,
                )
                .values(is_active=False, superseded_by=superseded_by, updated_at=datetime.now(UTC))
            )
            await session.commit()

    async def get_embedding_stats(self, namespace_id: UUID) -> dict:
        """Get statistics about embeddings in a namespace."""
        async with self._get_session() as session:
            # Count chunks with embeddings
            chunk_count = await session.execute(
                select(func.count(ChunkModel.id)).where(
                    ChunkModel.namespace_id == namespace_id,
                    ChunkModel.embedding.is_not(None),
                )
            )
            # Count entities with embeddings
            entity_count = await session.execute(
                select(func.count(EntityModel.id)).where(
                    EntityModel.namespace_id == namespace_id,
                    EntityModel.embedding.is_not(None),
                )
            )

            return {
                "chunk_embeddings": chunk_count.scalar_one(),
                "entity_embeddings": entity_count.scalar_one(),
            }
