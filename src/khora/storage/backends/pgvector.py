"""pgvector backend for vector embeddings storage.

Handles storage and retrieval of embeddings for semantic search
using pgvector extension in PostgreSQL.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from khora.core.models import Chunk, ChunkMetadata
from khora.db.models import Base, ChunkModel, EntityModel
from khora.db.schema import sync_enum_values
from khora.storage.backends.mixins import AsyncSessionMixin
from khora.telemetry import trace

if TYPE_CHECKING:
    pass

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


_DEADLOCK_MAX_RETRIES = 3

# Advisory lock key-space for entity upserts (avoids collision with other lock users).
_ENTITY_UPSERT_LOCK_KEY1 = 0x4B484F52  # "KHOR" in hex


def _namespace_lock_key(namespace_id: UUID) -> int:
    """Derive a stable 32-bit advisory lock key from a namespace UUID."""
    b = namespace_id.bytes
    chunks = struct.unpack(">IIII", b)
    folded = chunks[0] ^ chunks[1] ^ chunks[2] ^ chunks[3]
    return struct.unpack(">i", struct.pack(">I", folded))[0]


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
            chunk_index=chunk.metadata.chunk_index,
            start_char=chunk.metadata.start_char,
            end_char=chunk.metadata.end_char,
            token_count=chunk.metadata.token_count,
            metadata_=chunk.metadata.custom,
            embedding=chunk.embedding,
            embedding_model=chunk.embedding_model,
            created_at=chunk.created_at,
            source_timestamp=getattr(chunk, "source_timestamp", None),
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
                chunk_index=chunk.metadata.chunk_index,
                start_char=chunk.metadata.start_char,
                end_char=chunk.metadata.end_char,
                token_count=chunk.metadata.token_count,
                metadata_=chunk.metadata.custom,
                embedding=chunk.embedding,
                embedding_model=chunk.embedding_model,
                created_at=chunk.created_at,
                source_timestamp=getattr(chunk, "source_timestamp", None),
            )
            for chunk in chunks
        ]
        session.add_all(models)
        if commit:
            await session.commit()
        return chunks

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Get a chunk by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(ChunkModel).where(ChunkModel.id == chunk_id))
            model = result.scalar_one_or_none()
            return self._chunk_model_to_domain(model) if model else None

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        """Get multiple chunks by ID in a single query.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing chunks)
        """
        if not chunk_ids:
            return {}

        async with self._get_session() as session:
            result = await session.execute(select(ChunkModel).where(ChunkModel.id.in_(chunk_ids)))
            models = result.scalars().all()
            return {m.id: self._chunk_model_to_domain(m) for m in models}

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.document_id == document_id).order_by(ChunkModel.chunk_index)
            )
            return [self._chunk_model_to_domain(m) for m in result.scalars().all()]

    async def delete_chunks_by_document(self, document_id: UUID, *, session: AsyncSession | None = None) -> int:
        """Delete all chunks for a document.

        When *session* is provided the caller owns the transaction —
        no commit is issued.  When ``None``, a private session is used
        and committed automatically.
        """
        if session is not None:
            result = await session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
            return result.rowcount  # type: ignore[unresolved-attribute]
        async with self._get_session() as own_session:
            result = await own_session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
            await own_session.commit()
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

    def _chunk_model_to_domain(self, model: ChunkModel) -> Chunk:
        """Convert ChunkModel to domain Chunk."""
        return Chunk(
            id=model.id,
            namespace_id=model.namespace_id,
            document_id=model.document_id,
            content=model.content,
            metadata=ChunkMetadata(
                document_id=model.document_id,
                chunk_index=model.chunk_index,
                start_char=model.start_char,
                end_char=model.end_char,
                token_count=model.token_count,
                custom=model.metadata_,
            ),
            embedding=(
                np.asarray(model.embedding, dtype=np.float32)
                if (_HAS_NUMPY and model.embedding is not None)
                else (list(model.embedding) if model.embedding is not None else None)
            ),
            embedding_model=model.embedding_model,
            created_at=model.created_at,
            source_timestamp=getattr(model, "source_timestamp", None),
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
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search with ts_rank.

        Uses the content_tsv generated column and GIN index for efficient
        full-text matching.
        """
        async with self._get_session() as session:
            tsquery = func.plainto_tsquery(language, query_text)
            rank = func.ts_rank(ChunkModel.content_tsv, tsquery)

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

    async def update_entity(self, entity) -> None:
        """Update an entity record in PostgreSQL.

        Uses upsert to handle race conditions and entities created before
        dual-storage was implemented.
        """
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
            lock_key2 = _namespace_lock_key(entity.namespace_id)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                {"key1": _ENTITY_UPSERT_LOCK_KEY1, "key2": lock_key2},
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
                    "source_document_ids": stmt.excluded.source_document_ids,
                    "source_chunk_ids": stmt.excluded.source_chunk_ids,
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

    async def get_entity(self, entity_id: UUID):
        """Get an entity by ID from PostgreSQL."""
        async with self._get_session() as session:
            result = await session.execute(select(EntityModel).where(EntityModel.id == entity_id))
            model = result.scalar_one_or_none()
            if model is None:
                return None
            return self._entity_model_to_domain(model)

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict:
        """Fetch multiple entities by ID from pgvector storage.

        Provides a pgvector-backed fallback for Chronicle (which has no graph
        backend) so the entity co-occurrence channel can resolve entities.
        """
        if not entity_ids:
            return {}
        async with self._get_session() as session:
            result = await session.execute(select(EntityModel).where(EntityModel.id.in_(entity_ids)))
            return {model.id: self._entity_model_to_domain(model) for model in result.scalars()}

    async def entity_exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL."""
        async with self._get_session() as session:
            result = await session.execute(select(func.count(EntityModel.id)).where(EntityModel.id == entity_id))
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

        Returns list of (entity, is_new) tuples (is_new is approximate).
        """
        if not entities:
            return []

        async def _do_upsert():
            from sqlalchemy.dialects.postgresql import insert

            # Sort by (namespace_id, name, entity_type) to ensure consistent lock ordering
            sorted_entities = sorted(entities, key=lambda e: (str(e.namespace_id), e.name, str(e.entity_type)))
            lock_key2 = _namespace_lock_key(namespace_id)

            async with self._get_session() as session:
                # Acquire namespace-scoped advisory lock for the duration of this
                # transaction.  pg_advisory_xact_lock auto-releases on commit/rollback.
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                    {"key1": _ENTITY_UPSERT_LOCK_KEY1, "key2": lock_key2},
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
                            "source_document_ids": stmt.excluded.source_document_ids,
                            "source_chunk_ids": stmt.excluded.source_chunk_ids,
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

                # Single commit for all sub-batches under the advisory lock
                await session.commit()

            return [(entity, True) for entity in sorted_entities]

        return await _retry_on_deadlock(_do_upsert)

    # =========================================================================
    # Entity embedding operations
    # =========================================================================

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Update the embedding for an entity."""
        async with self._get_session() as session:
            await session.execute(
                update(EntityModel)
                .where(EntityModel.id == entity_id)
                .values(
                    embedding=embedding,
                    embedding_model=model,
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        """Update embeddings for multiple entities in a single transaction.

        Uses executemany semantics to send all updates in a single round-trip
        instead of N individual UPDATE statements.

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
                .where(tbl.c.id == bindparam("eid"))
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
