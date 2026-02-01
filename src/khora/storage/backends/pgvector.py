"""pgvector backend for vector embeddings storage.

Handles storage and retrieval of embeddings for semantic search
using pgvector extension in PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models import Chunk, ChunkMetadata
from khora.db.models import Base, ChunkModel, EntityModel

if TYPE_CHECKING:
    pass


class PgVectorBackend:
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
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        """Initialize the pgvector backend.

        Args:
            database_url: PostgreSQL connection URL (with pgvector extension)
            embedding_dimension: Dimension of embedding vectors
            echo: Enable SQL echo logging
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
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
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Establish connection to the database."""
        if self._engine is not None:
            return

        logger.info("Connecting to pgvector...")
        self._engine = create_async_engine(
            self._database_url,
            echo=self._echo,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Ensure pgvector extension is enabled
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        logger.info("Connected to pgvector")

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            logger.info("Disconnecting from pgvector...")
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

    def _get_session(self) -> AsyncSession:
        """Get a new database session."""
        if self._session_factory is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._session_factory()

    async def create_tables(self) -> None:
        """Create all database tables (for testing/development)."""
        if self._engine is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # =========================================================================
    # Chunk operations
    # =========================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with its embedding."""
        async with self._get_session() as session:
            model = ChunkModel(
                id=str(chunk.id),
                namespace_id=str(chunk.namespace_id),
                document_id=str(chunk.document_id),
                content=chunk.content,
                chunk_index=chunk.metadata.chunk_index,
                start_char=chunk.metadata.start_char,
                end_char=chunk.metadata.end_char,
                token_count=chunk.metadata.token_count,
                metadata_=chunk.metadata.custom,
                embedding=chunk.embedding,
                embedding_model=chunk.embedding_model,
                created_at=chunk.created_at,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._chunk_model_to_domain(model)

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        if not chunks:
            return []

        async with self._get_session() as session:
            models = [
                ChunkModel(
                    id=str(chunk.id),
                    namespace_id=str(chunk.namespace_id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    chunk_index=chunk.metadata.chunk_index,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char,
                    token_count=chunk.metadata.token_count,
                    metadata_=chunk.metadata.custom,
                    embedding=chunk.embedding,
                    embedding_model=chunk.embedding_model,
                    created_at=chunk.created_at,
                )
                for chunk in chunks
            ]
            session.add_all(models)
            await session.commit()
            return chunks

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Get a chunk by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(ChunkModel).where(ChunkModel.id == str(chunk_id)))
            model = result.scalar_one_or_none()
            return self._chunk_model_to_domain(model) if model else None

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        async with self._get_session() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.document_id == str(document_id)).order_by(ChunkModel.chunk_index)
            )
            return [self._chunk_model_to_domain(m) for m in result.scalars().all()]

    async def delete_chunks_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document."""
        async with self._get_session() as session:
            result = await session.execute(delete(ChunkModel).where(ChunkModel.document_id == str(document_id)))
            await session.commit()
            return result.rowcount

    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity.

        Uses cosine similarity for matching. Returns list of (chunk, similarity_score) tuples.
        """
        async with self._get_session() as session:
            # Build cosine similarity expression
            # pgvector uses <=> for cosine distance, so similarity = 1 - distance
            similarity = 1 - ChunkModel.embedding.cosine_distance(query_embedding)

            query = (
                select(ChunkModel, similarity.label("similarity"))
                .where(
                    ChunkModel.namespace_id == str(namespace_id),
                    ChunkModel.embedding.is_not(None),
                )
                .order_by(similarity.desc())
                .limit(limit)
            )

            if filter_document_ids:
                query = query.where(ChunkModel.document_id.in_([str(d) for d in filter_document_ids]))

            if min_similarity > 0:
                query = query.where(similarity >= min_similarity)

            result = await session.execute(query)
            rows = result.all()

            return [(self._chunk_model_to_domain(row.ChunkModel), row.similarity) for row in rows]

    def _chunk_model_to_domain(self, model: ChunkModel) -> Chunk:
        """Convert ChunkModel to domain Chunk."""
        return Chunk(
            id=UUID(model.id),
            namespace_id=UUID(model.namespace_id),
            document_id=UUID(model.document_id),
            content=model.content,
            metadata=ChunkMetadata(
                document_id=UUID(model.document_id),
                chunk_index=model.chunk_index,
                start_char=model.start_char,
                end_char=model.end_char,
                token_count=model.token_count,
                custom=model.metadata_,
            ),
            embedding=list(model.embedding) if model.embedding is not None else None,
            embedding_model=model.embedding_model,
            created_at=model.created_at,
        )

    # =========================================================================
    # Full-text search operations
    # =========================================================================

    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
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
                    ChunkModel.namespace_id == str(namespace_id),
                    ChunkModel.content_tsv.op("@@")(tsquery),
                )
                .order_by(rank.desc())
                .limit(limit)
            )

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
        from sqlalchemy.dialects.postgresql import insert

        async with self._get_session() as session:
            stmt = insert(EntityModel).values(
                id=str(entity.id),
                namespace_id=str(entity.namespace_id),
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                attributes=entity.attributes,
                source_document_ids=[str(d) for d in entity.source_document_ids],
                source_chunk_ids=[str(c) for c in entity.source_chunk_ids],
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
            # On conflict (entity already exists), update all fields
            # Note: use database column name "metadata" not Python attribute "metadata_"
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": stmt.excluded.name,
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

    async def update_entity(self, entity) -> None:
        """Update an entity record in PostgreSQL.

        Uses upsert to handle race conditions and entities created before
        dual-storage was implemented.
        """
        from sqlalchemy.dialects.postgresql import insert

        async with self._get_session() as session:
            stmt = insert(EntityModel).values(
                id=str(entity.id),
                namespace_id=str(entity.namespace_id),
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                attributes=entity.attributes,
                source_document_ids=[str(d) for d in entity.source_document_ids],
                source_chunk_ids=[str(c) for c in entity.source_chunk_ids],
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
            # On conflict, update all fields
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": stmt.excluded.name,
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
            result = await session.execute(select(EntityModel).where(EntityModel.id == str(entity_id)))
            model = result.scalar_one_or_none()
            if model is None:
                return None
            return self._entity_model_to_domain(model)

    async def entity_exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL."""
        async with self._get_session() as session:
            result = await session.execute(select(func.count(EntityModel.id)).where(EntityModel.id == str(entity_id)))
            return result.scalar_one() > 0

    def _entity_model_to_domain(self, model: EntityModel):
        """Convert EntityModel to domain Entity."""
        from uuid import UUID as UUIDType

        from khora.core.models import Entity

        return Entity(
            id=UUIDType(model.id),
            namespace_id=UUIDType(model.namespace_id),
            name=model.name,
            entity_type=model.entity_type,
            description=model.description,
            attributes=model.attributes or {},
            source_document_ids=[UUIDType(d) for d in (model.source_document_ids or [])],
            source_chunk_ids=[UUIDType(c) for c in (model.source_chunk_ids or [])],
            mention_count=model.mention_count,
            embedding=list(model.embedding) if model.embedding is not None else None,
            embedding_model=model.embedding_model or "",
            valid_from=model.valid_from,
            valid_until=model.valid_until,
            confidence=model.confidence,
            metadata=model.metadata_ or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def upsert_entities_batch(self, namespace_id: UUID, entities: list) -> list[tuple]:
        """Batch upsert entity records in PostgreSQL.

        Uses INSERT ... ON CONFLICT DO UPDATE for each entity.
        Returns list of (entity, is_new) tuples (is_new is approximate).
        """
        if not entities:
            return []

        from sqlalchemy.dialects.postgresql import insert

        results: list[tuple] = []
        async with self._get_session() as session:
            for entity in entities:
                stmt = insert(EntityModel).values(
                    id=str(entity.id),
                    namespace_id=str(entity.namespace_id),
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    attributes=entity.attributes,
                    source_document_ids=[str(d) for d in entity.source_document_ids],
                    source_chunk_ids=[str(c) for c in entity.source_chunk_ids],
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
                    index_elements=["id"],
                    set_={
                        "name": stmt.excluded.name,
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
                results.append((entity, True))  # Approximate — we don't know if new or updated
            await session.commit()

        return results

    # =========================================================================
    # Entity embedding operations
    # =========================================================================

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Update the embedding for an entity."""
        async with self._get_session() as session:
            await session.execute(
                update(EntityModel)
                .where(EntityModel.id == str(entity_id))
                .values(
                    embedding=embedding,
                    embedding_model=model,
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()

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
        """
        async with self._get_session() as session:
            similarity = 1 - EntityModel.embedding.cosine_distance(query_embedding)

            query = (
                select(EntityModel.id, similarity.label("similarity"))
                .where(
                    EntityModel.namespace_id == str(namespace_id),
                    EntityModel.embedding.is_not(None),
                )
                .order_by(similarity.desc())
                .limit(limit)
            )

            if min_similarity > 0:
                query = query.where(similarity >= min_similarity)

            result = await session.execute(query)
            return [(UUID(row.id), row.similarity) for row in result.all()]

    # =========================================================================
    # Utility operations
    # =========================================================================

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count total chunks in a namespace."""
        async with self._get_session() as session:
            result = await session.execute(
                select(func.count(ChunkModel.id)).where(ChunkModel.namespace_id == str(namespace_id))
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
                .where(ChunkModel.namespace_id == str(namespace_id))
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
                    ChunkModel.namespace_id == str(namespace_id),
                    ChunkModel.embedding.is_not(None),
                )
            )
            # Count entities with embeddings
            entity_count = await session.execute(
                select(func.count(EntityModel.id)).where(
                    EntityModel.namespace_id == str(namespace_id),
                    EntityModel.embedding.is_not(None),
                )
            )

            return {
                "chunk_embeddings": chunk_count.scalar_one(),
                "entity_embeddings": entity_count.scalar_one(),
            }
