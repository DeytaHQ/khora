"""PostgreSQL backend for relational data storage.

Handles storage of documents, tenancy data, ACLs, and sync checkpoints
using SQLAlchemy async with asyncpg.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models import Document, DocumentMetadata, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.db.models import (
    Base,
    DocumentModel,
    MemoryNamespaceModel,
    SyncCheckpointModel,
)
from khora.db.schema import sync_enum_values
from khora.storage.backends.base import PaginatedResult
from khora.storage.backends.mixins import AsyncSessionMixin, retry_on_deadlock

if TYPE_CHECKING:
    pass


class PostgreSQLBackend(AsyncSessionMixin):
    """PostgreSQL backend for relational data.

    Handles all relational data operations including multi-tenancy
    hierarchy, documents, and sync checkpoints.
    """

    def __init__(
        self,
        database_url: str,
        *,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = False,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the PostgreSQL backend.

        Args:
            database_url: PostgreSQL connection URL
            echo: Enable SQL echo logging
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_pre_ping: Enable pool pre-ping to detect stale connections
            engine: Optional shared engine (skip dispose on disconnect)
        """
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

        self._database_url = database_url
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_pre_ping = pool_pre_ping
        self._engine: AsyncEngine | None = engine
        self._engine_shared: bool = engine is not None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Establish connection to the database."""
        if self._session_factory is not None:
            return

        logger.info("Connecting to PostgreSQL...")
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
        logger.info("Connected to PostgreSQL")

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            logger.info("Disconnecting from PostgreSQL...")
            if not self._engine_shared:
                await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Disconnected from PostgreSQL")

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        if self._engine is None or self._session_factory is None:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False

    async def create_tables(self) -> None:
        """Create all database tables (for testing/development)."""
        if self._engine is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await sync_enum_values(self._engine)

    # =========================================================================
    # Namespace operations
    # =========================================================================

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the active version's row id.

        Idempotent: if the input is already an internal row-level id,
        returns it as-is. This allows callers to safely pass either
        the stable namespace_id or the internal id.

        Called on every public API entry (remember, recall, forget, etc.).
        Hits the indexed ``namespace_id`` column so it's sub-millisecond,
        but still one extra query per call. If namespace versioning is
        removed in the future this resolution layer can be dropped entirely,
        collapsing to a single UUID used everywhere.

        Args:
            namespace_id: The stable namespace identifier (shared across versions)
                or an internal row-level id

        Returns:
            The row-level id of the active version

        Raises:
            ValueError: If no active version exists for the given namespace_id
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(MemoryNamespaceModel.id).where(
                    or_(
                        MemoryNamespaceModel.namespace_id == namespace_id,
                        MemoryNamespaceModel.id == namespace_id,
                    ),
                    MemoryNamespaceModel.is_active == True,  # noqa: E712
                )
            )
            row_id = result.scalar_one_or_none()
            if row_id is not None:
                return row_id

            raise ValueError(f"No active namespace found for namespace_id or id={namespace_id}")

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        async with self._get_session() as session:
            model = MemoryNamespaceModel(
                id=namespace.id,
                namespace_id=namespace.namespace_id,
                tenancy_mode=namespace.tenancy_mode,
                version=namespace.version,
                is_active=namespace.is_active,
                config_overrides=namespace.config_overrides,
                sync_checkpoints=namespace.sync_checkpoints,
                metadata_=namespace.metadata,
                created_at=namespace.created_at,
                updated_at=namespace.updated_at,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._namespace_model_to_domain(model)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(MemoryNamespaceModel).where(MemoryNamespaceModel.id == namespace_id))
            model = result.scalar_one_or_none()
            return self._namespace_model_to_domain(model) if model else None

    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination.

        Args:
            active_only: If True, only return active namespaces (default)
            limit: Maximum namespaces to return
            offset: Offset for pagination

        Returns:
            PaginatedResult with namespace items and total count
        """
        async with self._get_session() as session:
            base_filter = MemoryNamespaceModel.is_active == True if active_only else True  # noqa: E712
            count_query = select(func.count(MemoryNamespaceModel.id)).where(base_filter)
            total = (await session.execute(count_query)).scalar_one()

            query = (
                select(MemoryNamespaceModel)
                .where(base_filter)
                .order_by(MemoryNamespaceModel.id)
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            items = [self._namespace_model_to_domain(m) for m in result.scalars().all()]
            return PaginatedResult(items=items, total=total, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        async with self._get_session() as session:
            await session.execute(
                update(MemoryNamespaceModel)
                .where(MemoryNamespaceModel.id == namespace.id)
                .values(
                    version=namespace.version,
                    is_active=namespace.is_active,
                    config_overrides=namespace.config_overrides,
                    sync_checkpoints=namespace.sync_checkpoints,
                    metadata_=namespace.metadata,
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()
            return namespace

    def _namespace_model_to_domain(self, model: MemoryNamespaceModel) -> MemoryNamespace:
        """Convert MemoryNamespaceModel to domain MemoryNamespace."""
        return MemoryNamespace(
            id=model.id,
            namespace_id=model.namespace_id,
            tenancy_mode=TenancyMode(model.tenancy_mode) if isinstance(model.tenancy_mode, str) else model.tenancy_mode,
            version=model.version,
            is_active=model.is_active,
            config_overrides=model.config_overrides,
            sync_checkpoints=model.sync_checkpoints,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        If previous_version is provided, increments its version number and links to it.
        The previous version is marked as inactive.

        Args:
            previous_version: The previous version to supersede (if any)

        Returns:
            New namespace version
        """
        from uuid import uuid4

        new_version = 1

        if previous_version:
            new_version = previous_version.version + 1
            # Deactivate the old version
            await self.deactivate_namespace(previous_version.id)

        # Create new namespace with incremented version
        # New versions inherit the parent's namespace_id (stable identifier across versions)
        namespace = MemoryNamespace(
            id=uuid4(),
            namespace_id=previous_version.namespace_id if previous_version else uuid4(),
            version=new_version,
            is_active=True,
            config_overrides=previous_version.config_overrides if previous_version else {},
            metadata=previous_version.metadata if previous_version else {},
        )

        return await self.create_namespace(namespace)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        async with self._get_session() as session:
            await session.execute(
                update(MemoryNamespaceModel)
                .where(MemoryNamespaceModel.id == namespace_id)
                .values(is_active=False, updated_at=datetime.now(UTC))
            )
            await session.commit()
            logger.info(f"Deactivated namespace {namespace_id}")

    # =========================================================================
    # Document operations
    # =========================================================================

    @retry_on_deadlock
    async def create_document(self, document: Document, *, session: AsyncSession | None = None) -> Document:
        """Create a new document."""
        if session is not None:
            return await self._create_document_with(session, document)
        async with self._get_session() as own_session:
            return await self._create_document_with(own_session, document, commit=True)

    async def _create_document_with(
        self, session: AsyncSession, document: Document, *, commit: bool = False
    ) -> Document:
        model = DocumentModel(
            id=document.id,
            namespace_id=document.namespace_id,
            content=document.content,
            status=document.status,
            source=document.metadata.source,
            source_type=document.metadata.source_type,
            content_type=document.metadata.content_type,
            title=document.metadata.title,
            author=document.metadata.author,
            language=document.metadata.language,
            checksum=document.metadata.checksum,
            size_bytes=document.metadata.size_bytes,
            metadata_=document.metadata.custom,
            chunk_count=document.chunk_count,
            entity_count=document.entity_count,
            error_message=document.error_message,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
        )
        session.add(model)
        if commit:
            await session.commit()
        else:
            await session.flush()
        await session.refresh(model)
        return self._document_model_to_domain(model)

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(DocumentModel).where(DocumentModel.id == document_id))
            model = result.scalar_one_or_none()
            return self._document_model_to_domain(model) if model else None

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace."""
        async with self._get_session() as session:
            query = select(DocumentModel).where(DocumentModel.namespace_id == namespace_id)
            if status:
                query = query.where(DocumentModel.status == status)
            query = query.limit(limit).offset(offset).order_by(DocumentModel.created_at.desc())
            result = await session.execute(query)
            return [self._document_model_to_domain(m) for m in result.scalars().all()]

    @retry_on_deadlock
    async def update_document(self, document: Document, *, session: AsyncSession | None = None) -> Document:
        """Update a document."""
        if session is not None:
            return await self._update_document_with(session, document)
        async with self._get_session() as own_session:
            return await self._update_document_with(own_session, document, commit=True)

    async def _update_document_with(
        self, session: AsyncSession, document: Document, *, commit: bool = False
    ) -> Document:
        await session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document.id)
            .values(
                content=document.content,
                status=document.status,
                source=document.metadata.source,
                source_type=document.metadata.source_type,
                content_type=document.metadata.content_type,
                title=document.metadata.title,
                author=document.metadata.author,
                language=document.metadata.language,
                checksum=document.metadata.checksum,
                size_bytes=document.metadata.size_bytes,
                metadata_=document.metadata.custom,
                chunk_count=document.chunk_count,
                entity_count=document.entity_count,
                error_message=document.error_message,
                updated_at=datetime.now(UTC),
                processed_at=document.processed_at,
            )
        )
        if commit:
            await session.commit()
        return document

    @retry_on_deadlock
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document."""
        async with self._get_session() as session:
            result = await session.execute(select(DocumentModel).where(DocumentModel.id == document_id))
            model = result.scalar_one_or_none()
            if model:
                await session.delete(model)
                await session.commit()
                return True
            return False

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by its content checksum (for deduplication).

        Returns the first matching document if multiple exist with the same checksum.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id, DocumentModel.checksum == checksum
                )
            )
            model = result.scalars().first()
            return self._document_model_to_domain(model) if model else None

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to Document object
        """
        if not document_ids:
            return {}

        async with self._get_session() as session:
            result = await session.execute(select(DocumentModel).where(DocumentModel.id.in_(document_ids)))
            models = result.scalars().all()
            return {m.id: self._document_model_to_domain(m) for m in models}

    async def get_documents_by_checksums(self, namespace_id: UUID, checksums: list[str]) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not checksums:
            return {}

        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.checksum.in_(checksums),
                )
            )
            models = result.scalars().all()
            return {m.checksum: self._document_model_to_domain(m) for m in models}

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution.

        Uses a column-limited SELECT to avoid reading content, processing
        stats, and other heavy/mutable columns.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        if not document_ids:
            return {}

        async with self._get_session() as session:
            result = await session.execute(
                select(
                    DocumentModel.id,
                    DocumentModel.title,
                    DocumentModel.source,
                    DocumentModel.source_type,
                    DocumentModel.created_at,
                    DocumentModel.source_timestamp,
                ).where(DocumentModel.id.in_(document_ids))
            )
            rows = result.all()
            return {
                row.id: DocumentSource(
                    id=row.id,
                    title=row.title,
                    source=row.source,
                    source_type=row.source_type,
                    created_at=row.created_at,
                    source_timestamp=row.source_timestamp,
                )
                for row in rows
            }

    def _document_model_to_domain(self, model: DocumentModel) -> Document:
        """Convert DocumentModel to domain Document."""
        return Document(
            id=model.id,
            namespace_id=model.namespace_id,
            content=model.content,
            status=DocumentStatus(model.status) if isinstance(model.status, str) else model.status,
            metadata=DocumentMetadata(
                source=model.source,
                source_type=model.source_type,
                content_type=model.content_type,
                title=model.title,
                author=model.author,
                language=model.language,
                checksum=model.checksum,
                size_bytes=model.size_bytes,
                custom=model.metadata_,
            ),
            chunk_count=model.chunk_count,
            entity_count=model.entity_count,
            error_message=model.error_message,
            created_at=model.created_at,
            updated_at=model.updated_at,
            processed_at=model.processed_at,
        )

    # =========================================================================
    # Sync checkpoint operations
    # =========================================================================

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        async with self._get_session() as session:
            result = await session.execute(
                select(SyncCheckpointModel).where(
                    SyncCheckpointModel.namespace_id == namespace_id, SyncCheckpointModel.source == source
                )
            )
            model = result.scalar_one_or_none()
            return model.checkpoint if model else None

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        async with self._get_session() as session:
            result = await session.execute(
                select(SyncCheckpointModel).where(
                    SyncCheckpointModel.namespace_id == namespace_id, SyncCheckpointModel.source == source
                )
            )
            model = result.scalar_one_or_none()
            if model:
                model.checkpoint = checkpoint
                model.updated_at = datetime.now(UTC)
            else:
                model = SyncCheckpointModel(
                    namespace_id=namespace_id,
                    source=source,
                    checkpoint=checkpoint,
                )
                session.add(model)
            await session.commit()
