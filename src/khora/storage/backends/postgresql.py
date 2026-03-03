"""PostgreSQL backend for relational data storage.

Handles storage of documents, tenancy data, ACLs, and sync checkpoints
using SQLAlchemy async with asyncpg.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models import Document, DocumentMetadata, MemoryNamespace, Organization, TenancyMode, Workspace
from khora.core.models.document import DocumentStatus
from khora.db.models import (
    Base,
    DocumentModel,
    MemoryNamespaceModel,
    OrganizationModel,
    SyncCheckpointModel,
    WorkspaceModel,
)
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
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the PostgreSQL backend.

        Args:
            database_url: PostgreSQL connection URL
            echo: Enable SQL echo logging
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
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

    # =========================================================================
    # Organization operations
    # =========================================================================

    async def create_organization(self, org: Organization) -> Organization:
        """Create a new organization."""
        async with self._get_session() as session:
            model = OrganizationModel(
                id=org.id,
                name=org.name,
                slug=org.slug,
                tenancy_mode=org.tenancy_mode,
                metadata_=org.metadata,
                created_at=org.created_at,
                updated_at=org.updated_at,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._org_model_to_domain(model)

    async def get_organization(self, org_id: UUID) -> Organization | None:
        """Get an organization by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(OrganizationModel).where(OrganizationModel.id == org_id))
            model = result.scalar_one_or_none()
            return self._org_model_to_domain(model) if model else None

    async def get_organization_by_slug(self, slug: str) -> Organization | None:
        """Get an organization by slug."""
        async with self._get_session() as session:
            result = await session.execute(select(OrganizationModel).where(OrganizationModel.slug == slug))
            model = result.scalar_one_or_none()
            return self._org_model_to_domain(model) if model else None

    def _org_model_to_domain(self, model: OrganizationModel) -> Organization:
        """Convert OrganizationModel to domain Organization."""
        return Organization(
            id=model.id,
            name=model.name,
            slug=model.slug,
            tenancy_mode=TenancyMode(model.tenancy_mode) if isinstance(model.tenancy_mode, str) else model.tenancy_mode,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # =========================================================================
    # Workspace operations
    # =========================================================================

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create a new workspace."""
        async with self._get_session() as session:
            model = WorkspaceModel(
                id=workspace.id,
                organization_id=workspace.organization_id,
                name=workspace.name,
                slug=workspace.slug,
                description=workspace.description,
                metadata_=workspace.metadata,
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._workspace_model_to_domain(model)

    async def get_workspace(self, workspace_id: UUID) -> Workspace | None:
        """Get a workspace by ID."""
        async with self._get_session() as session:
            result = await session.execute(select(WorkspaceModel).where(WorkspaceModel.id == workspace_id))
            model = result.scalar_one_or_none()
            return self._workspace_model_to_domain(model) if model else None

    async def list_workspaces(self, organization_id: UUID) -> list[Workspace]:
        """List all workspaces in an organization."""
        async with self._get_session() as session:
            result = await session.execute(
                select(WorkspaceModel).where(WorkspaceModel.organization_id == organization_id)
            )
            return [self._workspace_model_to_domain(m) for m in result.scalars().all()]

    def _workspace_model_to_domain(self, model: WorkspaceModel) -> Workspace:
        """Convert WorkspaceModel to domain Workspace."""
        return Workspace(
            id=model.id,
            organization_id=model.organization_id,
            name=model.name,
            slug=model.slug,
            description=model.description,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # =========================================================================
    # Namespace operations
    # =========================================================================

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        async with self._get_session() as session:
            model = MemoryNamespaceModel(
                id=namespace.id,
                workspace_id=namespace.workspace_id,
                name=namespace.name,
                slug=namespace.slug,
                description=namespace.description,
                version=namespace.version,
                is_active=namespace.is_active,
                previous_version_id=namespace.previous_version_id,
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

    async def get_namespace_by_slug(
        self, workspace_id: UUID, slug: str, *, active_only: bool = True
    ) -> MemoryNamespace | None:
        """Get a namespace by workspace ID and slug.

        Args:
            workspace_id: Workspace ID
            slug: Namespace slug
            active_only: If True, only return active namespace (default)

        Returns:
            MemoryNamespace or None if not found
        """
        async with self._get_session() as session:
            query = select(MemoryNamespaceModel).where(
                MemoryNamespaceModel.workspace_id == workspace_id,
                MemoryNamespaceModel.slug == slug,
            )
            if active_only:
                query = query.where(MemoryNamespaceModel.is_active == True)  # noqa: E712
            result = await session.execute(query)
            model = result.scalar_one_or_none()
            return self._namespace_model_to_domain(model) if model else None

    async def list_namespaces(self, workspace_id: UUID, *, active_only: bool = True) -> list[MemoryNamespace]:
        """List all namespaces in a workspace.

        Args:
            workspace_id: Workspace ID
            active_only: If True, only return active namespaces (default)

        Returns:
            List of MemoryNamespace objects
        """
        async with self._get_session() as session:
            query = select(MemoryNamespaceModel).where(MemoryNamespaceModel.workspace_id == workspace_id)
            if active_only:
                query = query.where(MemoryNamespaceModel.is_active == True)  # noqa: E712
            result = await session.execute(query)
            return [self._namespace_model_to_domain(m) for m in result.scalars().all()]

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        async with self._get_session() as session:
            await session.execute(
                update(MemoryNamespaceModel)
                .where(MemoryNamespaceModel.id == namespace.id)
                .values(
                    name=namespace.name,
                    slug=namespace.slug,
                    description=namespace.description,
                    version=namespace.version,
                    is_active=namespace.is_active,
                    previous_version_id=namespace.previous_version_id,
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
            workspace_id=model.workspace_id,
            name=model.name,
            slug=model.slug,
            description=model.description,
            version=model.version,
            is_active=model.is_active,
            previous_version_id=model.previous_version_id,
            config_overrides=model.config_overrides,
            sync_checkpoints=model.sync_checkpoints,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create_namespace_version(
        self,
        workspace_id: UUID,
        slug: str,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        If previous_version is provided, increments its version number and links to it.
        The previous version is marked as inactive.

        Args:
            workspace_id: Workspace ID
            slug: Namespace slug
            previous_version: The previous version to supersede (if any)

        Returns:
            New namespace version
        """
        from uuid import uuid4

        new_version = 1
        previous_id = None

        if previous_version:
            new_version = previous_version.version + 1
            previous_id = previous_version.id
            # Deactivate the old version
            await self.deactivate_namespace(previous_version.id)

        # Create new namespace with incremented version
        namespace = MemoryNamespace(
            id=uuid4(),
            workspace_id=workspace_id,
            name=previous_version.name if previous_version else f"Namespace {slug}",
            slug=slug,
            description=previous_version.description if previous_version else "",
            version=new_version,
            is_active=True,
            previous_version_id=previous_id,
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

    async def get_document_by_source(self, namespace_id: UUID, source: str) -> Document | None:
        """Get a document by its source (for update detection).

        Returns None if source is empty or no match found.
        """
        if not source:
            return None

        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel)
                .where(DocumentModel.namespace_id == namespace_id, DocumentModel.source == source)
                .order_by(DocumentModel.updated_at.desc())
            )
            model = result.scalars().first()
            return self._document_model_to_domain(model) if model else None

    async def get_documents_by_sources(self, namespace_id: UUID, sources: list[str]) -> dict[str, Document]:
        """Fetch documents by source in a single query.

        Used for batch update detection. Filters out empty sources.

        Args:
            namespace_id: Namespace to search in
            sources: List of source identifiers to look up

        Returns:
            Dictionary mapping source to Document (only for existing documents)
        """
        non_empty = [s for s in sources if s]
        if not non_empty:
            return {}

        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel)
                .where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.source.in_(non_empty),
                )
                .order_by(DocumentModel.updated_at.desc())
            )
            models = result.scalars().all()
            result_dict: dict[str, Document] = {}
            for m in models:
                if m.source not in result_dict:  # keep first (newest due to DESC)
                    result_dict[m.source] = self._document_model_to_domain(m)
            return result_dict

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
