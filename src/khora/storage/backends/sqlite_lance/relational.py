"""SQLite relational adapter for the embedded sqlite_lance backend.

Implements :class:`~khora.storage.backends.base.RelationalBackendProtocol`
on top of SQLAlchemy + aiosqlite, reusing the same ORM models as the
PostgreSQL backend. The adapter opens its own :class:`AsyncEngine` against
the SQLite database file owned by :class:`EmbeddedStorageHandle` so that
:meth:`StorageCoordinator.transaction` can share a single session across
SQL-based adapters via the public ``_session_factory`` attribute.

Schema creation is handled by Alembic (dialect-gated migrations from
DYT-2727); this adapter assumes tables already exist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import delete, event, func, or_, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.core.models import Document, DocumentMetadata, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.db.models import DocumentModel, MemoryNamespaceModel, SyncCheckpointModel
from khora.storage.backends.base import PaginatedResult
from khora.storage.backends.mixins import AsyncSessionMixin

from ..._log_safe import _safe_url_for_log

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


# SQLite pragmas applied on every new aiosqlite connection the SQLAlchemy
# pool opens. Mirrors the pragmas ``EmbeddedStorageHandle`` applies to its
# own aiosqlite connection — the two connections open the same DB file so
# WAL is required for them to coexist without lock contention.
_SQLITE_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("mmap_size", "268435456"),
    ("cache_size", "-64000"),
    ("temp_store", "MEMORY"),
    ("busy_timeout", "5000"),
)


def _build_sqlite_url(db_path: str) -> str:
    """Build a ``sqlite+aiosqlite://`` URL from the handle's ``db_path``.

    ``EmbeddedStorageHandle`` accepts either a filesystem path or the
    special ``:memory:`` sentinel; both forms translate directly to a
    valid aiosqlite URL.
    """
    if db_path == ":memory:" or db_path.startswith("file::memory:"):
        return "sqlite+aiosqlite:///:memory:"
    return f"sqlite+aiosqlite:///{db_path}"


def _register_pragmas(engine: AsyncEngine) -> None:
    """Apply ``_SQLITE_PRAGMAS`` to every new pool connection.

    SQLAlchemy's ``connect`` event fires on the sync DBAPI connection,
    so we attach the listener to ``engine.sync_engine``.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            for pragma, value in _SQLITE_PRAGMAS:
                cursor.execute(f"PRAGMA {pragma}={value}")
        finally:
            cursor.close()


class SQLiteLanceRelationalAdapter(AsyncSessionMixin):
    """Relational backend backed by SQLite + SQLAlchemy.

    Implements :class:`RelationalBackendProtocol`.  Shares the SQLite
    database file with :class:`EmbeddedStorageHandle` but manages its
    own :class:`AsyncEngine` so the coordinator can reach a session
    factory via ``self._session_factory``.
    """

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the SQLAlchemy async engine.

        Idempotent — a second call is a no-op.  Also connects the shared
        handle so aiosqlite-based sibling adapters are ready.
        """
        await self._handle.connect()

        if self._session_factory is not None:
            return

        url = _build_sqlite_url(self._handle.config.db_path)
        logger.info("Opening SQLAlchemy async engine for {url}", url=_safe_url_for_log(url))
        self._engine = create_async_engine(url, future=True)
        _register_pragmas(self._engine)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def disconnect(self) -> None:
        """Dispose the engine and session factory."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        # The shared handle is owned by the factory; don't close it here.

    async def is_healthy(self) -> bool:
        """Run a trivial SELECT to confirm the engine is usable."""
        if self._session_factory is None:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as exc:
            logger.debug(f"SQLite relational health check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Namespace operations
    # ------------------------------------------------------------------

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the active version's row id."""
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
        async with self._get_session() as session:
            result = await session.execute(select(MemoryNamespaceModel).where(MemoryNamespaceModel.id == namespace_id))
            model = result.scalar_one_or_none()
            return self._namespace_model_to_domain(model) if model else None

    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
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

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        Previous version (if any) is deactivated and the new row inherits
        its ``namespace_id``.  Config/metadata are carried over verbatim.
        """
        new_version = 1
        if previous_version:
            new_version = previous_version.version + 1
            await self.deactivate_namespace(previous_version.id)

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
        async with self._get_session() as session:
            await session.execute(
                update(MemoryNamespaceModel)
                .where(MemoryNamespaceModel.id == namespace_id)
                .values(is_active=False, updated_at=datetime.now(UTC))
            )
            await session.commit()

    def _namespace_model_to_domain(self, model: MemoryNamespaceModel) -> MemoryNamespace:
        return MemoryNamespace(
            id=model.id,
            namespace_id=model.namespace_id,
            tenancy_mode=TenancyMode(model.tenancy_mode) if isinstance(model.tenancy_mode, str) else model.tenancy_mode,
            version=model.version,
            is_active=model.is_active,
            config_overrides=model.config_overrides or {},
            sync_checkpoints=model.sync_checkpoints or {},
            metadata=model.metadata_ or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def create_document(self, document: Document, *, session: AsyncSession | None = None) -> Document:
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
            relationship_count=document.relationship_count,
            error_message=document.error_message,
            extraction_config_hash=document.extraction_config_hash,
            extraction_params=document.extraction_params,
            external_id=document.external_id,
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
        async with self._get_session() as session:
            result = await session.execute(select(DocumentModel).where(DocumentModel.id == document_id))
            model = result.scalar_one_or_none()
            return self._document_model_to_domain(model) if model else None

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        async with self._get_session() as session:
            query = select(DocumentModel).where(DocumentModel.namespace_id == namespace_id)
            if status:
                query = query.where(DocumentModel.status == status)
            if updated_before is not None:
                query = query.where(DocumentModel.updated_at < updated_before)
            query = query.limit(limit).offset(offset).order_by(DocumentModel.created_at.desc())
            result = await session.execute(query)
            return [self._document_model_to_domain(m) for m in result.scalars().all()]

    async def update_document(self, document: Document, *, session: AsyncSession | None = None) -> Document:
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
                relationship_count=document.relationship_count,
                error_message=document.error_message,
                extraction_config_hash=document.extraction_config_hash,
                extraction_params=document.extraction_params,
                external_id=document.external_id,
                updated_at=datetime.now(UTC),
                processed_at=document.processed_at,
            )
        )
        if commit:
            await session.commit()
        return document

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document.

        Uses a core ``DELETE`` rather than ORM cascade because the
        ORM ``ChunkModel`` has Postgres-only columns (pgvector
        ``embedding``, ``TSVECTOR`` ``content_tsv``) that don't exist
        in the SQLite schema — loading chunks for cascade would raise
        "no such column" on SQLite.  The SQLite ``chunks`` FK has
        ``ON DELETE CASCADE`` so cleanup happens at the DB level.
        """
        async with self._get_session() as session:
            result = await session.execute(delete(DocumentModel).where(DocumentModel.id == document_id))
            await session.commit()
            rowcount: int = getattr(result, "rowcount", 0)
            return rowcount > 0

    async def count_documents(self, namespace_id: UUID) -> int:
        async with self._get_session() as session:
            result = await session.execute(
                select(func.count(DocumentModel.id)).where(DocumentModel.namespace_id == namespace_id)
            )
            return result.scalar_one()

    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        async with self._get_session() as session:
            result = await session.execute(
                select(func.max(DocumentModel.created_at)).where(DocumentModel.namespace_id == namespace_id)
            )
            return result.scalar_one_or_none()

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        async with self._get_session() as session:
            result = await session.execute(
                select(
                    func.count(DocumentModel.id),
                    func.max(DocumentModel.created_at),
                ).where(DocumentModel.namespace_id == namespace_id)
            )
            row = result.one()
            return row[0], row[1]

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Return the first non-failed document matching ``checksum``.

        Failed documents are excluded so re-ingestion of previously failed
        content is allowed, matching the PostgreSQL backend semantics.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.checksum == checksum,
                    DocumentModel.status != DocumentStatus.FAILED,
                )
            )
            model = result.scalars().first()
            return self._document_model_to_domain(model) if model else None

    async def get_document_by_external_id(self, namespace_id: UUID, external_id: str | None) -> Document | None:
        """Get a document by (namespace_id, external_id) — ADR-056 dispatch.

        Status is NOT filtered so FAILED rows can self-heal on the next
        successful replace (ADR-056 §Decision #8).
        """
        if external_id is None:
            return None
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.external_id == external_id,
                )
            )
            model = result.scalars().first()
            return self._document_model_to_domain(model) if model else None

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        if not document_ids:
            return {}
        async with self._get_session() as session:
            result = await session.execute(select(DocumentModel).where(DocumentModel.id.in_(document_ids)))
            models = result.scalars().all()
            return {m.id: self._document_model_to_domain(m) for m in models}

    async def get_documents_by_checksums(self, namespace_id: UUID, checksums: list[str]) -> dict[str, Document]:
        if not checksums:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.checksum.in_(checksums),
                    DocumentModel.status != DocumentStatus.FAILED,
                )
            )
            models = result.scalars().all()
            return {m.checksum: self._document_model_to_domain(m) for m in models}

    async def get_documents_by_external_ids(self, namespace_id: UUID, external_ids: list[str]) -> dict[str, Document]:
        """Batch lookup by ``(namespace_id, external_id)`` — ADR-056. Status-agnostic."""
        filtered = [e for e in external_ids if e]
        if not filtered:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.namespace_id == namespace_id,
                    DocumentModel.external_id.in_(filtered),
                )
            )
            models = result.scalars().all()
            return {m.external_id: self._document_model_to_domain(m) for m in models if m.external_id}

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
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
                custom=model.metadata_ or {},
            ),
            chunk_count=model.chunk_count,
            entity_count=model.entity_count,
            relationship_count=model.relationship_count,
            error_message=model.error_message,
            extraction_config_hash=model.extraction_config_hash,
            extraction_params=model.extraction_params,
            external_id=model.external_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            processed_at=model.processed_at,
        )

    # ------------------------------------------------------------------
    # Sync checkpoint operations
    # ------------------------------------------------------------------

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        async with self._get_session() as session:
            result = await session.execute(
                select(SyncCheckpointModel).where(
                    SyncCheckpointModel.namespace_id == namespace_id,
                    SyncCheckpointModel.source == source,
                )
            )
            model = result.scalar_one_or_none()
            return model.checkpoint if model else None

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """UPSERT a sync checkpoint.

        SQLite's ORM ``merge``/``INSERT OR REPLACE`` paths would clobber
        non-key columns; use an explicit select-then-write round-trip to
        preserve row identity and ``created_at``.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(SyncCheckpointModel).where(
                    SyncCheckpointModel.namespace_id == namespace_id,
                    SyncCheckpointModel.source == source,
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
