"""SQLite relational adapter for the embedded sqlite_lance backend.

Implements :class:`~khora.storage.backends.base.RelationalBackendProtocol`
on top of SQLAlchemy + aiosqlite, reusing the same ORM models as the
PostgreSQL backend. The adapter opens its own :class:`AsyncEngine` against
the SQLite database file owned by :class:`EmbeddedStorageHandle` so that
:meth:`StorageCoordinator.transaction` can share a single session across
SQL-based adapters via the public ``_session_factory`` attribute.

Schema creation is handled by Alembic (dialect-gated migrations);
this adapter assumes tables already exist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
    delete,
    event,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.types import Uuid

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.core.models.recall import DocumentProjection
from khora.db.models import DocumentModel, MemoryNamespaceModel, SyncCheckpointModel
from khora.engines.chronicle.compression import MemoryFact
from khora.engines.chronicle.events import ChronicleEvent
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


# SQLite-specific Table mirrors for the Chronicle schema.
#
# The ORM models in khora.db.models declare Postgres-only types on these
# tables — ``ChronicleEventModel.embedding: Vector(1536)`` and
# ``MemoryFactModel.source_chunk_ids: ARRAY(UUID)``. Migration 024 already
# gates these out of the SQLite schema (no embedding column, source_chunk_ids
# becomes ``sa.JSON``). But the ORM column types' bind processors run in
# Python before SQLAlchemy emits SQL — so an ORM-style insert routes the
# values through the Postgres processor and fails. Defining standalone
# ``Table`` objects with SQLite-shaped column types lets Core inserts/selects
# use the right processors. We rely on ``Uuid`` (SQLAlchemy's dialect-aware
# UUID type), which stores as TEXT(32) on SQLite, matching migration 024's
# ``sa.String(36)`` close enough for round-trips.
_chronicle_metadata = MetaData()
_chronicle_events_sqlite = Table(
    "chronicle_events",
    _chronicle_metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column("namespace_id", Uuid(as_uuid=True), nullable=False),
    Column("chunk_id", Uuid(as_uuid=True), nullable=False),
    Column("subject", String(512), nullable=False),
    Column("verb", String(255), nullable=False),
    Column("object", String(512), nullable=True),
    Column("observation_date", DateTime(timezone=True), nullable=False),
    Column("referenced_date", DateTime(timezone=True), nullable=True),
    Column("relative_offset", String(255), nullable=True),
    Column("confidence", Float, nullable=False),
    Column("source_text", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
_memory_facts_sqlite = Table(
    "memory_facts",
    _chronicle_metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column("namespace_id", Uuid(as_uuid=True), nullable=False),
    Column("subject", String(512), nullable=False),
    Column("predicate", String(255), nullable=False),
    Column("object", String(512), nullable=False),
    Column("fact_text", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("is_active", Boolean, nullable=False),
    Column("superseded_by", Uuid(as_uuid=True), nullable=True),
    Column("source_chunk_ids", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
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
            source=document.source,
            source_type=document.source_type,
            source_name=document.source_name,
            source_url=document.source_url,
            content_type=document.content_type,
            title=document.title,
            author=document.author,
            language=document.language,
            checksum=document.checksum,
            size_bytes=document.size_bytes,
            metadata_=document.metadata,
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
            source_timestamp=document.source_timestamp,
            session_id=document.session_id,
        )
        session.add(model)
        if commit:
            await session.commit()
        else:
            await session.flush()
        await session.refresh(model)
        return self._document_model_to_domain(model)

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Returns ``None`` if the document does not exist OR belongs to a
        different namespace. Prevents cross-tenant document access by id
        (IDOR — IGR-221).
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.id == document_id,
                    DocumentModel.namespace_id == namespace_id,
                )
            )
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
                source=document.source,
                source_type=document.source_type,
                source_name=document.source_name,
                source_url=document.source_url,
                content_type=document.content_type,
                title=document.title,
                author=document.author,
                language=document.language,
                checksum=document.checksum,
                size_bytes=document.size_bytes,
                metadata_=document.metadata,
                chunk_count=document.chunk_count,
                entity_count=document.entity_count,
                relationship_count=document.relationship_count,
                error_message=document.error_message,
                extraction_config_hash=document.extraction_config_hash,
                extraction_params=document.extraction_params,
                external_id=document.external_id,
                updated_at=datetime.now(UTC),
                processed_at=document.processed_at,
                source_timestamp=document.source_timestamp,
                session_id=document.session_id,
            )
        )
        if commit:
            await session.commit()
        return document

    async def delete_document(self, document_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a document, scoped to ``namespace_id`` (IGR-226).

        Uses a core ``DELETE`` rather than ORM cascade because the
        ORM ``ChunkModel`` has Postgres-only columns (pgvector
        ``embedding``, ``TSVECTOR`` ``content_tsv``) that don't exist
        in the SQLite schema — loading chunks for cascade would raise
        "no such column" on SQLite.  The SQLite ``chunks`` FK has
        ``ON DELETE CASCADE`` so cleanup happens at the DB level.
        """
        async with self._get_session() as session:
            result = await session.execute(
                delete(DocumentModel).where(
                    DocumentModel.id == document_id,
                    DocumentModel.namespace_id == namespace_id,
                )
            )
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

    async def get_document_by_external_id(self, external_id: str | None, *, namespace_id: UUID) -> Document | None:
        """Get a document by (namespace_id, external_id).

        Status is NOT filtered so FAILED rows can self-heal on the next
        successful replace.
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

    async def get_documents_batch(self, document_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query, scoped to ``namespace_id``.

        Documents belonging to any other namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IGR-221).
        """
        if not document_ids:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.id.in_(document_ids),
                    DocumentModel.namespace_id == namespace_id,
                )
            )
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

    async def get_documents_by_external_ids(
        self, external_ids: list[str], *, namespace_id: UUID
    ) -> dict[str, Document]:
        """Batch lookup by ``(namespace_id, external_id)``. Status-agnostic."""
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

    async def get_document_sources_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution,
        scoped to ``namespace_id``.

        Documents in other namespaces are silently dropped from the
        result (IGR-221).
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
                ).where(
                    DocumentModel.id.in_(document_ids),
                    DocumentModel.namespace_id == namespace_id,
                )
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

    async def get_document_projections_batch(
        self,
        document_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> dict[UUID, DocumentProjection]:
        if not document_ids:
            return {}
        async with self._get_session() as session:
            result = await session.execute(
                select(
                    DocumentModel.id,
                    DocumentModel.created_at,
                    DocumentModel.source_type,
                    DocumentModel.title,
                    DocumentModel.external_id,
                    DocumentModel.source,
                    DocumentModel.source_name,
                    DocumentModel.source_url,
                    DocumentModel.content_type,
                    DocumentModel.source_timestamp,
                    DocumentModel.metadata_,
                ).where(
                    DocumentModel.id.in_(document_ids),
                    DocumentModel.namespace_id == namespace_id,
                )
            )
            rows = result.all()
            return {
                row.id: DocumentProjection(
                    id=row.id,
                    created_at=row.created_at or datetime.now(UTC),
                    source_type=row.source_type or "library",
                    title=row.title or None,
                    external_id=row.external_id or None,
                    source=row.source or None,
                    source_name=row.source_name or None,
                    source_url=row.source_url or None,
                    content_type=row.content_type or None,
                    source_timestamp=row.source_timestamp,
                    metadata=dict(row.metadata_ or {}),
                )
                for row in rows
            }

    def _document_model_to_domain(self, model: DocumentModel) -> Document:
        def _none_if_empty(v: str | None) -> str | None:
            return v if v else None

        return Document(
            id=model.id,
            namespace_id=model.namespace_id,
            content=model.content,
            status=DocumentStatus(model.status) if isinstance(model.status, str) else model.status,
            title=_none_if_empty(model.title),
            source=_none_if_empty(model.source),
            source_type=model.source_type or "library",
            source_name=_none_if_empty(getattr(model, "source_name", None)),
            source_url=_none_if_empty(getattr(model, "source_url", None)),
            content_type=_none_if_empty(model.content_type),
            author=_none_if_empty(model.author),
            language=_none_if_empty(model.language),
            checksum=_none_if_empty(model.checksum),
            size_bytes=model.size_bytes,
            metadata=dict(model.metadata_) if model.metadata_ else {},
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
            source_timestamp=getattr(model, "source_timestamp", None),
            session_id=getattr(model, "session_id", None),
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

    # =========================================================================
    # Chronicle engine: events + facts
    #
    # pgvector implements these on its vector adapter because it owns the
    # `embedding` column on chronicle_events. The sqlite_lance vector adapter
    # is LanceDB and has no SQL session, so the coordinator falls back here.
    #
    # Migration 024 already created chronicle_events + memory_facts on SQLite
    # (sans `embedding` column, dialect-gated). We use SQLAlchemy Core inserts/
    # selects with explicit column lists rather than ORM ``session.add`` /
    # ``select(Model)`` so the SQL emitted matches the SQLite schema — the ORM
    # models declare a Postgres-only ``embedding Vector(1536)`` column and a
    # ``source_chunk_ids ARRAY(UUID)`` column that would otherwise blow up
    # against the SQLite tables.
    # =========================================================================

    async def write_events(
        self,
        events: list[ChronicleEvent],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert chronicle_events rows; returns inserted IDs in input order."""
        if not events:
            return []
        now = datetime.now(UTC)
        ids: list[UUID] = []
        rows: list[dict[str, object]] = []
        for ev in events:
            ev_id = ev.id or uuid4()
            ids.append(ev_id)
            rows.append(
                {
                    "id": ev_id,
                    "namespace_id": namespace_id,
                    "chunk_id": ev.chunk_id,
                    "subject": ev.subject,
                    "verb": ev.verb,
                    "object": ev.object or None,
                    "observation_date": ev.observation_date or now,
                    "referenced_date": ev.referenced_date,
                    "relative_offset": ev.relative_offset or None,
                    "confidence": float(ev.confidence),
                    "source_text": ev.source_text or "",
                    "created_at": now,
                }
            )
        async with self._get_session() as session:
            await session.execute(insert(_chronicle_events_sqlite), rows)
            await session.commit()
        return ids

    async def write_facts(
        self,
        facts: list[MemoryFact],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert memory_facts rows; returns inserted IDs in input order."""
        if not facts:
            return []
        now = datetime.now(UTC)
        ids: list[UUID] = []
        rows: list[dict[str, object]] = []
        for f in facts:
            fact_id = f.id or uuid4()
            ids.append(fact_id)
            # SQLite stores source_chunk_ids as JSON; the JSON column type
            # serialises a Python list of UUID-strings on bind.
            chunk_ids_json = [str(cid) for cid in (f.source_chunk_ids or [])]
            rows.append(
                {
                    "id": fact_id,
                    "namespace_id": namespace_id,
                    "subject": f.subject or "",
                    "predicate": f.predicate or "",
                    "object": f.object_ or "",
                    "fact_text": f.fact_text or "",
                    "confidence": float(f.confidence),
                    "is_active": bool(f.is_active),
                    "superseded_by": f.superseded_by,
                    "source_chunk_ids": chunk_ids_json,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        async with self._get_session() as session:
            await session.execute(insert(_memory_facts_sqlite), rows)
            await session.commit()
        return ids

    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[ChronicleEvent]:
        """Query chronicle_events filtered by subject and referenced_date range."""
        t = _chronicle_events_sqlite
        stmt = select(t).where(t.c.namespace_id == namespace_id)
        if subject is not None:
            stmt = stmt.where(t.c.subject == subject)
        if since is not None:
            stmt = stmt.where(t.c.referenced_date >= since)
        if until is not None:
            stmt = stmt.where(t.c.referenced_date <= until)
        stmt = stmt.order_by(t.c.referenced_date.desc().nullslast()).limit(limit)
        async with self._get_session() as session:
            result = await session.execute(stmt)
            return [
                ChronicleEvent(
                    id=row.id,
                    chunk_id=row.chunk_id,
                    namespace_id=row.namespace_id,
                    subject=row.subject,
                    verb=row.verb,
                    object=row.object or "",
                    observation_date=row.observation_date,
                    referenced_date=row.referenced_date,
                    relative_offset=row.relative_offset or "",
                    confidence=float(row.confidence),
                    source_text=row.source_text or "",
                )
                for row in result.all()
            ]

    async def query_active_facts_for_subject(
        self,
        namespace_id: UUID,
        subject: str,
    ) -> list[MemoryFact]:
        """Return all active (not superseded) memory facts for a subject."""
        t = _memory_facts_sqlite
        stmt = (
            select(t)
            .where(
                t.c.namespace_id == namespace_id,
                t.c.subject == subject,
                t.c.is_active.is_(True),
            )
            .order_by(t.c.created_at.desc())
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            return [_row_to_memory_fact(row) for row in result.all()]

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID, *, namespace_id: UUID) -> None:
        """Mark a fact inactive and link it to its replacement.

        Scoped to ``namespace_id`` (IGR-226) — no-op when the fact belongs
        to a different namespace.
        """
        t = _memory_facts_sqlite
        async with self._get_session() as session:
            await session.execute(
                update(t)
                .where(
                    t.c.id == fact_id,
                    t.c.namespace_id == namespace_id,
                )
                .values(is_active=False, superseded_by=superseded_by, updated_at=datetime.now(UTC))
            )
            await session.commit()


def _row_to_memory_fact(row: object) -> MemoryFact:
    """Build a MemoryFact dataclass from a SQLAlchemy Core ``Row``.

    Tolerates ``source_chunk_ids`` arriving as either a JSON-decoded list of
    UUID strings (SQLite) or already-parsed list of ``UUID`` objects (some
    JSON deserialisers).
    """
    raw_chunk_ids = getattr(row, "source_chunk_ids", None) or []
    chunk_ids: list[UUID] = []
    for cid in raw_chunk_ids:
        if isinstance(cid, UUID):
            chunk_ids.append(cid)
        else:
            try:
                chunk_ids.append(UUID(str(cid)))
            except (ValueError, TypeError):
                continue
    return MemoryFact(
        id=row.id,
        namespace_id=row.namespace_id,
        subject=row.subject or "",
        predicate=row.predicate or "",
        object_=row.object or "",
        fact_text=row.fact_text or "",
        confidence=float(row.confidence),
        is_active=bool(row.is_active),
        superseded_by=row.superseded_by,
        source_chunk_ids=chunk_ids,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
