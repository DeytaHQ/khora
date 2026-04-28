"""SurrealDB relational adapter for Khora.

Implements RelationalBackendProtocol using SurrealQL, delegating connection
lifecycle to SurrealDBConnection.  Record IDs follow the SurrealDB convention:
``table:⟨uuid⟩``.  All UUIDs are converted to ``str`` at the boundary and
parsed back on read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models import Document, DocumentMetadata, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.storage.backends.base import PaginatedResult
from khora.storage.backends.surrealdb._helpers import (
    _parse_dt,
    _parse_uuid,
    _record_id,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SurrealDBRelationalAdapter:
    """Relational backend backed by SurrealDB.

    Fulfils :class:`~khora.storage.backends.base.RelationalBackendProtocol`
    without importing SQLAlchemy.  The adapter delegates all I/O to a
    :class:`SurrealDBConnection` instance.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBRelationalAdapter:
        """Create an adapter from a configuration dictionary.

        Expected keys mirror :class:`SurrealDBConnection.__init__` kwargs:
        ``mode``, ``path``, ``url``, ``namespace``, ``database``, ``user``,
        ``password``.  All are optional and fall back to SurrealDBConnection
        defaults.
        """
        conn = SurrealDBConnection(
            mode=config.get("mode", "memory"),
            path=config.get("path"),
            url=config.get("url"),
            namespace=config.get("namespace", "khora"),
            database=config.get("database", "default"),
            user=config.get("user", "root"),
            password=config.get("password", "root"),
        )
        return cls(connection=conn)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def create_tables(self) -> None:
        """Create SurrealDB tables and indexes (idempotent).

        Schema is also auto-initialized on connect(), so this is
        safe to call multiple times.
        """
        from .schema import initialize_schema

        await initialize_schema(self._conn)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to SurrealDB."""
        await self._conn.connect()

    async def disconnect(self) -> None:
        """Close the SurrealDB connection."""
        await self._conn.disconnect()

    async def is_healthy(self) -> bool:
        """Delegate health check to the connection."""
        return await self._conn.is_healthy()

    # ------------------------------------------------------------------
    # Namespace operations
    # ------------------------------------------------------------------

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a namespace identifier to the ID used by chunk/entity records.

        In SurrealDB, chunks and entities store namespace references using the
        stable ``namespace_id`` (not the row-level ``id``).  This method
        validates that an active namespace exists and returns the stable
        ``namespace_id`` so that search filters match stored data.
        """
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT id, namespace_id FROM memory_namespace "
            "WHERE (namespace_id = $ns OR id = $rid) AND is_active = true "
            "LIMIT 1",
            {"ns": ns_str, "rid": _record_id("memory_namespace", namespace_id)},
        )
        if row is not None:
            # Return the stable namespace_id — this is what chunks/entities
            # use as their namespace record reference.
            return UUID(row["namespace_id"])
        raise ValueError(f"No active namespace found for namespace_id or id={namespace_id}")

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace record."""
        rid = _record_id("memory_namespace", namespace.id)
        now_iso = namespace.created_at
        upd_iso = namespace.updated_at

        row = await self._conn.query_one(
            "CREATE $rid SET "
            "namespace_id = $namespace_id, "
            "tenancy_mode = $tenancy_mode, "
            "version = $version, "
            "is_active = $is_active, "
            "config_overrides = $config_overrides, "
            "sync_checkpoints = $sync_checkpoints, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "updated_at = $updated_at",
            {
                "rid": rid,
                "namespace_id": str(namespace.namespace_id),
                "tenancy_mode": (
                    namespace.tenancy_mode.value
                    if isinstance(namespace.tenancy_mode, TenancyMode)
                    else namespace.tenancy_mode
                ),
                "version": namespace.version,
                "is_active": namespace.is_active,
                "config_overrides": namespace.config_overrides or {},
                "sync_checkpoints": namespace.sync_checkpoints or {},
                "metadata_": namespace.metadata or {},
                "created_at": now_iso,
                "updated_at": upd_iso,
            },
        )
        if row is None:
            raise RuntimeError(f"Failed to create namespace {namespace.id}")
        return self._row_to_namespace(row)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by its row-level id."""
        rid = _record_id("memory_namespace", namespace_id)
        row = await self._conn.query_one(
            "SELECT * FROM $rid",
            {"rid": rid},
        )
        if row is None:
            return None
        return self._row_to_namespace(row)

    async def list_namespaces(
        self,
        *,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        where = "WHERE is_active = true" if active_only else ""

        count_row = await self._conn.query_one(
            f"SELECT count() AS total FROM memory_namespace {where} GROUP ALL",  # noqa: S608
        )
        total = count_row["total"] if count_row else 0

        rows = await self._conn.query(
            f"SELECT * FROM memory_namespace {where} ORDER BY id ASC LIMIT $lim START $off",  # noqa: S608
            {"lim": limit, "off": offset},
        )
        items = [self._row_to_namespace(r) for r in rows]
        return PaginatedResult(items=items, total=total, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update mutable namespace fields."""
        rid = _record_id("memory_namespace", namespace.id)
        await self._conn.execute(
            "UPDATE $rid SET "
            "version = $version, "
            "is_active = $is_active, "
            "config_overrides = $config_overrides, "
            "sync_checkpoints = $sync_checkpoints, "
            "metadata_ = $metadata_, "
            "updated_at = $updated_at",
            {
                "rid": rid,
                "version": namespace.version,
                "is_active": namespace.is_active,
                "config_overrides": namespace.config_overrides or {},
                "sync_checkpoints": namespace.sync_checkpoints or {},
                "metadata_": namespace.metadata or {},
                "updated_at": datetime.now(UTC),
            },
        )
        return namespace

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version, deactivating the previous one."""
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
        """Mark a namespace version as inactive."""
        rid = _record_id("memory_namespace", namespace_id)
        await self._conn.execute(
            "UPDATE $rid SET is_active = false, updated_at = $updated_at",
            {"rid": rid, "updated_at": datetime.now(UTC)},
        )
        logger.info(f"Deactivated namespace {namespace_id}")

    # -- namespace row → domain model --

    def _row_to_namespace(self, row: dict[str, Any]) -> MemoryNamespace:
        tenancy_raw = row.get("tenancy_mode", "shared")
        return MemoryNamespace(
            id=_parse_uuid(row["id"]),
            namespace_id=UUID(row["namespace_id"]) if isinstance(row["namespace_id"], str) else row["namespace_id"],
            tenancy_mode=TenancyMode(tenancy_raw) if isinstance(tenancy_raw, str) else tenancy_raw,
            version=row.get("version", 1),
            is_active=row.get("is_active", True),
            config_overrides=row.get("config_overrides") or {},
            sync_checkpoints=row.get("sync_checkpoints") or {},
            metadata=row.get("metadata_") or {},
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def create_document(self, document: Document) -> Document:
        """Create a new document record."""
        rid = _record_id("document", document.id)
        row = await self._conn.query_one(
            "CREATE $rid SET "
            "namespace_id = $namespace_id, "
            "content = $content, "
            "status = $status, "
            "source = $source, "
            "source_type = $source_type, "
            "content_type = $content_type, "
            "title = $title, "
            "author = $author, "
            "language = $language, "
            "checksum = $checksum, "
            "size_bytes = $size_bytes, "
            "metadata_ = $metadata_, "
            "chunk_count = $chunk_count, "
            "entity_count = $entity_count, "
            "relationship_count = $relationship_count, "
            "error_message = $error_message, "
            "extraction_config_hash = $extraction_config_hash, "
            "extraction_params = $extraction_params, "
            "external_id = $external_id, "
            "created_at = $created_at, "
            "updated_at = $updated_at, "
            "processed_at = $processed_at, "
            "source_timestamp = $source_timestamp",
            {
                "rid": rid,
                "namespace_id": str(document.namespace_id),
                "content": document.content,
                "status": document.status.value if isinstance(document.status, DocumentStatus) else document.status,
                "source": document.metadata.source,
                "source_type": document.metadata.source_type,
                "content_type": document.metadata.content_type,
                "title": document.metadata.title,
                "author": document.metadata.author,
                "language": document.metadata.language,
                "checksum": document.metadata.checksum,
                "size_bytes": document.metadata.size_bytes,
                "metadata_": document.metadata.custom or {},
                "chunk_count": document.chunk_count,
                "entity_count": document.entity_count,
                "relationship_count": document.relationship_count,
                "error_message": document.error_message,
                "extraction_config_hash": document.extraction_config_hash,
                "extraction_params": document.extraction_params,
                "external_id": document.external_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "processed_at": document.processed_at,
                "source_timestamp": document.source_timestamp,
            },
        )
        if row is None:
            raise RuntimeError(f"Failed to create document {document.id}")
        return self._row_to_document(row)

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        rid = _record_id("document", document_id)
        row = await self._conn.query_one(
            "SELECT * FROM $rid",
            {"rid": rid},
        )
        if row is None:
            return None
        return self._row_to_document(row)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace, newest first."""
        ns_str = str(namespace_id)
        conditions = ["namespace_id = $ns"]
        params: dict = {"ns": ns_str, "lim": limit, "off": offset}
        if status:
            conditions.append("status = $status")
            params["status"] = status
        if updated_before is not None:
            conditions.append("updated_at < $updated_before")
            params["updated_before"] = updated_before.isoformat()
        where = " AND ".join(conditions)
        rows = await self._conn.query(
            f"SELECT * FROM document WHERE {where} ORDER BY created_at DESC LIMIT $lim START $off",  # noqa: S608
            params,
        )
        return [self._row_to_document(r) for r in rows]

    async def update_document(self, document: Document) -> Document:
        """Update a document's mutable fields."""
        rid = _record_id("document", document.id)
        await self._conn.execute(
            "UPDATE $rid SET "
            "content = $content, "
            "status = $status, "
            "source = $source, "
            "source_type = $source_type, "
            "content_type = $content_type, "
            "title = $title, "
            "author = $author, "
            "language = $language, "
            "checksum = $checksum, "
            "size_bytes = $size_bytes, "
            "metadata_ = $metadata_, "
            "chunk_count = $chunk_count, "
            "entity_count = $entity_count, "
            "relationship_count = $relationship_count, "
            "error_message = $error_message, "
            "extraction_config_hash = $extraction_config_hash, "
            "extraction_params = $extraction_params, "
            "external_id = $external_id, "
            "updated_at = $updated_at, "
            "processed_at = $processed_at, "
            "source_timestamp = $source_timestamp",
            {
                "rid": rid,
                "content": document.content,
                "status": document.status.value if isinstance(document.status, DocumentStatus) else document.status,
                "source": document.metadata.source,
                "source_type": document.metadata.source_type,
                "content_type": document.metadata.content_type,
                "title": document.metadata.title,
                "author": document.metadata.author,
                "language": document.metadata.language,
                "checksum": document.metadata.checksum,
                "size_bytes": document.metadata.size_bytes,
                "metadata_": document.metadata.custom or {},
                "chunk_count": document.chunk_count,
                "entity_count": document.entity_count,
                "relationship_count": document.relationship_count,
                "error_message": document.error_message,
                "extraction_config_hash": document.extraction_config_hash,
                "extraction_params": document.extraction_params,
                "external_id": document.external_id,
                "updated_at": datetime.now(UTC),
                "processed_at": document.processed_at,
                "source_timestamp": document.source_timestamp,
            },
        )
        return document

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document, returning True if it existed."""
        rid = _record_id("document", document_id)
        deleted = await self._conn.query("DELETE $rid RETURN BEFORE", {"rid": rid})
        return bool(deleted)

    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT count() AS cnt FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        return (row["cnt"] or 0) if row else 0

    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        """Get the most recent document creation timestamp in a namespace."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT math::max(created_at) AS latest FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        return row["latest"] if row else None

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Get document count and last activity timestamp in a single query."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT count() AS cnt, math::max(created_at) AS latest FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        if not row:
            return 0, None
        return (row["cnt"] or 0), row["latest"]

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by content checksum within a namespace.

        FAILED documents are excluded to allow re-ingestion of previously failed content.
        """
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT * FROM document WHERE namespace_id = $ns AND checksum = $checksum AND status != 'failed' LIMIT 1",
            {"ns": ns_str, "checksum": checksum},
        )
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_document_by_external_id(self, namespace_id: UUID, external_id: str | None) -> Document | None:
        """Get a document by (namespace_id, external_id) — ADR-056 dispatch.

        Status is NOT filtered so FAILED rows can self-heal on the next
        successful replace (ADR-056 §Decision #8).
        """
        if external_id is None:
            return None
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT * FROM document WHERE namespace_id = $ns AND external_id = $external_id LIMIT 1",
            {"ns": ns_str, "external_id": external_id},
        )
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_documents_by_checksums(self, namespace_id: UUID, checksums: list[str]) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.
        FAILED documents are excluded to allow re-ingestion of previously failed content.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not checksums:
            return {}
        ns_str = str(namespace_id)
        rows = await self._conn.query(
            "SELECT * FROM document WHERE namespace_id = $ns AND checksum IN $checksums AND status != 'failed'",
            {"ns": ns_str, "checksums": checksums},
        )
        result: dict[str, Document] = {}
        for r in rows:
            doc = self._row_to_document(r)
            cs = r.get("checksum", "")
            if cs:
                result[cs] = doc
        return result

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query."""
        if not document_ids:
            return {}
        id_strs = [_record_id("document", uid) for uid in document_ids]
        rows = await self._conn.query(
            "SELECT * FROM document WHERE id IN $ids",
            {"ids": id_strs},
        )
        return {_parse_uuid(r["id"]): self._row_to_document(r) for r in rows}

    async def get_documents_by_external_ids(self, namespace_id: UUID, external_ids: list[str]) -> dict[str, Document]:
        """Batch lookup by ``(namespace_id, external_id)`` — ADR-056. Status-agnostic."""
        filtered = [e for e in external_ids if e]
        if not filtered:
            return {}
        ns_str = str(namespace_id)
        rows = await self._conn.query(
            "SELECT * FROM document WHERE namespace_id = $ns AND external_id IN $external_ids",
            {"ns": ns_str, "external_ids": filtered},
        )
        result: dict[str, Document] = {}
        for r in rows:
            ext = r.get("external_id")
            if ext:
                result[ext] = self._row_to_document(r)
        return result

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution."""
        if not document_ids:
            return {}
        id_strs = [_record_id("document", uid) for uid in document_ids]
        rows = await self._conn.query(
            "SELECT id, title, source, source_type, created_at, source_timestamp FROM document WHERE id IN $ids",
            {"ids": id_strs},
        )
        result: dict[UUID, DocumentSource] = {}
        for r in rows:
            uid = _parse_uuid(r["id"])
            result[uid] = DocumentSource(
                id=uid,
                title=r.get("title", ""),
                source=r.get("source", ""),
                source_type=r.get("source_type", ""),
                created_at=_parse_dt(r.get("created_at")),
                source_timestamp=_parse_dt(r.get("source_timestamp")),
            )
        return result

    # -- document row → domain model --

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        status_raw = row.get("status", "pending")
        return Document(
            id=_parse_uuid(row["id"]),
            namespace_id=UUID(row["namespace_id"]) if isinstance(row["namespace_id"], str) else row["namespace_id"],
            content=row.get("content", ""),
            status=DocumentStatus(status_raw) if isinstance(status_raw, str) else status_raw,
            metadata=DocumentMetadata(
                source=row.get("source", ""),
                source_type=row.get("source_type", ""),
                content_type=row.get("content_type", ""),
                title=row.get("title", ""),
                author=row.get("author", ""),
                language=row.get("language", "en"),
                checksum=row.get("checksum", ""),
                size_bytes=row.get("size_bytes", 0),
                custom=row.get("metadata_") or {},
            ),
            chunk_count=row.get("chunk_count", 0),
            entity_count=row.get("entity_count", 0),
            relationship_count=row.get("relationship_count", 0),
            error_message=row.get("error_message"),
            extraction_config_hash=row.get("extraction_config_hash"),
            extraction_params=row.get("extraction_params"),
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
            processed_at=_parse_dt(row.get("processed_at")),
            source_timestamp=_parse_dt(row.get("source_timestamp")),
            external_id=row.get("external_id"),
        )

    # ------------------------------------------------------------------
    # Sync checkpoint operations
    # ------------------------------------------------------------------

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT checkpoint FROM sync_checkpoint WHERE namespace_id = $ns AND source = $source LIMIT 1",
            {"ns": ns_str, "source": source},
        )
        if row is None:
            return None
        return row.get("checkpoint")

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Upsert a sync checkpoint for a namespace+source pair.

        Uses SurrealDB's UPSERT with a deterministic record ID derived
        from namespace and source so that repeated calls overwrite rather
        than duplicate.
        """
        ns_str = str(namespace_id)
        # Deterministic record ID avoids duplicates
        upsert_id = f"sync_checkpoint:⟨{ns_str}_{source}⟩"
        await self._conn.execute(
            "UPSERT $rid SET namespace_id = $ns, source = $source, checkpoint = $checkpoint, updated_at = $updated_at",
            {
                "rid": upsert_id,
                "ns": ns_str,
                "source": source,
                "checkpoint": checkpoint,
                "updated_at": datetime.now(UTC),
            },
        )

    # ------------------------------------------------------------------
    # SQLAlchemy compatibility shim
    # ------------------------------------------------------------------

    def _get_session(self) -> None:
        """No-op — SurrealDB does not use SQLAlchemy sessions."""
        return None
