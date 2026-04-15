"""SQLite backend for zero-infrastructure mode.

Provides both relational and vector storage using a single SQLite database
via aiosqlite. Vector search uses brute-force cosine similarity from
khora._accel (Rust/NumPy/pure-Python cascade). Full-text search uses FTS5.

No ORM — raw SQL throughout. The PostgreSQL ORM models use PG-specific types
(JSONB, ARRAY, Vector, TSVECTOR) that don't translate to SQLite cleanly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, MemoryNamespace
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.core.models.entity import Entity
from khora.core.models.tenancy import TenancyMode
from khora.storage.backends.base import PaginatedResult

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS memory_namespaces (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    tenancy_mode TEXT DEFAULT 'shared',
    name TEXT DEFAULT '',
    description TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    is_active INTEGER DEFAULT 1,
    config_overrides TEXT DEFAULT '{}',
    sync_checkpoints TEXT DEFAULT '{}',
    metadata_ TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ns_namespace_id ON memory_namespaces(namespace_id);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    content TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    source TEXT DEFAULT '',
    source_type TEXT DEFAULT '',
    content_type TEXT DEFAULT '',
    title TEXT DEFAULT '',
    author TEXT DEFAULT '',
    language TEXT DEFAULT 'en',
    checksum TEXT DEFAULT '',
    size_bytes INTEGER DEFAULT 0,
    metadata_ TEXT DEFAULT '{}',
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    error_message TEXT,
    extraction_config_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    processed_at TEXT,
    source_timestamp TEXT,
    external_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_docs_ns ON documents(namespace_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_checksum
    ON documents(namespace_id, checksum) WHERE checksum != '';
CREATE INDEX IF NOT EXISTS idx_docs_ns_external_id
    ON documents(namespace_id, external_id) WHERE external_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    metadata_ TEXT DEFAULT '{}',
    embedding TEXT,
    embedding_model TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    source_timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_ns ON chunks(namespace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_index ON chunks(document_id, chunk_index);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'CONCEPT',
    description TEXT DEFAULT '',
    attributes TEXT DEFAULT '{}',
    source_tool TEXT DEFAULT '',
    source_document_ids TEXT DEFAULT '[]',
    source_chunk_ids TEXT DEFAULT '[]',
    mention_count INTEGER DEFAULT 1,
    embedding TEXT,
    embedding_model TEXT DEFAULT '',
    valid_from TEXT,
    valid_until TEXT,
    confidence REAL DEFAULT 1.0,
    metadata_ TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_ns ON entities(namespace_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique
    ON entities(namespace_id, name, entity_type);

CREATE TABLE IF NOT EXISTS sync_checkpoints (
    namespace_id TEXT NOT NULL,
    source TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace_id, source)
);
"""

# FTS5 virtual table — must be created separately (can't use IF NOT EXISTS
# in some SQLite builds, so we guard with a try/except).
_FTS5_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    namespace_id UNINDEXED
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    return datetime.fromisoformat(val)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _json_loads(text: str | None) -> Any:
    if not text:
        return {}
    return json.loads(text)


def _uuid_list_dumps(uuids: list[UUID]) -> str:
    return json.dumps([str(u) for u in uuids])


def _uuid_list_loads(text: str | None) -> list[UUID]:
    if not text:
        return []
    return [UUID(s) for s in json.loads(text)]


# ---------------------------------------------------------------------------
# SQLiteRelationalBackend
# ---------------------------------------------------------------------------


class SQLiteRelationalBackend:
    """SQLite relational backend for zero-infrastructure mode.

    Implements :class:`RelationalBackendProtocol` using raw SQL via aiosqlite.
    UUIDs stored as TEXT, JSON fields as TEXT, datetimes as ISO-8601 TEXT.
    """

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        self._conn: Any = None  # aiosqlite.Connection

    @classmethod
    def from_config(cls, config: Any) -> SQLiteRelationalBackend:
        url = getattr(config, "url", "") or ""
        path = url.replace("sqlite:///", "").replace("sqlite://", "") or ":memory:"
        return cls(database_path=path)

    async def connect(self) -> None:
        import aiosqlite

        self._conn = await aiosqlite.connect(self._database_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        logger.debug("SQLite relational backend connected: {}", self._database_path)

    async def _create_schema(self) -> None:
        for statement in _SCHEMA_SQL.split(";"):
            stmt = statement.strip()
            if stmt:
                await self._conn.execute(stmt)
        await self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def is_healthy(self) -> bool:
        if not self._conn:
            return False
        try:
            await self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Namespace operations
    # ------------------------------------------------------------------

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        cursor = await self._conn.execute(
            "SELECT id FROM memory_namespaces WHERE (namespace_id = ? OR id = ?) AND is_active = 1 LIMIT 1",
            (str(namespace_id), str(namespace_id)),
        )
        row = await cursor.fetchone()
        if row is None:
            msg = f"No active namespace found for namespace_id or id={namespace_id}"
            raise ValueError(msg)
        return UUID(row["id"])

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        now = _now_iso()
        await self._conn.execute(
            "INSERT INTO memory_namespaces "
            "(id, namespace_id, tenancy_mode, version, is_active, config_overrides, "
            "sync_checkpoints, metadata_, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(namespace.id),
                str(namespace.namespace_id),
                namespace.tenancy_mode.value
                if isinstance(namespace.tenancy_mode, TenancyMode)
                else namespace.tenancy_mode,
                namespace.version,
                1 if namespace.is_active else 0,
                _json_dumps(namespace.config_overrides),
                _json_dumps(namespace.sync_checkpoints),
                _json_dumps(namespace.metadata),
                _dt_to_str(namespace.created_at) or now,
                _dt_to_str(namespace.updated_at) or now,
            ),
        )
        await self._conn.commit()
        return namespace

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        cursor = await self._conn.execute(
            "SELECT * FROM memory_namespaces WHERE id = ? LIMIT 1",
            (str(namespace_id),),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_namespace(row)

    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        where = "WHERE is_active = 1" if active_only else ""
        cursor = await self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM memory_namespaces {where}"  # noqa: S608
        )
        total_row = await cursor.fetchone()
        total = total_row["cnt"] if total_row else 0

        cursor = await self._conn.execute(
            f"SELECT * FROM memory_namespaces {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # noqa: S608
            (limit, offset),
        )
        rows = await cursor.fetchall()
        items = [self._row_to_namespace(r) for r in rows]
        return PaginatedResult(items=items, total=total, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        now = _now_iso()
        await self._conn.execute(
            "UPDATE memory_namespaces SET "
            "tenancy_mode = ?, version = ?, is_active = ?, config_overrides = ?, "
            "sync_checkpoints = ?, metadata_ = ?, updated_at = ? "
            "WHERE id = ?",
            (
                namespace.tenancy_mode.value
                if isinstance(namespace.tenancy_mode, TenancyMode)
                else namespace.tenancy_mode,
                namespace.version,
                1 if namespace.is_active else 0,
                _json_dumps(namespace.config_overrides),
                _json_dumps(namespace.sync_checkpoints),
                _json_dumps(namespace.metadata),
                now,
                str(namespace.id),
            ),
        )
        await self._conn.commit()
        namespace.updated_at = datetime.fromisoformat(now)
        return namespace

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        now = datetime.now(UTC)
        ns_id = previous_version.namespace_id if previous_version else uuid4()
        version = (previous_version.version + 1) if previous_version else 1

        if previous_version:
            await self._conn.execute(
                "UPDATE memory_namespaces SET is_active = 0, updated_at = ? WHERE id = ?",
                (_dt_to_str(now), str(previous_version.id)),
            )

        new_ns = MemoryNamespace(
            id=uuid4(),
            namespace_id=ns_id,
            version=version,
            is_active=True,
            config_overrides=previous_version.config_overrides if previous_version else {},
            created_at=now,
            updated_at=now,
        )
        return await self.create_namespace(new_ns)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        await self._conn.execute(
            "UPDATE memory_namespaces SET is_active = 0, updated_at = ? WHERE id = ?",
            (_now_iso(), str(namespace_id)),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def create_document(self, document: Document) -> Document:
        now = _now_iso()
        status = document.status.value if isinstance(document.status, DocumentStatus) else document.status
        await self._conn.execute(
            "INSERT INTO documents "
            "(id, namespace_id, content, status, source, source_type, content_type, "
            "title, author, language, checksum, size_bytes, metadata_, "
            "chunk_count, entity_count, error_message, extraction_config_hash, "
            "created_at, updated_at, processed_at, source_timestamp, external_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(document.id),
                str(document.namespace_id),
                document.content,
                status,
                document.metadata.source,
                document.metadata.source_type,
                document.metadata.content_type,
                document.metadata.title,
                document.metadata.author,
                document.metadata.language,
                document.metadata.checksum,
                document.metadata.size_bytes,
                _json_dumps(document.metadata.custom),
                document.chunk_count,
                document.entity_count,
                document.error_message,
                document.extraction_config_hash,
                _dt_to_str(document.created_at) or now,
                _dt_to_str(document.updated_at) or now,
                _dt_to_str(document.processed_at),
                _dt_to_str(document.source_timestamp),
                document.external_id,
            ),
        )
        await self._conn.commit()
        return document

    async def get_document(self, document_id: UUID) -> Document | None:
        cursor = await self._conn.execute("SELECT * FROM documents WHERE id = ?", (str(document_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        ns = str(namespace_id)
        if status:
            cursor = await self._conn.execute(
                "SELECT * FROM documents WHERE namespace_id = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (ns, status, limit, offset),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM documents WHERE namespace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (ns, limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_document(r) for r in rows]

    async def update_document(self, document: Document) -> Document:
        now = _now_iso()
        status = document.status.value if isinstance(document.status, DocumentStatus) else document.status
        await self._conn.execute(
            "UPDATE documents SET "
            "content = ?, status = ?, source = ?, source_type = ?, content_type = ?, "
            "title = ?, author = ?, language = ?, checksum = ?, size_bytes = ?, "
            "metadata_ = ?, chunk_count = ?, entity_count = ?, error_message = ?, "
            "extraction_config_hash = ?, external_id = ?, updated_at = ?, processed_at = ?, source_timestamp = ? "
            "WHERE id = ?",
            (
                document.content,
                status,
                document.metadata.source,
                document.metadata.source_type,
                document.metadata.content_type,
                document.metadata.title,
                document.metadata.author,
                document.metadata.language,
                document.metadata.checksum,
                document.metadata.size_bytes,
                _json_dumps(document.metadata.custom),
                document.chunk_count,
                document.entity_count,
                document.error_message,
                document.extraction_config_hash,
                document.external_id,
                now,
                _dt_to_str(document.processed_at),
                _dt_to_str(document.source_timestamp),
                str(document.id),
            ),
        )
        await self._conn.commit()
        document.updated_at = datetime.fromisoformat(now)
        return document

    async def delete_document(self, document_id: UUID) -> bool:
        cursor = await self._conn.execute("DELETE FROM documents WHERE id = ?", (str(document_id),))
        await self._conn.commit()
        return cursor.rowcount > 0

    async def count_documents(self, namespace_id: UUID) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) as cnt FROM documents WHERE namespace_id = ?",
            (str(namespace_id),),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        cursor = await self._conn.execute(
            "SELECT MAX(created_at) as last_at FROM documents WHERE namespace_id = ?",
            (str(namespace_id),),
        )
        row = await cursor.fetchone()
        if row and row["last_at"]:
            return _parse_dt(row["last_at"])
        return None

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) as cnt, MAX(created_at) as last_at FROM documents WHERE namespace_id = ?",
            (str(namespace_id),),
        )
        row = await cursor.fetchone()
        if not row:
            return 0, None
        return row["cnt"], _parse_dt(row["last_at"])

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by content checksum. FAILED documents are excluded."""
        # Note: SQLite (dev/test only) relies on full table scan; no partial index optimization
        cursor = await self._conn.execute(
            "SELECT * FROM documents WHERE namespace_id = ? AND checksum = ? AND status != 'failed' LIMIT 1",
            (str(namespace_id), checksum),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        cursor = await self._conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            [str(d) for d in document_ids],
        )
        rows = await cursor.fetchall()
        result: dict[UUID, Document] = {}
        for r in rows:
            doc = self._row_to_document(r)
            result[doc.id] = doc
        return result

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        cursor = await self._conn.execute(
            f"SELECT id, title, source, source_type, created_at, source_timestamp "  # noqa: S608
            f"FROM documents WHERE id IN ({placeholders})",
            [str(d) for d in document_ids],
        )
        rows = await cursor.fetchall()
        result: dict[UUID, DocumentSource] = {}
        for r in rows:
            uid = UUID(r["id"])
            result[uid] = DocumentSource(
                id=uid,
                title=r["title"] or "",
                source=r["source"] or "",
                source_type=r["source_type"] or "",
                created_at=_parse_dt(r["created_at"]),
                source_timestamp=_parse_dt(r["source_timestamp"]),
            )
        return result

    # ------------------------------------------------------------------
    # Sync checkpoints
    # ------------------------------------------------------------------

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        cursor = await self._conn.execute(
            "SELECT checkpoint FROM sync_checkpoints WHERE namespace_id = ? AND source = ?",
            (str(namespace_id), source),
        )
        row = await cursor.fetchone()
        return row["checkpoint"] if row else None

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        await self._conn.execute(
            "INSERT INTO sync_checkpoints (namespace_id, source, checkpoint, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(namespace_id, source) DO UPDATE SET checkpoint = ?, updated_at = ?",
            (str(namespace_id), source, checkpoint, _now_iso(), checkpoint, _now_iso()),
        )
        await self._conn.commit()

    def _get_session(self) -> Any:
        return None  # no ORM session for SQLite

    # ------------------------------------------------------------------
    # Row → domain model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_namespace(row: Any) -> MemoryNamespace:
        tenancy_raw = row["tenancy_mode"] if row["tenancy_mode"] else "shared"
        return MemoryNamespace(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            tenancy_mode=TenancyMode(tenancy_raw),
            version=row["version"] or 1,
            is_active=bool(row["is_active"]),
            config_overrides=_json_loads(row["config_overrides"]),
            sync_checkpoints=_json_loads(row["sync_checkpoints"]),
            metadata=_json_loads(row["metadata_"]),
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
        )

    @staticmethod
    def _row_to_document(row: Any) -> Document:
        status_raw = row["status"] or "pending"
        return Document(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            content=row["content"] or "",
            status=DocumentStatus(status_raw),
            metadata=DocumentMetadata(
                source=row["source"] or "",
                source_type=row["source_type"] or "",
                content_type=row["content_type"] or "",
                title=row["title"] or "",
                author=row["author"] or "",
                language=row["language"] or "en",
                checksum=row["checksum"] or "",
                size_bytes=row["size_bytes"] or 0,
                custom=_json_loads(row["metadata_"]),
            ),
            chunk_count=row["chunk_count"] or 0,
            entity_count=row["entity_count"] or 0,
            error_message=row["error_message"],
            extraction_config_hash=row["extraction_config_hash"],
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
            processed_at=_parse_dt(row["processed_at"]),
            source_timestamp=_parse_dt(row["source_timestamp"]),
            external_id=row["external_id"],
        )


# ---------------------------------------------------------------------------
# SQLiteVectorBackend
# ---------------------------------------------------------------------------


class SQLiteVectorBackend:
    """SQLite vector backend using brute-force cosine similarity.

    Implements :class:`VectorBackendProtocol` using raw SQL via aiosqlite.
    Embeddings are stored as JSON text (list of floats).  Vector search is
    done in Python via ``khora._accel.batch_cosine_similarity``.
    Full-text search uses FTS5.
    """

    def __init__(self, database_path: str, *, embedding_dimension: int = 1536) -> None:
        self._database_path = database_path
        self._embedding_dim = embedding_dimension
        self._conn: Any = None  # aiosqlite.Connection

    @classmethod
    def from_config(cls, config: Any) -> SQLiteVectorBackend:
        url = getattr(config, "url", "") or ""
        path = url.replace("sqlite:///", "").replace("sqlite://", "") or ":memory:"
        dim = getattr(config, "embedding_dimension", 1536)
        return cls(database_path=path, embedding_dimension=dim)

    async def connect(self) -> None:
        import aiosqlite

        self._conn = await aiosqlite.connect(self._database_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        logger.debug("SQLite vector backend connected: {}", self._database_path)

    async def _create_schema(self) -> None:
        # Main tables (idempotent — shared with relational backend)
        for statement in _SCHEMA_SQL.split(";"):
            stmt = statement.strip()
            if stmt:
                await self._conn.execute(stmt)
        # FTS5 virtual table
        try:
            await self._conn.executescript(_FTS5_SQL)
        except Exception:
            logger.debug("FTS5 table already exists or unsupported")
        await self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def is_healthy(self) -> bool:
        if not self._conn:
            return False
        try:
            await self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        now = _now_iso()
        embedding_json = json.dumps(chunk.embedding) if chunk.embedding else None
        await self._conn.execute(
            "INSERT INTO chunks "
            "(id, namespace_id, document_id, content, chunk_index, start_char, "
            "end_char, token_count, metadata_, embedding, embedding_model, "
            "created_at, source_timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(chunk.id),
                str(chunk.namespace_id),
                str(chunk.document_id),
                chunk.content,
                chunk.metadata.chunk_index,
                chunk.metadata.start_char,
                chunk.metadata.end_char,
                chunk.metadata.token_count,
                _json_dumps(chunk.metadata.custom),
                embedding_json,
                chunk.embedding_model,
                _dt_to_str(chunk.created_at) or now,
                _dt_to_str(chunk.source_timestamp),
            ),
        )
        # Index in FTS5
        await self._fts_insert(chunk)
        await self._conn.commit()
        return chunk

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        now = _now_iso()
        rows = []
        for c in chunks:
            embedding_json = json.dumps(c.embedding) if c.embedding else None
            rows.append(
                (
                    str(c.id),
                    str(c.namespace_id),
                    str(c.document_id),
                    c.content,
                    c.metadata.chunk_index,
                    c.metadata.start_char,
                    c.metadata.end_char,
                    c.metadata.token_count,
                    _json_dumps(c.metadata.custom),
                    embedding_json,
                    c.embedding_model,
                    _dt_to_str(c.created_at) or now,
                    _dt_to_str(c.source_timestamp),
                )
            )
        await self._conn.executemany(
            "INSERT INTO chunks "
            "(id, namespace_id, document_id, content, chunk_index, start_char, "
            "end_char, token_count, metadata_, embedding, embedding_model, "
            "created_at, source_timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        for c in chunks:
            await self._fts_insert(c)
        await self._conn.commit()
        return chunks

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        cursor = await self._conn.execute("SELECT * FROM chunks WHERE id = ?", (str(chunk_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = await self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            [str(c) for c in chunk_ids],
        )
        rows = await cursor.fetchall()
        result: dict[UUID, Chunk] = {}
        for r in rows:
            chunk = self._row_to_chunk(r)
            result[chunk.id] = chunk
        return result

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        cursor = await self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (str(document_id),),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chunk(r) for r in rows]

    async def delete_chunks_by_document(self, document_id: UUID) -> int:
        # Remove from FTS first
        cursor = await self._conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?",
            (str(document_id),),
        )
        fts_rows = await cursor.fetchall()
        for fr in fts_rows:
            await self._fts_delete(fr["id"])

        cursor = await self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (str(document_id),))
        await self._conn.commit()
        return cursor.rowcount

    async def count_chunks(self, namespace_id: UUID) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE namespace_id = ?",
            (str(namespace_id),),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        cursor = await self._conn.execute(
            "SELECT * FROM chunks WHERE namespace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (str(namespace_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Brute-force cosine similarity search over chunk embeddings."""
        ns = str(namespace_id)
        if filter_document_ids:
            placeholders = ",".join("?" for _ in filter_document_ids)
            cursor = await self._conn.execute(
                f"SELECT * FROM chunks WHERE namespace_id = ? AND embedding IS NOT NULL "  # noqa: S608
                f"AND document_id IN ({placeholders})",
                [ns, *[str(d) for d in filter_document_ids]],
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM chunks WHERE namespace_id = ? AND embedding IS NOT NULL",
                (ns,),
            )
        rows = await cursor.fetchall()
        if not rows:
            return []

        # Parse embeddings
        embeddings: list[list[float]] = []
        valid_rows: list[Any] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            embeddings.append(emb)
            valid_rows.append(row)

        if not embeddings:
            return []

        from khora._accel import batch_cosine_similarity

        scored_indices = batch_cosine_similarity(query_embedding, embeddings, min_similarity)

        results: list[tuple[Chunk, float]] = []
        for idx, score in scored_indices[:limit]:
            chunk = self._row_to_chunk(valid_rows[idx])
            results.append((chunk, float(score)))
        return results

    # ------------------------------------------------------------------
    # Entity operations (vector side)
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> None:
        now = _now_iso()
        embedding_json = json.dumps(entity.embedding) if entity.embedding else None
        await self._conn.execute(
            "INSERT OR REPLACE INTO entities "
            "(id, namespace_id, name, entity_type, description, attributes, "
            "source_tool, source_document_ids, source_chunk_ids, mention_count, "
            "embedding, embedding_model, valid_from, valid_until, confidence, "
            "metadata_, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(entity.id),
                str(entity.namespace_id),
                entity.name,
                entity.entity_type,
                entity.description,
                _json_dumps(entity.attributes),
                entity.source_tool,
                _uuid_list_dumps(entity.source_document_ids),
                _uuid_list_dumps(entity.source_chunk_ids),
                entity.mention_count,
                embedding_json,
                entity.embedding_model,
                _dt_to_str(entity.valid_from),
                _dt_to_str(entity.valid_until),
                entity.confidence,
                _json_dumps(entity.metadata),
                _dt_to_str(entity.created_at) or now,
                _dt_to_str(entity.updated_at) or now,
            ),
        )
        await self._conn.commit()

    async def update_entity(self, entity: Entity) -> None:
        now = _now_iso()
        embedding_json = json.dumps(entity.embedding) if entity.embedding else None
        await self._conn.execute(
            "UPDATE entities SET "
            "name = ?, entity_type = ?, description = ?, attributes = ?, "
            "source_tool = ?, source_document_ids = ?, source_chunk_ids = ?, "
            "mention_count = ?, embedding = ?, embedding_model = ?, "
            "valid_from = ?, valid_until = ?, confidence = ?, metadata_ = ?, updated_at = ? "
            "WHERE id = ?",
            (
                entity.name,
                entity.entity_type,
                entity.description,
                _json_dumps(entity.attributes),
                entity.source_tool,
                _uuid_list_dumps(entity.source_document_ids),
                _uuid_list_dumps(entity.source_chunk_ids),
                entity.mention_count,
                embedding_json,
                entity.embedding_model,
                _dt_to_str(entity.valid_from),
                _dt_to_str(entity.valid_until),
                entity.confidence,
                _json_dumps(entity.metadata),
                now,
                str(entity.id),
            ),
        )
        await self._conn.commit()

    async def entity_exists(self, entity_id: UUID) -> bool:
        cursor = await self._conn.execute("SELECT 1 FROM entities WHERE id = ?", (str(entity_id),))
        row = await cursor.fetchone()
        return row is not None

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        now = _now_iso()
        await self._conn.execute(
            "UPDATE entities SET embedding = ?, embedding_model = ?, updated_at = ? WHERE id = ?",
            (json.dumps(embedding), model, now, str(entity_id)),
        )
        await self._conn.commit()

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        now = _now_iso()
        count = 0
        for entity_id, embedding, model in updates:
            await self._conn.execute(
                "UPDATE entities SET embedding = ?, embedding_model = ?, updated_at = ? WHERE id = ?",
                (json.dumps(embedding), model, now, str(entity_id)),
            )
            count += 1
        await self._conn.commit()
        return count

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """Brute-force cosine similarity search over entity embeddings."""
        cursor = await self._conn.execute(
            "SELECT id, embedding FROM entities WHERE namespace_id = ? AND embedding IS NOT NULL",
            (str(namespace_id),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        embeddings: list[list[float]] = []
        entity_ids: list[UUID] = []
        for row in rows:
            embeddings.append(json.loads(row["embedding"]))
            entity_ids.append(UUID(row["id"]))

        from khora._accel import batch_cosine_similarity

        scored = batch_cosine_similarity(query_embedding, embeddings, min_similarity)

        results: list[tuple[UUID, float]] = []
        for idx, score in scored[:limit]:
            results.append((entity_ids[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
    ) -> list[tuple[Chunk, float]]:
        """Full-text search via FTS5."""
        # FTS5 MATCH uses simple tokenizer; sanitize query for safety
        safe_query = query_text.replace('"', '""')
        cursor = await self._conn.execute(
            "SELECT f.chunk_id, f.rank FROM chunks_fts f "
            "WHERE chunks_fts MATCH ? AND f.namespace_id = ? "
            "ORDER BY f.rank LIMIT ?",
            (safe_query, str(namespace_id), limit),
        )
        fts_rows = await cursor.fetchall()
        if not fts_rows:
            return []

        # Fetch full chunk data for matched IDs
        chunk_ids = [r["chunk_id"] for r in fts_rows]
        ranks = {r["chunk_id"]: r["rank"] for r in fts_rows}
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = await self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        )
        chunk_rows = await cursor.fetchall()
        chunk_map = {row["id"]: row for row in chunk_rows}

        results: list[tuple[Chunk, float]] = []
        for cid in chunk_ids:
            if cid in chunk_map:
                chunk = self._row_to_chunk(chunk_map[cid])
                # FTS5 rank is negative (lower = better), negate for score
                results.append((chunk, -ranks[cid]))
        return results

    # ------------------------------------------------------------------
    # FTS5 helpers
    # ------------------------------------------------------------------

    async def _fts_insert(self, chunk: Chunk) -> None:
        """Insert a chunk into the FTS5 index."""
        try:
            await self._conn.execute(
                "INSERT INTO chunks_fts(content, chunk_id, namespace_id) VALUES (?, ?, ?)",
                (chunk.content, str(chunk.id), str(chunk.namespace_id)),
            )
        except Exception:
            # FTS5 may not be available in all SQLite builds
            logger.debug("FTS5 insert failed (may be unsupported)")

    async def _fts_delete(self, chunk_id: str) -> None:
        """Remove a chunk from the FTS5 index."""
        try:
            await self._conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                (chunk_id,),
            )
        except Exception:
            logger.debug("FTS5 delete failed (may be unsupported)")

    # ------------------------------------------------------------------
    # Row → domain model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: Any) -> Chunk:
        raw_embedding = row["embedding"]
        embedding: list[float] | None = None
        if raw_embedding:
            embedding = json.loads(raw_embedding)

        return Chunk(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            document_id=UUID(row["document_id"]),
            content=row["content"] or "",
            metadata=ChunkMetadata(
                document_id=UUID(row["document_id"]),
                chunk_index=row["chunk_index"] or 0,
                start_char=row["start_char"] or 0,
                end_char=row["end_char"] or 0,
                token_count=row["token_count"] or 0,
                custom=_json_loads(row["metadata_"]),
            ),
            embedding=embedding,
            embedding_model=row["embedding_model"] or "",
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            source_timestamp=_parse_dt(row["source_timestamp"]),
        )

    @staticmethod
    def _row_to_entity(row: Any) -> Entity:
        raw_embedding = row["embedding"]
        embedding: list[float] | None = None
        if raw_embedding:
            embedding = json.loads(raw_embedding)

        return Entity(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            name=row["name"] or "",
            entity_type=row["entity_type"] or "CONCEPT",
            description=row["description"] or "",
            attributes=_json_loads(row["attributes"]),
            source_tool=row["source_tool"] or "",
            source_document_ids=_uuid_list_loads(row["source_document_ids"]),
            source_chunk_ids=_uuid_list_loads(row["source_chunk_ids"]),
            mention_count=row["mention_count"] or 1,
            embedding=embedding,
            embedding_model=row["embedding_model"] or "",
            valid_from=_parse_dt(row["valid_from"]),
            valid_until=_parse_dt(row["valid_until"]),
            confidence=row["confidence"] if row["confidence"] is not None else 1.0,
            metadata=_json_loads(row["metadata_"]),
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
        )
