"""LanceDB embedded vector store for Chronicle engine.

Provides a zero-infrastructure alternative to pgvector for local/embedded
deployments. LanceDB is file-backed (no server), supports HNSW indexing,
and handles millions of vectors efficiently.

Install: ``pip install khora[lancedb]`` (optional dependency).

Usage::

    engine = ChronicleEngine(config, vector_backend="lancedb", lancedb_path="./data/chronicle.lance")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import lancedb as _lancedb
    import pyarrow as pa

    _HAS_LANCEDB = True
except ImportError:
    _lancedb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]
    _HAS_LANCEDB = False


class LanceDBVectorStore:
    """Embedded vector store backed by LanceDB.

    Stores chunk embeddings in a local Lance file with HNSW indexing.
    No server required — the database is a directory on disk.

    This is a lightweight vector-only store. It does NOT replace the
    full StorageCoordinator — Chronicle still uses PostgreSQL for
    relational data (documents, namespaces, entities). LanceDB only
    handles the embedding similarity search path.
    """

    def __init__(
        self,
        path: str | Path = "./chronicle.lance",
        *,
        embedding_dim: int = 1536,
    ) -> None:
        if not _HAS_LANCEDB:
            raise ImportError("LanceDB is not installed. Install with: pip install khora[lancedb]")

        self._path = Path(path)
        self._embedding_dim = embedding_dim
        self._db: Any = None
        self._chunks_table: Any = None
        self._connected = False

    async def connect(self) -> None:
        """Open or create the LanceDB database."""
        if self._connected:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = _lancedb.connect(str(self._path))

        # Create or open the chunks table
        try:
            self._chunks_table = self._db.open_table("chunks")
            logger.info(f"Opened LanceDB table 'chunks' ({self._chunks_table.count_rows()} rows)")
        except Exception:
            # Create with schema
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("namespace_id", pa.string()),
                    pa.field("document_id", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), self._embedding_dim)),
                    pa.field("created_at", pa.float64()),  # epoch seconds
                ]
            )
            self._chunks_table = self._db.create_table("chunks", schema=schema)
            logger.info("Created LanceDB table 'chunks'")

        self._connected = True

    async def disconnect(self) -> None:
        """Close the database connection."""
        self._db = None
        self._chunks_table = None
        self._connected = False

    async def add_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Add chunk embeddings to the store.

        Args:
            chunks: List of dicts with keys: id, namespace_id, document_id,
                    content, embedding, created_at

        Returns:
            Number of chunks added.
        """
        if not chunks or not self._chunks_table:
            return 0

        records = []
        for c in chunks:
            embedding = c.get("embedding")
            if embedding is None:
                continue
            records.append(
                {
                    "id": str(c["id"]),
                    "namespace_id": str(c.get("namespace_id", "")),
                    "document_id": str(c.get("document_id", "")),
                    "content": c.get("content", ""),
                    "embedding": list(embedding),
                    "created_at": (
                        c.get("created_at", datetime.now(UTC)).timestamp()
                        if isinstance(c.get("created_at"), datetime)
                        else float(c.get("created_at", 0))
                    ),
                }
            )

        if records:
            self._chunks_table.add(records)

        return len(records)

    async def search_similar(
        self,
        query_embedding: list[float],
        *,
        namespace_id: str | None = None,
        limit: int = 10,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search for similar chunks by embedding.

        Args:
            query_embedding: Query vector.
            namespace_id: Optional namespace filter.
            limit: Max results.
            created_after: Optional temporal lower bound.
            created_before: Optional temporal upper bound.

        Returns:
            List of (chunk_dict, distance) tuples, sorted by similarity.
        """
        if not self._chunks_table:
            return []

        query = self._chunks_table.search(query_embedding).limit(limit)

        # Apply filters
        filters = []
        if namespace_id:
            filters.append(f"namespace_id = '{namespace_id}'")
        if created_after:
            filters.append(f"created_at >= {created_after.timestamp()}")
        if created_before:
            filters.append(f"created_at <= {created_before.timestamp()}")

        if filters:
            query = query.where(" AND ".join(filters))

        try:
            results = query.to_pandas()
        except Exception:
            return []

        output = []
        for _, row in results.iterrows():
            # LanceDB returns _distance (lower = more similar)
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance)  # Convert to similarity
            output.append(
                (
                    {
                        "id": row.get("id", ""),
                        "namespace_id": row.get("namespace_id", ""),
                        "document_id": row.get("document_id", ""),
                        "content": row.get("content", ""),
                        "created_at": row.get("created_at", 0),
                    },
                    similarity,
                )
            )

        return output

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Returns number of rows deleted.
        """
        if not self._chunks_table:
            return 0

        try:
            before = self._chunks_table.count_rows()
            self._chunks_table.delete(f"document_id = '{document_id}'")
            after = self._chunks_table.count_rows()
            return before - after
        except Exception:
            return 0

    async def count(self, namespace_id: str | None = None) -> int:
        """Count chunks, optionally filtered by namespace."""
        if not self._chunks_table:
            return 0

        try:
            if namespace_id:
                # LanceDB doesn't have a direct filtered count, do a search
                return self._chunks_table.count_rows(f"namespace_id = '{namespace_id}'")
            return self._chunks_table.count_rows()
        except Exception:
            return 0

    @property
    def is_available(self) -> bool:
        """Check if LanceDB is installed and the store is connected."""
        return _HAS_LANCEDB and self._connected
