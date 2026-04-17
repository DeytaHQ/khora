"""LanceDB table schemas for the sqlite_lance unified backend.

Defines the Arrow schemas for the chunks and entities vector tables and
provides an idempotent ``ensure_lance_tables`` helper used during
``EmbeddedStorageHandle.connect``.  Table bodies and index creation
beyond the core vector column are deferred to later tickets
(DYT-2728/2729/2730/2731).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
from loguru import logger

if TYPE_CHECKING:
    from lancedb.db import AsyncConnection


def _vector_type(dim: int, use_halfvec: bool) -> pa.DataType:
    """Return the fixed-size list type used for embeddings."""
    value_type = pa.float16() if use_halfvec else pa.float32()
    return pa.list_(value_type, list_size=dim)


def chunks_vec_schema(dim: int, use_halfvec: bool) -> pa.Schema:
    """Arrow schema for the ``chunks_vec`` LanceDB table."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("namespace_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=True),
            pa.field("created_at", pa.timestamp("us"), nullable=True),
            pa.field("vector", _vector_type(dim, use_halfvec), nullable=False),
        ]
    )


def entities_vec_schema(dim: int, use_halfvec: bool) -> pa.Schema:
    """Arrow schema for the ``entities_vec`` LanceDB table."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("namespace_id", pa.string(), nullable=False),
            pa.field("vector", _vector_type(dim, use_halfvec), nullable=False),
        ]
    )


async def ensure_lance_tables(
    conn: AsyncConnection,
    dim: int,
    use_halfvec: bool,
) -> None:
    """Create LanceDB vector tables if they don't already exist.

    Idempotent — uses ``exist_ok=True`` so repeated calls are safe.
    Vector indexes (IVF/HNSW) are created lazily by the vector adapter
    in later tickets.

    Args:
        conn: An active ``lancedb.AsyncConnection``.
        dim: Embedding dimension.
        use_halfvec: If True, store embeddings as float16 (half precision).
    """
    logger.info(
        f"Ensuring LanceDB tables (dim={dim}, halfvec={use_halfvec})",
    )
    await conn.create_table(
        "chunks_vec",
        schema=chunks_vec_schema(dim, use_halfvec),
        exist_ok=True,
    )
    await conn.create_table(
        "entities_vec",
        schema=entities_vec_schema(dim, use_halfvec),
        exist_ok=True,
    )
    logger.debug("LanceDB tables ensured")
