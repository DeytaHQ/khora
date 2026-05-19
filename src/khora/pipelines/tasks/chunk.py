"""Chunking task for document processing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khora.core.models import Chunk, Document


async def chunk_document(
    document: Document,
    *,
    strategy: str = "semantic",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk a document into smaller pieces.

    Args:
        document: Document to chunk
        strategy: Chunking strategy (fixed, semantic, recursive)
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    from khora.core.models import Chunk
    from khora.extraction.chunkers import create_chunker

    # Create chunker
    chunker = create_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chunk the document
    chunk_results = await asyncio.to_thread(chunker.chunk, document.content)

    # Convert to Chunk objects
    chunks = []
    for result in chunk_results:
        chunk = Chunk(
            namespace_id=document.namespace_id,
            document_id=document.id,
            content=result.content,
            chunk_index=result.index,
            start_char=result.start_char,
            end_char=result.end_char,
            token_count=result.token_count,
            metadata=dict(document.metadata),
            chunker_info=dict(result.metadata),
            created_at=document.created_at,  # Inherit doc created_at (which is source_timestamp when known)
            # Propagate the parsed source_timestamp so date-bounded
            # recalls don't fall back to chunk.created_at and surface
            # historical rows for "last week"-style queries (#615).
            source_timestamp=document.source_timestamp,
            # Inherit session_id so session-scoped recalls can hit the
            # partial index on (namespace_id, session_id) (#620).
            session_id=document.session_id,
        )
        chunks.append(chunk)

    return chunks
