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
    from khora.core.models import Chunk, ChunkMetadata
    from khora.extraction.chunkers import create_chunker

    # Create chunker
    chunker = create_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chunk the document
    chunk_results = await asyncio.to_thread(chunker.chunk, document.content)

    # Convert to Chunk objects
    # Inherit document timestamp and custom metadata so they propagate to search results
    doc_custom = document.metadata.custom if document.metadata else {}
    chunks = []
    for result in chunk_results:
        # Merge document custom metadata with any chunk-level metadata
        custom = {**doc_custom, **result.metadata} if doc_custom else result.metadata
        chunk = Chunk(
            namespace_id=document.namespace_id,
            document_id=document.id,
            content=result.content,
            metadata=ChunkMetadata(
                document_id=document.id,
                chunk_index=result.index,
                start_char=result.start_char,
                end_char=result.end_char,
                token_count=result.token_count,
                custom=custom,
            ),
            created_at=document.created_at,  # Inherit source timestamp from document
        )
        chunks.append(chunk)

    return chunks
