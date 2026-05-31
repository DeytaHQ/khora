"""Embedding task for chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from khora.core.models import Chunk


async def embed_chunks(
    chunks: list[Chunk],
    *,
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
    batch_size: int = 100,
    shared_embedder: Any | None = None,
) -> list[Chunk]:
    """Generate embeddings for chunks.

    Args:
        chunks: Chunks to embed
        model: Embedding model to use
        dimension: Embedding vector dimension
        batch_size: Batch size for embedding
        shared_embedder: Optional shared embedder instance (preserves LRU cache across calls)

    Returns:
        Chunks with embeddings set
    """
    from khora.extraction.embedders import LiteLLMEmbedder

    if not chunks:
        return []

    # Use shared embedder if provided, otherwise create a fresh one
    embedder = shared_embedder or LiteLLMEmbedder(model=model, dimension=dimension, batch_size=batch_size)

    # Extract texts
    texts = [chunk.content for chunk in chunks]

    # Generate embeddings
    embeddings = await embedder.embed_batch(texts)

    # Update chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.embedding_model = model

    return chunks
