"""Embedding task for chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prefect import task

if TYPE_CHECKING:
    from khora.core.models import Chunk


@task(name="embed_chunks", retries=3, retry_delay_seconds=10)
async def embed_chunks(
    chunks: list[Chunk],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[Chunk]:
    """Generate embeddings for chunks.

    Args:
        chunks: Chunks to embed
        model: Embedding model to use
        batch_size: Batch size for embedding

    Returns:
        Chunks with embeddings set
    """
    from khora.extraction.embedders import LiteLLMEmbedder

    if not chunks:
        return []

    # Create embedder
    embedder = LiteLLMEmbedder(model=model, batch_size=batch_size)

    # Extract texts
    texts = [chunk.content for chunk in chunks]

    # Generate embeddings
    embeddings = await embedder.embed_batch(texts)

    # Update chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.embedding_model = model

    return chunks
