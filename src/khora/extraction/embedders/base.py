"""Base embedder protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...
