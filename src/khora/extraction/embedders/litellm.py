"""LiteLLM-based embedder for unified embedding generation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from .base import Embedder

if TYPE_CHECKING:
    from khora.config import LiteLLMConfig


class LiteLLMEmbedder(Embedder):
    """LiteLLM-based embedder for text embeddings.

    Uses LiteLLM to generate embeddings from various providers
    (OpenAI, Cohere, etc.) through a unified interface.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        *,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 100,
    ) -> None:
        """Initialize the LiteLLM embedder.

        Args:
            model: Embedding model name
            dimension: Embedding vector dimension
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            batch_size: Maximum batch size for embed_batch
        """
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size

    @classmethod
    def from_config(cls, config: LiteLLMConfig) -> LiteLLMEmbedder:
        """Create embedder from LiteLLM configuration.

        Args:
            config: LiteLLMConfig instance

        Returns:
            Configured LiteLLMEmbedder
        """
        return cls(
            model=config.embedding_model,
            dimension=config.embedding_dimension,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            import litellm  # noqa: F401
        except ImportError:
            raise RuntimeError("litellm package not installed. Run: pip install litellm")

        # Process in batches if needed
        if len(texts) > self._batch_size:
            all_embeddings = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                batch_embeddings = await self._embed_batch_internal(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        return await self._embed_batch_internal(texts)

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Internal batch embedding without chunking."""
        import time as _time

        import litellm

        for attempt in range(self._max_retries):
            try:
                _t0 = _time.perf_counter()
                response = await litellm.aembedding(
                    model=self._model,
                    input=texts,
                    timeout=self._timeout,
                )
                _latency = (_time.perf_counter() - _t0) * 1000

                # Record telemetry
                from khora.telemetry import get_collector

                usage = getattr(response, "usage", None)
                get_collector().record_llm_call(
                    operation="embedding",
                    model=self._model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    latency_ms=_latency,
                    metadata={"batch_size": len(texts)},
                )

                return [item["embedding"] for item in response.data]
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Embedding failed after {self._max_retries} attempts: {e}")
                    raise
