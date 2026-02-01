"""LiteLLM-based embedder for unified embedding generation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from hashlib import sha256
from typing import TYPE_CHECKING

from loguru import logger

from .base import Embedder

if TYPE_CHECKING:
    from khora.config import LiteLLMConfig


class LiteLLMEmbedder(Embedder):
    """LiteLLM-based embedder for text embeddings.

    Uses LiteLLM to generate embeddings from various providers
    (OpenAI, Cohere, etc.) through a unified interface.

    Includes an in-memory embedding cache to avoid re-embedding
    identical texts (e.g. entity mentions that recur across queries).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        *,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 100,
        cache_max_size: int = 10000,
    ) -> None:
        """Initialize the LiteLLM embedder.

        Args:
            model: Embedding model name
            dimension: Embedding vector dimension
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            batch_size: Maximum batch size for embed_batch
            cache_max_size: Maximum cached embeddings (0 to disable)
        """
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return sha256(f"{self._model}:{text}".encode()).hexdigest()

    def _cache_get(self, text: str) -> list[float] | None:
        """Look up a cached embedding."""
        if not self._cache_max_size:
            return None
        key = self._cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_put(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        if not self._cache_max_size:
            return
        key = self._cache_key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

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
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses an in-memory cache to skip API calls for previously seen texts.

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

        # Separate cached vs uncached texts
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Fetch uncached embeddings
        if uncached_texts:
            if len(uncached_texts) > self._batch_size:
                all_embeddings: list[list[float]] = []
                for i in range(0, len(uncached_texts), self._batch_size):
                    batch = uncached_texts[i : i + self._batch_size]
                    batch_embeddings = await self._embed_batch_internal(batch)
                    all_embeddings.extend(batch_embeddings)
            else:
                all_embeddings = await self._embed_batch_internal(uncached_texts)

            # Populate results and cache
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                self._cache_put(texts[idx], embedding)

        return results  # type: ignore[return-value]

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
