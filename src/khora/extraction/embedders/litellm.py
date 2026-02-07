"""LiteLLM-based embedder for unified embedding generation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from hashlib import sha256
from typing import TYPE_CHECKING

from loguru import logger
from tenacity import AsyncRetrying, before_sleep_log, stop_after_attempt, wait_exponential

from .base import Embedder

if TYPE_CHECKING:
    from khora.config import LiteLLMConfig


class LiteLLMEmbedder(Embedder):
    """LiteLLM-based embedder for text embeddings.

    Uses LiteLLM to generate embeddings from various providers
    (OpenAI, Cohere, etc.) through a unified interface.

    Includes an in-memory embedding cache to avoid re-embedding
    identical texts (e.g. entity mentions that recur across queries).

    Cache Behavior:
        The cache persists across multiple embed_batch() calls within
        the embedder's lifetime. This enables cross-document embedding
        deduplication when processing document batches - if chunk text
        appears in multiple documents, it's only embedded once. For
        optimal batch processing, reuse the same embedder instance
        across all documents in a session.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        *,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 200,
        cache_max_size: int = 10000,
        embed_concurrency: int = 20,
        retry_wait: float = 1.0,
    ) -> None:
        """Initialize the LiteLLM embedder.

        Args:
            model: Embedding model name
            dimension: Embedding vector dimension
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            batch_size: Maximum batch size for embed_batch
            cache_max_size: Maximum cached embeddings (0 to disable)
            embed_concurrency: Maximum concurrent embedding sub-batch API calls
            retry_wait: Base wait time (seconds) for exponential backoff between retries
        """
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._embed_concurrency = embed_concurrency
        self._retry_wait = retry_wait
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return sha256(f"{self._model}:{text}".encode()).hexdigest()

    def _cache_get(self, text: str, *, key: str | None = None) -> list[float] | None:
        """Look up a cached embedding."""
        if not self._cache_max_size:
            return None
        key = key or self._cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_put(self, text: str, embedding: list[float], *, key: str | None = None) -> None:
        """Store an embedding in the cache."""
        if not self._cache_max_size:
            return
        key = key or self._cache_key(text)
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
            retry_wait=config.retry_wait,
            embed_concurrency=config.embed_concurrency,
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

        # Separate cached vs uncached texts; compute cache keys once
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        uncached_keys: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache_get(text, key=key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                uncached_keys.append(key)

        # Record embedding cache statistics
        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            from khora.telemetry import get_collector

            get_collector().record_llm_call(
                operation="embedding",
                model=self._model,
                cache_hit=True,
                batch_size=cache_hits,
                latency_ms=0.0,
            )

        # Fetch uncached embeddings with deduplication
        if uncached_texts:
            # Deduplicate: same text appearing multiple times only needs one API call
            unique_text_map: dict[str, int] = {}  # key -> first occurrence index in unique list
            unique_texts: list[str] = []
            dedup_indices: list[int] = []  # maps uncached position -> unique_texts position

            for key, text in zip(uncached_keys, uncached_texts):
                if key not in unique_text_map:
                    unique_text_map[key] = len(unique_texts)
                    unique_texts.append(text)
                dedup_indices.append(unique_text_map[key])

            if len(unique_texts) > self._batch_size:
                sub_batches = [
                    unique_texts[i : i + self._batch_size] for i in range(0, len(unique_texts), self._batch_size)
                ]
                sem = asyncio.Semaphore(self._embed_concurrency)

                async def _embed_sub(batch: list[str]) -> list[list[float]]:
                    async with sem:
                        return await self._embed_batch_internal(batch)

                sub_results = await asyncio.gather(*[_embed_sub(b) for b in sub_batches])
                unique_embeddings: list[list[float]] = [emb for result in sub_results for emb in result]
            else:
                unique_embeddings = await self._embed_batch_internal(unique_texts)

            # Map deduplicated results back to original positions and populate cache
            for i, (idx, key) in enumerate(zip(uncached_indices, uncached_keys)):
                embedding = unique_embeddings[dedup_indices[i]]
                results[idx] = embedding
                self._cache_put(texts[idx], embedding, key=key)

        return results  # type: ignore[return-value]

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Internal batch embedding without chunking."""
        import time as _time

        import litellm

        # Sanitize inputs: replace None/empty strings with a placeholder to avoid
        # OpenAI '$.input' is invalid errors
        sanitized = [t if t and t.strip() else " " for t in texts]

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        ):
            with attempt:
                _t0 = _time.perf_counter()
                response = await litellm.aembedding(
                    model=self._model,
                    input=sanitized,
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
                    batch_size=len(texts),
                    cache_hit=False,
                )

                return [item["embedding"] for item in response.data]
