"""LiteLLM-based embedder for unified embedding generation."""

from __future__ import annotations

import asyncio
import re
import time as _time_mod
from collections import OrderedDict
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from khora.config.llm import get_shared_session, llm_call_timeout
from khora.exceptions import EmbeddingError
from khora.telemetry import trace_span

from ._request_telemetry import set_connector_attributes, set_rate_limit_attributes
from .base import Embedder

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    from khora.config import LiteLLMConfig

# Token limits per embedding model family.  OpenAI text-embedding-3-*
# accepts up to 8191 tokens.  We leave a small margin.
_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}
_DEFAULT_TOKEN_LIMIT = 8191

# Cached tiktoken encoding (lazy-loaded, thread-safe after first call)
_tiktoken_encoding = None


def _get_encoding():
    """Get or create the cached tiktoken encoding for cl100k_base."""
    global _tiktoken_encoding  # noqa: PLW0603
    if _tiktoken_encoding is None:
        try:
            import tiktoken

            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None
    return _tiktoken_encoding


def _truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within the token limit.

    Uses tiktoken for accurate counting; falls back to a conservative
    character-based estimate (~3.5 chars/token) if tiktoken is unavailable.
    """
    enc = _get_encoding()
    if enc is not None:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    # Fallback: conservative chars-per-token estimate
    char_limit = int(max_tokens * 3.5)
    if len(text) <= char_limit:
        return text
    return text[:char_limit]


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
        max_batch_tokens: int = 50_000,
        cache_max_size: int = 50000,
        embed_concurrency: int = 20,
        retry_wait: float = 1.0,
        cache_ttl_hours: int | None = None,
        max_token_limit: int | None = None,
    ) -> None:
        """Initialize the LiteLLM embedder.

        Args:
            model: Embedding model name
            dimension: Embedding vector dimension
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            batch_size: Maximum texts per sub-batch (hard cap)
            max_batch_tokens: Maximum estimated tokens per sub-batch.
                Dynamically sizes batches so short texts get large batches
                and long texts get small batches, keeping API response
                payloads under ~2MB to avoid HTTP transfer corruption.
            cache_max_size: Maximum cached embeddings (0 to disable). Each entry
                uses ~13 KB (numpy float64) when numpy is available, or ~50 KB
                (Python list[float]) otherwise. At 50,000 entries: ~650 MB.
            embed_concurrency: Maximum concurrent embedding sub-batch API calls
            retry_wait: Base wait time (seconds) for exponential backoff between retries
            cache_ttl_hours: Cache entry TTL in hours (None = no expiry)
            max_token_limit: Per-text token limit; auto-detected from model if None
        """
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._max_batch_tokens = max_batch_tokens
        self._embed_concurrency = embed_concurrency
        self._retry_wait = retry_wait
        # Stores numpy arrays when numpy is available (~13 KB/entry vs ~50 KB for list[float])
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_ttl_seconds: float | None = cache_ttl_hours * 3600.0 if cache_ttl_hours is not None else None
        self._cache_hits = 0
        self._cache_misses = 0
        # Resolve per-text token limit from model name or explicit override
        if max_token_limit is not None:
            self._max_token_limit = max_token_limit
        else:
            # Match model name against known limits (supports prefixed model names)
            base = model.split("/")[-1] if "/" in model else model
            self._max_token_limit = _MODEL_TOKEN_LIMITS.get(base, _DEFAULT_TOKEN_LIMIT)
        self._truncation_count = 0

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return sha256(f"{self._model}:{text}".encode()).hexdigest()

    def _cache_get(self, text: str, *, key: str | None = None) -> list[float] | None:
        """Look up a cached embedding, respecting TTL if configured."""
        if not self._cache_max_size:
            return None
        key = key or self._cache_key(text)
        if key in self._cache:
            embedding, stored_at = self._cache[key]
            # Check TTL expiry
            if self._cache_ttl_seconds is not None:
                if (_time_mod.monotonic() - stored_at) > self._cache_ttl_seconds:
                    del self._cache[key]
                    self._cache_misses += 1
                    return None
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return embedding.tolist() if _HAS_NUMPY and hasattr(embedding, "tolist") else embedding
        self._cache_misses += 1
        return None

    def _cache_put(self, text: str, embedding: list[float], *, key: str | None = None) -> None:
        """Store an embedding in the cache with a timestamp."""
        if not self._cache_max_size:
            return
        key = key or self._cache_key(text)
        # Store as numpy float64 array when available: ~13 KB/entry vs ~50 KB for list[float]
        stored: Any = np.array(embedding, dtype=np.float64) if _HAS_NUMPY else embedding
        self._cache[key] = (stored, _time_mod.monotonic())
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
            batch_size=getattr(config, "embed_batch_size", 200),
            max_batch_tokens=getattr(config, "embed_batch_tokens", 50_000),
            cache_max_size=getattr(config, "embed_cache_max_size", 50000),
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
        # Delegate entirely to embed_batch so cache accounting (hits/misses) happens
        # in exactly one place.  A direct _cache_get call here would double-count
        # misses: once here and once inside embed_batch's cache loop (#1231).
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    def _split_by_budget(self, texts: list[str]) -> list[list[str]]:
        """Split texts into sub-batches respecting both count and token limits.

        Uses ``len(text) // 4`` as a cheap token estimate (avoids tiktoken
        overhead).  Each sub-batch stays under *both* ``_batch_size`` texts
        and ``_max_batch_tokens`` estimated tokens.  Short texts get large
        batches; long texts get small batches — keeping API response payloads
        under ~2 MB regardless of input content length.
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            est_tokens = max(len(text) // 4, 1)
            # Start a new batch if adding this text would exceed either limit
            if current and (len(current) >= self._batch_size or current_tokens + est_tokens > self._max_batch_tokens):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += est_tokens

        if current:
            batches.append(current)
        return batches

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
        with trace_span("khora.embedder.cache_lookup") as cache_span:
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

            cache_hits = len(texts) - len(uncached_texts)
            cache_span.set_attribute("total", len(texts))
            cache_span.set_attribute("hits", cache_hits)
            cache_span.set_attribute("misses", len(uncached_texts))

        # Record embedding cache statistics
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

            with trace_span("khora.embedder.api_call") as api_span:
                api_span.set_attribute("model", self._model)
                api_span.set_attribute("unique_texts", len(unique_texts))
                api_span.set_attribute("batch_size", len(uncached_texts))
                api_span.set_attribute("deduplicated", len(uncached_texts) - len(unique_texts))

                sub_batches = self._split_by_budget(unique_texts)
                if len(sub_batches) > 1:
                    api_span.set_attribute("sub_batches", len(sub_batches))
                    sem = asyncio.Semaphore(self._embed_concurrency)

                    async def _embed_sub(batch: list[str]) -> list[list[float]]:
                        async with sem:
                            return await self._embed_with_bisect(batch)

                    sub_results = await asyncio.gather(*[_embed_sub(b) for b in sub_batches])
                    unique_embeddings: list[list[float]] = [emb for result in sub_results for emb in result]
                else:
                    unique_embeddings = await self._embed_with_bisect(unique_texts)

            # L2-normalize all embeddings so dot product == cosine similarity
            # downstream (batch_dot_product is ~3x faster than batch_cosine).
            with trace_span("khora.embedder.normalize") as norm_span:
                from khora._accel import normalize_embeddings_batch

                unique_embeddings = normalize_embeddings_batch(unique_embeddings)
                norm_span.set_attribute("count", len(unique_embeddings))

            # Map deduplicated results back to original positions and populate cache
            for i, (idx, key) in enumerate(zip(uncached_indices, uncached_keys)):
                embedding = unique_embeddings[dedup_indices[i]]
                results[idx] = embedding
                self._cache_put(texts[idx], embedding, key=key)

        return results  # type: ignore[return-value]

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:  # type: ignore[invalid-return-type]
        """Internal batch embedding without chunking."""
        import time as _time

        import litellm

        # Sanitize inputs: strip control characters that break JSON serialization
        # (preserving \t, \n, \r) and replace empty strings with a placeholder.
        _ctrl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        sanitized = [_ctrl_re.sub("", t) if t and t.strip() else " " for t in texts]

        # Truncate texts that exceed the model's per-text token limit to prevent
        # 400 Bad Request errors from the embedding API.
        truncated = 0
        for i, text in enumerate(sanitized):
            result = _truncate_text(text, self._max_token_limit)
            if len(result) < len(text):
                sanitized[i] = result
                truncated += 1
        if truncated:
            self._truncation_count += truncated
            logger.warning(
                f"Truncated {truncated}/{len(sanitized)} texts to {self._max_token_limit} tokens "
                f"for model {self._model}"
            )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
            before_sleep=lambda retry_state: logger.opt(depth=1).warning(
                "Retrying embedding (attempt {}) after {!s}",
                retry_state.attempt_number,
                retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else "unknown",
            ),
            reraise=True,
        ):
            with attempt:
                with trace_span("khora.embedder.litellm_request") as req_span:
                    req_span.set_attribute("model", self._model)
                    req_span.set_attribute("batch_size", len(texts))
                    req_span.set_attribute("attempt", attempt.retry_state.attempt_number)
                    req_span.set_attribute("timeout", self._timeout)

                    _t0 = _time.perf_counter()
                    set_connector_attributes(req_span, get_shared_session())
                    response = await asyncio.wait_for(
                        litellm.aembedding(
                            model=self._model,
                            input=sanitized,
                            timeout=self._timeout,
                            dimensions=self._dimension,
                            shared_session=get_shared_session(),
                        ),
                        llm_call_timeout(self._timeout),
                    )
                    set_rate_limit_attributes(req_span, response)
                    _latency = (_time.perf_counter() - _t0) * 1000
                    req_span.set_attribute("latency_ms", round(_latency, 2))

                    # Record telemetry
                    from khora.telemetry import get_collector

                    usage = getattr(response, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", 0) or 0
                    req_span.set_attribute("prompt_tokens", prompt_tokens)
                    req_span.set_attribute("total_tokens", total_tokens)

                    from khora.khora import LLMUsage, _safe_completion_cost

                    _cost = _safe_completion_cost(response, model=self._model, call_type="aembedding")
                    get_collector().record_llm_call(
                        operation="embedding",
                        model=self._model,
                        prompt_tokens=prompt_tokens,
                        total_tokens=total_tokens,
                        latency_ms=_latency,
                        batch_size=len(texts),
                        cache_hit=False,
                        cost_usd=_cost,
                    )

                    from khora.telemetry.context import record_usage

                    record_usage(
                        LLMUsage(
                            operation="embedding",
                            model=self._model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            total_tokens=total_tokens,
                            latency_ms=_latency,
                            batch_size=len(texts),
                            cost_usd=_cost,
                        )
                    )

                result = [item["embedding"] for item in response.data]

                # Validate the returned embedding dimension on every batch.
                # A mismatch means the model returned a different dimension
                # than the one configured (and requested via dimensions=).
                # Raise rather than silently mutating self._dimension - a
                # silent overwrite turns a config error into a downstream
                # store-time crash (e.g. Postgres Vector(1536) columns).
                if result:
                    actual_dim = len(result[0])
                    if actual_dim != self._dimension:
                        raise EmbeddingError(
                            f"Embedding dimension mismatch: configured={self._dimension}, "
                            f"actual={actual_dim} (model {self._model!r}). The model returned a "
                            f"different dimension than requested; set the embedder dimension to "
                            f"match the model, or use a model that honors dimensions={self._dimension}."
                        )

                return result

    async def _embed_with_bisect(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch with bisect-on-failure for transient API errors.

        When the embedding API returns a malformed JSON response (common
        with large batches), splits the batch in half and retries each
        half independently instead of retrying the entire batch.
        """
        try:
            return await self._embed_batch_internal(texts)
        except Exception as e:
            error_str = str(e)
            is_parse_error = any(
                phrase in error_str for phrase in ("Expecting ',' delimiter", "Expecting value", "JSONDecodeError")
            )
            if not is_parse_error or len(texts) <= 2:
                raise

            mid = len(texts) // 2
            logger.warning(
                f"Embedding batch ({len(texts)} texts) hit JSON parse error, bisecting into {mid} + {len(texts) - mid}"
            )
            left = await self._embed_with_bisect(texts[:mid])
            right = await self._embed_with_bisect(texts[mid:])
            return left + right

    async def close(self) -> None:
        """Close underlying litellm HTTP sessions to avoid 'Unclosed client session' warnings."""
        try:
            import litellm as _litellm

            for attr in ("acache", "client_session"):
                session = getattr(_litellm, attr, None)
                if session is not None:
                    try:
                        await session.close()
                    except Exception as e:
                        logger.debug(f"Failed to close litellm session ({attr}): {e}")
        except ImportError:
            pass

    async def __aenter__(self) -> LiteLLMEmbedder:
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        await self.close()
