"""Unit tests for extraction/embedders/litellm.py — LiteLLM embedder."""

from __future__ import annotations

import time as _time_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.extraction.embedders.litellm import LiteLLMEmbedder


class TestLiteLLMEmbedder:
    """Tests for LiteLLMEmbedder initialization and properties."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        embedder = LiteLLMEmbedder()
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.dimension == 1536
        assert embedder._batch_size == 200
        assert embedder._cache_max_size == 10000

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        embedder = LiteLLMEmbedder(
            model="custom-model",
            dimension=768,
            batch_size=50,
            cache_max_size=500,
        )
        assert embedder.model_name == "custom-model"
        assert embedder.dimension == 768
        assert embedder._batch_size == 50
        assert embedder._cache_max_size == 500

    def test_from_config(self) -> None:
        """from_config creates embedder with config values."""
        config = MagicMock()
        config.embedding_model = "test-embed"
        config.embedding_dimension = 512
        config.timeout = 60
        config.max_retries = 5
        embedder = LiteLLMEmbedder.from_config(config)
        assert embedder.model_name == "test-embed"
        assert embedder.dimension == 512


class TestCache:
    """Tests for embedding cache."""

    def test_cache_key_deterministic(self) -> None:
        """Same text produces same cache key."""
        embedder = LiteLLMEmbedder()
        k1 = embedder._cache_key("hello")
        k2 = embedder._cache_key("hello")
        assert k1 == k2

    def test_cache_key_different(self) -> None:
        """Different text produces different key."""
        embedder = LiteLLMEmbedder()
        k1 = embedder._cache_key("hello")
        k2 = embedder._cache_key("world")
        assert k1 != k2

    def test_cache_miss(self) -> None:
        """Cache get for uncached text returns None."""
        embedder = LiteLLMEmbedder()
        assert embedder._cache_get("uncached") is None
        assert embedder._cache_misses == 1

    def test_cache_hit(self) -> None:
        """Cache put then get returns embedding."""
        embedder = LiteLLMEmbedder()
        embedding = [0.1, 0.2, 0.3]
        embedder._cache_put("test", embedding)
        result = embedder._cache_get("test")
        assert result == embedding
        assert embedder._cache_hits == 1

    def test_cache_eviction(self) -> None:
        """Cache evicts oldest entries when max_size is exceeded."""
        embedder = LiteLLMEmbedder(cache_max_size=2)
        embedder._cache_put("a", [1.0])
        embedder._cache_put("b", [2.0])
        embedder._cache_put("c", [3.0])  # Should evict "a"
        assert embedder._cache_get("a") is None
        assert embedder._cache_get("c") == [3.0]

    def test_cache_disabled(self) -> None:
        """Cache disabled when max_size=0."""
        embedder = LiteLLMEmbedder(cache_max_size=0)
        embedder._cache_put("test", [1.0])
        assert embedder._cache_get("test") is None

    def test_cache_stats(self) -> None:
        """Cache stats report correct values."""
        embedder = LiteLLMEmbedder()
        embedder._cache_put("a", [1.0])
        embedder._cache_get("a")  # hit
        embedder._cache_get("b")  # miss
        stats = embedder.cache_stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestEmbed:
    """Tests for the embed method."""

    @pytest.mark.asyncio
    async def test_single_text(self) -> None:
        """embed delegates to embed_batch."""
        embedder = LiteLLMEmbedder(model="test-model", max_retries=1)
        # Use a pre-normalized vector so L2-normalization is a no-op
        expected = [0.0, 0.0, 1.0]

        mock_response = MagicMock()
        mock_response.data = [{"embedding": expected}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed("hello world")

        assert result == expected

    @pytest.mark.asyncio
    async def test_cached_embed(self) -> None:
        """embed returns cached result without API call."""
        embedder = LiteLLMEmbedder()
        embedder._cache_put("hello", [0.1, 0.2])
        result = await embedder.embed("hello")
        assert result == [0.1, 0.2]


class TestEmbedBatch:
    """Tests for embed_batch method."""

    @pytest.mark.asyncio
    async def test_empty_texts(self) -> None:
        """Empty list returns empty list."""
        embedder = LiteLLMEmbedder()
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_small_batch(self) -> None:
        """Small batch (no chunking needed) calls API once."""
        embedder = LiteLLMEmbedder(model="test-model", batch_size=100, max_retries=1)

        mock_response = MagicMock()
        # Use pre-normalized vectors so L2-normalization is a no-op
        mock_response.data = [
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
        ]
        mock_response.usage = MagicMock(prompt_tokens=20, total_tokens=20)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_caching_integration(self) -> None:
        """Previously cached texts are not re-embedded."""
        embedder = LiteLLMEmbedder(model="test-model", batch_size=100, max_retries=1)
        # Cached values are already normalized (stored post-normalization)
        embedder._cache_put("cached", [1.0, 0.0])

        mock_response = MagicMock()
        # Use pre-normalized vector so L2-normalization is a no-op
        mock_response.data = [{"embedding": [0.0, 1.0]}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["cached", "new_text"])

        assert result[0] == [1.0, 0.0]  # From cache
        assert result[1] == [0.0, 1.0]  # From API (normalized)


class TestEmbedBatchInternal:
    """Tests for _embed_batch_internal (input sanitization)."""

    @pytest.mark.asyncio
    async def test_sanitizes_empty_inputs(self) -> None:
        """Empty/None inputs are replaced with space placeholder."""
        embedder = LiteLLMEmbedder(model="test-model", max_retries=1)

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1]},
            {"embedding": [0.2]},
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_api,
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder._embed_batch_internal(["", "  "])

        # Verify sanitized inputs were sent to API
        call_args = mock_api.call_args
        assert call_args.kwargs["input"] == [" ", " "]
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Retries on transient failure."""
        embedder = LiteLLMEmbedder(model="test-model", max_retries=2)

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch(
                "litellm.aembedding",
                new_callable=AsyncMock,
                side_effect=[Exception("transient"), mock_response],
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder._embed_batch_internal(["test"])

        assert result == [[0.1]]

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        """Raises after exhausting retries."""
        embedder = LiteLLMEmbedder(model="test-model", max_retries=2)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, side_effect=Exception("persistent")),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(Exception, match="persistent"):
                await embedder._embed_batch_internal(["test"])


class TestDimensionValidation:
    """Tests for embedding dimension validation (M-2)."""

    @pytest.mark.asyncio
    async def test_matching_dimension_no_warning(self) -> None:
        """No warning when actual dimension matches configured."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=3, max_retries=1)

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
            patch("khora.extraction.embedders.litellm.logger") as mock_logger,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder._embed_batch_internal(["hello"])

        assert result == [[0.1, 0.2, 0.3]]
        assert embedder.dimension == 3
        assert embedder._dimension_validated is True
        mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_mismatched_dimension_warns_and_updates(self) -> None:
        """Warning logged and dimension updated when mismatch detected."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=1536, max_retries=1)

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]  # dim=3, not 1536
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
            patch("khora.extraction.embedders.litellm.logger") as mock_logger,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder._embed_batch_internal(["hello"])

        assert result == [[0.1, 0.2, 0.3]]
        assert embedder.dimension == 3  # Updated to actual
        assert embedder._dimension_validated is True
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "dimension mismatch" in warning_msg.lower()

    @pytest.mark.asyncio
    async def test_dimension_validated_only_once(self) -> None:
        """Dimension validation only runs on first call."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=3, max_retries=1)

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await embedder._embed_batch_internal(["hello"])
            assert embedder._dimension_validated is True

            # Second call: change response dimension — should NOT trigger re-validation
            mock_response2 = MagicMock()
            mock_response2.data = [{"embedding": [0.1, 0.2]}]  # dim=2
            mock_response2.usage = MagicMock(prompt_tokens=10, total_tokens=10)

            with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response2):
                await embedder._embed_batch_internal(["world"])

            # Dimension should still be 3 (not updated to 2)
            assert embedder.dimension == 3


class TestCacheTTL:
    """Tests for cache TTL support (M-3)."""

    def test_no_ttl_entries_never_expire(self) -> None:
        """With TTL=None, entries never expire."""
        embedder = LiteLLMEmbedder(cache_ttl_hours=None)
        embedder._cache_put("test", [1.0, 2.0])

        # Even with time advancing, entry should still be valid
        result = embedder._cache_get("test")
        assert result == [1.0, 2.0]

    def test_ttl_entry_valid_before_expiry(self) -> None:
        """Entry is returned when within TTL."""
        embedder = LiteLLMEmbedder(cache_ttl_hours=1)
        embedder._cache_put("test", [1.0, 2.0])

        # Immediately after put, should be available
        result = embedder._cache_get("test")
        assert result == [1.0, 2.0]

    def test_ttl_entry_expires(self) -> None:
        """Entry expires after TTL elapses."""
        embedder = LiteLLMEmbedder(cache_ttl_hours=1)  # 1 hour = 3600 seconds
        embedder._cache_put("test", [1.0, 2.0])

        # Patch monotonic to simulate time passing beyond TTL
        current_time = _time_mod.monotonic()
        with patch.object(
            _time_mod,
            "monotonic",
            return_value=current_time + 3601,
        ):
            result = embedder._cache_get("test")

        assert result is None
        assert embedder._cache_misses == 1

    def test_ttl_entry_valid_just_before_expiry(self) -> None:
        """Entry is still valid just before TTL expires."""
        embedder = LiteLLMEmbedder(cache_ttl_hours=1)
        embedder._cache_put("test", [1.0, 2.0])

        current_time = _time_mod.monotonic()
        with patch.object(
            _time_mod,
            "monotonic",
            return_value=current_time + 3599,
        ):
            result = embedder._cache_get("test")

        assert result == [1.0, 2.0]

    def test_ttl_expired_entry_evicted_from_cache(self) -> None:
        """Expired entry is removed from cache on access."""
        embedder = LiteLLMEmbedder(cache_ttl_hours=1)
        embedder._cache_put("test", [1.0, 2.0])
        assert embedder.cache_stats["size"] == 1

        current_time = _time_mod.monotonic()
        with patch.object(
            _time_mod,
            "monotonic",
            return_value=current_time + 3601,
        ):
            embedder._cache_get("test")

        assert embedder.cache_stats["size"] == 0
