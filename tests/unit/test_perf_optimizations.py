"""Tests for performance optimizations (B1-B16 from PERFORMANCE_PLAN.md).

Covers:
- B5: Parallel graph+vector writes in create_entity
- B6: Semaphore release during retry sleep
- B7: Shared embedder across documents
- B11: Cache key computed once
- B12: _parse_response accepts dict (no JSON round-trip)
- B10: Embedding input deduplication
- B2: Batch entity upsert (replaces N+1)
- B8: Batch relationship storage in expansion
- B9: Selective entity updates in expansion
- B4: Parallel entity embedding + relationship storage
- B14: Parallel entity/relationship loading in expansion
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Entity
from khora.extraction.embedders.litellm import LiteLLMEmbedder
from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.storage.coordinator import StorageCoordinator

# =========================================================================
# B5: Parallel graph+vector writes in create_entity
# =========================================================================


class TestB5ParallelCreateEntity:
    """B5: create_entity should run graph+vector writes in parallel."""

    @pytest.mark.asyncio
    async def test_create_entity_both_backends_called(self) -> None:
        """Both graph and vector are called when both configured."""
        entity = MagicMock(spec=Entity, namespace_id=uuid4())
        graph = MagicMock()
        graph.create_entity = AsyncMock(return_value=entity)
        vec = MagicMock()
        vec.create_entity = AsyncMock()

        coord = StorageCoordinator(graph=graph, vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            result = await coord.create_entity(entity)

        graph.create_entity.assert_awaited_once()
        vec.create_entity.assert_awaited_once()
        assert result is entity

    @pytest.mark.asyncio
    async def test_create_entity_graph_only(self) -> None:
        """Works with graph backend only."""
        entity = MagicMock(spec=Entity, namespace_id=uuid4())
        graph = MagicMock()
        graph.create_entity = AsyncMock(return_value=entity)

        coord = StorageCoordinator(graph=graph)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            result = await coord.create_entity(entity)

        graph.create_entity.assert_awaited_once()
        assert result is entity

    @pytest.mark.asyncio
    async def test_create_entity_vector_only(self) -> None:
        """Works with vector backend only."""
        entity = MagicMock(spec=Entity, namespace_id=uuid4())
        vec = MagicMock()
        vec.create_entity = AsyncMock()

        coord = StorageCoordinator(vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            await coord.create_entity(entity)

        vec.create_entity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_entity_parallel_execution(self) -> None:
        """Verify graph and vector run concurrently (via asyncio.gather)."""
        entity = MagicMock(spec=Entity, namespace_id=uuid4())
        call_order = []

        async def slow_graph_create(e):
            call_order.append("graph_start")
            await asyncio.sleep(0.01)
            call_order.append("graph_end")
            return e

        async def slow_vec_create(e):
            call_order.append("vec_start")
            await asyncio.sleep(0.01)
            call_order.append("vec_end")

        graph = MagicMock()
        graph.create_entity = slow_graph_create
        vec = MagicMock()
        vec.create_entity = slow_vec_create

        coord = StorageCoordinator(graph=graph, vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            await coord.create_entity(entity)

        # Both should start before either finishes (parallel execution)
        assert "graph_start" in call_order
        assert "vec_start" in call_order
        # vec should start before graph finishes (parallel execution)
        assert call_order.index("vec_start") < call_order.index("graph_end")


# =========================================================================
# B6: Release semaphore during retry sleep
# =========================================================================


class TestB6SemaphoreReleaseDuringRetry:
    """B6: Semaphore should be released before retry sleep."""

    @pytest.mark.asyncio
    async def test_semaphore_released_during_retry_sleep(self) -> None:
        """Semaphore slot is freed while sleeping between retries."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=2, max_concurrent=1)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"entities": [{"name": "Test", "entity_type": "CONCEPT"}], "relationships": []}
        )
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        # First call fails, second succeeds
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient")
            return mock_response

        semaphore_was_free = False

        original_sleep = asyncio.sleep

        async def check_semaphore_during_sleep(duration):
            nonlocal semaphore_was_free
            # During sleep, the semaphore should be released
            # Try to acquire it to verify
            acquired = extractor._semaphore._value > 0
            if acquired:
                semaphore_was_free = True
            await original_sleep(0)  # Don't actually sleep

        with (
            patch("litellm.acompletion", side_effect=mock_completion),
            patch("asyncio.sleep", side_effect=check_semaphore_during_sleep),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract(
                "test text",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
            )

        assert semaphore_was_free, "Semaphore should be free during retry sleep"
        assert len(result.entities) == 1

    @pytest.mark.asyncio
    async def test_extract_multi_semaphore_released_during_retry(self) -> None:
        """Multi-batch extraction also releases semaphore during retry."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=2, max_concurrent=1)

        section_data = {"sections": [{"entities": [{"name": "A", "entity_type": "PERSON"}], "relationships": []}]}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient")
            return mock_response

        semaphore_was_free = False

        async def check_semaphore(duration):
            nonlocal semaphore_was_free
            if extractor._semaphore._value > 0:
                semaphore_was_free = True

        with (
            patch("litellm.acompletion", side_effect=mock_completion),
            patch("asyncio.sleep", side_effect=check_semaphore),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["text1"],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        assert semaphore_was_free, "Semaphore should be free during retry sleep"
        assert len(results) == 1


# =========================================================================
# B11: Cache key computed once + B12: _parse_response accepts dict
# =========================================================================


class TestB11CacheKeyOptimization:
    """B11: Cache key should be computed once per text in embed_batch."""

    def test_cache_get_with_precomputed_key(self) -> None:
        """_cache_get accepts a pre-computed key."""
        embedder = LiteLLMEmbedder()
        key = embedder._cache_key("test")
        embedder._cache_put("test", [1.0, 2.0])
        result = embedder._cache_get("test", key=key)
        assert result == [1.0, 2.0]

    def test_cache_put_with_precomputed_key(self) -> None:
        """_cache_put accepts a pre-computed key."""
        embedder = LiteLLMEmbedder()
        key = embedder._cache_key("test")
        embedder._cache_put("test", [1.0, 2.0], key=key)
        result = embedder._cache_get("test")
        assert result == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_embed_batch_uses_precomputed_keys(self) -> None:
        """embed_batch should compute keys once (verified by cache working correctly)."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=2, max_retries=1)

        mock_response = MagicMock()
        # Use pre-normalized vector so L2-normalization is a no-op
        mock_response.data = [{"embedding": [1.0, 0.0]}]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["hello"])

        assert result == [[1.0, 0.0]]
        # Verify it was cached
        assert embedder._cache_get("hello") == [1.0, 0.0]


class TestB12ParseResponseAcceptsDict:
    """B12: _parse_response should accept pre-parsed dicts (no JSON round-trip)."""

    def test_parse_response_with_dict(self) -> None:
        """_parse_response accepts a dict directly."""
        extractor = LLMEntityExtractor(model="test-model")
        data = {
            "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "A person"}],
            "relationships": [],
        }
        result = extractor._parse_response(data)
        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"

    def test_parse_response_with_string(self) -> None:
        """_parse_response still works with JSON strings."""
        extractor = LLMEntityExtractor(model="test-model")
        data = {
            "entities": [{"name": "Bob", "entity_type": "ORGANIZATION"}],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.entities) == 1
        assert result.entities[0].name == "Bob"

    def test_parse_response_dict_with_events(self) -> None:
        """Dict input with events parses correctly."""
        extractor = LLMEntityExtractor(model="test-model")
        data = {
            "entities": [],
            "relationships": [],
            "events": [{"description": "Meeting", "event_type": "EVENT"}],
        }
        result = extractor._parse_response(data)
        assert len(result.events) == 1

    def test_parse_response_dict_with_temporal(self) -> None:
        """Dict input with temporal info parses correctly."""
        extractor = LLMEntityExtractor(model="test-model")
        data = {
            "entities": [
                {
                    "name": "Meeting",
                    "entity_type": "EVENT",
                    "temporal": {"mentioned_at": "2024-01-01"},
                }
            ],
            "relationships": [],
        }
        result = extractor._parse_response(data)
        assert result.entities[0].temporal is not None


# =========================================================================
# B10: Embedding input deduplication
# =========================================================================


class TestB10EmbeddingDeduplication:
    """B10: Duplicate texts in embed_batch should only be embedded once."""

    @pytest.mark.asyncio
    async def test_duplicate_texts_single_api_call(self) -> None:
        """Same text appearing multiple times only gets embedded once."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=2, batch_size=100, max_retries=1)

        mock_response = MagicMock()
        # Only ONE embedding returned because dedup reduces to 1 unique text
        # Use pre-normalized vector so L2-normalization is a no-op
        mock_response.data = [{"embedding": [1.0, 0.0]}]
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_api,
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["same text", "same text", "same text"])

        # All three should get the same embedding
        assert len(result) == 3
        assert result[0] == [1.0, 0.0]
        assert result[1] == [1.0, 0.0]
        assert result[2] == [1.0, 0.0]

        # API should have been called with only 1 unique text
        call_args = mock_api.call_args
        assert len(call_args.kwargs["input"]) == 1

    @pytest.mark.asyncio
    async def test_mixed_unique_and_duplicate(self) -> None:
        """Mix of unique and duplicate texts deduplicates correctly."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=2, batch_size=100, max_retries=1)

        mock_response = MagicMock()
        # 2 unique texts → 2 pre-normalized embeddings
        mock_response.data = [
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_api,
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["alpha", "beta", "alpha"])

        assert len(result) == 3
        assert result[0] == [1.0, 0.0]  # alpha
        assert result[1] == [0.0, 1.0]  # beta
        assert result[2] == [1.0, 0.0]  # alpha (dedup)

        # Only 2 unique texts sent to API
        call_args = mock_api.call_args
        assert len(call_args.kwargs["input"]) == 2

    @pytest.mark.asyncio
    async def test_dedup_with_cache(self) -> None:
        """Deduplication works alongside cache."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=2, batch_size=100, max_retries=1)
        # Cached value is already normalized
        embedder._cache_put("cached", [1.0, 0.0])

        mock_response = MagicMock()
        # 1 unique uncached text — pre-normalized vector
        mock_response.data = [{"embedding": [0.0, 1.0]}]
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_api,
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(["cached", "new_text", "new_text"])

        assert result[0] == [1.0, 0.0]  # from cache
        assert result[1] == [0.0, 1.0]  # from API (normalized)
        assert result[2] == [0.0, 1.0]  # dedup'd from API

        # Only 1 text sent to API (new_text, deduplicated)
        call_args = mock_api.call_args
        assert len(call_args.kwargs["input"]) == 1


# =========================================================================
# B2: Batch entity upsert (replaces N+1)
# =========================================================================


class TestB2BatchEntityUpsert:
    """B2: upsert_entities_batch should handle both graph and vector in parallel."""

    @pytest.mark.asyncio
    async def test_upsert_batch_parallel(self) -> None:
        """Batch upsert uses asyncio.gather for graph+vector."""
        ns_id = uuid4()
        entity = MagicMock(spec=Entity, namespace_id=ns_id, id=uuid4())
        entities = [entity]

        graph = MagicMock()
        graph.upsert_entities_batch = AsyncMock(return_value=[(entity, True)])
        vec = MagicMock()
        vec.upsert_entities_batch = AsyncMock(return_value=[(entity, True)])

        coord = StorageCoordinator(graph=graph, vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            results = await coord.upsert_entities_batch(ns_id, entities)

        assert len(results) == 1
        graph.upsert_entities_batch.assert_awaited_once()
        vec.upsert_entities_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_batch_graph_only(self) -> None:
        """Batch upsert works with graph backend only."""
        ns_id = uuid4()
        entity = MagicMock(spec=Entity)
        entities = [entity]

        graph = MagicMock()
        graph.upsert_entities_batch = AsyncMock(return_value=[(entity, True)])

        coord = StorageCoordinator(graph=graph)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            results = await coord.upsert_entities_batch(ns_id, entities)

        assert len(results) == 1


# =========================================================================
# B8: Batch relationship storage in expansion
# =========================================================================


class TestB8BatchRelationshipsInExpansion:
    """B8: Expansion should use create_relationships_batch instead of individual calls."""

    @pytest.mark.asyncio
    async def test_store_expansion_results_batches_relationships(self) -> None:
        """store_expansion_results calls create_relationships_batch."""
        # Import the underlying function, bypassing Prefect @task
        from khora.pipelines.flows.expansion import store_expansion_results

        storage = MagicMock()
        storage.update_entity = AsyncMock()
        # #1320: returns (relationship, is_new) per edge; the flow counts via len().
        storage.create_relationships_batch = AsyncMock(side_effect=lambda rels, **kw: [(r, True) for r in rels])

        rels = [MagicMock(), MagicMock(), MagicMock()]

        # Call the raw function, not the Prefect task wrapper
        result_obj = MagicMock()
        result_obj.merged_entity_count = 0
        result_obj.entity_mapping = {}
        result_obj.inferred_relationships = rels

        stats = await store_expansion_results(result_obj, storage)

        storage.create_relationships_batch.assert_awaited_once_with(rels)
        assert stats["stored_relationships"] == 3

    @pytest.mark.asyncio
    async def test_store_expansion_no_relationships(self) -> None:
        """No relationships means no batch call."""
        from khora.pipelines.flows.expansion import store_expansion_results

        storage = MagicMock()
        storage.update_entity = AsyncMock()
        storage.create_relationships_batch = AsyncMock()

        result_obj = MagicMock()
        result_obj.merged_entity_count = 0
        result_obj.entity_mapping = {}
        result_obj.inferred_relationships = []

        await store_expansion_results(result_obj, storage)
        storage.create_relationships_batch.assert_not_awaited()


# =========================================================================
# B9: Selective entity updates in expansion
# =========================================================================


class TestB9SelectiveEntityUpdates:
    """B9: Only update entities that were actually merged, not all."""

    @pytest.mark.asyncio
    async def test_only_merged_entities_updated(self) -> None:
        """When merges occur, only merged target entities are updated."""
        from khora.pipelines.flows.expansion import store_expansion_results

        entity1 = MagicMock(spec=Entity)
        entity1.id = uuid4()
        entity2 = MagicMock(spec=Entity)
        entity2.id = uuid4()
        entity3 = MagicMock(spec=Entity)
        entity3.id = uuid4()

        storage = MagicMock()
        storage.update_entity = AsyncMock()
        storage.create_relationships_batch = AsyncMock(return_value=0)

        result_obj = MagicMock()
        result_obj.merged_entity_count = 1
        # entity1 was merged into entity2
        result_obj.entity_mapping = {entity1.id: entity2.id}
        result_obj.entities = [entity1, entity2, entity3]
        result_obj.inferred_relationships = []

        await store_expansion_results(result_obj, storage)

        # Only entity2 (the merge target) should be updated, not entity1 or entity3
        assert storage.update_entity.await_count == 1
        updated_entity = storage.update_entity.call_args_list[0][0][0]
        assert updated_entity.id == entity2.id

    @pytest.mark.asyncio
    async def test_no_merges_no_updates(self) -> None:
        """When no merges occur, no entity updates happen."""
        from khora.pipelines.flows.expansion import store_expansion_results

        storage = MagicMock()
        storage.update_entity = AsyncMock()
        storage.create_relationships_batch = AsyncMock(return_value=0)

        result_obj = MagicMock()
        result_obj.merged_entity_count = 0
        result_obj.entity_mapping = {}
        result_obj.entities = [MagicMock(), MagicMock()]
        result_obj.inferred_relationships = []

        await store_expansion_results(result_obj, storage)
        storage.update_entity.assert_not_awaited()


# =========================================================================
# B7: Shared embedder instance across documents
# =========================================================================


class TestB7SharedEmbedder:
    """B7: Shared embedder preserves cache across documents."""

    def test_shared_embedder_retains_cache(self) -> None:
        """Single embedder instance retains cached embeddings."""
        embedder = LiteLLMEmbedder()
        embedder._cache_put("entity:Alice", [0.1, 0.2])

        # Same instance used for second document
        result = embedder._cache_get("entity:Alice")
        assert result == [0.1, 0.2]

    def test_separate_embedders_no_cache_sharing(self) -> None:
        """Separate instances don't share cache (the old behavior)."""
        embedder1 = LiteLLMEmbedder()
        embedder1._cache_put("entity:Alice", [0.1, 0.2])

        embedder2 = LiteLLMEmbedder()
        result = embedder2._cache_get("entity:Alice")
        assert result is None  # Cache not shared


# =========================================================================
# Integration: Multi-batch extraction with dict passthrough (B6 + B12)
# =========================================================================


class TestMultiBatchOptimizations:
    """Integration tests for multi-batch extraction optimizations."""

    @pytest.mark.asyncio
    async def test_extract_multi_passes_dict_to_parse(self) -> None:
        """extract_multi passes section dicts directly (no JSON round-trip)."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        section_data = {
            "sections": [
                {
                    "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "Dev"}],
                    "relationships": [
                        {
                            "source_entity": "Alice",
                            "target_entity": "Acme",
                            "relationship_type": "WORKS_FOR",
                        }
                    ],
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["text about Alice"],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        assert len(results) == 1
        assert results[0].entities[0].name == "Alice"
        assert results[0].relationships[0].relationship_type == "WORKS_FOR"

    @pytest.mark.asyncio
    async def test_extract_multi_missing_sections(self) -> None:
        """Missing sections return empty ExtractionResult."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        section_data = {"sections": []}  # No sections returned
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=10, total_tokens=60)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["text1", "text2"],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        assert len(results) == 2
        assert len(results[0].entities) == 0
        assert len(results[1].entities) == 0


# =========================================================================
# Embedding batch with sub-batches and dedup (B3 + B10 integration)
# =========================================================================


class TestEmbeddingBatchDedup:
    """Integration: Large batches with deduplication and sub-batching."""

    @pytest.mark.asyncio
    async def test_large_batch_with_duplicates(self) -> None:
        """Large batch with duplicates: dedup reduces API calls."""
        embedder = LiteLLMEmbedder(model="test-model", dimension=1, batch_size=2, max_retries=1, embed_concurrency=2)

        # 4 texts, but only 2 unique after dedup
        texts = ["alpha", "beta", "alpha", "beta"]

        mock_response = MagicMock()
        # Use pre-normalized 1-D unit vectors
        mock_response.data = [
            {"embedding": [1.0]},
            {"embedding": [1.0]},
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_api,
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embedder.embed_batch(texts)

        assert len(result) == 4
        assert result[0] == [1.0]  # alpha
        assert result[1] == [1.0]  # beta
        assert result[2] == [1.0]  # alpha (dedup)
        assert result[3] == [1.0]  # beta (dedup)

        # API called once (2 unique texts fit in batch_size=2)
        assert mock_api.await_count == 1
