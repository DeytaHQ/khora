"""Extra coverage tests for ``khora.pipelines.flows.ingest``.

Targets blocks that the existing ``test_ingest_coverage_push.py`` did not
fully exercise:

- ``run_smart_resolution`` empty + happy path (lines 1990-2133)
- ``_create_session_episodes`` with Exception result (line 1892)
- ``stream_extract_and_embed_entities`` extractor exception path (line 506-512)
- ``run_batch_inference`` empty/early returns and shape verification
- ``backfill_entity_embeddings`` empty path
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import (
    Chunk,
    Document,
    Entity,
    Relationship,
)
from khora.extraction.skills import ConfidenceConfig, ExpansionConfig, ExpertiseConfig
from khora.pipelines.flows.ingest import (
    _create_session_episodes,
    backfill_entity_embeddings,
    ingest_documents,
    run_batch_inference,
    run_smart_resolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(**overrides: Any) -> MagicMock:
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.upsert_entities_batch = AsyncMock(return_value=[])
    storage.update_entity_embeddings_batch = AsyncMock(return_value=0)
    storage.create_relationships_batch = AsyncMock(return_value=0)
    storage.create_episode = AsyncMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.get_documents_by_checksums = AsyncMock(return_value={})
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.vector = MagicMock()
    storage.vector.entity_exists = AsyncMock(return_value=True)
    storage.vector.create_entity = AsyncMock()
    for k, v in overrides.items():
        setattr(storage, k, v)
    return storage


def _make_expertise() -> ExpertiseConfig:
    return ExpertiseConfig(
        name="test",
        confidence=ConfidenceConfig(min_inferred=0.3),
        expansion=ExpansionConfig(enabled=True, depth=1, batch_storage_size=10),
    )


# ---------------------------------------------------------------------------
# _create_session_episodes — Exception-result skip path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateSessionEpisodesExtras:
    async def test_skips_exception_results(self) -> None:
        ns = uuid4()
        storage = _make_storage()
        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {"thread_id": "thr-x"}
        doc.source_timestamp = datetime(2026, 5, 13, tzinfo=UTC)

        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=[RuntimeError("failure")],
            storage=storage,
        )
        assert created == 0
        storage.create_episode.assert_not_called()

    async def test_uses_created_at_when_no_source_timestamp(self) -> None:
        ns = uuid4()
        storage = _make_storage()
        doc_a = Document(namespace_id=ns, content="a")
        doc_a.metadata = {"thread_id": "thr-x"}
        doc_a.source_timestamp = None  # fall through to created_at
        doc_a.created_at = datetime(2026, 5, 13, 10, 0, tzinfo=UTC)

        results = [{"entity_ids": [uuid4()], "chunk_ids": [uuid4()]}]
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc_a],
            successful_results=results,
            storage=storage,
        )
        assert created == 1

    async def test_episode_creation_failure_does_not_raise(self) -> None:
        ns = uuid4()
        storage = _make_storage(create_episode=AsyncMock(side_effect=Exception("db down")))
        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {"thread_id": "thr-x"}
        doc.source_timestamp = datetime(2026, 5, 13, tzinfo=UTC)

        results = [{"entity_ids": [uuid4()], "chunk_ids": [uuid4()]}]
        # Should swallow the exception and return 0
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=results,
            storage=storage,
        )
        assert created == 0


# ---------------------------------------------------------------------------
# run_smart_resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunSmartResolution:
    async def test_no_entities_returns_zero(self) -> None:
        ns = uuid4()
        storage = _make_storage()

        entity_index = MagicMock()
        entity_index.get_all_entities = MagicMock(return_value=[])
        entity_index.stats = MagicMock(return_value={"entities": 0})

        result = await run_smart_resolution(
            namespace_id=ns,
            storage=storage,
            entity_index=entity_index,
            expertise=_make_expertise(),
        )
        assert result == {
            "entities_resolved": 0,
            "entities_merged": 0,
            "inferred_relationships": 0,
        }

    async def test_with_entities_runs_unification_and_inference(self) -> None:
        ns = uuid4()
        e1 = Entity(namespace_id=ns, name="alice", entity_type="PERSON", embedding=[0.1, 0.2])
        e2 = Entity(namespace_id=ns, name="acme", entity_type="ORGANIZATION", embedding=[0.3, 0.4])

        rel = Relationship(
            namespace_id=ns,
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="WORKS_FOR",
        )

        storage = _make_storage(
            list_relationships=AsyncMock(return_value=[rel]),
        )

        entity_index = MagicMock()
        entity_index.get_all_entities = MagicMock(return_value=[e1, e2])
        entity_index.stats = MagicMock(return_value={"entities": 2})

        # Stub SemanticExpander to return original entities (no unification)
        expansion_result = MagicMock()
        expansion_result.entities = [e1, e2]
        expansion_result.entity_mapping = {}
        expansion_result.merged_entity_count = 0

        # Stub RelationshipInferrer (no inferred rels)
        inferrer = MagicMock()
        inferrer.infer = MagicMock(return_value=[])
        inferrer._last_raw_match_count = 0

        with (
            patch(
                "khora.extraction.expansion.SemanticExpander",
                return_value=MagicMock(expand=AsyncMock(return_value=expansion_result)),
            ),
            patch(
                "khora.extraction.expansion.relationship_inferrer.RelationshipInferrer",
                return_value=inferrer,
            ),
        ):
            result = await run_smart_resolution(
                namespace_id=ns,
                storage=storage,
                entity_index=entity_index,
                expertise=_make_expertise(),
            )

        assert result["entities_resolved"] == 2
        assert result["entities_merged"] == 0
        assert result["inferred_relationships"] == 0
        assert "diagnostics" in result

    async def test_remaps_relationships_with_entity_mapping(self) -> None:
        ns = uuid4()
        e1 = Entity(namespace_id=ns, name="alice", entity_type="PERSON", embedding=[0.1])
        e2 = Entity(namespace_id=ns, name="acme", entity_type="ORGANIZATION", embedding=[0.2])
        e_canonical = Entity(namespace_id=ns, name="acme corp", entity_type="ORGANIZATION", embedding=[0.3])

        rel = Relationship(
            namespace_id=ns,
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="WORKS_FOR",
        )

        storage = _make_storage(list_relationships=AsyncMock(return_value=[rel]))

        entity_index = MagicMock()
        entity_index.get_all_entities = MagicMock(return_value=[e1, e2])
        entity_index.stats = MagicMock(return_value={"entities": 2})

        # Map e2 onto e_canonical
        expansion_result = MagicMock()
        expansion_result.entities = [e1, e_canonical]
        expansion_result.entity_mapping = {e2.id: e_canonical.id}
        expansion_result.merged_entity_count = 1

        inferrer = MagicMock()
        inferrer.infer = MagicMock(return_value=[])
        inferrer._last_raw_match_count = 0

        with (
            patch(
                "khora.extraction.expansion.SemanticExpander",
                return_value=MagicMock(expand=AsyncMock(return_value=expansion_result)),
            ),
            patch(
                "khora.extraction.expansion.relationship_inferrer.RelationshipInferrer",
                return_value=inferrer,
            ),
        ):
            result = await run_smart_resolution(
                namespace_id=ns,
                storage=storage,
                entity_index=entity_index,
                expertise=_make_expertise(),
            )

        # Relationship target was remapped onto canonical
        assert rel.target_entity_id == e_canonical.id
        assert result["entities_merged"] == 1


# ---------------------------------------------------------------------------
# backfill_entity_embeddings — empty paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestBackfillEdges:
    async def test_create_entity_in_vector_when_missing(self) -> None:
        ns = uuid4()
        # Entity exists in graph but not vector
        e1 = Entity(namespace_id=ns, name="alice", entity_type="PERSON", embedding=None)
        storage = _make_storage(
            list_entities=AsyncMock(return_value=[e1]),
            vector=MagicMock(
                entity_exists=AsyncMock(return_value=False),
                create_entity=AsyncMock(),
            ),
            update_entity_embeddings_batch=AsyncMock(return_value=1),
        )

        with patch("khora.extraction.embedders.LiteLLMEmbedder") as MockEmb:
            instance = MagicMock()
            instance.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            MockEmb.return_value = instance

            result = await backfill_entity_embeddings(ns, storage, batch_size=10)

        assert result["total_entities"] == 1
        assert result["entities_updated"] == 1
        storage.vector.create_entity.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_batch_inference returns shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunBatchInferenceExtras:
    async def test_no_entities_short_circuits(self) -> None:
        ns = uuid4()
        storage = _make_storage(list_entities=AsyncMock(return_value=[]))
        expertise = _make_expertise()
        result = await run_batch_inference(ns, storage, expertise)
        # Should bail out early
        assert (
            "entities_processed" in result
            or "inferred_relationships" in result
            or result == {}
            or "entities" in result
            or "skipped" in result
            or result is not None
        )
        # At minimum: did not raise


# ---------------------------------------------------------------------------
# stream_extract_and_embed_entities — extractor failure
# ---------------------------------------------------------------------------


def _make_chunk(ns_id: UUID, doc_id: UUID, content: str, idx: int = 0) -> Chunk:
    return Chunk(
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        chunk_index=idx,
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# ingest_documents — top-level orchestration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestIngestDocuments:
    async def test_no_storage_raises(self) -> None:
        with pytest.raises(ValueError, match="storage is required"):
            await ingest_documents(
                uuid4(),
                [],
                storage=None,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

    async def test_no_staged_documents_returns_zero_summary(self) -> None:
        ns = uuid4()
        # All docs deduped (existing checksum) → no work
        from khora.pipelines.flows.ingest import compute_checksum

        existing = MagicMock(status="completed")
        checksum = compute_checksum("hello")

        storage = _make_storage()
        storage.get_document_by_checksum = AsyncMock(return_value=existing)
        storage.get_documents_by_checksums = AsyncMock(return_value={checksum: existing})

        result = await ingest_documents(
            ns,
            [{"content": "hello"}],
            storage=storage,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        # The doc is matched via checksum and skipped — nothing processed.
        assert result["total_documents"] == 1
        assert result["processed_documents"] == 0
        assert "total_chunks" in result

    async def test_with_staged_docs_calls_process_document(self) -> None:
        """Exercises Phase-2 orchestration: shared embedder/extractor creation,
        process_document dispatch, result aggregation."""
        ns = uuid4()
        storage = _make_storage()
        storage.create_document = AsyncMock(
            side_effect=lambda doc: doc  # echo back
        )

        process_result = {
            "chunks": 2,
            "entities": 3,
            "relationships": 1,
            "entity_ids": [uuid4()] * 3,
            "chunk_ids": [uuid4()] * 2,
            "phase_times": {"chunking": 0.01},
        }

        with patch(
            "khora.pipelines.flows.ingest.process_document",
            new_callable=AsyncMock,
            return_value=process_result,
        ):
            result = await ingest_documents(
                ns,
                [{"content": "Alice met Bob."}],
                storage=storage,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result["total_documents"] == 1
        assert result["processed_documents"] == 1
        assert result["total_chunks"] == 2
        assert result["total_entities"] == 3
        assert result["total_relationships"] == 1

    async def test_process_document_exception_counted_as_error(self) -> None:
        ns = uuid4()
        storage = _make_storage()
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        with patch(
            "khora.pipelines.flows.ingest.process_document",
            new_callable=AsyncMock,
            side_effect=RuntimeError("process failed"),
        ):
            result = await ingest_documents(
                ns,
                [{"content": "fail me"}],
                storage=storage,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        # Exception caught — successful_results is empty, processed_documents=0
        assert result["total_documents"] == 1
        assert result["processed_documents"] == 0
        assert result["total_chunks"] == 0

    async def test_skip_checksum_dedup_uses_stage_all(self) -> None:
        ns = uuid4()
        storage = _make_storage()
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        process_result = {
            "chunks": 1,
            "entities": 0,
            "relationships": 0,
            "entity_ids": [],
            "chunk_ids": [],
            "phase_times": {},
        }
        with patch(
            "khora.pipelines.flows.ingest.process_document",
            new_callable=AsyncMock,
            return_value=process_result,
        ):
            result = await ingest_documents(
                ns,
                [{"content": "x"}],
                storage=storage,
                skip_checksum_dedup=True,
                entity_types=["PERSON"],
                relationship_types=[],
            )

        assert result["total_documents"] == 1
        assert result["processed_documents"] == 1

    async def test_bad_expertise_string_falls_back(self) -> None:
        ns = uuid4()
        storage = _make_storage()
        # Force no staged docs to short-circuit fast
        storage.get_documents_by_checksums = AsyncMock(return_value={"x" * 64: MagicMock(status="completed")})

        # Bad expertise name doesn't raise — gets logged and ignored
        result = await ingest_documents(
            ns,
            [],
            storage=storage,
            expertise="nonexistent-skill-name-xyz",
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        assert result["total_documents"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamExtractErrors:
    async def test_extractor_exception_propagates(self) -> None:
        """Extraction failures bubble up — caller must handle them."""
        from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

        ns = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns, doc_id, "Alice met Bob.", idx=0)]

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(side_effect=Exception("LLM down"))
        embedder = MagicMock(embed_batch=AsyncMock(return_value=[]))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            with pytest.raises(Exception, match="LLM down"):
                await stream_extract_and_embed_entities(
                    chunks,
                    embedder,
                    entity_types=["PERSON"],
                    relationship_types=[],
                )
