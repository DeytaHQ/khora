"""#1408: skeleton PageRank is the SINGLE core-chunk selector on the VC path.

Before the fix, the VectorCypher engine selected core chunks via skeleton
PageRank (``select_core_chunk_ids``) and then called ``extract_entities``
without ``selective_extraction``, so ``ChunkImportanceScorer`` re-selected
the top 70% of the already-selected 70% (~0.49 effective LLM coverage).
These tests pin:

- ``ChunkImportanceScorer`` never runs on the VC single-doc or streaming
  batch paths (the skeleton selection IS the selective step).
- The streaming batch call sites pass ``selective_extraction=False``.
- ``config.pipeline.selective_extraction=False`` now gates the skeleton
  selection itself: every chunk goes to LLM extraction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.temporal import TemporalChunk
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.extraction.extractors.base import ExtractionResult

pytestmark = pytest.mark.unit


def _make_config() -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "password"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    config.llm.max_tokens = 4096
    config.llm.extraction_wave_size = 20
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "fixed"
    config.pipeline.chunk_size = 10
    config.pipeline.chunk_overlap = 0
    config.pipeline.ketrag_skeleton_channel = False
    config.pipeline.selective_extraction = True
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _make_engine() -> VectorCypherEngine:
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()
    return engine


def _make_streaming_engine() -> VectorCypherEngine:
    """Engine wired for the streaming batch path (mirrors test_engine_coverage)."""
    engine = _make_engine()
    engine._vc_config.min_extraction_tokens = 0

    engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
    engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

    async def _create_document(doc):
        return doc

    engine._storage.create_document = AsyncMock(side_effect=_create_document)
    engine._storage.update_document = AsyncMock(return_value=None)

    async def _create_chunks(chunks):
        for c in chunks:
            if c.id is None:
                c.id = uuid4()
        return list(chunks)

    engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks)
    engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=None)
    engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock(return_value=None)

    async def _embed_batch(texts):
        return [[0.0] * 4 for _ in texts]

    engine._embedder.embed_batch = AsyncMock(side_effect=_embed_batch)
    engine._embedder.model_name = "mock-embed"
    return engine


def _make_temporal_chunks(n: int, ns) -> list[TemporalChunk]:
    doc_id = uuid4()
    return [
        TemporalChunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=doc_id,
            content=f"topic {i} alpha beta gamma delta shared keyword project meeting notes",
        )
        for i in range(n)
    ]


# ~80 words -> 8 fixed chunks at chunk_size=10 tokens
_LONG_CONTENT = " ".join(f"sentence {i} about the shared project alpha beta gamma delta review" for i in range(8))


def _patch_scorer_spy(monkeypatch) -> MagicMock:
    """Spy on ChunkImportanceScorer.select_for_extraction (must never fire)."""
    from khora.extraction.importance import ChunkImportanceScorer

    spy = MagicMock(side_effect=AssertionError("ChunkImportanceScorer must not run on the VC path (#1408)"))
    monkeypatch.setattr(ChunkImportanceScorer, "select_for_extraction", spy)
    return spy


def _patch_extract_multi(monkeypatch) -> list[list[str]]:
    """Stub the LLM boundary; extract_entities itself runs for real."""
    from khora.extraction.extractors import LLMEntityExtractor

    captured: list[list[str]] = []

    async def fake_extract_multi(self, texts, **kwargs):
        captured.append(list(texts))
        return [ExtractionResult() for _ in texts]

    monkeypatch.setattr(LLMEntityExtractor, "extract_multi", fake_extract_multi)
    return captured


class TestImportanceScorerNeverRunsOnVCPath:
    @pytest.mark.asyncio
    async def test_single_doc_skeleton_path(self, monkeypatch) -> None:
        """_run_skeleton_extraction runs real extract_entities without re-selection."""
        engine = _make_engine()
        scorer_spy = _patch_scorer_spy(monkeypatch)
        captured = _patch_extract_multi(monkeypatch)

        ns = uuid4()
        chunks = _make_temporal_chunks(6, ns)
        result = await engine._run_skeleton_extraction(
            chunks, ns, entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result == (0, 0)
        scorer_spy.assert_not_called()
        # The skeleton selected a strict subset (max(1, int(6 * 0.7)) == 4)
        # and all of it reached the LLM boundary.
        assert len(captured) == 1
        assert 1 <= len(captured[0]) < 6

    @pytest.mark.asyncio
    async def test_streaming_batch_path(self, monkeypatch) -> None:
        """remember_batch (streaming) runs real extract_entities without re-selection."""
        engine = _make_streaming_engine()
        scorer_spy = _patch_scorer_spy(monkeypatch)
        captured = _patch_extract_multi(monkeypatch)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result.processed == 1
        scorer_spy.assert_not_called()
        assert len(captured) == 1  # extraction actually ran


class TestBatchCallSiteContract:
    @pytest.mark.asyncio
    async def test_streaming_batch_passes_selective_extraction_false(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        captured_kwargs: list[dict] = []

        async def fake_extract(chunks, **kw):
            captured_kwargs.append(kw)
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result.processed == 1
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["selective_extraction"] is False


class TestSelectiveExtractionConfigGate:
    @pytest.mark.asyncio
    async def test_config_false_sends_all_chunks_to_llm(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        engine._config.pipeline.selective_extraction = False
        captured_chunks: list[list] = []

        async def fake_extract(chunks, **kw):
            captured_chunks.append(list(chunks))
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result.chunks > 2  # gate is meaningful only above the small-doc fast path
        assert len(captured_chunks) == 1
        assert len(captured_chunks[0]) == result.chunks

    @pytest.mark.asyncio
    async def test_config_true_skeleton_selects_subset(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        engine._config.pipeline.selective_extraction = True
        captured_chunks: list[list] = []

        async def fake_extract(chunks, **kw):
            captured_chunks.append(list(chunks))
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result.chunks > 2
        assert len(captured_chunks) == 1
        # skeleton_core_ratio=0.70 selects a strict subset
        assert 1 <= len(captured_chunks[0]) < result.chunks
