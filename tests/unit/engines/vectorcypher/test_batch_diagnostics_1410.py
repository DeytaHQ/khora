"""#1410: remember_batch() surfaces extraction diagnostics on BatchResult.metadata.

Before the fix, the streaming batch pipeline called ``extract_entities``
without ``out_diagnostics`` and ``BatchResult.metadata`` was never populated,
so a batch whose LLM extraction entirely failed (truncation, parse failure,
retry exhaustion) returned ``entities=0`` and looked successful. These tests
pin the ADR-001 surface: failures land in ``BatchResult.metadata``
(``extraction_errors`` + ``degradations``, mirroring the #889
``RememberResult.metadata`` shape) and the happy path stays clean.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.extraction.extractors.base import ExtractionResult
from khora.khora import RememberResult
from tests.test_helpers.diagnostics import assert_no_silent_degradation

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
    config.pipeline.extraction_second_pass = False
    config.query.lexical_channel = "bm25"
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _make_streaming_engine() -> VectorCypherEngine:
    """Engine wired for the streaming batch path (mirrors test_selective_extraction_1408)."""
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()
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


# ~80 words -> 8 fixed chunks at chunk_size=10 tokens
_LONG_CONTENT = " ".join(f"sentence {i} about the shared project alpha beta gamma delta review" for i in range(8))


def _patch_extract_multi(monkeypatch, *, fail: bool) -> None:
    """Stub the LLM boundary; ``extract_entities`` itself runs for real.

    ``LLMEntityExtractor`` signals a failed chunk by returning an empty
    ``ExtractionResult`` with ``metadata["error"]`` set (truncation /
    post-retry) — see ``pipelines/tasks/extract.py`` #889.
    """
    from khora.extraction.extractors import LLMEntityExtractor

    async def fake_extract_multi(self, texts, **kwargs):
        if fail:
            return [ExtractionResult(metadata={"error": "LLM response truncated"}) for _ in texts]
        return [ExtractionResult() for _ in texts]

    monkeypatch.setattr(LLMEntityExtractor, "extract_multi", fake_extract_multi)


class TestStreamingBatchExtractionFailure:
    @pytest.mark.asyncio
    async def test_failed_extraction_surfaces_degradations(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=True)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        # The batch "succeeded" operationally (chunks stored) ...
        assert result.processed == 1
        assert result.entities == 0
        # ... but it must NOT look successful: extraction failures are visible.
        assert result.metadata["extraction_errors"] >= 1
        degradations = result.metadata["degradations"]
        assert degradations
        assert degradations[0]["component"] == "extraction.llm"
        assert degradations[0]["reason"] == "extraction_failed"
        with pytest.raises(AssertionError, match="silent degradation"):
            assert_no_silent_degradation(result)

    @pytest.mark.asyncio
    async def test_happy_path_metadata_stays_empty(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=False)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert result.processed == 1
        assert result.metadata == {}
        assert_no_silent_degradation(result)

    @pytest.mark.asyncio
    async def test_conversation_mode_aggregates_per_document_failures(self, monkeypatch) -> None:
        """Conversation mode extracts per-document; failures fold into one aggregate."""
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=True)

        # >50% of docs with occurred_at + avg content < 200 chars triggers
        # conversation mode (per-document _extract_one_doc calls).
        docs = [
            {"content": "alice met bob to review the launch", "metadata": {"occurred_at": "2026-07-01T10:00:00Z"}},
            {"content": "bob shipped the release with carol", "metadata": {"occurred_at": "2026-07-01T11:00:00Z"}},
        ]
        result = await engine.remember_batch(docs, uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"])

        assert result.processed == 2
        # One failed chunk per document, aggregated across concurrent calls.
        assert result.metadata["extraction_errors"] == 2
        assert len(result.metadata["degradations"]) == 2


class TestBatchCallSiteContract:
    @pytest.mark.asyncio
    async def test_streaming_call_site_threads_out_diagnostics(self, monkeypatch) -> None:
        """The batch call site hands extract_entities a dict and surfaces what lands in it."""
        engine = _make_streaming_engine()
        captured_kwargs: list[dict[str, Any]] = []

        async def fake_extract(chunks, **kw):
            captured_kwargs.append(kw)
            out = kw["out_diagnostics"]
            out["extraction_errors"] = 3
            out.setdefault("degradations", []).append(
                {"component": "extraction.llm", "reason": "extraction_failed", "detail": "boom", "exception": None}
            )
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        result = await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        assert len(captured_kwargs) == 1
        assert isinstance(captured_kwargs[0]["out_diagnostics"], dict)
        assert result.metadata["extraction_errors"] == 3
        assert len(result.metadata["degradations"]) == 1


class TestLegacyBatchPath:
    @pytest.mark.asyncio
    async def test_legacy_path_aggregates_remember_metadata(self) -> None:
        engine = _make_streaming_engine()
        engine._vc_config.streaming_pipeline = False
        ns = uuid4()

        degradation = {
            "component": "extraction.llm",
            "reason": "extraction_failed",
            "detail": "truncated",
            "exception": None,
        }

        async def fake_remember(content, namespace_id, **kwargs):
            return RememberResult(
                document_id=uuid4(),
                namespace_id=namespace_id,
                chunks_created=1,
                entities_extracted=0,
                relationships_created=0,
                metadata={"extraction_errors": 1, "degradations": [degradation]},
            )

        engine.remember = fake_remember  # type: ignore[method-assign]

        result = await engine.remember_batch(
            [{"content": "doc one"}, {"content": "doc two"}],
            ns,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert result.processed == 2
        assert result.metadata["extraction_errors"] == 2
        assert len(result.metadata["degradations"]) == 2
