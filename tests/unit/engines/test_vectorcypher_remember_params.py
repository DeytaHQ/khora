"""VectorCypher remember/remember_batch parameter threading + per_document.

Covers the Solomon-integration surface:

- ``expertise`` reaches ``extract_entities`` on BOTH ingest paths — the
  single-document ``remember()`` path and the streaming ``remember_batch()``
  path (callers with an individual-retry fallback depend on both).
- ``chunk_size`` overrides the configured pipeline chunk size for a single
  call on both paths (default stays the configured value when omitted).
- ``BatchResult.per_document`` carries a per-input breakdown in input order,
  INCLUDING checksum-skipped duplicates whose entry resolves to the already
  stored document's id (via ``get_documents_by_checksums``) and intra-batch
  duplicates whose entry resolves to the batch winner's id.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Document
from khora.engines.vectorcypher.engine import VectorCypherEngine

# A content string comfortably above the min_extraction_tokens gate.
_LONG_CONTENT = "Alice from Acme Robotics uses LangGraph for agent orchestration. " * 20


def _make_connected_engine() -> VectorCypherEngine:
    """Mock-connected engine (mirrors TestVectorCypherEngineRemember)."""
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
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "semantic"
    config.pipeline.chunk_size = 512
    config.pipeline.chunk_overlap = 50
    config.query.lexical_channel = "bm25"

    engine = VectorCypherEngine(config)
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()
    # Disable the word-count extraction gate so single-chunk test docs
    # still reach extract_entities.
    engine._vc_config.min_extraction_tokens = 0
    return engine


def _wire_ingest_mocks(engine: VectorCypherEngine) -> None:
    """Wire storage/embedder mocks so the real pipelines run end to end."""

    async def _echo_document(doc: Document) -> Document:
        return doc

    async def _create_chunks_batch(chunks):
        for c in chunks:
            c.id = uuid4()
        return list(chunks)

    async def _embed_batch(texts):
        return [[0.1] * 8 for _ in texts]

    engine._storage.get_document_by_checksum = AsyncMock(return_value=None)
    engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
    engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})
    engine._storage.create_document = AsyncMock(side_effect=_echo_document)
    engine._storage.update_document = AsyncMock()
    engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
    engine._embedder.embed_batch = AsyncMock(side_effect=_embed_batch)
    engine._embedder.model_name = "text-embedding-3-small"


def _make_raw_chunk(content: str) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.start_char = 0
    chunk.end_char = len(content)
    chunk.metadata = {}
    return chunk


def _make_existing_doc(**overrides) -> MagicMock:
    doc = MagicMock()
    doc.id = uuid4()
    doc.external_id = None
    doc.session_id = None
    doc.status = "completed"
    doc.chunk_count = 3
    doc.entity_count = 2
    doc.relationship_count = 1
    for key, value in overrides.items():
        setattr(doc, key, value)
    return doc


@pytest.mark.unit
class TestExpertiseReachesExtractor:
    """expertise threads through to extract_entities on both remember paths."""

    @pytest.mark.asyncio
    async def test_remember_passes_expertise_to_extract_entities(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [_make_raw_chunk(_LONG_CONTENT)]

        extract_mock = AsyncMock(return_value=([], []))
        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch("khora.pipelines.tasks.extract.extract_entities", extract_mock),
        ):
            result = await engine.remember(
                _LONG_CONTENT,
                namespace_id,
                entity_types=["PERSON", "COMPANY"],
                relationship_types=["WORKS_FOR"],
                expertise="lead_intel",
            )

        assert result.chunks_created == 1
        extract_mock.assert_awaited_once()
        assert extract_mock.call_args.kwargs["expertise"] == "lead_intel"

    @pytest.mark.asyncio
    async def test_remember_batch_passes_expertise_to_extract_entities(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [_make_raw_chunk(_LONG_CONTENT)]

        extract_mock = AsyncMock(return_value=([], []))
        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch("khora.pipelines.tasks.extract.extract_entities", extract_mock),
        ):
            result = await engine.remember_batch(
                [{"content": _LONG_CONTENT, "source": "solomon://company/1"}],
                namespace_id,
                entity_types=["PERSON", "COMPANY"],
                relationship_types=["WORKS_FOR"],
                expertise="lead_intel",
            )

        assert result.processed == 1
        extract_mock.assert_awaited_once()
        assert extract_mock.call_args.kwargs["expertise"] == "lead_intel"


@pytest.mark.unit
class TestChunkSizeOverride:
    """chunk_size overrides the configured chunker size for a single call."""

    @pytest.mark.asyncio
    async def test_remember_chunk_size_override_reaches_chunker(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker) as factory:
            await engine.remember(
                "content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_size=2000,
            )

        assert factory.call_args.kwargs["chunk_size"] == 2000

    @pytest.mark.asyncio
    async def test_remember_default_chunk_size_unchanged(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker) as factory:
            await engine.remember(
                "content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert factory.call_args.kwargs["chunk_size"] == 512

    @pytest.mark.asyncio
    async def test_remember_batch_chunk_size_override_reaches_chunker(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker) as factory:
            await engine.remember_batch(
                [{"content": "content"}],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_size=2000,
            )

        assert factory.call_args.kwargs["chunk_size"] == 2000

    @pytest.mark.asyncio
    async def test_remember_batch_legacy_chunk_size_forwarded_to_remember(self) -> None:
        engine = _make_connected_engine()
        engine._vc_config.streaming_pipeline = False
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        namespace_id = uuid4()

        remember_mock = AsyncMock(
            return_value=MagicMock(
                document_id=uuid4(),
                chunks_created=1,
                entities_extracted=0,
                relationships_created=0,
                metadata={},
            )
        )
        with patch.object(engine, "remember", remember_mock):
            await engine.remember_batch(
                [{"content": "content"}],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_size=2000,
            )

        remember_mock.assert_awaited_once()
        assert remember_mock.call_args.kwargs["chunk_size"] == 2000


@pytest.mark.unit
class TestPerDocumentBreakdown:
    """BatchResult.per_document maps every input doc to a stored document id."""

    @pytest.mark.asyncio
    async def test_per_document_includes_skipped_and_intra_batch_dup_ids(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        namespace_id = uuid4()

        new_content = _LONG_CONTENT
        dup_content = "This document already exists in the lake."
        dup_checksum = hashlib.sha256(dup_content.encode("utf-8")).hexdigest()

        existing = _make_existing_doc()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={dup_checksum: existing})

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [_make_raw_chunk(new_content)]

        extract_mock = AsyncMock(return_value=([], []))
        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch("khora.pipelines.tasks.extract.extract_entities", extract_mock),
        ):
            result = await engine.remember_batch(
                [
                    {"content": new_content, "source": "solomon://company/1"},
                    {"content": dup_content, "source": "solomon://company/2"},
                    {"content": new_content, "source": "solomon://company/3"},  # intra-batch dup of doc 0
                ],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result.processed == 1
        assert result.skipped == 2
        assert len(result.per_document) == 3

        processed_entry, db_dup_entry, intra_dup_entry = result.per_document

        # Doc 0: processed — carries the created document's id + chunk count.
        assert processed_entry["source"] == "solomon://company/1"
        assert processed_entry["skipped"] is False
        assert processed_entry["chunks"] == 1
        assert processed_entry["document_id"] is not None

        # Doc 1: checksum-skipped — carries the EXISTING document's id.
        assert db_dup_entry["source"] == "solomon://company/2"
        assert db_dup_entry["skipped"] is True
        assert db_dup_entry["chunks"] == 0
        assert db_dup_entry["document_id"] == existing.id

        # Doc 2: intra-batch duplicate — resolves to the batch winner's id.
        assert intra_dup_entry["source"] == "solomon://company/3"
        assert intra_dup_entry["skipped"] is True
        assert intra_dup_entry["document_id"] == processed_entry["document_id"]

    @pytest.mark.asyncio
    async def test_per_document_failed_doc_reports_none_id(self) -> None:
        engine = _make_connected_engine()
        _wire_ingest_mocks(engine)
        engine._storage.create_document = AsyncMock(side_effect=RuntimeError("db down"))
        namespace_id = uuid4()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            result = await engine.remember_batch(
                [{"content": "content", "source": "solomon://company/1"}],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result.failed == 1
        assert len(result.per_document) == 1
        assert result.per_document[0]["document_id"] is None
        assert result.per_document[0]["skipped"] is False

    @pytest.mark.asyncio
    async def test_per_document_legacy_path_includes_skipped_ids(self) -> None:
        engine = _make_connected_engine()
        engine._vc_config.streaming_pipeline = False
        namespace_id = uuid4()

        dup_content = "This document already exists in the lake."
        dup_checksum = hashlib.sha256(dup_content.encode("utf-8")).hexdigest()
        existing = _make_existing_doc()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={dup_checksum: existing})

        new_doc_id = uuid4()
        remember_mock = AsyncMock(
            return_value=MagicMock(
                document_id=new_doc_id,
                chunks_created=2,
                entities_extracted=1,
                relationships_created=0,
                metadata={},
            )
        )
        with patch.object(engine, "remember", remember_mock):
            result = await engine.remember_batch(
                [
                    {"content": "brand new content", "source": "solomon://company/1"},
                    {"content": dup_content, "source": "solomon://company/2"},
                ],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result.processed == 1
        assert result.skipped == 1
        assert len(result.per_document) == 2
        assert result.per_document[0]["document_id"] == new_doc_id
        assert result.per_document[0]["chunks"] == 2
        assert result.per_document[0]["skipped"] is False
        assert result.per_document[1]["document_id"] == existing.id
        assert result.per_document[1]["skipped"] is True
