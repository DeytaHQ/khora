"""Unit tests for ``chunker_info`` propagation through the VectorCypher engine.

The contract under test:

* A chunker emits ``ChunkResult.metadata = {"chunker": "<strategy>", ...}``
  (see ``khora.extraction.chunkers.base.ChunkResult`` docstring — every
  chunker MUST stamp ``metadata["chunker"]`` with its strategy name).
* The VectorCypher engine must copy that dict, verbatim, onto the
  ``TemporalChunk.chunker_info`` field before handing the chunk to the
  temporal store.

These tests pin that wire at the engine boundary. They mock the storage,
temporal store, dual-node, and embedder layers — no DB, no Neo4j, no
LiteLLM. The single observable is the ``TemporalChunk`` list passed into
``temporal_store.create_chunks_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.document import Document, DocumentStatus
from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.vectorcypher.engine import VectorCypherEngine

# ---------------------------------------------------------------------------
# Stub chunker — emits ChunkResult-shaped objects with a known metadata dict
# ---------------------------------------------------------------------------


@dataclass
class _StubChunkResult:
    """Duck-typed stand-in for ``ChunkResult``.

    The engine reads ``content``, ``start_char``, ``end_char``, and
    ``metadata``; we provide exactly those.
    """

    content: str
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class _StubChunker:
    """Returns a fixed list of ``_StubChunkResult`` instances on ``chunk()``.

    Mirrors the ``Chunker.chunk(text)`` contract from
    ``khora.extraction.chunkers.base.Chunker``.
    """

    def __init__(self, results: list[_StubChunkResult]) -> None:
        self._results = results

    def chunk(self, text: str) -> list[_StubChunkResult]:
        return list(self._results)


# ---------------------------------------------------------------------------
# Engine scaffolding (mirrors patterns in test_engine_coverage.py)
# ---------------------------------------------------------------------------


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
    config.storage.embedding_dimension = 4
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    # Extraction off — we only care about the chunk-write path.
    config.pipeline.extract_entities = False
    config.pipeline.chunking_strategy = "fixed"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 0
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _make_connected_engine() -> VectorCypherEngine:
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


def _make_document(content: str = "hello world") -> Document:
    return Document(
        id=uuid4(),
        namespace_id=uuid4(),
        content=content,
        status=DocumentStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkerInfoPropagation:
    """``ChunkResult.metadata`` must arrive on ``TemporalChunk.chunker_info``."""

    @pytest.mark.asyncio
    async def test_single_chunk_copies_chunker_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One chunk in → one ``TemporalChunk`` out with matching ``chunker_info``."""
        engine = _make_connected_engine()

        marker = {"chunker": "fixed", "size": 100}
        stub_chunker = _StubChunker(
            [_StubChunkResult(content="hello world", start_char=0, end_char=11, metadata=marker)]
        )
        monkeypatch.setattr(
            "khora.extraction.chunkers.create_chunker",
            lambda *args, **kwargs: stub_chunker,
        )

        # Capture the TemporalChunks handed to the temporal store.
        captured: list[list[TemporalChunk]] = []

        async def _capture_create_chunks(chunks: list[TemporalChunk]) -> list[TemporalChunk]:
            captured.append(list(chunks))
            for c in chunks:
                if c.id is None:
                    c.id = uuid4()
            return list(chunks)

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_capture_create_chunks)
        engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=None)
        engine._embedder.embed_batch = AsyncMock(return_value=[[0.0] * 4])
        engine._embedder.model_name = "mock-embed"
        engine._storage.update_document = AsyncMock(return_value=None)

        document = _make_document(content="hello world")
        from datetime import UTC, datetime

        await engine._process_document(
            document,
            skill_name="default",
            expertise=None,
            extraction_model=None,
            occurred_at=datetime.now(UTC),
            entity_types=[],
            relationship_types=[],
        )

        # Exactly one batch with one chunk.
        assert len(captured) == 1, f"expected 1 batch call, got {len(captured)}"
        assert len(captured[0]) == 1, f"expected 1 chunk in batch, got {len(captured[0])}"

        temporal_chunk = captured[0][0]
        assert isinstance(temporal_chunk, TemporalChunk)
        # The contract: chunker_info is a (defensively copied) duplicate of
        # ChunkResult.metadata.
        assert temporal_chunk.chunker_info == marker, (
            f"chunker_info mismatch: expected {marker}, got {temporal_chunk.chunker_info}"
        )
        # Defensive copy: mutating the original chunker metadata must not
        # leak into the persisted TemporalChunk.
        marker["after_persist"] = "leaked"
        assert "after_persist" not in temporal_chunk.chunker_info, (
            "chunker_info must be a copy of the chunker's metadata, not a shared reference"
        )

    @pytest.mark.asyncio
    async def test_multiple_chunks_each_carry_their_own_chunker_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A multi-chunk document → each ``TemporalChunk`` carries its own dict."""
        engine = _make_connected_engine()

        results = [
            _StubChunkResult(content=f"chunk {i}", start_char=i, end_char=i + 7, metadata={"chunker": "fixed", "i": i})
            for i in range(3)
        ]
        stub_chunker = _StubChunker(results)
        monkeypatch.setattr(
            "khora.extraction.chunkers.create_chunker",
            lambda *args, **kwargs: stub_chunker,
        )

        captured: list[list[TemporalChunk]] = []

        async def _capture_create_chunks(chunks: list[TemporalChunk]) -> list[TemporalChunk]:
            captured.append(list(chunks))
            for c in chunks:
                if c.id is None:
                    c.id = uuid4()
            return list(chunks)

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_capture_create_chunks)
        engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=None)
        engine._embedder.embed_batch = AsyncMock(return_value=[[0.0] * 4, [0.0] * 4, [0.0] * 4])
        engine._embedder.model_name = "mock-embed"
        engine._storage.update_document = AsyncMock(return_value=None)

        document = _make_document(content="one two three")
        from datetime import UTC, datetime

        await engine._process_document(
            document,
            skill_name="default",
            expertise=None,
            extraction_model=None,
            occurred_at=datetime.now(UTC),
            entity_types=[],
            relationship_types=[],
        )

        # All three chunks land in one batch on the default code path.
        all_chunks = [c for batch in captured for c in batch]
        assert len(all_chunks) == 3, f"expected 3 chunks, got {len(all_chunks)}"

        for i, tc in enumerate(all_chunks):
            assert tc.chunker_info == {"chunker": "fixed", "i": i}, (
                f"chunk #{i} has unexpected chunker_info: {tc.chunker_info}"
            )

    @pytest.mark.asyncio
    async def test_empty_chunker_metadata_yields_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A chunker with empty ``metadata={}`` → ``chunker_info == {}`` (not None)."""
        engine = _make_connected_engine()

        stub_chunker = _StubChunker([_StubChunkResult(content="bare content", start_char=0, end_char=12, metadata={})])
        monkeypatch.setattr(
            "khora.extraction.chunkers.create_chunker",
            lambda *args, **kwargs: stub_chunker,
        )

        captured: list[list[TemporalChunk]] = []

        async def _capture_create_chunks(chunks: list[TemporalChunk]) -> list[TemporalChunk]:
            captured.append(list(chunks))
            for c in chunks:
                if c.id is None:
                    c.id = uuid4()
            return list(chunks)

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_capture_create_chunks)
        engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=None)
        engine._embedder.embed_batch = AsyncMock(return_value=[[0.0] * 4])
        engine._embedder.model_name = "mock-embed"
        engine._storage.update_document = AsyncMock(return_value=None)

        from datetime import UTC, datetime

        await engine._process_document(
            _make_document(content="bare content"),
            skill_name="default",
            expertise=None,
            extraction_model=None,
            occurred_at=datetime.now(UTC),
            entity_types=[],
            relationship_types=[],
        )

        all_chunks = [c for batch in captured for c in batch]
        assert len(all_chunks) == 1
        tc = all_chunks[0]
        # Must be a dict (default_factory), never None.
        assert tc.chunker_info == {}
        assert isinstance(tc.chunker_info, dict)
