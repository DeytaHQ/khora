"""Unit tests for per-call chunk_strategy override.

Tests verify:
- Khora facade threads chunk_strategy to engine (remember + remember_batch)
- VectorCypher _process_document uses override when set, config when None
- Skeleton _process_document uses override when set, config when None
- Protocol and all engine signatures accept chunk_strategy
- Invalid strategy names raise ValueError via create_chunker()
- Empty string is not silently treated as None (M1 correctness)
- ChunkStrategy Literal type covers all valid values (L2)
"""

from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.extraction.chunkers import ChunkStrategy, create_chunker
from khora.khora import RememberResult

from .helpers import RESOLVE_ROW_ID, make_kb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skeleton_engine_with_mocks():
    """Build a mock SkeletonConstructionEngine with controllable _process_document."""
    from khora.engines.skeleton.engine import SkeletonConstructionEngine

    engine = SkeletonConstructionEngine.__new__(SkeletonConstructionEngine)
    engine._config = MagicMock()
    engine._config.pipeline.chunking_strategy = "semantic"
    engine._config.pipeline.chunk_size = 512
    engine._config.pipeline.chunk_overlap = 50
    engine._config.pipeline.extract_entities = False
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda d: d)
    storage.update_document = AsyncMock()
    engine._storage = storage
    engine._embedder = MagicMock()
    engine._temporal_store = MagicMock()
    engine._connected = True
    return engine


def _vectorcypher_engine_with_mocks():
    """Build a mock VectorCypherEngine with controllable internals."""
    from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine

    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._config = MagicMock()
    engine._config.pipeline.chunking_strategy = "semantic"
    engine._config.pipeline.chunk_size = 512
    engine._config.pipeline.chunk_overlap = 50
    engine._config.pipeline.extract_entities = False
    engine._vc_config = VectorCypherConfig()
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda d: d)
    storage.update_document = AsyncMock()
    engine._storage = storage
    engine._embedder = MagicMock()
    engine._temporal_store = MagicMock()
    engine._neo4j_driver = None
    engine._retriever = None
    engine._dual_nodes = None
    engine._router = None
    from khora.engines.vectorcypher.recall_cache import RecallResultCache

    engine._recall_cache = RecallResultCache(max_size=0)  # #1469: writes bump the epoch
    engine._connected = True
    return engine


# ===========================================================================
# Khora facade threading tests
# ===========================================================================


@pytest.mark.unit
class TestKhoraFacadeThreading:
    """Verify Khora passes chunk_strategy through to the engine."""

    @pytest.mark.asyncio
    async def test_remember_threads_chunk_strategy(self) -> None:
        """remember() forwards chunk_strategy to engine.remember()."""
        kb = make_kb(connected=True)
        engine = kb._engine
        ns_id = uuid4()
        engine.remember.return_value = RememberResult(
            document_id=uuid4(),
            namespace_id=RESOLVE_ROW_ID,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )

        await kb.remember(
            "hello world",
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="conversation",
        )

        engine.remember.assert_awaited_once()
        call_kwargs = engine.remember.call_args.kwargs
        assert call_kwargs["chunk_strategy"] == "conversation"

    @pytest.mark.asyncio
    async def test_remember_default_chunk_strategy_is_none(self) -> None:
        """remember() passes None when chunk_strategy not specified."""
        kb = make_kb(connected=True)
        engine = kb._engine
        engine.remember.return_value = RememberResult(
            document_id=uuid4(),
            namespace_id=RESOLVE_ROW_ID,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )

        await kb.remember(
            "hello world",
            namespace=uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        call_kwargs = engine.remember.call_args.kwargs
        assert call_kwargs["chunk_strategy"] is None

    @pytest.mark.asyncio
    async def test_remember_batch_threads_chunk_strategy(self) -> None:
        """remember_batch() forwards chunk_strategy to engine.remember_batch()."""
        from khora.khora import BatchResult

        kb = make_kb(connected=True)
        engine = kb._engine
        engine.remember_batch.return_value = BatchResult(
            total=1,
            processed=1,
            skipped=0,
            failed=0,
            chunks=2,
            entities=0,
            relationships=0,
        )

        await kb.remember_batch(
            [{"content": "doc1"}],
            namespace=uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )

        call_kwargs = engine.remember_batch.call_args.kwargs
        assert call_kwargs["chunk_strategy"] == "fixed"

    @pytest.mark.asyncio
    async def test_remember_batch_default_chunk_strategy_is_none(self) -> None:
        """remember_batch() passes None when chunk_strategy not specified."""
        from khora.khora import BatchResult

        kb = make_kb(connected=True)
        engine = kb._engine
        engine.remember_batch.return_value = BatchResult(
            total=1,
            processed=1,
            skipped=0,
            failed=0,
            chunks=2,
            entities=0,
            relationships=0,
        )

        await kb.remember_batch(
            [{"content": "doc1"}],
            namespace=uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        call_kwargs = engine.remember_batch.call_args.kwargs
        assert call_kwargs["chunk_strategy"] is None


# ===========================================================================
# VectorCypher _process_document chunker override tests
# ===========================================================================


@pytest.mark.unit
class TestVectorCypherChunkStrategy:
    """Verify VectorCypher uses the chunk_strategy override correctly."""

    @pytest.mark.asyncio
    async def test_process_document_uses_override(self) -> None:
        """_process_document creates chunker with overridden strategy."""
        engine = _vectorcypher_engine_with_mocks()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []
            mock_cc.return_value = mock_chunker

            from datetime import UTC, datetime

            from khora.core.models import Document

            doc = Document(
                namespace_id=uuid4(),
                content="test content",
            )
            doc.id = uuid4()

            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy="fixed",
            )

            mock_cc.assert_called_once()
            assert mock_cc.call_args.kwargs.get("strategy") == "fixed"

    @pytest.mark.asyncio
    async def test_process_document_falls_back_to_config(self) -> None:
        """_process_document uses config strategy when chunk_strategy is None."""
        engine = _vectorcypher_engine_with_mocks()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []
            mock_cc.return_value = mock_chunker

            from datetime import UTC, datetime

            from khora.core.models import Document

            doc = Document(
                namespace_id=uuid4(),
                content="test content",
            )
            doc.id = uuid4()

            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy=None,
            )

            mock_cc.assert_called_once()
            assert mock_cc.call_args.kwargs.get("strategy") == "semantic"  # config default


# ===========================================================================
# Skeleton _process_document chunker override tests
# ===========================================================================


@pytest.mark.unit
class TestSkeletonChunkStrategy:
    """Verify Skeleton uses the chunk_strategy override correctly."""

    @pytest.mark.asyncio
    async def test_process_document_uses_override(self) -> None:
        """_process_document creates chunker with overridden strategy."""
        engine = _skeleton_engine_with_mocks()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []
            mock_cc.return_value = mock_chunker

            from datetime import UTC, datetime

            from khora.core.models import Document

            doc = Document(
                namespace_id=uuid4(),
                content="test content",
            )
            doc.id = uuid4()

            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                chunk_strategy="recursive",
            )

            mock_cc.assert_called_once()
            assert mock_cc.call_args.kwargs.get("strategy") == "recursive"

    @pytest.mark.asyncio
    async def test_process_document_falls_back_to_config(self) -> None:
        """_process_document uses config strategy when chunk_strategy is None."""
        engine = _skeleton_engine_with_mocks()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []
            mock_cc.return_value = mock_chunker

            from datetime import UTC, datetime

            from khora.core.models import Document

            doc = Document(
                namespace_id=uuid4(),
                content="test content",
            )
            doc.id = uuid4()

            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                chunk_strategy=None,
            )

            mock_cc.assert_called_once()
            assert mock_cc.call_args.kwargs.get("strategy") == "semantic"  # config default


# ===========================================================================
# Protocol and engine signature verification
# ===========================================================================


@pytest.mark.unit
class TestProtocolSignatures:
    """Verify chunk_strategy exists in protocol and all engine signatures."""

    def test_protocol_remember_has_chunk_strategy(self) -> None:
        """MemoryEngineProtocol.remember includes chunk_strategy parameter."""
        from khora.engines.protocol import MemoryEngineProtocol

        sig = inspect.signature(MemoryEngineProtocol.remember)
        assert "chunk_strategy" in sig.parameters
        param = sig.parameters["chunk_strategy"]
        assert param.default is None

    def test_protocol_remember_batch_has_chunk_strategy(self) -> None:
        """MemoryEngineProtocol.remember_batch includes chunk_strategy parameter."""
        from khora.engines.protocol import MemoryEngineProtocol

        sig = inspect.signature(MemoryEngineProtocol.remember_batch)
        assert "chunk_strategy" in sig.parameters
        param = sig.parameters["chunk_strategy"]
        assert param.default is None

    @pytest.mark.parametrize(
        "engine_path",
        [
            "khora.engines.vectorcypher.engine.VectorCypherEngine",
            "khora.engines.skeleton.engine.SkeletonConstructionEngine",
        ],
    )
    def test_engine_remember_has_chunk_strategy(self, engine_path: str) -> None:
        """All engine implementations accept chunk_strategy in remember()."""
        module_path, cls_name = engine_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        sig = inspect.signature(cls.remember)
        assert "chunk_strategy" in sig.parameters
        assert sig.parameters["chunk_strategy"].default is None

    @pytest.mark.parametrize(
        "engine_path",
        [
            "khora.engines.vectorcypher.engine.VectorCypherEngine",
            "khora.engines.skeleton.engine.SkeletonConstructionEngine",
        ],
    )
    def test_engine_remember_batch_has_chunk_strategy(self, engine_path: str) -> None:
        """All engine implementations accept chunk_strategy in remember_batch()."""
        module_path, cls_name = engine_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        sig = inspect.signature(cls.remember_batch)
        assert "chunk_strategy" in sig.parameters
        assert sig.parameters["chunk_strategy"].default is None


# ===========================================================================
# M2: Edge case tests — invalid strategies and empty string
# ===========================================================================


@pytest.mark.unit
class TestChunkStrategyEdgeCases:
    """Edge cases for chunk_strategy validation."""

    def test_invalid_strategy_raises_value_error(self) -> None:
        """create_chunker raises ValueError for unknown strategy names."""
        with pytest.raises(ValueError, match="Unknown chunking strategy: invalid"):
            create_chunker(strategy="invalid")

    def test_empty_string_raises_value_error(self) -> None:
        """create_chunker raises ValueError for empty string (not silently treated as None)."""
        with pytest.raises(ValueError, match="Unknown chunking strategy: "):
            create_chunker(strategy="")

    @pytest.mark.asyncio
    async def test_empty_string_not_treated_as_none_in_skeleton(self) -> None:
        """M1: Empty string chunk_strategy is not silently treated as None.

        With the `is not None` check, empty string is passed through to
        create_chunker, which raises ValueError (correct behavior).
        """
        engine = _skeleton_engine_with_mocks()

        from datetime import UTC, datetime

        from khora.core.models import Document

        doc = Document(
            namespace_id=uuid4(),
            content="test content",
        )
        doc.id = uuid4()

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                chunk_strategy="",
            )

    @pytest.mark.asyncio
    async def test_empty_string_not_treated_as_none_in_vectorcypher(self) -> None:
        """M1: Empty string chunk_strategy is not silently treated as None."""
        engine = _vectorcypher_engine_with_mocks()

        from datetime import UTC, datetime

        from khora.core.models import Document

        doc = Document(
            namespace_id=uuid4(),
            content="test content",
        )
        doc.id = uuid4()

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            await engine._process_document(
                doc,
                skill_name="general_entities",
                occurred_at=datetime.now(UTC),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy="",
            )

    def test_all_valid_strategies_accepted(self) -> None:
        """All ChunkStrategy literal values produce a valid chunker."""
        for strategy in ("fixed", "semantic", "recursive", "conversation"):
            chunker = create_chunker(strategy=strategy)
            assert chunker is not None

    def test_typo_strategy_raises_value_error(self) -> None:
        """Typos in strategy name are caught."""
        with pytest.raises(ValueError, match="Unknown chunking strategy: semnatic"):
            create_chunker(strategy="semnatic")


# ===========================================================================
# L2: ChunkStrategy Literal type verification
# ===========================================================================


@pytest.mark.unit
class TestChunkStrategyType:
    """Verify ChunkStrategy is a proper Literal type."""

    def test_chunk_strategy_is_literal(self) -> None:
        """ChunkStrategy is defined as a Literal type with expected values."""
        import typing

        args = typing.get_args(ChunkStrategy)
        assert set(args) == {"fixed", "semantic", "recursive", "conversation"}

    def test_create_chunker_uses_chunk_strategy_type(self) -> None:
        """create_chunker's strategy parameter uses ChunkStrategy type."""
        hints = get_type_hints(create_chunker)
        assert hints["strategy"] is ChunkStrategy
