"""Unit tests for DYT-697: Expertise API pass-through.

Tests the full chain from MemoryLake → Engine → process_document
for expertise and extraction_config_hash parameters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.extraction.skills import (
    EntityTypeConfig,
    ExpansionConfig,
    ExpertiseConfig,
    RelationshipTypeConfig,
)
from khora.memory_lake import BatchResult, RememberResult

from .helpers import make_lake

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_expertise(*, expansion_enabled: bool = False) -> ExpertiseConfig:
    """Build a minimal ExpertiseConfig for testing."""
    return ExpertiseConfig(
        name="test-expertise",
        entity_types=[
            EntityTypeConfig(name="PERSON", description="A human person"),
            EntityTypeConfig(name="COMPANY", description="A business organization"),
        ],
        relationship_types=[
            RelationshipTypeConfig(
                name="WORKS_FOR",
                description="Employment relationship",
                source_types=["PERSON"],
                target_types=["COMPANY"],
            ),
        ],
        expansion=ExpansionConfig(enabled=expansion_enabled),
    )


# ---------------------------------------------------------------------------
# 1. Top-level exports
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    """DYT-699: ExpertiseConfig types importable from khora top-level."""

    def test_expertise_config_importable(self) -> None:
        from khora import ExpertiseConfig as EC

        assert EC is ExpertiseConfig

    def test_entity_type_config_importable(self) -> None:
        from khora import EntityTypeConfig as ETC

        assert ETC is EntityTypeConfig

    def test_relationship_type_config_importable(self) -> None:
        from khora import RelationshipTypeConfig as RTC

        assert RTC is RelationshipTypeConfig

    def test_all_exports_present(self) -> None:
        import khora

        assert "ExpertiseConfig" in khora.__all__
        assert "EntityTypeConfig" in khora.__all__
        assert "RelationshipTypeConfig" in khora.__all__


# ---------------------------------------------------------------------------
# 2. MemoryLake.remember() with expertise
# ---------------------------------------------------------------------------


class TestRememberWithExpertise:
    """DYT-700: remember() accepts expertise and extraction_config_hash."""

    @pytest.mark.asyncio
    async def test_remember_passes_expertise_to_engine(self) -> None:
        """expertise param is forwarded to the engine."""
        lake = make_lake(connected=True)
        ns_id = uuid4()
        expertise = _sample_expertise()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=3,
            entities_extracted=2,
            relationships_created=1,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember(
                "Alice works for Acme Corp",
                namespace=ns_id,
                entity_types=["PERSON", "COMPANY"],
                relationship_types=["WORKS_FOR"],
                expertise=expertise,
            )

        assert result == mock_result
        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["expertise"] is expertise

    @pytest.mark.asyncio
    async def test_remember_passes_extraction_config_hash(self) -> None:
        """extraction_config_hash param is forwarded to the engine."""
        lake = make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                extraction_config_hash="abc123def456",
            )

        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["extraction_config_hash"] == "abc123def456"

    @pytest.mark.asyncio
    async def test_remember_none_expertise_backward_compat(self) -> None:
        """Calling without expertise (None) preserves backward compatibility."""
        lake = make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result == mock_result
        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["expertise"] is None
        assert call_kwargs["extraction_config_hash"] is None


# ---------------------------------------------------------------------------
# 3. MemoryLake.remember_batch() with expertise
# ---------------------------------------------------------------------------


class TestRememberBatchWithExpertise:
    """DYT-700: remember_batch() accepts expertise and extraction_config_hash."""

    @pytest.mark.asyncio
    async def test_remember_batch_passes_expertise(self) -> None:
        """expertise param is forwarded to the engine for batch."""
        lake = make_lake(connected=True)
        ns_id = uuid4()
        expertise = _sample_expertise()

        mock_result = BatchResult(
            total=2,
            processed=2,
            skipped=0,
            failed=0,
            chunks=6,
            entities=4,
            relationships=2,
        )
        lake._engine.remember_batch = AsyncMock(return_value=mock_result)

        docs = [
            {"content": "Alice works for Acme"},
            {"content": "Bob works for Globex"},
        ]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON", "COMPANY"],
                relationship_types=["WORKS_FOR"],
                expertise=expertise,
                extraction_config_hash="batch_hash_123",
            )

        assert result == mock_result
        call_kwargs = lake._engine.remember_batch.call_args.kwargs
        assert call_kwargs["expertise"] is expertise
        assert call_kwargs["extraction_config_hash"] == "batch_hash_123"

    @pytest.mark.asyncio
    async def test_remember_batch_none_expertise_backward_compat(self) -> None:
        """Calling remember_batch without expertise preserves backward compat."""
        lake = make_lake(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(total=1, processed=1, skipped=0, failed=0, chunks=2, entities=1, relationships=0)
        lake._engine.remember_batch = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                [{"content": "test"}],
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result == mock_result
        call_kwargs = lake._engine.remember_batch.call_args.kwargs
        assert call_kwargs["expertise"] is None
        assert call_kwargs["extraction_config_hash"] is None


# ---------------------------------------------------------------------------
# 4. DocumentModel extraction_config_hash
# ---------------------------------------------------------------------------


class TestDocumentModelExtractionConfigHash:
    """DYT-701: extraction_config_hash on DocumentModel and domain Document."""

    def test_domain_document_has_extraction_config_hash(self) -> None:
        """Domain Document supports extraction_config_hash field."""
        from khora.core.models.document import Document

        doc = Document(content="test", extraction_config_hash="sha256_abc")
        assert doc.extraction_config_hash == "sha256_abc"

    def test_domain_document_defaults_to_none(self) -> None:
        """extraction_config_hash defaults to None for legacy docs."""
        from khora.core.models.document import Document

        doc = Document(content="test")
        assert doc.extraction_config_hash is None

    def test_orm_model_has_column(self) -> None:
        """DocumentModel ORM has extraction_config_hash column."""
        from khora.db.models import DocumentModel

        assert hasattr(DocumentModel, "extraction_config_hash")
        col = DocumentModel.__table__.columns["extraction_config_hash"]
        assert col.nullable is True
        assert col.type.length == 255


# ---------------------------------------------------------------------------
# 5. Expansion control via expertise
# ---------------------------------------------------------------------------


class TestExpansionControl:
    """DYT-700: Expansion enabled when expertise.expansion.enabled is True."""

    def test_expansion_config_enabled_flag(self) -> None:
        """ExpansionConfig enabled flag is accessible."""
        expertise = _sample_expertise(expansion_enabled=True)
        assert expertise.expansion.enabled is True

    def test_expansion_config_disabled_by_default(self) -> None:
        """ExpansionConfig is disabled by default."""
        expertise = _sample_expertise(expansion_enabled=False)
        assert expertise.expansion.enabled is False

    @pytest.mark.asyncio
    async def test_graphrag_batch_enables_expansion_from_expertise(self) -> None:
        """GraphRAGEngine.remember_batch() sets enable_expansion=True when
        expertise.expansion.enabled is True, even if infer_relationships=False."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        expertise = _sample_expertise(expansion_enabled=True)

        # Build a minimal GraphRAGEngine with mocked internals
        engine = object.__new__(GraphRAGEngine)
        mock_storage = AsyncMock()
        mock_storage.get_documents_by_checksums = AsyncMock(return_value={})
        mock_storage.list_entities = AsyncMock(return_value=[])
        engine._storage = mock_storage
        engine._config = type(
            "C",
            (),
            {
                "llm": type(
                    "L",
                    (),
                    {
                        "embedding_model": "text-embedding-3-small",
                        "extraction_model": "gpt-4o-mini",
                        "model": "gpt-4o-mini",
                    },
                )(),
            },
        )()
        engine._query_engine = AsyncMock()
        engine._query_engine.invalidate_caches = lambda *a: None

        mock_ingest = AsyncMock(
            return_value={
                "total_documents": 1,
                "processed_documents": 1,
                "skipped_documents": 0,
                "failed_documents": 0,
                "total_chunks": 2,
                "total_entities": 1,
                "total_relationships": 0,
                "total_inferred_relationships": 0,
            }
        )

        with patch("khora.pipelines.flows.ingest.ingest_documents", mock_ingest):
            # Note: infer_relationships=False, but expertise.expansion.enabled=True
            await engine.remember_batch(
                [{"content": "test"}],
                uuid4(),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                expertise=expertise,
                infer_relationships=False,
            )

        # The key assertion: enable_expansion should be True because
        # expertise.expansion.enabled overrides infer_relationships=False
        call_kwargs = mock_ingest.call_args.kwargs
        assert call_kwargs["enable_expansion"] is True
        assert call_kwargs["expertise"] is expertise

    @pytest.mark.asyncio
    async def test_graphrag_batch_no_expansion_when_disabled(self) -> None:
        """GraphRAGEngine.remember_batch() respects infer_relationships=False
        when expertise.expansion.enabled is also False."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        expertise = _sample_expertise(expansion_enabled=False)

        engine = object.__new__(GraphRAGEngine)
        mock_storage = AsyncMock()
        mock_storage.get_documents_by_checksums = AsyncMock(return_value={})
        mock_storage.list_entities = AsyncMock(return_value=[])
        engine._storage = mock_storage
        engine._config = type(
            "C",
            (),
            {
                "llm": type(
                    "L",
                    (),
                    {
                        "embedding_model": "text-embedding-3-small",
                        "extraction_model": "gpt-4o-mini",
                        "model": "gpt-4o-mini",
                    },
                )(),
            },
        )()
        engine._query_engine = AsyncMock()
        engine._query_engine.invalidate_caches = lambda *a: None

        mock_ingest = AsyncMock(
            return_value={
                "total_documents": 1,
                "processed_documents": 1,
                "skipped_documents": 0,
                "failed_documents": 0,
                "total_chunks": 2,
                "total_entities": 1,
                "total_relationships": 0,
                "total_inferred_relationships": 0,
            }
        )

        with patch("khora.pipelines.flows.ingest.ingest_documents", mock_ingest):
            await engine.remember_batch(
                [{"content": "test"}],
                uuid4(),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                expertise=expertise,
                infer_relationships=False,
            )

        call_kwargs = mock_ingest.call_args.kwargs
        assert call_kwargs["enable_expansion"] is False


# ---------------------------------------------------------------------------
# 6. Engine protocol compatibility
# ---------------------------------------------------------------------------


class TestEngineProtocolCompat:
    """Engines accept expertise and extraction_config_hash without breaking."""

    def test_protocol_remember_signature(self) -> None:
        """MemoryEngineProtocol.remember() includes expertise params."""
        import inspect

        from khora.engines.protocol import MemoryEngineProtocol

        sig = inspect.signature(MemoryEngineProtocol.remember)
        params = sig.parameters
        assert "expertise" in params
        assert "extraction_config_hash" in params

    def test_protocol_remember_batch_signature(self) -> None:
        """MemoryEngineProtocol.remember_batch() includes expertise params."""
        import inspect

        from khora.engines.protocol import MemoryEngineProtocol

        sig = inspect.signature(MemoryEngineProtocol.remember_batch)
        params = sig.parameters
        assert "expertise" in params
        assert "extraction_config_hash" in params


# ---------------------------------------------------------------------------
# 7. Alembic migration exists
# ---------------------------------------------------------------------------


class TestMigration:
    """DYT-701: Migration adds extraction_config_hash column."""

    def test_migration_file_exists(self) -> None:
        """Migration 015 exists and has correct revision chain."""
        import importlib

        m = importlib.import_module("khora.db.migrations.versions.015_add_extraction_config_hash")

        assert m.revision == "015_add_extraction_config_hash"
        assert m.down_revision == "014_sync_document_status_enum"
