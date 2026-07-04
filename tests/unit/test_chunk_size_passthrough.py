"""Unit tests for: chunk_size / per_document API pass-through.

Tests the Khora facade chain for the Solomon-integration parameters:
``chunk_size`` on remember()/remember_batch(), string ``expertise`` on the
facade type surface, and ``BatchResult.per_document`` propagation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.khora import BatchResult, RememberResult

from .helpers import make_kb


@pytest.mark.unit
class TestRememberChunkSizePassthrough:
    """remember() forwards chunk_size to the engine."""

    @pytest.mark.asyncio
    async def test_remember_passes_chunk_size_to_engine(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await kb.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_size=2000,
            )

        call_kwargs = kb._engine.remember.call_args.kwargs
        assert call_kwargs["chunk_size"] == 2000

    @pytest.mark.asyncio
    async def test_remember_chunk_size_defaults_to_none(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await kb.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        call_kwargs = kb._engine.remember.call_args.kwargs
        assert call_kwargs["chunk_size"] is None

    @pytest.mark.asyncio
    async def test_remember_passes_string_expertise_to_engine(self) -> None:
        """The facade forwards a string expertise (name or YAML path) as-is."""
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await kb.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                expertise="lead_intel",
            )

        call_kwargs = kb._engine.remember.call_args.kwargs
        assert call_kwargs["expertise"] == "lead_intel"


@pytest.mark.unit
class TestRememberBatchChunkSizePassthrough:
    """remember_batch() forwards chunk_size and returns per_document intact."""

    @pytest.mark.asyncio
    async def test_remember_batch_passes_chunk_size_to_engine(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(total=1, processed=1, skipped=0, failed=0, chunks=2, entities=1, relationships=0)
        kb._engine.remember_batch = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await kb.remember_batch(
                [{"content": "test"}],
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_size=2000,
            )

        call_kwargs = kb._engine.remember_batch.call_args.kwargs
        assert call_kwargs["chunk_size"] == 2000

    @pytest.mark.asyncio
    async def test_remember_batch_chunk_size_defaults_to_none(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(total=1, processed=1, skipped=0, failed=0, chunks=2, entities=1, relationships=0)
        kb._engine.remember_batch = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await kb.remember_batch(
                [{"content": "test"}],
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        call_kwargs = kb._engine.remember_batch.call_args.kwargs
        assert call_kwargs["chunk_size"] is None

    @pytest.mark.asyncio
    async def test_remember_batch_returns_per_document_from_engine(self) -> None:
        """The facade's llm_usage replace() must not drop per_document."""
        kb = make_kb(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        per_document = [
            {"document_id": doc_id, "source": "solomon://company/1", "chunks": 2, "entities": 1, "skipped": False},
        ]
        mock_result = BatchResult(
            total=1,
            processed=1,
            skipped=0,
            failed=0,
            chunks=2,
            entities=1,
            relationships=0,
            per_document=per_document,
        )
        kb._engine.remember_batch = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await kb.remember_batch(
                [{"content": "test"}],
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        assert result.per_document == per_document

    def test_batch_result_per_document_defaults_empty(self) -> None:
        result = BatchResult(total=0, processed=0, skipped=0, failed=0, chunks=0, entities=0, relationships=0)
        assert result.per_document == []
