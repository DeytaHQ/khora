"""Unit tests for: external_id pass-through.

Tests the full chain from Khora → Engine for the external_id parameter.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.khora import BatchResult, RememberResult

from .helpers import RESOLVE_ROW_ID, make_kb

# ---------------------------------------------------------------------------
# 1. Khora.remember() with external_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberWithExternalId:
    """remember() accepts and forwards external_id."""

    @pytest.mark.asyncio
    async def test_remember_passes_external_id_to_engine(self) -> None:
        """external_id param is forwarded to the engine."""
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=RESOLVE_ROW_ID,
            chunks_created=3,
            entities_extracted=2,
            relationships_created=1,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        result = await kb.remember(
            "Alice works for Acme Corp",
            namespace=ns_id,
            entity_types=["PERSON", "COMPANY"],
            relationship_types=["WORKS_FOR"],
            external_id="ext-123",
        )

        assert result == mock_result
        call_kwargs = kb._engine.remember.call_args.kwargs
        assert call_kwargs["external_id"] == "ext-123"

    @pytest.mark.asyncio
    async def test_remember_without_external_id_backward_compat(self) -> None:
        """Calling without external_id omits it from engine kwargs (backward compat)."""
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=RESOLVE_ROW_ID,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        result = await kb.remember(
            "test content",
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert result == mock_result
        call_kwargs = kb._engine.remember.call_args.kwargs
        assert call_kwargs["external_id"] is None


# ---------------------------------------------------------------------------
# 2. Khora.remember_batch() with external_id in doc dicts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchWithExternalId:
    """remember_batch() passes doc dicts through (including external_id)."""

    @pytest.mark.asyncio
    async def test_remember_batch_passes_docs_with_external_id(self) -> None:
        """Doc dicts with external_id are forwarded unchanged to the engine."""
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(
            total=2,
            processed=2,
            skipped=0,
            failed=0,
            chunks=4,
            entities=2,
            relationships=1,
        )
        kb._engine.remember_batch = AsyncMock(return_value=mock_result)

        docs = [
            {"content": "Alice works for Acme", "external_id": "ext-1"},
            {"content": "Bob works for Globex", "external_id": "ext-2"},
        ]

        result = await kb.remember_batch(
            docs,
            namespace=ns_id,
            entity_types=["PERSON", "COMPANY"],
            relationship_types=["WORKS_FOR"],
        )

        assert result == mock_result
        call_args = kb._engine.remember_batch.call_args
        passed_docs = call_args.args[0]
        assert passed_docs[0]["external_id"] == "ext-1"
        assert passed_docs[1]["external_id"] == "ext-2"

    @pytest.mark.asyncio
    async def test_remember_batch_mixed_docs(self) -> None:
        """Batch with some docs having external_id and some without."""
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(
            total=2,
            processed=2,
            skipped=0,
            failed=0,
            chunks=4,
            entities=2,
            relationships=1,
        )
        kb._engine.remember_batch = AsyncMock(return_value=mock_result)

        docs = [
            {"content": "Alice works for Acme", "external_id": "ext-1"},
            {"content": "Bob works for Globex"},
        ]

        result = await kb.remember_batch(
            docs,
            namespace=ns_id,
            entity_types=["PERSON", "COMPANY"],
            relationship_types=["WORKS_FOR"],
        )

        assert result == mock_result
        call_args = kb._engine.remember_batch.call_args
        passed_docs = call_args.args[0]
        assert passed_docs[0]["external_id"] == "ext-1"
        assert "external_id" not in passed_docs[1]


# ---------------------------------------------------------------------------
# 3. Protocol and engine signature verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExternalIdSignatures:
    """Verify external_id exists in protocol and engine signatures."""

    def test_protocol_remember_has_external_id(self) -> None:
        """MemoryEngineProtocol.remember includes external_id parameter."""
        from khora.engines.protocol import MemoryEngineProtocol

        sig = inspect.signature(MemoryEngineProtocol.remember)
        assert "external_id" in sig.parameters
        param = sig.parameters["external_id"]
        assert param.default is None

    def test_vectorcypher_remember_has_external_id(self) -> None:
        """VectorCypherEngine.remember includes external_id parameter."""
        from khora.engines.vectorcypher.engine import VectorCypherEngine

        sig = inspect.signature(VectorCypherEngine.remember)
        assert "external_id" in sig.parameters
        param = sig.parameters["external_id"]
        assert param.default is None

    def test_khora_remember_has_external_id(self) -> None:
        """Khora.remember includes external_id parameter."""
        from khora.khora import Khora

        sig = inspect.signature(Khora.remember)
        assert "external_id" in sig.parameters
        param = sig.parameters["external_id"]
        assert param.default is None


# ---------------------------------------------------------------------------
# 4. Domain model accepts external_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentModelExternalId:
    """Document domain model supports external_id."""

    def test_document_accepts_external_id(self) -> None:
        """Document can be constructed with external_id."""
        from khora.core.models.document import Document

        doc = Document(content="test", external_id="ext-abc")
        assert doc.external_id == "ext-abc"

    def test_document_external_id_defaults_to_none(self) -> None:
        """Document external_id defaults to None for backward compatibility."""
        from khora.core.models.document import Document

        doc = Document(content="test")
        assert doc.external_id is None
