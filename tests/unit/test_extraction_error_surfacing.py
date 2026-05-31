"""Unit tests for #889: surface LLM extraction errors on RememberResult.

Pre-fix, ``LLMEntityExtractor`` would return an
``ExtractionResult(metadata={"error": ...})`` on truncated responses or
retry exhaustion; ``extract_entities`` and downstream callers iterated
the empty ``entities`` / ``relationships`` lists and produced a
RememberResult with ``entities_extracted=0`` - looking identical to
"the chunks had no extractable entities". This file pins the fix:

1. ``extract_entities`` writes ``extraction_errors`` (int) and
   ``degradations`` (ADR-001 list) onto the caller-supplied
   ``out_diagnostics`` dict when one or more chunks errored.
2. ``RememberResult.metadata`` carries the same diagnostics so a caller
   sees the silent-degradation signal without scraping logs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.extraction.extractors.base import ExtractionResult


@pytest.mark.unit
class TestExtractEntitiesSurfacesErrors:
    """``extract_entities`` populates ``out_diagnostics`` on chunk failure."""

    async def test_vectorcypher_extraction_failure_surfaces_in_metadata(self) -> None:
        """A single failed chunk yields ``extraction_errors=1`` + one degradation.

        Mocks ``LLMEntityExtractor.extract_multi`` to return an
        ``ExtractionResult`` carrying ``metadata={"error": ...}`` -
        the same shape ``llm.py`` produces on a truncated response.
        Asserts the caller-supplied dict picks up the count + an
        ADR-001 degradation entry.
        """
        from khora.pipelines.tasks.extract import extract_entities

        mock_chunk = MagicMock()
        mock_chunk.content = "alpha beta gamma"
        mock_chunk.namespace_id = uuid4()
        mock_chunk.document_id = uuid4()
        mock_chunk.id = uuid4()
        mock_chunk.created_at = None

        failed_result = ExtractionResult(
            entities=[],
            relationships=[],
            events=[],
            metadata={"error": "truncated_response", "finish_reason": "length"},
        )

        diagnostics: dict = {}
        with patch(
            "khora.extraction.extractors.LLMEntityExtractor.extract_multi",
            new_callable=AsyncMock,
            return_value=[failed_result],
        ):
            entities, relationships = await extract_entities(
                [mock_chunk],
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
                out_diagnostics=diagnostics,
            )

        # Pre-fix: empty lists and no signal. Post-fix: empty lists
        # AND the caller sees that 1/1 chunks failed.
        assert entities == []
        assert relationships == []
        assert diagnostics["extraction_errors"] == 1
        assert diagnostics["llm_chunks"] == 1
        degradations = diagnostics["degradations"]
        assert len(degradations) == 1
        entry = degradations[0]
        assert entry["component"] == "extraction.llm"
        assert entry["reason"] == "extraction_failed"
        # detail carries the original ``error`` value so operators can
        # tell truncation apart from parse errors.
        assert "truncated_response" in entry["detail"]

    async def test_extract_entities_no_errors_writes_zero(self) -> None:
        """Happy path: ``extraction_errors=0`` and no degradations entry.

        The dict is still populated (with zero) so callers can
        distinguish "extraction ran" from "extraction was skipped".
        """
        from khora.pipelines.tasks.extract import extract_entities

        mock_chunk = MagicMock()
        mock_chunk.content = "alpha beta gamma"
        mock_chunk.namespace_id = uuid4()
        mock_chunk.document_id = uuid4()
        mock_chunk.id = uuid4()
        mock_chunk.created_at = None

        good_result = ExtractionResult(entities=[], relationships=[], events=[])

        diagnostics: dict = {}
        with patch(
            "khora.extraction.extractors.LLMEntityExtractor.extract_multi",
            new_callable=AsyncMock,
            return_value=[good_result],
        ):
            await extract_entities(
                [mock_chunk],
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
                out_diagnostics=diagnostics,
            )

        assert diagnostics["extraction_errors"] == 0
        assert diagnostics.get("degradations", []) == []

    async def test_extract_entities_without_out_diagnostics_is_backward_compatible(
        self,
    ) -> None:
        """Pre-fix callers (no ``out_diagnostics``) keep the same 2-tuple shape.

        Adding the kwarg must not break existing callers that didn't
        opt in. The function still returns ``(entities, relationships)``;
        nothing observable changes for those callers.
        """
        from khora.pipelines.tasks.extract import extract_entities

        mock_chunk = MagicMock()
        mock_chunk.content = "alpha beta gamma"
        mock_chunk.namespace_id = uuid4()
        mock_chunk.document_id = uuid4()
        mock_chunk.id = uuid4()
        mock_chunk.created_at = None

        failed_result = ExtractionResult(
            entities=[],
            relationships=[],
            events=[],
            metadata={"error": "truncated_response"},
        )

        with patch(
            "khora.extraction.extractors.LLMEntityExtractor.extract_multi",
            new_callable=AsyncMock,
            return_value=[failed_result],
        ):
            result = await extract_entities(
                [mock_chunk],
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
            )

        # Still a 2-tuple; nothing crashes despite the missing dict.
        entities, relationships = result
        assert entities == []
        assert relationships == []


@pytest.mark.unit
class TestRememberResultExtractionMetadata:
    """``_build_remember_metadata`` projects diagnostics onto a result payload."""

    def test_happy_path_metadata_is_empty(self) -> None:
        """No errors -> empty metadata, preserving pre-fix behavior."""
        from khora.engines.vectorcypher.engine import _build_remember_metadata

        metadata = _build_remember_metadata({"extraction_errors": 0, "llm_chunks": 4})
        assert metadata == {}

    def test_failures_surface_count_and_degradations(self) -> None:
        """Errors -> ``extraction_errors`` + ``degradations`` on metadata.

        Mirrors the chronicle channel-degradation convention: the same
        ADR-001 shape on the result's observability dict.
        """
        from khora.engines.vectorcypher.engine import _build_remember_metadata

        degradations = [
            {
                "component": "extraction.llm",
                "reason": "extraction_failed",
                "detail": "truncated_response",
                "exception": None,
            }
        ]
        metadata = _build_remember_metadata(
            {
                "extraction_errors": 1,
                "llm_chunks": 1,
                "degradations": degradations,
            }
        )
        assert metadata["extraction_errors"] == 1
        assert metadata["degradations"] == degradations

    def test_none_or_empty_input_returns_empty(self) -> None:
        """Pre-fix callers that don't allocate a dict still get an empty payload."""
        from khora.engines.vectorcypher.engine import _build_remember_metadata

        assert _build_remember_metadata(None) == {}
        assert _build_remember_metadata({}) == {}
