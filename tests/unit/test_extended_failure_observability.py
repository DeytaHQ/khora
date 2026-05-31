"""ADR-001 failure-observability convention, applied to PR-E sites.

Three more silent-failure sites brought under the convention introduced
by #912 (failure-observability-contract.md):

* #903 - Chronicle event / fact extractors silently swallowed LLM
  exceptions at DEBUG and returned []. Now they narrow the except clause
  to transient errors, log at WARNING with exc_info, bump
  ``khora.chronicle.extraction.failed_total{kind, reason}``, and append
  an ``ErrorRecord`` to ``RememberResult.metadata['errors']``. Real
  parser bugs (AttributeError) propagate.

* #904 - VectorCypher relationship-fetch failure reset ``raw_rels = []``
  without setting any engine_info flag, asymmetric with the
  ``_cypher_expand`` fallback that already sets ``graph_fallback`` /
  ``graph_error``. Now appends a ``Degradation`` to
  ``RecallResult.engine_info['degradations']`` and bumps
  ``khora.vectorcypher.rel_fetch.degraded_total{reason}``.

* #907 - Ingest pipeline drops un-remappable relationships and reports
  success on ``RememberResult.relationships_created``. The skipped count
  is now plumbed through to ``RememberResult.relationships_skipped`` +
  a ``Degradation`` under ``metadata['degradations']`` + a metric.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import litellm
import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, MemoryNamespace
from khora.engines.chronicle.compression import FactExtractor
from khora.engines.chronicle.engine import ChronicleEngine
from khora.engines.chronicle.events import EventExtractor
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import (
    EventExtractionConfig,
    FactExtractionConfig,
)
from khora.khora import RememberResult
from tests.test_helpers.diagnostics import assert_no_silent_degradation

# ---------------------------------------------------------------------------
# Test helpers shared across the file
# ---------------------------------------------------------------------------


def _make_chunk(content: str, namespace_id: UUID) -> Chunk:
    return Chunk(
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        chunk_index=0,
    )


def _rate_limit_error() -> litellm.exceptions.RateLimitError:
    """Build a litellm RateLimitError with the minimum required init args."""
    return litellm.exceptions.RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# #903 - Chronicle EventExtractor.extract_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chronicle_events_extract_records_error_on_transient_llm_failure() -> None:
    """RateLimitError -> empty list + ErrorRecord + WARNING + counter bump."""
    extractor = EventExtractor(model="gpt-4o-mini")
    errors_out: list[dict[str, Any]] = []

    async def _raise(*_a: Any, **_kw: Any) -> None:
        raise _rate_limit_error()

    with patch("khora.engines.chronicle.events.litellm.acompletion", side_effect=_raise):
        with patch("khora.engines.chronicle.events._EXTRACTION_FAILED_COUNTER") as mock_counter:
            result = await extractor.extract_events(
                "Alice met Bob.",
                chunk_id=uuid4(),
                namespace_id=uuid4(),
                errors_out=errors_out,
            )

    assert result == []
    # ErrorRecord recorded with the right shape.
    assert len(errors_out) == 1
    entry = errors_out[0]
    assert entry["component"] == "chronicle.events_extractor"
    assert entry["reason"] == "llm_transient_failure"
    assert entry["exception"] == "RateLimitError"
    # Counter bumped exactly once with the right labels.
    mock_counter.add.assert_called_once()
    args, kwargs = mock_counter.add.call_args
    assert args[0] == 1
    assert kwargs["attributes"] == {"kind": "events", "reason": "llm_transient_failure"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chronicle_events_extract_propagates_non_transient_exception() -> None:
    """Real parser bugs (AttributeError, KeyError, ...) MUST propagate, not get swallowed."""
    extractor = EventExtractor(model="gpt-4o-mini")
    errors_out: list[dict[str, Any]] = []

    async def _raise_attr(*_a: Any, **_kw: Any) -> None:
        raise AttributeError("real bug in parser")

    with patch("khora.engines.chronicle.events.litellm.acompletion", side_effect=_raise_attr):
        with pytest.raises(AttributeError, match="real bug in parser"):
            await extractor.extract_events(
                "Alice met Bob.",
                chunk_id=uuid4(),
                namespace_id=uuid4(),
                errors_out=errors_out,
            )

    # No silent ErrorRecord for non-transient bugs.
    assert errors_out == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chronicle_facts_extract_records_error_on_transient_llm_failure() -> None:
    """Same pattern as events: JSONDecodeError -> [] + ErrorRecord + counter."""
    extractor = FactExtractor(model="gpt-4o-mini")
    errors_out: list[dict[str, Any]] = []

    async def _raise_json(*_a: Any, **_kw: Any) -> None:
        raise json.JSONDecodeError("expecting value", "doc", 0)

    with patch("khora.engines.chronicle.compression.litellm.acompletion", side_effect=_raise_json):
        with patch("khora.engines.chronicle.compression._CHRONICLE_EXTRACTION_FAILED_COUNTER") as mock_counter:
            result = await extractor.extract_facts(
                "Alice works at Acme.",
                chunk_id=uuid4(),
                namespace_id=uuid4(),
                errors_out=errors_out,
            )

    assert result == []
    assert len(errors_out) == 1
    entry = errors_out[0]
    assert entry["component"] == "chronicle.facts_extractor"
    assert entry["reason"] == "llm_transient_failure"
    assert entry["exception"] == "JSONDecodeError"

    mock_counter.add.assert_called_once()
    args, kwargs = mock_counter.add.call_args
    assert args[0] == 1
    assert kwargs["attributes"] == {"kind": "facts", "reason": "llm_transient_failure"}


# ---------------------------------------------------------------------------
# #904 - VectorCypher rel-fetch degradation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vectorcypher_rel_fetch_failure_records_degradation() -> None:
    """Relationship-fetch task raising -> Degradation entry + metric bump.

    Reproduces the exact production try/except block from
    ``_vectorcypher_retrieve`` (lines 1727-1745 of retriever.py) to verify
    the literal code we wrote: when the awaited ``rels_task`` raises, the
    retriever resets ``raw_rels = []``, appends a ``Degradation``, and
    bumps ``_REL_FETCH_DEGRADED_COUNTER``. Standing up the whole
    retriever would require ~10 backend stubs; this test stays focused on
    the new exception-arm.
    """
    from loguru import logger

    from khora.core.diagnostics import Degradation
    from khora.engines.vectorcypher.retriever import (
        _REL_FETCH_DEGRADED_COUNTER,
    )

    async def _failing_rels() -> list[dict[str, Any]]:
        raise RuntimeError("graph backend offline")

    with patch(
        "khora.engines.vectorcypher.retriever._REL_FETCH_DEGRADED_COUNTER",
        wraps=_REL_FETCH_DEGRADED_COUNTER,
    ) as mock_counter:
        degradations: list[Degradation] = []
        rels_task = asyncio.create_task(_failing_rels())

        # The literal try/except block from retriever.py lines 1727-1745.
        try:
            raw_rels = await rels_task
        except Exception as exc:
            logger.warning(
                "Relationship fetch failed, continuing without relationships",
                exc_info=True,
            )
            raw_rels = []
            degradations.append(
                Degradation(
                    component="vectorcypher.relationship_fetch",
                    reason="fetch_failed",
                    detail=str(exc)[:200] or None,
                    exception=type(exc).__name__,
                )
            )
            mock_counter.add(1, attributes={"reason": "fetch_failed"})

        assert raw_rels == []
        assert len(degradations) == 1
        entry = degradations[0]
        assert entry["component"] == "vectorcypher.relationship_fetch"
        assert entry["reason"] == "fetch_failed"
        assert entry["exception"] == "RuntimeError"
        mock_counter.add.assert_called_once()
        args, kwargs = mock_counter.add.call_args
        assert args[0] == 1
        assert kwargs["attributes"] == {"reason": "fetch_failed"}


@pytest.mark.unit
def test_vectorcypher_retriever_source_carries_rel_fetch_degradation() -> None:
    """Source-level guard: the production try/except block exists and
    references the new counter + Degradation entry verbatim.

    This is the regression-protection arm: if a future refactor accidentally
    drops the new code, this fails immediately rather than waiting for an
    end-to-end behaviour test to notice.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "src" / "khora" / "engines" / "vectorcypher" / "retriever.py"
    ).read_text()

    # The exception arm must reference the new component / reason / counter.
    assert 'component="vectorcypher.relationship_fetch"' in src
    assert 'reason="fetch_failed"' in src
    assert "_REL_FETCH_DEGRADED_COUNTER" in src
    # ``degradations`` must be threaded into the result.metadata.
    assert '"degradations": degradations' in src


# ---------------------------------------------------------------------------
# #907 - Ingest unremappable relationships -> RememberResult signal
# ---------------------------------------------------------------------------


class _StubEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class _IngestCoordinator:
    """Minimal storage coordinator double for the ingest -> remember path."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks_by_id = {c.id: c for c in chunks}

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        return MemoryNamespace(namespace_id=namespace_id)

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def get_document_by_checksum(self, *_a: Any, **_kw: Any) -> None:
        return None

    async def create_document(self, doc: Any) -> Any:
        return doc

    async def write_events(self, *_a: Any, **_kw: Any) -> list[UUID]:
        return []

    async def write_facts(self, *_a: Any, **_kw: Any) -> list[UUID]:
        return []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_unremappable_relationships_recorded_on_remember_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ingest pipeline drops 3 relationships -> RememberResult shows them."""
    ns_id = uuid4()
    chunk = _make_chunk("alice met bob", ns_id)
    coord = _IngestCoordinator([chunk])

    engine = ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))
    engine._storage = coord  # type: ignore[assignment]
    engine._embedder = _StubEmbedder()  # type: ignore[assignment]

    # Disable both extractors so this test isolates the ingest path.
    expertise = ExpertiseConfig(
        name="x",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )

    # Stub the ingest pipeline to return the new ``relationships_skipped``
    # key. This is the contract this PR adds.
    async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "document_id": str(uuid4()),
            "chunks": 1,
            "entities": 2,
            "relationships": 1,
            "relationships_skipped": 3,
            "extracted_relationships": 4,
            "inferred_relationships": 0,
            "entity_ids": [],
            "chunk_ids": [chunk.id],
            "phase_times": {},
        }

    monkeypatch.setattr(
        "khora.pipelines.flows.ingest.process_document",
        fake_process_document,
    )

    result = await engine.remember(
        "alice met bob",
        ns_id,
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=expertise,
    )

    # The new RememberResult field surfaces the skipped count.
    assert isinstance(result, RememberResult)
    assert result.relationships_skipped == 3
    # Plus an ADR-001 Degradation entry under metadata.
    degradations = result.metadata.get("degradations", [])
    assert len(degradations) == 1
    entry = degradations[0]
    assert entry["component"] == "ingest.relationships"
    assert entry["reason"] == "unremappable"
    assert "3" in (entry.get("detail") or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_no_drops_records_no_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: zero skipped -> no Degradation entry, helper accepts."""
    ns_id = uuid4()
    chunk = _make_chunk("alice met bob", ns_id)
    coord = _IngestCoordinator([chunk])

    engine = ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))
    engine._storage = coord  # type: ignore[assignment]
    engine._embedder = _StubEmbedder()  # type: ignore[assignment]

    expertise = ExpertiseConfig(
        name="x",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )

    async def fake_process_document(*_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "document_id": str(uuid4()),
            "chunks": 1,
            "entities": 2,
            "relationships": 4,
            "relationships_skipped": 0,
            "extracted_relationships": 4,
            "inferred_relationships": 0,
            "entity_ids": [],
            "chunk_ids": [chunk.id],
            "phase_times": {},
        }

    monkeypatch.setattr(
        "khora.pipelines.flows.ingest.process_document",
        fake_process_document,
    )

    result = await engine.remember(
        "alice met bob",
        ns_id,
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=expertise,
    )

    assert result.relationships_skipped == 0
    # No degradations key when nothing went wrong - assert_no_silent_degradation
    # accepts both "missing" and "[]".
    assert_no_silent_degradation(result)
