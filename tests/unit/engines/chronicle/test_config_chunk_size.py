"""Chronicle honors the configured pipeline chunk_size (#1426).

``KhoraConfig.pipeline.chunk_size`` must reach the shared ingest pipeline on
both chronicle write paths - ``remember()`` (``process_document``) and
``remember_batch()`` (``ingest_documents``) - exactly like VectorCypher and
Skeleton resolve it. A per-call ``chunk_size=`` override still wins.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config.schema import KhoraConfig
from khora.engines.chronicle.engine import ChronicleEngine

_CONFIGURED_CHUNK_SIZE = 777


def _make_engine() -> ChronicleEngine:
    """Bare engine (no connect()) with a non-default configured chunk_size."""
    config = KhoraConfig(database_url="postgresql://localhost/test")
    config.pipeline.chunk_size = _CONFIGURED_CHUNK_SIZE
    return ChronicleEngine(config)


class _StubCoordinator:
    """Minimal storage stub for the remember()/remember_batch() happy path."""

    async def get_document_by_checksum(self, *_a: Any, **_kw: Any) -> None:
        return None

    async def create_document(self, doc: Any) -> Any:
        return doc

    async def get_namespace(self, _namespace_id: UUID) -> None:
        return None


def _patch_process_document(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the shared single-doc pipeline; capture the kwargs it receives."""
    captured: dict[str, Any] = {}

    async def fake_process_document(*_a: Any, **kw: Any) -> dict[str, Any]:
        captured.update(kw)
        return {
            "document_id": str(uuid4()),
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
            "extracted_relationships": 0,
            "inferred_relationships": 0,
            "entity_ids": [],
            "chunk_ids": [],
            "phase_times": {},
        }

    monkeypatch.setattr("khora.pipelines.flows.ingest.process_document", fake_process_document)
    return captured


def _patch_ingest_documents(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the shared batch pipeline; capture the kwargs it receives."""
    captured: dict[str, Any] = {}

    async def fake_ingest_documents(
        _namespace_id: UUID,
        doc_inputs: list[dict[str, Any]],
        _storage: Any,
        **kw: Any,
    ) -> dict[str, Any]:
        captured.update(kw)
        return {
            "total_documents": len(doc_inputs),
            "processed_documents": len(doc_inputs),
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "total_inferred_relationships": 0,
            "episodes_created": 0,
            "per_document_results": [{"chunk_ids": [], "phase_times": {}} for _ in doc_inputs],
            "timing": {"staging_s": 0.0, "processing_s": 0.0, "phase_totals": {}},
        }

    monkeypatch.setattr("khora.pipelines.flows.ingest.ingest_documents", fake_ingest_documents)
    return captured


@pytest.mark.unit
class TestRememberChunkSizeResolution:
    @pytest.mark.asyncio
    async def test_configured_chunk_size_reaches_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine()
        engine._storage = _StubCoordinator()  # type: ignore[assignment]
        captured = _patch_process_document(monkeypatch)

        await engine.remember(
            "some content",
            uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert captured["chunk_size"] == _CONFIGURED_CHUNK_SIZE

    @pytest.mark.asyncio
    async def test_per_call_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine()
        engine._storage = _StubCoordinator()  # type: ignore[assignment]
        captured = _patch_process_document(monkeypatch)

        await engine.remember(
            "some content",
            uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_size=2000,
        )

        assert captured["chunk_size"] == 2000


@pytest.mark.unit
class TestRememberBatchChunkSizeResolution:
    @pytest.mark.asyncio
    async def test_configured_chunk_size_reaches_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine()
        engine._storage = _StubCoordinator()  # type: ignore[assignment]
        captured = _patch_ingest_documents(monkeypatch)

        await engine.remember_batch(
            [{"content": "doc"}],
            uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            deduplicate=False,
        )

        assert captured["chunk_size"] == _CONFIGURED_CHUNK_SIZE

    @pytest.mark.asyncio
    async def test_per_call_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine()
        engine._storage = _StubCoordinator()  # type: ignore[assignment]
        captured = _patch_ingest_documents(monkeypatch)

        await engine.remember_batch(
            [{"content": "doc"}],
            uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            deduplicate=False,
            chunk_size=2000,
        )

        assert captured["chunk_size"] == 2000
