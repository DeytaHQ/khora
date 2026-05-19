"""Integration tests for windowed processing and submit_batch.

Eight scenarios:
1. Windowed processing splits correctly — total chunks > max_chunks_in_flight
   triggers multiple windows; all documents complete.
2. Single-document exceeds window — one document produces more chunks than
   max_chunks_in_flight; logged as a warning, still processes successfully.
3. Window=None preserves current behavior — no windowing, single pass.
4. submit_batch returns immediately — BatchHandle returned before processing;
   documents are PENDING in the DB at return time.
5. on_result callback fires per document — once per doc with correct
   DocumentResult (status, chunks_created, entities_extracted).
6. Multiple submit_batch calls — two independent batches; each callback
   receives only its own documents.
7. Crash recovery — cancel the background task; PENDING documents survive
   in the DB.
8. Entity dedup across windows — two separate submit_batch calls producing
   the same entity; upsert_entities_batch deduplicates to one entity row.

Requires Docker Compose stack (make dev) and NEO4J_INTEGRATION_TEST=1.
Connection strings can be overridden via env vars:
    KHORA_DATABASE_URL  (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
    KHORA_NEO4J_URL     (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME (default: neo4j)
    KHORA_NEO4J_PASSWORD (default: password)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from loguru import logger

from khora.config import KhoraConfig
from khora.core.models.document import DocumentStatus
from khora.engines.vectorcypher.engine import VectorCypherConfig
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import DocumentResult, Khora

# Use full embedding dimension to match the deployed PostgreSQL schema (vector(1536))
EMBED_DIM = 1536

# ---------------------------------------------------------------------------
# Registry-based extraction stub
# ---------------------------------------------------------------------------

_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def _plan_extraction(marker: str, entities: list[tuple[str, str]]) -> None:
    """Stage an entity extraction result for texts containing ``marker``."""
    _EXTRACTION_REGISTRY[marker] = ExtractionResult(
        entities=[ExtractedEntity(name=n, entity_type=t, confidence=0.99) for n, t in entities],
    )


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    out = []
    for text in texts:
        matched = next(
            (r for marker, r in _EXTRACTION_REGISTRY.items() if marker in text),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Return deterministic 1536-dim unit vectors matching the deployed schema."""
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return [unit[:] for _ in texts]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(*, chunk_size: int = 1024, chunk_overlap: int = 50) -> KhoraConfig:
    """Build KhoraConfig pointing at the Docker Compose stack."""
    database_url = os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5432/khora",
    )
    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

    config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
    config.storage.neo4j_user = neo4j_user
    config.storage.neo4j_password = neo4j_password
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False
    config.pipeline.chunk_size = chunk_size
    config.pipeline.chunk_overlap = chunk_overlap
    return config


@contextmanager
def _capture_loguru(level: str = "DEBUG"):
    """Synchronous loguru sink that captures messages for assertions."""
    messages: list[str] = []
    handler_id = logger.add(lambda record: messages.append(record), level=level, format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def _multi_chunk_content() -> str:
    """Content that produces >= 3 chunks with chunk_size=100 (≈400 chars/chunk)."""
    # ~1600 chars → 4+ chunks with character fallback (400 chars/chunk)
    sentence = "The quick brown fox jumped over the lazy dog. "
    return sentence * 36  # ~1620 characters


def _short_content(idx: int) -> str:
    """Short content that produces exactly 1 chunk (well under chunk_size)."""
    return f"Note {idx}: a brief single-sentence record about topic {idx}."


# ---------------------------------------------------------------------------
# Pytest markers — skip unless NEO4J_INTEGRATION_TEST=1
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("NEO4J_INTEGRATION_TEST"),
        reason="set NEO4J_INTEGRATION_TEST=1 to run (requires make dev)",
    ),
]


# ---------------------------------------------------------------------------
# Scenario 1–3: Windowed processing via remember_batch
# ---------------------------------------------------------------------------


class TestWindowedProcessingIntegration:
    """Tests 1–3: remember_batch windowing via VectorCypherConfig.max_chunks_in_flight."""

    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _EXTRACTION_REGISTRY.clear()
        monkeypatch.setattr(
            "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
            _stub_extract_multi,
        )
        monkeypatch.setattr(
            "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
            _stub_embed_batch,
        )

    # ------------------------------------------------------------------
    # 1. Windowed processing splits correctly
    # ------------------------------------------------------------------

    async def test_windowed_splits_correctly(self) -> None:
        """Three 1-chunk docs with max_chunks_in_flight=2 → two windows, all complete."""
        config = _make_config()
        vc_config = VectorCypherConfig(max_chunks_in_flight=2)

        kb = Khora(config, run_migrations=False, engine_kwargs={"vectorcypher_config": vc_config})
        await kb.connect()
        try:
            ns = await kb.create_namespace()
            ns_id = ns.namespace_id
            ext = uuid4().hex[:8]

            docs = [{"content": _short_content(i), "external_id": f"winsplit-{ext}-{i}"} for i in range(3)]

            with _capture_loguru("DEBUG") as messages:
                result = await kb.remember_batch(
                    docs,
                    namespace=ns_id,
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    chunk_strategy="fixed",
                )

            assert result.processed == 3, f"expected 3 processed, got {result.processed}"
            assert result.failed == 0, f"unexpected failures: {result.failed}"
            # 3 docs, max 2 per window → 2 windows logged
            assert any("Windowed processing: 3 docs" in m and "2 windows" in m for m in messages), (
                "expected windowed-processing debug log; got:\n" + "\n".join(messages)
            )
        finally:
            await kb.disconnect()

    # ------------------------------------------------------------------
    # 2. Single-document exceeds window
    # ------------------------------------------------------------------

    async def test_single_document_exceeds_window(self) -> None:
        """One doc with 3+ chunks and max_chunks_in_flight=2 → warning + success."""
        # Small chunk_size so _multi_chunk_content() produces multiple chunks
        config = _make_config(chunk_size=100, chunk_overlap=0)
        vc_config = VectorCypherConfig(max_chunks_in_flight=2)

        kb = Khora(config, run_migrations=False, engine_kwargs={"vectorcypher_config": vc_config})
        await kb.connect()
        try:
            ns = await kb.create_namespace()
            ns_id = ns.namespace_id
            ext = uuid4().hex[:8]

            docs = [{"content": _multi_chunk_content(), "external_id": f"exceeds-{ext}"}]

            with _capture_loguru("WARNING") as messages:
                result = await kb.remember_batch(
                    docs,
                    namespace=ns_id,
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    chunk_strategy="fixed",
                )

            assert result.processed == 1, f"expected 1 processed, got {result.processed}"
            assert result.failed == 0, f"unexpected failures: {result.failed}"
            assert any(
                "exceeds" in m and "max_chunks_in_flight" in m and "single-document window" in m for m in messages
            ), "expected single-document-window warning; got:\n" + "\n".join(messages)
        finally:
            await kb.disconnect()

    # ------------------------------------------------------------------
    # 9. Cross-window entity count not inflated
    # ------------------------------------------------------------------

    async def test_cross_window_entity_count_not_inflated(self) -> None:
        """BatchResult.entities must not double-count shared entities across windows.

        With max_chunks_in_flight=1, three 1-chunk documents are processed in
        three separate windows.  Two of the documents extract the same entity
        ("Alice").  upsert_entities_batch() ensures a single DB row, so
        BatchResult.entities must be 2 (Alice + unique entity from doc 0),
        not 3 (naive per-window count that double-counts Alice).
        """
        config = _make_config()
        vc_config = VectorCypherConfig(max_chunks_in_flight=1, enable_smart_resolution=False)

        kb = Khora(config, run_migrations=False, engine_kwargs={"vectorcypher_config": vc_config})
        await kb.connect()
        try:
            ns = await kb.create_namespace()
            ns_id = ns.namespace_id
            ext = uuid4().hex[:8]

            # Three docs; doc 0 extracts "UniqueEntity", docs 1 and 2 both extract "Alice"
            marker_unique = f"unique-entity-{ext}"
            marker_alice = f"alice-shared-{ext}"
            _plan_extraction(marker_unique, entities=[("UniqueEntity", "CONCEPT")])
            _plan_extraction(marker_alice, entities=[("Alice", "PERSON")])

            docs = [
                {"content": f"{marker_unique} only in doc 0", "external_id": f"cw-{ext}-0"},
                {"content": f"{marker_alice} first time", "external_id": f"cw-{ext}-1"},
                {"content": f"{marker_alice} second time", "external_id": f"cw-{ext}-2"},
            ]

            result = await kb.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON", "CONCEPT"],
                relationship_types=["KNOWS"],
                chunk_strategy="fixed",
            )

            assert result.processed == 3, f"expected 3 processed, got {result.processed}"
            assert result.failed == 0, f"unexpected failures: {result.failed}"
            # 2 unique entities extracted (UniqueEntity + Alice), not 3
            assert result.entities == 2, (
                f"expected 2 unique entities (UniqueEntity + Alice), got {result.entities}. "
                "Cross-window entity count inflation not fixed."
            )
        finally:
            await kb.disconnect()

    # ------------------------------------------------------------------
    # 3. Window=None preserves current behavior
    # ------------------------------------------------------------------

    async def test_window_none_preserves_behavior(self) -> None:
        """max_chunks_in_flight=None → no windowing; all docs process in one pass."""
        config = _make_config()
        # Default VectorCypherConfig has max_chunks_in_flight=None
        vc_config = VectorCypherConfig(max_chunks_in_flight=None)

        kb = Khora(config, run_migrations=False, engine_kwargs={"vectorcypher_config": vc_config})
        await kb.connect()
        try:
            ns = await kb.create_namespace()
            ns_id = ns.namespace_id
            ext = uuid4().hex[:8]

            docs = [{"content": _short_content(i), "external_id": f"nowin-{ext}-{i}"} for i in range(3)]

            result = await kb.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy="fixed",
            )

            assert result.processed == 3
            assert result.failed == 0
            assert result.total == 3
        finally:
            await kb.disconnect()


# ---------------------------------------------------------------------------
# Scenario 4–8: submit_batch API
# ---------------------------------------------------------------------------


class TestSubmitBatchIntegration:
    """Tests 4–8: submit_batch API and deferred processing."""

    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _EXTRACTION_REGISTRY.clear()
        monkeypatch.setattr(
            "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
            _stub_extract_multi,
        )
        monkeypatch.setattr(
            "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
            _stub_embed_batch,
        )

    @pytest.fixture
    async def kb_ns(self) -> Any:
        """Yield (kb, stable_ns_id, row_ns_id) and disconnect after each test.

        - stable_ns_id: pass to submit_batch / remember_batch (public API)
        - row_ns_id: use for direct storage queries (get_document_by_external_id,
          count_entities, etc.) — the documents are stored under this ID
        """
        config = _make_config()
        _kb = Khora(config, run_migrations=False)
        await _kb.connect()
        _kb.start_pending_processor()
        ns = await _kb.create_namespace()
        stable_id = ns.namespace_id
        row_id = await _kb.storage.resolve_namespace(stable_id)
        try:
            yield _kb, stable_id, row_id
        finally:
            await _kb.disconnect()

    # ------------------------------------------------------------------
    # 4. submit_batch returns immediately; documents PENDING at return
    # ------------------------------------------------------------------

    async def test_returns_handle_documents_pending(self, kb_ns: Any) -> None:
        """submit_batch returns a BatchHandle before processing; docs are PENDING in DB."""
        kb, stable_ns_id, row_ns_id = kb_ns
        processing_started = asyncio.Event()
        processing_can_proceed = asyncio.Event()

        original_process = kb._get_engine().process_staged_document

        async def _gated_process(doc: Any, **kwargs: Any) -> Any:
            processing_started.set()
            await processing_can_proceed.wait()
            return await original_process(doc, **kwargs)

        kb._get_engine().process_staged_document = _gated_process
        ext_id = f"pending-test-{uuid4().hex[:8]}"
        handle = None
        try:
            docs = [{"content": _short_content(0), "external_id": ext_id}]
            handle = await kb.submit_batch(
                docs,
                on_result=lambda c, t, r: None,
                namespace=stable_ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy="fixed",
            )

            assert handle is not None
            assert handle.total == 1
            assert not handle.is_done

            # Wait until processing starts, then inspect DB (use row_ns_id for direct query)
            await asyncio.wait_for(processing_started.wait(), timeout=5.0)
            doc_in_db = await kb.storage.get_document_by_external_id(ext_id, namespace_id=row_ns_id)
            assert doc_in_db is not None, "document must exist in DB after submit_batch returns"
            assert doc_in_db.status in (
                DocumentStatus.PENDING,
                DocumentStatus.PROCESSING,
            ), f"expected PENDING or PROCESSING, got {doc_in_db.status}"
        finally:
            processing_can_proceed.set()
            kb._get_engine().process_staged_document = original_process
            if handle is not None:
                await asyncio.wait_for(handle.wait(), timeout=30.0)

    # ------------------------------------------------------------------
    # 5. on_result callback fires per document
    # ------------------------------------------------------------------

    async def test_on_result_fires_per_document(self, kb_ns: Any) -> None:
        """on_result fires once per document with correct DocumentResult fields."""
        kb, stable_ns_id, _row_ns_id = kb_ns
        namespace_id = stable_ns_id
        results: list[DocumentResult] = []
        call_args: list[tuple[int, int]] = []

        def _on_result(completed: int, total: int, doc_result: DocumentResult) -> None:
            results.append(doc_result)
            call_args.append((completed, total))

        ext = uuid4().hex[:8]
        docs = [{"content": _short_content(i), "external_id": f"cb-{ext}-{i}"} for i in range(3)]

        handle = await kb.submit_batch(
            docs,
            on_result=_on_result,
            namespace=namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )
        await asyncio.wait_for(handle.wait(), timeout=60.0)

        assert len(results) == 3, f"expected 3 callbacks, got {len(results)}"
        assert all(r.success for r in results), f"unexpected failures: {results}"
        assert all(r.chunks_created >= 1 for r in results), "each doc must produce >= 1 chunk"
        assert call_args[-1] == (3, 3), f"expected last call (3,3), got {call_args[-1]}"
        assert handle.completed == 3
        assert handle.failed == 0

    # ------------------------------------------------------------------
    # 6. Multiple submit_batch calls — independent callbacks
    # ------------------------------------------------------------------

    async def test_multiple_batches_dont_interfere(self, kb_ns: Any) -> None:
        """Two concurrent submit_batch calls each receive only their own callbacks."""
        kb, stable_ns_id, _row_ns_id = kb_ns
        namespace_id = stable_ns_id
        results_a: list[DocumentResult] = []
        results_b: list[DocumentResult] = []

        ext_a = uuid4().hex[:8]
        ext_b = uuid4().hex[:8]

        handle_a = await kb.submit_batch(
            [
                {"content": _short_content(0), "external_id": f"multi-a-{ext_a}-0"},
                {"content": _short_content(1), "external_id": f"multi-a-{ext_a}-1"},
            ],
            on_result=lambda c, t, r: results_a.append(r),
            namespace=namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )
        handle_b = await kb.submit_batch(
            [
                {"content": _short_content(2), "external_id": f"multi-b-{ext_b}-0"},
            ],
            on_result=lambda c, t, r: results_b.append(r),
            namespace=namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )

        await asyncio.wait_for(asyncio.gather(handle_a.wait(), handle_b.wait()), timeout=120.0)

        assert handle_a.total == 2
        assert handle_b.total == 1
        assert len(results_a) == 2, f"batch A saw {len(results_a)} results, expected 2"
        assert len(results_b) == 1, f"batch B saw {len(results_b)} results, expected 1"
        assert handle_a.batch_id != handle_b.batch_id
        assert handle_a.failed == 0
        assert handle_b.failed == 0

    # ------------------------------------------------------------------
    # 7. Crash recovery — PENDING documents survive task cancellation
    # ------------------------------------------------------------------

    async def test_crash_recovery_pending_documents_survive(self, kb_ns: Any) -> None:
        """PENDING documents written before submit_batch returns survive cancellation."""
        kb, stable_ns_id, row_ns_id = kb_ns
        processing_started = asyncio.Event()
        original_process = kb._get_engine().process_staged_document
        ext_id = f"crash-{uuid4().hex[:8]}"

        async def _blocking_process(doc: Any, **kwargs: Any) -> Any:
            processing_started.set()
            await asyncio.sleep(3600)  # blocks until cancelled
            return await original_process(doc, **kwargs)

        kb._get_engine().process_staged_document = _blocking_process
        try:
            docs = [{"content": _short_content(0), "external_id": ext_id}]

            await kb.submit_batch(
                docs,
                on_result=lambda c, t, r: None,
                namespace=stable_ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                chunk_strategy="fixed",
            )

            # Document must be in DB immediately (durability contract)
            doc_before = await kb.storage.get_document_by_external_id(ext_id, namespace_id=row_ns_id)
            assert doc_before is not None, "document must be in DB after submit_batch returns"

            # Wait until processing has started before cancelling
            await asyncio.wait_for(processing_started.wait(), timeout=5.0)

            # Simulate crash: cancel background tasks
            for task in list(kb._bg_tasks):
                task.cancel()
            await asyncio.sleep(0.1)

            # Document must still be in DB (PENDING)
            doc_after = await kb.storage.get_document_by_external_id(ext_id, namespace_id=row_ns_id)
            assert doc_after is not None, "PENDING document must survive task cancellation"
            assert doc_after.status in (
                DocumentStatus.PENDING,
                DocumentStatus.PROCESSING,
            ), f"expected PENDING/PROCESSING after crash, got {doc_after.status}"
        finally:
            kb._get_engine().process_staged_document = original_process
            # Cancel any remaining tasks to avoid warnings
            for task in list(kb._bg_tasks):
                task.cancel()

    # ------------------------------------------------------------------
    # 8. Entity dedup across separate submit_batch calls
    # ------------------------------------------------------------------

    async def test_entity_dedup_across_separate_batches(self, kb_ns: Any) -> None:
        """Same entity in two separate submit_batch calls is deduplicated to one row."""
        kb, stable_ns_id, row_ns_id = kb_ns
        namespace_id = stable_ns_id
        marker = f"alice-dedup-{uuid4().hex[:8]}"
        _plan_extraction(marker, entities=[("Alice", "PERSON")])

        ext = uuid4().hex[:8]

        handle_a = await kb.submit_batch(
            [{"content": f"{marker} first mention", "external_id": f"dedup-a-{ext}"}],
            on_result=lambda c, t, r: None,
            namespace=namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )
        await asyncio.wait_for(handle_a.wait(), timeout=60.0)

        handle_b = await kb.submit_batch(
            [{"content": f"{marker} second mention", "external_id": f"dedup-b-{ext}"}],
            on_result=lambda c, t, r: None,
            namespace=namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            chunk_strategy="fixed",
        )
        await asyncio.wait_for(handle_b.wait(), timeout=60.0)

        assert handle_a.failed == 0
        assert handle_b.failed == 0

        graph = kb.storage.graph
        assert graph is not None, "graph backend required for entity count check"
        entity_count = await graph.count_entities(row_ns_id)
        assert entity_count == 1, f"expected 1 entity (Alice deduplicated), got {entity_count}"
