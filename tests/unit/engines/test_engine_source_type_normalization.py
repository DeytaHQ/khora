"""Engine-level ``source_type`` normalization for per-doc batch dicts.

``documents.source_type`` is NOT NULL as of migration 055. The engines that
build their own per-document input dicts changed
``doc_data.get("source_type", source_type)`` to
``doc_data.get("source_type") or source_type`` so a doc dict carrying an
explicit ``None`` (or ``""``) inherits the batch-level value instead of
forwarding the falsy one. ``dict.get(key, default)`` returns ``None`` when the
key is *present* with value ``None`` — the default only fires on key absence —
which is the whole bug.

All five engine sites are covered here, table-driven:

===================================================  ==========================
site                                                 seam the test captures at
===================================================  ==========================
``ChronicleEngine.remember_batch``                   ``ingest_documents``
``SkeletonConstructionEngine.remember_batch``        ``Document(...)``
``VectorCypherEngine._remember_batch_impl`` (stream) ``Document(...)``
``VectorCypherEngine._remember_batch_impl`` (direct) ``self.remember(...)``
``VectorCypherEngine._remember_batch_legacy``        ``self.remember(...)``
===================================================  ==========================

Each test captures the value at the seam and then stops the flow, so no
extraction, embedding or storage work runs.

Worth being precise about what this layer buys, because it is easy to
overstate: it is **not** the thing standing between a caller and an
``IntegrityError``. These dicts flow into ``pipelines/flows/ingest.py``, whose
own ``or "manual"`` already rules out NULL (covered in
``tests/unit/test_pipelines_ingest.py``), and the four ``Khora`` entry points
normalize before any engine is reached. What the engine normalization decides
is the *value*: without it, a doc dict carrying ``source_type=None`` silently
lands as ``'manual'`` — the ingest-pipeline default — instead of the
batch-level value the caller passed. The reviewer classed these sites as not
publicly reachable; they are covered anyway because pinning them is cheaper
than re-deriving the reachability argument every time someone edits one.
"""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

FALSY = [None, ""]


def _mock_config() -> MagicMock:
    """Minimal config for constructing an engine — nothing is read for real."""
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = None
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.llm.model = "gpt-4o-mini"
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_dimension = 1536
    config.llm.extraction_model = None
    config.llm.timeout = 30
    config.telemetry_database_url = None
    # Real values, not MagicMocks: the batch paths build a chunker before they
    # build a Document, and create_chunker() rejects an unknown strategy.
    config.pipeline.chunking_strategy = "fixed"
    config.pipeline.chunk_size = 512
    config.pipeline.chunk_overlap = 50
    return config


class _StopAfterCapture(Exception):
    """Sentinel: the value under test is captured, nothing past here matters."""


def _capturing_document(captured: list[dict]):
    """A ``Document`` stand-in that records kwargs then aborts the flow."""

    def _factory(**kwargs):
        captured.append(kwargs)
        raise _StopAfterCapture

    return _factory


# ---------------------------------------------------------------------------
# Chronicle — captures at ingest_documents
# ---------------------------------------------------------------------------


async def _capture_chronicle_doc_inputs(documents: list[dict], **kwargs) -> list[dict]:
    """Run ``ChronicleEngine.remember_batch`` far enough to see ``doc_inputs``."""
    from khora.engines.chronicle.engine import ChronicleEngine

    engine = ChronicleEngine(_mock_config())
    engine._connected = True
    engine._storage = AsyncMock()

    captured: list[list[dict]] = []

    async def _fake_ingest(namespace_id, doc_inputs, storage, **_kwargs):
        captured.append(doc_inputs)
        raise _StopAfterCapture

    with (
        patch("khora.pipelines.flows.ingest.ingest_documents", side_effect=_fake_ingest),
        patch("khora.engines.chronicle.engine.LiteLLMEmbedder", MagicMock()),
    ):
        with pytest.raises(_StopAfterCapture):
            await engine.remember_batch(
                documents,
                namespace_id=uuid4(),
                deduplicate=False,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                **kwargs,
            )

    assert captured, "ingest_documents was never reached"
    return captured[0]


@pytest.mark.unit
class TestChronicleRememberBatchSourceType:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_falsy_per_doc_source_type_inherits_batch_value(self, falsy) -> None:
        doc_inputs = await _capture_chronicle_doc_inputs(
            [{"content": "body", "source_type": falsy}],
            source_type="slack",
        )

        assert doc_inputs[0]["source_type"] == "slack"

    @pytest.mark.asyncio
    async def test_explicit_per_doc_source_type_still_wins(self) -> None:
        """The normalization must not swallow a real per-doc override."""
        doc_inputs = await _capture_chronicle_doc_inputs(
            [{"content": "body", "source_type": "file"}],
            source_type="slack",
        )

        assert doc_inputs[0]["source_type"] == "file"


# ---------------------------------------------------------------------------
# Skeleton — captures at Document(...)
# ---------------------------------------------------------------------------


async def _capture_skeleton_document(documents: list[dict], **kwargs) -> dict:
    from khora.engines.skeleton.engine import SkeletonConstructionEngine

    engine = SkeletonConstructionEngine(_mock_config())
    engine._connected = True
    # The batch path fetches these collaborators before building any Document;
    # without them the flow raises before reaching the seam under test.
    engine._embedder = MagicMock()
    engine._temporal_store = AsyncMock()
    storage = AsyncMock()
    storage.get_documents_by_checksums = AsyncMock(return_value={})
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    engine._storage = storage

    captured: list[dict] = []
    with (
        patch("khora.engines.skeleton.engine.Document", _capturing_document(captured)),
        patch("khora.engines.skeleton.engine.LiteLLMEmbedder", MagicMock()),
        suppress(BaseException),
    ):
        await engine.remember_batch(
            documents,
            namespace_id=uuid4(),
            deduplicate=False,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            **kwargs,
        )

    assert captured, "Document(...) was never constructed"
    return captured[0]


@pytest.mark.unit
class TestSkeletonRememberBatchSourceType:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_falsy_per_doc_source_type_inherits_batch_value(self, falsy) -> None:
        kwargs = await _capture_skeleton_document(
            [{"content": "body", "source_type": falsy}],
            source_type="slack",
        )

        assert kwargs["source_type"] == "slack"

    @pytest.mark.asyncio
    async def test_explicit_per_doc_source_type_still_wins(self) -> None:
        kwargs = await _capture_skeleton_document(
            [{"content": "body", "source_type": "file"}],
            source_type="slack",
        )

        assert kwargs["source_type"] == "file"


# ---------------------------------------------------------------------------
# VectorCypher — three sites, two seams
# ---------------------------------------------------------------------------


def _vectorcypher_engine(*, streaming: bool):
    from khora.engines.vectorcypher.engine import VectorCypherEngine

    engine = VectorCypherEngine(_mock_config())
    engine._connected = True
    engine._embedder = MagicMock()
    engine._temporal_store = AsyncMock()
    storage = AsyncMock()
    storage.get_documents_by_checksums = AsyncMock(return_value={})
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    engine._storage = storage
    engine._vc_config.streaming_pipeline = streaming
    return engine


async def _capture_vectorcypher_remember_kwargs(documents: list[dict], *, legacy: bool, **kwargs) -> dict:
    """Drive the two branches that forward through ``self.remember``."""
    engine = _vectorcypher_engine(streaming=False)

    captured: list[dict] = []

    async def _fake_remember(*_args, **remember_kwargs):
        captured.append(remember_kwargs)
        raise _StopAfterCapture

    method = engine._remember_batch_legacy if legacy else engine._remember_batch_impl

    # The engines wrap per-document work in try/except and record a failure
    # rather than propagating, so the sentinel does not reach us. Suppressing
    # broadly here is deliberate: the assertion below is on what was captured
    # at the seam, not on how the flow ended.
    with patch.object(type(engine), "remember", side_effect=_fake_remember), suppress(BaseException):
        await method(
            documents,
            uuid4(),
            deduplicate=False,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            **kwargs,
        )

    assert captured, "self.remember() was never reached"
    return captured[0]


async def _capture_vectorcypher_document(documents: list[dict], **kwargs) -> dict:
    """Drive the streaming branch, which constructs ``Document`` directly."""
    engine = _vectorcypher_engine(streaming=True)

    captured: list[dict] = []
    with patch("khora.engines.vectorcypher.engine.Document", _capturing_document(captured)), suppress(BaseException):
        await engine._remember_batch_impl(
            documents,
            uuid4(),
            deduplicate=False,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            **kwargs,
        )

    assert captured, "Document(...) was never constructed"
    return captured[0]


@pytest.mark.unit
class TestVectorCypherRememberBatchSourceType:
    """The default engine — the highest-exposure of the five sites."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    @pytest.mark.parametrize("legacy", [False, True], ids=["direct", "legacy"])
    async def test_falsy_per_doc_source_type_inherits_batch_value(self, falsy, legacy) -> None:
        kwargs = await _capture_vectorcypher_remember_kwargs(
            [{"content": "body", "source_type": falsy}],
            legacy=legacy,
            source_type="slack",
        )

        assert kwargs["source_type"] == "slack"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy", [False, True], ids=["direct", "legacy"])
    async def test_explicit_per_doc_source_type_still_wins(self, legacy) -> None:
        kwargs = await _capture_vectorcypher_remember_kwargs(
            [{"content": "body", "source_type": "file"}],
            legacy=legacy,
            source_type="slack",
        )

        assert kwargs["source_type"] == "file"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_streaming_branch_falsy_inherits_batch_value(self, falsy) -> None:
        kwargs = await _capture_vectorcypher_document(
            [{"content": "body", "source_type": falsy}],
            source_type="slack",
        )

        assert kwargs["source_type"] == "slack"

    @pytest.mark.asyncio
    async def test_streaming_branch_explicit_value_still_wins(self) -> None:
        kwargs = await _capture_vectorcypher_document(
            [{"content": "body", "source_type": "file"}],
            source_type="slack",
        )

        assert kwargs["source_type"] == "file"
