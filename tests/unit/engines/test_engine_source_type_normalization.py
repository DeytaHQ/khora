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

Only Chronicle routes through ``pipelines/flows/ingest.py``; VectorCypher and
Skeleton do not. VectorCypher's streaming branch constructs ``Document(...)``
itself and its two other sites forward to ``self.remember(...)``, so for those
four sites there is no downstream ``or "manual"`` to fall back on — the value
these expressions produce is the value that reaches ``create_document``.

Two levels are covered, because they fail differently:

* **per-doc** — ``doc_data.get("source_type") or source_type``. Without the
  ``or``, a dict carrying an explicit ``None`` forwards the falsy value
  instead of inheriting the batch-level one.
* **batch-level** — ``source_type = source_type or "library"`` at each batch
  method's entry. The per-doc expressions only rule out a falsy *per-doc*
  value; a caller passing ``source_type=""`` would otherwise have the empty
  string preserved all the way to the column. ``documents.source_type`` is
  NOT NULL but not non-empty — migration 055 deliberately adds no
  ``CHECK (source_type <> '')`` — so nothing downstream would reject it.

The reviewer classed the engine entry points as not publicly reachable; they
are covered anyway because pinning them is cheaper than re-deriving the
reachability argument every time someone edits one.
"""

from __future__ import annotations

from contextlib import nullcontext, suppress
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
    """Drive the two branches that forward through ``self.remember``.

    The two are reached by genuinely different routes, and conflating them is
    easy to do by accident: with ``streaming_pipeline=False``,
    ``_remember_batch_impl`` immediately delegates to ``_remember_batch_legacy``,
    so calling it captures the *legacy* site twice and never exercises its own.

    ``_remember_batch_impl``'s own ``self.remember(...)`` is the Stage-0a
    external-id dispatch: a document whose ``external_id`` already exists in
    the namespace is routed to the replace path. Reaching it needs streaming
    **on** and a storage stub that reports the id as already present.
    """
    engine = _vectorcypher_engine(streaming=not legacy)

    captured: list[dict] = []

    async def _fake_remember(*_args, **remember_kwargs):
        captured.append(remember_kwargs)
        raise _StopAfterCapture

    delegated: list[int] = []

    async def _spy_legacy(*_args, **_kwargs):
        delegated.append(1)
        raise _StopAfterCapture

    if legacy:
        method = engine._remember_batch_legacy
    else:
        method = engine._remember_batch_impl
        # Make every doc look like an existing external_id so Stage 0a fires.
        documents = [{**doc, "external_id": "ext-1"} for doc in documents]
        engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-1": MagicMock()})

    # The engines wrap per-document work in try/except and record a failure
    # rather than propagating, so the sentinel does not reach us. Suppressing
    # broadly here is deliberate: the assertion below is on what was captured
    # at the seam, not on how the flow ended.
    with patch.object(type(engine), "remember", side_effect=_fake_remember):
        with (
            patch.object(type(engine), "_remember_batch_legacy", side_effect=_spy_legacy)
            if not legacy
            else nullcontext()
        ):
            with suppress(BaseException):
                await method(
                    documents,
                    uuid4(),
                    deduplicate=False,
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    **kwargs,
                )

    assert captured, "self.remember() was never reached"
    if not legacy:
        # Guards the mistake this helper used to make: with streaming off,
        # _remember_batch_impl delegates to _remember_batch_legacy and the
        # "direct" case silently captured the legacy site instead of its own.
        assert not delegated, "the direct case delegated to _remember_batch_legacy — its own site was not exercised"
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


# ---------------------------------------------------------------------------
# Batch-level normalization — the falsy value the per-doc `or` cannot catch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchLevelSourceTypeNormalization:
    """A falsy *batch-level* ``source_type`` collapses to the default.

    Distinct from the per-doc cases above, and not covered by them: the per-doc
    expressions are ``doc_data.get("source_type") or source_type``, so they only
    rule out a falsy per-doc value. With ``source_type=""`` and no per-doc key,
    every one of those expressions evaluates to ``""`` and writes it. Nothing
    downstream rejects it — ``documents.source_type`` is NOT NULL but not
    non-empty, and migration 055 deliberately adds no
    ``CHECK (source_type <> '')``.

    ``None`` is included alongside ``""`` even though the parameter is annotated
    ``str``: passing it was always a type violation, but before migration 055 it
    wrote SQL NULL silently rather than raising, so a caller may well be doing
    it today.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_chronicle_falsy_batch_source_type(self, falsy) -> None:
        doc_inputs = await _capture_chronicle_doc_inputs([{"content": "body"}], source_type=falsy)

        assert doc_inputs[0]["source_type"] == "library"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_skeleton_falsy_batch_source_type(self, falsy) -> None:
        kwargs = await _capture_skeleton_document([{"content": "body"}], source_type=falsy)

        assert kwargs["source_type"] == "library"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    @pytest.mark.parametrize("legacy", [False, True], ids=["direct", "legacy"])
    async def test_vectorcypher_falsy_batch_source_type(self, falsy, legacy) -> None:
        kwargs = await _capture_vectorcypher_remember_kwargs(
            [{"content": "body"}],
            legacy=legacy,
            source_type=falsy,
        )

        assert kwargs["source_type"] == "library"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy", FALSY)
    async def test_vectorcypher_streaming_falsy_batch_source_type(self, falsy) -> None:
        kwargs = await _capture_vectorcypher_document([{"content": "body"}], source_type=falsy)

        assert kwargs["source_type"] == "library"

    @pytest.mark.asyncio
    async def test_a_real_batch_value_is_not_collapsed(self) -> None:
        """Positive control: the normalization must only touch falsy values."""
        doc_inputs = await _capture_chronicle_doc_inputs([{"content": "body"}], source_type="slack")

        assert doc_inputs[0]["source_type"] == "slack"
