"""Unit tests for ``khora.integrations.llamaindex.KhoraRetriever``.

Runs against an ``AsyncMock(spec=Khora)`` so no infrastructure is
required. Integration coverage against a real khora lives in
``tests/integration/integrations/llamaindex/test_e2e.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("llama_index.core")


from khora import Khora  # noqa: E402
from khora.core.models import DocumentProjection, RecallChunk, RecallEntity  # noqa: E402


def _mk_kb(*, recall_result) -> Khora:
    """Build an ``AsyncMock(spec=Khora)`` whose ``recall`` returns the given result."""
    kb = AsyncMock(spec=Khora)
    kb.recall = AsyncMock(return_value=recall_result)
    return kb


def _mk_chunk(content: str = "alpha", score: float = 0.9) -> tuple[RecallChunk, DocumentProjection]:
    """Build a (RecallChunk, DocumentProjection) pair linked by document_id."""
    document_id = uuid4()
    now = datetime.now(UTC)
    chunk = RecallChunk(
        id=uuid4(),
        document_id=document_id,
        content=content,
        score=score,
        created_at=now,
    )
    doc = DocumentProjection(id=document_id, created_at=now)
    return chunk, doc


def _mk_entity(name: str = "Alice", score: float = 0.3) -> RecallEntity:
    return RecallEntity(
        id=uuid4(),
        name=name,
        entity_type="PERSON",
        description="example",
        score=score,
        attributes={},
        mention_count=0,
        source_document_ids=[],
        source_chunk_ids=[],
    )


def _mk_recall_result(*, chunks=None, entities=None, abstain=False):
    """Minimal stub matching the public fields of ``RecallResult``.

    ``chunks`` is a list of ``RecallChunk`` (or a list of (RecallChunk,
    DocumentProjection) tuples — in that case we split them).
    """
    chunk_list: list[RecallChunk] = []
    doc_list: list[DocumentProjection] = []
    for item in chunks or []:
        if isinstance(item, tuple):
            ch, doc = item
            chunk_list.append(ch)
            doc_list.append(doc)
        else:
            chunk_list.append(item)
    engine_info = {"abstention_signals": {"should_abstain": abstain}} if abstain else {}
    return MagicMock(
        namespace_id=uuid4(),
        documents=doc_list,
        chunks=chunk_list,
        entities=entities or [],
        relationships=[],
        engine_info=engine_info,
    )


@pytest.mark.asyncio
async def test_aretrieve_returns_nodes_for_each_chunk():
    """One chunk in → one ``NodeWithScore`` out."""
    from khora.integrations.llamaindex import KhoraRetriever

    chunk_pair = _mk_chunk("hello", score=0.9)
    kb = _mk_kb(recall_result=_mk_recall_result(chunks=[chunk_pair]))
    ns = uuid4()
    retriever = KhoraRetriever(kb, namespace_id=ns, similarity_top_k=3)

    nodes = await retriever.aretrieve("hello")

    assert len(nodes) == 1
    assert nodes[0].score == pytest.approx(0.9)
    assert nodes[0].node.text == "hello"
    assert nodes[0].node.metadata["khora_kind"] == "chunk"

    # Recall was called with the configured namespace + top_k.
    kwargs = kb.recall.call_args.kwargs
    assert kwargs["namespace"] == ns
    assert kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_aretrieve_excludes_entities_by_default():
    """Default ``include_entities=False`` — entity hits are ignored."""
    from khora.integrations.llamaindex import KhoraRetriever

    chunk_pair = _mk_chunk(score=0.5)
    entity = _mk_entity()
    kb = _mk_kb(recall_result=_mk_recall_result(chunks=[chunk_pair], entities=[entity]))
    retriever = KhoraRetriever(kb, namespace_id=uuid4())

    nodes = await retriever.aretrieve("hi")

    kinds = {n.node.metadata["khora_kind"] for n in nodes}
    assert kinds == {"chunk"}


@pytest.mark.asyncio
async def test_aretrieve_includes_entities_when_opted_in():
    """``include_entities=True`` surfaces entities alongside chunks."""
    from khora.integrations.llamaindex import KhoraRetriever

    chunk_pair = _mk_chunk(score=0.5)
    entity = _mk_entity()
    kb = _mk_kb(recall_result=_mk_recall_result(chunks=[chunk_pair], entities=[entity]))
    retriever = KhoraRetriever(kb, namespace_id=uuid4(), include_entities=True)

    nodes = await retriever.aretrieve("hi")

    kinds = sorted(n.node.metadata["khora_kind"] for n in nodes)
    assert kinds == ["chunk", "entity"]


@pytest.mark.asyncio
async def test_aretrieve_propagates_abstention_signal():
    """When khora flags abstain, every node carries ``khora_should_abstain=True``."""
    from khora.integrations.llamaindex import KhoraRetriever

    chunk_pair = _mk_chunk(score=0.1)
    kb = _mk_kb(recall_result=_mk_recall_result(chunks=[chunk_pair], abstain=True))
    retriever = KhoraRetriever(kb, namespace_id=uuid4())

    nodes = await retriever.aretrieve("hi")
    assert nodes[0].node.metadata["khora_should_abstain"] is True


@pytest.mark.asyncio
async def test_sync_retrieve_raises_not_implemented():
    """``_retrieve`` (and the public ``retrieve``) must refuse sync calls."""
    from llama_index.core.schema import QueryBundle

    from khora.integrations.llamaindex import KhoraRetriever

    kb = _mk_kb(recall_result=_mk_recall_result())
    retriever = KhoraRetriever(kb, namespace_id=uuid4())

    with pytest.raises(NotImplementedError, match="async-only"):
        retriever._retrieve(QueryBundle("hi"))


def test_isinstance_base_retriever():
    """``isinstance(retriever, BaseRetriever)`` passes — required by LlamaIndex."""
    from llama_index.core.base.base_retriever import BaseRetriever

    from khora.integrations.llamaindex import KhoraRetriever

    kb = AsyncMock(spec=Khora)
    retriever = KhoraRetriever(kb, namespace_id=uuid4())
    assert isinstance(retriever, BaseRetriever)


def test_invalid_top_k_rejected():
    """``similarity_top_k <= 0`` raises immediately."""
    from khora.integrations.llamaindex import KhoraRetriever

    kb = AsyncMock(spec=Khora)
    with pytest.raises(ValueError, match="similarity_top_k"):
        KhoraRetriever(kb, namespace_id=uuid4(), similarity_top_k=0)


def test_namespace_id_property_exposes_bound_uuid():
    """The ``namespace_id`` property is the marker-Protocol surface."""
    from khora.integrations.llamaindex import KhoraRetriever

    kb = AsyncMock(spec=Khora)
    ns = uuid4()
    retriever = KhoraRetriever(kb, namespace_id=ns)
    assert isinstance(retriever.namespace_id, UUID)
    assert retriever.namespace_id == ns


@pytest.mark.asyncio
async def test_recall_kwargs_forwarded():
    """Extra kwargs in ``recall_kwargs`` are passed straight to ``Khora.recall``."""
    from khora.integrations.llamaindex import KhoraRetriever

    kb = _mk_kb(recall_result=_mk_recall_result())
    retriever = KhoraRetriever(
        kb,
        namespace_id=uuid4(),
        recall_kwargs={"min_similarity": 0.42},
    )
    await retriever.aretrieve("q")
    assert kb.recall.call_args.kwargs["min_similarity"] == pytest.approx(0.42)
