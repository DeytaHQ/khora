"""End-to-end integration: ``KhoraRetriever`` + ``KhoraMemoryBlock`` over a real khora.

Runs against an in-memory ``sqlite_lance`` khora (no Postgres, no
Neo4j). The mock LLM helper patches ``litellm.acompletion`` /
``litellm.aembedding`` so no API keys are needed.

Proves the adapter is wired up end-to-end:

1. Build a real ``Khora`` on sqlite_lance.
2. Stash two documents via ``Khora.remember``.
3. Wrap khora in ``KhoraRetriever`` and verify ``aretrieve`` returns
   correctly-typed ``NodeWithScore`` objects with the stored chunk text.
4. Wrap khora in a ``KhoraMemoryBlock`` and verify ``aput`` then ``aget``
   round-trips a chat message through khora's recall.
"""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    import llama_index.core  # noqa: F401

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
    pytest.mark.skipif(not _HAS_LLAMAINDEX, reason="llama_index not installed"),
]


@pytest.mark.asyncio
async def test_khora_retriever_returns_chunk_for_verbatim_query(monkeypatch):
    """Verbatim recall returns a NodeWithScore whose text matches the stored doc."""
    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.llamaindex import KhoraRetriever

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        text_one = "We decided to use PostgreSQL for the user database."
        text_two = "The release window is the third week of every month."
        await kb.remember(text_one, namespace=ns_id, entity_types=[], relationship_types=[])
        await kb.remember(text_two, namespace=ns_id, entity_types=[], relationship_types=[])

        retriever = KhoraRetriever(kb, namespace_id=ns_id, similarity_top_k=3)
        nodes = await retriever.aretrieve(text_one)

        assert nodes, "expected at least one retrieved node"
        texts = [n.node.text for n in nodes]
        assert text_one in texts
        # Every node carries chunk-shaped metadata.
        for node in nodes:
            assert node.node.metadata["khora_kind"] == "chunk"
            assert "chunk_id" in node.node.metadata
            assert "document_id" in node.node.metadata


@pytest.mark.asyncio
async def test_khora_memory_block_put_then_get_roundtrip(monkeypatch):
    """``aput`` then ``aget`` exposes the persisted message through recall."""
    from llama_index.core.llms import ChatMessage

    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.llamaindex import KhoraMemoryBlock

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        block = KhoraMemoryBlock(kb=kb, namespace_id=ns_id, similarity_top_k=3)

        original = "I prefer PostgreSQL for relational workloads."
        await block.aput(messages=[ChatMessage(role="user", content=original)])

        out = await block.aget(messages=[ChatMessage(role="user", content=original)])
        # The block returns its rendered envelope; the recalled text
        # should appear inside it (deterministic mock LLM means
        # verbatim cosine = 1.0).
        assert out.startswith("<khora_memory>")
        assert original in out
