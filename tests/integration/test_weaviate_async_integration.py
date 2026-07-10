"""Real-Weaviate integration test for ``WeaviateTemporalStore`` (#783).

Exercises the v4 async client end-to-end against a running Weaviate
cluster. Gated behind ``WEAVIATE_INTEGRATION_TEST=1`` so CI does not
flake when the optional service is unavailable.

How to run locally::

    docker compose up -d weaviate  # via the `weaviate` profile in compose.yaml
    WEAVIATE_INTEGRATION_TEST=1 \\
        WEAVIATE_URL=http://localhost:8090 \\
        uv run pytest tests/integration/test_weaviate_async_integration.py -v

In CI: the ``weaviate-integration`` workflow job sets both env vars
automatically.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from pydantic_settings import BaseSettings

from khora.config import KhoraConfig
from khora.storage.temporal import TemporalChunk
from khora.storage.temporal.weaviate import (
    WeaviateBackendConfig,
    WeaviateTemporalStore,
)

_GATE = os.environ.get("WEAVIATE_INTEGRATION_TEST") == "1"
_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8090")
_API_KEY = os.environ.get("WEAVIATE_API_KEY")  # optional - empty for anonymous

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _GATE, reason="set WEAVIATE_INTEGRATION_TEST=1 to run"),
]


@pytest.fixture
async def store() -> WeaviateTemporalStore:
    """Connected store against a live Weaviate cluster."""
    # weaviate-client must be present when this test runs - bail clean if not.
    pytest.importorskip("weaviate")

    # Use a minimal in-memory KhoraConfig (no DB) - the backend only
    # touches Weaviate, not the storage coordinator.
    config = KhoraConfig.__new__(KhoraConfig)
    BaseSettings.__init__(config)
    backend_config = WeaviateBackendConfig(
        url=_URL,
        api_key=_API_KEY if _API_KEY else None,
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50061")),
    )
    s = WeaviateTemporalStore(config, backend_config)
    await s.connect()
    yield s
    await s.disconnect()


@pytest.mark.asyncio
async def test_health_check_reports_healthy(store: WeaviateTemporalStore) -> None:
    out = await store.health_check()
    assert out["status"] == "healthy"
    assert out["backend"] == "weaviate"


@pytest.mark.asyncio
async def test_create_get_delete_roundtrip(store: WeaviateTemporalStore) -> None:
    namespace_id = uuid4()
    chunk_id = uuid4()
    doc_id = uuid4()

    chunk = TemporalChunk(
        id=chunk_id,
        namespace_id=namespace_id,
        document_id=doc_id,
        content="Alice deployed v0.16.2 on Tuesday.",
        embedding=[0.01] * 1536,
        source_system="test",
        author="Alice",
        tags=["release"],
        chunker_info={"chunker": "semantic"},
    )

    out = await store.create_chunk(chunk)
    assert out.id == chunk_id

    fetched = await store.get_chunk(chunk_id, namespace_id)
    assert fetched is not None
    assert fetched.content == chunk.content
    assert fetched.author == "Alice"
    assert fetched.chunker_info == {"chunker": "semantic"}

    deleted = await store.delete_chunk(chunk_id, namespace_id)
    assert deleted is True

    gone = await store.get_chunk(chunk_id, namespace_id)
    assert gone is None


@pytest.mark.asyncio
async def test_batch_insert_search_with_temporal_filter(
    store: WeaviateTemporalStore,
) -> None:
    namespace_id = uuid4()
    doc_id = uuid4()
    chunks = [
        TemporalChunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=doc_id,
            content=f"chunk {i} body text",
            embedding=[float(i) * 0.01] * 1536,
            source_system=f"sys{i % 3}",
            tags=["batch"],
        )
        for i in range(8)
    ]
    out = await store.create_chunks_batch(chunks)
    assert len(out) == 8

    results = await store.search(
        namespace_id=namespace_id,
        query_embedding=[0.04] * 1536,
        limit=4,
    )
    assert results, "expected at least one hit"
    assert len(results) <= 4

    # Cleanup
    cleared = await store.delete_chunks_by_document(doc_id, namespace_id)
    assert cleared == 8


@pytest.mark.asyncio
async def test_tenant_isolation(store: WeaviateTemporalStore) -> None:
    """A chunk written under tenant A is not visible to tenant B."""
    ns_a = uuid4()
    ns_b = uuid4()
    doc_id = uuid4()
    chunk_id = uuid4()

    await store.create_chunk(
        TemporalChunk(
            id=chunk_id,
            namespace_id=ns_a,
            document_id=doc_id,
            content="tenant-A-secret",
            embedding=[0.5] * 1536,
        )
    )

    visible_in_a = await store.get_chunk(chunk_id, ns_a)
    assert visible_in_a is not None

    visible_in_b = await store.get_chunk(chunk_id, ns_b)
    assert visible_in_b is None

    await store.delete_chunk(chunk_id, ns_a)


@pytest.mark.asyncio
async def test_search_fulltext_returns_bm25_hits(store: WeaviateTemporalStore) -> None:
    """search_fulltext dispatches to Weaviate BM25 and returns scored Chunk tuples."""
    namespace_id = uuid4()
    doc_id = uuid4()

    chunks = [
        TemporalChunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=doc_id,
            content="turbopuffer is a serverless vector database",
            embedding=[0.01] * 1536,
        ),
        TemporalChunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=doc_id,
            content="weaviate supports hybrid search with BM25 and vector",
            embedding=[0.02] * 1536,
        ),
    ]
    await store.create_chunks_batch(chunks)

    results = await store.search_fulltext(namespace_id, "weaviate hybrid", limit=5)

    # At minimum the weaviate chunk should surface - BM25 exact match
    assert len(results) >= 1
    contents = [chunk.content for chunk, _score in results]
    assert any("weaviate" in c for c in contents)
    # Scores should be non-negative floats
    for _chunk, score in results:
        assert score >= 0.0

    await store.delete_chunks_by_document(doc_id, namespace_id)


@pytest.mark.asyncio
async def test_search_fulltext_empty_query_returns_empty(store: WeaviateTemporalStore) -> None:
    result = await store.search_fulltext(uuid4(), "")
    assert result == []
