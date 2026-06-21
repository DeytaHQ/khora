"""Live pg+Neo4j projection of materialized communities onto recall() (#1308).

The query-time half of #1276: after the VectorCypher engine assembles a recall's
result entities, it fetches the materialized dream :Community summaries those
entities belong to via the live ``get_entity_communities`` reader and surfaces
them on ``RecallResult.communities`` (de-duped + capped).

This drives the PUBLIC ``Khora.recall()`` path end-to-end against a real pg+Neo4j
stack (entity ingest -> real ``HAS_MEMBER`` Cypher reader -> projection onto the
public result), so it protects the ``RecallResult.communities`` + ``engine_info``
wiring, not just the private helper. LLM calls are stubbed (deterministic unit
embedding + a fixed extracted entity), so no API key is required.

With materialized communities: a recall touching the member entity surfaces the
community summary. Without: ``RecallResult.communities`` is empty and recall is
otherwise unaffected (zero added cost).

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_USERNAME=neo4j KHORA_NEO4J_PASSWORD=pleaseletmein \\
    NEO4J_INTEGRATION_TEST=1 \\
        uv run pytest tests/integration/dream/test_neo4j_community_recall_projection.py -v
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models.entity import CommunityNode
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import Khora
from khora.query import SearchMode
from tests.test_helpers.diagnostics import assert_no_silent_degradation

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")

EMBED_DIM = 1536
_ENTITY_NAME = "Falcon"
_MARKER = "graphdoc"


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_reachable(DATABASE_URL, 5432) and _reachable(NEO4J_URL, 7687)),
        reason="pg+neo4j not reachable (run `make dev`)",
    ),
]


# Deterministic LLM stubs — no OPENAI_API_KEY required. Every text maps to the
# SAME unit vector, so the query embedding equals the seeded entity's embedding
# (cosine 1.0): the entry-entity vector search returns the entity as the
# graph-expansion seed. The extractor emits the shared entity only for
# marker-carrying docs, so the doc gets a real MENTIONED_IN edge.
def _unit_vector() -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_unit_vector() for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _unit_vector()


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    out: list[ExtractionResult] = []
    for text in texts:
        if _MARKER in text:
            out.append(
                ExtractionResult(entities=[ExtractedEntity(name=_ENTITY_NAME, entity_type="PERSON", confidence=0.99)])
            )
        else:
            out.append(ExtractionResult())
    return out


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.storage.neo4j_user = NEO4J_USER
    config.storage.neo4j_password = NEO4J_PASSWORD
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False
    config.query.min_entity_similarity = 0.0
    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


def _graph_backend(kb: Khora):
    graph = kb.storage.graph
    return getattr(graph, "_backend", graph)


async def _ingest_entity(kb: Khora, namespace_id: UUID) -> UUID:
    """Ingest a marker doc so the entity lands in both stores; return its id."""
    await kb.remember(
        content=f"{_ENTITY_NAME} {_MARKER}: orbital launch alpha bravo charlie delta.",
        namespace=namespace_id,
        title="graph-doc",
        source_name="linear",
        entity_types=["PERSON"],
        relationship_types=[],
    )
    ns_row_id = await kb.storage.resolve_namespace(namespace_id)
    entities = await kb.storage.list_entities(ns_row_id, limit=50)
    # Entity names are normalized (lowercased) at ingest, so match case-insensitively.
    match = next(e for e in entities if e.name.lower() == _ENTITY_NAME.lower())
    return match.id


@pytest.mark.asyncio
async def test_recall_projects_materialized_community(kb: Khora) -> None:
    """A recall touching a member entity surfaces the community summary (#1308)."""
    ns = await kb.create_namespace()
    namespace_id = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(namespace_id)

    entity_id = await _ingest_entity(kb, namespace_id)

    community = CommunityNode(
        id=uuid4(),
        namespace_id=ns_row_id,
        summary="the falcon launch community summary",
        member_ids=[entity_id],
        summary_depth=1,
    )
    count = await _graph_backend(kb).materialize_communities_batch(
        [community], namespace_id=ns_row_id, materialized_at=datetime.now(UTC)
    )
    assert count == 1

    result = await kb.recall(_ENTITY_NAME, namespace=namespace_id, limit=20, mode=SearchMode.GRAPH)

    assert entity_id in {e.id for e in result.entities}, "pre-flight: the member entity must surface in recall"
    assert [c.id for c in result.communities] == [community.id]
    assert result.communities[0].summary == "the falcon launch community summary"
    assert_no_silent_degradation(result)


@pytest.mark.asyncio
async def test_recall_without_materialized_community_is_empty(kb: Khora) -> None:
    """A recall with no materialized communities returns empty (zero added cost)."""
    ns = await kb.create_namespace()
    namespace_id = ns.namespace_id

    entity_id = await _ingest_entity(kb, namespace_id)

    result = await kb.recall(_ENTITY_NAME, namespace=namespace_id, limit=20, mode=SearchMode.GRAPH)

    assert entity_id in {e.id for e in result.entities}, "pre-flight: the entity must surface in recall"
    assert result.communities == []
    assert_no_silent_degradation(result)
