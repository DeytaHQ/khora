"""Live PG+Neo4j proof that ``graph_channel_empty`` fires on a real graph path.

This is the real-backend companion to the hermetic
``tests/unit/engines/vectorcypher/test_filter_counters.py``. The hermetic suite
pins the counter's call-site logic with a mocked retriever; this file proves the
same counter fires when a GENUINE graph channel — built from real
``MENTIONED_IN`` edges over live Postgres + Neo4j — is emptied by a metadata
post-filter.

Execution contract: this file is collected by the CI ``test-integration`` job,
which provisions Postgres + Neo4j and sets ``KHORA_PG_REQUIRED=1`` /
``NEO4J_INTEGRATION_TEST=1`` (the ``tests/integration/conftest.py`` guard then
aborts the session LOUDLY if either backend is unreachable, so this never passes
by silently skipping in CI). Locally, without ``NEO4J_INTEGRATION_TEST=1``, it
self-skips — run it via ``make dev`` + ``NEO4J_INTEGRATION_TEST=1``.

Markers: ``integration`` (so the integration job collects it) + ``filter_enforcement``.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \\
        tests/integration/test_vectorcypher_filter_counters.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5434/khora)
    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest

import khora.filter.telemetry as filter_telemetry
from khora.config import KhoraConfig
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import Khora
from khora.query import SearchMode

# Postgres ``chunks.embedding`` / ``entities.embedding`` are ``vector(1536)``, so
# the stub embedder must emit 1536-dim vectors to match the deployed schema.
EMBED_DIM = 1536

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filter_enforcement,
    pytest.mark.skipif(
        not os.environ.get("NEO4J_INTEGRATION_TEST"),
        reason="set NEO4J_INTEGRATION_TEST=1 to run against real Postgres + Neo4j (requires make dev)",
    ),
]


# This repo's compose puts Postgres on 5434 (see compose.yaml). Honor an explicit
# KHORA_DATABASE_URL override (the CI integration job sets it), else default to it.
_DEFAULT_PG_URL = "postgresql+asyncpg://khora:khora@localhost:5434/khora"
_DEFAULT_NEO4J_URL = "bolt://localhost:7687"


def _database_url() -> str:
    url = os.environ.get("KHORA_DATABASE_URL", _DEFAULT_PG_URL)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _neo4j_url() -> str:
    return os.environ.get("KHORA_NEO4J_URL", _DEFAULT_NEO4J_URL)


# ---------------------------------------------------------------------------
# Deterministic LLM stubs — no OPENAI_API_KEY required.
# ---------------------------------------------------------------------------
#
# Every text maps to the SAME 1536-dim unit vector, so the query embedding equals
# the seeded entity's embedding (cosine 1.0): the entry-entity vector search
# returns the entity as the seed the graph expansion needs. The extractor emits a
# shared entity ONLY for marker-carrying docs, so those docs get real
# MENTIONED_IN edges and surface through the graph channel.


def _unit_vector() -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_unit_vector() for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _unit_vector()


_GRAPH_ENTITY_NAME = "Falcon"
_GRAPH_MARKER = "graphdoc"


async def _stub_extract_multi_with_entity(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    """Emit the shared entity for marker-carrying docs (real MENTIONED_IN edges)."""
    out: list[ExtractionResult] = []
    for text in texts:
        if _GRAPH_MARKER in text:
            out.append(
                ExtractionResult(
                    entities=[ExtractedEntity(name=_GRAPH_ENTITY_NAME, entity_type="PERSON", confidence=0.99)]
                )
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
        _stub_extract_multi_with_entity,
    )


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


@pytest.fixture
async def kb_graph() -> AsyncIterator[Khora]:
    """A connected VectorCypher Khora (live PG + Neo4j) with entity extraction ON.

    ``extract_entities`` is enabled so the ingest builds real MENTIONED_IN edges;
    ``min_entity_similarity = 0.0`` floors the entry-entity vector gate so the
    seeded entity is always returned as a graph-expansion seed. Reranking is off
    to keep the live path light (no cross-encoder load in CI).
    """
    config = KhoraConfig(database_url=_database_url(), neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
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
        try:
            await instance.disconnect()
        except Exception:
            pass


async def test_graph_channel_empty_counter_fires_on_real_graph_path(
    kb_graph: Khora, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``graph_channel_empty`` fires when the REAL graph channel (built from
    genuine MENTIONED_IN edges) is emptied by the metadata post-filter.

    Mode is ``SearchMode.GRAPH`` ON PURPOSE: the retriever's mode dispatch sets
    ``force_graph`` only for ``GRAPH``, which routes deterministically to
    ``_vectorcypher_retrieve`` regardless of the query router's complexity
    heuristic (``HYBRID`` would let the router classify a short entity query as
    SIMPLE / ``use_graph=False`` and fall to the graph-less ``_simple_retrieve``,
    making this proof vacuous).

    Corpus: 3 docs that mention the shared entity (``_GRAPH_MARKER`` -> extraction
    emits it -> real MENTIONED_IN edges), all tagged ``"noise"``. A pre-flight
    GRAPH recall with NO filter asserts ``graph_chunk_count > 0`` — proving the
    channel really HELD candidates (never vacuous). The filter ``tag IN {"urgent"}``
    then drops every graph chunk (all ``"noise"``), so ``graph_channel_empty``
    increments.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counter)

    ns = await kb_graph.create_namespace()
    namespace_id: UUID = ns.namespace_id

    # Graph-side docs: mention the entity (real MENTIONED_IN edges), tag "noise".
    for i in range(3):
        await kb_graph.remember(
            content=f"{_GRAPH_ENTITY_NAME} {_GRAPH_MARKER} entry {i}: orbital launch alpha bravo charlie.",
            namespace=namespace_id,
            title=f"graph-doc-{i}",
            source_name="linear",
            metadata={"tag": "noise"},  # violates the filter below
            entity_types=["PERSON"],
            relationship_types=[],
        )

    # Pre-flight: a GRAPH recall with NO filter must surface graph-channel
    # candidates, so the emptied-under-filter signal is provably non-vacuous.
    baseline = await kb_graph.recall(
        _GRAPH_ENTITY_NAME,
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.GRAPH,
    )
    assert baseline.engine_info.get("graph_chunk_count", 0) > 0, (
        "pre-flight: the real graph channel must hold candidates before the filter "
        f"(graph_chunk_count={baseline.engine_info.get('graph_chunk_count')}); "
        "the entity-level vector index did not seed the expansion"
    )
    assert counter.adds == [], "no filter supplied → graph_channel_empty must not fire on the pre-flight"

    # Filtered GRAPH recall: tag must be in {"urgent"}, but every graph-side doc is
    # "noise", so the full-AST post-filter empties the (real, non-empty) channel.
    result = await kb_graph.recall(
        _GRAPH_ENTITY_NAME,
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.GRAPH,
        filter={"metadata.tag": {"$in": ["urgent"]}},
    )

    assert counter.adds, (
        "graph_channel_empty must fire when a non-empty REAL graph channel is "
        f"emptied by the metadata post-filter; got {counter.adds}"
    )
    assert result.engine_info.get("graph_chunk_count", 0) == 0, "the graph channel must be empty after the filter"
