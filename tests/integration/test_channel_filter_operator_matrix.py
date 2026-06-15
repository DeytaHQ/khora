"""Channel × filter-operator coverage matrix — RECENCY + GRAPH cells (live PG+Neo4j).

GitHub issue #1223 follow-up. The live half of the matrix whose registry, meta-tests,
and shared cell driver live (hermetic) in
``tests/recall/test_channel_filter_operator_matrix.py``. This module holds the
RECENCY and GRAPH row cells: each drives ONE channel × ONE operator class through a
real ``Khora.recall()`` on the live VectorCypher (PG + Neo4j) stack and asserts, per
cell, the four contracts the driver enforces — (1) the violating chunk is ABSENT,
(2) no private ``_filter_channel_plans`` leak, (3) the ``engine_info["filter"]``
report obeys the engine-independent invariants + the pinned leaf is accounted for,
and (4) the channel under test actually FIRED (non-vacuity).

WHY ``tests/integration/`` (not ``tests/e2e/``): CI's ``e2e.yml`` ``vc_full`` leg
(the only e2e leg with Neo4j) path-pins its pytest selection to specific modules, so
a new ``tests/e2e/`` module would never be selected there → its cells would only ever
SKIP (vacuous green). CI's ``test-integration`` job runs ``pytest tests/integration/
-m integration`` with PG+Neo4j provisioned and ``NEO4J_INTEGRATION_TEST=1`` set, so
these cells actually execute — no workflow change.

Self-skip: the module is gated on ``NEO4J_INTEGRATION_TEST`` + Postgres + Neo4j
reachability (the same integration-style guards the sibling live filter tests use —
``test_vectorcypher_recency_channel_pg.py`` / ``test_filter_pushdown_graph.py``), so a
no-Docker / no-flag run collects-and-skips cleanly, and the integration conftest's
loud-DB-down tripwire turns a missing required store into a RED session on the CI leg.

Run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688)
    NEO4J_INTEGRATION_TEST=1 KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
        KHORA_NEO4J_URL=bolt://localhost:7688 KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/test_channel_filter_operator_matrix.py -m integration
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest
from pydantic import SecretStr

from khora import Khora
from khora.config.schema import KhoraConfig
from tests.recall.test_channel_filter_operator_matrix import (
    _GRAPH_QUERY,
    _PG_EMBED_DIM,
    _RECENCY_QUERY,
    _assert_channel_cell,
    _lower_entity_floor,
    _satisfying,
    _violating,
)
from tests.test_helpers.filter_spy import stub_llm

# --------------------------------------------------------------------------- #
# Self-skip guards — mirror tests/integration/test_filter_pushdown_graph.py and
# test_vectorcypher_recency_channel_pg.py (NOT the e2e lane_skip/lane_reachable).
# --------------------------------------------------------------------------- #


def _pg_reachable() -> bool:
    url = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False


def _neo4j_reachable() -> bool:
    url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 7687), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.filter_enforcement,
    pytest.mark.skipif(
        not os.environ.get("NEO4J_INTEGRATION_TEST") or not _pg_reachable() or not _neo4j_reachable(),
        reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live matrix cells",
    ),
]


# --------------------------------------------------------------------------- #
# Live VectorCypher kb fixtures (PG + Neo4j). Mirror the sibling integration
# tests' fixtures: deterministic extractor+embedder sized at 1536, graph-write
# config on, HyDE + reranking off. The recency fixture additionally flips the
# recency channel ON at build time.
# --------------------------------------------------------------------------- #


def _base_config() -> KhoraConfig:
    database_url = os.environ.get("KHORA_DATABASE_URL", "postgresql+asyncpg://khora:khora@localhost:5434/khora")
    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = SecretStr(os.environ.get("KHORA_NEO4J_PASSWORD", "password"))
    config.llm.embedding_dimension = _PG_EMBED_DIM
    config.storage.embedding_dimension = _PG_EMBED_DIM
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False
    config.query.enable_hyde = "never"
    config.query.enable_hyde_cypher = False
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False
    return config


@pytest.fixture
async def vectorcypher_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora on the live PG+Neo4j stack (graph row cells).

    Mirrors ``test_filter_pushdown_graph.py::kb``: deterministic extractor+embedder
    sized at 1536, graph-write config on, HyDE + reranking off. The entity floor is
    lowered so the deterministic-embedder entities clear entry-entity search.
    """
    stub_llm(monkeypatch, dim=_PG_EMBED_DIM)
    config = _base_config()
    kb = Khora(config, engine="vectorcypher", run_migrations=False)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def vectorcypher_recency_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora (live PG+Neo4j) with the recency channel ON.

    Same wiring as :func:`vectorcypher_kb` plus ``temporal_recency_channel_enabled=True``
    and a low relevance floor so the recency channel is active and both keyword-sharing
    docs clear the cosine gate (mirrors ``test_vectorcypher_recency_channel_pg.py``).
    """
    stub_llm(monkeypatch, dim=_PG_EMBED_DIM)
    config = _base_config()
    config.query.temporal_recency_channel_enabled = True
    config.query.temporal_query_relevance_floor = 0.30
    kb = Khora(config, engine="vectorcypher", run_migrations=False)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


# ===========================================================================
# RECENCY ROW — the #1223 seam. ``_recency_channel_chunks`` pushes every leaf
# into the khora_chunks WHERE (post-#1236), so the pinned leaf is PUSHED and the
# recency channel's ``post_filtered_keys`` is empty (asserted in the driver's
# MECHANISM block).
# ===========================================================================


async def test_recency_a_cell(vectorcypher_recency_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recency × A: provenance ``$ne`` over ``source_name`` (the #1223 vector).

    The freshest doc carries ``source_name="leakdoc"`` and VIOLATES
    ``{"source_name": {"$ne": "leakdoc"}}``; the older doc carries ``source_name="cleandoc"``
    and satisfies it. The recency channel pushes ``source_name`` into the SQL so the
    violating freshest doc is never fetched — and the report credits it pushed.
    """
    await _assert_channel_cell(
        vectorcypher_recency_kb,
        row="recency",
        filter_spec={"source_name": {"$ne": "leakdoc"}},
        violating_doc=_violating("recent leak note", source_name="leakdoc"),
        satisfying_doc=_satisfying("clean note", source_name="cleandoc"),
        query=_RECENCY_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_recency_b_cell(vectorcypher_recency_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recency × B: ``$exists`` over ``source_url``.

    The freshest doc OMITS ``source_url`` and VIOLATES ``{"source_url": {"$exists": true}}``;
    the older doc sets ``source_url`` and satisfies it.
    """
    await _assert_channel_cell(
        vectorcypher_recency_kb,
        row="recency",
        filter_spec={"source_url": {"$exists": True}},
        violating_doc=_violating("recent no-url note"),  # source_url ABSENT → violates $exists true
        satisfying_doc=_satisfying("clean url note", source_url="https://example.test/clean"),
        query=_RECENCY_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_recency_c_cell(vectorcypher_recency_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recency × C: ``$ne`` MISSING-INCLUSION over ``source_name``.

    The freshest doc has ``source_name="leakdoc"`` (the excluded value → VIOLATES);
    the satisfying doc OMITS ``source_name`` entirely and MUST SURVIVE — a ``$ne``
    never matches an absent key (the exact #1223 leak shape, but proving the
    survivor side too).
    """
    await _assert_channel_cell(
        vectorcypher_recency_kb,
        row="recency",
        filter_spec={"source_name": {"$ne": "leakdoc"}},
        violating_doc=_violating("recent leak note", source_name="leakdoc"),
        satisfying_doc=_satisfying("absent-source note"),  # source_name ABSENT → must SURVIVE the $ne
        query=_RECENCY_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_recency_d_cell(vectorcypher_recency_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recency × D: metadata present-JSON-null vs absent on ``metadata.tier``.

    The freshest doc sets ``metadata.tier`` to ``"gold"`` and VIOLATES
    ``{"metadata.tier": {"$eq": null}}`` (it is present-and-non-null); the satisfying
    doc OMITS the key, which an ``$eq null`` matches (null-or-missing).
    """
    await _assert_channel_cell(
        vectorcypher_recency_kb,
        row="recency",
        filter_spec={"metadata.tier": {"$eq": None}},
        # present non-null → violates $eq null; tier ABSENT → matches $eq null
        violating_doc=_violating("recent tiered note", metadata={"tier": "gold"}),
        satisfying_doc=_satisfying("untiered note", metadata={"other": "x"}),
        query=_RECENCY_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


# ===========================================================================
# GRAPH ROW — ``_fetch_chunks_from_entities`` over-fetches and post-filters. The
# provenance columns (A/B/C) PUSH via compile_cypher (pushed_keys carries them);
# the metadata sub-path column (D) is unpushable to Cypher → over-fetch +
# in-memory post-filter (post_filtered_keys carries it).
# ===========================================================================


async def test_graph_a_cell(vectorcypher_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph × A: provenance ``$ne`` over ``source_name`` (PUSHED via Cypher).

    The graph over-fetch path pushes the provenance leaf into the Cypher ``WHERE``,
    so the violating doc is never fetched and the report credits ``source_name``
    pushed.
    """
    await _assert_channel_cell(
        vectorcypher_kb,
        row="graph",
        filter_spec={"source_name": {"$ne": "leakdoc"}},
        violating_doc=_violating("graph leak note", source_name="leakdoc"),
        satisfying_doc=_satisfying("graph clean note", source_name="cleandoc"),
        query=_GRAPH_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_graph_b_cell(vectorcypher_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph × B: ``$exists`` over ``source_url`` (PUSHED via Cypher).

    The freshest doc omits ``source_url`` and violates ``$exists true``; the other
    sets it and survives. The graph channel pushes the presence predicate to Cypher.
    """
    await _assert_channel_cell(
        vectorcypher_kb,
        row="graph",
        filter_spec={"source_url": {"$exists": True}},
        violating_doc=_violating("graph no-url note"),  # source_url ABSENT → violates $exists true
        satisfying_doc=_satisfying("graph url note", source_url="https://example.test/graph"),
        query=_GRAPH_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_graph_c_cell(vectorcypher_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph × C: ``$ne`` MISSING-INCLUSION over ``source_name``.

    The violating doc has ``source_name="leakdoc"`` (the excluded value); the
    satisfying doc OMITS ``source_name`` and MUST SURVIVE — the missing-inclusion
    contract on the graph push path.
    """
    await _assert_channel_cell(
        vectorcypher_kb,
        row="graph",
        filter_spec={"source_name": {"$ne": "leakdoc"}},
        violating_doc=_violating("graph leak note", source_name="leakdoc"),
        satisfying_doc=_satisfying("graph absent-source note"),  # source_name ABSENT → must SURVIVE the $ne
        query=_GRAPH_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


async def test_graph_d_cell(vectorcypher_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph × D: metadata present-JSON-null vs absent on ``metadata.tier`` (POST-FILTERED).

    A metadata sub-path is unpushable to Cypher, so the graph channel over-fetches
    and applies an in-memory ``compile_python`` post-filter — the report credits
    ``metadata.tier`` as POST-FILTERED (not pushed). The violating doc sets
    ``metadata.tier="gold"`` (present non-null → violates ``$eq null``); the satisfying
    doc omits the key (matches ``$eq null``).
    """
    await _assert_channel_cell(
        vectorcypher_kb,
        row="graph",
        filter_spec={"metadata.tier": {"$eq": None}},
        # present non-null → violates $eq null; tier ABSENT → matches $eq null
        violating_doc=_violating("graph tiered note", metadata={"tier": "gold"}),
        satisfying_doc=_satisfying("graph untiered note", metadata={"other": "x"}),
        query=_GRAPH_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )
