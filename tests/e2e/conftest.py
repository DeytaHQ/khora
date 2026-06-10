"""Fixtures + self-skip guards for the deterministic e2e recall-filter suite.

``@internal``. Backend-owned. Provides the three per-engine ``Khora`` fixtures
(embedded ``sqlite_lance`` / live ``vectorcypher`` PG+Neo4j / live ``chronicle``
PG-only), the deterministic extractor+embedder install (``stub_llm``), fresh
per-test namespaces, and the reachability guards the parametrized test modules
self-skip on. The test modules under ``tests/e2e/test_*.py`` (QA-owned) consume
these fixtures and the ``_harness`` engine layer; they never reach into ``src/``.

Determinism: every fixture installs ``stub_llm`` (no network, SHA-256-derived
embeddings), disables HyDE explicitly (``enable_hyde="never"``; the default is
``"auto"``), and sends every chunk to the stub (``selective_extraction=False``)
so the hand-counted ``expected_ids`` stay exact.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import UUID

import pytest
from pydantic import SecretStr

from khora import Khora
from khora.config.schema import KhoraConfig, SQLiteLanceConfig
from tests.test_helpers.filter_spy import EMBED_DIM, stub_llm

pytestmark = pytest.mark.e2e

# The Postgres pgvector column is fixed at 1536 (KhoraConfig rejects any other
# dimension on the PG backend), so the live-DB fixtures size their deterministic
# vectors at 1536 rather than the small embedded ``EMBED_DIM`` (32).
PG_EMBED_DIM = 1536


# --------------------------------------------------------------------------- #
# Self-skip guards — mirror the existing tests/integration skip pattern so the
# default no-Docker run collects-and-skips the live legs cleanly.
# --------------------------------------------------------------------------- #


def _pg_reachable() -> bool:
    """Whether the compose Postgres is reachable (copied from the graph spy)."""
    url = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False


def _neo4j_reachable() -> bool:
    """Whether the compose Neo4j is reachable (copied from tests/integration/conftest)."""
    url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 7687), timeout=2):
            return True
    except OSError:
        return False


def _embedded_available() -> bool:
    """Whether the no-Docker embedded stack (aiosqlite + lancedb) is importable."""
    try:
        import aiosqlite  # noqa: F401
        import lancedb  # noqa: F401
    except ImportError:
        return False
    return True


def _database_url() -> str:
    return os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5434/khora",
    )


# --------------------------------------------------------------------------- #
# Shared determinism config — applied to every engine fixture.
# --------------------------------------------------------------------------- #


def _apply_deterministic_query(config: KhoraConfig) -> None:
    """Pin the recall path deterministic: extraction on, no HyDE, no reranking.

    ``extract_entities`` on + ``selective_extraction`` off send every seed chunk to
    the stub so the graph is populated and ``expected_ids`` stay exact. HyDE and both
    reranking paths are disabled so no LLM rewriting/reordering perturbs the frozen
    deterministic-embedding scores.
    """
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False
    config.query.enable_hyde = "never"
    config.query.enable_hyde_cypher = False
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False


def _lower_entity_floor(kb: Khora) -> None:
    """Lower the VectorCypher entity-similarity floor to 0 on a connected kb.

    The deterministic hash embedder produces vectors with no semantic meaning, so a
    query-to-entity cosine sits below the default floor and entity vector search
    returns nothing — short-circuiting the graph path. Lowering the floor lets the
    seeded entities clear it (a test knob, not a product change — same as the
    existing graph filter spies). No-op when the engine exposes no retriever.
    """
    retriever = getattr(kb._engine, "_retriever", None)
    if retriever is not None and getattr(retriever, "_config", None) is not None:
        retriever._config.min_entity_similarity = 0.0


# --------------------------------------------------------------------------- #
# Embedded sqlite_lance fixture — the no-Docker MAIN lane.
# --------------------------------------------------------------------------- #


@pytest.fixture
async def sqlite_lance_kb(monkeypatch: pytest.MonkeyPatch, tmp_path) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora on an embedded sqlite_lance stack.

    No Postgres, no Neo4j, no Docker. Migrated via Alembic on entry. Installs the
    deterministic extractor+embedder and the required graph-write config so the
    real ingest path populates entities. HyDE off.
    """
    stub_llm(monkeypatch, dim=EMBED_DIM)

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    _apply_deterministic_query(config)

    kb = Khora(config, engine="vectorcypher", run_migrations=True)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


# --------------------------------------------------------------------------- #
# Live vectorcypher fixture — PG (pgvector) + Neo4j. Self-skips without services.
# --------------------------------------------------------------------------- #


@pytest.fixture
async def vectorcypher_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora on the live PG+Neo4j stack.

    Mirrors ``tests/integration/test_filter_pushdown_graph.py::kb``: deterministic
    extractor+embedder sized at 1536, graph-write config on, HyDE off. The module
    that uses this fixture carries the ``NEO4J_INTEGRATION_TEST`` + reachability
    self-skip mark from ``ENGINE_PARAMS`` so a no-Docker run skips it cleanly.
    """
    stub_llm(monkeypatch, dim=PG_EMBED_DIM)

    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    config = KhoraConfig(database_url=_database_url(), neo4j_url=neo4j_url)
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = SecretStr(os.environ.get("KHORA_NEO4J_PASSWORD", "password"))
    config.llm.embedding_dimension = PG_EMBED_DIM
    config.storage.embedding_dimension = PG_EMBED_DIM
    _apply_deterministic_query(config)

    kb = Khora(config, engine="vectorcypher", run_migrations=False)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


# --------------------------------------------------------------------------- #
# Live chronicle fixture — PG-only (pgvector). Self-skips without Postgres.
# --------------------------------------------------------------------------- #


@pytest.fixture
async def chronicle_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``chronicle`` Khora on a PG-only stack (no graph backend).

    Mirrors ``tests/integration/matrix/test_chronicle_pg.py::kb``: deterministic
    extractor+embedder sized at 1536, no Neo4j URL, HyDE off. Self-skips via the
    ``ENGINE_PARAMS`` reachability mark when Postgres is down.
    """
    stub_llm(monkeypatch, dim=PG_EMBED_DIM)

    config = KhoraConfig(database_url=_database_url())
    config.llm.embedding_dimension = PG_EMBED_DIM
    config.storage.embedding_dimension = PG_EMBED_DIM
    # Chronicle's channels are PG-only; no graph URL.
    config.neo4j_url = None
    _apply_deterministic_query(config)

    kb = Khora(config, engine="chronicle", run_migrations=False)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


# --------------------------------------------------------------------------- #
# Engine indirection + fresh per-test namespace (no cross-test bleed).
# --------------------------------------------------------------------------- #


@pytest.fixture
def kb(request: pytest.FixtureRequest) -> Khora:
    """Resolve the per-engine kb fixture named by ``ENGINE_PARAMS``.

    A test parametrizes ``kb`` indirectly over ``_harness.ENGINE_PARAMS`` (each
    param is the name of one engine fixture above); this resolves that fixture so
    the test body and the ``namespace_id`` fixture are engine-agnostic.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    """A fresh namespace on the selected engine — random id, no cross-test bleed."""
    ns = await kb.create_namespace()
    return ns.namespace_id
