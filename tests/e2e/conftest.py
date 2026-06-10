"""Fixtures + self-skip guards for the deterministic e2e recall-filter suite.

``@internal``. Backend-owned. Provides the three per-engine ``Khora`` fixtures
(embedded ``sqlite_lance`` / live ``vectorcypher`` PG+Neo4j / live ``chronicle``
PG-only) with the deterministic extractor+embedder install (``stub_llm``) and the
reachability guards the live modules self-skip on. Each test module pins the
engine fixture it needs directly and mints its own fresh namespace. The test
modules under ``tests/e2e/test_*.py`` (QA-owned) consume these fixtures and the
``_harness`` engine layer; they never reach into ``src/``.

Determinism: every engine fixture installs ``stub_llm`` (no network, SHA-256-derived
embeddings) before the ``Khora`` is built, disables HyDE explicitly
(``enable_hyde="never"``; the default is ``"auto"``), and sends every chunk to the
stub (``selective_extraction=False``) so the hand-counted ``expected_ids`` stay
exact.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest
from pydantic import SecretStr

from khora import Khora
from khora.config.schema import KhoraConfig, SQLiteLanceConfig, SurrealDBConfig
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


def _weaviate_reachable() -> bool:
    """Whether the live Weaviate at ``WEAVIATE_URL`` is reachable (socket-probe).

    Mirrors ``_pg_reachable`` / ``_neo4j_reachable``: parse host+port off the
    configured URL and TCP-probe it so the live Skeleton-Weaviate lane collects
    and self-skips cleanly on a no-Docker run. ``WEAVIATE_URL`` defaults to the
    compose HTTP port (8090) when unset.
    """
    url = os.environ.get("WEAVIATE_URL", "http://localhost:8090")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 8080), timeout=2):
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
    that uses this fixture carries the ``NEO4J_INTEGRATION_TEST`` + Postgres
    reachability self-skip mark so a no-Docker run skips it cleanly.
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
    extractor+embedder sized at 1536, no Neo4j URL, HyDE off. The module that uses
    this fixture self-skips via its Postgres reachability mark when Postgres is down.
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
# Skeleton lanes — the Skeleton engine (VECTOR / HYBRID / KEYWORD, no graph).
#
# The Skeleton engine has no graph channel (supported_modes omits GRAPH), so
# ``_lower_entity_floor`` is a harmless no-op on it (the retriever getattr guard
# returns early). Skeleton recall is the vector-only path: doc-level
# ``external_id`` reconciliation works identically to live pgvector, since the
# survivor set is keyed off the returned chunks' ``document_id``s either way.
# Two of these run container-free (embedded sqlite_lance / in-process surrealdb
# memory mode); two are live (pgvector / weaviate) and self-skip without Postgres
# (and Weaviate). The Skeleton backend auto-detects sqlite_lance / surrealdb from
# ``config.storage.backend`` (engine.py); pgvector is the default; weaviate is
# selected via the ``backend`` + ``weaviate_url`` engine kwargs.
# --------------------------------------------------------------------------- #


@pytest.fixture
async def skeleton_sqlite_lance_kb(monkeypatch: pytest.MonkeyPatch, tmp_path) -> AsyncIterator[Khora]:
    """Connected ``skeleton`` Khora on an embedded sqlite_lance stack (container-free).

    Like ``sqlite_lance_kb`` but ``engine="skeleton"``. No Postgres, no Neo4j, no
    Docker. Migrated via Alembic on entry. Skeleton auto-detects the sqlite_lance
    backend from ``config.storage.backend``.
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

    kb = Khora(config, engine="skeleton", run_migrations=True)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def skeleton_surrealdb_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``skeleton`` Khora on an in-process SurrealDB (memory mode, container-free).

    ``config.storage.backend = "surrealdb"`` with the default ``memory`` mode runs
    SurrealDB in-process (no container, no on-disk file). Schema initialises
    declaratively on ``connect()`` (no Alembic), so ``run_migrations`` is a no-op
    here. Skeleton auto-detects the surrealdb backend from ``config.storage.backend``.
    """
    stub_llm(monkeypatch, dim=EMBED_DIM)

    config = KhoraConfig()
    config.storage.backend = "surrealdb"
    config.storage.surrealdb = SurrealDBConfig(mode="memory", embedding_dimension=EMBED_DIM)
    config.llm.embedding_dimension = EMBED_DIM
    _apply_deterministic_query(config)

    kb = Khora(config, engine="skeleton")
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def skeleton_pgvector_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``skeleton`` Khora on the live PG (pgvector) stack. Self-skips without PG.

    Mirrors the ``chronicle_kb`` shape (live DSN, dim 1536, ``run_migrations=False``)
    but ``engine="skeleton"`` on the default pgvector backend. Skeleton recall uses
    the vector-only path; doc-level ``external_id`` reconciliation works identically
    to live pgvector. The module that uses this fixture self-skips via its Postgres
    reachability mark when Postgres is down.
    """
    stub_llm(monkeypatch, dim=PG_EMBED_DIM)

    config = KhoraConfig(database_url=_database_url())
    config.llm.embedding_dimension = PG_EMBED_DIM
    config.storage.embedding_dimension = PG_EMBED_DIM
    _apply_deterministic_query(config)

    kb = Khora(config, engine="skeleton", run_migrations=False)
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def skeleton_weaviate_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``skeleton`` Khora on live Weaviate (vectors) + live PG (documents).

    The Skeleton-Weaviate backend keeps vectors in Weaviate and documents /
    namespaces in Postgres (the relational coordinator falls through to the PG
    path), so dim is 1536 (the PG pgvector column constraint applies). ``backend``
    + ``weaviate_url`` are forwarded to the Skeleton engine constructor.
    ``WEAVIATE_URL`` is read from env into a local str (no Pydantic field, so no
    ``SecretStr`` obligation — it is a service endpoint, not a credential). The
    module that uses this fixture self-skips when Postgres or Weaviate is down.
    """
    stub_llm(monkeypatch, dim=PG_EMBED_DIM)

    config = KhoraConfig(database_url=_database_url())
    config.llm.embedding_dimension = PG_EMBED_DIM
    config.storage.embedding_dimension = PG_EMBED_DIM
    _apply_deterministic_query(config)

    kb = Khora(
        config,
        engine="skeleton",
        run_migrations=False,
        engine_kwargs={"backend": "weaviate", "weaviate_url": os.environ["WEAVIATE_URL"]},
    )
    await kb.connect()
    _lower_entity_floor(kb)
    try:
        yield kb
    finally:
        await kb.disconnect()


# --------------------------------------------------------------------------- #
# CI fail-loud tripwire — convert skip → hard error on the container legs.
#
# The live fixtures above self-skip via ``_pg_reachable()`` / ``_weaviate_reachable()``
# so a no-Docker dev run collects-and-skips them cleanly. But in CI a container
# leg whose store is down must FAIL RED, not skip green — a silent skip would let
# a broken-infra leg pass. Mirrors ``tests/integration/conftest.py``'s
# ``KHORA_PG_REQUIRED`` pattern: when the leg's "required" flag is set and the
# store is unreachable, abort the whole session with a red exit.
#
#   KHORA_E2E_PG_REQUIRED=1        -> Postgres must be reachable (vc_full,
#                                     skeleton_pgvector, skeleton_weaviate,
#                                     chronicle legs).
#   KHORA_E2E_NEO4J_REQUIRED=1     -> Neo4j must be reachable (vc_full leg — the
#                                     only lane with a live graph backend).
#   KHORA_E2E_WEAVIATE_REQUIRED=1  -> Weaviate must be reachable (skeleton_weaviate
#                                     leg).
#
# The devops e2e workflow sets these on the matching container legs only; the
# default no-Docker job leaves them unset so the per-fixture self-skip still wins.
# --------------------------------------------------------------------------- #


def pytest_configure(config: pytest.Config) -> None:
    """Fail loudly at session start if a CI-required e2e backend is unreachable."""
    if os.environ.get("KHORA_E2E_PG_REQUIRED") == "1" and not _pg_reachable():
        pytest.exit(
            f"KHORA_E2E_PG_REQUIRED=1 but Postgres is unreachable at {_database_url()}. "
            "The e2e CI leg provisions Postgres; a skip here would hide real failures.",
            returncode=1,
        )

    if os.environ.get("KHORA_E2E_NEO4J_REQUIRED") == "1" and not _neo4j_reachable():
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        pytest.exit(
            f"KHORA_E2E_NEO4J_REQUIRED=1 but Neo4j is unreachable at {neo4j_url}. "
            "The e2e CI leg provisions Neo4j; a skip here would hide real failures.",
            returncode=1,
        )

    if os.environ.get("KHORA_E2E_WEAVIATE_REQUIRED") == "1" and not _weaviate_reachable():
        weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8090")
        pytest.exit(
            f"KHORA_E2E_WEAVIATE_REQUIRED=1 but Weaviate is unreachable at {weaviate_url}. "
            "The e2e CI leg provisions Weaviate; a skip here would hide real failures.",
            returncode=1,
        )
