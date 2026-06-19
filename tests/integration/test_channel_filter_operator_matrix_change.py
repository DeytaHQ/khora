"""Channel x filter-operator coverage matrix — SESSION + CHANGE rows (integration).

The live-stack companion to the hermetic registry module
``tests/recall/test_channel_filter_operator_matrix.py`` (which owns the shared
registry, the ``_assert_channel_cell`` driver + all four row recipes, the
``_SeedDoc`` carrier + the ``_violating`` / ``_satisfying`` factories, and the
registry/partition unit meta-tests). This module owns the two BEHAVIORAL graph-only
rows whose firing recipe needs a populated graph and which enforce VIA the pushdown
``_vector_search_chunks`` channel (they are not a separate post-filter seam —
``is_partition_member=False``):

* **SESSION fan-out** — when the entry entity is ``MENTIONED_IN`` chunks spanning
  >= 2 distinct ``c.channel`` values, a temporal query fans the vector search out
  per-channel; each per-channel ``_vector_search_chunks`` (plus the unscoped
  fallback) carries the caller filter into the ``khora_chunks`` WHERE so a
  filter-violating chunk is never fetched (GitHub issue #1223 generalized). The
  cell arms the fan-out by seeding its two docs in DISTINCT ``metadata["channel"]``
  values.
* **CHANGE decomposition** — a CHANGE-classified query over an entity with version
  history runs a *second* current-state ``_vector_search_chunks`` sub-search; that
  sub-search also carries the caller filter.

Each cell drives the row's firing recipe through the shared ``_assert_channel_cell``
helper and asserts the four per-cell invariants: non-vacuity (the channel actually
fanned out / decomposed and fired the spied seam), enforcement (the violating doc
is absent; the satisfying doc survives), and report honesty
(``engine_info["filter"]`` accounts for the pinned leaf with no unenforced leaves
and no private-plan leak). Both rows cover operator-class **A** — the representative
leak-prone provenance ``$ne`` over ``source_name``.

The module also carries the **recency pre-fix proof** (acceptance criterion): a
single self-contained test that demonstrates the recency × provenance-``$ne`` cell
FAILS on pre-GitHub-issue-#1236 behavior (reproduced by a monkeypatch shim that
strips ``filter_ast`` before the recency SQL) and PASSES on current code. The
pre-fix path is reproduced ONLY by an auto-undone monkeypatch — nothing is
reverted in committed source.

Self-skip: ``@pytest.mark.integration``, gated on ``NEO4J_INTEGRATION_TEST`` +
Postgres reachability (mirrors ``tests/integration/test_filter_pushdown_graph.py``),
so a no-Docker run collects-and-skips this module cleanly. ``ci.yml``'s
``test-integration`` job provisions PG+Neo4j and runs ``pytest tests/integration/
-m integration``, so these cells execute on every PR with zero workflow change.
Run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688)
    NEO4J_INTEGRATION_TEST=1 KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
        KHORA_NEO4J_URL=bolt://localhost:7688 KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/test_channel_filter_operator_matrix_change.py -m integration
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest
from pydantic import SecretStr

from khora import Khora
from khora.config import KhoraConfig
from khora.storage.temporal.pgvector import PgVectorTemporalStore
from tests.recall.test_channel_filter_operator_matrix import (
    _CHANGE_QUERY,
    _assert_channel_cell,
    _lower_entity_floor,
    _satisfying,
    _violating,
)
from tests.test_helpers.filter_spy import stub_llm

pytestmark = [pytest.mark.integration, pytest.mark.filter_enforcement]

# The Postgres pgvector column is fixed at 1536, so the live-DB suite sizes its
# deterministic vectors at 1536 (mirrors test_filter_pushdown_graph.py).
_PG_EMBED_DIM = 1536


# --------------------------------------------------------------------------- #
# Self-skip guards — mirror tests/integration/test_filter_pushdown_graph.py so a
# no-Docker run collects-and-skips cleanly.
# --------------------------------------------------------------------------- #


def _pg_reachable() -> bool:
    url = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False


_SKIP = pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST") or not _pg_reachable(),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (PG+Neo4j) to exercise the channel-matrix cells",
)


# --------------------------------------------------------------------------- #
# Live PG+Neo4j kb fixtures (mirror tests/integration/test_filter_pushdown_graph.py::kb).
# Two variants: the plain graph kb (session / change) and a recency-on kb (the
# pre-fix proof) with the recency channel flag flipped on at build time.
# --------------------------------------------------------------------------- #


def _build_config(*, recency_on: bool) -> KhoraConfig:
    """A vc_full ``KhoraConfig`` wired for the deterministic-stub live stack.

    Mirrors the integration graph fixture: 1536-dim embeddings, extraction on,
    selective-extraction off, HyDE / reranking off, the entity-similarity floor
    dropped so the deterministic-embedder entry entities clear it. When
    ``recency_on`` the recency channel flag is enabled with a low relevance floor
    so both keyword-sharing docs clear the cosine gate.
    """
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
    config.query.min_entity_similarity = 0.0
    if recency_on:
        config.query.temporal_recency_channel_enabled = True
        config.query.temporal_query_relevance_floor = 0.30
    return config


@pytest.fixture
async def kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora on the live PG+Neo4j stack (session / change)."""
    stub_llm(monkeypatch, dim=_PG_EMBED_DIM)
    instance = Khora(_build_config(recency_on=False), engine="vectorcypher", run_migrations=False)
    await instance.connect()
    _lower_entity_floor(instance)
    try:
        yield instance
    finally:
        await instance.disconnect()


@pytest.fixture
async def recency_kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """Connected ``vectorcypher`` Khora (live PG+Neo4j) with the recency channel ON.

    The pre-fix proof needs ``temporal_recency_channel_enabled=True`` so the recency
    channel runs and the (filter-stripped, pre-fix) shim can leak the freshest doc.
    """
    stub_llm(monkeypatch, dim=_PG_EMBED_DIM)
    instance = Khora(_build_config(recency_on=True), engine="vectorcypher", run_migrations=False)
    await instance.connect()
    _lower_entity_floor(instance)
    try:
        yield instance
    finally:
        await instance.disconnect()


# --------------------------------------------------------------------------- #
# Operator-class A representative — the leak-prone provenance ``$ne`` filter.
# --------------------------------------------------------------------------- #

_SOURCE_NAME_NE = {"source_name": {"$ne": "leakdoc"}}


# --------------------------------------------------------------------------- #
# (4) SESSION fan-out — every per-channel + fallback search threads + enforces.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_session_a_cell(kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Session × A: session fan-out enforces a ``source_name`` ``$ne`` on every per-channel search.

    The two docs name the same entity but land in DISTINCT ``metadata["channel"]``
    values (``chan-a`` / ``chan-b``), so the entity is ``MENTIONED_IN`` chunks
    spanning >= 2 channels and the temporal query fans out per-channel (retriever
    fan-out gate ``len(fanout_channels) >= 2``). The VIOLATING doc carries
    ``source_name="leakdoc"`` (forbidden); the SATISFYING doc carries
    ``source_name="cleandoc"`` and must survive. ``_assert_channel_cell`` installs
    the row spy on ``_vector_search_chunks``, runs the temporal recall, and asserts
    non-vacuity (>= 2 captures: per-channel + fallback), enforcement, and report
    honesty.
    """
    await _assert_channel_cell(
        kb,
        row="session",
        filter_spec=_SOURCE_NAME_NE,
        violating_doc=_violating("session leak note", source_name="leakdoc", metadata={"channel": "chan-a"}),
        satisfying_doc=_satisfying("session clean note", source_name="cleandoc", metadata={"channel": "chan-b"}),
        query="what did falcon do recently",
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


# --------------------------------------------------------------------------- #
# (7) CHANGE decomposition — the current-state sub-search threads + enforces.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_change_a_cell(kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Change × A: CHANGE decomposition enforces a ``source_name`` ``$ne`` on the sub-search.

    Both docs name the shared entity, so ``_fetch_version_history`` returns a row
    (its ``OPTIONAL MATCH`` surfaces the current entity) and the CHANGE-classified
    query ``"what did falcon used to make"`` is REWRITTEN by ``_decompose_change_query``
    to a distinct current-state sub-query (``"What does falcon make now?"``), firing a
    second current-state ``_vector_search_chunks`` carrying ``filter_ast``. The
    VIOLATING doc carries ``source_name="leakdoc"``; the SATISFYING doc carries
    ``source_name="cleandoc"`` and must survive. Non-vacuity (``_assert_change_fired``)
    is "a sub-search ran with a query_text != the original" — the rewrite proves the
    decomposition specifically fired AND threaded the filter (in GRAPH mode the main
    vector search is skipped, so the sub-search is the ONLY vector call).
    """
    await _assert_channel_cell(
        kb,
        row="change",
        filter_spec=_SOURCE_NAME_NE,
        violating_doc=_violating("falcon now makes cloud software", source_name="leakdoc"),
        satisfying_doc=_satisfying("falcon used to make hardware", source_name="cleandoc"),
        query=_CHANGE_QUERY,
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )


# --------------------------------------------------------------------------- #
# RECENCY pre-fix proof — the GitHub issue #1236 regression demonstration.
# --------------------------------------------------------------------------- #


def _install_prefix_recency_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the pre-GitHub-issue-#1236 recency leak via a monkeypatch shim.

    Before the fix, ``PgVectorTemporalStore.search_recent_chunks`` did not push the
    recall filter into the ``khora_chunks`` WHERE — it returned the freshest rows
    unfiltered (provenance-blank), so a ``source_name`` ``$ne`` filter matched-all
    and the freshest filter-violating chunk surfaced. This shim wraps the REAL
    method but DROPS ``filter_ast`` / ``filter_plan_out`` before delegating, so the
    recency SQL runs without the pushed predicate — the exact old behavior. Patched
    at the class level so the bound call ``self._vector_store.search_recent_chunks``
    (retriever recency channel, retriever.py:3700) picks it up; ``monkeypatch``
    undoes it at teardown so nothing is reverted in committed source.
    """
    real = PgVectorTemporalStore.search_recent_chunks

    async def _leaky_search_recent_chunks(
        self: PgVectorTemporalStore,
        namespace_id,
        limit,
        *,
        created_after=None,
        filter_ast=None,  # noqa: ARG001 — deliberately dropped to reproduce the leak
        filter_plan_out=None,  # noqa: ARG001 — deliberately dropped
    ):
        # Strip the filter: delegate to the real recency SQL WITHOUT filter_ast /
        # filter_plan_out, reproducing the provenance-blank pre-fix result set.
        return await real(self, namespace_id, limit, created_after=created_after)

    monkeypatch.setattr(PgVectorTemporalStore, "search_recent_chunks", _leaky_search_recent_chunks)


@_SKIP
@pytest.mark.asyncio
async def test_recency_prefix_leaks_postfix_clean(recency_kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recency × ``source_name``-``$ne`` cell leaks pre-fix and is clean post-fix.

    Two halves over the SAME recency firing recipe + the SAME provenance-``$ne``
    filter. The recency recipe's ``prepare`` hook backdates the ``violating`` doc to
    be the freshest on the recency axis; it carries ``source_name="leakdoc"`` and
    VIOLATES ``{"source_name": {"$ne": "leakdoc"}}``:

    1. PRE-FIX — with the leak shim installed (``filter_ast`` stripped before the
       recency SQL), the cell's "violating chunk absent" enforcement assertion
       FIRES: the freshest leak doc surfaces unfiltered. Wrapped in
       ``pytest.raises(AssertionError)`` so the leak is PROVEN, not silently
       tolerated. The captured message is pinned (it must be the recency leak) and
       is the regression evidence for the PR note.
    2. POST-FIX — after the shim's scoped ``MonkeyPatch.context()`` exits, the
       IDENTICAL cell runs on real code and PASSES: the leak doc is absent (the
       recency channel pushes the filter into the khora_chunks WHERE) and the report
       is honest.

    The shim is installed on its OWN ``MonkeyPatch.context()`` (``shim_mp``), NOT the
    test's function-scoped ``monkeypatch`` — because the ``recency_kb`` fixture
    installed ``stub_llm`` on that shared ``monkeypatch`` instance, so a bare
    ``monkeypatch.undo()`` would tear down the deterministic LLM stub too and the
    POST-FIX half's ``remember`` would hit the real extractor (no "Falcon" entity →
    the graph pre-flight fails). Scoping the shim leaves ``stub_llm`` intact: exiting
    the context undoes ONLY the leak shim + the recency spy (also installed on
    ``shim_mp`` inside the cell), and the POST-FIX half runs with the fixture's stub
    still live. No revert is left in committed source.
    """

    def _recency_docs() -> tuple:
        # The same violating (freshest) + satisfying docs both halves run, built the
        # SAME way the recency cell builds them (the ``violating`` external id is the
        # one the recency backdating prepare-hook keys off).
        return (
            _violating("recent leak note", source_name="leakdoc"),
            _satisfying("clean note", source_name="cleandoc"),
        )

    # 1. PRE-FIX — the leak shim makes the cell's enforcement assertion fire. Scope
    # the shim to its OWN MonkeyPatch so tearing it down does NOT touch the fixture's
    # stub_llm (installed on the test's shared ``monkeypatch``). The cell installs its
    # recency spy via the passed monkeypatch, so it lands on ``shim_mp`` here too and
    # is torn down with the context — the shim (installed first) is wrapped by the spy.
    with pytest.MonkeyPatch.context() as shim_mp:
        _install_prefix_recency_leak(shim_mp)
        violating, satisfying = _recency_docs()
        with pytest.raises(AssertionError) as excinfo:
            await _assert_channel_cell(
                recency_kb,
                row="recency",
                filter_spec=_SOURCE_NAME_NE,
                violating_doc=violating,
                satisfying_doc=satisfying,
                query="latest falcon launch update",
                monkeypatch=shim_mp,
                expect_satisfying_present=True,
            )
        # Pin the failure to the recency leak so a DIFFERENT failure (a vacuity guard,
        # a missing entry entity, etc.) cannot masquerade as the #1236 regression proof.
        assert "leaked through the 'recency' channel" in str(excinfo.value), (
            "the pre-fix half failed for a reason OTHER than the recency leak — the proof is "
            f"not demonstrating the #1236 regression. Got: {excinfo.value}"
        )

    # 2. POST-FIX — the shim context has exited (shim + its spy undone, stub_llm still
    # live on the fixture monkeypatch). Run the IDENTICAL cell on real code; it passes.
    violating, satisfying = _recency_docs()
    await _assert_channel_cell(
        recency_kb,
        row="recency",
        filter_spec=_SOURCE_NAME_NE,
        violating_doc=violating,
        satisfying_doc=satisfying,
        query="latest falcon launch update",
        monkeypatch=monkeypatch,
        expect_satisfying_present=True,
    )
