"""Recall-filter pushdown/threading spies on the embedded sqlite_lance stack.

These are the NO-DOCKER critical-path spies for the filter-enforcement
feature (#1051). They run the real ``Khora(engine="vectorcypher")`` against
a fully-embedded SQLite+LanceDB coordinator (per-test ``tmp_path``), drive a
genuine ``kb.recall(..., filter=...)``, and OBSERVE the ``filter_ast`` that
flows past each channel boundary — asserting it is the SAME canonical AST the
facade built, via ``canonical_hash`` equality plus a ``len(calls) >= N``
vacuity guard.

Scope discipline (mirrors the mock-level pushdown suite): we pin the WIRING
contract — the validated filter reaches the channel unchanged — NOT the row
set. The end-to-end "the rows are actually narrowed" proof is the
filter-conformance suite's job. Spying here passes through to the real method,
so the recall executes its genuine logic; we only read what was threaded.

What enforces vs. only wires on THIS embedded backend (verified against
source — the no-Docker stack is the point):

* VECTOR channel — ``SQLiteLanceTemporalStore.search`` builds a
  ``compile_python`` post-filter and a ``compile_lance`` JSON1 pushdown, so the
  filter is genuinely ENFORCED. The spy asserts the wiring; enforcement is the
  conformance suite's row-proof.
* BM25 channel — the embedded ``SQLiteLanceTemporalStore.search_fulltext`` now
  forwards ``filter_ast`` into ``_bm25_search``, which pushes the compilable
  leaves into SQL and re-checks the rest against the decoded chunk, so the
  embedded BM25 channel now ENFORCES the recall filter (matching the pgvector
  sibling; see the PG variant in the qa-graph suite). The spy proves the
  retriever THREADS the right filter to that boundary, and an enforcement probe
  proves a contradiction filter yields an empty BM25-only recall.

No Docker / Postgres / Neo4j — pure in-process aiosqlite + lancedb.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:  # Module-level import gate matches the sibling sqlite_lance suites.
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.extraction.skills import ExpertiseConfig
from khora.khora import Khora
from khora.query import SearchMode
from khora.query.temporal_detection import TemporalCategory
from tests.test_helpers.filter_spy import (
    EMBED_DIM,
    assert_filter_threaded,
    plan_extraction,
    seed_corpus,
    spy_on,
    stub_llm,
)

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.filter_enforcement,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic embedder + content-keyed entity extractor (no API key)."""
    stub_llm(monkeypatch)


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    """Per-test VectorCypher Khora on a fresh embedded stack.

    Mirrors the sibling sqlite_lance suite: own ``tmp_path`` per test because
    ``StorageFactory`` caches engine pools by URL.
    """
    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.neo4j_url = None
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="vectorcypher", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        try:
            await kb.disconnect()
        except Exception:
            pass


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


def _expertise() -> ExpertiseConfig:
    return ExpertiseConfig(name="vc-filter-enforcement-embedded")


def _remember(kb: Khora) -> Any:
    """Bind the per-test remember wiring for ``seed_corpus``."""
    return partial(
        kb.remember,
        title="",
        entity_types=["PERSON", "CONCEPT", "EVENT", "ORG"],
        relationship_types=["KNOWS", "RELATES_TO", "MENTIONS"],
        expertise=_expertise(),
    )


def _retriever(kb: Khora) -> Any:
    """The live VectorCypherRetriever instance (built at connect)."""
    return kb._engine._get_retriever()  # type: ignore[union-attr]


# A representative system-key filter that compiles cleanly on every backend:
# a typed source column plus a date range. Both keys are projected onto the
# Chunk row, so the vector channel pushes them down.
_FILTER = {
    "source_name": "linear",
    "occurred_at": {"$gte": "2026-01-01"},
}


async def _seed(kb: Khora, namespace_id: UUID) -> None:
    """Seed an entity-bearing corpus shared across the path spies."""
    plan_extraction(
        "Falcon",
        entities=[("Falcon", "EVENT"), ("SpaceX", "ORG")],
        relationships=[("SpaceX", "Falcon", "RELATES_TO")],
    )
    await seed_corpus(
        _remember(kb),
        namespace_id,
        [
            "Falcon launch coordinated by SpaceX in early 2026.",
            "Falcon recovery operations reported by SpaceX teams.",
            "Falcon mission telemetry archived after the SpaceX review.",
        ],
    )


# --------------------------------------------------------------------------- #
# Path 1 — VECTOR channel (_vector_search_chunks -> temporal store search)
# --------------------------------------------------------------------------- #


async def test_vector_channel_threads_filter(kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch) -> None:
    """The vector chunk channel receives the facade's exact filter AST.

    ENFORCES on embedded (compile_lance + compile_python). Here we pin the
    wiring: ``_vector_search_chunks`` is called and every call carries a
    ``filter_ast`` whose canonical_hash matches the facade-built AST.
    """
    await _seed(kb, namespace_id)
    records = spy_on(monkeypatch, _retriever(kb), "_vector_search_chunks")

    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter=_FILTER)

    assert_filter_threaded(records, _FILTER, min_calls=1)


# --------------------------------------------------------------------------- #
# Path 2 — BM25 channel (_bm25_search_chunks -> temporal store search_fulltext)
# --------------------------------------------------------------------------- #


async def test_bm25_channel_threads_filter(kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch) -> None:
    """The BM25 channel receives the facade's exact filter AST.

    Embedded ``SQLiteLanceTemporalStore.search_fulltext`` now forwards
    ``filter_ast`` into ``_bm25_search`` (which compiles + post-filters it), so
    the channel ENFORCES the filter. This spy still pins the WIRING contract:
    the retriever THREADS the correct filter to that boundary. (The row-level
    enforcement proof is the probe below.)

    DIFFERENTIAL gate-fire check (avoids a vacuous wiring spy on a dead
    channel): the BM25 channel is default-off in ``RetrieverConfig``. We first
    confirm that with the flag OFF the channel does NOT fire (control), then
    enable it and confirm it fires AND threads the filter. So the spy is proven
    to be observing a genuinely-active channel, not silently passing because
    ``_bm25_search_chunks`` was never reached.
    """
    await _seed(kb, namespace_id)
    retriever = _retriever(kb)

    # Control: flag OFF -> channel does not fire.
    retriever._config.enable_bm25_channel = False
    off_records = spy_on(monkeypatch, retriever, "_bm25_search_chunks")
    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter=_FILTER)
    assert not off_records, (
        f"BM25 channel fired with enable_bm25_channel=False ({len(off_records)} call(s)); "
        f"the flag is not the gate, so the wiring assertion below would not prove the channel is live."
    )

    # Flag ON -> channel fires AND threads the facade filter.
    retriever._config.enable_bm25_channel = True
    on_records = spy_on(monkeypatch, retriever, "_bm25_search_chunks")
    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter=_FILTER)
    assert_filter_threaded(on_records, _FILTER, min_calls=1)


async def test_bm25_channel_enforces_filter_embedded(kb: Khora, namespace_id: UUID) -> None:
    """A filter that excludes every chunk must yield an empty BM25-only recall.

    Embedded BM25 now forwards ``filter_ast`` into ``_bm25_search``, so a
    contradiction filter (a ``source_name`` that matches nothing in the corpus)
    excludes every row through the BM25 channel and the recall is empty. Mirrors
    the pgvector sibling's enforcement contract.
    """
    await _seed(kb, namespace_id)
    retriever = _retriever(kb)
    retriever._config.enable_bm25_channel = True

    # A source_name that matches NOTHING in the seed corpus.
    contradiction = {"source_name": "no-such-source-xyzzy"}
    result = await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        mode=SearchMode.KEYWORD,
        filter=contradiction,
    )
    assert result.chunks == [], f"filter-excluded chunks leaked through embedded BM25: {len(result.chunks)}"


# --------------------------------------------------------------------------- #
# Path 6 — EXPLICIT-synthesis decision (engine.py date-key gate)
# --------------------------------------------------------------------------- #


async def test_date_filter_synthesizes_explicit_signal(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A date-key caller filter drives an EXPLICIT (source="api") temporal signal.

    The engine gates EXPLICIT synthesis on ``filter_constrains_date_key`` — a
    date predicate makes the recall an explicit temporal intent. We capture the
    ``temporal_signal`` the retriever receives and assert the category, plus the
    threaded filter hash (same facade AST).
    """
    await _seed(kb, namespace_id)
    records = spy_on(monkeypatch, _retriever(kb), "retrieve")

    # occurred_at predicate -> date key -> EXPLICIT.
    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter={"occurred_at": {"$gte": "2026-01-01"}})

    assert_filter_threaded(records, {"occurred_at": {"$gte": "2026-01-01"}}, min_calls=1)
    signal = records[0].kwargs["temporal_signal"]
    assert signal is not None and signal.category == TemporalCategory.EXPLICIT, (
        f"date-key filter must synthesize EXPLICIT, got {getattr(signal, 'category', None)}"
    )
    assert signal.source == "api", f"EXPLICIT from a caller filter must be source='api', got {signal.source!r}"


async def test_non_date_filter_does_not_synthesize_explicit(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pure-metadata caller filter must NOT trigger EXPLICIT synthesis.

    DIFFERENTIAL (avoids a vacuous negative): the SAME non-temporal query
    ``"Falcon launch"`` is run twice against the SAME stack — once with a DATE
    filter (positive control: MUST synthesize EXPLICIT) and once with a
    non-date filter (MUST NOT). If the date-key gate were dead (always or never
    EXPLICIT), one of the two halves fails. So a passing test proves the
    EXPLICIT decision is driven by the date key, not an incidental constant.
    """
    await _seed(kb, namespace_id)
    retriever = _retriever(kb)

    # Positive control: a date-key filter on this exact query DOES go EXPLICIT.
    pos_records = spy_on(monkeypatch, retriever, "retrieve")
    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter={"occurred_at": {"$gte": "2026-01-01"}})
    pos_signal = pos_records[0].kwargs["temporal_signal"]
    assert pos_signal is not None and pos_signal.category == TemporalCategory.EXPLICIT, (
        "positive control failed: a date-key filter must synthesize EXPLICIT on this query, "
        f"got {getattr(pos_signal, 'category', None)} — the gate is not firing, so the "
        "negative half below would be vacuous."
    )

    # Negative case: same query, non-date filter -> must NOT go EXPLICIT.
    neg_records = spy_on(monkeypatch, retriever, "retrieve")
    await kb.recall("Falcon launch", namespace=namespace_id, limit=10, filter={"source_name": "linear"})
    assert_filter_threaded(neg_records, {"source_name": "linear"}, min_calls=1)
    neg_signal = neg_records[0].kwargs["temporal_signal"]
    assert neg_signal is None or neg_signal.category != TemporalCategory.EXPLICIT, (
        f"non-date filter must NOT synthesize EXPLICIT, got {getattr(neg_signal, 'category', None)}"
    )


# --------------------------------------------------------------------------- #
# Path 5 — restrictive-fallback guard, embedded fail-fast contract.
#
# The restrictive-filter unfiltered re-run (retriever.py:1479) only fires when a
# point-in-time ``temporal_filter`` (occurred bounds) reached the retriever. The
# embedded sqlite_lance backend lacks bi-temporal version columns, so it FAILS
# FAST on any point-in-time temporal query *before* retrieval (retriever.py:601)
# rather than silently returning current-state rows. That fail-fast makes the
# re-run path itself unreachable embedded — so the re-run-guard enforcement
# (filter_ast suppresses the unfiltered re-run) is proven on the PG+Neo4j stack
# (qa-graph), where point-in-time is honored. Here we pin the embedded promise
# the fail-fast actually makes: a point-in-time query raises cleanly and never
# degrades into an unfiltered current-state recall.
# --------------------------------------------------------------------------- #


async def test_embedded_point_in_time_fails_fast(kb: Khora, namespace_id: UUID) -> None:
    """A point-in-time temporal query raises NotImplementedError on embedded.

    This is the embedded backstop behind path 5: the embedded stack cannot honor
    occurred-bounds temporal queries, so it refuses them rather than returning
    unfiltered current-state rows (which would silently violate any concurrent
    recall filter). The clean raise is the contract; this guards it against a
    regression that would let those queries through unfiltered.
    """
    await _seed(kb, namespace_id)

    # "last week" resolves to an occurred-bounds temporal_filter -> point-in-time.
    with pytest.raises(NotImplementedError, match="sqlite_lance"):
        await kb.recall("Falcon launch last week", namespace=namespace_id, limit=10)
