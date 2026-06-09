"""Full-stack integration tests for the recall-filter on VectorCypher.

INTERIM TEST — SUPERSEDED BY THE FILTER-CONFORMANCE SUITE. This is the
VectorCypher counterpart to ``tests/recall/test_filter_skeleton_pgvector.py``.
Per the project's filter-verification strategy, the live row-set assertions
here reduce to the dedicated filter-conformance corpus: their permanent home
is a separate filter-conformance CI job (``tests/integration/matrix/``, its own
conformance marker, per-engine databases) that is not built yet. That job is
deliberately excluded from the main test job so the conformance cases never
double-run. This file lives in ``tests/recall/`` and is collected by the main
test job's unit step, where the LIVE scenarios SELF-SKIP because that job
provisions no Postgres/Neo4j — they gate locally via ``make dev`` only. The
two HERMETIC telemetry tests (Scenario 4) carry no skip marker and run as real
gates in every job, since they mock the retriever/engine internals and touch no
database. When the filter-conformance suite lands, MIGRATE/REMOVE the live
scenarios so the assertions do not double-run against the conformance corpus.
Do not entrench them (no Postgres/Neo4j service should be added to the main
test job for the live tests).

Where ``tests/unit/engines/test_vectorcypher_filter_pushdown.py`` exercises the
engine→channel WIRING with mocks (that the validated ``filter_ast`` reaches both
chunk channels), this file drives the *public* API end-to-end on the live
scenarios: real ``Khora.remember()`` writes through the VectorCypher engine into
real ``khora_chunks`` + Neo4j, and ``Khora.recall(filter=...)`` must narrow the
result to exactly the in-scope rows.

Scenarios (mirroring V1's S1/S2/S5 + hermetic telemetry and partial-failure pairs):

* S1 — row-set (LIVE): 5 in-scope + 5 out-of-scope chunks, each out-of-scope row
  violating EXACTLY ONE of the three predicates (wrong ``source_name``;
  ``occurred_at`` too old; ``metadata.tag`` out of set / missing). A single
  three-predicate ``recall(filter=...)`` must return EXACTLY the 5 in-scope
  chunk ids. The per-row single-violation design proves each predicate bites
  independently. A no-filter control proves the split is a property of the
  FILTER, not retrieval reachability.
* S2 — engine_info carrier + validation (LIVE): ``engine_info["filter"]`` reports
  the VectorCypher carrier with HONEST flags (``engine="vectorcypher"``,
  ``supported=False``, ``pushed_down=False`` — VectorCypher fans the filter
  across multiple channels and does NOT claim whole-AST pushdown), and an invalid
  filter raises ``RecallFilterValidationError`` through the ``recall()`` façade.
* S5 — backward-compat (LIVE): the deprecated ``start_time``/``end_time`` shim
  filters rows end-to-end on the live VectorCypher lake; passing BOTH ``filter=``
  and the deprecated bounds raises ``ValueError``.
* S4 — filter telemetry: the two service-level filter counters fire for the
  RIGHT reason, spied via the ``_RecordingCounter`` singleton hook. Two layers:
  a LIVE real-graph-path proof and HERMETIC wiring tests.
  4a-LIVE — ``graph_channel_empty`` increments when the REAL graph channel,
  built from genuine MENTIONED_IN edges (a deterministic entity-emitting
  extractor + a ``SearchMode.GRAPH`` recall, which deterministically routes to
  the graph path regardless of the router's complexity heuristic), is emptied by
  the metadata post-filter; a pre-flight asserts the channel actually HELD
  candidates so the signal is non-vacuous. Self-skips without the stack.
  4a-HERMETIC — the same counter at the retriever seam with mocked channels (runs
  everywhere), plus a survivor control. 4b-HERMETIC — ``under_filled`` increments
  when a filtered VectorCypher recall returns fewer chunks than the requested
  limit (driven through the real engine with a mocked retriever), plus a
  no-filter control.
* S6 — partial-failure vs transient-fallback (HERMETIC): a filter the Cypher
  compiler cannot honor raises ``RecallFilterUnsupportedError`` out of
  ``retrieve()`` (NOT masked as a vector-only fallback — the over-fetch probe runs
  the compiler outside the transient-error handler), while a TRANSIENT Neo4j error
  from ``_cypher_expand`` IS caught and degrades to vector-only (``graph_fallback``
  set, chunks still returned). Both run everywhere.

ENVIRONMENT: the live scenarios need the Docker Compose Postgres + Neo4j from
THIS repo (``make dev``, Postgres port 5434, Neo4j bolt 7687). They skip cleanly
when either is unreachable (e.g. the CI lint sandbox) and RUN as the real gate in
the CI integration job. The public API is HARD-imported (not ``importorskip``) so
a broken import is a LOUD error, never a silent skip.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from neo4j.exceptions import ServiceUnavailable

import khora.filter.telemetry as filter_telemetry

# Hard import (NOT importorskip): the filter surface and the Khora façade are on
# the branch, so an import failure must be a LOUD test error — never a silent
# module skip. (The live scenarios still skip when the databases are unreachable,
# via the live_db fixture's skip below.)
from khora.config import KhoraConfig
from khora.core.models import Chunk
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.extraction.extractors.base import ExtractionResult
from khora.filter import (
    RecallFilter,
    RecallFilterUnsupportedError,
    RecallFilterValidationError,
    parse_to_ast,
)
from khora.khora import Khora
from khora.query import SearchMode

# Postgres backend supports only embedding_dimension=1536 (schema.py), so the
# stub embedder must emit 1536-dim vectors to match the deployed schema.
EMBED_DIM = 1536


# This repo's compose puts Postgres on 5434 (see compose.yaml). Honor an explicit
# override, else default to the compose port — never another project's container.
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


def _tcp_reachable(url: str, default_port: int) -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _live_stack_reachable() -> bool:
    # VectorCypher's primary recall path needs BOTH Postgres (chunk channels) and
    # Neo4j (graph expansion); require both before running the live scenarios.
    return _tcp_reachable(_database_url(), 5432) and _tcp_reachable(_neo4j_url(), 7687)


# Only the LIVE scenarios (S1/S2/S5) carry this skip. The hermetic Scenario-4
# telemetry tests run everywhere — they are decorated individually, NOT via this
# module-level mark, so they gate in the main test job too.
_LIVE_DB = pytest.mark.skipif(
    not _live_stack_reachable(),
    # Not accidental missing coverage: this is an interim smoke test whose
    # permanent CI coverage is the dedicated filter-conformance job. It
    # self-skips without the local Postgres + Neo4j stack (run `make dev`).
    reason=(
        "interim smoke test; permanent CI coverage is the dedicated "
        "filter-conformance job. Self-skips without the local Postgres + Neo4j "
        "stack (run `make dev`)."
    ),
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Deterministic LLM stubs — no OPENAI_API_KEY required.
# ---------------------------------------------------------------------------
#
# Every text maps to the SAME 1536-dim unit vector, so query and all chunk
# embeddings are identical → cosine similarity is 1.0 for every row. Retrieval
# is therefore filter-bound, not similarity-bound: the WHERE predicate is the
# only thing that narrows the candidate set, which is exactly what S1 must
# isolate. Entity extraction is irrelevant to the filter contract, so the LLM
# extractor is stubbed to return nothing — the chunk channels carry the filter.


def _unit_vector() -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_unit_vector() for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _unit_vector()


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    return [ExtractionResult() for _ in texts]


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


# ---------------------------------------------------------------------------
# VectorCypher Khora fixture (live Postgres + Neo4j).
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    """A connected Khora on the VectorCypher engine over the compose stack.

    ``run_migrations=True`` materializes the schema, including ``khora_chunks``,
    on a fresh DB. Entity extraction is disabled — the filter contract lives on
    the chunk channels and the stubbed extractor returns nothing anyway.
    """
    config = KhoraConfig(database_url=_database_url(), neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
    config.pipeline.extract_entities = False
    config.pipeline.selective_extraction = False
    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        try:
            await instance.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Seed corpus — 5 in-scope, 5 out-of-scope (one predicate violation each).
# ---------------------------------------------------------------------------
#
# The three-predicate filter under test:
#   source_name == "linear"  AND  occurred_at >= 2026-04-05
#   AND  metadata.tag IN {"urgent", "release"}
#
# Reused VERBATIM from the V1 skeleton corpus so the two engines are pinned
# against an identical row-set — any divergence is an engine difference, not a
# corpus difference. Content is DISTINCT per row so the dedupe-by-checksum write
# path keeps every row as its own chunk.

_IN_BOUND = "2026-04-05T00:00:00Z"  # the occurred_at lower bound (inclusive)

# label -> (source_name, occurred_at ISO, metadata.tag | None)
_IN_SCOPE: dict[str, tuple[str, str, str | None]] = {
    # All three predicates satisfied. Boundary, mid, and future occurred_at;
    # both allowed tag values are represented.
    "in_boundary": ("linear", _IN_BOUND, "urgent"),  # exactly at the >= bound
    "in_recent": ("linear", "2026-05-01T12:00:00Z", "release"),
    "in_future": ("linear", "2026-06-01T00:00:00Z", "urgent"),
    "in_release": ("linear", "2026-04-10T00:00:00Z", "release"),
    "in_urgent": ("linear", "2026-04-20T09:30:00Z", "urgent"),
}

# Each out-of-scope row violates EXACTLY ONE predicate; every other field is
# in-scope, so a leak pins which predicate failed to bite.
_OUT_OF_SCOPE: dict[str, tuple[str | None, str, str | None]] = {
    # Predicate 1 (source_name) violated; date + tag are in-scope.
    "out_wrong_source": ("slack", "2026-05-15T00:00:00Z", "urgent"),
    # Predicate 2 (occurred_at) violated: one day before the inclusive bound.
    "out_too_old": ("linear", "2026-04-04T23:59:59Z", "release"),
    # Predicate 3 (tag) violated: tag present but outside the allowed set.
    "out_wrong_tag": ("linear", "2026-05-20T00:00:00Z", "backlog"),
    # Predicate 3 (tag) violated: tag key absent entirely.
    "out_missing_tag": ("linear", "2026-05-25T00:00:00Z", None),
    # Predicate 1 (source_name) violated via NULL: source_name omitted at write.
    "out_null_source": (None, "2026-05-30T00:00:00Z", "release"),
}

_RECALL_FILTER = {
    "source_name": "linear",
    "occurred_at": {"$gte": _IN_BOUND},
    "metadata.tag": {"$in": ["urgent", "release"]},
}


def _content_for(label: str) -> str:
    # Distinct, short (single-chunk) content per row, with a stable token so a
    # keyword channel would also surface it if a mode ever uses BM25.
    return f"chunk {label} alpha bravo charlie {hashlib.sha256(label.encode()).hexdigest()[:8]}"


async def _seed(kb: Khora, namespace_id: UUID) -> dict[str, UUID]:
    """Remember every row and return a ``label -> chunk_id`` map.

    Each ``remember`` produces exactly one chunk (content < chunk_size). The
    chunk id is read back from ``khora_chunks`` by ``document_id`` so the tests
    assert on real, server-assigned chunk ids rather than guessing them.
    """
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    doc_to_label: dict[UUID, str] = {}
    for label, (source_name, occurred_at, tag) in {**_IN_SCOPE, **_OUT_OF_SCOPE}.items():
        metadata: dict[str, Any] = {"occurred_at": occurred_at}
        if tag is not None:
            metadata["tag"] = tag
        result = await kb.remember(
            content=_content_for(label),
            namespace=namespace_id,
            title=label,
            source_name=source_name,
            metadata=metadata,
            entity_types=[],
            relationship_types=[],
        )
        doc_to_label[result.document_id] = label

    # Resolve each document's single chunk id from the live khora_chunks table.
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.text("SELECT id, document_id FROM khora_chunks WHERE namespace_id = :ns"),
                    {"ns": namespace_id},
                )
            ).fetchall()
    finally:
        await engine.dispose()

    label_to_chunk: dict[str, UUID] = {}
    for chunk_id, document_id in rows:
        label_to_chunk[doc_to_label[document_id]] = chunk_id

    assert len(label_to_chunk) == len(doc_to_label), "each remembered doc must yield exactly one chunk"
    return label_to_chunk


def _ids_for(chunk_ids: dict[str, UUID], labels: dict[str, Any]) -> set[UUID]:
    return {chunk_ids[label] for label in labels}


# ===========================================================================
# S1 — row-set: the three-predicate filter returns EXACTLY the in-scope ids.
# ===========================================================================


@_LIVE_DB
async def test_filter_returns_exactly_in_scope_chunks(kb: Khora) -> None:
    """``recall(filter=...)`` narrows the live result to exactly the 5 in-scope
    chunks — no out-of-scope leak.

    This is the end-to-end proof that the VectorCypher engine→channel wiring
    actually filters rows: the facade builds the AST, the engine threads it into
    the vector chunk channel, and the pgvector store compiles a ``khora_chunks``
    WHERE predicate. ``SearchMode.VECTOR`` routes to the graph-less
    ``_simple_retrieve`` path (retriever.py:737 ``force_simple``) with the BM25
    channel and pgvector-internal BM25 fusion both OFF (retriever.py:2390/2396),
    so VectorCypher here is pure pgvector WHERE-push — structurally the same
    filter path as the V1 Skeleton engine, which is why the V1 corpus is reused
    verbatim and the same exact in-scope set is expected. Every out-of-scope row
    violates exactly one predicate, so a leak would name the broken one.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    chunk_ids = await _seed(kb, namespace_id)
    in_scope = _ids_for(chunk_ids, _IN_SCOPE)
    out_of_scope = _ids_for(chunk_ids, _OUT_OF_SCOPE)
    assert len(in_scope) == 5, "expected 5 in-scope chunks seeded"
    assert len(out_of_scope) == 5, "expected 5 out-of-scope chunks seeded"

    # limit comfortably exceeds the corpus so the filter, not the limit, bounds
    # the result. VECTOR mode keeps retrieval purely embedding+filter (no BM25
    # keyword channel) so the filter is the only narrowing force.
    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    returned = {c.id for c in result.chunks}
    assert returned == in_scope, (
        f"filter must return EXACTLY the in-scope chunk ids; "
        f"leaked={returned & out_of_scope}, missing={in_scope - returned}"
    )


@_LIVE_DB
async def test_no_filter_returns_all_chunks(kb: Khora) -> None:
    """Control: with no filter the same recall returns the whole corpus.

    Proves the in-scope/out-of-scope split is a property of the FILTER, not of
    retrieval reachability — every seeded chunk is recallable absent the filter,
    so S1's narrowing is attributable to the predicate alone.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    chunk_ids = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    returned = {c.id for c in result.chunks}
    assert returned == set(chunk_ids.values()), "unfiltered recall must reach every seeded chunk"


# ===========================================================================
# S2 — engine_info carrier + facade-level filter validation.
# ===========================================================================


@_LIVE_DB
async def test_engine_info_reports_vectorcypher_filter_carrier(kb: Khora) -> None:
    """``engine_info['filter']`` carries the VectorCypher carrier with HONEST flags.

    Unlike the skeleton-pgvector path (whole-filter SQL pushdown, so
    ``supported=True`` / ``pushed_down=True``), VectorCypher fans the filter
    across multiple channels (vector + BM25 push down; the graph channel applies
    an in-memory post-filter) and does NOT claim whole-AST pushdown. The façade
    therefore surfaces the honest carrier: ``engine="vectorcypher"``,
    ``supported=False`` (only the skeleton engine is the declared pushdown
    target), ``pushed_down=False`` (the VectorCypher engine never stamps a
    ``pushed_down=True``, so the façade default holds). The assertion that
    actually matters — that the filter narrows to exactly the in-scope rows — is
    re-checked here too.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    chunk_ids = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    info = (result.engine_info or {}).get("filter")
    assert info is not None, "engine_info['filter'] carrier must be present on a filtered recall"
    assert info["engine"] == "vectorcypher"
    assert info["supported"] is False, "only the skeleton engine declares filter support"
    assert info["pushed_down"] is False, "VectorCypher does not claim whole-AST pushdown"

    # The assertion that actually matters: the filter narrows to exactly the
    # in-scope rows (same contract as S1, re-pinned alongside the carrier).
    assert {c.id for c in result.chunks} == _ids_for(chunk_ids, _IN_SCOPE)


@_LIVE_DB
async def test_invalid_filter_raises_validation_error(kb: Khora) -> None:
    """An invalid filter raises ``RecallFilterValidationError`` through recall().

    Covers both validation failure shapes: an unknown top-level key and an
    illegal operator for an otherwise-valid key. Both must surface the typed
    error from ``khora.filter`` before any retrieval work.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    with pytest.raises(RecallFilterValidationError):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            filter={"not_a_real_key": "x"},
        )

    with pytest.raises(RecallFilterValidationError):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            # $contains is not a legal operator on the occurred_at system key.
            filter={"occurred_at": {"$contains": "x"}},
        )


# ===========================================================================
# S5 — backward-compat: the deprecated start_time/end_time bounds, live path.
# ===========================================================================
#
# The shim MECHANICS (DeprecationWarning emission, fold-to-AST, the
# filter=+bounds ValueError) are already covered against a mock engine in the
# unit suite. These tests add the gap that mocks can't reach: that the deprecated
# bounds actually FILTER ROWS end-to-end on the live VectorCypher lake.


@_LIVE_DB
async def test_start_time_bound_filters_rows_live(kb: Khora) -> None:
    """``start_time=`` (deprecated) filters rows end-to-end on the live lake.

    The bound folds to ``occurred_at >= start`` and AND-s into the same
    khora_chunks predicate path the public ``filter=`` uses. With the bound set
    to the in-scope lower edge, only rows at/after it survive — the same
    occurred_at split S1 exercises, but reached through the legacy kwarg. The
    DeprecationWarning is asserted here too so the live path is exercised under
    the warning contract, not just the mocked unit path.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    chunk_ids = await _seed(kb, namespace_id)

    # occurred_at >= 2026-04-05 excludes only "out_too_old" (2026-04-04). Every
    # other seeded row (in-scope + the other out-of-scope rows) is at/after the
    # bound, so 9 of 10 survive — this isolates the temporal predicate alone.
    start = datetime(2026, 4, 5, tzinfo=UTC)

    with pytest.warns(DeprecationWarning):
        result = await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            start_time=start,
        )

    returned = {c.id for c in result.chunks}
    expected = set(chunk_ids.values()) - {chunk_ids["out_too_old"]}
    assert chunk_ids["out_too_old"] not in returned, "start_time= must exclude the pre-bound row"
    assert returned == expected, f"expected 9 of 10 rows at/after the bound, got {len(returned)}"


@_LIVE_DB
async def test_filter_and_bounds_conflict_raises_live(kb: Khora) -> None:
    """Passing BOTH filter= and the deprecated bounds raises ValueError.

    Asserted on the live façade to confirm the guard fires before any engine/DB
    work on the real path, complementing the mocked unit coverage.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    with pytest.raises(ValueError, match="filter= or the deprecated start_time/end_time"):
        await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            mode=SearchMode.VECTOR,
            filter=_RECALL_FILTER,
            start_time=datetime(2026, 4, 5, tzinfo=UTC),
        )


# ===========================================================================
# S4 — filter telemetry. The two service-level filter counters fire for the
# RIGHT reason. Two layers:
#   * a LIVE real-graph-path proof (4a-live, self-skips) that drives the
#     ``graph_channel_empty`` counter through genuine MENTIONED_IN edges built by
#     a deterministic entity-emitting extractor + a graph-driving SearchMode —
#     the graph channel really holds candidates before the metadata post-filter
#     empties them, so the signal is NOT vacuous; and
#   * HERMETIC wiring tests (4a/4b, no DB, run everywhere) that pin the same two
#     counters at the retriever/engine seam with mocked channels.
# ===========================================================================
#
# Both counters live in ``khora.filter.telemetry`` as lazily-built module
# singletons. The retriever calls ``record_graph_channel_empty`` and the engine
# calls ``record_under_filled``; each helper routes through the lazy getter,
# which returns an already-set singleton. So pre-seeding the module global with a
# recording fake (the same hook the existing recall-filter telemetry tests use)
# captures the ``.add(...)`` calls without a real MeterProvider. The fake is the
# canonical ``_RecordingCounter`` from the chronicle filter suite.


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


# ---------------------------------------------------------------------------
# 4a-LIVE — real graph path: a deterministic entity-emitting extractor builds
# genuine MENTIONED_IN edges so the graph channel really holds candidates, then
# a metadata predicate they all violate empties it. Self-skips without the stack.
# ---------------------------------------------------------------------------
#
# This is the NON-VACUOUS proof: the graph channel must have candidates BEFORE
# the metadata post-filter runs. We get them through the real machinery —
# extraction writes an entity per doc (so the entity-level vector index has a
# seed), the ingest wires MENTIONED_IN edges, and a ``SearchMode.GRAPH`` recall
# (which deterministically routes to the graph path) expands those edges into
# graph-channel chunks. Every seeded doc carries ``metadata.tag == "noise"``; the
# filter demands ``tag IN {"urgent"}``, so the full-AST post-filter drops every
# graph chunk and ``record_graph_channel_empty`` fires for the genuine reason.

# The shared entity the graph-side docs mention. Under the autouse unit-vector
# embedder, the entity embedding equals the query embedding (cosine 1.0), so the
# entry-entity vector search returns it as the seed the graph expansion needs.
# The marker keeps the entity-emitting extractor scoped to these docs (a hook for
# a future off-graph control); every doc in this test carries it.
_GRAPH_ENTITY_NAME = "Falcon"
_GRAPH_MARKER = "graphdoc"


async def _stub_extract_multi_with_entity(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    """Emit the shared entity for graph-side docs (those carrying the marker).

    Marker-carrying docs get a real MENTIONED_IN edge to ``_GRAPH_ENTITY_NAME`` so
    the graph expansion surfaces them; any doc without the marker extracts nothing
    and never enters the graph expansion.
    """
    from khora.extraction.extractors.base import ExtractedEntity

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


@pytest.fixture
async def kb_graph(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    """A connected VectorCypher Khora with entity extraction ON.

    Re-patches ``extract_multi`` AFTER the autouse ``_patch_llm`` (so the
    entity-emitting stub wins) and enables ``extract_entities`` so the ingest
    builds real MENTIONED_IN edges. Reranking is disabled to keep the live path
    light and avoid loading a cross-encoder model in CI.
    """
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi_with_entity,
    )
    config = KhoraConfig(database_url=_database_url(), neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False
    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        try:
            await instance.disconnect()
        except Exception:
            pass


@_LIVE_DB
async def test_graph_channel_empty_counter_fires_on_real_graph_path(
    kb_graph: Khora, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4a-LIVE — ``graph_channel_empty`` fires when the REAL graph channel (built
    from genuine MENTIONED_IN edges) is emptied by the metadata post-filter.

    Mode is ``SearchMode.GRAPH`` ON PURPOSE: the retriever's mode dispatch sets
    ``force_graph`` only for ``GRAPH`` (retriever.py:738), which routes
    deterministically to ``_vectorcypher_retrieve`` regardless of the query
    router's complexity heuristic. ``HYBRID`` would be fragile here — the router
    classifies a short entity query as SIMPLE / ``use_graph=False`` and the HYBRID
    dispatch then falls to the graph-less ``_simple_retrieve``, leaving the graph
    channel empty and this proof vacuous (verified against the live router).
    ``GRAPH`` mode does skip the vector + BM25 chunk channels, so the
    ``empty_under_filter`` degradation (which needs a surviving vector/BM25 row)
    is NOT exercised here — that arm is pinned deterministically by the hermetic
    ``test_graph_channel_empty_counter_fires_for_emptied_channel`` below. This
    test's unique job is the COUNTER on a genuinely-populated real graph channel.

    Corpus: 3 docs that mention the shared entity (``_GRAPH_MARKER`` → extraction
    emits it → real MENTIONED_IN edges), all tagged ``"noise"``. Under the
    unit-vector embedder the entity embedding equals the query embedding, so the
    entry-entity vector search seeds the graph expansion. A pre-flight GRAPH recall
    with NO filter asserts ``graph_chunk_count > 0`` — proving the channel really
    HELD candidates (never vacuous). The filter ``tag IN {"urgent"}`` then drops
    every graph chunk (all ``"noise"``), so ``graph_channel_empty`` increments.
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


def _graph_chunk(ns_id: UUID, *, tag: str) -> Chunk:
    """A chunk the graph channel returns, carrying a ``metadata.tag`` value."""
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"graph chunk tagged {tag}",
        metadata={"tag": tag},
    )


def _make_retriever(ns_id: UUID) -> VectorCypherRetriever:
    """A retriever wired so BOTH the vector and BM25 chunk channels fire.

    Mirrors ``_make_retriever`` in
    ``tests/unit/engines/test_vectorcypher_filter_pushdown.py`` (the MODERATE /
    ``_vectorcypher_retrieve`` path): the vector channel returns one row and the
    BM25 channel returns one row, so the "vector/BM25 returned rows" arm of the
    graph-channel-empty condition genuinely holds. Graph helpers are stubbed so
    the cypher-expansion path completes without Neo4j; the caller overrides
    ``_fetch_chunks_from_entities`` to seed the graph channel.
    """
    vector_store = AsyncMock()
    neo4j_driver = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    vec_result = MagicMock()
    vec_result.chunk = MagicMock()
    vec_result.chunk.id = uuid4()
    vec_result.chunk.namespace_id = ns_id
    vec_result.chunk.document_id = uuid4()
    vec_result.chunk.content = "vector channel chunk"
    vec_result.chunk.occurred_at = None
    vec_result.chunk.created_at = None
    vec_result.chunk.source_timestamp = None
    vec_result.chunk.metadata = {}
    vec_result.chunk.chunker_info = {}
    vec_result.combined_score = 0.85
    vec_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[vec_result])

    bm25_chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="bm25 channel chunk")
    vector_store.search_fulltext = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    storage = AsyncMock()
    storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    storage.get_entities_batch = AsyncMock(return_value={})
    storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    config = RetrieverConfig(enable_bm25_channel=True, enable_session_aware_search=False)
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=neo4j_driver,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="moderate",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._version_filter_entities = AsyncMock(return_value=[])
    return retriever


async def test_graph_channel_empty_counter_fires_for_emptied_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """4a — ``graph_channel_empty`` fires when the graph channel HELD candidates
    that the metadata post-filter then emptied — never vacuously.

    The graph channel is seeded with two chunks whose tags violate the filter, so
    the channel is genuinely NON-EMPTY before the full-AST in-memory post-filter
    runs; the post-filter then drops both, narrowing the graph channel to empty.
    Only THAT condition (``graph_chunks`` truthy → post-filter → empty) fires the
    service-level ``khora.recall.filter.graph_channel_empty`` counter. We spy on
    the lazily-built singleton (pre-seeding the module global makes the helper's
    getter return the fake) and assert it incremented exactly once with no labels.
    The negative control below proves it stays silent when a graph chunk survives.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counter)

    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    # Both graph chunks violate the filter → the channel is non-empty on fetch,
    # then the metadata post-filter empties it. This is the genuine "channel held
    # candidates and the predicate emptied it" trigger, not a vacuous empty.
    violators = [_graph_chunk(ns_id, tag="noise"), _graph_chunk(ns_id, tag="other")]
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(c.id, 0.9, c) for c in violators])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
    result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    assert counter.adds == [(1, {})], (
        f"graph_channel_empty must fire exactly once (no labels) when a non-empty "
        f"graph channel is emptied under the filter; got {counter.adds}"
    )
    # Corroborate the RIGHT reason: the graph channel did hold candidates and was
    # narrowed to empty (the fused provenance carries zero graph chunks, and the
    # failure-observability degradation entry names the emptied channel).
    assert result.metadata["graph_chunk_count"] == 0
    degradations = result.metadata.get("degradations", [])
    assert any(
        d.get("component") == "vectorcypher.graph_channel" and d.get("reason") == "empty_under_filter"
        for d in degradations
    ), f"expected the graph-channel-empty degradation; got {degradations}"


async def test_graph_channel_empty_counter_silent_when_chunk_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """4a control — when a graph chunk survives the post-filter, no counter fires.

    Proves 4a's positive signal comes from the channel genuinely emptying, not
    from the mere presence of a filter: a single graph chunk whose tag satisfies
    the predicate survives, so ``graph_chunks`` stays non-empty after the
    post-filter and the counter must NOT increment.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counter)

    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    survivor = _graph_chunk(ns_id, tag="urgent")  # satisfies the filter
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(survivor.id, 0.9, survivor)])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
    await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    assert counter.adds == [], "graph_channel_empty must stay silent while a graph chunk survives the filter"


def _make_engine_with_stub_retriever(
    ns_id: UUID, *, chunks: list[tuple[Chunk, float]]
) -> tuple[VectorCypherEngine, VectorCypherResult]:
    """A VectorCypher engine whose retriever is stubbed to return ``chunks``.

    The engine's ``recall`` is pure after the retriever returns (validation,
    abstention signals, and document projection read only the in-memory
    ``VectorCypherResult``), so a stubbed ``_get_retriever`` is enough to drive
    the real ``record_under_filled`` call site with no database. The stub
    retriever carries a real ``RetrieverConfig`` because ``recall`` saves/restores
    ``retriever._config.hybrid_alpha`` around the call.
    """
    engine = VectorCypherEngine(KhoraConfig())

    routing = RoutingDecision(
        complexity=QueryComplexity.MODERATE,
        use_graph=True,
        graph_depth=2,
        confidence=0.8,
        reasoning="moderate",
    )
    vc_result = VectorCypherResult(
        chunks=chunks,
        entities=[],
        routing_decision=routing,
        relationships=[],
        metadata={
            "max_raw_vector_score": 0.9,
            "vector_chunk_count": len(chunks),
            "graph_chunk_count": 0,
            "bm25_chunk_count": 0,
        },
    )

    stub_retriever = MagicMock()
    stub_retriever._config = RetrieverConfig()
    stub_retriever.retrieve = AsyncMock(return_value=vc_result)
    engine._get_retriever = MagicMock(return_value=stub_retriever)  # type: ignore[method-assign]
    return engine, vc_result


async def test_under_filled_counter_fires_when_filtered_recall_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """4b — ``under_filled`` fires when a FILTERED recall returns fewer than the limit.

    The engine records ``khora.recall.filter.under_filled`` once per call when a
    caller filter is present AND the result has fewer chunks than the requested
    ``limit``. We stub the retriever to return a single chunk and request a much
    larger limit, with a filter supplied — the real engine call site must fire the
    counter exactly once with no labels. Spied via the lazily-built singleton.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counter)

    ns_id = uuid4()
    chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="alpha bravo charlie content")
    engine, _ = _make_engine_with_stub_retriever(ns_id, chunks=[(chunk, 0.9)])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))
    result = await engine.recall("alpha bravo charlie", ns_id, limit=10, mode=SearchMode.VECTOR, filter_ast=ast)

    assert len(result.chunks) < 10, "precondition: the filtered result is under the requested limit"
    assert counter.adds == [(1, {})], (
        f"under_filled must fire exactly once (no labels) when a filtered recall "
        f"returns fewer than the requested limit; got {counter.adds}"
    )


async def test_under_filled_counter_silent_without_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """4b control — a short result with NO filter leaves ``under_filled`` silent.

    The under-filled counter is owned by the filter subsystem: it fires only when
    a caller filter narrowed the candidate set. A short result with no
    ``filter_ast`` is ordinary low recall, not filter-induced under-fill, so the
    counter must NOT increment — even though the same single-chunk result is below
    the requested limit.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counter)

    ns_id = uuid4()
    chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="alpha bravo charlie content")
    engine, _ = _make_engine_with_stub_retriever(ns_id, chunks=[(chunk, 0.9)])

    result = await engine.recall("alpha bravo charlie", ns_id, limit=10, mode=SearchMode.VECTOR)

    assert len(result.chunks) < 10, "precondition: the unfiltered result is under the requested limit"
    assert counter.adds == [], "under_filled must stay silent when no caller filter is supplied"


# ===========================================================================
# S6 — partial-failure vs transient-Neo4j fallback (HERMETIC, no DB).
# ===========================================================================
#
# The capability boundary the engine must honor:
#   * A filter the Cypher compiler CANNOT honor raises
#     ``RecallFilterUnsupportedError`` and that error PROPAGATES out of
#     ``retrieve()`` — it is NOT masked as a vector-only fallback (a capability
#     gap must surface, not silently under-recall). The over-fetch probe
#     (retriever.py:1083-1093) runs ``compile_cypher`` at method-body
#     indentation, OUTSIDE any transient-error handler, so a raise there escapes.
#   * A TRANSIENT Neo4j error (one of ``_NEO4J_TRANSIENT_ERRORS`` =
#     ``(ServiceUnavailable, ConnectionPoolError, SessionExpired, TransientError)``)
#     from ``_cypher_expand`` IS caught (retriever.py:1344) and degrades to
#     vector-only: ``graph_fallback=True``, ``graph_chunks=[]``, and the vector +
#     BM25 channels still carry recall — so the call returns chunks, never raises.
# Both run everywhere (no ``_LIVE_DB`` mark); they reuse the hermetic
# ``_make_retriever`` from the S4 section.


async def test_partial_failure_compile_cypher_raises_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 3a — a filter whose Cypher compilation RAISES surfaces
    RecallFilterUnsupportedError out of retrieve(), NOT a vector-only fallback.

    The over-fetch probe (retriever.py:1083-1093) runs compile_cypher at
    method-body scope, outside any broad except, so a compile error propagates.
    This is the deliberate contrast with a transient Neo4j error (next test),
    which IS swallowed into a vector-only fallback. We force the raise by
    monkeypatching the compiler the retriever imports inside retrieve().
    """

    def _raise_unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise RecallFilterUnsupportedError("metadata.tag", "forced unsupported for test")

    monkeypatch.setattr("khora.filter.compilers.cypher.compile_cypher", _raise_unsupported)

    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    retriever._vector_only_fallback = AsyncMock()  # spy: a compile-error must NOT degrade to fallback
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))

    with pytest.raises(RecallFilterUnsupportedError):
        await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    retriever._vector_only_fallback.assert_not_awaited()


async def test_transient_neo4j_error_falls_back_to_vector_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 3b — a TRANSIENT Neo4j error degrades to vector-only, never raises.

    _cypher_expand is wrapped by `except _NEO4J_TRANSIENT_ERRORS` (retriever.py:1344);
    a ServiceUnavailable raised there sets graph_fallback and the vector + BM25
    channels still carry recall. Proves the fallback path is intact and is the
    deliberate contrast with the compile-error propagation above.
    """
    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    retriever._cypher_expand = AsyncMock(side_effect=ServiceUnavailable("simulated transient outage"))
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))

    # Must NOT raise — the transient error is caught and the recall degrades.
    result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    assert result.metadata.get("graph_fallback") is True, "transient Neo4j error must set graph_fallback"
    assert result.metadata.get("graph_error") == "ServiceUnavailable", "the degraded-channel error type must surface"
    assert result.chunks, "transient Neo4j error must fall back to vector-only and still return chunks"
