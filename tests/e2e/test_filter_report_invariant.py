"""Cross-engine filter-report invariant gate — rides the e2e matrix.

Every Khora engine that runs a structured recall filter surfaces an honest
``FilterPushdownReport`` as ``RecallResult.engine_info["filter"]``. The shape is
backend-agnostic (built by the single ``build_filter_report`` fold), so a set of
engine-INDEPENDENT invariants must hold no matter which engine / store produced
the report: the pushed/post-filtered key lists partition the filter's constraint
leaves (total + disjoint), ``pushed_down`` derives from the LIST partition, every
channel addresses only real leaves with sorted lists, and a no-filter recall is
the canonical empty carrier. :func:`_harness.assert_filter_report_invariants`
encodes those invariants once; this module drives them against the report a REAL
recall emits on each engine lane.

Why this rides the e2e matrix (not the compile-level conformance matrix):
``engine_info["filter"]`` is produced only on the recall path — the compile-only
filter-conformance suite never runs a recall and emits no report, so it is the
wrong matrix. This module reuses the e2e harness lane selectors and the per-lane
recall-capable ``Khora`` fixtures (``Khora.remember()`` + ``Khora.recall(filter=)``)
so the gate sees the same engine configs the row-set proof does.

One test pair per lane — NOT one parametrized function spanning lanes. pytest's
async fixtures cannot be resolved with ``request.getfixturevalue`` from inside a
running event loop, so each lane takes its recall-capable ``Khora`` fixture as a
DIRECT argument exactly as the sibling row-set modules do. The shared per-recall
body lives in the two ``_assert_*`` coroutines so the per-lane functions stay one
line each. Each lane's node ids lead with the lane token, so the e2e workflow's
per-leg ``-k <token>`` selects exactly that lane (the ``vc_full`` leg, which
selects by file path, appends this module to its target). The reachability /
``lane_skip`` gating mirrors the sibling row-set module for the same lane, so a
no-Docker run collects-and-skips identically.

The representative filter slice per lane is ``_harness.lane_rowset_cases(lane)``
VERBATIM (the same curated, raise-on-empty corpus the row-set proof reconciles,
carrying system + metadata keys including the chronicle partial-pushdown shapes),
plus the no-filter and ``filter={}`` empty-carrier cases (invariant e). All
per-lane reachability / fixture-name wiring lives in ``_harness`` so this module
names no live-backend signal — it is inferred embedded-only by the
verification-coverage gate, which the path-selecting ``vc_full`` leg satisfies
while the conftest tripwire still enforces the live store per leg.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora import Khora
from khora.filter.conformance import ConformanceCase
from khora.query import SearchMode
from tests.e2e import _harness
from tests.test_helpers.filter_spy import plan_extraction

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# The lanes this gate covers — every engine config that emits engine_info["filter"]
# on a real recall, i.e. every _E2E_BACKEND_MAP lane (the report is produced on
# the recall path, never at filter-compile time). Imported by the hermetic drift
# guard (tests/unit/filter/test_filter_report_invariant_coverage.py), which ties
# this set to the conformance backend tokens and the e2e workflow matrix so a new
# lane cannot ship without invariant coverage. GATE_EXCLUSIONS is the documented
# escape hatch for a future compile-only engine that emits no report on recall —
# empty today (every engine emits one), so GATE_LANES covers every lane.
GATE_LANES: frozenset[str] = frozenset(_harness._E2E_BACKEND_MAP)
GATE_EXCLUSIONS: frozenset[str] = frozenset()

# A generous limit + a negative similarity floor so the meaningless hash-cosine
# never narrows the recall — the filter (when present) is the only narrowing
# force, exactly as the row-set proof runs it.
_RECALL_LIMIT = 100
_RECALL_MIN_SIMILARITY = -1.0

# Empty-carrier filter specs (invariant e): a no-filter recall and a filter={}
# recall must both emit the canonical all-False, leafless report.
_EMPTY_CARRIER_SPECS: list[dict | None] = [None, {}]
_EMPTY_CARRIER_IDS = ["no-filter", "empty-filter"]

# --------------------------------------------------------------------------- #
# Shared per-recall bodies — one real recall, then the engine-independent
# invariant assertion. Reused by every lane's test pair so the per-lane functions
# carry only their fixture + the lane's case slice.
# --------------------------------------------------------------------------- #


def _assert_report(result, filter_spec: dict | None) -> None:
    """Assert a recall result's emitted filter report against the invariants.

    Extracts the leaf set the recall-facade way (``filter_spec_leaves`` →
    ``parse_to_ast`` → ``filter_leaf_keys``) and passes the report + leaves to the
    pure :func:`_harness.assert_filter_report_invariants`. Also pins invariant (g)
    here — the private ``_filter_channel_plans`` carrier must NOT leak into public
    ``engine_info`` — because only the caller holds the full ``engine_info`` dict.
    """
    assert "filter" in result.engine_info, "engine emitted no engine_info['filter'] on a recall"
    assert "_filter_channel_plans" not in result.engine_info, (
        "private _filter_channel_plans carrier leaked into public engine_info"
    )
    leaves = _harness.filter_spec_leaves(filter_spec)
    _harness.assert_filter_report_invariants(result.engine_info["filter"], leaves)


async def _assert_rowset(kb: Khora, case: ConformanceCase) -> None:
    """Seed a row-set case, run its filtered recall, assert the emitted report.

    Asserts the report (NOT the survivor set — that is the row-set proof's job):
    the filter is lowered to its constraint-leaf set the same way the recall facade
    lowers it, and ``engine_info["filter"]`` is checked against the invariants.
    """
    namespace_id = (await kb.create_namespace()).namespace_id
    await _harness.seed_records(kb, case.seed_records, namespace_id)
    result = await kb.recall(
        case.seed_records[0].content,
        namespace=namespace_id,
        mode=SearchMode.HYBRID,
        limit=_RECALL_LIMIT,
        min_similarity=_RECALL_MIN_SIMILARITY,
        filter=case.filter,
    )
    _assert_report(result, case.filter)


async def _assert_empty_carrier(kb: Khora, spec: dict | None) -> None:
    """Run a no-filter / filter={} recall, assert the canonical empty-carrier report (invariant d/e)."""
    namespace_id = (await kb.create_namespace()).namespace_id
    result = await kb.recall(
        "filter report invariant probe",
        namespace=namespace_id,
        mode=SearchMode.HYBRID,
        limit=_RECALL_LIMIT,
        min_similarity=_RECALL_MIN_SIMILARITY,
        filter=spec,
    )
    _assert_report(result, spec)


# --------------------------------------------------------------------------- #
# vc_full — live VectorCypher (PG + Neo4j). Path-selected by the e2e workflow.
# --------------------------------------------------------------------------- #


@_harness.lane_skip("vc_full")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_full"),
    reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live graph lane",
)
@pytest.mark.parametrize("case", _harness.lane_rowset_cases("vc_full"), ids=lambda c: f"vc_full-{c.id}")
async def test_report_invariants_rowset_vc_full(vectorcypher_kb, case) -> None:
    """The live VectorCypher recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(vectorcypher_kb, case)


@_harness.lane_skip("vc_full")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_full"),
    reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live graph lane",
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"vc_full-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_vc_full(vectorcypher_kb, spec) -> None:
    """A no-filter / filter={} recall on live VectorCypher is the canonical empty carrier."""
    await _assert_empty_carrier(vectorcypher_kb, spec)


# --------------------------------------------------------------------------- #
# vc_embedded — VectorCypher on the embedded sqlite_lance stack (container-free).
# --------------------------------------------------------------------------- #


@_harness.lane_skip("vc_embedded")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_embedded"),
    reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
)
@pytest.mark.parametrize("case", _harness.lane_rowset_cases("vc_embedded"), ids=lambda c: f"vc_embedded-{c.id}")
async def test_report_invariants_rowset_vc_embedded(sqlite_lance_kb, case) -> None:
    """The embedded VectorCypher recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(sqlite_lance_kb, case)


@_harness.lane_skip("vc_embedded")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_embedded"),
    reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"vc_embedded-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_vc_embedded(sqlite_lance_kb, spec) -> None:
    """A no-filter / filter={} recall on embedded VectorCypher is the canonical empty carrier."""
    await _assert_empty_carrier(sqlite_lance_kb, spec)


# --------------------------------------------------------------------------- #
# skeleton_pgvector — live Skeleton on pgvector.
# --------------------------------------------------------------------------- #


@_harness.lane_skip("skeleton_pgvector")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_pgvector"), reason="start Postgres (make dev) to exercise this live lane"
)
@pytest.mark.parametrize(
    "case", _harness.lane_rowset_cases("skeleton_pgvector"), ids=lambda c: f"skeleton_pgvector-{c.id}"
)
async def test_report_invariants_rowset_skeleton_pgvector(skeleton_pgvector_kb, case) -> None:
    """The live Skeleton-pgvector recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(skeleton_pgvector_kb, case)


@_harness.lane_skip("skeleton_pgvector")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_pgvector"), reason="start Postgres (make dev) to exercise this live lane"
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"skeleton_pgvector-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_skeleton_pgvector(skeleton_pgvector_kb, spec) -> None:
    """A no-filter / filter={} recall on live Skeleton-pgvector is the canonical empty carrier."""
    await _assert_empty_carrier(skeleton_pgvector_kb, spec)


# --------------------------------------------------------------------------- #
# skeleton_surrealdb — Skeleton on in-process SurrealDB (memory://, container-free).
# --------------------------------------------------------------------------- #


@_harness.lane_skip("skeleton_surrealdb")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_surrealdb"),
    reason="embedded SurrealDB SDK not installed (pip install khora[surrealdb])",
)
@pytest.mark.parametrize(
    "case", _harness.lane_rowset_cases("skeleton_surrealdb"), ids=lambda c: f"skeleton_surrealdb-{c.id}"
)
async def test_report_invariants_rowset_skeleton_surrealdb(skeleton_surrealdb_kb, case) -> None:
    """The Skeleton-SurrealDB recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(skeleton_surrealdb_kb, case)


@_harness.lane_skip("skeleton_surrealdb")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_surrealdb"),
    reason="embedded SurrealDB SDK not installed (pip install khora[surrealdb])",
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"skeleton_surrealdb-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_skeleton_surrealdb(skeleton_surrealdb_kb, spec) -> None:
    """A no-filter / filter={} recall on Skeleton-SurrealDB is the canonical empty carrier."""
    await _assert_empty_carrier(skeleton_surrealdb_kb, spec)


# --------------------------------------------------------------------------- #
# skeleton_weaviate — live Skeleton-Weaviate (vectors in Weaviate, docs in PG).
# --------------------------------------------------------------------------- #


@_harness.lane_skip("skeleton_weaviate")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_weaviate"),
    reason="start Postgres + Weaviate (make dev) to exercise the live Skeleton-Weaviate lane",
)
@pytest.mark.parametrize(
    "case", _harness.lane_rowset_cases("skeleton_weaviate"), ids=lambda c: f"skeleton_weaviate-{c.id}"
)
async def test_report_invariants_rowset_skeleton_weaviate(skeleton_weaviate_kb, case) -> None:
    """The live Skeleton-Weaviate recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(skeleton_weaviate_kb, case)


@_harness.lane_skip("skeleton_weaviate")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_weaviate"),
    reason="start Postgres + Weaviate (make dev) to exercise the live Skeleton-Weaviate lane",
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"skeleton_weaviate-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_skeleton_weaviate(skeleton_weaviate_kb, spec) -> None:
    """A no-filter / filter={} recall on live Skeleton-Weaviate is the canonical empty carrier."""
    await _assert_empty_carrier(skeleton_weaviate_kb, spec)


# --------------------------------------------------------------------------- #
# skeleton_sqlite_lance — Skeleton on the embedded sqlite_lance stack (container-free).
# --------------------------------------------------------------------------- #


@_harness.lane_skip("skeleton_sqlite_lance")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_sqlite_lance"),
    reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
)
@pytest.mark.parametrize(
    "case", _harness.lane_rowset_cases("skeleton_sqlite_lance"), ids=lambda c: f"skeleton_sqlite_lance-{c.id}"
)
async def test_report_invariants_rowset_skeleton_sqlite_lance(skeleton_sqlite_lance_kb, case) -> None:
    """The embedded Skeleton-sqlite_lance recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(skeleton_sqlite_lance_kb, case)


@_harness.lane_skip("skeleton_sqlite_lance")
@pytest.mark.skipif(
    not _harness.lane_reachable("skeleton_sqlite_lance"),
    reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"skeleton_sqlite_lance-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_skeleton_sqlite_lance(skeleton_sqlite_lance_kb, spec) -> None:
    """A no-filter / filter={} recall on embedded Skeleton-sqlite_lance is the canonical empty carrier."""
    await _assert_empty_carrier(skeleton_sqlite_lance_kb, spec)


# --------------------------------------------------------------------------- #
# chronicle — live Chronicle engine (PG-only, partial-pushdown shapes).
# --------------------------------------------------------------------------- #


@_harness.lane_skip("chronicle")
@pytest.mark.skipif(
    not _harness.lane_reachable("chronicle"), reason="start Postgres (make dev) to exercise this live lane"
)
@pytest.mark.parametrize("case", _harness.lane_rowset_cases("chronicle"), ids=lambda c: f"chronicle-{c.id}")
async def test_report_invariants_rowset_chronicle(chronicle_kb, case) -> None:
    """The live Chronicle recall emits an invariant-clean filter report per row-set case."""
    await _assert_rowset(chronicle_kb, case)


@_harness.lane_skip("chronicle")
@pytest.mark.skipif(
    not _harness.lane_reachable("chronicle"), reason="start Postgres (make dev) to exercise this live lane"
)
@pytest.mark.parametrize("spec", _EMPTY_CARRIER_SPECS, ids=[f"chronicle-{i}" for i in _EMPTY_CARRIER_IDS])
async def test_report_invariants_empty_chronicle(chronicle_kb, spec) -> None:
    """A no-filter / filter={} recall on live Chronicle is the canonical empty carrier."""
    await _assert_empty_carrier(chronicle_kb, spec)


# --------------------------------------------------------------------------- #
# Entity-bearing graph/chronicle leak cases (#1457 / #1458).
#
# The row-set cases above seed generic ``"conformance anchor"`` content with the
# extractor staging NOTHING (no ``plan_extraction``), so those recalls return an
# EMPTY entity surface and the surface-coverage rule stays inert — they remain
# invariant-clean. These two cases are the opposite: they stage a real entity
# corpus so the graph (VectorCypher) / entity (Chronicle) channel surfaces
# entities the date filter never constrained. Before the #1457 / #1458 fix the
# non-empty uncovered entity surface forced every filter leaf into
# ``unenforced_keys``; the fix re-applies the "Chunk" predicate to each entity's
# provenance (∃-over-provenance) and marks the entity surface covered, so the
# report is CLEAN again.
#
# Each body asserts the CLEAN post-fix invariant (``unenforced_keys == []``)
# directly, with an anti-vacuity gate that fails loud unless the recall actually
# surfaced entities first (so a no-entity path cannot pass the assertion for the
# wrong reason). Both self-skip on a no-Docker run via the lane reachability
# marks, exactly like the row-set cases.
# --------------------------------------------------------------------------- #

_LEAK_MARKER = "leakmark"
_LEAK_ENTITIES = [("Falcon", "PERSON"), ("Orbit", "ORG")]
_LEAK_RELATIONSHIPS = [("Falcon", "Orbit", "WORKS_ON")]
# The seed's event time, seeded via ``source_timestamp`` so every chunk carries a
# real event-time anchor after the filter horizon below. VectorCypher chunks pick
# up an event time from ingest either way; Chronicle chunks only carry one when
# ``source_timestamp`` is supplied (its recall resolves ``occurred_at`` as
# ``COALESCE(occurred_at, source_timestamp)``), so an anchor-less seed would make
# the ``occurred_at`` filter empty every chunk AND drop every provenance entity.
_LEAK_SOURCE_TIMESTAMP = datetime(2025, 6, 1, tzinfo=UTC)
# A date filter whose horizon predates the seed's event time: a date-key predicate
# (drives the VectorCypher EXPLICIT graph path) that the entity surface never
# gates. The seed's ``source_timestamp`` clears it, so the clean recall keeps the
# chunks; the leaf is the one forced unenforced by an uncovered entity surface
# before the #1457 / #1458 provenance fix.
_LEAK_DATE_FILTER: dict = {"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}}


async def _seed_leak_corpus(kb: Khora, namespace_id) -> None:
    """Seed an entity-bearing corpus through the real ingest path (populates entities)."""
    plan_extraction(_LEAK_MARKER, _LEAK_ENTITIES, _LEAK_RELATIONSHIPS)
    for content in _harness.entity_seed_docs(_LEAK_MARKER, count=3):
        await kb.remember(
            content=content,
            namespace=namespace_id,
            source_timestamp=_LEAK_SOURCE_TIMESTAMP,
            entity_types=[t for _, t in _LEAK_ENTITIES],
            relationship_types=[rt for _, _, rt in _LEAK_RELATIONSHIPS],
        )


@_harness.lane_skip("vc_full")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_full"),
    reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live graph lane",
)
async def test_report_invariant_graph_entity_bearing_date_filter_vc_full(vectorcypher_kb) -> None:
    """A graph-path VectorCypher recall over an entity corpus + date filter reports clean.

    The seeded entities make the graph channel surface a non-empty entity set the
    chunk-side date filter does not cover, so the emitted report currently forces
    the date leaf into ``unenforced_keys``. This asserts the CLEAN invariant the
    #1457 fix restores (``unenforced_keys == []``), gated on a real entity surface
    so it is not vacuous.
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    await _seed_leak_corpus(kb, namespace_id)

    # Anti-vacuity gate: the GRAPH channel really surfaced entities (fails loud if
    # the graph never fired — then the leak below would be vacuous).
    gate = await _harness.assert_graph_contributes(kb, namespace_id, _LEAK_MARKER)
    if gate.engine_info.get("graph_chunk_count", 0) <= 0:
        pytest.fail("graph channel spliced no chunks — the graph-path leak is not exercised (vacuous)")

    result = await kb.recall(
        _LEAK_MARKER,
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=_RECALL_LIMIT,
        min_similarity=_RECALL_MIN_SIMILARITY,
        filter=_LEAK_DATE_FILTER,
    )
    if not result.entities:
        pytest.fail("graph recall surfaced no entities — the surface-coverage rule is inert (vacuous)")
    _assert_report(result, _LEAK_DATE_FILTER)


# --------------------------------------------------------------------------- #
# #1494: a compound ``source_name`` + date filter over a TWO-SOURCE entity corpus
# narrows the graph-path entity surface to exactly the matching source's entities,
# AND reports clean (``unenforced_keys == []``).
#
# This is the entity-surface counterpart of the doc-key leak: the seed carries two
# ``source_name``-tagged corpora, each contributing its own entity set the date
# filter never gates. Pre-#1494 the ∃-over-provenance pass evaluated the
# ``source_name`` leaf against a bare provenance chunk that lacked the denormalized
# document keys (``source_name`` isn't even on ``DocumentSource``), so it dropped
# EVERY entity (both sources) and forced the leaves unenforced. Post-fix the
# provenance pass hydrates each entity's parent-document projection, so the entity
# surface narrows to the ``linear`` source's entities while the ``slack`` ones drop,
# and the report is clean. The anti-vacuity gate fails loud unless BOTH sources'
# entities actually surface first.
# --------------------------------------------------------------------------- #

# Two source-tagged corpora, each with its own marker + entity set. The markers are
# distinct so ``plan_extraction`` stages each source's entities independently; both
# share the ``_SRC_RECALL_MARKER`` substring so a single GRAPH recall surfaces BOTH
# sources' entities (making the ``source_name`` narrowing non-vacuous).
_SRC_RECALL_MARKER = "srcnarrow"
_SRC_MARKER_LINEAR = "srcnarrow-linear"
_SRC_MARKER_SLACK = "srcnarrow-slack"
_SRC_ENTITIES_LINEAR = [("Meridian", "PERSON"), ("Beacon", "ORG")]
_SRC_RELATIONSHIPS_LINEAR = [("Meridian", "Beacon", "WORKS_ON")]
_SRC_ENTITIES_SLACK = [("Cascade", "PERSON"), ("Summit", "ORG")]
_SRC_RELATIONSHIPS_SLACK = [("Cascade", "Summit", "WORKS_ON")]
_SRC_SOURCE_NAME_LINEAR = "linear"
_SRC_SOURCE_NAME_SLACK = "slack"
# ``source_name`` for the matching source + a date bound the seed's event time clears
# (so the date leaf never narrows and the ``source_name`` leaf is the only force).
_SRC_COMPOUND_FILTER: dict = {
    "source_name": _SRC_SOURCE_NAME_LINEAR,
    "occurred_at": {"$gte": "2020-01-01T00:00:00Z"},
}


async def _seed_two_source_corpus(kb: Khora, namespace_id) -> None:
    """Seed two ``source_name``-tagged entity corpora through the real ingest path.

    The ``linear`` source contributes ``_SRC_ENTITIES_LINEAR`` and the ``slack``
    source ``_SRC_ENTITIES_SLACK``; each document carries the shared
    ``_SRC_RECALL_MARKER`` so one GRAPH recall surfaces both sources' entities.
    """
    plan_extraction(_SRC_MARKER_LINEAR, _SRC_ENTITIES_LINEAR, _SRC_RELATIONSHIPS_LINEAR)
    plan_extraction(_SRC_MARKER_SLACK, _SRC_ENTITIES_SLACK, _SRC_RELATIONSHIPS_SLACK)
    for marker, source_name, entities, relationships in (
        (_SRC_MARKER_LINEAR, _SRC_SOURCE_NAME_LINEAR, _SRC_ENTITIES_LINEAR, _SRC_RELATIONSHIPS_LINEAR),
        (_SRC_MARKER_SLACK, _SRC_SOURCE_NAME_SLACK, _SRC_ENTITIES_SLACK, _SRC_RELATIONSHIPS_SLACK),
    ):
        for content in _harness.entity_seed_docs(marker, count=3):
            await kb.remember(
                content=content,
                namespace=namespace_id,
                source_name=source_name,
                source_timestamp=_LEAK_SOURCE_TIMESTAMP,
                entity_types=[t for _, t in entities],
                relationship_types=[rt for _, _, rt in relationships],
            )


@_harness.lane_skip("vc_full")
@pytest.mark.skipif(
    not _harness.lane_reachable("vc_full"),
    reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live graph lane",
)
async def test_report_invariant_graph_source_name_narrows_entities_vc_full(vectorcypher_kb) -> None:
    """A graph recall + compound ``source_name`` filter narrows entities and reports clean.

    Seeds two ``source_name``-tagged corpora (``linear`` / ``slack``), each with its
    own entity set. A compound ``{"source_name": "linear", "occurred_at": {"$gte":
    2020}}`` filter must narrow the graph-path entity surface to exactly the
    ``linear`` source's entities (the ``slack`` ones drop) AND report clean
    (``unenforced_keys == []``). The #1494 fix's projection hydration is what lets the
    ∃-over-provenance pass resolve ``source_name`` (a key absent from ``DocumentSource``)
    on each entity's provenance. Gated on BOTH sources surfacing first so the narrowing
    is not vacuous.
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    await _seed_two_source_corpus(kb, namespace_id)

    # Anti-vacuity gate: an UNFILTERED graph recall surfaces BOTH sources' entities,
    # so the narrowing below is a real prune (not an already-empty-or-single set).
    gate = await _harness.assert_graph_contributes(kb, namespace_id, _SRC_RECALL_MARKER)
    if gate.engine_info.get("graph_chunk_count", 0) <= 0:
        pytest.fail("graph channel spliced no chunks — the source-narrowing leak is not exercised (vacuous)")
    # Ingest canonicalizes entity names to lowercase, so compare case-insensitively.
    gate_names = {e.name.lower() for e in gate.entities}
    linear_names = {n.lower() for n, _ in _SRC_ENTITIES_LINEAR}
    slack_names = {n.lower() for n, _ in _SRC_ENTITIES_SLACK}
    if not (linear_names <= gate_names and slack_names <= gate_names):
        pytest.fail(
            f"both sources' entities must surface UNFILTERED for the narrowing to be meaningful; "
            f"got {sorted(gate_names)} (need {sorted(linear_names | slack_names)})"
        )

    result = await kb.recall(
        _SRC_RECALL_MARKER,
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=_RECALL_LIMIT,
        min_similarity=_RECALL_MIN_SIMILARITY,
        filter=_SRC_COMPOUND_FILTER,
    )

    filtered_names = {e.name.lower() for e in result.entities}
    if not filtered_names:
        pytest.fail("filtered graph recall surfaced no entities — the surface-coverage rule is inert (vacuous)")
    # Narrowed to exactly the linear source's entities; the slack ones dropped.
    assert linear_names <= filtered_names, (
        f"the matching-source (linear) entities were over-dropped — #1494 not fixed: got {sorted(filtered_names)}"
    )
    assert not (slack_names & filtered_names), (
        f"a non-matching-source (slack) entity leaked past the source_name filter: got {sorted(filtered_names)}"
    )
    # The entity surface is covered by the ∃ pass, so the report is clean.
    _assert_report(result, _SRC_COMPOUND_FILTER)


@_harness.lane_skip("chronicle")
@pytest.mark.skipif(
    not _harness.lane_reachable("chronicle"), reason="start Postgres (make dev) to exercise this live lane"
)
async def test_report_invariant_entity_bearing_date_filter_chronicle(chronicle_kb) -> None:
    """A HYBRID Chronicle recall over an entity corpus + date filter reports clean.

    Chronicle's entity channel surfaces a non-empty entity set the chunk-side date
    filter does not cover, so before the #1458 fix the emitted report forced the
    date leaf into ``unenforced_keys``. The fix re-applies the "Chunk" predicate to
    each entity's provenance (∃-over-provenance) and marks the entity surface
    covered, so the report is CLEAN (``unenforced_keys == []``). Chronicle has no
    GRAPH mode — the entity channel runs in HYBRID.

    Two conditions must both hold for the entity surface to be non-empty here (so
    the coverage rule is exercised, not vacuously inert):

    1. The query-complexity router is disabled. The single-token ``_LEAK_MARKER``
       query routes SIMPLE, which sets ``run_entity=False`` and skips the entity
       channel entirely in HYBRID; turning the router off is the documented
       "run all channels" knob (mirrors the hermetic composition test).
    2. The seed carries an event time after the filter horizon. Chronicle chunks
       land with ``occurred_at=NULL`` and take their event time from
       ``source_timestamp`` (its recall resolves ``occurred_at`` as
       ``COALESCE(occurred_at, source_timestamp)``). ``_seed_leak_corpus`` stamps
       ``source_timestamp=_LEAK_SOURCE_TIMESTAMP`` (2025, past the 2020 horizon), so
       the ∃-over-provenance pass keeps the entities instead of dropping them for a
       missing event time. The pass evaluates each provenance chunk through the same
       ``_chunk_to_record`` COALESCE view the chunk channel uses (#1458), so the
       entity surface enforces ``occurred_at`` identically to the chunk surface.
    """
    kb = chronicle_kb
    # Keep every channel live: the single-token marker query would otherwise route
    # SIMPLE and skip the entity channel (run_entity=False), leaving result.entities
    # empty and the surface-coverage rule inert. Turning the router off is the
    # documented "run all channels" knob (mirrors the hermetic composition test).
    kb._engine._router_enabled = False
    namespace_id = (await kb.create_namespace()).namespace_id
    await _seed_leak_corpus(kb, namespace_id)

    result = await kb.recall(
        _LEAK_MARKER,
        namespace=namespace_id,
        mode=SearchMode.HYBRID,
        limit=_RECALL_LIMIT,
        min_similarity=_RECALL_MIN_SIMILARITY,
        filter=_LEAK_DATE_FILTER,
    )
    # Anti-vacuity gate: the entity channel really surfaced entities.
    if not result.entities:
        pytest.fail("chronicle recall surfaced no entities — the surface-coverage rule is inert (vacuous)")
    _assert_report(result, _LEAK_DATE_FILTER)
