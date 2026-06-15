"""Row-set recall-filter proof on the live VectorCypher (PG + Neo4j) stack.

The graph-lane companion to ``test_filter_rowset_embedded.py``: it drives the same
``Khora.remember()`` -> ``Khora.recall(filter=...)`` row-set proof on the full
VectorCypher engine, where the graph channel is live. Two things this lane adds
over the embedded one:

* AC2 — the graph channel FIRES and Neo4j is actually POPULATED: an entity-bearing
  ingest writes genuine entities + ``MENTIONED_IN`` edges, asserted via both the
  ``graph_chunk_count`` contribution signal and the ``graph_counts`` persistence
  probe.
* the row-set reconciliation runs through the PG/pgvector read path (1536-dim).

Self-skip: gated on ``NEO4J_INTEGRATION_TEST`` + Postgres reachability via the
``vectorcypher_kb`` fixture's guards, so a no-Docker run collects and skips this
module cleanly. Run under ``make dev`` + ``NEO4J_INTEGRATION_TEST=1``.
"""

from __future__ import annotations

import os

import pytest

from khora.filter import conformance
from khora.query import SearchMode
from tests.e2e import _harness
from tests.test_helpers.filter_spy import plan_extraction

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    _harness.lane_skip("vc_full"),
    pytest.mark.skipif(
        not os.environ.get("NEO4J_INTEGRATION_TEST") or not _harness._pg_reachable(),
        reason="set NEO4J_INTEGRATION_TEST=1 and start PG+Neo4j (make dev) to exercise the live graph lane",
    ),
]


# --------------------------------------------------------------------------- #
# AC2 — the graph channel fires AND Neo4j is populated after a real ingest.
# --------------------------------------------------------------------------- #


_MARKER = "falconmark"
_ENTITIES = [("Falcon", "PERSON"), ("Orbit", "ORG")]
_RELATIONSHIPS = [("Falcon", "Orbit", "WORKS_ON")]


async def test_graph_channel_fires_and_neo4j_populated(vectorcypher_kb) -> None:
    """An entity-bearing ingest populates Neo4j and the graph channel feeds recall.

    Asserts BOTH design signals: PERSISTENCE (``graph_counts`` shows the entities +
    relationships were written to the graph) and CONTRIBUTION (a no-filter
    ``mode=GRAPH`` recall returns entities and ``graph_chunk_count > 0`` — the graph
    channel actually fed the recall, not just exists).
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    plan_extraction(_MARKER, _ENTITIES, _RELATIONSHIPS)
    for content in _harness.entity_seed_docs(_MARKER, count=3):
        await kb.remember(
            content=content,
            namespace=namespace_id,
            entity_types=[t for _, t in _ENTITIES],
            relationship_types=[rt for _, _, rt in _RELATIONSHIPS],
        )

    # PERSISTENCE — the graph was written with EXACTLY the planned entities. The
    # namespace is fresh per test, so these counts are a delta from zero (no nodes
    # inherited from another test/namespace), and we pin the exact set + types rather
    # than a bare ``> 0`` so a stray inherited or mis-typed node cannot satisfy it.
    entity_count, relationship_count = await _harness.graph_counts(kb, namespace_id)
    assert entity_count == len(_ENTITIES), f"expected {len(_ENTITIES)} entities, got {entity_count}"
    assert relationship_count >= len(_RELATIONSHIPS), (
        f"expected >= {len(_RELATIONSHIPS)} relationships, got {relationship_count}"
    )
    # Entity names come back normalized (lowercased) by the production
    # ``normalize_entity_name`` step — comparing against the normalized planned set
    # proves the real graph-write path ran, not a persistence bypass.
    resolved = await kb.storage.resolve_namespace(namespace_id)
    persisted = {(e.name, e.entity_type) for e in await kb.storage.list_entities(resolved)}
    expected = {(name.lower(), etype) for name, etype in _ENTITIES}
    assert persisted == expected, f"persisted entities {persisted} != planned {expected}"

    # CONTRIBUTION — the graph channel fed the recall (non-vacuous). Query the exact
    # seeded entity name so the deterministic embedder clears the entity vector floor
    # and the graph entry gate reliably fires (an empty query crashes graph expansion).
    gate = await _harness.assert_graph_contributes(kb, namespace_id, _ENTITIES[0][0])
    assert gate.engine_info.get("graph_chunk_count", 0) > 0
    assert "graph" in gate.engine_info.get("channels_used", [])


# --------------------------------------------------------------------------- #
# Row-set reconciliation through the live PG/pgvector read path.
# --------------------------------------------------------------------------- #


def _graph_rowset_cases() -> list[conformance.ConformanceCase]:
    """The row-set cases for the live VectorCypher (``cypher``) lane.

    The live PG/pgvector chunk row carries the denormalized document system keys, so
    this lane narrows on them as well as the dotted-``metadata`` families
    (``include_system_keys=True``): the ``remember``-threadable system-key ``F-OP``
    families and the two ``source_name`` ``F-EXISTS`` presence states the embedded
    lane defers here. See ``_harness.rowset_cases``.
    """
    return _harness.rowset_cases("cypher", include_system_keys=True)


@pytest.mark.parametrize("case", _graph_rowset_cases(), ids=lambda c: c.id)
async def test_rowset_reconciliation_graph(vectorcypher_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the live graph stack."""
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


def test_lane_selection_matches_shipped_vc_full() -> None:
    """The resolver lane selection equals this module's shipped precedent (Layer-3).

    The Layer-1 empty-raise in ``lane_rowset_cases`` catches a token that selects
    NOTHING, but not a valid-but-wrong token that happens to select a different
    non-empty set. This pins the resolver path (``lane_rowset_cases("vc_full")``)
    to the shipped selection this module has always parametrized over
    (``rowset_cases("cypher", include_system_keys=True)``), so a future
    ``_E2E_BACKEND_MAP`` drift that re-points ``vc_full`` at a different token (the
    ``cypher``->``postgres`` slip that would silently drop 81->46 cases) or flips its
    system-key flag fails LOUD here instead of silently under-covering.
    """
    resolver_ids = {c.id for c in _harness.lane_rowset_cases("vc_full")}
    shipped_ids = {c.id for c in _graph_rowset_cases()}
    assert resolver_ids == shipped_ids

    # The F-LOGIC lane selection is pinned the same way (Layer-3): the resolver path
    # (``lane_logic_cases("vc_full")``) must equal the threadable subset this module
    # parametrizes ``test_logic_reconciliation_graph`` over, so a token / corpus drift
    # that re-points ``vc_full`` at a different non-empty F-LOGIC selection fails LOUD.
    logic_resolver_ids = {c.id for c in _harness.lane_logic_cases("vc_full")}
    logic_shipped_ids = {c.id for c in _graph_logic_cases()}
    assert logic_resolver_ids == logic_shipped_ids


# --------------------------------------------------------------------------- #
# F-LOGIC boolean-composition reconciliation (system + metadata compositions).
# --------------------------------------------------------------------------- #


def _graph_logic_cases() -> list[conformance.ConformanceCase]:
    """The threadable F-LOGIC boolean-composition cases for the live (``cypher``) lane.

    The live PG/pgvector chunk row carries the denormalized document system keys, so
    the engine recall path narrows on the system-key compositions (``$and`` / ``$or``
    / ``$not`` / De Morgan / distributivity over ``source_name`` / ``source_type`` /
    ``source_timestamp``) as well as the ``metadata`` ones. See
    ``_harness.engine_logic_cases``. The empty-raise is the same Layer-1 anti-vacuity
    guard the row-set helpers carry — a corpus shrink that drops every threadable
    F-LOGIC case for this lane fails RED rather than parametrizing a vacuous lane.
    """
    cases = [c for c in _harness.engine_logic_cases(conformance.f_logic_cases()) if "cypher" in c.backends]
    if not cases:
        raise RuntimeError(
            "engine_logic_cases selected zero F-LOGIC cases for the cypher lane — refusing a vacuously-green lane."
        )
    return cases


@pytest.mark.parametrize("case", _graph_logic_cases(), ids=lambda c: c.id)
async def test_logic_reconciliation_graph(vectorcypher_kb, case) -> None:
    """A boolean-composition filter's survivors reconcile to the case's expected ids on the live graph stack."""
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)
