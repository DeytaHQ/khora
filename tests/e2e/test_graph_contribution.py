"""Graph-contribution honesty proof — live PG + Neo4j, self-skips without services.

A row-set filter proof passes VACUOUSLY if the graph channel never contributed
(entity vector search returned nothing and recall short-circuited to the simple
path). This module closes that hole with a positive pre-flight + a negative
tripwire, so the filtered assertions in the graph row-set module are non-vacuous by
construction:

* POSITIVE — after seeding an entity-bearing corpus through the real
  ``Khora.remember()`` ingest path, a no-filter ``mode=GRAPH`` recall MUST return a
  non-empty entity set (``assert_graph_contributes``) AND the graph must be
  populated (``graph_counts`` > 0). If the graph never fired, this fails LOUD.
* NEGATIVE TRIPWIRE — the SAME corpus shape with the extractor staging NOTHING (no
  ``plan_extraction``) must populate ZERO entities, proving the graph contribution
  is CAUSED by the seeded entities, not an always-on fallback.
* FILTER ENFORCEMENT — every doc is entity-bearing (so the graph channel feeds them
  all), but only some satisfy the filter; the filtered ``mode=GRAPH`` recall must
  return ONLY the satisfying docs. A regression where the graph channel ignores
  ``filter_ast`` and smuggles a violating graph-fed chunk into the fused result fails
  the set-equality assertion. (The firing tripwire proves *causation of firing*; this
  proves *filter enforcement* — the two are distinct holes.)

Self-skip: gated on ``NEO4J_INTEGRATION_TEST`` + Postgres reachability (the
``vectorcypher_kb`` fixture's guards), so a no-Docker ``uv run pytest -m e2e``
collects and skips this module cleanly. Run it under ``make dev`` with
``NEO4J_INTEGRATION_TEST=1``.
"""

from __future__ import annotations

import os

import pytest

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

_MARKER = "falconmark"
_ENTITIES = [("Falcon", "PERSON"), ("Orbit", "ORG")]
_RELATIONSHIPS = [("Falcon", "Orbit", "WORKS_ON")]


async def _seed_entity_corpus(kb, namespace_id) -> None:
    """Seed an entity-bearing corpus through the real ingest path (populates the graph)."""
    for content in _harness.entity_seed_docs(_MARKER, count=3):
        await kb.remember(
            content=content,
            namespace=namespace_id,
            entity_types=[t for _, t in _ENTITIES],
            relationship_types=[rt for _, _, rt in _RELATIONSHIPS],
        )


async def test_graph_channel_contributes_preflight(vectorcypher_kb) -> None:
    """A no-filter GRAPH recall returns entities AND the graph is populated (the anchor).

    This is the load-bearing positive gate: it proves the real graph-write path
    (genuine extracted entities + edges) populated Neo4j and that the GRAPH channel
    surfaces them. Every filtered graph assertion depends on this holding.
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    plan_extraction(_MARKER, _ENTITIES, _RELATIONSHIPS)
    await _seed_entity_corpus(kb, namespace_id)

    # CONTRIBUTION: the GRAPH channel returns entities (fails loud if it never fired).
    # Query the exact seeded entity name so the deterministic embedder clears the
    # entity vector floor and the graph entry gate fires (an empty query crashes graph
    # expansion). assert_graph_contributes pins the entity-set signal; graph_chunk_count
    # is the lane-isolated proof that the GRAPH channel specifically held candidates.
    gate = await _harness.assert_graph_contributes(kb, namespace_id, _MARKER)
    assert gate.engine_info.get("graph_chunk_count", 0) > 0, (
        "graph_chunk_count must be > 0 on a GRAPH recall — the graph channel did not contribute"
    )

    # PERSISTENCE: the graph was really written (entities + relationships in Neo4j).
    entity_count, relationship_count = await _harness.graph_counts(kb, namespace_id)
    assert entity_count > 0, "no entities were persisted to the graph"
    assert relationship_count > 0, "no relationships were persisted to the graph"


async def test_graph_contribution_negative_tripwire(vectorcypher_kb) -> None:
    """The SAME corpus with NO staged entities populates zero graph — proving causation.

    Without a ``plan_extraction`` call the stub extractor emits nothing, so the
    ingest path writes no entities. A GRAPH recall must then return an empty entity
    set and the graph counts must be zero. This proves the positive test's graph
    contribution is CAUSED by the seeded entities, not an always-on fallback that
    would make the positive assertion vacuous.
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    # Deliberately NO plan_extraction(_MARKER, ...) — the registry stays empty.
    await _seed_entity_corpus(kb, namespace_id)

    entity_count, relationship_count = await _harness.graph_counts(kb, namespace_id)
    assert entity_count == 0, f"no entities were staged, yet {entity_count} were persisted"
    assert relationship_count == 0, f"no relationships were staged, yet {relationship_count} were persisted"

    # A real (non-empty) query so we don't depend on empty-query graph-expansion
    # behavior; with no entities staged the GRAPH lane has nothing to contribute.
    result = await kb.recall(
        _MARKER,
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=_harness._RECALL_LIMIT,
        min_similarity=0.0,
    )
    # The lane-isolated signal: the GRAPH channel held zero candidates. This is the
    # load-bearing causation proof — the positive test's contribution is caused by the
    # seeded entities, not an always-on fallback.
    assert result.engine_info.get("graph_chunk_count", 0) == 0, (
        "graph_chunk_count must be 0 when no entities were staged (the graph contribution must be caused by extraction)"
    )
    assert not result.entities, "the graph channel returned entities for an un-seeded namespace (vacuity risk)"


async def test_graph_channel_drops_violating_chunks(vectorcypher_kb) -> None:
    """The graph channel DROPS chunks that violate the filter — enforcement, not just firing.

    The firing tripwire proves the graph contribution is *caused* by extraction; this
    proves the graph channel *honors* ``filter_ast``. EVERY seeded doc mentions the
    marker, so the stub extractor emits the shared entity for all of them and the graph
    channel feeds every doc into the fused result. Half are tagged ``keep`` and half
    ``drop``. A no-filter ``GRAPH`` pre-flight confirms the channel held candidates;
    the filtered ``GRAPH`` recall (``metadata.tag`` in ``{"keep"}``) must then return
    ONLY the keep docs. A regression where the graph channel ignores the filter and
    smuggles a violating graph-fed chunk into the result fails the set-equality assert.
    """
    kb = vectorcypher_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    plan_extraction(_MARKER, _ENTITIES, _RELATIONSHIPS)

    keep_ids = {"graph-keep-0", "graph-keep-1"}
    drop_ids = {"graph-drop-0", "graph-drop-1"}
    for tag, ext_ids in (("keep", sorted(keep_ids)), ("drop", sorted(drop_ids))):
        for i, ext_id in enumerate(ext_ids):
            await kb.remember(
                content=f"{_MARKER} {tag} document {i}.",
                namespace=namespace_id,
                external_id=ext_id,
                metadata={"tag": tag},
                entity_types=[t for _, t in _ENTITIES],
                relationship_types=[rt for _, _, rt in _RELATIONSHIPS],
            )

    # PRE-FLIGHT: the graph channel held candidates (so the drop assertion is non-vacuous).
    gate = await _harness.assert_graph_contributes(kb, namespace_id, _ENTITIES[0][0])
    assert gate.engine_info.get("graph_chunk_count", 0) > 0

    # FILTERED: the filter must drop the violating (drop-tagged) graph-fed docs.
    result = await kb.recall(
        _ENTITIES[0][0],
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=_harness._RECALL_LIMIT,
        min_similarity=0.0,
        filter={"metadata.tag": {"$in": ["keep"]}},
    )
    survivors = _harness.reconcile(result)
    assert survivors == frozenset(keep_ids), (
        f"the graph channel must drop filter-violating chunks: expected the keep set, got {set(survivors)}"
    )
