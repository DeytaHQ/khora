"""``_cypher_expand`` neighborhood-shape handling on the storage path (#1086).

On the embedded (graph-less-Neo4j) path, ``_cypher_expand`` reads
neighborhoods from the storage coordinator's ``get_neighborhoods_batch``,
which returns ``{seed: {"entities": [...], "relationships": [...]}}``. The
embedded sqlite_lance backend puts ``Entity`` domain objects in that
``entities`` list, but the normalization loop's scoring step reads dict
fields — so the loop now maps ``Entity`` -> dict before the
``isinstance(..., dict)`` check.

Any neighborhood entry that is neither an ``Entity`` nor a dict is a
genuinely unrecognized shape: it cannot be scored and is dropped. When a
``degradations`` list is supplied, the loop appends a structured
``Degradation`` so the dropped expansion hop is observable rather than a
silent empty graph channel (ADR-001 failure-observability convention).

These are pure-unit tests against ``_cypher_expand`` with a mocked
storage coordinator — no embedded stack, no LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.diagnostics import Degradation
from khora.core.models.entity import Entity
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever

pytestmark = pytest.mark.unit


def _retriever_with_graph_storage(neighborhoods: dict) -> VectorCypherRetriever:
    """A retriever whose storage path returns ``neighborhoods``.

    ``neo4j_driver=None`` forces ``_dual_nodes`` to ``None`` so
    ``_cypher_expand`` takes the ``self._storage and self._storage._graph``
    branch. ``storage._graph`` is a truthy marker and
    ``get_neighborhoods_batch`` is stubbed to return the supplied shape.
    """
    storage = MagicMock()
    storage._graph = MagicMock()  # truthy — selects the storage-coordinator branch
    storage.get_neighborhoods_batch = AsyncMock(return_value=neighborhoods)

    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )


async def test_cypher_expand_records_degradation_for_unrecognized_shape() -> None:
    """A neighborhood entry that is neither Entity nor dict appends a Degradation."""
    seed = uuid4()
    # 42 is neither an ``Entity`` nor a dict — the normalization loop cannot
    # score it and drops it.
    retriever = _retriever_with_graph_storage({seed: {"entities": [42], "relationships": []}})

    degradations: list[Degradation] = []
    scores, info = await retriever._cypher_expand(
        [seed],
        uuid4(),
        depth=1,
        degradations=degradations,
    )

    # The unrecognized entry produced no scored entity ...
    assert scores == {}
    assert info == {}
    # ... but the drop is observable, not silent.
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.cypher_expand"
    assert deg["reason"] == "unrecognized_neighborhood_shape"
    # The detail names the dropped type so an operator can diagnose it.
    assert "int" in deg.get("detail", "")


async def test_cypher_expand_no_degradation_when_list_not_passed() -> None:
    """Without a ``degradations`` sink the loop still drops cleanly (no crash)."""
    seed = uuid4()
    retriever = _retriever_with_graph_storage({seed: {"entities": ["not-an-entity"], "relationships": []}})

    # ``degradations`` defaults to ``None`` — the elif guard must short-circuit.
    scores, info = await retriever._cypher_expand([seed], uuid4(), depth=1)

    assert scores == {}
    assert info == {}


async def test_cypher_expand_maps_entity_objects_to_scores() -> None:
    """An ``Entity`` domain object (the embedded backend's shape) is scored,
    and a co-located unrecognized entry is the only thing degraded."""
    seed = uuid4()
    carol_id = uuid4()
    carol = Entity(id=carol_id, namespace_id=uuid4(), name="carol", entity_type="PERSON")
    retriever = _retriever_with_graph_storage({seed: {"entities": [carol, object()], "relationships": []}})

    degradations: list[Degradation] = []
    scores, info = await retriever._cypher_expand(
        [seed],
        uuid4(),
        depth=1,
        degradations=degradations,
    )

    # The Entity is scored (proves Entity->dict mapping landed before the
    # dict check) ...
    assert carol_id in scores, f"Entity object was dropped instead of scored: {scores!r}"
    assert info[str(carol_id)]["name"] == "carol"
    # ... and only the genuinely unrecognized ``object()`` was degraded.
    assert len(degradations) == 1
    assert degradations[0]["reason"] == "unrecognized_neighborhood_shape"
