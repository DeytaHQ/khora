"""Leg 2 of #1474: typed/confidence-weighted graph expansion.

Two surfaces:

* ``_build_neighborhood_query`` gains an ``include_edge_metadata`` flag. OFF
  (default) it is byte-identical to the pre-#1474 query - so the #1419
  min-distance contract and the existing dual_nodes query-shape tests are
  untouched. ON it additionally returns (rel_type, rel_confidence,
  rel_direction) per reported hop, with ``distance`` kept last in the map so the
  BFS structure and distances are unchanged.
* ``_cypher_expand`` scores each hop as ``1/(1+distance)`` when the flag is OFF
  (byte-identical legacy) and ``1/(1+distance) * confidence * type_prior`` when
  ON.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.dual_nodes import _build_neighborhood_query
from khora.engines.vectorcypher.retriever import (
    _DEFAULT_RELATIONSHIP_TYPE_PRIOR,
    _RELATIONSHIP_TYPE_PRIORS,
    RetrieverConfig,
    VectorCypherRetriever,
    _relationship_type_prior,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Query builder: flag OFF is byte-identical; flag ON adds edge metadata.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("prefer_current", [False, True])
def test_flag_off_is_byte_identical_to_legacy(depth: int, prefer_current: bool) -> None:
    """Default (include_edge_metadata=False) reproduces the pre-#1474 query.

    The two-arg call is exactly what the existing dual_nodes tests use, so this
    pins that the new default arg does not perturb the legacy query text.
    """
    two_arg = _build_neighborhood_query(depth, prefer_current)
    explicit_off = _build_neighborhood_query(depth, prefer_current, include_edge_metadata=False)
    assert two_arg == explicit_off
    # No edge-metadata keys leak into the default query.
    assert "rel_type" not in two_arg
    assert "rel_confidence" not in two_arg
    assert "rel_direction" not in two_arg
    assert "_cands" not in two_arg


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_flag_on_adds_edge_metadata_keys(depth: int) -> None:
    """include_edge_metadata=True returns type/confidence/direction per hop."""
    query = _build_neighborhood_query(depth, prefer_current=False, include_edge_metadata=True)
    assert "rel_type: head(" in query
    assert "rel_confidence: head(" in query
    assert "rel_direction: head(" in query
    # Direction is computed from the traversed edge orientation.
    assert "startNode(_r" in query
    assert "type(_r" in query


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_flag_on_preserves_min_distance_contract_shape(depth: int) -> None:
    """The #1419 BFS shape (one hop-block per depth, distance last) is intact."""
    query = _build_neighborhood_query(depth, prefer_current=False, include_edge_metadata=True)
    # The exponential all-paths pattern must still be absent.
    assert "[*1.." not in query
    # One expansion block + one DISTINCT frontier collect per hop (the extra
    # metadata collect is NOT ``collect(DISTINCT`` so the count is unchanged).
    assert query.count("OPTIONAL MATCH") == depth
    assert query.count("collect(DISTINCT") == depth
    assert query.count("[0..$hop_limit]") == depth
    # ``distance`` stays the LAST key of each reported map (so the shape
    # assertion ``distance: {i}}}`` used by the dual_nodes tests still holds).
    for i in range(1, depth + 1):
        assert f"distance: {i}}}" in query
    assert f"distance: {depth + 1}}}" not in query


def test_flag_on_prefer_current_keeps_temporal_filters() -> None:
    """Edge metadata coexists with the per-hop valid_until filtering."""
    depth = 2
    query = _build_neighborhood_query(depth, prefer_current=True, include_edge_metadata=True)
    for i in range(1, depth + 1):
        assert f"_r{i}.valid_until IS NULL OR _r{i}.valid_until > _now" in query
    assert query.count("x.valid_until IS NULL OR x.valid_until > _now") == depth
    # No per-row cast of the stored property (would defeat the index).
    assert "datetime(_r" not in query


# --------------------------------------------------------------------------
# Type-prior table.
# --------------------------------------------------------------------------


def test_type_prior_downweights_cooccurrence() -> None:
    """ASSOCIATED_WITH (co-occurrence) gets a lower prior than a typed relation."""
    assert _relationship_type_prior("ASSOCIATED_WITH") == _RELATIONSHIP_TYPE_PRIORS["ASSOCIATED_WITH"]
    assert _relationship_type_prior("ASSOCIATED_WITH") < _relationship_type_prior("WORKS_WITH")


def test_type_prior_unknown_and_none_fall_back_to_default() -> None:
    assert _relationship_type_prior("SOME_UNSEEN_TYPE") == _DEFAULT_RELATIONSHIP_TYPE_PRIOR
    assert _relationship_type_prior(None) == _DEFAULT_RELATIONSHIP_TYPE_PRIOR
    assert _relationship_type_prior("") == _DEFAULT_RELATIONSHIP_TYPE_PRIOR


# --------------------------------------------------------------------------
# _cypher_expand scoring: flag OFF = legacy decay; flag ON = decay*conf*prior.
# --------------------------------------------------------------------------


def _retriever(*, typed_weighted: bool) -> VectorCypherRetriever:
    """A retriever whose ``_dual_nodes.get_entity_neighborhoods`` is mocked."""
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(enable_typed_weighted_expansion=typed_weighted),
        storage=None,
    )


def _neighborhood(seed, related_id, *, rel_type, rel_confidence, distance=1):
    return {
        str(seed): [
            {
                "id": str(related_id),
                "name": "bob",
                "entity_type": "PERSON",
                "description": "",
                "source_tool": "",
                "rel_type": rel_type,
                "rel_confidence": rel_confidence,
                "rel_direction": "out",
                "distance": distance,
            }
        ]
    }


async def test_scoring_flag_off_uses_pure_distance_decay() -> None:
    """Flag OFF: score == 1/(1+distance), ignoring type/confidence entirely."""
    seed, related = uuid4(), uuid4()
    retriever = _retriever(typed_weighted=False)
    retriever._dual_nodes = AsyncMock()
    retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
        return_value=_neighborhood(seed, related, rel_type="ASSOCIATED_WITH", rel_confidence=0.1, distance=1)
    )

    scores, _info = await retriever._cypher_expand([seed], uuid4(), depth=1)

    assert scores[related] == pytest.approx(0.5)  # 1 / (1 + 1); type/conf ignored
    # Flag off must NOT request edge metadata from the neighborhood query.
    kwargs = retriever._dual_nodes.get_entity_neighborhoods.await_args.kwargs
    assert kwargs["include_edge_metadata"] is False


async def test_scoring_flag_on_applies_confidence_and_type_prior() -> None:
    """Flag ON: score == 1/(1+distance) * confidence * type_prior."""
    seed, related = uuid4(), uuid4()
    retriever = _retriever(typed_weighted=True)
    retriever._dual_nodes = AsyncMock()
    retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
        return_value=_neighborhood(seed, related, rel_type="ASSOCIATED_WITH", rel_confidence=0.8, distance=1)
    )

    scores, _info = await retriever._cypher_expand([seed], uuid4(), depth=1)

    expected = 0.5 * 0.8 * _RELATIONSHIP_TYPE_PRIORS["ASSOCIATED_WITH"]
    assert scores[related] == pytest.approx(expected)
    kwargs = retriever._dual_nodes.get_entity_neighborhoods.await_args.kwargs
    assert kwargs["include_edge_metadata"] is True


async def test_scoring_flag_on_ranks_typed_above_cooccurrence() -> None:
    """A typed edge outranks a co-occurrence edge at the same distance/confidence."""
    seed = uuid4()
    typed_id, cooc_id = uuid4(), uuid4()
    retriever = _retriever(typed_weighted=True)
    retriever._dual_nodes = AsyncMock()
    retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
        return_value={
            str(seed): [
                {
                    "id": str(typed_id),
                    "name": "typed",
                    "entity_type": "PERSON",
                    "rel_type": "WORKS_WITH",
                    "rel_confidence": 0.9,
                    "rel_direction": "out",
                    "distance": 1,
                },
                {
                    "id": str(cooc_id),
                    "name": "cooc",
                    "entity_type": "PERSON",
                    "rel_type": "ASSOCIATED_WITH",
                    "rel_confidence": 0.9,
                    "rel_direction": "out",
                    "distance": 1,
                },
            ]
        }
    )

    scores, _info = await retriever._cypher_expand([seed], uuid4(), depth=1)

    assert scores[typed_id] > scores[cooc_id]


async def test_scoring_flag_on_missing_metadata_degrades_to_decay() -> None:
    """Flag ON but no rel metadata (e.g. SurrealDB): score falls back to decay."""
    seed, related = uuid4(), uuid4()
    retriever = _retriever(typed_weighted=True)
    retriever._dual_nodes = AsyncMock()
    retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
        return_value={
            str(seed): [
                {
                    "id": str(related),
                    "name": "bob",
                    "entity_type": "PERSON",
                    # No rel_type / rel_confidence keys at all.
                    "distance": 2,
                }
            ]
        }
    )

    scores, _info = await retriever._cypher_expand([seed], uuid4(), depth=2)

    # decay (1/3) * 1.0 (default confidence) * 1.0 (default prior) == 1/3.
    assert scores[related] == pytest.approx(1.0 / 3)
