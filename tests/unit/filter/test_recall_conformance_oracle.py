"""Recall-result conformance oracle over filtered recalls — ``@internal``.

The compile-side conformance harness (:mod:`tests.unit.filter.test_conformance_catalog`
+ :mod:`~tests.unit.filter.test_conformance_harness`) checks that each backend's
filter COMPILE agrees with the Python oracle over seed records. This module checks
the complementary, OUTPUT-side invariant: a real filtered recall's returned
surfaces conform to the SAME compiled predicate, WITHOUT knowing how the engine
produced them.

For every filtered corpus case it drives a real recall (engine-level, no live DB)
and calls the implementation-blind oracle
:func:`khora.filter.conformance.assert_recall_conformance`:

  1. every RETURNED chunk passes the compiled ``"Chunk"`` predicate, and
  2. every RETURNED entity has ≥1 provenance chunk that passes (the ∃-over-
     provenance rule #1457 / #1458 enforce on the entity surface).

Both multi-surface engines are exercised on the SAME corpus:

* the **VectorCypher** leg reuses the report suite's fully-wired retriever + the
  #1457 provenance wiring (``TestRecallConformanceOracleVectorCypher``), and
* the **Chronicle** leg drives ``ChronicleEngine.recall`` with the #1458
  ∃-over-provenance entity filter live (``TestRecallConformanceOracleChronicle``).

The oracle is fed the SEEDED ``Chunk`` objects the returned ids point back to (the
``RecallChunk`` / ``RecallEntity`` projections drop the metadata / document-key
fields a filter can address), so the predicate sees the full filterable surface —
the "implementation-blind" contract: seed known chunks, recall, verify the
invariant over what came back. The oracle asserts the INVARIANT, never the
mechanism, so the SAME check runs unchanged against either engine's output.

A falsifiability test proves the oracle has teeth (it must REJECT a recall that
surfaces a chunk / entity the predicate excludes), mirroring the compile harness's
``test_assert_case_fails_on_wrong_expected_ids``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk
from khora.core.models.document import DocumentSource
from khora.core.models.entity import Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.filter import parse_to_ast
from khora.filter.ast import FilterNode
from khora.filter.conformance import assert_recall_conformance
from khora.filter.model import RecallFilter
from khora.query import SearchMode
from tests.unit.engines.test_vectorcypher_filter_report import (
    _build_engine,
    _make_retriever,
)

pytestmark = pytest.mark.unit


def _ast(spec: dict[str, Any]) -> FilterNode:
    """Lower a wire filter to its canonical AST exactly as the facade does."""
    return parse_to_ast(RecallFilter.model_validate(spec))


def _chunk(
    ns_id: UUID,
    *,
    year: int,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """A seed chunk carrying the filterable fields a recall filter can address."""
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"seed chunk {year}",
        occurred_at=datetime(year, 6, 1, tzinfo=UTC),
        metadata=metadata or {},
        source_document=(DocumentSource(id=uuid4(), source=source) if source is not None else None),
    )


async def _recall_surfaces(
    filter_spec: dict[str, Any],
    *,
    ns_id: UUID,
    passing_entity_chunk: Chunk,
    failing_entity_chunk: Chunk | None,
    chunk_registry: dict[UUID, Chunk],
) -> tuple[list[Chunk], list[list[Chunk]]]:
    """Drive a filtered recall and map the returned surfaces back to seed chunks.

    Wires the graph path with one entity whose provenance is ``passing_entity_chunk``
    and (optionally) one whose provenance is ``failing_entity_chunk``, runs the
    recall, and returns ``(returned_chunk_objects, entity_provenance_objects)`` — the
    exact inputs :func:`assert_recall_conformance` consumes. Returned/provenance ids
    are resolved back to the seeded ``Chunk`` objects via ``chunk_registry`` (the
    ``RecallChunk`` / ``RecallEntity`` projections drop the fields a filter reads).
    """
    retriever = _make_retriever(ns_id, enable_bm25=True)

    passing_entity = Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name="Passing",
        entity_type="EVENT",
        source_chunk_ids=[passing_entity_chunk.id],
    )
    sim: list[tuple[UUID, float]] = [(passing_entity.id, 0.9)]
    ents: dict[UUID, Entity] = {passing_entity.id: passing_entity}
    if failing_entity_chunk is not None:
        failing_entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Failing",
            entity_type="EVENT",
            source_chunk_ids=[failing_entity_chunk.id],
        )
        sim.append((failing_entity.id, 0.8))
        ents[failing_entity.id] = failing_entity

    retriever._storage.search_similar_entities = AsyncMock(return_value=sim)
    retriever._storage.get_entities_batch = AsyncMock(return_value=ents)
    retriever._storage.get_chunks_batch = AsyncMock(return_value=dict(chunk_registry))
    retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])
    engine = _build_engine(retriever)

    result = await engine.recall(
        "alpha bravo charlie",
        ns_id,
        limit=10,
        mode=SearchMode.HYBRID,
        filter_ast=_ast(filter_spec),
    )

    returned_chunks = [chunk_registry[c.id] for c in result.chunks if c.id in chunk_registry]
    entity_provenance = [
        [chunk_registry[cid] for cid in ent.source_chunk_ids if cid in chunk_registry] for ent in result.entities
    ]
    return returned_chunks, entity_provenance


async def _recall_surfaces_chronicle(
    filter_spec: dict[str, Any],
    *,
    ns_id: UUID,
    passing_entity_chunk: Chunk,
    failing_entity_chunk: Chunk,
    chunk_registry: dict[UUID, Chunk],
) -> tuple[list[Chunk], list[list[Chunk]]]:
    """Drive a filtered Chronicle recall and map the returned surfaces back to seed chunks.

    The Chronicle counterpart of :func:`_recall_surfaces`. Wires the semantic
    channel to return the passing chunk (the chunk surface the filter narrows) and
    the entity channel to resolve one entity whose provenance is
    ``passing_entity_chunk`` and one whose provenance is ``failing_entity_chunk``;
    the #1458 ∃-over-provenance pass drops the decoy so exactly the passing entity
    survives. Runs a HYBRID recall with the router disabled (a single-token query
    would otherwise route SIMPLE and skip the entity channel), then maps the
    returned ``RecallChunk`` / ``RecallEntity`` ids back to the seeded ``Chunk``
    objects via ``chunk_registry`` — the exact inputs
    :func:`assert_recall_conformance` consumes.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))

    passing_entity = Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name="Passing",
        entity_type="EVENT",
        source_chunk_ids=[passing_entity_chunk.id],
    )
    failing_entity = Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name="Failing",
        entity_type="EVENT",
        source_chunk_ids=[failing_entity_chunk.id],
    )

    storage = AsyncMock()
    # Semantic channel returns the passing chunk; bm25 returns nothing so the
    # chunk surface is the semantic result narrowed by the post-filter alone.
    storage.search_similar_chunks = AsyncMock(return_value=[(passing_entity_chunk, 0.9)])
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    # Entity channel: both entities surface; ∃-over-provenance drops the decoy.
    storage.search_similar_entities = AsyncMock(return_value=[(passing_entity.id, 0.95), (failing_entity.id, 0.9)])
    storage.get_entities_batch = AsyncMock(
        return_value={passing_entity.id: passing_entity, failing_entity.id: failing_entity}
    )
    storage.get_chunks_batch = AsyncMock(return_value=dict(chunk_registry))
    storage.get_entities_by_names_batch = AsyncMock(return_value={})
    # Doc-key hydration falls back to each chunk's source_document (a source filter
    # resolves off there); an empty projection map keeps that fallback path clean.
    storage.get_document_projections_batch = AsyncMock(return_value={})
    engine._storage = storage

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    engine._embedder = embedder
    engine._connected = True
    # Keep every channel live: the entity channel runs only in HYBRID/ALL and is
    # gated off when the router classifies the query SIMPLE (mirrors the composition
    # test's ``_engine_with_entity``).
    engine._router_enabled = False

    result = await engine.recall(
        "alpha bravo charlie",
        ns_id,
        limit=10,
        mode=SearchMode.HYBRID,
        filter_ast=_ast(filter_spec),
    )

    returned_chunks = [chunk_registry[c.id] for c in result.chunks if c.id in chunk_registry]
    entity_provenance = [
        [chunk_registry[cid] for cid in ent.source_chunk_ids if cid in chunk_registry] for ent in result.entities
    ]
    return returned_chunks, entity_provenance


# The filtered corpus: each case is a filter whose passing/failing chunk fields the
# oracle re-checks over the recall's surfaces. Every case surfaces exactly one
# entity (its provenance passes) and drops a decoy entity (its provenance fails),
# so the ∃-over-provenance leg is exercised, not vacuous. Both engine legs share it.
_CASES = ("date_gte_2027", "source_linear", "metadata_tier_gold")


def _seed_for_case(case: str, ns_id: UUID) -> tuple[dict[str, Any], Chunk, Chunk]:
    """Return ``(filter_spec, passing_chunk, failing_chunk)`` for a corpus case."""
    if case == "date_gte_2027":
        return (
            {"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}},
            _chunk(ns_id, year=2027),
            _chunk(ns_id, year=2026),
        )
    if case == "source_linear":
        return (
            {"source": "linear", "occurred_at": {"$gte": "2027-01-01T00:00:00Z"}},
            _chunk(ns_id, year=2027, source="linear"),
            _chunk(ns_id, year=2027, source="slack"),
        )
    if case == "metadata_tier_gold":
        return (
            {"metadata.tier": "gold", "occurred_at": {"$gte": "2027-01-01T00:00:00Z"}},
            _chunk(ns_id, year=2027, metadata={"tier": "gold"}),
            _chunk(ns_id, year=2027, metadata={"tier": "silver"}),
        )
    raise AssertionError(f"unknown case {case!r}")


class TestRecallConformanceOracleVectorCypher:
    """Every filtered corpus case: the VectorCypher recall's surfaces conform to the predicate."""

    @pytest.mark.parametrize("case", _CASES)
    async def test_recall_surfaces_conform(self, case: str) -> None:
        """The returned chunks + entity provenance pass the compiled recall predicate.

        Drives a real filtered VectorCypher recall, then asserts the
        implementation-blind oracle: every returned chunk passes the ``"Chunk"``
        predicate AND every returned entity has ≥1 provenance chunk that passes. The
        decoy entity (failing provenance) is dropped, so exactly the passing entity
        survives — the ∃ leg is genuinely exercised.
        """
        ns_id = uuid4()
        filter_spec, passing_chunk, failing_chunk = _seed_for_case(case, ns_id)
        registry = {passing_chunk.id: passing_chunk, failing_chunk.id: failing_chunk}

        returned_chunks, entity_provenance = await _recall_surfaces(
            filter_spec,
            ns_id=ns_id,
            passing_entity_chunk=passing_chunk,
            failing_entity_chunk=failing_chunk,
            chunk_registry=registry,
        )

        # Precondition: the ∃ leg is not vacuous — exactly the passing entity
        # survived (the decoy with failing provenance was dropped).
        assert len(entity_provenance) == 1, (
            f"expected exactly the passing entity to survive, got {len(entity_provenance)} entities"
        )
        assert entity_provenance[0], "the surviving entity carried no provenance — the ∃ leg is vacuous"

        # The implementation-blind oracle: chunk surface + ∃-over-provenance.
        assert_recall_conformance(
            _ast(filter_spec),
            returned_chunks=returned_chunks,
            entity_provenance=entity_provenance,
        )


class TestRecallConformanceOracleChronicle:
    """Every filtered corpus case: the Chronicle recall's surfaces conform to the predicate.

    The Chronicle mirror of :class:`TestRecallConformanceOracleVectorCypher`,
    running the SAME implementation-blind oracle over the SAME corpus against the
    #1458 ∃-over-provenance entity filter. Chronicle returns ``relationships=[]``
    and covers its ``chunks`` surface unconditionally; the entity surface is covered
    (and its provenance filtered) only when a filter is present, which every case
    here supplies — so the ∃ leg is live, not inert.
    """

    @pytest.mark.parametrize("case", _CASES)
    async def test_recall_surfaces_conform(self, case: str) -> None:
        """The returned chunks + entity provenance pass the compiled recall predicate.

        Drives a real filtered Chronicle recall, then asserts the same
        implementation-blind oracle as the VectorCypher leg: every returned chunk
        passes the ``"Chunk"`` predicate AND every returned entity has ≥1 provenance
        chunk that passes. The decoy entity (failing provenance) is dropped by the
        #1458 pass, so exactly the passing entity survives — the ∃ leg is exercised.
        """
        ns_id = uuid4()
        filter_spec, passing_chunk, failing_chunk = _seed_for_case(case, ns_id)
        registry = {passing_chunk.id: passing_chunk, failing_chunk.id: failing_chunk}

        returned_chunks, entity_provenance = await _recall_surfaces_chronicle(
            filter_spec,
            ns_id=ns_id,
            passing_entity_chunk=passing_chunk,
            failing_entity_chunk=failing_chunk,
            chunk_registry=registry,
        )

        # Precondition: the chunk surface is non-empty (the semantic channel's
        # passing chunk survived the post-filter) so the chunk-surface leg is live.
        assert returned_chunks, "no chunk survived the Chronicle post-filter — the chunk-surface leg is vacuous"
        # Precondition: the ∃ leg is not vacuous — exactly the passing entity
        # survived (the decoy with failing provenance was dropped).
        assert len(entity_provenance) == 1, (
            f"expected exactly the passing entity to survive, got {len(entity_provenance)} entities"
        )
        assert entity_provenance[0], "the surviving entity carried no provenance — the ∃ leg is vacuous"

        # The implementation-blind oracle: chunk surface + ∃-over-provenance.
        assert_recall_conformance(
            _ast(filter_spec),
            returned_chunks=returned_chunks,
            entity_provenance=entity_provenance,
        )


class TestRecallConformanceOracleIsFalsifiable:
    """The oracle must REJECT a recall surface the predicate excludes (it has teeth)."""

    def test_oracle_rejects_a_chunk_that_fails_the_predicate(self) -> None:
        """A returned chunk outside the filter must raise — the chunk-surface leg has teeth."""
        ns_id = uuid4()
        filter_ast = _ast({"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}})
        # A 2026 chunk fails the 2027 lower bound; surfacing it violates the invariant.
        bad_chunk = _chunk(ns_id, year=2026)
        with pytest.raises(AssertionError, match="does not pass the recall filter predicate"):
            assert_recall_conformance(filter_ast, returned_chunks=[bad_chunk], entity_provenance=[])

    def test_oracle_rejects_an_entity_with_no_passing_provenance(self) -> None:
        """An entity whose every provenance chunk fails must raise — the ∃ leg has teeth."""
        ns_id = uuid4()
        filter_ast = _ast({"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}})
        # The entity's only provenance chunk is 2026 (fails); ∃ over it is False.
        failing_prov = _chunk(ns_id, year=2026)
        with pytest.raises(AssertionError, match="no provenance chunk passing"):
            assert_recall_conformance(filter_ast, returned_chunks=[], entity_provenance=[[failing_prov]])

    def test_oracle_rejects_an_entity_with_empty_provenance(self) -> None:
        """An entity with empty provenance fails the existential (a filtered recall never surfaces one)."""
        filter_ast = _ast({"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}})
        with pytest.raises(AssertionError, match="no provenance chunk passing"):
            assert_recall_conformance(filter_ast, returned_chunks=[], entity_provenance=[[]])

    def test_oracle_accepts_a_conformant_recall(self) -> None:
        """The control: a recall whose surfaces all pass must NOT raise (no degenerate always-raise)."""
        ns_id = uuid4()
        filter_ast = _ast({"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}})
        good_chunk = _chunk(ns_id, year=2027)
        prov = _chunk(ns_id, year=2027)
        assert_recall_conformance(filter_ast, returned_chunks=[good_chunk], entity_provenance=[[prov]])
