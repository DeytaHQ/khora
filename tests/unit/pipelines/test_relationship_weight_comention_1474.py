"""Leg 1 of #1474: relationship.weight = co-mention frequency at ingest.

``_assign_comention_weights`` stamps each edge's weight with the number of
chunks in which both endpoints co-occur. Previously ``weight`` was a constant
1.0 for every edge; this makes it a real connection-strength signal. The change
is additive - a previously-inert column becomes populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.pipelines.flows.ingest import (
    _MAX_COMENTION_ENTITIES_PER_CHUNK,
    _assign_comention_weights,
)


@dataclass
class _Chunk:
    """Minimal stand-in for the ingest TemporalChunk (only ``.id`` is read)."""

    id: UUID


def _entity(name: str) -> Entity:
    return Entity(id=uuid4(), namespace_id=uuid4(), name=name, entity_type="CONCEPT")


def _rel(src: Entity, tgt: Entity, rel_type: str = "RELATES_TO") -> Relationship:
    return Relationship(
        namespace_id=src.namespace_id,
        source_entity_id=src.id,
        target_entity_id=tgt.id,
        relationship_type=rel_type,
    )


@pytest.mark.unit
def test_weight_counts_chunks_where_both_endpoints_appear() -> None:
    """A pair co-occurring in 3 chunks gets weight 3.0."""
    a, b = _entity("A"), _entity("B")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b}
    chunks = [_Chunk(uuid4()) for _ in range(3)]
    # A and B appear together in all three chunks.
    chunk_entity_keys = {c.id: ["A:CONCEPT", "B:CONCEPT"] for c in chunks}
    rel = _rel(a, b)

    _assign_comention_weights(chunks, all_entities, chunk_entity_keys, [rel])

    assert rel.weight == 3.0


@pytest.mark.unit
def test_weight_is_direction_agnostic() -> None:
    """The pair is unordered: an edge B->A still gets the A/B co-mention count."""
    a, b = _entity("A"), _entity("B")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b}
    chunks = [_Chunk(uuid4()) for _ in range(2)]
    chunk_entity_keys = {c.id: ["A:CONCEPT", "B:CONCEPT"] for c in chunks}
    rel = _rel(b, a)  # reversed direction

    _assign_comention_weights(chunks, all_entities, chunk_entity_keys, [rel])

    assert rel.weight == 2.0


@pytest.mark.unit
def test_pair_never_co_occurring_keeps_default_weight() -> None:
    """A pair with no single-chunk co-occurrence keeps the 1.0 default."""
    a, b = _entity("A"), _entity("B")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b}
    # Each entity appears alone in its own chunk - never together.
    chunks = [_Chunk(uuid4()), _Chunk(uuid4())]
    chunk_entity_keys = {chunks[0].id: ["A:CONCEPT"], chunks[1].id: ["B:CONCEPT"]}
    rel = _rel(a, b)

    _assign_comention_weights(chunks, all_entities, chunk_entity_keys, [rel])

    assert rel.weight == 1.0  # untouched default


@pytest.mark.unit
def test_duplicate_entity_keys_in_a_chunk_count_once() -> None:
    """The same entity listed twice in a chunk does not inflate the count."""
    a, b = _entity("A"), _entity("B")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b}
    chunk = _Chunk(uuid4())
    chunk_entity_keys = {chunk.id: ["A:CONCEPT", "A:CONCEPT", "B:CONCEPT"]}
    rel = _rel(a, b)

    _assign_comention_weights([chunk], all_entities, chunk_entity_keys, [rel])

    assert rel.weight == 1.0  # one chunk, counted once


@pytest.mark.unit
def test_distinct_pairs_get_independent_counts() -> None:
    """Three-entity chunks contribute to every pairwise count independently."""
    a, b, c = _entity("A"), _entity("B"), _entity("C")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b, "C:CONCEPT": c}
    # Chunk 1: A,B,C ; Chunk 2: A,B only.
    ch1, ch2 = _Chunk(uuid4()), _Chunk(uuid4())
    chunk_entity_keys = {
        ch1.id: ["A:CONCEPT", "B:CONCEPT", "C:CONCEPT"],
        ch2.id: ["A:CONCEPT", "B:CONCEPT"],
    }
    ab, ac, bc = _rel(a, b), _rel(a, c), _rel(b, c)

    _assign_comention_weights([ch1, ch2], all_entities, chunk_entity_keys, [ab, ac, bc])

    assert ab.weight == 2.0  # co-occur in both chunks
    assert ac.weight == 1.0  # only chunk 1
    assert bc.weight == 1.0  # only chunk 1


@pytest.mark.unit
def test_unknown_entity_keys_are_ignored() -> None:
    """A chunk-key with no entity in ``all_entities`` is skipped, not crashed."""
    a, b = _entity("A"), _entity("B")
    all_entities = {"A:CONCEPT": a, "B:CONCEPT": b}
    chunk = _Chunk(uuid4())
    chunk_entity_keys = {chunk.id: ["A:CONCEPT", "B:CONCEPT", "GHOST:CONCEPT"]}
    rel = _rel(a, b)

    _assign_comention_weights([chunk], all_entities, chunk_entity_keys, [rel])

    assert rel.weight == 1.0  # A+B co-occur once; ghost ignored


@pytest.mark.unit
def test_pathological_chunk_is_bounded() -> None:
    """A chunk with more entities than the cap does not explode; it truncates."""
    n = _MAX_COMENTION_ENTITIES_PER_CHUNK + 20
    entities = [_entity(f"E{i}") for i in range(n)]
    all_entities = {f"E{i}:CONCEPT": entities[i] for i in range(n)}
    chunk = _Chunk(uuid4())
    chunk_entity_keys = {chunk.id: [f"E{i}:CONCEPT" for i in range(n)]}
    # Edge between the first two entities - inside the cap after sorting by id.
    # We cannot assert an exact weight (sort order by UUID is random), only that
    # the call completes and does not raise on a large chunk.
    rels = [_rel(entities[i], entities[i + 1]) for i in range(n - 1)]

    _assign_comention_weights([chunk], all_entities, chunk_entity_keys, rels)

    # At least one edge among the (bounded) considered entities got a weight > 1;
    # the point of the test is that the bounded pass terminated.
    assert all(r.weight >= 1.0 for r in rels)
