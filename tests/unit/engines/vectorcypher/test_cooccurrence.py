"""Provenance tests for VectorCypher's ``_build_cooccurrence_relationships``.

The builder's signature takes a ``chunks: list[Chunk]`` argument so each
emitted ``ASSOCIATED_WITH`` edge can carry ``source_chunk_ids`` /
``source_document_ids`` provenance back to the chunk (and document) it
was synthesized from.

These tests pin the provenance contract — the cap and pair-dedup
behaviour is covered separately in
``test_engine_coverage.py::TestBuildCooccurrenceRelationships``; this
module focuses on the new fields.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from loguru import logger as loguru_logger

from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher.engine import (
    _MAX_COOCCURRENCE_PER_CHUNK,
    _build_cooccurrence_relationships,
)


def _chunk(document_id=None) -> Chunk:
    """Make a Chunk with a fresh id, attached to ``document_id`` (or a fresh doc)."""
    return Chunk(document_id=document_id or uuid4())


@pytest.mark.unit
class TestCooccurrenceProvenance:
    def test_source_document_ids_populated_for_single_chunk(self) -> None:
        """Pair from one chunk gets that chunk's id + its document's id stamped on."""
        ns = uuid4()
        doc = uuid4()
        chunk = Chunk(document_id=doc)
        e_a = Entity(name="A", entity_type="PERSON", source_chunk_ids=[chunk.id])
        e_b = Entity(name="B", entity_type="PERSON", source_chunk_ids=[chunk.id])

        rels = _build_cooccurrence_relationships([e_a, e_b], [chunk], ns, [])

        assert len(rels) == 1
        rel = rels[0]
        assert rel.source_chunk_ids == [chunk.id]
        assert rel.source_document_ids == [doc]

    def test_multi_chunk_pair_dedup_keeps_first_seen_chunk(self) -> None:
        """When A,B co-occur in two chunks, exactly one edge is emitted and it
        carries the FIRST chunk's (and document's) provenance.

        Pair-dedup uses ``existing_pairs.add(pair)`` after the first emission,
        so the second chunk's iteration short-circuits before producing a
        duplicate edge.  The first-seen chunk's id/document_id wins.
        """
        ns = uuid4()
        d1, d2 = uuid4(), uuid4()
        chunk_x = Chunk(document_id=d1)
        chunk_y = Chunk(document_id=d2)
        e_a = Entity(
            name="A",
            entity_type="PERSON",
            source_chunk_ids=[chunk_x.id, chunk_y.id],
        )
        e_b = Entity(
            name="B",
            entity_type="PERSON",
            source_chunk_ids=[chunk_x.id, chunk_y.id],
        )

        rels = _build_cooccurrence_relationships([e_a, e_b], [chunk_x, chunk_y], ns, [])

        # Pair-dedup → exactly one edge survives.
        assert len(rels) == 1
        rel = rels[0]
        # The first chunk iterated (whichever the dict ordering hands back
        # first; in CPython 3.7+ that is insertion order from entity's
        # source_chunk_ids, so chunk_x) wins the provenance race.
        assert rel.source_chunk_ids == [chunk_x.id]
        assert rel.source_document_ids == [d1]

    def test_chunk_without_doc_fallback_emits_warning(self) -> None:
        """Entity references a chunk_id not present in ``chunks`` → relationship
        is still created with empty ``source_document_ids`` and a loguru
        WARNING is emitted.

        This is the defensive fallback path: extraction-time entity
        ``source_chunk_ids`` should always be a subset of the chunks passed
        in, but we must not crash if a caller passes mismatched inputs.
        """
        ns = uuid4()
        missing_chunk_id = uuid4()
        e_a = Entity(name="A", entity_type="PERSON", source_chunk_ids=[missing_chunk_id])
        e_b = Entity(name="B", entity_type="PERSON", source_chunk_ids=[missing_chunk_id])

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            # chunks is empty — missing_chunk_id is not in the map.
            rels = _build_cooccurrence_relationships([e_a, e_b], [], ns, [])
        finally:
            loguru_logger.remove(sink_id)

        assert len(rels) == 1
        rel = rels[0]
        assert rel.source_document_ids == []
        # The chunk_id we DO know about (the one the entity references) is
        # still recorded — only the document mapping is missing.
        assert rel.source_chunk_ids == [missing_chunk_id]
        assert any("missing from chunks list" in m and str(missing_chunk_id) in m for m in records), (
            f"expected warning about missing chunk, got: {records}"
        )

    def test_no_duplicate_pairs_across_chunks(self) -> None:
        """Two entities sharing two chunks must produce exactly one edge.

        This is the dedup invariant — duplicate ``(min(id), max(id))``
        pairs across chunks must collapse.  Without dedup the builder
        would emit two ``ASSOCIATED_WITH`` edges between A and B (one
        per chunk), bloating the graph.
        """
        ns = uuid4()
        chunk_x = Chunk(document_id=uuid4())
        chunk_y = Chunk(document_id=uuid4())
        e_a = Entity(
            name="A",
            entity_type="PERSON",
            source_chunk_ids=[chunk_x.id, chunk_y.id],
        )
        e_b = Entity(
            name="B",
            entity_type="PERSON",
            source_chunk_ids=[chunk_x.id, chunk_y.id],
        )

        rels = _build_cooccurrence_relationships([e_a, e_b], [chunk_x, chunk_y], ns, [])

        assert len(rels) == 1, (
            f"expected pair-dedup → 1 edge, got {len(rels)}: {[(r.source_entity_id, r.target_entity_id) for r in rels]}"
        )

    def test_max_cooccurrence_per_chunk_cap_respected(self) -> None:
        """``_MAX_COOCCURRENCE_PER_CHUNK`` caps the pair count per chunk.

        10 entities in a single chunk → C(10,2) = 45 raw pairs.  The cap
        must keep emissions at exactly ``_MAX_COOCCURRENCE_PER_CHUNK``.
        """
        ns = uuid4()
        chunk = Chunk(document_id=uuid4())
        entities = [Entity(name=f"E{i}", entity_type="CONCEPT", source_chunk_ids=[chunk.id]) for i in range(10)]

        rels = _build_cooccurrence_relationships(entities, [chunk], ns, [])

        assert len(rels) == _MAX_COOCCURRENCE_PER_CHUNK, (
            f"expected cap at {_MAX_COOCCURRENCE_PER_CHUNK}, got {len(rels)}"
        )
        # Even the capped edges carry provenance.
        for rel in rels:
            assert rel.source_chunk_ids == [chunk.id]
            assert rel.source_document_ids == [chunk.document_id]

    def test_existing_pair_skipped_does_not_emit_provenance(self) -> None:
        """If a pair already exists in ``existing_relationships``, the
        co-occurrence builder skips it entirely — no edge, no provenance.

        Guards against the bug where the builder would emit a duplicate
        edge with new provenance, overwriting the original extraction
        relationship's source link.
        """
        ns = uuid4()
        chunk = Chunk(document_id=uuid4())
        e_a = Entity(name="A", entity_type="PERSON", source_chunk_ids=[chunk.id])
        e_b = Entity(name="B", entity_type="PERSON", source_chunk_ids=[chunk.id])
        existing = Relationship(
            source_entity_id=e_b.id,
            target_entity_id=e_a.id,
            relationship_type="KNOWS",
            namespace_id=ns,
        )

        rels = _build_cooccurrence_relationships([e_a, e_b], [chunk], ns, [existing])

        assert rels == []
