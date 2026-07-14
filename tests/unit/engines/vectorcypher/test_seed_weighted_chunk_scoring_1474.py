"""Leg 3 of #1474: seed-relevance-weighted graph-chunk scoring.

``_fetch_chunks_from_entities`` scored graph-channel chunks as
``total_mentions * (1 + 0.1 * entity_count)`` - fully query-agnostic. Leg 3
adds an optional ``seed_relevance`` map (entity_id -> query relevance); when
supplied it scores each chunk ``total_mentions * sum(relevance of its connected
entities)``, so a chunk connected to highly-query-relevant entities ranks above
one connected to many irrelevant ones. ``None`` keeps the legacy score
byte-identical.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever

pytestmark = pytest.mark.unit


def _retriever() -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=None,
    )


def _chunk_record(chunk_id, entity_ids, total_mentions):
    return {
        "chunk_id": str(chunk_id),
        "document_id": str(uuid4()),
        "content": "body",
        "total_mentions": total_mentions,
        "entity_ids": [str(e) for e in entity_ids],
        "metadata": {},
        "occurred_at": None,
        "source_timestamp": None,
        "chunker_info": {},
    }


async def _fetch(retriever, records, *, seed_relevance):
    """Drive ``_fetch_chunks_from_entities`` against a mocked dual_nodes."""
    retriever._dual_nodes = AsyncMock()
    retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=records)
    return await retriever._fetch_chunks_from_entities(
        entity_ids=[uuid4()],
        namespace_id=uuid4(),
        temporal_filter=None,
        limit=10,
        seed_relevance=seed_relevance,
    )


async def test_flag_off_reproduces_legacy_score() -> None:
    """seed_relevance=None -> total_mentions * (1 + 0.1 * entity_count)."""
    retriever = _retriever()
    cid = uuid4()
    e1, e2 = uuid4(), uuid4()
    records = [_chunk_record(cid, [e1, e2], total_mentions=3)]

    results = await _fetch(retriever, records, seed_relevance=None)

    (_id, score, _chunk) = results[0]
    assert score == pytest.approx(3.0 * (1 + 0.1 * 2))  # legacy formula


async def test_flag_on_weights_by_seed_relevance() -> None:
    """seed_relevance -> total_mentions * sum(connected entities' relevance)."""
    retriever = _retriever()
    cid = uuid4()
    e1, e2 = uuid4(), uuid4()
    records = [_chunk_record(cid, [e1, e2], total_mentions=3)]
    seed_relevance = {e1: 0.9, e2: 0.1}

    results = await _fetch(retriever, records, seed_relevance=seed_relevance)

    (_id, score, _chunk) = results[0]
    assert score == pytest.approx(3.0 * (0.9 + 0.1))  # mentions * relevance sum


async def test_flag_on_ranks_relevant_entity_chunk_above_noisy_one() -> None:
    """A chunk on one highly-relevant entity beats a chunk on two noisy ones."""
    retriever = _retriever()
    relevant_chunk, noisy_chunk = uuid4(), uuid4()
    e_rel, e_noise1, e_noise2 = uuid4(), uuid4(), uuid4()
    records = [
        _chunk_record(relevant_chunk, [e_rel], total_mentions=1),
        _chunk_record(noisy_chunk, [e_noise1, e_noise2], total_mentions=1),
    ]
    seed_relevance = {e_rel: 0.95, e_noise1: 0.05, e_noise2: 0.05}

    results = await _fetch(retriever, records, seed_relevance=seed_relevance)

    score_by_id = {cid: s for cid, s, _ in results}
    assert score_by_id[relevant_chunk] > score_by_id[noisy_chunk]


async def test_flag_on_no_relevance_signal_falls_back_to_legacy() -> None:
    """Connected entities absent from the map -> legacy score, not zero."""
    retriever = _retriever()
    cid = uuid4()
    e1, e2 = uuid4(), uuid4()
    records = [_chunk_record(cid, [e1, e2], total_mentions=4)]
    # None of the chunk's entities appear in the relevance map.
    seed_relevance = {uuid4(): 0.7}

    results = await _fetch(retriever, records, seed_relevance=seed_relevance)

    (_id, score, _chunk) = results[0]
    assert score == pytest.approx(4.0 * (1 + 0.1 * 2))  # fell back to legacy


async def test_flag_on_empty_entity_ids_falls_back_to_legacy() -> None:
    """A chunk with no connected entities (SurrealDB fallback) uses legacy score."""
    retriever = _retriever()
    cid = uuid4()
    records = [_chunk_record(cid, [], total_mentions=1)]

    results = await _fetch(retriever, records, seed_relevance={uuid4(): 0.5})

    (_id, score, _chunk) = results[0]
    assert score == pytest.approx(1.0 * (1 + 0.1 * 0))  # == 1.0, legacy
