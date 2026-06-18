"""Unit tests for the leaf core-chunk ranking util ``khora.core.ranking``.

The util was extracted from ``SkeletonIndexer.build_skeleton`` (task #29). These
tests lock in:

* **Parity** — ``select_core_chunk_ids`` returns exactly the same core ids, in
  the same order, as the legacy ``SkeletonIndexer`` path, including the
  insertion-order tie-break under identical-content (tied-score) chunks.
* **CoreSelection** shape — ``scores`` covers every input id (floats),
  ``core_ids`` is a score-descending top-n subset of size ``max(1, int(n*r))``.
* The thin wrapper equals ``CoreSelection.core_ids``.
* Edge cases — empty input and single-chunk input.
* ``SkeletonIndexer`` delegation — ``get_core_chunks`` / ``is_core_chunk`` /
  ``get_pagerank_score`` still behave correctly after ``build_skeleton``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from khora.core.ranking import CoreSelection, select_core_chunk_ids, select_core_chunks
from khora.core.temporal import TemporalChunk
from khora.engines.skeleton.skeleton import SkeletonIndexer

_NS = uuid4()
_DOC = uuid4()


def _chunk(content: str) -> TemporalChunk:
    """Build a TemporalChunk with a fresh uuid4 id and the given content."""
    return TemporalChunk(
        id=uuid4(),
        namespace_id=_NS,
        document_id=_DOC,
        content=content,
    )


def _varied_chunks() -> list[TemporalChunk]:
    """A fixed, varied multi-chunk input (>= 7 chunks) for parity testing."""
    return [
        _chunk("The quick brown fox jumps over the lazy dog near the river bank."),
        _chunk("Machine learning models require large datasets for effective training."),
        _chunk("PostgreSQL provides robust transactional guarantees and indexing."),
        _chunk("The river bank flooded after heavy rain caused the dog to flee."),
        _chunk("Neural networks learn hierarchical feature representations from data."),
        _chunk("Vector search retrieves nearest neighbors using cosine similarity."),
        _chunk("Knowledge graphs encode entities and relationships as nodes and edges."),
        _chunk("The lazy dog slept while the quick fox hunted near the river."),
        _chunk("Embeddings map text into a dense continuous vector space for search."),
    ]


def _legacy_core_ids(chunks: list[TemporalChunk], ratio: float) -> list[UUID]:
    """Run the OLD SkeletonIndexer path and return its core ids."""
    indexer = SkeletonIndexer(core_ratio=ratio)
    indexer.add_chunks_batch(chunks)
    return indexer.build_skeleton()


# ---------------------------------------------------------------------------
# Parity (most important)
# ---------------------------------------------------------------------------


def test_parity_varied_chunks_same_ids_and_order() -> None:
    """Util core ids EQUAL legacy core ids — same elements AND same order."""
    chunks = _varied_chunks()
    ratio = 0.4

    legacy = _legacy_core_ids(chunks, ratio)
    util = select_core_chunk_ids(chunks, ratio)

    assert util == legacy, f"order/element mismatch: util={util} legacy={legacy}"


def test_parity_tied_scores_insertion_order_tiebreak() -> None:
    """Identical-content chunks tie on score; insertion order must be the
    tie-break, identically across both paths."""
    # Several identical-content chunks plus a couple of distinct ones. The
    # identical chunks score equally, so ordering among them is decided purely
    # by the stable sort over the input (insertion) order.
    chunks = [
        _chunk("identical content shared across several tied chunks here"),
        _chunk("identical content shared across several tied chunks here"),
        _chunk("identical content shared across several tied chunks here"),
        _chunk("identical content shared across several tied chunks here"),
        _chunk("a wholly different sentence about databases and indexes"),
        _chunk("another distinct sentence concerning vectors and embeddings"),
        _chunk("identical content shared across several tied chunks here"),
    ]
    ratio = 0.5

    legacy = _legacy_core_ids(chunks, ratio)
    util = select_core_chunk_ids(chunks, ratio)

    assert util == legacy, f"tie-break mismatch: util={util} legacy={legacy}"


def test_parity_default_ratio() -> None:
    """Parity holds at the SkeletonIndexer default core_ratio (0.1)."""
    chunks = _varied_chunks()
    ratio = 0.1

    legacy = _legacy_core_ids(chunks, ratio)
    util = select_core_chunk_ids(chunks, ratio)

    assert util == legacy


def test_parity_two_chunk_ids_and_scores() -> None:
    """The n==2 small-doc case: util matches legacy on BOTH ids and scores.

    The two-chunk path is the smallest case the engine's len<=2 fast path
    guards; this pins core-id ordering AND the per-chunk PageRank score map
    against the legacy ``SkeletonIndexer`` (real ``khora._accel``, not mocked).
    """
    chunks = [
        _chunk("Machine learning models require large datasets for training."),
        _chunk("PostgreSQL provides transactional guarantees and rich indexing."),
    ]
    ratio = 0.5

    indexer = SkeletonIndexer(core_ratio=ratio)
    indexer.add_chunks_batch(chunks)
    legacy_ids = indexer.build_skeleton()

    result = select_core_chunks(chunks, ratio)

    # Core-id ordering parity.
    assert result.core_ids == legacy_ids

    # Per-chunk score parity against the legacy indexer's node state.
    for c in chunks:
        assert result.scores[c.id] == indexer.get_pagerank_score(c.id)


# ---------------------------------------------------------------------------
# CoreSelection shape
# ---------------------------------------------------------------------------


def test_core_selection_scores_cover_every_input_id() -> None:
    chunks = _varied_chunks()
    result = select_core_chunks(chunks, 0.4)

    input_ids = {c.id for c in chunks}
    assert set(result.scores.keys()) == input_ids
    assert all(isinstance(v, float) for v in result.scores.values())


def test_core_selection_core_ids_subset_and_size() -> None:
    chunks = _varied_chunks()
    ratio = 0.4
    n = len(chunks)
    result = select_core_chunks(chunks, ratio)

    assert set(result.core_ids).issubset(set(result.scores.keys()))
    assert len(result.core_ids) == max(1, int(n * ratio))


def test_core_selection_core_ids_are_top_n_by_score() -> None:
    """core_ids are the highest-scoring ids, in descending score order."""
    chunks = _varied_chunks()
    result = select_core_chunks(chunks, 0.5)

    core_scores = [result.scores[cid] for cid in result.core_ids]
    # Descending.
    assert core_scores == sorted(core_scores, reverse=True)
    # No non-core chunk outscores the lowest core chunk.
    non_core = [s for cid, s in result.scores.items() if cid not in set(result.core_ids)]
    if non_core:
        assert min(core_scores) >= max(non_core)


# ---------------------------------------------------------------------------
# Thin wrapper
# ---------------------------------------------------------------------------


def test_thin_wrapper_equals_core_selection_core_ids() -> None:
    chunks = _varied_chunks()
    ratio = 0.4

    full = select_core_chunks(chunks, ratio)
    wrapper = select_core_chunk_ids(chunks, ratio)

    assert wrapper == full.core_ids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_selection() -> None:
    result = select_core_chunks([], 0.4)
    assert result == CoreSelection([], {})
    assert select_core_chunk_ids([], 0.4) == []


def test_single_chunk_yields_exactly_one_core() -> None:
    chunks = [_chunk("a single lonely chunk with some words in it")]
    result = select_core_chunks(chunks, 0.1)  # 0.1 * 1 == 0 -> max(1, 0) == 1

    assert len(result.core_ids) == 1
    assert result.core_ids[0] == chunks[0].id
    assert set(result.scores.keys()) == {chunks[0].id}


# ---------------------------------------------------------------------------
# SkeletonIndexer delegation
# ---------------------------------------------------------------------------


def test_skeleton_delegation_state_after_build() -> None:
    """After build_skeleton, the indexer's per-chunk state reflects the util's
    result: core flags, pagerank scores, and the lookup helpers."""
    chunks = _varied_chunks()
    ratio = 0.4

    indexer = SkeletonIndexer(core_ratio=ratio)
    indexer.add_chunks_batch(chunks)
    core_ids = indexer.build_skeleton()

    # get_core_chunks matches the returned core ids (set-wise).
    assert set(indexer.get_core_chunks()) == set(core_ids)

    # is_core_chunk: True for core ids, False otherwise.
    core_set = set(core_ids)
    for c in chunks:
        assert indexer.is_core_chunk(c.id) == (c.id in core_set)

    # get_pagerank_score matches the util's scores for the same input.
    util_scores = select_core_chunks(chunks, ratio).scores
    for c in chunks:
        assert indexer.get_pagerank_score(c.id) == util_scores[c.id]


def test_skeleton_delegation_unknown_id_defaults() -> None:
    """Helpers degrade gracefully for an id never added to the indexer."""
    chunks = _varied_chunks()
    indexer = SkeletonIndexer(core_ratio=0.4)
    indexer.add_chunks_batch(chunks)
    indexer.build_skeleton()

    missing = uuid4()
    assert indexer.is_core_chunk(missing) is False
    assert indexer.get_pagerank_score(missing) == 0.0
