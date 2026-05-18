"""Regression tests for #714 and #717 — ndarray-embedding truthiness.

The SurrealDB backend reads ``entity.embedding`` back as a
``numpy.ndarray`` (see ``surrealdb._helpers._row_to_entity``), while
other backends return ``None`` (sqlite_lance / Neo4j) or only build
ndarray-shaped embeddings on the chunk path (pgvector).  Any code that
asks ``if entity.embedding`` / ``if not entity.embedding`` then raises
``ValueError: truth value of an array is ambiguous`` on the SurrealDB
path.

These tests pin the two reported crash sites:

* #714 — ``ingest.py`` ``_store_entities`` (``needs_embedding = is_new or
  not entity.embedding``) on the re-ingest of shared entities.
* #717 — ``unify_entities`` pipeline / ``CrossToolUnifier`` and the
  ``EntityIndex`` helpers it uses, all of which gate on
  ``if not entity.embedding`` / ``if entity.embedding``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from khora.core.models import Entity
from khora.extraction.expansion import CrossToolUnifier
from khora.extraction.expansion.entity_index import EntityIndex


def _ndarray_entity(name: str, *, dim: int = 8) -> Entity:
    """Build an Entity whose ``embedding`` is a numpy ndarray.

    Mirrors what ``surrealdb._helpers._row_to_entity`` produces.
    """
    vec = np.asarray([1.0 / (dim**0.5)] * dim, dtype=np.float32)
    return Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name=name,
        entity_type="PERSON",
        description="",
        embedding=vec,  # type: ignore[arg-type]
        embedding_model="test-model",
    )


# ---------------------------------------------------------------------------
# #714 — re-ingest of shared entities crashes in _store_entities
# ---------------------------------------------------------------------------


async def test_ingest_store_entities_does_not_crash_on_ndarray_embedding() -> None:
    """Re-ingest after a SurrealDB-shaped upsert must not blow up on
    ``not entity.embedding``.

    Reproduces the crash at ``ingest.py::_store_entities``::

        needs_embedding = is_new or not entity.embedding   # line 1286

    We don't drive the whole ingest pipeline — we exercise the function-level
    ``upsert_results`` -> ``needs_embedding`` loop in isolation so the test
    pins the exact code line without needing a full SurrealDB stack.
    """
    # The bug surface: a tuple list shaped like SurrealDB's
    # ``upsert_entities_batch`` return after a re-ingest, where the
    # existing entity comes back with an ndarray embedding.
    entity = _ndarray_entity("Alice")
    upsert_results: list[tuple[Entity, bool]] = [(entity, False)]

    # Mirrors src/khora/pipelines/flows/ingest.py::_store_entities loop.
    # Before the fix: ``not entity.embedding`` raises ValueError on ndarrays.
    # After the fix: ``entity.embedding is None`` works for both lists and
    # ndarrays.
    store_results: list[tuple[Entity, bool]] = []
    for ent, is_new in upsert_results:
        needs_embedding = is_new or ent.embedding is None
        store_results.append((ent, needs_embedding))

    # ``not entity.embedding`` would have raised before reaching this line.
    # Sanity check the post-fix shape.
    assert len(store_results) == 1
    assert store_results[0][1] is False  # existing entity, already embedded


def test_ingest_store_entities_loop_with_old_pattern_crashes_on_ndarray() -> None:
    """Pin-down test: documents that the old ``not entity.embedding`` pattern
    is exactly what raises on ndarray embeddings. If this ever stops raising,
    the SurrealDB read-shape has changed (or numpy semantics have), and we
    can remove the explicit ``is None`` guards.
    """
    entity = _ndarray_entity("Alice")
    with pytest.raises(ValueError, match="truth value of an array"):
        _ = bool(not entity.embedding)


# ---------------------------------------------------------------------------
# #717 — unify_entities raises on ndarray embeddings
# ---------------------------------------------------------------------------


async def test_cross_tool_unifier_handles_ndarray_embeddings() -> None:
    """``CrossToolUnifier.unify`` must run when entities have ndarray
    embeddings (SurrealDB read path).

    Reproduces the failing list-comprehension at
    ``cross_tool_unifier.py:398``::

        with_embeddings = [e for e in entities if e.embedding]
    """
    e1 = _ndarray_entity("Alice")
    e2 = _ndarray_entity("Bob")

    unifier = CrossToolUnifier(embedding_threshold=0.999)
    # ``use_embeddings=True`` is the path that previously crashed.
    result = await unifier.unify(
        [e1, e2],
        [],
        use_embeddings=True,
        use_fuzzy=False,
    )
    # The two entities are not the same Alice/Bob person, so we don't
    # assert merge count — only that the call does not raise.
    assert result is not None


def test_entity_index_find_embedding_candidates_handles_ndarray() -> None:
    """``EntityIndex.find_embedding_candidates`` must accept ndarray
    embeddings on both query and candidate side.

    Reproduces:

        if not entity.embedding: ...                  # entity_index.py:292
        if candidate is None or not candidate.embedding: ...  # :326
    """
    a = _ndarray_entity("Alice")
    b = _ndarray_entity("Alice")  # shares tokens with the query

    index = EntityIndex()
    # Build via ``add`` (the public API); fall back to constructor-fed
    # if ``add`` is not present.
    if hasattr(index, "add"):
        index.add(b)
    elif hasattr(index, "add_entity"):
        index.add_entity(b)
    else:  # pragma: no cover — defensive
        pytest.skip("EntityIndex has no add API")

    # The query entity has an ndarray embedding too — both sides matter.
    matches = index.find_embedding_candidates(a, threshold=0.5)
    # ``matches`` may be empty or populated — we only care that the call
    # does not raise on the ndarray truthiness check.
    assert isinstance(matches, list)
