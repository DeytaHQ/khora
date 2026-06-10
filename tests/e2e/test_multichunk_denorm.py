"""Multi-chunk denormalization uniformity — embedded sqlite_lance, no Docker.

When one document chunks into several pieces, the document-grained fields must be
denormalized identically onto EVERY chunk: a filter that addresses a document key
(``occurred_at`` / ``source_timestamp`` / the document projection's denormalized
keys) must treat every chunk of the document the same way, or a multi-chunk
document would filter inconsistently across its own chunks.

This drives the real ``Khora.remember()`` ingest path with content long enough to
split into ``>= 2`` chunks, then asserts via recall that every returned chunk of the
document carries the same document-grained event time and resolves to one document
projection whose denormalized keys equal the seeded values. No Docker — runs on the
default ``uv run pytest -m e2e`` path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.query import SearchMode
from tests.e2e import _harness

pytestmark = pytest.mark.e2e

_SOURCE_TIMESTAMP = datetime(2026, 1, 2, tzinfo=UTC)


async def test_multichunk_denormalization_is_uniform(sqlite_lance_kb) -> None:
    """Every chunk of a multi-chunk document shares its document-grained fields.

    Seeds one long document (``>= 2`` chunks) with a ``source_timestamp``, a
    denormalized ``source_name`` / ``external_id``, and a ``metadata`` blob, then
    recalls it. The assertion is uniformity: all returned chunks belong to the SAME
    document and carry the SAME event time (``occurred_at`` falls back to the
    document ``source_timestamp``), and they resolve to exactly ONE document
    projection whose denormalized keys equal the seeded values — so a document-key
    filter narrows every chunk of the document identically.
    """
    kb = sqlite_lance_kb
    namespace_id = (await kb.create_namespace()).namespace_id

    content = _harness.multi_chunk_doc("denormmark", chunks=4)
    result = await kb.remember(
        content=content,
        namespace=namespace_id,
        source_timestamp=_SOURCE_TIMESTAMP,
        source_name="linear",
        external_id="denorm-doc",
        metadata={"tier": "gold"},
        entity_types=["ENTITY"],
        relationship_types=["RELATED_TO"],
    )
    assert result.chunks_created >= 2, (
        f"the document must split into >= 2 chunks to test uniformity, got {result.chunks_created}"
    )

    recalled = await kb.recall(
        "denormmark",
        namespace=namespace_id,
        mode=SearchMode.VECTOR,
        limit=_harness._RECALL_LIMIT,
        min_similarity=0.0,
    )

    # All recalled chunks belong to the one seeded document.
    chunk_doc_ids = {chunk.document_id for chunk in recalled.chunks}
    assert len(recalled.chunks) >= 2, f"expected >= 2 recalled chunks, got {len(recalled.chunks)}"
    assert chunk_doc_ids == {result.document_id}, "recalled chunks span more than the one seeded document"

    # The document-grained event time is denormalized identically onto every chunk.
    occurred_at_values = {chunk.occurred_at for chunk in recalled.chunks}
    assert occurred_at_values == {_SOURCE_TIMESTAMP}, (
        f"occurred_at must be uniform across the document's chunks, got {occurred_at_values}"
    )

    # The chunks resolve to exactly one document projection carrying the seeded
    # denormalized keys — the whole-document filterable surface.
    projections = [doc for doc in recalled.documents if doc.id == result.document_id]
    assert len(projections) == 1, "the multi-chunk document must resolve to exactly one projection"
    projection = projections[0]
    assert projection.external_id == "denorm-doc"
    assert projection.source_name == "linear"
    # The embedded SQLite store round-trips datetimes tz-naive, so compare wall-clock
    # values (tzinfo-insensitive) — the recall-filter boundary normalizes tz-naive too.
    assert projection.source_timestamp is not None
    assert projection.source_timestamp.replace(tzinfo=None) == _SOURCE_TIMESTAMP.replace(tzinfo=None)
    assert projection.metadata == {"tier": "gold"}
