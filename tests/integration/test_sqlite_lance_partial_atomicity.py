"""Cross-store partial-atomicity contract test (sqlite_lance).

The embedded coordinator documents (CLAUDE.md gotchas) that
``coordinator.transaction()`` does NOT promise full ACID across SQLite
and LanceDB. The specific contract for the chunk-write path
(``vector.py:create_chunks_batch``) is:

1. SQLite metadata commits FIRST.
2. LanceDB embedding insert SECOND.
3. If LanceDB raises after the SQLite commit, the exception is logged
   and re-raised. SQLite stays consistent; LanceDB has no vector for
   the orphan chunk row. Recall paths skip rows missing from LanceDB.

This module codifies that behaviour so a future refactor that "fixes"
partial atomicity (e.g., by rolling back SQLite on Lance failure)
trips an explicit failure and forces a deliberate decision rather than
silently breaking the documented contract.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, MemoryNamespace
from tests.integration._sqlite_lance_fixtures import (
    build_sqlite_lance_coordinator,
    fake_embedding,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed",
    ),
]


async def test_lance_add_failure_leaves_sqlite_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``tbl.add`` raises after the SQLite commit, the chunk row stays
    in SQLite. The exception propagates so the caller can reconcile.
    """
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        doc = Document(
            namespace_id=ns.id,
            content="partial-atomicity-test",
            external_id="atomicity-1",
            metadata=DocumentMetadata(source="test", title="atomicity"),
        )
        await coord.create_document(doc)

        chunk = Chunk(
            namespace_id=ns.id,
            document_id=doc.id,
            content="partial-atomicity content",
            metadata=ChunkMetadata(document_id=doc.id, chunk_index=0),
            embedding=fake_embedding("partial-atomicity content"),
            embedding_model="fake",
        )

        # Open the LanceDB chunks table once and replace its `add` method with
        # a raiser. The vector adapter caches the table handle, so subsequent
        # calls reuse this poisoned table.
        tbl = await coord.vector._chunks_table()  # type: ignore[union-attr]
        original_add = tbl.add

        async def _poisoned_add(*args, **kwargs):
            raise RuntimeError("simulated LanceDB add failure")

        monkeypatch.setattr(tbl, "add", _poisoned_add)

        with pytest.raises(RuntimeError, match="simulated LanceDB add failure"):
            await coord.create_chunks_batch([chunk])

        # SQLite must have the chunk row even though LanceDB is empty.
        sqlite_chunk = await coord.vector.get_chunk(chunk.id, namespace_id=ns.id)  # type: ignore[union-attr]
        assert sqlite_chunk is not None, (
            "Partial-atomicity contract violated: SQLite chunk row was rolled back "
            "after LanceDB failure. The documented behaviour is SQLite stays consistent."
        )
        assert sqlite_chunk.content == "partial-atomicity content"

        # Restore the real add and verify LanceDB never received the row.
        monkeypatch.setattr(tbl, "add", original_add)
        results = await coord.search_similar_chunks(
            ns.id,
            fake_embedding("partial-atomicity content"),
            limit=10,
        )
        # Recall must skip the orphan rather than crashing or fabricating.
        assert chunk.id not in {c.id for c, _ in results}, (
            "Recall returned a chunk that's only in SQLite — LanceDB orphan "
            "filter is broken. Search would surface chunks with no vector."
        )
    finally:
        with contextlib.suppress(Exception):
            await coord.disconnect()
