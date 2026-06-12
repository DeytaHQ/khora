"""sqlite_lance parity: ``get_chunks_batch`` for temporal engines (#1086).

The skeleton / vectorcypher engines write ingested chunks to the
``khora_chunks`` temporal-store table, while chronicle writes to
``chunks``. ``SQLiteLanceVectorAdapter.get_chunks_batch`` only read the
``chunks`` table, so it returned ``{}`` for the temporal engines — which
broke VectorCypher's graph-channel chunk fetch (the surfaced entity's
chunk ids resolved to nothing) on the embedded stack.

The fix mirrors the existing ``get_chunks_by_document`` read-side
fallback (#905): when an id is not satisfied by ``chunks``, look it up in
``khora_chunks`` and decode it via ``_temporal_row_to_chunk`` (no
dual-write). A chronicle-only stack has no ``khora_chunks`` table; the
fallback swallows the missing-table error and returns what ``chunks``
yielded instead of crashing.

This is the storage-layer sibling of
``tests/unit/storage/test_get_chunks_by_document_temporal_fallback.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402

pytestmark = pytest.mark.embedded

_CONTENT = "PostgreSQL was chosen for the user database."


async def _remember(kb, namespace_id):
    return await kb.remember(
        _CONTENT,
        namespace=namespace_id,
        entity_types=[],
        relationship_types=[],
    )


@pytest.mark.parametrize("engine", ["skeleton", "vectorcypher"])
async def test_get_chunks_batch_reads_temporal_table(engine: str) -> None:
    """skeleton / vectorcypher write khora_chunks; get_chunks_batch finds them."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine=engine) as kb:
        ns = await kb.create_namespace()
        result = await _remember(kb, ns.namespace_id)
        assert result.chunks_created >= 1

        resolved = await kb.storage.resolve_namespace(ns.namespace_id)
        # Learn the chunk ids the way the graph channel does — via the document
        # read path (already khora_chunks-aware), then exercise the batch path.
        by_doc = await kb.storage.get_chunks_by_document(result.document_id, namespace_id=resolved)
        assert by_doc, f"{engine}: expected temporal chunks for the ingested doc"
        chunk_ids = [c.id for c in by_doc]

        # Anti-vacuity guard: these ids live ONLY in khora_chunks, never in the
        # plain ``chunks`` table for the temporal engines, so a ``chunks``-only
        # query would return ``{}``. The fallback is the sole reason the batch
        # resolves them.
        vec = kb.storage._vector  # type: ignore[union-attr]
        plain_cur = await vec._sqlite.execute("SELECT count(*) FROM chunks")  # type: ignore[attr-defined]
        assert (await plain_cur.fetchone())[0] == 0, f"{engine}: expected an empty `chunks` table"

        fetched = await kb.storage.get_chunks_batch(chunk_ids, namespace_id=resolved)
        assert set(fetched.keys()) == set(chunk_ids), (
            f"{engine}: get_chunks_batch did not resolve khora_chunks ids via the fallback; "
            f"got {set(fetched.keys())} vs {set(chunk_ids)}"
        )
        assert all(c.content == _CONTENT for c in fetched.values())


async def test_get_chunks_batch_chronicle_uses_chunks_table() -> None:
    """chronicle writes the ``chunks`` table directly; the fallback is a no-op."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="chronicle") as kb:
        ns = await kb.create_namespace()
        result = await _remember(kb, ns.namespace_id)
        assert result.chunks_created >= 1

        resolved = await kb.storage.resolve_namespace(ns.namespace_id)
        by_doc = await kb.storage.get_chunks_by_document(result.document_id, namespace_id=resolved)
        assert by_doc
        chunk_ids = [c.id for c in by_doc]

        fetched = await kb.storage.get_chunks_batch(chunk_ids, namespace_id=resolved)
        assert set(fetched.keys()) == set(chunk_ids)
        assert all(c.content == _CONTENT for c in fetched.values())
