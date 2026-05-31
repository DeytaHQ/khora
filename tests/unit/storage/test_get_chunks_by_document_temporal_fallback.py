"""sqlite_lance parity: ``get_chunks_by_document`` for temporal engines (#905).

The skeleton / vectorcypher engines write ingested chunks to the
``khora_chunks`` temporal-store table, while chronicle writes to
``chunks``. The sqlite_lance vector adapter's ``get_chunks_by_document``
only read ``chunks``, so it returned ``[]`` for the temporal engines
even though recall (which reads ``khora_chunks``) found the chunks.

The fix is a read-side fallback: when the ``chunks`` query is empty,
read ``khora_chunks`` and map rows to ``Chunk`` (no dual-write). A
chronicle-only stack has no ``khora_chunks`` table; the fallback
swallows the missing-table error and returns ``[]`` instead of crashing.
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
async def test_get_chunks_by_document_reads_temporal_table(engine: str) -> None:
    """skeleton / vectorcypher write khora_chunks; the read-side fallback finds them."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine=engine) as kb:
        ns = await kb.create_namespace()
        result = await _remember(kb, ns.namespace_id)
        assert result.chunks_created >= 1

        # ``get_chunks_by_document`` filters on the resolved (row-level)
        # namespace id, the same id the ingest path stamps onto chunks.
        resolved = await kb.storage.resolve_namespace(ns.namespace_id)
        chunks = await kb.storage.get_chunks_by_document(result.document_id, namespace_id=resolved)

        assert len(chunks) >= 1, f"{engine}: expected temporal chunks, got {len(chunks)}"
        chunk = chunks[0]
        assert chunk.document_id == result.document_id
        assert chunk.content == _CONTENT


async def test_get_chunks_by_document_chronicle_uses_chunks_table() -> None:
    """chronicle writes the ``chunks`` table directly and never hits the fallback."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="chronicle") as kb:
        ns = await kb.create_namespace()
        result = await _remember(kb, ns.namespace_id)
        assert result.chunks_created >= 1

        resolved = await kb.storage.resolve_namespace(ns.namespace_id)
        chunks = await kb.storage.get_chunks_by_document(result.document_id, namespace_id=resolved)

        assert len(chunks) >= 1
        assert chunks[0].content == _CONTENT
