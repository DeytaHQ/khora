"""End-to-end doc_title hydration through the rerank stage.

`hydrate_doc_titles` is the helper that both `HybridQueryEngine` and
`VectorCypherRetriever` call before invoking a reranker — it issues a
single batched fetch against `StorageCoordinator.get_document_sources_batch`
and mutates each candidate's `doc_title` in place. This integration test
exercises the full chain — real SQLite-backed coordinator, real document
rows, real batch fetch — for both candidate shapes the production code
passes in:

* `item=Chunk` with a direct `chunk.document_id` (the engine path).
* `item=FusedResult` whose `.item` is the chunk (the retriever path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, Document, MemoryNamespace
from khora.engines.vectorcypher.fusion import FusedResult
from khora.query.reranking import RerankCandidate, hydrate_doc_titles
from tests.integration._sqlite_lance_fixtures import (
    build_sqlite_lance_coordinator,
    fake_embedding,
)

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


async def _seed_two_titled_docs(coord, namespace_id) -> list[tuple[Document, Chunk]]:
    pairs: list[tuple[Document, Chunk]] = []
    for idx, title in enumerate(["Acme Q2 Review", "Globex Annual Report"]):
        doc = Document(
            namespace_id=namespace_id,
            content=f"Body for {title} — chunk {idx}.",
            external_id=f"doc-{idx}",
            source="test",
            title=title,
        )
        await coord.create_document(doc)
        chunk = Chunk(
            namespace_id=namespace_id,
            document_id=doc.id,
            content=doc.content,
            chunk_index=0,
            embedding=fake_embedding(doc.content),
            embedding_model="fake",
        )
        pairs.append((doc, chunk))
    await coord.create_chunks_batch([c for _, c in pairs])
    return pairs


class TestDocTitleHydration:
    """`hydrate_doc_titles` populates titles via one batched coordinator call."""

    async def test_hydrates_titles_for_chunk_and_fused_result_candidates(self, tmp_path: Path) -> None:
        """End-to-end: engine-shaped + retriever-shaped candidates both get titles.

        Builds candidates that mirror the two production call sites:
        * `HybridQueryEngine` reranking — `item` is the Chunk itself, so the
          extractor reads `chunk.document_id`.
        * `VectorCypherRetriever` reranking — `item` is a `FusedResult`, so
          the extractor reads `fr.item.document_id`.

        Both forms must end up with `doc_title` equal to the seeded
        `Document.title`, and a missing-document candidate must remain empty
        without raising.
        """
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            pairs = await _seed_two_titled_docs(coord, ns.id)
            (doc_a, chunk_a), (doc_b, chunk_b) = pairs

            # Engine-shape: item=Chunk; extractor reads chunk.document_id.
            engine_candidates: list[RerankCandidate] = [
                RerankCandidate(item=chunk_a, original_score=0.9, content=chunk_a.content),
                RerankCandidate(item=chunk_b, original_score=0.8, content=chunk_b.content),
            ]
            assert all(c.doc_title == "" for c in engine_candidates)

            await hydrate_doc_titles(
                engine_candidates,
                coord,
                lambda c: getattr(c, "document_id", None),
                namespace_id=ns.id,
            )

            assert engine_candidates[0].doc_title == doc_a.title
            assert engine_candidates[1].doc_title == doc_b.title

            # Retriever-shape: item=FusedResult wrapping the chunk; extractor
            # reaches through fr.item.document_id. Also include one candidate
            # whose document was never persisted — hydration must leave its
            # title untouched ("") and must not raise.
            from uuid import uuid4

            missing_chunk = Chunk(
                namespace_id=ns.id,
                document_id=uuid4(),
                content="orphan",
                chunk_index=0,
                embedding=fake_embedding("orphan"),
                embedding_model="fake",
            )
            fused_a = FusedResult(item_id=chunk_a.id, item=chunk_a, rrf_score=0.5)
            fused_b = FusedResult(item_id=chunk_b.id, item=chunk_b, rrf_score=0.4)
            fused_missing = FusedResult(item_id=missing_chunk.id, item=missing_chunk, rrf_score=0.3)
            retriever_candidates: list[RerankCandidate] = [
                RerankCandidate(item=fused_a, original_score=0.9, content=chunk_a.content),
                RerankCandidate(item=fused_b, original_score=0.8, content=chunk_b.content),
                RerankCandidate(item=fused_missing, original_score=0.7, content=missing_chunk.content),
            ]

            await hydrate_doc_titles(
                retriever_candidates,
                coord,
                lambda fr: getattr(getattr(fr, "item", None), "document_id", None),
                namespace_id=ns.id,
            )

            assert retriever_candidates[0].doc_title == doc_a.title
            assert retriever_candidates[1].doc_title == doc_b.title
            # Missing document: fallback to empty, no exception.
            assert retriever_candidates[2].doc_title == ""
        finally:
            await coord.disconnect()
