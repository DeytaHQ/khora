"""Cross-channel recency-visibility contract for first-class ``occurred_at``.

The recall-path recency reader (``VectorCypherRetriever._extract_occurred_at``
/ ``_calculate_recency_scores``) now reads the first-class ``Chunk.occurred_at``
column instead of the ``metadata['occurred_at']`` blob. This test pins the
consequence for the non-vector channels: chunks surfaced through the **BM25**
lexical channel and the **keyword_ppr** channel are hydrated straight from
storage (``search_fulltext`` / ``get_chunks_batch``), so they must carry a
populated first-class ``occurred_at`` and be recency-scored on the same footing
as vector-channel chunks - not stranded at the 0.5 "missing date" default.

Before the reader flip a channel that surfaced chunks without stamping the
``metadata`` blob would have been recency-invisible; this exercises the real
embedded storage round-trip + the real channels + the real recency scorer to
prove they now line up.

The entity-PPR channel (``ppr_retrieval.ppr_retrieve_chunks``, #1492 §(b)) runs
PageRank over the graph and is not exercisable on the embedded stack, so it is
covered compositionally rather than driven here: stage-1 re-wrap tests pin that
its re-wrapped chunks carry the first-class ``occurred_at`` / ``source_timestamp``
fields, and this PR's reader/scorer tests pin that ``_calculate_recency_scores``
reads that column - the two compose to the same recency visibility proven here
for the keyword_ppr channel.

Hermetic - deterministic fake embeddings, real SQLite + LanceDB in tmp_path.
Mirrors ``test_keyword_ppr_channel.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, Document, MemoryNamespace
from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.keyword_edges import persist_keyword_chunk_edges
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever
from khora.extraction.tokenize import tokenize_multilingual
from khora.query.keyword_ppr import keyword_ppr_retrieve_chunks
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator, fake_embedding

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


def _recency_retriever() -> VectorCypherRetriever:
    """A retriever whose only exercised surface is ``_calculate_recency_scores``.

    ``temporal_reference_wall_clock=True`` anchors recency to ``now`` so a
    genuinely old chunk scores near 0 (not 1.0 via the relative newest-in-set
    anchor) - the discriminating reference for "recency-visible".
    """
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(temporal_reference_wall_clock=True, recency_decay_days=30),
    )


async def test_bm25_and_keyword_ppr_chunks_are_recency_visible(tmp_path: Path) -> None:
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        ns_id = await coord.resolve_namespace(ns.namespace_id)

        now = datetime.now(UTC)
        fresh_ts = now - timedelta(days=2)
        stale_ts = now - timedelta(days=400)

        # Both chunks mention "photosynthesis" so the lexical channels surface
        # both; their occurred_at columns are far apart so recency must separate
        # them. The metadata blob is deliberately left empty - the reader must
        # rely on the first-class column alone.
        doc = Document(namespace_id=ns_id, content="bio", external_id="doc-0", title="bio")
        await coord.create_document(doc)
        fresh = Chunk(
            namespace_id=ns_id,
            document_id=doc.id,
            content="photosynthesis converts sunlight into chemical energy in plants",
            chunk_index=0,
            embedding=fake_embedding("fresh"),
            embedding_model="fake",
            occurred_at=fresh_ts,
        )
        stale = Chunk(
            namespace_id=ns_id,
            document_id=doc.id,
            content="photosynthesis was catalogued by early botanists long ago",
            chunk_index=1,
            embedding=fake_embedding("stale"),
            embedding_model="fake",
            occurred_at=stale_ts,
        )
        await coord.create_chunks_batch([fresh, stale])

        # --- keyword_ppr channel: rank -> hydrate via get_chunks_batch --------
        await persist_keyword_chunk_edges(coord, ns_id, [fresh, stale])
        ranked = await keyword_ppr_retrieve_chunks(
            coord,
            ns_id,
            "tell me about photosynthesis",
            tokenizer=tokenize_multilingual,
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            limit=10,
            max_edges=50_000,
        )
        assert ranked, "keyword_ppr channel returned no chunks"
        ppr_map = await coord.get_chunks_batch([cid for cid, _ in ranked], namespace_id=ns_id)
        ppr_chunks = {c.content: c for c in ppr_map.values()}
        assert set(ppr_chunks) == {fresh.content, stale.content}
        # The channel-hydrated chunks carry the first-class column (blob empty).
        for c in ppr_chunks.values():
            assert c.occurred_at is not None
            assert (c.metadata or {}).get("occurred_at") is None

        # --- bm25 channel: search_fulltext_chunks -> (Chunk, score) -----------
        bm25 = await coord.search_fulltext_chunks(ns_id, "photosynthesis", limit=10)
        bm25_chunks = {c.content: c for c, _score in bm25}
        assert set(bm25_chunks) == {fresh.content, stale.content}
        for c in bm25_chunks.values():
            assert c.occurred_at is not None

        # --- recency scorer reads the column for BOTH channels' chunks --------
        retriever = _recency_retriever()
        for label, chunk_by_content in (("keyword_ppr", ppr_chunks), ("bm25", bm25_chunks)):
            fresh_c = chunk_by_content[fresh.content]
            stale_c = chunk_by_content[stale.content]
            scores = retriever._calculate_recency_scores(
                [
                    FusedResult(item_id=fresh_c.id, item=fresh_c, rrf_score=0.5),
                    FusedResult(item_id=stale_c.id, item=stale_c, rrf_score=0.5),
                ]
            )
            # Fresh scores near 1.0, stale near 0.0. Both are strictly on either
            # side of the 0.5 missing-date default, proving neither channel's
            # chunk fell through to "no occurred_at".
            assert scores[fresh_c.id] > 0.5 > scores[stale_c.id], f"{label}: {scores}"
    finally:
        await coord.disconnect()
