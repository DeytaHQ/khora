"""Skeleton SQLite + LanceDB integration tests (DYT-3545).

Mirrors ``tests/integration/matrix/test_skeleton_pg.py`` (PR #474) for the
embedded stack. Skeleton has no graph component, so the embedded subset is
**SQLite (relational + FTS5 + graph adapter) + LanceDB (vectors)**.

How the LLM is mocked:
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return content-derived unit
  vectors. The test rig uses a 32-dim embedding so LanceDB indexes stay
  fast in tmpdir; ``config.llm.embedding_dimension`` and
  ``config.storage.embedding_dimension`` are aligned to 32.
* Skeleton does **no** entity extraction, so no extractor stub is needed.

How to run locally::

    uv run pytest tests/integration/matrix/test_skeleton_sqlite_lance.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.engines.skeleton.backends import TemporalFilter
from khora.memory_lake import MemoryLake
from khora.query import SearchMode

EMBED_DIM = 32  # small dim keeps LanceDB index build cheap in tmp_path


# ---------------------------------------------------------------------------
# Fixtures: skip-if-no-embedded-deps, deterministic embedder
# ---------------------------------------------------------------------------


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# Keyword vocabulary: deterministic, content-aware embeddings. Slot count
# stays well below EMBED_DIM=32 so we never collide.
_KEYWORD_SLOTS: dict[str, int] = {
    "alpha": 0,
    "bravo": 1,
    "charlie": 2,
    "delta": 3,
    "echo": 4,
    "kangaroo": 5,
    "kangaroos": 5,
    "penguin": 6,
    "penguins": 6,
    "widget": 7,
    "falcon": 8,
    "launch": 9,
    "rocket": 10,
    "tag": 11,
    "metadata": 12,
    "filter": 13,
    "concurrent": 14,
    "batch": 15,
    "bulk": 16,
    "recent": 17,
    "old": 18,
    "first": 19,
    "second": 20,
    "third": 21,
    "fourth": 22,
    "fifth": 23,
    "animals": 24,
    "document": 25,
    "group": 26,
}


def _embed_for(text_in: str) -> list[float]:
    """Deterministic 32-dim unit vector derived from ``text_in``.

    Small constant baseline component so the all-zero edge case (a query
    that matches no vocabulary) still gets a defined vector.
    """
    vec = [0.0] * EMBED_DIM
    vec[EMBED_DIM - 1] = 0.01

    lower = text_in.lower()
    for kw, slot in _KEYWORD_SLOTS.items():
        if kw in lower:
            vec[slot] = 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the embedder so no real LLM is called.

    Skeleton does no entity extraction, so the extractor doesn't need a stub.
    """
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


@pytest.fixture
async def lake(tmp_path: Path) -> AsyncIterator[MemoryLake]:
    """Per-test Skeleton MemoryLake bound to an embedded SQLite+LanceDB pair.

    The fixture allocates a fresh tmp_path per test; ``run_migrations=True``
    builds the alembic schema in the SQLite file before the coordinator
    opens it (mirrors :func:`build_sqlite_lance_coordinator` in
    ``tests/integration/_sqlite_lance_fixtures.py``).
    """
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=db_path,
        lance_path=lance_path,
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024

    lake = MemoryLake(config, engine="skeleton", run_migrations=True)
    await lake.connect()
    try:
        yield lake
    finally:
        await lake.disconnect()


@pytest.fixture
async def namespace_id(lake: MemoryLake) -> UUID:
    ns = await lake.create_namespace()
    return ns.namespace_id


async def _remember(
    lake: MemoryLake,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> Any:
    return await lake.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        metadata=metadata,
        # Skeleton accepts these for protocol compliance but doesn't use them.
        entity_types=["PERSON", "CONCEPT"],
        relationship_types=["RELATES_TO"],
    )


async def _recall(lake: MemoryLake, query: str, **kwargs: Any) -> Any:
    """Recall wrapper that pins ``mode=SearchMode.VECTOR`` for deterministic ranking.

    Mirrors ``_recall`` in ``test_skeleton_pg.py``: the wrapper sidesteps
    BM25 weighting under HYBRID so top-k ordering tests aren't affected by
    the blend weight. ``test_skeleton_recall_default_hybrid_mode`` exercises
    the default-HYBRID path explicitly (DYT-3555).
    """
    kwargs.setdefault("mode", SearchMode.VECTOR)
    return await lake.recall(query, **kwargs)


# ---------------------------------------------------------------------------
# Tests — mirror test_skeleton_pg.py one-for-one
# ---------------------------------------------------------------------------


async def test_skeleton_remember_recall_roundtrip(lake: MemoryLake, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in context."""
    contents = [
        "alpha document mentions the falcon launch in detail.",
        "bravo document covers a different rocket programme entirely.",
        "charlie document is a side note unrelated to anything else.",
    ]
    for c in contents:
        await _remember(lake, namespace_id=namespace_id, content=c)

    result = await _recall(lake, "falcon launch", namespace=namespace_id, limit=10)

    # Skeleton reports its backend type in metadata. Once DYT-3561 lands,
    # the value should be "lancedb" (or whatever name the new backend
    # registers under) — assert non-empty rather than pin the string.
    assert result.metadata.get("backend") is not None
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    assert "falcon" in result.context_text.lower()


async def test_skeleton_namespace_isolation(lake: MemoryLake) -> None:
    """Two namespaces, queries don't cross-bleed."""
    ns_a = (await lake.create_namespace()).namespace_id
    ns_b = (await lake.create_namespace()).namespace_id

    await _remember(lake, namespace_id=ns_a, content="alpha document about kangaroos in the outback.")
    await _remember(lake, namespace_id=ns_b, content="bravo document about penguins on the ice.")

    result_a = await _recall(lake, "animals", namespace=ns_a, limit=10)
    result_b = await _recall(lake, "animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c, _ in result_a.chunks)
    b_text = " ".join(c.content for c, _ in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b content leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a content leaked into namespace_b"


async def test_skeleton_recall_top_k_ordering(lake: MemoryLake, namespace_id: UUID) -> None:
    """Results ordered by descending similarity (combined_score)."""
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha bravo charlie delta echo (high overlap with query)",
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha bravo charlie (medium overlap with query)",
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha (low overlap with query)",
    )

    result = await _recall(lake, "alpha bravo charlie delta echo", namespace=namespace_id, limit=10)

    assert len(result.chunks) >= 3
    scores = [score for _, score in result.chunks]
    for prev, curr in zip(scores, scores[1:]):
        assert prev >= curr, f"similarity ordering violated: {prev} < {curr} in {scores}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Engine-level namespace resolution gap (out of scope for DYT-3561): "
        "the test fixture passes the stable ``ns.namespace_id`` straight to "
        "``engine.recall`` while ``lake.remember`` resolves it to the "
        "row-level ``id`` before persisting. ``khora_chunks.namespace_id`` "
        "therefore stores the row-level id and the recall filter misses. "
        "Same issue lurks behind PG's DYT-3556 xfail; needs a separate "
        "ticket to either resolve at the engine boundary or update the "
        "test contract."
    ),
)
async def test_skeleton_recall_with_metadata_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Tag filter restricts recall to chunks carrying the requested tag.

    Unlike the PG sibling, the embedded path doesn't hit DYT-3556 (SQLite
    serializes ``tags`` as JSON-text — no ``ARRAY(String).contains``
    incompatibility). The current xfail tracks a separate engine-layer
    namespace-resolution gap surfaced by this test bypassing
    :meth:`MemoryLake.recall` and calling ``engine.recall`` directly.
    """
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha document tagged group A",
        metadata={"tags": ["group-A"]},
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha document tagged group B",
        metadata={"tags": ["group-B"]},
    )

    engine = lake._get_engine()  # type: ignore[attr-defined]
    result = await engine.recall(
        "alpha document",
        namespace_id,
        limit=10,
        temporal_filter=TemporalFilter(tags=["group-A"]),
        hybrid_alpha=1.0,
    )

    assert len(result.chunks) >= 1
    for chunk, _score in result.chunks:
        assert "group A" in chunk.content, f"non-group-A leaked in: {chunk.content!r}"
    assert all("group B" not in c.content for c, _ in result.chunks), "group-B chunk leaked through tag filter"


async def test_skeleton_temporal_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Two docs 5d vs 20d apart, "last 7 days" → recent only.

    Skeleton's single-doc ``remember()`` ignores ``metadata['occurred_at']``
    (DYT-3557 — only ``remember_batch`` reads it). To dodge that here we
    use ``remember_batch``, which reads the field and passes it through to
    ``TemporalChunk.occurred_at``. Mirrors the dodge documented in
    ``test_skeleton_pg.py`` but avoids the SQL-update workaround since
    backdating chunks via direct SQL would couple this test to the
    sqlite_lance internals.
    """
    now = datetime.now(UTC)
    ts_recent = (now - timedelta(days=5)).isoformat()
    ts_old = (now - timedelta(days=20)).isoformat()

    batch = await lake.remember_batch(
        [
            {
                "content": "recent document about the falcon launch",
                "title": "recent",
                "metadata": {"occurred_at": ts_recent},
            },
            {
                "content": "old document about the falcon launch",
                "title": "old",
                "metadata": {"occurred_at": ts_old},
            },
        ],
        namespace=namespace_id,
        entity_types=["PERSON"],
        relationship_types=["RELATES_TO"],
    )
    assert batch.processed == 2
    assert batch.failed == 0

    seven_days_ago = now - timedelta(days=7)
    result = await _recall(
        lake,
        "falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    contents = [c.content for c, _ in result.chunks]
    assert any("recent" in c for c in contents), f"recent doc not returned: {contents}"
    assert not any("old document" in c for c in contents), (
        f"20-day-old document leaked through occurred_after filter: {contents}"
    )


async def test_skeleton_remember_batch(lake: MemoryLake, namespace_id: UUID) -> None:
    """Bulk-ingest 20 docs in a single ``remember_batch`` call."""
    documents = [
        {
            "content": f"batch document number {i} contains widget-{i} content",
            "title": f"doc-{i}",
        }
        for i in range(20)
    ]
    batch = await lake.remember_batch(
        documents,
        namespace=namespace_id,
        entity_types=["PERSON"],
        relationship_types=["RELATES_TO"],
    )

    assert batch.processed == 20, f"expected 20 processed, got {batch}"
    assert batch.failed == 0, f"unexpected failures: {batch}"
    assert batch.chunks >= 20, f"expected ≥20 chunks (one per doc), got {batch.chunks}"

    result = await _recall(lake, "widget batch document", namespace=namespace_id, limit=25)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 20, f"expected ≥20 distinct chunks returned, got {len(contents_returned)}"


async def test_skeleton_recall_empty_namespace(lake: MemoryLake) -> None:
    """Recall against an empty namespace returns an empty chunks list."""
    ns = (await lake.create_namespace()).namespace_id

    result = await _recall(lake, "anything at all", namespace=ns, limit=10)

    assert result.chunks == []
    assert result.entities == []  # Skeleton never returns entities anyway.
    assert result.metadata.get("backend") is not None


async def test_skeleton_recall_metadata_keys(lake: MemoryLake, namespace_id: UUID) -> None:
    """RecallResult.metadata exposes the keys the Skeleton engine documents."""
    await _remember(lake, namespace_id=namespace_id, content="alpha simple sentence")

    result = await _recall(lake, "alpha", namespace=namespace_id, limit=5)

    md = result.metadata
    expected = {"backend", "hybrid_alpha", "temporal_filter"}
    missing = expected - md.keys()
    assert not missing, f"missing skeleton metadata keys: {missing}"
    assert md["hybrid_alpha"] == 1.0
    assert md["temporal_filter"] is None


async def test_skeleton_recall_default_hybrid_mode(lake: MemoryLake, namespace_id: UUID) -> None:
    """Default ``MemoryLake.recall(...)`` works on HYBRID — DYT-3555 regression.

    Pre-fix, ``SkeletonConstructionEngine.recall`` referenced a non-existent
    ``SearchMode.KEYWORD`` member, crashing on default HYBRID. DYT-3555
    added the enum member; this test exercises the default path on the
    embedded backend so the regression coverage extends past PG.
    """
    await _remember(lake, namespace_id=namespace_id, content="alpha simple sentence")
    result = await lake.recall("alpha", namespace=namespace_id, limit=5)

    assert result.metadata.get("backend") is not None
    # HYBRID maps to ``hybrid_alpha=0.7`` per engine.py:444.
    assert result.metadata.get("hybrid_alpha") == 0.7


async def test_skeleton_concurrent_remember(lake: MemoryLake, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(lake, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"

    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"

    result = await _recall(lake, "widget", namespace=namespace_id, limit=20)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 5
