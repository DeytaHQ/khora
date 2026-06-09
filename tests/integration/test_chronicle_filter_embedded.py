"""End-to-end recall-filter tests for the Chronicle engine over a real embedded store.

Exercises the deterministic recall filter against a live SQLite + LanceDB pair in
``tmp_path`` — no storage-layer mocks. This complements the mocked engine-composition
unit tests (``tests/recall/test_chronicle_filter_composition.py``) by driving the
filter through real persistence, which is where the storage-coupled parts most likely
break:

* **Metadata serialization round-trip** — chunk ``metadata`` is written to and read
  back from the real ``chunks.metadata`` column, so a ``metadata.<path>`` predicate is
  evaluated against a genuinely round-tripped blob, not a hand-built dict.
* **Date columns** — ``source_timestamp`` pushes down to the real recency window
  (``COALESCE(source_timestamp, created_at)``) at the SQL layer, and ``occurred_at`` is
  enforced by the post-filter against the chunk read back from storage. The embedded
  sqlite_lance write path now persists a distinct ``occurred_at`` column (migration
  ``046``), so a chunk's effective event time ``COALESCE(occurred_at, source_timestamp)``
  honors an in-range ``occurred_at`` even when ``source_timestamp`` is out of range, and
  falls back to ``source_timestamp`` when ``occurred_at`` is unset.

Seeding goes through the coordinator's own write API (``create_chunks_batch``) with
deterministic fake embeddings, exactly like the sibling sqlite_lance ingest suite, so
the suite stays hermetic (no LLM, no network). All seed chunks share one embedding so
the vector channel returns the whole seed set and the filter is the only narrowing
force.

Cross-compiler parity (the Chronicle path agreeing with the in-process
``compile_python`` oracle for the same filter) is asserted in
``test_engine_recall_matches_compile_python_oracle``.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from khora.config.schema import QuerySettings
from khora.core.models import Chunk, Document, MemoryNamespace
from khora.engines.chronicle.engine import ChronicleEngine
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.compilers.python import compile_python
from khora.filter.context import CompileContext
from khora.query import SearchMode
from tests.integration._sqlite_lance_fixtures import (
    EMBED_DIM,
    build_sqlite_lance_coordinator,
    fake_embedding,
)

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

# One shared embedding so every seed chunk matches the query equally — the vector
# channel returns the whole seed set, leaving the filter as the only narrowing force.
_QUERY_TEXT = "shared retrieval anchor"
_SHARED_EMBEDDING = fake_embedding(_QUERY_TEXT, dim=EMBED_DIM)

_IN_RANGE = datetime(2026, 6, 1, tzinfo=UTC)
_OUT_OF_RANGE = datetime(2020, 1, 1, tzinfo=UTC)
_FILTER_LB = "2026-01-01T00:00:00Z"


def _filter_ast(wire: dict) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _seed(
    coord: Any,
    namespace_id: UUID,
    specs: list[dict[str, Any]],
) -> list[Chunk]:
    """Insert one document + one chunk per spec via the real coordinator write API.

    Each ``spec`` carries ``content`` plus any of ``metadata`` / ``source_timestamp`` /
    ``occurred_at`` / ``created_at``. All chunks share ``_SHARED_EMBEDDING`` so the
    vector channel returns them all.
    """
    chunks: list[Chunk] = []
    for spec in specs:
        doc = Document(
            namespace_id=namespace_id,
            content=spec["content"],
            source="test",
            title=spec["content"][:32],
        )
        await coord.create_document(doc)
        chunk_kwargs: dict[str, Any] = {
            "namespace_id": namespace_id,
            "document_id": doc.id,
            "content": spec["content"],
            "chunk_index": 0,
            "embedding": _SHARED_EMBEDDING,
            "embedding_model": "fake",
            "metadata": spec.get("metadata", {}),
        }
        for date_key in ("source_timestamp", "occurred_at", "created_at"):
            if date_key in spec:
                chunk_kwargs[date_key] = spec[date_key]
        chunks.append(Chunk(**chunk_kwargs))
    await coord.create_chunks_batch(chunks)
    return chunks


def _engine_over(coord: Any) -> ChronicleEngine:
    """A ChronicleEngine bound to the real embedded coordinator.

    Reranking is disabled (it would lazily pull a cross-encoder on first recall);
    it only reorders candidates, never adds/drops a row, so the filter contract is
    unaffected. The embedder returns the shared query embedding so the vector channel
    retrieves the whole seed set.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))
    engine._storage = coord
    embedder = _FakeEmbedder()
    engine._embedder = embedder
    engine._connected = True
    return engine


class _FakeEmbedder:
    async def embed(self, _text: str) -> list[float]:
        return _SHARED_EMBEDDING


async def _recall_ids(engine: ChronicleEngine, namespace_id: UUID, wire: dict) -> set[UUID]:
    result = await engine.recall(
        _QUERY_TEXT,
        namespace_id,
        limit=50,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast(wire),
    )
    return {c.id for c in result.chunks}


@pytest.mark.asyncio
async def test_metadata_filter_round_trips_through_real_store(tmp_path: Path) -> None:
    # metadata.tier == "gold" must select exactly the gold chunks after a real
    # write→read round-trip of the metadata blob.
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        chunks = await _seed(
            coord,
            ns.id,
            [
                {"content": "gold one", "metadata": {"tier": "gold"}},
                {"content": "gold two", "metadata": {"tier": "gold"}},
                {"content": "silver one", "metadata": {"tier": "silver"}},
                {"content": "no tier", "metadata": {}},
            ],
        )
        gold_ids = {chunks[0].id, chunks[1].id}

        returned = await _recall_ids(_engine_over(coord), ns.id, {"metadata.tier": "gold"})
        assert returned == gold_ids
    finally:
        await coord.disconnect()


@pytest.mark.asyncio
async def test_source_timestamp_pushdown_narrows_against_real_window(tmp_path: Path) -> None:
    # source_timestamp >= 2026-01-01 pushes down to the real recency window; only the
    # in-range chunk survives.
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        chunks = await _seed(
            coord,
            ns.id,
            [
                {"content": "recent", "source_timestamp": _IN_RANGE},
                {"content": "ancient", "source_timestamp": _OUT_OF_RANGE},
            ],
        )
        in_range_id = chunks[0].id

        returned = await _recall_ids(_engine_over(coord), ns.id, {"source_timestamp": {"$gte": _FILTER_LB}})
        assert returned == {in_range_id}
    finally:
        await coord.disconnect()


@pytest.mark.asyncio
async def test_occurred_at_coalesce_recovery_against_real_columns(tmp_path: Path) -> None:
    # occurred_at is post-filtered against the chunk read back from storage, using
    # the effective event time COALESCE(occurred_at, source_timestamp). The embedded
    # sqlite_lance write path now persists a distinct occurred_at column (migration
    # 046 → create_chunks_batch / _row_to_chunk in sqlite_lance/vector.py), so this
    # asserts two halves of the contract against what the store actually round-trips:
    #
    #   1. HONORED: a chunk whose occurred_at is in range but whose source_timestamp
    #      is out of range still SURVIVES — the effective event time resolves to the
    #      in-range occurred_at, NOT the out-of-range source_timestamp. This is the
    #      regression guard for the persist-occurred_at fix: if the write path dropped
    #      occurred_at (read back as NULL), COALESCE would collapse to the out-of-range
    #      source_timestamp and this chunk would be (wrongly) filtered out.
    #   2. FALLBACK: a chunk with NO occurred_at but an in-range source_timestamp still
    #      SURVIVES — COALESCE recovers via source_timestamp (no false-empty).
    #
    # A chunk with neither anchor in range is dropped (negative case).
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        chunks = await _seed(
            coord,
            ns.id,
            [
                # occurred_at in range, source_timestamp out of range → effective event
                # time honors the persisted occurred_at → survives. Drops to a false
                # negative if occurred_at is not round-tripped (the regression guard).
                {"content": "occurred honored", "occurred_at": _IN_RANGE, "source_timestamp": _OUT_OF_RANGE},
                # no occurred_at, source_timestamp in range → COALESCE recovers via
                # source_timestamp → survives (proves NO false-empty when occurred_at
                # is unset).
                {"content": "fallback recover", "source_timestamp": _IN_RANGE},
                # neither anchor in range → dropped.
                {"content": "no anchor in range", "source_timestamp": _OUT_OF_RANGE},
            ],
        )
        honored_id = chunks[0].id
        fallback_id = chunks[1].id

        returned = await _recall_ids(_engine_over(coord), ns.id, {"occurred_at": {"$gte": _FILTER_LB}})
        assert returned == {honored_id, fallback_id}, (
            "occurred_at filter must (1) honor a persisted in-range occurred_at even when "
            "source_timestamp is out of range, and (2) recover event time from "
            "source_timestamp when occurred_at is unset (no false-empty); rows whose "
            "effective event time is out of range are dropped"
        )
        # Explicit regression guard: the honored chunk would be dropped if the write
        # path failed to round-trip occurred_at (effective time would fall back to its
        # out-of-range source_timestamp). Assert it survives on its own merits.
        assert honored_id in returned, (
            "chunk with in-range occurred_at + out-of-range source_timestamp must survive "
            "— a regression in occurred_at persistence would drop it"
        )
    finally:
        await coord.disconnect()


@pytest.mark.asyncio
async def test_occurred_at_round_trips_through_real_store(tmp_path: Path) -> None:
    # Direct write→read round-trip of the distinct occurred_at column through the real
    # coordinator/vector adapter (no filter, no engine). A chunk seeded with an
    # occurred_at that differs from both created_at and source_timestamp must read back
    # with that exact occurred_at — proving migration 046 + create_chunks_batch /
    # _row_to_chunk persist and restore the column, not silently coalesce it away.
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        chunks = await _seed(
            coord,
            ns.id,
            [
                {
                    "content": "distinct occurred_at",
                    "occurred_at": _IN_RANGE,
                    "source_timestamp": _OUT_OF_RANGE,
                },
            ],
        )
        written = chunks[0]
        assert written.occurred_at == _IN_RANGE  # sanity: seeded as expected

        read_back = await coord.get_chunk(written.id, namespace_id=ns.id)
        assert read_back is not None
        assert read_back.occurred_at == _IN_RANGE, (
            "occurred_at must round-trip through the real store unchanged "
            f"(wrote {written.occurred_at!r}, read back {read_back.occurred_at!r})"
        )
        # source_timestamp stays distinct — occurred_at is not derived from it.
        assert read_back.source_timestamp == _OUT_OF_RANGE
        assert read_back.occurred_at != read_back.source_timestamp
    finally:
        await coord.disconnect()


@pytest.mark.asyncio
async def test_engine_recall_matches_compile_python_oracle(tmp_path: Path) -> None:
    # Cross-compiler parity: the Chronicle engine's real-storage recall returns the
    # SAME surviving chunk IDs as the in-process compile_python oracle applied to the
    # same seed, for the same composed filter. The oracle is the reference all
    # per-backend compilers must agree with.
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        specs = [
            {"content": "gold recent", "metadata": {"tier": "gold"}, "source_timestamp": _IN_RANGE},
            {"content": "gold ancient", "metadata": {"tier": "gold"}, "source_timestamp": _OUT_OF_RANGE},
            {"content": "silver recent", "metadata": {"tier": "silver"}, "source_timestamp": _IN_RANGE},
        ]
        chunks = await _seed(coord, ns.id, specs)

        wire = {"metadata.tier": "gold", "source_timestamp": {"$gte": _FILTER_LB}}
        ast = _filter_ast(wire)

        # Oracle: apply compile_python directly to the seed records.
        predicate = compile_python(ast, CompileContext(backend_target="chunks")).predicate
        oracle_ids = {
            c.id
            for c in chunks
            if predicate(
                {
                    "metadata": c.metadata,
                    "source_timestamp": c.source_timestamp,
                    "occurred_at": c.occurred_at if c.occurred_at is not None else c.source_timestamp,
                    "created_at": c.created_at,
                }
            )
        }

        engine_ids = await _recall_ids(_engine_over(coord), ns.id, wire)

        assert engine_ids == oracle_ids
        # Sanity: exactly the gold + in-range chunk.
        assert engine_ids == {chunks[0].id}
    finally:
        await coord.disconnect()
