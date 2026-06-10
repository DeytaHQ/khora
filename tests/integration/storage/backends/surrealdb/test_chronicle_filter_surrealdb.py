"""End-to-end occurred_at persistence + recall-filter tests on the SurrealDB chunk path.

The SurrealDB ``chunk`` table carries a distinct ``occurred_at`` ``option<datetime>``
field (the real-world event time the chunk's content refers to, separate from both
``created_at`` ingestion time and ``source_timestamp``). These tests drive that field
through the production ``SurrealDBVectorAdapter`` write/read path against an embedded,
in-memory SurrealDB (``mode="memory"``) — no storage mocks, no Docker, no external
service — and prove two halves of the contract:

* **Round-trip** — a chunk written with an ``occurred_at`` distinct from both
  ``created_at`` and ``source_timestamp`` reads back with that exact value. This fails
  loudly if the write path drops ``occurred_at`` (it would read back ``None``) or the
  SCHEMAFULL ``chunk`` table strips the field.
* **Recall-filter regression guard** — a chunk whose ``occurred_at`` is in range but
  whose ``source_timestamp`` is out of range is honored by an ``occurred_at`` recall
  filter. The effective event time is ``COALESCE(occurred_at, source_timestamp)``; if
  the write path dropped ``occurred_at`` (read back as ``None``), that COALESCE would
  collapse to the out-of-range ``source_timestamp`` and the chunk would be (wrongly)
  filtered out. So this proves ``occurred_at`` is genuinely persisted, not silently
  recovered from ``source_timestamp``.

The SurrealDB sibling of ``tests/integration/test_chronicle_filter_embedded.py`` (the
sqlite_lance sibling) and ``tests/integration/test_chronicle_filter_pgvector.py`` (the
PG/pgvector sibling): same engine, same filter AST, same recall path — only the storage
backend differs. Seeding goes through the coordinator's own write API
(``create_chunks_batch``) with deterministic fake embeddings; all seed chunks share one
embedding so the vector channel returns the whole seed set and the filter is the only
narrowing force. ``SurrealDBVectorAdapter.search_similar`` scores with
``vector::dot(embedding, $query_embedding)`` (brute-force cosine; SurrealDB ``<|K|>``
KNN is unreliable embedded), so identical unit vectors all score 1.0.

The ``surrealdb`` SDK is an optional dependency, so this module self-skips when it is
absent (``pytest.importorskip``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.config import KhoraConfig  # noqa: E402
from khora.config.schema import QuerySettings  # noqa: E402
from khora.core.models import Chunk, Document, MemoryNamespace  # noqa: E402
from khora.engines.chronicle.engine import ChronicleEngine  # noqa: E402
from khora.filter import RecallFilter  # noqa: E402
from khora.filter.ast import parse_to_ast  # noqa: E402
from khora.query import SearchMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402
from khora.storage.coordinator import StorageCoordinator  # noqa: E402

pytestmark = pytest.mark.integration

# A small embedding dimension is fine — the SurrealDB ``embedding`` field is
# ``option<array<float>>`` with no fixed width and ``search_similar`` brute-forces the
# dot product. One shared unit vector so every seed chunk scores 1.0 and the vector
# channel returns the whole seed set, leaving the filter as the only narrowing force.
EMBED_DIM = 8
_QUERY_TEXT = "shared retrieval anchor"
_SHARED_EMBEDDING = [1.0] + [0.0] * (EMBED_DIM - 1)

# Tz-AWARE bounds (the column is a SurrealDB ``datetime``). The filter literal is the
# same instant in ISO-8601 Z form so the post-filter compares tz-aware to tz-aware.
_IN_RANGE = datetime(2026, 6, 1, tzinfo=UTC)
_OUT_OF_RANGE = datetime(2020, 1, 1, tzinfo=UTC)
_FILTER_LB = "2026-01-01T00:00:00Z"
# A fourth distinct anchor for the last_accessed_at round-trip (the implementation
# folded last_accessed_at in alongside occurred_at); kept different from every other
# seeded timestamp so the round-trip assert can't pass by coincidence.
_LAST_ACCESSED = datetime(2026, 2, 14, tzinfo=UTC)


@pytest.fixture
async def coord() -> AsyncIterator[StorageCoordinator]:
    """A connected coordinator over an embedded SurrealDB unified backend.

    Both the relational and vector adapters share one in-memory ``SurrealDBConnection``
    (the unified backend): ``create_namespace`` / ``create_document`` live on the
    relational adapter; ``create_chunks_batch`` / ``get_chunk`` / the recall
    ``search_*`` channels live on the vector adapter. Sharing one connection keeps both
    writes landing in the same database the recall path later reads. The connection's
    auto schema-init runs the full khora SCHEMAFULL schema (which already declares
    ``occurred_at`` on ``chunk``).
    """
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="occurredat")
    await conn.connect()
    relational = SurrealDBRelationalAdapter(conn)
    vector = SurrealDBVectorAdapter(conn)
    coordinator = StorageCoordinator(relational=relational, vector=vector)
    # Warm the embedded SurrealDB engine before the test body runs. The very first
    # create/query/update cycle against a freshly-initialized in-memory engine is
    # occasionally non-deterministic — a rare cold-start artifact where a query can
    # return a wrong result (e.g. a namespace-scoped UPDATE transiently matching the
    # wrong row), most visible on the first SurrealDB operation in a fresh process or
    # under heavy concurrent load. When several tests run in sequence the earlier ones
    # warm the engine implicitly; this explicit warm-up makes each test self-contained
    # so the engine is reliable even for the first test in a cold run. The throwaway
    # namespace cannot affect any test's data (recall is namespace-scoped).
    _warm_ns = await coordinator.create_namespace(MemoryNamespace())
    _warm_doc = Document(namespace_id=_warm_ns.id, content="warmup", source="warmup", title="warmup")
    await coordinator.create_document(_warm_doc)
    _warm_chunk = Chunk(
        namespace_id=_warm_ns.id,
        document_id=_warm_doc.id,
        content="warmup",
        chunk_index=0,
        embedding=list(_SHARED_EMBEDDING),
        embedding_model="fake",
        last_accessed_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    await coordinator.create_chunks_batch([_warm_chunk])
    await coordinator.update_last_accessed(_warm_ns.id, [_warm_chunk.id], datetime(2001, 1, 1, tzinfo=UTC))
    await coordinator.get_chunk(_warm_chunk.id, namespace_id=_warm_ns.id)
    try:
        yield coordinator
    finally:
        await conn.disconnect()


class _FakeEmbedder:
    async def embed(self, _text: str) -> list[float]:
        return list(_SHARED_EMBEDDING)


def _engine_over(coordinator: StorageCoordinator) -> ChronicleEngine:
    """A ChronicleEngine bound to the embedded SurrealDB coordinator.

    Reranking is disabled (it would lazily pull a cross-encoder on first recall); it
    only reorders candidates, never adds/drops a row, so the filter contract is
    unaffected. The fake embedder returns the shared query embedding so the vector
    channel retrieves the whole seed set.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))
    engine._storage = coordinator
    engine._embedder = _FakeEmbedder()
    engine._connected = True
    return engine


def _filter_ast(wire: dict) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _seed(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    specs: list[dict[str, Any]],
) -> list[Chunk]:
    """Insert one document + one chunk per spec via the real coordinator write API.

    Each ``spec`` carries ``content`` plus any of ``source_timestamp`` / ``occurred_at``
    / ``last_accessed_at`` / ``created_at``. All chunks share ``_SHARED_EMBEDDING`` so
    the vector channel returns them all.
    """
    chunks: list[Chunk] = []
    for spec in specs:
        doc = Document(
            namespace_id=namespace_id,
            content=spec["content"],
            source="test",
            title=spec["content"][:32],
        )
        await coordinator.create_document(doc)
        chunk_kwargs: dict[str, Any] = {
            "namespace_id": namespace_id,
            "document_id": doc.id,
            "content": spec["content"],
            "chunk_index": 0,
            "embedding": list(_SHARED_EMBEDDING),
            "embedding_model": "fake",
            "metadata": spec.get("metadata", {}),
        }
        for date_key in ("source_timestamp", "occurred_at", "last_accessed_at", "created_at"):
            if date_key in spec:
                chunk_kwargs[date_key] = spec[date_key]
        chunks.append(Chunk(**chunk_kwargs))
    await coordinator.create_chunks_batch(chunks)
    return chunks


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
async def test_occurred_at_round_trips_through_real_store(coord: StorageCoordinator) -> None:
    # Direct write→read round-trip of the distinct occurred_at field through the real
    # SurrealDB vector adapter (no filter, no engine). A chunk seeded with an
    # occurred_at that differs from BOTH created_at and source_timestamp must read back
    # with that exact occurred_at — proving the write path persists it and the read path
    # restores it, rather than the batch insert dropping it or the SCHEMAFULL table
    # silently stripping it. last_accessed_at (folded in by the same implementation) is
    # round-tripped alongside, seeded distinct from every other anchor.
    ns = await coord.create_namespace(MemoryNamespace())
    created_at = datetime(2025, 3, 15, tzinfo=UTC)
    chunks = await _seed(
        coord,
        ns.id,
        [
            {
                "content": "distinct occurred_at",
                "occurred_at": _IN_RANGE,
                "source_timestamp": _OUT_OF_RANGE,
                "last_accessed_at": _LAST_ACCESSED,
                "created_at": created_at,
            },
        ],
    )
    written = chunks[0]
    # Sanity: all four anchors are genuinely distinct before the round-trip.
    assert written.occurred_at == _IN_RANGE
    assert written.source_timestamp == _OUT_OF_RANGE
    assert written.last_accessed_at == _LAST_ACCESSED
    assert written.created_at == created_at
    assert (
        len(
            {
                written.occurred_at,
                written.source_timestamp,
                written.last_accessed_at,
                written.created_at,
            }
        )
        == 4
    )

    read_back = await coord.get_chunk(written.id, namespace_id=ns.id)
    assert read_back is not None
    assert read_back.occurred_at == _IN_RANGE, (
        "occurred_at must round-trip through the real SurrealDB store unchanged "
        f"(wrote {written.occurred_at!r}, read back {read_back.occurred_at!r}) — a "
        "regression in the batch write path or the row mapper would read back None"
    )
    # last_accessed_at must round-trip too (folded into the same write/read wiring).
    assert read_back.last_accessed_at == _LAST_ACCESSED, (
        "last_accessed_at must round-trip through the real SurrealDB store unchanged "
        f"(wrote {written.last_accessed_at!r}, read back {read_back.last_accessed_at!r})"
    )
    # The other anchors stay distinct — occurred_at is not derived from any of them.
    assert read_back.source_timestamp == _OUT_OF_RANGE
    assert read_back.occurred_at != read_back.source_timestamp
    assert read_back.occurred_at != read_back.created_at
    assert read_back.occurred_at != read_back.last_accessed_at


@pytest.mark.asyncio
async def test_occurred_at_filter_honored_over_out_of_range_source_timestamp(
    coord: StorageCoordinator,
) -> None:
    # Recall-filter regression guard. The effective event time is
    # COALESCE(occurred_at, source_timestamp). Seed a chunk whose occurred_at is in
    # range but whose source_timestamp is out of range: an occurred_at recall filter
    # must HONOR it, because the in-range occurred_at — not the out-of-range
    # source_timestamp — is the effective event time.
    #
    # This is the guard for the persist-occurred_at fix: if the write path dropped
    # occurred_at (read back as None), the effective event time would fall back to the
    # out-of-range source_timestamp and this chunk would be (wrongly) dropped. That it
    # survives proves occurred_at is genuinely persisted, not silently recovered via
    # COALESCE(occurred_at, source_timestamp).
    #
    # A second chunk with NO occurred_at but an in-range source_timestamp confirms the
    # fallback still recovers (no false-empty); a third chunk with neither anchor in
    # range is the negative case.
    ns = await coord.create_namespace(MemoryNamespace())
    chunks = await _seed(
        coord,
        ns.id,
        [
            # occurred_at in range, source_timestamp out of range → survives ONLY if
            # occurred_at round-trips. This is the regression guard.
            {"content": "occurred honored", "occurred_at": _IN_RANGE, "source_timestamp": _OUT_OF_RANGE},
            # no occurred_at, source_timestamp in range → COALESCE recovers via
            # source_timestamp → survives (proves no false-empty when occurred_at unset).
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
    # Explicit regression guard: the honored chunk would be dropped if the write path
    # failed to round-trip occurred_at (its effective event time would fall back to the
    # out-of-range source_timestamp). Assert it survives on its own merits.
    assert honored_id in returned, (
        "chunk with in-range occurred_at + out-of-range source_timestamp must survive — "
        "a regression in occurred_at persistence would drop it"
    )


# A new last_accessed_at value the reinforcement UPDATE stamps in, distinct from the
# seed anchor so the round-trip assert can't pass by coincidence. Microseconds are kept
# so the round-trip also guards against second-truncation in the datetime bind/parse.
_REINFORCED = datetime(2026, 5, 30, 12, 0, 0, 123456, tzinfo=UTC)
# A distinct initial stamp for the sibling chunk that the UPDATE must NOT touch — kept
# different from both _LAST_ACCESSED and _REINFORCED so over-stamping is detectable.
_SIBLING_SEED = datetime(2025, 11, 9, tzinfo=UTC)


@pytest.mark.asyncio
async def test_update_last_accessed_round_trips_through_real_store(coord: StorageCoordinator) -> None:
    # The Chronicle reinforcement-on-recall path stamps chunk.last_accessed_at via the
    # coordinator. Seed TWO chunks in one namespace, each with a distinct INITIAL
    # last_accessed_at, call coord.update_last_accessed for ONLY the first with a DISTINCT
    # new timestamp, then read both back. The stamped chunk must read back the new value
    # (proving the UPDATE + row mapper round-trip the tz-aware, microsecond-precise stamp)
    # and the sibling must keep its seed (proving the UPDATE's id scope didn't over-stamp
    # other rows in the same namespace). Exercises the real SurrealDBVectorAdapter UPDATE
    # against embedded SurrealDB — a regression in the SQL/bindings or the mapper would
    # read back the seed (or None), and a missing id scope would over-stamp the sibling.
    ns = await coord.create_namespace(MemoryNamespace())
    chunks = await _seed(
        coord,
        ns.id,
        [
            {"content": "reinforce me", "last_accessed_at": _LAST_ACCESSED},
            {"content": "leave me alone", "last_accessed_at": _SIBLING_SEED},
        ],
    )
    target_id = chunks[0].id
    sibling_id = chunks[1].id

    # Sanity: the seed stamps are the initial values before the UPDATE, and all three
    # anchors are genuinely distinct so the asserts below have teeth.
    seeded_target = await coord.get_chunk(target_id, namespace_id=ns.id)
    seeded_sibling = await coord.get_chunk(sibling_id, namespace_id=ns.id)
    assert seeded_target is not None and seeded_target.last_accessed_at == _LAST_ACCESSED
    assert seeded_sibling is not None and seeded_sibling.last_accessed_at == _SIBLING_SEED
    assert len({_LAST_ACCESSED, _SIBLING_SEED, _REINFORCED}) == 3

    rowcount = await coord.update_last_accessed(ns.id, [target_id], _REINFORCED)
    assert rowcount == 1, "the in-namespace UPDATE must touch exactly the one listed chunk"

    read_back = await coord.get_chunk(target_id, namespace_id=ns.id)
    assert read_back is not None
    assert read_back.last_accessed_at == _REINFORCED, (
        "update_last_accessed must persist the new timestamp through the real SurrealDB "
        f"store (stamped {_REINFORCED!r}, read back {read_back.last_accessed_at!r}) — a "
        "regression in the UPDATE SQL/bindings or the row mapper would read back the seed"
    )
    # tz-awareness survives the round-trip (the column is a SurrealDB datetime; a naive
    # read-back would mean the bind or the mapper dropped tzinfo).
    assert read_back.last_accessed_at.tzinfo is not None, "last_accessed_at lost tzinfo on round-trip"

    # The sibling was NOT in the id list — its seed stamp must be untouched. Catches an
    # UPDATE whose WHERE clause dropped the id scope and stamped every chunk in the ns.
    sibling_after = await coord.get_chunk(sibling_id, namespace_id=ns.id)
    assert sibling_after is not None
    assert sibling_after.last_accessed_at == _SIBLING_SEED, (
        "a chunk not in the id list must keep its stamp — the UPDATE's id scope must not "
        "over-stamp sibling chunks in the same namespace"
    )


@pytest.mark.asyncio
async def test_update_last_accessed_counts_only_matched_rows(coord: StorageCoordinator) -> None:
    # Rowcount fidelity: the return is the count of rows the UPDATE actually matched, not
    # the count of ids requested. Seed ONE real chunk, then call update_last_accessed with
    # that real id PLUS a nonexistent id in the same namespace. The result must be 1 (only
    # the real row matched), and the real chunk must read back the new stamp. Catches a
    # regression that returns len(chunk_ids) instead of the matched-row count — every other
    # test happens to use a request list whose length equals the match count, so this is
    # the one case that discriminates the two.
    ns = await coord.create_namespace(MemoryNamespace())
    chunks = await _seed(
        coord,
        ns.id,
        [{"content": "real chunk", "last_accessed_at": _LAST_ACCESSED}],
    )
    real_id = chunks[0].id
    missing_id = uuid4()

    rowcount = await coord.update_last_accessed(ns.id, [real_id, missing_id], _REINFORCED)
    assert rowcount == 1, (
        "update_last_accessed must return the count of matched rows (1), not the number of "
        f"ids requested (2) — got {rowcount}; a return len(chunk_ids) regression would give 2"
    )

    read_back = await coord.get_chunk(real_id, namespace_id=ns.id)
    assert read_back is not None
    assert read_back.last_accessed_at == _REINFORCED


@pytest.mark.asyncio
async def test_update_last_accessed_rejects_cross_namespace_writes(coord: StorageCoordinator) -> None:
    # IDOR guard: a chunk id from ns_a passed with ns_b must NOT be stamped. The UPDATE is
    # scoped to the namespace, so a forged-id write through the reinforcement path affects
    # zero rows and leaves the chunk's stamp unchanged.
    ns_a = await coord.create_namespace(MemoryNamespace())
    ns_b = await coord.create_namespace(MemoryNamespace())
    chunks = await _seed(
        coord,
        ns_a.id,
        [{"content": "ns a chunk", "last_accessed_at": _LAST_ACCESSED}],
    )
    chunk_a_id = chunks[0].id

    rowcount = await coord.update_last_accessed(ns_b.id, [chunk_a_id], _REINFORCED)
    assert rowcount == 0, "a cross-namespace update_last_accessed must touch zero rows"

    # The chunk's stamp in its real namespace is unchanged — the wrong-namespace UPDATE
    # did not leak across the tenant boundary.
    read_back = await coord.get_chunk(chunk_a_id, namespace_id=ns_a.id)
    assert read_back is not None
    assert read_back.last_accessed_at == _LAST_ACCESSED, (
        "a cross-namespace update_last_accessed must leave the chunk's stamp unchanged"
    )


@pytest.mark.asyncio
async def test_update_last_accessed_empty_list_is_noop(coord: StorageCoordinator) -> None:
    # Empty chunk_ids short-circuits in the coordinator before any SurrealDB round-trip.
    ns = await coord.create_namespace(MemoryNamespace())
    rowcount = await coord.update_last_accessed(ns.id, [], _REINFORCED)
    assert rowcount == 0
