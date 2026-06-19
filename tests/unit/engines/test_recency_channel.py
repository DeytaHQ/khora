"""Unit tests for the parallel recency channel (Issue #567 A3).

Devil's-Advocate demand #3: a chunk with cosine similarity below the
relevance floor must NOT enter the merged pool, even if it's
today-dated. This test pins the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever
from khora.filter import FilterNode, RecallFilter, parse_to_ast
from khora.filter.compilers.python import compile_python
from khora.filter.execute import build_compile_context
from khora.filter.report import ChannelPlan
from khora.storage.temporal import TemporalVectorStore


def _make_chunk(content: str, occurred_at: datetime | None, embedding: list[float]) -> Chunk:
    """Build a Chunk shaped like what pgvector.search_recent_chunks returns."""
    chunk = MagicMock(spec=Chunk)
    chunk.id = uuid4()
    chunk.namespace_id = uuid4()
    chunk.document_id = uuid4()
    chunk.content = content
    chunk.embedding = embedding
    chunk.occurred_at = occurred_at
    chunk.created_at = occurred_at
    chunk.metadata = None
    return chunk


@pytest.fixture
def retriever_with_mocked_store():
    """A VectorCypherRetriever stub wired with a mock vector_store.

    We bypass the real engine plumbing — only ``_recency_channel_chunks``
    is under test, and it depends solely on ``self._config`` and
    ``self._vector_store.search_recent_chunks``.
    """
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
        temporal_recency_channel_limit=10,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg
    retriever._vector_store = MagicMock()
    return retriever


@pytest.mark.asyncio
async def test_below_floor_chunk_excluded(retriever_with_mocked_store) -> None:
    """A today-dated chunk with cosine=0.20 must NOT enter the pool when
    the floor is 0.30 (Devil's-Advocate demand #3, exact scenario)."""
    retriever = retriever_with_mocked_store

    # Two chunks: one above floor (0.5), one below (0.2). Both today-dated.
    query_emb = [1.0, 0.0, 0.0]
    above_floor_chunk = _make_chunk("recent and relevant", datetime.now(UTC), [0.5, 0.866, 0.0])
    below_floor_chunk = _make_chunk("recent and irrelevant", datetime.now(UTC), [0.2, 0.0, 0.979])

    retriever._vector_store.search_recent_chunks = AsyncMock(
        return_value=[(above_floor_chunk, None), (below_floor_chunk, None)]
    )

    result = await retriever._recency_channel_chunks(
        query_embedding=query_emb,
        namespace_id=uuid4(),
        temporal_filter=None,
    )

    # Only the above-floor chunk survives.
    assert len(result) == 1
    assert result[0][0] == above_floor_chunk.id


@pytest.mark.asyncio
async def test_empty_store_returns_empty(retriever_with_mocked_store) -> None:
    """No candidates from SQL → empty result, no crash."""
    retriever = retriever_with_mocked_store
    retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[])

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_sql_failure_returns_empty_does_not_raise(retriever_with_mocked_store) -> None:
    """If search_recent_chunks throws (missing index, etc.), the channel
    must degrade silently — the caller's retrieve() must not fail."""
    retriever = retriever_with_mocked_store
    retriever._vector_store.search_recent_chunks = AsyncMock(side_effect=RuntimeError("boom"))

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_no_search_recent_chunks_method_returns_empty() -> None:
    """If the vector store doesn't implement search_recent_chunks (e.g.
    SurrealDB path), the channel returns [] without raising."""
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg

    # A vector_store stub WITHOUT search_recent_chunks attribute.
    # Use a real object (not MagicMock — which auto-creates attrs).
    class _NoRecency:
        pass

    retriever._vector_store = _NoRecency()

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_chunks_without_embeddings_skipped(retriever_with_mocked_store) -> None:
    """A chunk that came back from SQL with embedding=None can't be
    cosine-filtered, so the gate must skip it rather than treating it as
    above-floor."""
    retriever = retriever_with_mocked_store
    no_emb = _make_chunk("today no embedding", datetime.now(UTC), [])
    no_emb.embedding = None

    retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[(no_emb, None)])

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


class _ProtocolDefaultStore(TemporalVectorStore):
    """A concrete TemporalVectorStore that relies on the Protocol's DEFAULT
    ``search_recent_chunks`` (which returns ``[]``).

    Unlike SurrealDB or pgvector this store does NOT override the recency
    method — it inherits the no-op default. The only abstract members it
    implements are the bare minimum to instantiate; none are exercised by
    the recency channel under test, so they raise if mis-called.
    """

    async def connect(self) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError

    async def create_chunk(self, chunk: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def create_chunks_batch(self, chunks: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:  # pragma: no cover
        raise NotImplementedError

    async def search(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def health_check(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def _filter_ast() -> FilterNode:
    """A real recall-filter AST so the plan-recording branch is reachable."""
    return parse_to_ast(RecallFilter.model_validate({"metadata.tier": "gold"}))


@pytest.mark.asyncio
async def test_protocol_default_store_returns_empty_and_records_no_plan() -> None:
    """A real ``TemporalVectorStore`` that inherits the Protocol DEFAULT
    ``search_recent_chunks`` (returns ``[]``) contributes nothing: the
    channel returns ``[]`` and records NO ChannelPlan, even when a caller
    filter is present (the early-return on empty SQL rows precedes the
    plan-recording site). A channel that never produced post-filtered
    chunks must never be credited with a disposition in the report."""
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg
    retriever._vector_store = _ProtocolDefaultStore()

    filter_channel_plans: dict[str, Any] = {}
    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=_filter_ast(),
        filter_channel_plans=filter_channel_plans,
    )

    assert result == []
    # The recency channel honestly never appears in the report.
    assert "recency" not in filter_channel_plans
    assert filter_channel_plans == {}


# --------------------------------------------------------------------------- #
# GitHub issue #1223 — the recency channel must push the recall filter into the
# SQL so a filter-violating recent chunk is never fetched, and must report the
# pushdown honestly. Before the fix the channel rebuilt each row into a
# provenance-blank public ``Chunk`` and ran an in-memory post-filter that could
# not read ``source_name`` — so a chunk whose ``source_name`` violated the
# filter leaked into the merged pool. These tests pin both the no-leak behavior
# and the honest ChannelPlan recording.
# --------------------------------------------------------------------------- #


class _FilterAwareRecencyStore:
    """A stub temporal store that simulates the pgvector SQL pushdown.

    ``search_recent_chunks`` honors ``filter_ast`` by evaluating it against the
    stored chunk's REAL ``source_name`` (the denormalized provenance field that
    lives on the SQL row, not on the public ``Chunk`` the channel rebuilds) —
    mirroring the WHERE predicate the real ``khora_chunks`` compile produces. A
    chunk whose ``source_name`` fails the predicate is dropped before it is ever
    returned, exactly as the SQL would never fetch it. When ``filter_ast`` is set
    the store appends ``ChannelPlan(pushed_keys=frozenset({"source_name"}))`` to
    ``filter_plan_out``, matching the ``consumed_keys`` the pg compiler reports.
    """

    def __init__(self, chunk: Chunk, source_name: str) -> None:
        self._chunk = chunk
        self._source_name = source_name

    async def search_recent_chunks(
        self,
        namespace_id: UUID,
        limit: int,
        *,
        created_after: datetime | None = None,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[tuple[Chunk, float | None]]:
        rows: list[tuple[Chunk, float | None]] = [(self._chunk, None)]
        if filter_ast is not None:
            # Compile the filter the same way the pg backend does and run it
            # against a record carrying the SQL-row ``source_name``. This is the
            # in-SQL WHERE the real backend pushes down — a violating row is
            # never fetched.
            predicate = compile_python(
                filter_ast,
                build_compile_context("khora_chunks", on_unsupported="raise"),
            ).predicate

            class _Row:
                def __init__(self, source_name: str) -> None:
                    self.source_name = source_name
                    self.metadata = None

            rows = [r for r in rows if predicate(_Row(self._source_name))]
            if filter_plan_out is not None:
                filter_plan_out.append(ChannelPlan(pushed_keys=frozenset({"source_name"})))
        elif filter_plan_out is not None:
            # No caller filter: the real pgvector store still appends an (empty)
            # ChannelPlan to the sink. The channel must NOT credit a "recency"
            # entry in that case (gated on ``filter_ast is not None``).
            filter_plan_out.append(ChannelPlan())
        return rows


def _aligned_chunk() -> Chunk:
    """A recent chunk whose embedding aligns with the query (cosine ≥ 0).

    The channel's relevance floor is set to 0.0 in these tests, so any
    non-negative cosine survives the gate — the test isolates the filter
    behavior from the relevance gate.
    """
    return _make_chunk("recent secret content", datetime.now(UTC), [1.0, 0.0, 0.0])


def _recency_retriever() -> VectorCypherRetriever:
    """A retriever whose relevance floor is 0.0 so the aligned chunk survives."""
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.0,
        temporal_recency_channel_limit=10,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg
    return retriever


@pytest.mark.asyncio
async def test_ne_filter_excludes_chunk_records_no_plan() -> None:
    """``$ne`` on the stored chunk's ``source_name`` → SQL fetches 0 rows, so
    the channel returns nothing AND records no recency plan (a channel that
    produced no surviving chunks gates nothing and must not be credited)."""
    retriever = _recency_retriever()
    chunk = _aligned_chunk()
    retriever._vector_store = _FilterAwareRecencyStore(chunk, source_name="secret")

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$ne": "secret"}}))
    filter_channel_plans: dict[str, Any] = {}

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=filter_ast,
        filter_channel_plans=filter_channel_plans,
    )

    assert result == []
    assert "recency" not in filter_channel_plans


@pytest.mark.asyncio
async def test_ne_filter_does_not_leak_secret_chunk() -> None:
    """Regression for GitHub issue #1223: under ``{"source_name": {"$ne":
    "secret"}}`` the recent "secret" chunk must NOT appear in the channel's
    results. Pre-fix the channel post-filtered a rebuilt provenance-blank
    ``Chunk`` (no ``source_name``), so the predicate matched-all and the secret
    chunk leaked into the merged pool."""
    retriever = _recency_retriever()
    secret_chunk = _aligned_chunk()
    retriever._vector_store = _FilterAwareRecencyStore(secret_chunk, source_name="secret")

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$ne": "secret"}}))

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=filter_ast,
    )

    leaked_ids = {cid for cid, _score, _chunk in result}
    assert secret_chunk.id not in leaked_ids


@pytest.mark.asyncio
async def test_eq_filter_includes_chunk_records_pushed_plan() -> None:
    """``$eq`` matching the stored chunk's ``source_name`` → the chunk survives
    the SQL, and the recorded recency ChannelPlan credits ``source_name`` as
    pushed (and post-filters nothing — the predicate ran entirely in SQL)."""
    retriever = _recency_retriever()
    chunk = _aligned_chunk()
    retriever._vector_store = _FilterAwareRecencyStore(chunk, source_name="secret")

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$eq": "secret"}}))
    filter_channel_plans: dict[str, Any] = {}

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=filter_ast,
        filter_channel_plans=filter_channel_plans,
    )

    assert [cid for cid, _score, _chunk in result] == [chunk.id]
    assert "recency" in filter_channel_plans
    plan = filter_channel_plans["recency"]
    assert "source_name" in plan.pushed_keys
    assert sorted(plan.post_filtered_keys) == []


class _RaisingRecencyStore:
    """Temporal store whose ``search_recent_chunks`` raises ``exc`` — models a
    backend that rejects a filter leaf it cannot push (``RecallFilterUnsupportedError``)
    or hits an operational fault (a generic exception)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def search_recent_chunks(
        self,
        namespace_id: UUID,
        limit: int,
        *,
        created_after: datetime | None = None,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[tuple[Chunk, float | None]]:
        raise self._exc


@pytest.mark.asyncio
async def test_unsupported_filter_leaf_propagates_fail_loud() -> None:
    """A filter leaf the recency SQL cannot push raises ``RecallFilterUnsupportedError``,
    and the channel re-raises it rather than swallowing it — matching the vector
    channel's fail-loud contract so an unenforceable filter never silently no-ops."""
    from khora.filter.model import RecallFilterUnsupportedError

    retriever = _recency_retriever()
    retriever._vector_store = _RaisingRecencyStore(RecallFilterUnsupportedError("<leaf>", "unsupported"))

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$ne": "secret"}}))

    with pytest.raises(RecallFilterUnsupportedError):
        await retriever._recency_channel_chunks(
            query_embedding=[1.0, 0.0, 0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            filter_ast=filter_ast,
        )


@pytest.mark.asyncio
async def test_operational_fault_degrades_to_empty() -> None:
    """An operational fault (DB / network) degrades the recency channel to ``[]``
    — it is pool augmentation only, so the vector + BM25 channels still enforce
    the filter and carry the report. A generic exception must NOT propagate."""
    retriever = _recency_retriever()
    retriever._vector_store = _RaisingRecencyStore(RuntimeError("connection reset"))

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$ne": "secret"}}))

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=filter_ast,
    )

    assert result == []


@pytest.mark.asyncio
async def test_operational_fault_records_degradation() -> None:
    """On an operational fault the channel appends a structured Degradation to the
    caller's ``degradations`` list (ADR-001) so the silently-dropped channel is
    observable in ``RecallResult.engine_info['degradations']``."""
    retriever = _recency_retriever()
    retriever._vector_store = _RaisingRecencyStore(RuntimeError("connection reset"))

    filter_ast = parse_to_ast(RecallFilter.model_validate({"source_name": {"$ne": "secret"}}))
    degradations: list[Any] = []

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=filter_ast,
        degradations=degradations,
    )

    assert result == []
    assert len(degradations) == 1
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.recency_channel"
    assert deg["reason"] == "channel_exception"
    assert deg["exception"] == "RuntimeError"


@pytest.mark.asyncio
async def test_no_filter_recall_records_no_recency_plan() -> None:
    """A no-filter recency recall must NOT credit a ``recency`` channel entry,
    even though the store appends an (empty) ChannelPlan to the sink — symmetric
    with the vector / BM25 channels, which the caller gates on ``filter_ast is not
    None``. Guards the plan-recording branch added for that symmetry."""
    retriever = _recency_retriever()
    chunk = _aligned_chunk()
    retriever._vector_store = _FilterAwareRecencyStore(chunk, source_name="secret")

    filter_channel_plans: dict[str, Any] = {}

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=None,
        filter_channel_plans=filter_channel_plans,
    )

    # The chunk still surfaces (no filter to exclude it) ...
    assert [cid for cid, _score, _chunk in result] == [chunk.id]
    # ... but no recency disposition is recorded without a caller filter.
    assert "recency" not in filter_channel_plans
