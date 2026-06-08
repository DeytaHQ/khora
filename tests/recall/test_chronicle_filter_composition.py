"""Engine-composition tests for the Chronicle recall-filter (Layer 4 → engine).

``@internal``. Pins task part (c): how the Chronicle engine *composes* the two
compiled halves of a recall filter at ``recall()`` time —

1. **Date bounds intersect the recency window (narrow only).** The
   :func:`compile_chronicle` date-bound is folded into the engine's existing
   ``created_after`` / ``created_before`` recency window via ``_intersect_lower`` /
   ``_intersect_upper`` — ``max`` of the lower bounds, ``min`` of the upper bounds.
   The filter can only SHRINK the window, never widen it.
2. **The metadata predicate post-filters chunk candidates.** The
   :func:`compile_python` predicate is applied to the fused candidates so an
   out-of-scope chunk is dropped before top-k.
3. **``khora.recall.filter.unindexed_metadata`` increments on the post-filter
   path.** ``compile_python`` emits the counter per metadata leaf at compile time
   (mirroring ``compile_postgres``), so a metadata-bearing filter recall fires it.

The window-intersection assertions exercise the pure ``_intersect_*`` helpers
directly (deterministic, no infra). The post-filter / telemetry assertions drive
``ChronicleEngine.recall()`` over a mocked storage + embedder (the established
unit pattern from ``tests/unit/engines/test_chronicle_abstention_signals.py``):
the semantic channel returns a known in-scope / out-of-scope chunk mix and the
filter must narrow the result.

INTERIM SKIP: self-skips if the filter compilers / engine wiring are not yet on
the branch, so the suite stays green until Backend's slice lands.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

# Self-skip until the engine wiring + compilers are on the branch.
pytest.importorskip(
    "khora.filter.compilers.python",
    reason="recall-filter compilers not yet on the branch (Backend task in flight)",
)
pytest.importorskip(
    "khora.filter.compilers.chronicle",
    reason="Chronicle compiler not yet on the branch (Backend task in flight)",
)

from khora.config import KhoraConfig  # noqa: E402
from khora.config.schema import QuerySettings  # noqa: E402
from khora.core.models import Chunk  # noqa: E402
from khora.core.models.document import DocumentSource  # noqa: E402
from khora.engines.chronicle import engine as chronicle_engine  # noqa: E402
from khora.engines.chronicle.engine import ChronicleEngine  # noqa: E402
from khora.filter import telemetry as filter_telemetry  # noqa: E402
from khora.query import SearchMode  # noqa: E402

# The narrow-only intersection helpers are required for part (c) assertion 1; if
# they are renamed this import (and the tests) need a one-line update.
_intersect_lower = chronicle_engine._intersect_lower
_intersect_upper = chronicle_engine._intersect_upper


# ===========================================================================
# (1) Date bounds intersect the recency window — NARROW ONLY.
# ===========================================================================
#
# The engine folds the compiled date-bound into the existing recency window with
# _intersect_lower (max of lowers) / _intersect_upper (min of uppers). These pure
# helpers are the load-bearing "narrow only" guarantee.


_EARLY = datetime(2026, 1, 1, tzinfo=UTC)
_MID = datetime(2026, 6, 1, tzinfo=UTC)
_LATE = datetime(2026, 12, 1, tzinfo=UTC)


def test_lower_intersection_takes_the_later_bound() -> None:
    # max of the two lower bounds — the filter tightens the window's start.
    assert _intersect_lower(_EARLY, _MID) == _MID
    assert _intersect_lower(_MID, _EARLY) == _MID


def test_upper_intersection_takes_the_earlier_bound() -> None:
    # min of the two upper bounds — the filter tightens the window's end.
    assert _intersect_upper(_LATE, _MID) == _MID
    assert _intersect_upper(_MID, _LATE) == _MID


def test_none_filter_bound_leaves_window_unchanged() -> None:
    # No filter bound on a side → that side of the window is untouched.
    assert _intersect_lower(_MID, None) == _MID
    assert _intersect_upper(_MID, None) == _MID


def test_none_window_adopts_filter_bound() -> None:
    # An unbounded window side adopts the filter's bound (still a narrowing — it
    # goes from "unbounded" to "bounded").
    assert _intersect_lower(None, _MID) == _MID
    assert _intersect_upper(None, _MID) == _MID


@pytest.mark.parametrize("filter_lo", [_EARLY, _MID, _LATE, None])
def test_lower_intersection_never_widens(filter_lo: datetime | None) -> None:
    # Property: the resulting lower bound is always >= the incoming window lower
    # bound (a None result only when BOTH sides are unbounded).
    window = _MID
    result = _intersect_lower(window, filter_lo)
    assert result is not None
    assert result >= window, "lower-bound intersection must never move the window start earlier"


@pytest.mark.parametrize("filter_hi", [_EARLY, _MID, _LATE, None])
def test_upper_intersection_never_widens(filter_hi: datetime | None) -> None:
    # Property: the resulting upper bound is always <= the incoming window upper.
    window = _MID
    result = _intersect_upper(window, filter_hi)
    assert result is not None
    assert result <= window, "upper-bound intersection must never move the window end later"


# ===========================================================================
# Mocked-engine harness for the post-filter / telemetry assertions.
# ===========================================================================


def _chunk(
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
    created_at: datetime | None = None,
    source_timestamp: datetime | None = None,
) -> Chunk:
    """A minimal Chunk with optional metadata + the three date fields.

    ``created_at`` defaults to now() when unset (the dataclass default); the
    explicit param lets a test pin a specific created_at for the literal-date-key
    assertions.
    """
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        created_at=created_at if created_at is not None else datetime.now(UTC),
        metadata=metadata or {},
        occurred_at=occurred_at,
        source_timestamp=source_timestamp,
    )


def _engine_with_semantic(results: list[tuple[Chunk, float]]) -> ChronicleEngine:
    """A connected ChronicleEngine whose semantic channel returns ``results``.

    BM25 / entity channels return empty so the semantic channel is the only
    candidate source; VECTOR mode is used at the call site to keep retrieval
    purely embedding + filter (no keyword channel), so the filter is the only
    narrowing force on the candidate set.

    Reranking is disabled (``enable_reranking=False``): it defaults to True and
    would lazily download / load the BAAI/bge-reranker-v2-m3 cross-encoder on the
    first recall, making these tests slow, network-dependent, and cold-cache
    flaky. The reranker only reorders candidates — it never adds or drops a row —
    so disabling it leaves the filter-narrowing contract under test unchanged
    while keeping the tests hermetic.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    storage.search_similar_chunks = AsyncMock(return_value=results)
    storage.search_similar_entities = AsyncMock(return_value=[])
    engine._storage = storage

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    engine._embedder = embedder
    engine._connected = True
    return engine


def _filter_ast(wire: dict) -> Any:
    from khora.filter import RecallFilter
    from khora.filter.ast import parse_to_ast

    return parse_to_ast(RecallFilter.model_validate(wire))


# ===========================================================================
# (2) The metadata predicate post-filters chunk candidates.
# ===========================================================================


@pytest.mark.asyncio
async def test_metadata_filter_drops_out_of_scope_chunks() -> None:
    # Two in-scope (tier=gold) + two out-of-scope (tier=silver / missing). A
    # metadata.tier == gold filter must return ONLY the in-scope chunks.
    in1 = _chunk("alpha gold one", metadata={"tier": "gold"})
    in2 = _chunk("alpha gold two", metadata={"tier": "gold"})
    out_silver = _chunk("alpha silver", metadata={"tier": "silver"})
    out_missing = _chunk("alpha none", metadata={})
    results = [(in1, 0.9), (in2, 0.85), (out_silver, 0.8), (out_missing, 0.75)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"metadata.tier": "gold"}),
    )

    returned = {c.id for c in result.chunks}
    assert returned == {in1.id, in2.id}, f"metadata post-filter must keep exactly the in-scope chunks; got {returned}"


@pytest.mark.asyncio
async def test_no_filter_returns_all_candidates() -> None:
    # Control: without a filter, the same candidate set comes back in full — so the
    # narrowing above is attributable to the FILTER, not retrieval reachability.
    chunks = [_chunk(f"alpha-{i}", metadata={"tier": "silver"}) for i in range(4)]
    results = [(c, 0.9 - 0.1 * i) for i, c in enumerate(chunks)]

    engine = _engine_with_semantic(results)
    result = await engine.recall("alpha", uuid4(), limit=10, mode=SearchMode.VECTOR)

    assert {c.id for c in result.chunks} == {c.id for c in chunks}


@pytest.mark.asyncio
async def test_filter_excluding_all_yields_empty() -> None:
    # A filter no candidate satisfies narrows the result to empty (the post-filter
    # is applied even when it removes everything).
    chunks = [_chunk(f"alpha-{i}", metadata={"tier": "silver"}) for i in range(3)]
    results = [(c, 0.9) for c in chunks]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"metadata.tier": "gold"}),
    )

    assert result.chunks == []


# ===========================================================================
# occurred_at filtering narrows (not empties) — against the EFFECTIVE event time.
# ===========================================================================
#
# Semantics (engine._chunk_to_record): the post-filter
# record's occurred_at is the EFFECTIVE EVENT TIME = COALESCE(occurred_at,
# source_timestamp) — the chunk's literal occurred_at, falling back to
# source_timestamp when occurred_at is None (the pgvector DTO always nulls
# occurred_at). There is deliberately NO created_at fallback: ingest time is not
# event time, so a chunk with neither occurred_at nor source_timestamp has no
# effective occurred_at and a positive filter excludes it. These tests drive scope
# via occurred_at / source_timestamp (the COALESCE inputs) accordingly. (Contrast:
# the eight denormalized document keys are NOT carried on the legacy DTO, so a
# positive predicate on them returns empty — an accepted, documented limitation we
# deliberately do NOT assert as "returns rows".)


@pytest.mark.asyncio
async def test_occurred_at_filter_narrows_against_effective_event_time() -> None:
    # The record's occurred_at = source_timestamp or created_at. Drive in/out of
    # scope via source_timestamp (the precedence input). in_scope's source_timestamp
    # is after the bound; too_old's is before it; both have created_at irrelevant
    # because source_timestamp takes precedence in the COALESCE.
    in_scope = _chunk(
        "alpha recent",
        source_timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    too_old = _chunk(
        "alpha old",
        source_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    results = [(in_scope, 0.9), (too_old, 0.8)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}}),
    )

    returned = {c.id for c in result.chunks}
    assert in_scope.id in returned, "occurred_at filtering must NOT silently drop the in-scope row"
    assert returned == {in_scope.id}, "the pre-bound row (source_timestamp too old) must be excluded"


@pytest.mark.asyncio
async def test_occurred_at_upper_bound_filter_narrows() -> None:
    # The upper-bound direction also narrows (not empties): the row whose effective
    # event time is at/before the bound survives, the later one is dropped.
    in_scope = _chunk("alpha early", source_timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    too_new = _chunk("alpha late", source_timestamp=datetime(2026, 12, 1, tzinfo=UTC))
    results = [(in_scope, 0.9), (too_new, 0.8)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"occurred_at": {"$lte": "2026-06-01T00:00:00Z"}}),
    )

    returned = {c.id for c in result.chunks}
    assert in_scope.id in returned
    assert returned == {in_scope.id}


@pytest.mark.asyncio
async def test_occurred_at_filter_falls_back_to_source_timestamp_pg_shape() -> None:
    # REGRESSION (formerly the masked PG-shape bug, now fixed by _chunk_to_record):
    # the pgvector DTO has chunk.occurred_at=None, so the record's occurred_at falls
    # back to source_timestamp (record["occurred_at"] = occurred_at or
    # source_timestamp). A PG row whose source_timestamp is in range MUST survive an
    # occurred_at filter (not false-empty); one whose source_timestamp is out of
    # range is dropped. This is the case my earlier populated-occurred_at fixture
    # MASKED. (created_at is deliberately set out of range to prove it is NOT a
    # fallback — see the no-anchor test below.)
    pg_in_scope = _chunk(
        "alpha pg in",
        occurred_at=None,
        source_timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    pg_too_old = _chunk(
        "alpha pg old",
        occurred_at=None,
        source_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    results = [(pg_in_scope, 0.9), (pg_too_old, 0.8)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}}),
    )

    returned = {c.id for c in result.chunks}
    assert pg_in_scope.id in returned, "PG-shape occurred_at filter must fall back to source_timestamp, not empty"
    # The out-of-range PG row is dropped EVEN THOUGH its created_at is in range —
    # proving created_at is NOT part of the occurred_at fallback.
    assert returned == {pg_in_scope.id}, "occurred_at fallback is source_timestamp only; created_at must not rescue"


@pytest.mark.asyncio
async def test_occurred_at_filter_excludes_unanchored_chunk_no_created_at_fallback() -> None:
    # A chunk with NO event-time anchor (occurred_at=None AND source_timestamp=None)
    # has record["occurred_at"] = None, so a positive occurred_at $gte EXCLUDES it —
    # even though created_at is in range. Ingest time is deliberately NOT an
    # occurred_at fallback (engine._chunk_to_record: "No created_at fallback —
    # ingest time is not event time").
    unanchored = _chunk(
        "alpha unanchored",
        occurred_at=None,
        source_timestamp=None,
        created_at=datetime(2026, 6, 1, tzinfo=UTC),  # in range, but NOT a fallback
    )
    results = [(unanchored, 0.9)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}}),
    )

    assert result.chunks == [], (
        "an unanchored chunk must be excluded by a positive occurred_at filter (no created_at fallback)"
    )


@pytest.mark.asyncio
async def test_created_at_filter_works_against_literal_created_at() -> None:
    # created_at is post-filtered (cross-dimension, not pushed) against the chunk's
    # LITERAL created_at field — the filter narrows correctly and does not empty.
    in_scope = _chunk("alpha recent", created_at=datetime(2026, 6, 1, tzinfo=UTC))
    too_old = _chunk("alpha old", created_at=datetime(2020, 1, 1, tzinfo=UTC))
    results = [(in_scope, 0.9), (too_old, 0.8)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"created_at": {"$gte": "2026-01-01T00:00:00Z"}}),
    )

    returned = {c.id for c in result.chunks}
    assert in_scope.id in returned, "created_at filtering must narrow, not silently empty"
    assert returned == {in_scope.id}


# ===========================================================================
# (3) unindexed_metadata counter increments on the post-filter path.
# ===========================================================================


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


@pytest.fixture
def recording_counters(monkeypatch: pytest.MonkeyPatch) -> dict[str, _RecordingCounter]:
    """Replace the filter-telemetry counter singletons with recording fakes.

    Same monkeypatch-the-singleton hook the existing recall-filter telemetry tests
    use (tests/unit/filter/test_filter_telemetry.py): pre-seeding each module
    global makes ``record_unindexed_metadata`` land on the fake.
    """
    counters = {
        "unindexed_metadata": _RecordingCounter(),
        "under_filled": _RecordingCounter(),
        "graph_channel_empty": _RecordingCounter(),
    }
    monkeypatch.setattr(filter_telemetry, "_unindexed_metadata_counter", counters["unindexed_metadata"])
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counters["under_filled"])
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counters["graph_channel_empty"])
    return counters


@pytest.mark.asyncio
async def test_unindexed_metadata_fires_on_post_filter_recall(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # A metadata-bearing filter recall fires the unindexed_metadata counter (the
    # post-filter compiles the metadata leaf to a Python predicate). Assert >= 1
    # observation carrying the leaf's op — robust whether emission is per-compile
    # or (hypothetically) per-evaluation.
    chunks = [_chunk("alpha", metadata={"tier": "gold"})]
    engine = _engine_with_semantic([(chunks[0], 0.9)])

    await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"metadata.tier": "gold"}),
    )

    adds = recording_counters["unindexed_metadata"].adds
    assert len(adds) >= 1, "a metadata-bearing filter recall must fire unindexed_metadata"
    assert any(a[1].get("op") == "$eq" for a in adds)


@pytest.mark.asyncio
async def test_unindexed_metadata_silent_for_system_key_only_recall(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # A system-key-only filter (occurred_at date bound) does not touch metadata, so
    # the unindexed_metadata counter stays quiet on the post-filter path.
    chunks = [_chunk("alpha", metadata={"tier": "gold"})]
    engine = _engine_with_semantic([(chunks[0], 0.9)])

    await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}}),
    )

    assert recording_counters["unindexed_metadata"].adds == []


@pytest.mark.asyncio
async def test_no_filter_recall_does_not_fire_unindexed_metadata(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # No filter → no compile → no counter.
    chunks = [_chunk("alpha", metadata={"tier": "gold"})]
    engine = _engine_with_semantic([(chunks[0], 0.9)])

    await engine.recall("alpha", uuid4(), limit=10, mode=SearchMode.VECTOR)

    assert recording_counters["unindexed_metadata"].adds == []


# ===========================================================================
# Denorm doc-key carrier resolution (engine._chunk_to_record).
# ===========================================================================
#
# The post-filter record resolves a system key off chunk.source_document
# (DocumentSource carries ONLY id/title/source/source_type/created_at/
# source_timestamp). So on the Chronicle path:
#   * title / source / source_type RESOLVE from source_document when present.
#   * source_name / source_url / external_id / content_type have NO carrier and
#     are ALWAYS absent — a DOCUMENTED gap, not a bug. A positive predicate on a
#     no-carrier key returns empty; a negative predicate ($ne) matches all.
# We do NOT assert the no-carrier keys "return rows" (the lead's documented
# limitation); we assert exactly the absent-key polarity.


@pytest.mark.asyncio
async def test_source_document_fallback_resolves_projected_keys() -> None:
    # A chunk with source_document populated resolves source/source_type/title from
    # it, so a filter on those keys narrows correctly.
    sd = DocumentSource(id=uuid4(), title="Release notes", source="linear", source_type="issue")
    match = _chunk("alpha match")
    match.source_document = sd
    other = _chunk("alpha other")
    other.source_document = DocumentSource(id=uuid4(), title="x", source="slack", source_type="msg")
    results = [(match, 0.9), (other, 0.8)]

    engine = _engine_with_semantic(results)
    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"source": "linear"}),
    )

    assert {c.id for c in result.chunks} == {match.id}, "source must resolve off source_document for the post-filter"


@pytest.mark.asyncio
async def test_no_carrier_doc_key_positive_predicate_returns_empty() -> None:
    # source_name has NO carrier (not on DocumentSource), so it is always absent on
    # the Chronicle path → a positive $eq predicate returns empty. DOCUMENTED gap.
    sd = DocumentSource(id=uuid4(), title="t", source="linear", source_type="issue")
    chunk = _chunk("alpha")
    chunk.source_document = sd
    engine = _engine_with_semantic([(chunk, 0.9)])

    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"source_name": "linear"}),
    )

    assert result.chunks == [], "a positive predicate on a no-carrier key returns empty (documented gap)"


@pytest.mark.asyncio
async def test_no_carrier_doc_key_negative_predicate_matches_all() -> None:
    # The $ne mirror: a no-carrier (always-absent) key satisfies $ne <value> for
    # every row (Rule 2 polarity — absent is "not equal"), so the row survives.
    sd = DocumentSource(id=uuid4(), title="t", source="linear", source_type="issue")
    chunk = _chunk("alpha")
    chunk.source_document = sd
    engine = _engine_with_semantic([(chunk, 0.9)])

    result = await engine.recall(
        "alpha",
        uuid4(),
        limit=10,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast({"source_name": {"$ne": "linear"}}),
    )

    assert {c.id for c in result.chunks} == {chunk.id}, "$ne on a no-carrier key matches all (absent is not-equal)"
