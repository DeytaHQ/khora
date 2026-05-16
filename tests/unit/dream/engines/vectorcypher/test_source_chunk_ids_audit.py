"""Unit tests for the source_chunk_ids audit dream op (#659).

The SQL paths (Postgres ``unnest`` and SQLite JSON anti-join) are
exercised by integration tests against the live fixture stacks. These
unit tests pin the report-building logic by monkeypatching
``_collect_entity_rows`` so every test feeds curated per-entity rows
into the public coroutine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher import source_chunk_ids_audit
from khora.dream.engines.vectorcypher.source_chunk_ids_audit import (
    plan_vectorcypher_source_chunk_ids_audit,
)
from khora.dream.plan import DreamOp, OpKind

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCoordinator:
    """Just enough surface for ``plan_vectorcypher_source_chunk_ids_audit``.

    The audit op only uses ``resolve_namespace`` (to translate the stable
    id to the row id) plus, via the patched ``_collect_entity_rows``, an
    optional ``transaction`` context.
    """

    resolved_id: UUID = field(default_factory=uuid4)
    resolve_calls: list[UUID] = field(default_factory=list)

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        self.resolve_calls.append(namespace_id)
        return self.resolved_id


def _patch_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Replace the SQL collector with one that returns the given rows."""

    async def _fake_collect(coordinator: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr(source_chunk_ids_audit, "_collect_entity_rows", _fake_collect)


def _row(*, length: int, dead: int, name: str | None = None) -> dict[str, Any]:
    """Shape that ``_collect_entity_rows`` returns."""
    return {
        "entity_id": uuid4(),
        "name": name or f"e-{length}",
        "length": length,
        "dead_uuid_count": dead,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counts_dead_uuids(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entity with 5 source_chunk_ids but only 3 chunks alive → dead_uuid_count == 2."""
    rows = [_row(length=5, dead=2, name="five")]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    report = op.outputs[0]

    assert report["total_entities"] == 1
    assert report["total_dead_uuids"] == 2
    assert report["top_offenders"][0]["dead_uuid_count"] == 2
    assert report["top_offenders"][0]["length"] == 5
    assert report["top_offenders"][0]["name"] == "five"


@pytest.mark.asyncio
async def test_length_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """100 entities with lengths 1..100 → p50/p90/p99/max match nearest-rank quantiles."""
    rows = [_row(length=i, dead=0) for i in range(1, 101)]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    report = op.outputs[0]

    assert report["total_entities"] == 100
    assert report["length_p50"] == 50
    assert report["length_p90"] == 90
    assert report["length_p99"] == 99
    assert report["length_max"] == 100
    assert report["total_dead_uuids"] == 0


@pytest.mark.asyncio
async def test_top_offenders_sorted_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-20 offenders are returned longest-first and capped to top_k_offenders."""
    # 50 entities with shuffled lengths so the sort actually has to do work.
    lengths = list(range(1, 51))
    # interleave so the input isn't already sorted descending
    interleaved = lengths[::2] + lengths[1::2]
    rows = [_row(length=length, dead=0) for length in interleaved]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    report = op.outputs[0]

    top = report["top_offenders"]
    assert len(top) == 20
    assert [item["length"] for item in top] == list(range(50, 30, -1))


@pytest.mark.asyncio
async def test_top_k_offenders_param_caps_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller can shrink the top-K via the ``top_k_offenders`` kwarg."""
    rows = [_row(length=i, dead=0) for i in range(1, 11)]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(
        uuid4(),
        coordinator=coord,
        top_k_offenders=3,
    )
    report = op.outputs[0]

    assert len(report["top_offenders"]) == 3
    assert [item["length"] for item in report["top_offenders"]] == [10, 9, 8]


@pytest.mark.asyncio
async def test_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit must be read-only.

    Implemented as a structural check: the public coroutine receives no
    object that exposes a write surface — only the coordinator (whose
    only method we touch is ``resolve_namespace``) and the patched row
    collector (which we control here).
    """
    rows = [_row(length=2, dead=1)]
    _patch_rows(monkeypatch, rows)

    writes: list[str] = []

    class _RecordingCoordinator(_FakeCoordinator):
        async def create_document(self, *args: Any, **kwargs: Any) -> None:
            writes.append("create_document")
            raise AssertionError("audit op must not call create_document")

        async def create_entity(self, *args: Any, **kwargs: Any) -> None:
            writes.append("create_entity")
            raise AssertionError("audit op must not call create_entity")

        async def replace_chunks(self, *args: Any, **kwargs: Any) -> None:
            writes.append("replace_chunks")
            raise AssertionError("audit op must not call replace_chunks")

    coord = _RecordingCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)

    assert op.decision == "audit_complete"
    assert writes == []


@pytest.mark.asyncio
async def test_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entity with an empty ``source_chunk_ids`` doesn't crash and reports zero."""
    rows = [_row(length=0, dead=0, name="empty")]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    report = op.outputs[0]

    assert report["total_entities"] == 1
    assert report["total_dead_uuids"] == 0
    assert report["length_p50"] == 0
    assert report["length_p90"] == 0
    assert report["length_p99"] == 0
    assert report["length_max"] == 0
    assert report["top_offenders"][0]["length"] == 0
    assert report["top_offenders"][0]["dead_uuid_count"] == 0


@pytest.mark.asyncio
async def test_empty_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero entities is not an error — all stats are zero, top_offenders is empty."""
    _patch_rows(monkeypatch, [])

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    report = op.outputs[0]

    assert report["total_entities"] == 0
    assert report["total_dead_uuids"] == 0
    assert report["length_p50"] == 0
    assert report["top_offenders"] == []


@pytest.mark.asyncio
async def test_dream_op_round_trips_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The op's outputs must JSON-serialize cleanly for the file/event sinks."""
    rows = [_row(length=3, dead=1), _row(length=7, dead=2)]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)

    blob = json.dumps(op.outputs[0])
    round_tripped = json.loads(blob)

    assert round_tripped["total_entities"] == 2
    assert round_tripped["total_dead_uuids"] == 3
    assert round_tripped["length_max"] == 7
    # entity_id must already be a string so the dict is json-safe.
    for offender in round_tripped["top_offenders"]:
        UUID(offender["entity_id"])  # parseable


@pytest.mark.asyncio
async def test_resolves_namespace_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stable namespace_id must be resolved to the row id before any audit work."""
    rows = [_row(length=1, dead=0)]
    _patch_rows(monkeypatch, rows)

    coord = _FakeCoordinator()
    stable = uuid4()
    op = await plan_vectorcypher_source_chunk_ids_audit(stable, coordinator=coord)

    assert coord.resolve_calls == [stable]
    assert op.namespace_id == stable


def test_op_kind_is_registered() -> None:
    """The new OpKind value must exist and use the spec'd string."""
    assert OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT.value == "vectorcypher_source_chunk_ids_audit"


@pytest.mark.asyncio
async def test_op_type_and_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """The returned DreamOp carries the right OpKind and phase tag."""
    _patch_rows(monkeypatch, [])
    coord = _FakeCoordinator()
    op = await plan_vectorcypher_source_chunk_ids_audit(uuid4(), coordinator=coord)
    assert isinstance(op, DreamOp)
    assert op.op_type == OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT
    assert op.phase == "audit"
    assert op.duration_ms is not None and op.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Pure-helper tests — pin the percentile + SQLite-parse helpers.
# ---------------------------------------------------------------------------


def test_length_percentiles_handles_empty() -> None:
    assert source_chunk_ids_audit._length_percentiles([]) == (0, 0, 0, 0)


def test_length_percentiles_single_value() -> None:
    assert source_chunk_ids_audit._length_percentiles([7]) == (7, 7, 7, 7)


def test_length_percentiles_two_values() -> None:
    # Nearest-rank with [1, 10]: p50 picks the lower (rank 1).
    p50, p90, p99, mx = source_chunk_ids_audit._length_percentiles([1, 10])
    assert p50 == 1
    assert p90 == 10
    assert p99 == 10
    assert mx == 10


@pytest.mark.parametrize(
    "value, expected_len",
    [
        (None, 0),
        ("[]", 0),
        ("not-json", 0),
        ("{}", 0),  # JSON but not a list
        ('["00000000-0000-0000-0000-000000000001"]', 1),
        (["00000000-0000-0000-0000-000000000001"], 1),
        ('["not-a-uuid", "00000000-0000-0000-0000-000000000001"]', 1),
    ],
)
def test_parse_sqlite_uuid_list_handles_edge_cases(value: Any, expected_len: int) -> None:
    out = source_chunk_ids_audit._parse_sqlite_uuid_list(value)
    assert len(out) == expected_len
    for item in out:
        assert isinstance(item, UUID)
