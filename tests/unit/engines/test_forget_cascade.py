"""Unit tests for the forget-cascade malformed-orphan sweep (#1237).

A relationship whose endpoint node is missing its id cannot be deserialized
by ``list_relationships`` (skip-and-warn), so its id never reaches the
enumerate-and-delete path and it would survive ``forget()``. The cascade now
sweeps such edges directly in the graph store and surfaces a ``Degradation``.

The sweep is gated on the namespace having ≥1 deserializable relationship
(#1241), so the fake graph below returns one unrelated edge to exercise it.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

import khora.engines._forget_cascade as fc
from khora.engines._forget_cascade import cascade_forget_extraction


class _FakeGraph:
    """Graph backend exposing the cascade primitives + the sweep hook."""

    def __init__(self, *, swept: int, has_sweep: bool = True) -> None:
        self._swept = swept
        self.sweep_calls: list[tuple] = []
        # One relationship unrelated to the forgotten doc: keeps the namespace
        # non-empty (so the #1241 early-out lets the sweep run) without being
        # classified as an orphan/survivor of the document being forgotten.
        self._rels = [SimpleNamespace(id=uuid4(), source_document_ids=[uuid4()])]
        if has_sweep:
            self.delete_malformed_orphan_relationships = self._sweep  # type: ignore[attr-defined]

    async def list_entities(self, namespace_id, limit=0):
        return []

    async def list_relationships(self, namespace_id, limit=0):
        return list(self._rels)

    async def delete_entities_batch(self, ids, *, namespace_id):
        return None

    async def delete_relationships_batch(self, ids, *, namespace_id):
        return None

    async def _sweep(self, document_id, *, namespace_id):
        self.sweep_calls.append((document_id, namespace_id))
        return self._swept


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sweep_records_degradation_when_malformed_edges_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id, ns_id = uuid4(), uuid4()
    graph = _FakeGraph(swept=3)

    counter_calls: list[tuple] = []
    monkeypatch.setattr(
        fc,
        "_FORGET_DEGRADED_COUNTER",
        SimpleNamespace(add=lambda value, attrs: counter_calls.append((value, attrs))),
    )

    degradations = await cascade_forget_extraction(
        graph=graph,
        vector=None,
        document_id=doc_id,
        namespace_id=ns_id,
        engine="vectorcypher",
    )

    assert graph.sweep_calls == [(doc_id, ns_id)]
    # Counter incremented by the swept magnitude (3), labelled by reason only.
    assert counter_calls == [(3, {"reason": "malformed_relationship_swept"})]
    assert len(degradations) == 1
    deg = degradations[0]
    assert deg["component"] == "forget_cascade"
    assert deg["reason"] == "malformed_relationship_swept"
    assert "3 orphan relationship" in (deg["detail"] or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_degradation_when_nothing_swept() -> None:
    graph = _FakeGraph(swept=0)

    degradations = await cascade_forget_extraction(
        graph=graph,
        vector=None,
        document_id=uuid4(),
        namespace_id=uuid4(),
        engine="vectorcypher",
    )

    assert graph.sweep_calls  # sweep still attempted (namespace had relationships)
    assert degradations == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backend_without_sweep_hook_is_a_noop() -> None:
    """Non-Neo4j graph backends lack the sweep method → getattr miss, no error."""
    graph = _FakeGraph(swept=99, has_sweep=False)

    degradations = await cascade_forget_extraction(
        graph=graph,
        vector=None,
        document_id=uuid4(),
        namespace_id=uuid4(),
        engine="chronicle",
    )

    assert degradations == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sweep_skipped_when_namespace_has_no_relationships() -> None:
    """#1241 early-out: a namespace with zero deserializable edges skips the
    sweep (the common forget_session/expire_sessions case), avoiding the
    unindexed full-relationship scan."""
    graph = _FakeGraph(swept=5)
    graph._rels = []  # namespace has no deserializable relationships

    degradations = await cascade_forget_extraction(
        graph=graph,
        vector=None,
        document_id=uuid4(),
        namespace_id=uuid4(),
        engine="vectorcypher",
    )

    assert graph.sweep_calls == []  # sweep not attempted
    assert degradations == []
