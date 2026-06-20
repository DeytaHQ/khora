"""sqlite_lance parity for the checkpoint / resume path (#1274).

#896 made ``dream_history`` / ``dream_status`` persist a run row on the
embedded SQLite stack, but the resume cursor (``last_committed_op_seq``)
was still gated to Postgres - on sqlite_lance the checkpoint never
advanced, so a crashed APPLY pass could not resume. The DreamRunStore
abstraction routes run-state through a SQLite-backed store on any non-PG
SQL stack, so the checkpoint now advances and ``resume_from`` works.

These tests drive a real dream run on the embedded stack and assert the
orchestrator's run-state store advances the checkpoint and exposes the
``graph_mirror_pending`` accessors the #1272 reconciler builds on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402
from khora.dream.orchestrator import DreamOrchestrator  # noqa: E402
from khora.dream.runstore import GraphMirrorPending  # noqa: E402

pytestmark = pytest.mark.embedded


async def _remember(kb, namespace_id):
    return await kb.remember(
        "PostgreSQL was chosen for the user database.",
        namespace=namespace_id,
        entity_types=[],
        relationship_types=[],
    )


async def test_checkpoint_advances_on_sqlite_lance() -> None:
    """The resume cursor persists on the embedded SQLite stack."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
        store = orch._run_store()

        run_id = uuid4()
        await store.record_run(run_id, ns.namespace_id, mode="apply")
        assert await store.read_last_committed(run_id) == -1, "checkpoint not initialized on sqlite_lance"

        await store.advance_checkpoint(run_id, 0)
        assert await store.read_last_committed(run_id) == 0, "checkpoint did not advance on sqlite_lance"

        await store.advance_checkpoint(run_id, 3)
        assert await store.read_last_committed(run_id) == 3


async def test_graph_mirror_pending_on_sqlite_lance() -> None:
    """graph_mirror_pending set/get/clear round-trips on the embedded stack."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
        store = orch._run_store()

        run_id = uuid4()
        await store.record_run(run_id, ns.namespace_id, mode="apply")
        op = uuid4()
        await store.mark_graph_mirror_pending(
            run_id,
            GraphMirrorPending(op_seq=0, op_id=op, op_type="prune_edges", payload={"edge_ids": [7]}),
        )

        pending = await store.get_graph_mirror_pending(run_id)
        assert len(pending) == 1
        assert pending[0].op_id == op
        assert pending[0].payload == {"edge_ids": [7]}

        await store.clear_graph_mirror_pending(run_id, 0)
        assert await store.get_graph_mirror_pending(run_id) == []


async def test_open_pending_by_namespace_on_sqlite_lance() -> None:
    """#1292: the namespace-scoped drain query spans runs on the embedded stack."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
        store = orch._run_store()

        run_a = uuid4()
        run_b = uuid4()
        await store.record_run(run_a, ns.namespace_id, mode="apply")
        await store.record_run(run_b, ns.namespace_id, mode="apply")
        await store.mark_graph_mirror_pending(
            run_a, GraphMirrorPending(op_seq=0, op_id=uuid4(), op_type="vectorcypher_prune_edges", payload={"a": 1})
        )
        await store.mark_graph_mirror_pending(
            run_b, GraphMirrorPending(op_seq=1, op_id=uuid4(), op_type="vectorcypher_dedupe_entities", payload={"b": 2})
        )

        open_pending = await store.get_open_graph_mirror_pending(ns.namespace_id)
        assert {rid for rid, _ in open_pending} == {run_a, run_b}


async def test_dream_history_still_works_on_sqlite_lance() -> None:
    """Regression: the #896 history/status path is unchanged by the store."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        result = await kb.dream(ns.namespace_id, mode="dry-run", config=DreamConfig(enabled=True))
        run_id = result.run.run_id

        history = await kb.dream_history(ns.namespace_id)
        assert len(history) >= 1
        assert history[0].run_id == run_id

        status = await kb.dream_status(run_id)
        assert status
        assert status["run_id"] == str(run_id)
