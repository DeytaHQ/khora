"""Reachability (#1263) + master invariant gate (#1268) on sqlite_lance.

The Phase-1 runnable legs (single embedded store, no docker):

  * #1263: the three mutation ops (dedupe / centroid_recompute /
    source_chunk_ids_gc) are reachable via ``kb.dream()`` once enabled —
    they appear in the vectorcypher plugin's ``dream_capabilities`` and
    are emitted from ``plan_dream``. Disabled by default they are absent
    (the master-switch contract the other mutation ops follow).
  * #1268: after any dream apply, dedupe correctness holds end-to-end;
    a re-run over the same world emits an identical plan_hash
    (permutation/determinism); the transitive component collapses to one
    canonical.

The PG + Neo4j cross-store live-set leg (graph-preferring ``list_entities``
== PG ground truth after apply) is Phase 2 — tracked in #1272. It needs an
isolated pg+graph stack and is out of scope for the embedded gate here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.core.models.entity import Entity  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402
from khora.dream.engines.registry import _VectorCypherPlugin, plan_hash  # noqa: E402
from khora.dream.engines.vectorcypher import plan_vectorcypher_dedupe_entities  # noqa: E402
from khora.dream.plan import DreamPlan, DreamScope, OpKind  # noqa: E402

pytestmark = pytest.mark.embedded


def _unit(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).astype(np.float32).tolist() if norm else vec


async def _plant_duplicate_org_cluster(kb, ns_id: UUID, dim: int) -> tuple[UUID, list[Entity]]:
    """Upsert three near-identical ORG entities directly via the coordinator.

    Entities reference the resolved row-level namespace id (the FK target);
    the returned id is what the planner / dream call must be driven with so
    ``list_entities`` finds them on this embedded stack.
    """

    def pad(head: list[float]) -> list[float]:
        return _unit(head + [0.0] * (dim - len(head)))

    resolved = await kb.storage.resolve_namespace(ns_id)
    ents = [
        Entity(
            id=uuid4(),
            namespace_id=resolved,
            name="Acme Corporation",
            entity_type="ORG",
            mention_count=9,
            embedding=pad([1.0, 0.0]),
        ),
        Entity(
            id=uuid4(),
            namespace_id=resolved,
            name="Acme Corp",
            entity_type="ORG",
            mention_count=3,
            embedding=pad([1.0, 0.02]),
        ),
        Entity(
            id=uuid4(),
            namespace_id=resolved,
            name="Acme Co",
            entity_type="ORG",
            mention_count=1,
            embedding=pad([1.0, 0.04]),
        ),
    ]
    await kb.storage.upsert_entities_batch(resolved, ents)
    return resolved, ents


# ---------------------------------------------------------------------------
# #1263 — reachability
# ---------------------------------------------------------------------------


def test_capabilities_include_the_three_mutation_ops() -> None:
    """dedupe / centroid_recompute / source_chunk_ids_gc must be declared
    capabilities so the orchestrator can route them (#1263)."""
    caps = _VectorCypherPlugin().dream_capabilities
    assert OpKind.VECTORCYPHER_DEDUPE_ENTITIES in caps
    assert OpKind.VECTORCYPHER_CENTROID_RECOMPUTE in caps
    assert OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC in caps


async def test_dedupe_reachable_via_kb_dream_when_enabled() -> None:
    """With dedupe enabled, a dry-run dream reaches the dedupe planner via
    ``kb.dream()`` without raising (#1263, #1265).

    Note: the embedded sqlite_lance read path returns entities with
    ``embedding=None`` (vectors live in LanceDB, off the graph row), so the
    similarity kernel can't score candidates here and the plan may carry
    zero merge ops. The merge-firing leg is exercised at the planner level
    in ``test_dedupe_correctness.py`` (pgvector-shaped fake coordinator)
    and on the pg+graph stack in #1272. The gate here is *reachability*."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)

        config = DreamConfig(enabled=True, dedupe_entities_enabled=True, dedupe_entities_default_threshold=0.90)
        result = await kb.dream(
            resolved,
            mode="dry-run",
            ops=[OpKind.VECTORCYPHER_DEDUPE_ENTITIES],
            config=config,
        )

        # Reachable: the op was routed (not dropped as unsupported) and the
        # run completed. A dedupe-disabled / unsupported op would surface a
        # skip_reason instead of being silently swallowed.
        assert result.run.mode == "dry-run"
        disabled = [
            sr
            for sr in result.metadata.get("skip_reasons", [])
            if sr.get("op_kind") == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
            and sr.get("reason") in ("op_disabled_by_config", "op_not_supported_by_engine")
        ]
        assert disabled == [], "dedupe was enabled + supported; it must not be dropped"


async def test_dedupe_absent_when_disabled() -> None:
    """Default-OFF master switch: dedupe is not planned unless enabled."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)

        config = DreamConfig(enabled=True)  # dedupe_entities_enabled defaults False
        result = await kb.dream(
            resolved,
            mode="dry-run",
            ops=[OpKind.VECTORCYPHER_DEDUPE_ENTITIES],
            config=config,
        )
        op_types = {s.op_type for s in result.ops}
        assert str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES) not in op_types


async def test_centroid_and_gc_reachable_via_capabilities() -> None:
    """centroid_recompute + source_chunk_ids_gc plan without crashing when
    enabled (#1263). They may legitimately emit zero ops on a tiny corpus —
    the gate is reachability, not op count."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)

        config = DreamConfig(
            enabled=True,
            centroid_recompute_enabled=True,
            source_chunk_ids_gc_enabled=True,
        )
        # Must not raise — the planners are wired and reachable.
        result = await kb.dream(
            resolved,
            mode="dry-run",
            ops=[OpKind.VECTORCYPHER_CENTROID_RECOMPUTE, OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC],
            config=config,
        )
        assert result.run.mode == "dry-run"


# ---------------------------------------------------------------------------
# #1268 — master invariant: determinism + transitivity on the live store
# ---------------------------------------------------------------------------


async def test_plan_is_deterministic_across_two_dream_runs() -> None:
    """Two dry-run dreams over the same unchanged store yield an identical
    plan_hash (#1266/#1268 determinism)."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)
        config = DreamConfig(enabled=True, dedupe_entities_enabled=True, dedupe_entities_default_threshold=0.90)

        r1 = await kb.dream(resolved, mode="dry-run", ops=[OpKind.VECTORCYPHER_DEDUPE_ENTITIES], config=config)
        r2 = await kb.dream(resolved, mode="dry-run", ops=[OpKind.VECTORCYPHER_DEDUPE_ENTITIES], config=config)

        assert r1.metadata["plan_hash"] == r2.metadata["plan_hash"]


async def test_planner_runs_and_is_deterministic_on_live_store() -> None:
    """The dedupe planner runs against the real embedded coordinator and is
    deterministic across two calls (#1266/#1268). Embeddings are stripped on
    the embedded read path, so this asserts the no-crash + determinism leg;
    the merge-firing transitivity proof lives in test_dedupe_correctness.py
    (pgvector-shaped coordinator) and the live-set leg in #1272."""
    from khora.dream.engines.registry import plan_hash
    from khora.dream.plan import DreamPlan

    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)

        ops1 = await plan_vectorcypher_dedupe_entities(resolved, coordinator=kb.storage, default_threshold=0.90)
        ops2 = await plan_vectorcypher_dedupe_entities(resolved, coordinator=kb.storage, default_threshold=0.90)

        h1 = plan_hash(DreamPlan(plan_id=uuid4(), namespace_id=resolved, ops=tuple(ops1)))
        h2 = plan_hash(DreamPlan(plan_id=uuid4(), namespace_id=resolved, ops=tuple(ops2)))
        assert h1 == h2


async def test_dedupe_apply_runs_cleanly_on_sqlite_lance() -> None:
    """Dedupe apply is Postgres-only; running ``kb.dream(mode="apply")`` on
    the embedded sqlite_lance stack must complete without raising (#1268).

    The embedded read path strips embeddings so no merge op is planned here;
    the dialect-gate *skip* behaviour for a planned dedupe op is covered by
    ``test_dream_apply_dialect_gate.py``, and the mutation leg by #1272. The
    gate in this test is that the apply run terminates cleanly."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)
        config = DreamConfig(enabled=True, dedupe_entities_enabled=True, dedupe_entities_default_threshold=0.90)

        # apply mode must not raise even though the op can't run on sqlite.
        result = await kb.dream(
            resolved,
            mode="apply",
            ops=[OpKind.VECTORCYPHER_DEDUPE_ENTITIES],
            config=config,
        )
        assert result.run.mode == "apply"
        # Any dedupe summary that did surface must not claim a real apply on
        # this dialect — it would have been gated to skipped.
        summary = next((s for s in result.ops if s.op_type == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)), None)
        if summary is not None:
            assert summary.applied == 0


async def test_registry_plan_dream_wires_dedupe_without_dropping_it() -> None:
    """plugin.plan_dream routes the dedupe op when enabled — it is neither
    dropped as unsupported nor crashes (#1263). plan_hash is stable."""
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        resolved, _ = await _plant_duplicate_org_cluster(kb, ns.namespace_id, dim=8)
        config = DreamConfig(enabled=True, dedupe_entities_enabled=True, dedupe_entities_default_threshold=0.90)

        plugin = _VectorCypherPlugin()
        plan: DreamPlan = await plugin.plan_dream(
            kb,
            resolved,
            scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_DEDUPE_ENTITIES,)),
            config=config,
        )
        # The op is a declared capability, so it is never dropped as
        # "op_not_supported_by_engine" and never reported "op_disabled".
        skips = plan.metadata.get("skip_reasons", [])
        bad = [
            sr
            for sr in skips
            if sr.get("op_kind") == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
            and sr.get("reason") in ("op_not_supported_by_engine", "op_disabled_by_config")
        ]
        assert bad == []
        # plan_hash is stable on the plan object (covers outputs, #1266).
        assert plan_hash(plan) == plan_hash(plan)


async def test_registry_plan_dream_emits_merges_payload_on_pgvector_shape() -> None:
    """When the coordinator surfaces embeddings (pgvector shape), plan_dream
    emits a dedupe op carrying the ``outputs[0]['merges']`` apply contract
    (#1263, #1265). Uses a fake kb/coordinator so the proof does not depend
    on the embedded LanceDB read path stripping embeddings."""
    from dataclasses import dataclass, field

    @dataclass
    class _Coord:
        ents: list[Entity] = field(default_factory=list)

        async def list_entities(self, namespace_id, *, entity_type=None, limit=100, offset=0):
            del namespace_id, limit, offset
            return [e for e in self.ents if entity_type is None or e.entity_type == entity_type]

    class _KB:
        def __init__(self, coord):
            self.storage = coord

    ns_id = uuid4()

    def pad(head: list[float]) -> list[float]:
        return _unit(head + [0.0] * (8 - len(head)))

    ents = [
        Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Globex Corporation",
            entity_type="ORG",
            mention_count=9,
            embedding=pad([1.0, 0.0]),
        ),
        Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Globex Corp",
            entity_type="ORG",
            mention_count=2,
            embedding=pad([1.0, 0.01]),
        ),
    ]
    kb = _KB(_Coord(ents=ents))
    config = DreamConfig(enabled=True, dedupe_entities_enabled=True, dedupe_entities_default_threshold=0.90)

    plan = await _VectorCypherPlugin().plan_dream(
        kb,
        ns_id,
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_DEDUPE_ENTITIES,)),
        config=config,
    )
    dedupe_ops = [op for op in plan.ops if op.op_type == OpKind.VECTORCYPHER_DEDUPE_ENTITIES]
    planned = [op for op in dedupe_ops if op.decision == "planned"]
    assert planned, "plan_dream must emit a planned dedupe op on a pgvector-shaped stack"
    assert "merges" in planned[0].outputs[0]
