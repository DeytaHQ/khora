"""Unit tests for the vectorcypher centroid-recompute op (#660)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Entity
from khora.dream.engines.vectorcypher import plan_vectorcypher_centroid_recompute
from khora.dream.plan import OpKind


@dataclass
class _FakeCoordinator:
    entities: list[Entity] = field(default_factory=list)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        del namespace_id, entity_type, limit, offset
        return list(self.entities)


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def _entity(ns: UUID, name: str, vec: list[float], *, mention_count: int = 1) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns,
        name=name,
        entity_type="ORGANIZATION",
        embedding=_norm(vec),
        mention_count=mention_count,
    )


@pytest.mark.asyncio
async def test_centroid_path_for_lexically_close_names() -> None:
    """Two near-identical names with similar embeddings → 'centroid'."""
    ns = uuid4()
    a = _entity(ns, "OpenAI", [1.0, 0.05, 0.0], mention_count=3)
    b = _entity(ns, "Open AI", [0.98, 0.07, 0.0], mention_count=1)
    coord = _FakeCoordinator(entities=[a, b])

    (op,) = await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id, b.id]])

    assert op.op_type == OpKind.VECTORCYPHER_CENTROID_RECOMPUTE
    assert op.decision == "centroid"
    assert len(op.outputs) == 1
    payload = op.outputs[0]
    vec = payload["new_embedding_vector"]
    # L2-normalised (dot-product semantics with pre-normalised store).
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-5)
    assert payload["source_count"] == 2
    # Canonical entity is the one with highest mention_count.
    assert op.inputs[0]["canonical_entity_id"] == str(a.id)


@pytest.mark.asyncio
async def test_re_embed_path_for_lexically_distant_synonyms() -> None:
    """Lexically distant names but high cosine → 're_embed' canonical name."""
    ns = uuid4()
    # Identical embeddings → cosine 1.0 → not multimodal.
    # Lev("IBM", "International Business Machines") = 28 → above threshold.
    a = _entity(ns, "IBM", [0.6, 0.8, 0.0], mention_count=10)
    b = _entity(ns, "International Business Machines", [0.6, 0.8, 0.0], mention_count=2)
    coord = _FakeCoordinator(entities=[a, b])

    (op,) = await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id, b.id]])

    assert op.decision == "re_embed"
    payload = op.outputs[0]
    # Canonical name = highest mention_count.
    assert payload["new_embedding_text"] == "IBM"
    assert payload["embedding_model"] == "text-embedding-3-small"
    assert payload["source_count"] == 2
    # No embedding vector was computed on this path (deferred to v0.15).
    assert "new_embedding_vector" not in payload


@pytest.mark.asyncio
async def test_skip_multimodal_when_intra_cluster_cosine_below_floor() -> None:
    """Intra-cluster cosine < 0.88 → 'skip_multimodal', no outputs."""
    ns = uuid4()
    # Orthogonal embeddings → cosine 0.0.
    a = _entity(ns, "Apple", [1.0, 0.0, 0.0], mention_count=5)
    b = _entity(ns, "Apple", [0.0, 1.0, 0.0], mention_count=5)
    coord = _FakeCoordinator(entities=[a, b])

    (op,) = await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id, b.id]])

    assert op.decision == "skip_multimodal"
    assert op.outputs == ()
    assert "multimodal" in op.rationale.lower()


@pytest.mark.asyncio
async def test_apply_mode_raises_not_implemented() -> None:
    """v0.14 dry-run only — apply mode must raise."""
    ns = uuid4()
    coord = _FakeCoordinator(entities=[])

    with pytest.raises(NotImplementedError, match="apply mode lands in v0.15"):
        await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[], mode="apply")


@pytest.mark.asyncio
async def test_no_writes_to_coordinator() -> None:
    """The op must not mutate the coordinator's entity set."""
    ns = uuid4()
    a = _entity(ns, "OpenAI", [1.0, 0.0, 0.0], mention_count=3)
    b = _entity(ns, "Open AI", [0.99, 0.01, 0.0], mention_count=1)
    coord = _FakeCoordinator(entities=[a, b])
    before = list(coord.entities)
    snapshot = [(e.id, e.name, list(e.embedding or []), e.mention_count) for e in before]

    await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id, b.id]])

    after = [(e.id, e.name, list(e.embedding or []), e.mention_count) for e in coord.entities]
    assert after == snapshot


@pytest.mark.asyncio
async def test_dream_op_round_trips_json() -> None:
    """Outputs are JSON-serialisable for the sink layer."""
    ns = uuid4()
    a = _entity(ns, "OpenAI", [1.0, 0.05, 0.0], mention_count=3)
    b = _entity(ns, "Open AI", [0.98, 0.07, 0.0], mention_count=1)
    coord = _FakeCoordinator(entities=[a, b])

    (op,) = await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id, b.id]])

    payload = json.dumps({"inputs": list(op.inputs), "outputs": list(op.outputs)})
    restored = json.loads(payload)
    assert "new_embedding_vector" in restored["outputs"][0]
    assert restored["inputs"][0]["cluster_size"] == 2


@pytest.mark.asyncio
async def test_skip_singleton_when_cluster_has_fewer_than_two_members() -> None:
    """A 1-member cluster is a no-op — decision='skip_singleton'."""
    ns = uuid4()
    a = _entity(ns, "OpenAI", [1.0, 0.0, 0.0], mention_count=3)
    coord = _FakeCoordinator(entities=[a])

    (op,) = await plan_vectorcypher_centroid_recompute(ns, coordinator=coord, merge_clusters=[[a.id]])

    assert op.decision == "skip_singleton"
    assert op.outputs == ()


@pytest.mark.asyncio
async def test_multiple_clusters_emit_one_op_each() -> None:
    """One DreamOp is emitted per input cluster."""
    ns = uuid4()
    a1 = _entity(ns, "OpenAI", [1.0, 0.0, 0.0], mention_count=2)
    a2 = _entity(ns, "Open AI", [0.99, 0.05, 0.0], mention_count=1)
    b1 = _entity(ns, "Apple", [1.0, 0.0, 0.0], mention_count=4)
    b2 = _entity(ns, "Apple", [0.0, 1.0, 0.0], mention_count=4)
    coord = _FakeCoordinator(entities=[a1, a2, b1, b2])

    ops = await plan_vectorcypher_centroid_recompute(
        ns, coordinator=coord, merge_clusters=[[a1.id, a2.id], [b1.id, b2.id]]
    )

    assert len(ops) == 2
    assert ops[0].decision == "centroid"
    assert ops[1].decision == "skip_multimodal"
