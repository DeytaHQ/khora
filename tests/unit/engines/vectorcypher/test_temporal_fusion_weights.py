"""Temporal-branch fusion weights are config-reachable (plumbing only).

``RetrieverConfig.temporal_vector_weight`` / ``temporal_graph_weight`` were
always live (the temporally-detected branch of ``_fuse_results`` swaps to
them) but config-unreachable: ``VectorCypherConfig`` had no corresponding
fields and ``_assemble_retriever_config`` never mapped them, so every
deployment ran the hardcoded 0.3/0.7 defaults (surfaced by the
khora-benchmarks deyta_multisource fusion review - a "vector-only" ablation
config still ran graph-heavy 0.3/0.7 fusion on temporal queries).

Contract: defaults stay byte-identical (0.3/0.7 end to end); explicit
``fusion_temporal_*`` values reach the assembled RetrieverConfig. Pure
config-assembly tests, no DB.
"""

from __future__ import annotations

from khora.config.schema import KhoraConfig
from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine
from khora.engines.vectorcypher.retriever import RetrieverConfig


def _assembled_config(vc_config: VectorCypherConfig | None = None) -> RetrieverConfig:
    engine = VectorCypherEngine(KhoraConfig(), vectorcypher_config=vc_config)
    return engine._assemble_retriever_config()


def test_defaults_byte_identical() -> None:
    """Default config produces exactly the previous hardcoded behavior."""
    rc = _assembled_config()
    defaults = RetrieverConfig()
    assert rc.temporal_vector_weight == defaults.temporal_vector_weight == 0.3
    assert rc.temporal_graph_weight == defaults.temporal_graph_weight == 0.7
    vc = VectorCypherConfig()
    assert vc.fusion_temporal_vector_weight == 0.3
    assert vc.fusion_temporal_graph_weight == 0.7


def test_explicit_values_reach_retriever_config() -> None:
    rc = _assembled_config(VectorCypherConfig(fusion_temporal_vector_weight=1.0, fusion_temporal_graph_weight=0.0))
    assert rc.temporal_vector_weight == 1.0
    assert rc.temporal_graph_weight == 0.0
