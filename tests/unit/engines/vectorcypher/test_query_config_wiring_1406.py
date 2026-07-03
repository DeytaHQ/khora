"""#1406 - query-tuning knobs reach the executed VectorCypher retriever config.

``query.vector_weight`` / ``graph_weight``, ``recency_weight`` /
``recency_decay_days``, ``keyword_weight`` and ``min_chunk_similarity`` were
silently inert on the default recall() path: ``_assemble_retriever_config``
read only ``VectorCypherConfig`` fields whose names differ from the query.*
family, so the #1017/#1330-style reconcile could never bridge them.

Contract: for each affected ``KHORA_QUERY_*`` env var, construct KhoraConfig
from the environment and assert the assembled RetrieverConfig reflects it.
No DB connections involved (``_assemble_retriever_config`` is pure).
"""

from __future__ import annotations

import pytest

from khora.config.schema import KhoraConfig
from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever


def _assembled_config(vc_config: VectorCypherConfig | None = None) -> RetrieverConfig:
    engine = VectorCypherEngine(KhoraConfig(), vectorcypher_config=vc_config)
    return engine._assemble_retriever_config()


# --------------------------------------------------------------------------- #
# Env-var contract: KHORA_QUERY_* -> assembled RetrieverConfig.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("env_var", "raw", "rc_field", "expected"),
    [
        ("KHORA_QUERY_VECTOR_WEIGHT", "0.9", "vector_weight", 0.9),
        ("KHORA_QUERY_GRAPH_WEIGHT", "0.1", "graph_weight", 0.1),
        ("KHORA_QUERY_RECENCY_WEIGHT", "0.55", "recency_weight", 0.55),
        ("KHORA_QUERY_RECENCY_DECAY_DAYS", "14", "recency_decay_days", 14.0),
        # keyword_weight fills the BM25 (lexical) fusion slot.
        ("KHORA_QUERY_KEYWORD_WEIGHT", "0.45", "bm25_weight", 0.45),
        ("KHORA_QUERY_MIN_CHUNK_SIMILARITY", "0.2", "min_chunk_similarity", 0.2),
    ],
)
def test_query_env_var_reaches_retriever_config(
    monkeypatch: pytest.MonkeyPatch, env_var: str, raw: str, rc_field: str, expected: float
) -> None:
    monkeypatch.setenv(env_var, raw)
    rc = _assembled_config()
    assert getattr(rc, rc_field) == expected


# --------------------------------------------------------------------------- #
# Precedence: a caller-supplied VectorCypherConfig value beats query.*.
# --------------------------------------------------------------------------- #


def test_caller_vc_config_wins_over_query_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KHORA_QUERY_VECTOR_WEIGHT", "0.9")
    monkeypatch.setenv("KHORA_QUERY_RECENCY_DECAY_DAYS", "21")
    rc = _assembled_config(VectorCypherConfig(fusion_vector_weight=0.55, temporal_recency_decay_days=3.0))
    assert rc.vector_weight == 0.55
    assert rc.recency_decay_days == 3.0


def test_vc_config_left_at_default_takes_query_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed VectorCypherConfig that does NOT touch a field still gets the
    query.* value for it (the #1017 partial-reconcile semantics)."""
    monkeypatch.setenv("KHORA_QUERY_GRAPH_WEIGHT", "0.15")
    rc = _assembled_config(VectorCypherConfig(fusion_vector_weight=0.55))
    assert rc.graph_weight == 0.15


# --------------------------------------------------------------------------- #
# Default-divergence resolution: one canonical default across all tiers.
# --------------------------------------------------------------------------- #


def test_defaults_are_canonical_across_tiers() -> None:
    """QuerySettings, VectorCypherConfig, and the assembled RetrieverConfig
    agree at defaults - no silent behavior change from the #1406 wiring.

    Canonical values: fusion 0.6/0.4 (the engine's previous effective
    behavior), recency 0.35/7 (the post-BEAM-100k QuerySettings values,
    previously shadowed), lexical/BM25 weight 0.3, chunk floor 0.0 (no floor;
    the previous effective behavior - a 0.05 default would deterministically
    drop ~2/3 of hash-embedding mock chunks in the embedded test lanes, and
    activating a floor is an opt-in accuracy experiment, not this wiring fix).
    """
    q = KhoraConfig().query
    vc = VectorCypherConfig()
    rc = _assembled_config()

    assert q.vector_weight == vc.fusion_vector_weight == rc.vector_weight == 0.6
    assert q.graph_weight == vc.fusion_graph_weight == rc.graph_weight == 0.4
    assert q.recency_weight == vc.temporal_recency_weight == rc.recency_weight == 0.35
    assert q.recency_decay_days == vc.temporal_recency_decay_days == rc.recency_decay_days == 7.0
    assert q.keyword_weight == vc.bm25_weight == rc.bm25_weight == 0.3
    assert q.min_chunk_similarity == rc.min_chunk_similarity == 0.0


# --------------------------------------------------------------------------- #
# min_chunk_similarity floor: per-call min_similarity resolution.
# --------------------------------------------------------------------------- #


def _floor_retriever(min_chunk_similarity: float) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(min_chunk_similarity=min_chunk_similarity)
    return retriever


def test_unset_min_similarity_falls_back_to_configured_floor() -> None:
    assert _floor_retriever(0.05)._effective_min_similarity(0.0) == 0.05


def test_explicit_min_similarity_wins_over_configured_floor() -> None:
    assert _floor_retriever(0.05)._effective_min_similarity(0.3) == 0.3


def test_floor_disabled_when_config_is_zero() -> None:
    assert _floor_retriever(0.0)._effective_min_similarity(0.0) == 0.0
