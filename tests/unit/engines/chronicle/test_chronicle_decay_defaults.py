"""Chronicle #851 / #853: regression-pin the user-visible decay defaults.

These tests guard against silent drift between the constants in
``chronicle.engine`` and the pydantic field defaults in ``config.schema``,
which were out of sync prior to v0.17.3 (#851 had three places disagreeing
between 24h and 168h; #853 had 0.10 vs 0.30).
"""

from __future__ import annotations

from khora.config import KhoraConfig
from khora.engines.chronicle.engine import (
    DEFAULT_CHRONICLE_DECAY_WEIGHT,
    DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
)


def test_default_chronicle_half_life_is_168_hours() -> None:
    cfg = KhoraConfig()
    assert cfg.query.temporal_half_life_hours == 168.0
    assert cfg.query.temporal_half_life_hours == DEFAULT_CHRONICLE_HALF_LIFE_HOURS


def test_default_chronicle_decay_weight_is_0_30() -> None:
    cfg = KhoraConfig()
    assert cfg.query.chronicle_decay_weight == 0.30
    assert cfg.query.chronicle_decay_weight == DEFAULT_CHRONICLE_DECAY_WEIGHT


def test_vectorcypher_queryconfig_half_life_matches_chronicle() -> None:
    """Pins the second duplicated literal: QueryConfig dataclass default in
    `khora.query.engine`. Devil's-advocate caught this site at v0.17.3 - it
    must stay aligned with `DEFAULT_CHRONICLE_HALF_LIFE_HOURS`.
    """
    from khora.query.engine import QueryConfig

    assert QueryConfig.__dataclass_fields__["temporal_half_life_hours"].default == 168.0
    assert QueryConfig.__dataclass_fields__["temporal_half_life_hours"].default == DEFAULT_CHRONICLE_HALF_LIFE_HOURS


def test_soft_temporal_score_default_matches_chronicle() -> None:
    """Pins the third duplicated literal: `_soft_temporal_score`'s function
    default. Without this guard the VectorCypher recall path silently
    decoupled from chronicle's half-life - see v0.17.3 devil's-advocate
    review.
    """
    import inspect

    from khora.query.engine import HybridQueryEngine

    sig = inspect.signature(HybridQueryEngine._soft_temporal_score)
    assert sig.parameters["half_life_hours"].default == 168.0
    assert sig.parameters["half_life_hours"].default == DEFAULT_CHRONICLE_HALF_LIFE_HOURS
