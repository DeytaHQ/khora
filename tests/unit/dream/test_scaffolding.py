"""Scaffolding tests for ``khora.dream`` (#650).

Validates the type surface, env-var precedence, and that the stub API
raises :class:`NotImplementedError` with the expected message.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

import khora
from khora import DreamConfig, DreamMode, DreamResult, Khora, OpKind
from khora.config import KhoraConfig
from khora.dream import (
    DreamCapable,
    DreamDiff,
    DreamOp,
    DreamOpsConfig,
    DreamPlan,
    DreamProgress,
    DreamRunInfo,
    DreamScope,
)


def test_import_public_surface() -> None:
    """Public symbols must be importable from the top-level package."""
    assert Khora is not None
    assert DreamConfig is not None
    assert DreamResult is not None
    assert DreamMode is not None
    assert OpKind is not None
    # OpKind members are stable strings.
    assert OpKind.DEDUPE_ENTITIES.value == "dedupe_entities"


def test_dream_config_defaults() -> None:
    """A bare ``DreamConfig()`` must be safe — nothing destructive on."""
    cfg = DreamConfig()
    assert cfg.enabled is False
    assert cfg.default_mode == "dry-run"
    assert cfg.ops.dedupe_entities is False
    assert cfg.ops.prune_edges is False
    assert cfg.ops.compact_facts is False
    assert cfg.ops.cluster_events is False
    assert cfg.ops.recompute_centroids is False
    # Sinks default off.
    assert cfg.report_file_sink_enabled is False
    assert cfg.report_event_sink_enabled is False
    assert cfg.report_collector_sink_enabled is False


def test_dream_config_env_var_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-vars must populate both flat and nested DreamConfig fields."""
    monkeypatch.setenv("KHORA_DREAM_ENABLED", "true")
    monkeypatch.setenv("KHORA_DREAM_LLM_MAX_TOKENS_PER_RUN", "300000")
    monkeypatch.setenv("KHORA_DREAM_OPS__DEDUPE_ENTITIES", "true")

    cfg = KhoraConfig()
    assert cfg.dream.enabled is True
    assert cfg.dream.llm_max_tokens_per_run == 300_000
    assert cfg.dream.ops.dedupe_entities is True
    # Other ops still default off.
    assert cfg.dream.ops.prune_edges is False


def test_khora_config_nested() -> None:
    """KhoraConfig must accept a DreamConfig and round-trip cleanly."""
    cfg = KhoraConfig(dream=DreamConfig(enabled=True))
    assert cfg.dream.enabled is True
    assert cfg.dream.default_mode == "dry-run"


@pytest.mark.asyncio
async def test_dream_stubs_raise_not_implemented() -> None:
    """``Khora.dream`` must raise NotImplementedError pointing at the orchestrator ticket."""
    kb = Khora.__new__(Khora)
    # Inject a minimal config so the orchestrator constructor can pull dream out.
    kb._config = KhoraConfig()

    with pytest.raises(NotImplementedError, match="Dream orchestrator not yet wired"):
        await kb.dream(uuid4())

    with pytest.raises(NotImplementedError, match="Dream orchestrator not yet wired"):
        await kb.dream_status(uuid4())

    with pytest.raises(NotImplementedError, match="Dream orchestrator not yet wired"):
        await kb.dream_history(uuid4())


def test_internal_symbols_not_in_top_level_all() -> None:
    """Internal dataclasses must stay out of ``khora.__all__``."""
    for name in ("DreamOp", "DreamPlan", "DreamProgress", "DreamDiff", "DreamCapable"):
        assert name not in khora.__all__, f"{name} leaked into khora.__all__"


def test_public_symbols_in_top_level_all() -> None:
    """Public dataclasses must be advertised in ``khora.__all__``."""
    for name in ("DreamConfig", "DreamResult", "DreamMode", "DreamScope", "DreamRunInfo", "OpKind"):
        assert name in khora.__all__, f"{name} missing from khora.__all__"


def test_internal_symbols_importable_from_dream() -> None:
    """Internal dataclasses must still be importable from khora.dream."""
    # If any of these isn't defined, the module-level import at the top
    # of this test file would have failed. Reference them so static
    # analysis doesn't flag the imports as unused.
    assert DreamOp.__name__ == "DreamOp"
    assert DreamPlan.__name__ == "DreamPlan"
    assert DreamProgress.__name__ == "DreamProgress"
    assert DreamDiff.__name__ == "DreamDiff"
    assert DreamScope.__name__ == "DreamScope"
    assert DreamRunInfo.__name__ == "DreamRunInfo"
    assert DreamOpsConfig.__name__ == "DreamOpsConfig"
    assert DreamCapable.__name__ == "DreamCapable"
