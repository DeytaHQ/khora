"""Unit tests for Skeleton engine kwarg-drift refusal (#890, #891).

Pre-fix, ``SkeletonConstructionEngine.remember`` silently accepted
``entity_types`` / ``relationship_types`` (it has no entity extraction)
and ``recall`` silently accepted ``recency_bias`` (it has no temporal
decay). Both are now refused via ``UnsupportedEngineKwargError`` so
the caller is forced to either drop the kwarg or pick an engine that
honors it.

The tests exercise the validation branch directly - the engine never
touches storage when the refusal fires, so we can drive ``remember`` /
``recall`` with shallow stubs and skip ``connect()``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.engine import SkeletonConstructionEngine
from khora.exceptions import UnsupportedEngineKwargError
from khora.query import SearchMode


def _build_engine_with_stubs() -> tuple[SkeletonConstructionEngine, AsyncMock, AsyncMock]:
    """Construct an engine with embedder + temporal store + storage stubs.

    Mirrors ``test_skeleton_search_mode._build_engine_with_stubs`` but
    also wires a ``_storage`` stub because the refusal in ``remember``
    happens before storage is touched - we still keep the stub so a
    regression that orders things wrong fails loudly instead of running
    real I/O.
    """
    cfg = MagicMock()
    cfg.storage.backend = "pgvector"

    engine = SkeletonConstructionEngine.__new__(SkeletonConstructionEngine)
    engine._config = cfg
    engine._backend_type = "pgvector"
    engine._weaviate_url = None
    engine._storage_config = MagicMock()
    engine._connected = True

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    engine._embedder = embedder

    temporal_store = AsyncMock()
    temporal_store.search = AsyncMock(return_value=[])
    engine._temporal_store = temporal_store

    storage = AsyncMock()
    # If the refusal-order regresses and we reach storage, this would
    # raise rather than silently succeed.
    storage.get_document_by_checksum = AsyncMock(
        side_effect=AssertionError("storage must not be touched when remember refuses kwargs"),
    )
    storage.create_document = AsyncMock(
        side_effect=AssertionError("storage must not be touched when remember refuses kwargs"),
    )
    engine._storage = storage

    return engine, temporal_store, storage


# ---------------------------------------------------------------------------
# #890: remember() refuses non-empty entity_types / relationship_types
# ---------------------------------------------------------------------------


async def test_skeleton_remember_raises_on_entity_types_kwarg() -> None:
    """Non-empty ``entity_types`` raises ``UnsupportedEngineKwargError``.

    Skeleton has no entity extraction; previously the kwarg was silently
    dropped (#890) - callers saw ``entities_extracted=0`` and assumed
    no entities matched the whitelist. Now the engine refuses up front.
    """
    engine, _temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError) as excinfo:
        await engine.remember(
            "alpha beta gamma",
            namespace_id,
            entity_types=["PERSON", "ORG"],
            relationship_types=[],
        )
    assert excinfo.value.engine_name == "skeleton"
    assert excinfo.value.kwarg == "entity_types"
    assert "extraction" in excinfo.value.reason.lower()


async def test_skeleton_remember_raises_on_relationship_types_kwarg() -> None:
    """Non-empty ``relationship_types`` raises ``UnsupportedEngineKwargError``.

    Symmetric to ``entity_types`` (#890). The relationship-types branch
    has its own check so callers see exactly which kwarg the engine
    refused.
    """
    engine, _temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError) as excinfo:
        await engine.remember(
            "alpha beta gamma",
            namespace_id,
            entity_types=[],
            relationship_types=["WORKS_FOR"],
        )
    assert excinfo.value.engine_name == "skeleton"
    assert excinfo.value.kwarg == "relationship_types"
    assert "extraction" in excinfo.value.reason.lower()


async def test_skeleton_remember_accepts_empty_type_lists() -> None:
    """Empty lists are still the documented happy-path contract.

    ``Khora.remember`` always forwards ``entity_types`` /
    ``relationship_types``; callers targeting Skeleton must hand in
    empty lists. The refusal must not trip on the protocol-default
    empty-list case, otherwise every Skeleton ingest fails.
    """
    engine, _temporal_store, storage = _build_engine_with_stubs()
    # Replace storage stub with one that lets the call proceed - we
    # only care that the validator does NOT raise on empty lists.
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    # Short-circuit _process_document so we exit cleanly without
    # exercising the full pipeline.
    storage.create_document = AsyncMock(
        side_effect=AssertionError("expected to short-circuit before create_document"),
    )
    # Patch _process_document on the instance to skip the rest.
    engine._process_document = AsyncMock(return_value=(0, 0, 0))  # type: ignore[method-assign]
    # storage.create_document is still wired but the engine calls it
    # before _process_document; patch storage.create_document to return
    # a Document instead. Build a minimal stand-in.
    from datetime import UTC, datetime

    from khora.core.models import Document

    doc = Document(
        namespace_id=uuid4(),
        content="alpha",
        checksum="x",
        size_bytes=5,
        metadata={},
        created_at=datetime.now(UTC),
    )
    storage.create_document = AsyncMock(return_value=doc)

    # Should NOT raise.
    result = await engine.remember(
        "alpha beta gamma",
        uuid4(),
        entity_types=[],
        relationship_types=[],
    )
    assert result is not None
    assert result.entities_extracted == 0


# ---------------------------------------------------------------------------
# #1431: remember() refuses non-None expertise
# ---------------------------------------------------------------------------


async def test_skeleton_remember_raises_on_expertise_kwarg() -> None:
    """Non-None ``expertise`` raises ``UnsupportedEngineKwargError``.

    ``expertise`` is the third ontology-guidance kwarg in the same family
    as ``entity_types`` / ``relationship_types``. #890 made those two
    loud; ``expertise`` stayed silently accepted-and-ignored (#1431).
    """
    from khora.extraction.skills.base import EntityTypeConfig, ExpertiseConfig

    engine, _temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError) as excinfo:
        await engine.remember(
            "alpha beta gamma",
            namespace_id,
            entity_types=[],
            relationship_types=[],
            expertise=ExpertiseConfig(name="physics", entity_types=[EntityTypeConfig(name="PARTICLE")]),
        )
    assert excinfo.value.engine_name == "skeleton"
    assert excinfo.value.kwarg == "expertise"
    assert "extraction" in excinfo.value.reason.lower()


async def test_skeleton_remember_raises_on_string_expertise() -> None:
    """A string expertise (registered name / YAML path) is also refused.

    The signature accepts ``ExpertiseConfig | str | None``; any non-None
    form asks for ontology-guided extraction the engine cannot honor.
    """
    engine, _temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError) as excinfo:
        await engine.remember(
            "alpha beta gamma",
            namespace_id,
            entity_types=[],
            relationship_types=[],
            expertise="lead_intel",
        )
    assert excinfo.value.kwarg == "expertise"


# ---------------------------------------------------------------------------
# #891: recall() refuses non-None recency_bias
# ---------------------------------------------------------------------------


async def test_skeleton_recall_raises_on_recency_bias() -> None:
    """Non-None ``recency_bias`` raises ``UnsupportedEngineKwargError``.

    Pre-fix Skeleton accepted ``recency_bias`` and silently ignored it
    (#891). Now the engine refuses any concrete value so the caller
    either drops the kwarg (Skeleton, no decay) or routes to an engine
    that actually applies decay (Chronicle).
    """
    engine, temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError) as excinfo:
        await engine.recall(
            "alpha",
            namespace_id,
            mode=SearchMode.VECTOR,
            recency_bias=0.5,
        )
    assert excinfo.value.engine_name == "skeleton"
    assert excinfo.value.kwarg == "recency_bias"
    # Storage must not be touched on refusal.
    temporal_store.search.assert_not_awaited()


async def test_skeleton_recall_recency_bias_none_is_silent_no_op() -> None:
    """``recency_bias=None`` (the default) must NOT raise.

    Skeleton has to accept ``None`` for protocol parity - callers that
    don't pass the kwarg get ``None`` by default and should not be
    forced to drop the kwarg from their routing layer just because the
    engine doesn't honor it.
    """
    engine, temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    # Should not raise.
    result = await engine.recall(
        "alpha",
        namespace_id,
        mode=SearchMode.VECTOR,
        recency_bias=None,
    )
    assert result is not None
    temporal_store.search.assert_awaited_once()


async def test_skeleton_recall_zero_recency_bias_still_raises() -> None:
    """A zero ``recency_bias`` is non-None and must raise.

    Zero is a meaningful value (no decay) - the engine could in principle
    accept it as a happy-path special case, but DA's directive is "raise
    on non-None" for clarity. The test pins that behavior so a future
    contributor doesn't quietly broaden the accepted set.
    """
    engine, temporal_store, _storage = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(UnsupportedEngineKwargError):
        await engine.recall(
            "alpha",
            namespace_id,
            mode=SearchMode.VECTOR,
            recency_bias=0.0,
        )
    temporal_store.search.assert_not_awaited()
