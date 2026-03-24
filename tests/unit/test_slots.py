"""Verify slotted dataclasses reject arbitrary attributes and omit __dict__.

These models are instantiated in bulk during ingestion and retrieval (thousands
of Chunk / Entity / Relationship objects per recall).  The slots=True
optimisation saves ~40-60 bytes per instance.  If someone accidentally drops
it, these tests catch the regression via observable behaviour — not by
inspecting the class attribute.
"""

from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, ChunkMetadata, DocumentMetadata
from khora.core.models.entity import Entity, Episode, Relationship
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats

# ── helpers ──────────────────────────────────────────────────────────

# High-frequency models that are allocated in hot loops.
_HOT_PATH_CLASSES = [
    DocumentMetadata,
    ChunkMetadata,
    Chunk,
    Entity,
    Relationship,
    Episode,
]

# Frozen result types returned from public API methods.
_FROZEN_RESULT_CLASSES = [
    RememberResult,
    BatchResult,
    RecallResult,
    Stats,
]


# ── behavioural tests ───────────────────────────────────────────────


class TestSlottedInstancesOmitDict:
    """Instances of slotted dataclasses must not carry a per-instance __dict__.

    The absence of __dict__ is the observable memory-saving behaviour that
    slots=True provides.  If __dict__ reappears, every instance grows by
    ~100+ bytes — a meaningful regression when thousands are in flight.
    """

    @pytest.mark.parametrize("cls", _HOT_PATH_CLASSES, ids=lambda c: c.__name__)
    def test_hot_path_model_instance_has_no_dict(self, cls):
        instance = cls()
        assert not hasattr(instance, "__dict__"), (
            f"{cls.__name__} instance has __dict__ — " "slots=True may have been removed from the dataclass decorator"
        )

    @pytest.mark.parametrize("cls", _FROZEN_RESULT_CLASSES, ids=lambda c: c.__name__)
    def test_frozen_result_instance_has_no_dict(self, cls):
        # Frozen dataclasses require all fields at construction; use
        # sensible defaults to avoid importing extra types.
        kwargs = _minimal_kwargs(cls)
        instance = cls(**kwargs)
        assert not hasattr(instance, "__dict__"), (
            f"{cls.__name__} instance has __dict__ — " "slots=True may have been removed from the dataclass decorator"
        )


class TestSlottedInstancesRejectArbitraryAttrs:
    """Slotted classes must raise AttributeError on undeclared attributes.

    This protects against silent data corruption where a typo like
    ``chunk.metdata = ...`` silently creates a new attribute instead of
    raising immediately.
    """

    @pytest.mark.parametrize("cls", _HOT_PATH_CLASSES, ids=lambda c: c.__name__)
    def test_hot_path_model_rejects_extra_attr(self, cls):
        instance = cls()
        with pytest.raises(AttributeError):
            instance._undeclared_test_attr = "oops"  # type: ignore[attr-defined]

    @pytest.mark.parametrize("cls", _FROZEN_RESULT_CLASSES, ids=lambda c: c.__name__)
    def test_frozen_result_rejects_extra_attr(self, cls):
        kwargs = _minimal_kwargs(cls)
        instance = cls(**kwargs)
        # Frozen+slotted dataclasses may raise AttributeError or TypeError
        # depending on the Python version and dataclass internals.
        with pytest.raises((AttributeError, TypeError)):
            instance._undeclared_test_attr = "oops"  # type: ignore[attr-defined]


# ── minimal construction helpers ─────────────────────────────────────


def _minimal_kwargs(cls):
    """Return the smallest set of kwargs needed to construct *cls*."""
    name = cls.__name__
    ns = uuid4()

    if name == "RememberResult":
        return {
            "document_id": uuid4(),
            "namespace_id": ns,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
        }
    if name == "BatchResult":
        return {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
        }
    if name == "RecallResult":
        return {
            "query": "",
            "namespace_id": ns,
            "chunks": [],
            "entities": [],
            "context_text": "",
        }
    if name == "Stats":
        return {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
        }
    msg = f"No minimal kwargs defined for {name}"
    raise ValueError(msg)
