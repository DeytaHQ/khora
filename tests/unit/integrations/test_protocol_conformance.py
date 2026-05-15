"""Protocol-conformance smoke tests for khora.integrations.

The Protocols are ``@runtime_checkable``, so the contract these tests
encode is exactly: ``isinstance(adapter, MemoryAdapter)`` must pass for
adapter classes that declare the right method signatures, and fail for
classes that don't.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.integrations import (
    KhoraIntegration,
    MemoryAdapter,
    RetrievedNode,
    RetrieverAdapter,
)


class _FakeKhora:
    """Stand-in for the real Khora — Protocols only check shape."""


class _ConformingMemoryAdapter:
    name = "fake-memory"

    def __init__(self) -> None:
        self.kb = _FakeKhora()
        self.namespace_id = uuid4()

    async def asave(self, content: str, *, metadata: dict[str, Any] | None = None) -> UUID:
        return uuid4()

    async def asearch(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return []


class _ConformingRetrieverAdapter:
    name = "fake-retriever"

    def __init__(self) -> None:
        self.kb = _FakeKhora()
        self.namespace_id = uuid4()

    async def aretrieve(self, query: str, *, limit: int = 10) -> list[RetrievedNode]:
        return []


class _MissingAsearch:
    name = "broken"

    def __init__(self) -> None:
        self.kb = _FakeKhora()
        self.namespace_id = uuid4()

    async def asave(self, content: str, *, metadata: dict[str, Any] | None = None) -> UUID:
        return uuid4()


def test_memory_adapter_isinstance_passes_for_conforming():
    assert isinstance(_ConformingMemoryAdapter(), MemoryAdapter)


def test_retriever_adapter_isinstance_passes_for_conforming():
    assert isinstance(_ConformingRetrieverAdapter(), RetrieverAdapter)


def test_memory_adapter_isinstance_fails_when_method_missing():
    # _MissingAsearch has asave but no asearch — Protocol check should fail.
    assert not isinstance(_MissingAsearch(), MemoryAdapter)


def test_khora_integration_marker_accepts_anything_with_attrs():
    # Marker protocols only check attribute presence, not types.
    adapter = _ConformingMemoryAdapter()
    assert isinstance(adapter, KhoraIntegration)


def test_khora_integration_rejects_when_attrs_missing():
    class _Bare:
        pass

    assert not isinstance(_Bare(), KhoraIntegration)


def test_retrieved_node_is_frozen_dataclass():
    node = RetrievedNode(id=uuid4(), text="hi", score=0.5)
    with pytest.raises((AttributeError, TypeError)):
        node.text = "mutated"  # type: ignore[misc]


def test_retrieved_node_metadata_defaults_empty():
    node = RetrievedNode(id=uuid4(), text="hi", score=0.5)
    assert node.metadata == {}
