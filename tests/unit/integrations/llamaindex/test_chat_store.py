"""Unit tests for the deprecated ``KhoraChatStore``.

Exercises all 7 ``BaseChatStore`` abstract methods round-trip plus the
deprecation warning on instantiation.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("llama_index.core")


from khora import Khora  # noqa: E402
from khora.core.models.document import (  # noqa: E402
    Document,
    DocumentStatus,
)


class _ShadowStore:
    """Tiny in-memory replacement for the parts of Khora the chat store touches."""

    def __init__(self) -> None:
        self.docs: dict[UUID, Document] = {}

    async def remember(self, content: str, **kwargs: Any) -> Any:
        document_id = uuid4()
        metadata_custom = dict(kwargs.get("metadata") or {})
        doc = Document(
            id=document_id,
            namespace_id=kwargs["namespace"],
            content=content,
            external_id=kwargs.get("external_id"),
            title=kwargs.get("title", ""),
            source=kwargs.get("source", ""),
            metadata=metadata_custom,
            status=DocumentStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.docs[document_id] = doc
        return MagicMock(document_id=document_id)

    async def list_documents(self, *, namespace: UUID, limit: int = 100) -> list[Document]:
        return list(self.docs.values())[:limit]

    async def forget(self, document_id: UUID, *, namespace: UUID) -> bool:
        return self.docs.pop(document_id, None) is not None


@pytest.fixture
def store_factory():
    """Yields ``(make_store, shadow)`` — call ``make_store()`` for a fresh store."""
    shadow = _ShadowStore()
    kb = AsyncMock(spec=Khora)
    kb.remember = AsyncMock(side_effect=shadow.remember)
    kb.list_documents = AsyncMock(side_effect=shadow.list_documents)
    kb.forget = AsyncMock(side_effect=shadow.forget)

    def _make() -> Any:
        from khora.integrations.llamaindex import KhoraChatStore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return KhoraChatStore(kb=kb, namespace_id=uuid4())

    return _make, shadow


def test_instantiation_emits_deprecation_warning():
    """Per issue #627 acceptance criteria — DeprecationWarning on every call."""
    from khora.integrations.llamaindex import KhoraChatStore

    kb = AsyncMock(spec=Khora)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        KhoraChatStore(kb=kb, namespace_id=uuid4())
    deprecations = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected at least one DeprecationWarning"
    assert "KhoraMemoryBlock" in str(deprecations[0].message)


def test_isinstance_base_chat_store(store_factory):
    """``isinstance(store, BaseChatStore)`` passes — LlamaIndex relies on it."""
    from llama_index.core.storage.chat_store.base import BaseChatStore

    make, _shadow = store_factory
    assert isinstance(make(), BaseChatStore)


def test_class_name_overridden(store_factory):
    make, _ = store_factory
    assert make().class_name() == "KhoraChatStore"


def test_set_and_get_messages_roundtrip(store_factory):
    """set_messages then get_messages returns the same role + content."""
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there", additional_kwargs={"k": "v"}),
    ]
    store.set_messages("conv-1", messages)

    out = store.get_messages("conv-1")
    assert len(out) == 2
    assert str(out[0].role.value if hasattr(out[0].role, "value") else out[0].role) == "user"
    assert out[0].content == "hello"
    assert out[1].content == "hi there"
    assert out[1].additional_kwargs == {"k": "v"}


def test_set_messages_replaces_previous(store_factory):
    """set_messages drops any pre-existing messages under the same key."""
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages("k", [ChatMessage(role="user", content="v1")])
    store.set_messages("k", [ChatMessage(role="user", content="v2")])

    out = store.get_messages("k")
    assert len(out) == 1
    assert out[0].content == "v2"


def test_add_message_appends_after_existing(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages("k", [ChatMessage(role="user", content="first")])
    store.add_message("k", ChatMessage(role="assistant", content="second"))

    out = store.get_messages("k")
    assert [m.content for m in out] == ["first", "second"]


def test_delete_messages_returns_payload_and_clears(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages(
        "k",
        [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")],
    )

    deleted = store.delete_messages("k")
    assert deleted is not None
    assert [m.content for m in deleted] == ["a", "b"]
    assert store.get_messages("k") == []


def test_delete_messages_missing_returns_none(store_factory):
    make, _ = store_factory
    assert make().delete_messages("never-existed") is None


def test_delete_message_by_index(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages(
        "k",
        [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="assistant", content="b"),
            ChatMessage(role="user", content="c"),
        ],
    )

    deleted = store.delete_message("k", 1)
    assert deleted is not None
    assert deleted.content == "b"
    assert [m.content for m in store.get_messages("k")] == ["a", "c"]


def test_delete_message_out_of_range_returns_none(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages("k", [ChatMessage(role="user", content="a")])
    assert store.delete_message("k", 99) is None


def test_delete_last_message(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages(
        "k",
        [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")],
    )

    deleted = store.delete_last_message("k")
    assert deleted is not None
    assert deleted.content == "b"
    assert [m.content for m in store.get_messages("k")] == ["a"]


def test_get_keys_returns_distinct(store_factory):
    from llama_index.core.llms import ChatMessage

    make, _ = store_factory
    store = make()
    store.set_messages("k1", [ChatMessage(role="user", content="a")])
    store.set_messages("k2", [ChatMessage(role="user", content="b")])
    keys = store.get_keys()
    assert sorted(keys) == ["k1", "k2"]
