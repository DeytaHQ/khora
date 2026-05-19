"""``KhoraChatStore`` — LlamaIndex ``BaseChatStore`` over khora.

**Deprecated since v0.13.** Modern LlamaIndex agents use
``BaseMemoryBlock`` (see :class:`KhoraMemoryBlock` in this package)
plus ``Memory`` for long-term memory. ``BaseChatStore`` is the legacy
flat-list-of-ChatMessages surface used by ``ChatMemoryBuffer``; we ship
this adapter for compatibility with code that still depends on it, and
emit a ``DeprecationWarning`` on instantiation.

The store maps LlamaIndex's ``(key: str, messages: list[ChatMessage])``
shape onto khora's document/chunk model. One ChatMessage = one
``Khora.remember(...)`` document, with metadata that round-trips role,
position in the conversation, and ``additional_kwargs``.

Sync surface (every ``BaseChatStore`` abstract method) bridges through
:func:`khora.integrations._sync.run_sync`. That bridge rejects reentry
from inside a running event loop — do not call these methods from
inside an ``async def``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.integrations._sync import run_sync
from khora.integrations.llamaindex._mapping import chat_message_metadata, message_to_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_index.core.llms import ChatMessage
    from llama_index.core.storage.chat_store.base import BaseChatStore

    from khora.khora import Khora


# Lazy-resolved base class cache.
_BaseChatStoreCls: type | None = None


def _import_base_chat_store() -> type[BaseChatStore]:
    """Lazy import of ``llama_index.core.storage.chat_store.base.BaseChatStore``."""
    try:
        from llama_index.core.storage.chat_store.base import (  # noqa: PLC0415
            BaseChatStore as _Base,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "KhoraChatStore requires the optional `llamaindex` extra. Install with: pip install 'khora[llamaindex]'"
        ) from exc
    return _Base


def _import_chat_message() -> type[ChatMessage]:
    """Lazy import of ``llama_index.core.llms.ChatMessage``."""
    from llama_index.core.llms import ChatMessage as _CM  # noqa: PLC0415

    return _CM


def _get_base_class() -> type:
    """Return (and cache) the dynamically-resolved ``BaseChatStore`` class."""
    global _BaseChatStoreCls
    if _BaseChatStoreCls is None:
        _BaseChatStoreCls = _import_base_chat_store()
    return _BaseChatStoreCls


# Metadata keys this adapter uses — mirrors _mapping._KEY_* but inlined
# here to keep the SQL-search filter explicit.
_KEY_SOURCE = "llamaindex_source"
_KEY_CHAT_KEY = "llamaindex_chat_key"
_KEY_CHAT_INDEX = "llamaindex_chat_index"
_KEY_CHAT_ROLE = "llamaindex_chat_role"
_KEY_CHAT_ADDITIONAL = "llamaindex_chat_additional_kwargs"


def KhoraChatStore(  # noqa: N802 — class-shaped factory
    *,
    kb: Khora,
    namespace_id: UUID,
) -> BaseChatStore:
    """Build a deprecated ``BaseChatStore`` instance bound to a khora.

    Emits ``DeprecationWarning`` on call. New code should use
    :class:`KhoraMemoryBlock` plus ``llama_index.core.memory.Memory``
    instead.

    Args:
        kb: A connected :class:`khora.Khora` instance.
        namespace_id: Stable khora namespace UUID this store reads/writes.

    Returns:
        A ``BaseChatStore`` instance (dynamically subclassed so
        ``isinstance(store, BaseChatStore)`` passes).
    """
    warnings.warn(
        "KhoraChatStore is deprecated — modern LlamaIndex agents should use "
        "KhoraMemoryBlock + llama_index.core.memory.Memory. KhoraChatStore "
        "is shipped only for compatibility with legacy ChatMemoryBuffer "
        "users and may be removed in a future khora minor.",
        DeprecationWarning,
        stacklevel=2,
    )

    base_cls = _get_base_class()

    class _KhoraChatStore(base_cls):  # type: ignore[misc, valid-type]
        """Runtime class — subclass of ``BaseChatStore`` with our methods."""

        @classmethod
        def class_name(cls) -> str:
            return "KhoraChatStore"

        # -- BaseChatStore abstract sync methods --

        def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
            run_sync(self._aset_messages(key, messages))

        def get_messages(self, key: str) -> list[ChatMessage]:
            return run_sync(self._aget_messages(key))

        def add_message(self, key: str, message: ChatMessage) -> None:
            run_sync(self._aadd_message(key, message))

        def delete_messages(self, key: str) -> list[ChatMessage] | None:
            return run_sync(self._adelete_messages(key))

        def delete_message(self, key: str, idx: int) -> ChatMessage | None:
            return run_sync(self._adelete_message(key, idx))

        def delete_last_message(self, key: str) -> ChatMessage | None:
            return run_sync(self._adelete_last_message(key))

        def get_keys(self) -> list[str]:
            return run_sync(self._aget_keys())

        # -- Internal async implementations --

        async def _list_for_key(self, key: str) -> list[Any]:
            """List documents stamped with the given chat key, ordered by index.

            Returns the raw khora ``Document`` objects so callers can
            decide whether to project to ``ChatMessage`` or use the
            ``id`` for deletion.
            """
            # khora has no per-metadata-key pushdown filter on
            # list_documents; we over-fetch and filter client-side. The
            # store is intended for bounded chat-history workloads (one
            # key = one conversation) so this is fine.
            documents = await self._kb.list_documents(
                namespace=self._namespace_id,
                limit=10_000,
            )
            matched: list[Any] = []
            for document in documents:
                custom = document.metadata or {}
                if not custom:
                    continue
                if custom.get(_KEY_SOURCE) != "chat_store":
                    continue
                if custom.get(_KEY_CHAT_KEY) != key:
                    continue
                matched.append(document)
            matched.sort(key=lambda d: int((d.metadata or {}).get(_KEY_CHAT_INDEX, 0)))
            return matched

        def _document_to_message(self, document: Any) -> ChatMessage:
            """Project a stored khora ``Document`` back to a ``ChatMessage``."""
            ChatMessage = _import_chat_message()  # noqa: N806
            custom = document.metadata or {}
            role = custom.get(_KEY_CHAT_ROLE, "user")
            additional = dict(custom.get(_KEY_CHAT_ADDITIONAL) or {})
            return ChatMessage(
                role=role,
                content=document.content,
                additional_kwargs=additional,
            )

        async def _aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
            # Replace semantics: drop any existing messages for this key
            # then append the new list.
            existing = await self._list_for_key(key)
            for document in existing:
                await self._kb.forget(document.id, namespace=self._namespace_id)
            for index, message in enumerate(messages):
                await self._remember_message(key, index, message)

        async def _aget_messages(self, key: str) -> list[ChatMessage]:
            documents = await self._list_for_key(key)
            return [self._document_to_message(d) for d in documents]

        async def _aadd_message(self, key: str, message: ChatMessage) -> None:
            existing = await self._list_for_key(key)
            next_index = (
                int((existing[-1].metadata or {}).get(_KEY_CHAT_INDEX, len(existing) - 1)) + 1 if existing else 0
            )
            await self._remember_message(key, next_index, message)

        async def _adelete_messages(self, key: str) -> list[ChatMessage] | None:
            existing = await self._list_for_key(key)
            if not existing:
                return None
            projected = [self._document_to_message(d) for d in existing]
            for document in existing:
                await self._kb.forget(document.id, namespace=self._namespace_id)
            return projected

        async def _adelete_message(self, key: str, idx: int) -> ChatMessage | None:
            existing = await self._list_for_key(key)
            if not existing or idx < 0 or idx >= len(existing):
                return None
            target = existing[idx]
            projected = self._document_to_message(target)
            await self._kb.forget(target.id, namespace=self._namespace_id)
            return projected

        async def _adelete_last_message(self, key: str) -> ChatMessage | None:
            existing = await self._list_for_key(key)
            if not existing:
                return None
            target = existing[-1]
            projected = self._document_to_message(target)
            await self._kb.forget(target.id, namespace=self._namespace_id)
            return projected

        async def _aget_keys(self) -> list[str]:
            documents = await self._kb.list_documents(
                namespace=self._namespace_id,
                limit=10_000,
            )
            keys: set[str] = set()
            for document in documents:
                custom = document.metadata or {}
                if custom.get(_KEY_SOURCE) != "chat_store":
                    continue
                key = custom.get(_KEY_CHAT_KEY)
                if isinstance(key, str):
                    keys.add(key)
            return sorted(keys)

        async def _remember_message(self, key: str, index: int, message: ChatMessage) -> None:
            text = message_to_text(message)
            if not text.strip():
                # khora.remember rejects empty content; persist a single
                # space so the slot still round-trips.
                text = " "
            metadata = chat_message_metadata(chat_key=key, index=index, message=message)
            await self._kb.remember(
                text,
                namespace=self._namespace_id,
                metadata=metadata,
                skill_name="general_entities",
                entity_types=[],
                relationship_types=[],
            )

    store = _KhoraChatStore()
    # Bind runtime state via object.__setattr__ — BaseChatStore inherits
    # from pydantic's BaseComponent and would validate stray attrs.
    object.__setattr__(store, "_kb", kb)
    object.__setattr__(store, "_namespace_id", namespace_id)
    return store


__all__ = ["KhoraChatStore"]
