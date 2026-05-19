"""``KhoraMemoryBlock`` — LlamaIndex long-term memory block backed by khora.

Implements the ``BaseMemoryBlock[str]`` contract from
``llama_index.core.memory.memory`` (LlamaIndex 0.14). The block is
pure-async: ``_aget`` runs ``Khora.recall`` against the last user
message and renders the hits as a single text payload; ``_aput`` runs
``Khora.remember`` over each inbound message so they become long-term
memory; ``atruncate`` returns ``None`` to drop the block content (khora
itself is the persistent store; truncation only clears the in-flight
payload).

Per issue #627, the design defers framework loading until first
instantiation. We can't subclass ``BaseMemoryBlock`` at module load
without importing ``llama_index``, so we resolve the base lazily and
build a dynamic subclass — same trick the langgraph adapter uses.
``isinstance(block, BaseMemoryBlock)`` still passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.integrations.llamaindex._mapping import message_to_text, stamp_event_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_index.core.llms import ChatMessage
    from llama_index.core.memory.memory import BaseMemoryBlock

    from khora.khora import Khora


# Lazy-resolved base class cache. Populated on first ``KhoraMemoryBlock``
# instantiation; thereafter all instances share the same dynamic subclass
# so ``isinstance`` checks are cheap.
_BaseMemoryBlockCls: type | None = None


def _import_base_memory_block() -> type[BaseMemoryBlock]:
    """Lazy import of ``llama_index.core.memory.memory.BaseMemoryBlock``."""
    try:
        from llama_index.core.memory.memory import (  # noqa: PLC0415
            BaseMemoryBlock as _Base,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraMemoryBlock requires the optional `llamaindex` extra. Install with: pip install 'khora[llamaindex]'"
        ) from exc
    return _Base


def _get_base_class() -> type:
    """Return (and cache) the dynamically-resolved ``BaseMemoryBlock`` class."""
    global _BaseMemoryBlockCls
    if _BaseMemoryBlockCls is None:
        _BaseMemoryBlockCls = _import_base_memory_block()
    return _BaseMemoryBlockCls


# Header line prepended to the rendered recall result. Keeps the block's
# text disambiguated from the rest of the prompt template.
_RECALL_HEADER = "<khora_memory>"
_RECALL_FOOTER = "</khora_memory>"


def KhoraMemoryBlock(  # noqa: N802 — factory matches class-like usage
    *,
    kb: Khora,
    namespace_id: UUID,
    name: str = "khora_memory",
    description: str | None = None,
    priority: int = 1,
    similarity_top_k: int = 5,
    session_id: UUID | None = None,
    skill_name: str = "general_entities",
    entity_types: list[str] | None = None,
    relationship_types: list[str] | None = None,
) -> BaseMemoryBlock:
    """Build a ``BaseMemoryBlock[str]`` instance bound to a khora.

    The factory is a function (not a class) so the framework import
    happens inside the function body — same module-load discipline the
    rest of the package enforces. The returned object is still an
    ``isinstance(..., BaseMemoryBlock)`` for LlamaIndex's purposes.

    Args:
        kb: A connected :class:`khora.Khora` instance. Caller owns the
            lifecycle.
        namespace_id: Stable khora namespace this block reads/writes.
        name: LlamaIndex memory block name. Default ``"khora_memory"``.
        description: Optional description forwarded to LlamaIndex.
        priority: LlamaIndex truncation priority. Lower values are kept
            longer when the agent's memory budget tightens. Default
            ``1`` (don't truncate before normal blocks).
        similarity_top_k: Max chunks pulled per recall. Default ``5``.
        session_id: Optional khora session UUID stamped on every
            ``remember`` call. Useful for ``Khora.forget_session(...)``
            cleanup at the end of an agent run.
        skill_name: khora extraction skill name forwarded to
            ``Khora.remember``. Default ``"general_entities"``.
        entity_types / relationship_types: extraction whitelist
            forwarded to ``Khora.remember``. Empty lists / ``None``
            disable extraction entirely — useful for cheap, frequent
            chat-history writes where entity extraction isn't worth the
            LLM tokens.

    Returns:
        A populated ``BaseMemoryBlock[str]`` instance.

    Raises:
        ImportError: If the ``[llamaindex]`` extra is not installed.
    """
    base_cls = _get_base_class()

    class _KhoraMemoryBlock(base_cls):  # type: ignore[misc, valid-type]
        """The runtime class — subclass of ``BaseMemoryBlock[str]``."""

        # Pydantic model config: BaseMemoryBlock allows arbitrary types,
        # but we still need to declare our own fields so pydantic can
        # validate the constructor kwargs. Mark them as exclude-from-
        # serialization where appropriate.
        model_config = {"arbitrary_types_allowed": True}

        async def _aget(  # type: ignore[override]
            self,
            messages: list[ChatMessage] | None = None,
            **block_kwargs: Any,
        ) -> str:
            """Recall against the last user message and render as text.

            If ``messages`` is empty or contains no user-role message,
            returns an empty string — there's nothing to recall against.
            """
            query = _pick_query(messages)
            if not query:
                return ""
            result = await self._kb.recall(
                query,
                namespace=self._namespace_id,
                limit=self._top_k,
            )
            return _format_recall(result)

        async def _aput(  # type: ignore[override]
            self,
            messages: list[ChatMessage],
        ) -> None:
            """Persist each message into khora as a long-term memory."""
            if not messages:
                return
            for index, message in enumerate(messages):
                text = message_to_text(message)
                if not text.strip():
                    # Skip empty messages — no value in indexing them and
                    # khora's remember rejects empty content.
                    continue
                metadata: dict[str, Any] = {
                    "llamaindex_source": "memory_block",
                    "llamaindex_role": str(getattr(message.role, "value", message.role)),
                    "llamaindex_msg_index": int(index),
                }
                result = await self._kb.remember(
                    text,
                    namespace=self._namespace_id,
                    metadata=metadata,
                    skill_name=self._skill_name,
                    entity_types=list(self._entity_types),
                    relationship_types=list(self._relationship_types),
                    session_id=self._session_id,
                )
                # Round-trip a delete handle through the message's own
                # additional_kwargs (per the issue design note). Mutation
                # is in-place — LlamaIndex's BaseMemoryBlock.aput already
                # stamps session_id on the same dict.
                message.additional_kwargs.update(stamp_event_id({}, result.document_id))

        async def atruncate(  # type: ignore[override]
            self,
            content: str,
            tokens_to_truncate: int,
        ) -> str | None:
            """Drop the in-flight payload entirely.

            khora is the persistent store — we lose nothing by dropping
            the rendered text; the next ``_aget`` call rebuilds it from
            khora. Returning ``None`` tells LlamaIndex the block is now
            empty.
            """
            return None

    block = _KhoraMemoryBlock(name=name, description=description, priority=priority)
    # We attach the bound state via object.__setattr__ because
    # BaseMemoryBlock is a frozen-ish pydantic model and direct setattr
    # would trigger validation. The fields don't need to be schema-
    # validated — they're runtime-only handles.
    object.__setattr__(block, "_kb", kb)
    object.__setattr__(block, "_namespace_id", namespace_id)
    object.__setattr__(block, "_top_k", int(similarity_top_k))
    object.__setattr__(block, "_session_id", session_id)
    object.__setattr__(block, "_skill_name", skill_name)
    object.__setattr__(block, "_entity_types", list(entity_types) if entity_types is not None else [])
    object.__setattr__(
        block,
        "_relationship_types",
        list(relationship_types) if relationship_types is not None else [],
    )
    return block


def _pick_query(messages: list[ChatMessage] | None) -> str:
    """Pick the recall query — the last user-role message's text.

    Returns an empty string when there is no usable message. Falls back
    to the last message of any role when no user-role message is present
    so the block still recalls something on assistant-only flows.
    """
    if not messages:
        return ""
    # Walk in reverse — most recent user message wins.
    for message in reversed(messages):
        role = getattr(message.role, "value", message.role)
        if str(role).lower() == "user":
            text = message_to_text(message)
            if text.strip():
                return text
    # No user-role match — fall back to the last non-empty message.
    for message in reversed(messages):
        text = message_to_text(message)
        if text.strip():
            return text
    return ""


def _format_recall(result: Any) -> str:
    """Render a ``RecallResult`` as a bounded text block.

    Joins the chunk contents into a single block. The output is wrapped
    in a ``<khora_memory>`` envelope so a downstream prompt template can
    spot it.
    """
    parts = [chunk.content for chunk in result.chunks]
    context = "\n".join(p for p in parts if p).strip()
    if not context:
        return ""
    return f"{_RECALL_HEADER}\n{context}\n{_RECALL_FOOTER}"


__all__ = ["KhoraMemoryBlock"]
