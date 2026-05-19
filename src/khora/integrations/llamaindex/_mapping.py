"""Translation helpers between LlamaIndex types and khora models.

Owns the two narrow shape conversions the adapter performs:

* ``khora.Chunk`` (+ score) → LlamaIndex ``NodeWithScore(node=TextNode(...))``
* ``khora.Entity`` (+ score) → LlamaIndex ``NodeWithScore(node=TextNode(...))``
  (only emitted when ``KhoraRetriever(include_entities=True)``)
* ``ChatMessage`` → text payload suitable for ``Khora.remember``

Kept in one file so future LlamaIndex API drift only needs to be threaded
through here. Adapter-internal helpers — not part of the public surface.

Module-load discipline: no top-level ``import llama_index``. Type-only
imports live behind ``if TYPE_CHECKING``; runtime imports happen inside
function bodies. Enforced by ``tools/check_optional_imports.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_index.core.llms import ChatMessage
    from llama_index.core.schema import NodeWithScore

    from khora.core.models.document import Chunk
    from khora.core.models.entity import Entity


# Metadata key the adapter stamps on every document it writes via
# ``KhoraMemoryBlock`` / ``KhoraChatStore``. Used to skip foreign docs in
# read paths so the adapter never returns data it didn't author.
_KEY_SOURCE = "llamaindex_source"

# Metadata keys used by the chat store to round-trip ChatMessage identity
# back from the stored khora document.
_KEY_CHAT_KEY = "llamaindex_chat_key"
_KEY_CHAT_INDEX = "llamaindex_chat_index"
_KEY_CHAT_ROLE = "llamaindex_chat_role"
_KEY_CHAT_ADDITIONAL = "llamaindex_chat_additional_kwargs"
_KEY_EVENT_ID = "khora_event_id"


def message_to_text(message: ChatMessage) -> str:
    """Render a ChatMessage as the plain text we hand to ``Khora.remember``.

    Uses the canonical ``message.content`` accessor — LlamaIndex 0.14
    joins all ``TextBlock`` parts into a single string for us. Non-text
    blocks (images, audio, etc.) contribute nothing here; the original
    ``message.blocks`` payload is preserved verbatim in the document's
    ``additional_kwargs`` so callers can reconstruct the message if they
    need to.
    """
    text = message.content
    if text is None:
        return ""
    return str(text)


def chunk_to_node_with_score(
    chunk: Chunk,
    score: float,
    *,
    abstention_signals: dict[str, Any] | None = None,
) -> NodeWithScore:
    """Convert a scored khora ``Chunk`` to a LlamaIndex ``NodeWithScore``.

    The ``TextNode.metadata`` carries the chunk's source document ID and
    any custom metadata the document was stored with, plus a flag that
    surfaces khora's abstention signals to downstream postprocessors /
    response synthesizers (``khora_should_abstain`` per the issue's
    acceptance criteria).

    Args:
        chunk: A khora ``Chunk`` returned from ``Khora.recall``.
        score: The retrieval score paired with ``chunk`` by khora.
        abstention_signals: Optional ``RecallResult.metadata["abstention_signals"]``
            dict — if present, ``khora_should_abstain`` is propagated to
            every node so consumers can short-circuit answer generation.

    Returns:
        A populated ``NodeWithScore`` whose ``.node`` is a ``TextNode``.
    """
    from llama_index.core.schema import NodeWithScore, TextNode  # noqa: PLC0415

    custom = chunk.metadata or {}
    metadata: dict[str, Any] = {
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.id),
        "namespace_id": str(chunk.namespace_id),
        "khora_kind": "chunk",
    }
    # Forward the user's own custom metadata. We intentionally don't
    # strip ``llamaindex_*`` keys — if the chunk was written by the
    # adapter, surfacing them back is harmless and helps debugging.
    metadata.update(custom)
    if abstention_signals is not None:
        metadata["khora_should_abstain"] = bool(abstention_signals.get("should_abstain", False))

    node = TextNode(
        id_=str(chunk.id),
        text=chunk.content,
        metadata=metadata,
    )
    return NodeWithScore(node=node, score=float(score))


def entity_to_node_with_score(
    entity: Entity,
    score: float,
    *,
    abstention_signals: dict[str, Any] | None = None,
) -> NodeWithScore:
    """Convert a scored khora ``Entity`` to a LlamaIndex ``NodeWithScore``.

    Entity nodes carry a short summary as their text payload (``"<name>
    (<type>): <description>"``) so a downstream LLM can use them in the
    same context window as chunk text. Identity round-trips through the
    metadata.
    """
    from llama_index.core.schema import NodeWithScore, TextNode  # noqa: PLC0415

    summary_parts = [entity.name]
    if entity.entity_type:
        summary_parts.append(f"({entity.entity_type})")
    description = (entity.description or "").strip()
    summary = " ".join(summary_parts)
    text = f"{summary}: {description}" if description else summary

    metadata: dict[str, Any] = {
        "entity_id": str(entity.id),
        "entity_name": entity.name,
        "entity_type": entity.entity_type,
        "namespace_id": str(entity.namespace_id),
        "khora_kind": "entity",
    }
    if abstention_signals is not None:
        metadata["khora_should_abstain"] = bool(abstention_signals.get("should_abstain", False))

    node = TextNode(
        id_=str(entity.id),
        text=text,
        metadata=metadata,
    )
    return NodeWithScore(node=node, score=float(score))


def chat_message_metadata(
    *,
    chat_key: str,
    index: int,
    message: ChatMessage,
) -> dict[str, Any]:
    """Build the metadata dict we stamp on a ChatMessage-backed document.

    The fields are read back by ``KhoraChatStore.get_messages`` to
    reconstruct the original message order, role, and additional_kwargs.
    """
    # ChatMessage.additional_kwargs is a dict of arbitrary JSON-safe
    # values per the LlamaIndex contract. We persist it verbatim — khora
    # stores metadata.custom in a JSONB column.
    additional = dict(message.additional_kwargs or {})
    return {
        _KEY_SOURCE: "chat_store",
        _KEY_CHAT_KEY: chat_key,
        _KEY_CHAT_INDEX: int(index),
        _KEY_CHAT_ROLE: str(getattr(message.role, "value", message.role)),
        _KEY_CHAT_ADDITIONAL: additional,
    }


def stamp_event_id(metadata: dict[str, Any], event_id: Any) -> dict[str, Any]:
    """Return ``metadata`` extended with the khora ``RememberResult.document_id``.

    Used by ``KhoraMemoryBlock`` to round-trip a delete handle through
    ``ChatMessage.additional_kwargs`` (per the issue's design note).
    """
    new_meta = dict(metadata)
    new_meta[_KEY_EVENT_ID] = str(event_id)
    return new_meta


__all__ = [
    "chat_message_metadata",
    "chunk_to_node_with_score",
    "entity_to_node_with_score",
    "message_to_text",
    "stamp_event_id",
]
