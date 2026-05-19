"""Extra coverage tests for the llamaindex adapter submodules.

Targets the remaining uncovered branches:

* ``_mapping.message_to_text`` with ``content`` = None (line 56)
* ``_mapping.entity_to_node_with_score`` for an entity with no
  ``entity_type`` (line 124->126)
* ``_mapping.chunk_to_node_with_score`` / ``entity_to_node_with_score``
  with ``abstention_signals`` carrying ``should_abstain=False`` (line 138)
* ``memory._aput`` empty-message-list early return (line 150)
* ``memory._pick_query`` user-role search + fallback-to-any-role exits
  (lines 224, 229, 231)
* ``memory._format_recall`` empty-context, chunk-content fallback,
  empty-everything (lines 245-246, 248)
* ``chat_store._list_for_key`` ignoring documents with no metadata /
  wrong source / no key (lines 159, 161)
* ``chat_store._adelete_last_message`` empty case
* ``chat_store._aget_keys`` non-string key skip (line 236, 238)
* ``chat_store._remember_message`` empty-text fallback to single space
  (line 247)
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.llms import ChatMessage  # noqa: E402

from khora.core.models.entity import Entity  # noqa: E402
from khora.integrations.llamaindex._mapping import (  # noqa: E402
    chunk_to_node_with_score,
    entity_to_node_with_score,
    message_to_text,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _mapping
# ---------------------------------------------------------------------------


def test_message_to_text_handles_none_content() -> None:
    """ChatMessage with no content (None) → empty string, not a crash."""
    # ChatMessage(content=None) is permitted (e.g. tool_call shells).
    msg = ChatMessage(role="assistant")
    out = message_to_text(msg)
    assert isinstance(out, str)
    assert out == ""


def test_entity_to_node_with_score_no_entity_type_renders_bare_name() -> None:
    """Entity with empty ``entity_type`` skips the ``(TYPE)`` suffix."""
    entity = Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name="Bob",
        entity_type="",
        description="a guy",
    )
    node = entity_to_node_with_score(entity, 0.5)
    # No parens-wrapped type segment.
    assert "(" not in node.node.text
    assert node.node.text.strip() == "Bob: a guy"


def test_chunk_to_node_with_score_should_abstain_false_keeps_key() -> None:
    """When signals exist but ``should_abstain`` is False, the metadata key still appears."""
    from datetime import UTC, datetime

    from khora.core.models.document import Chunk

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        namespace_id=uuid4(),
        content="hello",
        chunk_index=0,
        metadata={},
        created_at=datetime.now(UTC),
    )
    node = chunk_to_node_with_score(chunk, 0.5, abstention_signals={"should_abstain": False})
    assert node.node.metadata["khora_should_abstain"] is False


def test_entity_to_node_with_score_should_abstain_signal_propagates() -> None:
    entity = Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name="Alice",
        entity_type="PERSON",
        description="x",
    )
    node = entity_to_node_with_score(entity, 0.5, abstention_signals={"should_abstain": True})
    assert node.node.metadata["khora_should_abstain"] is True


# ---------------------------------------------------------------------------
# memory.py — _aput, _pick_query, _format_recall
# ---------------------------------------------------------------------------


def _mk_kb(**recall_attrs: Any) -> Any:
    """Build an AsyncMock Khora.

    ``recall_attrs`` controls the return value of ``recall``.
    """
    from khora.khora import Khora

    kb = AsyncMock(spec=Khora)
    recall_result = MagicMock(
        chunks=recall_attrs.get("chunks", []),
        entities=recall_attrs.get("entities", []),
        context_text=recall_attrs.get("context_text", ""),
        metadata=recall_attrs.get("metadata", {}),
    )
    kb.recall = AsyncMock(return_value=recall_result)
    kb.remember = AsyncMock(return_value=MagicMock(document_id=uuid4()))
    return kb


async def test_memory_aput_empty_messages_is_noop() -> None:
    """``_aput`` with no messages must short-circuit before touching kb."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb()
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    await block.aput(messages=[])
    kb.remember.assert_not_awaited()


async def test_memory_aget_falls_back_when_user_messages_have_only_empty_text() -> None:
    """User-role match found but text empty → fall through to "any role"  loop."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb(context_text="recalled")
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    messages = [
        ChatMessage(role="assistant", content="assistant content"),
        ChatMessage(role="user", content="   "),  # whitespace-only user msg
    ]
    out = await block.aget(messages=messages)
    # Falls back to last non-empty message of any role.
    assert kb.recall.call_args.args[0] == "assistant content"
    assert "recalled" in out


async def test_memory_aget_returns_empty_when_all_messages_whitespace() -> None:
    """Both loops exhaust → return ''. ``_aget`` therefore returns ''."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb()
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    messages = [
        ChatMessage(role="user", content=""),
        ChatMessage(role="assistant", content="   "),
    ]
    out = await block.aget(messages=messages)
    assert out == ""
    kb.recall.assert_not_awaited()


async def test_memory_aget_falls_back_to_chunk_content_when_context_text_empty() -> None:
    """``_format_recall`` joins chunk.content when context_text is empty."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    chunks = [
        (MagicMock(content="chunk A"), 0.9),
        (MagicMock(content="chunk B"), 0.8),
    ]
    kb = _mk_kb(chunks=chunks, context_text="")
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    messages = [ChatMessage(role="user", content="why?")]
    out = await block.aget(messages=messages)
    # Built from chunk.content join.
    assert "chunk A" in out
    assert "chunk B" in out
    assert out.startswith("<khora_memory>")


async def test_memory_aget_returns_empty_when_recall_has_no_context_and_no_chunks() -> None:
    """No context_text + no chunks → empty payload (no envelope)."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb(chunks=[], context_text="")
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    out = await block.aget(messages=[ChatMessage(role="user", content="why?")])
    assert out == ""


async def test_memory_aget_chunk_fallback_drops_empty_chunk_text() -> None:
    """Chunks with empty ``content`` are dropped from the chunk-join fallback."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    chunks = [
        (MagicMock(content="real chunk"), 0.9),
        (MagicMock(content=""), 0.8),
    ]
    kb = _mk_kb(chunks=chunks, context_text="")
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    out = await block.aget(messages=[ChatMessage(role="user", content="q")])
    assert "real chunk" in out
    # Empty-content chunk doesn't introduce a blank line.
    assert "\n\n" not in out.replace("<khora_memory>\n", "").rstrip("\n</khora_memory>")


# ---------------------------------------------------------------------------
# chat_store.py — gaps
# ---------------------------------------------------------------------------


def _mk_chat_store() -> Any:
    from khora.integrations.llamaindex import KhoraChatStore
    from khora.khora import Khora

    kb = AsyncMock(spec=Khora)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return KhoraChatStore(kb=kb, namespace_id=uuid4()), kb


def test_chat_store_list_for_key_skips_no_metadata_and_wrong_source() -> None:
    """The internal _list_for_key filter must reject foreign docs."""
    from khora.core.models.document import Document

    store, kb = _mk_chat_store()

    # 3 docs: one with empty metadata, one with wrong source, one valid.
    ns = uuid4()
    no_meta = Document(id=uuid4(), namespace_id=ns, metadata={})
    foreign = Document(
        id=uuid4(),
        namespace_id=ns,
        metadata={"llamaindex_source": "memory_block", "llamaindex_chat_key": "k"},
    )
    own = Document(
        id=uuid4(),
        namespace_id=ns,
        metadata={
            "llamaindex_source": "chat_store",
            "llamaindex_chat_key": "k",
            "llamaindex_chat_index": 0,
            "llamaindex_chat_role": "user",
            "llamaindex_chat_additional_kwargs": {},
        },
        content="hi",
    )
    kb.list_documents = AsyncMock(return_value=[no_meta, foreign, own])

    out = store.get_messages("k")
    assert len(out) == 1
    assert out[0].content == "hi"


def test_chat_store_delete_last_message_empty_key_returns_none() -> None:
    store, kb = _mk_chat_store()
    kb.list_documents = AsyncMock(return_value=[])
    assert store.delete_last_message("never-existed") is None


def test_chat_store_get_keys_skips_non_string_keys() -> None:
    """A document with a non-string chat_key value must be skipped."""
    from khora.core.models.document import Document

    store, kb = _mk_chat_store()
    ns = uuid4()
    bad_key_doc = Document(
        id=uuid4(),
        namespace_id=ns,
        metadata={"llamaindex_source": "chat_store", "llamaindex_chat_key": 42},
    )
    good_key_doc = Document(
        id=uuid4(),
        namespace_id=ns,
        metadata={"llamaindex_source": "chat_store", "llamaindex_chat_key": "real"},
    )
    foreign_doc = Document(
        id=uuid4(),
        namespace_id=ns,
        metadata={"llamaindex_source": "memory_block", "llamaindex_chat_key": "k"},
    )
    kb.list_documents = AsyncMock(return_value=[bad_key_doc, good_key_doc, foreign_doc])

    keys = store.get_keys()
    assert keys == ["real"]


def test_chat_store_remember_message_substitutes_space_for_empty_content() -> None:
    """An empty-content ChatMessage must be persisted as a single space."""
    from llama_index.core.llms import ChatMessage

    store, kb = _mk_chat_store()
    kb.list_documents = AsyncMock(return_value=[])
    # Capture the content passed to kb.remember.
    seen: dict[str, Any] = {}

    async def _fake_remember(content: str, **kwargs: Any) -> Any:
        seen["content"] = content
        return MagicMock(document_id=uuid4())

    kb.remember = AsyncMock(side_effect=_fake_remember)
    store.set_messages("k", [ChatMessage(role="user", content="")])
    assert seen["content"] == " "
