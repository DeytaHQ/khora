"""Unit tests for ``khora.integrations.llamaindex.KhoraMemoryBlock``.

Runs against an ``AsyncMock(spec=Khora)`` — no infrastructure required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("llama_index.core")


from khora import Khora  # noqa: E402


def _mk_kb(*, recall_result=None, document_id=None) -> Khora:
    kb = AsyncMock(spec=Khora)
    kb.recall = AsyncMock(
        return_value=recall_result or MagicMock(chunks=[], entities=[], context_text="recalled text", metadata={})
    )
    kb.remember = AsyncMock(return_value=MagicMock(document_id=document_id or uuid4()))
    return kb


def test_block_is_base_memory_block():
    """``isinstance(block, BaseMemoryBlock)`` passes — required by LlamaIndex."""
    from llama_index.core.memory.memory import BaseMemoryBlock

    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb()
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())
    assert isinstance(block, BaseMemoryBlock)


def test_block_name_priority_description_propagate():
    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb()
    block = KhoraMemoryBlock(
        kb=kb,
        namespace_id=uuid4(),
        name="custom_block",
        description="some description",
        priority=3,
    )
    assert block.name == "custom_block"
    assert block.description == "some description"
    assert block.priority == 3


@pytest.mark.asyncio
async def test_aget_calls_recall_with_last_user_message():
    """The block recalls against the last user-role message's text."""
    from llama_index.core.llms import ChatMessage

    from khora.integrations.llamaindex import KhoraMemoryBlock

    recall = MagicMock(chunks=[], entities=[], context_text="rendered chunk", metadata={})
    kb = _mk_kb(recall_result=recall)
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4(), similarity_top_k=4)

    out = await block.aget(
        messages=[
            ChatMessage(role="user", content="what database did we pick?"),
            ChatMessage(role="assistant", content="postgres"),
            ChatMessage(role="user", content="why?"),
        ]
    )

    # The last user message wins.
    kb.recall.assert_called_once()
    kwargs = kb.recall.call_args.kwargs
    assert kb.recall.call_args.args[0] == "why?"
    assert kwargs["limit"] == 4
    assert "rendered chunk" in out
    assert out.startswith("<khora_memory>")
    assert out.rstrip().endswith("</khora_memory>")


@pytest.mark.asyncio
async def test_aget_no_messages_returns_empty():
    from khora.integrations.llamaindex import KhoraMemoryBlock

    block = KhoraMemoryBlock(kb=_mk_kb(), namespace_id=uuid4())
    assert await block.aget(messages=None) == ""
    assert await block.aget(messages=[]) == ""


@pytest.mark.asyncio
async def test_aput_persists_each_message():
    """Every non-empty message becomes a ``Khora.remember`` call."""
    from llama_index.core.llms import ChatMessage

    from khora.integrations.llamaindex import KhoraMemoryBlock

    kb = _mk_kb()
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())

    messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
        ChatMessage(role="user", content=""),  # skipped
    ]
    await block.aput(messages=messages)

    assert kb.remember.call_count == 2
    contents = [call.args[0] for call in kb.remember.call_args_list]
    assert contents == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_aput_stamps_event_id_on_each_message():
    """The returned document_id is round-tripped through additional_kwargs."""
    from llama_index.core.llms import ChatMessage

    from khora.integrations.llamaindex import KhoraMemoryBlock

    doc_id = uuid4()
    kb = _mk_kb(document_id=doc_id)
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())

    msg = ChatMessage(role="user", content="hello")
    await block.aput(messages=[msg])

    assert msg.additional_kwargs["khora_event_id"] == str(doc_id)


@pytest.mark.asyncio
async def test_atruncate_returns_none():
    """``atruncate`` clears the in-flight payload (khora is the truth store)."""
    from khora.integrations.llamaindex import KhoraMemoryBlock

    block = KhoraMemoryBlock(kb=_mk_kb(), namespace_id=uuid4())
    assert await block.atruncate("some long content", 1024) is None


@pytest.mark.asyncio
async def test_session_id_forwarded_to_remember():
    from llama_index.core.llms import ChatMessage

    from khora.integrations.llamaindex import KhoraMemoryBlock

    session_id = uuid4()
    kb = _mk_kb()
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4(), session_id=session_id)

    await block.aput(messages=[ChatMessage(role="user", content="hi")])
    assert kb.remember.call_args.kwargs["session_id"] == session_id


@pytest.mark.asyncio
async def test_aget_falls_back_to_non_user_message():
    """Assistant-only flows still recall — fallback picks the last non-empty message."""
    from llama_index.core.llms import ChatMessage

    from khora.integrations.llamaindex import KhoraMemoryBlock

    recall = MagicMock(chunks=[], entities=[], context_text="from-assistant", metadata={})
    kb = _mk_kb(recall_result=recall)
    block = KhoraMemoryBlock(kb=kb, namespace_id=uuid4())

    out = await block.aget(messages=[ChatMessage(role="assistant", content="some assistant statement")])
    assert kb.recall.call_args.args[0] == "some assistant statement"
    assert "from-assistant" in out
