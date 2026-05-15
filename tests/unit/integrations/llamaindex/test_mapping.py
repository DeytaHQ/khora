"""Unit tests for ``khora.integrations.llamaindex._mapping``.

Exercises chunk → NodeWithScore, entity → NodeWithScore, message →
text, and the ``stamp_event_id`` round-trip helper.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

pytest.importorskip("llama_index.core")

from khora.core.models.document import Chunk, ChunkMetadata  # noqa: E402
from khora.core.models.entity import Entity  # noqa: E402
from khora.integrations.llamaindex._mapping import (  # noqa: E402
    chat_message_metadata,
    chunk_to_node_with_score,
    entity_to_node_with_score,
    message_to_text,
    stamp_event_id,
)


def _mk_chunk(*, content: str = "hello world", custom: dict | None = None) -> Chunk:
    document_id = uuid4()
    return Chunk(
        id=uuid4(),
        document_id=document_id,
        namespace_id=uuid4(),
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0, custom=dict(custom or {})),
        created_at=datetime.now(UTC),
    )


def _mk_entity(name: str = "Alice", entity_type: str = "PERSON") -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name=name,
        entity_type=entity_type,
        description="example description",
    )


def test_chunk_to_node_with_score_basic():
    """A chunk maps to a TextNode with content + metadata + score."""
    chunk = _mk_chunk(content="hello", custom={"source": "test"})
    node = chunk_to_node_with_score(chunk, 0.87)

    assert node.score == pytest.approx(0.87)
    assert node.node.text == "hello"
    assert node.node.id_ == str(chunk.id)
    assert node.node.metadata["khora_kind"] == "chunk"
    assert node.node.metadata["chunk_id"] == str(chunk.id)
    assert node.node.metadata["document_id"] == str(chunk.document_id)
    assert node.node.metadata["namespace_id"] == str(chunk.namespace_id)
    # User-supplied custom metadata flows through.
    assert node.node.metadata["source"] == "test"


def test_chunk_to_node_with_score_abstention_signal():
    """Abstention signals propagate to every node."""
    chunk = _mk_chunk()
    signals = {"should_abstain": True, "combined_score": 0.91}
    node = chunk_to_node_with_score(chunk, 0.4, abstention_signals=signals)
    assert node.node.metadata["khora_should_abstain"] is True


def test_chunk_to_node_with_score_no_signals_no_flag():
    """No abstention signals → no ``khora_should_abstain`` key."""
    chunk = _mk_chunk()
    node = chunk_to_node_with_score(chunk, 0.4)
    assert "khora_should_abstain" not in node.node.metadata


def test_entity_to_node_with_score_renders_summary():
    """Entity text combines name, type, and description."""
    entity = _mk_entity(name="Alice", entity_type="PERSON")
    node = entity_to_node_with_score(entity, 0.55)
    assert node.score == pytest.approx(0.55)
    assert node.node.metadata["khora_kind"] == "entity"
    assert node.node.metadata["entity_name"] == "Alice"
    assert node.node.metadata["entity_type"] == "PERSON"
    assert "Alice" in node.node.text
    assert "PERSON" in node.node.text
    assert "example description" in node.node.text


def test_entity_to_node_with_score_no_description():
    """Entity without description: text is just the summary."""
    entity = Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name="Bob",
        entity_type="PERSON",
        description=None,
    )
    node = entity_to_node_with_score(entity, 0.1)
    # Just the summary, no trailing colon or description.
    assert node.node.text.strip() == "Bob (PERSON)"


def test_message_to_text_simple():
    """ChatMessage.content (the canonical text accessor) is forwarded verbatim."""
    from llama_index.core.llms import ChatMessage

    msg = ChatMessage(role="user", content="hello world")
    assert message_to_text(msg) == "hello world"


def test_message_to_text_empty():
    from llama_index.core.llms import ChatMessage

    msg = ChatMessage(role="user", content="")
    assert message_to_text(msg) == ""


def test_chat_message_metadata_roundtrips_role_and_kwargs():
    from llama_index.core.llms import ChatMessage

    msg = ChatMessage(role="assistant", content="hi", additional_kwargs={"tool": "search"})
    meta = chat_message_metadata(chat_key="key-1", index=3, message=msg)
    assert meta["llamaindex_source"] == "chat_store"
    assert meta["llamaindex_chat_key"] == "key-1"
    assert meta["llamaindex_chat_index"] == 3
    assert meta["llamaindex_chat_role"] == "assistant"
    assert meta["llamaindex_chat_additional_kwargs"] == {"tool": "search"}


def test_stamp_event_id_returns_new_dict_with_event_id():
    """stamp_event_id never mutates its input dict."""
    src = {"a": 1}
    event_id = uuid4()
    out = stamp_event_id(src, event_id)
    assert out["khora_event_id"] == str(event_id)
    assert out["a"] == 1
    assert "khora_event_id" not in src
