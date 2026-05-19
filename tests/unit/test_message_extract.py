"""Tests for message extraction utilities."""

from __future__ import annotations

from khora.query.message_extract import (
    extract_message_text,
    extract_messages_from_chunk,
    find_message_in_chunk,
)


def _sample_chunk():
    """Return a (content, metadata) pair for a conversation chunk."""
    line1 = "[10:00] alice: hello world"
    line2 = "[10:01] bob: hi there"
    content = f"{line1}\n{line2}"
    metadata = {
        "chunker": "conversation",
        "messages": [
            {
                "id": "m1",
                "author": "alice",
                "timestamp": "2025-01-15T10:00:00+00:00",
                "start_char": 0,
                "end_char": len(line1),
            },
            {
                "id": "m2",
                "author": "bob",
                "timestamp": "2025-01-15T10:01:00+00:00",
                "start_char": len(line1) + 1,
                "end_char": len(line1) + 1 + len(line2),
            },
        ],
    }
    return content, metadata


class TestExtractMessages:
    def test_extract_messages(self):
        _, meta = _sample_chunk()
        msgs = extract_messages_from_chunk(meta)
        assert len(msgs) == 2
        assert msgs[0]["id"] == "m1"

    def test_extract_from_empty_metadata(self):
        assert extract_messages_from_chunk({}) == []

    def test_extract_from_non_conversation(self):
        assert extract_messages_from_chunk({"source_type": "document"}) == []


class TestFindMessage:
    def test_find_by_id(self):
        content, meta = _sample_chunk()
        result = find_message_in_chunk(content, meta, "m1")
        assert result is not None
        assert result["author"] == "alice"
        assert "hello world" in result["text"]

    def test_find_second_message(self):
        content, meta = _sample_chunk()
        result = find_message_in_chunk(content, meta, "m2")
        assert result is not None
        assert result["author"] == "bob"
        assert "hi there" in result["text"]

    def test_find_missing_id(self):
        content, meta = _sample_chunk()
        assert find_message_in_chunk(content, meta, "nonexistent") is None


class TestExtractText:
    def test_extract_text_by_offsets(self):
        content, meta = _sample_chunk()
        m = meta["messages"][0]
        text = extract_message_text(content, m["start_char"], m["end_char"])
        assert text == "[10:00] alice: hello world"

    def test_extract_second_message(self):
        content, meta = _sample_chunk()
        m = meta["messages"][1]
        text = extract_message_text(content, m["start_char"], m["end_char"])
        assert text == "[10:01] bob: hi there"
