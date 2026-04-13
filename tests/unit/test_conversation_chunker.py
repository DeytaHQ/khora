"""Tests for the ConversationChunker."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from khora.extraction.chunkers.conversation import (
    ConversationChunker,
    ConversationChunkerConfig,
    SlackMessage,
)


def _ts(minutes: int = 0) -> datetime:
    """Helper: return a UTC datetime offset by *minutes* from a fixed base."""
    base = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    return base + timedelta(minutes=minutes)


def _msg(
    text: str,
    author: str = "alice",
    minutes: int = 0,
    msg_id: str | None = None,
    thread_ts: str | None = None,
    channel: str | None = "general",
) -> SlackMessage:
    """Helper: build a SlackMessage."""
    return SlackMessage(
        text=text,
        author=author,
        timestamp=_ts(minutes),
        message_id=msg_id or f"msg-{minutes}",
        thread_ts=thread_ts,
        channel=channel,
    )


class TestSlackMessage:
    def test_construction(self):
        msg = _msg("hello")
        assert msg.text == "hello"
        assert msg.author == "alice"

    def test_from_dict(self):
        data = {
            "text": "hi",
            "author": "bob",
            "timestamp": "2025-01-15T10:00:00+00:00",
            "message_id": "m1",
            "thread_ts": "123.456",
            "channel": "random",
            "reactions": ["+1"],
        }
        msg = SlackMessage.from_dict(data)
        assert msg.author == "bob"
        assert msg.thread_ts == "123.456"
        assert msg.reactions == ["+1"]

    def test_sorting_by_timestamp(self):
        msgs = [_msg("c", minutes=10), _msg("a", minutes=0), _msg("b", minutes=5)]
        sorted_msgs = sorted(msgs, key=lambda m: m.timestamp)
        assert [m.text for m in sorted_msgs] == ["a", "b", "c"]


class TestConversationChunker:
    def test_chunk_empty(self):
        chunker = ConversationChunker()
        assert chunker.chunk_messages([]) == []

    def test_single_message(self):
        chunker = ConversationChunker(config=ConversationChunkerConfig(min_group_size=1))
        results = chunker.chunk_messages([_msg("hello")])
        assert len(results) == 1
        assert "hello" in results[0].content

    def test_thread_grouping(self):
        """Messages sharing a thread_ts are grouped together."""
        msgs = [
            _msg("thread start", author="alice", minutes=0, thread_ts="t1"),
            _msg("unrelated top-level", author="carol", minutes=1),
            _msg("thread reply", author="bob", minutes=2, thread_ts="t1"),
        ]
        chunker = ConversationChunker(config=ConversationChunkerConfig(min_group_size=1))
        results = chunker.chunk_messages(msgs)
        # Thread group + top-level group
        assert len(results) == 2
        # Thread group should contain both thread messages
        thread_chunk = next(r for r in results if r.metadata.get("thread_ts") == "t1")
        assert thread_chunk.metadata["message_count"] == 2

    def test_time_gap_splitting(self):
        """A 30-min gap with 15-min threshold splits into two groups."""
        msgs = [
            _msg("morning 1", minutes=0),
            _msg("morning 2", minutes=5),
            _msg("afternoon 1", minutes=35),
            _msg("afternoon 2", minutes=40),
        ]
        chunker = ConversationChunker(config=ConversationChunkerConfig(time_gap_minutes=15))
        results = chunker.chunk_messages(msgs)
        assert len(results) == 2
        assert results[0].metadata["message_count"] == 2
        assert results[1].metadata["message_count"] == 2

    def test_mixed_threads_and_toplevel(self):
        """Threads are separated from top-level conversation."""
        msgs = [
            _msg("top1", minutes=0),
            _msg("top2", minutes=1),
            _msg("thread msg", minutes=2, thread_ts="t1"),
        ]
        chunker = ConversationChunker(config=ConversationChunkerConfig(min_group_size=1))
        results = chunker.chunk_messages(msgs)
        source_types = {r.metadata.get("thread_ts") for r in results}
        assert "t1" in source_types
        assert None in source_types

    def test_max_group_size_splits(self):
        """A group of 60 messages with max=50 splits into two."""
        msgs = [_msg(f"msg {i}", minutes=i, msg_id=f"m{i}") for i in range(60)]
        config = ConversationChunkerConfig(max_group_size=50, min_group_size=2, time_gap_minutes=9999)
        chunker = ConversationChunker(config=config)
        results = chunker.chunk_messages(msgs)
        assert len(results) == 2
        total = sum(r.metadata["message_count"] for r in results)
        assert total == 60

    def test_min_group_size_merges(self):
        """A tiny trailing group merges with its neighbour."""
        # 3 messages then a gap then 1 message → should merge the 1 into the 3
        msgs = [
            _msg("a", minutes=0),
            _msg("b", minutes=1),
            _msg("c", minutes=2),
            _msg("d", minutes=20),  # gap triggers split, but group of 1 < min_group_size=2
        ]
        config = ConversationChunkerConfig(time_gap_minutes=15, min_group_size=2)
        chunker = ConversationChunker(config=config)
        results = chunker.chunk_messages(msgs)
        assert len(results) == 1
        assert results[0].metadata["message_count"] == 4

    def test_format_output(self):
        """Rendered text matches [HH:MM] author: text format."""
        msgs = [_msg("hello world", author="alice", minutes=0)]
        chunker = ConversationChunker(config=ConversationChunkerConfig(min_group_size=1))
        results = chunker.chunk_messages(msgs)
        assert results[0].content == "[10:00] alice: hello world"

    def test_message_metadata_present(self):
        """Chunk metadata contains messages list."""
        msgs = [_msg("hi", minutes=0), _msg("hey", author="bob", minutes=1)]
        chunker = ConversationChunker()
        results = chunker.chunk_messages(msgs)
        assert len(results) == 1
        meta = results[0].metadata
        assert "messages" in meta
        assert len(meta["messages"]) == 2
        assert meta["messages"][0]["author"] == "alice"
        assert meta["messages"][1]["author"] == "bob"

    def test_char_offsets_correct(self):
        """start_char/end_char correctly index into content."""
        msgs = [
            _msg("first message", author="alice", minutes=0),
            _msg("second message", author="bob", minutes=1),
        ]
        chunker = ConversationChunker()
        results = chunker.chunk_messages(msgs)
        chunk = results[0]
        for msg_meta in chunk.metadata["messages"]:
            extracted = chunk.content[msg_meta["start_char"] : msg_meta["end_char"]]
            assert msg_meta["author"] in extracted
            assert "] " in extracted  # has time prefix

    def test_json_input(self):
        """chunk(text) parses a JSON array correctly."""
        data = [
            {
                "text": "hello",
                "author": "alice",
                "timestamp": "2025-01-15T10:00:00+00:00",
                "message_id": "m1",
            },
            {
                "text": "world",
                "author": "bob",
                "timestamp": "2025-01-15T10:01:00+00:00",
                "message_id": "m2",
            },
        ]
        chunker = ConversationChunker()
        results = chunker.chunk(json.dumps(data))
        assert len(results) == 1
        assert "hello" in results[0].content
        assert "world" in results[0].content

    def test_configurable_time_gap(self):
        """5-min vs 60-min gap threshold produces different groupings."""
        msgs = [
            _msg("a", minutes=0),
            _msg("b", minutes=10),
            _msg("c", minutes=20),
        ]
        # 5-min gap → 3 groups (each gap > 5)
        config_tight = ConversationChunkerConfig(time_gap_minutes=5, min_group_size=1)
        results_tight = ConversationChunker(config=config_tight).chunk_messages(msgs)

        # 60-min gap → 1 group (no gap > 60)
        config_wide = ConversationChunkerConfig(time_gap_minutes=60, min_group_size=1)
        results_wide = ConversationChunker(config=config_wide).chunk_messages(msgs)

        assert len(results_tight) == 3
        assert len(results_wide) == 1

    def test_authors_deduplicated(self):
        """metadata authors list contains unique names."""
        msgs = [
            _msg("hi", author="alice", minutes=0),
            _msg("hey", author="bob", minutes=1),
            _msg("yo", author="alice", minutes=2),
        ]
        chunker = ConversationChunker()
        results = chunker.chunk_messages(msgs)
        authors = results[0].metadata["authors"]
        assert authors == ["alice", "bob"]

    def test_chunk_empty_string(self):
        chunker = ConversationChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("  ") == []

    def test_metadata_fields(self):
        """Chunk metadata has all expected top-level fields."""
        msgs = [_msg("hello", channel="dev", minutes=0)]
        chunker = ConversationChunker(config=ConversationChunkerConfig(min_group_size=1))
        results = chunker.chunk_messages(msgs)
        meta = results[0].metadata
        assert meta["source_type"] == "slack_conversation"
        assert meta["channel"] == "dev"
        assert meta["message_count"] == 1
        assert "time_start" in meta
        assert "time_end" in meta
