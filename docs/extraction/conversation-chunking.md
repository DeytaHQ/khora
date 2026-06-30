# Conversation Chunking

Standard text chunkers split on token count or sentence boundaries. That works for articles and docs, but conversations are different - meaning lives in the **exchange** between participants, not in isolated sentences. Splitting a Slack thread mid-reply destroys context.

The `ConversationChunker` groups Slack messages into coherent conversation chunks using three layers of grouping, while preserving per-message metadata so individual messages can still be retrieved from search results.

## The Three-Layer Strategy

### 1. Thread Grouping

Messages that share a `thread_ts` are always kept together. A Slack thread is an intentional conversational unit - splitting it would break context.

```
Channel timeline:
  alice: "Should we deploy today?"        ← top-level
  bob: "Let me check CI" (thread reply)   ← grouped with alice's message
  carol: "Lunch anyone?"                  ← separate top-level
  bob: "CI is green" (thread reply)       ← grouped with alice's message
```

Result: one chunk for the deploy thread, one for Carol's message.

### 2. Temporal Windowing

Top-level messages (not in threads) are split when the time gap between consecutive messages exceeds a threshold (default: 15 minutes). Conversations naturally have pauses - a 30-minute gap likely means a topic change.

```
10:00 alice: "Working on the API"
10:02 bob: "Need help?"
10:05 alice: "I'm good"
                                    ← 25 minute gap → split here
10:30 carol: "Database is slow"
10:32 bob: "Checking"
```

### 3. Semantic Similarity (Optional)

When `semantic_threshold` is set, groups are further split if cosine similarity between adjacent messages drops below the threshold. This catches topic changes that happen without a time gap. Disabled by default.

## Per-Message Metadata

Each chunk stores metadata about every message it contains, including character offsets into the chunk content:

```python
chunk.metadata = {
    "chunker": "conversation",
    "channel": "general",
    "thread_ts": "1234567890.123456",  # None for top-level groups
    "session_id": None,                # Forwarded from the ingest call if set
    "message_count": 3,
    "time_start": "2025-01-15T10:00:00+00:00",
    "time_end": "2025-01-15T10:05:00+00:00",
    "authors": ["alice", "bob"],
    "messages": [                      # Only present when include_message_metadata=True
        {"id": "msg1", "author": "alice", "timestamp": "...", "start_char": 0, "end_char": 42},
        {"id": "msg2", "author": "bob", "timestamp": "...", "start_char": 43, "end_char": 89},
        ...
    ]
}
```

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `time_gap_minutes` | `int` | `15` | Gap threshold to split conversations |
| `session_gap_minutes` | `int` | `30` | Larger gap that marks a new session boundary |
| `max_group_size` | `int` | `50` | Max messages per chunk |
| `min_group_size` | `int` | `2` | Below this, merge with adjacent group |
| `semantic_threshold` | `float \| None` | `None` | Cosine similarity split (None = disabled) |
| `include_message_metadata` | `bool` | `True` | Store per-message data in chunk metadata |

These can also be set via `KhoraConfig`:

| Env var | Config field |
|---------|-------------|
| `KHORA_PIPELINES_CONVERSATION_TIME_GAP_MINUTES` | `pipelines.conversation_time_gap_minutes` |
| `KHORA_PIPELINES_CONVERSATION_MAX_GROUP_SIZE` | `pipelines.conversation_max_group_size` |
| `KHORA_PIPELINES_CONVERSATION_MIN_GROUP_SIZE` | `pipelines.conversation_min_group_size` |
| `KHORA_PIPELINES_CONVERSATION_SEMANTIC_THRESHOLD` | `pipelines.conversation_semantic_threshold` |

## Usage

### Programmatic

```python
from khora.extraction.chunkers import ConversationChunker, ConversationChunkerConfig, SlackMessage
from datetime import datetime, timezone

messages = [
    SlackMessage(
        text="Should we deploy today?",
        author="alice",
        timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        message_id="m1",
        channel="engineering",
    ),
    SlackMessage(
        text="CI is green, let's go",
        author="bob",
        timestamp=datetime(2025, 1, 15, 10, 2, tzinfo=timezone.utc),
        message_id="m2",
        channel="engineering",
    ),
]

config = ConversationChunkerConfig(time_gap_minutes=15)
chunker = ConversationChunker(config=config)
chunks = chunker.chunk_messages(messages)
```

### Via Factory

```python
from khora.extraction.chunkers import create_chunker

chunker = create_chunker("conversation", time_gap_minutes=10, max_group_size=30)
```

### JSON Input

The `chunk()` method accepts a JSON array of message dicts:

```python
import json

data = [
    {"text": "hello", "author": "alice", "timestamp": "2025-01-15T10:00:00+00:00", "message_id": "m1"},
    {"text": "hey!", "author": "bob", "timestamp": "2025-01-15T10:01:00+00:00", "message_id": "m2"},
]
chunks = chunker.chunk(json.dumps(data))
```

## Retrieving Individual Messages from Search Results

After searching, you can extract individual messages from conversation chunks:

```python
from khora.query.message_extract import find_message_in_chunk, extract_messages_from_chunk

# From a search result chunk
messages = extract_messages_from_chunk(chunk.metadata)
for msg in messages:
    print(f"{msg['author']} at {msg['timestamp']}")

# Find a specific message by ID
result = find_message_in_chunk(chunk.content, chunk.metadata, "m1")
if result:
    print(f"{result['author']}: {result['text']}")
```

## Integration with Ingestion Pipeline

When ingesting Slack data, use the conversation chunking strategy:

```yaml
# khora.yaml
pipelines:
  chunking_strategy: conversation
  conversation_time_gap_minutes: 15
  conversation_max_group_size: 50
```

Or programmatically when calling the ingestion pipeline, pass `chunk_strategy="conversation"` to use this chunker instead of the default semantic chunker.
